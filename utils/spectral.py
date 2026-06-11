"""
Spectral Analysis Module for LLM Training

This module tracks spectral properties of weight matrices (W), gradient matrices (G),
and weight update matrices (ΔW = W_t - W_0) throughout training.

Goal: Compare how different optimizers (Adam vs Muon) converge to different spectral solutions.
- Adam hypothesized to produce sparse/low-rank spectra
- Muon hypothesized to produce dense/full-rank spectra
"""

import os
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from collections import defaultdict
import json
import copy
import re


# Layer type categories for stratified analysis
LAYER_TYPES = {
    'embedding': ['wte', 'wpe', 'embed', 'embedding'],
    'attn_qkv': ['c_q', 'c_k', 'c_v', 'q_proj', 'k_proj', 'v_proj', 'query', 'key', 'value'],
    'attn_out': ['c_proj', 'o_proj', 'out_proj', 'attn.c_proj'],
    'mlp_up': ['c_fc', 'fc1', 'up_proj', 'gate_proj', 'w1', 'w3'],
    'mlp_down': ['c_proj', 'fc2', 'down_proj', 'w2'],
    'lm_head': ['lm_head', 'head'],
}


def categorize_layer(layer_name: str) -> str:
    """
    Categorize a layer by its type based on naming patterns.

    Args:
        layer_name: Full parameter name (e.g., 'transformer.h.0.attn.c_q.weight')

    Returns:
        Layer type string: 'embedding', 'attn_qkv', 'attn_out', 'mlp_up', 'mlp_down', 'lm_head', or 'other'
    """
    name_lower = layer_name.lower()

    # Check for embedding layers
    if any(pat in name_lower for pat in LAYER_TYPES['embedding']):
        return 'embedding'

    # Check for LM head
    if any(pat in name_lower for pat in LAYER_TYPES['lm_head']):
        return 'lm_head'

    # Check for attention layers (order matters - check specific patterns first)
    if 'attn' in name_lower or 'attention' in name_lower:
        # Check for output projection in attention
        if any(pat in name_lower for pat in ['c_proj', 'o_proj', 'out_proj']):
            # Make sure it's not in MLP
            if 'mlp' not in name_lower:
                return 'attn_out'
        # Check for Q/K/V
        if any(pat in name_lower for pat in LAYER_TYPES['attn_qkv']):
            return 'attn_qkv'

    # Check for MLP layers
    if 'mlp' in name_lower or 'ffn' in name_lower or 'feed_forward' in name_lower:
        # Up projection patterns
        if any(pat in name_lower for pat in ['c_fc', 'fc1', 'up_proj', 'gate_proj', 'w1', 'w3']):
            return 'mlp_up'
        # Down projection patterns
        if any(pat in name_lower for pat in ['c_proj', 'fc2', 'down_proj', 'w2']):
            return 'mlp_down'

    return 'other'


def get_layer_type_display_name(layer_type: str) -> str:
    """Get human-readable display name for layer type."""
    names = {
        'embedding': 'Embedding',
        'attn_qkv': 'Attention Q/K/V',
        'attn_out': 'Attention Output',
        'mlp_up': 'MLP Up/Gate',
        'mlp_down': 'MLP Down',
        'lm_head': 'LM Head',
        'other': 'Other',
    }
    return names.get(layer_type, layer_type)


def compute_spectral_metrics(
    matrix: torch.Tensor,
    normalize: bool = True,
    eps: float = 1e-6,
    energy_thresholds: Tuple[float, ...] = (0.90, 0.99),
) -> Dict[str, Any]:
    """
    Compute comprehensive spectral metrics for a 2D matrix.

    Args:
        matrix: 2D tensor to analyze (weight, gradient, or weight update)
        normalize: If True, compute metrics on Frobenius-normalized singular values
        eps: Floor for singular values to avoid numerical issues with near-zero values
        energy_thresholds: Thresholds for effective rank computation (e.g., 90%, 99%)

    Returns:
        Dictionary containing:
        - singular_values: Full array of singular values (raw)
        - singular_values_normalized: Frobenius-normalized singular values
        - singular_values_minmax: Min-max normalized singular values
        - spectral_entropy: H = -sum(p_i * log(p_i))
        - normalized_spectral_entropy: H / log(rank) ∈ [0,1]
        - stable_rank: ||M||_F^2 / σ_1^2
        - participation_ratio: (sum(σ_i^2))^2 / sum(σ_i^4)
        - effective_rank_90: Rank at 90% energy threshold
        - effective_rank_99: Rank at 99% energy threshold
        - sigma_max: Largest singular value
        - sigma_min: Smallest singular value
        - condition_number: σ_max / σ_min
        - frobenius_norm: ||M||_F
        - numerical_rank: Number of singular values above eps threshold
    """
    if matrix.dim() != 2:
        raise ValueError(f"Expected 2D matrix, got {matrix.dim()}D tensor")

    # Cast to float32 for numerical stability (important for float16/bfloat16 training)
    matrix = matrix.detach().float()

    # Compute SVD - only singular values needed
    try:
        singular_values = torch.linalg.svdvals(matrix)
    except RuntimeError:
        # Fallback for edge cases
        singular_values = torch.zeros(min(matrix.shape), device=matrix.device)

    # Move to CPU for computations
    sv = singular_values.cpu().numpy()

    # Apply floor to avoid numerical issues
    sv_floored = np.maximum(sv, eps)

    # Basic statistics
    frobenius_norm = np.sqrt(np.sum(sv ** 2))
    sigma_max = sv[0] if len(sv) > 0 else 0.0
    sigma_min = sv[-1] if len(sv) > 0 else 0.0

    # Numerical rank (singular values above threshold)
    numerical_rank = int(np.sum(sv > eps))

    # Frobenius-normalized singular values (comparable across layers)
    if frobenius_norm > eps:
        sv_frobenius_normalized = sv / frobenius_norm
    else:
        sv_frobenius_normalized = sv.copy()

    # Min-max normalized singular values (curves from 1 to ~0)
    if sigma_max > eps:
        sv_minmax = sv / sigma_max
    else:
        sv_minmax = sv.copy()

    # Compute metrics on the appropriate version
    sv_for_metrics = sv_frobenius_normalized if normalize else sv_floored

    # Spectral entropy: H = -sum(p_i * log(p_i)) where p_i = σ_i^2 / sum(σ_j^2)
    sv_squared = sv_for_metrics ** 2
    total_energy = np.sum(sv_squared)

    if total_energy > eps:
        p = sv_squared / total_energy
        # Avoid log(0) by filtering out near-zero probabilities
        p_nonzero = p[p > eps]
        spectral_entropy = -np.sum(p_nonzero * np.log(p_nonzero))
    else:
        spectral_entropy = 0.0

    # Normalized spectral entropy: H / log(rank)
    if numerical_rank > 1:
        normalized_spectral_entropy = spectral_entropy / np.log(numerical_rank)
    else:
        normalized_spectral_entropy = 0.0

    # Clip to [0, 1] for numerical stability
    normalized_spectral_entropy = np.clip(normalized_spectral_entropy, 0.0, 1.0)

    # Stable rank: ||M||_F^2 / σ_1^2
    if sigma_max > eps:
        stable_rank = (frobenius_norm ** 2) / (sigma_max ** 2)
    else:
        stable_rank = 0.0

    # Participation ratio: (sum(σ_i^2))^2 / sum(σ_i^4)
    sv_4 = sv_floored ** 4
    sum_sv_4 = np.sum(sv_4)
    if sum_sv_4 > eps:
        participation_ratio = (frobenius_norm ** 4) / sum_sv_4
    else:
        participation_ratio = 0.0

    # Effective rank at energy thresholds
    effective_ranks = {}
    if total_energy > eps:
        cumulative_energy = np.cumsum(sv_squared) / total_energy
        for threshold in energy_thresholds:
            key = f"effective_rank_{int(threshold * 100)}"
            rank_at_threshold = int(np.searchsorted(cumulative_energy, threshold) + 1)
            effective_ranks[key] = min(rank_at_threshold, len(sv))
    else:
        for threshold in energy_thresholds:
            key = f"effective_rank_{int(threshold * 100)}"
            effective_ranks[key] = 0

    # Condition number
    if sigma_min > eps:
        condition_number = sigma_max / sigma_min
    else:
        condition_number = float('inf')

    # Min dimension for normalization (maximum possible rank)
    min_dim = min(matrix.shape)

    # Normalized stable rank ∈ [0, 1] - divide by min dimension
    normalized_stable_rank = stable_rank / min_dim if min_dim > 0 else 0.0

    # Normalized effective ranks ∈ [0, 1] - divide by min dimension
    normalized_effective_ranks = {}
    for key, value in effective_ranks.items():
        normalized_effective_ranks[f'normalized_{key}'] = value / min_dim if min_dim > 0 else 0.0

    # Normalized participation ratio ∈ [0, 1]
    normalized_participation_ratio = participation_ratio / min_dim if min_dim > 0 else 0.0

    return {
        # Raw singular values (store as list for JSON serialization)
        'singular_values': sv.tolist(),
        'singular_values_normalized': sv_frobenius_normalized.tolist(),
        'singular_values_minmax': sv_minmax.tolist(),

        # Scalar metrics
        'spectral_entropy': float(spectral_entropy),
        'normalized_spectral_entropy': float(normalized_spectral_entropy),
        'stable_rank': float(stable_rank),
        'normalized_stable_rank': float(normalized_stable_rank),  # ∈ [0, 1]
        'participation_ratio': float(participation_ratio),
        'normalized_participation_ratio': float(normalized_participation_ratio),  # ∈ [0, 1]
        **effective_ranks,
        **normalized_effective_ranks,  # normalized_effective_rank_90, normalized_effective_rank_99 ∈ [0, 1]
        'sigma_max': float(sigma_max),
        'sigma_min': float(sigma_min),
        'condition_number': float(condition_number) if not np.isinf(condition_number) else None,
        'frobenius_norm': float(frobenius_norm),
        'numerical_rank': numerical_rank,
        'min_dim': min_dim,
        'matrix_shape': list(matrix.shape),
    }


def reshape_to_2d(weight: torch.Tensor, module: nn.Module) -> Optional[torch.Tensor]:
    """
    Reshape weight tensor to 2D for spectral analysis.

    Args:
        weight: Weight tensor from the module
        module: The nn.Module containing the weight

    Returns:
        2D tensor or None if not applicable
    """
    if weight is None:
        return None

    if isinstance(module, (nn.Linear, nn.Embedding)):
        # Already 2D: (out_features, in_features) or (num_embeddings, embedding_dim)
        if weight.dim() == 2:
            return weight
        else:
            return weight.view(weight.size(0), -1)

    elif isinstance(module, nn.Conv2d):
        # Shape: (out_channels, in_channels, kH, kW)
        # Flatten to: (out_channels, in_channels * kH * kW)
        return weight.view(weight.size(0), -1)

    elif weight.dim() == 2:
        # Already 2D
        return weight

    elif weight.dim() > 2:
        # Generic flattening for other multi-dimensional weights
        return weight.view(weight.size(0), -1)

    return None


def is_trackable_layer(
    name: str,
    module: nn.Module,
    weight: torch.Tensor,
    min_matrix_dim: int = 64,
) -> bool:
    """
    Check if a layer should be tracked for spectral analysis.

    Args:
        name: Parameter name
        module: Parent module
        weight: Weight tensor
        min_matrix_dim: Minimum dimension for meaningful spectral analysis

    Returns:
        True if layer should be tracked
    """
    # Skip biases, norms, and other 1D tensors
    if weight.dim() < 2:
        return False

    # Skip very small matrices (no meaningful spectra)
    if min(weight.shape) < min_matrix_dim:
        return False

    # Skip norm layers (typically have 1D parameters)
    if isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
        return False

    # Track Linear, Embedding, Conv2d, and other 2D+ weight matrices
    if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv2d)):
        return True

    # Generic check for 2D+ weights with sufficient size
    return True


class SpectralTracker:
    """
    Tracks spectral properties of weights, gradients, and weight updates during training.

    Hypothesis:
    - Adam: Sparse/low-rank spectra (low spectral entropy, low effective rank)
    - Muon: Dense/full-rank spectra (high spectral entropy, high effective rank)

    Usage:
        tracker = SpectralTracker(model, track_every=100)

        for step in range(num_steps):
            loss.backward()

            # Before optimizer.step() - record gradients
            tracker.step_pre_optimizer(step)

            optimizer.step()

            # After optimizer.step() - record weights and delta_w
            tracker.step_post_optimizer(step)
    """

    # Matrix types for tracking
    WEIGHT = 'W'
    GRADIENT = 'G'
    DELTA_W = 'delta_W'  # Cumulative: W_t - W_0
    STEP_UPDATE = 'step_update'  # One-step: W_{t+1} - W_t

    def __init__(
        self,
        model: nn.Module,
        track_every: int = 100,
        layer_filter: Optional[callable] = None,
        min_matrix_dim: int = 64,
        eps: float = 1e-6,
        delta_w_min_norm: float = 1e-8,
    ):
        """
        Initialize the spectral tracker.

        Args:
            model: PyTorch model to track
            track_every: Track spectral metrics every N steps
            layer_filter: Optional callable(name, module) -> bool for filtering layers
            min_matrix_dim: Minimum matrix dimension for meaningful spectral analysis
            eps: Epsilon for numerical stability
            delta_w_min_norm: Skip ΔW metrics if ||ΔW||_F < this threshold
        """
        self.model = model
        self.track_every = track_every
        self.layer_filter = layer_filter
        self.min_matrix_dim = min_matrix_dim
        self.eps = eps
        self.delta_w_min_norm = delta_w_min_norm

        # Snapshot W_0 at initialization (detached clones)
        self.w0_snapshots: Dict[str, torch.Tensor] = {}

        # Pre-step weight snapshots for computing step_update = W_{t+1} - W_t
        self.w_pre_step: Dict[str, torch.Tensor] = {}

        # History storage: history[layer_name][matrix_type] -> list of (step, metrics_dict)
        self.history: Dict[str, Dict[str, List[Tuple[int, Dict]]]] = defaultdict(
            lambda: defaultdict(list)
        )

        # Module references for reshape operations
        self.tracked_params: Dict[str, Tuple[nn.Module, str]] = {}

        # Initialize snapshots and tracked parameters
        self._init_tracking()

        # Track if we've done any tracking yet
        self.last_tracked_step = -1

    def _init_tracking(self):
        """Initialize W_0 snapshots and identify tracked parameters."""
        for name, module in self.model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                full_name = f"{name}.{param_name}" if name else param_name

                # Apply layer filter if provided
                if self.layer_filter is not None:
                    if not self.layer_filter(full_name, module):
                        continue

                # Check if layer is trackable
                if not is_trackable_layer(full_name, module, param, self.min_matrix_dim):
                    continue

                # Store module reference for reshaping
                self.tracked_params[full_name] = (module, param_name)

                # Snapshot W_0
                weight_2d = reshape_to_2d(param.data, module)
                if weight_2d is not None:
                    self.w0_snapshots[full_name] = weight_2d.detach().clone()

    def should_track(self, step: int) -> bool:
        """Check if we should track at this step."""
        return step % self.track_every == 0

    def step_pre_optimizer(self, step: int):
        """
        Call AFTER loss.backward() but BEFORE optimizer.step().
        Records gradient metrics and snapshots weights for step_update computation.

        Args:
            step: Current training step (global step count)
        """
        if not self.should_track(step):
            return

        for name, (module, param_name) in self.tracked_params.items():
            param = getattr(module, param_name)

            # Snapshot current weights for step_update computation
            weight_2d = reshape_to_2d(param.data, module)
            if weight_2d is not None:
                self.w_pre_step[name] = weight_2d.detach().clone()

            # Track gradient if available
            if param.grad is not None:
                grad_2d = reshape_to_2d(param.grad, module)
                if grad_2d is not None:
                    metrics = compute_spectral_metrics(grad_2d, normalize=True, eps=self.eps)
                    self.history[name][self.GRADIENT].append((step, metrics))

    def step_post_optimizer(self, step: int):
        """
        Call AFTER optimizer.step().
        Records weight metrics, delta_W (cumulative), and step_update (one-step).

        Args:
            step: Current training step (global step count)
        """
        if not self.should_track(step):
            return

        self.last_tracked_step = step

        for name, (module, param_name) in self.tracked_params.items():
            param = getattr(module, param_name)
            weight_2d = reshape_to_2d(param.data, module)

            if weight_2d is None:
                continue

            # Track weights
            metrics = compute_spectral_metrics(weight_2d, normalize=True, eps=self.eps)
            self.history[name][self.WEIGHT].append((step, metrics))

            # Track delta_W = W_t - W_0 (cumulative change from initialization)
            if name in self.w0_snapshots:
                delta_w = weight_2d - self.w0_snapshots[name]
                delta_w_norm = torch.norm(delta_w).item()

                if delta_w_norm > self.delta_w_min_norm:
                    delta_metrics = compute_spectral_metrics(delta_w, normalize=True, eps=self.eps)
                    self.history[name][self.DELTA_W].append((step, delta_metrics))

            # Track step_update = W_{t+1} - W_t (one-step change)
            if name in self.w_pre_step:
                step_update = weight_2d - self.w_pre_step[name]
                step_update_norm = torch.norm(step_update).item()

                if step_update_norm > self.delta_w_min_norm:
                    step_metrics = compute_spectral_metrics(step_update, normalize=True, eps=self.eps)
                    self.history[name][self.STEP_UPDATE].append((step, step_metrics))

    def step(self, step: int, phase: str = 'post'):
        """
        Single-method interface for tracking.

        Args:
            step: Current training step
            phase: 'pre' (before optimizer.step) or 'post' (after optimizer.step)
        """
        if phase == 'pre':
            self.step_pre_optimizer(step)
        elif phase == 'post':
            self.step_post_optimizer(step)
        else:
            raise ValueError(f"Invalid phase: {phase}. Use 'pre' or 'post'.")

    def get_tracked_layers(self) -> List[str]:
        """Return list of tracked layer names."""
        return list(self.tracked_params.keys())

    def get_latest_metrics(self, layer_name: str, matrix_type: str) -> Optional[Dict]:
        """Get the most recent metrics for a specific layer and matrix type."""
        if layer_name in self.history and matrix_type in self.history[layer_name]:
            entries = self.history[layer_name][matrix_type]
            if entries:
                return entries[-1][1]  # Return metrics dict from (step, metrics) tuple
        return None

    def get_metrics_at_step(self, step: int) -> Dict[str, Dict[str, Dict]]:
        """
        Get all metrics at a specific step.

        Returns:
            Nested dict: {layer_name: {matrix_type: metrics_dict}}
        """
        result = {}
        for layer_name, matrix_types in self.history.items():
            result[layer_name] = {}
            for matrix_type, entries in matrix_types.items():
                for entry_step, metrics in entries:
                    if entry_step == step:
                        result[layer_name][matrix_type] = metrics
                        break
        return result

    def aggregate(self, step: Optional[int] = None) -> Dict[str, Dict[str, Any]]:
        """
        Compute aggregated metrics across all tracked layers for a given step.

        Args:
            step: Training step to aggregate (None = latest)

        Returns:
            Dictionary with per-matrix-type aggregations:
            {
                'W': {
                    'mean_spectral_entropy': ...,
                    'std_spectral_entropy': ...,
                    'weighted_mean_spectral_entropy': ...,
                    ... (same for other metrics)
                },
                'G': {...},
                'delta_W': {...},
                'step_update': {...}
            }
        """
        if step is None:
            step = self.last_tracked_step

        if step < 0:
            return {}

        # Scalar metrics to aggregate
        scalar_metrics = [
            'spectral_entropy',
            'normalized_spectral_entropy',
            'stable_rank',
            'normalized_stable_rank',  # ∈ [0, 1]
            'participation_ratio',
            'normalized_participation_ratio',  # ∈ [0, 1]
            'effective_rank_90',
            'effective_rank_99',
            'normalized_effective_rank_90',  # ∈ [0, 1]
            'normalized_effective_rank_99',  # ∈ [0, 1]
            'sigma_max',
            'condition_number',
            'frobenius_norm',
            'numerical_rank',
        ]

        result = {}

        for matrix_type in [self.WEIGHT, self.GRADIENT, self.DELTA_W, self.STEP_UPDATE]:
            values_per_metric = defaultdict(list)
            weights = []  # Frobenius norms for weighted averaging

            for layer_name, matrix_types in self.history.items():
                if matrix_type not in matrix_types:
                    continue

                # Find metrics at the requested step
                for entry_step, metrics in matrix_types[matrix_type]:
                    if entry_step == step:
                        weights.append(metrics.get('frobenius_norm', 1.0))
                        for metric_name in scalar_metrics:
                            if metric_name in metrics and metrics[metric_name] is not None:
                                values_per_metric[metric_name].append(metrics[metric_name])
                        break

            if not values_per_metric:
                continue

            # Compute aggregations
            agg = {'step': step, 'num_layers': len(weights)}
            weights = np.array(weights)

            for metric_name, values in values_per_metric.items():
                values = np.array(values)

                # Unweighted statistics
                agg[f'mean_{metric_name}'] = float(np.mean(values))
                agg[f'std_{metric_name}'] = float(np.std(values))
                agg[f'min_{metric_name}'] = float(np.min(values))
                agg[f'max_{metric_name}'] = float(np.max(values))

                # Weighted mean (larger layers count more)
                if len(weights) == len(values) and np.sum(weights) > 0:
                    weighted_mean = np.average(values, weights=weights)
                    agg[f'weighted_mean_{metric_name}'] = float(weighted_mean)

            result[matrix_type] = agg

        return result

    def get_history_for_layer(
        self,
        layer_name: str,
        matrix_type: str
    ) -> List[Tuple[int, Dict]]:
        """Get full history for a specific layer and matrix type."""
        return self.history.get(layer_name, {}).get(matrix_type, [])

    def get_layer_types(self) -> Dict[str, List[str]]:
        """
        Group tracked layers by their type.

        Returns:
            Dict mapping layer_type -> list of layer names
        """
        groups = defaultdict(list)
        for layer_name in self.tracked_params.keys():
            layer_type = categorize_layer(layer_name)
            groups[layer_type].append(layer_name)
        return dict(groups)

    def aggregate_by_layer_type(
        self,
        step: Optional[int] = None,
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Compute aggregated metrics stratified by layer type.

        Args:
            step: Training step to aggregate (None = latest)
            metrics: List of metric names to include (None = all normalized metrics)

        Returns:
            Nested dict: {matrix_type: {layer_type: {metric_name: {
                'mean': ..., 'std': ..., 'median': ..., 'q25': ..., 'q75': ...,
                'weighted_mean': ..., 'values': [...], 'weights': [...],
                'layer_names': [...]
            }}}}
        """
        if step is None:
            step = self.last_tracked_step

        if step < 0:
            return {}

        if metrics is None:
            # Default to normalized metrics for fair comparison
            metrics = [
                'normalized_stable_rank',
                'normalized_effective_rank_90',
                'normalized_effective_rank_99',
                'normalized_spectral_entropy',
                'normalized_participation_ratio',
            ]

        layer_groups = self.get_layer_types()
        result = {}

        for matrix_type in [self.WEIGHT, self.GRADIENT, self.DELTA_W, self.STEP_UPDATE]:
            result[matrix_type] = {}

            for layer_type, layer_names in layer_groups.items():
                type_result = {}

                for metric_name in metrics:
                    values = []
                    weights = []  # Parameter counts for weighting
                    layer_names_with_data = []

                    for layer_name in layer_names:
                        if matrix_type not in self.history.get(layer_name, {}):
                            continue

                        # Find metrics at the requested step
                        for entry_step, entry_metrics in self.history[layer_name][matrix_type]:
                            if entry_step == step:
                                if metric_name in entry_metrics and entry_metrics[metric_name] is not None:
                                    values.append(entry_metrics[metric_name])
                                    # Use min_dim * max_dim as weight (parameter count proxy)
                                    shape = entry_metrics.get('matrix_shape', [1, 1])
                                    weights.append(shape[0] * shape[1])
                                    layer_names_with_data.append(layer_name)
                                break

                    if not values:
                        continue

                    values = np.array(values)
                    weights = np.array(weights)

                    type_result[metric_name] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'median': float(np.median(values)),
                        'q25': float(np.percentile(values, 25)),
                        'q75': float(np.percentile(values, 75)),
                        'min': float(np.min(values)),
                        'max': float(np.max(values)),
                        'weighted_mean': float(np.average(values, weights=weights)) if np.sum(weights) > 0 else float(np.mean(values)),
                        'n_layers': len(values),
                        'values': values.tolist(),
                        'weights': weights.tolist(),
                        'layer_names': layer_names_with_data,
                    }

                if type_result:
                    result[matrix_type][layer_type] = type_result

        return result

    def get_stratified_history(
        self,
        metric_name: str = 'normalized_stable_rank',
        matrix_type: str = 'W',
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get time series of a metric stratified by layer type.

        Args:
            metric_name: Which metric to track
            matrix_type: Which matrix type (W, G, delta_W, step_update)

        Returns:
            Dict: {layer_type: {
                'steps': [...],
                'mean': [...],
                'std': [...],
                'q25': [...],
                'q75': [...],
                'weighted_mean': [...],
            }}
        """
        all_steps = self.get_all_steps()
        layer_groups = self.get_layer_types()
        result = {}

        for layer_type, layer_names in layer_groups.items():
            steps_data = []
            means = []
            stds = []
            q25s = []
            q75s = []
            weighted_means = []

            for step in all_steps:
                values = []
                weights = []

                for layer_name in layer_names:
                    if matrix_type not in self.history.get(layer_name, {}):
                        continue

                    for entry_step, entry_metrics in self.history[layer_name][matrix_type]:
                        if entry_step == step:
                            if metric_name in entry_metrics and entry_metrics[metric_name] is not None:
                                values.append(entry_metrics[metric_name])
                                shape = entry_metrics.get('matrix_shape', [1, 1])
                                weights.append(shape[0] * shape[1])
                            break

                if values:
                    values = np.array(values)
                    weights = np.array(weights)
                    steps_data.append(step)
                    means.append(float(np.mean(values)))
                    stds.append(float(np.std(values)))
                    q25s.append(float(np.percentile(values, 25)))
                    q75s.append(float(np.percentile(values, 75)))
                    weighted_means.append(
                        float(np.average(values, weights=weights)) if np.sum(weights) > 0 else float(np.mean(values))
                    )

            if steps_data:
                result[layer_type] = {
                    'steps': steps_data,
                    'mean': means,
                    'std': stds,
                    'q25': q25s,
                    'q75': q75s,
                    'weighted_mean': weighted_means,
                }

        return result

    def get_optimizer_comparison_data(
        self,
        metric_name: str = 'normalized_stable_rank',
        matrix_type: str = 'W',
    ) -> Dict[str, Any]:
        """
        Get data formatted for optimizer comparison plots.

        Returns aggregate weighted mean with IQR bands across all layers.

        Returns:
            Dict: {
                'steps': [...],
                'weighted_mean': [...],
                'q25': [...],  # 25th percentile across layers
                'q75': [...],  # 75th percentile across layers
                'per_layer_type': {layer_type: {...}},
            }
        """
        all_steps = self.get_all_steps()

        steps_out = []
        weighted_means = []
        q25s = []
        q75s = []

        for step in all_steps:
            all_values = []
            all_weights = []

            for layer_name, matrix_types in self.history.items():
                if matrix_type not in matrix_types:
                    continue

                for entry_step, entry_metrics in matrix_types[matrix_type]:
                    if entry_step == step:
                        if metric_name in entry_metrics and entry_metrics[metric_name] is not None:
                            all_values.append(entry_metrics[metric_name])
                            shape = entry_metrics.get('matrix_shape', [1, 1])
                            all_weights.append(shape[0] * shape[1])
                        break

            if all_values:
                values = np.array(all_values)
                weights = np.array(all_weights)

                steps_out.append(step)
                weighted_means.append(
                    float(np.average(values, weights=weights)) if np.sum(weights) > 0 else float(np.mean(values))
                )
                q25s.append(float(np.percentile(values, 25)))
                q75s.append(float(np.percentile(values, 75)))

        return {
            'steps': steps_out,
            'weighted_mean': weighted_means,
            'q25': q25s,
            'q75': q75s,
            'per_layer_type': self.get_stratified_history(metric_name, matrix_type),
        }

    def get_all_steps(self) -> List[int]:
        """Get all steps where tracking occurred."""
        steps = set()
        for layer_name, matrix_types in self.history.items():
            for matrix_type, entries in matrix_types.items():
                for step, _ in entries:
                    steps.add(step)
        return sorted(steps)

    def state_dict(self) -> Dict[str, Any]:
        """Return tracker state for checkpointing."""
        return {
            'w0_snapshots': {k: v.cpu() for k, v in self.w0_snapshots.items()},
            'history': dict(self.history),  # Convert defaultdict to dict
            'track_every': self.track_every,
            'min_matrix_dim': self.min_matrix_dim,
            'eps': self.eps,
            'delta_w_min_norm': self.delta_w_min_norm,
            'last_tracked_step': self.last_tracked_step,
            'tracked_params_names': list(self.tracked_params.keys()),
        }

    def load_state_dict(self, state: Dict[str, Any]):
        """Load tracker state from checkpoint."""
        # Restore W_0 snapshots (move to same device as model)
        device = next(self.model.parameters()).device
        for name, tensor in state['w0_snapshots'].items():
            if name in self.tracked_params:
                self.w0_snapshots[name] = tensor.to(device)

        # Restore history
        self.history = defaultdict(lambda: defaultdict(list))
        for layer_name, matrix_types in state['history'].items():
            for matrix_type, entries in matrix_types.items():
                self.history[layer_name][matrix_type] = entries

        self.last_tracked_step = state.get('last_tracked_step', -1)

    def get_aggregate_history(self) -> List[Dict[str, Any]]:
        """
        Compute aggregated metrics for all tracked steps.

        Returns:
            List of aggregate dictionaries, one per step
        """
        all_steps = self.get_all_steps()
        return [{'step': step, **self.aggregate(step)} for step in all_steps]

    def export_to_json(self, path: str):
        """Export all history to JSON file."""
        # Get layer type groupings
        layer_types = self.get_layer_types()

        # Get stratified histories for key metrics
        stratified_data = {}
        for metric in ['normalized_stable_rank', 'normalized_effective_rank_99', 'normalized_spectral_entropy']:
            for matrix_type in [self.WEIGHT, self.GRADIENT, self.DELTA_W, self.STEP_UPDATE]:
                key = f'{matrix_type}_{metric}'
                stratified_data[key] = self.get_stratified_history(metric, matrix_type)

        # Get optimizer comparison data
        comparison_data = {}
        for metric in ['normalized_stable_rank', 'normalized_effective_rank_99', 'normalized_spectral_entropy']:
            for matrix_type in [self.WEIGHT, self.GRADIENT]:
                key = f'{matrix_type}_{metric}'
                comparison_data[key] = self.get_optimizer_comparison_data(metric, matrix_type)

        # Convert history to JSON-serializable format
        export_data = {
            'tracked_layers': list(self.tracked_params.keys()),
            'layer_types': layer_types,
            'track_every': self.track_every,
            'min_matrix_dim': self.min_matrix_dim,
            'history': {},
            'aggregates': self.get_aggregate_history(),
            'stratified': stratified_data,
            'comparison_data': comparison_data,
        }

        for layer_name, matrix_types in self.history.items():
            export_data['history'][layer_name] = {
                'layer_type': categorize_layer(layer_name),
            }
            for matrix_type, entries in matrix_types.items():
                export_data['history'][layer_name][matrix_type] = [
                    {'step': step, 'metrics': metrics}
                    for step, metrics in entries
                ]

        with open(path, 'w') as f:
            json.dump(export_data, f, indent=2)

    @staticmethod
    def load_from_json(path: str) -> Dict[str, Any]:
        """Load exported history from JSON file."""
        with open(path, 'r') as f:
            return json.load(f)


class SpectralLogger:
    """
    Logger for spectral metrics during training.

    Writes metrics incrementally using JSONL format (one JSON object per line)
    for efficient appending without rewriting the entire file.

    Also maintains in-memory history for final JSON export with full structure.
    """

    def __init__(self, output_dir: str, optimizer_name: str):
        """
        Initialize spectral logger.

        Args:
            output_dir: Directory to save logs
            optimizer_name: Name of optimizer (for file naming)
        """
        import os
        self.output_dir = output_dir
        self.optimizer_name = optimizer_name

        # In-memory history (for final structured export)
        self.scalar_history: List[Dict] = []
        self.layer_history: Dict[str, List[Dict]] = defaultdict(list)

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # File paths for incremental logging (JSONL format)
        self.aggregate_jsonl_path = os.path.join(
            self.output_dir,
            f'spectral_aggregate_{self.optimizer_name}.jsonl'
        )
        self.layers_jsonl_path = os.path.join(
            self.output_dir,
            f'spectral_layers_{self.optimizer_name}.jsonl'
        )

        # Clear existing files (fresh start)
        for path in [self.aggregate_jsonl_path, self.layers_jsonl_path]:
            if os.path.exists(path):
                os.remove(path)

    def log_aggregate(self, step: int, aggregate: Dict[str, Dict]):
        """
        Log aggregated metrics for a step.

        Immediately appends to JSONL file and stores in memory.
        """
        row = {'step': step}

        for matrix_type, metrics in aggregate.items():
            for key, value in metrics.items():
                if key != 'step' and key != 'num_layers':
                    row[f'{matrix_type}_{key}'] = value

        # Store in memory
        self.scalar_history.append(row)

        # Append to JSONL file immediately
        with open(self.aggregate_jsonl_path, 'a') as f:
            f.write(json.dumps(row) + '\n')

    def log_layer_metrics(self, step: int, layer_name: str, matrix_type: str, metrics: Dict):
        """
        Log per-layer metrics for a step.

        Immediately appends to JSONL file and stores in memory.
        """
        entry = {
            'step': step,
            'layer_name': layer_name,
            'matrix_type': matrix_type,
            **{k: v for k, v in metrics.items() if not k.startswith('singular_values')}
        }

        # Store in memory (grouped by layer)
        self.layer_history[layer_name].append({
            'step': step,
            'matrix_type': matrix_type,
            **{k: v for k, v in metrics.items() if not k.startswith('singular_values')}
        })

        # Append to JSONL file immediately
        with open(self.layers_jsonl_path, 'a') as f:
            f.write(json.dumps(entry) + '\n')

    def save_all(self):
        """
        Save final structured JSON files for visualization compatibility.

        The JSONL files are already written incrementally, but this creates
        the structured JSON format expected by the visualization scripts.
        """
        import os
        os.makedirs(self.output_dir, exist_ok=True)

        # Save scalar aggregates as structured JSON (for visualization)
        scalar_path = os.path.join(
            self.output_dir,
            f'spectral_aggregate_{self.optimizer_name}.json'
        )
        with open(scalar_path, 'w') as f:
            json.dump(self.scalar_history, f, indent=2)

        # Save per-layer metrics as structured JSON (for visualization)
        layer_path = os.path.join(
            self.output_dir,
            f'spectral_layers_{self.optimizer_name}.json'
        )
        with open(layer_path, 'w') as f:
            json.dump(dict(self.layer_history), f, indent=2)

    @staticmethod
    def load_from_jsonl(jsonl_path: str) -> List[Dict]:
        """
        Load data from a JSONL file.

        Args:
            jsonl_path: Path to JSONL file

        Returns:
            List of dictionaries (one per line)
        """
        entries = []
        if os.path.exists(jsonl_path):
            with open(jsonl_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entries.append(json.loads(line))
        return entries

    def get_current_aggregate_history(self) -> List[Dict]:
        """
        Get current aggregate history by reading from JSONL file.

        Useful for monitoring during training.
        """
        return self.load_from_jsonl(self.aggregate_jsonl_path)


# Convenience function to load JSONL files
def load_spectral_jsonl(results_dir: str, optimizer: str, data_type: str = 'aggregate') -> List[Dict]:
    """
    Load spectral data from JSONL files.

    Args:
        results_dir: Directory containing spectral results
        optimizer: Optimizer name
        data_type: 'aggregate' or 'layers'

    Returns:
        List of dictionaries from JSONL file
    """
    import os
    if data_type == 'aggregate':
        path = os.path.join(results_dir, f'spectral_aggregate_{optimizer}.jsonl')
    else:
        path = os.path.join(results_dir, f'spectral_layers_{optimizer}.jsonl')

    entries = []
    if os.path.exists(path):
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
    return entries
