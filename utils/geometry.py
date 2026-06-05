"""
Optimization Geometry Analysis Module

This module provides comprehensive tools for analyzing the geometry of
neural network optimization, including:
- Hessian eigenvalue spectrum via Lanczos iteration
- Per-layer gradient analysis (norms, effective rank, subspace overlap)
- Gradient similarity metrics across layers and training steps
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


# =============================================================================
# HESSIAN EIGENVALUE SPECTRUM (LANCZOS ALGORITHM)
# =============================================================================

def compute_hvp(model, data, targets, v):
    """
    Compute Hessian-Vector Product using reverse-over-reverse autograd.

    Note: Disables flash/efficient attention during computation since those
    kernels don't support second-order gradients.
    """
    params = [p for p in model.parameters() if p.requires_grad]

    # Disable efficient attention backends that don't support double backward
    # Use the math backend which does support second-order derivatives
    with torch.enable_grad(), \
         torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
        loss = model(data, targets=targets)
        grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
        dot_product = sum([(g * v_).sum() for g, v_ in zip(grads, v)])
        hvp = torch.autograd.grad(dot_product, params, retain_graph=True)

    return [h.detach() for h in hvp]


def lanczos_iteration(model, data_loader, k=10, num_iterations=50, device='cuda'):
    """
    Lanczos algorithm for computing top-k eigenvalues of the Hessian.

    Returns:
        eigenvalues: Array of top-k eigenvalues (sorted descending)
        T_matrix: Tridiagonal matrix (for computing full spectrum if needed)
    """
    params = [p for p in model.parameters() if p.requires_grad]
    param_count = sum(p.numel() for p in params)

    # Get a batch for HVP computation
    x, y = next(data_loader)

    # Initialize random starting vector
    v = [torch.randn_like(p, device=device) for p in params]
    norm_v = torch.sqrt(sum([torch.sum(v_ ** 2) for v_ in v]))
    v = [v_ / norm_v for v_ in v]

    # Lanczos vectors and tridiagonal matrix elements
    V = [v]  # List of orthonormal vectors
    alphas = []  # Diagonal elements
    betas = []   # Off-diagonal elements

    v_prev = None

    for j in range(min(num_iterations, k + 20)):  # Extra iterations for convergence
        # w = H @ v[j]
        w = compute_hvp(model, x, y, v)

        # alpha_j = v[j]^T @ w
        alpha = sum([(v_ * w_).sum() for v_, w_ in zip(v, w)]).item()
        alphas.append(alpha)

        # w = w - alpha_j * v[j] - beta_{j-1} * v[j-1]
        w = [w_ - alpha * v_ for w_, v_ in zip(w, v)]
        if v_prev is not None:
            w = [w_ - betas[-1] * vp_ for w_, vp_ in zip(w, v_prev)]

        # beta_j = ||w||
        beta = torch.sqrt(sum([torch.sum(w_ ** 2) for w_ in w])).item()

        if beta < 1e-10:
            break

        betas.append(beta)

        # v[j+1] = w / beta_j
        v_prev = v
        v = [w_ / beta for w_ in w]

        # Re-orthogonalize against all previous vectors (for numerical stability)
        for v_old in V:
            proj = sum([(v_ * vo_).sum() for v_, vo_ in zip(v, v_old)])
            v = [v_ - proj * vo_ for v_, vo_ in zip(v, v_old)]

        norm_v = torch.sqrt(sum([torch.sum(v_ ** 2) for v_ in v]))
        if norm_v > 1e-10:
            v = [v_ / norm_v for v_ in v]
            V.append(v)

    # Build tridiagonal matrix
    m = len(alphas)
    T = np.zeros((m, m))
    for i in range(m):
        T[i, i] = alphas[i]
        if i < m - 1 and i < len(betas):
            T[i, i+1] = betas[i]
            T[i+1, i] = betas[i]

    # Compute eigenvalues of tridiagonal matrix
    eigenvalues = np.linalg.eigvalsh(T)
    eigenvalues = np.sort(eigenvalues)[::-1]  # Sort descending

    return eigenvalues[:k], T


def compute_spectral_density(eigenvalues: np.ndarray, num_bins: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute histogram-based spectral density from eigenvalues.

    Returns:
        bin_centers: Center points of histogram bins
        density: Normalized density values
    """
    counts, bin_edges = np.histogram(eigenvalues, bins=num_bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return bin_centers, counts


# =============================================================================
# PER-LAYER GRADIENT ANALYSIS
# =============================================================================

def compute_layer_gradient_norms(model: nn.Module) -> Dict[str, float]:
    """
    Compute L2 gradient norm for each layer.

    Returns:
        Dictionary mapping layer names to their gradient norms
    """
    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms[name] = param.grad.norm(2).item()
    return grad_norms


def compute_effective_rank(weight_matrix: torch.Tensor, eps: float = 1e-10) -> float:
    """
    Compute effective rank of a weight matrix using entropy of normalized singular values.

    effective_rank = exp(-sum(p_i * log(p_i)))
    where p_i = sigma_i / sum(sigma_j)

    This measures how many singular directions are actually being used.
    """
    if weight_matrix.dim() != 2:
        # For non-2D tensors (like biases), return 1
        return 1.0

    # Compute SVD (only need singular values)
    try:
        S = torch.linalg.svdvals(weight_matrix.float())
    except:
        return 1.0

    # Normalize to get probability distribution
    S = S + eps  # Avoid log(0)
    p = S / S.sum()

    # Compute entropy
    entropy = -(p * torch.log(p)).sum().item()

    # Effective rank = exp(entropy)
    return np.exp(entropy)


def compute_layer_effective_ranks(model: nn.Module) -> Dict[str, float]:
    """
    Compute effective rank for each 2D weight matrix in the model.
    """
    effective_ranks = {}
    for name, param in model.named_parameters():
        if param.dim() == 2:
            effective_ranks[name] = compute_effective_rank(param.data)
    return effective_ranks


# =============================================================================
# GRADIENT SUBSPACE ANALYSIS
# =============================================================================

def compute_gradient_svd(grad_matrix: torch.Tensor, k: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute top-k left singular vectors of a gradient matrix.

    Returns:
        U_k: Top-k left singular vectors (columns)
        S_k: Top-k singular values
    """
    if grad_matrix.dim() != 2:
        return None, None

    try:
        # Use randomized SVD for efficiency on large matrices
        U, S, Vh = torch.linalg.svd(grad_matrix.float(), full_matrices=False)
        return U[:, :k], S[:k]
    except:
        return None, None


def compute_principal_angles(U1: torch.Tensor, U2: torch.Tensor) -> torch.Tensor:
    """
    Compute principal angles between two subspaces spanned by column vectors.

    The principal angles theta_i are defined by:
    cos(theta_i) = singular values of U1^T @ U2

    Returns:
        Principal angles in radians (sorted ascending)
    """
    if U1 is None or U2 is None:
        return None

    # Compute U1^T @ U2
    M = U1.T @ U2

    # Singular values give cosines of principal angles
    try:
        S = torch.linalg.svdvals(M)
        # Clamp to valid range for arccos
        S = torch.clamp(S, -1.0, 1.0)
        angles = torch.acos(S)
        return angles
    except:
        return None


def compute_subspace_overlap(U1: torch.Tensor, U2: torch.Tensor) -> float:
    """
    Compute subspace overlap as average cosine of principal angles.

    Overlap = 1 means identical subspaces
    Overlap = 0 means orthogonal subspaces
    """
    angles = compute_principal_angles(U1, U2)
    if angles is None:
        return 0.0

    # Average cosine of principal angles
    return torch.cos(angles).mean().item()


class GradientSubspaceTracker:
    """
    Track gradient subspace evolution across training steps for each layer.
    """

    def __init__(self, model: nn.Module, top_k: int = 10):
        self.model = model
        self.top_k = top_k
        self.prev_subspaces = {}  # Store previous step's subspaces per layer

    def update(self) -> Dict[str, float]:
        """
        Compute subspace overlap with previous step for each layer.

        Returns:
            Dictionary mapping layer names to subspace overlap scores
        """
        overlaps = {}
        current_subspaces = {}

        for name, param in self.model.named_parameters():
            if param.grad is not None and param.grad.dim() == 2:
                U_k, S_k = compute_gradient_svd(param.grad, self.top_k)

                if U_k is not None:
                    current_subspaces[name] = U_k

                    if name in self.prev_subspaces:
                        overlap = compute_subspace_overlap(self.prev_subspaces[name], U_k)
                        overlaps[name] = overlap

        # Update stored subspaces
        self.prev_subspaces = current_subspaces

        return overlaps

    def reset(self):
        """Clear stored subspaces."""
        self.prev_subspaces = {}


# =============================================================================
# GRADIENT SIMILARITY METRICS
# =============================================================================

def compute_gradient_cosine_similarity(grad1: torch.Tensor, grad2: torch.Tensor) -> float:
    """
    Compute cosine similarity between two gradient tensors.
    """
    g1_flat = grad1.flatten().float()
    g2_flat = grad2.flatten().float()

    norm1 = g1_flat.norm()
    norm2 = g2_flat.norm()

    if norm1 < 1e-10 or norm2 < 1e-10:
        return 0.0

    return (g1_flat @ g2_flat / (norm1 * norm2)).item()


class GradientSimilarityTracker:
    """
    Track gradient similarity across layers and consecutive steps.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.prev_gradients = {}  # Store previous step's gradients per layer

    def compute_cross_layer_similarity(self) -> np.ndarray:
        """
        Compute pairwise similarity matrix across all 2D layers.

        For layers of the same shape: uses cosine similarity of flattened gradients.
        For layers of different shapes: uses correlation of gradient statistics
        (sign pattern histogram correlation as a shape-agnostic measure).

        Returns:
            Similarity matrix (num_layers x num_layers)
            Layer names list
        """
        layer_grads = []
        layer_names = []
        layer_shapes = []

        for name, param in self.model.named_parameters():
            if param.grad is not None and param.dim() == 2:
                layer_grads.append(param.grad.float())
                layer_names.append(name)
                layer_shapes.append(param.grad.shape)

        if len(layer_grads) < 2:
            return np.array([]), layer_names

        n = len(layer_grads)
        sim_matrix = np.zeros((n, n))

        # Precompute gradient statistics for shape-agnostic comparison
        # Use histogram of normalized gradient values as a "fingerprint"
        def gradient_fingerprint(grad, num_bins=50):
            """Compute normalized histogram of gradient values."""
            flat = grad.flatten()
            # Normalize to zero mean, unit variance
            flat = (flat - flat.mean()) / (flat.std() + 1e-8)
            # Clamp outliers and compute histogram
            flat = torch.clamp(flat, -3, 3)
            hist = torch.histc(flat, bins=num_bins, min=-3, max=3)
            hist = hist / (hist.sum() + 1e-8)  # Normalize to probability
            return hist.cpu().numpy()

        fingerprints = [gradient_fingerprint(g) for g in layer_grads]

        for i in range(n):
            for j in range(n):
                if i == j:
                    sim_matrix[i, j] = 1.0
                elif layer_shapes[i] == layer_shapes[j]:
                    # Same shape: use cosine similarity
                    g1 = layer_grads[i].flatten()
                    g2 = layer_grads[j].flatten()
                    norm1, norm2 = g1.norm(), g2.norm()
                    if norm1 > 1e-10 and norm2 > 1e-10:
                        sim_matrix[i, j] = (g1 @ g2 / (norm1 * norm2)).item()
                    else:
                        sim_matrix[i, j] = 0.0
                else:
                    # Different shapes: use histogram correlation
                    # Pearson correlation between fingerprints
                    fp1, fp2 = fingerprints[i], fingerprints[j]
                    fp1_centered = fp1 - fp1.mean()
                    fp2_centered = fp2 - fp2.mean()
                    num = np.sum(fp1_centered * fp2_centered)
                    denom = np.sqrt(np.sum(fp1_centered**2) * np.sum(fp2_centered**2) + 1e-10)
                    sim_matrix[i, j] = num / denom

        return sim_matrix, layer_names

    def compute_temporal_similarity(self) -> Dict[str, float]:
        """
        Compute gradient similarity between current and previous step for each layer.

        Returns:
            Dictionary mapping layer names to temporal similarity scores
        """
        temporal_sims = {}
        current_gradients = {}

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                current_gradients[name] = param.grad.clone()

                if name in self.prev_gradients:
                    sim = compute_gradient_cosine_similarity(
                        self.prev_gradients[name],
                        param.grad
                    )
                    temporal_sims[name] = sim

        # Update stored gradients
        self.prev_gradients = current_gradients

        return temporal_sims

    def reset(self):
        """Clear stored gradients."""
        self.prev_gradients = {}


# =============================================================================
# COMPREHENSIVE GEOMETRY TRACKER
# =============================================================================

class OptimizationGeometryTracker:
    """
    Unified tracker for all optimization geometry metrics.

    Tracks:
    - Hessian eigenvalue spectrum (via Lanczos)
    - Per-layer gradient norms
    - Weight matrix effective ranks
    - Gradient subspace overlap (per layer, across steps)
    - Gradient similarity (across layers and steps)
    """

    def __init__(self, model: nn.Module, top_k_hessian: int = 20, top_k_subspace: int = 10):
        self.model = model
        self.top_k_hessian = top_k_hessian
        self.top_k_subspace = top_k_subspace

        self.subspace_tracker = GradientSubspaceTracker(model, top_k_subspace)
        self.similarity_tracker = GradientSimilarityTracker(model)

        # History for analysis
        self.history = defaultdict(list)

    def compute_all_metrics(
        self,
        data_loader,
        device: str = 'cuda',
        compute_hessian: bool = True,
        lanczos_iterations: int = 30
    ) -> Dict:
        """
        Compute all geometry metrics for the current training state.

        Returns:
            Dictionary containing all computed metrics
        """
        metrics = {}

        # 1. Hessian eigenvalue spectrum
        if compute_hessian:
            eigenvalues, T_matrix = lanczos_iteration(
                self.model, data_loader,
                k=self.top_k_hessian,
                num_iterations=lanczos_iterations,
                device=device
            )
            metrics['hessian_eigenvalues'] = eigenvalues
            metrics['lambda_max'] = eigenvalues[0] if len(eigenvalues) > 0 else 0.0
            metrics['lambda_min_top_k'] = eigenvalues[-1] if len(eigenvalues) > 0 else 0.0
            metrics['spectral_gap'] = (eigenvalues[0] - eigenvalues[1]) if len(eigenvalues) > 1 else 0.0

            # Spectral density
            if len(eigenvalues) > 5:
                bin_centers, density = compute_spectral_density(eigenvalues)
                metrics['spectral_density'] = (bin_centers, density)

        # 2. Per-layer gradient norms
        grad_norms = compute_layer_gradient_norms(self.model)
        metrics['layer_grad_norms'] = grad_norms
        metrics['total_grad_norm'] = sum(grad_norms.values()) ** 0.5 if grad_norms else 0.0

        # 3. Weight matrix effective ranks
        effective_ranks = compute_layer_effective_ranks(self.model)
        metrics['layer_effective_ranks'] = effective_ranks
        metrics['avg_effective_rank'] = np.mean(list(effective_ranks.values())) if effective_ranks else 0.0

        # 4. Gradient subspace overlap (temporal)
        subspace_overlaps = self.subspace_tracker.update()
        metrics['layer_subspace_overlaps'] = subspace_overlaps
        metrics['avg_subspace_overlap'] = np.mean(list(subspace_overlaps.values())) if subspace_overlaps else 1.0

        # 5. Cross-layer gradient similarity
        cross_layer_sim, layer_names = self.similarity_tracker.compute_cross_layer_similarity()
        metrics['cross_layer_similarity'] = cross_layer_sim
        metrics['cross_layer_names'] = layer_names

        # 6. Temporal gradient similarity
        temporal_sims = self.similarity_tracker.compute_temporal_similarity()
        metrics['layer_temporal_similarity'] = temporal_sims
        metrics['avg_temporal_similarity'] = np.mean(list(temporal_sims.values())) if temporal_sims else 1.0

        return metrics

    def get_layer_summary(self) -> Dict[str, List[str]]:
        """
        Get summary of tracked layers by category.
        """
        layers_2d = []
        layers_1d = []
        layers_other = []

        for name, param in self.model.named_parameters():
            if param.dim() == 2:
                layers_2d.append(name)
            elif param.dim() == 1:
                layers_1d.append(name)
            else:
                layers_other.append(name)

        return {
            '2D_layers': layers_2d,
            '1D_layers': layers_1d,
            'other_layers': layers_other
        }

    def reset(self):
        """Reset all trackers."""
        self.subspace_tracker.reset()
        self.similarity_tracker.reset()
        self.history = defaultdict(list)


def categorize_layers(model: nn.Module) -> Dict[str, List[str]]:
    """
    Categorize model layers by type (attention, mlp, embedding, etc.)

    Useful for per-category analysis.
    """
    categories = defaultdict(list)

    for name, param in model.named_parameters():
        if 'attn' in name or 'c_q' in name or 'c_k' in name or 'c_v' in name:
            categories['attention'].append(name)
        elif 'mlp' in name or 'c_fc' in name or 'c_proj' in name:
            categories['mlp'].append(name)
        elif 'wte' in name or 'embed' in name:
            categories['embedding'].append(name)
        elif 'lm_head' in name or 'head' in name:
            categories['head'].append(name)
        else:
            categories['other'].append(name)

    return dict(categories)
