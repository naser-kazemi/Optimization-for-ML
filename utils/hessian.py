"""
Hessian Spectrum Analysis Module

Computes Hessian spectral properties including:
- Largest eigenvalue (λ_max) via Power Iteration
- Trace estimation via Hutchinson's method
- Effective rank of the Hessian
- Full spectrum estimation via Lanczos iteration

These metrics characterize the loss landscape curvature and help ensure
the model converges to wide, generalizable minima rather than sharp, brittle ones.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict


def compute_hvp(model, data, targets, v, device='cuda'):
    """
    Compute Hessian-Vector Product using reverse-over-reverse autodiff.

    Args:
        model: PyTorch model
        data: Input tokens (B, T)
        targets: Target tokens (B, T)
        v: List of tensors (same shape as model parameters)
        device: Device string

    Returns:
        List of tensors representing H @ v
    """
    params = [p for p in model.parameters() if p.requires_grad]

    with torch.enable_grad(), \
         torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
        loss = model(data, targets=targets)
        grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)

        dot_product = sum([(g * v_).sum() for g, v_ in zip(grads, v)])
        hvp = torch.autograd.grad(dot_product, params, retain_graph=True)

    return hvp


def power_iteration_multi(model, data_loader, num_eigenvalues=5, num_iterations=20, device='cuda'):
    """
    Compute top-k eigenvalues using deflated Power Iteration.

    Args:
        model: PyTorch model
        data_loader: Data loader iterator
        num_eigenvalues: Number of top eigenvalues to compute
        num_iterations: Power iteration iterations per eigenvalue
        device: Device string

    Returns:
        List of eigenvalues (descending order), List of eigenvectors
    """
    params = [p for p in model.parameters() if p.requires_grad]
    x, y = next(data_loader)

    eigenvalues = []
    eigenvectors = []

    for k in range(num_eigenvalues):
        # Initialize random vector
        v = [torch.randn_like(p).to(device) for p in params]

        # Orthogonalize against previous eigenvectors
        for prev_v in eigenvectors:
            dot = sum([(v_i * pv_i).sum() for v_i, pv_i in zip(v, prev_v)])
            v = [v_i - dot * pv_i for v_i, pv_i in zip(v, prev_v)]

        # Normalize
        norm_v = torch.sqrt(sum([torch.sum(v_**2) for v_ in v]))
        v = [v_ / (norm_v + 1e-10) for v_ in v]

        # Power iteration
        for _ in range(num_iterations):
            hvp = compute_hvp(model, x, y, v, device)

            # Orthogonalize against previous eigenvectors
            for prev_v in eigenvectors:
                dot = sum([(h_i * pv_i).sum() for h_i, pv_i in zip(hvp, prev_v)])
                hvp = tuple(h_i - dot * pv_i for h_i, pv_i in zip(hvp, prev_v))

            norm_hvp = torch.sqrt(sum([torch.sum(h_**2) for h_ in hvp]))
            if norm_hvp.item() < 1e-10:
                break
            v = [h_ / norm_hvp for h_ in hvp]

        # Compute Rayleigh quotient
        hvp_final = compute_hvp(model, x, y, v, device)
        lambda_k = sum([(v_i * h_i).sum() for v_i, h_i in zip(v, hvp_final)]).item()

        eigenvalues.append(lambda_k)
        eigenvectors.append(v)

    return eigenvalues, eigenvectors


def hutchinson_trace_estimator(model, data_loader, num_samples=10, device='cuda'):
    """
    Estimate Hessian trace using Hutchinson's stochastic trace estimator.

    tr(H) ≈ (1/n) Σ v_i^T H v_i, where v_i are random Rademacher vectors

    Args:
        model: PyTorch model
        data_loader: Data loader iterator
        num_samples: Number of random vectors for estimation
        device: Device string

    Returns:
        Estimated trace value
    """
    params = [p for p in model.parameters() if p.requires_grad]
    x, y = next(data_loader)

    trace_estimates = []

    for _ in range(num_samples):
        # Rademacher random vector (±1 with equal probability)
        v = [torch.randint(0, 2, p.shape, device=device).float() * 2 - 1 for p in params]

        hvp = compute_hvp(model, x, y, v, device)

        # v^T H v
        trace_sample = sum([(v_i * h_i).sum() for v_i, h_i in zip(v, hvp)]).item()
        trace_estimates.append(trace_sample)

    return np.mean(trace_estimates), np.std(trace_estimates)


def lanczos_algorithm(model, data_loader, num_iterations=30, device='cuda'):
    """
    Lanczos algorithm for tridiagonalization of the Hessian.

    Returns alpha (diagonal) and beta (off-diagonal) of tridiagonal matrix,
    from which eigenvalues can be computed.

    Args:
        model: PyTorch model
        data_loader: Data loader iterator
        num_iterations: Number of Lanczos iterations
        device: Device string

    Returns:
        alpha: Diagonal elements
        beta: Off-diagonal elements
        eigenvalues: Eigenvalues of tridiagonal matrix
    """
    params = [p for p in model.parameters() if p.requires_grad]
    x, y = next(data_loader)

    # Initialize random starting vector
    v = [torch.randn_like(p).to(device) for p in params]
    norm_v = torch.sqrt(sum([torch.sum(v_**2) for v_ in v]))
    v = [v_ / (norm_v + 1e-10) for v_ in v]

    v_prev = [torch.zeros_like(p) for p in params]

    alpha = []
    beta = [0.0]

    for j in range(num_iterations):
        # w = H @ v_j
        w = compute_hvp(model, x, y, v, device)

        # alpha_j = v_j^T @ w
        alpha_j = sum([(v_i * w_i).sum() for v_i, w_i in zip(v, w)]).item()
        alpha.append(alpha_j)

        # w = w - alpha_j * v_j - beta_j * v_{j-1}
        w = [w_i - alpha_j * v_i - beta[j] * vp_i for w_i, v_i, vp_i in zip(w, v, v_prev)]

        # beta_{j+1} = ||w||
        beta_next = torch.sqrt(sum([torch.sum(w_i**2) for w_i in w])).item()

        if beta_next < 1e-10:
            break

        beta.append(beta_next)

        # v_{j+1} = w / beta_{j+1}
        v_prev = v
        v = [w_i / beta_next for w_i in w]

    # Compute eigenvalues of tridiagonal matrix
    n = len(alpha)
    T = np.zeros((n, n))
    for i in range(n):
        T[i, i] = alpha[i]
        if i < n - 1:
            T[i, i+1] = beta[i+1]
            T[i+1, i] = beta[i+1]

    eigenvalues = np.linalg.eigvalsh(T)
    eigenvalues = sorted(eigenvalues, reverse=True)

    return np.array(alpha), np.array(beta[1:]), eigenvalues


def compute_hessian_spectrum(model, data_loader, num_eigenvalues=10,
                             lanczos_iterations=30, trace_samples=10,
                             device='cuda'):
    """
    Compute comprehensive Hessian spectrum metrics.

    Args:
        model: PyTorch model
        data_loader: Data loader iterator
        num_eigenvalues: Top-k eigenvalues via power iteration
        lanczos_iterations: Lanczos iterations for spectrum estimation
        trace_samples: Hutchinson trace estimation samples
        device: Device string

    Returns:
        Dictionary with:
        - lambda_max: Largest eigenvalue
        - lambda_min: Smallest eigenvalue (from Lanczos)
        - top_eigenvalues: List of top-k eigenvalues
        - trace: Estimated trace
        - trace_std: Standard deviation of trace estimate
        - effective_rank: tr(H)^2 / ||H||_F^2 (approximated)
        - lanczos_eigenvalues: Full spectrum from Lanczos
        - spectral_norm: ||H||_2 = λ_max
        - condition_number: λ_max / λ_min (if λ_min > 0)
    """
    model.eval()

    # Top eigenvalues via power iteration
    top_eigs, top_vecs = power_iteration_multi(
        model, data_loader,
        num_eigenvalues=min(num_eigenvalues, 5),  # Limit for efficiency
        num_iterations=15,
        device=device
    )

    lambda_max = top_eigs[0] if top_eigs else 0.0

    # Trace estimation
    trace_mean, trace_std = hutchinson_trace_estimator(
        model, data_loader,
        num_samples=trace_samples,
        device=device
    )

    # Lanczos for full spectrum
    alpha, beta, lanczos_eigs = lanczos_algorithm(
        model, data_loader,
        num_iterations=lanczos_iterations,
        device=device
    )

    # Lambda min from Lanczos (smallest eigenvalue)
    lambda_min = lanczos_eigs[-1] if len(lanczos_eigs) > 0 else 0.0

    # Effective rank approximation: tr(H) / λ_max
    # This gives a sense of how many directions have significant curvature
    effective_rank = trace_mean / (abs(lambda_max) + 1e-10) if lambda_max != 0 else 0.0

    # Frobenius norm approximation from eigenvalues
    frob_norm_sq = sum([e**2 for e in lanczos_eigs])

    # Condition number
    if lambda_min > 1e-10:
        condition_number = abs(lambda_max) / lambda_min
    else:
        condition_number = float('inf')

    # Spectral entropy from eigenvalue distribution
    pos_eigs = [e for e in lanczos_eigs if e > 1e-10]
    if pos_eigs:
        total = sum(pos_eigs)
        probs = [e / total for e in pos_eigs]
        spectral_entropy = -sum(p * np.log(p + 1e-10) for p in probs)
        max_entropy = np.log(len(pos_eigs))
        normalized_spectral_entropy = spectral_entropy / (max_entropy + 1e-10)
    else:
        spectral_entropy = 0.0
        normalized_spectral_entropy = 0.0

    model.train()

    return {
        'lambda_max': float(lambda_max),
        'lambda_min': float(lambda_min),
        'top_eigenvalues': [float(e) for e in top_eigs],
        'trace': float(trace_mean),
        'trace_std': float(trace_std),
        'effective_rank': float(effective_rank),
        'spectral_norm': float(abs(lambda_max)),
        'condition_number': float(condition_number) if not np.isinf(condition_number) else None,
        'frobenius_norm_sq': float(frob_norm_sq),
        'spectral_entropy': float(spectral_entropy),
        'normalized_spectral_entropy': float(normalized_spectral_entropy),
        'lanczos_eigenvalues': [float(e) for e in lanczos_eigs],
        'num_positive_eigenvalues': len([e for e in lanczos_eigs if e > 0]),
        'num_negative_eigenvalues': len([e for e in lanczos_eigs if e < 0]),
    }


class LayerWiseHessianTracker:
    """
    Tracks Hessian properties stratified by layer type (QKV, MLP, etc.)

    Uses parameter-wise Hessian diagonal approximation for efficient
    per-layer analysis.
    """

    LAYER_TYPES = {
        'attn_qkv': ['c_q', 'c_k', 'c_v', 'q_proj', 'k_proj', 'v_proj'],
        'attn_out': ['c_proj', 'o_proj', 'out_proj'],
        'mlp_up': ['c_fc', 'fc1', 'up_proj', 'gate_proj'],
        'mlp_down': ['mlp.c_proj', 'fc2', 'down_proj'],
        'embedding': ['wte', 'wpe', 'embed'],
        'lm_head': ['lm_head'],
    }

    def __init__(self, model):
        self.model = model
        self.layer_params = self._categorize_parameters()

    def _categorize_parameters(self):
        """Group parameters by layer type."""
        categories = defaultdict(list)

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            layer_type = 'other'
            name_lower = name.lower()

            # Check for specific layer types
            if 'attn' in name_lower:
                if any(pat in name_lower for pat in ['c_q', 'c_k', 'c_v', 'q_proj', 'k_proj', 'v_proj']):
                    layer_type = 'attn_qkv'
                elif 'c_proj' in name_lower or 'o_proj' in name_lower:
                    layer_type = 'attn_out'
            elif 'mlp' in name_lower:
                if 'c_fc' in name_lower or 'fc1' in name_lower or 'up_proj' in name_lower:
                    layer_type = 'mlp_up'
                elif 'c_proj' in name_lower or 'fc2' in name_lower or 'down_proj' in name_lower:
                    layer_type = 'mlp_down'
            elif any(pat in name_lower for pat in ['wte', 'wpe', 'embed']):
                layer_type = 'embedding'
            elif 'lm_head' in name_lower:
                layer_type = 'lm_head'

            categories[layer_type].append((name, param))

        return dict(categories)

    def compute_diagonal_hessian(self, data_loader, device='cuda'):
        """
        Compute diagonal Hessian approximation using gradient squared.

        This is Fisher information approximation: diag(H) ≈ E[g^2]
        """
        x, y = next(data_loader)

        self.model.zero_grad()
        with torch.enable_grad():
            loss = self.model(x, targets=y)
            loss.backward()

        layer_metrics = {}

        for layer_type, params in self.layer_params.items():
            grad_squared_sum = 0.0
            grad_abs_sum = 0.0
            num_params = 0

            for name, param in params:
                if param.grad is not None:
                    grad_sq = param.grad.detach() ** 2
                    grad_squared_sum += grad_sq.sum().item()
                    grad_abs_sum += param.grad.detach().abs().sum().item()
                    num_params += param.numel()

            if num_params > 0:
                layer_metrics[layer_type] = {
                    'mean_grad_squared': grad_squared_sum / num_params,
                    'mean_grad_abs': grad_abs_sum / num_params,
                    'total_grad_squared': grad_squared_sum,
                    'num_params': num_params,
                    'estimated_trace': grad_squared_sum,  # Fisher approx
                }

        return layer_metrics

    def compute_layer_curvature(self, data_loader, num_hvp_samples=5, device='cuda'):
        """
        Estimate per-layer curvature using random HVP projections.
        """
        x, y = next(data_loader)

        layer_curvatures = {}

        for layer_type, params in self.layer_params.items():
            curvature_samples = []

            for _ in range(num_hvp_samples):
                # Random vector for this layer's parameters
                v_layer = {}
                for name, param in params:
                    v_layer[name] = torch.randn_like(param)

                # Normalize
                norm = sum(v.pow(2).sum() for v in v_layer.values()).sqrt()
                for name in v_layer:
                    v_layer[name] = v_layer[name] / (norm + 1e-10)

                # Compute v^T H v using HVP
                self.model.zero_grad()
                with torch.enable_grad(), \
                     torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
                    loss = self.model(x, targets=y)

                    # Get gradients only for this layer
                    layer_params_list = [p for _, p in params]
                    grads = torch.autograd.grad(loss, layer_params_list, create_graph=True, retain_graph=True)

                    # Dot product g^T v
                    dot = sum((g * v_layer[name]).sum() for g, (name, _) in zip(grads, params))

                    # Second derivative
                    hvp = torch.autograd.grad(dot, layer_params_list, retain_graph=True)

                    # v^T H v
                    curvature = sum((v_layer[name] * h).sum() for h, (name, _) in zip(hvp, params)).item()
                    curvature_samples.append(curvature)

            layer_curvatures[layer_type] = {
                'mean_curvature': np.mean(curvature_samples),
                'std_curvature': np.std(curvature_samples),
                'max_curvature': max(curvature_samples),
                'min_curvature': min(curvature_samples),
            }

        return layer_curvatures


def compute_full_hessian_metrics(model, data_loader, device='cuda',
                                  num_top_eigenvalues=5,
                                  lanczos_iterations=20,
                                  trace_samples=5,
                                  compute_layerwise=True):
    """
    Comprehensive Hessian analysis combining global and layer-wise metrics.

    Args:
        model: PyTorch model
        data_loader: Data loader iterator
        device: Device string
        num_top_eigenvalues: Number of top eigenvalues to compute
        lanczos_iterations: Lanczos iterations
        trace_samples: Hutchinson trace samples
        compute_layerwise: Whether to compute per-layer metrics

    Returns:
        Dictionary with global and layer-wise Hessian metrics
    """
    # Global Hessian spectrum
    global_metrics = compute_hessian_spectrum(
        model, data_loader,
        num_eigenvalues=num_top_eigenvalues,
        lanczos_iterations=lanczos_iterations,
        trace_samples=trace_samples,
        device=device
    )

    result = {'global': global_metrics}

    # Layer-wise metrics
    if compute_layerwise:
        tracker = LayerWiseHessianTracker(model)

        # Diagonal Hessian approximation
        diag_metrics = tracker.compute_diagonal_hessian(data_loader, device)
        result['layer_diagonal'] = diag_metrics

        # Layer curvature (expensive, optional)
        # curvature_metrics = tracker.compute_layer_curvature(data_loader, num_hvp_samples=3, device=device)
        # result['layer_curvature'] = curvature_metrics

    return result
