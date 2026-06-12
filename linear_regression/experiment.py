"""
Linear Regression Experiment: Adam vs Muon on Low-Rank vs Full-Rank Problems

Goal: Evaluate how Adam and Muon converge to different spectral solutions
depending on the problem structure (over/under-parameterized).

Hypothesis:
- Adam: May prefer sparse/low-rank solutions
- Muon: May prefer dense/full-rank solutions due to orthogonalization

Usage:
    python experiment.py
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
import os
import json
import sys
from dataclasses import dataclass

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from optim.adam_ns import AdamNS


# ============================================================================
# Spectral Analysis Utilities
# ============================================================================

def compute_spectral_entropy(matrix: torch.Tensor, eps: float = 1e-8) -> float:
    """
    Compute spectral entropy: H = -sum(p_i * log(p_i))
    where p_i = sigma_i^2 / sum(sigma_j^2)

    Returns normalized entropy in [0, 1].
    """
    if matrix.dim() != 2:
        return 0.0

    sv = torch.linalg.svdvals(matrix.float().detach())
    sv_sq = sv ** 2
    total = sv_sq.sum()

    if total < eps:
        return 0.0

    p = sv_sq / total
    p = p[p > eps]
    entropy = -torch.sum(p * torch.log(p)).item()

    # Normalize by log(rank) to get [0, 1]
    rank = min(matrix.shape)
    if rank > 1:
        entropy = entropy / np.log(rank)

    return np.clip(entropy, 0, 1)


def compute_stable_rank(matrix: torch.Tensor, eps: float = 1e-8) -> float:
    """Compute stable rank: ||M||_F^2 / sigma_max^2"""
    if matrix.dim() != 2:
        return 0.0

    sv = torch.linalg.svdvals(matrix.float().detach())
    sigma_max = sv[0].item()
    frob_sq = (sv ** 2).sum().item()

    if sigma_max < eps:
        return 0.0

    return frob_sq / (sigma_max ** 2)


def compute_normalized_stable_rank(matrix: torch.Tensor, eps: float = 1e-8) -> float:
    """Stable rank normalized to [0, 1] by dividing by min dimension."""
    sr = compute_stable_rank(matrix, eps)
    min_dim = min(matrix.shape)
    return sr / min_dim if min_dim > 0 else 0.0


def get_singular_values(matrix: torch.Tensor) -> np.ndarray:
    """Get singular values as numpy array."""
    return torch.linalg.svdvals(matrix.float().detach()).cpu().numpy()


def analyze_null_space_component(W: torch.Tensor, A: torch.Tensor, eps: float = 1e-5) -> Dict[str, float]:
    """
    Decompose W into row-space and null-space components w.r.t. A.

    A has shape (n_samples, input_dim).
    W has shape (input_dim, output_dim).

    The row space of A spans the "data-relevant" directions.
    The null space of A is the orthogonal complement.

    Returns dict with:
        - rank_A: numerical rank of A
        - null_dim: dimension of null space
        - row_energy_frac: fraction of ||W||^2 in row space
        - null_energy_frac: fraction of ||W||^2 in null space
        - row_norm: ||W_row||_F
        - null_norm: ||W_null||_F
    """
    W = W.float().detach()
    A = A.float().detach()

    # SVD of A to get its row space
    U, S, Vh = torch.linalg.svd(A, full_matrices=True)
    # Vh is (input_dim, input_dim), rows are right singular vectors

    rank_A = (S > eps * S[0]).sum().item()  # numerical rank

    V_row = Vh[:rank_A]       # (rank_A, input_dim) — row space basis
    V_null = Vh[rank_A:]      # (input_dim - rank_A, input_dim) — null space basis

    # Project W columns onto each subspace
    # W is (input_dim, output_dim), project each output dimension
    W_row_proj = V_row.T @ (V_row @ W)    # row space component
    W_null_proj = V_null.T @ (V_null @ W) if V_null.shape[0] > 0 else torch.zeros_like(W)

    row_energy = (W_row_proj ** 2).sum().item()
    null_energy = (W_null_proj ** 2).sum().item()
    total_energy = (W ** 2).sum().item()

    if total_energy < 1e-10:
        return {
            'rank_A': int(rank_A),
            'null_dim': int(Vh.shape[0] - rank_A),
            'row_energy_frac': 0.0,
            'null_energy_frac': 0.0,
            'row_norm': 0.0,
            'null_norm': 0.0,
        }

    return {
        'rank_A': int(rank_A),
        'null_dim': int(Vh.shape[0] - rank_A),
        'row_energy_frac': row_energy / total_energy,
        'null_energy_frac': null_energy / total_energy,
        'row_norm': row_energy ** 0.5,
        'null_norm': null_energy ** 0.5,
    }


# ============================================================================
# JSON Serialization Utilities
# ============================================================================

def metrics_to_dict(metrics: 'TrainingMetrics') -> Dict:
    """Convert TrainingMetrics to JSON-serializable dictionary."""
    return {
        'steps': metrics.steps,
        'losses': metrics.losses,
        'weight_spectral_entropy': metrics.weight_spectral_entropy,
        'grad_spectral_entropy': metrics.grad_spectral_entropy,
        'update_spectral_entropy': metrics.update_spectral_entropy,
        'weight_stable_rank': metrics.weight_stable_rank,
        'grad_stable_rank': metrics.grad_stable_rank,
        'update_stable_rank': metrics.update_stable_rank,
        'weight_singular_values': [sv.tolist() for sv in metrics.weight_singular_values],
        'grad_singular_values': [sv.tolist() for sv in metrics.grad_singular_values],
        'update_singular_values': [sv.tolist() for sv in metrics.update_singular_values],
        'lr_history': metrics.lr_history,
        'null_energy_frac': metrics.null_energy_frac,
        'row_energy_frac': metrics.row_energy_frac,
        'null_norm': metrics.null_norm,
        'row_norm': metrics.row_norm,
    }


def save_metrics_json(metrics: 'TrainingMetrics', filepath: str, config: Optional[Dict] = None):
    """Save TrainingMetrics to JSON file."""
    data = metrics_to_dict(metrics)
    if config is not None:
        data['config'] = config
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  Saved metrics to {filepath}")


def save_metrics_jsonl(metrics: 'TrainingMetrics', filepath: str, config: Optional[Dict] = None):
    """
    Save TrainingMetrics to JSONL file (one JSON object per logged step).
    Each line contains all metrics for that step.
    """
    with open(filepath, 'w') as f:
        # Write config as first line if provided
        if config is not None:
            f.write(json.dumps({'type': 'config', **config}) + '\n')

        # Write one line per logged step with all metrics
        n_steps = len(metrics.steps)
        for i in range(n_steps):
            entry = {
                'type': 'metrics',
                'step': metrics.steps[i],
                'loss': metrics.losses[i],
                'weight_spectral_entropy': metrics.weight_spectral_entropy[i],
                'grad_spectral_entropy': metrics.grad_spectral_entropy[i],
                'update_spectral_entropy': metrics.update_spectral_entropy[i],
                'weight_stable_rank': metrics.weight_stable_rank[i],
                'grad_stable_rank': metrics.grad_stable_rank[i],
                'update_stable_rank': metrics.update_stable_rank[i],
                'weight_singular_values': metrics.weight_singular_values[i].tolist(),
                'grad_singular_values': metrics.grad_singular_values[i].tolist(),
                'update_singular_values': metrics.update_singular_values[i].tolist(),
                'lr': metrics.lr_history[i],
                'null_energy_frac': metrics.null_energy_frac[i],
                'row_energy_frac': metrics.row_energy_frac[i],
                'null_norm': metrics.null_norm[i],
                'row_norm': metrics.row_norm[i],
            }
            f.write(json.dumps(entry) + '\n')

    print(f"  Saved metrics to {filepath}")


def load_metrics_json(filepath: str) -> Dict:
    """Load metrics from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    # Convert singular values back to numpy arrays
    for key in ['weight_singular_values', 'grad_singular_values', 'update_singular_values']:
        if key in data:
            data[key] = [np.array(sv) for sv in data[key]]
    return data


# ============================================================================
# Problem Generation
# ============================================================================

@dataclass
class LinearProblem:
    """A linear regression problem Ax = b"""
    A: torch.Tensor  # (n_samples, input_dim)
    b: torch.Tensor  # (n_samples, output_dim)
    W_true: torch.Tensor  # True solution (input_dim, output_dim)
    name: str
    rank: int


def generate_low_rank_problem(
    n_samples: int = 1000,
    input_dim: int = 128,
    output_dim: int = 64,
    true_rank: int = 8,
    noise_std: float = 0.01,
    device: str = 'cuda',
) -> LinearProblem:
    """
    Generate a low-rank linear regression problem.

    The true solution W_true has rank = true_rank << min(input_dim, output_dim).
    This represents an overparameterized problem with implicit low-rank structure.
    """
    # Create low-rank ground truth: W = U @ V.T
    U = torch.randn(input_dim, true_rank, device=device)
    V = torch.randn(output_dim, true_rank, device=device)
    W_true = U @ V.T  # Shape: (input_dim, output_dim), rank = true_rank

    # Generate input data
    A = torch.randn(n_samples, input_dim, device=device)

    # Generate targets with noise
    b = A @ W_true + noise_std * torch.randn(n_samples, output_dim, device=device)

    return LinearProblem(A, b, W_true, f"low_rank_{true_rank}", true_rank)


def generate_full_rank_problem(
    n_samples: int = 1000,
    input_dim: int = 128,
    output_dim: int = 64,
    noise_std: float = 0.01,
    device: str = 'cuda',
) -> LinearProblem:
    """
    Generate a full-rank linear regression problem.

    The true solution W_true has rank = min(input_dim, output_dim).
    """
    # Full-rank ground truth with well-conditioned spectrum
    W_true = torch.randn(input_dim, output_dim, device=device)

    # Make it well-conditioned by adding identity-like structure
    min_dim = min(input_dim, output_dim)
    scale = 0.5
    W_true[:min_dim, :min_dim] += scale * torch.eye(min_dim, device=device)

    # Generate input data
    A = torch.randn(n_samples, input_dim, device=device)

    # Generate targets
    b = A @ W_true + noise_std * torch.randn(n_samples, output_dim, device=device)

    true_rank = min(input_dim, output_dim)
    return LinearProblem(A, b, W_true, f"full_rank_{true_rank}", true_rank)


# ============================================================================
# Linear Model
# ============================================================================

class LinearModel(nn.Module):
    """Simple linear model W for regression: y = x @ W"""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)
        # Small initialization
        nn.init.normal_(self.linear.weight, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    @property
    def W(self) -> torch.Tensor:
        return self.linear.weight.T  # Return (input_dim, output_dim) shape


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def create_model_with_init(
    input_dim: int,
    output_dim: int,
    device: str,
    init_state: Optional[Dict] = None,
    seed: Optional[int] = None,
) -> LinearModel:
    """Create a LinearModel, optionally loading from saved initial state."""
    if seed is not None:
        set_seed(seed)

    model = LinearModel(input_dim, output_dim).to(device)

    if init_state is not None:
        model.load_state_dict(init_state)

    return model


# ============================================================================
# Training Loop with Spectral Tracking
# ============================================================================

@dataclass
class TrainingMetrics:
    """Container for training metrics over time."""
    steps: List[int]
    losses: List[float]
    weight_spectral_entropy: List[float]
    grad_spectral_entropy: List[float]
    update_spectral_entropy: List[float]
    weight_stable_rank: List[float]
    grad_stable_rank: List[float]
    update_stable_rank: List[float]
    weight_singular_values: List[np.ndarray]
    grad_singular_values: List[np.ndarray]
    update_singular_values: List[np.ndarray]
    lr_history: List[float]
    # Null space analysis
    null_energy_frac: List[float]
    row_energy_frac: List[float]
    null_norm: List[float]
    row_norm: List[float]


def train(
    model: LinearModel,
    problem: LinearProblem,
    optimizer: torch.optim.Optimizer,
    optimizer_name: str = "optimizer",
    n_steps: int = 2000,
    log_every: int = 20,
    print_every: int = 100,
    batch_size: int = 256,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
) -> TrainingMetrics:
    """Train model and track spectral properties."""

    metrics = TrainingMetrics(
        steps=[], losses=[],
        weight_spectral_entropy=[], grad_spectral_entropy=[], update_spectral_entropy=[],
        weight_stable_rank=[], grad_stable_rank=[], update_stable_rank=[],
        weight_singular_values=[], grad_singular_values=[], update_singular_values=[],
        lr_history=[],
        null_energy_frac=[], row_energy_frac=[], null_norm=[], row_norm=[],
    )

    n_samples = problem.A.shape[0]
    device = problem.A.device

    # Store previous weights for computing updates
    W_prev = model.W.data.clone()

    for step in range(n_steps):
        # Sample batch
        idx = torch.randint(0, n_samples, (batch_size,), device=device)
        x_batch = problem.A[idx]
        y_batch = problem.b[idx]

        # Forward pass
        optimizer.zero_grad()
        pred = model(x_batch)
        loss = torch.nn.functional.mse_loss(pred, y_batch)

        # Backward pass
        loss.backward()

        # Log metrics before optimizer step
        if step % log_every == 0:
            with torch.no_grad():
                # Get current learning rate
                lr = optimizer.param_groups[0]['lr']

                # Compute spectral metrics for weights
                W = model.W
                w_entropy = compute_spectral_entropy(W)
                w_stable_rank = compute_normalized_stable_rank(W)

                # Compute spectral metrics for gradients
                G = model.linear.weight.grad
                if G is not None:
                    g_entropy = compute_spectral_entropy(G)
                    g_stable_rank = compute_normalized_stable_rank(G)
                    g_sv = get_singular_values(G)
                else:
                    g_entropy = 0.0
                    g_stable_rank = 0.0
                    g_sv = np.zeros(min(W.shape))

                # Compute update (delta W from last step)
                delta_W = W - W_prev
                u_entropy = compute_spectral_entropy(delta_W)
                u_stable_rank = compute_normalized_stable_rank(delta_W)
                u_sv = get_singular_values(delta_W)

                # Compute null space decomposition
                null_info = analyze_null_space_component(W, problem.A)

                # Store metrics
                metrics.steps.append(step)
                metrics.losses.append(loss.item())
                metrics.weight_spectral_entropy.append(w_entropy)
                metrics.grad_spectral_entropy.append(g_entropy)
                metrics.update_spectral_entropy.append(u_entropy)
                metrics.weight_stable_rank.append(w_stable_rank)
                metrics.grad_stable_rank.append(g_stable_rank)
                metrics.update_stable_rank.append(u_stable_rank)
                metrics.weight_singular_values.append(get_singular_values(W))
                metrics.grad_singular_values.append(g_sv)
                metrics.update_singular_values.append(u_sv)
                metrics.lr_history.append(lr)
                metrics.null_energy_frac.append(null_info['null_energy_frac'])
                metrics.row_energy_frac.append(null_info['row_energy_frac'])
                metrics.null_norm.append(null_info['null_norm'])
                metrics.row_norm.append(null_info['row_norm'])

                # Print progress
                if step % print_every == 0:
                    print(
                        f"  [{optimizer_name}] step={step:5d} | "
                        f"loss={loss.item():.6f} | lr={lr:.6f} | "
                        f"W_entropy={w_entropy:.4f} | W_rank={w_stable_rank:.4f} | "
                        f"null_frac={null_info['null_energy_frac']:.4f} | row_frac={null_info['row_energy_frac']:.4f}"
                    )

        # Store weights before update
        W_prev = model.W.data.clone()

        # Optimizer step
        optimizer.step()

        # Scheduler step (if provided)
        if scheduler is not None:
            scheduler.step()

    return metrics


# ============================================================================
# Plotting
# ============================================================================

def plot_lr_sweep(
    lr_results: Dict[str, Dict[float, 'TrainingMetrics']],
    save_dir: str,
):
    """
    Plot LR sweep results with Adam in blue and Muon in red.

    Args:
        lr_results: Dict mapping optimizer name -> {lr: metrics}
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)

    # Color scheme: Adam=blue, Muon=red
    opt_colors = {'adam': '#1f77b4', 'muon': '#d62728', 'adamns': '#2ca02c'}

    # Create figure with 2x3 subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Plot 1: Loss curves
    ax = axes[0, 0]
    for opt_name, lr_dict in lr_results.items():
        color = opt_colors.get(opt_name.lower(), '#333333')
        for i, (lr, metrics) in enumerate(sorted(lr_dict.items())):
            alpha = 0.3 + 0.7 * (i / max(1, len(lr_dict) - 1))  # Vary alpha by LR
            ax.semilogy(metrics.steps, metrics.losses,
                       color=color, alpha=alpha, linewidth=1.5,
                       label=f'{opt_name.upper()} lr={lr}')
    ax.set_xlabel('Step')
    ax.set_ylabel('MSE Loss')
    ax.set_title('Training Loss (all LRs)')
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.3)

    # Plot 2: Final loss vs LR
    ax = axes[0, 1]
    for opt_name, lr_dict in lr_results.items():
        color = opt_colors.get(opt_name.lower(), '#333333')
        lrs = sorted(lr_dict.keys())
        final_losses = [lr_dict[lr].losses[-1] for lr in lrs]
        ax.semilogy(lrs, final_losses, 'o-', color=color, linewidth=2,
                   markersize=8, label=opt_name.upper())
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Final MSE Loss')
    ax.set_title('Final Loss vs Learning Rate')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Weight spectral entropy
    ax = axes[0, 2]
    for opt_name, lr_dict in lr_results.items():
        color = opt_colors.get(opt_name.lower(), '#333333')
        for i, (lr, metrics) in enumerate(sorted(lr_dict.items())):
            alpha = 0.3 + 0.7 * (i / max(1, len(lr_dict) - 1))
            ax.plot(metrics.steps, metrics.weight_spectral_entropy,
                   color=color, alpha=alpha, linewidth=1.5,
                   label=f'{opt_name.upper()} lr={lr}')
    ax.set_xlabel('Step')
    ax.set_ylabel('Spectral Entropy')
    ax.set_title('Weight Spectral Entropy')
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(True, alpha=0.3)

    # Plot 4: Weight stable rank
    ax = axes[1, 0]
    for opt_name, lr_dict in lr_results.items():
        color = opt_colors.get(opt_name.lower(), '#333333')
        for i, (lr, metrics) in enumerate(sorted(lr_dict.items())):
            alpha = 0.3 + 0.7 * (i / max(1, len(lr_dict) - 1))
            ax.plot(metrics.steps, metrics.weight_stable_rank,
                   color=color, alpha=alpha, linewidth=1.5,
                   label=f'{opt_name.upper()} lr={lr}')
    ax.set_xlabel('Step')
    ax.set_ylabel('Stable Rank')
    ax.set_title('Weight Stable Rank')
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(True, alpha=0.3)

    # Plot 5: Null space energy fraction
    ax = axes[1, 1]
    for opt_name, lr_dict in lr_results.items():
        color = opt_colors.get(opt_name.lower(), '#333333')
        for i, (lr, metrics) in enumerate(sorted(lr_dict.items())):
            alpha = 0.3 + 0.7 * (i / max(1, len(lr_dict) - 1))
            ax.plot(metrics.steps, metrics.null_energy_frac,
                   color=color, alpha=alpha, linewidth=1.5,
                   label=f'{opt_name.upper()} lr={lr}')
    ax.set_xlabel('Step')
    ax.set_ylabel('Null Space Energy Fraction')
    ax.set_title('Null Space Energy')
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.3)

    # Plot 6: Final spectra for best LR of each optimizer
    ax = axes[1, 2]
    for opt_name, lr_dict in lr_results.items():
        color = opt_colors.get(opt_name.lower(), '#333333')
        # Find best LR (lowest final loss)
        best_lr = min(lr_dict.keys(), key=lambda lr: lr_dict[lr].losses[-1])
        metrics = lr_dict[best_lr]
        sv = metrics.weight_singular_values[-1]
        sv_normalized = sv / sv[0] if sv[0] > 1e-10 else sv
        ax.semilogy(sv_normalized[:32], color=color, linewidth=2,
                   label=f'{opt_name.upper()} (best lr={best_lr})')
    ax.set_xlabel('Singular Value Index')
    ax.set_ylabel('Normalized SV')
    ax.set_title('Final Weight Spectrum (best LR)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'lr_sweep_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"LR sweep plot saved to {save_dir}/lr_sweep_comparison.png")


def plot_comparison(
    results: Dict[str, Dict[str, TrainingMetrics]],
    save_dir: str,
):
    """Plot comprehensive comparison of optimizers on different problems."""

    os.makedirs(save_dir, exist_ok=True)

    problems = list(results.keys())
    optimizers = list(results[problems[0]].keys())
    colors = {'adam': '#1f77b4', 'muon': '#ff7f0e', 'adamns': '#2ca02c'}

    # 1. Loss curves
    fig, axes = plt.subplots(1, len(problems), figsize=(5 * len(problems), 4))
    if len(problems) == 1:
        axes = [axes]

    for ax, prob in zip(axes, problems):
        for opt in optimizers:
            m = results[prob][opt]
            ax.semilogy(m.steps, m.losses, label=opt.upper(), color=colors[opt], linewidth=2)
        ax.set_xlabel('Step')
        ax.set_ylabel('MSE Loss')
        ax.set_title(f'{prob.replace("_", " ").title()}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'loss_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Spectral entropy comparison (weights, gradients, updates)
    fig, axes = plt.subplots(3, len(problems), figsize=(5 * len(problems), 10))
    if len(problems) == 1:
        axes = axes.reshape(-1, 1)

    titles = ['Weight Spectral Entropy', 'Gradient Spectral Entropy', 'Update Spectral Entropy']
    data_keys = ['weight_spectral_entropy', 'grad_spectral_entropy', 'update_spectral_entropy']

    for col, prob in enumerate(problems):
        for row, (title, key) in enumerate(zip(titles, data_keys)):
            ax = axes[row, col]
            for opt in optimizers:
                m = results[prob][opt]
                ax.plot(m.steps, getattr(m, key), label=opt.upper(), color=colors[opt], linewidth=2)
            ax.set_xlabel('Step')
            ax.set_ylabel('Normalized Entropy')
            ax.set_title(f'{title}\n({prob.replace("_", " ").title()})')
            ax.set_ylim(0, 1.1)
            ax.legend()
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'spectral_entropy.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 3. Stable rank comparison
    fig, axes = plt.subplots(3, len(problems), figsize=(5 * len(problems), 10))
    if len(problems) == 1:
        axes = axes.reshape(-1, 1)

    titles = ['Weight Stable Rank', 'Gradient Stable Rank', 'Update Stable Rank']
    data_keys = ['weight_stable_rank', 'grad_stable_rank', 'update_stable_rank']

    for col, prob in enumerate(problems):
        for row, (title, key) in enumerate(zip(titles, data_keys)):
            ax = axes[row, col]
            for opt in optimizers:
                m = results[prob][opt]
                ax.plot(m.steps, getattr(m, key), label=opt.upper(), color=colors[opt], linewidth=2)
            ax.set_xlabel('Step')
            ax.set_ylabel('Normalized Stable Rank')
            ax.set_title(f'{title}\n({prob.replace("_", " ").title()})')
            ax.set_ylim(0, 1.1)
            ax.legend()
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'stable_rank.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 4. Null space analysis
    fig, axes = plt.subplots(2, len(problems), figsize=(5 * len(problems), 8))
    if len(problems) == 1:
        axes = axes.reshape(-1, 1)

    for col, prob in enumerate(problems):
        # Null energy fraction over training
        ax = axes[0, col]
        for opt in optimizers:
            m = results[prob][opt]
            ax.plot(m.steps, m.null_energy_frac, label=opt.upper(), color=colors[opt], linewidth=2)
        ax.set_xlabel('Step')
        ax.set_ylabel('Null Space Energy Fraction')
        ax.set_title(f'Null Space Energy\n({prob.replace("_", " ").title()})')
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Row space energy fraction over training
        ax = axes[1, col]
        for opt in optimizers:
            m = results[prob][opt]
            ax.plot(m.steps, m.row_energy_frac, label=opt.upper(), color=colors[opt], linewidth=2)
        ax.set_xlabel('Step')
        ax.set_ylabel('Row Space Energy Fraction')
        ax.set_title(f'Row Space Energy\n({prob.replace("_", " ").title()})')
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'null_space_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 5. Singular value spectra at end of training (weights, gradients, updates)
    fig, axes = plt.subplots(3, len(problems), figsize=(5 * len(problems), 10))
    if len(problems) == 1:
        axes = axes.reshape(-1, 1)

    sv_data = [
        ('weight_singular_values', 'Final Weight Spectrum'),
        ('grad_singular_values', 'Final Gradient Spectrum'),
        ('update_singular_values', 'Final Update (dW) Spectrum'),
    ]

    for col, prob in enumerate(problems):
        for row, (sv_key, title) in enumerate(sv_data):
            ax = axes[row, col]
            for opt in optimizers:
                m = results[prob][opt]
                sv = getattr(m, sv_key)[-1]
                if sv[0] > 1e-10:
                    sv_normalized = sv / sv[0]
                else:
                    sv_normalized = sv
                ax.semilogy(sv_normalized, label=opt.upper(), color=colors[opt], linewidth=2)
            ax.set_xlabel('Singular Value Index')
            ax.set_ylabel('Normalized Singular Value')
            ax.set_title(f'{title}\n({prob.replace("_", " ").title()})')
            ax.legend()
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'singular_values.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 5. Combined summary plot
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Loss
    ax = axes[0, 0]
    for prob in problems:
        for opt in optimizers:
            m = results[prob][opt]
            linestyle = '-' if 'low' in prob else '--'
            ax.semilogy(m.steps, m.losses, label=f'{opt.upper()} ({prob.split("_")[0]})',
                       color=colors[opt], linestyle=linestyle, linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('MSE Loss')
    ax.set_title('Training Loss')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Weight spectral entropy
    ax = axes[0, 1]
    for prob in problems:
        for opt in optimizers:
            m = results[prob][opt]
            linestyle = '-' if 'low' in prob else '--'
            ax.plot(m.steps, m.weight_spectral_entropy,
                   label=f'{opt.upper()} ({prob.split("_")[0]})',
                   color=colors[opt], linestyle=linestyle, linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Normalized Entropy')
    ax.set_title('Weight Spectral Entropy')
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Weight stable rank
    ax = axes[1, 0]
    for prob in problems:
        for opt in optimizers:
            m = results[prob][opt]
            linestyle = '-' if 'low' in prob else '--'
            ax.plot(m.steps, m.weight_stable_rank,
                   label=f'{opt.upper()} ({prob.split("_")[0]})',
                   color=colors[opt], linestyle=linestyle, linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Normalized Stable Rank')
    ax.set_title('Weight Stable Rank')
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Final singular value spectra (weight)
    ax = axes[1, 1]
    for prob in problems:
        for opt in optimizers:
            m = results[prob][opt]
            sv = m.weight_singular_values[-1]
            sv_normalized = sv / sv[0] if sv[0] > 0 else sv
            linestyle = '-' if 'low' in prob else '--'
            ax.semilogy(sv_normalized[:32],  # First 32 for clarity
                       label=f'{opt.upper()} ({prob.split("_")[0]})',
                       color=colors[opt], linestyle=linestyle, linewidth=2)
    ax.set_xlabel('Singular Value Index')
    ax.set_ylabel('Normalized SV')
    ax.set_title('Final Weight Spectrum (first 32)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'summary.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 6. Spectra comparison plot (W, G, dW side by side for final step)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))

    for row, prob in enumerate(problems):
        # Weight spectrum
        ax = axes[row, 0]
        for opt in optimizers:
            m = results[prob][opt]
            sv = m.weight_singular_values[-1]
            sv_normalized = sv / sv[0] if sv[0] > 1e-10 else sv
            ax.semilogy(sv_normalized, label=opt.upper(), color=colors[opt], linewidth=2)
        ax.set_xlabel('Index')
        ax.set_ylabel('Normalized SV')
        ax.set_title(f'Weight Spectrum\n({prob.replace("_", " ").title()})')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Gradient spectrum
        ax = axes[row, 1]
        for opt in optimizers:
            m = results[prob][opt]
            sv = m.grad_singular_values[-1]
            sv_normalized = sv / sv[0] if sv[0] > 1e-10 else sv
            ax.semilogy(sv_normalized, label=opt.upper(), color=colors[opt], linewidth=2)
        ax.set_xlabel('Index')
        ax.set_ylabel('Normalized SV')
        ax.set_title(f'Gradient Spectrum\n({prob.replace("_", " ").title()})')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Update spectrum
        ax = axes[row, 2]
        for opt in optimizers:
            m = results[prob][opt]
            sv = m.update_singular_values[-1]
            sv_normalized = sv / sv[0] if sv[0] > 1e-10 else sv
            ax.semilogy(sv_normalized, label=opt.upper(), color=colors[opt], linewidth=2)
        ax.set_xlabel('Index')
        ax.set_ylabel('Normalized SV')
        ax.set_title(f'Update (dW) Spectrum\n({prob.replace("_", " ").title()})')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'spectra_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Plots saved to {save_dir}")


# ============================================================================
# Hyperparameter Sweep
# ============================================================================

def run_lr_sweep(
    input_dim: int = 512,
    output_dim: int = 64,
    n_samples: int = 200,
    low_rank: int = 8,
    n_steps: int = 2000,
    batch_size: int = 256,
    adam_lrs: List[float] = None,
    muon_lrs: List[float] = None,
    log_every: int = 20,
    print_every: int = 100,
    use_muon_schedule: bool = False,
    device: str = 'cuda',
    save_dir: str = 'linear_regression/lr_sweep_results',
    seed: int = 42,
):
    """
    Run learning rate sweep for both Adam and Muon.

    Uses same model initialization across all runs for fair comparison.
    Saves metrics to JSON and generates combined comparison plots.
    """
    if adam_lrs is None:
        adam_lrs = [0.001, 0.005, 0.01, 0.02, 0.05]
    if muon_lrs is None:
        muon_lrs = [0.005, 0.01, 0.02, 0.05, 0.1]

    os.makedirs(save_dir, exist_ok=True)

    print("=" * 60)
    print("Learning Rate Hyperparameter Sweep")
    print("=" * 60)
    print(f"Input dim: {input_dim}, Output dim: {output_dim}")
    print(f"N samples: {n_samples}, Low rank: {low_rank}")
    print(f"Adam LRs: {adam_lrs}")
    print(f"Muon LRs: {muon_lrs}")
    print(f"Muon LR schedule: {use_muon_schedule}")
    print(f"Seed: {seed}")
    print("=" * 60)

    # Generate problem with fixed seed
    set_seed(seed)
    problem = generate_low_rank_problem(
        n_samples=n_samples,
        input_dim=input_dim,
        output_dim=output_dim,
        true_rank=low_rank,
        device=device,
    )

    # Create initial model state (same for all runs)
    set_seed(seed)
    init_model = LinearModel(input_dim, output_dim).to(device)
    init_state = init_model.state_dict()

    # Store all results
    lr_results = {'adam': {}, 'muon': {}}

    # Config for JSON saving
    base_config = {
        'input_dim': input_dim,
        'output_dim': output_dim,
        'n_samples': n_samples,
        'low_rank': low_rank,
        'n_steps': n_steps,
        'batch_size': batch_size,
        'seed': seed,
    }

    # Run Adam sweeps
    print("\n--- Adam Learning Rate Sweep ---")
    for lr in adam_lrs:
        print(f"\nAdam lr={lr}")
        model = create_model_with_init(input_dim, output_dim, device, init_state)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        metrics = train(
            model, problem, optimizer,
            optimizer_name=f"Adam(lr={lr})",
            n_steps=n_steps,
            log_every=log_every,
            print_every=print_every,
            batch_size=batch_size,
        )
        lr_results['adam'][lr] = metrics

        # Save to JSONL
        config = {**base_config, 'optimizer': 'adam', 'lr': lr}
        save_metrics_jsonl(metrics, os.path.join(save_dir, f'metrics_adam_lr_{lr}.jsonl'), config)

    # Run Muon sweeps
    print("\n--- Muon Learning Rate Sweep ---")
    for lr in muon_lrs:
        print(f"\nMuon lr={lr}")
        model = create_model_with_init(input_dim, output_dim, device, init_state)
        optimizer = torch.optim.Muon(model.parameters(), lr=lr, momentum=0.95)

        scheduler = None
        if use_muon_schedule:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps)

        metrics = train(
            model, problem, optimizer,
            optimizer_name=f"Muon(lr={lr})",
            n_steps=n_steps,
            log_every=log_every,
            print_every=print_every,
            batch_size=batch_size,
            scheduler=scheduler,
        )
        lr_results['muon'][lr] = metrics

        # Save to JSONL
        config = {**base_config, 'optimizer': 'muon', 'lr': lr, 'use_schedule': use_muon_schedule}
        save_metrics_jsonl(metrics, os.path.join(save_dir, f'metrics_muon_lr_{lr}.jsonl'), config)

    # Generate comparison plots
    print("\nGenerating LR sweep comparison plots...")
    plot_lr_sweep(lr_results, save_dir)

    # Print summary
    print("\n" + "=" * 60)
    print("LR Sweep Summary")
    print("=" * 60)

    print("\nAdam best LRs by final loss:")
    adam_sorted = sorted(lr_results['adam'].items(), key=lambda x: x[1].losses[-1])
    for lr, m in adam_sorted[:3]:
        print(f"  lr={lr}: final_loss={m.losses[-1]:.6f}, entropy={m.weight_spectral_entropy[-1]:.4f}")

    print("\nMuon best LRs by final loss:")
    muon_sorted = sorted(lr_results['muon'].items(), key=lambda x: x[1].losses[-1])
    for lr, m in muon_sorted[:3]:
        print(f"  lr={lr}: final_loss={m.losses[-1]:.6f}, entropy={m.weight_spectral_entropy[-1]:.4f}")

    return lr_results


# ============================================================================
# Main Experiment
# ============================================================================

def run_experiment(
    input_dim: int = 128,
    output_dim: int = 64,
    true_rank_low: int = 8,
    n_samples: int = 2000,
    n_steps: int = 3000,
    batch_size: int = 256,
    adam_lr: float = 0.01,
    muon_lr: float = 0.02,
    log_every: int = 20,
    print_every: int = 100,
    use_muon_schedule: bool = False,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    save_dir: str = 'results',
    seed: int = 42,
):
    """Run the full comparison experiment."""

    os.makedirs(save_dir, exist_ok=True)

    print(f"Running experiment on {device}")
    print(f"Input dim: {input_dim}, Output dim: {output_dim}")
    print(f"Low rank target: {true_rank_low}, Full rank target: {min(input_dim, output_dim)}")
    print(f"Seed: {seed}")
    print()

    # Set seed for reproducibility
    set_seed(seed)

    # Generate problems
    print("Generating problems...")
    problem_low_rank = generate_low_rank_problem(
        n_samples=n_samples,
        input_dim=input_dim,
        output_dim=output_dim,
        true_rank=true_rank_low,
        device=device,
    )

    problem_full_rank = generate_full_rank_problem(
        n_samples=n_samples,
        input_dim=input_dim,
        output_dim=output_dim,
        device=device,
    )

    problems = {
        'low_rank': problem_low_rank,
        'full_rank': problem_full_rank,
    }

    # Create initial model state (same for both optimizers)
    set_seed(seed)
    init_model = LinearModel(input_dim, output_dim).to(device)
    init_state = init_model.state_dict()

    # Config for JSON saving
    base_config = {
        'input_dim': input_dim,
        'output_dim': output_dim,
        'n_samples': n_samples,
        'true_rank_low': true_rank_low,
        'n_steps': n_steps,
        'batch_size': batch_size,
        'adam_lr': adam_lr,
        'muon_lr': muon_lr,
        'use_muon_schedule': use_muon_schedule,
        'seed': seed,
    }

    results = {}

    for prob_name, problem in problems.items():
        print(f"\n{'='*60}")
        print(f"Problem: {prob_name} (true rank = {problem.rank})")
        print(f"{'='*60}")

        results[prob_name] = {}

        # Train with Adam
        print(f"\nTraining with Adam (lr={adam_lr})...")
        model_adam = create_model_with_init(input_dim, output_dim, device, init_state)
        optimizer_adam = torch.optim.Adam(model_adam.parameters(), lr=adam_lr)
        metrics_adam = train(
            model_adam, problem, optimizer_adam,
            optimizer_name="Adam",
            n_steps=n_steps, log_every=log_every, print_every=print_every, batch_size=batch_size
        )
        results[prob_name]['adam'] = metrics_adam
        print(f"  Final loss: {metrics_adam.losses[-1]:.6f}")
        print(f"  Final weight entropy: {metrics_adam.weight_spectral_entropy[-1]:.4f}")
        print(f"  Final weight stable rank: {metrics_adam.weight_stable_rank[-1]:.4f}")
        print(f"  Final null space frac: {metrics_adam.null_energy_frac[-1]:.4f}")
        print(f"  Final row space frac: {metrics_adam.row_energy_frac[-1]:.4f}")

        # Save Adam metrics to JSON
        config = {**base_config, 'problem': prob_name, 'optimizer': 'adam'}
        save_metrics_json(metrics_adam, os.path.join(save_dir, f'metrics_adam_{prob_name}.json'), config)

        # Train with Muon
        print(f"\nTraining with Muon (lr={muon_lr})...")
        model_muon = create_model_with_init(input_dim, output_dim, device, init_state)
        optimizer_muon = torch.optim.Muon(model_muon.parameters(), lr=muon_lr, momentum=0.95)

        scheduler = None
        if use_muon_schedule:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_muon, T_max=n_steps)

        metrics_muon = train(
            model_muon, problem, optimizer_muon,
            optimizer_name="Muon",
            n_steps=n_steps, log_every=log_every, print_every=print_every, batch_size=batch_size,
            scheduler=scheduler,
        )
        results[prob_name]['muon'] = metrics_muon
        print(f"  Final loss: {metrics_muon.losses[-1]:.6f}")
        print(f"  Final weight entropy: {metrics_muon.weight_spectral_entropy[-1]:.4f}")
        print(f"  Final weight stable rank: {metrics_muon.weight_stable_rank[-1]:.4f}")
        print(f"  Final null space frac: {metrics_muon.null_energy_frac[-1]:.4f}")
        print(f"  Final row space frac: {metrics_muon.row_energy_frac[-1]:.4f}")

        # Save Muon metrics to JSONL
        config = {**base_config, 'problem': prob_name, 'optimizer': 'muon'}
        save_metrics_jsonl(metrics_muon, os.path.join(save_dir, f'metrics_muon_{prob_name}.jsonl'), config)

        # Train with AdamNS (momentum orthogonalization mode)
        print(f"\nTraining with AdamNS (lr={adam_lr}, mode=momentum)...")
        model_adamns = create_model_with_init(input_dim, output_dim, device, init_state)
        optimizer_adamns = AdamNS(model_adamns.parameters(), lr=adam_lr, ns_mode='momentum', warmup_steps=100)
        metrics_adamns = train(
            model_adamns, problem, optimizer_adamns,
            optimizer_name="AdamNS",
            n_steps=n_steps, log_every=log_every, print_every=print_every, batch_size=batch_size
        )
        results[prob_name]['adamns'] = metrics_adamns
        print(f"  Final loss: {metrics_adamns.losses[-1]:.6f}")
        print(f"  Final weight entropy: {metrics_adamns.weight_spectral_entropy[-1]:.4f}")
        print(f"  Final weight stable rank: {metrics_adamns.weight_stable_rank[-1]:.4f}")
        print(f"  Final null space frac: {metrics_adamns.null_energy_frac[-1]:.4f}")
        print(f"  Final row space frac: {metrics_adamns.row_energy_frac[-1]:.4f}")

        # Save AdamNS metrics to JSONL
        config = {**base_config, 'problem': prob_name, 'optimizer': 'adamns'}
        save_metrics_jsonl(metrics_adamns, os.path.join(save_dir, f'metrics_adamns_{prob_name}.jsonl'), config)

        # Print null space comparison
        print(f"\n  Null Space Comparison for {prob_name}:")
        print(f"    Adam   — null: {metrics_adam.null_energy_frac[-1]:.4f}, row: {metrics_adam.row_energy_frac[-1]:.4f}")
        print(f"    Muon   — null: {metrics_muon.null_energy_frac[-1]:.4f}, row: {metrics_muon.row_energy_frac[-1]:.4f}")
        print(f"    AdamNS — null: {metrics_adamns.null_energy_frac[-1]:.4f}, row: {metrics_adamns.row_energy_frac[-1]:.4f}")

    # Plot results
    print(f"\n{'='*60}")
    print("Generating plots...")
    print(f"{'='*60}")
    plot_comparison(results, save_dir)

    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Linear Regression: Adam vs Muon')
    parser.add_argument('--input-dim', type=int, default=256, help='Input dimension')
    parser.add_argument('--output-dim', type=int, default=128, help='Output dimension')
    parser.add_argument('--low-rank', type=int, default=16, help='True rank for low-rank problem')
    parser.add_argument('--n-samples', type=int, default=2000, help='Number of training samples')
    parser.add_argument('--overparameterized', action='store_true',
                        help='Use overparameterized setting (input_dim=512, n_samples=200) for meaningful null space analysis')
    parser.add_argument('--n-steps', type=int, default=4000, help='Number of training steps')
    parser.add_argument('--batch-size', type=int, default=2000, help='Batch size')
    parser.add_argument('--adam-lr', type=float, default=0.01, help='Adam learning rate')
    parser.add_argument('--muon-lr', type=float, default=0.02, help='Muon learning rate')
    parser.add_argument('--log-every', type=int, default=20, help='Log metrics every N steps')
    parser.add_argument('--print-every', type=int, default=100, help='Print progress every N steps')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save-dir', type=str, default='linear_regression/results')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    # Hyperparameter sweep flags
    parser.add_argument('--hyps', action='store_true',
                        help='Run LR hyperparameter sweep instead of single experiment')
    parser.add_argument('--adam-lrs', type=float, nargs='+', default=[0.001, 0.005, 0.01, 0.02, 0.05],
                        help='Adam learning rates to sweep (for --hyps mode)')
    parser.add_argument('--muon-lrs', type=float, nargs='+', default=[0.005, 0.01, 0.02, 0.05, 0.1],
                        help='Muon learning rates to sweep (for --hyps mode)')

    # LR schedule flag
    parser.add_argument('--muon-lr-schedule', action='store_true',
                        help='Use cosine annealing LR schedule for Muon (default: False)')

    args = parser.parse_args()

    # Apply overparameterized preset if requested
    input_dim = args.input_dim
    n_samples = args.n_samples
    if args.overparameterized:
        input_dim = 2048
        n_samples = 32
        print("Using OVERPARAMETERIZED setting for meaningful null space analysis:")
        print(f"  input_dim={input_dim}, n_samples={n_samples}")
        print(f"  Expected null space dimension: {input_dim - n_samples}")
        print()

    # Check if overparameterized
    if input_dim > n_samples:
        print(f"Note: System is overparameterized (input_dim={input_dim} > n_samples={n_samples})")
        print(f"  Null space dimension: {input_dim - n_samples}")
        print()
    else:
        print(f"Note: System is NOT overparameterized (input_dim={input_dim} <= n_samples={n_samples})")
        print(f"  Null space is trivial - both optimizers forced into row space")
        print()

    if args.hyps:
        # Run LR hyperparameter sweep
        run_lr_sweep(
            input_dim=input_dim,
            output_dim=args.output_dim,
            n_samples=n_samples,
            low_rank=args.low_rank,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            adam_lrs=args.adam_lrs,
            muon_lrs=args.muon_lrs,
            log_every=args.log_every,
            print_every=args.print_every,
            use_muon_schedule=args.muon_lr_schedule,
            device=args.device,
            save_dir=args.save_dir,
            seed=args.seed,
        )
    else:
        # Run single experiment
        run_experiment(
            input_dim=input_dim,
            output_dim=args.output_dim,
            true_rank_low=args.low_rank,
            n_samples=n_samples,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            adam_lr=args.adam_lr,
            muon_lr=args.muon_lr,
            log_every=args.log_every,
            print_every=args.print_every,
            use_muon_schedule=args.muon_lr_schedule,
            device=args.device,
            save_dir=args.save_dir,
            seed=args.seed,
        )
