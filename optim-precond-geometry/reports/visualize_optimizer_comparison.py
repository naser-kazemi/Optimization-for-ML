#!/usr/bin/env python3
"""
Optimizer Comparison Visualization

Generates comprehensive comparison plots across multiple optimizers for:
- Training/validation loss and perplexity
- Hessian eigenvalue (lambda_max) - curvature tracking
- All spectral metrics for W, G, delta_W, step_update matrices
- Singular value distributions

Metrics visualized:
- singular_values (raw, normalized, minmax)
- spectral_entropy, normalized_spectral_entropy
- stable_rank, normalized_stable_rank
- participation_ratio, normalized_participation_ratio
- effective_rank_90, effective_rank_99
- sigma_max, sigma_min, condition_number
- frobenius_norm, numerical_rank

Usage:
    python reports/visualize_optimizer_comparison.py --results-dir optimizer_comparison --save-dir optimizer_comparison/plots
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# Color palette for optimizers (colorblind-friendly)
OPTIMIZER_COLORS = {
    'adam': '#1f77b4',        # Blue
    'adamw': '#2ca02c',       # Green
    'muon': '#d62728',        # Red
    'adam_ns': '#9467bd',     # Purple
    'adam_ns_momentum': '#9467bd',  # Purple (same as adam_ns)
    'adam_ns_grad': '#ff7f0e',     # Orange
    'adam_ns_update': '#8c564b',   # Brown
    'sgd': '#7f7f7f',         # Gray
}

OPTIMIZER_LABELS = {
    'adam': 'Adam',
    'adamw': 'AdamW',
    'muon': 'Muon',
    'adam_ns': 'AdamNS (mom)',
    'adam_ns_momentum': 'AdamNS (mom)',
    'adam_ns_grad': 'AdamNS (grad)',
    'adam_ns_update': 'AdamNS (upd)',
    'sgd': 'SGD',
}

# All matrix types to visualize
MATRIX_TYPES = ['W', 'G', 'delta_W', 'step_update']

# All spectral metrics
SPECTRAL_METRICS = [
    ('mean_normalized_spectral_entropy', 'Norm. Spectral Entropy'),
    ('mean_spectral_entropy', 'Spectral Entropy'),
    ('mean_stable_rank', 'Stable Rank'),
    ('mean_normalized_stable_rank', 'Norm. Stable Rank'),
    ('mean_participation_ratio', 'Participation Ratio'),
    ('mean_normalized_participation_ratio', 'Norm. Participation Ratio'),
    ('mean_effective_rank_90', 'Effective Rank 90%'),
    ('mean_effective_rank_99', 'Effective Rank 99%'),
    ('mean_normalized_effective_rank_90', 'Norm. Eff. Rank 90%'),
    ('mean_normalized_effective_rank_99', 'Norm. Eff. Rank 99%'),
    ('mean_sigma_max', 'Sigma Max'),
    ('mean_sigma_min', 'Sigma Min'),
    ('mean_condition_number', 'Condition Number'),
    ('mean_frobenius_norm', 'Frobenius Norm'),
    ('mean_numerical_rank', 'Numerical Rank'),
]


def load_metrics_csv(filepath: str) -> pd.DataFrame:
    """Load metrics from CSV file."""
    return pd.read_csv(filepath)


def load_spectral_full(filepath: str) -> Dict[str, Any]:
    """Load full spectral data from JSON."""
    with open(filepath, 'r') as f:
        return json.load(f)


def find_optimizer_files(results_dir: str, optimizers: List[str]) -> Dict[str, Dict[str, str]]:
    """Find metrics and spectral files for each optimizer."""
    results_dir = Path(results_dir)
    files = {}

    for opt in optimizers:
        opt = opt.strip()
        opt_files = {}

        # Look for metrics CSV
        csv_path = results_dir / f'metrics_{opt}.csv'
        if csv_path.exists():
            opt_files['metrics'] = str(csv_path)

        # Look for spectral full JSON
        json_path = results_dir / f'spectral_full_{opt}.json'
        if json_path.exists():
            opt_files['spectral_full'] = str(json_path)

        # Look for spectral aggregate JSON
        agg_path = results_dir / f'spectral_aggregate_{opt}.json'
        if agg_path.exists():
            opt_files['spectral_aggregate'] = str(agg_path)

        if opt_files:
            files[opt] = opt_files

    return files


def plot_metric(ax, data: Dict[str, pd.DataFrame], col: str, title: str, log_scale: bool = False):
    """Helper to plot a single metric across optimizers."""
    has_data = False
    for opt, df in data.items():
        if col in df.columns:
            color = OPTIMIZER_COLORS.get(opt, '#333333')
            label = OPTIMIZER_LABELS.get(opt, opt)
            ax.plot(df['step'], df[col], label=label, color=color, linewidth=1.5)
            has_data = True

    if has_data:
        ax.set_xlabel('Step', fontsize=9)
        ax.set_ylabel(title, fontsize=9)
        ax.set_title(title, fontsize=10)
        if log_scale:
            ax.set_yscale('log')
        ax.legend(loc='best', fontsize=7)
        ax.grid(True, alpha=0.3)
    return has_data


def plot_training_curves(data: Dict[str, pd.DataFrame], save_dir: str):
    """Plot training curves for multiple optimizers."""
    metrics = [
        ('train_loss', 'Training Loss', False),
        ('val_loss', 'Validation Loss', False),
        ('val_perplexity', 'Validation Perplexity', True),
        ('lambda_max', 'Hessian Top Eigenvalue (λ_max)', False),
        ('cos_sim', 'Cos Similarity (update, eigenvector)', False),
    ]

    os.makedirs(save_dir, exist_ok=True)

    for col, title, log_scale in metrics:
        fig, ax = plt.subplots(figsize=(10, 6))

        for opt, df in data.items():
            if col in df.columns:
                color = OPTIMIZER_COLORS.get(opt, '#333333')
                label = OPTIMIZER_LABELS.get(opt, opt)
                ax.plot(df['step'], df[col], label=label, color=color, linewidth=2)

        ax.set_xlabel('Step', fontsize=12)
        ax.set_ylabel(title, fontsize=12)
        ax.set_title(f'{title} vs Training Step', fontsize=14)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        if log_scale:
            ax.set_yscale('log')

        plt.tight_layout()
        safe_name = col.replace('/', '_')
        plt.savefig(os.path.join(save_dir, f'comparison_{safe_name}.png'), dpi=150)
        plt.close()

    print(f"Saved training curve plots to {save_dir}/")


def plot_spectral_metrics_grid(
    data: Dict[str, pd.DataFrame],
    save_dir: str,
    matrix_type: str = 'W'
):
    """Plot all spectral metrics for a given matrix type in a grid."""
    os.makedirs(save_dir, exist_ok=True)

    # Build column names and filter to available
    available_metrics = []
    for metric_key, metric_label in SPECTRAL_METRICS:
        col = f'spectral_{matrix_type}_{metric_key}'
        for df in data.values():
            if col in df.columns:
                available_metrics.append((col, f'{metric_label} ({matrix_type})'))
                break

    if not available_metrics:
        print(f"No spectral metrics found for matrix type {matrix_type}")
        return

    # Create grid
    n_metrics = len(available_metrics)
    n_cols = 4
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.5 * n_rows))
    axes = axes.flatten() if n_metrics > 1 else [axes]

    for idx, (col, title) in enumerate(available_metrics):
        ax = axes[idx]
        log_scale = 'condition' in col.lower()
        plot_metric(ax, data, col, title.split('(')[0].strip(), log_scale)

    # Hide unused subplots
    for idx in range(len(available_metrics), len(axes)):
        axes[idx].set_visible(False)

    matrix_name = {'W': 'Weights', 'G': 'Gradients', 'delta_W': 'Weight Updates (ΔW)', 'step_update': 'Step Updates'}
    plt.suptitle(f'Spectral Metrics - {matrix_name.get(matrix_type, matrix_type)}', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'spectral_all_{matrix_type}.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved spectral metrics grid for {matrix_type} to {save_dir}/")


def plot_sv_distributions_comparison(
    spectral_data: Dict[str, Dict],
    save_dir: str,
    matrix_type: str = 'W',
    steps: List[int] = None
):
    """Plot singular value distributions across optimizers at specific steps."""
    os.makedirs(save_dir, exist_ok=True)

    # Find common steps
    all_steps = set()
    for opt, data in spectral_data.items():
        for layer_name, layer_data in data.items():
            if not isinstance(layer_data, dict):
                continue
            if matrix_type in layer_data:
                for entry in layer_data[matrix_type]:
                    if isinstance(entry, dict) and 'step' in entry:
                        all_steps.add(entry['step'])

    if not all_steps:
        print(f"No singular value data found for matrix type {matrix_type}")
        return

    all_steps = sorted(all_steps)

    # Select steps to plot
    if steps is None:
        if len(all_steps) >= 5:
            indices = [0, len(all_steps)//4, len(all_steps)//2, 3*len(all_steps)//4, -1]
            steps = [all_steps[i] for i in indices]
        elif len(all_steps) >= 3:
            steps = [all_steps[0], all_steps[len(all_steps)//2], all_steps[-1]]
        else:
            steps = all_steps

    # Create plot
    fig, axes = plt.subplots(1, len(steps), figsize=(4.5 * len(steps), 4.5))
    if len(steps) == 1:
        axes = [axes]

    for step_idx, step in enumerate(steps):
        ax = axes[step_idx]

        for opt, data in spectral_data.items():
            all_svs = []

            for layer_name, layer_data in data.items():
                if not isinstance(layer_data, dict):
                    continue
                if matrix_type not in layer_data:
                    continue

                for entry in layer_data[matrix_type]:
                    if isinstance(entry, dict) and entry.get('step') == step:
                        if 'singular_values_normalized' in entry:
                            svs = entry['singular_values_normalized']
                            if svs:
                                all_svs.extend(svs[:50])
                        break

            if all_svs:
                all_svs = sorted(all_svs, reverse=True)[:100]
                color = OPTIMIZER_COLORS.get(opt, '#333333')
                label = OPTIMIZER_LABELS.get(opt, opt)
                ax.plot(range(len(all_svs)), all_svs, label=label, color=color, linewidth=2)

        ax.set_xlabel('Singular Value Index', fontsize=10)
        ax.set_ylabel('Normalized SV', fontsize=10)
        ax.set_title(f'Step {step}', fontsize=11)
        ax.set_yscale('log')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

    matrix_name = {'W': 'Weights', 'G': 'Gradients', 'delta_W': 'ΔW', 'step_update': 'Step Update'}
    plt.suptitle(f'Singular Value Decay - {matrix_name.get(matrix_type, matrix_type)}', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'sv_decay_{matrix_type}.png'), dpi=150)
    plt.close()

    print(f"Saved SV decay comparison for {matrix_type} to {save_dir}/")


def plot_comprehensive_dashboard(data: Dict[str, pd.DataFrame], save_dir: str):
    """Create a comprehensive dashboard with key metrics from all matrix types."""
    os.makedirs(save_dir, exist_ok=True)

    fig = plt.figure(figsize=(24, 20))
    gs = GridSpec(5, 4, figure=fig, hspace=0.35, wspace=0.3)

    # Row 1: Training metrics
    row1_metrics = [
        ('train_loss', 'Train Loss'),
        ('val_loss', 'Val Loss'),
        ('val_perplexity', 'Val Perplexity'),
        ('lambda_max', 'Hessian λ_max'),
    ]

    for col_idx, (col, title) in enumerate(row1_metrics):
        ax = fig.add_subplot(gs[0, col_idx])
        log_scale = 'perplexity' in col.lower()
        plot_metric(ax, data, col, title, log_scale)

    # Row 2: Weight matrix (W) spectral metrics
    row2_metrics = [
        ('spectral_W_mean_normalized_spectral_entropy', 'Entropy (W)'),
        ('spectral_W_mean_normalized_stable_rank', 'Stable Rank (W)'),
        ('spectral_W_mean_normalized_effective_rank_99', 'Eff Rank 99% (W)'),
        ('spectral_W_mean_frobenius_norm', 'Frobenius (W)'),
    ]

    for col_idx, (col, title) in enumerate(row2_metrics):
        ax = fig.add_subplot(gs[1, col_idx])
        plot_metric(ax, data, col, title)

    # Row 3: Gradient matrix (G) spectral metrics
    row3_metrics = [
        ('spectral_G_mean_normalized_spectral_entropy', 'Entropy (G)'),
        ('spectral_G_mean_normalized_stable_rank', 'Stable Rank (G)'),
        ('spectral_G_mean_normalized_effective_rank_99', 'Eff Rank 99% (G)'),
        ('spectral_G_mean_frobenius_norm', 'Frobenius (G)'),
    ]

    for col_idx, (col, title) in enumerate(row3_metrics):
        ax = fig.add_subplot(gs[2, col_idx])
        plot_metric(ax, data, col, title)

    # Row 4: Weight update matrix (delta_W) spectral metrics
    row4_metrics = [
        ('spectral_delta_W_mean_normalized_spectral_entropy', 'Entropy (ΔW)'),
        ('spectral_delta_W_mean_normalized_stable_rank', 'Stable Rank (ΔW)'),
        ('spectral_delta_W_mean_normalized_effective_rank_99', 'Eff Rank 99% (ΔW)'),
        ('spectral_delta_W_mean_frobenius_norm', 'Frobenius (ΔW)'),
    ]

    for col_idx, (col, title) in enumerate(row4_metrics):
        ax = fig.add_subplot(gs[3, col_idx])
        plot_metric(ax, data, col, title)

    # Row 5: Step update spectral metrics
    row5_metrics = [
        ('spectral_step_update_mean_normalized_spectral_entropy', 'Entropy (step)'),
        ('spectral_step_update_mean_normalized_stable_rank', 'Stable Rank (step)'),
        ('spectral_step_update_mean_normalized_effective_rank_99', 'Eff Rank 99% (step)'),
        ('spectral_step_update_mean_frobenius_norm', 'Frobenius (step)'),
    ]

    for col_idx, (col, title) in enumerate(row5_metrics):
        ax = fig.add_subplot(gs[4, col_idx])
        plot_metric(ax, data, col, title)

    plt.suptitle('Optimizer Comparison Dashboard - All Matrix Types', fontsize=16, y=0.99)
    plt.savefig(os.path.join(save_dir, 'dashboard_full.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved comprehensive dashboard to {save_dir}/")


def plot_delta_w_focused(data: Dict[str, pd.DataFrame], save_dir: str):
    """Create a focused plot on delta_W (weight update) metrics - key for optimizer comparison."""
    os.makedirs(save_dir, exist_ok=True)

    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.3)

    # Row 1: Key delta_W metrics
    row1_metrics = [
        ('spectral_delta_W_mean_normalized_spectral_entropy', 'Spectral Entropy'),
        ('spectral_delta_W_mean_normalized_stable_rank', 'Stable Rank'),
        ('spectral_delta_W_mean_participation_ratio', 'Participation Ratio'),
        ('spectral_delta_W_mean_normalized_effective_rank_99', 'Effective Rank 99%'),
    ]

    for col_idx, (col, title) in enumerate(row1_metrics):
        ax = fig.add_subplot(gs[0, col_idx])
        plot_metric(ax, data, col, title)

    # Row 2: More delta_W metrics
    row2_metrics = [
        ('spectral_delta_W_mean_effective_rank_90', 'Effective Rank 90%'),
        ('spectral_delta_W_mean_numerical_rank', 'Numerical Rank'),
        ('spectral_delta_W_mean_sigma_max', 'Sigma Max'),
        ('spectral_delta_W_mean_frobenius_norm', 'Frobenius Norm'),
    ]

    for col_idx, (col, title) in enumerate(row2_metrics):
        ax = fig.add_subplot(gs[1, col_idx])
        plot_metric(ax, data, col, title)

    # Row 3: Comparison with training metrics
    row3_metrics = [
        ('train_loss', 'Train Loss'),
        ('val_loss', 'Val Loss'),
        ('lambda_max', 'Hessian λ_max'),
        ('spectral_W_mean_normalized_spectral_entropy', 'Entropy (W)'),
    ]

    for col_idx, (col, title) in enumerate(row3_metrics):
        ax = fig.add_subplot(gs[2, col_idx])
        plot_metric(ax, data, col, title)

    plt.suptitle('Weight Update (ΔW) Spectral Analysis - Optimizer Comparison', fontsize=16, y=0.99)
    plt.savefig(os.path.join(save_dir, 'delta_w_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved delta_W analysis to {save_dir}/")


def plot_final_metrics_bar(data: Dict[str, pd.DataFrame], save_dir: str):
    """Create bar charts comparing final metric values across optimizers."""
    os.makedirs(save_dir, exist_ok=True)

    # Extract final values
    final_metrics = {}
    for opt, df in data.items():
        if len(df) > 0:
            final_metrics[opt] = df.iloc[-1].to_dict()

    if not final_metrics:
        return

    # Metrics to compare (including delta_W)
    metrics_to_plot = [
        ('val_loss', 'Val Loss'),
        ('lambda_max', 'Hessian λ_max'),
        ('spectral_W_mean_normalized_spectral_entropy', 'Entropy (W)'),
        ('spectral_W_mean_normalized_stable_rank', 'Stable Rank (W)'),
        ('spectral_delta_W_mean_normalized_spectral_entropy', 'Entropy (ΔW)'),
        ('spectral_delta_W_mean_normalized_stable_rank', 'Stable Rank (ΔW)'),
        ('spectral_G_mean_normalized_spectral_entropy', 'Entropy (G)'),
        ('spectral_G_mean_frobenius_norm', 'Frobenius (G)'),
    ]

    # Filter to available
    available = [(m, t) for m, t in metrics_to_plot if any(m in fm for fm in final_metrics.values())]

    if not available:
        return

    n_metrics = len(available)
    n_cols = 4
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    axes = axes.flatten()

    optimizers = list(final_metrics.keys())
    x = np.arange(len(optimizers))

    for idx, (metric, title) in enumerate(available):
        ax = axes[idx]
        values = [final_metrics[opt].get(metric, 0) for opt in optimizers]
        colors = [OPTIMIZER_COLORS.get(opt, '#333333') for opt in optimizers]
        labels = [OPTIMIZER_LABELS.get(opt, opt) for opt in optimizers]

        ax.bar(x, values, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel(title, fontsize=9)
        ax.set_title(f'Final {title}', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')

    for idx in range(len(available), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('Final Metrics Comparison', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'final_metrics.png'), dpi=150)
    plt.close()

    print(f"Saved final metrics comparison to {save_dir}/")


def main():
    parser = argparse.ArgumentParser(description='Visualize optimizer comparison results')
    parser.add_argument('--results-dir', type=str, default='optimizer_comparison',
                        help='Directory containing training results')
    parser.add_argument('--save-dir', type=str, default=None,
                        help='Directory to save plots (default: results-dir/plots)')
    parser.add_argument('--optimizers', type=str, default='adam,adamw,muon,adam_ns,adam_ns_grad,adam_ns_update',
                        help='Comma-separated list of optimizers to compare')
    args = parser.parse_args()

    results_dir = args.results_dir
    save_dir = args.save_dir or os.path.join(results_dir, 'plots')
    optimizers = [o.strip() for o in args.optimizers.split(',')]

    print("=" * 60)
    print("Optimizer Comparison Visualization")
    print("=" * 60)
    print(f"Results directory: {results_dir}")
    print(f"Save directory: {save_dir}")
    print(f"Optimizers: {optimizers}")
    print()

    # Find available files
    files = find_optimizer_files(results_dir, optimizers)

    if not files:
        print(f"No result files found in {results_dir}")
        return

    print(f"Found results for optimizers: {list(files.keys())}")

    # Load data
    metrics_data = {}
    spectral_data = {}

    for opt, opt_files in files.items():
        if 'metrics' in opt_files:
            print(f"Loading metrics for {opt}...")
            metrics_data[opt] = load_metrics_csv(opt_files['metrics'])

        if 'spectral_full' in opt_files:
            print(f"Loading spectral data for {opt}...")
            spectral_data[opt] = load_spectral_full(opt_files['spectral_full'])

    # Generate all plots
    if metrics_data:
        print("\n--- Generating plots ---\n")

        print("Training curves...")
        plot_training_curves(metrics_data, save_dir)

        # Spectral metrics for ALL matrix types
        for matrix_type in MATRIX_TYPES:
            print(f"Spectral metrics grid ({matrix_type})...")
            plot_spectral_metrics_grid(metrics_data, save_dir, matrix_type=matrix_type)

        print("Comprehensive dashboard...")
        plot_comprehensive_dashboard(metrics_data, save_dir)

        print("Delta W focused analysis...")
        plot_delta_w_focused(metrics_data, save_dir)

        print("Final metrics comparison...")
        plot_final_metrics_bar(metrics_data, save_dir)

    if spectral_data:
        # SV distributions for ALL matrix types
        for matrix_type in MATRIX_TYPES:
            print(f"SV distributions ({matrix_type})...")
            plot_sv_distributions_comparison(spectral_data, save_dir, matrix_type=matrix_type)

    print("\n" + "=" * 60)
    print("Visualization complete!")
    print("=" * 60)
    print(f"\nPlots saved to: {save_dir}/")
    print("\nGenerated plots:")
    for f in sorted(Path(save_dir).glob('*.png')):
        print(f"  - {f.name}")


if __name__ == '__main__':
    main()