"""
Ablation Visualization Script

Generates comprehensive visualizations for ablation experiments:
- Learning rate ablations
- Model depth ablations
- Regularization comparison
- Cross-optimizer comparison

Metrics visualized:
- Train/val loss and perplexity
- Gradient norms
- Spectral metrics (entropy, stable rank, effective rank)
- Frobenius norms
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Plot style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 10

# Color palettes
OPTIMIZER_COLORS = {
    'adam': '#1f77b4',
    'adamw': '#2ca02c',
    'muon': '#d62728',
    'sgd': '#9467bd',
    'adam_reg': '#ff7f0e',  # Adam with regularization
}

# Generate distinct colors for ablations
def get_ablation_colors(n: int) -> List[str]:
    """Generate n distinct colors for ablation curves."""
    cmap = plt.cm.viridis
    return [mcolors.to_hex(cmap(i / max(n - 1, 1))) for i in range(n)]


def load_metrics_jsonl(path: str) -> List[Dict]:
    """Load metrics from JSONL file."""
    entries = []
    if os.path.exists(path):
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    return entries


def load_experiment_data(results_dir: str) -> Dict[str, Dict[str, Any]]:
    """
    Load all experiment data from a results directory.

    Returns:
        Dict mapping experiment_name -> {
            'metrics': [...],
            'config': {...},
            'ablation_type': 'lr' | 'depth' | 'reg',
            'ablation_value': ...,
        }
    """
    experiments = {}

    for subdir in Path(results_dir).iterdir():
        if not subdir.is_dir():
            continue

        exp_name = subdir.name

        # Find metrics file
        metrics_files = list(subdir.glob('metrics_*.jsonl'))
        if not metrics_files:
            continue

        metrics_path = metrics_files[0]
        metrics = load_metrics_jsonl(str(metrics_path))

        if not metrics:
            continue

        # Determine ablation type and value
        ablation_type = None
        ablation_value = None

        if exp_name.startswith('lr_'):
            ablation_type = 'lr'
            ablation_value = float(exp_name.replace('lr_', ''))
        elif exp_name.startswith('depth_'):
            ablation_type = 'depth'
            ablation_value = int(exp_name.replace('depth_', ''))
        elif exp_name.startswith('reg_'):
            ablation_type = 'reg'
            val = exp_name.replace('reg_', '')
            ablation_value = 0.0 if val == 'none' else float(val)

        # Load summary if available
        summary_files = list(subdir.glob('summary_*.json'))
        config = {}
        if summary_files:
            with open(summary_files[0], 'r') as f:
                summary = json.load(f)
                config = summary.get('config', {})

        experiments[exp_name] = {
            'metrics': metrics,
            'config': config,
            'ablation_type': ablation_type,
            'ablation_value': ablation_value,
            'path': str(subdir),
        }

    return experiments


def extract_time_series(
    experiments: Dict[str, Dict],
    metric_name: str,
    ablation_type: Optional[str] = None,
) -> Dict[str, Tuple[List[float], List[float]]]:
    """
    Extract time series data for a metric.

    Returns:
        Dict mapping experiment_name -> (steps, values)
    """
    result = {}

    for exp_name, exp_data in experiments.items():
        if ablation_type and exp_data.get('ablation_type') != ablation_type:
            continue

        metrics = exp_data['metrics']
        steps = [m['step'] for m in metrics if metric_name in m]
        values = [m[metric_name] for m in metrics if metric_name in m]

        if steps:
            result[exp_name] = (steps, values)

    return result


def plot_metric_comparison(
    experiments: Dict[str, Dict],
    metric_name: str,
    ablation_type: str,
    title: str,
    ylabel: str,
    save_path: Optional[str] = None,
    log_scale: bool = False,
):
    """Plot a metric comparison across ablation values."""
    data = extract_time_series(experiments, metric_name, ablation_type)

    if not data:
        print(f"No data found for {metric_name} with ablation_type={ablation_type}")
        return

    # Sort by ablation value
    sorted_items = sorted(
        [(name, d) for name, d in data.items()],
        key=lambda x: experiments[x[0]].get('ablation_value', 0)
    )

    fig, ax = plt.subplots(figsize=(12, 7))
    colors = get_ablation_colors(len(sorted_items))

    for idx, (exp_name, (steps, values)) in enumerate(sorted_items):
        ablation_val = experiments[exp_name].get('ablation_value', exp_name)

        # Format label
        if ablation_type == 'lr':
            label = f"lr={ablation_val:.0e}"
        elif ablation_type == 'depth':
            label = f"depth={ablation_val}"
        elif ablation_type == 'reg':
            label = f"λ={ablation_val}" if ablation_val > 0 else "no reg"
        else:
            label = exp_name

        ax.plot(steps, values, color=colors[idx], linewidth=2, label=label, marker='o', markersize=2)

    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    if log_scale:
        ax.set_yscale('log')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")

    plt.close(fig)


def plot_loss_and_perplexity(
    experiments: Dict[str, Dict],
    ablation_type: str,
    save_dir: str,
):
    """Plot train/val loss and perplexity."""
    # 2x2 grid: train loss, val loss, train ppl, val ppl
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    metrics_info = [
        ('train_loss', 'Train Loss', axes[0, 0]),
        ('val_loss', 'Validation Loss', axes[0, 1]),
        ('train_perplexity', 'Train Perplexity', axes[1, 0]),
        ('val_perplexity', 'Validation Perplexity', axes[1, 1]),
    ]

    data_by_metric = {
        name: extract_time_series(experiments, name, ablation_type)
        for name, _, _ in metrics_info
    }

    # Sort experiments by ablation value
    exp_names = set()
    for data in data_by_metric.values():
        exp_names.update(data.keys())

    sorted_exps = sorted(
        exp_names,
        key=lambda x: experiments[x].get('ablation_value', 0)
    )
    colors = get_ablation_colors(len(sorted_exps))

    for metric_name, title, ax in metrics_info:
        data = data_by_metric[metric_name]

        for idx, exp_name in enumerate(sorted_exps):
            if exp_name not in data:
                continue

            steps, values = data[exp_name]
            ablation_val = experiments[exp_name].get('ablation_value', exp_name)

            if ablation_type == 'lr':
                label = f"lr={ablation_val:.0e}"
            elif ablation_type == 'depth':
                label = f"d={ablation_val}"
            elif ablation_type == 'reg':
                label = f"λ={ablation_val}" if ablation_val > 0 else "no reg"
            else:
                label = exp_name

            ax.plot(steps, values, color=colors[idx], linewidth=2, label=label, marker='o', markersize=2)

        ax.set_xlabel('Step', fontsize=11)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)

    ablation_labels = {'lr': 'Learning Rate', 'depth': 'Model Depth', 'reg': 'Regularization'}
    fig.suptitle(f'Loss & Perplexity - {ablation_labels.get(ablation_type, ablation_type)} Ablation',
                 fontsize=14, fontweight='bold', y=1.01)

    plt.tight_layout()

    path = os.path.join(save_dir, f'loss_perplexity_{ablation_type}.png')
    plt.savefig(path)
    print(f"Saved: {path}")
    plt.close(fig)


def plot_gradient_metrics(
    experiments: Dict[str, Dict],
    ablation_type: str,
    save_dir: str,
):
    """Plot gradient norm metrics."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    metrics_info = [
        ('grad_norm', 'Gradient Norm', axes[0]),
        ('weight_norm', 'Weight Norm', axes[1]),
    ]

    for metric_name, title, ax in metrics_info:
        data = extract_time_series(experiments, metric_name, ablation_type)

        if not data:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
            continue

        sorted_exps = sorted(
            data.keys(),
            key=lambda x: experiments[x].get('ablation_value', 0)
        )
        colors = get_ablation_colors(len(sorted_exps))

        for idx, exp_name in enumerate(sorted_exps):
            steps, values = data[exp_name]
            ablation_val = experiments[exp_name].get('ablation_value', exp_name)

            if ablation_type == 'lr':
                label = f"lr={ablation_val:.0e}"
            elif ablation_type == 'depth':
                label = f"d={ablation_val}"
            elif ablation_type == 'reg':
                label = f"λ={ablation_val}" if ablation_val > 0 else "no reg"
            else:
                label = exp_name

            ax.plot(steps, values, color=colors[idx], linewidth=2, label=label, marker='o', markersize=2)

        ax.set_xlabel('Step', fontsize=11)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)

    ablation_labels = {'lr': 'Learning Rate', 'depth': 'Model Depth', 'reg': 'Regularization'}
    fig.suptitle(f'Gradient & Weight Norms - {ablation_labels.get(ablation_type, ablation_type)} Ablation',
                 fontsize=14, fontweight='bold', y=1.01)

    plt.tight_layout()

    path = os.path.join(save_dir, f'grad_weight_norms_{ablation_type}.png')
    plt.savefig(path)
    print(f"Saved: {path}")
    plt.close(fig)


def plot_spectral_metrics(
    experiments: Dict[str, Dict],
    ablation_type: str,
    save_dir: str,
):
    """Plot spectral metrics for W and delta_W."""
    # 2x3 grid: W metrics (entropy, stable rank, eff rank) and delta_W metrics
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    metrics_info = [
        ('spectral_W_mean_normalized_spectral_entropy', 'H̃(W)', axes[0, 0]),
        ('spectral_W_mean_normalized_stable_rank', 'r̃(W)', axes[0, 1]),
        ('spectral_W_mean_normalized_effective_rank_99', 'eff_r̃₉₉(W)', axes[0, 2]),
        ('spectral_delta_W_mean_normalized_spectral_entropy', 'H̃(ΔW)', axes[1, 0]),
        ('spectral_delta_W_mean_normalized_stable_rank', 'r̃(ΔW)', axes[1, 1]),
        ('spectral_delta_W_mean_normalized_effective_rank_99', 'eff_r̃₉₉(ΔW)', axes[1, 2]),
    ]

    for metric_name, title, ax in metrics_info:
        data = extract_time_series(experiments, metric_name, ablation_type)

        if not data:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
            continue

        sorted_exps = sorted(
            data.keys(),
            key=lambda x: experiments[x].get('ablation_value', 0)
        )
        colors = get_ablation_colors(len(sorted_exps))

        for idx, exp_name in enumerate(sorted_exps):
            steps, values = data[exp_name]
            ablation_val = experiments[exp_name].get('ablation_value', exp_name)

            if ablation_type == 'lr':
                label = f"lr={ablation_val:.0e}"
            elif ablation_type == 'depth':
                label = f"d={ablation_val}"
            elif ablation_type == 'reg':
                label = f"λ={ablation_val}" if ablation_val > 0 else "no reg"
            else:
                label = exp_name

            ax.plot(steps, values, color=colors[idx], linewidth=2, label=label, marker='o', markersize=2)

        ax.set_xlabel('Step', fontsize=11)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1.05)
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

    ablation_labels = {'lr': 'Learning Rate', 'depth': 'Model Depth', 'reg': 'Regularization'}
    fig.suptitle(f'Spectral Metrics - {ablation_labels.get(ablation_type, ablation_type)} Ablation',
                 fontsize=14, fontweight='bold', y=1.01)

    plt.tight_layout()

    path = os.path.join(save_dir, f'spectral_metrics_{ablation_type}.png')
    plt.savefig(path)
    print(f"Saved: {path}")
    plt.close(fig)


def plot_frobenius_norms(
    experiments: Dict[str, Dict],
    ablation_type: str,
    save_dir: str,
):
    """Plot Frobenius norm metrics."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    metrics_info = [
        ('spectral_W_mean_frobenius_norm', '||W||_F', axes[0]),
        ('spectral_delta_W_mean_frobenius_norm', '||ΔW||_F', axes[1]),
    ]

    for metric_name, title, ax in metrics_info:
        data = extract_time_series(experiments, metric_name, ablation_type)

        if not data:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
            continue

        sorted_exps = sorted(
            data.keys(),
            key=lambda x: experiments[x].get('ablation_value', 0)
        )
        colors = get_ablation_colors(len(sorted_exps))

        for idx, exp_name in enumerate(sorted_exps):
            steps, values = data[exp_name]
            ablation_val = experiments[exp_name].get('ablation_value', exp_name)

            if ablation_type == 'lr':
                label = f"lr={ablation_val:.0e}"
            elif ablation_type == 'depth':
                label = f"d={ablation_val}"
            elif ablation_type == 'reg':
                label = f"λ={ablation_val}" if ablation_val > 0 else "no reg"
            else:
                label = exp_name

            ax.plot(steps, values, color=colors[idx], linewidth=2, label=label, marker='o', markersize=2)

        ax.set_xlabel('Step', fontsize=11)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)

    ablation_labels = {'lr': 'Learning Rate', 'depth': 'Model Depth', 'reg': 'Regularization'}
    fig.suptitle(f'Frobenius Norms - {ablation_labels.get(ablation_type, ablation_type)} Ablation',
                 fontsize=14, fontweight='bold', y=1.01)

    plt.tight_layout()

    path = os.path.join(save_dir, f'frobenius_norms_{ablation_type}.png')
    plt.savefig(path)
    print(f"Saved: {path}")
    plt.close(fig)


def plot_final_metrics_comparison(
    experiments: Dict[str, Dict],
    save_dir: str,
):
    """Plot bar chart comparing final metrics across all experiments."""
    # Extract final metrics
    final_data = []
    for exp_name, exp_data in experiments.items():
        metrics = exp_data['metrics']
        if metrics:
            final = metrics[-1]
            final_data.append({
                'name': exp_name,
                'ablation_type': exp_data.get('ablation_type', 'unknown'),
                'ablation_value': exp_data.get('ablation_value', 0),
                'val_loss': final.get('val_loss', float('nan')),
                'val_perplexity': final.get('val_perplexity', float('nan')),
                'delta_w_entropy': final.get('spectral_delta_W_mean_normalized_spectral_entropy', float('nan')),
                'delta_w_stable_rank': final.get('spectral_delta_W_mean_normalized_stable_rank', float('nan')),
            })

    if not final_data:
        return

    # Sort by ablation type and value
    final_data.sort(key=lambda x: (x['ablation_type'], x['ablation_value']))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    metrics_to_plot = [
        ('val_loss', 'Final Val Loss', axes[0, 0]),
        ('val_perplexity', 'Final Val Perplexity', axes[0, 1]),
        ('delta_w_entropy', 'Final H̃(ΔW)', axes[1, 0]),
        ('delta_w_stable_rank', 'Final r̃(ΔW)', axes[1, 1]),
    ]

    for metric_key, title, ax in metrics_to_plot:
        names = [d['name'] for d in final_data]
        values = [d[metric_key] for d in final_data]

        # Color by ablation type
        colors = []
        for d in final_data:
            if d['ablation_type'] == 'lr':
                colors.append('#1f77b4')
            elif d['ablation_type'] == 'depth':
                colors.append('#2ca02c')
            elif d['ablation_type'] == 'reg':
                colors.append('#d62728')
            else:
                colors.append('#7f7f7f')

        bars = ax.bar(range(len(names)), values, color=colors)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Final Metrics Comparison', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()

    path = os.path.join(save_dir, 'final_metrics_comparison.png')
    plt.savefig(path)
    print(f"Saved: {path}")
    plt.close(fig)


def plot_optimizer_comparison(
    adam_dir: str,
    muon_dir: str,
    save_dir: str,
):
    """Compare Adam and Muon results."""
    adam_exps = load_experiment_data(adam_dir) if os.path.exists(adam_dir) else {}
    muon_exps = load_experiment_data(muon_dir) if os.path.exists(muon_dir) else {}

    if not adam_exps or not muon_exps:
        print("Need both Adam and Muon results for comparison")
        return

    # Find comparable experiments (same depth)
    adam_depth_exps = {exp['ablation_value']: (name, exp) for name, exp in adam_exps.items()
                       if exp.get('ablation_type') == 'depth'}
    muon_depth_exps = {exp['ablation_value']: (name, exp) for name, exp in muon_exps.items()
                       if exp.get('ablation_type') == 'depth'}

    common_depths = set(adam_depth_exps.keys()) & set(muon_depth_exps.keys())

    if not common_depths:
        print("No common depths found for comparison")
        return

    # Plot comparison for each common depth
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    metrics_to_compare = [
        ('val_loss', 'Validation Loss', axes[0, 0]),
        ('val_perplexity', 'Validation Perplexity', axes[0, 1]),
        ('spectral_delta_W_mean_normalized_spectral_entropy', 'H̃(ΔW)', axes[1, 0]),
        ('spectral_delta_W_mean_normalized_stable_rank', 'r̃(ΔW)', axes[1, 1]),
    ]

    for metric_name, title, ax in metrics_to_compare:
        for depth in sorted(common_depths):
            adam_name, adam_exp = adam_depth_exps[depth]
            muon_name, muon_exp = muon_depth_exps[depth]

            # Adam
            adam_metrics = adam_exp['metrics']
            adam_steps = [m['step'] for m in adam_metrics if metric_name in m]
            adam_values = [m[metric_name] for m in adam_metrics if metric_name in m]

            # Muon
            muon_metrics = muon_exp['metrics']
            muon_steps = [m['step'] for m in muon_metrics if metric_name in m]
            muon_values = [m[metric_name] for m in muon_metrics if metric_name in m]

            if adam_steps:
                ax.plot(adam_steps, adam_values, color=OPTIMIZER_COLORS['adam'],
                       linewidth=2, label=f'Adam d={depth}', linestyle='-')
            if muon_steps:
                ax.plot(muon_steps, muon_values, color=OPTIMIZER_COLORS['muon'],
                       linewidth=2, label=f'Muon d={depth}', linestyle='--')

        ax.set_xlabel('Step', fontsize=11)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Adam vs Muon Comparison', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()

    path = os.path.join(save_dir, 'optimizer_comparison.png')
    plt.savefig(path)
    print(f"Saved: {path}")
    plt.close(fig)


def plot_regularization_effect(
    experiments: Dict[str, Dict],
    save_dir: str,
):
    """Plot the effect of spectral regularization on Adam."""
    reg_exps = {name: exp for name, exp in experiments.items()
                if exp.get('ablation_type') == 'reg'}

    if not reg_exps:
        print("No regularization experiments found")
        return

    # Sort by lambda
    sorted_exps = sorted(reg_exps.items(), key=lambda x: x[1].get('ablation_value', 0))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    metrics_info = [
        ('val_loss', 'Validation Loss', axes[0, 0]),
        ('val_perplexity', 'Validation Perplexity', axes[0, 1]),
        ('spectral_delta_W_mean_normalized_stable_rank', 'r̃(ΔW) - Target of Regularization', axes[1, 0]),
        ('spectral_delta_W_mean_normalized_spectral_entropy', 'H̃(ΔW)', axes[1, 1]),
    ]

    colors = get_ablation_colors(len(sorted_exps))

    for metric_name, title, ax in metrics_info:
        for idx, (exp_name, exp_data) in enumerate(sorted_exps):
            metrics = exp_data['metrics']
            steps = [m['step'] for m in metrics if metric_name in m]
            values = [m[metric_name] for m in metrics if metric_name in m]

            if not steps:
                continue

            ablation_val = exp_data.get('ablation_value', 0)
            label = f"λ={ablation_val}" if ablation_val > 0 else "no reg (baseline)"

            ax.plot(steps, values, color=colors[idx], linewidth=2, label=label, marker='o', markersize=2)

        ax.set_xlabel('Step', fontsize=11)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Effect of Spectral Regularization on Adam\nL = L_ce + λ·Σ(1 - r̃(ΔW))',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    path = os.path.join(save_dir, 'regularization_effect.png')
    plt.savefig(path)
    print(f"Saved: {path}")
    plt.close(fig)


def generate_all_visualizations(
    results_dir: str,
    optimizer: str,
    save_dir: str,
):
    """Generate all visualizations for an optimizer's ablation results."""
    os.makedirs(save_dir, exist_ok=True)

    print(f"\nLoading experiments from: {results_dir}")
    experiments = load_experiment_data(results_dir)
    print(f"Found {len(experiments)} experiments")

    if not experiments:
        print("No experiments found!")
        return

    # Determine which ablation types are present
    ablation_types = set(exp.get('ablation_type') for exp in experiments.values() if exp.get('ablation_type'))
    print(f"Ablation types: {ablation_types}")

    # Generate plots for each ablation type
    for ablation_type in ablation_types:
        print(f"\nGenerating plots for {ablation_type} ablation...")

        plot_loss_and_perplexity(experiments, ablation_type, save_dir)
        plot_gradient_metrics(experiments, ablation_type, save_dir)
        plot_spectral_metrics(experiments, ablation_type, save_dir)
        plot_frobenius_norms(experiments, ablation_type, save_dir)

    # Regularization-specific plots
    if 'reg' in ablation_types:
        plot_regularization_effect(experiments, save_dir)

    # Final metrics comparison
    plot_final_metrics_comparison(experiments, save_dir)

    print(f"\nAll plots saved to: {save_dir}")


def main():
    parser = argparse.ArgumentParser(description='Visualize ablation experiment results')
    parser.add_argument('--results-dir', type=str, required=True,
                        help='Directory containing ablation results')
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='Optimizer name (adam, muon)')
    parser.add_argument('--save-dir', type=str, default=None,
                        help='Directory to save plots (defaults to results-dir/plots)')
    parser.add_argument('--compare-with', type=str, default=None,
                        help='Path to other optimizer results for comparison')

    args = parser.parse_args()

    save_dir = args.save_dir or os.path.join(args.results_dir, 'plots')

    generate_all_visualizations(args.results_dir, args.optimizer, save_dir)

    # Cross-optimizer comparison
    if args.compare_with:
        print("\nGenerating cross-optimizer comparison...")
        plot_optimizer_comparison(args.results_dir, args.compare_with, save_dir)


if __name__ == '__main__':
    main()
