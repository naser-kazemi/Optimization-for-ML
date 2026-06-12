"""
Comprehensive Visualization System for Optimization Geometry Analysis

Visualizations include:
1. Multi-panel training dynamics (loss, grad norm, eigenvalues - shared x-axis)
2. Per-layer gradient norm heatmaps (with proper layer naming)
3. Hessian eigenvalue spectrum evolution + spectral density
4. Weight matrix effective rank dynamics
5. Gradient subspace stability analysis over time
6. Cross-layer gradient similarity by matrix type (Q, K, V, MLP)
7. Consecutive layer gradient similarity
8. Validation perplexity over time
9. Comprehensive optimizer comparison dashboard
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import argparse

# Set publication-quality style
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.titlesize": 14,
    "figure.dpi": 150,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "axes.spines.top": False,
    "axes.spines.right": False
})

# Color scheme for optimizers
OPTIMIZER_COLORS = {
    'sgd': '#E74C3C',      # Red
    'adam': '#3498DB',      # Blue
    'adamw': '#2ECC71',     # Green
    'muon': '#9B59B6'       # Purple
}

OPTIMIZER_LABELS = {
    'sgd': 'SGD',
    'adam': 'Adam',
    'adamw': 'AdamW',
    'muon': 'Muon'
}

# Color scheme for layer types
LAYER_TYPE_COLORS = {
    'c_q': '#E74C3C',       # Query - Red
    'c_k': '#3498DB',       # Key - Blue
    'c_v': '#2ECC71',       # Value - Green
    'c_fc': '#9B59B6',      # MLP FC - Purple
    'c_proj': '#F39C12',    # Projection - Orange
    'wte': '#1ABC9C',       # Embedding - Teal
    'lm_head': '#34495E'    # LM Head - Dark Gray
}


def load_scalar_metrics(output_dir: str, optimizer: str) -> Optional[pd.DataFrame]:
    """Load scalar metrics CSV for an optimizer."""
    path = os.path.join(output_dir, f"metrics_{optimizer}.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def load_layer_metrics(output_dir: str, optimizer: str) -> Optional[List[Dict]]:
    """Load per-layer metrics JSON for an optimizer."""
    path = os.path.join(output_dir, f"layer_metrics_{optimizer}.json")
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None


def load_eigenvalue_history(output_dir: str, optimizer: str) -> Optional[List[Dict]]:
    """Load eigenvalue history for an optimizer."""
    path = os.path.join(output_dir, f"eigenvalues_{optimizer}.json")
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None


def load_cross_layer_sim(output_dir: str, optimizer: str) -> Optional[List[Dict]]:
    """Load cross-layer similarity history for an optimizer."""
    path = os.path.join(output_dir, f"cross_layer_sim_{optimizer}.json")
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None


def get_layer_display_name(full_name: str) -> str:
    """Convert full layer name to readable display name."""
    # transformer.h.0.attn.c_q.weight -> L0.attn.c_q
    # transformer.h.5.mlp.c_fc.weight -> L5.mlp.c_fc
    parts = full_name.split('.')

    # Find layer index
    layer_idx = None
    for i, p in enumerate(parts):
        if p == 'h' and i + 1 < len(parts):
            try:
                layer_idx = int(parts[i + 1])
            except:
                pass

    # Extract component type
    if 'c_q' in full_name:
        comp = 'Q'
    elif 'c_k' in full_name:
        comp = 'K'
    elif 'c_v' in full_name:
        comp = 'V'
    elif 'c_fc' in full_name:
        comp = 'FC'
    elif 'c_proj' in full_name:
        if 'attn' in full_name:
            comp = 'Attn.Proj'
        else:
            comp = 'MLP.Proj'
    elif 'wte' in full_name:
        return 'Embed'
    elif 'lm_head' in full_name:
        return 'LM_Head'
    elif 've_gate' in full_name:
        comp = 'VE_Gate'
    else:
        comp = parts[-2] if len(parts) > 1 else parts[-1]

    if layer_idx is not None:
        return f"L{layer_idx}.{comp}"
    return comp


def categorize_layer(name: str) -> str:
    """Categorize layer by type."""
    if 'c_q' in name:
        return 'Query (Q)'
    elif 'c_k' in name:
        return 'Key (K)'
    elif 'c_v' in name:
        return 'Value (V)'
    elif 'c_fc' in name:
        return 'MLP FC'
    elif 'c_proj' in name:
        if 'attn' in name:
            return 'Attn Proj'
        return 'MLP Proj'
    elif 'wte' in name or 'embed' in name:
        return 'Embedding'
    elif 'lm_head' in name:
        return 'LM Head'
    return 'Other'


# =============================================================================
# PLOT 1: VALIDATION PERPLEXITY OVER TIME
# =============================================================================

def plot_validation_perplexity(output_dir: str, optimizers: List[str], save_path: str):
    """
    Plot validation perplexity evolution across training for all optimizers.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for opt in optimizers:
        df = load_scalar_metrics(output_dir, opt)
        if df is None or 'val_perplexity' not in df.columns:
            continue

        color = OPTIMIZER_COLORS.get(opt, '#333333')
        label = OPTIMIZER_LABELS.get(opt, opt.upper())

        ax.plot(df['step'], df['val_perplexity'],
                color=color, label=label, linewidth=2, marker='o', markersize=4)

    ax.set_xlabel("Training Step", fontweight='bold')
    ax.set_ylabel("Validation Perplexity", fontweight='bold')
    ax.set_title("Validation Perplexity Evolution", fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')

    # Use log scale if perplexity varies a lot
    if ax.get_ylim()[1] / ax.get_ylim()[0] > 10:
        ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# =============================================================================
# PLOT 2: MULTI-PANEL TRAINING DYNAMICS (Shared X-axis)
# =============================================================================

def plot_training_dynamics(output_dir: str, optimizers: List[str], save_path: str):
    """
    Create a multi-panel figure showing training dynamics with shared x-axis.
    """
    fig, axes = plt.subplots(5, 1, figsize=(12, 14), sharex=True)
    fig.suptitle("Training Dynamics: Optimizer Comparison", fontsize=16, fontweight='bold')

    for opt in optimizers:
        df = load_scalar_metrics(output_dir, opt)
        if df is None:
            continue

        color = OPTIMIZER_COLORS.get(opt, '#333333')
        label = OPTIMIZER_LABELS.get(opt, opt.upper())
        steps = df['step']

        # Panel 1: Loss curves
        axes[0].plot(steps, df['train_loss'], color=color, alpha=0.7, linestyle='-', label=f'{label} (train)')
        axes[0].plot(steps, df['val_loss'], color=color, alpha=1.0, linestyle='--', label=f'{label} (val)', linewidth=2)

        # Panel 2: Gradient norm
        if 'total_grad_norm' in df.columns:
            axes[1].plot(steps, df['total_grad_norm'], color=color, label=label, linewidth=1.5)

        # Panel 3: Lambda max
        if 'lambda_max' in df.columns:
            lambda_vals = df['lambda_max'].replace(0, np.nan)
            axes[2].plot(steps, lambda_vals, color=color, label=label, linewidth=1.5, marker='o', markersize=3)

        # Panel 4: Effective rank
        if 'avg_effective_rank' in df.columns:
            axes[3].plot(steps, df['avg_effective_rank'], color=color, label=label, linewidth=1.5)

        # Panel 5: Subspace overlap
        if 'avg_subspace_overlap' in df.columns:
            axes[4].plot(steps, df['avg_subspace_overlap'], color=color, label=label, linewidth=1.5)

    # Configure each panel
    axes[0].set_ylabel("Loss", fontweight='bold')
    axes[0].set_title("Training & Validation Loss", fontsize=11, loc='left')
    axes[0].legend(loc='upper right', ncol=2, fontsize=8)

    axes[1].set_ylabel("Gradient Norm", fontweight='bold')
    axes[1].set_title("Total Gradient L2 Norm", fontsize=11, loc='left')
    axes[1].legend(loc='upper right')
    axes[1].set_yscale('log')

    axes[2].set_ylabel(r"$\lambda_{max}$", fontweight='bold')
    axes[2].set_title("Hessian Top Eigenvalue (Sharpness)", fontsize=11, loc='left')
    axes[2].legend(loc='upper right')

    axes[3].set_ylabel("Effective Rank", fontweight='bold')
    axes[3].set_title("Average Weight Matrix Effective Rank", fontsize=11, loc='left')
    axes[3].legend(loc='upper right')

    axes[4].set_ylabel("Overlap", fontweight='bold')
    axes[4].set_xlabel("Training Step", fontweight='bold')
    axes[4].set_title("Gradient Subspace Overlap (Stability)", fontsize=11, loc='left')
    axes[4].legend(loc='lower right')
    axes[4].set_ylim(-0.1, 1.1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# =============================================================================
# PLOT 3: GRADIENT NORM HEATMAP (Layers x Steps) - FIXED NAMING
# =============================================================================

def plot_gradient_norm_heatmap(output_dir: str, optimizer: str, save_path: str):
    """
    Create a heatmap showing gradient norm evolution across layers and training steps.
    Uses readable layer names instead of showing 'weight'.
    """
    layer_metrics = load_layer_metrics(output_dir, optimizer)
    if layer_metrics is None or len(layer_metrics) == 0:
        print(f"No layer metrics found for {optimizer}")
        return

    # Extract gradient norms per layer per step
    all_layers = set()
    for entry in layer_metrics:
        if 'grad_norms' in entry:
            all_layers.update(entry['grad_norms'].keys())

    # Sort layers by type and index for logical ordering
    def layer_sort_key(name):
        # Extract layer index
        layer_idx = -1
        for part in name.split('.'):
            try:
                layer_idx = int(part)
                break
            except:
                pass

        # Determine type order: embed, attn (q,k,v,proj), mlp (fc, proj), head
        if 'wte' in name or 'embed' in name:
            type_order = 0
        elif 'c_q' in name:
            type_order = 1
        elif 'c_k' in name:
            type_order = 2
        elif 'c_v' in name:
            type_order = 3
        elif 'attn' in name and 'c_proj' in name:
            type_order = 4
        elif 'c_fc' in name:
            type_order = 5
        elif 'mlp' in name and 'c_proj' in name:
            type_order = 6
        elif 'lm_head' in name:
            type_order = 100
        else:
            type_order = 50

        return (type_order, layer_idx, name)

    layer_list = sorted(list(all_layers), key=layer_sort_key)

    # Filter to keep only 2D weight matrices
    layer_list = [l for l in layer_list if 'weight' in l.lower() or any(x in l for x in ['c_q', 'c_k', 'c_v', 'c_fc', 'c_proj'])]

    if len(layer_list) == 0:
        print(f"No 2D layers found for {optimizer}")
        return

    steps = [entry['step'] for entry in layer_metrics if 'grad_norms' in entry]

    # Build heatmap matrix
    heatmap_data = np.zeros((len(layer_list), len(steps)))
    for j, entry in enumerate(layer_metrics):
        if 'grad_norms' not in entry:
            continue
        for i, layer in enumerate(layer_list):
            if layer in entry['grad_norms']:
                heatmap_data[i, j] = entry['grad_norms'][layer]

    # Create readable display names
    display_names = [get_layer_display_name(l) for l in layer_list]

    # If too many layers, sample them
    max_layers = 40
    if len(layer_list) > max_layers:
        indices = np.linspace(0, len(layer_list)-1, max_layers, dtype=int)
        layer_list = [layer_list[i] for i in indices]
        display_names = [display_names[i] for i in indices]
        heatmap_data = heatmap_data[indices, :]

    fig, ax = plt.subplots(figsize=(14, 10))

    # Log scale for better visualization
    heatmap_log = np.log10(heatmap_data + 1e-10)

    im = ax.imshow(heatmap_log, aspect='auto', cmap='viridis', interpolation='nearest')

    # Configure axes
    ax.set_xlabel("Training Step", fontweight='bold')
    ax.set_ylabel("Layer", fontweight='bold')
    ax.set_title(f"Gradient Norm Evolution - {OPTIMIZER_LABELS.get(optimizer, optimizer.upper())}",
                 fontsize=14, fontweight='bold')

    # X-axis ticks
    num_xticks = min(10, len(steps))
    xtick_indices = np.linspace(0, len(steps)-1, num_xticks, dtype=int)
    ax.set_xticks(xtick_indices)
    ax.set_xticklabels([steps[i] for i in xtick_indices])

    # Y-axis ticks with readable names
    ax.set_yticks(range(len(display_names)))
    ax.set_yticklabels(display_names, fontsize=8)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, label='log10(Gradient Norm)')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# =============================================================================
# PLOT 4: HESSIAN SPECTRAL DENSITY
# =============================================================================

def plot_hessian_spectral_density(output_dir: str, optimizers: List[str], save_path: str):
    """
    Plot Hessian eigenvalue spectral density (histogram) at different training stages.
    """
    n_opts = len(optimizers)
    fig, axes = plt.subplots(2, n_opts, figsize=(5*n_opts, 8))
    if n_opts == 1:
        axes = axes.reshape(2, 1)

    for col, opt in enumerate(optimizers):
        eigenvalue_history = load_eigenvalue_history(output_dir, opt)
        if eigenvalue_history is None or len(eigenvalue_history) == 0:
            axes[0, col].text(0.5, 0.5, 'No data', ha='center', va='center', transform=axes[0, col].transAxes)
            axes[1, col].text(0.5, 0.5, 'No data', ha='center', va='center', transform=axes[1, col].transAxes)
            axes[0, col].set_title(f"{OPTIMIZER_LABELS.get(opt, opt.upper())} - Spectrum")
            axes[1, col].set_title(f"{OPTIMIZER_LABELS.get(opt, opt.upper())} - Density")
            continue

        # Top row: Eigenvalue spectrum at checkpoints
        ax_spectrum = axes[0, col]
        max_checkpoints = 6
        if len(eigenvalue_history) > max_checkpoints:
            indices = np.linspace(0, len(eigenvalue_history)-1, max_checkpoints, dtype=int)
            selected = [eigenvalue_history[i] for i in indices]
        else:
            selected = eigenvalue_history

        cmap = plt.cm.viridis
        colors = [cmap(i / len(selected)) for i in range(len(selected))]

        for i, entry in enumerate(selected):
            eigenvalues = np.array(entry['eigenvalues'])
            step = entry['step']
            ax_spectrum.plot(range(1, len(eigenvalues)+1), eigenvalues,
                    color=colors[i], marker='o', markersize=3,
                    label=f'Step {step}', alpha=0.8, linewidth=1.5)

        ax_spectrum.set_xlabel("Eigenvalue Index")
        ax_spectrum.set_ylabel("Eigenvalue")
        ax_spectrum.set_title(f"{OPTIMIZER_LABELS.get(opt, opt.upper())} - Spectrum", fontweight='bold')
        ax_spectrum.legend(loc='upper right', fontsize=7)
        ax_spectrum.set_yscale('symlog', linthresh=0.1)

        # Bottom row: Spectral density (histogram)
        ax_density = axes[1, col]

        # Plot density for first, middle, and last checkpoint
        checkpoints_for_density = [selected[0], selected[len(selected)//2], selected[-1]]
        density_colors = ['#3498DB', '#F39C12', '#E74C3C']

        for entry, color in zip(checkpoints_for_density, density_colors):
            eigenvalues = np.array(entry['eigenvalues'])
            step = entry['step']

            # Compute histogram density
            if len(eigenvalues) > 1:
                # Use KDE-like smoothed histogram
                ax_density.hist(eigenvalues, bins=20, alpha=0.5, color=color,
                               label=f'Step {step}', density=True, edgecolor='white')

        ax_density.set_xlabel("Eigenvalue")
        ax_density.set_ylabel("Density")
        ax_density.set_title(f"{OPTIMIZER_LABELS.get(opt, opt.upper())} - Spectral Density", fontweight='bold')
        ax_density.legend(loc='upper right', fontsize=8)

    fig.suptitle("Hessian Eigenvalue Spectrum & Density", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# =============================================================================
# PLOT 5: GRADIENT SUBSPACE ANALYSIS OVER TIME
# =============================================================================

def plot_gradient_subspace_analysis(output_dir: str, optimizers: List[str], save_path: str):
    """
    Plot gradient subspace stability metrics over time.
    Shows subspace overlap and temporal similarity by layer type.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top left: Average subspace overlap
    ax1 = axes[0, 0]
    for opt in optimizers:
        df = load_scalar_metrics(output_dir, opt)
        if df is None or 'avg_subspace_overlap' not in df.columns:
            continue
        color = OPTIMIZER_COLORS.get(opt, '#333')
        label = OPTIMIZER_LABELS.get(opt, opt.upper())
        ax1.plot(df['step'], df['avg_subspace_overlap'], color=color, label=label, linewidth=2)

    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Subspace Overlap")
    ax1.set_title("Gradient Subspace Overlap (Stability)", fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.set_ylim(-0.1, 1.1)
    ax1.axhline(0.5, color='gray', linestyle='--', alpha=0.5)

    # Top right: Temporal gradient similarity
    ax2 = axes[0, 1]
    for opt in optimizers:
        df = load_scalar_metrics(output_dir, opt)
        if df is None or 'avg_temporal_similarity' not in df.columns:
            continue
        color = OPTIMIZER_COLORS.get(opt, '#333')
        label = OPTIMIZER_LABELS.get(opt, opt.upper())
        ax2.plot(df['step'], df['avg_temporal_similarity'], color=color, label=label, linewidth=2)

    ax2.set_xlabel("Training Step")
    ax2.set_ylabel("Temporal Similarity")
    ax2.set_title("Gradient Temporal Similarity", fontweight='bold')
    ax2.legend(loc='lower right')
    ax2.set_ylim(-0.1, 1.1)

    # Bottom left: Subspace overlap by layer type (Attention vs MLP)
    ax3 = axes[1, 0]
    for opt in optimizers:
        layer_metrics = load_layer_metrics(output_dir, opt)
        if layer_metrics is None:
            continue

        steps = [e['step'] for e in layer_metrics if 'subspace_overlap_attn' in e]
        attn_overlaps = [e.get('subspace_overlap_attn', np.nan) for e in layer_metrics if 'subspace_overlap_attn' in e]
        mlp_overlaps = [e.get('subspace_overlap_mlp', np.nan) for e in layer_metrics if 'subspace_overlap_mlp' in e]

        if steps:
            color = OPTIMIZER_COLORS.get(opt, '#333')
            label = OPTIMIZER_LABELS.get(opt, opt.upper())
            ax3.plot(steps, attn_overlaps, color=color, linestyle='-', label=f'{label} Attn', linewidth=1.5)
            ax3.plot(steps, mlp_overlaps, color=color, linestyle='--', label=f'{label} MLP', linewidth=1.5, alpha=0.7)

    ax3.set_xlabel("Training Step")
    ax3.set_ylabel("Subspace Overlap")
    ax3.set_title("Subspace Overlap: Attention vs MLP", fontweight='bold')
    ax3.legend(loc='lower right', fontsize=8, ncol=2)
    ax3.set_ylim(-0.1, 1.1)

    # Bottom right: Temporal similarity by layer type
    ax4 = axes[1, 1]
    for opt in optimizers:
        layer_metrics = load_layer_metrics(output_dir, opt)
        if layer_metrics is None:
            continue

        steps = [e['step'] for e in layer_metrics if 'temporal_sim_attn' in e]
        attn_sims = [e.get('temporal_sim_attn', np.nan) for e in layer_metrics if 'temporal_sim_attn' in e]
        mlp_sims = [e.get('temporal_sim_mlp', np.nan) for e in layer_metrics if 'temporal_sim_mlp' in e]

        if steps:
            color = OPTIMIZER_COLORS.get(opt, '#333')
            label = OPTIMIZER_LABELS.get(opt, opt.upper())
            ax4.plot(steps, attn_sims, color=color, linestyle='-', label=f'{label} Attn', linewidth=1.5)
            ax4.plot(steps, mlp_sims, color=color, linestyle='--', label=f'{label} MLP', linewidth=1.5, alpha=0.7)

    ax4.set_xlabel("Training Step")
    ax4.set_ylabel("Temporal Similarity")
    ax4.set_title("Temporal Similarity: Attention vs MLP", fontweight='bold')
    ax4.legend(loc='lower right', fontsize=8, ncol=2)
    ax4.set_ylim(-0.1, 1.1)

    fig.suptitle("Gradient Subspace Analysis Over Training", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# =============================================================================
# PLOT 6: CONSECUTIVE LAYER GRADIENT SIMILARITY
# =============================================================================

def plot_consecutive_layer_similarity(output_dir: str, optimizer: str, save_path: str):
    """
    Plot gradient similarity between consecutive layers of the same type.
    Shows how similar gradients are between L0.Q and L1.Q, L1.Q and L2.Q, etc.
    """
    layer_metrics = load_layer_metrics(output_dir, optimizer)
    if layer_metrics is None or len(layer_metrics) == 0:
        print(f"No layer metrics found for {optimizer}")
        return

    # Group layers by type
    layer_types = ['c_q', 'c_k', 'c_v', 'c_fc', 'c_proj']
    type_labels = ['Query (Q)', 'Key (K)', 'Value (V)', 'MLP FC', 'Projection']

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, (layer_type, label) in enumerate(zip(layer_types, type_labels)):
        if idx >= len(axes):
            break
        ax = axes[idx]

        # Find all layers of this type
        all_layers = set()
        for entry in layer_metrics:
            if 'grad_norms' in entry:
                for layer in entry['grad_norms'].keys():
                    if layer_type in layer:
                        all_layers.add(layer)

        if len(all_layers) < 2:
            ax.text(0.5, 0.5, f'Not enough {label} layers', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"{label} - Consecutive Similarity")
            continue

        # Sort by layer index
        def get_layer_idx(name):
            for part in name.split('.'):
                try:
                    return int(part)
                except:
                    pass
            return -1

        sorted_layers = sorted(list(all_layers), key=get_layer_idx)

        # Compute similarity between consecutive layers over time
        steps = []
        consecutive_sims = defaultdict(list)

        for entry in layer_metrics:
            if 'grad_norms' not in entry:
                continue
            steps.append(entry['step'])

            # For each consecutive pair
            for i in range(len(sorted_layers) - 1):
                l1, l2 = sorted_layers[i], sorted_layers[i+1]
                idx1, idx2 = get_layer_idx(l1), get_layer_idx(l2)

                # Get temporal similarities if available
                if 'temporal_similarities' in entry:
                    sim1 = entry['temporal_similarities'].get(l1, np.nan)
                    sim2 = entry['temporal_similarities'].get(l2, np.nan)
                    # Use average of temporal similarities as proxy
                    avg_sim = (sim1 + sim2) / 2 if not (np.isnan(sim1) or np.isnan(sim2)) else np.nan
                    consecutive_sims[f"L{idx1}-L{idx2}"].append(avg_sim)

        if not consecutive_sims:
            ax.text(0.5, 0.5, 'No similarity data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"{label} - Consecutive Similarity")
            continue

        # Plot each consecutive pair
        colors = plt.cm.viridis(np.linspace(0, 1, len(consecutive_sims)))
        for (pair_name, sims), color in zip(consecutive_sims.items(), colors):
            ax.plot(steps[:len(sims)], sims, label=pair_name, color=color, linewidth=1.5, alpha=0.8)

        ax.set_xlabel("Training Step")
        ax.set_ylabel("Similarity")
        ax.set_title(f"{label} - Consecutive Layer Similarity", fontweight='bold')
        ax.legend(loc='lower right', fontsize=7)
        ax.set_ylim(-0.1, 1.1)

    # Hide unused subplot
    if len(layer_types) < len(axes):
        for i in range(len(layer_types), len(axes)):
            axes[i].axis('off')

    fig.suptitle(f"Consecutive Layer Gradient Similarity - {OPTIMIZER_LABELS.get(optimizer, optimizer.upper())}",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# =============================================================================
# PLOT 7: CROSS-LAYER SIMILARITY BY MATRIX TYPE
# =============================================================================

def plot_cross_layer_similarity_by_type(output_dir: str, optimizer: str, save_path: str):
    """
    Plot cross-layer gradient similarity separately for Q, K, V, and MLP matrices.
    """
    sim_history = load_cross_layer_sim(output_dir, optimizer)
    if sim_history is None or len(sim_history) == 0:
        print(f"No cross-layer similarity data for {optimizer}")
        return

    # Use the last recorded similarity matrix
    last_entry = sim_history[-1]
    full_sim_matrix = np.array(last_entry['similarity_matrix'])
    layer_names = last_entry.get('layer_names', [])
    step = last_entry['step']

    if len(layer_names) == 0:
        print(f"No layer names in cross-layer data for {optimizer}")
        return

    # Categorize layers
    categories = {
        'Query (Q)': [],
        'Key (K)': [],
        'Value (V)': [],
        'MLP FC': [],
        'MLP/Attn Proj': []
    }

    for i, name in enumerate(layer_names):
        if 'c_q' in name:
            categories['Query (Q)'].append(i)
        elif 'c_k' in name:
            categories['Key (K)'].append(i)
        elif 'c_v' in name:
            categories['Value (V)'].append(i)
        elif 'c_fc' in name:
            categories['MLP FC'].append(i)
        elif 'c_proj' in name:
            categories['MLP/Attn Proj'].append(i)

    # Filter categories with enough layers
    valid_categories = {k: v for k, v in categories.items() if len(v) >= 2}

    if len(valid_categories) == 0:
        print(f"Not enough layers per category for {optimizer}")
        return

    n_cats = len(valid_categories)
    cols = min(3, n_cats)
    rows = (n_cats + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if n_cats == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for ax_idx, (cat_name, indices) in enumerate(valid_categories.items()):
        ax = axes[ax_idx]

        # Extract sub-matrix
        sub_matrix = full_sim_matrix[np.ix_(indices, indices)]

        # Get display names
        sub_names = [get_layer_display_name(layer_names[i]) for i in indices]

        # Plot heatmap
        im = ax.imshow(sub_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')

        ax.set_xticks(range(len(sub_names)))
        ax.set_xticklabels(sub_names, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(len(sub_names)))
        ax.set_yticklabels(sub_names, fontsize=8)
        ax.set_title(f"{cat_name}", fontweight='bold')

        plt.colorbar(im, ax=ax, shrink=0.8)

    # Hide unused subplots
    for i in range(len(valid_categories), len(axes)):
        axes[i].axis('off')

    fig.suptitle(f"Cross-Layer Similarity by Matrix Type - {OPTIMIZER_LABELS.get(optimizer, optimizer.upper())} (Step {step})",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# =============================================================================
# PLOT 8: GRADIENT NORMS BY LAYER TYPE OVER TIME
# =============================================================================

def plot_gradient_norms_by_type(output_dir: str, optimizers: List[str], save_path: str):
    """
    Plot gradient norms evolution grouped by layer type (Q, K, V, FC, Proj).
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    layer_types = [
        ('grad_norm_attn_q', 'Query (Q)'),
        ('grad_norm_attn_k', 'Key (K)'),
        ('grad_norm_attn_v', 'Value (V)'),
        ('grad_norm_mlp_fc', 'MLP FC'),
        ('grad_norm_mlp_proj', 'MLP/Attn Proj')
    ]

    for idx, (metric_key, label) in enumerate(layer_types):
        ax = axes[idx]

        for opt in optimizers:
            layer_metrics = load_layer_metrics(output_dir, opt)
            if layer_metrics is None:
                continue

            steps = [e['step'] for e in layer_metrics if metric_key in e]
            values = [e.get(metric_key, np.nan) for e in layer_metrics if metric_key in e]

            if steps:
                color = OPTIMIZER_COLORS.get(opt, '#333')
                label_opt = OPTIMIZER_LABELS.get(opt, opt.upper())
                ax.plot(steps, values, color=color, label=label_opt, linewidth=1.5)

        ax.set_xlabel("Training Step")
        ax.set_ylabel("Gradient Norm")
        ax.set_title(f"{label} Gradient Norm", fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        ax.set_yscale('log')

    # Hide unused subplot
    axes[-1].axis('off')

    fig.suptitle("Gradient Norms by Layer Type Over Training", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# =============================================================================
# PLOT 9: COMPREHENSIVE DASHBOARD
# =============================================================================

def plot_comprehensive_dashboard(output_dir: str, optimizers: List[str], save_path: str):
    """
    Create a comprehensive dashboard comparing all optimizers.
    """
    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.35, wspace=0.3)

    # Row 1: Loss, Perplexity, Gradient Norm
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    for opt in optimizers:
        df = load_scalar_metrics(output_dir, opt)
        if df is None:
            continue
        color = OPTIMIZER_COLORS.get(opt, '#333')
        label = OPTIMIZER_LABELS.get(opt, opt.upper())

        ax1.plot(df['step'], df['train_loss'], color=color, label=label, linewidth=1.5)

        if 'val_perplexity' in df.columns:
            ax2.plot(df['step'], df['val_perplexity'], color=color, label=label, linewidth=1.5)

        if 'total_grad_norm' in df.columns:
            ax3.plot(df['step'], df['total_grad_norm'], color=color, label=label, linewidth=1.5)

    ax1.set_ylabel("Training Loss")
    ax1.set_title("Training Loss")
    ax1.legend(loc='upper right', fontsize=8)

    ax2.set_ylabel("Validation Perplexity")
    ax2.set_title("Validation Perplexity")
    ax2.legend(loc='upper right', fontsize=8)

    ax3.set_ylabel("Gradient Norm")
    ax3.set_title("Total Gradient Norm")
    ax3.set_yscale('log')
    ax3.legend(loc='upper right', fontsize=8)

    # Row 2: Lambda Max, Effective Rank, Cosine Alignment
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])

    for opt in optimizers:
        df = load_scalar_metrics(output_dir, opt)
        if df is None:
            continue
        color = OPTIMIZER_COLORS.get(opt, '#333')
        label = OPTIMIZER_LABELS.get(opt, opt.upper())

        if 'lambda_max' in df.columns:
            lambda_vals = df['lambda_max'].replace(0, np.nan)
            ax4.plot(df['step'], lambda_vals, color=color, label=label, linewidth=1.5, marker='o', markersize=2)

        if 'avg_effective_rank' in df.columns:
            ax5.plot(df['step'], df['avg_effective_rank'], color=color, label=label, linewidth=1.5)

        if 'cos_sim' in df.columns:
            cos_vals = df['cos_sim'].replace(0, np.nan)
            ax6.plot(df['step'], cos_vals, color=color, label=label, linewidth=1.5, marker='s', markersize=2)

    ax4.set_ylabel(r"$\lambda_{max}$")
    ax4.set_title("Hessian Sharpness")
    ax4.legend(loc='upper right', fontsize=8)

    ax5.set_ylabel("Effective Rank")
    ax5.set_title("Avg Weight Matrix Effective Rank")
    ax5.legend(loc='upper right', fontsize=8)

    ax6.set_ylabel("Cosine Similarity")
    ax6.set_title(r"Step Alignment with $v_{max}$")
    ax6.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax6.legend(loc='upper right', fontsize=8)

    # Row 3: Subspace Overlap, Temporal Similarity, Cross-Layer Similarity
    ax7 = fig.add_subplot(gs[2, 0])
    ax8 = fig.add_subplot(gs[2, 1])
    ax9 = fig.add_subplot(gs[2, 2])

    for opt in optimizers:
        df = load_scalar_metrics(output_dir, opt)
        if df is None:
            continue
        color = OPTIMIZER_COLORS.get(opt, '#333')
        label = OPTIMIZER_LABELS.get(opt, opt.upper())

        if 'avg_subspace_overlap' in df.columns:
            ax7.plot(df['step'], df['avg_subspace_overlap'], color=color, label=label, linewidth=1.5)

        if 'avg_temporal_similarity' in df.columns:
            ax8.plot(df['step'], df['avg_temporal_similarity'], color=color, label=label, linewidth=1.5)

        if 'avg_cross_layer_similarity' in df.columns:
            ax9.plot(df['step'], df['avg_cross_layer_similarity'], color=color, label=label, linewidth=1.5)

    ax7.set_xlabel("Step")
    ax7.set_ylabel("Subspace Overlap")
    ax7.set_title("Gradient Subspace Stability")
    ax7.set_ylim(-0.1, 1.1)
    ax7.legend(loc='lower right', fontsize=8)

    ax8.set_xlabel("Step")
    ax8.set_ylabel("Temporal Similarity")
    ax8.set_title("Gradient Temporal Stability")
    ax8.set_ylim(-0.1, 1.1)
    ax8.legend(loc='lower right', fontsize=8)

    ax9.set_xlabel("Step")
    ax9.set_ylabel("Cross-Layer Similarity")
    ax9.set_title("Gradient Cross-Layer Correlation")
    ax9.legend(loc='upper right', fontsize=8)

    # Row 4: Validation metrics
    ax10 = fig.add_subplot(gs[3, 0])
    ax11 = fig.add_subplot(gs[3, 1])
    ax12 = fig.add_subplot(gs[3, 2])

    for opt in optimizers:
        df = load_scalar_metrics(output_dir, opt)
        if df is None:
            continue
        color = OPTIMIZER_COLORS.get(opt, '#333')
        label = OPTIMIZER_LABELS.get(opt, opt.upper())

        if 'val_loss' in df.columns:
            ax10.plot(df['step'], df['val_loss'], color=color, label=label, linewidth=1.5)

        if 'lr_multiplier' in df.columns:
            ax11.plot(df['step'], df['lr_multiplier'], color=color, label=label, linewidth=1.5)

    ax10.set_xlabel("Step")
    ax10.set_ylabel("Validation Loss")
    ax10.set_title("Validation Loss")
    ax10.legend(loc='upper right', fontsize=8)

    ax11.set_xlabel("Step")
    ax11.set_ylabel("LR Multiplier")
    ax11.set_title("Learning Rate Schedule")
    ax11.legend(loc='upper right', fontsize=8)

    # Sharpness vs Generalization scatter
    for opt in optimizers:
        df = load_scalar_metrics(output_dir, opt)
        if df is None or 'lambda_max' not in df.columns:
            continue
        color = OPTIMIZER_COLORS.get(opt, '#333')
        label = OPTIMIZER_LABELS.get(opt, opt.upper())

        mask = df['lambda_max'] > 0
        if mask.sum() > 0:
            ax12.scatter(df.loc[mask, 'lambda_max'], df.loc[mask, 'val_loss'],
                        color=color, label=label, alpha=0.6, s=30)

    ax12.set_xlabel(r"$\lambda_{max}$")
    ax12.set_ylabel("Validation Loss")
    ax12.set_title("Sharpness vs Generalization")
    ax12.legend(loc='upper right', fontsize=8)

    fig.suptitle("Optimization Geometry: Comprehensive Dashboard", fontsize=16, fontweight='bold')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def generate_all_plots(output_dir: str, optimizers: List[str], report_dir: str):
    """Generate all visualization plots."""
    os.makedirs(report_dir, exist_ok=True)

    print(f"\nGenerating visualizations for optimizers: {optimizers}")
    print(f"Input directory: {output_dir}")
    print(f"Output directory: {report_dir}\n")

    # 1. Validation perplexity
    plot_validation_perplexity(
        output_dir, optimizers,
        os.path.join(report_dir, "validation_perplexity.png")
    )

    # 2. Multi-panel training dynamics
    plot_training_dynamics(
        output_dir, optimizers,
        os.path.join(report_dir, "training_dynamics.png")
    )

    # 3. Gradient norm heatmaps (per optimizer)
    for opt in optimizers:
        plot_gradient_norm_heatmap(
            output_dir, opt,
            os.path.join(report_dir, f"grad_norm_heatmap_{opt}.png")
        )

    # 4. Hessian spectral density
    plot_hessian_spectral_density(
        output_dir, optimizers,
        os.path.join(report_dir, "hessian_spectral_density.png")
    )

    # 5. Gradient subspace analysis
    plot_gradient_subspace_analysis(
        output_dir, optimizers,
        os.path.join(report_dir, "gradient_subspace_analysis.png")
    )

    # 6. Consecutive layer similarity (per optimizer)
    for opt in optimizers:
        plot_consecutive_layer_similarity(
            output_dir, opt,
            os.path.join(report_dir, f"consecutive_layer_sim_{opt}.png")
        )

    # 7. Cross-layer similarity by type (per optimizer)
    for opt in optimizers:
        plot_cross_layer_similarity_by_type(
            output_dir, opt,
            os.path.join(report_dir, f"cross_layer_sim_by_type_{opt}.png")
        )

    # 8. Gradient norms by layer type
    plot_gradient_norms_by_type(
        output_dir, optimizers,
        os.path.join(report_dir, "gradient_norms_by_type.png")
    )

    # 9. Comprehensive dashboard
    plot_comprehensive_dashboard(
        output_dir, optimizers,
        os.path.join(report_dir, "comprehensive_dashboard.png")
    )

    print(f"\nAll visualizations saved to {report_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Generate optimization geometry visualizations")
    parser.add_argument("--input-dir", type=str, default="geometry_results",
                       help="Directory containing geometry metrics")
    parser.add_argument("--output-dir", type=str, default="reports/geometry",
                       help="Directory to save visualization plots")
    parser.add_argument("--optimizers", type=str, nargs="+",
                       default=["sgd", "adam", "adamw"],
                       help="List of optimizers to compare")

    args = parser.parse_args()
    generate_all_plots(args.input_dir, args.optimizers, args.output_dir)


if __name__ == "__main__":
    main()
