import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Set publication-quality style parameters
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 16,
    "figure.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--"
})

def main():
    csv_path = "metrics.csv"
    output_dir = "reports"
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return
        
    df = pd.read_csv(csv_path)
    
    # Identify distinct runs based on step resets (step = 0)
    # We want to find the longest run to plot
    df['run_id'] = (df['step'] == 0).cumsum()
    run_lengths = df.groupby('run_id').size()
    best_run_id = run_lengths.idxmax()
    
    print(f"Identified {df['run_id'].nunique()} runs in the CSV file.")
    print(f"Run sizes:\n{run_lengths}")
    print(f"Selecting Run ID {best_run_id} (size {run_lengths[best_run_id]}) as the main training run to plot.")
    
    run_df = df[df['run_id'] == best_run_id].copy().reset_index(drop=True)
    
    # Let's clean up any duplicate steps within the run if they exist (like the duplicate step 0s)
    run_df = run_df.drop_duplicates(subset=['step'], keep='last').reset_index(drop=True)
    
    steps = run_df['step']
    train_loss = run_df['train_loss']
    val_loss = run_df['val_loss']
    perplexity = run_df['val_perplexity']
    lambda_max = run_df['lambda_max']
    cos_sim = run_df['cos_sim']
    lr_mult = run_df['lr_multiplier']
    
    # -------------------------------------------------------------
    # Plot 1: Loss & Perplexity (with LR Multiplier)
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Subplot 1: Train & Val Loss
    ax1.plot(steps, train_loss, label="Training Loss", color="#1F77B4", linewidth=2.5, marker='o', markersize=5)
    ax1.plot(steps, val_loss, label="Validation Loss", color="#FF7F0E", linewidth=2.5, marker='s', markersize=5)
    ax1.set_ylabel("Cross Entropy Loss", fontweight='bold')
    ax1.set_title("Training and Validation Loss Trajectories", pad=10)
    ax1.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none")
    
    # Subplot 2: Perplexity & LR Multiplier
    color_perp = "#2CA02C"
    ax2.plot(steps, perplexity, label="Validation Perplexity", color=color_perp, linewidth=2.5, marker='^', markersize=5)
    ax2.set_ylabel("Perplexity", color=color_perp, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=color_perp)
    
    # Add a dual axis for the learning rate multiplier to highlight the WSD phases
    ax2_right = ax2.twinx()
    color_lr = "#D62728"
    ax2_right.plot(steps, lr_mult, label="LR Multiplier (WSD)", color=color_lr, linewidth=1.8, linestyle='--', marker='x', alpha=0.8)
    ax2_right.set_ylabel("LR Multiplier", color=color_lr, fontweight='bold')
    ax2_right.tick_params(axis='y', labelcolor=color_lr)
    ax2_right.set_ylim(-0.05, 1.05)
    
    # Set labels & titles
    ax2.set_xlabel("Optimization Steps", fontweight='bold')
    ax2.set_title("Validation Perplexity and WSD Learning Rate Schedule", pad=10)
    
    # Combine legends for ax2 and ax2_right
    lines, labels = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_right.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="upper right", frameon=True, facecolor="white", edgecolor="none")
    
    # Annotate WSD Phases on ax2_right
    # Let's find where Warmup, Stable, Decay occur
    # Step 0-6: LR is 0.0 (Warmup) or building up, Step 7-13: LR is 1.0 (Stable), Step 14-17: LR decays
    ax2.axvspan(0, 6, color='#CCCCCC', alpha=0.15)
    ax2.text(3, perplexity.max() * 0.9, "WARMUP /\nSTABILIZE", color="#777777", fontsize=9, ha="center", fontweight="bold")
    ax2.axvspan(7, 13, color='#94C11F', alpha=0.08)
    ax2.text(10, perplexity.max() * 0.9, "STABLE PHASE\n(Peak Learning Rate)", color="#558833", fontsize=9, ha="center", fontweight="bold")
    ax2.axvspan(13, steps.max(), color='#FFDDDD', alpha=0.15)
    ax2.text(15, perplexity.max() * 0.9, "DECAY PHASE\n(Cosine Warmdown)", color="#AA3333", fontsize=9, ha="center", fontweight="bold")
    
    plt.tight_layout()
    plot1_path = os.path.join(output_dir, "loss_curves.png")
    plt.savefig(plot1_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved loss & perplexity curves to: {plot1_path}")
    
    # -------------------------------------------------------------
    # Plot 2: Hessian Geometry (Lambda Max & Cosine Alignment)
    # -------------------------------------------------------------
    fig, (ax3, ax4) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Subplot 3: Lambda Max (Curvature Sharpness)
    # Filter step > 5 to show training curvature (step < 6 has initial/pre-warmup spikes or negative values)
    ax3.plot(steps, lambda_max, label="Top Eigenvalue ($\lambda_{max}$)", color="#9467BD", linewidth=2.5, marker='D', markersize=5)
    ax3.set_ylabel("Hessian Max Eigenvalue ($\lambda_{max}$)", fontweight='bold')
    ax3.set_title("Loss Landscape Curvature Sharpness ($\lambda_{max}$)", pad=10)
    ax3.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none")
    
    # Annotate peak curvature
    peak_idx = lambda_max.idxmax()
    peak_step = steps[peak_idx]
    peak_val = lambda_max[peak_idx]
    ax3.annotate(f"Peak Sharpness: {peak_val:.2f}\n(Step {peak_step})", 
                 xy=(peak_step, peak_val), 
                 xytext=(peak_step + 1.5, peak_val - 0.2),
                 arrowprops=dict(facecolor='black', shrink=0.08, width=1, headwidth=6),
                 fontsize=9, fontweight="bold")
                 
    # Highlight the drop in sharpness during decay phase
    ax3.axvspan(13, steps.max(), color='#FFDDDD', alpha=0.15)
    ax3.text(15, lambda_max.min() + 0.2, "Sharpness Drop\nin Decay Phase", color="#AA3333", fontsize=9, ha="center", fontweight="bold")
    
    # Subplot 4: Cosine Alignment Similarity
    # Note: step 0-6 has cos_sim = 0 because training hasn't progressed to actual parameter updates yet
    valid_alignment_steps = steps[steps >= 7]
    valid_cos_sim = cos_sim[steps >= 7]
    
    ax4.plot(valid_alignment_steps, valid_cos_sim, label="Cosine Similarity $\cos(\Delta\\theta, v_{max})$", color="#8C564B", linewidth=2, marker='o', markersize=5)
    ax4.axhline(0, color="grey", linestyle="--", linewidth=1, alpha=0.7)
    ax4.set_ylabel("Cosine Alignment Score", fontweight='bold')
    ax4.set_xlabel("Optimization Steps", fontweight='bold')
    ax4.set_title("Optimizer Step Alignment with Dominant Hessian Curvature Direction ($v_{max}$)", pad=10)
    ax4.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none")
    
    plt.tight_layout()
    plot2_path = os.path.join(output_dir, "hessian_geometry.png")
    plt.savefig(plot2_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved Hessian geometry metrics to: {plot2_path}")

if __name__ == "__main__":
    main()
