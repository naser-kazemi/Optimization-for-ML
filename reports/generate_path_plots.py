"""
Plot the AutoNEB mode-connectivity results written by connect.py.

Reads `path_profile.csv` (per-sample loss along the connected and linear paths;
columns: pair, kind [within|cross], curve [connected|linear],
landscape [train|val], position, loss) and, if present, `connect_summary.csv`
(per-pair barrier stats). Produces one loss-along-path figure per
(kind, landscape) group, comparing the AutoNEB path against the naive
linear-interpolation baseline, averaged over pairs with a +/- std band.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
    "grid.linestyle": "--",
})

_STYLE = {
    "connected": {"color": "#1F77B4", "label": "AutoNEB path"},
    "linear": {"color": "#D62728", "label": "Linear interpolation"},
}

_KIND_TITLE = {
    "within": "same optimizer, different inits",
    "cross": "different optimizers, shared init",
}


def _aggregate(df, curve, grid):
    """Interpolate each pair's profile onto a common grid; return mean, std."""
    sub = df[df["curve"] == curve]
    if sub.empty:
        return None, None
    curves = []
    for _, g in sub.groupby("pair"):
        g = g.sort_values("position")
        curves.append(np.interp(grid, g["position"].to_numpy(), g["loss"].to_numpy()))
    arr = np.vstack(curves)
    return arr.mean(axis=0), arr.std(axis=0)


def _plot_group(df, summary, kind, landscape, output_dir):
    grid = np.linspace(0.0, 1.0, 101)
    fig, ax = plt.subplots(figsize=(10, 6))
    plotted = False
    for curve, style in _STYLE.items():
        mean, std = _aggregate(df, curve, grid)
        if mean is None:
            continue
        plotted = True
        ax.plot(grid, mean, color=style["color"], linewidth=2.5, label=style["label"])
        ax.fill_between(grid, mean - std, mean + std, color=style["color"], alpha=0.15)
    if not plotted:
        plt.close()
        return

    title_extra = ""
    if summary is not None:
        s = summary[summary["kind"] == kind]
        bc = s[f"barrier_{landscape}_connected"]
        bl = s[f"barrier_{landscape}_linear"]
        if len(s):
            title_extra = (f"  |  barrier: AutoNEB {bc.mean():.3f}±{bc.std():.3f}, "
                           f"linear {bl.mean():.3f}±{bl.std():.3f}")

    ax.set_xlabel("Normalized path position (arc length)", fontweight="bold")
    ax.set_ylabel("Cross Entropy Loss", fontweight="bold")
    ax.set_title(f"{_KIND_TITLE.get(kind, kind)} — {landscape} landscape{title_extra}", pad=10)
    ax.legend(loc="upper center", frameon=True, facecolor="white", edgecolor="none")

    plt.tight_layout()
    out_path = os.path.join(output_dir, f"path_profile_{kind}_{landscape}.png")
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved path profile to: {out_path}")


def main():
    path_csv = "path_profile.csv"
    summary_csv = "connect_summary.csv"
    output_dir = "reports"
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(path_csv):
        print(f"Error: {path_csv} not found. Run connect.py first.")
        return

    df = pd.read_csv(path_csv)
    summary = pd.read_csv(summary_csv) if os.path.exists(summary_csv) else None

    for kind in sorted(df["kind"].unique()):
        for landscape in sorted(df[df["kind"] == kind]["landscape"].unique()):
            sub = df[(df["kind"] == kind) & (df["landscape"] == landscape)]
            _plot_group(sub, summary, kind, landscape, output_dir)


if __name__ == "__main__":
    main()
