"""Tier 1a: analytic potentials — gradient correctness and ground-truth barrier."""

import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from optim.potentials import double_well_2d, grid_search_mep, muller_brown_2d


def _finite_diff_grad(energy, coords, eps=1e-5):
    g = torch.zeros_like(coords)
    for i in range(coords.numel()):
        cp = coords.clone(); cp[i] += eps
        cm = coords.clone(); cm[i] -= eps
        lp, _ = energy(cp)
        lm, _ = energy(cm)
        g[i] = (lp - lm) / (2 * eps)
    return g


def test_double_well_gradient_matches_finite_diff():
    energy = double_well_2d()
    for pt in [[0.3, 0.4], [-0.7, 0.2], [1.1, -0.5]]:
        coords = torch.tensor(pt, dtype=torch.float64)
        _, grad = energy(coords)
        fd = _finite_diff_grad(energy, coords)
        assert torch.allclose(grad, fd, atol=1e-5), f"{grad} vs {fd}"


def test_muller_brown_gradient_matches_finite_diff():
    energy = muller_brown_2d()
    for pt in [[-0.5, 1.4], [0.0, 0.5], [0.6, 0.03]]:
        coords = torch.tensor(pt, dtype=torch.float64)
        _, grad = energy(coords)
        fd = _finite_diff_grad(energy, coords, eps=1e-4)
        assert torch.allclose(grad, fd, rtol=1e-3, atol=1e-3), f"{grad} vs {fd}"


def test_double_well_grid_barrier_is_one():
    energy = double_well_2d()
    barrier, _ = grid_search_mep(
        energy, bounds=((-2.0, 2.0), (-2.0, 2.0)), resolution=201,
        start=(-1.0, 0.0), end=(1.0, 0.0),
    )
    # Exact saddle barrier is 1.0 at (0, 0); grid discretization keeps it close.
    assert abs(barrier - 1.0) < 0.05, barrier
