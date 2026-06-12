"""Tier 1b: NEB / AutoNEB on analytic potentials with known minimum-energy paths.

This is the primary correctness gate for the NEB math, fully on CPU.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from optim.autoneb import AutoNEBConfig, autoneb_connect, insert_pivots
from optim.neb import NEBConfig, Path, measure_path, neb_optimize
from optim.potentials import double_well_2d, grid_search_mep, muller_brown_2d


def _scaled(energy, factor):
    def e(coords):
        loss, grad = energy(coords)
        return loss / factor, (grad / factor if grad is not None else None)
    return e


def test_linear_init_endpoints_and_spacing():
    x0 = torch.tensor([-1.0, 0.0])
    x1 = torch.tensor([1.0, 0.0])
    path = Path.linear_init(x0, x1, n_interior=8)
    assert path.n_pivots == 10
    assert torch.allclose(path.coords[0], x0)
    assert torch.allclose(path.coords[-1], x1)
    seg = path.segment_lengths()
    assert torch.allclose(seg, seg.mean() * torch.ones_like(seg), atol=1e-5)


def test_reparametrize_equalizes_spacing_and_keeps_endpoints():
    x0 = torch.tensor([-1.0, 0.0])
    x1 = torch.tensor([1.0, 0.0])
    path = Path.linear_init(x0, x1, n_interior=8)
    # Bunch the interior pivots up near the start, then re-space.
    path.coords[1:-1] = path.coords[1:-1] * 0.1 + x0 * 0.9
    path.reparametrize()
    seg = path.segment_lengths()
    assert torch.allclose(seg, seg.mean() * torch.ones_like(seg), atol=1e-4)
    assert torch.allclose(path.coords[0], x0)
    assert torch.allclose(path.coords[-1], x1)


def test_neb_double_well_reduces_a_detour_barrier():
    energy = double_well_2d()
    x0 = torch.tensor([-1.0, 0.0])
    x1 = torch.tensor([1.0, 0.0])

    path = Path.linear_init(x0, x1, n_interior=9)
    path.coords[1:-1, 1] += 1.5  # push interior pivots off the valley (high detour)
    barrier_before = measure_path(path, energy, n_interp=5).barrier()

    neb_optimize(path, energy, NEBConfig(n_steps=600, lr=0.05, momentum=0.9))
    profile = measure_path(path, energy, n_interp=5)
    barrier_after = profile.barrier()

    assert barrier_after < barrier_before
    assert abs(barrier_after - 1.0) < 0.15, barrier_after

    # Endpoints stayed fixed.
    assert torch.allclose(path.coords[0], x0)
    assert torch.allclose(path.coords[-1], x1)

    # NEB barrier agrees with the dense-grid ground truth.
    grid_barrier, _ = grid_search_mep(
        energy, bounds=((-2.0, 2.0), (-2.0, 2.0)), resolution=201,
        start=(-1.0, 0.0), end=(1.0, 0.0),
    )
    assert abs(barrier_after - grid_barrier) < 0.15


def test_autoneb_muller_brown_below_linear():
    # Scale down the steep Müller-Brown potential for friendlier step sizes.
    energy = _scaled(muller_brown_2d(), factor=100.0)
    # Two of the well locations (approximate).
    x0 = torch.tensor([-0.558, 1.442])
    x1 = torch.tensor([0.623, 0.028])

    linear = Path.linear_init(x0, x1, n_interior=11)
    barrier_linear = measure_path(linear, energy, n_interp=5).barrier()

    cfg = AutoNEBConfig(
        n_pivots_interior=11,
        cycles_lr=(0.01, 0.005, 0.002),
        steps_per_cycle=400,
        momentum=0.9,
        insert_threshold=0.25,
        max_pivots=40,
        n_interp=5,
    )
    _, profile = autoneb_connect(x0, x1, energy, energy, cfg)
    barrier_neb = profile.barrier()

    # The straight line crosses the central ridge; the curved MEP is lower.
    assert barrier_neb < barrier_linear, (barrier_neb, barrier_linear)


def _dense_profile(path, losses, n_interp):
    from optim.neb import PathProfile
    m = path.n_pivots
    n = (m - 1) * (n_interp + 1) + 1
    assert len(losses) == n
    return PathProfile(positions=torch.linspace(0, 1, n), losses=torch.tensor(losses))


def test_insert_pivots_catches_hidden_bump_between_equal_loss_pivots():
    # a bump between two equal-loss pivots is invisible to pivot losses alone
    x0 = torch.tensor([0.0, 0.0])
    x1 = torch.tensor([3.0, 0.0])
    path = Path.linear_init(x0, x1, n_interior=2)  # 4 pivots, 3 segments
    k = 2  # dense layout: pivot i at index 3*i, sub-samples at alphas 1/3, 2/3
    losses = [
        0.0, 0.0, 0.0,
        0.0, 1.0, 0.9,   # hidden bump, equal pivot losses
        0.0, 0.0, 0.0,
        0.0,
    ]
    profile = _dense_profile(path, losses, n_interp=k)

    new_path = insert_pivots(path, profile, n_interp=k, threshold_frac=0.1,
                             insert_count=2, max_pivots=32)
    assert new_path.n_pivots == path.n_pivots + 1
    assert torch.allclose(new_path.coords[2], torch.tensor([1.0 + 1.0 / 3.0, 0.0]), atol=1e-5)
    w = new_path.target_weights
    assert abs(float(w[1]) / float(w[2]) - 0.5) < 1e-5


def test_insert_pivots_respects_count_and_cap():
    x0 = torch.tensor([0.0, 0.0])
    x1 = torch.tensor([3.0, 0.0])
    path = Path.linear_init(x0, x1, n_interior=2)  # 4 pivots
    k = 1
    losses = [0.0, 1.0, 0.1, 1.0, 0.1, 1.0, 0.0]
    profile = _dense_profile(path, losses, n_interp=k)

    one = insert_pivots(path, profile, n_interp=k, threshold_frac=0.1,
                        insert_count=1, max_pivots=32)
    assert one.n_pivots == path.n_pivots + 1

    capped = insert_pivots(path, profile, n_interp=k, threshold_frac=0.1,
                           insert_count=3, max_pivots=4)
    assert capped.n_pivots <= 4


def test_reparametrize_preserves_target_weights():
    x0 = torch.tensor([0.0, 0.0])
    x1 = torch.tensor([4.0, 0.0])
    coords = torch.stack([x0, torch.tensor([3.0, 0.0]), torch.tensor([3.5, 0.0]), x1])
    path = Path(coords, target_weights=torch.tensor([2.0, 1.0, 1.0]))
    path.reparametrize()
    seg = path.segment_lengths()
    assert torch.allclose(seg, torch.tensor([2.0, 1.0, 1.0]), atol=1e-5)


def test_tangent_blends_at_local_maximum():
    # pivot 1 is a local max: tangent must blend toward the higher-loss side
    coords = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
    path = Path(coords)
    losses = torch.tensor([0.4, 1.0, 0.0])
    tang = path.tangents(losses)
    t = tang[1]
    assert abs(float(t.norm()) - 1.0) < 1e-5
    t_bwd = (coords[1] - coords[0]) / (coords[1] - coords[0]).norm()
    t_fwd = (coords[2] - coords[1]) / (coords[2] - coords[1]).norm()
    cos_bwd = float(t @ t_bwd)
    cos_fwd = float(t @ t_fwd)
    assert cos_bwd > 0.0 and cos_fwd > 0.0
    assert cos_bwd > cos_fwd
