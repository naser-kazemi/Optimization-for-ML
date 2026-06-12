"""Tier 2: NEB machinery wired to the actual GPT loss (tiny model, mostly CPU)."""

import os
import sys
from contextlib import nullcontext

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from optim.autoneb import AutoNEBConfig, autoneb_connect
from optim.param_vector import flatten_state_dict, make_nn_energy
from conftest import make_tiny_gpt, make_token_batches

_CPU = torch.device("cpu")


def test_energy_matches_direct_model_call():
    model, cfg = make_tiny_gpt(seed=3)
    batches = make_token_batches(cfg, n_batches=3, batch_size=4, seed=3)
    vec, specs = flatten_state_dict(model.state_dict())

    energy = make_nn_energy(model, specs, batches, _CPU, nullcontext(), with_grad=False)
    loss_via_energy, _ = energy(vec)

    model.eval()
    with torch.no_grad():
        direct = sum(model(x, y).item() for x, y in batches) / len(batches)
    assert abs(loss_via_energy - direct) < 1e-5


def test_energy_grad_matches_finite_diff_directional():
    model, cfg = make_tiny_gpt(seed=4)
    batches = make_token_batches(cfg, n_batches=2, batch_size=4, seed=4)
    vec, specs = flatten_state_dict(model.state_dict())

    energy_g = make_nn_energy(model, specs, batches, _CPU, nullcontext(), with_grad=True)
    energy_0 = make_nn_energy(model, specs, batches, _CPU, nullcontext(), with_grad=False)

    _, grad = energy_g(vec)
    torch.manual_seed(0)
    eps = 1e-3
    for _ in range(3):
        d = torch.randn_like(vec)
        d = d / d.norm()
        lp, _ = energy_0(vec + eps * d)
        lm, _ = energy_0(vec - eps * d)
        fd = (lp - lm) / (2 * eps)
        analytic = float(grad @ d)
        assert abs(fd - analytic) < 5e-2 + 0.05 * abs(analytic), (fd, analytic)


def test_self_connection_barrier_is_zero():
    model, cfg = make_tiny_gpt(seed=5)
    batches = make_token_batches(cfg, n_batches=2, batch_size=4, seed=5)
    vec, specs = flatten_state_dict(model.state_dict())

    energy_g = make_nn_energy(model, specs, batches, _CPU, nullcontext(), with_grad=True)
    energy_0 = make_nn_energy(model, specs, batches, _CPU, nullcontext(), with_grad=False)

    cfg_neb = AutoNEBConfig(n_pivots_interior=5, cycles_lr=(0.1,), steps_per_cycle=5,
                            insert_threshold=0.2, max_pivots=16, n_interp=3)
    # Connect a point to itself: the path collapses, the loss is flat.
    _, profile = autoneb_connect(vec, vec.clone(), energy_g, energy_0, cfg_neb)
    assert abs(profile.barrier()) < 1e-4


def test_energy_full_is_deterministic():
    model, cfg = make_tiny_gpt(seed=6)
    batches = make_token_batches(cfg, n_batches=2, batch_size=4, seed=6)
    vec, specs = flatten_state_dict(model.state_dict())
    energy = make_nn_energy(model, specs, batches, _CPU, nullcontext(), with_grad=False)
    l1, _ = energy(vec)
    l2, _ = energy(vec)
    assert l1 == l2


@pytest.mark.gpu
def test_saddle_curvature_runs_on_accelerator():
    if not (torch.backends.mps.is_available() or torch.cuda.is_available()):
        pytest.skip("needs mps/cuda for double-backward HVP")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda")

    from optim.param_vector import load_vector_into_model
    from utils.hvp import power_iteration

    model, cfg = make_tiny_gpt(seed=7)
    model = model.to(device)
    batches = [(x.to(device), y.to(device)) for x, y in
               make_token_batches(cfg, n_batches=1, batch_size=2, seed=7)]
    vec, specs = flatten_state_dict(model.state_dict())
    load_vector_into_model(model, vec, specs)

    def _loader():
        while True:
            yield batches[0]

    lambda_max, _ = power_iteration(model, _loader(), num_iterations=3, device=device.type)
    assert lambda_max == lambda_max  # finite (not NaN)
