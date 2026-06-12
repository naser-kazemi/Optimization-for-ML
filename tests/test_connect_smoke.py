"""Tier 3: end-to-end smoke of the connect pipeline on CPU with synthetic data.

Bypasses the HuggingFace download by injecting a random token tensor, and uses
plain torch optimizers so no Hydra config is needed.
"""

import os
import sys
from contextlib import nullcontext

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data import make_dataloader, make_fixed_eval_batches
from optim.autoneb import AutoNEBConfig, autoneb_connect
from optim.neb import Path, measure_path
from optim.param_vector import flatten_state_dict, make_nn_energy, make_nn_energy_stream
from connect import make_pairs, train_endpoint
from utils.logging import CSVLogger
from conftest import make_tiny_gpt


def test_connect_pipeline_smoke(tmp_path):
    model_a, cfg = make_tiny_gpt(seed=11)
    seq_len = cfg.sequence_len

    # Synthetic corpus (no network).
    torch.manual_seed(0)
    train_data = torch.randint(0, cfg.vocab_size, (5000,))
    val_data = torch.randint(0, cfg.vocab_size, (2000,))

    eval_batches = make_fixed_eval_batches(val_data, 4, seq_len, 4,
                                           torch.device("cpu"), seed=123)

    # Shared init: build B from A's initial weights.
    init_vec, specs = flatten_state_dict(model_a.state_dict())
    model_b, _ = make_tiny_gpt(seed=11)

    from optim.param_vector import load_vector_into_model
    load_vector_into_model(model_b, init_vec, specs)

    loader_a = make_dataloader(train_data, 4, seq_len, torch.device("cpu"))
    loader_b = make_dataloader(train_data, 4, seq_len, torch.device("cpu"))
    train_endpoint(model_a, loader_a, [torch.optim.AdamW(model_a.parameters(), lr=1e-3)],
                   nullcontext(), grad_accum_steps=1, max_steps=5)
    train_endpoint(model_b, loader_b, [torch.optim.SGD(model_b.parameters(), lr=1e-2)],
                   nullcontext(), grad_accum_steps=1, max_steps=5)

    theta_a, _ = flatten_state_dict(model_a.state_dict())
    theta_b, _ = flatten_state_dict(model_b.state_dict())

    relax_loader = make_dataloader(train_data, 4, seq_len, torch.device("cpu"), seed=7)
    energy_mb = make_nn_energy_stream(model_a, specs, relax_loader,
                                      nullcontext(), with_grad=True)
    energy_mb_eval = make_nn_energy_stream(model_a, specs, relax_loader,
                                           nullcontext(), with_grad=False)
    energy_full = make_nn_energy(model_a, specs, eval_batches, torch.device("cpu"),
                                 nullcontext(), with_grad=False)

    cfg_neb = AutoNEBConfig(n_pivots_interior=3, cycles_lr=(0.1, 0.05), steps_per_cycle=5,
                            insert_count=2, insert_threshold=0.2, max_pivots=16, n_interp=2)
    n_pivots_initial = cfg_neb.n_pivots_interior + 2

    path, profile = autoneb_connect(theta_a, theta_b, energy_mb, energy_full, cfg_neb,
                                    energy_loss_only=energy_mb_eval)
    linear = Path.linear_init(theta_a, theta_b, cfg_neb.n_pivots_interior)
    linear_profile = measure_path(linear, energy_full, n_interp=cfg_neb.n_interp)

    barrier_connected = profile.barrier()
    barrier_linear = linear_profile.barrier()

    assert torch.isfinite(torch.tensor(barrier_connected))
    assert torch.isfinite(torch.tensor(barrier_linear))
    # autoneb_connect seeds its best-path tracker with the linear init,
    # so it can never return worse than the linear baseline
    assert barrier_connected <= barrier_linear + 1e-6
    assert path.n_pivots >= n_pivots_initial

    # Logging round-trip.
    path_csv = tmp_path / "path_profile.csv"
    logger = CSVLogger(str(path_csv))
    for pos, loss in zip(profile.positions.tolist(), profile.losses.tolist()):
        logger.log({"repeat": 0, "kind": "connected", "position": pos, "loss": loss})
    assert path_csv.exists()
    assert len(path_csv.read_text().strip().splitlines()) >= 2  # header + >=1 row


def test_make_pairs_within_and_cross():
    pairs = make_pairs(["adamw", "sgd"], [1, 2, 3], within=True, cross=True)
    within = [p for p in pairs if p["kind"] == "within"]
    cross = [p for p in pairs if p["kind"] == "cross"]
    assert len(within) == 6 and len(cross) == 3
    assert all(p["opt_a"] == p["opt_b"] and p["seed_a"] != p["seed_b"] for p in within)
    assert all(p["opt_a"] != p["opt_b"] and p["seed_a"] != p["seed_b"] for p in cross)
    assert {(p["seed_a"], p["seed_b"]) for p in cross} == {(1, 2), (2, 3), (3, 1)}

    # single seed: cross pair degenerates to a shared init
    only_cross = make_pairs(["adamw", "sgd"], [1], within=False, cross=True)
    assert len(only_cross) == 1
    assert only_cross[0]["seed_a"] == only_cross[0]["seed_b"] == 1


def test_seeded_dataloader_is_reproducible():
    data = torch.arange(2000)
    a = make_dataloader(data, 4, 16, torch.device("cpu"), seed=5)
    b = make_dataloader(data, 4, 16, torch.device("cpu"), seed=5)
    for _ in range(3):
        xa, ya = next(a)
        xb, yb = next(b)
        assert torch.equal(xa, xb) and torch.equal(ya, yb)


def test_train_endpoint_early_stops_on_plateau():
    model, cfg = make_tiny_gpt(seed=3)
    loader = make_dataloader(torch.randint(0, cfg.vocab_size, (3000,)), 2,
                             cfg.sequence_len, torch.device("cpu"), seed=1)
    opt = [torch.optim.AdamW(model.parameters(), lr=1e-3)]
    # one improvement, then a plateau: decay triggers at step 20, stop at 25
    evals = iter([5.0, 4.0, 4.0, 4.0, 4.0, 4.0])
    steps_run = train_endpoint(model, loader, opt, nullcontext(),
                               grad_accum_steps=1, max_steps=100,
                               eval_fn=lambda: next(evals), eval_every=5,
                               patience=2, min_delta=0.01,
                               warmup_steps=2, decay_steps=5)
    assert steps_run == 25

    model2, cfg2 = make_tiny_gpt(seed=3)
    opt2 = [torch.optim.AdamW(model2.parameters(), lr=1e-3)]
    steps_run2 = train_endpoint(model2, loader, opt2, nullcontext(),
                                grad_accum_steps=1, max_steps=8)
    assert steps_run2 == 8
