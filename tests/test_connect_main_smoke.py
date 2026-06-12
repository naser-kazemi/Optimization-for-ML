"""End-to-end smoke of connect.run_experiment through the real Hydra config,
with prepare_data stubbed out (no download)."""

import os
import sys

import hydra
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import connect


def test_run_experiment_end_to_end_via_hydra_config(tmp_path, monkeypatch):
    if not hasattr(torch.optim, "Muon"):
        import pytest
        pytest.skip("torch.optim.Muon not available in this torch build")

    torch.manual_seed(0)
    train_data = torch.randint(0, 256, (4000,))
    val_data = torch.randint(0, 256, (2000,))
    monkeypatch.setattr(connect, "prepare_data",
                        lambda **kw: (train_data, val_data, None, None))

    overrides = [
        "model.depth=2", "model.aspect_ratio=16", "model.head_dim=16",
        "model.max_seq_len=32", "model.vocab_size=256",
        "training.device=cpu", "training.device_batch_size=2",
        "training.total_batch_size=64",
        "connect.seeds=[1]", "connect.optimizers=[adamw,muon]",
        "connect.within_pairs=false", "connect.cross_pairs=true",
        "connect.endpoint_steps=12", "connect.endpoint_eval_every=4",
        "connect.endpoint_patience=1", "connect.endpoint_warmup_steps=2",
        "connect.endpoint_decay_steps=3",
        "connect.measure_batches=2", "connect.val_batches=2",
        "connect.measure_batch_size=2", "connect.relax_batch_size=2",
        "connect.compute_saddle_curvature=false",
        "connect.autoneb.n_pivots_interior=3",
        "connect.autoneb.cycles_lr=[0.1]",
        "connect.autoneb.steps_per_cycle=3", "connect.autoneb.n_interp=2",
        f"connect.endpoint_dir={tmp_path}/endpoints/",
        f"logging.path_csv={tmp_path}/path.csv",
        f"logging.summary_csv={tmp_path}/summary.csv",
        f"logging.cycles_csv={tmp_path}/cycles.csv",
        "logging.use_wandb=false",
    ]
    with hydra.initialize(version_base=None, config_path="../config"):
        cfg = hydra.compose(config_name="connect", overrides=overrides)

    results = connect.run_experiment(cfg)

    assert len(results) == 1  # one adamw<->muon cross pair
    row = results[0]
    assert {row["optimizer_A"], row["optimizer_B"]} == {"adamw", "muon"}
    assert torch.isfinite(torch.tensor(row["barrier_train_connected"]))
    for f in ("path.csv", "summary.csv", "cycles.csv"):
        assert (tmp_path / f).exists(), f
    cached = sorted(os.listdir(tmp_path / "endpoints"))
    assert any(c.startswith("adamw_seed1_lr") for c in cached)
    assert any(c.startswith("muon_seed1_lr") for c in cached)

    # second invocation must reuse the cache
    results2 = connect.run_experiment(cfg)
    assert results2[0]["loss_A"] == row["loss_A"]
    assert results2[0]["loss_B"] == row["loss_B"]
