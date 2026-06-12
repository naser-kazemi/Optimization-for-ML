"""Unit tests for the Adam+Newton-Schulz family (optim/adamns.py)."""

import os
import sys

import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from optim.adamns import AdamNSFamily, zeropower_via_newtonschulz5
from train import build_optimizer
from conftest import make_tiny_gpt


def test_newtonschulz_flattens_singular_values():
    torch.manual_seed(0)
    G = torch.randn(8, 16) @ torch.diag(torch.logspace(0, -2, 16)) @ torch.randn(16, 16)
    O = zeropower_via_newtonschulz5(G, steps=5)
    sv = torch.linalg.svdvals(O.float())
    # NS5 is a loose orthogonalization: singular values land in a band around 1
    assert float(sv.max()) < 1.6 and float(sv.min()) > 0.3, sv
    assert float((O * G).sum()) > 0


def _one_step(ns_point, seed=0):
    torch.manual_seed(seed)
    p = torch.nn.Parameter(torch.randn(8, 16))
    opt = AdamNSFamily([p], lr=0.01, ns_point=ns_point)
    x = torch.randn(4, 16)
    for _ in range(3):
        opt.zero_grad()
        loss = (x @ p.T).pow(2).sum()
        loss.backward()
        opt.step()
    return p.detach()


def test_all_variants_step_and_differ():
    results = {pt: _one_step(pt) for pt in AdamNSFamily.NS_POINTS}
    for pt, p in results.items():
        assert torch.isfinite(p).all(), pt
    assert not torch.allclose(results["grad"], results["momentum"])
    assert not torch.allclose(results["momentum"], results["update"])
    assert not torch.allclose(results["grad"], results["update"])


def test_build_optimizer_wires_adamns_variants():
    for opt_type in ("adamns", "adamgradns", "adamupdns"):
        model, gcfg = make_tiny_gpt(seed=1)
        cfg = OmegaConf.create({"optimizer": {
            "type": opt_type, "lr": 3e-4, "weight_decay": 0.0,
            "adamns_lr": 3e-4, "adamgradns_lr": 3e-4, "adamupdns_lr": 0.02,
        }})
        opts = build_optimizer(model, cfg)
        assert len(opts) == 2
        assert isinstance(opts[0], AdamNSFamily)

        x = torch.randint(0, gcfg.vocab_size, (2, gcfg.sequence_len))
        y = torch.randint(0, gcfg.vocab_size, (2, gcfg.sequence_len))
        loss = model(x, y)
        loss.backward()
        for o in opts:
            o.step()
        for p in model.parameters():
            assert torch.isfinite(p).all()
