"""Tier 0: state_dict <-> flat-vector round-trip and the NN energy closure."""

import os
import sys
from contextlib import nullcontext

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from optim.param_vector import (
    flatten_state_dict,
    load_vector_into_model,
    make_nn_energy,
    unflatten_vector,
)
from conftest import make_tiny_gpt, make_token_batches


def test_flatten_unflatten_roundtrip(tiny_gpt):
    model, _ = tiny_gpt
    sd = model.state_dict()
    vec, specs = flatten_state_dict(sd)

    assert vec.dtype == torch.float32
    assert vec.device.type == "cpu"
    assert vec.numel() == sum(s.numel for s in specs)

    sd2 = unflatten_vector(vec, specs)
    assert set(sd2.keys()) == set(sd.keys())
    for k in sd:
        assert torch.equal(sd2[k], sd[k]), f"mismatch on {k}"


def test_flatten_order_is_stable(tiny_gpt):
    model, _ = tiny_gpt
    _, specs_a = flatten_state_dict(model.state_dict())
    _, specs_b = flatten_state_dict(model.state_dict())
    assert [s.name for s in specs_a] == [s.name for s in specs_b]
    assert [s.name for s in specs_a] == sorted(s.name for s in specs_a)


def test_load_vector_roundtrips_and_leaves_buffers(tiny_gpt):
    model, _ = tiny_gpt
    vec, specs = flatten_state_dict(model.state_dict())

    cos_before = model.cos.clone()
    sin_before = model.sin.clone()

    perturbed = vec + 0.01 * torch.randn_like(vec)
    load_vector_into_model(model, perturbed, specs)

    vec_after, _ = flatten_state_dict(model.state_dict())
    assert torch.allclose(vec_after, perturbed, atol=1e-6)

    # RoPE caches are non-persistent buffers: not in the vector, untouched.
    assert "cos" not in {s.name for s in specs}
    assert "sin" not in {s.name for s in specs}
    assert torch.equal(model.cos, cos_before)
    assert torch.equal(model.sin, sin_before)


def test_energy_grad_shape_and_finite():
    model, cfg = make_tiny_gpt(seed=1)
    batches = make_token_batches(cfg, n_batches=2, batch_size=4, seed=1)
    vec, specs = flatten_state_dict(model.state_dict())

    energy = make_nn_energy(model, specs, batches, torch.device("cpu"),
                            nullcontext(), with_grad=True)
    loss, grad = energy(vec)

    assert isinstance(loss, float) and loss == loss  # not NaN
    assert grad.numel() == vec.numel()
    assert torch.isfinite(grad).all()
    assert grad.abs().sum() > 0  # some gradient signal


def test_energy_without_grad_returns_none():
    model, cfg = make_tiny_gpt(seed=2)
    batches = make_token_batches(cfg, n_batches=1, batch_size=2, seed=2)
    vec, specs = flatten_state_dict(model.state_dict())

    energy = make_nn_energy(model, specs, batches, torch.device("cpu"),
                            nullcontext(), with_grad=False)
    loss, grad = energy(vec)
    assert isinstance(loss, float)
    assert grad is None
