"""Shared pytest fixtures/helpers. Puts the repo root on sys.path so `models`,
`optim`, and `utils` import cleanly when running `pytest` from anywhere."""

import os
import sys

import pytest
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from models import GPT, GPTConfig  # noqa: E402


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: requires mps/cuda (double-backward)")


def make_tiny_gpt(seed=0):
    """A small GPT for fast CPU tests."""
    torch.manual_seed(seed)
    cfg = GPTConfig(
        sequence_len=16,
        vocab_size=64,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=32,
    )
    model = GPT(cfg)
    model.init_weights()
    return model, cfg


def make_token_batches(cfg, n_batches=2, batch_size=4, seed=0):
    """Random (x, y) next-token batches on CPU for the tiny model."""
    g = torch.Generator().manual_seed(seed)
    batches = []
    for _ in range(n_batches):
        x = torch.randint(0, cfg.vocab_size, (batch_size, cfg.sequence_len), generator=g)
        y = torch.randint(0, cfg.vocab_size, (batch_size, cfg.sequence_len), generator=g)
        batches.append((x, y))
    return batches


@pytest.fixture
def tiny_gpt():
    return make_tiny_gpt(seed=0)


@pytest.fixture
def tiny_batches(tiny_gpt):
    _, cfg = tiny_gpt
    return make_token_batches(cfg, n_batches=2, batch_size=4, seed=0)
