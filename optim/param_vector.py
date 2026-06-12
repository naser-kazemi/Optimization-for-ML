"""Flat-vector <-> state_dict utilities and the energy closures used by NEB.

Vectors are float32 on CPU, flattened in sorted state_dict key order. Note
this ordering differs from model.parameters(); never mix the two — load the
vector into the model first.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ParamSpec:
    name: str
    shape: torch.Size
    numel: int
    dtype: torch.dtype


def flatten_state_dict(state_dict):
    specs = []
    chunks = []
    for name in sorted(state_dict.keys()):
        t = state_dict[name]
        specs.append(ParamSpec(name=name, shape=t.shape, numel=t.numel(), dtype=t.dtype))
        chunks.append(t.detach().to(torch.float32).reshape(-1).cpu())
    vec = torch.cat(chunks) if chunks else torch.zeros(0, dtype=torch.float32)
    return vec, specs


def unflatten_vector(vec, specs):
    out = {}
    offset = 0
    for s in specs:
        segment = vec[offset:offset + s.numel].reshape(s.shape).to(s.dtype)
        out[s.name] = segment
        offset += s.numel
    assert offset == vec.numel()
    return out


def load_vector_into_model(model, vec, specs):
    state_dict = unflatten_vector(vec, specs)
    model.load_state_dict(state_dict, strict=True)


def _collect_flat_grad(model, specs, coords):
    named = dict(model.named_parameters())
    grad_chunks = []
    for s in specs:
        p = named.get(s.name, None)
        if p is None or p.grad is None:
            grad_chunks.append(torch.zeros(s.numel, dtype=torch.float32))
        else:
            grad_chunks.append(p.grad.detach().to(torch.float32).reshape(-1).cpu())
    return torch.cat(grad_chunks).to(coords.device, coords.dtype)


def make_nn_energy_stream(model, specs, batch_iter, autocast_ctx,
                          with_grad=True, weight_decay=0.0):
    """energy(coords) -> (loss, grad) consuming one batch from batch_iter per
    call, so the relaxation objective stays distinct from the fixed
    measurement landscape."""

    def energy(coords):
        load_vector_into_model(model, coords, specs)
        model.eval()
        x, y = next(batch_iter)

        if not with_grad:
            with torch.no_grad():
                with autocast_ctx:
                    loss = model(x, y)
            total = loss.item()
            if weight_decay:
                total += 0.5 * weight_decay * float((coords * coords).sum())
            return total, None

        model.zero_grad(set_to_none=True)
        with autocast_ctx:
            loss = model(x, y)
        loss.backward()
        total = loss.detach().item()
        grad = _collect_flat_grad(model, specs, coords)

        if weight_decay:
            total += 0.5 * weight_decay * float((coords * coords).sum())
            grad = grad + weight_decay * coords

        return total, grad

    return energy


def make_nn_energy(model, specs, eval_batches, device, autocast_ctx,
                   with_grad=True, weight_decay=0.0):
    """energy(coords) -> (loss, grad) over a fixed batch set (mean loss),
    giving a deterministic landscape for measuring barriers."""
    n = len(eval_batches)
    assert n > 0

    def energy(coords):
        load_vector_into_model(model, coords, specs)
        model.eval()

        if not with_grad:
            total = 0.0
            with torch.no_grad():
                for x, y in eval_batches:
                    with autocast_ctx:
                        loss = model(x, y)
                    total += loss.item() / n
            if weight_decay:
                total += 0.5 * weight_decay * float((coords * coords).sum())
            return total, None

        model.zero_grad(set_to_none=True)
        total = 0.0
        for x, y in eval_batches:
            with autocast_ctx:
                loss = model(x, y) / n
            loss.backward()
            total += loss.detach().item()

        grad = _collect_flat_grad(model, specs, coords)

        if weight_decay:
            total += 0.5 * weight_decay * float((coords * coords).sum())
            grad = grad + weight_decay * coords

        return total, grad

    return energy
