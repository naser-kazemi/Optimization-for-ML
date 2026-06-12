"""String-method NEB over a chain of coordinate vectors, model-agnostic:
only needs an energy(coords) -> (loss, grad) closure."""

from dataclasses import dataclass

import torch

_EPS = 1e-12


class Path:
    """Chain of M pivots, (M, D) float32 on CPU. Rows 0 and M-1 are fixed
    endpoints. target_weights (M-1,) is each segment's share of the arc length
    after reparametrization, so inserted resolution stays where it was added."""

    def __init__(self, coords, target_weights=None):
        assert coords.dim() == 2, "coords must be (M, D)"
        self.coords = coords
        if target_weights is None:
            target_weights = torch.ones(coords.shape[0] - 1)
        assert target_weights.shape == (coords.shape[0] - 1,)
        self.target_weights = target_weights / target_weights.sum()

    @property
    def n_pivots(self):
        return self.coords.shape[0]

    @property
    def dim(self):
        return self.coords.shape[1]

    @staticmethod
    def linear_init(x0, x1, n_interior):
        x0 = x0.reshape(-1).to(torch.float32).cpu()
        x1 = x1.reshape(-1).to(torch.float32).cpu()
        m = n_interior + 2
        alphas = torch.linspace(0.0, 1.0, m).unsqueeze(1)
        coords = (1.0 - alphas) * x0.unsqueeze(0) + alphas * x1.unsqueeze(0)
        return Path(coords.contiguous())

    def segment_lengths(self):
        diffs = self.coords[1:] - self.coords[:-1]
        return diffs.pow(2).sum(dim=1).sqrt()

    def tangents(self, losses):
        """Upwind tangents with the Henkelman & Jonsson (2000) blend at local
        extrema, where the plain upwind rule flips discontinuously."""
        coords = self.coords
        m = coords.shape[0]
        tang = torch.zeros_like(coords)
        for i in range(1, m - 1):
            forward = coords[i + 1] - coords[i]
            backward = coords[i] - coords[i - 1]
            l_prev, l_i, l_next = float(losses[i - 1]), float(losses[i]), float(losses[i + 1])
            is_extremum = (l_prev < l_i > l_next) or (l_prev > l_i < l_next)
            if is_extremum:
                t_fwd = forward / (forward.norm() + _EPS)
                t_bwd = backward / (backward.norm() + _EPS)
                dl_max = max(abs(l_next - l_i), abs(l_prev - l_i))
                dl_min = min(abs(l_next - l_i), abs(l_prev - l_i))
                if l_next > l_prev:
                    t = dl_max * t_fwd + dl_min * t_bwd
                else:
                    t = dl_min * t_fwd + dl_max * t_bwd
            elif l_next > l_prev:
                t = forward
            else:
                t = backward
            norm = t.norm()
            if norm > _EPS:
                tang[i] = t / norm
        return tang

    def reparametrize(self):
        """Re-space interior pivots so segment k occupies target_weights[k] of
        the total arc length; endpoints unchanged."""
        coords = self.coords
        m = coords.shape[0]
        seg = self.segment_lengths()
        cum = torch.cat([torch.zeros(1), torch.cumsum(seg, dim=0)])
        total = cum[-1]
        if total < _EPS:
            return
        targets = torch.cat([torch.zeros(1), torch.cumsum(self.target_weights, dim=0)]) * total
        new = coords.clone()
        for k in range(1, m - 1):
            t = targets[k]
            j = int(torch.searchsorted(cum, t, right=True).item()) - 1
            j = max(0, min(j, m - 2))
            denom = seg[j]
            frac = ((t - cum[j]) / denom) if denom > _EPS else 0.0
            new[k] = coords[j] + frac * (coords[j + 1] - coords[j])
        self.coords = new


@dataclass
class NEBConfig:
    n_steps: int = 200
    lr: float = 0.1
    momentum: float = 0.9
    reparametrize_every: int = 1


def neb_optimize(path, energy, cfg, callback=None, progress=False, desc="NEB",
                 energy_loss_only=None):
    """Relax interior pivots: descend the component of the loss gradient
    perpendicular to the path tangent (heavy-ball), then re-space. Endpoint
    losses are re-evaluated each step so a stochastic energy doesn't pin the
    tangent decision to one stale batch; energy_loss_only is an optional
    cheaper closure for those evaluations."""
    m = path.n_pivots
    velocity = torch.zeros_like(path.coords)
    eval_endpoint = energy_loss_only if energy_loss_only is not None else energy

    steps = range(cfg.n_steps)
    if progress:
        from tqdm.auto import tqdm
        steps = tqdm(steps, desc=desc, leave=False)

    for step in steps:
        losses = torch.zeros(m)
        grads = torch.zeros_like(path.coords)
        losses[0] = eval_endpoint(path.coords[0])[0]
        losses[-1] = eval_endpoint(path.coords[-1])[0]
        for i in range(1, m - 1):
            li, gi = energy(path.coords[i])
            losses[i] = li
            grads[i] = gi

        tang = path.tangents(losses)

        forces = torch.zeros_like(path.coords)
        for i in range(1, m - 1):
            g = grads[i]
            t = tang[i]
            g_perp = g - (g @ t) * t
            forces[i] = -g_perp

        velocity = cfg.momentum * velocity + forces
        path.coords[1:-1] = path.coords[1:-1] + cfg.lr * velocity[1:-1]

        if cfg.reparametrize_every > 0 and (step + 1) % cfg.reparametrize_every == 0:
            path.reparametrize()

        if progress:
            steps.set_postfix(max_loss=f"{float(losses.max()):.3f}")

        if callback is not None:
            callback(step, path, losses)

    return path


@dataclass
class PathProfile:
    positions: torch.Tensor  # (S,) in [0, 1]
    losses: torch.Tensor     # (S,)

    def barrier(self):
        """Max loss on the path minus the higher of the two endpoint losses."""
        endpoint = max(float(self.losses[0]), float(self.losses[-1]))
        return float(self.losses.max()) - endpoint

    def argmax(self):
        return int(torch.argmax(self.losses).item())


def measure_path(path, energy_full, n_interp=0):
    """Loss along the path: every pivot plus n_interp interpolated points per
    segment (pivot i lands at index i*(n_interp+1))."""
    coords = path.coords
    m = coords.shape[0]
    seg = path.segment_lengths()
    cum = torch.cat([torch.zeros(1), torch.cumsum(seg, dim=0)])
    total = float(cum[-1])

    positions = []
    losses = []
    for i in range(m):
        loss_i, _ = energy_full(coords[i])
        positions.append(float(cum[i]))
        losses.append(loss_i)
        if n_interp > 0 and i < m - 1:
            for k in range(1, n_interp + 1):
                frac = k / (n_interp + 1)
                pt = coords[i] + frac * (coords[i + 1] - coords[i])
                loss_k, _ = energy_full(pt)
                positions.append(float(cum[i] + frac * seg[i]))
                losses.append(loss_k)

    positions = torch.tensor(positions)
    if total > _EPS:
        positions = positions / total
    losses = torch.tensor(losses)
    return PathProfile(positions=positions, losses=losses)
