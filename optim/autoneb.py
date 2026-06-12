"""AutoNEB control loop: relax, measure, insert pivots where the path is
under-resolved, repeat with decreasing lr (Draxler et al. 2018, Alg. 2)."""

from dataclasses import dataclass

import torch

from optim.neb import NEBConfig, Path, measure_path, neb_optimize

_EPS = 1e-12


@dataclass
class AutoNEBConfig:
    n_pivots_interior: int = 7
    cycles_lr: tuple = (0.01, 0.003, 0.001)
    steps_per_cycle: int = 200
    momentum: float = 0.9
    reparametrize_every: int = 1
    insert_count: int = 2
    insert_threshold: float = 0.10
    max_pivots: int = 32
    n_interp: int = 5


def insert_pivots(path, profile, n_interp, threshold_frac, insert_count, max_pivots):
    """Split segments where the densely-sampled loss deviates most from the
    linear interpolation of the pivot losses. profile must come from
    measure_path(path, ..., n_interp=n_interp). At most insert_count pivots are
    added, each at the worst deviation of its segment, if that deviation
    exceeds threshold_frac of the profile's loss range. Segment weights are
    split with the insertion so reparametrization keeps the new resolution."""
    coords = path.coords
    m = coords.shape[0]
    k = n_interp
    losses = profile.losses
    assert losses.numel() == (m - 1) * (k + 1) + 1
    if k == 0 or insert_count <= 0 or m >= max_pivots:
        return path

    loss_range = float(losses.max() - losses.min())
    if loss_range <= _EPS:
        return path

    seg_dev = torch.zeros(m - 1)
    seg_alpha = torch.zeros(m - 1)
    for i in range(m - 1):
        la = float(losses[i * (k + 1)])
        lb = float(losses[(i + 1) * (k + 1)])
        best = (-float("inf"), 0.0)
        for j in range(1, k + 1):
            alpha = j / (k + 1)
            interp = (1.0 - alpha) * la + alpha * lb
            dev = float(losses[i * (k + 1) + j]) - interp
            if dev > best[0]:
                best = (dev, alpha)
        seg_dev[i] = best[0] / loss_range
        seg_alpha[i] = best[1]

    budget = min(insert_count, max_pivots - m)
    order = torch.argsort(seg_dev, descending=True)
    chosen = {int(i): float(seg_alpha[i]) for i in order[:budget]
              if float(seg_dev[i]) > threshold_frac}
    if not chosen:
        return path

    rows = [coords[0]]
    weights = []
    w = path.target_weights
    for i in range(m - 1):
        if i in chosen:
            a = chosen[i]
            rows.append((1.0 - a) * coords[i] + a * coords[i + 1])
            weights.extend([a * float(w[i]), (1.0 - a) * float(w[i])])
        else:
            weights.append(float(w[i]))
        rows.append(coords[i + 1])
    return Path(torch.stack(rows).contiguous(), torch.tensor(weights))


def autoneb_connect(x0, x1, energy_minibatch, energy_full, cfg, callback=None,
                    progress=False, energy_loss_only=None):
    """Find a low-barrier path from x0 to x1.

    Relaxes with energy_minibatch (may be stochastic), measures and decides
    insertions with energy_full. Returns the best (path, profile) over the
    linear initialization and all cycles, so the result is never worse than
    the straight line it starts from.
    """
    path = Path.linear_init(x0, x1, cfg.n_pivots_interior)
    n_cycles = len(cfg.cycles_lr)

    profile = measure_path(path, energy_full, n_interp=cfg.n_interp)
    best_path = Path(path.coords.clone(), path.target_weights.clone())
    best_profile = profile

    for cycle_idx, lr in enumerate(cfg.cycles_lr):
        if cycle_idx > 0:
            path = insert_pivots(path, profile, cfg.n_interp,
                                 cfg.insert_threshold, cfg.insert_count,
                                 cfg.max_pivots)
        neb_cfg = NEBConfig(
            n_steps=cfg.steps_per_cycle,
            lr=lr,
            momentum=cfg.momentum,
            reparametrize_every=cfg.reparametrize_every,
        )
        neb_optimize(path, energy_minibatch, neb_cfg, progress=progress,
                     desc=f"AutoNEB cycle {cycle_idx + 1}/{n_cycles} (lr={lr})",
                     energy_loss_only=energy_loss_only)

        profile = measure_path(path, energy_full, n_interp=cfg.n_interp)
        if callback is not None:
            callback(cycle_idx, lr, path, profile)

        if profile.barrier() < best_profile.barrier():
            # snapshot: later cycles keep relaxing `path` in place
            best_path = Path(path.coords.clone(), path.target_weights.clone())
            best_profile = profile

    return best_path, best_profile
