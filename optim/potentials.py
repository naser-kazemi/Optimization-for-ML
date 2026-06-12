"""
Analytic 2-D test potentials with the same `energy(coords) -> (loss, grad)`
contract as the NN energy closure, plus a dense-grid ground-truth barrier
search. Used by the NEB unit tests to validate the path optimizer against
problems with a known minimum-energy path before touching the GPT.
"""

import heapq

import torch


def double_well_2d():
    """E(x, y) = (x^2 - 1)^2 + y^2.

    Minima at (-1, 0) and (+1, 0) with E = 0; the minimum-energy path runs
    along y = 0 through the saddle at (0, 0) with E = 1. Hence the exact
    barrier between the two minima is 1.0.
    """
    def energy(coords):
        x, y = coords[0], coords[1]
        loss = (x * x - 1.0) ** 2 + y * y
        grad = torch.stack([4.0 * x * (x * x - 1.0), 2.0 * y]).to(coords.dtype)
        return float(loss), grad
    return energy


def muller_brown_2d():
    """The classic 4-Gaussian Müller-Brown potential (analytic gradient)."""
    # float64 constants so the potential is high precision (keeps finite-diff
    # gradient checks meaningful); the returned grad is cast back to coords.dtype.
    A = torch.tensor([-200.0, -100.0, -170.0, 15.0], dtype=torch.float64)
    a = torch.tensor([-1.0, -1.0, -6.5, 0.7], dtype=torch.float64)
    b = torch.tensor([0.0, 0.0, 11.0, 0.6], dtype=torch.float64)
    c = torch.tensor([-10.0, -10.0, -6.5, 0.7], dtype=torch.float64)
    x0 = torch.tensor([1.0, 0.0, -0.5, -1.0], dtype=torch.float64)
    y0 = torch.tensor([0.0, 0.5, 1.5, 1.0], dtype=torch.float64)

    def energy(coords):
        x, y = coords[0], coords[1]
        dx = x - x0
        dy = y - y0
        expo = a * dx * dx + b * dx * dy + c * dy * dy
        terms = A * torch.exp(expo)
        loss = terms.sum()
        # d/dx and d/dy of each term
        dterm_dx = terms * (2.0 * a * dx + b * dy)
        dterm_dy = terms * (b * dx + 2.0 * c * dy)
        grad = torch.stack([dterm_dx.sum(), dterm_dy.sum()]).to(coords.dtype)
        return float(loss), grad
    return energy


def grid_search_mep(energy, bounds, resolution, start, end):
    """Dense-grid minimax (bottleneck) path search for a ground-truth barrier.

    Finds the path between the grid cells nearest `start` and `end` that
    minimizes the maximum energy encountered, via a Dijkstra-style search on the
    bottleneck cost. Returns (barrier, path_points) where `barrier` is the
    minimax energy minus the higher endpoint energy.

    Args:
        energy: energy(coords) -> (loss, grad).
        bounds: ((xmin, xmax), (ymin, ymax)).
        resolution: number of grid points per axis.
        start, end: (x, y) endpoints.
    """
    (xmin, xmax), (ymin, ymax) = bounds
    xs = torch.linspace(xmin, xmax, resolution)
    ys = torch.linspace(ymin, ymax, resolution)

    E = torch.zeros(resolution, resolution)
    for i in range(resolution):
        for j in range(resolution):
            E[i, j], _ = energy(torch.tensor([xs[i], ys[j]]))

    def nearest(pt):
        i = int(torch.argmin((xs - pt[0]).abs()).item())
        j = int(torch.argmin((ys - pt[1]).abs()).item())
        return i, j

    si, sj = nearest(start)
    ei, ej = nearest(end)

    # Bottleneck shortest path: cost(node) = min over paths of max energy on path.
    INF = float("inf")
    best = torch.full((resolution, resolution), INF)
    best[si, sj] = float(E[si, sj])
    pq = [(best[si, sj].item(), si, sj)]
    while pq:
        cost, i, j = heapq.heappop(pq)
        if cost > best[i, j]:
            continue
        if (i, j) == (ei, ej):
            break
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ni, nj = i + di, j + dj
            if 0 <= ni < resolution and 0 <= nj < resolution:
                new_cost = max(cost, float(E[ni, nj]))
                if new_cost < best[ni, nj]:
                    best[ni, nj] = new_cost
                    heapq.heappush(pq, (new_cost, ni, nj))

    minimax_energy = float(best[ei, ej])
    endpoint = max(float(E[si, sj]), float(E[ei, ej]))
    return minimax_energy - endpoint, (minimax_energy, (si, sj), (ei, ej))
