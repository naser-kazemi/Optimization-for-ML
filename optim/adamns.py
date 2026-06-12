"""Adam variants with Newton-Schulz orthogonalization (Muon-style) applied at
one of three points in the update: the gradient, the momentum, or the final
update. 2-D parameters only; callers keep embeddings/heads in a separate
AdamW group, as with Muon."""

import torch


def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    """Approximate UV^T from the SVD G = USV^T (quintic iteration, coefficients
    from the Muon reference implementation; float32 is fine at this scale)."""
    assert G.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.to(torch.float32)
    X = X / (X.norm() + eps)
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


class AdamNSFamily(torch.optim.Optimizer):
    """Adam with NS orthogonalization at ns_point in {grad, momentum, update}.

    momentum mode tracks the second moment of the orthogonalized momentum
    rather than the raw gradient (as in AdaMuon); otherwise sqrt(v) scales
    with the gradient magnitude while the NS output is O(1), and the ratio
    blows up as gradients shrink. Weight decay is decoupled.
    """

    NS_POINTS = ("grad", "momentum", "update")

    def __init__(self, params, lr=3e-4, betas=(0.9, 0.95), eps=1e-8,
                 weight_decay=0.0, ns_point="momentum", ns_steps=5):
        assert ns_point in self.NS_POINTS
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        ns_point=ns_point, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            eps = group['eps']
            ns_point = group['ns_point']
            ns_steps = group['ns_steps']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                apply_ns = p.ndim == 2

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                state['step'] += 1
                t = state['step']
                bc1 = 1 - beta1 ** t
                bc2 = 1 - beta2 ** t
                m, v = state['exp_avg'], state['exp_avg_sq']

                if ns_point == "grad" and apply_ns:
                    g = zeropower_via_newtonschulz5(g, ns_steps)

                m.mul_(beta1).add_(g, alpha=1 - beta1)

                if ns_point == "momentum" and apply_ns:
                    o = zeropower_via_newtonschulz5(m / bc1, ns_steps)
                    v.mul_(beta2).addcmul_(o, o, value=1 - beta2)
                    update = o / ((v / bc2).sqrt() + eps)
                else:
                    v.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                    update = (m / bc1) / ((v / bc2).sqrt() + eps)
                    if ns_point == "update" and apply_ns:
                        update = zeropower_via_newtonschulz5(update, ns_steps)
                        update = update * max(1.0, p.size(0) / p.size(1)) ** 0.5

                if group['weight_decay'] > 0:
                    p.mul_(1 - group['lr'] * group['weight_decay'])
                p.add_(update, alpha=-group['lr'])
        return loss
