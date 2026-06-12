import torch

def compute_hvp_reverse_over_reverse(model, data, targets, v):
    """
    Compute Hessian-Vector Product using reverse-over-reverse (standard autograd).
    model: PyTorch model that returns a scalar loss when called as model(data, targets=targets)
    data: input tokens (e.g. B, T)
    targets: target tokens (e.g. B, T)
    v: list of tensors representing the vector v (same shape as model parameters)

    Note: Disables flash/efficient attention during computation since those
    kernels don't support second-order gradients.
    """
    params = [p for p in model.parameters() if p.requires_grad]

    # We must explicitly enable gradients to construct the computational graph,
    # as this function is often called within no_grad() evaluation blocks.
    # Disable efficient attention backends that don't support double backward.
    with torch.enable_grad(), \
         torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
        loss = model(data, targets=targets)
        grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)

        # HVP = d/dtheta (grad . v): differentiate the grad-vector dot product again.
        dot_product = sum([(g * v_).sum() for g, v_ in zip(grads, v)])
        hvp = torch.autograd.grad(dot_product, params, retain_graph=True)

    return hvp


@torch.no_grad()
def power_iteration(model, data_loader, num_iterations=10, device='cuda'):
    """
    Compute top eigenvalue and eigenvector of the Hessian using Power Iteration + HVP.
    Returns: lambda_max, v_max
    """
    params = [p for p in model.parameters() if p.requires_grad]

    # Start from a random unit vector and repeatedly apply H, renormalizing.
    v = [torch.randn_like(p).to(device) for p in params]
    norm_v = torch.sqrt(sum([torch.sum(v_ ** 2) for v_ in v]))
    v = [v_ / norm_v for v_ in v]

    x, y = next(data_loader)

    for _ in range(num_iterations):
        hvp = compute_hvp_reverse_over_reverse(model, x, y, v)
        norm_hvp = torch.sqrt(sum([torch.sum(h_ ** 2) for h_ in hvp]))
        if norm_hvp.item() == 0:
            break
        v = [h_ / norm_hvp for h_ in hvp]

    # Rayleigh quotient v^T H v gives the eigenvalue for the converged v.
    hvp = compute_hvp_reverse_over_reverse(model, x, y, v)
    lambda_max = sum([(v_ * h_).sum() for v_, h_ in zip(v, hvp)]).item()
    return lambda_max, v
