import torch

def compute_cosine_similarity(delta_theta, v_max):
    """
    Compute cosine similarity between the parameter update step and the top eigenvector of the Hessian.
    delta_theta: list of tensors representing the update step
    v_max: list of tensors representing the top eigenvector
    """
    dot_product = sum([(dt * vm).sum() for dt, vm in zip(delta_theta, v_max)])
    
    norm_delta = torch.sqrt(sum([torch.sum(dt ** 2) for dt in delta_theta]))
    norm_v = torch.sqrt(sum([torch.sum(vm ** 2) for vm in v_max]))
    
    if norm_delta == 0 or norm_v == 0:
        return 0.0
    
    return (dot_product / (norm_delta * norm_v)).item()
