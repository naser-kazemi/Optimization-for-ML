import torch

def quantize_to_int8(tensor):
    """
    Quantize a float tensor to int8 representation and dequantize back to float.
    This simulates the precision loss of 8-bit gradient quantization.
    
    scale = max(abs(tensor)) / 127
    q_tensor = round(tensor / scale)
    dq_tensor = q_tensor * scale
    """
    if tensor is None:
        return None
    
    max_val = tensor.abs().max()
    if max_val == 0:
        return tensor
    
    scale = max_val / 127.0
    q_tensor = torch.round(tensor / scale).clamp(-128, 127)
    dq_tensor = q_tensor * scale
    return dq_tensor

def register_quantization_hooks(model):
    """
    Registers a backward hook on all parameters that require gradients.
    The hook will quantize the gradients to INT8 precision during the backward pass.
    """
    handles = []
    for name, p in model.named_parameters():
        if p.requires_grad:
            def hook(grad):
                return quantize_to_int8(grad)
            handle = p.register_hook(hook)
            handles.append(handle)
    return handles
