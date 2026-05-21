def get_wsd_lr_multiplier(progress, warmup_ratio=0.05, warmdown_ratio=0.3, final_lr_frac=0.1):
    """
    Compute WSD (Warmup-Stable-Decay) learning rate multiplier.
    progress: float between 0.0 and 1.0
    warmup_ratio: fraction of total steps spent warming up
    warmdown_ratio: fraction of total steps spent cooling down (Decay phase)
    final_lr_frac: LR at the end of training as a fraction of peak LR
    """
    if progress < warmup_ratio:
        return progress / warmup_ratio if warmup_ratio > 0 else 1.0
    elif progress < 1.0 - warmdown_ratio:
        return 1.0
    else:
        cooldown = (1.0 - progress) / warmdown_ratio
        return cooldown * 1.0 + (1 - cooldown) * final_lr_frac
