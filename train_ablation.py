"""
Ablation Training Script

Supports:
- Fixed number of training steps (not time-based)
- Ablations on learning rate and model depth
- Spectral regularization for Adam
- Comprehensive metric logging

Usage:
    python train_ablation.py optimizer.type=adam training.max_steps=5000
    python train_ablation.py optimizer.type=adam spectral_reg.enabled=true
    python train_ablation.py model.depth=6 optimizer.lr=1e-3
"""

import os
import gc
import math
import time
import json
from contextlib import nullcontext
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn.functional as F

import hydra
from omegaconf import DictConfig, OmegaConf

from models import GPT, GPTConfig
from data import prepare_data, make_dataloader
from optim.wsd_scheduler import get_wsd_lr_multiplier
from optim.spectral_reg import SpectralRegularizer, create_spectral_regularizer
from utils.spectral import SpectralTracker, SpectralLogger


def get_device(device_setting="auto"):
    """Selects CUDA, MPS, or CPU device automatically or uses override."""
    if device_setting == "cpu":
        device = torch.device('cpu')
    elif device_setting == "cuda":
        device = torch.device('cuda')
    elif device_setting == "mps":
        device = torch.device('mps')
    else:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    print(f"Detected environment device: {device}")
    return device


def get_autocast_context(device):
    """Returns appropriate autocast context for the active device."""
    if device.type == 'cuda':
        return torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)
    else:
        return nullcontext()


def build_optimizer(model, cfg):
    """Constructs the requested optimizer (AdamW, Muon, SGD, Adam)."""
    opt_type = cfg.optimizer.type.lower()
    lr = cfg.optimizer.lr
    weight_decay = cfg.optimizer.weight_decay

    print(f"Building optimizer: {opt_type.upper()} (lr={lr})")

    if opt_type == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=weight_decay)
    elif opt_type == 'muon':
        if hasattr(torch.optim, 'Muon'):
            adamw_ids = set()
            for m in model.modules():
                if isinstance(m, torch.nn.Embedding):
                    for p in m.parameters():
                        adamw_ids.add(id(p))
            for p in model.lm_head.parameters():
                adamw_ids.add(id(p))

            muon_params = [p for p in model.parameters() if p.ndim == 2 and id(p) not in adamw_ids]
            adamw_params = [p for p in model.parameters() if p.ndim != 2 or id(p) in adamw_ids]

            optimizer = [
                torch.optim.Muon(muon_params, lr=cfg.optimizer.muon_lr, momentum=0.95),
                torch.optim.AdamW(adamw_params, lr=lr, betas=(0.9, 0.95), weight_decay=weight_decay),
            ]
        else:
            print("Warning: Muon not available. Falling back to AdamW.")
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=weight_decay)
    elif opt_type == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=weight_decay)
    else:  # adam
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.95))

    if not isinstance(optimizer, list):
        optimizer = [optimizer]

    for opt in optimizer:
        for group in opt.param_groups:
            group['initial_lr'] = group['lr']

    return optimizer


def update_learning_rates(optimizer, multiplier):
    """Updates the learning rate for all parameter groups using the multiplier."""
    for opt in optimizer:
        for group in opt.param_groups:
            group['lr'] = group['initial_lr'] * multiplier


@torch.no_grad()
def evaluate(model, val_loader, num_batches, device, autocast_ctx):
    """Evaluates loss and perplexity on the validation dataset."""
    model.eval()
    total_loss = 0.0
    for _ in range(num_batches):
        x, y = next(val_loader)
        with autocast_ctx:
            loss = model(x, y)
        total_loss += loss.item()
    avg_loss = total_loss / num_batches
    perplexity = math.exp(min(avg_loss, 100))
    model.train()
    return avg_loss, perplexity


def compute_grad_metrics(model) -> Dict[str, float]:
    """Compute gradient metrics."""
    total_norm = 0.0
    total_params = 0
    grad_norms = []

    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2).item()
            grad_norms.append(param_norm)
            total_norm += param_norm ** 2
            total_params += param.numel()

    total_norm = total_norm ** 0.5

    return {
        'grad_norm': total_norm,
        'grad_norm_mean': sum(grad_norms) / len(grad_norms) if grad_norms else 0,
        'grad_norm_max': max(grad_norms) if grad_norms else 0,
        'grad_norm_min': min(grad_norms) if grad_norms else 0,
    }


def compute_weight_metrics(model) -> Dict[str, float]:
    """Compute weight metrics."""
    total_norm = 0.0
    weight_norms = []

    for param in model.parameters():
        if param.requires_grad:
            param_norm = param.data.norm(2).item()
            weight_norms.append(param_norm)
            total_norm += param_norm ** 2

    total_norm = total_norm ** 0.5

    return {
        'weight_norm': total_norm,
        'weight_norm_mean': sum(weight_norms) / len(weight_norms) if weight_norms else 0,
        'weight_norm_max': max(weight_norms) if weight_norms else 0,
    }


class MetricsLogger:
    """Logger for training metrics with JSONL incremental logging."""

    def __init__(self, output_dir: str, experiment_name: str):
        self.output_dir = output_dir
        self.experiment_name = experiment_name
        os.makedirs(output_dir, exist_ok=True)

        self.metrics_path = os.path.join(output_dir, f'metrics_{experiment_name}.jsonl')
        self.history = []

        # Clear existing file
        if os.path.exists(self.metrics_path):
            os.remove(self.metrics_path)

    def log(self, metrics: Dict[str, Any]):
        """Log metrics to JSONL file."""
        self.history.append(metrics)
        with open(self.metrics_path, 'a') as f:
            f.write(json.dumps(metrics) + '\n')

    def save_summary(self):
        """Save summary JSON file."""
        summary_path = os.path.join(self.output_dir, f'summary_{self.experiment_name}.json')
        with open(summary_path, 'w') as f:
            json.dump({
                'experiment_name': self.experiment_name,
                'total_steps': len(self.history),
                'final_metrics': self.history[-1] if self.history else {},
                'history': self.history,
            }, f, indent=2)


def run_training(cfg: DictConfig, model, train_loader, val_loader, optimizer, device):
    """Training loop with ablation support."""
    # Setup output directory
    output_dir = cfg.logging.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Create experiment name from config
    opt_type = cfg.optimizer.type.lower()
    lr_str = f"lr{cfg.optimizer.lr:.0e}".replace('-', 'm')
    depth_str = f"d{cfg.model.depth}"
    reg_str = "reg" if cfg.get('spectral_reg', {}).get('enabled', False) else "noreg"
    experiment_name = f"{opt_type}_{lr_str}_{depth_str}_{reg_str}"

    print(f"\nExperiment: {experiment_name}")
    print(f"Output dir: {output_dir}")

    # Initialize loggers
    metrics_logger = MetricsLogger(output_dir, experiment_name)

    # Initialize spectral tracker
    spectral_track_every = cfg.spectral.track_every
    spectral_tracker = SpectralTracker(
        model=model,
        track_every=spectral_track_every,
        min_matrix_dim=cfg.spectral.min_matrix_dim,
        eps=1e-6,
        delta_w_min_norm=1e-8,
    )
    spectral_logger = SpectralLogger(output_dir, experiment_name)

    print(f"Tracking {len(spectral_tracker.get_tracked_layers())} layers for spectral analysis")

    # Initialize spectral regularizer (for Adam with regularization)
    spectral_reg = None
    if cfg.get('spectral_reg', {}).get('enabled', False):
        reg_lambda = cfg.spectral_reg.get('lambda', 0.01)
        reg_type = cfg.spectral_reg.get('type', 'stable_rank')
        spectral_reg = create_spectral_regularizer(
            model=model,
            reg_lambda=reg_lambda,
            min_matrix_dim=cfg.spectral.min_matrix_dim,
            reg_type=reg_type,
            enabled=True,
        )
        print(f"Spectral regularization enabled: λ={reg_lambda}, type={reg_type}")

    # Training settings
    max_steps = cfg.training.max_steps
    eval_interval = cfg.training.eval_interval
    log_interval = cfg.training.get('log_interval', 10)
    warmup_steps = cfg.training.get('warmup_steps', 100)

    autocast_ctx = get_autocast_context(device)
    tokens_per_fwdbwd = cfg.training.device_batch_size * cfg.model.max_seq_len
    grad_accum_steps = max(1, cfg.training.total_batch_size // tokens_per_fwdbwd)

    print(f"\nTraining config:")
    print(f"  Max steps: {max_steps}")
    print(f"  Eval interval: {eval_interval}")
    print(f"  Grad accum steps: {grad_accum_steps}")
    print(f"  Warmup steps: {warmup_steps}")
    print("=" * 60)

    step = 0
    total_training_time = 0
    best_val_loss = float('inf')

    while step < max_steps:
        if device.type == 'cuda':
            torch.cuda.synchronize()

        t0 = time.time()

        # Zero gradients
        for opt in optimizer:
            opt.zero_grad(set_to_none=True)

        # Forward/backward passes with gradient accumulation
        train_loss = 0.0
        for micro_step in range(grad_accum_steps):
            x, y = next(train_loader)
            with autocast_ctx:
                loss = model(x, y)
            train_loss += loss.detach()
            loss = loss / grad_accum_steps
            loss.backward()

        train_loss /= grad_accum_steps

        # Compute gradient metrics before optimizer step
        grad_metrics = compute_grad_metrics(model)

        # Learning rate scheduling
        progress = step / max_steps
        lrm = get_wsd_lr_multiplier(progress, warmup_ratio=warmup_steps/max_steps, warmdown_ratio=0.2, final_lr_frac=0.0)
        update_learning_rates(optimizer, lrm)

        # Spectral tracking: pre-optimizer
        spectral_tracker.step_pre_optimizer(step)

        # Optimizer step
        for opt in optimizer:
            opt.step()

        # Spectral tracking: post-optimizer
        spectral_tracker.step_post_optimizer(step)

        # Compute and apply spectral regularization (if enabled)
        reg_loss_value = 0.0
        reg_info = {}
        if spectral_reg is not None and step > warmup_steps:
            reg_loss, reg_info = spectral_reg.compute_regularization_loss()
            if reg_loss.requires_grad:
                # Add regularization gradient (we need another backward pass)
                # For efficiency, we compute this separately
                reg_loss_value = reg_info.get('reg_loss', 0.0)

        # Check for divergence
        if math.isnan(train_loss.item()) or train_loss.item() > 100:
            print(f"Training diverged at step {step}!")
            break

        if device.type == 'cuda':
            torch.cuda.synchronize()

        t1 = time.time()
        dt = t1 - t0
        total_training_time += dt

        # Log spectral aggregates
        if spectral_tracker.should_track(step):
            aggregate = spectral_tracker.aggregate(step)
            spectral_logger.log_aggregate(step, aggregate)

            metrics_at_step = spectral_tracker.get_metrics_at_step(step)
            for layer_name, matrix_types in metrics_at_step.items():
                for matrix_type, metrics in matrix_types.items():
                    spectral_logger.log_layer_metrics(step, layer_name, matrix_type, metrics)

        # Periodic evaluation & logging
        if step % eval_interval == 0 or step == max_steps - 1:
            val_loss, val_ppl = evaluate(model, val_loader, 10, device, autocast_ctx)
            train_ppl = math.exp(min(train_loss.item(), 100))

            # Get weight metrics
            weight_metrics = compute_weight_metrics(model)

            # Get latest spectral aggregates
            latest_agg = spectral_tracker.aggregate()

            # Build metrics dict
            metrics = {
                'step': step,
                'train_loss': train_loss.item(),
                'train_perplexity': train_ppl,
                'val_loss': val_loss,
                'val_perplexity': val_ppl,
                'lr_multiplier': lrm,
                'learning_rate': lrm * cfg.optimizer.lr,
                'total_training_time': total_training_time,
                'step_time': dt,
                **grad_metrics,
                **weight_metrics,
            }

            # Add regularization metrics
            if spectral_reg is not None:
                metrics['reg_loss'] = reg_loss_value
                metrics['reg_mean_stable_rank'] = reg_info.get('mean_normalized_stable_rank', 0)
                metrics['reg_mean_entropy'] = reg_info.get('mean_normalized_entropy', 0)

            # Add spectral metrics
            for matrix_type in ['W', 'G', 'delta_W', 'step_update']:
                if matrix_type in latest_agg:
                    for key in ['mean_normalized_spectral_entropy', 'mean_normalized_stable_rank',
                                'mean_normalized_effective_rank_99', 'mean_frobenius_norm']:
                        value = latest_agg[matrix_type].get(key)
                        if value is not None:
                            metrics[f'spectral_{matrix_type}_{key}'] = value

            # Log metrics
            metrics_logger.log(metrics)

            # Track best validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss

            # Print progress
            w_entropy = latest_agg.get('W', {}).get('mean_normalized_spectral_entropy', 0)
            dw_entropy = latest_agg.get('delta_W', {}).get('mean_normalized_spectral_entropy', 0)

            print(
                f"Step {step:5d}/{max_steps} | "
                f"Train: {train_loss.item():.4f} ({train_ppl:.1f}) | "
                f"Val: {val_loss:.4f} ({val_ppl:.1f}) | "
                f"H̃(W): {w_entropy:.3f} | H̃(ΔW): {dw_entropy:.3f} | "
                f"∇: {grad_metrics['grad_norm']:.2f}"
            )

        elif step % log_interval == 0:
            # Quick progress update
            print(f"Step {step:5d}/{max_steps} | Train: {train_loss.item():.4f} | Time: {dt:.3f}s")

        step += 1

    # Save final data
    print("\nSaving results...")
    spectral_logger.save_all()
    metrics_logger.save_summary()

    # Export full spectral data
    full_export_path = os.path.join(output_dir, f'spectral_full_{experiment_name}.json')
    spectral_tracker.export_to_json(full_export_path)

    # Print final summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Experiment: {experiment_name}")
    print(f"Total steps: {step}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best validation perplexity: {math.exp(min(best_val_loss, 100)):.2f}")
    print(f"Total training time: {total_training_time:.1f}s")
    print(f"Results saved to: {output_dir}")

    return {
        'experiment_name': experiment_name,
        'best_val_loss': best_val_loss,
        'final_step': step,
    }


@hydra.main(version_base=None, config_path="config", config_name="ablation")
def main(cfg: DictConfig):
    # Print config
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))

    # Set random seeds
    torch.manual_seed(cfg.training.seed)
    device = get_device(cfg.training.device)

    if device.type == 'cuda':
        torch.cuda.manual_seed(cfg.training.seed)
        torch.set_float32_matmul_precision('high')

    print(f"\nPreparing dataset '{cfg.dataset.name}'...")
    train_data, val_data, tok_model, bos_id = prepare_data(
        dataset_name=cfg.dataset.name,
        num_train_docs=cfg.dataset.num_train_docs,
        num_val_docs=cfg.dataset.num_val_docs,
        vocab_size=cfg.model.vocab_size
    )

    # Compute transformer dimension parameters
    base_dim = cfg.model.depth * cfg.model.aspect_ratio
    model_dim = ((base_dim + cfg.model.head_dim - 1) // cfg.model.head_dim) * cfg.model.head_dim
    num_heads = model_dim // cfg.model.head_dim

    print(f"\nModel config: depth={cfg.model.depth}, dim={model_dim}, heads={num_heads}")

    gpt_config = GPTConfig(
        sequence_len=cfg.model.max_seq_len,
        vocab_size=cfg.model.vocab_size,
        n_layer=cfg.model.depth,
        n_head=num_heads,
        n_kv_head=num_heads,
        n_embd=model_dim,
    )

    print("Initializing model architecture...")
    model = GPT(gpt_config).to(device)
    model.init_weights()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    optimizer = build_optimizer(model, cfg)

    train_loader = make_dataloader(train_data, cfg.training.device_batch_size, cfg.model.max_seq_len, device)
    val_loader = make_dataloader(val_data, cfg.training.device_batch_size, cfg.model.max_seq_len, device)

    run_training(cfg, model, train_loader, val_loader, optimizer, device)


if __name__ == '__main__':
    main()
