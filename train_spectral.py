"""
Training script with Spectral Analysis Tracking

Tracks spectral properties of weights (W), gradients (G), and weight updates (ΔW)
throughout training to compare optimizer spectral solutions.

Hypothesis:
- Adam: Sparse/low-rank spectra (low spectral entropy, low effective rank)
- Muon: Dense/full-rank spectra (high spectral entropy, high effective rank)

Usage:
    python train_spectral.py optimizer.type=adam training.max_steps=5000
    python train_spectral.py optimizer.type=muon training.max_steps=3000
"""

import os
import gc
import math
import time
import json
from contextlib import nullcontext

import torch
import torch.nn.functional as F

import hydra
from omegaconf import DictConfig, OmegaConf

from models import GPT, GPTConfig
from data import prepare_data, make_dataloader
from optim.wsd_scheduler import get_wsd_lr_multiplier
from optim.quantization import register_quantization_hooks
from optim.adam_ns import AdamNS
from utils.hvp import power_iteration
from utils.metrics import compute_cosine_similarity
from utils.logging import CSVLogger, WandbLogger
from utils.spectral import SpectralTracker, SpectralLogger
from utils.hessian import compute_hessian_spectrum, LayerWiseHessianTracker


def get_device(device_setting="auto"):
    """
    Selects CUDA, MPS, or CPU device automatically or uses override.
    """
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
    """
    Returns appropriate autocast context for the active device.
    """
    if device.type == 'cuda':
        return torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)
    else:
        return nullcontext()


def build_optimizer(model, cfg):
    """
    Constructs the requested optimizer.

    Supported types:
    - adamw: Standard AdamW
    - adam: Standard Adam
    - muon: Muon (Newton-Schulz) for matrix params + AdamW for others
    - sgd: SGD with momentum
    - adam_ns: AdamNS with momentum orthogonalization (default)
    - adam_ns_grad: AdamNS with gradient orthogonalization
    - adam_ns_update: AdamNS with update orthogonalization
    - adam_ns_momentum: AdamNS with momentum orthogonalization (explicit)
    """
    opt_type = cfg.optimizer.type.lower()
    lr = cfg.optimizer.lr
    weight_decay = cfg.optimizer.weight_decay

    print(f"Building optimizer: {opt_type.upper()}")

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

    elif opt_type.startswith('adam_ns'):
        # AdamNS variants: adam_ns, adam_ns_grad, adam_ns_update, adam_ns_momentum
        ns_cfg = cfg.get('adam_ns', {})
        ns_iters = ns_cfg.get('ns_iters', 5)
        warmup_steps = ns_cfg.get('warmup_steps', cfg.training.get('warmup_steps', 200))

        # Determine NS mode from optimizer type
        if opt_type == 'adam_ns_grad':
            ns_mode = 'grad'
        elif opt_type == 'adam_ns_update':
            ns_mode = 'update'
        else:  # adam_ns or adam_ns_momentum
            ns_mode = 'momentum'

        print(f"  NS mode: {ns_mode}, NS iters: {ns_iters}, warmup: {warmup_steps}")

        optimizer = AdamNS(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=weight_decay,
            ns_mode=ns_mode,
            ns_iters=ns_iters,
            warmup_steps=warmup_steps,
            matrix_only=True,
        )

    else:  # adam
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.95))

    if not isinstance(optimizer, list):
        optimizer = [optimizer]

    for opt in optimizer:
        for group in opt.param_groups:
            group['initial_lr'] = group['lr']

    return optimizer


def update_learning_rates(optimizer, multiplier):
    """
    Updates the learning rate for all parameter groups using the multiplier.
    """
    for opt in optimizer:
        for group in opt.param_groups:
            group['lr'] = group['initial_lr'] * multiplier


@torch.no_grad()
def evaluate(model, val_loader, num_batches, device, autocast_ctx):
    """
    Evaluates loss and perplexity on the validation dataset.
    """
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


def run_training(cfg, model, train_loader, val_loader, optimizer, device):
    """
    Training loop with integrated spectral tracking.
    """
    # Setup loggers
    output_dir = cfg.logging.output_dir
    os.makedirs(output_dir, exist_ok=True)

    optimizer_name = cfg.optimizer.type.lower()
    csv_logger = CSVLogger(os.path.join(output_dir, f'metrics_{optimizer_name}.csv'))

    wandb_logger = WandbLogger(
        use_wandb=cfg.logging.use_wandb,
        project=cfg.logging.wandb_project,
        entity=cfg.logging.wandb_entity,
        config=OmegaConf.to_container(cfg, resolve=True)
    )

    # Initialize spectral tracker
    spectral_track_every = cfg.spectral.track_every
    spectral_min_dim = cfg.spectral.min_matrix_dim

    print(f"Initializing SpectralTracker (track every {spectral_track_every} steps, min_dim={spectral_min_dim})")
    spectral_tracker = SpectralTracker(
        model=model,
        track_every=spectral_track_every,
        min_matrix_dim=spectral_min_dim,
        eps=1e-6,
        delta_w_min_norm=1e-8,
    )

    spectral_logger = SpectralLogger(output_dir, optimizer_name)

    print(f"Tracking {len(spectral_tracker.get_tracked_layers())} layers:")
    for layer_name in spectral_tracker.get_tracked_layers()[:5]:
        print(f"  - {layer_name}")
    if len(spectral_tracker.get_tracked_layers()) > 5:
        print(f"  ... and {len(spectral_tracker.get_tracked_layers()) - 5} more")

    total_training_time = 0
    step = 0
    max_steps = cfg.training.max_steps
    eval_interval = cfg.training.eval_interval

    # Initialize Hessian tracking
    hessian_cfg = cfg.get('hessian', {})
    hessian_enabled = hessian_cfg.get('enabled', True)
    hessian_interval = hessian_cfg.get('compute_interval', eval_interval)
    hessian_top_k = hessian_cfg.get('top_k', 5)
    hessian_lanczos_iters = hessian_cfg.get('lanczos_iterations', 20)
    hessian_trace_samples = hessian_cfg.get('trace_samples', 5)

    if hessian_enabled:
        hessian_tracker = LayerWiseHessianTracker(model)
        hessian_log_path = os.path.join(output_dir, f'hessian_{optimizer_name}.jsonl')
        print(f"Hessian tracking enabled (every {hessian_interval} steps)")
    else:
        hessian_tracker = None
        hessian_log_path = None
    log_interval = cfg.training.get('log_interval', 10)
    warmup_steps = cfg.training.get('warmup_steps', 100)

    autocast_ctx = get_autocast_context(device)
    tokens_per_fwdbwd = cfg.training.device_batch_size * cfg.model.max_seq_len
    grad_accum_steps = max(1, cfg.training.total_batch_size // tokens_per_fwdbwd)

    print(f"\nStarting training:")
    print(f"  Max steps: {max_steps}")
    print(f"  Eval interval: {eval_interval}")
    print(f"  Log interval: {log_interval}")
    print(f"  Grad accum steps: {grad_accum_steps}")
    print(f"  Spectral tracking every {spectral_track_every} steps")
    print("=" * 60)

    while step < max_steps:
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elif device.type == 'mps' and hasattr(torch.mps, 'synchronize'):
            torch.mps.synchronize()

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

        # Learning rate scheduling (step-based)
        progress = step / max_steps
        warmup_ratio = warmup_steps / max_steps
        lrm = get_wsd_lr_multiplier(progress, warmup_ratio=warmup_ratio, warmdown_ratio=0.2, final_lr_frac=0.0)

        update_learning_rates(optimizer, lrm)

        # ===== SPECTRAL TRACKING: PRE-OPTIMIZER =====
        # Record gradient metrics and snapshot weights for step_update
        spectral_tracker.step_pre_optimizer(step)

        # Track delta_theta for geometric alignment
        prev_params = [p.clone().detach() for p in model.parameters() if p.requires_grad]

        # Optimizer step
        for opt in optimizer:
            opt.step()

        # ===== SPECTRAL TRACKING: POST-OPTIMIZER =====
        # Record weight metrics, delta_W, and step_update
        spectral_tracker.step_post_optimizer(step)

        # Log spectral aggregates
        if spectral_tracker.should_track(step):
            aggregate = spectral_tracker.aggregate(step)
            spectral_logger.log_aggregate(step, aggregate)

            # Log per-layer metrics for detailed analysis
            metrics_at_step = spectral_tracker.get_metrics_at_step(step)
            for layer_name, matrix_types in metrics_at_step.items():
                for matrix_type, metrics in matrix_types.items():
                    spectral_logger.log_layer_metrics(step, layer_name, matrix_type, metrics)

        # Calculate parameter updates (delta_theta)
        delta_theta = [
            p.detach() - prev for p, prev in zip((p for p in model.parameters() if p.requires_grad), prev_params)
        ]

        # Check for divergence
        if math.isnan(train_loss.item()) or train_loss.item() > 100:
            print(f"Training diverged at step {step}!")
            break

        if device.type == 'cuda':
            torch.cuda.synchronize()
        elif device.type == 'mps' and hasattr(torch.mps, 'synchronize'):
            torch.mps.synchronize()

        t1 = time.time()
        dt = t1 - t0

        total_training_time += dt

        # Periodic validation & logging (at multiples of eval_interval: k, 2k, 3k, ...)
        if step > 0 and step % eval_interval == 0:
            val_loss, val_ppl = evaluate(model, val_loader, 10, device, autocast_ctx)

            # Compute Hessian curvature metrics (basic - lambda_max via power iteration)
            lambda_max, v_max = power_iteration(model, train_loader, num_iterations=5, device=device.type)
            cos_sim = compute_cosine_similarity(delta_theta, v_max)

            # Full Hessian spectrum computation (trace, effective rank, Lanczos spectrum)
            hessian_metrics = {}
            if hessian_enabled and step % hessian_interval == 0:
                try:
                    hessian_spectrum = compute_hessian_spectrum(
                        model, train_loader,
                        num_eigenvalues=hessian_top_k,
                        lanczos_iterations=hessian_lanczos_iters,
                        trace_samples=hessian_trace_samples,
                        device=device.type
                    )

                    # Layer-wise diagonal Hessian approximation
                    layer_diag = hessian_tracker.compute_diagonal_hessian(train_loader, device.type)

                    hessian_metrics = {
                        'hessian_lambda_max': hessian_spectrum['lambda_max'],
                        'hessian_lambda_min': hessian_spectrum['lambda_min'],
                        'hessian_trace': hessian_spectrum['trace'],
                        'hessian_trace_std': hessian_spectrum['trace_std'],
                        'hessian_effective_rank': hessian_spectrum['effective_rank'],
                        'hessian_spectral_norm': hessian_spectrum['spectral_norm'],
                        'hessian_condition_number': hessian_spectrum['condition_number'],
                        'hessian_spectral_entropy': hessian_spectrum['spectral_entropy'],
                        'hessian_normalized_spectral_entropy': hessian_spectrum['normalized_spectral_entropy'],
                        'hessian_num_positive_eigs': hessian_spectrum['num_positive_eigenvalues'],
                        'hessian_num_negative_eigs': hessian_spectrum['num_negative_eigenvalues'],
                    }

                    # Log full Hessian record to JSONL
                    hessian_record = {
                        'step': step,
                        **hessian_spectrum,
                        'layer_diagonal': layer_diag,
                    }
                    with open(hessian_log_path, 'a') as f:
                        f.write(json.dumps(hessian_record) + '\n')

                    print(f"    Hessian: λ_max={hessian_spectrum['lambda_max']:.2f}, "
                          f"trace={hessian_spectrum['trace']:.2f}, "
                          f"eff_rank={hessian_spectrum['effective_rank']:.2f}")

                except Exception as e:
                    print(f"    Hessian computation failed: {e}")

            # Get latest spectral aggregates for logging
            latest_agg = spectral_tracker.aggregate()

            metrics = {
                'step': step,
                'train_loss': train_loss.item(),
                'val_loss': val_loss,
                'val_perplexity': val_ppl,
                'lambda_max': lambda_max,
                'cos_sim': cos_sim,
                'lr_multiplier': lrm,
                'total_training_time': total_training_time,
                **hessian_metrics,  # Include full Hessian spectrum metrics
            }

            # Add spectral metrics to log for ALL matrix types
            csv_matrix_types = ['W', 'G', 'delta_W', 'step_update']
            csv_metric_keys = [
                'mean_spectral_entropy',
                'mean_normalized_spectral_entropy',
                'mean_stable_rank',
                'mean_normalized_stable_rank',
                'mean_participation_ratio',
                'mean_normalized_participation_ratio',
                'mean_effective_rank_90',
                'mean_effective_rank_99',
                'mean_normalized_effective_rank_90',
                'mean_normalized_effective_rank_99',
                'mean_sigma_max',
                'mean_sigma_min',
                'mean_condition_number',
                'mean_frobenius_norm',
                'mean_numerical_rank',
            ]
            for matrix_type in csv_matrix_types:
                if matrix_type in latest_agg:
                    for key in csv_metric_keys:
                        value = latest_agg[matrix_type].get(key)
                        if value is not None:
                            metrics[f'spectral_{matrix_type}_{key}'] = value

            # Print progress with spectral summary (using normalized metrics)
            w_entropy = latest_agg.get('W', {}).get('mean_normalized_spectral_entropy', 0)
            w_stable_rank = latest_agg.get('W', {}).get('mean_normalized_stable_rank', 0)
            dw_entropy = latest_agg.get('delta_W', {}).get('mean_normalized_spectral_entropy', 0)
            dw_stable_rank = latest_agg.get('delta_W', {}).get('mean_normalized_stable_rank', 0)

            print(
                f"Step {step:04d} | Train: {train_loss.item():.4f} | Val: {val_loss:.4f} | "
                f"λ_max: {lambda_max:.2f} | H̃(W): {w_entropy:.3f} | r̃(W): {w_stable_rank:.3f} | "
                f"H̃(ΔW): {dw_entropy:.3f} | r̃(ΔW): {dw_stable_rank:.3f} | Time: {total_training_time:.1f}s"
            )

            csv_logger.log(metrics)
            wandb_logger.log(metrics)

        elif step > 0 and step % log_interval == 0:
            # Quick progress update
            print(f"Step {step:5d}/{max_steps} | Train: {train_loss.item():.4f} | Time: {dt:.3f}s")

        step += 1

    # ===== SAVE SPECTRAL DATA =====
    print("\nSaving spectral tracking data...")

    # Save aggregate and layer metrics
    spectral_logger.save_all()

    # Save full spectral export (including singular values)
    full_export_path = os.path.join(output_dir, f'spectral_full_{optimizer_name}.json')
    spectral_tracker.export_to_json(full_export_path)
    print(f"Saved full spectral data to: {full_export_path}")

    # Save tracker state for potential resumption
    tracker_state_path = os.path.join(output_dir, f'spectral_tracker_state_{optimizer_name}.pt')
    torch.save(spectral_tracker.state_dict(), tracker_state_path)

    print(f"Training completed. All data saved to: {output_dir}")

    # Print final spectral summary
    print("\n" + "=" * 60)
    print("FINAL SPECTRAL SUMMARY")
    print("=" * 60)

    final_agg = spectral_tracker.aggregate()
    for matrix_type, metrics in final_agg.items():
        if metrics:
            print(f"\n{matrix_type}:")
            print(f"  Normalized Spectral Entropy: {metrics.get('mean_normalized_spectral_entropy', 'N/A'):.4f}")
            print(f"  Stable Rank: {metrics.get('mean_stable_rank', 'N/A'):.2f}")
            print(f"  Participation Ratio: {metrics.get('mean_participation_ratio', 'N/A'):.2f}")
            print(f"  Effective Rank (90%): {metrics.get('mean_effective_rank_90', 'N/A'):.1f}")
            print(f"  Effective Rank (99%): {metrics.get('mean_effective_rank_99', 'N/A'):.1f}")


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    # Set random seeds
    torch.manual_seed(cfg.training.seed)
    device = get_device(cfg.training.device)

    if device.type == 'cuda':
        torch.cuda.manual_seed(cfg.training.seed)
        torch.set_float32_matmul_precision('high')
    elif device.type == 'mps':
        if hasattr(torch.mps, 'manual_seed'):
            torch.mps.manual_seed(cfg.training.seed)

    print(f"Preparing dataset '{cfg.dataset.name}'...")
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

    if cfg.training.quantize_grads:
        print("Registering INT8 Gradient Quantization hooks...")
        register_quantization_hooks(model)

    optimizer = build_optimizer(model, cfg)

    train_loader = make_dataloader(train_data, cfg.training.device_batch_size, cfg.model.max_seq_len, device)
    val_loader = make_dataloader(val_data, cfg.training.device_batch_size, cfg.model.max_seq_len, device)

    run_training(cfg, model, train_loader, val_loader, optimizer, device)


if __name__ == '__main__':
    main()
