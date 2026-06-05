"""
Enhanced Training Script with Comprehensive Optimization Geometry Logging

This script extends the base training loop with detailed tracking of:
- Hessian eigenvalue spectrum evolution (via Lanczos)
- Per-layer gradient norms and their evolution
- Weight matrix effective rank dynamics
- Gradient subspace stability (principal angle tracking)
- Cross-layer and temporal gradient similarity

Designed for comparing optimization geometry across SGD, Adam, AdamW.
"""

import os
import gc
import json
import math
import time
from contextlib import nullcontext
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np

import hydra
from omegaconf import DictConfig, OmegaConf

from models import GPT, GPTConfig
from data import prepare_data, make_dataloader
from optim.wsd_scheduler import get_wsd_lr_multiplier
from optim.quantization import register_quantization_hooks
from utils.hvp import power_iteration
from utils.metrics import compute_cosine_similarity
from utils.logging import CSVLogger, WandbLogger
from utils.geometry import (
    OptimizationGeometryTracker,
    lanczos_iteration,
    compute_layer_gradient_norms,
    compute_layer_effective_ranks,
    categorize_layers
)


def get_device(device_setting="auto"):
    """Select CUDA, MPS, or CPU device."""
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
    """Construct the requested optimizer (AdamW, Muon, SGD, Adam)."""
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
    else:  # adam
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.95))

    if not isinstance(optimizer, list):
        optimizer = [optimizer]

    for opt in optimizer:
        for group in opt.param_groups:
            group['initial_lr'] = group['lr']

    return optimizer


def update_learning_rates(optimizer, multiplier):
    """Update learning rate for all parameter groups."""
    for opt in optimizer:
        for group in opt.param_groups:
            group['lr'] = group['initial_lr'] * multiplier


@torch.no_grad()
def evaluate(model, val_loader, num_batches, device, autocast_ctx):
    """Evaluate loss and perplexity on validation set."""
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


class GeometryLogger:
    """
    Extended logger for geometry metrics with support for:
    - Scalar metrics (CSV compatible)
    - Per-layer metrics (JSON for detailed analysis)
    - Eigenvalue spectra (NumPy arrays)
    """

    def __init__(self, output_dir: str, optimizer_name: str):
        self.output_dir = output_dir
        self.optimizer_name = optimizer_name
        os.makedirs(output_dir, exist_ok=True)

        # Scalar metrics logger
        self.scalar_csv_path = os.path.join(output_dir, f"metrics_{optimizer_name}.csv")
        self.scalar_logger = CSVLogger(self.scalar_csv_path)

        # Per-layer metrics storage
        self.layer_metrics_path = os.path.join(output_dir, f"layer_metrics_{optimizer_name}.json")
        self.layer_metrics = []

        # Eigenvalue spectra storage
        self.eigenvalue_history = []

        # Cross-layer similarity matrices
        self.cross_layer_sim_history = []

    def log_scalar(self, metrics: dict):
        """Log scalar metrics to CSV."""
        self.scalar_logger.log(metrics)

    def log_layer_metrics(self, step: int, layer_data: dict):
        """Store per-layer metrics."""
        self.layer_metrics.append({
            'step': step,
            **layer_data
        })

    def log_eigenvalue_spectrum(self, step: int, eigenvalues: np.ndarray):
        """Store eigenvalue spectrum."""
        self.eigenvalue_history.append({
            'step': step,
            'eigenvalues': eigenvalues.tolist() if isinstance(eigenvalues, np.ndarray) else eigenvalues
        })

    def log_cross_layer_similarity(self, step: int, sim_matrix: np.ndarray, layer_names: list):
        """Store cross-layer similarity matrix."""
        self.cross_layer_sim_history.append({
            'step': step,
            'similarity_matrix': sim_matrix.tolist() if isinstance(sim_matrix, np.ndarray) else sim_matrix,
            'layer_names': layer_names
        })

    def save_all(self):
        """Save all accumulated data to disk."""
        # Save layer metrics
        with open(self.layer_metrics_path, 'w') as f:
            json.dump(self.layer_metrics, f, indent=2)

        # Save eigenvalue history
        eigenvalue_path = os.path.join(self.output_dir, f"eigenvalues_{self.optimizer_name}.json")
        with open(eigenvalue_path, 'w') as f:
            json.dump(self.eigenvalue_history, f, indent=2)

        # Save cross-layer similarity history
        sim_path = os.path.join(self.output_dir, f"cross_layer_sim_{self.optimizer_name}.json")
        with open(sim_path, 'w') as f:
            json.dump(self.cross_layer_sim_history, f, indent=2)

        print(f"Saved all geometry data to {self.output_dir}/")


def run_geometry_training(cfg, model, train_loader, val_loader, optimizer, device):
    """
    Training loop with comprehensive geometry tracking.
    """
    opt_name = cfg.optimizer.type.lower()
    output_dir = cfg.logging.get('output_dir', 'geometry_results')
    logger = GeometryLogger(output_dir, opt_name)

    # Initialize geometry tracker
    geometry_tracker = OptimizationGeometryTracker(
        model,
        top_k_hessian=cfg.geometry.get('top_k_hessian', 20),
        top_k_subspace=cfg.geometry.get('top_k_subspace', 10)
    )

    # Get layer categorization for analysis
    layer_categories = categorize_layers(model)
    print(f"Layer categories: {[(k, len(v)) for k, v in layer_categories.items()]}")

    wandb_logger = WandbLogger(
        use_wandb=cfg.logging.use_wandb,
        project=cfg.logging.wandb_project,
        entity=cfg.logging.wandb_entity,
        config=OmegaConf.to_container(cfg, resolve=True)
    )

    t_start = time.time()
    total_training_time = 0
    step = 0
    warmup_steps = 5

    autocast_ctx = get_autocast_context(device)
    tokens_per_fwdbwd = cfg.training.device_batch_size * cfg.model.max_seq_len
    grad_accum_steps = max(1, cfg.training.total_batch_size // tokens_per_fwdbwd)

    # Geometry logging settings
    geometry_interval = cfg.geometry.get('log_interval', 10)
    hessian_interval = cfg.geometry.get('hessian_interval', 50)  # Hessian is expensive
    lanczos_iterations = cfg.geometry.get('lanczos_iterations', 30)

    print(f"Starting geometry-tracked training: optimizer={opt_name}, grad_accum={grad_accum_steps}")
    print(f"Geometry logging every {geometry_interval} steps, Hessian every {hessian_interval} steps")

    while True:
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elif device.type == 'mps' and hasattr(torch.mps, 'synchronize'):
            torch.mps.synchronize()

        t0 = time.time()

        # Zero gradients
        for opt in optimizer:
            opt.zero_grad(set_to_none=True)

        # Gradient accumulation
        train_loss = 0.0
        for micro_step in range(grad_accum_steps):
            x, y = next(train_loader)
            with autocast_ctx:
                loss = model(x, y)
            train_loss += loss.detach()
            loss = loss / grad_accum_steps
            loss.backward()

        train_loss /= grad_accum_steps

        # Learning rate schedule
        progress = min(total_training_time / cfg.training.time_budget, 1.0) if cfg.training.time_budget > 0 else 0
        if cfg.training.use_wsd:
            lrm = get_wsd_lr_multiplier(progress, warmup_ratio=0.05, warmdown_ratio=0.2, final_lr_frac=0.0)
        else:
            lrm = get_wsd_lr_multiplier(progress, warmup_ratio=0.05, warmdown_ratio=0.3, final_lr_frac=0.1)

        update_learning_rates(optimizer, lrm)

        # Track delta_theta for alignment analysis
        prev_params = [p.clone().detach() for p in model.parameters() if p.requires_grad]

        # Optimizer step
        for opt in optimizer:
            opt.step()

        # Calculate parameter updates
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

        if step > warmup_steps:
            total_training_time += dt

        # =====================================================================
        # GEOMETRY LOGGING
        # =====================================================================
        if (step + 1) % geometry_interval == 0:
            # Per-layer gradient norms
            grad_norms = compute_layer_gradient_norms(model)
            total_grad_norm = sum(v ** 2 for v in grad_norms.values()) ** 0.5

            # Weight matrix effective ranks
            effective_ranks = compute_layer_effective_ranks(model)
            avg_eff_rank = np.mean(list(effective_ranks.values())) if effective_ranks else 0

            # Subspace overlap (temporal)
            subspace_overlaps = geometry_tracker.subspace_tracker.update()
            avg_subspace_overlap = np.mean(list(subspace_overlaps.values())) if subspace_overlaps else 1.0

            # Temporal gradient similarity
            temporal_sims = geometry_tracker.similarity_tracker.compute_temporal_similarity()
            avg_temporal_sim = np.mean(list(temporal_sims.values())) if temporal_sims else 1.0

            # Cross-layer similarity
            cross_layer_sim, layer_names = geometry_tracker.similarity_tracker.compute_cross_layer_similarity()
            if len(cross_layer_sim) > 0:
                # Off-diagonal mean (excludes self-similarity)
                mask = ~np.eye(cross_layer_sim.shape[0], dtype=bool)
                avg_cross_layer_sim = cross_layer_sim[mask].mean() if mask.sum() > 0 else 0
                logger.log_cross_layer_similarity(step, cross_layer_sim, layer_names)
            else:
                avg_cross_layer_sim = 0

            # Compute per-layer-type averages for detailed analysis
            def get_layer_type_avg(metrics_dict, pattern):
                vals = [v for k, v in metrics_dict.items() if pattern in k]
                return np.mean(vals) if vals else 0.0

            # Store per-layer data with type breakdowns
            layer_data = {
                'grad_norms': grad_norms,
                'effective_ranks': effective_ranks,
                'subspace_overlaps': subspace_overlaps,
                'temporal_similarities': temporal_sims,
                # Aggregates by layer type
                'grad_norm_attn_q': get_layer_type_avg(grad_norms, 'c_q'),
                'grad_norm_attn_k': get_layer_type_avg(grad_norms, 'c_k'),
                'grad_norm_attn_v': get_layer_type_avg(grad_norms, 'c_v'),
                'grad_norm_mlp_fc': get_layer_type_avg(grad_norms, 'c_fc'),
                'grad_norm_mlp_proj': get_layer_type_avg(grad_norms, 'c_proj'),
                'subspace_overlap_attn': get_layer_type_avg(subspace_overlaps, 'attn'),
                'subspace_overlap_mlp': get_layer_type_avg(subspace_overlaps, 'mlp'),
                'temporal_sim_attn': get_layer_type_avg(temporal_sims, 'attn'),
                'temporal_sim_mlp': get_layer_type_avg(temporal_sims, 'mlp'),
            }
            logger.log_layer_metrics(step, layer_data)

        # =====================================================================
        # HESSIAN EIGENVALUE SPECTRUM (expensive, less frequent)
        # =====================================================================
        compute_hessian = (step + 1) % hessian_interval == 0
        lambda_max = 0.0
        cos_sim = 0.0

        if compute_hessian:
            # Full Lanczos for eigenvalue spectrum
            eigenvalues, T_matrix = lanczos_iteration(
                model, train_loader,
                k=cfg.geometry.get('top_k_hessian', 20),
                num_iterations=lanczos_iterations,
                device=device.type
            )

            lambda_max = eigenvalues[0] if len(eigenvalues) > 0 else 0.0
            lambda_2 = eigenvalues[1] if len(eigenvalues) > 1 else 0.0
            spectral_gap = lambda_max - lambda_2

            # Power iteration for eigenvector (for alignment)
            _, v_max = power_iteration(model, train_loader, num_iterations=5, device=device.type)
            cos_sim = compute_cosine_similarity(delta_theta, v_max)

            # Log eigenvalue spectrum
            logger.log_eigenvalue_spectrum(step, eigenvalues)

            print(f"Step {step:04d} | Hessian: lambda_max={lambda_max:.4f}, gap={spectral_gap:.4f}, cos_sim={cos_sim:.4f}")

        # =====================================================================
        # VALIDATION & SCALAR LOGGING
        # =====================================================================
        if (step + 1) % cfg.training.eval_interval == 0:
            val_loss, val_ppl = evaluate(model, val_loader, 10, device, autocast_ctx)

            # Compile scalar metrics
            scalar_metrics = {
                'step': step,
                'train_loss': train_loss.item(),
                'val_loss': val_loss,
                'val_perplexity': val_ppl,
                'lr_multiplier': lrm,
                'total_training_time': total_training_time,
                'total_grad_norm': total_grad_norm if 'total_grad_norm' in dir() else 0,
                'avg_effective_rank': avg_eff_rank if 'avg_eff_rank' in dir() else 0,
                'avg_subspace_overlap': avg_subspace_overlap if 'avg_subspace_overlap' in dir() else 1,
                'avg_temporal_similarity': avg_temporal_sim if 'avg_temporal_sim' in dir() else 1,
                'avg_cross_layer_similarity': avg_cross_layer_sim if 'avg_cross_layer_sim' in dir() else 0,
                'lambda_max': lambda_max,
                'cos_sim': cos_sim
            }

            logger.log_scalar(scalar_metrics)
            wandb_logger.log(scalar_metrics)

            print(
                f"Step {step:04d} | Loss: {train_loss.item():.4f} | Val: {val_loss:.4f} | "
                f"GradNorm: {scalar_metrics['total_grad_norm']:.4f} | "
                f"EffRank: {scalar_metrics['avg_effective_rank']:.2f} | "
                f"SubspaceOvl: {scalar_metrics['avg_subspace_overlap']:.4f}"
            )

        step += 1

        if step > warmup_steps and total_training_time >= cfg.training.time_budget:
            break

    # Save all accumulated geometry data
    logger.save_all()
    print(f"Training completed. Results saved to {output_dir}/")


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

    # Compute transformer dimensions
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

    # Run geometry-tracked training
    run_geometry_training(cfg, model, train_loader, val_loader, optimizer, device)


if __name__ == '__main__':
    main()
