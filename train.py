import os
import gc
import math
import time
from contextlib import nullcontext

import torch
import torch.nn.functional as F

import hydra
from omegaconf import DictConfig, OmegaConf

from models import GPT, GPTConfig
from data import prepare_data, make_dataloader
from optim.wsd_scheduler import get_wsd_lr_multiplier
from optim.quantization import register_quantization_hooks
from utils.hvp import power_iteration
from utils.metrics import compute_cosine_similarity
from utils.logging import CSVLogger, WandbLogger

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
        # MPS or CPU - fallback to nullcontext to avoid precision/support issues
        return nullcontext()


def build_optimizer(model, cfg):
    """
    Constructs the requested optimizer (AdamW, Muon, SGD, Adam).
    """
    opt_type = cfg.optimizer.type.lower()
    lr = cfg.optimizer.lr
    weight_decay = cfg.optimizer.weight_decay
    
    print(f"Building optimizer: {opt_type.upper()}")
    if opt_type == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=weight_decay)
    elif opt_type == 'muon':
        if hasattr(torch.optim, 'Muon'):
            # Precondition embeddings & heads with AdamW, rest with Muon
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
            print("Warning: Muon not available in this PyTorch distribution. Falling back to AdamW.")
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=weight_decay)
    elif opt_type == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.95))

    if not isinstance(optimizer, list):
        optimizer = [optimizer]
        
    # Setup initial learning rate for scheduler tracking
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
    Orchestrates the modularized training loop and metrics recording.
    """
    csv_logger = CSVLogger(cfg.logging.csv_log_path)
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

    print(f"Starting training sweep: grad_accum_steps={grad_accum_steps}")
    
    while True:
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elif device.type == 'mps' and hasattr(torch.mps, 'synchronize'):
            torch.mps.synchronize()
            
        t0 = time.time()

        # Zero gradients across all optimizers
        for opt in optimizer:
            opt.zero_grad(set_to_none=True)

        # Micro-stepping gradient accumulation
        train_loss = 0.0
        for micro_step in range(grad_accum_steps):
            x, y = next(train_loader)
            with autocast_ctx:
                loss = model(x, y)
            train_loss += loss.detach()
            loss = loss / grad_accum_steps
            loss.backward()

        train_loss /= grad_accum_steps

        # Learning rate scheduler
        progress = min(total_training_time / cfg.training.time_budget, 1.0) if cfg.training.time_budget > 0 else 0
        if cfg.training.use_wsd:
            lrm = get_wsd_lr_multiplier(progress, warmup_ratio=0.05, warmdown_ratio=0.2, final_lr_frac=0.0)
        else:
            lrm = get_wsd_lr_multiplier(progress, warmup_ratio=0.05, warmdown_ratio=0.3, final_lr_frac=0.1)
        
        update_learning_rates(optimizer, lrm)

        # Track delta_theta for geometric alignment
        prev_params = [p.clone().detach() for p in model.parameters() if p.requires_grad]

        for opt in optimizer:
            opt.step()

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

        if step > warmup_steps:
            total_training_time += dt

        # Periodic validation & Hessian curvature logging
        if (step + 1) % cfg.training.eval_interval == 0:
            val_loss, val_ppl = evaluate(model, val_loader, 10, device, autocast_ctx)
            
            # Compute top Hessian eigenvalue (lambda_max) and eigenvector (v_max)
            lambda_max, v_max = power_iteration(model, train_loader, num_iterations=5, device=device.type)
            
            # Compute similarity of update step with top eigenvector
            cos_sim = compute_cosine_similarity(delta_theta, v_max)
            
            metrics = {
                'step': step,
                'train_loss': train_loss.item(),
                'val_loss': val_loss,
                'val_perplexity': val_ppl,
                'lambda_max': lambda_max,
                'cos_sim': cos_sim,
                'lr_multiplier': lrm,
                'total_training_time': total_training_time
            }
            
            print(
                f"Step {step:04d} | Train Loss: {train_loss.item():.4f} | "
                f"Val Loss: {val_loss:.4f} | Lambda_max: {lambda_max:.4f} | "
                f"Cos Sim: {cos_sim:.4f} | Time: {total_training_time:.1f}s"
            )
            
            csv_logger.log(metrics)
            wandb_logger.log(metrics)

        step += 1

        if step > warmup_steps and total_training_time >= cfg.training.time_budget:
            break

    print(f"Training completed successfully. Metrics written to '{cfg.logging.csv_log_path}'.")


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    # Set random seeds for reproducibility
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

    # Build and prepare optimizer
    optimizer = build_optimizer(model, cfg)

    # Dataloaders
    train_loader = make_dataloader(train_data, cfg.training.device_batch_size, cfg.model.max_seq_len, device)
    val_loader = make_dataloader(val_data, cfg.training.device_batch_size, cfg.model.max_seq_len, device)

    # Start training sweep
    run_training(cfg, model, train_loader, val_loader, optimizer, device)


if __name__ == '__main__':
    main()
