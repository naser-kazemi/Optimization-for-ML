# OptML: Transformer Loss Landscape and Curvature Dynamics

Code for our Optimization for Machine Learning (OptML) miniproject. We train a
small GPT-style transformer and track optimization dynamics: learning rate
schedules, preconditioning (Muon vs. AdamW), gradient quantization, and loss
landscape curvature (the top Hessian eigenvalue $\lambda_{max}$ and how well the
optimizer step aligns with it).

The project started as a single notebook (`demo.ipynb`) and was split into
modules with a Hydra config so experiments can be run and overridden from the
command line.

## Layout

```
config/config.yaml       Hydra config (optimizer, schedule, device, ...)
optim/wsd_scheduler.py    Warmup-Stable-Decay learning rate schedule
optim/quantization.py     INT8 gradient quantization hook
utils/hvp.py              Hessian-vector products and power iteration
utils/metrics.py          cosine alignment of update step with top eigenvector
utils/logging.py          CSV and Weights & Biases logging
data.py                   DCLM-edu streaming and tokenizer training
models.py                 GPT model (RoPE, value embeddings, residual gates)
train.py                  training entry point
reports/                  report, plotting script, figures
metrics.csv               per-step metrics from the latest run
```

## What's implemented

- **Hydra config.** Model, dataset, training, optimizer, and logging options
  live in `config/config.yaml` and can be overridden on the command line
  (e.g. `optimizer.type=muon`, `training.quantize_grads=true`).
- **WSD scheduler** (`optim/wsd_scheduler.py`): linear warmup to the peak
  learning rate, a stable phase, then a cosine decay down to a fractional
  minimum.
- **Hessian curvature tracking** (`utils/hvp.py`, `utils/metrics.py`):
  Hessian-vector products via double backprop, power iteration to estimate the
  top eigenvalue $\lambda_{max}$ and eigenvector $v_{max}$, and the cosine
  similarity between the parameter step $\Delta\theta$ and $v_{max}$.
- **INT8 gradient quantization** (`optim/quantization.py`): a backward hook that
  quantizes and dequantizes gradients to simulate 8-bit precision.
- **Plotting** (`reports/generate_plots.py`): reads `metrics.csv` and writes the
  loss and curvature figures under `reports/`.

## Setup

Install dependencies into your active conda environment:

```bash
bash install.sh
```

## Running

Quick local check on Apple Silicon (`mps`) or CPU:

```bash
python train.py training.device=mps training.time_budget=5 training.eval_interval=1 \
  dataset.num_train_docs=100 dataset.num_val_docs=10 \
  training.total_batch_size=2 training.device_batch_size=2
```

Full runs on a CUDA node:

```bash
# AdamW baseline
python train.py optimizer.type=adamw training.device=cuda logging.use_wandb=true

# Muon
python train.py optimizer.type=muon training.device=cuda logging.use_wandb=true

# INT8 gradients
python train.py training.quantize_grads=true training.device=cuda logging.use_wandb=true
```

## Results

See [reports/project_report.md](reports/project_report.md) for the write-up and
the loss and Hessian-geometry figures.
