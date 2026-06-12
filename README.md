# 📊 OptML: Transformer Loss Landscape Optimization & Curvature Dynamics

This repository contains our modular framework for the **Optimization for Machine Learning (OptML)** miniproject, focusing on transformer training dynamics, learning rate scheduling variations, preconditioning, gradient quantization, and empirical loss landscape geometry. 

We have refactored a monolithic notebook baseline into a premium, command-line-driven, and highly parameterized codebase decorated with Meta's **Hydra** configurations. This architecture is designed to monitor, track, and visualize training metrics alongside deep loss landscape traits like the dominant Hessian eigenvalue ($\lambda_{max}$) and optimizer step alignment.

---

## 📂 Repository Architecture

The codebase is organized as follows:

```
[Project Root]/
├── config/
│   └── config.yaml          # Meta's Hydra configurations (optimizers, schedules, devices, etc.)
├── optim/
│   ├── wsd_scheduler.py     # Warmup-Stable-Decay (WSD) scheduler
│   └── quantization.py      # INT8 backward gradient quantization hook simulator
├── reports/
│   ├── project_report.md    # Comprehensive Implementation Report
│   ├── generate_plots.py    # Automated visualization rendering engine
│   ├── loss_curves.png      # Loss, perplexity & WSD schedule chart
│   └── hessian_geometry.png # Lambda_max & step cosine alignment chart
├── utils/
│   ├── hvp.py               # Hessian-Vector Product (HVP) and Power Iteration engine
│   ├── metrics.py           # Update/eigenvector cosine alignment metrics
│   └── logging.py           # Unified CSV & Weights & Biases (wandb) logger
├── data.py                  # DCLM-Edu streaming & tokenizer compilation pipeline
├── models.py                # GPT architecture (RoPE, Value Embeddings, residual gates)
├── train.py                 # Primary training entry point (Hydra decorated)
├── install.sh               # Dependency installer
├── proposal.md              # Research proposal
├── README.md                # [This File] Project guide
└── metrics.csv              # step-by-step local logged metrics
```

---

## 🛠️ Implemented Features

### 1. Robust Parametrization (Meta's Hydra Config)
Decoupled hardcoded values into `config/config.yaml`. Supports command-line overrides (e.g. `optimizer.type=muon`, `training.quantize_grads=true`) without modifying source scripts.

### 2. Warmup-Stable-Decay (WSD) Scheduler
Implements a multi-stage training schedule under `optim/wsd_scheduler.py` consisting of:
* A linear warmup to target learning rate.
* A stable phase where peak learning rate is maintained.
* A sharp cosine decay (warmdown) phase down to a configured fractional minimum.

### 3. Hessian Curvature Geometry Tracker
Tracks the structural landscape of the loss function in real-time under `utils/hvp.py` and `utils/metrics.py`:
* **Hessian-Vector Products (HVP):** Reverse-over-reverse double autograd directly over the model's unified forward loss graph.
* **Power Iteration:** Estimates the top eigenvalue ($\lambda_{max}$) and eigenvector ($v_{max}$) of the parameter Hessian.
* **Optimizer Step Alignment:** Computes the cosine similarity $\cos(\Delta\theta, v_{max})$ between the actual parameter step ($\Delta\theta = \theta_{t+1} - \theta_t$) and the dominant high-curvature direction.

### 4. Low-Precision Simulated Quantization Hook
Simulates low-precision (INT8) gradients using custom PyTorch backward hooks (`optim/quantization.py`) registered to parameters, evaluating implicit regularization effects.

### 5. Automated Visualization Suite
Includes a Python plotting engine (`reports/generate_plots.py`) that reads local `metrics.csv` files, isolates training runs, and outputs publication-quality performance graphs under `reports/`.

---

## 🚀 Getting Started

### 1. Installation
To install the required dependencies (such as Hydra, Matplotlib, and Tokenizers) in your active conda environment, run:
```bash
bash install.sh
```

### 2. Local Validation Run
To run an ultra-lightweight training check on your local machine using Apple Silicon GPU (`mps`) or CPU:
```bash
python train.py training.device=mps training.time_budget=5 training.eval_interval=1 dataset.num_train_docs=100 dataset.num_val_docs=10 training.total_batch_size=2 training.device_batch_size=2
```

### 3. Cluster Deployment
For full-scale high-performance sweeps on NVIDIA GPUs, Hydra enables clean overrides:
* **AdamW Baseline:**
  ```bash
  python train.py optimizer.type=adamw training.device=cuda logging.use_wandb=true
  ```
* **Muon Preconditioned Sweep:**
  ```bash
  python train.py optimizer.type=muon training.device=cuda logging.use_wandb=true
  ```
* **Gradient Quantization Sweep:**
  ```bash
  python train.py training.quantize_grads=true training.device=cuda logging.use_wandb=true
  ```

---

## 🔁 Reproducibility

All experiment families in this repository can be reproduced with the shell scripts below.

* Optimizer comparison experiments:
  ```bash
  /data/knezevic/Optimization-for-ML/scripts/run_optimizer_comparison.sh
  ```

* Ablation experiments:
  ```bash
  /data/knezevic/Optimization-for-ML/scripts/run_ablation_adam.sh
  /data/knezevic/Optimization-for-ML/scripts/run_ablation_muon.sh
  ```

* Linear regression experiments:
  ```bash
  /data/knezevic/Optimization-for-ML/linear_regression/run_linear.sh
  ```

For a one-command launcher that runs the experiment scripts in sequence, use:
```bash
python /data/knezevic/Optimization-for-ML/run.py
```

### Visualizations

After the experiment scripts finish, open the notebooks below to reproduce the analysis figures:

* /data/knezevic/Optimization-for-ML/notebooks/ablation_analysis.ipynb
* /data/knezevic/Optimization-for-ML/notebooks/optimizer_comparison_analysis.ipynb
* /data/knezevic/Optimization-for-ML/linear_regression/linear_regression/final_results/visualize_results.ipynb

### Note

Part of the codes in the repo, where the codes we have used for exploring the ideas and they were not used in the final report.
