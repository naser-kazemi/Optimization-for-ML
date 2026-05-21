# Research Proposal: Empirical Analysis of Optimizer Geometry and Stability Under State-of-the-Art Training Regimes

## 1. Abstract

This project benchmarks modern optimization methods (SGD, AdamW, Muon) to evaluate their behavior under contemporary training conditions. While standard benchmarks focus purely on convergence speed, this study investigates the geometric mechanisms underlying optimizer stability and final task performance. We constrain the experimental scope to track the loss landscape geometry at critical training junctures. Specifically, we will analyze the Hessian eigenspectrum during Warmup-Stable-Decay (WSD) phase transitions, compare the update vector alignment of orthogonal versus diagonal preconditioners, and evaluate the implicit regularization effects of gradient quantization. The goal is to provide a rigorous, empirical explanation for why certain optimizers navigate non-convex landscapes more effectively.

## 2. Core Research Questions

The analysis will focus on three specific empirical ablations to address the following questions:

* **RQ1 (Phase Transitions):** Does the rapid reduction in learning rate during the 'Decay' phase of a WSD schedule trap the optimizer in a fundamentally sharper minimum, and do adaptive methods (AdamW) resist this sharpening better than SGD?
* **RQ2 (Preconditioning Geometry):** Do orthogonal optimizers (Muon) inherently orthogonalize their update directions away from the sharpest curvature of the loss landscape faster than diagonal methods (AdamW)?
* **RQ3 (Implicit Regularization):** Does the noise injected by coarse gradient quantization (8-bit) flatten the loss landscape, mimicking the implicit regularization typically associated with small mini-batch sizes?

## 3. Methodology and Experimental Design

To avoid the computational burden of continuous landscape mapping, we will employ targeted metrics evaluated strictly at predefined epoch boundaries using the provided single-GPU LLM codebase.

### 3.1. WSD Schedule Phase-Transition Analysis

We will track the loss landscape geometry across the boundaries of a WSD schedule.

* **Metric:** Top Hessian eigenvalue $\lambda_{max}$.
* **Implementation:** We will use Pearlmutter's forward-over-reverse automatic differentiation to compute Hessian-vector products (HVPs). We apply the Power Iteration algorithm:

$$v_{k+1} = \frac{H v_k}{\|H v_k\|}$$



Upon convergence to the principal eigenvector $v_{max}$, we compute the scalar curvature:

$$\lambda_{max} = v_{max}^T H v_{max}$$



This metric will be recorded precisely at the boundary separating the 'Stable' and 'Decay' phases.

### 3.2. Orthogonal vs. Diagonal Preconditioning Geometry

We will mathematically quantify 'stability' by comparing how different optimizers interact with the steepest directions of the loss basin.

* **Metric:** Cosine similarity between the parameter update step $\Delta \theta$ and the top eigenvector of the Hessian $v_{max}$.

$$\text{Sim} = \frac{\langle \Delta \theta, v_{max} \rangle}{\|\Delta \theta\| \|v_{max}\|}$$


* **Implementation:** We will compute this scalar dot product every N steps to determine if orthogonal optimizers (Muon) actively avoid the highest-curvature directions compared to independent coordinate-wise scaling (AdamW).

### 3.3. Gradient Quantization Ablation

We will test gradient quantization as a computationally cheap analogue to batch-size regularization.

* **Metric:** Final validation loss and the trace of the Hessian (approximated via Hutchinson's estimator) or $\lambda_{max}$ to measure basin flatness.
* **Implementation:** A PyTorch wrapper will intercept the backward pass, quantizing gradients to INT8 before passing them to the optimizer. We will compare AdamW (FP32 gradients) against AdamW (INT8 gradients).

## 4. Execution Plan and Timeline

This schedule assumes a 4-week window and execution on a single GPU node.

**Week 1: Infrastructure and Baselines**

* Clone the `llm_getting_started_notebook` baseline.
* Implement the WSD learning rate scheduler.
* Integrate the HVP computation hooks for $\lambda_{max}$ and $v_{max}$ extraction.
* Write the gradient quantization wrapper.

**Week 2: Execution of Training Sweeps**

* Run baseline training for SGD, AdamW, and Muon using standard FP32 gradients.
* Run the INT8 gradient ablation for AdamW.
* Ensure all logging strictly captures the evaluation metrics (loss, gradient norm, $\lambda_{max}$, update cosine similarity) at the predefined epoch boundaries.

**Week 3: Data Extraction and Analysis**

* Aggregate the time-series data.
* Compute the variance of the metrics across different runs if multiple seeds are feasible within compute constraints.
* Generate Wandb-style smoothed plots comparing the geometric metrics ($\lambda_{max}$ trajectories) between optimizers.

**Week 4: Synthesis and Reporting**

* Draft the 3-page report using the required LaTeX template.
* Structure the report to directly address RQ1, RQ2, and RQ3 with empirical evidence.
* Finalize the codebase, ensuring the GitHub repository contains a self-sufficient `run.py` script for complete reproducibility.
