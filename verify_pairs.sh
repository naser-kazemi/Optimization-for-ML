#!/usr/bin/env bash
set -e

# Dense linear-path check for all pairs among {adamw, adam, muon} x seeds {1,2,3},
# each with a random-direction control of the same length.

# within-optimizer, different inits

python verify_linear.py --device cuda --points 201 --batches 32 \
  --a endpoints/adamw_seed1_lr0.0003_steps8000.pt --b endpoints/adamw_seed2_lr0.0003_steps8000.pt

python verify_linear.py --device cuda --points 201 --batches 32 \
  --a endpoints/adamw_seed1_lr0.0003_steps8000.pt --b endpoints/adamw_seed3_lr0.0003_steps8000.pt

python verify_linear.py --device cuda --points 201 --batches 32 \
  --a endpoints/adamw_seed2_lr0.0003_steps8000.pt --b endpoints/adamw_seed3_lr0.0003_steps8000.pt

python verify_linear.py --device cuda --points 201 --batches 32 \
  --a endpoints/adam_seed1_lr0.0003_steps8000.pt --b endpoints/adam_seed2_lr0.0003_steps8000.pt

python verify_linear.py --device cuda --points 201 --batches 32 \
  --a endpoints/adam_seed1_lr0.0003_steps8000.pt --b endpoints/adam_seed3_lr0.0003_steps8000.pt

python verify_linear.py --device cuda --points 201 --batches 32 \
  --a endpoints/adam_seed2_lr0.0003_steps8000.pt --b endpoints/adam_seed3_lr0.0003_steps8000.pt

python verify_linear.py --device cuda --points 201 --batches 32 \
  --a endpoints/muon_seed1_lr0.05_steps8000.pt --b endpoints/muon_seed2_lr0.05_steps8000.pt

python verify_linear.py --device cuda --points 201 --batches 32 \
  --a endpoints/muon_seed1_lr0.05_steps8000.pt --b endpoints/muon_seed3_lr0.05_steps8000.pt

python verify_linear.py --device cuda --points 201 --batches 32 \
  --a endpoints/muon_seed2_lr0.05_steps8000.pt --b endpoints/muon_seed3_lr0.05_steps8000.pt

# cross-optimizer, shared init

python verify_linear.py --device cuda --points 201 --batches 32 \
  --a endpoints/adamw_seed1_lr0.0003_steps8000.pt --b endpoints/adam_seed1_lr0.0003_steps8000.pt

python verify_linear.py --device cuda --points 201 --batches 32 \
  --a endpoints/adamw_seed2_lr0.0003_steps8000.pt --b endpoints/adam_seed2_lr0.0003_steps8000.pt

python verify_linear.py --device cuda --points 201 --batches 32 \
  --a endpoints/adamw_seed3_lr0.0003_steps8000.pt --b endpoints/adam_seed3_lr0.0003_steps8000.pt

python verify_linear.py --device cuda --points 201 --batches 32 \
  --a endpoints/adamw_seed1_lr0.0003_steps8000.pt --b endpoints/muon_seed1_lr0.05_steps8000.pt

python verify_linear.py --device cuda --points 201 --batches 32 \
  --a endpoints/adamw_seed2_lr0.0003_steps8000.pt --b endpoints/muon_seed2_lr0.05_steps8000.pt

python verify_linear.py --device cuda --points 201 --batches 32 \
  --a endpoints/adamw_seed3_lr0.0003_steps8000.pt --b endpoints/muon_seed3_lr0.05_steps8000.pt

python verify_linear.py --device cuda --points 201 --batches 32 \
  --a endpoints/adam_seed1_lr0.0003_steps8000.pt --b endpoints/muon_seed1_lr0.05_steps8000.pt

python verify_linear.py --device cuda --points 201 --batches 32 \
  --a endpoints/adam_seed2_lr0.0003_steps8000.pt --b endpoints/muon_seed2_lr0.05_steps8000.pt

python verify_linear.py --device cuda --points 201 --batches 32 \
  --a endpoints/adam_seed3_lr0.0003_steps8000.pt --b endpoints/muon_seed3_lr0.05_steps8000.pt
