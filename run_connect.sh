#!/usr/bin/env bash
set -e

python -m pytest tests/ -q

# AdamW
python connect.py training.device=cuda 'connect.seeds=[3]' \
  'connect.optimizers=[adamw]' connect.within_pairs=false connect.cross_pairs=false \
  connect.endpoint_dir=endpoints/ logging.use_wandb=true

# Adam
python connect.py training.device=cuda 'connect.seeds=[3]' \
  'connect.optimizers=[adam]' connect.within_pairs=false connect.cross_pairs=false \
  connect.endpoint_dir=endpoints/ logging.use_wandb=true

# Muon
python connect.py training.device=cuda 'connect.seeds=[3]' \
  'connect.optimizers=[muon]' connect.within_pairs=false connect.cross_pairs=false \
  optimizer.muon_lr=0.05 connect.endpoint_dir=endpoints/ logging.use_wandb=true

# AdamNS (NS on momentum)
python connect.py training.device=cuda 'connect.seeds=[3]' \
  'connect.optimizers=[adamns]' connect.within_pairs=false connect.cross_pairs=false \
  connect.endpoint_dir=endpoints/ logging.use_wandb=true

# AdamGradNS (NS on gradient)
python connect.py training.device=cuda 'connect.seeds=[3]' \
  'connect.optimizers=[adamgradns]' connect.within_pairs=false connect.cross_pairs=false \
  connect.endpoint_dir=endpoints/ logging.use_wandb=true

# AdamUpdNS (NS on update)
python connect.py training.device=cuda 'connect.seeds=[3]' \
  'connect.optimizers=[adamupdns]' connect.within_pairs=false connect.cross_pairs=false \
  connect.endpoint_dir=endpoints/ logging.use_wandb=true

# Full run, reuses the cached endpoints:
# python connect.py training.device=cuda optimizer.muon_lr=0.05 \
#   'connect.optimizers=[adamw,muon,adam,adamns,adamgradns,adamupdns]' \
#   logging.use_wandb=true logging.wandb_entity=<your-entity>
