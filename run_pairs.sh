#!/usr/bin/env bash
set -e

# AutoNEB connections between cached endpoints. Stage 1 must print
# "loaded endpoint ... from cache" for every endpoint.

python connect.py training.device=cuda 'connect.seeds=[1]' \
  'connect.optimizers=[adamw,adam]' connect.within_pairs=false connect.cross_pairs=true \
  optimizer.muon_lr=0.05 connect.endpoint_dir=endpoints/ logging.use_wandb=true
  
python connect.py training.device=cuda 'connect.seeds=[1,2]' \
  'connect.optimizers=[adam]' connect.within_pairs=true connect.cross_pairs=false \
  connect.endpoint_dir=endpoints/ logging.use_wandb=true

python connect.py training.device=cuda 'connect.seeds=[1]' \
  'connect.optimizers=[adamw,muon]' connect.within_pairs=false connect.cross_pairs=true \
  optimizer.muon_lr=0.05 connect.endpoint_dir=endpoints/ logging.use_wandb=true

python connect.py training.device=cuda 'connect.seeds=[1]' \
  'connect.optimizers=[adamw,adamns]' connect.within_pairs=false connect.cross_pairs=true \
  optimizer.muon_lr=0.05 connect.endpoint_dir=endpoints/ logging.use_wandb=true

python connect.py training.device=cuda 'connect.seeds=[1]' \
  'connect.optimizers=[adamw,adamgradns]' connect.within_pairs=false connect.cross_pairs=true \
  optimizer.muon_lr=0.05 connect.endpoint_dir=endpoints/ logging.use_wandb=true

python connect.py training.device=cuda 'connect.seeds=[1]' \
  'connect.optimizers=[adamw,adamupdns]' connect.within_pairs=false connect.cross_pairs=true \
  optimizer.muon_lr=0.05 connect.endpoint_dir=endpoints/ logging.use_wandb=true

python connect.py training.device=cuda 'connect.seeds=[1]' \
  'connect.optimizers=[adam,muon]' connect.within_pairs=false connect.cross_pairs=true \
  optimizer.muon_lr=0.05 connect.endpoint_dir=endpoints/ logging.use_wandb=true

python connect.py training.device=cuda 'connect.seeds=[1]' \
  'connect.optimizers=[adam,adamns]' connect.within_pairs=false connect.cross_pairs=true \
  optimizer.muon_lr=0.05 connect.endpoint_dir=endpoints/ logging.use_wandb=true

python connect.py training.device=cuda 'connect.seeds=[1]' \
  'connect.optimizers=[adam,adamgradns]' connect.within_pairs=false connect.cross_pairs=true \
  optimizer.muon_lr=0.05 connect.endpoint_dir=endpoints/ logging.use_wandb=true

python connect.py training.device=cuda 'connect.seeds=[1]' \
  'connect.optimizers=[adam,adamupdns]' connect.within_pairs=false connect.cross_pairs=true \
  optimizer.muon_lr=0.05 connect.endpoint_dir=endpoints/ logging.use_wandb=true

python connect.py training.device=cuda 'connect.seeds=[1]' \
  'connect.optimizers=[muon,adamns]' connect.within_pairs=false connect.cross_pairs=true \
  optimizer.muon_lr=0.05 connect.endpoint_dir=endpoints/ logging.use_wandb=true

python connect.py training.device=cuda 'connect.seeds=[1]' \
  'connect.optimizers=[muon,adamgradns]' connect.within_pairs=false connect.cross_pairs=true \
  optimizer.muon_lr=0.05 connect.endpoint_dir=endpoints/ logging.use_wandb=true

python connect.py training.device=cuda 'connect.seeds=[1]' \
  'connect.optimizers=[muon,adamupdns]' connect.within_pairs=false connect.cross_pairs=true \
  optimizer.muon_lr=0.05 connect.endpoint_dir=endpoints/ logging.use_wandb=true

python connect.py training.device=cuda 'connect.seeds=[1]' \
  'connect.optimizers=[adamns,adamgradns]' connect.within_pairs=false connect.cross_pairs=true \
  optimizer.muon_lr=0.05 connect.endpoint_dir=endpoints/ logging.use_wandb=true

python connect.py training.device=cuda 'connect.seeds=[1]' \
  'connect.optimizers=[adamns,adamupdns]' connect.within_pairs=false connect.cross_pairs=true \
  optimizer.muon_lr=0.05 connect.endpoint_dir=endpoints/ logging.use_wandb=true

python connect.py training.device=cuda 'connect.seeds=[1]' \
  'connect.optimizers=[adamgradns,adamupdns]' connect.within_pairs=false connect.cross_pairs=true \
  optimizer.muon_lr=0.05 connect.endpoint_dir=endpoints/ logging.use_wandb=true

# Plots from the accumulated CSVs:
# python reports/generate_path_plots.py




