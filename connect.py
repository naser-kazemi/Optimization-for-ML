"""Mode-connectivity experiment: train endpoints with different optimizers and
seeds, then connect pairs of them with AutoNEB (Draxler et al., 2018) and
compare against linear interpolation. See config/connect.yaml."""

import math
import os
from contextlib import nullcontext
from itertools import combinations

import hydra
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from data import make_dataloader, make_fixed_eval_batches, prepare_data
from models import GPT
from optim.autoneb import AutoNEBConfig, autoneb_connect
from optim.neb import Path, measure_path
from optim.param_vector import (flatten_state_dict, load_vector_into_model,
                                make_nn_energy, make_nn_energy_stream)
from _train import build_gpt_config, build_optimizer, get_autocast_context, get_device
from utils.logging import CSVLogger, WandbLogger


def build_optimizer_for(model, cfg, opt_type):
    if opt_type == "muon" and not hasattr(torch.optim, "Muon"):
        # don't let build_optimizer silently fall back to AdamW here
        raise RuntimeError("torch.optim.Muon is not available in this PyTorch build")
    sub = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    sub.optimizer.type = opt_type
    return build_optimizer(model, sub)


def train_endpoint(model, train_loader, optimizer, autocast_ctx, grad_accum_steps,
                   max_steps, log_fn=None, eval_fn=None, eval_every=100,
                   patience=5, min_delta=0.01, warmup_steps=100, decay_steps=500,
                   final_lr_frac=0.1, grad_clip=0.0):
    """Warmup -> stable lr until eval_fn plateaus -> linear decay tail -> stop.

    max_steps is a hard cap (decay is forced to fit before it). Returns the
    number of steps actually run.
    """
    from tqdm.auto import tqdm

    warmup_steps = min(warmup_steps, max(1, max_steps // 10))
    decay_steps = min(decay_steps, max(1, max_steps // 3))
    decay_start = max_steps - decay_steps
    best = float("inf")
    stale = 0

    model.train()
    pbar = tqdm(total=max_steps, desc="  train endpoint", leave=False)
    step = 0
    while step < max_steps:
        if step < warmup_steps:
            mult = (step + 1) / warmup_steps
        elif step < decay_start:
            mult = 1.0
        else:
            t = min(1.0, (step - decay_start) / max(1, decay_steps))
            mult = 1.0 + (final_lr_frac - 1.0) * t
        for opt in optimizer:
            for group in opt.param_groups:
                base = group.setdefault('initial_lr', group['lr'])
                group['lr'] = base * mult

        for opt in optimizer:
            opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(grad_accum_steps):
            x, y = next(train_loader)
            with autocast_ctx:
                loss = model(x, y) / grad_accum_steps
            loss.backward()
            step_loss += loss.detach().item()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        for opt in optimizer:
            opt.step()
        step += 1
        pbar.update(1)
        pbar.set_postfix(loss=f"{step_loss:.4f}", lr_mult=f"{mult:.2f}",
                         best_eval=(f"{best:.4f}" if best < float("inf") else "n/a"))
        if log_fn is not None:
            log_fn(step, step_loss)

        if eval_fn is not None and step < decay_start and step % eval_every == 0:
            eval_loss = eval_fn()
            model.train()
            if eval_loss < best - min_delta:
                best = eval_loss
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    decay_start = step

        if step >= decay_start + decay_steps:
            break
    pbar.close()
    return step


def _saddle_curvature(eval_model, specs, path, energy_full, eval_batches, n_iters):
    """Top Hessian eigenvalue at the highest-loss pivot (NaN if HVP unsupported)."""
    from utils.hvp import power_iteration

    pivot_losses = [energy_full(path.coords[i])[0] for i in range(path.n_pivots)]
    idx = max(range(len(pivot_losses)), key=lambda i: pivot_losses[i])
    load_vector_into_model(eval_model, path.coords[idx], specs)

    def _loader():
        while True:
            for b in eval_batches:
                yield b

    try:
        lambda_max, _ = power_iteration(eval_model, _loader(), num_iterations=n_iters,
                                        device=eval_model.cos.device.type)
        return lambda_max
    except RuntimeError as e:
        print(f"  [warn] saddle curvature skipped: {e}")
        return float("nan")


def make_pairs(optimizers, seeds, within, cross):
    """within: same optimizer, two seeds. cross: two optimizers, rotated seeds
    (seed_i vs seed_{i+1}) so cross endpoints come from independent inits."""
    pairs = []
    if within:
        for opt in optimizers:
            for s1, s2 in combinations(seeds, 2):
                pairs.append({"kind": "within", "opt_a": opt, "seed_a": s1,
                              "opt_b": opt, "seed_b": s2})
    if cross:
        for o1, o2 in combinations(optimizers, 2):
            for i, s in enumerate(seeds):
                pairs.append({"kind": "cross", "opt_a": o1, "seed_a": s,
                              "opt_b": o2, "seed_b": seeds[(i + 1) % len(seeds)]})
    return pairs


def _endpoint_lr(cfg, opt_type):
    if opt_type == "sgd":
        return cfg.optimizer.get("sgd_lr", cfg.optimizer.lr)
    if opt_type == "muon":
        return cfg.optimizer.muon_lr
    if opt_type in ("adamns", "adamgradns", "adamupdns"):
        return cfg.optimizer.get(f"{opt_type}_lr", cfg.optimizer.lr)
    return cfg.optimizer.lr


def _endpoint_steps(cfg, opt_type):
    per_opt = cfg.connect.get("endpoint_steps_per_opt", None) or {}
    return int(per_opt.get(opt_type, cfg.connect.endpoint_steps))


@torch.no_grad()
def _mean_eval_loss(model, batches):
    model.eval()
    return sum(model(x, y).item() for x, y in batches) / len(batches)


def get_endpoints(cfg, gpt_config, train_data, device, autocast_ctx, specs,
                  eval_model, measure_batches, wandb_logger=None):
    """Train or load one endpoint per (optimizer, seed). Cached on disk; the
    filename encodes optimizer, seed, lr and step cap, so existing files are
    loaded and never overwritten."""
    seq_len = cfg.model.max_seq_len
    tokens_per_fwdbwd = cfg.training.device_batch_size * seq_len
    grad_accum_steps = max(1, cfg.training.total_batch_size // tokens_per_fwdbwd)
    endpoint_dir = to_absolute_path(cfg.connect.endpoint_dir)
    os.makedirs(endpoint_dir, exist_ok=True)

    inits, endpoints = {}, {}
    for seed in cfg.connect.seeds:
        init_file = os.path.join(endpoint_dir, f"init_seed{seed}.pt")
        if os.path.exists(init_file):
            init_vec = torch.load(init_file, map_location="cpu")
        else:
            torch.manual_seed(seed)
            init_model = GPT(gpt_config).to(device)
            init_model.init_weights()
            init_vec, _ = flatten_state_dict(init_model.state_dict())
            del init_model
            torch.save(init_vec, init_file)
        inits[seed] = init_vec

        for opt_type in cfg.connect.optimizers:
            lr = _endpoint_lr(cfg, opt_type)
            n_steps = _endpoint_steps(cfg, opt_type)
            ckpt = os.path.join(
                endpoint_dir,
                f"{opt_type}_seed{seed}_lr{lr:g}_steps{n_steps}.pt")
            if os.path.exists(ckpt):
                vec = torch.load(ckpt, map_location="cpu")
                print(f"  loaded endpoint {opt_type}/seed{seed} from cache")
            else:
                print(f"  Training endpoint {opt_type}/seed{seed} "
                      f"(lr={lr:g}, cap {n_steps} steps)...")
                model = GPT(gpt_config).to(device)
                load_vector_into_model(model, init_vec, specs)
                optimizer = build_optimizer_for(model, cfg, opt_type)
                # same batch order for every optimizer at a given seed
                loader = make_dataloader(train_data, cfg.training.device_batch_size,
                                         seq_len, device, seed=10_000 + seed)
                tag = f"endpoint/{opt_type}_s{seed}"
                log_fn = None
                if wandb_logger is not None:
                    log_fn = (lambda step, loss, _tag=tag:
                              wandb_logger.log({f"{_tag}/loss": loss, f"{_tag}/step": step}))

                def eval_fn(_model=model, _tag=tag):
                    eval_loss = _mean_eval_loss(_model, measure_batches)
                    if wandb_logger is not None:
                        wandb_logger.log({f"{_tag}/eval_loss": eval_loss})
                    return eval_loss

                steps_run = train_endpoint(
                    model, loader, optimizer, autocast_ctx, grad_accum_steps,
                    n_steps, log_fn=log_fn, eval_fn=eval_fn,
                    eval_every=cfg.connect.endpoint_eval_every,
                    patience=cfg.connect.endpoint_patience,
                    min_delta=cfg.connect.endpoint_min_delta,
                    warmup_steps=cfg.connect.endpoint_warmup_steps,
                    decay_steps=cfg.connect.endpoint_decay_steps,
                    final_lr_frac=cfg.connect.endpoint_final_lr_frac,
                    grad_clip=cfg.connect.endpoint_grad_clip)
                print(f"  endpoint {opt_type}/seed{seed}: stopped after {steps_run} steps"
                      + (" (hit cap)" if steps_run >= n_steps else " (converged)"))
                vec, _ = flatten_state_dict(model.state_dict())
                torch.save(vec, ckpt)
                del model
            endpoints[(opt_type, seed)] = vec

            load_vector_into_model(eval_model, vec, specs)
            final_loss = _mean_eval_loss(eval_model, measure_batches)
            print(f"  endpoint {opt_type}/seed{seed}: train-landscape loss = {final_loss:.4f}")
            if wandb_logger is not None:
                wandb_logger.log({f"endpoint/{opt_type}_s{seed}/final_loss": final_loss})
    return inits, endpoints


def run_experiment(cfg: DictConfig):
    device = get_device(cfg.training.device)
    autocast_ctx = get_autocast_context(device)
    seq_len = cfg.model.max_seq_len

    print(f"Preparing shared dataset '{cfg.dataset.name}' (cache_dir={cfg.dataset.cache_dir})...")
    train_data, val_data, _, _ = prepare_data(
        dataset_name=cfg.dataset.name,
        num_train_docs=cfg.dataset.num_train_docs,
        num_val_docs=cfg.dataset.num_val_docs,
        vocab_size=cfg.model.vocab_size,
        cache_dir=cfg.dataset.cache_dir,
    )

    gpt_config = build_gpt_config(cfg)
    eval_model = GPT(gpt_config).to(device)
    _, specs = flatten_state_dict(eval_model.state_dict())

    # fixed batch sets for measuring barriers; relaxation streams fresh
    # minibatches instead, so the path is never optimized on these
    measure_batches = make_fixed_eval_batches(
        train_data, cfg.connect.measure_batch_size, seq_len,
        cfg.connect.measure_batches, device, seed=cfg.connect.eval_seed,
    )
    val_batches = make_fixed_eval_batches(
        val_data, cfg.connect.measure_batch_size, seq_len,
        cfg.connect.val_batches, device, seed=cfg.connect.eval_seed + 1,
    )

    autoneb_cfg = AutoNEBConfig(**OmegaConf.to_container(cfg.connect.autoneb, resolve=True))
    wd = cfg.connect.weight_decay

    path_logger = CSVLogger(to_absolute_path(cfg.logging.path_csv))
    summary_logger = CSVLogger(to_absolute_path(cfg.logging.summary_csv))
    cycle_logger = CSVLogger(to_absolute_path(cfg.logging.cycles_csv))
    wandb_logger = WandbLogger(
        use_wandb=cfg.logging.use_wandb,
        project=cfg.logging.wandb_project,
        entity=cfg.logging.wandb_entity,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    print("=== Stage 1: endpoints ===")
    inits, endpoints = get_endpoints(cfg, gpt_config, train_data, device,
                                     autocast_ctx, specs, eval_model,
                                     measure_batches, wandb_logger=wandb_logger)

    energy_train = make_nn_energy(eval_model, specs, measure_batches, device,
                                  nullcontext(), with_grad=False, weight_decay=wd)
    energy_val = make_nn_energy(eval_model, specs, val_batches, device,
                                nullcontext(), with_grad=False, weight_decay=wd)

    pairs = make_pairs(list(cfg.connect.optimizers), list(cfg.connect.seeds),
                       cfg.connect.within_pairs, cfg.connect.cross_pairs)
    print(f"\n=== Stage 2: connecting {len(pairs)} pairs ===")

    results = []
    for pair_idx, pair in enumerate(pairs):
        kind, opt_a, seed_a = pair["kind"], pair["opt_a"], pair["seed_a"]
        opt_b, seed_b = pair["opt_b"], pair["seed_b"]
        pair_id = f"{opt_a}-s{seed_a}__{opt_b}-s{seed_b}"
        print(f"\n--- [{pair_idx + 1}/{len(pairs)}] {kind}: {pair_id} ---")

        theta_a = endpoints[(opt_a, seed_a)]
        theta_b = endpoints[(opt_b, seed_b)]

        relax_loader = make_dataloader(train_data, cfg.connect.relax_batch_size,
                                       seq_len, device,
                                       seed=cfg.connect.eval_seed + 100 + pair_idx)
        energy_relax = make_nn_energy_stream(eval_model, specs, relax_loader,
                                             autocast_ctx, with_grad=True,
                                             weight_decay=wd)
        energy_relax_eval = make_nn_energy_stream(eval_model, specs, relax_loader,
                                                  autocast_ctx, with_grad=False,
                                                  weight_decay=wd)

        def cycle_cb(cycle_idx, lr, p, prof, _pair_id=pair_id, _kind=kind):
            cycle_logger.log({"pair": _pair_id, "kind": _kind, "cycle": cycle_idx,
                              "lr": lr, "n_pivots": p.n_pivots,
                              "barrier_train": prof.barrier()})
            wandb_logger.log({f"autoneb/{_pair_id}/barrier_train": prof.barrier(),
                              f"autoneb/{_pair_id}/n_pivots": p.n_pivots,
                              f"autoneb/{_pair_id}/cycle": cycle_idx})
            print(f"  cycle {cycle_idx}: lr={lr} pivots={p.n_pivots} "
                  f"barrier(train)={prof.barrier():.4f}")

        path, profile_train = autoneb_connect(theta_a, theta_b, energy_relax,
                                              energy_train, autoneb_cfg,
                                              callback=cycle_cb, progress=True,
                                              energy_loss_only=energy_relax_eval)
        profile_val = measure_path(path, energy_val, n_interp=autoneb_cfg.n_interp)

        linear_path = Path.linear_init(theta_a, theta_b, autoneb_cfg.n_pivots_interior)
        linear_train = measure_path(linear_path, energy_train, n_interp=autoneb_cfg.n_interp)
        linear_val = measure_path(linear_path, energy_val, n_interp=autoneb_cfg.n_interp)

        loss_a = energy_train(theta_a)[0]
        loss_b = energy_train(theta_b)[0]
        loss_init_a = energy_train(inits[seed_a])[0]
        loss_init_b = energy_train(inits[seed_b])[0]
        dist_ab = float((theta_a - theta_b).norm())
        dist_a0 = float((theta_a - inits[seed_a]).norm())
        dist_b0 = float((theta_b - inits[seed_b]).norm())
        path_len = float(path.segment_lengths().sum())

        saddle_lambda = float("nan")
        if cfg.connect.compute_saddle_curvature and device.type != "cpu":
            print("  Estimating lambda_max at the saddle...")
            saddle_lambda = _saddle_curvature(eval_model, specs, path, energy_train,
                                              measure_batches, cfg.connect.power_iters)

        for curve, landscape, prof in (("connected", "train", profile_train),
                                       ("connected", "val", profile_val),
                                       ("linear", "train", linear_train),
                                       ("linear", "val", linear_val)):
            for pos, loss in zip(prof.positions.tolist(), prof.losses.tolist()):
                path_logger.log({"pair": pair_id, "kind": kind, "curve": curve,
                                 "landscape": landscape, "position": pos, "loss": loss})
            wandb_logger.log_curve(
                f"profiles/{pair_id}/{curve}_{landscape}",
                prof.positions.tolist(), prof.losses.tolist(),
                x_label="position", y_label="loss",
                title=f"{pair_id} {curve} ({landscape})")

        row = {
            "pair": pair_id, "kind": kind,
            "optimizer_A": opt_a, "seed_A": seed_a,
            "optimizer_B": opt_b, "seed_B": seed_b,
            "barrier_train_connected": profile_train.barrier(),
            "barrier_train_linear": linear_train.barrier(),
            "barrier_val_connected": profile_val.barrier(),
            "barrier_val_linear": linear_val.barrier(),
            "saddle_lambda_max": saddle_lambda, "n_pivots_final": path.n_pivots,
            "loss_A": loss_a, "loss_B": loss_b,
            "loss_init_A": loss_init_a, "loss_init_B": loss_init_b,
            "dist_AB": dist_ab, "dist_A_init": dist_a0, "dist_B_init": dist_b0,
            "path_length": path_len,
        }
        summary_logger.log(row)
        wandb_logger.log({f"pairs/{pair_id}/{k}": v for k, v in row.items()
                          if isinstance(v, (int, float))})
        results.append(row)
        print(f"  barrier(train): connected={row['barrier_train_connected']:.4f}  "
              f"linear={row['barrier_train_linear']:.4f}  |  "
              f"barrier(val): connected={row['barrier_val_connected']:.4f}  "
              f"linear={row['barrier_val_linear']:.4f}")
        print(f"  loss: A={loss_a:.4f} B={loss_b:.4f} (init {loss_init_a:.4f}/{loss_init_b:.4f})  "
              f"dist A<->B={dist_ab:.3f}  path_len={path_len:.3f}  "
              f"pivots={path.n_pivots}")

    def _mean_std(xs):
        m = sum(xs) / len(xs)
        v = sum((x - m) ** 2 for x in xs) / len(xs)
        return m, math.sqrt(v)

    print("\n=== Aggregate ===")
    for kind in ("within", "cross"):
        rows = [r for r in results if r["kind"] == kind]
        if not rows:
            continue
        groups = sorted({(r["optimizer_A"], r["optimizer_B"]) for r in rows})
        for oa, ob in groups:
            sub = [r for r in rows if (r["optimizer_A"], r["optimizer_B"]) == (oa, ob)]
            mc, sc = _mean_std([r["barrier_train_connected"] for r in sub])
            ml, sl = _mean_std([r["barrier_train_linear"] for r in sub])
            label = oa if oa == ob else f"{oa}-{ob}"
            wandb_logger.log({
                f"aggregate/{kind}/{label}/barrier_connected_mean": mc,
                f"aggregate/{kind}/{label}/barrier_connected_std": sc,
                f"aggregate/{kind}/{label}/barrier_linear_mean": ml,
                f"aggregate/{kind}/{label}/barrier_linear_std": sl,
            })
            print(f"  [{kind}] {label} (n={len(sub)}): "
                  f"barrier(connected) = {mc:.4f} +/- {sc:.4f}  "
                  f"barrier(linear) = {ml:.4f} +/- {sl:.4f}")
    print(f"Profiles -> '{cfg.logging.path_csv}', summary -> '{cfg.logging.summary_csv}', "
          f"cycles -> '{cfg.logging.cycles_csv}'")
    return results


@hydra.main(version_base=None, config_path="config", config_name="connect")
def main(cfg: DictConfig):
    run_experiment(cfg)


if __name__ == "__main__":
    main()
