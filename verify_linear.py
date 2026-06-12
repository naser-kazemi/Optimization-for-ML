"""Re-measure the straight segment between two cached endpoints at high
resolution, plus a control walk of the same length in a random direction.

    python verify_linear.py --a endpoints/<A>.pt --b endpoints/<B>.pt \
        --points 201 --batches 32 --device cuda
"""

import argparse
from contextlib import nullcontext

import torch
from omegaconf import OmegaConf

from data import make_fixed_eval_batches, prepare_data
from models import GPT
from optim.neb import Path, measure_path
from optim.param_vector import flatten_state_dict, make_nn_energy
from _train import build_gpt_config, get_device


def profile_segment(x0, x1, energy, points):
    path = Path.linear_init(x0, x1, points - 2)
    return measure_path(path, energy, n_interp=0)


def describe(name, prof):
    losses = prof.losses
    i = int(torch.argmax(losses))
    print(f"[{name}]")
    print(f"  endpoints: {float(losses[0]):.4f} / {float(losses[-1]):.4f}")
    print(f"  max: {float(losses.max()):.4f} at position {float(prof.positions[i]):.3f}"
          f"  |  min: {float(losses.min()):.4f}")
    print(f"  barrier (max - worse endpoint): {prof.barrier():.6f}")
    qs = [0.1, 0.25, 0.5, 0.75, 0.9]
    samples = [float(losses[int(q * (losses.numel() - 1))]) for q in qs]
    print("  loss at positions", qs, ":", [f"{s:.4f}" for s in samples])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--points", type=int, default=201)
    ap.add_argument("--batches", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--config", default="config/connect.yaml")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    device = get_device(args.device)
    seq_len = cfg.model.max_seq_len

    train_data, _, _, _ = prepare_data(
        dataset_name=cfg.dataset.name,
        num_train_docs=cfg.dataset.num_train_docs,
        num_val_docs=cfg.dataset.num_val_docs,
        vocab_size=cfg.model.vocab_size,
        cache_dir=cfg.dataset.cache_dir,
    )
    batches = make_fixed_eval_batches(train_data, args.batch_size, seq_len,
                                      args.batches, device, seed=cfg.connect.eval_seed)

    model = GPT(build_gpt_config(cfg)).to(device)
    _, specs = flatten_state_dict(model.state_dict())
    energy = make_nn_energy(model, specs, batches, device, nullcontext(), with_grad=False)

    theta_a = torch.load(args.a, map_location="cpu")
    theta_b = torch.load(args.b, map_location="cpu")
    dist = float((theta_a - theta_b).norm())
    tokens = args.batches * args.batch_size * seq_len
    print(f"|A-B| = {dist:.3f}   resolution: {args.points} points "
          f"({dist / (args.points - 1):.2f} apart)   landscape: {tokens} tokens\n")

    describe("A -> B (linear)", profile_segment(theta_a, theta_b, energy, args.points))
    print()

    g = torch.Generator().manual_seed(args.seed)
    u = torch.randn(theta_a.numel(), generator=g)
    theta_rand = theta_a + dist * u / u.norm()
    describe("A -> A + |A-B| * random_dir (control)",
             profile_segment(theta_a, theta_rand, energy, max(21, args.points // 4)))


if __name__ == "__main__":
    main()
