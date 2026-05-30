"""
Run the full experiment: baselines vs. ablations of the three-level model.

Usage:
    uv run python scripts/run_experiment.py --tasks 10 --epochs 5 --device cuda

    # multi-seed (produces mean ± std)
    uv run python scripts/run_experiment.py --tasks 10 --epochs 5 --device cuda --seeds 0 1 2

Produces results/results.csv (single seed) or results/results_multiseed.csv (multi-seed).
Gate values for the learned model are saved to results/gate_log_seed<N>.npy.

For a fast sanity run on CPU:
    uv run python scripts/run_experiment.py --tasks 3 --epochs 3 --device cpu --quick
"""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np
import torch

from src.data import make_split_cifar100
from src.model import ThreeLevelNet
from src.baselines import make_flat_baseline, run_ewc
from src.train import run_continual
from src.metrics import summarize


def build_configs(num_classes, num_slots, concept_dim):
    return {
        "finetune": (
            lambda: make_flat_baseline(num_classes, num_slots, concept_dim),
            "continual",
        ),
        "ewc": (
            lambda: make_flat_baseline(num_classes, num_slots, concept_dim),
            "ewc",
        ),
        "L1+L2 (no control)": (
            lambda: ThreeLevelNet(num_classes, num_slots, concept_dim,
                                  gate_type="none", use_adaptation=True),
            "continual",
        ),
        "L1+L2+L3 handcrafted": (
            lambda: ThreeLevelNet(num_classes, num_slots, concept_dim,
                                  gate_type="handcrafted", use_adaptation=True),
            "continual",
        ),
        "L1+L2+L3 learned (ours)": (
            lambda: ThreeLevelNet(num_classes, num_slots, concept_dim,
                                  gate_type="learned", use_adaptation=True),
            "continual",
        ),
    }


def run_one_seed(args, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    num_classes = 100
    random_acc = 1.0 / (num_classes // args.tasks)

    train_loaders, test_loaders, groups = make_split_cifar100(
        n_experiences=args.tasks, batch_size=args.batch, seed=seed,
    )

    configs = build_configs(num_classes, args.slots, args.dim)
    if args.configs:
        configs = {k: v for k, v in configs.items()
                   if any(kw.lower() in k.lower() for kw in args.configs)}
    os.makedirs("results", exist_ok=True)
    seed_results = {}

    for name, (factory, runner) in configs.items():
        print(f"\n===== {name} (seed={seed}) =====")
        model = factory()
        if runner == "ewc":
            R, gate_logs = run_ewc(model, train_loaders, test_loaders, groups, args.device,
                                   epochs_per_task=args.epochs, lr=args.lr)
        else:
            R, gate_logs = run_continual(model, train_loaders, test_loaders, groups, args.device,
                                         epochs_per_task=args.epochs, lr=args.lr)

        if gate_logs:
            flat = np.array([v for task_log in gate_logs.values() for v in task_log])
            tag = name.replace(" ", "_").replace("+", "").replace("(", "").replace(")", "")
            np.save(f"results/gate_log_{tag}_seed{seed}.npy", flat)

        m = summarize(R, random_acc=random_acc)
        print(f"{name}: ACC={m['ACC']:.3f}  BWT={m['BWT']:+.3f}  FWT={m['FWT']:+.3f}")
        seed_results[name] = m
        np.save(f"results/R_{name.replace(' ', '_').replace('+', '').replace('(', '').replace(')', '')}_seed{seed}.npy", R)

    return seed_results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--slots", type=int, default=8)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0, help="single seed (ignored if --seeds is set)")
    ap.add_argument("--seeds", type=int, nargs="+", help="run multiple seeds, e.g. --seeds 0 1 2")
    ap.add_argument("--quick", action="store_true", help="tiny run for debugging")
    ap.add_argument("--configs", type=str, nargs="+",
                    help="keywords to filter configs, e.g. --configs handcrafted learned")
    args = ap.parse_args()

    seeds = args.seeds if args.seeds else [args.seed]
    print(f"Loading Split-CIFAR-100 ({args.tasks} tasks) on {args.device} ...")

    all_results: dict[str, list[dict]] = {}
    for seed in seeds:
        seed_results = run_one_seed(args, seed)
        for name, m in seed_results.items():
            all_results.setdefault(name, []).append(m)

    os.makedirs("results", exist_ok=True)

    if len(seeds) == 1:
        rows = [{"config": name, **{k: round(v, 4) for k, v in runs[0].items()}}
                for name, runs in all_results.items()]
        with open("results/results.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["config", "ACC", "BWT", "FWT"])
            w.writeheader()
            w.writerows(rows)
        print("\nWrote results/results.csv")
    else:
        print("\n--- Multi-seed summary ---")
        rows = []
        for name, runs in all_results.items():
            row = {"config": name}
            for metric in ("ACC", "BWT", "FWT"):
                vals = [r[metric] for r in runs]
                row[f"{metric}_mean"] = round(float(np.mean(vals)), 4)
                row[f"{metric}_std"] = round(float(np.std(vals)), 4)
            rows.append(row)
            print(f"{name}: ACC={row['ACC_mean']:.3f}±{row['ACC_std']:.3f}  "
                  f"BWT={row['BWT_mean']:+.3f}±{row['BWT_std']:.3f}")
        with open("results/results_multiseed.csv", "w", newline="") as f:
            fields = ["config", "ACC_mean", "ACC_std", "BWT_mean", "BWT_std", "FWT_mean", "FWT_std"]
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print("\nWrote results/results_multiseed.csv")


if __name__ == "__main__":
    main()
