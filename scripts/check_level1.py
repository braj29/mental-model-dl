"""
Sanity check: can Level 1 alone reach ~60% on a single CIFAR-100 task?

If not, the whole stack is bottlenecked and no control story can show up.
Run before committing to a full multi-seed experiment.

Usage:
    uv run python scripts/check_level1.py
    uv run python scripts/check_level1.py --epochs 100 --device cuda
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F

from src.data import make_split_cifar100
from src.baselines import make_flat_baseline


def accuracy(model, loader, classes, device):
    model.eval()
    cls = torch.tensor(sorted(classes), device=device)
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, _, _ = model(x)
            masked = logits[:, cls]
            pred = cls[masked.argmax(dim=1)]
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--slots", type=int, default=8)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_loaders, test_loaders, groups = make_split_cifar100(
        n_experiences=10, batch_size=args.batch, seed=args.seed,
    )
    train_loader = train_loaders[0]
    test_loader = test_loaders[0]
    task_classes = groups[0]
    num_classes_in_task = len(task_classes)
    random_acc = 1.0 / num_classes_in_task

    model = make_flat_baseline(num_classes=100, num_slots=args.slots, concept_dim=args.dim)
    model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"Task 0: {num_classes_in_task} classes, random chance = {random_acc:.3f}")
    print(f"Training Level 1 alone for {args.epochs} epochs on {args.device} ...\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(args.device), y.to(args.device)
            optimizer.zero_grad()
            logits, _, _ = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()

        if epoch % 10 == 0 or epoch == 1:
            acc = accuracy(model, test_loader, task_classes, args.device)
            print(f"epoch {epoch:3d}  loss={total_loss / len(train_loader):.3f}  acc={acc:.3f}")

    final_acc = accuracy(model, test_loader, task_classes, args.device)
    target = 0.60
    status = "PASS" if final_acc >= target else "BELOW TARGET"
    print(f"\nFinal acc={final_acc:.3f}  target={target:.2f}  [{status}]")
    if final_acc < target:
        print("Level 1 is bottlenecked. Consider: more slots, larger dim, deeper encoder, more epochs.")


if __name__ == "__main__":
    main()
