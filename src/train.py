"""
Continual-learning training / evaluation loop for ThreeLevelNet.

Trains sequentially on each task and, after every task, evaluates on all tasks
seen so far to build the accuracy matrix R used by metrics.summarize.

The loss supervises the refined (post-adaptation) prediction by default, with an
optional auxiliary loss on the base prediction (helps stabilise Level 1).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def task_incremental_accuracy(model, loader, classes, device):
    """Task-incremental eval: restrict logits to the task's class set."""
    model.eval()
    correct = total = 0
    cls = torch.tensor(sorted(classes), device=device)
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            refined, _, _ = model(x)
            masked = refined[:, cls]                       # (B, |classes|)
            pred_local = masked.argmax(dim=1)
            pred = cls[pred_local]
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / max(total, 1)


def train_task(model, loader, optimizer, device, epochs=1, aux_weight=0.3, gate_sparsity=0.1):
    """Train for one task. Returns (final_loss, gate_log).

    gate_log is a list of mean gate values per batch (empty if model has no gate).
    """
    model.train()
    gate_log = []
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            refined, base, info = model(x, y)
            loss = F.cross_entropy(refined, y)
            if aux_weight > 0:
                loss = loss + aux_weight * F.cross_entropy(base, y)
            gate = info.get("gate_value")
            if gate is not None and torch.is_tensor(gate):
                gate_log.append(float(gate.detach().mean()))
                # penalise a learned gate that stays open — prevents collapse to always-on
                if gate_sparsity > 0 and gate.requires_grad:
                    loss = loss + gate_sparsity * gate
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
    return float(loss.detach()), gate_log


def run_continual(model, train_loaders, test_loaders, class_groups, device,
                  epochs_per_task=1, lr=1e-3, aux_weight=0.3, verbose=True):
    """Returns (R, gate_logs) where gate_logs maps task_index -> list of per-batch gate values."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    T = len(train_loaders)
    R = np.zeros((T, T), dtype=np.float64)
    gate_logs = {}

    for i in range(T):
        last_loss, gate_log = train_task(model, train_loaders[i], optimizer, device,
                                         epochs=epochs_per_task, aux_weight=aux_weight)
        gate_logs[i] = gate_log
        for j in range(i + 1):
            R[i, j] = task_incremental_accuracy(model, test_loaders[j], class_groups[j], device)
        if verbose:
            seen = R[i, :i + 1]
            gate_summary = f"  gate={np.mean(gate_log):.3f}±{np.std(gate_log):.3f}" if gate_log else ""
            print(f"[task {i}] loss={last_loss:.3f}  mean_acc_so_far={seen.mean():.3f}{gate_summary}")
    return R, gate_logs
