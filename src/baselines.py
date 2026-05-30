"""
Baselines for comparison.

  * FinetuneBaseline -- plain sequential training (catastrophic-forgetting lower
    bound). Implemented simply by running ThreeLevelNet with gate_type="none"
    and use_adaptation=False, i.e. a flat concept network.

  * EWCBaseline -- Elastic Weight Consolidation (Kirkpatrick et al., 2017) on the
    same flat concept network, so the comparison isolates the contribution of the
    three-level adaptation/control rather than backbone differences.

These let your results table read:
    Finetune  <  EWC  <  (L1+L2)  <  (L1+L2+L3)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .model import ThreeLevelNet
from .train import task_incremental_accuracy


def make_flat_baseline(num_classes, num_slots=8, concept_dim=64):
    """A flat Level-1-only concept network (no adaptation, no control)."""
    return ThreeLevelNet(
        num_classes=num_classes, num_slots=num_slots, concept_dim=concept_dim,
        gate_type="none", use_adaptation=False,
    )


class EWC:
    """Elastic Weight Consolidation penalty over a flat model's parameters."""

    def __init__(self, model, lam=400.0):
        self.model = model
        self.lam = lam
        self.fisher = {}
        self.opt_params = {}

    def _named_trainable(self):
        return {n: p for n, p in self.model.named_parameters() if p.requires_grad}

    def consolidate(self, loader, device, n_batches=50):
        fisher = {n: torch.zeros_like(p) for n, p in self._named_trainable().items()}
        self.model.eval()
        count = 0
        for x, y in loader:
            if count >= n_batches:
                break
            x, y = x.to(device), y.to(device)
            self.model.zero_grad()
            refined, _, _ = self.model(x)
            logp = F.log_softmax(refined, dim=-1)
            samp = logp.argmax(dim=-1)
            loss = F.nll_loss(logp, samp)
            loss.backward()
            for n, p in self._named_trainable().items():
                if p.grad is not None:
                    fisher[n] += p.grad.detach() ** 2
            count += 1
        for n in fisher:
            fisher[n] /= max(count, 1)
        self.fisher = fisher
        self.opt_params = {n: p.detach().clone() for n, p in self._named_trainable().items()}

    def penalty(self):
        if not self.fisher:
            return torch.tensor(0.0)
        loss = 0.0
        for n, p in self._named_trainable().items():
            loss = loss + (self.fisher[n] * (p - self.opt_params[n]) ** 2).sum()
        return self.lam * loss


def run_ewc(model, train_loaders, test_loaders, class_groups, device,
            epochs_per_task=1, lr=1e-3, lam=400.0, verbose=True):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ewc = EWC(model, lam=lam)
    T = len(train_loaders)
    R = np.zeros((T, T))
    for i in range(T):
        model.train()
        for _ in range(epochs_per_task):
            for x, y in train_loaders[i]:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                refined, _, _ = model(x)
                loss = F.cross_entropy(refined, y) + ewc.penalty()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
        ewc.consolidate(train_loaders[i], device)
        for j in range(i + 1):
            R[i, j] = task_incremental_accuracy(model, test_loaders[j], class_groups[j], device)
        if verbose:
            print(f"[EWC task {i}] mean_acc_so_far={R[i, :i+1].mean():.3f}")
    return R, {}  # no gate in EWC; {} keeps API consistent with run_continual
