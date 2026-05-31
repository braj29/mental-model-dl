"""
Level 3 — Control of Adaptation (Second-order / self-modeling)
==============================================================

Level 3 regulates *when* and *how much* Level 2 is allowed to change Level 1.
This is the self-modeling level: the system maintains a representation of its
own learning state (uncertainty, novelty, conflict) and uses it to gate
adaptation -- directly mirroring the plasticity-vs-stability control in
Bhalwankar & Treur (2021).

Both gates now operate per-slot: each concept slot gets its own adaptation gate
based on how uncertain that slot's predictions are. Slots representing familiar
concepts stay closed (protected); slots encoding novel inputs open (adapt).

Two versions are provided:

  * HandcraftedGate   -- per-slot uncertainty-threshold rule (Paper 1).
  * LearnedGate       -- per-slot self-model with GRU self-state (Paper 2).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def predictive_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Normalised predictive entropy in [0, 1] -- a proxy for epistemic state.
    logits: (..., C) — works for any leading batch dimensions.
    """
    p = F.softmax(logits, dim=-1)
    ent = -(p * (p + 1e-9).log()).sum(dim=-1)
    return ent / torch.log(torch.tensor(logits.size(-1), dtype=logits.dtype, device=logits.device))


class HandcraftedGate(nn.Module):
    """Level 3 (Paper 1): per-slot uncertainty-threshold gate.

    For each slot, compute mean predictive entropy across the batch.
    gate_s = 1 if slot_entropy_s > threshold else 0.
    Returns (num_slots,) gate vector.
    """

    def __init__(self, threshold: float = 0.2):
        super().__init__()
        self.threshold = threshold

    def forward(self, per_slot_logits: torch.Tensor):
        """per_slot_logits: (B, S, C)"""
        B, S, C = per_slot_logits.shape
        slot_ent = predictive_entropy(per_slot_logits.reshape(B * S, C)).reshape(B, S)
        mean_slot_ent = slot_ent.mean(dim=0)          # (S,)
        gates = (mean_slot_ent > self.threshold).float()
        return gates, {"entropy": mean_slot_ent.mean().detach()}


class LearnedGate(nn.Module):
    """Level 3 (Paper 2): per-slot learned self-model gate.

    For each slot, computes uncertainty features (entropy, confidence, margin)
    and uses a shared network + GRU self-state to output a per-slot gate in [0.1, 1].
    The floor of 0.1 ensures the gate never fully closes (preserves plasticity).

    GRU state is shared across slots (captures global task-novelty context)
    but gate predictions are slot-specific (enables selective concept protection).
    """

    def __init__(self, state_dim: int = 16):
        super().__init__()
        self.state_dim = state_dim
        self.register_buffer("running_state", torch.zeros(state_dim))
        in_dim = 3 + state_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )
        self.update = nn.GRUCell(3, state_dim)
        nn.init.normal_(self.net[-1].weight, std=0.01)
        nn.init.zeros_(self.net[-1].bias)

    def _per_slot_features(self, per_slot_logits: torch.Tensor) -> torch.Tensor:
        """(B, S, C) → (S, 3) per-slot uncertainty features averaged over batch."""
        B, S, C = per_slot_logits.shape
        flat = per_slot_logits.reshape(B * S, C)
        p = F.softmax(flat, dim=-1)
        ent = predictive_entropy(flat).reshape(B, S).mean(0)      # (S,)
        conf = p.max(dim=-1).values.reshape(B, S).mean(0)         # (S,)
        top2 = p.topk(2, dim=-1).values.reshape(B, S, 2)
        margin = (top2[:, :, 0] - top2[:, :, 1]).mean(0)          # (S,)
        return torch.stack([ent, conf, margin], dim=-1)             # (S, 3)

    def forward(self, per_slot_logits: torch.Tensor):
        """per_slot_logits: (B, S, C)"""
        feats = self._per_slot_features(per_slot_logits)            # (S, 3)
        batch_feats = feats.mean(dim=0)                             # (3,) for GRU
        new_state = self.update(
            batch_feats.unsqueeze(0), self.running_state.unsqueeze(0)
        ).squeeze(0)
        state_exp = new_state.unsqueeze(0).expand(feats.size(0), -1)  # (S, state_dim)
        gate_in = torch.cat([feats, state_exp], dim=-1)              # (S, 3+state_dim)
        gate_per_slot = torch.sigmoid(self.net(gate_in)).squeeze(-1) # (S,)
        gate = 0.1 + 0.9 * gate_per_slot                            # floor at 0.1
        self.running_state = new_state.detach()
        return gate, {"entropy": batch_feats[0].detach(), "gate": gate_per_slot.detach()}
