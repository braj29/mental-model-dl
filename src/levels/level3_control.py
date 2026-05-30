"""
Level 3 — Control of Adaptation (Second-order / self-modeling)
==============================================================

Level 3 regulates *when* and *how much* Level 2 is allowed to change Level 1.
This is the self-modeling level: the system maintains a representation of its
own learning state (uncertainty, novelty, conflict) and uses it to gate
adaptation -- directly mirroring the plasticity-vs-stability control in
Bhalwankar & Treur (2021).

Two versions are provided:

  * HandcraftedGate   -- uncertainty-threshold rule. Use for Paper 1: simple,
                         interpretable, stable. A genuine Level-3 controller
                         whose policy is fixed rather than learned.

  * LearnedGate       -- a small self-model that reads the learning state and
                         outputs a continuous adaptation gate in [0, 1].
                         This is the full self-modeling vision (Paper 2).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def predictive_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Normalised predictive entropy in [0, 1] -- a proxy for epistemic state."""
    p = F.softmax(logits, dim=-1)
    ent = -(p * (p + 1e-9).log()).sum(dim=-1)
    return ent / torch.log(torch.tensor(logits.size(-1), dtype=logits.dtype, device=logits.device))


class HandcraftedGate(nn.Module):
    """Level 3 (Paper 1): allow adaptation only when the model is uncertain.

    gate = 1 if mean predictive entropy > threshold else 0.
    Returns a scalar gate that scales the Level 2 delta.
    """

    def __init__(self, threshold: float = 0.2):
        super().__init__()
        self.threshold = threshold

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        ent = predictive_entropy(logits).mean()
        gate = (ent > self.threshold).float()
        return gate, {"entropy": ent.detach()}


class LearnedGate(nn.Module):
    """Level 3 (Paper 2): a learned self-model producing a soft adaptation gate.

    Computes per-sample features so uncertain samples get a higher gate value
    than confident ones within the same batch — genuine selective gating.

    Input features (per sample):
        - predictive entropy                 (uncertainty)
        - max softmax probability            (confidence)
        - logit margin (top1 - top2)         (decision conflict)
        - a learned running summary vector   (recurrent self-state, shared across batch)
    Output: gate per sample in [0, 1], mean is used to scale the Level 2 delta.
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
        # small random init so gate starts near 0.5 but can move immediately
        nn.init.normal_(self.net[-1].weight, std=0.01)
        nn.init.zeros_(self.net[-1].bias)

    def _per_sample_features(self, logits: torch.Tensor) -> torch.Tensor:
        """Returns (B, 3) per-sample uncertainty features."""
        p = F.softmax(logits, dim=-1)
        ent = predictive_entropy(logits)                          # (B,)
        conf = p.max(dim=-1).values                               # (B,)
        top2 = p.topk(2, dim=-1).values
        margin = top2[:, 0] - top2[:, 1]                         # (B,)
        return torch.stack([ent, conf, margin], dim=-1)           # (B, 3)

    def forward(self, logits: torch.Tensor):
        feats = self._per_sample_features(logits)                 # (B, 3)
        batch_feats = feats.mean(dim=0)                           # (3,) for GRU update
        new_state = self.update(
            batch_feats.unsqueeze(0), self.running_state.unsqueeze(0)
        ).squeeze(0)                                              # (state_dim,)
        state_expanded = new_state.unsqueeze(0).expand(feats.size(0), -1)  # (B, state_dim)
        gate_in = torch.cat([feats, state_expanded], dim=-1)     # (B, 3+state_dim)
        gate_per_sample = torch.sigmoid(self.net(gate_in)).squeeze(-1)  # (B,)
        gate = gate_per_sample.mean()                            # scalar for model.py compatibility
        # update self-state (detached so it acts as slow-moving memory)
        self.running_state = new_state.detach()
        return gate, {"entropy": batch_feats[0].detach(), "gate": gate_per_sample.detach()}
