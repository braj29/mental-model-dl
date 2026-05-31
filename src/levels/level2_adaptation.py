"""
Level 2 — Adaptation (First-order mental model change)
======================================================

Level 2 is the process that *changes* the Level 1 mental model in response to
experience. Instead of changing Level 1 weights via ordinary backprop alone,
a hypernetwork generates a task/context-conditioned set of weights for the
Level 1 classifier head AND per-slot concept adjustments.

The crucial design choice that distinguishes this from von Oswald et al. (2020):
the hypernetwork is conditioned NOT on a task ID, but on the *current state of
the mental model* -- the pooled concept activations and the current prediction
error. This makes adaptation responsive to what the model currently believes
and how wrong it is, which is exactly the first-order adaptation described in
Bhalwankar & Treur (2021).

Swap-in note: chrhenning/hypnettorch offers production hypernetworks; this local
version keeps the generated-weight mechanics explicit for the thesis.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AdaptationHyperNet(nn.Module):
    """Generates (delta) weights for the Level 1 classifier head and concept slots.

    Two output heads sharing a backbone:
      - head_net: delta_w (num_classes × concept_dim) and delta_b (num_classes,)
        for the classifier head — controls how predictions are made.
      - slot_net: delta_slots (num_slots × concept_dim)
        for the slot representations — controls what concepts are formed.

    Per-slot deltas let Level 3 gate adaptation selectively per concept,
    rather than all-or-nothing on the entire head.
    """

    def __init__(
        self,
        context_dim: int,
        num_classes: int,
        concept_dim: int,
        num_slots: int,
        hidden: int = 128,
        delta_scale: float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.concept_dim = concept_dim
        self.num_slots = num_slots
        self.delta_scale = delta_scale

        self.backbone = nn.Sequential(
            nn.Linear(context_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
        )
        head_out = num_classes * concept_dim + num_classes
        self.head_net = nn.Linear(hidden, head_out)
        self.slot_net = nn.Linear(hidden, num_slots * concept_dim)

        # small non-zero init — gives gate a gradient signal while keeping deltas small
        nn.init.normal_(self.head_net.weight, std=0.01)
        nn.init.zeros_(self.head_net.bias)
        nn.init.normal_(self.slot_net.weight, std=0.01)
        nn.init.zeros_(self.slot_net.bias)

    def forward(self, context: torch.Tensor):
        """context: (context_dim,) or (B, context_dim) -> per-batch we average.

        Returns:
            delta_w:     (num_classes, concept_dim)
            delta_b:     (num_classes,)
            delta_slots: (num_slots, concept_dim)
        """
        if context.dim() == 2:
            context = context.mean(dim=0)
        h = self.backbone(context)
        flat = self.head_net(h) * self.delta_scale
        w_size = self.num_classes * self.concept_dim
        delta_w = flat[:w_size].view(self.num_classes, self.concept_dim)
        delta_b = flat[w_size:]
        delta_slots = self.slot_net(h).view(self.num_slots, self.concept_dim) * self.delta_scale
        return delta_w, delta_b, delta_slots


def build_context(concept_pool: torch.Tensor, prediction_error: torch.Tensor) -> torch.Tensor:
    """Assemble the Level 2 conditioning signal from the mental model's state.

    concept_pool:     (B, concept_dim)   what the model currently represents
    prediction_error: (B, num_classes)   how wrong the current prediction is
    Returns (B, concept_dim + num_classes)
    """
    return torch.cat([concept_pool, prediction_error], dim=-1)
