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

    One backbone with one head:
      - head_net: per-slot head deltas (num_slots × num_classes × concept_dim) and
        a shared bias delta (num_classes,).

    Generating a separate delta_w per slot lets Level 3's per-slot gate control
    exactly how much each concept slot contributes to the head update — without
    touching the slot representations themselves (which would cause forgetting).
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
        # per-slot head weight delta + shared bias delta
        per_slot_w = num_slots * num_classes * concept_dim
        self.head_net = nn.Linear(hidden, per_slot_w + num_classes)

        nn.init.normal_(self.head_net.weight, std=0.01)
        nn.init.zeros_(self.head_net.bias)

    def forward(self, context: torch.Tensor):
        """context: (context_dim,) or (B, context_dim) -> per-batch we average.

        Returns:
            delta_w_per_slot: (num_slots, num_classes, concept_dim)
            delta_b:          (num_classes,)
        """
        if context.dim() == 2:
            context = context.mean(dim=0)
        h = self.backbone(context)
        flat = self.head_net(h) * self.delta_scale
        w_size = self.num_slots * self.num_classes * self.concept_dim
        delta_w_per_slot = flat[:w_size].view(self.num_slots, self.num_classes, self.concept_dim)
        delta_b = flat[w_size:]
        return delta_w_per_slot, delta_b


def build_context(concept_pool: torch.Tensor, prediction_error: torch.Tensor) -> torch.Tensor:
    """Assemble the Level 2 conditioning signal from the mental model's state.

    concept_pool:     (B, concept_dim)   what the model currently represents
    prediction_error: (B, num_classes)   how wrong the current prediction is
    Returns (B, concept_dim + num_classes)
    """
    return torch.cat([concept_pool, prediction_error], dim=-1)
