"""
Level 2 — Adaptation (First-order mental model change)
======================================================

Level 2 is the process that *changes* the Level 1 mental model in response to
experience. Instead of changing Level 1 weights via ordinary backprop alone,
a hypernetwork generates a task/context-conditioned set of weights for the
Level 1 classifier head.

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
    """Generates (delta) weights for the Level 1 classifier head.

    Given a context vector summarising the mental model's current state, it
    outputs a weight matrix and bias for the head. We generate a *delta* that
    is added to the resident head weights, which empirically trains far more
    stably than generating weights from scratch.
    """

    def __init__(
        self,
        context_dim: int,
        num_classes: int,
        concept_dim: int,
        hidden: int = 128,
        delta_scale: float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.concept_dim = concept_dim
        self.delta_scale = delta_scale

        out_size = num_classes * concept_dim + num_classes   # weight + bias
        self.net = nn.Sequential(
            nn.Linear(context_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_size),
        )
        # small non-zero init so the gate gets a gradient signal through delta
        nn.init.normal_(self.net[-1].weight, std=0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, context: torch.Tensor):
        """context: (context_dim,) or (B, context_dim) -> per-batch we average.

        Returns delta_w: (num_classes, concept_dim), delta_b: (num_classes,)
        """
        if context.dim() == 2:
            context = context.mean(dim=0)     # one shared update per batch
        out = self.net(context) * self.delta_scale
        w_size = self.num_classes * self.concept_dim
        delta_w = out[:w_size].view(self.num_classes, self.concept_dim)
        delta_b = out[w_size:]
        return delta_w, delta_b


def build_context(concept_pool: torch.Tensor, prediction_error: torch.Tensor) -> torch.Tensor:
    """Assemble the Level 2 conditioning signal from the mental model's state.

    concept_pool:     (B, concept_dim)   what the model currently represents
    prediction_error: (B, num_classes)   how wrong the current prediction is
    Returns (B, concept_dim + num_classes)
    """
    return torch.cat([concept_pool, prediction_error], dim=-1)
