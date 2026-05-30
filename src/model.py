"""
The Three-Level Mental Model Network
====================================

Combines:
    Level 1 (ConceptLayer)        -- the mental model that is used
    Level 2 (AdaptationHyperNet)  -- generates weight deltas to adapt Level 1
    Level 3 (Handcrafted/Learned) -- gates how much Level 2 is applied

Forward pass:
    1. Level 1 forms concepts and makes a *base* prediction.
    2. Level 3 inspects the prediction to decide an adaptation gate.
    3. Level 2 generates a head-weight delta from the mental-model state.
    4. The gated delta is applied; a refined prediction is produced.

Both predictions are returned so the training loop can optionally supervise the
base prediction, the refined one, or both.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .levels.level1_concepts import ConceptLayer
from .levels.level2_adaptation import AdaptationHyperNet, build_context
from .levels.level3_control import HandcraftedGate, LearnedGate


class ThreeLevelNet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_slots: int = 8,
        concept_dim: int = 64,
        in_channels: int = 3,
        gate_type: str = "handcrafted",   # "handcrafted" | "learned" | "none"
        gate_threshold: float = 0.4,
        use_adaptation: bool = True,
    ):
        super().__init__()
        self.use_adaptation = use_adaptation
        self.num_classes = num_classes

        # Level 1
        self.level1 = ConceptLayer(
            num_classes=num_classes, num_slots=num_slots,
            dim=concept_dim, in_channels=in_channels,
        )

        # Level 2
        context_dim = concept_dim + num_classes
        self.level2 = AdaptationHyperNet(
            context_dim=context_dim, num_classes=num_classes, concept_dim=concept_dim,
        )

        # Level 3
        if gate_type == "handcrafted":
            self.level3 = HandcraftedGate(threshold=gate_threshold)
        elif gate_type == "learned":
            self.level3 = LearnedGate()
        elif gate_type == "none":
            self.level3 = None
        else:
            raise ValueError(f"unknown gate_type {gate_type}")

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None):
        # ---- Level 1: form concepts + base prediction ----
        slots = self.level1.concepts(x)                       # (B, S, D)
        base_logits = self.level1.predict_from_concepts(slots)
        concept_pool = slots.mean(dim=1)                      # (B, D)

        info = {"base_logits": base_logits}

        if not self.use_adaptation or self.level3 is None:
            return base_logits, base_logits, info

        # ---- Level 3: decide adaptation gate from self-state ----
        gate, gate_info = self.level3(base_logits)
        info.update(gate_info)

        # ---- Level 2: generate weight delta from mental-model state ----
        # prediction error proxy: 1 - softmax (or true error if y provided)
        with torch.no_grad():
            probs = F.softmax(base_logits, dim=-1)
        if y is not None:
            onehot = F.one_hot(y, self.num_classes).float()
            pred_error = onehot - probs                       # signed error
        else:
            pred_error = -probs                               # uncertainty signal
        context = build_context(concept_pool, pred_error)     # (B, ctx_dim)
        delta_w, delta_b = self.level2(context)

        # ---- apply gated delta to Level 1 head ----
        new_w = self.level1.head_w + gate * delta_w
        new_b = self.level1.head_b + gate * delta_b
        refined_logits = self.level1.predict_from_concepts(slots, new_w, new_b)

        info["gate_value"] = gate  # kept with grad for sparsity penalty; detach in logging if needed
        return refined_logits, base_logits, info
