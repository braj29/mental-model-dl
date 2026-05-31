"""
The Three-Level Mental Model Network
====================================

Combines:
    Level 1 (ConceptLayer)        -- the mental model that is used
    Level 2 (AdaptationHyperNet)  -- generates weight deltas to adapt Level 1
    Level 3 (Handcrafted/Learned) -- gates how much Level 2 is applied, per slot

Forward pass:
    1. Level 1 forms concept slots and makes a *base* prediction.
    2. Level 3 inspects per-slot predictions to compute per-slot adaptation gates.
    3. Level 2 generates a head-weight delta and per-slot slot deltas.
    4. Each slot is adapted by its own gate × slot delta.
    5. The adapted slots are pooled; the head delta (scaled by mean gate) is applied.
    6. Both base and refined predictions are returned.
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
        self.num_slots = num_slots

        # Level 1
        self.level1 = ConceptLayer(
            num_classes=num_classes, num_slots=num_slots,
            dim=concept_dim, in_channels=in_channels,
        )

        # Level 2 — now generates both head delta and per-slot deltas
        context_dim = concept_dim + num_classes
        self.level2 = AdaptationHyperNet(
            context_dim=context_dim, num_classes=num_classes,
            concept_dim=concept_dim, num_slots=num_slots,
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
        # ---- Level 1: form concept slots + base prediction ----
        slots = self.level1.concepts(x)                            # (B, S, D)
        base_logits = self.level1.predict_from_concepts(slots)     # (B, C)
        concept_pool = slots.mean(dim=1)                           # (B, D)

        info = {"base_logits": base_logits}

        if not self.use_adaptation or self.level3 is None:
            return base_logits, base_logits, info

        # ---- Level 3: per-slot gate from each slot's prediction uncertainty ----
        per_slot_logits = F.linear(
            slots, self.level1.head_w, self.level1.head_b
        )                                                          # (B, S, C)
        gates, gate_info = self.level3(per_slot_logits)            # (S,)
        info.update(gate_info)

        # ---- Level 2: head delta + per-slot concept deltas ----
        with torch.no_grad():
            probs = F.softmax(base_logits, dim=-1)
        if y is not None:
            onehot = F.one_hot(y, self.num_classes).float()
            pred_error = onehot - probs
        else:
            pred_error = -probs
        context = build_context(concept_pool, pred_error)          # (B, ctx_dim)
        delta_w_per_slot, delta_b = self.level2(context)           # (S, C, D), (C,)

        # ---- apply per-slot gated head delta (slots are never mutated) ----
        # Effective head delta = sum over slots of gate_s * delta_w_s, normalised by S
        S = slots.size(1)
        # gates: (S,), delta_w_per_slot: (S, C, D) -> weighted sum -> (C, D)
        gated_delta_w = (gates.view(S, 1, 1) * delta_w_per_slot).mean(dim=0)
        gate_mean = gates.mean()
        new_w = self.level1.head_w + gated_delta_w
        new_b = self.level1.head_b + gate_mean * delta_b
        refined_logits = self.level1.predict_from_concepts(slots, new_w, new_b)

        info["gate_value"] = gate_mean
        return refined_logits, base_logits, info
