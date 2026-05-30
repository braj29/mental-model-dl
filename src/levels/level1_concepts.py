"""
Level 1 — Mental Model Use (Concept Representations)
=====================================================

This level holds the *current mental model*: a structured set of concept
representations that are applied to the present input to produce a prediction.

Following mental-model theory (Bhalwankar & Treur, 2021), Level 1 is the model
that is *used*. It does not decide when to change itself -- that is the job of
Level 2 (adaptation) and Level 3 (control of adaptation).

We implement the concept layer with Slot Attention (Locatello et al., 2020),
which learns a set of object/concept slots in an unsupervised, permutation-
invariant way. A simpler Concept-Bottleneck-Model (CBM) style linear bottleneck
is also provided for the very first experiment.

Swap-in note: lucidrains/slot-attention provides a drop-in SlotAttention; this
local implementation is kept so the mechanics are transparent and dependency-free.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvEncoder(nn.Module):
    """Shared backbone that maps an image to a set of spatial feature vectors.

    Output shape: (batch, num_locations, feature_dim), suitable for slot attention.
    """

    def __init__(self, in_channels: int = 3, feature_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, feature_dim, 5, stride=1, padding=2),
            nn.ReLU(inplace=True),
        )
        self.feature_dim = feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        h = self.net(x)                      # (B, D, H', W')
        b, d, hh, ww = h.shape
        h = h.permute(0, 2, 3, 1).reshape(b, hh * ww, d)   # (B, N, D)
        return h


class SlotAttention(nn.Module):
    """Slot Attention (Locatello et al., 2020).

    Learns `num_slots` concept vectors that compete to explain the input
    features through iterative attention. Each slot is one concept in the
    mental model.
    """

    def __init__(self, num_slots: int, dim: int, iters: int = 3, hidden_dim: int = 128, eps: float = 1e-8):
        super().__init__()
        self.num_slots = num_slots
        self.dim = dim
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.xavier_uniform_(self.slots_logsigma)

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        self.gru = nn.GRUCell(dim, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim),
        )
        self.norm_input = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (B, N, D)
        b, n, d = inputs.shape
        mu = self.slots_mu.expand(b, self.num_slots, -1)
        sigma = self.slots_logsigma.exp().expand(b, self.num_slots, -1)
        slots = mu + sigma * torch.randn_like(mu)

        inputs = self.norm_input(inputs)
        k = self.to_k(inputs)
        v = self.to_v(inputs)

        for _ in range(self.iters):
            slots_prev = slots
            slots_n = self.norm_slots(slots)
            q = self.to_q(slots_n)

            attn_logits = torch.einsum("bid,bjd->bij", q, k) * self.scale
            attn = attn_logits.softmax(dim=1) + self.eps           # compete over slots
            attn = attn / attn.sum(dim=-1, keepdim=True)           # normalise over inputs
            updates = torch.einsum("bij,bjd->bid", attn, v)

            slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d),
            ).reshape(b, self.num_slots, d)
            slots = slots + self.mlp(self.norm_pre_ff(slots))

        return slots   # (B, num_slots, D)


class ConceptLayer(nn.Module):
    """Level 1 module: encoder -> slot concepts -> task prediction.

    The classifier head weights can be *overwritten* at adaptation time by the
    Level 2 hypernetwork. We therefore expose the head as a functional call so
    externally generated weights can be injected.
    """

    def __init__(self, num_classes: int, num_slots: int = 8, dim: int = 64, in_channels: int = 3):
        super().__init__()
        self.encoder = ConvEncoder(in_channels=in_channels, feature_dim=dim)
        self.slot_attn = SlotAttention(num_slots=num_slots, dim=dim)
        self.dim = dim
        self.num_slots = num_slots
        self.num_classes = num_classes

        # Default (resident) head. Adaptation may replace these weights.
        self.head_w = nn.Parameter(torch.randn(num_classes, dim) * (dim ** -0.5))
        self.head_b = nn.Parameter(torch.zeros(num_classes))

    def concepts(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)              # (B, N, D)
        slots = self.slot_attn(feats)        # (B, S, D)
        return slots

    def predict_from_concepts(
        self,
        slots: torch.Tensor,
        head_w: torch.Tensor | None = None,
        head_b: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pooled = slots.mean(dim=1)           # (B, D)  aggregate concepts
        w = self.head_w if head_w is None else head_w
        b = self.head_b if head_b is None else head_b
        return F.linear(pooled, w, b)        # (B, num_classes)

    def forward(self, x: torch.Tensor, head_w=None, head_b=None):
        slots = self.concepts(x)
        logits = self.predict_from_concepts(slots, head_w, head_b)
        return logits, slots
