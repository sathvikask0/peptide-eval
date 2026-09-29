"""Cross-attention from peptide residues onto target residues.

The pooled models could not use the target. One candidate explanation is that
pooling destroys the thing that matters: a binding site is a handful of residues,
so averaging an 888-residue protein divides its signal by a hundred. This model
removes that step and lets each peptide residue look at each target residue
directly. If the target still adds nothing, pooling was not the reason.
"""
from __future__ import annotations

import math

import torch
from torch import nn

D_IN = 1280


class CrossAttentionScorer(nn.Module):
    """One attention block, peptide residues as queries, target residues as keys.

    Deliberately small. There are 3,788 training pairs over 122 targets, so a
    deep model would fit them long before it learned anything transferable; the
    forest already showed that fitting these pairs harder does not help. The
    projection width is the one capacity knob and the head is two layers.
    """

    def __init__(self, width: int = 256, dropout: float = 0.2, use_target: bool = True):
        super().__init__()
        self.use_target = use_target
        self.query = nn.Linear(D_IN, width)
        self.key = nn.Linear(D_IN, width)
        self.value = nn.Linear(D_IN, width)
        # The peptide path exists in both variants, so the ablation that drops
        # the target changes what the model can see and not how big it is.
        self.peptide = nn.Linear(D_IN, width)
        self.head = nn.Sequential(
            nn.LayerNorm(width),
            nn.Dropout(dropout),
            nn.Linear(width, width // 2),
            nn.GELU(),
            nn.Linear(width // 2, 1),
        )
        self.width = width

    def forward(self, peptide, peptide_mask, target, target_mask):
        """peptide: (B, Lp, 1280)   target: (1, Lt, 1280), shared by the batch."""
        context = self.peptide(peptide)
        if self.use_target:
            scores = self.query(peptide) @ self.key(target).transpose(-1, -2)
            scores = scores / math.sqrt(self.width)
            scores = scores.masked_fill(~target_mask.unsqueeze(1), float("-inf"))
            context = context + torch.softmax(scores, dim=-1) @ self.value(target)
        mask = peptide_mask.unsqueeze(-1).float()
        pooled = (context * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.head(pooled).squeeze(-1)
