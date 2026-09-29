"""Cross-attention head over ESM-2 with its last layers unfrozen.

Everything else in this repository keeps the encoder frozen, so its conclusion
is about frozen ESM-2 features rather than about the problem. This is the
experiment that tests the encoder itself: if the binding information is present
in ESM-2 but not in a linearly readable form, letting the top layers move should
surface it.
"""
from __future__ import annotations

import math

import torch
from torch import nn

MODEL = "facebook/esm2_t33_650M_UR50D"
WINDOW = 1022
D_IN = 1280


class FineTuneScorer(nn.Module):
    """ESM-2 with the top `unfrozen` layers trainable, plus the attention head.

    Only the top layers move. The lower ones hold the general protein statistics
    learned from 65 million sequences, which 3,788 binding pairs cannot improve
    on and would readily destroy; the top layers are where task-specific
    structure is usually found.
    """

    def __init__(self, unfrozen: int = 2, width: int = 256, dropout: float = 0.2,
                 use_target: bool = True):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL)
        self.esm = AutoModel.from_pretrained(MODEL)
        self.use_target = use_target

        for parameter in self.esm.parameters():
            parameter.requires_grad = False
        layers = self.esm.encoder.layer
        for layer in layers[len(layers) - unfrozen :]:
            for parameter in layer.parameters():
                parameter.requires_grad = True
        # Autograd builds no graph for the frozen prefix, because its parameters
        # and its inputs both stop gradients, so activations are only stored from
        # the first unfrozen layer onward.

        self.query = nn.Linear(D_IN, width)
        self.key = nn.Linear(D_IN, width)
        self.value = nn.Linear(D_IN, width)
        self.peptide = nn.Linear(D_IN, width)
        self.head = nn.Sequential(
            nn.LayerNorm(width), nn.Dropout(dropout),
            nn.Linear(width, width // 2), nn.GELU(), nn.Linear(width // 2, 1),
        )
        self.width = width

    def embed(self, sequences: list[str], device: str):
        """Residue representations for a batch, windowed for anything over the context."""
        if len(sequences) == 1 and len(sequences[0]) > WINDOW:
            sequence = sequences[0]
            pieces = [sequence[i : i + WINDOW] for i in range(0, len(sequence), WINDOW)]
            parts = [self.embed([piece], device)[0] for piece in pieces]
            joined = torch.cat([part[0] for part in parts], dim=0).unsqueeze(0)
            return joined, torch.ones(1, joined.shape[1], dtype=torch.bool, device=device)
        tokens = self.tokenizer(sequences, return_tensors="pt", padding=True).to(device)
        states = self.esm(**tokens).last_hidden_state
        # Drop the start token and keep one row per residue.
        mask = tokens["attention_mask"].bool()[:, 1:-1] if states.shape[1] > 2 else tokens["attention_mask"].bool()
        return states[:, 1 : 1 + mask.shape[1]], mask

    def forward(self, peptides: list[str], target: str, device: str):
        peptide_states, peptide_mask = self.embed(peptides, device)
        context = self.peptide(peptide_states)
        if self.use_target:
            target_states, target_mask = self.embed([target], device)
            scores = self.query(peptide_states) @ self.key(target_states).transpose(-1, -2)
            scores = (scores / math.sqrt(self.width)).masked_fill(
                ~target_mask.unsqueeze(1), float("-inf")
            )
            context = context + torch.softmax(scores, dim=-1) @ self.value(target_states)
        mask = peptide_mask.unsqueeze(-1).float()
        pooled = (context * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.head(pooled).squeeze(-1)

    def trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
