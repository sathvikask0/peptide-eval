"""Residue-level ESM-2 representations, stored as one flat array plus offsets.

Pooling to a single vector averages a binding site over the whole protein. For
an 888-residue target the site is a handful of residues, so mean pooling divides
its signal by a hundred. Cross-attention needs the residues kept apart, which
means storing a ragged set of matrices rather than a rectangular one.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from peptide_eval.embed import MODEL, WINDOW, device


@torch.no_grad()
def encode(sequences: list[str], batch_size: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Return residue vectors concatenated end to end, and each sequence's start offset.

    Stored as float16: these are inputs to a small trainable head, not values we
    do arithmetic on, and halving a gigabyte matters more than the last few bits
    of a representation that is itself an approximation.
    """
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).to(device()).eval()

    chunks: list[np.ndarray] = []
    lengths = np.zeros(len(sequences), dtype=np.int64)

    # A sequence longer than the context is encoded in consecutive windows and
    # its residue vectors are concatenated back in order. Residues near a window
    # edge lose the context that falls on the other side of it; nothing else
    # about their position changes.
    flat, owner = [], []
    for index, sequence in enumerate(sequences):
        for start in range(0, len(sequence), WINDOW):
            flat.append(sequence[start : start + WINDOW])
            owner.append(index)
        lengths[index] = len(sequence)

    parts: dict[int, list[np.ndarray]] = {}
    for start in range(0, len(flat), batch_size):
        batch = flat[start : start + batch_size]
        tokens = tokenizer(batch, return_tensors="pt", padding=True).to(device())
        states = model(**tokens).last_hidden_state
        for row, window in enumerate(batch):
            # Drop the start and end tokens so row r is residue r of the window.
            vectors = states[row, 1 : 1 + len(window)].half().cpu().numpy()
            parts.setdefault(owner[start + row], []).append(vectors)

    for index in range(len(sequences)):
        chunks.append(np.concatenate(parts[index]))
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    return np.concatenate(chunks), offsets


def cached(sequences: list[str], path: Path, batch_size: int = 4):
    if path.exists():
        store = np.load(path, allow_pickle=True)
        if list(store["sequences"]) == list(sequences):
            return store["vectors"], store["offsets"]
    vectors, offsets = encode(sequences, batch_size)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, sequences=np.array(sequences, dtype=object), vectors=vectors, offsets=offsets)
    return vectors, offsets
