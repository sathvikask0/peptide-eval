"""Group targets by sequence homology so a split can separate protein families.

Holding out a target by exact sequence is not enough. If a test target's close
homologue sits in training, the model has effectively seen it: a two-residue
variant of the same domain binds much the same peptides. The published
leave-target-out benchmarks impose exact-sequence exclusion only, and say so.
"""
from __future__ import annotations

import numpy as np
from Bio import Align

# Pairwise alignment over every pair is wasteful when most pairs are unrelated.
# Two sequences that share almost no 3-mers cannot align above any threshold we
# care about, so they are ruled out before alignment rather than by it.
KMER = 3
PREFILTER = 0.05


def _kmers(sequence: str) -> set[str]:
    return {sequence[i : i + KMER] for i in range(len(sequence) - KMER + 1)}


def _aligner() -> Align.PairwiseAligner:
    aligner = Align.PairwiseAligner(scoring="blastp")
    aligner.mode = "global"
    return aligner


def identity(a: str, b: str, aligner: Align.PairwiseAligner) -> float:
    """Fraction of the shorter sequence that aligns to an identical residue.

    Normalising by the shorter sequence means a short domain fully contained in
    a long protein counts as related, which is the behaviour we want: the domain
    is what binds the peptide.
    """
    alignment = aligner.align(a, b)[0]
    matches = sum(x == y for x, y in zip(*alignment))
    return matches / min(len(a), len(b))


def cluster(sequences: list[str], threshold: float = 0.30) -> dict[str, int]:
    """Single-linkage connected components at a sequence-identity threshold.

    Single linkage is deliberate and conservative: it merges two clusters when
    any pair across them is similar, so it errs towards declaring targets
    related. For a holdout that is the safe direction to err in, because the
    failure it prevents is silently testing on a near-copy of a training target.
    """
    kmer_sets = [_kmers(s) for s in sequences]
    aligner = _aligner()
    parent = list(range(len(sequences)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(sequences)):
        for j in range(i + 1, len(sequences)):
            if find(i) == find(j):
                continue
            shared = len(kmer_sets[i] & kmer_sets[j])
            if shared / max(1, min(len(kmer_sets[i]), len(kmer_sets[j]))) < PREFILTER:
                continue
            if identity(sequences[i], sequences[j], aligner) >= threshold:
                parent[find(i)] = find(j)

    labels, assignment = {}, {}
    for index, sequence in enumerate(sequences):
        root = find(index)
        if root not in labels:
            labels[root] = len(labels)
        assignment[sequence] = labels[root]
    return assignment


def split_by_cluster(assignment: dict[str, int], fractions=(0.6, 0.15, 0.25), seed: int = 42):
    """Assign whole clusters to train / validation / test.

    Clusters are shuffled and taken in order until each split has its share of
    *targets*, not of clusters, because cluster sizes are uneven and it is the
    target count that determines how many held-out rankings we can score.
    """
    sizes: dict[int, int] = {}
    for cluster_id in assignment.values():
        sizes[cluster_id] = sizes.get(cluster_id, 0) + 1

    order = list(sizes)
    np.random.default_rng(seed).shuffle(order)

    total = sum(sizes.values())
    wanted = [f * total for f in fractions]
    buckets: list[list[int]] = [[], [], []]
    filled = [0.0, 0.0, 0.0]
    for cluster_id in order:
        # Give the cluster to whichever split is furthest from its quota, so a
        # single large cluster cannot swamp one split.
        target = int(np.argmax([w - f for w, f in zip(wanted, filled)]))
        buckets[target].append(cluster_id)
        filled[target] += sizes[cluster_id]

    names = ("train", "validation", "test")
    return {
        sequence: names[next(i for i, b in enumerate(buckets) if cluster_id in b)]
        for sequence, cluster_id in assignment.items()
    }
