"""Group targets by sequence homology so a split can separate protein families.

Holding out a target by exact sequence is not enough. If a test target's close
homologue sits in training, the model has effectively seen it: a two-residue
variant of the same domain binds much the same peptides. The published
leave-target-out benchmarks impose exact-sequence exclusion only, and say so.
"""
from __future__ import annotations

import numpy as np
from Bio import Align

# Two sequences sharing almost no 3-mers cannot align well, so they are ruled
# out before alignment rather than by it. This is an optimisation; the identity
# rule below is the similarity measure.
KMER = 3
PREFILTER = 0.05

# Percent identity on its own does not separate related proteins from unrelated
# ones. Measured on random target pairs from this dataset, the identity of the
# best local alignment has a median of 0.33 and reaches 0.83, because a short
# high-scoring stretch can always be found somewhere. What distinguishes the
# random pairs is that those stretches are short: their alignments cover a
# median of 8% of the shorter sequence. Requiring both a high identity and a
# long alignment calls 0.8% of random pairs related, while an identical
# sequence and a 20%-mutated copy both still pass.
MIN_IDENTITY = 0.30
MIN_COVERAGE = 0.50


def _kmers(sequence: str) -> set[str]:
    return {sequence[i : i + KMER] for i in range(len(sequence) - KMER + 1)}


def _aligner() -> Align.PairwiseAligner:
    aligner = Align.PairwiseAligner(scoring="blastp")
    # Local, not global: these targets run from 72 to 7,073 residues, and a
    # global alignment of a domain against a large protein is dominated by the
    # gaps needed to span it rather than by the homology we are looking for.
    aligner.mode = "local"
    return aligner


def similarity(a: str, b: str, aligner: Align.PairwiseAligner) -> tuple[float, float]:
    """Identity within the aligned region, and how much of the shorter sequence it covers."""
    alignment = aligner.align(a, b)[0]
    blocks_a, blocks_b = alignment.aligned
    aligned_length = sum(end - start for start, end in blocks_a)
    if aligned_length == 0:
        return 0.0, 0.0
    matches = sum(
        a[start_a + offset] == b[start_b + offset]
        for (start_a, end_a), (start_b, _) in zip(blocks_a, blocks_b)
        for offset in range(end_a - start_a)
    )
    return matches / aligned_length, aligned_length / min(len(a), len(b))


def related(a: str, b: str, aligner: Align.PairwiseAligner) -> bool:
    identity, coverage = similarity(a, b, aligner)
    return identity >= MIN_IDENTITY and coverage >= MIN_COVERAGE


def cluster(sequences: list[str]) -> dict[str, int]:
    """Single-linkage connected components over the relatedness rule.

    Single linkage merges two clusters when any pair across them is related, so
    it errs towards declaring targets related. For a holdout that is the safe
    direction: the failure it prevents is silently testing on a near-copy of a
    training target. It is also the reason the rule above has to be strict, as
    single linkage will chain a whole dataset together through a permissive one.
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
            if related(sequences[i], sequences[j], aligner):
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

    Quotas are in targets rather than clusters, because cluster sizes are uneven
    and it is the target count that sets how many held-out rankings we can
    score. Each cluster goes to whichever split is furthest below its quota, so
    one large cluster cannot swamp a split.
    """
    sizes: dict[int, int] = {}
    for cluster_id in assignment.values():
        sizes[cluster_id] = sizes.get(cluster_id, 0) + 1

    order = list(sizes)
    np.random.default_rng(seed).shuffle(order)

    total = sum(sizes.values())
    wanted = [fraction * total for fraction in fractions]
    filled = [0.0, 0.0, 0.0]
    names = ("train", "validation", "test")
    placement: dict[int, str] = {}
    for cluster_id in order:
        choice = int(np.argmax([w - f for w, f in zip(wanted, filled)]))
        placement[cluster_id] = names[choice]
        filled[choice] += sizes[cluster_id]

    return {sequence: placement[cluster_id] for sequence, cluster_id in assignment.items()}
