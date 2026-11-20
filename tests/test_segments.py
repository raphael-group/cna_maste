"""`segments`' phasing segmentation, as `run_cnamaste` builds it from `initial_phase_given_partition`'s output.

The run's chain (RC:174-425): `TrackedSegmentation` with `initial_blocks`, `phasing_segments` on the refined lengths,
`omics.create_bin_ranges` on them, `phased_bins`, and the bins' counts by `omics.summarize_counts_for_bins` under the
phase indicator. Referee: a seeded genome with planted blocks, phase segments, phase switches and minor BAF per
segment, so every boundary, label and count sum is known. The phase recovery itself is PR- #19's.
"""

from __future__ import annotations

from typing import NamedTuple

import anndata
import numpy as np
import pandas as pd
import pytest

from cnamaste.omics import create_bin_ranges, summarize_counts_for_bins
from cnamaste.segments import TrackedSegmentation, get_dropped, phasing_segments
from labelled_counts import get_labelled_counts

BLOCKS_PER_CONTIG = np.array([40, 30, 20])
REFINED_LENGTHS = np.array([15, 25, 30, 8, 12])
"""Planted phase segments, in blocks: breakpoints at block 15 of contig 1 and block 8 of contig 3."""
SEGMENT_BAF = np.array([0.2, 0.4, 0.5, 0.1, 0.3])
"""The minor BAF of each phase segment, as phased."""
GENES_PER_BLOCK, N_SPOTS = 2, 4
DROPPED_GENE = 90
"""Contig 2's 11th gene, between blocks 44 and 45 and in neither, as `assign_initial_blocks` leaves one: it belongs to
no segment and no bin."""


class Genome(NamedTuple):
    """The gene table, the gene counts and the block counts `create_bin_ranges` reads, and the planted phase."""

    df_gene_snp: pd.DataFrame
    adata: anndata.AnnData
    single_X: np.ndarray
    single_total_bb_RD: np.ndarray
    phase_indicator: np.ndarray
    block_segment: np.ndarray


def get_genome(seed: int) -> Genome:
    """Seeded: 90 blocks of 2 genes over 3 contigs, and a 181st gene in none; 4 spots. Each block's B count is drawn at
    its segment's minor BAF where the planted phase is 1 and at its mirror where 0, phase switching with probability
    0.2 per block, so the phase-corrected B count is at the minor BAF everywhere.
    """
    rng = np.random.default_rng(seed)
    n_blocks = BLOCKS_PER_CONTIG.sum()
    block_contig = np.repeat(np.arange(1, 4), BLOCKS_PER_CONTIG)
    block_id = np.repeat(np.arange(n_blocks, dtype=float), GENES_PER_BLOCK)
    contig = np.repeat(block_contig, GENES_PER_BLOCK)
    block_id, contig = np.insert(block_id, DROPPED_GENE, np.nan), np.insert(contig, DROPPED_GENE, 2)
    start = np.concatenate([1e5 * np.arange(np.sum(contig == c)) for c in (1, 2, 3)]).astype(np.int64)
    genes = [f"g{i}" for i in range(contig.size)]
    df = pd.DataFrame(
        {"CHR": contig, "START": start, "END": start + 50_000, "is_interval": True, "gene": genes, "block_id": block_id}
    )
    gene_counts = rng.poisson(20, (N_SPOTS, contig.size))
    adata = anndata.AnnData(gene_counts.astype(float), var=pd.DataFrame(index=genes))
    adata.layers["count"] = gene_counts
    in_block = ~np.isnan(block_id)
    rdr = np.zeros((n_blocks, N_SPOTS), dtype=np.int64)
    np.add.at(rdr, block_id[in_block].astype(int), gene_counts[:, in_block].T)

    block_segment = np.repeat(np.arange(REFINED_LENGTHS.size), REFINED_LENGTHS)
    phase_indicator = np.cumsum(rng.random(n_blocks) < 0.2) % 2
    baf = np.where(phase_indicator == 1, SEGMENT_BAF[block_segment], 1.0 - SEGMENT_BAF[block_segment])
    total = rng.poisson(150, (n_blocks, N_SPOTS))
    single_X = np.stack([rdr, rng.binomial(total, baf[:, None])], axis=1)
    return Genome(df, adata, single_X, total, phase_indicator, block_segment)


def track(genome: Genome) -> TrackedSegmentation:
    """`TrackedSegmentation` with `initial_blocks` and `phasing_segments`, as RC:174-175 and RC:369-372."""
    tracked = TrackedSegmentation(genome.df_gene_snp)
    tracked.update_from_table(genome.df_gene_snp, "block_id", "initial_blocks")
    tracked.update(phasing_segments(tracked["initial_blocks"], REFINED_LENGTHS), "phasing_segments")
    return tracked


def test_phasing_segments_are_the_refined_runs_of_blocks() -> None:
    """Each gene takes its block's planted segment, the gene in no block `DROPPED`; segments start at the planted
    breakpoints, blocks 15 and 78, exactly; 2, 1 and 2 segments per contig, none spanning two; the blocks refine
    them. Fails where a run of `refined_lengths` is misread, a block lands in its neighbour's segment, or a
    dropped gene is labelled."""
    genome = get_genome(seed=0)
    tracked = track(genome)
    segments, blocks = tracked["phasing_segments"], tracked["initial_blocks"]
    block_id = genome.df_gene_snp.block_id.to_numpy()
    expected = np.full(block_id.size, get_dropped())
    expected[~np.isnan(block_id)] = genome.block_segment[block_id[~np.isnan(block_id)].astype(int)]
    np.testing.assert_array_equal(segments.label, expected)
    first_block = block_id[segments.first].astype(int)
    assert first_block.tolist() == np.r_[0, np.cumsum(REFINED_LENGTHS)[:-1]].tolist() == [0, 15, 40, 70, 78]
    assert segments.lengths.tolist() == [2, 1, 2]
    assert np.array_equal(segments.contig, genome.df_gene_snp.CHR.to_numpy()[segments.last])
    assert blocks.refines(segments) and not segments.refines(blocks)


@pytest.mark.parametrize(
    ("refined_lengths", "refusal"),
    [([15, 30, 25, 8, 12], "spans two contigs"), ([15, 25, 30, 8, 11], "cover 89 blocks of 90")],
    ids=["across a contig", "short a block"],
)
def test_phasing_segments_refuse_lengths_that_do_not_tile_each_contig(refined_lengths: list[int], refusal: str) -> None:
    """90 blocks again, but a segment of blocks 15-44 crossing from contig 1 to 2, or lengths summing to 89: each
    `ValueError`. Fails where a phase segment may span two contigs or blocks are left unsegmented."""
    tracked = TrackedSegmentation(get_genome(seed=0).df_gene_snp)
    blocks = tracked.update_from_table(get_genome(seed=0).df_gene_snp, "block_id", "initial_blocks")
    with pytest.raises(ValueError, match=refusal):
        tracked.update(phasing_segments(blocks, np.array(refined_lengths)), "phasing_segments")


def test_bins_never_cross_a_phase_segment_and_conserve_the_phased_counts() -> None:
    """`create_bin_ranges` at 1,500 SNP UMIs per bin (about 3 blocks) on the planted segments, then
    `summarize_counts_for_bins` under the planted phase: the phase segments refine into the bins, none crossing a
    contig; the bins' read depth sums each spot's genes in blocks, the dropped gene in none; their SNP depth and
    phased B counts, `B` where the phase is 1 and `total - B` where 0, sum the blocks', exactly. Each of the 29 bins'
    BAF lies within 0.05 of its segment's planted minor BAF (measured 0.033; binomial sd 0.012 at 1,682 reads or
    more), and, ignoring the phase, a bin is 0.1 or more off (0.48). Fails where a bin crosses a phase breakpoint or
    contig, a count is lost or doubled, or the phase is ignored or inverted."""
    genome = get_genome(seed=0)
    tracked = track(genome)
    block_counts = get_labelled_counts(genome.single_X[:, 0], genome.single_X[:, 1], genome.single_total_bb_RD)
    df = create_bin_ranges(
        genome.df_gene_snp.copy(),
        genome.adata,
        None,
        None,
        None,
        block_counts,
        REFINED_LENGTHS,
        0,
        1_500,
        0,
        max_binlength=5e6,
    )
    bins = tracked.update_from_table(df, "bin_id", "phased_bins")
    assert bins.refines(tracked["phasing_segments"]) and bins.n_segments > REFINED_LENGTHS.size
    assert bins.label[DROPPED_GENE] == get_dropped()

    def binned(phase_indicator: np.ndarray) -> np.ndarray:
        """The bins' (read depth, phased B, SNP depth), each (n_bins, n_spots)."""
        counts = summarize_counts_for_bins(df, genome.adata, block_counts, phase_indicator)
        return np.stack([c.toarray().T for c in counts.multi_counts])

    rdr, b_allele, total = binned(genome.phase_indicator)
    in_block = np.ones(genome.adata.n_vars, dtype=bool)
    in_block[DROPPED_GENE] = False
    np.testing.assert_array_equal(rdr.sum(axis=0), genome.adata.layers["count"][:, in_block].sum(axis=1))
    np.testing.assert_array_equal(total.sum(axis=0), genome.single_total_bb_RD.sum(axis=0))
    phase = genome.phase_indicator[:, None] == 1
    observed = genome.single_X[:, 1]
    phased = np.where(phase, observed, genome.single_total_bb_RD - observed)
    np.testing.assert_array_equal(b_allele.sum(axis=0), phased.sum(axis=0))

    bin_segment = bins.aggregate(tracked["phasing_segments"].label, np.maximum)
    baf = b_allele.sum(axis=1) / total.sum(axis=1)
    assert np.abs(baf - SEGMENT_BAF[bin_segment]).max() <= 0.05
    _, unphased_b, _ = binned(np.ones_like(genome.phase_indicator))
    assert np.abs(unphased_b.sum(axis=1) / total.sum(axis=1) - SEGMENT_BAF[bin_segment]).max() >= 0.1
