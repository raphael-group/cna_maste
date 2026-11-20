"""`omics`: genes and SNPs into blocks, blocks into bins, and their counts, on a genome small enough to write out.

Referees: the genome below, whose genes, SNPs, blocks and bins are stated by hand, and sums taken directly over it.
Contig boundaries and length caps are checked where a function draws segments. `tests/calicost/test_omics.py`
compares the same functions to CalicoST's. It costs 1 s.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import anndata
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from cnamaste.omics import (
    assign_initial_blocks,
    binned_gene_snp,
    create_bin_ranges,
    form_gene_snp_table,
    greedy_binning_nobreak,
    summarize_counts_for_bins,
    summarize_counts_for_blocks,
)
from cnamaste.segments import Segmentation
from cnamaste.spatio_genomic_counts import SpatioGenomicCounts
from labelled_counts import get_labelled_counts

GENES = [
    ("g1", "chr1", 100, 200),
    ("g2", "chr1", 150, 300),
    ("g3", "chr1", 500, 10_000),
    ("gX", "chrX", 1, 900),
    ("g4", "chr2", 50, 150),
    ("g5", "chr2", 1_000, 2_000),
    ("g_absent", "chr2", 3_000, 4_000),
]
"""(name, contig, start, end): g1 and g2 overlap; g3 spans contig 2's coordinates, on contig 1; chrX is dropped;
`g_absent` is not in the AnnData."""

SNPS = [
    "1_120_A_C",
    "1_160_A_C",
    "1_400_A_C",
    "1_550_A_C",
    "2_60_A_C",
    "2_600_A_C",
    "2_1001_A_C",
    "2_1002_A_C",
    "2_1003_A_C",
    "2_1004_A_C",
    "2_3500_A_C",
]
"""In g1; in g1 and g2 (g2 closest in start); in no gene; in g3; in g4; in no gene on contig 2, though inside g3's
span; four in g5; in `g_absent`."""

GENE_OF = {
    "1_120_A_C": "g1",
    "1_160_A_C": "g2",
    "1_550_A_C": "g3",
    "2_60_A_C": "g4",
    "2_1001_A_C": "g5",
    "2_1002_A_C": "g5",
    "2_1003_A_C": "g5",
    "2_1004_A_C": "g5",
}
"""Each SNP the table keeps, and its gene."""

N_SPOTS = 3


def get_written_spans(table: pd.DataFrame, key: str) -> np.ndarray:
    """Per segment of `key`, as `segmentation.tsv.gz`'s gene rows state it: `CHR`, the first `START`, the last `END`."""
    genes = table[table.is_interval.astype(bool)].groupby(key)
    return np.stack([genes.CHR.first(), genes.START.first(), genes.END.last()], axis=1)


def hgtable(path: Path) -> str:
    """`GENES` as a legacy hgTables file, the columns `reference.get_reference_genes` reads."""
    rows = "".join(f"{i}\t{name}\t{chrom}\t{start}\t{end}\n" for i, (name, chrom, start, end) in enumerate(GENES))
    path.write_text("#bin\tname2\tchrom\tcdsStart\tcdsEnd\n" + rows)
    return str(path)


def counts(seed: int = 0) -> dict[str, Any]:
    """Integer gene counts (an AnnData without `g_absent`) and A and B counts per SNP, over `N_SPOTS` spots."""
    rng = np.random.default_rng(seed)
    genes = [g[0] for g in GENES if g[0] != "g_absent"]
    adata = anndata.AnnData(X=np.zeros((N_SPOTS, len(genes))), var=pd.DataFrame(index=genes))
    adata.layers["count"] = rng.integers(0, 50, (N_SPOTS, len(genes)))
    a_counts = rng.integers(0, 20, (N_SPOTS, len(SNPS)))
    b_counts = rng.integers(0, 20, (N_SPOTS, len(SNPS)))
    return {"adata": adata, "A": a_counts, "B": b_counts}


@pytest.fixture
def table(tmp_path: Path) -> pd.DataFrame:
    """`form_gene_snp_table` on the genome."""
    return form_gene_snp_table(np.array(SNPS), hgtable(tmp_path / "hg.tsv"), counts()["adata"])


# -- greedy_binning_nobreak ----------------------------------------------------------------------------------------


def binning(lengths: list[int], umi: list[int], floor: int = 10, max_binlength: float = 1e9) -> list[int]:
    """`greedy_binning_nobreak` with `floor` on the SNP UMIs alone."""
    zeros = np.zeros(len(lengths), dtype=int)
    return greedy_binning_nobreak(np.array(lengths), zeros, np.array(umi), zeros, 0, floor, 0, max_binlength).tolist()


@pytest.mark.parametrize(
    ("lengths", "umi", "floor", "max_binlength", "expected"),
    [
        ([1] * 6, [5] * 6, 10, 1e9, [0, 0, 1, 1, 2, 2]),
        ([1] * 3, [10, 10, 3], 10, 1e9, [0, 1, 1]),
        ([1], [3], 10, 1e9, [0]),
        ([1] * 4, [1] * 4, 0, 1e9, [0, 1, 2, 3]),
        ([3, 3, 3], [5, 5, 5], 10, 5, [0, 1, 1]),
        ([10], [50], 10, 5, [0]),
    ],
    ids=["shortest runs", "tail joins previous", "lone tail stays", "no floor", "length cap", "one long block"],
)
def test_greedy_binning_takes_the_shortest_runs_meeting_the_floors(
    lengths: list[int], umi: list[int], floor: int, max_binlength: float, expected: list[int]
) -> None:
    """Each bin is the shortest run of blocks meeting the floors; a failing tail joins the previous bin, where there is
    one; a run reaching `max_binlength` closes a block early, so a long block is a bin alone. Fails where a bin grows
    past its floor or its length, or a tail is left under the floor beside a bin it could join."""
    assert binning(lengths, umi, floor, max_binlength) == expected


def test_greedy_binning_holds_each_floor() -> None:
    """The UMI, SNP-UMI and normal-UMI floors each hold a bin open until met. Fails where a floor is not read."""
    ones = np.ones(4, dtype=int)
    for which in range(3):
        floors = [0, 0, 0]
        floors[which] = 2
        bins = greedy_binning_nobreak(ones, ones, ones, ones, *floors, 1e9)
        assert bins.tolist() == [0, 0, 1, 1], which


# -- SpatioGenomicCounts.from_grouped_columns ----------------------------------------------------------------------


def test_grouped_columns_sum_each_column_in_its_segment() -> None:
    """Each column summed into its segment, an unlabelled segment a zero column; at 200 spots and 40 of 50 shuffled
    columns in 7 segments, `np.add.at`'s exactly. A column given a second segment is refused. Fails where the incidence
    drops a column, a segment, or a term."""
    rng = np.random.default_rng(1)
    matrix = rng.integers(0, 2**40, (200, 50))
    columns, groups = rng.permutation(50)[:40], rng.integers(0, 6, 40)
    expected = np.zeros((200, 7), dtype=matrix.dtype)
    np.add.at(expected, (slice(None), groups), matrix[:, columns])
    segmentation = Segmentation.from_terms(50, columns, groups, 7, name="columns")
    found = SpatioGenomicCounts.from_grouped_columns(matrix, segmentation, np.int64)
    assert np.array_equal(found.toarray(), expected) and not found.toarray()[:, 6].any()
    with pytest.raises(ValueError, match="1 rows take a second label"):
        Segmentation.from_terms(50, np.r_[columns, columns[:1]], np.r_[groups, 0], 7, name="columns")


# -- form_gene_snp_table -------------------------------------------------------------------------------------------


def test_each_kept_snp_takes_its_overlapping_gene_on_its_contig(table: pd.DataFrame) -> None:
    """A SNP takes the overlapping gene closest in start (g2 over g1 at 160), never one on another contig (2_600
    lies in g3's span on contig 1, and is dropped); SNPs in no gene, and genes not in the AnnData or off the
    autosomes, are dropped. Fails where a SNP is placed across a contig or outside its gene."""
    snps = table[~table.is_interval]
    assert dict(zip(snps.snp_id, snps.gene)) == GENE_OF
    assert set(table[table.is_interval].gene) == {"g1", "g2", "g3", "g4", "g5"}
    assert list(table.CHR) == sorted(table.CHR) and table.groupby("CHR").START.is_monotonic_increasing.all()


def test_only_genes_within_num_preceeding_rows_are_searched(tmp_path: Path) -> None:
    """g5's four SNPs follow its row: with `num_preceeding_rows=3` the first three find it and the fourth is
    dropped. Fails where the search reaches further back than asked."""
    table = form_gene_snp_table(np.array(SNPS), hgtable(tmp_path / "hg.tsv"), counts()["adata"], num_preceeding_rows=3)
    kept = set(table[~table.is_interval].snp_id)
    assert {"2_1001_A_C", "2_1002_A_C", "2_1003_A_C"} <= kept and "2_1004_A_C" not in kept


# -- assign_initial_blocks -----------------------------------------------------------------------------------------


def blocks(table: pd.DataFrame, floor: int) -> pd.DataFrame:
    given = counts()
    return assign_initial_blocks(table.copy(), given["adata"], given["A"], given["B"], np.array(SNPS), floor)


def test_overlapping_genes_are_one_block_and_no_block_crosses_a_contig(table: pd.DataFrame) -> None:
    """Floor 0: g1 and g2 (overlapping) and their SNPs are one block; g3, g4 and g5 each another; block ids ascend
    along the table, every row has one, and no block holds two contigs. Fails where overlap is missed or a block
    spans a contig."""
    found = blocks(table, 0)
    ids = dict(zip(found.gene.where(found.is_interval, found.snp_id), found.block_id))
    assert ids["g1"] == ids["g2"] == ids["1_120_A_C"] == ids["1_160_A_C"]
    assert len({ids["g1"], ids["g3"], ids["g4"], ids["g5"]}) == 4
    assert (found.block_id >= 0).all() and found.block_id.is_monotonic_increasing
    assert (found.groupby("block_id").CHR.nunique() == 1).all()


def test_a_floor_merges_blocks_up_to_the_contig_end(table: pd.DataFrame) -> None:
    """A floor no block meets: each contig is one block, the last block of contig 1 not joined to contig 2's. Fails
    where an under-floor block at a contig's end merges across it."""
    found = blocks(table, 10**9)
    assert found.groupby("CHR").block_id.nunique().tolist() == [1, 1]
    assert found.block_id.nunique() == 2


@pytest.mark.parametrize("position", [10, 20_000], ids=["before the first gene", "after the last"])
def test_a_snp_in_no_gene_is_refused(table: pd.DataFrame, position: int) -> None:
    """A SNP row outside every merged gene interval, before contig 1's first gene or after contig 2's last: `ValueError`
    naming it, rather than its UMIs summed into another block. Fails where the row is placed silently."""
    chrom = 1 if position == 10 else 2
    stray = pd.DataFrame(
        [
            {
                "CHR": chrom,
                "START": position,
                "END": position + 1,
                "snp_id": "1_120_A_C",
                "gene": None,
                "is_interval": False,
            }
        ]
    )
    given = pd.concat([table, stray]).sort_values(["CHR", "START"], kind="stable")
    with pytest.raises(ValueError, match="1 SNP rows lie in no merged gene interval"):
        blocks(given, 0)


def test_known_ids_map_to_dense_ids_in_order(table: pd.DataFrame) -> None:
    """A `known_id` column is the segmentation: its ids, sorted, become 0, 1, ...; a row without one is -1. Fails where
    the order or the missing value is lost."""
    given = table.copy()
    given["known_id"] = [7 if c == 1 else 3 for c in given.CHR]
    given.iloc[0, given.columns.get_loc("known_id")] = None
    found = blocks(given, 0)
    assert found.block_id.tolist() == [-1] + [1 if c == 1 else 0 for c in found.CHR.iloc[1:]]


# -- summarize_counts_for_blocks -----------------------------------------------------------------------------------


@pytest.mark.parametrize("sparse", [False, True], ids=["dense", "csr"])
def test_block_counts_are_their_genes_and_snps_sums(table: pd.DataFrame, sparse: bool) -> None:
    """Per block and spot: its genes' counts (each gene once), its SNPs' A counts, and A + B; `contig`, `start` and
    `end` its genes' span; `lengths` the blocks per contig. Checked against sums taken directly; dense and CSR inputs
    alike. Fails where a gene or SNP is placed in another block, or counted twice."""
    given, found = counts(), blocks(table, 0)
    to = sp.csr_matrix if sparse else np.asarray
    adata = given["adata"].copy()
    adata.layers["count"] = to(adata.layers["count"])
    binned = summarize_counts_for_blocks(found, adata, to(given["A"]), to(given["B"]), np.array(SNPS))
    index = {s: i for i, s in enumerate(SNPS)}
    genes = list(given["adata"].var.index)
    for b, rows in found.groupby("block_id"):
        snp = [index[s] for s in rows.snp_id.dropna()]
        cols = [genes.index(g) for g in rows[rows.is_interval].gene.unique()]
        assert np.array_equal(binned.umis.toarray()[:, b], given["adata"].layers["count"][:, cols].sum(axis=1))
        assert np.array_equal(binned.alt_snp_umis.toarray()[:, b], given["A"][:, snp].sum(axis=1))
        assert np.array_equal(binned.total_snp_umis.toarray()[:, b], (given["A"] + given["B"])[:, snp].sum(axis=1))
    assert np.array_equal(
        np.stack([binned.contig, binned.start, binned.end], axis=1), get_written_spans(found, "block_id")
    )
    assert binned.lengths.tolist() == found.groupby("CHR").block_id.nunique().tolist()


# -- create_bin_ranges ---------------------------------------------------------------------------------------------


def segments(lengths: list[int], chrs: list[int]) -> pd.DataFrame:
    """One row per block: `CHR`, `START`, `END` (abutting, each `lengths[i]` long), `block_id`."""
    ends = np.cumsum(lengths)
    return pd.DataFrame({"CHR": chrs, "START": ends - np.array(lengths), "END": ends, "block_id": range(len(lengths))})


def bin_ranges(frame: pd.DataFrame, umi: list[int], refined: list[int], max_binlength: float = 1e9, **floors) -> Any:
    """`create_bin_ranges` on `frame`, each block's read and SNP UMIs `umi`, over one spot."""
    x = np.array(umi)[:, None]
    settings = {"secondary_min_umi": 0, "secondary_min_snp_umi": 10, "secondary_min_normal_umi": 0} | floors
    return create_bin_ranges(
        frame.copy(),
        None,
        None,
        None,
        None,
        get_labelled_counts(x, 0 * x, x),
        np.array(refined),
        max_binlength=max_binlength,
        **settings,
    )


def test_no_bin_crosses_a_refined_segment_or_a_contig() -> None:
    """Six blocks, two contigs, the refined segments the contigs, a floor no run meets: one bin per contig. Fails where
    a bin crosses a segment, and with it a contig."""
    found = bin_ranges(segments([1] * 6, [1, 1, 1, 2, 2, 2]), [1] * 6, [3, 3])
    assert found.bin_id.tolist() == [0, 0, 0, 1, 1, 1]
    assert (found.groupby("bin_id").CHR.nunique() == 1).all()


def test_a_block_over_max_binlength_is_a_bin_alone() -> None:
    """The middle of five blocks is longer than `max_binlength`: a bin of its own, under the floor or not. Fails where
    it shares a bin."""
    found = bin_ranges(segments([1, 1, 10, 1, 1], [1] * 5), [1] * 5, [5], max_binlength=5)
    assert found.bin_id.tolist() == [0, 0, 1, 2, 2]


def test_normal_umis_gate_a_bin_only_when_normal_candidates_are_given() -> None:
    """A normal-UMI floor bins blocks in pairs given every spot normal, and is ignored without `normal_candidates`.
    Fails where the floor is read without candidates, or ignored with them."""
    frame, umi = segments([1] * 4, [1] * 4), [1] * 4
    floors = {"secondary_min_snp_umi": 0, "secondary_min_normal_umi": 2}
    assert bin_ranges(frame, umi, [4], **floors).bin_id.tolist() == [0, 1, 2, 3]
    x = np.array(umi)[:, None]
    gated = create_bin_ranges(
        frame.copy(),
        None,
        None,
        None,
        None,
        get_labelled_counts(x, 0 * x, x),
        np.array([4]),
        0,
        0,
        2,
        normal_candidates=np.array([True]),
    )
    assert gated.bin_id.tolist() == [0, 0, 1, 1]


# -- summarize_counts_for_bins -------------------------------------------------------------------------------------


def test_bin_counts_flip_b_where_the_phase_is_zero(table: pd.DataFrame) -> None:
    """Bins of whole blocks: each bin's B is its blocks' B where `phase_indicator` is 1 and total - B where 0; its
    read counts are its genes' in the AnnData; `contig`, `start` and `end` carried from its blocks are its genes' span;
    `lengths` the bins per contig. Fails where a block is phased the wrong way, a gene counted in another bin, or a
    span read from another bin."""
    given, found = counts(), blocks(table, 0)
    binned = summarize_counts_for_blocks(found, given["adata"], given["A"], given["B"], np.array(SNPS))
    found["bin_id"] = found.block_id // 2
    phase = np.arange(found.block_id.nunique()) % 2
    bins = summarize_counts_for_bins(found, given["adata"], binned, phase)
    b_counts, total = binned.alt_snp_umis.toarray().T, binned.total_snp_umis.toarray().T
    phased = np.where(phase[:, None] == 1, b_counts, total - b_counts)
    genes = list(given["adata"].var.index)
    for k, rows in found.groupby("bin_id"):
        members = rows.block_id.unique()
        assert np.array_equal(bins.alt_snp_umis.toarray()[:, k], phased[members].sum(axis=0))
        assert np.array_equal(bins.total_snp_umis.toarray()[:, k], total[members].sum(axis=0))
        cols = [genes.index(g) for g in rows.gene.dropna().unique()]
        assert np.array_equal(bins.umis.toarray()[:, k], given["adata"].layers["count"][:, cols].sum(axis=1))
    assert np.array_equal(np.stack([bins.contig, bins.start, bins.end], axis=1), get_written_spans(found, "bin_id"))
    assert bins.lengths.tolist() == found.groupby("CHR").bin_id.nunique().tolist()


# -- binned_gene_snp -----------------------------------------------------------------------------------------------


def test_binned_gene_snp_states_each_bin_s_span_and_contents(table: pd.DataFrame) -> None:
    """Per bin: its first CHR and START, last END, its genes and SNPs in genomic order, `N_SNPS`; the caller's table
    unchanged. Fails where a bin's span or contents are read from another bin, or the table is written."""
    found = blocks(table, 0)
    found["bin_id"] = found.block_id
    before = found.copy()
    info = binned_gene_snp(found)
    pd.testing.assert_frame_equal(found, before)
    for _, row in info.iterrows():
        rows = found[found.bin_id == row.bin_id]
        assert (row.CHR, row.START, row.END) == (rows.CHR.iloc[0], rows.START.iloc[0], rows.END.iloc[-1])
        assert row.INCLUDED_GENES.split(",") == list(rows.gene.dropna().unique())
        assert row.INCLUDED_SNP_IDS.split(",") == list(rows.snp_id.dropna().unique())
        assert row.N_SNPS == rows.snp_id.notna().sum()
