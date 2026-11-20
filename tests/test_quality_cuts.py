"""`quality_cuts`: the filter files as `load_input_data` reads them, and the gene cuts it makes on the read depth.

Referees: planted counts, dense and CSR, whose cut genes are known; the binomial thinning's mean and standard
deviation for `downsample_genes`. Normal spots and the per-bin `quality.min_normal_count_perbin` cut are
`test_normal_spot.py`'s.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from cnamaste.quality_cuts import downsample_genes, get_filter_genes, get_filter_ranges, zero_genes

N_SPOTS, N_GENES = 2_000, 6
CAP = 5_000.0
OVER_CAP = np.array([1, 4])
"""Genes planted at 10 UMIs per spot, 20,000 in all, 4x `CAP`; gene 2 sits at `CAP` exactly, the rest below it."""


def get_planted_counts(backing: str) -> np.ndarray | sp.csr_matrix:
    """Seeded int64 counts, `(N_SPOTS, N_GENES)`, a third of entries zero.

    Args:
        backing: `dense` or `csr`.

    Returns:
        The counts.
    """
    rng = np.random.default_rng(3)
    counts = rng.poisson(1.0, (N_SPOTS, N_GENES)) * (rng.random((N_SPOTS, N_GENES)) > 1 / 3)
    counts[:, OVER_CAP] = 10
    counts[:, 2] = 0
    counts[: int(CAP) // 5, 2] = 5
    return counts if backing == "dense" else sp.csr_matrix(counts)


def as_dense(counts: np.ndarray | sp.csr_matrix) -> np.ndarray:
    return counts.toarray() if sp.issparse(counts) else np.asarray(counts)


@pytest.mark.parametrize("backing", ["dense", "csr"])
def test_zero_genes_zeroes_the_masked_genes_alone(backing: str) -> None:
    """Genes 0 and 3 zeroed in every spot, the others and the input unchanged; a CSR result stores no zero."""
    counts = get_planted_counts(backing)
    before = as_dense(counts).copy()
    genes = np.isin(np.arange(N_GENES), [0, 3])
    found = zero_genes(counts, genes)

    expected = before.copy()
    expected[:, genes] = 0
    np.testing.assert_array_equal(as_dense(found), expected)
    np.testing.assert_array_equal(as_dense(counts), before)
    if sp.issparse(found):
        assert (found.data != 0).all()


@pytest.mark.parametrize("backing", ["dense", "csr"])
def test_downsample_genes_thins_the_genes_over_the_cap_to_it(backing: str) -> None:
    """Genes over `CAP` thinned at `CAP / total` = 1/4: each entry at most its input, the total within 4 sd,
    sqrt(20,000 x 1/4 x 3/4) = 61.2, of `CAP`. Gene 2 at `CAP` and the genes below it are bitwise the input, the
    draw is the seed's alone, and NumPy's global state is not consumed."""
    counts = get_planted_counts(backing)
    before = as_dense(counts).copy()
    state = np.random.get_state()
    found = as_dense(downsample_genes(counts, CAP))
    assert all(np.array_equal(a, b) for a, b in zip(np.random.get_state(), state, strict=True))

    kept = np.setdiff1d(np.arange(N_GENES), OVER_CAP)
    np.testing.assert_array_equal(found[:, kept], before[:, kept])
    assert (found[:, OVER_CAP] <= before[:, OVER_CAP]).all()
    assert (np.abs(found[:, OVER_CAP].sum(axis=0) - CAP) < 4 * np.sqrt(20_000 * 0.25 * 0.75)).all()
    np.testing.assert_array_equal(as_dense(downsample_genes(counts, CAP)), found)
    np.testing.assert_array_equal(as_dense(counts), before)


def test_the_filter_files_read_every_line_ranges_in_numeric_genome_order(tmp_path: Path) -> None:
    """A gene file has no header, so its first gene is kept; ranges drop `chr` and sort by contig as an integer, so
    chr2 precedes chr10, then by start."""
    genes = tmp_path / "genes.txt"
    genes.write_text("HBB\nMT-CO1\n")
    assert get_filter_genes(genes).iloc[:, 0].tolist() == ["HBB", "MT-CO1"]

    ranges = tmp_path / "ranges.bed"
    ranges.write_text("chr10\t500\t600\nchr2\t900\t1000\nchr2\t100\t200\n")
    found = get_filter_ranges(ranges)
    assert found[["Chr", "Start", "End"]].to_numpy().tolist() == [[2, 100, 200], [2, 900, 1000], [10, 500, 600]]
