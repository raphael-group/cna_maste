"""`SpatioGenomicCounts`' sums of a float, spot-major: the dense array's bitwise numpy's of its bin-major transpose, and
a CSR's refereed by numpy's to rtol 1e-12.

The CSR sums each row and column in scipy's order, not numpy's pairwise one (T- #118): on non-negative values over
16 decades, as counts and baselines are, the two orders agree to rounding.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from cnamaste.spatio_genomic_counts import (
    NormalBaselineSized,
    SpatioGenomicCounts,
    SpatioGenomicLabelledMultiCounts,
    get_label_sums,
)

N_ROWS = 5
WIDTHS = (1, 7, 8, 127, 128, 129, 1000, 4099)
"""Below numpy's 8-way unroll, at it, at and about its 128 leaf, and splits two and five deep."""


def get_spread_values(n_columns: int, seed: int = 3) -> np.ndarray:
    """`(N_ROWS, n_columns)` C-order non-negative floats, magnitudes 1e-8 to 1e8, ~50% zeros, one empty row (1)."""
    rng = np.random.default_rng(seed)
    values = 10.0 ** rng.uniform(-8, 8, (N_ROWS, n_columns))
    values *= rng.random((N_ROWS, n_columns)) < 0.5
    values[1] = 0.0
    return np.ascontiguousarray(values)


@pytest.mark.parametrize("n_columns", WIDTHS)
def test_the_sums_are_numpy_s_of_the_bin_major_array(n_columns: int) -> None:
    """On `(n_columns, N_ROWS)` spot-major floats: `sum(axis=0)`, `sum(axis=1)` and `spot_sums(idx)` of the dense array
    are numpy's of its bin-major transpose bitwise, of the CSR to rtol 1e-12, and exactly 0.0 on the empty bin. Fails
    where a sum leaves the bin-major order, drops a stored entry or reads a spot outside `idx`."""
    values = get_spread_values(n_columns).T
    bin_major = np.ascontiguousarray(values.T)
    idx = np.sort(np.random.default_rng(5).permutation(n_columns)[: max(1, n_columns // 2)])
    dense, csr = SpatioGenomicCounts(values), SpatioGenomicCounts(sp.csr_matrix(values))
    assert csr.sparse and not dense.sparse
    for found, rtol in ((dense, 0.0), (csr, 1e-12)):
        np.testing.assert_allclose(found.sum(axis=0), np.sum(bin_major, axis=1), rtol=rtol, atol=0.0)
        np.testing.assert_allclose(found.sum(axis=1), np.sum(bin_major, axis=0), rtol=rtol, atol=0.0)
        np.testing.assert_allclose(found.spot_sums(idx), np.sum(bin_major[:, idx], axis=1), rtol=rtol, atol=0.0)
        assert found.sum(axis=0)[1] == 0.0


def test_the_sized_baseline_is_its_dense_product_summed_and_counted() -> None:
    """On 4099 spots by `N_ROWS` segments, coverage with zeros, a zero segment and labels 0 to 3 with -1 dropped and
    label 2 empty: `get_label_sums` of `NormalBaselineSized` is `np.sum(toarray()[labels == k], axis=0)` bitwise, zero at
    label 2, `toarray()` the `@` product the dense baseline was, and `count_positive` the dense count at every axis.
    Fails where a spot is added out of spot order or to the wrong label, -1 is read, or a factor's zero is counted."""
    rng = np.random.default_rng(7)
    coverage = rng.integers(0, 5000, 4099) * (rng.random(4099) < 0.9)
    pooled = 10.0 ** rng.uniform(0, 6, N_ROWS)
    pooled[1] = 0.0
    baseline = NormalBaselineSized(coverage.astype(np.float64), pooled / np.sum(pooled))
    dense = coverage.reshape(-1, 1) @ baseline.rdr_normal.reshape(1, -1)
    assert np.array_equal(baseline.toarray(), dense)
    labels = rng.choice(np.array([-1, 0, 1, 3]), size=coverage.size)
    sums = get_label_sums(baseline, labels, 4)
    for label in range(4):
        assert np.array_equal(sums[label], np.sum(dense[labels == label], axis=0))
    assert not sums[2].any()
    for axis in (None, 0, 1):
        assert np.array_equal(baseline.count_positive(axis), (dense > 0).sum(axis=axis))


def get_labelled_counts(n_spots: int, n_segments: int, seed: int) -> SpatioGenomicLabelledMultiCounts:
    """Integer CSR channels at about 8% density, as binned at 100K spots, with a `normal_spot` labelling."""
    rng = np.random.default_rng(seed)

    def draw(scale: int) -> sp.csr_matrix:
        values = rng.poisson(scale, (n_spots, n_segments)) * (rng.random((n_spots, n_segments)) < 0.08)
        return sp.csr_matrix(values.astype(np.int64))

    total = draw(6)
    alt = sp.csr_matrix((rng.binomial(total.data, 0.4), total.indices, total.indptr), shape=total.shape)
    normal = rng.random(n_spots) < 0.2
    return SpatioGenomicLabelledMultiCounts(
        *(SpatioGenomicCounts(m) for m in (draw(30), alt, total)),
        np.ones(n_segments, dtype=np.int64),
        np.arange(n_segments),
        np.arange(n_segments) + 1,
        {"normal_spot": normal},
        {},
        {},
    )


@pytest.mark.parametrize("dense", [False, True], ids=["csr", "dense"])
def test_the_integer_label_sums_are_each_label_s_rows_summed(dense: bool) -> None:
    """On 2,000 spots by 300 segments, labels 0 to 4 with -1 dropped and label 3 empty: `get_label_sums` of the
    integer channels equals `np.sum(toarray()[labels == k], axis=0)` exactly, CSR or dense, zero at label 3. Fails
    where a spot is summed into the wrong label, -1 is read or the product's orientation drops an entry."""
    counts = get_labelled_counts(2000, 300, seed=11)
    labels = np.random.default_rng(2).choice(np.array([-1, 0, 1, 2, 4]), size=counts.n_spots)
    for channel in (counts.umis, counts.alt_snp_umis, counts.total_snp_umis):
        found = SpatioGenomicCounts(channel.toarray()) if dense else channel
        sums = get_label_sums(found, labels, 5)
        dense_values = channel.toarray()
        for label in range(5):
            assert np.array_equal(sums[label], np.sum(dense_values[labels == label], axis=0))
        assert not sums[3].any()


@pytest.mark.parametrize("n_moved", [0, 1, 50, 2000], ids=["none", "one", "some", "all"])
def test_the_updated_pseudobulk_is_the_pseudobulk(n_moved: int) -> None:
    """On 2,000 spots by 300 segments, 4 labels, `n_moved` spots relabelled, -1 to and from included: every channel of
    `get_updated_pseudobulk` from the previous labelling's pseudobulk is `get_pseudobulk`'s bitwise, the sized baseline
    included; a labelling with fewer labels is summed in full. Fails where a moved spot's row is added to the wrong
    label, not removed from its old one, or a -1 is read."""
    counts = get_labelled_counts(2000, 300, seed=13)
    rng = np.random.default_rng(n_moved)
    before = rng.choice(np.array([-1, 0, 1, 2, 3]), size=counts.n_spots, p=[0.05, 0.25, 0.25, 0.25, 0.2])
    before[:4] = np.arange(4)
    after = before.copy()
    moved = rng.choice(np.arange(4, counts.n_spots), size=min(n_moved, counts.n_spots - 4), replace=False)
    after[moved] = rng.choice(np.array([-1, 0, 1, 2, 3]), size=moved.size)
    previous = counts.with_labelling("clone", before).get_pseudobulk("clone")
    relabelled = counts.with_labelling("clone", after)
    found, expected = relabelled.get_updated_pseudobulk("clone", previous, before), relabelled.get_pseudobulk("clone")
    for got, want in zip(found[:4], expected[:4], strict=True):
        assert got.dtype == want.dtype and np.array_equal(got, want)
    fewer = np.where(after == 3, 2, after)
    found = counts.with_labelling("clone", fewer).get_updated_pseudobulk("clone", previous, before)
    expected = counts.with_labelling("clone", fewer).get_pseudobulk("clone")
    assert found.umis.shape[0] == 3
    for got, want in zip(found[:4], expected[:4], strict=True):
        assert np.array_equal(got, want)
