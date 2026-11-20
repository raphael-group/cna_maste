# ruff: noqa
"""Frozen copies of `cnamaste.spatio_genomic_counts`' `SpatioGenomicPairCounts`, `BinnedCounts` and `npz_entries` as
PR- #101 held them, which T- #84's `SpatioGenomicLabelledMultiCounts` replaces, and of the pairs form of
`SpatioGenomicCounts.from_grouped_columns` (as a function of `cls`) and `get_segmented` as PR- #103 held them, which a
`Segmentation` replaces, and the bin-major dask `SpatioGenomicCounts` they ran on, as PR- #143 held it, which the
spot-major CSR replaces; verbatim. `tests/legacy/test_legacy.py` holds the new code byte-equal to them."""

from __future__ import annotations

import time
from typing import Any, NamedTuple, Self

import numpy as np
import scipy.sparse as sp

import dask.array as da
from dask import delayed

from cnamaste.config import start_time
from cnamaste.logger import get_logger

logger = get_logger("legacy.spatio_genomic_counts", start_time=start_time)

CHUNK_BINS = 1024
"""Bins (rows) per dask block; spots are never split."""


def get_chunk_bins() -> int:
    """Return the chunk bins, 1024."""
    return CHUNK_BINS


def canonical(m: Any, dtype: Any = None) -> sp.csr_matrix:
    """Return `m` as canonical CSR, with no explicit zeros, cast to `dtype` where given.

    Args:
        m: a sparse matrix.
        dtype: optional.

    Returns:
        The canonical CSR.
    """
    out = sp.csr_matrix(m, dtype=dtype if dtype is not None else m.dtype)
    out.eliminate_zeros()
    out.sort_indices()
    out.indptr, out.indices = out.indptr.astype(np.int64), out.indices.astype(np.int64)
    return out


def get_lazy_block(block: Any, shape: tuple[int, int], dtype: Any) -> da.Array:
    """One dask block: a CSR block, or a delayed one, of `shape`.

    Args:
        block: the CSR matrix or its `dask.delayed`.
        shape: its shape.
        dtype: its dtype.

    Returns:
        The single-block dask array.
    """
    meta = sp.csr_matrix((0, 0), dtype=dtype)
    return da.from_delayed(block if hasattr(block, "dask") else delayed(block), shape, dtype=dtype, meta=meta)


def get_grouped_block(counts: sp.csr_matrix, columns: np.ndarray, groups: np.ndarray, n_groups: int, dtype: Any) -> Any:
    """Rows `n_groups` of grouped column sums: `(counts @ incidence).T` for the (column, group) incidence.

    Args:
        counts: `(n_spots, n_columns)` integer CSR.
        columns: each term's column.
        groups: each term's group, in `[0, n_groups)`.
        n_groups: the rows.
        dtype: the sums' dtype.

    Returns:
        The `(n_groups, n_spots)` canonical CSR.
    """
    incidence = sp.csr_matrix(
        (np.ones(len(columns), dtype=counts.dtype), (columns, groups)), shape=(counts.shape[1], n_groups)
    )
    return canonical((counts @ incidence).T, dtype)


# --- the per-spot array -----------------------------------------------------------------------------------------------


class SpatioGenomicCounts:
    """One per-spot quantity `(n_obs, n_spots)`, dense or canonical CSR; every operation gives the same bits from either.

    Args:
        values: a dense array or sparse matrix (made canonical CSR).
    """

    ndim = 2

    def __init__(self, values: Any) -> None:
        # NB a dense array kept as given: a spot-major view (`by_spot`) is no copy, and `sum` keeps numpy's order.
        self.blocks: da.Array | None = values if isinstance(values, da.Array) else None
        self.computed: Any = None
        if self.blocks is None:
            self.computed = canonical(values) if sp.issparse(values) else np.asarray(values)

    @classmethod
    def from_grouped_columns(cls, counts: Any, segmentation: Segmentation, dtype: Any) -> SpatioGenomicCounts:
        """The lazy `(n_segments, n_spots)` sums of integer `counts` columns, each column in its segment. One CSR
        product per block of `CHUNK_BINS` segments.

        Args:
            counts: `(n_spots, n_columns)` integer counts, CSR or dense.
            segmentation: of the `n_columns` columns.
            dtype: the sums' dtype.

        Returns:
            The lazy `SpatioGenomicCounts`.
        """
        columns, groups = segmentation.terms
        n_groups = segmentation.n_segments
        counts = sp.csr_matrix(counts)
        blocks = []
        for start in range(0, n_groups, CHUNK_BINS):
            stop = min(start + CHUNK_BINS, n_groups)
            terms = (groups >= start) & (groups < stop)
            block = delayed(get_grouped_block)(counts, columns[terms], groups[terms] - start, stop - start, dtype)
            blocks.append(get_lazy_block(block, (stop - start, counts.shape[0]), dtype))
        return cls(da.concatenate(blocks, axis=0) if blocks else get_lazy_block(sp.csr_matrix((0, 0)), (0, 0), dtype))

    def compute(self) -> Any:
        """The array, canonical CSR or dense: the lazy blocks computed once, timed in the log.

        Returns:
            The `(n_obs, n_spots)` array.
        """
        if self.computed is None:
            start = time.perf_counter()
            found = self.blocks.compute(scheduler="synchronous")  # type: ignore[union-attr]
            self.computed = canonical(found) if sp.issparse(found) else np.asarray(found)
            logger.debug(f"Computed {self.shape} counts in {time.perf_counter() - start:.3f} s.")
        return self.computed

    @property
    def values(self) -> Any:
        return self.compute()

    @classmethod
    def of(cls, a: Any) -> SpatioGenomicCounts:
        return a if isinstance(a, cls) else cls(a)

    @property
    def sparse(self) -> bool:
        return sp.issparse(self.blocks._meta) if self.blocks is not None else sp.issparse(self.computed)

    @property
    def shape(self) -> tuple[int, int]:
        found = self.blocks if self.blocks is not None else self.computed
        return (int(found.shape[0]), int(found.shape[1]))

    @property
    def dtype(self) -> np.dtype:
        return self.blocks.dtype if self.blocks is not None else self.computed.dtype

    def get_row_blocks(self) -> da.Array:
        """The array as dask blocks of `CHUNK_BINS` rows: its own where lazy.

        Returns:
            The dask array.
        """
        if self.blocks is not None:
            return self.blocks
        starts = range(0, self.shape[0], CHUNK_BINS)
        rows = [self.computed[start : start + CHUNK_BINS] for start in starts]
        return da.concatenate([get_lazy_block(row, row.shape, self.dtype) for row in rows], axis=0)

    def toarray(self) -> np.ndarray:
        return self.values.toarray() if self.sparse else self.values.copy()

    def like(self, values: Any) -> SpatioGenomicCounts:
        """Return `values` in this array's backing.

        Args:
            values: an array of this shape.

        Returns:
            A `SpatioGenomicCounts`.
        """
        if self.sparse:
            return SpatioGenomicCounts(sp.csr_matrix(values))
        return SpatioGenomicCounts(values.toarray() if sp.issparse(values) else values)

    def zeros(self, dtype: Any = np.float64) -> SpatioGenomicCounts:
        """Return zeros of this shape in this backing; a dense-backed run's unset baseline stays dense zeros.

        Args:
            dtype: of the zeros.

        Returns:
            A `SpatioGenomicCounts`.
        """
        return SpatioGenomicCounts(
            sp.csr_matrix(self.shape, dtype=dtype) if self.sparse else np.zeros(self.shape, dtype=dtype)
        )

    def astype(self, dtype: Any) -> SpatioGenomicCounts:
        """Cast as numpy casts: a float to an integer truncated, as assignment into an integer array.

        Args:
            dtype: the target.

        Returns:
            A `SpatioGenomicCounts`.
        """
        if self.sparse:
            m = self.values.copy()
            m.data = m.data.astype(dtype)
            return SpatioGenomicCounts(m)
        return SpatioGenomicCounts(self.values.astype(dtype))

    def rows(self, index: Any) -> SpatioGenomicCounts:
        return SpatioGenomicCounts(self.values[index, :])

    def spots(self, index: Any) -> SpatioGenomicCounts:
        return SpatioGenomicCounts(self.values[:, index])

    def zero_rows(self, index: Any) -> SpatioGenomicCounts:
        """Set rows `index` to 0, in a new array.

        Args:
            index: the rows.

        Returns:
            A `SpatioGenomicCounts`.
        """
        if self.sparse:
            keep = np.ones(self.shape[0], dtype=self.dtype)
            keep[index] = 0
            return SpatioGenomicCounts(sp.diags(keep, format="csr", dtype=self.dtype) @ self.values)
        out = self.values.copy()
        out[index, :] = 0
        return SpatioGenomicCounts(out)

    def product(self, left: Any) -> SpatioGenomicCounts:
        """Compute `left @ self` for integer counts, exact in any order.

        Args:
            left: a sparse integer `(n, n_obs)`, an incidence or a selection.

        Returns:
            A `SpatioGenomicCounts` `(n, n_spots)`.
        """
        if self.dtype.kind not in "iu":
            raise TypeError("product of float values: its order is not numpy's.")
        out = left @ self.values
        return SpatioGenomicCounts(out if self.sparse else np.asarray(out, dtype=self.dtype))

    def __add__(self, other: SpatioGenomicCounts) -> SpatioGenomicCounts:
        return self.like(self.values + other.values) if self.dtype.kind in "iu" else NotImplemented

    def __sub__(self, other: SpatioGenomicCounts) -> SpatioGenomicCounts:
        return self.like(self.values - other.values) if self.dtype.kind in "iu" else NotImplemented

    def map(self, f: Any) -> SpatioGenomicCounts:
        """Apply `f` elementwise; on CSR to the stored entries only, so `f(0)` must be 0.

        Args:
            f: a numpy ufunc or vectorized callable.

        Returns:
            A `SpatioGenomicCounts`.
        """
        if not self.sparse:
            return SpatioGenomicCounts(f(self.values))
        if f(np.zeros(1, self.dtype))[0] != 0:
            raise ValueError("map of CSR: f(0) != 0, the result is dense.")
        m = self.values.copy()
        m.data = f(m.data)
        return SpatioGenomicCounts(m)

    def sum(self, axis: int | None = None) -> Any:
        """Compute `np.sum(values, axis)` on either backing; a float CSR's sums agree with the dense to rounding.

        Args:
            axis: 0, 1 or None.

        Returns:
            The sum.
        """
        if not self.sparse:
            return np.sum(self.values, axis=axis)
        dtype = self.dtype if self.dtype.kind in "iub" else np.float64
        if axis is None:
            return self.values.sum(dtype=dtype)
        return np.asarray(self.values.sum(axis=axis, dtype=dtype)).ravel()

    def spot_sums(self, idx: Any = None) -> np.ndarray:
        """Compute `np.sum(values[:, idx], axis=1)`: per segment, the spots `idx` summed; every spot where None.

        Args:
            idx: spot indices or None.

        Returns:
            The (n_obs,) sums.
        """
        if idx is None:
            return self.sum(axis=1)
        return SpatioGenomicCounts(self.values[:, idx]).sum(axis=1)

    def mean(self) -> float:
        return float(np.mean(self.values)) if not self.sparse else float(self.sum()) / (self.shape[0] * self.shape[1])

    def count_positive(self, axis: int | None = None) -> Any:
        """Compute `(values > 0).sum(axis)`.

        Args:
            axis: 0, 1 or None.

        Returns:
            The counts of positive entries.
        """
        if not self.sparse:
            return (self.values > 0).sum(axis=axis)
        m = self.values
        positive = m.data > 0
        if axis is None:
            return int(np.count_nonzero(positive))
        if axis == 0:
            return np.bincount(m.indices[positive], minlength=m.shape[1])
        return (
            np.add.reduceat(positive.astype(np.int64), m.indptr[:-1]) * (np.diff(m.indptr) > 0)
            if m.nnz
            else np.zeros(m.shape[0], np.int64)
        )

    def is_zero(self) -> bool:
        return self.values.count_nonzero() == 0 if self.sparse else bool(np.all(self.values == 0))

    def by_spot(self) -> Any:
        """Return spot-major values: CSR `(n_spots, n_obs)`, or the dense array's transposed view (no copy).

        Returns:
            The `(n_spots, n_obs)` array.
        """
        return canonical(self.values.T) if self.sparse else self.values.T

    def ratio(self, other: Any, fill_value: float = np.nan) -> np.ndarray:
        """Return this array over `other`, dense; a ratio is undefined, not zero, where `other` is zero.

        Args:
            other: the `(n_obs, n_spots)` denominator, a `SpatioGenomicCounts`, CSR or dense array.
            fill_value: for undefined ratios.

        Returns:
            The `(n_obs, n_spots)` ratio.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = self.toarray() / as_dense_array(other)
        return np.nan_to_num(ratio, nan=fill_value)


def get_segmented_block(fine: Any, segments: np.ndarray, rows: np.ndarray, n_segments: int) -> sp.csr_matrix:
    """Rows of a coarser segmentation's sums: segment `segments[k]` adds fine row `rows[k]`.

    Args:
        fine: `(n_fine, n_spots)` CSR.
        segments: each term's segment, in `[0, n_segments)`.
        rows: each term's fine row.
        n_segments: the rows.

    Returns:
        The `(n_segments, n_spots)` canonical CSR, in `fine`'s dtype.
    """
    incidence = sp.csr_matrix(
        (np.ones(len(rows), dtype=fine.dtype), (segments, rows)), shape=(n_segments, fine.shape[0])
    )
    return canonical(incidence @ fine, fine.dtype)


class SpatioGenomicPairCounts:
    """`single_X` `(n_obs, 2, n_spots)`: channel 0 read depth (`rdr`), 1 BAF numerator (`baf`), one backing.

    Each channel is a `SpatioGenomicCounts`, one per-spot quantity; `SpatioGenomicPairCounts` pairs them as the HMM's
    two observations.

    Args: `rdr`, `baf` each `(n_obs, n_spots)`, a `SpatioGenomicCounts`, dense array or CSR.
    """

    ndim = 3

    def __init__(self, rdr: Any, baf: Any) -> None:
        self.rdr, self.baf = SpatioGenomicCounts.of(rdr), SpatioGenomicCounts.of(baf)

    @classmethod
    def of(cls, x: Any) -> SpatioGenomicPairCounts:
        """Return `x` as `SpatioGenomicPairCounts`: itself, or the dense `(n_obs, 2, n_spots)` a capture records, kept
        dense.

        Args: `x` a `SpatioGenomicPairCounts` or `(n_obs, 2, n_spots)` array.
        Returns: a `SpatioGenomicPairCounts`.
        """
        if isinstance(x, cls):
            return x
        x = np.asarray(x)
        return cls(x[:, 0, :], x[:, 1, :])

    @property
    def sparse(self) -> bool:
        return self.rdr.sparse

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.rdr.shape[0], 2, self.rdr.shape[1])

    @property
    def dtype(self) -> np.dtype:
        return self.rdr.dtype

    def toarray(self) -> np.ndarray:
        out = np.zeros(self.shape, dtype=self.dtype)
        out[:, 0, :] = self.rdr.toarray()
        out[:, 1, :] = self.baf.toarray()
        return out

    def rows(self, index: Any) -> SpatioGenomicPairCounts:
        return SpatioGenomicPairCounts(self.rdr.rows(index), self.baf.rows(index))

    def with_rdr(self, rdr: Any) -> SpatioGenomicPairCounts:
        """Replace channel 0 by `rdr` in this backing, cast to this dtype as assignment into the dense array casts.

        Args: `rdr` `(n_obs, n_spots)`.
        Returns: a new `SpatioGenomicPairCounts`.
        """
        rdr = SpatioGenomicCounts.of(rdr)
        return SpatioGenomicPairCounts(self.baf.like(rdr.astype(self.dtype).values), self.baf)

    def without_rdr(self) -> SpatioGenomicPairCounts:
        return SpatioGenomicPairCounts(self.rdr.zeros(self.dtype), self.baf)

    def spot_sums(self, idx: Any = None) -> np.ndarray:
        """Compute `np.sum(single_X[:, :, idx], axis=-1)`.

        Args: `idx` spot indices or None.
        Returns: the `(n_obs, 2)` sums.
        """
        return np.stack([self.rdr.spot_sums(idx), self.baf.spot_sums(idx)], axis=1)


class _BinnedCountsFields(NamedTuple):
    lengths: np.ndarray
    X: Any
    base_nb_mean: Any
    total_bb_RD: Any


class BinnedCounts(_BinnedCountsFields):
    """The counts a segmentation sums, per segment and spot; unpacks as `(lengths, X, base_nb_mean, total_bb_RD)`.

    `lengths` `(n_contigs,)` segments per contig; `X` `(n_segments, 2, n_spots)` a `SpatioGenomicPairCounts` or its
    dense array; `base_nb_mean` the `(n_segments, n_spots)` baseline expression, empty until it is set; `total_bb_RD`
    the `(n_segments, n_spots)` SNP-covering reads. Each a `SpatioGenomicCounts`, CSR or dense array. Validated at
    construction.

    Args: `lengths`, `X`, `base_nb_mean`, `total_bb_RD` as above.
    """

    __slots__ = ()

    def __new__(cls, lengths: Any, X: Any, base_nb_mean: Any, total_bb_RD: Any) -> Self:
        self = super().__new__(cls, lengths, X, base_nb_mean, total_bb_RD)
        self.validate()
        return self

    @property
    def n_segments(self) -> int:
        return self.X.shape[0]

    @property
    def n_spots(self) -> int:
        return self.X.shape[2]

    @property
    def transcript_counts(self) -> SpatioGenomicCounts:
        """Total gene expression UMIs per segment and spot.

        Returns: the `(n_segments, n_spots)` `SpatioGenomicCounts`.
        """
        return SpatioGenomicPairCounts.of(self.X).rdr

    @property
    def hap_counts(self) -> SpatioGenomicCounts:
        """Haplotype A (H0) counts per segment and spot.

        Returns: the `(n_segments, n_spots)` `SpatioGenomicCounts`.
        """
        return SpatioGenomicPairCounts.of(self.X).baf

    @property
    def alt_hap_counts(self) -> SpatioGenomicCounts:
        """Haplotype B (H1) counts per segment and spot.

        Returns: the `(n_segments, n_spots)` `SpatioGenomicCounts`.
        """
        return SpatioGenomicCounts.of(self.total_bb_RD) - self.hap_counts

    def baf(self, fill_value: float = np.nan) -> np.ndarray:
        """Per-spot BAF, dense: `hap_counts` over `total_bb_RD` (`SpatioGenomicCounts.ratio`).

        Args: `fill_value` for undefined ratios, where the depth is zero.
        Returns: the `(n_segments, n_spots)` BAF.
        """
        return self.hap_counts.ratio(self.total_bb_RD, fill_value)

    def rdr(self, fill_value: float = np.nan) -> np.ndarray:
        """Per-spot RDR, dense: `transcript_counts` over `base_nb_mean` (`SpatioGenomicCounts.ratio`).

        Args: `fill_value` for undefined ratios, where the baseline is zero.
        Returns: the `(n_segments, n_spots)` RDR.
        """
        return self.transcript_counts.ratio(self.base_nb_mean, fill_value)

    def validate(self) -> None:
        """Refuse lengths that do not sum to the segments, or a baseline or depth not `(n_segments, n_spots)`.

        Raises: `ValueError` naming the disagreement.
        """
        n_seg, n_spt = self.n_segments, self.n_spots
        if np.sum(self.lengths) != n_seg:
            raise ValueError(f"Lengths array sum ({np.sum(self.lengths)}) does not match X n_segments ({n_seg}).")
        if self.base_nb_mean.shape != (n_seg, n_spt):
            raise ValueError(f"base_nb_mean shape {self.base_nb_mean.shape} mismatch. Expected: ({n_seg}, {n_spt})")
        if self.total_bb_RD.shape != (n_seg, n_spt):
            raise ValueError(f"total_bb_RD shape {self.total_bb_RD.shape} mismatch. Expected: ({n_seg}, {n_spt})")


def as_dense_array(a: Any) -> Any:
    """Return `a` as the dense array a capture records.

    Args: `a` a `SpatioGenomicPairCounts` (to `(n_obs, 2, n_spots)`), `SpatioGenomicCounts`, CSR or array.
    Returns: the dense array.
    """
    if isinstance(a, (SpatioGenomicPairCounts, SpatioGenomicCounts)):
        return a.toarray()
    return a.toarray() if sp.issparse(a) else a


NPZ_CHANNELS = ("rdr", "baf")
"""`SpatioGenomicPairCounts`' channels, each written under `<name>__<channel>` in turn: read depth, then the BAF
numerator."""


def npz_entries(name: str, value: Any) -> dict[str, np.ndarray]:
    """`value` as the entries `binned_data.npz` carries it in, sparse where it is sparse.

    A sparse `SpatioGenomicPairCounts` goes as its two channels, `<name>__rdr` and `<name>__baf`, each in turn; a sparse
    `(n_obs, n_spots)` array as its CSR, `<name>__data`, `__indices`, `__indptr` and `__shape`. Anything dense goes
    as itself, under `name`: the entry CalicoST writes, and the one this repository wrote for every array before.

    Densifying a sparse array to write it costs `4 x n_obs x n_spots` float64s to hold and to compress -- 9.7 GiB
    at 1,848 bins and 175,000 spots -- against the `nnz` entries it has. The compressed file is not always smaller,
    since zlib carries a dense array's zeros cheaply; what the entries save is the holding and the compressing.

    Args: `name` the entry's name; `value` a `SpatioGenomicPairCounts`, `SpatioGenomicCounts`, sparse matrix or dense
    array.
    Returns: the entries, to pass to `utils.write_npz` as keyword arguments.
    """
    if isinstance(value, SpatioGenomicPairCounts):
        # NB dense, the file keeps CalicoST's own entry: one `(n_obs, 2, n_spots)` array under `name`.
        if not value.sparse:
            return {name: value.toarray()}
        return {k: v for c in NPZ_CHANNELS for k, v in npz_entries(f"{name}__{c}", getattr(value, c)).items()}
    values = value.values if isinstance(value, SpatioGenomicCounts) else value
    if not sp.issparse(values):
        return {name: np.asarray(values)}
    values = canonical(values)
    return {
        f"{name}__data": values.data,
        f"{name}__indices": values.indices,
        f"{name}__indptr": values.indptr,
        f"{name}__shape": np.asarray(values.shape, dtype=np.int64),
    }


def from_grouped_columns(
    cls: Any, counts: Any, columns: np.ndarray, groups: np.ndarray, n_groups: int, dtype: Any
) -> SpatioGenomicCounts:
    """The lazy `(n_groups, n_spots)` sums of integer `counts` columns: row `groups[k]` adds column `columns[k]`,
    a column listed twice in a group counting twice. One CSR product per block of `CHUNK_BINS` groups.

    Args:
        counts: `(n_spots, n_columns)` integer counts, CSR or dense.
        columns: each term's column.
        groups: each term's group.
        n_groups: the rows.
        dtype: the sums' dtype.

    Returns:
        The lazy `SpatioGenomicCounts`.
    """
    counts = sp.csr_matrix(counts)
    blocks = []
    for start in range(0, n_groups, get_chunk_bins()):
        stop = min(start + get_chunk_bins(), n_groups)
        terms = (groups >= start) & (groups < stop)
        block = delayed(get_grouped_block)(counts, columns[terms], groups[terms] - start, stop - start, dtype)
        blocks.append(get_lazy_block(block, (stop - start, counts.shape[0]), dtype))
    return cls(da.concatenate(blocks, axis=0) if blocks else get_lazy_block(sp.csr_matrix((0, 0)), (0, 0), dtype))


def get_segmented(
    counts: SpatioGenomicCounts, segments: np.ndarray, rows: np.ndarray, n_segments: int
) -> SpatioGenomicCounts:
    """The lazy sums of `counts` rows into a coarser segmentation, one block per `CHUNK_BINS` segments: integers in any
    order, floats only where each segment takes one row (a selection), so no float is summed.

    Args:
        counts: the fine `(n_fine, n_spots)` counts.
        segments: each term's segment.
        rows: each term's fine row.
        n_segments: the segmentation's rows.

    Returns:
        The `(n_segments, n_spots)` lazy `SpatioGenomicCounts`.
    """
    if counts.dtype.kind not in "iu" and np.bincount(segments, minlength=1).max(initial=0) > 1:
        raise TypeError("segmenting float counts into a segment of several rows: its order is not numpy's.")
    fine = delayed(canonical)(counts.get_row_blocks()) if counts.blocks is not None else counts.computed
    blocks = []
    for start in range(0, n_segments, get_chunk_bins()):
        stop = min(start + get_chunk_bins(), n_segments)
        terms = (segments >= start) & (segments < stop)
        block = delayed(get_segmented_block)(fine, segments[terms] - start, rows[terms], stop - start)
        blocks.append(get_lazy_block(block, (stop - start, counts.shape[1]), counts.dtype))
    return SpatioGenomicCounts(da.concatenate(blocks, axis=0))
