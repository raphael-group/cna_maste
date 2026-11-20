"""Per-spot arrays `(n_spots, n_bins)`, dense or CSR: `SpatioGenomicCounts`, and the containers of its channels.

`SpatioGenomicCounts` is one per-spot quantity, spot-major `(n_spots, n_bins)`: a read depth, an allele count, a baseline.

A run builds them as canonical CSR: `get_grouped_sums`, a product with a 0/1 incidence, sums gene or SNP columns into
bins (`from_grouped_columns`) and bins into coarser segments (`get_segmented`).
`SpatioGenomicMultiCounts` holds a segmentation's three channels, `umis`, `alt_snp_umis` and `total_snp_umis`, with the
views `get_segmented` (the counts according to a segmentation of these bins) and `get_phased` (the phasing flip).
`SpatioGenomicLabelledMultiCounts` adds each segment's `contig`, `start` and `end`, with the view `lengths` (segments
per contig), spot labellings and segmentations, the view `normal_baseline_sized`, the view `get_pseudobulk` (`PseudobulkCounts`, each label's summed counts) and encodings: per
label of a labelling, the bins with equal `(count, total)` share an equivalence class (`CountsEncoding`).

A capture records them bin-major `(n_bins, n_spots)`, and its replays hand them back spot-major.
Every operation computes the same bits on both backings: integers sum exactly in any order, a float sums in the
bin-major array's order (segments in sequence, spots pairwise), and maps keep zero at zero. The fused map-sums a kernel
folds over a row are `cnamaste.sparse`'s.
"""

from __future__ import annotations

import itertools
from functools import cached_property
from typing import Any, NamedTuple

import numpy as np
import scipy.sparse as sp
from numba import njit

from cnamaste.pseudobulk import log_pseudobulk
from cnamaste.segments import Segmentation, get_contig_lengths


def canonical(m: Any, dtype: Any = None) -> sp.csr_matrix:
    """Return `m` as canonical CSR, with no explicit zeros, cast to `dtype` where given.

    Args:
        m: a sparse matrix or dense array.
        dtype: optional.

    Returns:
        The canonical CSR.
    """
    out = sp.csr_matrix(m, dtype=dtype if dtype is not None else m.dtype)
    out.eliminate_zeros()
    out.sort_indices()
    out.indptr, out.indices = out.indptr.astype(np.int64), out.indices.astype(np.int64)
    return out


def get_incidence(rows: Any, groups: Any, shape: tuple[int, int], dtype: Any) -> sp.csr_matrix:
    """Return the 0/1 incidence of `rows` in `groups`; a row listed twice in a group counts twice.

    Args:
        rows: each term's row.
        groups: each term's group (column).
        shape: `(n_rows, n_groups)`.
        dtype: the entries' dtype.

    Returns:
        The `(n_rows, n_groups)` CSR.
    """
    return sp.csr_matrix((np.ones(len(rows), dtype=dtype), (rows, groups)), shape=shape)


def get_grouped_sums(X: Any, incidence: sp.csr_matrix) -> sp.csr_matrix:
    """Return `X @ incidence`: each group's columns of `X` summed, exact for integers.

    Args:
        X: `(n, n_columns)` CSR or dense.
        incidence: `(n_columns, n_groups)` 0/1 (`get_incidence`).

    Returns:
        The `(n, n_groups)` canonical CSR, in the product's dtype.
    """
    return canonical(sp.csr_matrix(X) @ incidence)


def get_segment_incidence(segmentation: Segmentation, dtype: Any) -> sp.csr_matrix:
    """Return the incidence of a segmentation's rows in its segments (`get_incidence`).

    Args:
        segmentation: of `n_rows` rows.
        dtype: the entries' dtype.

    Returns:
        The `(n_rows, n_segments)` CSR.
    """
    rows, segments = segmentation.terms
    return get_incidence(rows, segments, (segmentation.label.size, segmentation.n_segments), dtype)


# --- the per-spot array -----------------------------------------------------------------------------------------------


class SpatioGenomicCounts:
    """One per-spot quantity `(n_spots, n_bins)`, dense or canonical CSR; every operation gives the same bits from either.

    Args:
        values: a dense array (made C-order) or sparse matrix (made canonical CSR).
    """

    ndim = 2

    def __init__(self, values: Any) -> None:
        self.values: Any = canonical(values) if sp.issparse(values) else np.ascontiguousarray(values)

    @classmethod
    def from_grouped_columns(cls, counts: Any, segmentation: Segmentation, dtype: Any) -> SpatioGenomicCounts:
        """The `(n_spots, n_segments)` sums of integer `counts` columns, each column in its segment (`get_grouped_sums`).

        Args:
            counts: `(n_spots, n_columns)` integer counts, CSR or dense.
            segmentation: of the `n_columns` columns.
            dtype: the sums' dtype.

        Returns:
            The CSR `SpatioGenomicCounts`.
        """
        counts = sp.csr_matrix(counts)
        return cls(canonical(get_grouped_sums(counts, get_segment_incidence(segmentation, counts.dtype)), dtype))

    @classmethod
    def of(cls, a: Any) -> SpatioGenomicCounts:
        return a if isinstance(a, cls) else cls(a)

    @property
    def sparse(self) -> bool:
        return sp.issparse(self.values)

    def get_held_values_at_stored_entries(self, other: SpatioGenomicCounts) -> np.ndarray:
        """`other`'s values at this CSR's stored entries (`sparse.values_at_stored_entries`), held for the last `other`
        read: the field reads one depth's B counts once per counts, not once per labelling.

        Args:
            other: `(n_spots, n_bins)` counts, dense or CSR, not modified in place.

        Returns:
            The values, one per stored entry.
        """
        from cnamaste.sparse import values_at_stored_entries

        held = self.__dict__.get("held_values_at_stored_entries")
        if held is None or held[0] is not other:
            held = (other, values_at_stored_entries(self.values, other.values))
            self.__dict__["held_values_at_stored_entries"] = held
        return held[1]

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.values.shape[0]), int(self.values.shape[1]))

    @property
    def dtype(self) -> np.dtype:
        return self.values.dtype

    def toarray(self) -> np.ndarray:
        return self.values.toarray() if self.sparse else self.values.copy()

    def get_bin_major(self) -> Any:
        """Return the bin-major values, `(n_bins, n_spots)`: canonical CSR, or a C-order dense copy.

        Returns:
            The transposed values.
        """
        return canonical(self.values.T) if self.sparse else np.ascontiguousarray(self.values.T)

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

    def spots(self, index: Any) -> SpatioGenomicCounts:
        return SpatioGenomicCounts(self.values[index, :])

    def segments(self, index: Any) -> SpatioGenomicCounts:
        return SpatioGenomicCounts(self.values[:, index])

    def zero_segments(self, index: Any) -> SpatioGenomicCounts:
        """Set segments (columns) `index` to 0, in a new array.

        Args:
            index: the segments, indices or a bool mask.

        Returns:
            A `SpatioGenomicCounts`.
        """
        out = self.values.copy()
        if not self.sparse:
            out[:, index] = 0
            return SpatioGenomicCounts(out)
        zeroed = np.zeros(self.shape[1], dtype=bool)
        zeroed[index] = True
        out.data[zeroed[out.indices]] = 0
        return SpatioGenomicCounts(out)

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
        """Compute `np.sum(values, axis)`: integers exactly, a float in the bin-major order (`get_bin_major`).

        Args:
            axis: 0 over spots, 1 over segments, or None.

        Returns:
            The sum.
        """
        dtype = self.dtype if self.dtype.kind in "iub" else np.float64
        if self.dtype.kind in "iu":
            found = np.sum(self.values, axis=axis) if not self.sparse else self.values.sum(axis=axis, dtype=dtype)
            return found if axis is None else np.asarray(found).ravel()
        bin_major, bin_axis = self.get_bin_major(), None if axis is None else 1 - axis
        if not self.sparse:
            return np.sum(bin_major, axis=bin_axis)
        if axis is None:
            return bin_major.sum(dtype=dtype)
        return np.asarray(bin_major.sum(axis=bin_axis, dtype=dtype)).ravel()

    def spot_sums(self, idx: Any = None) -> np.ndarray:
        """Compute `np.sum(values[idx], axis=0)`: per segment, the spots `idx` summed; every spot where None.

        Args:
            idx: spot indices or None.

        Returns:
            The (n_bins,) sums.
        """
        if idx is None:
            return self.sum(axis=0)
        if not self.sparse:
            # NB the spots in sequence: numpy's order on the bin-major `values[:, idx]`, an F-order copy.
            return np.sum(self.values[idx], axis=0)
        return self.spots(idx).sum(axis=0)

    def mean(self) -> float:
        return float(self.sum()) / (self.shape[0] * self.shape[1])

    def count_positive(self, axis: int | None = None) -> Any:
        """Compute `(values > 0).sum(axis)`.

        Args:
            axis: 0 over spots, 1 over segments, or None.

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


class NormalBaselineSized(NamedTuple):
    """The normal baseline `(n_spots, n_segments)` held as its two factors: entry (i, j) is `spots_coverage[i] *
    rdr_normal[j]`, one product rounded once, the dense matrix's bits; the kernels read it without the matrix.

    Args:
        spots_coverage: `(n_spots,)` float64 each spot's `umis` total.
        rdr_normal: `(n_segments,)` float64 the normal spots' pooled `umis`, normalized.
    """

    spots_coverage: np.ndarray
    rdr_normal: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        return (self.spots_coverage.size, self.rdr_normal.size)

    def toarray(self) -> np.ndarray:
        return np.multiply.outer(self.spots_coverage, self.rdr_normal)

    def count_positive(self, axis: int | None = None) -> Any:
        """Compute `(toarray() > 0).sum(axis)`: an entry is positive where both factors are, the factors finite and
        non-negative.

        Args:
            axis: 0 over spots, 1 over segments, or None.

        Returns:
            The counts of positive entries.
        """
        spots, segments = self.spots_coverage > 0, self.rdr_normal > 0
        if axis is None:
            return int(np.count_nonzero(spots)) * int(np.count_nonzero(segments))
        if axis == 0:
            return np.where(segments, np.count_nonzero(spots), 0)
        return np.where(spots, np.count_nonzero(segments), 0)

    def is_zero(self) -> bool:
        return self.count_positive() == 0


def get_segmented(counts: SpatioGenomicCounts, segmentation: Segmentation) -> SpatioGenomicCounts:
    """The sums of `counts` bins into a coarser segmentation (`get_grouped_sums`): integers in any order, floats only
    where each segment takes one bin (a selection), so no float is summed.

    Args:
        counts: the fine `(n_spots, n_fine)` counts.
        segmentation: of the `n_fine` bins.

    Returns:
        The `(n_spots, n_segments)` CSR `SpatioGenomicCounts`.
    """
    _, segments = segmentation.terms
    if counts.dtype.kind not in "iu" and np.bincount(segments, minlength=1).max(initial=0) > 1:
        raise TypeError("segmenting float counts into a segment of several bins: its order is not numpy's.")
    return SpatioGenomicCounts(get_grouped_sums(counts.values, get_segment_incidence(segmentation, counts.dtype)))


class SpatioGenomicMultiCounts(NamedTuple):
    """A segmentation's counts per segment and spot: total UMIs, the alternate (haplotype A) SNP UMIs and the total SNP
    UMIs, each a `SpatioGenomicCounts` of one shape.

    Args:
        umis: `(n_spots, n_segments)` gene UMIs.
        alt_snp_umis: `(n_spots, n_segments)` haplotype A SNP UMIs.
        total_snp_umis: `(n_spots, n_segments)` SNP UMIs of both haplotypes.
    """

    umis: SpatioGenomicCounts
    alt_snp_umis: SpatioGenomicCounts
    total_snp_umis: SpatioGenomicCounts

    def get_phased(self, keep: np.ndarray) -> SpatioGenomicMultiCounts:
        """The phasing flip: `alt_snp_umis` kept where `keep`, else `total_snp_umis - alt_snp_umis`, a mask per segment.

        Args:
            keep: `(n_segments,)` bool.

        Returns:
            The phased counts, `alt_snp_umis` int64 CSR.
        """
        keep = np.asarray(keep, dtype=bool)
        flipped = SpatioGenomicCounts(self.total_snp_umis.values - self.alt_snp_umis.values)
        phased = self.alt_snp_umis.zero_segments(~keep).values + flipped.zero_segments(keep).values
        return self._replace(alt_snp_umis=SpatioGenomicCounts(canonical(phased, np.int64)))

    def get_segmented(self, segmentation: Segmentation) -> SpatioGenomicMultiCounts:
        """Each channel summed into a coarser segmentation (`get_segmented`).

        Args:
            segmentation: of these rows.

        Returns:
            The segmented counts.
        """
        return SpatioGenomicMultiCounts(*(get_segmented(c, segmentation) for c in self))

    def get_labelled(self, contig: Any, start: Any, end: Any) -> SpatioGenomicLabelledMultiCounts:
        """The counts with no labelling or segmentation yet, so their `normal_baseline_sized` is zero.

        Args:
            contig: `(n_segments,)` each segment's contig, the outputs' `CHR`.
            start: `(n_segments,)` each segment's first base, the outputs' `START`.
            end: `(n_segments,)` each segment's end, the outputs' `END`.

        Returns:
            The `SpatioGenomicLabelledMultiCounts`.
        """
        labels = (np.asarray(v, dtype=np.int64) for v in (contig, start, end))
        return SpatioGenomicLabelledMultiCounts(*self, *labels, {}, {}, {})


ENCODING_CHANNELS = {
    "rdr": ("umis", "normal_baseline_sized"),
    "baf": ("alt_snp_umis", "total_snp_umis"),
    "labelling_unary": ("umis", "alt_snp_umis", "total_snp_umis"),
}
"""Each encoding key's channels: `(count, total)`, or the counts the labelling unary's tables read."""


def get_encoding_channels() -> dict[str, tuple[str, ...]]:
    """Return the encoding channels, `ENCODING_CHANNELS`."""
    return ENCODING_CHANNELS


def get_encoded_pairs(counts: Any, totals: Any, common_zero_depth: bool) -> tuple[np.ndarray, np.ndarray]:
    """`(counts, totals)` as encoded: a zero total's count set to 0 where `common_zero_depth`; otherwise as given.

    Non-integer pairs are not rounded; the M step's objective scores these (`hmm_mstep`).

    Args:
        counts: `(n_bins, n_labels)` counts.
        totals: `(n_bins, n_labels)` totals.
        common_zero_depth: zero the counts at zero total.

    Returns:
        Copies `(counts, totals)`, `(n_bins, n_labels)` each.
    """
    counts = np.array(counts, copy=True)
    totals = np.array(totals, copy=True)

    # NB all entries with 0 total counts collapse into a single (0, 0) state.
    if common_zero_depth:
        zero_mask = totals == 0
        counts[zero_mask] = 0
        totals[zero_mask] = 0

    return counts, totals


class CountsEncoding(NamedTuple):
    """One class of equivalence labels: per label, its unique `(count, total)` pairs in `np.unique` order, and each
    bin's class, an index into the labels' pairs concatenated.

    Args:
        labelling: the labelling it was built within, or None for a pseudobulk's own columns.
        labels: that labelling's `(n_spots,)` labels when built, or None.
        channels: the `(count, total)` channels, or `labelling_unary`'s three.
        unique_counts: per label, its unique counts.
        unique_totals: per label, its unique totals.
        equivalence_class: `(n_labels, n_bins)` each bin's class.
    """

    labelling: str | None
    labels: np.ndarray | None
    channels: tuple[str, ...]
    unique_counts: list[np.ndarray]
    unique_totals: list[np.ndarray]
    equivalence_class: np.ndarray


def get_counts_encoding(counts: Any, totals: Any, channels: tuple[str, ...], common_zero_depth: bool) -> CountsEncoding:
    """Each label's (row's) unique `(count, total)` pairs, as `np.unique` orders them, and each bin's class.

    Args:
        counts: `(n_labels, n_bins)` counts.
        totals: `(n_labels, n_bins)` totals.
        channels: the channels' names.
        common_zero_depth: as `get_encoded_pairs`.

    Returns:
        The `CountsEncoding` of the rows.
    """
    counts, totals = get_encoded_pairs(counts, totals, common_zero_depth)
    unique_counts: list[np.ndarray] = []
    unique_totals: list[np.ndarray] = []
    inverses: list[np.ndarray] = []

    for s in range(counts.shape[0]):
        pairs, inverse = np.unique(np.vstack([counts[s], totals[s]]).T, axis=0, return_inverse=True)
        unique_counts.append(pairs[:, 0])
        unique_totals.append(pairs[:, 1])
        inverses.append(inverse.reshape(-1).astype(np.int32))

    offsets = np.cumsum([0] + [len(u) for u in unique_counts], dtype=np.int64)
    equivalence_class = np.zeros(counts.shape, dtype=np.int64)
    for s, inverse in enumerate(inverses):
        equivalence_class[s] = inverse + offsets[s]
    return CountsEncoding(None, None, channels, unique_counts, unique_totals, equivalence_class)


def get_labelling_unary_encoding(counts: tuple[Any, Any, Any], channels: tuple[str, ...]) -> CountsEncoding:
    """The `labelling_unary` encoding: one label, the integer counts the labelling unary's `lgamma` tables read.

    A count j is read as a read count, a B count, a depth, an A count `n - b` where `b <= n`, and as 0 at a switched
    segment's absent depth.

    Args:
        counts: the read counts, B counts and depths the unary reads, non-negative integers.
        channels: the channels' names.

    Returns:
        The `CountsEncoding`: the read counts ascending as both `unique_counts` and `unique_totals`, and each count j's
        class, `(1, 1 + max j)`, -1 where unread.
    """
    umis, b_counts, totals = (np.asarray(v).ravel() for v in counts)
    read = [v.astype(np.int64) for v in (umis, b_counts, totals, (totals - b_counts)[b_counts <= totals])]
    realized = np.zeros(1 + max(int(v.max(initial=0)) for v in read), dtype=np.bool_)
    realized[0] = True
    for values in read:
        realized[values] = True
    distinct = np.flatnonzero(realized)
    equivalence_class = np.full((1, realized.size), -1, dtype=np.int32)
    equivalence_class[0, distinct] = np.arange(distinct.size, dtype=np.int32)
    return CountsEncoding(None, None, channels, [distinct], [distinct], equivalence_class)


class EncodedCounts:
    """The methods on an encoding, `get_encoding`'s by its key: the unique pairs, the classes and the decode."""

    def get_encoding(self, encoding: str | None) -> CountsEncoding:
        """The encoding `encoding`, or the one held where None; each holder its own.

        Args:
            encoding: the key, or None.

        Returns:
            The `CountsEncoding`.
        """
        raise NotImplementedError

    def get_unique(self, encoding: str | None = None) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Each label's unique `(count, total)` pairs.

        Args:
            encoding: the key; None for the one encoding held.

        Returns:
            Per label, the unique counts, and per label, the unique totals.
        """
        found = self.get_encoding(encoding)
        return found.unique_counts, found.unique_totals

    def get_equivalence_class(self, encoding: str | None = None) -> np.ndarray:
        """Each bin's class, an index into the labels' unique pairs concatenated.

        Args:
            encoding: the key; None for the one encoding held.

        Returns:
            The `(n_labels, n_bins)` classes.
        """
        return self.get_encoding(encoding).equivalence_class

    def decode(self, values: np.ndarray, axis: int, encoding: str | None = None) -> np.ndarray:
        """Gather `values` over the labels' unique pairs, concatenated, to every bin of every label: one `np.take`.

        Args:
            values: `n_unique` entries along `axis`, the labels' unique pairs in turn.
            axis: the unique-pair axis.
            encoding: the key; None for the one encoding held.

        Returns:
            `values` with `axis` replaced by `(n_labels, n_bins)`.
        """
        found = self.get_encoding(encoding)
        n_unique = sum(len(u) for u in found.unique_counts)
        if values.shape[axis] != n_unique:
            raise ValueError(f"values {values.shape} have no axis {axis} of {n_unique} unique pairs.")
        return np.take(values, found.equivalence_class, axis=axis)

    def compression_rate(self, encoding: str | None = None) -> float:
        """One less the unique pairs over the bins of every label; 0.0 for no bins.

        Args:
            encoding: the key; None for the one encoding held.

        Returns:
            The rate.
        """
        found = self.get_encoding(encoding)
        if found.equivalence_class.size == 0:
            return 0.0
        return 1.0 - float(sum(u.shape[0] for u in found.unique_counts)) / found.equivalence_class.size


def get_held_encoding(encodings: dict[str, CountsEncoding], encoding: str | None) -> CountsEncoding:
    """The encoding `encoding`, or the one held where None.

    Args:
        encodings: key -> encoding.
        encoding: the key, or None.

    Returns:
        The `CountsEncoding`.

    Raises:
        KeyError: on a key not held, or None where not exactly one is held.
    """
    if encoding is None:
        if len(encodings) != 1:
            raise KeyError(f"encoding None names the one encoding held, of {sorted(encodings)}.")
        return next(iter(encodings.values()))
    if encoding not in encodings:
        raise KeyError(f"no encoding {encoding!r}, of {sorted(encodings)}.")
    return encodings[encoding]


class PseudobulkCountsFields(NamedTuple):
    umis: Any
    normal_baseline_sized: Any
    alt_snp_umis: Any
    total_snp_umis: Any
    clone_lengths: np.ndarray | None
    encodings: dict[str, CountsEncoding]


class PseudobulkCounts(PseudobulkCountsFields, EncodedCounts):
    """Each label's summed counts per bin, clone-major `(n_labels, n_bins)`, or stacked along bins `(1, n_labels n_bins)`,
    where a label may be a clone or a single spot; a channel None is not scored, except on encoded counts, which score
    their encodings.

    Args:
        umis: gene UMIs.
        normal_baseline_sized: the normal baseline.
        alt_snp_umis: haplotype A SNP UMIs, or the minor count where folded.
        total_snp_umis: SNP UMIs of both haplotypes.
        clone_lengths: bins per clone where clones are stacked along bins; None for one clone, or by spot.
        encodings: key -> `CountsEncoding`, per column; `{}` scores the counts densely.
    """

    @property
    def X(self) -> np.ndarray:
        """The read-depth and B-allele counts as `run_cnamaste` holds them, `(n_bins, 2, n_labels)`."""
        return np.stack([np.transpose(self.umis), np.transpose(self.alt_snp_umis)], axis=1)

    def compute(self) -> tuple[Any, Any, Any, Any]:
        """The channels' arrays.

        Returns:
            `umis`, `normal_baseline_sized`, `alt_snp_umis` and `total_snp_umis`, a channel None kept None.
        """
        return tuple(c.values if isinstance(c, SpatioGenomicCounts) else c for c in self[:4])

    def get_encoding(self, encoding: str | None) -> CountsEncoding:
        """The encoding `encoding`, or the one held where None (`get_held_encoding`).

        Args:
            encoding: the key, or None.

        Returns:
            The `CountsEncoding`.
        """
        return get_held_encoding(self.encodings, encoding)

    def get_clone_lengths(self) -> np.ndarray:
        """Bins per clone, clone c the c-th block of bins; one block of every bin where `clone_lengths` is None.

        Returns:
            The `(n_clones,)` int64 lengths.
        """
        if self.clone_lengths is not None:
            return np.asarray(self.clone_lengths, dtype=np.int64).reshape(-1)
        held = [c for c in self[:4] if c is not None]
        n_bins = held[0].shape[1] if held else self.get_equivalence_class(next(iter(self.encodings))).shape[1]
        return np.array([n_bins], dtype=np.int64)

    def get_log_lambda(self) -> np.ndarray:
        """`log lambda_g`, `eqn:nb_emission`: each clone's normal baseline over its sum over bins; `-inf` at 0.

        Returns:
            The log fractions, the baseline's shape.
        """
        base = np.asarray(self.normal_baseline_sized, dtype=np.float64)
        bounds = np.concatenate([[0], np.cumsum(self.get_clone_lengths())])
        blocks = []
        for a, b in itertools.pairwise(bounds):
            with np.errstate(divide="ignore", invalid="ignore"):
                blocks.append(np.log(base[:, a:b] / base[:, a:b].sum(axis=1, keepdims=True)))
        return np.concatenate(blocks, axis=1)

    def update_encoding(self, key: str, channels: tuple[str, ...], *, common_zero_depth: bool) -> PseudobulkCounts:
        """Build or replace the encoding `key` of `channels`, `(count, total)`, per column; `labelling_unary` of its
        three channels, one label (`get_labelling_unary_encoding`).

        Args:
            key: the encoding, `rdr`, `baf` or `labelling_unary` (`ENCODING_CHANNELS`).
            channels: the `(count, total)` channels, or `labelling_unary`'s three.
            common_zero_depth: as `get_encoded_pairs`; `labelling_unary` reads every count as given.

        Returns:
            The counts.
        """
        if key == "labelling_unary":
            found = get_labelling_unary_encoding(tuple(getattr(self, c) for c in channels), channels)
        else:
            found = get_counts_encoding(
                getattr(self, channels[0]), getattr(self, channels[1]), channels, common_zero_depth
            )
        return self._replace(encodings=self.encodings | {key: found})

    def get_stacked(self) -> PseudobulkCounts:
        """The labels stacked along bins, `(1, n_labels * n_bins)`, `clone_lengths` `(n_labels,)`; no encoding.

        Returns:
            The stacked counts.
        """
        n_labels, n_bins = self.umis.shape
        stacked = (np.reshape(v, (1, -1)) for v in self[:4])
        return PseudobulkCounts(*stacked, np.full(n_labels, n_bins), {})


@njit(nogil=True, cache=True, error_model="numpy")
def get_sized_label_sums(
    spots_coverage: np.ndarray, rdr_normal: np.ndarray, labels: np.ndarray, n_labels: int
) -> np.ndarray:
    """Each label's rows of the sized baseline summed in spot order, each row `spots_coverage[spot] * rdr_normal`,
    bitwise `np.sum(toarray()[idx], axis=0)`, with no matrix.

    Args:
        spots_coverage: `(n_spots,)` float64.
        rdr_normal: `(n_segments,)` float64.
        labels: `(n_spots,)` int64 labels 0 to `n_labels - 1`, -1 dropped.
        n_labels: the labels.

    Returns:
        The `(n_labels, n_segments)` sums, a label with no spot zero.
    """
    sums = np.zeros((n_labels, rdr_normal.size))
    for spot in range(spots_coverage.size):
        label = labels[spot]
        if label >= 0:
            for segment in range(rdr_normal.size):
                sums[label, segment] += spots_coverage[spot] * rdr_normal[segment]
    return sums


def get_label_sums(counts: SpatioGenomicCounts | NormalBaselineSized, labels: np.ndarray, n_labels: int) -> np.ndarray:
    """Each label's spots summed per segment: integers as one product `incidence_labels×spots @ X`
    (`get_grouped_sums`), exact; the sized baseline in spot order (`get_sized_label_sums`); a float label by label in
    `spot_sums`' order.

    Args:
        counts: `(n_spots, n_segments)` counts.
        labels: `(n_spots,)` int64 labels 0 to `n_labels - 1`, -1 dropped.
        n_labels: the labels.

    Returns:
        The `(n_labels, n_segments)` float64 sums, a label with no spot zero.
    """
    if isinstance(counts, NormalBaselineSized):
        return get_sized_label_sums(counts.spots_coverage, counts.rdr_normal, labels, n_labels)
    if counts.dtype.kind in "iu":
        spots = np.flatnonzero(labels >= 0)
        incidence = get_incidence(spots, labels[spots], (counts.shape[0], n_labels), counts.dtype)
        # NB the spots on the left: `X.T @ incidence` would transpose the whole CSR per call.
        sums = incidence.T.tocsr() @ counts.values
        return (sums.toarray() if sp.issparse(sums) else np.asarray(sums)).astype(np.float64)
    sums = np.zeros((n_labels, counts.shape[1]))
    for label in range(n_labels):
        idx = np.flatnonzero(labels == label)
        if len(idx) > 0:
            sums[label] = counts.spot_sums(idx)
    return sums


def get_updated_label_sums(
    counts: SpatioGenomicCounts | NormalBaselineSized, sums: np.ndarray, labels: np.ndarray, previous: np.ndarray
) -> np.ndarray:
    """`get_label_sums(counts, labels, len(sums))` from `sums`, those of `previous`: integers by the moved spots' rows
    alone, one signed product `incidence_labels×moved @ X[moved]`, exact below 2**53; anything else in full.

    Args:
        counts: `(n_spots, n_segments)` counts.
        sums: `(n_labels, n_segments)` float64 `get_label_sums` of `previous`.
        labels: `(n_spots,)` int64 labels 0 to `n_labels - 1`, -1 dropped.
        previous: `(n_spots,)` int64 the labels `sums` was summed under.

    Returns:
        The `(n_labels, n_segments)` float64 sums, bitwise `get_label_sums`'.
    """
    n_labels = sums.shape[0]
    if isinstance(counts, NormalBaselineSized) or counts.dtype.kind not in "iu":
        return get_label_sums(counts, labels, n_labels)
    moved = np.flatnonzero(labels != previous)
    if moved.size == 0:
        return sums.copy()
    into, out = labels[moved], previous[moved]
    rows = np.concatenate([np.flatnonzero(into >= 0), np.flatnonzero(out >= 0)])
    columns = np.concatenate([into[into >= 0], out[out >= 0]])
    signs = np.concatenate([np.ones(np.count_nonzero(into >= 0)), -np.ones(np.count_nonzero(out >= 0))])
    incidence = sp.csr_matrix((signs.astype(np.int64), (columns, rows)), shape=(n_labels, moved.size))
    delta = incidence @ counts.values[moved]
    delta = delta.toarray() if sp.issparse(delta) else np.asarray(delta)
    return sums + delta.astype(np.float64)


class LabelledMultiCountsFields(NamedTuple):
    umis: SpatioGenomicCounts
    alt_snp_umis: SpatioGenomicCounts
    total_snp_umis: SpatioGenomicCounts
    contig: np.ndarray
    start: np.ndarray
    end: np.ndarray
    labellings: dict[str, np.ndarray]
    segmentations: dict[str, np.ndarray]
    encodings: dict[str, CountsEncoding]


class SpatioGenomicLabelledMultiCounts(LabelledMultiCountsFields, EncodedCounts):
    """A segmentation's three channels (`SpatioGenomicMultiCounts`) with their spot labellings, segmentations and
    encodings.

    Args:
        umis: `(n_spots, n_segments)` gene UMIs.
        alt_snp_umis: `(n_spots, n_segments)` haplotype A SNP UMIs.
        total_snp_umis: `(n_spots, n_segments)` SNP UMIs of both haplotypes.
        contig: `(n_segments,)` each segment's contig, the outputs' `CHR`.
        start: `(n_segments,)` each segment's first base, the outputs' `START`.
        end: `(n_segments,)` each segment's end, the outputs' `END`.
        labellings: name -> `(n_spots,)` a label per spot, -1 dropped; `normal_spot` is a bool labelling.
        segmentations: name -> `(n_segments,)` a coarser segment per segment, -1 dropped.
        encodings: key -> `CountsEncoding` of a labelling's pseudobulk (`update_encoding`).
    """

    @property
    def n_segments(self) -> int:
        return self.umis.shape[1]

    @property
    def n_spots(self) -> int:
        return self.umis.shape[0]

    @property
    def multi_counts(self) -> SpatioGenomicMultiCounts:
        return SpatioGenomicMultiCounts(self.umis, self.alt_snp_umis, self.total_snp_umis)

    @property
    def lengths(self) -> np.ndarray:
        """Segments per contig, in segment order (`get_contig_lengths` of `contig`).

        Returns:
            The `(n_contigs,)` int64 lengths.
        """
        return get_contig_lengths(self.contig)

    def get_segment_labels(self, segmentation: Segmentation) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The labels carried to `segmentation`'s coarser segments: a coarse segment takes the `contig` of its segments,
        the `start` of its first and the `end` of its last.

        Args:
            segmentation: of these segments.

        Returns:
            The coarse segments' `contig`, `start` and `end`.

        Raises:
            ValueError: where a coarse segment holds no segment or spans two contigs.
        """
        rows, segments = segmentation.terms
        first = np.full(segmentation.n_segments, self.n_segments, dtype=np.int64)
        last = np.full(segmentation.n_segments, -1, dtype=np.int64)
        np.minimum.at(first, segments, rows)
        np.maximum.at(last, segments, rows)
        if np.any(last < 0):
            raise ValueError(f"{segmentation.name}: {np.count_nonzero(last < 0)} segments hold no segment.")
        if np.any(self.contig[first] != self.contig[last]):
            raise ValueError(f"{segmentation.name}: a segment spans two contigs.")
        return self.contig[first], self.start[first], self.end[last]

    def with_umis(self, umis: Any) -> SpatioGenomicLabelledMultiCounts:
        """Replace `umis`, cast to their dtype as assignment casts, in `alt_snp_umis`' backing.

        Args:
            umis: `(n_spots, n_segments)` counts.

        Returns:
            The counts.
        """
        cast = SpatioGenomicCounts.of(umis).astype(self.umis.dtype)
        if cast.sparse != self.alt_snp_umis.sparse:
            cast = SpatioGenomicCounts(sp.csr_matrix(cast.values) if self.alt_snp_umis.sparse else cast.toarray())
        return self._replace(umis=cast)

    def with_labelling(self, name: str, labels: np.ndarray) -> SpatioGenomicLabelledMultiCounts:
        """Add or replace the spot labelling `name`.

        Args:
            name: the labelling, e.g. `normal_spot`.
            labels: `(n_spots,)` a label per spot.

        Returns:
            The counts.
        """
        return self._replace(labellings=self.labellings | {name: np.asarray(labels)})

    def _replace(self, **fields: Any) -> SpatioGenomicLabelledMultiCounts:
        """`NamedTuple._replace`, keeping a computed `normal_baseline_sized` where `umis` and the `normal_spot`
        labelling are the same objects, so a relabelling does not pool the normal spots again; the `labelling_unary`
        encoding is dropped where a channel it reads is not the same object.

        Args:
            fields: the fields replaced.

        Returns:
            The counts.
        """
        replaced = super()._replace(**fields)
        if any(getattr(replaced, c) is not getattr(self, c) for c in ENCODING_CHANNELS["labelling_unary"]):
            unread = {key: found for key, found in replaced.encodings.items() if key != "labelling_unary"}
            replaced = LabelledMultiCountsFields._replace(replaced, encodings=unread)
        baseline = self.__dict__.get("normal_baseline_sized")
        same_normal = replaced.labellings.get("normal_spot") is self.labellings.get("normal_spot")
        if baseline is not None and replaced.umis is self.umis and same_normal:
            replaced.__dict__["normal_baseline_sized"] = baseline
        return replaced

    @cached_property
    def normal_baseline_sized(self) -> SpatioGenomicCounts | NormalBaselineSized:
        """The normal spots' pooled `umis`, normalized, times each spot's `umis` total; zero without `normal_spot`.

        Returns:
            The `(n_spots, n_segments)` baseline as its factors, `NormalBaselineSized`; the zero `SpatioGenomicCounts`
            without `normal_spot`.
        """
        if "normal_spot" not in self.labellings:
            return self.total_snp_umis.zeros()
        rdr_normal = self.umis.spot_sums(self.labellings["normal_spot"] == True)
        rdr_normal = rdr_normal / np.sum(rdr_normal)
        spots_coverage = np.asarray(self.umis.sum(axis=1), dtype=np.float64)
        return NormalBaselineSized(spots_coverage, np.asarray(rdr_normal, dtype=np.float64))

    def get_spot_weights(self) -> np.ndarray:
        """Each spot's read-depth weight in the field: its segments with SNP UMIs over its segments with a baseline, 1
        where either has none.

        Returns:
            The `(n_spots,)` weights.
        """
        # NB the weight of `sandbox.spot_assignment.compute_loglike_spot_assignment_given_emission`.
        nb_valid = self.normal_baseline_sized.count_positive(axis=1)
        bb_valid = self.total_snp_umis.count_positive(axis=1)
        weights = np.ones(self.n_spots)
        both = (nb_valid > 0) & (bb_valid > 0)
        weights[both] = bb_valid[both] / nb_valid[both]
        return weights

    def get_pseudobulk(self, labelling: str, encoding: str | None = None) -> PseudobulkCounts:
        """Each label's spots summed per segment, clone-major `(n_labels, n_segments)` (`get_label_sums`); a label with
        no spot zero.

        Args:
            labelling: the labelling, labels 0 to `n_labels - 1`, -1 dropped.
            encoding: the key of an encoding built within `labelling`, attached; None for none.

        Returns:
            The `PseudobulkCounts`, `clone_lengths` None.

        Raises:
            ValueError: on an encoding built within another labelling, or one whose labelling changed since.
        """
        labels = np.asarray(self.labellings[labelling]).astype(np.int64)
        n_labels = int(np.max(labels, initial=-1)) + 1
        umis, base, alt_snp_umis, total_snp_umis = (
            get_label_sums(c, labels, n_labels)
            for c in (self.umis, self.normal_baseline_sized, self.alt_snp_umis, self.total_snp_umis)
        )
        pseudobulk = PseudobulkCounts(umis, base, alt_snp_umis, total_snp_umis, None, {})
        log_pseudobulk(pseudobulk, np.bincount(labels[labels >= 0], minlength=n_labels).tolist())
        if encoding is None:
            return pseudobulk
        found = self.get_encoding(encoding)
        if found.labelling != labelling:
            raise ValueError(f"encoding {encoding!r} is built within {found.labelling!r}, not {labelling!r}.")
        return pseudobulk._replace(encodings={encoding: found})

    def get_updated_pseudobulk(
        self, labelling: str, previous: PseudobulkCounts, previous_labels: np.ndarray
    ) -> PseudobulkCounts:
        """`get_pseudobulk(labelling)` bitwise, from `previous`, the pseudobulk of `previous_labels`: each channel by
        `get_updated_label_sums`; in full where the number of labels differs.

        Args:
            labelling: the labelling, labels 0 to `n_labels - 1`, -1 dropped.
            previous: `get_pseudobulk` under `previous_labels`.
            previous_labels: `(n_spots,)` the labels `previous` was summed under.

        Returns:
            The `PseudobulkCounts`, `clone_lengths` None, no encoding.
        """
        labels = np.asarray(self.labellings[labelling]).astype(np.int64)
        n_labels = int(np.max(labels, initial=-1)) + 1
        if previous.umis.shape[0] != n_labels:
            return self.get_pseudobulk(labelling)
        before = np.asarray(previous_labels).astype(np.int64)
        channels = (self.umis, self.normal_baseline_sized, self.alt_snp_umis, self.total_snp_umis)
        umis, base, alt_snp_umis, total_snp_umis = (
            get_updated_label_sums(c, sums, labels, before) for c, sums in zip(channels, previous[:4], strict=True)
        )
        pseudobulk = PseudobulkCounts(umis, base, alt_snp_umis, total_snp_umis, None, {})
        log_pseudobulk(pseudobulk, np.bincount(labels[labels >= 0], minlength=n_labels).tolist())
        return pseudobulk

    def get_encoding(self, encoding: str | None) -> CountsEncoding:
        """The encoding `encoding`, or the one held where None.

        Args:
            encoding: the key, or None.

        Returns:
            The `CountsEncoding`.

        Raises:
            ValueError: where its labelling changed since `update_encoding`.
        """
        found = get_held_encoding(self.encodings, encoding)
        if found.labelling is not None and not np.array_equal(self.labellings.get(found.labelling, []), found.labels):  # type: ignore[arg-type]
            raise ValueError(f"encoding {encoding!r}: labelling {found.labelling!r} changed since update_encoding.")
        return found

    def update_encoding(
        self, key: str, channels: tuple[str, str], *, labelling: str, common_zero_depth: bool
    ) -> SpatioGenomicLabelledMultiCounts:
        """Build or replace the encoding `key` of `channels`, `(count, total)`, within each label of `labelling`: the
        label's pseudobulk bins with equal pairs share a class.

        Args:
            key: the encoding, e.g. `rdr` or `baf` (`ENCODING_CHANNELS`).
            channels: the `(count, total)` channels.
            labelling: the labelling.
            common_zero_depth: as `get_encoded_pairs`.

        Returns:
            The counts.
        """
        pseudobulk = self.get_pseudobulk(labelling).update_encoding(key, channels, common_zero_depth=common_zero_depth)
        found = pseudobulk.encodings[key]._replace(labelling=labelling, labels=self.labellings[labelling].copy())
        return self._replace(encodings=self.encodings | {key: found})
