"""The tests' `SpatioGenomicLabelledMultiCounts` from arrays."""

from __future__ import annotations

from typing import Any

import numpy as np

from cnamaste.spatio_genomic_counts import (
    SpatioGenomicCounts,
    SpatioGenomicLabelledMultiCounts,
    SpatioGenomicMultiCounts,
)


def get_labelled_counts(
    umis: Any, alt_snp_umis: Any, total_snp_umis: Any, lengths: Any = None, normal_spot: Any = None
) -> SpatioGenomicLabelledMultiCounts:
    """The three channels, dense or CSR, bin-major, as spot-major labelled counts.

    Args:
        umis: `(n_obs, n_spots)` gene UMIs.
        alt_snp_umis: `(n_obs, n_spots)` haplotype A SNP UMIs.
        total_snp_umis: `(n_obs, n_spots)` SNP UMIs.
        lengths: rows per contig, contigs 1, 2, ...; None is one contig.
        normal_spot: `(n_spots,)` bool, or None for no baseline.

    Returns:
        The counts, row `i` spanning `[i, i + 1)`.
    """
    channels = SpatioGenomicMultiCounts(*(SpatioGenomicCounts.of(c.T) for c in (umis, alt_snp_umis, total_snp_umis)))
    n_rows = channels.umis.shape[1]
    lengths = np.array([n_rows]) if lengths is None else np.asarray(lengths)
    contig = np.repeat(np.arange(1, len(lengths) + 1), lengths)
    counts = channels.get_labelled(contig, np.arange(n_rows), np.arange(1, n_rows + 1))
    return counts if normal_spot is None else counts.with_labelling("normal_spot", np.asarray(normal_spot, dtype=bool))
