"""The quality cuts `load_input_data` and `run_cnamaste` make on the counts: genes and ranges read from the filter
files, and genes zeroed.

- `get_filter_genes`, `get_filter_ranges` -- the gene list and the genomic ranges to filter, as the files hold them.
- `zero_genes` -- counts with a gene mask zeroed, CSR or dense.
"""

import logging

import numpy as np
import pandas as pd
import scipy.sparse as sp

logger = logging.getLogger(__name__)


def get_filter_genes(filter_gene_file):
    return pd.read_csv(filter_gene_file, header=None)


def get_filter_ranges(filter_range_file):
    ranges = pd.read_csv(filter_range_file, header=None, sep="\t", names=["Chr", "Start", "End"])

    if "chr" in ranges.Chr.iloc[0]:
        ranges["Chr"] = [int(x[3:]) for x in ranges.Chr.to_numpy()]

    ranges.sort_values(by=["Chr", "Start"], inplace=True)

    return ranges


def zero_genes(counts, genes):
    """`counts` with the columns where `genes` is True set to 0, as the dense `counts[:, genes] = 0`.

    Args:
        counts: (spots, genes), CSR or dense.
        genes: bool mask per gene.

    Returns:
        The zeroed counts; a CSR result keeps no explicit zero.
    """
    if not sp.issparse(counts):
        out = counts.copy()
        out[:, genes] = 0
        return out
    out = sp.csr_matrix(
        counts @ sp.diags((~np.asarray(genes)).astype(counts.dtype), dtype=counts.dtype), dtype=counts.dtype
    )
    out.eliminate_zeros()
    return out


def downsample_genes(counts, cap, seed=0):
    """`counts` with each gene whose total exceeds `cap` thinned binomially per spot at `cap / total`.

    A gene's expected total becomes `cap`; genes at or below it are unchanged. The draw is a local generator at `seed`,
    so the run's own random state is not consumed.

    Args:
        counts: (spots, genes), CSR or dense, integer.
        cap: the largest expected gene total.

    Returns:
        The thinned counts, in `counts`' form; a CSR result keeps no explicit zero.
    """
    rng = np.random.default_rng(seed)
    totals = np.asarray(counts.sum(axis=0), dtype=np.float64).ravel()
    rate = np.where(totals > cap, cap / np.maximum(totals, 1.0), 1.0)
    if not sp.issparse(counts):
        out = np.asarray(counts).copy()
        cols = np.flatnonzero(rate < 1.0)
        out[:, cols] = rng.binomial(out[:, cols].astype(np.int64), rate[cols]).astype(out.dtype)
        return out
    out = sp.csr_matrix(counts, copy=True)
    thin = rate[out.indices] < 1.0
    out.data[thin] = rng.binomial(out.data[thin].astype(np.int64), rate[out.indices[thin]]).astype(out.dtype)
    out.eliminate_zeros()
    return out
