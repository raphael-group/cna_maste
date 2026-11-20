# ruff: noqa: F541, F841
"""Frozen copies of `cnamaste.omics`' implementations PR- #62 held, verbatim, which T- #84's count containers replace;
`tests/legacy/test_legacy.py` holds the new code byte-equal to them."""

import numpy as np
import pandas as pd
import scipy.sparse
from numba import njit, prange

from cnamaste.omics import logger
from legacy.spatio_genomic_counts import BinnedCounts, SpatioGenomicCounts, SpatioGenomicPairCounts, canonical


@njit(parallel=True, cache=True)
def _column_sums(counts, columns, groups, n_groups):
    """Sum integer `counts[:, columns[k]]` into row `groups[k]`, exact in any order.

    Args: `counts` (n_spots, n_columns) integers; `columns`, `groups` per-term column and group; `n_groups`.
    Returns: the (n_groups, n_spots) sums.
    """
    out = np.zeros((n_groups, counts.shape[0]), dtype=counts.dtype)
    for i in prange(counts.shape[0]):
        for k in range(len(columns)):
            out[groups[k], i] += counts[i, columns[k]]
    return out


def grouped_column_sums(counts, columns, groups, n_groups, sparse=False):
    """Sum column `columns[k]` of integer `counts` into row `groups[k]`, for CSR or dense `counts`.

    CSR as one sparse product with the (column, group) incidence, dense by `_column_sums`; integer, so both agree.

    Args: `counts` (n_spots, n_columns) CSR or dense; `columns`, `groups`, `n_groups`; `sparse` returns CSR.
    Returns: the (n_groups, n_spots) sums, CSR where `sparse`, else dense.
    """

    if not scipy.sparse.issparse(counts):
        out = _column_sums(np.ascontiguousarray(counts), columns, groups, n_groups)
        return canonical(out) if sparse else out
    incidence = scipy.sparse.csr_matrix(
        (np.ones(len(columns), dtype=counts.dtype), (columns, groups)), shape=(counts.shape[1], n_groups)
    )
    product = (scipy.sparse.csr_matrix(counts) @ incidence).T
    return canonical(product) if sparse else np.ascontiguousarray(product.toarray())


def summarize_counts_for_blocks(
    df_gene_snp,
    adata,
    cell_snp_Aallele,
    cell_snp_Ballele,
    unique_snp_ids,
):
    """Aggregate gene-level total UMIs and SNP-level A/B allele counts into blocks.

    Args: `df_gene_snp` rows with `block_id`; `adata` counts; `cell_snp_*allele` (n_spots, n_snps); `unique_snp_ids`.
    Returns: a `BinnedCounts` of block lengths, `single_X`, `single_base_nb_mean`, `single_total_bb_RD`.
    """
    logger.info(f"Aggregating (snp, umi) counts for genome segmentation.")

    # NB each block's sites and genes as (column, block) pairs, the sums one pass each (`_column_sums`): the counts are
    #    integers, so bitwise the per-block loop this replaces (the replayed stage). A site listed twice in a block
    #    counts twice, as its index did; a gene counts once per block, as `np.isin` did.
    blocks = df_gene_snp.block_id.unique()
    n_blocks = len(blocks)
    n_spots = adata.shape[0]
    width = int(blocks.max()) + 1

    df_snps = df_gene_snp[df_gene_snp.snp_id.notna()]
    snp_idx = pd.Index(unique_snp_ids).get_indexer(df_snps.snp_id.to_numpy())
    assert np.all(snp_idx >= 0), "a site absent from unique_snp_ids"
    snp_block = df_snps.block_id.to_numpy().astype(np.int64)

    gene_names = adata.var.index.to_numpy()
    columns_of = {}
    for k, g in enumerate(gene_names):
        columns_of.setdefault(g, []).append(k)
    df_genes = df_gene_snp[df_gene_snp.is_interval == True][["gene", "block_id"]].drop_duplicates()
    pairs = np.array(
        [(c, b) for g, b in zip(df_genes.gene, df_genes.block_id) for c in columns_of.get(g, [])], dtype=np.int64
    ).reshape(-1, 2)

    # NB 0 is total umis;  1 index is haplotype 0 counts at each site. Row b of each is block b's sums, a row no block
    #    names left empty (`spatio_genomic_counts`): CSR, int64, as the dense int arrays were.
    a_sums = grouped_column_sums(cell_snp_Aallele, snp_idx.astype(np.int64), snp_block, width, sparse=True)
    b_sums = grouped_column_sums(cell_snp_Ballele, snp_idx.astype(np.int64), snp_block, width, sparse=True)
    gene_sums = grouped_column_sums(adata.layers["count"], pairs[:, 0].copy(), pairs[:, 1].copy(), width, sparse=True)

    # NB the selection of rows `blocks` into rows `blocks` of an (n_blocks, n_spots) array, as indexed assignment did.
    select = scipy.sparse.csr_matrix((np.ones(n_blocks, dtype=np.int64), (blocks, blocks)), shape=(n_blocks, width))
    single_X = SpatioGenomicPairCounts(canonical(select @ gene_sums, np.int64), canonical(select @ a_sums, np.int64))
    single_total_bb_RD = SpatioGenomicCounts(canonical(select @ (a_sums + b_sums), np.int64))
    single_base_nb_mean = single_total_bb_RD.zeros()

    # NB list of (unique) blocks grouped by contig.
    lengths = df_gene_snp.groupby("CHR")["block_id"].nunique().to_numpy()

    assert single_X.ndim == 3

    return BinnedCounts(lengths, single_X, single_base_nb_mean, single_total_bb_RD)


def summarize_counts_for_bins(
    df_gene_snp,
    adata,
    single_X,
    single_total_bb_RD,
    phase_indicator,
):
    """Aggregate block counts into the bins of `df_gene_snp`'s `bin_id`, phased by `phase_indicator`.

    Args: `df_gene_snp` with `block_id`, `bin_id`; `single_X` (n_blocks, 2, n_spots); `phase_indicator` per block.
    Returns: a `BinnedCounts` of bins per chromosome, `single_X`, `single_base_nb_mean`, `single_total_bb_RD`.
    """
    logger.info(f"Summarizing counts for bins.")

    has_assigned_bin = ~df_gene_snp.bin_id.isnull()

    bins = df_gene_snp.loc[has_assigned_bin, "bin_id"].unique()

    # NB last axis is the number of spot (barcodes).
    n_bins = len(bins)
    n_spots = adata.shape[0]

    logger.info(f"Retaining {100.0 * np.mean(has_assigned_bin):.2f}% of gene/snps with assigned bin.")

    # NB unique block ids and gene names per bin.
    df_bin_contents = df_gene_snp[has_assigned_bin].groupby("bin_id", sort=True).agg({"block_id": set, "gene": set})

    if df_bin_contents.index.isna().any():
        logger.warning(f"Found ill-defined group with None entries for group.")

    block_sets = df_bin_contents["block_id"].to_numpy()
    gene_sets = df_bin_contents["gene"].to_numpy()

    gene_names = adata.var.index.to_numpy()
    gene_index_map = {g: i for i, g in enumerate(gene_names)}
    count_matrix = adata.layers["count"]  # (n_spots, n_genes), sparse or dense.

    # NB bin sums as incidence products: integer, exact in any order; B counts phased per block; rows by sorted bin id.
    single_X, single_total_bb_RD = SpatioGenomicPairCounts.of(single_X), SpatioGenomicCounts.of(single_total_bb_RD)
    n_rows = single_X.shape[0]

    block_pairs = [(b, int(x)) for b in range(df_bin_contents.shape[0]) for x in block_sets[b] if x is not None]
    block_bins, block_ids = (
        (np.array(v, dtype=np.int64) for v in zip(*block_pairs)) if block_pairs else (np.zeros(0, np.int64),) * 2
    )
    by_block = scipy.sparse.csr_matrix(
        (np.ones(len(block_ids), dtype=np.int64), (block_bins, block_ids)), shape=(n_bins, n_rows)
    )

    used = np.unique(block_ids)
    h0 = np.zeros(n_rows, dtype=bool)
    h0[used] = np.asarray(phase_indicator)[used].astype(bool)
    keep = scipy.sparse.diags(h0.astype(np.int64), format="csr", dtype=np.int64)
    flip = scipy.sparse.diags((~h0).astype(np.int64), format="csr", dtype=np.int64)
    phased = single_X.baf.product(keep) + (single_total_bb_RD - single_X.baf).product(flip)

    gene_pairs = [
        (gene_index_map[g], b)
        for b in range(df_bin_contents.shape[0])
        for g in gene_sets[b]
        if g is not None and g in gene_index_map
    ]
    for b in range(df_bin_contents.shape[0]):
        if not any(g is not None for g in gene_sets[b]):
            logger.debug(f"No genes found for bin row {b}.")
    gene_cols, gene_bins = (
        (np.array(v, dtype=np.int64) for v in zip(*gene_pairs)) if gene_pairs else (np.zeros(0, np.int64),) * 2
    )
    by_gene = scipy.sparse.csr_matrix(
        (np.ones(len(gene_cols), dtype=count_matrix.dtype), (gene_cols, gene_bins)),
        shape=(count_matrix.shape[1], n_bins),
    )
    gene_sums = (scipy.sparse.csr_matrix(count_matrix) @ by_gene).T

    # NB int64, as the dense arrays the loop filled; the unset baseline float64 zeros; each in the input's backing.
    bin_single_X = SpatioGenomicPairCounts(
        single_X.baf.like(canonical(gene_sums, np.int64)), phased.product(by_block).astype(np.int64)
    )
    bin_single_total_bb_RD = single_total_bb_RD.product(by_block).astype(np.int64)
    bin_single_base_nb_mean = bin_single_total_bb_RD.zeros()

    chr_order = df_gene_snp.CHR.unique()
    lengths = (
        df_gene_snp.loc[has_assigned_bin].groupby("CHR")["bin_id"].nunique().reindex(chr_order, fill_value=0).to_numpy()
    )

    assert bin_single_X.ndim == 3

    return BinnedCounts(lengths, bin_single_X, bin_single_base_nb_mean, bin_single_total_bb_RD)
