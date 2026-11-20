import logging

import numpy as np
import pandas as pd
import scipy.sparse

from cnamaste.config import start_time
from cnamaste.logger import get_logger
from cnamaste.reference import get_reference_genes
from cnamaste.segments import Segmentation
from cnamaste.spatio_genomic_counts import (
    SpatioGenomicCounts,
    SpatioGenomicMultiCounts,
    get_segmented,
)

logger = get_logger(__name__, start_time=start_time)


def greedy_binning_nobreak(
    block_lengths,
    block_umi,
    block_snp_umi,
    block_normal_umi,
    secondary_min_umi,
    secondary_min_snp_umi,
    secondary_min_normal_umi,
    max_binlength,
):
    """Group consecutive blocks into bins meeting minimum UMI, SNP-UMI and normal-UMI counts and a maximum length.

    Args: `block_lengths` per-block lengths; `block_*umi` per-block counts; `secondary_min_*` floors; `max_binlength`.

    Returns:
        The bin id of each block.
    """
    assert len(block_lengths) == len(block_umi) == len(block_snp_umi) == len(block_normal_umi), (
        f"Block array length mismatch: "
        f"lengths={len(block_lengths)}, "
        f"umi={len(block_umi)}, "
        f"snp_umi={len(block_snp_umi)}, "
        f"normal_umi={len(block_normal_umi)}"
    )

    # NB (start, end) indices of new bins that aggregate old blocks to
    #    meet umi, length, etc. requirements.
    bin_ranges = []
    s = 0

    while s < len(block_lengths):
        t = s + 1

        # NB extend included blocks until meets required umi count.
        while t < len(block_lengths):
            total_umi = np.sum(block_umi[s:t])
            snp_umi = np.sum(block_snp_umi[s:t])
            normal_umi = np.sum(block_normal_umi[s:t])
            length = np.sum(block_lengths[s:t])

            # NB check if all min requirements are met
            meets_umi = total_umi >= secondary_min_umi
            meets_snp = snp_umi >= secondary_min_snp_umi
            meets_normal = normal_umi >= secondary_min_normal_umi
            all_criteria_met = meets_umi and meets_snp and meets_normal

            # NB break if bin is too long but meets UMI requirements
            if length >= max_binlength and all_criteria_met:
                logger.warning(
                    f"Solved for bin length={length / max_binlength:>6.2f} [max_binlength] "
                    f"(umi={total_umi:>8}, snp-umi={snp_umi:>8}, normal-umi={normal_umi:>8})"
                )
                t = max(t - 1, s + 1)
                break

            # NB continue if criteria not met and not too long
            if all_criteria_met:
                break

            t += 1

        # NB final counts for bin [s:t]
        total_umi = np.sum(block_umi[s:t])
        snp_umi = np.sum(block_snp_umi[s:t])
        normal_umi = np.sum(block_normal_umi[s:t])
        length = np.sum(block_lengths[s:t])

        # NB check if it's a small bin at the end that doesn't meet criteria
        if s > 0 and t == len(block_lengths):
            if (
                total_umi < secondary_min_umi
                or snp_umi < secondary_min_snp_umi
                or normal_umi < secondary_min_normal_umi
            ):
                logger.debug(
                    f"Last bin failed thresholds "
                    f"(UMI={total_umi:>8}/{secondary_min_umi:<8}, "
                    f"SNP-UMI={snp_umi:>8}/{secondary_min_snp_umi:<8}, "
                    f"normal-UMI={normal_umi:>8}/{secondary_min_normal_umi:<8}), "
                    f"merging with previous."
                )
                bin_ranges[-1][1] = t
            else:
                bin_ranges.append([s, t])
        else:
            bin_ranges.append([s, t])

        s = t

    bin_ids = np.zeros(len(block_lengths), dtype=int)

    for i, x in enumerate(bin_ranges):
        bin_ids[x[0] : x[1]] = i

    # NB return new bin ids for each block, where new bins meet umi, length, etc. requirements.
    return bin_ids


# TODO assumes reference gene contains all those present in Visium anndata.
def form_gene_snp_table(
    unique_snp_ids,
    hgtable_file,
    adata,
    verbose=False,
    num_preceeding_rows=100,  # MAGIC
):
    logger.info(f"Forming gene & snp meta data.")

    # NB includes both gene and SNP info: CHR, START, END, snp_id, gene, is_interval
    df_gene = get_reference_genes(hgtable_file)

    logger.info(f"Filtering reference genes to those in visium.")

    common_genes = set(df_gene.gene) & set(adata.var.index)
    genes_not_in_reference = set(adata.var.index) - common_genes

    logger.info(f"Found {100.0 * len(common_genes) / len(adata.var.index):.2f}% of visium genes to be in reference.")

    # NB enriched for sex-chromosome and mitochondrial genes, many lncRNAs/antisense/pseudogenes, and CT antigens commonly over-expressed in tumors.
    logger.info(f"Found {len(genes_not_in_reference):_} genes to be in visium but not in reference:")

    genes_sorted = sorted(genes_not_in_reference)

    if verbose:
        for i in range(0, len(genes_sorted), 10):
            chunk = genes_sorted[i : i + 10]
            logger.info(", ".join(chunk))

    # NB limits reference genes to those present in (filtered) AnnData UMIs.
    df_gene = df_gene[df_gene.gene.isin(adata.var.index)]

    # NB add SNP info: {contig}_{pos}_{ref}_{alt}.
    snp_chr = np.array([int(x.split("_")[0]) for x in unique_snp_ids])
    snp_pos = np.array([int(x.split("_")[1]) for x in unique_snp_ids])
    snp_end = snp_pos + 1

    # NB vertical concatenation
    df_gene_snp = pd.concat(
        [
            df_gene,
            pd.DataFrame(
                {
                    "CHR": snp_chr,
                    "START": snp_pos,
                    "END": snp_end,
                    "snp_id": unique_snp_ids,
                    "gene": None,
                    "is_interval": False,
                }
            ),
        ],
        ignore_index=True,
    )

    logger.debug(f"Sorting df_gene_snp")

    df_gene_snp.sort_values(by=["CHR", "START"], inplace=True)

    logger.debug(f"Assigning genes to SNPs")

    vec_is_interval = df_gene_snp.is_interval.to_numpy()

    vec_chr = df_gene_snp.CHR.to_numpy()
    vec_start = df_gene_snp.START.to_numpy()
    vec_end = df_gene_snp.END.to_numpy()

    # NB loops over sites, on arrays, the column written once: a gene row's name is read and never written, so the
    #    order of assignment does not matter (bitwise the per-row `iloc` form, the replayed stage).
    vec_gene = df_gene_snp.gene.to_numpy(dtype=object, copy=True)

    for i in np.where(df_gene_snp.gene.isnull())[0]:
        # TODO first SNP has no gene.
        if i == 0:
            continue

        this_pos = vec_start[i]

        # NB look for an overlapping gene, closest in START, in the previous {num_preceeding_rows} rows (on same contig).
        j = i - 1

        # NB assigns closest in start.
        while j >= 0 and j >= (i - num_preceeding_rows) and (vec_chr[j] == vec_chr[i]):
            if vec_is_interval[j] and vec_start[j] <= this_pos and vec_end[j] > this_pos:
                vec_gene[i] = vec_gene[j]
                break

            j -= 1

    df_gene_snp["gene"] = vec_gene

    logger.debug(f"Assigned SNPs to genes.")

    # NB remove SNPs that have no corresponding genes.
    isin = ~df_gene_snp.gene.isnull()

    # TODO retaining 84.623% of SNPs with known gene (given Gencode filtered by AnnData) for num_preceeding_rows=50.
    logger.info(
        f"Retaining {100.0 * np.mean(isin[~df_gene_snp.is_interval]):.3f}% of snps with matched gene (given reference filtered by visium panel) for num_preceeding_rows={num_preceeding_rows}."
    )

    logger.info(f"Failed to find overlapping gene for:\n{df_gene_snp[df_gene_snp.gene.isnull()]}")

    df_gene_snp = df_gene_snp[isin]

    logger.info(f"Created gene-snp query table:\n{df_gene_snp.head()}")

    return df_gene_snp


def get_unique_names(names: pd.Series) -> list[str]:
    """The distinct names, None dropped, in order of first appearance.

    Args:
        names: one bin's names, in the table's (genomic) row order.

    Returns:
        The names, each once.
    """
    return list(dict.fromkeys(name for name in names if name is not None))


def binned_gene_snp(df_gene_snp, key="bin_id"):
    # NB table with contig range + set of genes + snp_ids,
    #    only for those defined with key bin_id.
    table_bininfo = (
        df_gene_snp[~getattr(df_gene_snp, key).isnull()]
        .groupby(key)
        .agg(
            {
                "CHR": "first",
                "START": "first",
                "END": "last",
                "gene": get_unique_names,
                "snp_id": get_unique_names,
            }
        )
        .reset_index()  # TBC (0, ..., N-1).
    )
    table_bininfo["INCLUDED_GENES"] = [",".join(names) for names in table_bininfo.gene.values]
    table_bininfo["INCLUDED_SNP_IDS"] = [",".join(names) for names in table_bininfo.snp_id.values]
    table_bininfo["NORMAL_COUNT"] = np.nan
    table_bininfo["N_SNPS"] = [len(names) for names in table_bininfo.snp_id.values]

    table_bininfo.drop(columns=["gene", "snp_id"], inplace=True)

    logger.info(
        f"Finalizing genomic segment annotation for {table_bininfo.shape[0]} bins, given initial {len(df_gene_snp.gene.unique())} genes ({table_bininfo.shape[0] / len(df_gene_snp.gene.unique()):.2f} sampling)."
    )

    return table_bininfo


def summarize_blocks(
    gene_snp_table,
    adata,
    cell_snp_Aallele,
    cell_snp_Ballele,
    unique_snp_ids,
    block_key=None,
    normal_candidates=None,
    sort_key="total_umi",
):
    """Log each block's SNP, gene and UMI counts at DEBUG; skipped at any other level.

    Args:
        gene_snp_table: gene and SNP rows with `block_key`.
        adata: counts.
        cell_snp_*allele: (n_spots, n_snps).
    """
    if not logger.isEnabledFor(logging.DEBUG):
        return

    logger.info(f"Summarizing blocks ...")

    assert block_key is not None, "block_key must be specified"
    assert block_key in gene_snp_table.columns, f"{block_key} not in DataFrame"

    map_snp_index = {x: i for i, x in enumerate(unique_snp_ids)}

    block_summary = gene_snp_table.groupby(block_key).agg(
        num_snps=("snp_id", lambda x: x.notna().sum()),
        num_genes=("is_interval", "sum"),
        genes=("gene", lambda x: list({g for g in x if g is not None})),
        snp_ids=("snp_id", lambda x: [s for s in x if s is not None]),
        chr=("CHR", "first"),
        start=("START", "min"),
        end=("END", "max"),
    )

    # NB Mbp -> Kbp.
    block_summary["length"] = 1_000.0 * (block_summary["end"] - block_summary["start"])

    gene_names = adata.var.index.to_numpy()
    gene_index_map = {g: i for i, g in enumerate(gene_names)}
    count_matrix = adata.layers["count"]  # (n_spots, n_genes)

    total_umis = np.zeros(len(block_summary), dtype=int)
    snp_umis = np.zeros(len(block_summary), dtype=int)

    normal_umis = np.zeros(len(block_summary), dtype=int)
    normal_snp_umis = np.zeros(len(block_summary), dtype=int)

    if normal_candidates is not None:
        assert count_matrix.shape[0] == len(normal_candidates), f"{count_matrix.shape[0]} != {len(normal_candidates)}"

        assert cell_snp_Aallele.shape[0] == len(normal_candidates), (
            f"{cell_snp_Aallele.shape[0]} != {len(normal_candidates)}"
        )
        assert cell_snp_Ballele.shape[0] == len(normal_candidates), (
            f"{cell_snp_Ballele.shape[0]} != {len(normal_candidates)}"
        )

    for idx, (block_id, row) in enumerate(block_summary.iterrows()):
        genes = row["genes"]
        if genes:
            gene_idx = [gene_index_map[g] for g in genes if g in gene_index_map]
            if gene_idx:
                block_sum = count_matrix[:, gene_idx].sum()
                total_umis[idx] = int(block_sum)

                # Calculate normal spot UMIs
                if normal_candidates is not None:
                    normal_umis[idx] = int(count_matrix[normal_candidates, :][:, gene_idx].sum())

        # SNP-covering UMIs
        snp_ids = row["snp_ids"]
        if snp_ids:
            # TODO HACK?
            snp_idx = np.array([map_snp_index[s] for s in snp_ids if s])
            if len(snp_idx) > 0:
                snp_umis[idx] = int(cell_snp_Aallele[:, snp_idx].sum() + cell_snp_Ballele[:, snp_idx].sum())

                # Calculate SNP-covering UMIs for normal spots
                if normal_candidates is not None:
                    normal_snp_umis[idx] = int(
                        cell_snp_Aallele[normal_candidates][:, snp_idx].sum()
                        + cell_snp_Ballele[normal_candidates][:, snp_idx].sum()
                    )

    block_summary["total_umi"] = total_umis
    block_summary["snp_umi"] = snp_umis

    block_summary["normal_umi"] = normal_umis
    block_summary["normal_snp_umi"] = normal_snp_umis

    if sort_key is not None:
        block_summary = block_summary.sort_values(sort_key, ascending=False)

    # TODO MAGIC keyword
    max_rows = 25

    logger.info(f"Breakdown of genes/snps/umi per {block_key} sorted by {sort_key} (top {max_rows}):")
    logger.info(
        f"{'block id':<10}\t{'chr':>4}\t{'start':>12}\t{'length':>12} [Kbp]\t{'snps':>8}\t{'genes':>8}\t{'total umi':>12}\t{'snp umi':>12}\t{'normal umi':>12}\t{'normal snp umi':>12}"
    )
    logger.info("-" * 136)

    for ii, (block_id, row) in enumerate(block_summary.iterrows()):
        logger.info(
            f"{block_id:<10}\t{row['chr']:>4}\t{row['start']:>12}\t{row['length'] / 1.0e6:>12}\t{row['num_snps']:>8}\t{row['num_genes']:>8}\t"
            f"{row['total_umi']:>12}\t{row['snp_umi']:>12}\t{row['normal_umi']:>12}\t{row['normal_snp_umi']:>12}"
        )

        if ii > max_rows:
            break

    logger.info(
        f"\n"
        f"median block length: {block_summary['length'].median() / 1.0e6:.1f} [Kbp],\n"
        f"mean block length: {block_summary['length'].mean() / 1.0e6:.1f} [Kbp],\n"
        f"median snps/block: {block_summary['num_snps'].median():.1f},\n"
        f"median genes/block: {block_summary['num_genes'].median():.1f},\n"
        f"median umis/block: {block_summary['total_umi'].median():.1f},\n"
        f"median snp-umis/block: {block_summary['snp_umi'].median():.1f},\n"
        f"total blocks: {len(block_summary):_},\n"
        f"total umis: {block_summary['total_umi'].sum():_},\n"
        f"total snp-umis: {block_summary['snp_umi'].sum():_},\n"
        f"total normal umis: {block_summary['normal_umi'].sum():_},\n"
        f"total normal snp-umis: {block_summary['normal_snp_umi'].sum():_},\n"
        f"blocks with 0 umis: {(block_summary['total_umi'] == 0).mean():.1%},\n"
        f"blocks with <100 umis: {(block_summary['total_umi'] < 100).mean():.1%},\n"
        f"blocks with snp-umis, but no gene-umis: {((block_summary['snp_umi'] > 0) & (block_summary['total_umi'] == 0)).mean():.1%}\n"
    )

    if block_summary.index.isna().any():
        logger.warning(f"Found ill-defined group:/n{block_summary.loc[np.nan]}")


def summarize_counts_for_blocks(
    df_gene_snp,
    adata,
    cell_snp_Aallele,
    cell_snp_Ballele,
    unique_snp_ids,
):
    """Aggregate gene-level total UMIs and SNP-level A/B allele counts into blocks.

    Args:
        df_gene_snp: rows with `block_id`.
        adata: the counts.
        cell_snp_Aallele: (n_spots, n_snps) A allele counts.
        cell_snp_Ballele: (n_spots, n_snps) B allele counts.
        unique_snp_ids: the SNPs' ids.

    Returns:
        The blocks' `SpatioGenomicLabelledMultiCounts`, each row its genes' `contig`, `start` and `end`.
    """
    logger.info(f"Aggregating (snp, umi) counts for genome segmentation.")

    # NB integer counts, so the lazy per-block sums are bitwise in any order.
    n_blocks = len(df_gene_snp.block_id.unique())

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

    # NB the total SNP UMIs as one sum over both alleles' columns side by side.
    snp_columns = snp_idx.astype(np.int64)
    both_alleles = scipy.sparse.hstack(
        [scipy.sparse.csr_matrix(cell_snp_Aallele), scipy.sparse.csr_matrix(cell_snp_Ballele)], format="csr"
    )
    block_of_gene = Segmentation.from_terms(len(gene_names), pairs[:, 0], pairs[:, 1], n_blocks, name="block_of_gene")
    block_of_snp = Segmentation.from_terms(
        cell_snp_Aallele.shape[1], snp_columns, snp_block, n_blocks, name="block_of_snp"
    )
    block_of_allele = Segmentation.from_terms(
        both_alleles.shape[1],
        np.concatenate([snp_columns, snp_columns + cell_snp_Aallele.shape[1]]),
        np.concatenate([snp_block, snp_block]),
        n_blocks,
        name="block_of_allele",
    )
    counts = SpatioGenomicMultiCounts(
        umis=SpatioGenomicCounts.from_grouped_columns(adata.layers["count"], block_of_gene, np.int64),
        alt_snp_umis=SpatioGenomicCounts.from_grouped_columns(cell_snp_Aallele, block_of_snp, np.int64),
        total_snp_umis=SpatioGenomicCounts.from_grouped_columns(both_alleles, block_of_allele, np.int64),
    )

    blocks = Segmentation.from_table(df_gene_snp, "block_id")
    return counts.get_labelled(blocks.contig, blocks.start, blocks.end)


def assign_initial_blocks(
    df_gene_snp,
    adata,
    cell_snp_Aallele,
    cell_snp_Ballele,
    unique_snp_ids,
    initial_min_umi,
):
    """Assign SNPs to genome blocks by merging overlapping gene intervals, each with at least `initial_min_umi` reads.

    Blocks can be aggregated up to the scale of switch errors in the population-based phasing.

    Args: `df_gene_snp` gene and SNP rows; `adata`; `cell_snp_*allele` (n_spots, n_snps); `initial_min_umi` read floor.

    Returns:
        `df_gene_snp` sorted by (CHR, START) with `block_id`; `is_interval` True for a gene, else a SNP.
    """
    if "known_id" in df_gene_snp.columns:
        logger.warning("Assuming known genomic segmentation given 'known_id' in df_gene_snp.")

        df_gene_snp["known_id"] = df_gene_snp["known_id"].astype("Int64")

        unique_ids = df_gene_snp["known_id"].dropna().unique()
        unique_ids = np.sort(unique_ids)  # Ensure order is consistent

        mapping = {old_id: i for i, old_id in enumerate(unique_ids)}

        df_gene_snp["block_id"] = df_gene_snp["known_id"].map(mapping).fillna(-1).astype(int)

        return df_gene_snp

    logger.info(f"Creating initial genome segmentation based solely on overlapping genes.")

    # NB first level: partition of genome by gene range (if two genes overlap, they are grouped to one range);
    # NB == is_gene.
    is_interval = df_gene_snp.is_interval

    # NB merge overlapping genes.
    tmp_block_genome_intervals = list(
        zip(
            df_gene_snp[is_interval].CHR.to_numpy(),
            df_gene_snp[is_interval].START.to_numpy(),
            df_gene_snp[is_interval].END.to_numpy(),
        )
    )

    # NB (chr, start, end) for first gene.
    first_interval = tmp_block_genome_intervals[0]

    # NB list of intervals for initial block definition (merged overlapping genes).
    block_genome_intervals = [first_interval]
    merged = 0

    # NB called snps are limited to transcripts, ergo limited to genes.
    #
    #    initial intervals are gene ranges merged based on overlap.
    for next_interval in tmp_block_genome_intervals[1:]:
        contig, start, end = next_interval

        # NB check whether overlap with previous block
        if contig == block_genome_intervals[-1][0] and max(start, block_genome_intervals[-1][1]) < min(
            end, block_genome_intervals[-1][2]
        ):
            block_genome_intervals[-1] = (
                contig,
                min(start, block_genome_intervals[-1][1]),
                max(end, block_genome_intervals[-1][2]),
            )

            # TODO warn on excessive length;
            merged += 1
        else:
            block_genome_intervals.append(next_interval)

    # NB TODO 20%?
    logger.info(f"Merged {100.0 * merged / len(tmp_block_genome_intervals):.3f}% of genes to ranges as overlapping.")

    # NB map block_genome_intervals to block_ranges for rows of df_gene_snp: each row's interval is the last on its
    #    contig starting before the row ends, kept where the row starts before that interval ends -- the overlap
    #    max(START, start) < min(END, end) of disjoint, sorted intervals. A range is its rows' first and last.
    chrom, start, end = (df_gene_snp[c].to_numpy() for c in ("CHR", "START", "END"))
    interval_chrom = np.array([x[0] for x in block_genome_intervals])
    interval_start = np.array([x[1] for x in block_genome_intervals])
    interval_end = np.array([x[2] for x in block_genome_intervals])
    row_interval = np.full(len(chrom), -1)
    for contig in np.unique(interval_chrom):
        on, rows = np.flatnonzero(interval_chrom == contig), np.flatnonzero(chrom == contig)
        k = on[0] + np.searchsorted(interval_start[on], end[rows], "left") - 1
        inside = (k >= on[0]) & (start[rows] < interval_end[np.maximum(k, on[0])]) & (start[rows] < end[rows])
        later = (k - 1 >= on[0]) & (start[rows] < interval_end[np.maximum(k - 1, on[0])])
        assert not np.any(inside & later), "a row overlaps two merged gene intervals"
        row_interval[rows[inside]] = k[inside]
    assigned = np.flatnonzero(row_interval >= 0)
    first = np.full(len(block_genome_intervals), np.iinfo(np.int64).max)
    last = np.full(len(block_genome_intervals), -1)
    np.minimum.at(first, row_interval[assigned], assigned)
    np.maximum.at(last, row_interval[assigned], assigned)
    block_ranges = list(zip(first.tolist(), (last + 1).tolist()))

    assert np.all(np.array([x[1] for x in block_ranges[:-1]]) == np.array([x[0] for x in block_ranges[1:]]))

    # NB record the initial block id in df_gene_snps: -1 for a row before the first range or after the last.
    initial_block_id = np.full(len(chrom), -1)
    for i, x in enumerate(block_ranges):
        initial_block_id[x[0] : x[1]] = i
    df_gene_snp["initial_block_id"] = initial_block_id

    # NB a SNP in no block would have its UMIs summed into the last block (`np.add.at` at index -1): refused.
    #    `form_gene_snp_table` drops every SNP in no gene, so only a table built otherwise holds one.
    unblocked = df_gene_snp.snp_id.notnull().to_numpy() & (initial_block_id < 0)
    if np.any(unblocked):
        raise ValueError(
            f"{int(unblocked.sum())} SNP rows lie in no merged gene interval, the first "
            f"{df_gene_snp.snp_id.to_numpy()[unblocked][0]}: drop SNPs in no gene, as form_gene_snp_table does."
        )

    logger.info("Assigned snps to initial genome segments (segments == genes, merged on overlap).")

    summarize_blocks(
        df_gene_snp,
        adata,
        cell_snp_Aallele,
        cell_snp_Ballele,
        unique_snp_ids,
        block_key="initial_block_id",
    )

    logger.info(
        f"Updating genome segmentation to ensure min. snp-covering umi={initial_min_umi} threshold is satisfied for the new segments."
    )

    # NB second level: extend the first level blocks based on haplotype-aggregated counts such that the min. snp-covering umi counts >= initial_min_umi.
    #    Each initial block's snp-covering umis once, and blocks s to t as a difference of their running total: the counts
    #    are integers, so it is the sum the loop this replaces took over blocks s to t afresh for each t.
    snp_rows = np.flatnonzero(df_gene_snp.snp_id.notnull().to_numpy())
    snp_index = pd.Index(unique_snp_ids).get_indexer(df_gene_snp.snp_id.to_numpy()[snp_rows])
    assert np.all(snp_index >= 0), "a site absent from unique_snp_ids"
    site_umis = (np.asarray(cell_snp_Aallele.sum(axis=0)).ravel() + np.asarray(cell_snp_Ballele.sum(axis=0)).ravel())[
        snp_index
    ]
    block_umis = np.zeros(len(block_ranges), dtype=site_umis.dtype)
    np.add.at(block_umis, initial_block_id[snp_rows], site_umis)
    running = np.concatenate([np.zeros(1, dtype=block_umis.dtype), np.cumsum(block_umis)])

    initial_block_chr = chrom[np.array([x[0] for x in block_ranges])]
    block_ranges_new = []
    s = 0

    # NB s is the "lower" initial_block_id and t is the "upper" initial_block_id.
    while s < len(block_ranges):
        t = s

        while t <= len(block_ranges):
            t += 1

            reach_end = t == len(block_ranges)
            change_chr = initial_block_chr[s] != initial_block_chr[t - 1]

            # NB num. of snp-covering umis for initial block ids s to t.
            this_snp_umis = running[t] - running[s]

            if reach_end:
                logger.warning(f"Reached last block with {this_snp_umis}/{initial_min_umi} required snp umis.")
                break

            if change_chr:
                t -= 1
                this_snp_umis = running[t] - running[s]

                logger.warning(f"Reached contig end with {this_snp_umis}/{initial_min_umi} required snp umis.")

                break

            if this_snp_umis >= initial_min_umi:
                break

        # NB goal is to have assigned this_snp_umis, s and t: initial blocks s to t span rows
        #    block_ranges[s][0] to block_ranges[t - 1][1], contiguous as asserted above.
        if this_snp_umis < initial_min_umi and s > 0 and initial_block_chr[s - 1] == initial_block_chr[s]:
            block_ranges_new[-1] = (block_ranges_new[-1][0], block_ranges[t - 1][1])
        else:
            block_ranges_new.append((block_ranges[s][0], block_ranges[t - 1][1]))

        # NB fast-forward lower block id to upper.
        s = t

    # NB record the block id in df_gene_snp
    block_id = np.zeros(len(chrom), dtype=np.int64)
    for i, x in enumerate(block_ranges_new):
        block_id[x[0] : x[1]] = i
    df_gene_snp["block_id"] = block_id

    logger.info(
        f"Updated genome segmentation given (population phased) genotypes and min. snp-covering umi={initial_min_umi} per segment."
    )

    summarize_blocks(
        df_gene_snp,
        adata,
        cell_snp_Aallele,
        cell_snp_Ballele,
        unique_snp_ids,
        block_key="block_id",
    )

    return df_gene_snp.drop(columns=["initial_block_id"])


def create_bin_ranges(
    df_gene_snp,
    adata,
    cell_snp_Aallele,
    cell_snp_Ballele,
    unique_snp_ids,
    counts,
    refined_lengths,
    secondary_min_umi,
    secondary_min_snp_umi,
    secondary_min_normal_umi,
    normal_candidates=None,
    max_binlength=5e6,
    key="block_id",
):
    """Aggregate haplotype blocks into bins under UMI and length constraints, never across `refined_lengths`.

    Args:
        df_gene_snp: rows with `key`.
        adata: unused.
        cell_snp_Aallele: unused.
        cell_snp_Ballele: unused.
        unique_snp_ids: unused.
        counts: the blocks' `SpatioGenomicLabelledMultiCounts`.
        refined_lengths: blocks per phase segment.
        secondary_min_umi: the floor on UMIs per bin.
        secondary_min_snp_umi: the floor on SNP UMIs per bin.
        secondary_min_normal_umi: the floor on normal spots' UMIs per bin.
        normal_candidates: the normal spots, or None.
        max_binlength: the longest bin, in base pairs.
        key: the block column.

    Returns:
        `df_gene_snp` with a `bin_id` column.
    """
    logger.info(f"Aggregating blocks to bins given baf-inferred phasing to satisfy umi, length, etc. constraints.")

    # TODO BUG dropna?
    # NB block intervals: by key (e.g. block_id)
    sorted_chr_pos_both = df_gene_snp.groupby(key).agg({"CHR": "first", "START": "first", "END": "last"})

    # NB block intervals
    block_lengths = sorted_chr_pos_both.END.to_numpy() - sorted_chr_pos_both.START.to_numpy()
    n_blocks = len(block_lengths)

    # NB total umi per block (summed across spots)
    block_umi = counts.umis.sum(axis=0)

    # NB total snp-covering umi per block
    block_snp_umi = counts.total_snp_umis.sum(axis=0)

    # NB normal-spot umi per block
    if normal_candidates is not None:
        if isinstance(normal_candidates, (np.ndarray, pd.Series)) and normal_candidates.dtype == bool:
            normal_idx = np.flatnonzero(normal_candidates)
        else:
            normal_idx = np.asarray(normal_candidates, dtype=int)

        block_normal_umi = counts.umis.spot_sums(normal_idx)
    else:
        block_normal_umi = np.zeros(n_blocks, dtype=int)
        secondary_min_normal_umi = 0

    assert len(block_lengths) == len(block_umi) == len(block_snp_umi) == len(block_normal_umi), (
        f"Block array length mismatch: "
        f"lengths={len(block_lengths)}, "
        f"umi={len(block_umi)}, "
        f"snp_umi={len(block_snp_umi)}, "
        f"normal_umi={len(block_normal_umi)}"
    )

    frac_normal = normal_candidates.mean() if normal_candidates is not None else np.nan

    logger.info(
        f"Creating (phased) bin ranges: max_length={max_binlength:_}, "
        f"min_umi={secondary_min_umi}, "
        f"min_snp_umi={secondary_min_snp_umi}, "
        f"min_normal_umi={secondary_min_normal_umi}, "
        f"fraction normal={frac_normal:.3f}"
    )

    # NB breakpoints defined by jump in minor baf, and oversized blocks.
    breakpoints = np.concatenate(
        [
            np.cumsum(refined_lengths),
            np.where(block_lengths > max_binlength)[0],
            np.where(block_lengths > max_binlength)[0] + 1,
        ]
    )

    # NB sorted, unique.
    breakpoints = np.sort(np.unique(breakpoints))

    if breakpoints[0] != 0:
        breakpoints = np.append([0], breakpoints)

    assert np.all(breakpoints[:-1] < breakpoints[1:])

    # NB assign each block to a bin (that meets umi, length, etc. requirements)
    bin_ids = np.zeros(n_blocks, dtype=int)
    offset = 0

    for i in range(len(breakpoints) - 1):
        b1, b2 = breakpoints[i], breakpoints[i + 1]

        if b2 - b1 == 1:
            bin_ids[b1:b2] = offset
            offset += 1
        else:
            this_bin_ids = greedy_binning_nobreak(
                block_lengths[b1:b2],
                block_umi[b1:b2],
                block_snp_umi[b1:b2],
                block_normal_umi[b1:b2],
                secondary_min_umi,
                secondary_min_snp_umi,
                secondary_min_normal_umi,
                max_binlength,
            )

            bin_ids[b1:b2] = offset + this_bin_ids
            offset += np.max(this_bin_ids) + 1

    if "bin_id" in df_gene_snp.columns:
        logger.warning(f"Overwriting bin_id column, storing in block_id.")
        df_gene_snp["block_id"] = df_gene_snp["bin_id"]

    # NB map each block_id to its bin id
    df_gene_snp["bin_id"] = getattr(df_gene_snp, key).map({i: x for i, x in enumerate(bin_ids)})

    # NB return df_gene_snp with an updated bin_id column, and potentially deifned block_id column if it was overwritten.
    return df_gene_snp


def summarize_counts_for_bins(
    df_gene_snp,
    adata,
    counts,
    phase_indicator,
):
    """Aggregate block counts into the bins of `df_gene_snp`'s `bin_id`, phased by `phase_indicator`.

    Args:
        df_gene_snp: rows with `block_id` and `bin_id`.
        adata: the counts.
        counts: the blocks' `SpatioGenomicLabelledMultiCounts`.
        phase_indicator: (n_blocks,) phase per block.

    Returns:
        The bins' `SpatioGenomicLabelledMultiCounts`, each row's span carried from its blocks, and `counts`' spot labellings.
    """
    logger.info(f"Summarizing counts for bins.")

    has_assigned_bin = ~df_gene_snp.bin_id.isnull()

    bins = df_gene_snp.loc[has_assigned_bin, "bin_id"].unique()

    # NB last axis is the number of spot (barcodes).
    n_bins = len(bins)

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

    # NB bin sums as lazy segmentations: integer, exact in any order; B counts phased per block; rows by sorted bin id.
    n_rows = counts.n_segments

    block_pairs = [(b, int(x)) for b in range(df_bin_contents.shape[0]) for x in block_sets[b] if x is not None]
    block_bins, block_ids = (
        (np.array(v, dtype=np.int64) for v in zip(*block_pairs)) if block_pairs else (np.zeros(0, np.int64),) * 2
    )
    used = np.unique(block_ids)
    h0 = np.zeros(n_rows, dtype=bool)
    h0[used] = np.asarray(phase_indicator)[used].astype(bool)
    phased = counts.multi_counts.get_phased(h0)

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
    bin_of_gene = Segmentation.from_terms(len(gene_names), gene_cols, gene_bins, n_bins, name="bin_of_gene")
    bin_of_block = Segmentation.from_terms(n_rows, block_ids, block_bins, n_bins, name="bin_of_block")
    binned = SpatioGenomicMultiCounts(
        umis=SpatioGenomicCounts.from_grouped_columns(count_matrix, bin_of_gene, np.int64),
        alt_snp_umis=get_segmented(phased.alt_snp_umis, bin_of_block),
        total_snp_umis=get_segmented(phased.total_snp_umis, bin_of_block),
    )

    return binned.get_labelled(*counts.get_segment_labels(bin_of_block))._replace(labellings=counts.labellings)
