import ast

import anndata
import numpy as np
import scanpy as sc
import scipy
import scipy.sparse
import scipy.stats
from numba import njit, prange
from sklearn.cluster import KMeans

from cnamaste.config import get_global_config, start_time
from cnamaste.hmm_emission import WeightedBBFit
from cnamaste.hmm_utils import get_em_solver_params, get_normal_clone
from cnamaste.logger import get_logger
from cnamaste.segments import Segmentation
from cnamaste.sparse import sum_over_row_csr, sum_over_row_dense
from cnamaste.spatio_genomic_counts import SpatioGenomicCounts, get_grouped_sums

logger = get_logger(__name__, start_time=start_time)


def determine_normal_candidates(
    config,
    res,
    baf_profiles,
    umis,
):
    """Determine normal candidate spots from BAF profiles, or provided indices.

    Args:
        config: the run's.
        res: the BAF-only fit.
        baf_profiles: (n_clones, n_obs) per clone.
        umis: (n_spots, n_obs) read depth.

    Returns:
        The boolean normal-candidate mask, or None where `normalidx_file` is given.
    """
    logger.info(f"Determining normal spots based on BAF-only clones.")

    # NB no input file for barcodes of normal spots; a tumor proportion per spot is not read.
    if config.preprocessing.normalidx_file is None:
        eps_baf = config.quality.baf.eps
        PERCENT_NORMAL = 40  # MAGIC

        logger.info(
            f"Identifying normal spots based on estimated BAF given eps_baf={eps_baf} and PERCENT_NORMAL={PERCENT_NORMAL}."
        )

        # NB baf_profiles is (n_clones, n_obs).
        id_nearnormal_clone = get_normal_clone(np.asarray(baf_profiles, dtype=np.float64).T, eps_baf)

        logger.info(f"Found clone {id_nearnormal_clone} to be the most normal-like given BAF deviations.")

        # NB measure the standard deviation of log-transformed transcript counts for each spot.
        vec_stds = log1p_std_by_spot(umis)

        prior_stdthreshold = np.inf

        while True:
            stdthreshold = np.percentile(
                vec_stds[res["new_assignment"] == id_nearnormal_clone],
                PERCENT_NORMAL,
            )
            normal_candidate = (vec_stds < stdthreshold) & (res["new_assignment"] == id_nearnormal_clone)

            # DEPRECATE config.run_legacy.
            if stdthreshold > 1.5 * prior_stdthreshold:  # MAGIC
                logger.info(
                    f"Determined {PERCENT_NORMAL}% normal spots with sufficient UMIs, assigned to normal like clone."
                )
                logger.info(
                    f"BAF-clone breakdown:\n{np.unique(res['new_assignment'][normal_candidate], return_counts=True)}"
                )
                break
            elif PERCENT_NORMAL == 100:
                logger.warning(
                    f"All {np.count_nonzero(res['new_assignment'] == id_nearnormal_clone)} spots for clone {id_nearnormal_clone} considered to be normal."
                )
                break

            PERCENT_NORMAL += 10
        return normal_candidate

    # You may want to load normal_candidate from file here
    return None


@njit(cache=True)
def _squared_deviation(v, i, o, k, args):
    """Compute `(v - mean_i)^2` for spot `i`'s value `v` at segment `o`.

    Args:
        v: the value.
        i: the spot.
        o: the segment.
        k: unused.
        args: the per-spot means.

    Returns:
        The squared deviation.
    """
    dev = v - args[0][i]
    return dev * dev


@njit(parallel=True, cache=True)
def _squares_csr(n_obs, indptr, indices, data, means):
    out = np.empty(len(indptr) - 1)
    for i in prange(len(indptr) - 1):
        out[i] = sum_over_row_csr(_squared_deviation, n_obs, indptr, indices, data, i, 0, (means,))
    return out


@njit(parallel=True, cache=True)
def _squares_dense(values, means):
    out = np.empty(values.shape[0])
    for i in prange(values.shape[0]):
        out[i] = sum_over_row_dense(_squared_deviation, values, i, 0, (means,))
    return out


def log1p_std_by_spot(single_X_rdr):
    """Compute `np.std(np.log1p(single_X_rdr), axis=1)` bitwise in either backing.

    log1p a `map`, the mean a `sum(axis=1)` in sequence, squared deviations over every segment in sequence (a zero count
    deviates by the mean), then divided and rooted.

    Args:
        single_X_rdr: (n_spots, n_obs) dense or CSR.

    Returns:
        The (n_spots,) standard deviations.
    """
    logs = SpatioGenomicCounts.of(single_X_rdr).map(np.log1p)
    n_obs = logs.shape[1]
    means = logs.sum(axis=1) / n_obs
    if logs.sparse:
        squares = _squares_csr(n_obs, logs.values.indptr, logs.values.indices, logs.values.data, means)
    else:
        squares = _squares_dense(logs.values, means)
    return np.sqrt(squares / n_obs)


def determine_normal_baseline(counts, normal_candidate, config=None):
    """Zero the bins whose normal spots hold under `quality.min_normal_count_perbin` UMIs and label the normal spots.

    Args:
        counts: a `SpatioGenomicLabelledMultiCounts`.
        normal_candidate: (n_spots,) bool.
        config: the run's, or None for the global one.

    Returns:
        The counts, `umis` zeroed at those bins, labelled `normal_spot` (`normal_baseline_sized`).
    """
    if config is None:
        config = get_global_config()

    single_X_rdr = counts.umis

    logger.info(
        f"Found sparsity of normal spot set={100.0 * single_X_rdr.spots(normal_candidate == True).mean() == 0.0:.3f}%"
    )

    # NB normal baseline transcript count; unnormalized.
    rdr_normal = single_X_rdr.spot_sums(normal_candidate == True)

    bidx_inconfident = np.where(rdr_normal < config.quality.min_normal_count_perbin)[0]

    logger.info(
        f"Found {100.0 * np.mean(rdr_normal >= config.quality.min_normal_count_perbin):.3f}% of segments with confident normal baseline for MIN_NORMAL_COUNT_PERBIN={config.quality.min_normal_count_perbin}"
    )
    # NB the bins quality.min_normal_count_perbin zeroes, and the share of the spots' read depth they held.
    logger.info(
        f"quality.min_normal_count_perbin={config.quality.min_normal_count_perbin}: zeroing {len(bidx_inconfident)} of "
        f"{rdr_normal.size} bins, {100.0 * single_X_rdr.segments(bidx_inconfident).sum() / max(single_X_rdr.sum(), 1):.3f}% of "
        f"the spots' read depth and {100.0 * rdr_normal[bidx_inconfident].sum() / max(rdr_normal.sum(), 1):.3f}% of the "
        f"normal spots'."
    )

    # NB avoid ill-defined distributions if normal has 0 count in that bin: a new array, the input unchanged.
    zeroed = counts._replace(umis=single_X_rdr.zero_segments(bidx_inconfident))
    return zeroed.with_labelling("normal_spot", np.asarray(normal_candidate, dtype=bool))


def _log_normalized(counts, min_cells=10):
    """Filter genes in at least `min_cells` spots, normalize to the median spot total and `log1p`, on CSR counts.

    Bitwise scanpy 1.12's `filter_genes`, `normalize_total` and `log1p`, without scanpy's
    numba kernels; integer counts, so every sum is exact in any order.

    Args:
        counts: CSR (n_spots, n_genes) integer.
        min_cells: the floor.

    Returns:
        The kept genes' mask and the normalized CSR.
    """
    keep = np.asarray((counts > 0).sum(axis=0)).ravel() >= min_cells
    kept = counts[:, keep]
    totals = np.asarray(kept.sum(axis=1)).ravel()
    factor = totals / np.median(totals)
    factor = factor + (factor == 0)
    data = np.log1p(np.true_divide(kept.data, np.repeat(factor, np.diff(kept.indptr))))
    return keep, scipy.sparse.csr_matrix((data, kept.indices.copy(), kept.indptr.copy()), shape=kept.shape)


# NB equivalent to CalicoST utils_IO.filter_de_genes_tri.
def filter_normal_diffexp(
    exp_counts,
    df_bininfo,
    normal_candidate,
    sample_list=None,
    sample_ids=None,
    logfcthreshold_u=2,
    logfcthreshold_t=4,
    quantile_threshold=80,
    use_kmeans=True,
):
    """Drop differentially expressed genes between normal and tumor spots, clustered per slice by PCA + kmeans.

    Normal spots come from kmeans or the given `normal_candidate`. Counts stay sparse and integer; each bin's spot
    totals are one sparse product with the bins' gene incidence, bitwise the AnnData form.

    Args:
        exp_counts: gene counts.
        df_bininfo: bins.
        normal_candidate: mask.
        logfcthreshold_*: fold-change cuts.

    Returns:
        `new_single_X_rdr` (n_spots, n_bins) and the filtered-out genes.
    """
    genes = np.asarray(exp_counts.columns)
    counts = scipy.sparse.csr_matrix(anndata.AnnData(exp_counts).X)
    normal_candidate = np.asarray(normal_candidate, dtype=bool)
    gene_umis = np.asarray(counts.sum(axis=0)).ravel()

    if sample_list is None:
        sample_list = [None]

    filtered_out_set = set()

    for s, sname in enumerate(sample_list):
        index = np.arange(counts.shape[0]) if sname is None else np.where(sample_ids == s)[0]
        slice_counts, normal_mask = counts[index], normal_candidate[index]

        if slice_counts[normal_mask].sum() < slice_counts.shape[1] * 10:
            logger.warning(f"Insufficient normal spot UMIs for slice {sname}. Skipping.")
            continue

        umi_threshold = np.percentile(np.asarray(slice_counts.sum(axis=0)).ravel(), quantile_threshold)

        keep, expression = _log_normalized(slice_counts)
        kept_counts, kept_genes = slice_counts[:, keep], genes[keep]

        # ---------------------------------------------------------
        # CLUSTERING LOGIC: K-Means vs Pre-Defined
        # ---------------------------------------------------------
        if use_kmeans:
            pcs = anndata.AnnData(expression)
            sc.pp.pca(pcs, n_comps=4)
            kmeans = KMeans(n_clusters=2, random_state=0).fit(pcs.obsm["X_pca"])
            kmeans_labels = kmeans.predict(pcs.obsm["X_pca"])

            # Log cluster proportions
            n_spots_slice = len(kmeans_labels)
            label_counts = np.bincount(kmeans_labels, minlength=2)
            logger.info(
                f"K-means clustering for slice {sname}: "
                f"Cluster 0: {100.0 * label_counts[0] / n_spots_slice:.1f}% | "
                f"Cluster 1: {100.0 * label_counts[1] / n_spots_slice:.1f}%"
            )

            idx_kmeans_label = np.argmax(np.bincount(kmeans_labels[normal_mask], minlength=2))

            # Log overlap with normal candidates
            n_candidates = np.sum(normal_mask)
            overlap = np.sum(kmeans_labels[normal_mask] == idx_kmeans_label)

            if n_candidates > 0:
                logger.info(
                    f"Overlap: {overlap}/{n_candidates} ({100.0 * overlap / n_candidates:.1f}%) "
                    f"of pre-defined normal candidates fell into the designated 'normal' Cluster {idx_kmeans_label}."
                )

            clone = np.array(["normal"] * len(index), dtype=object)
            clone[(kmeans_labels != idx_kmeans_label) & (~normal_mask)] = "tumor"
            clone[(kmeans_labels == idx_kmeans_label) & (~normal_mask)] = "unsure"

        else:
            # Bypass K-means entirely, rely on pre-labeled normal candidates
            logger.info(f"Skipping K-means for slice {sname}. Using pre-defined normal candidates directly.")
            clone = np.array(["tumor"] * len(index), dtype=object)
            clone[normal_mask] = "normal"
            # Note: There are no "unsure" spots in this mode.

        # ---------------------------------------------------------
        # DIFFERENTIAL EXPRESSION CALCULATION
        # ---------------------------------------------------------
        agg_counts = np.vstack(
            [np.asarray(kept_counts[clone == label].sum(axis=0)).ravel() for label in ["normal", "unsure", "tumor"]]
        )
        # Avoid division by zero on empty slices/clusters
        row_sums = np.sum(agg_counts, axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        agg_counts = agg_counts / row_sums * 1e6

        geneumis = gene_umis[keep]

        def logfc(label):
            if not np.any(clone == ("unsure", "tumor")[label - 1]):
                return np.zeros(len(kept_genes))
            return np.where(
                (agg_counts[label, :] == 0) | (agg_counts[0, :] == 0),
                10,
                np.log2(agg_counts[label, :] / agg_counts[0, :]),
            )

        # NB the log of a zero share is masked by `where` and only warns.
        with np.errstate(divide="ignore", invalid="ignore"):
            logfc_u, logfc_t = logfc(1), logfc(2)

        this_filtered_out_set = set(
            kept_genes[(np.abs(logfc_u) > logfcthreshold_u) & (geneumis > umi_threshold)]
        ) | set(kept_genes[(np.abs(logfc_t) > logfcthreshold_t) & (geneumis > umi_threshold)])
        filtered_out_set = filtered_out_set | this_filtered_out_set

        logger.info(
            f"Removed {len(this_filtered_out_set)} genes with differential expression for slice {sname}. "
            f"Total unique removed across run: {len(filtered_out_set)}."
        )

    # NB each bin's genes as one column of a (genes, bins) incidence: a gene in two bins counts in both.
    column_of = {g: k for k, g in enumerate(genes)}
    rows, cols = [], []
    for b, genestr in enumerate(df_bininfo.INCLUDED_GENES.values):
        found = sorted(column_of[g] for g in set(genestr.split(",")) if g in column_of)
        rows += found
        cols += [b] * len(found)
    incidence = scipy.sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.int64), (rows, cols)), shape=(len(genes), df_bininfo.shape[0])
    )
    retained = np.array([g not in filtered_out_set for g in genes])

    # NB a CSR-backed `SpatioGenomicCounts` (`spatio_genomic_counts`): the integer sums, as float64.
    new_single_X_rdr = SpatioGenomicCounts(get_grouped_sums(counts[:, retained], incidence[retained])).astype(
        np.float64
    )
    total_counts = int(gene_umis @ np.asarray(incidence.sum(axis=1)).ravel())
    retained_counts = int(gene_umis[retained] @ np.asarray(incidence[retained].sum(axis=1)).ravel())

    logger.info(f"Retained {100.0 * retained_counts / max(total_counts, 1):.3f}% of bin UMIs.")

    return new_single_X_rdr, filtered_out_set


def fit_normal_spot_bb(baf_counts: np.ndarray, total_counts: np.ndarray) -> np.ndarray:
    """The beta-binomial fit to the normal spots' pooled B-allele counts per bin.

    Args:
        baf_counts: `(n_obs,)` B-allele counts summed over the normal spots.
        total_counts: `(n_obs,)` allele totals summed over the normal spots.

    Returns:
        The fit's parameters, `(p, tau)`.
    """
    model = WeightedBBFit(baf_counts, np.ones(len(baf_counts)), weights=np.ones(len(baf_counts)), exposure=total_counts)
    return np.asarray(model.fit(**get_em_solver_params()).params)


def normal_baf_bin_filter(
    df_gene_snp,
    counts,
    index_normal,
    confidence_interval=None,
    min_betabinom_tau=30,
):
    """Re-aggregate counts after dropping bins whose BAF is not normal-like in the normal clone.

    That may be from a misidentified normal clone, mixing with non-normal spots, or allele-specific expression.

    Args:
        df_gene_snp: the gene and SNP table, with `bin_id`.
        counts: a `SpatioGenomicLabelledMultiCounts`.
        index_normal: the normal spots.
        confidence_interval: the BAF band, or None for the config's.
        min_betabinom_tau: the floor on the fitted concentration.

    Returns:
        `df_gene_snp` and the kept bins' counts, with `counts`' spot labellings.
    """
    if confidence_interval is None:
        confidence_interval = ast.literal_eval(get_global_config().quality.normal_allele_specific_confidence)

    logger.info("Selecting bins for removal based on normal spot BAF.")

    # NB pool b-allele counts for each bin across all normal spots; 1D genomic segments.
    tmpX = counts.alt_snp_umis.spot_sums(index_normal)
    tmptotal_bb_RD = counts.total_snp_umis.spot_sums(index_normal)

    params = fit_normal_spot_bb(tmpX, tmptotal_bb_RD)

    logger.info(f"Best-fit BetaBinom model to normal spot BAF has parameters={params}")

    # TODO warn if patched.
    # NB patches parameters assuming min_betabinom_tau=30;
    params[0] = 0.5
    params[-1] = max(params[-1], min_betabinom_tau)

    # NB drop bins whose normal-spot BAF falls outside the confidence interval; cut by cdf, the bins ppf would cut.
    a, b = params[0] * params[1], (1.0 - params[0]) * params[1]
    removal_indicator1 = scipy.stats.betabinom.cdf(tmpX, tmptotal_bb_RD, a, b) < confidence_interval[0]
    removal_indicator2 = scipy.stats.betabinom.cdf(tmpX - 1, tmptotal_bb_RD, a, b) >= confidence_interval[1]

    index_removal = np.where(removal_indicator1 | removal_indicator2)[0]
    index_remaining = np.where(~(removal_indicator1 | removal_indicator2))[0]

    removal_indicator = removal_indicator1 | removal_indicator2

    logger.info(
        f"Removing {np.count_nonzero(removal_indicator)} [{100.0 * np.mean(removal_indicator):.4f}]% genomic segments with potential allele-specific expression, based on normal candidates --- confidence={confidence_interval} and min_betabinom_tau={min_betabinom_tau}."
    )

    # NB below constructs the counts with segments removed.
    col = np.where(df_gene_snp.columns == "bin_id")[0][0]
    df_gene_snp.iloc[np.where(df_gene_snp.bin_id.isin(index_removal))[0], col] = None

    # NB reassign bin_id to be unique integers in [0, n_bins_remaining) for downstream processing.
    df_gene_snp["bin_id"] = df_gene_snp["bin_id"].map({x: i for i, x in enumerate(index_remaining)})
    df_gene_snp.bin_id = df_gene_snp.bin_id.astype("Int64")

    logger.info(f"Solved for unique bin ids:\n{np.unique(df_gene_snp.bin_id)}")

    if df_gene_snp.bin_id.isnull().any():
        logger.warning(f"Detected {df_gene_snp.bin_id.isnull().sum()} bin_ids with NaN value.")

    # NB the kept bins as a segmentation of the bins, each kept bin its own segment, lazily.
    n_remaining = len(index_remaining)
    kept_bin_of_bin = Segmentation.from_terms(
        counts.n_segments, index_remaining, np.arange(n_remaining), n_remaining, name="kept_bin_of_bin"
    )
    kept = counts.multi_counts.get_segmented(kept_bin_of_bin)

    assert df_gene_snp["bin_id"].nunique(dropna=True) == n_remaining, (
        f"{df_gene_snp['bin_id'].notna().sum()} != {n_remaining}"
    )

    labelled = kept.get_labelled(*counts.get_segment_labels(kept_bin_of_bin))
    return df_gene_snp, labelled._replace(labellings=counts.labellings)
