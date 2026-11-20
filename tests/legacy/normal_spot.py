"""Frozen copies of `cnamaste.normal_spot.normal_baf_bin_filter` as PR- #62 held it and `determine_normal_baseline` as
PR- #101 held it, verbatim, which T- #84's count containers replace; `tests/legacy/test_legacy.py` holds the new code
byte-equal to them."""

import ast

import numpy as np
import scipy.stats

from cnamaste.config import get_global_config
from cnamaste.normal_spot import fit_normal_spot_bb, logger
from legacy.spatio_genomic_counts import BinnedCounts, SpatioGenomicCounts, SpatioGenomicPairCounts


def normal_baf_bin_filter(
    df_gene_snp,
    single_X,
    single_base_nb_mean,
    single_total_bb_RD,
    index_normal,
    confidence_interval=None,
    min_betabinom_tau=30,
):
    """Re-aggregate counts after dropping bins whose BAF is not normal-like in the normal clone.

    That may be from a misidentified normal clone, mixing with non-normal spots, or allele-specific expression.

    Args: `df_gene_snp`; `single_X` (n_obs, 2, n_spots); `index_normal` normal spots; `confidence_interval` BAF band.
    Returns: `df_gene_snp` and a `BinnedCounts` of the kept bins.
    """
    if confidence_interval is None:
        confidence_interval = ast.literal_eval(get_global_config().quality.normal_allele_specific_confidence)

    logger.info("Selecting bins for removal based on normal spot BAF.")

    # NB pool b-allele counts for each bin across all normal spots; 1D genomic segments.
    single_X, single_base_nb_mean, single_total_bb_RD = (
        SpatioGenomicPairCounts.of(single_X),
        SpatioGenomicCounts.of(single_base_nb_mean),
        SpatioGenomicCounts.of(single_total_bb_RD),
    )
    tmpX = single_X.baf.spot_sums(index_normal)
    tmptotal_bb_RD = single_total_bb_RD.spot_sums(index_normal)

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

    # NB below constructs single_X, single_base_nb_mean, single_total_bb_RD with segments removed.
    col = np.where(df_gene_snp.columns == "bin_id")[0][0]
    df_gene_snp.iloc[np.where(df_gene_snp.bin_id.isin(index_removal))[0], col] = None

    # NB reassign bin_id to be unique integers in [0, n_bins_remaining) for downstream processing.
    df_gene_snp["bin_id"] = df_gene_snp["bin_id"].map({x: i for i, x in enumerate(index_remaining)})
    df_gene_snp.bin_id = df_gene_snp.bin_id.astype("Int64")

    logger.info(f"Solved for unique bin ids:\n{np.unique(df_gene_snp.bin_id)}")

    if df_gene_snp.bin_id.isnull().any():
        logger.warning(f"Detected {df_gene_snp.bin_id.isnull().sum()} bin_ids with NaN value.")

    single_X = single_X.rows(index_remaining)
    single_base_nb_mean = single_base_nb_mean.rows(index_remaining)
    single_total_bb_RD = single_total_bb_RD.rows(index_remaining)

    lengths = np.zeros(len(df_gene_snp.CHR.unique()), dtype=int)

    for i, c in enumerate(df_gene_snp.CHR.unique()):
        lengths[i] = len(df_gene_snp[(df_gene_snp.CHR == c) & (~df_gene_snp.bin_id.isnull())].bin_id.unique())

    assert df_gene_snp["bin_id"].nunique(dropna=True) == single_X.shape[0], (
        f"{df_gene_snp['bin_id'].notna().sum()} != {single_X.shape[0]}"
    )
    assert df_gene_snp["bin_id"].nunique(dropna=True) == sum(lengths), (
        f"{df_gene_snp['bin_id'].notna().sum()} != {sum(lengths)}"
    )

    return df_gene_snp, BinnedCounts(lengths, single_X, single_base_nb_mean, single_total_bb_RD)


def determine_normal_baseline(single_X_rdr, normal_candidate, config=None):
    if config is None:
        config = get_global_config()

    logger.info(
        f"Found sparsity of normal spot set={100.0 * SpatioGenomicCounts.of(single_X_rdr).spots(normal_candidate == True).mean() == 0.0:.3f}%"
    )

    # NB normal baseline transcript count; unnormalized.
    # TODO normal_candidate clause
    single_X_rdr = SpatioGenomicCounts.of(single_X_rdr)
    rdr_normal = single_X_rdr.spot_sums(normal_candidate == True)

    bidx_inconfident = np.where(rdr_normal < config.quality.min_normal_count_perbin)[0]

    logger.info(
        f"Found {100.0 * np.mean(rdr_normal >= config.quality.min_normal_count_perbin):.3f}% of segments with confident normal baseline for MIN_NORMAL_COUNT_PERBIN={config.quality.min_normal_count_perbin}"
    )
    # NB the bins quality.min_normal_count_perbin zeroes, and the share of the spots' read depth they held.
    logger.info(
        f"quality.min_normal_count_perbin={config.quality.min_normal_count_perbin}: zeroing {len(bidx_inconfident)} of "
        f"{rdr_normal.size} bins, {100.0 * single_X_rdr.rows(bidx_inconfident).sum() / max(single_X_rdr.sum(), 1):.3f}% of "
        f"the spots' read depth and {100.0 * rdr_normal[bidx_inconfident].sum() / max(rdr_normal.sum(), 1):.3f}% of the "
        f"normal spots'."
    )

    # NB where normal transcript count < config.quality.min_normal_count_perbin, zero.
    rdr_normal[bidx_inconfident] = 0
    rdr_normal = rdr_normal / np.sum(rdr_normal)

    # NB avoid ill-defined distributions if normal has 0 count in that bin: a new array, the input unchanged.
    single_X_rdr = single_X_rdr.zero_rows(bidx_inconfident)

    # NB replicate and normalize rdr_normal to the per-spot total transcripts, T_n, after zero'ing.
    spots_coverage = single_X_rdr.sum(axis=0)

    single_base_nb_mean = rdr_normal.reshape(-1, 1) @ spots_coverage.reshape(1, -1)

    return rdr_normal, single_X_rdr, single_base_nb_mean
