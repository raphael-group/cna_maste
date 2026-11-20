import numpy as np

from cnamaste.config import get_global_config, start_time
from cnamaste.logger import get_logger
from cnamaste.reference import get_reference_recomb_rates

logger = get_logger(__name__, start_time=start_time)


def compute_numbat_phase_switch_prob(position_cM, chr_pos_vector, nu=1.0, min_prob=None):
    """Numbat's phase switch probability of adjacent SNPs, `(1 - exp(-2 nu d)) / 2` for d cM, floored at `min_prob`.

    Args: `position_cM` (SNPs,) cM; `chr_pos_vector` (chr, pos) per SNP; `nu`; `min_prob` floor (config's by default).

    Returns:
        (SNPs,) switch probabilities.
    """
    if min_prob is None:
        min_prob = get_global_config().phasing.min_prob

    # NB no SNPs, no switches: CalicoST's empty array, rather than the log line's max of nothing raising.
    if len(position_cM) == 0:
        return np.ones(0)

    logger.info_once(f"Computing numbat phase switch probabilities assuming nu={nu}.")
    logger.info(f"position_cM has {100.0 * np.mean(np.isnan(position_cM))}% NAN content.")

    phase_switch_prob = min_prob * np.ones(len(position_cM))

    for i, cm in enumerate(position_cM[:-1]):
        cm_next = position_cM[i + 1]

        # NB defaults to min_prob. in these cases.
        if np.isnan(cm) or np.isnan(cm_next) or chr_pos_vector[i][0] != chr_pos_vector[i + 1][0]:
            continue

        assert cm <= cm_next

        d = cm_next - cm

        # NB numbat definition;
        phase_switch_prob[i] = (1.0 - np.exp(-2.0 * nu * d)) / 2.0

    logger.info_once(f"Solved for max phase switch prob. = {np.max(phase_switch_prob)}")

    under_flowed = phase_switch_prob < min_prob

    logger.info(
        f"Reassigning under flowed phase_switch_prob. for {100.0 * np.mean(under_flowed):.6e}% given {min_prob} threshold."
    )

    phase_switch_prob[phase_switch_prob < min_prob] = min_prob

    return phase_switch_prob


def assert_logical_order(chr_pos_vector):
    """Assert (chr, pos) pairs non-decreasing in logical order: integer contigs 1, 2, ..., 22, then position.

    Args:
        chr_pos_vector: (chr, pos) pairs.

    Raises:
        AssertionError: where out of order.
    """
    keys = [(int(chrom), pos) for chrom, pos in chr_pos_vector]
    assert all(a <= b for a, b in zip(keys, keys[1:])), "(chr, pos) pairs are not in logical order"


def assign_centiMorgans(chr_pos_vector, ref_positions_cM):
    logger.info_once("Assigning centiMorgan recombination rates.")

    ref_chrom = np.array(ref_positions_cM.chrom).astype(int)
    ref_pos = np.array(ref_positions_cM.pos)
    ref_cm = np.array(ref_positions_cM.pos_cm)

    # NB the caller's order, asserted rather than imposed: position_cM is aligned to it.
    assert_logical_order(chr_pos_vector)

    # NB k, per query, is the first row at or after (contig, pos): the row the loop this replaces fast-forwarded to,
    #    found by a search within the query's contig. The arithmetic per branch is the loop's, element for element,
    #    so the result is bitwise it.
    chrom = np.array([x[0] for x in chr_pos_vector])
    pos = np.array([x[1] for x in chr_pos_vector])
    n_ref = len(ref_chrom)

    k = np.empty(len(pos), dtype=np.int64)
    for c in np.unique(chrom):
        at = chrom == c
        first, last = np.searchsorted(ref_chrom, c, "left"), np.searchsorted(ref_chrom, c, "right")
        k[at] = first + np.searchsorted(ref_pos[first:last], pos[at], "left")

    # NB k - 1 is -1 where k is 0, the last row, as the loop's ref_cm[k - 1] read it.
    inside = (k < n_ref) & (ref_chrom[np.minimum(k, n_ref - 1)] == chrom)
    after = inside & (k > 0) & (ref_chrom[k - 1] == chrom)
    start = inside & ~after

    position_cM = np.ones(len(chr_pos_vector)) * np.nan
    i, j = k[after], k[after] - 1
    position_cM[after] = ref_cm[j] + (pos[after] - ref_pos[j]) / (ref_pos[i] - ref_pos[j]) * (ref_cm[i] - ref_cm[j])

    # NB Extrapolation from Chromosome Start
    i = k[start]
    position_cM[start] = (pos[start] - 0) / (ref_pos[i] - 0) * (ref_cm[i] - 0)

    # NB Extrapolation Beyond Last Reference Point
    position_cM[~inside] = ref_cm[k[~inside] - 1]

    # NB given input [(chr1, start1), (chr1, end1), (chr2, start2), (chr2, end2), ...])
    #    return positions in centiMorgan.
    return position_cM


def get_sitewise_transmat(segment_key, df_gene_snp, geneticmap_file, nu, logphase_shift):
    """Phase switch probability per segment from recombination rate / genetic distance [cM].

    Args: `segment_key` e.g. block_id or bin_id; `df_gene_snp`; `geneticmap_file`; `nu`; `logphase_shift`.

    Returns:
        The log sitewise transition matrix.
    """
    logger.info(f"Constructing sitewise transition matrix for phasing given recombination rates.")

    ref_positions_cM = get_reference_recomb_rates(geneticmap_file)

    # NB sorted contig,start per block.
    sorted_chr_pos_first = df_gene_snp.groupby(segment_key).agg({"CHR": "first", "START": "first"})

    sorted_chr_pos_last = df_gene_snp.groupby(segment_key).agg({"CHR": "last", "END": "last"})

    if sorted_chr_pos_first.index.isna().any():
        logger.warning(f"Found ill-defined group with None entries for group.")

    # NB dataframe to list.
    sorted_chr_pos_first = list(zip(sorted_chr_pos_first.CHR.to_numpy(), sorted_chr_pos_first.START.to_numpy()))

    sorted_chr_pos_last = list(zip(sorted_chr_pos_last.CHR.to_numpy(), sorted_chr_pos_last.END.to_numpy()))

    # NB [(chr1, start1), (chr1, end1), (chr2, start2), (chr2, end2), ...]) construct ...
    chr_pos = [val for pair in zip(sorted_chr_pos_first, sorted_chr_pos_last) for val in pair]

    # NB blocks disjoint and in logical order, so [1::2] below selects each block's end.
    assert_logical_order(chr_pos)

    # NB positions in cM of [(chr1, start1), (chr1, end1), (chr2, start2), (chr2, end2), ...])
    position_cM = assign_centiMorgans(chr_pos, ref_positions_cM)

    # NB chr_pos used to identify chromosome switches.
    phase_switch_prob = compute_numbat_phase_switch_prob(position_cM, chr_pos, nu)

    # NB transition matrix for phasing.
    log_sitewise_transmat = np.minimum(np.log(0.5), np.log(phase_switch_prob) - logphase_shift)

    log_sitewise_transmat = log_sitewise_transmat[1::2]

    logger.info(
        f"Solved for (recombination based) sitewise transition matrix for phasing with shape={log_sitewise_transmat.shape}."
    )

    return log_sitewise_transmat
