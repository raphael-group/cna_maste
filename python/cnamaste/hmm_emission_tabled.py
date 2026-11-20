"""The clone decoded unary: each spot's log-likelihood under each clone's decoded path, without the dense emission.

The decoded path is over (state, phase) pairs where phased; unphased paths are at phase 0.
"""

from math import exp, lgamma, log, log1p
from typing import Any, NamedTuple

import numpy as np
from numba import njit, prange

from cnamaste.config import get_count_table_extent, get_global_config
from cnamaste.hmm_emission_bb import bb_logpmf_element, bb_switch_logpmf_element
from cnamaste.hmm_emission_nb import nb_logpmf_element
from cnamaste.hmm_utils import rising_factorial_threshold
from cnamaste.sparse import (
    sum_over_row_csr,
    sum_over_row_dense,
    sum_over_stored_entries,
    sum_over_stored_entries_and_columns,
    values_at_stored_entries,
)
from cnamaste.spatio_genomic_counts import (
    CountsEncoding,
    NormalBaselineSized,
    PseudobulkCounts,
    SpatioGenomicCounts,
    get_encoding_channels,
)


@njit(cache=True)
def count_table_size(counts, count_table_extent):
    """Return 1 + the largest count if every count is a non-negative integer up to `count_table_extent`, else 0.

    Args:
        counts: an array of counts.
        count_table_extent: the largest count a table covers, `hmm.count_table_extent`.

    Returns:
        The table extent, 0 for none.
    """
    top = 0.0
    for count in counts.ravel():
        if count < 0 or count != np.floor(count) or count > count_table_extent:
            return 0
        top = max(top, count)
    return int(top) + 1


class ClonePaths(NamedTuple):
    """Each clone's decoded state path, its phase and read-depth scale.

    Args:
        paths: (n_clones, n_obs) state per segment.
        phases: (n_clones, n_obs) phase per segment, 1 where the alleles are switched.
        scales: (n_clones,) read-depth scale.
    """

    paths: np.ndarray
    phases: np.ndarray
    scales: np.ndarray


class NBParameters(NamedTuple):
    """The negative binomial's parameters, per state.

    Args:
        mu: mean per unit baseline.
        alpha: dispersion.
        r: 1 / alpha.
        lgamma_r: lgamma(r).
    """

    mu: np.ndarray
    alpha: np.ndarray
    r: np.ndarray
    lgamma_r: np.ndarray


class BBParameters(NamedTuple):
    """The beta-binomial's parameters, per state.

    Args:
        alpha: B allele concentration.
        beta: A allele concentration.
        log_beta: log B(alpha, beta).
    """

    alpha: np.ndarray
    beta: np.ndarray
    log_beta: np.ndarray


class CountTables(NamedTuple):
    """The `lgamma` tables over the tabled integer counts j, read at column `equivalence_class[j]`, and whether the
    kernels read them.

    Args:
        tabled: whether the kernels read the tables.
        equivalence_class: the `labelling_unary` encoding's class of each count j, its column, -1 where unread.
        log_factorial: lgamma(j + 1).
        lgamma_count_plus_r: (n_states, n_slots) lgamma(j + r_s).
        lgamma_count_plus_alpha: (n_states, n_slots) lgamma(j + alpha_s).
        lgamma_count_plus_beta: (n_states, n_slots) lgamma(j + beta_s).
        lgamma_total_plus_alpha_beta: (n_states, n_slots) lgamma((j + alpha_s) + beta_s).
    """

    tabled: bool
    equivalence_class: np.ndarray
    log_factorial: np.ndarray
    lgamma_count_plus_r: np.ndarray
    lgamma_count_plus_alpha: np.ndarray
    lgamma_count_plus_beta: np.ndarray
    lgamma_total_plus_alpha_beta: np.ndarray


class DenseOperands(NamedTuple):
    """The dense arrays a kernel reads beside its driver, empty where unread.

    Args:
        baseline: (n_spots, n_obs) baseline, where the baseline is dense.
        b_counts: (n_spots, n_obs) B allele counts, where the allele depth is dense.
        spots_coverage: (n_spots,) the `NormalBaselineSized` factor, where the baseline is sized.
        rdr_normal: (n_obs,) the `NormalBaselineSized` factor, where the baseline is sized.
    """

    baseline: np.ndarray
    b_counts: np.ndarray
    spots_coverage: np.ndarray
    rdr_normal: np.ndarray


class FieldArgs(NamedTuple):
    """The kernels' shared arguments.

    Args:
        clones: the `ClonePaths`.
        nb: the `NBParameters`.
        bb: the `BBParameters`.
        tables: the `CountTables`.
        dense: the `DenseOperands`.
    """

    clones: ClonePaths
    nb: NBParameters
    bb: BBParameters
    tables: CountTables
    dense: DenseOperands


@njit(cache=True, inline="always")
def rdr_logpmf_tabled(count, baseline, segment, clone, args):
    """Return clone `clone`'s read-depth log-pmf at `segment`, `nb_logpmf_element`, tabled for integer counts.

    Args:
        count: the read count.
        baseline: the spot's baseline.
        segment: the segment index.
        clone: the clone index.
        args: a `FieldArgs`.

    Returns:
        The log-pmf.
    """
    state = args.clones.paths[clone, segment]
    nb_mean = baseline * args.clones.scales[clone] * args.nb.mu[state]
    if not args.tables.tabled:
        return nb_logpmf_element(
            count, nb_mean, args.nb.alpha[state], args.nb.r[state], args.nb.lgamma_r[state], lgamma(count + 1.0)
        )
    if nb_mean > 0.0:
        alpha_mean = args.nb.alpha[state] * nb_mean
        log1p_alpha_mean = log1p(alpha_mean)
        # NB at a zero count rise is lgamma(r) - lgamma(r) - lgamma(1) = 0.0, the last term a signed zero: bitwise.
        if count == 0 and 0.0 < alpha_mean < np.inf:
            return 0.0 - args.nb.r[state] * log1p_alpha_mean
        slot = args.tables.equivalence_class[int(count)]
        rise = args.tables.lgamma_count_plus_r[state, slot] - args.nb.lgamma_r[state] - args.tables.log_factorial[slot]
        return rise - args.nb.r[state] * log1p_alpha_mean + count * (log(alpha_mean) - log1p_alpha_mean)
    return 0.0


@njit(cache=True)
def baf_switch_logpmf_tabled(log_baf_nophase, b_count, total, state, args):
    """Return `bb_switch_logpmf_element` of the unphased log-pmf, tabled for integer counts, bitwise.

    Args:
        log_baf_nophase: the unphased log-pmf.
        b_count: the B allele count.
        total: the allele depth.
        state: the state index.
        args: a `FieldArgs`.

    Returns:
        The switched log-pmf.
    """
    alpha, beta = args.bb.alpha[state], args.bb.beta[state]
    if not args.tables.tabled or b_count > total:
        return bb_switch_logpmf_element(log_baf_nophase, b_count, total, alpha, beta)
    a_count = total - b_count
    b_slot, a_slot = args.tables.equivalence_class[int(b_count)], args.tables.equivalence_class[int(a_count)]
    lgamma_alpha, lgamma_beta = args.tables.lgamma_count_plus_alpha, args.tables.lgamma_count_plus_beta
    alpha_rest, beta_rest = alpha + total - b_count, beta + total - b_count
    # NB (alpha + n) - k is the table's (n - k) + alpha unless n + alpha rounds: lgamma it there.
    lgamma_alpha_rest = lgamma_alpha[state, a_slot] if alpha_rest == a_count + alpha else lgamma(alpha_rest)
    lgamma_beta_rest = lgamma_beta[state, a_slot] if beta_rest == a_count + beta else lgamma(beta_rest)
    return (
        log_baf_nophase
        + lgamma_beta[state, b_slot]
        - lgamma_alpha[state, b_slot]
        + lgamma_alpha_rest
        - lgamma_beta_rest
    )


@njit(cache=True, inline="always")
def baf_logpmf_tabled(b_count, total, segment, clone, args):
    """Return clone `clone`'s allele log-pmf at `segment` and its phase, `bb_logpmf_element`'s, tabled for integer counts.

    Exactly +0.0 at `total = b_count = 0` and phase 0, as `sum_over_stored_entries` requires.

    Args:
        b_count: the B allele count.
        total: the allele depth.
        segment: the segment index.
        clone: the clone index.
        args: a `FieldArgs`.

    Returns:
        The log-pmf.
    """
    state = args.clones.paths[clone, segment]
    if not args.tables.tabled:
        log_baf = bb_logpmf_element(b_count, total, args.bb.alpha[state], args.bb.beta[state])
    elif b_count > total:
        log_baf = 0.0
    else:
        classes, log_factorial = args.tables.equivalence_class, args.tables.log_factorial
        b_slot, total_slot, a_slot = classes[int(b_count)], classes[int(total)], classes[int(total - b_count)]
        coeff = log_factorial[total_slot] - log_factorial[b_slot] - log_factorial[a_slot]
        num = (
            args.tables.lgamma_count_plus_alpha[state, b_slot]
            + args.tables.lgamma_count_plus_beta[state, a_slot]
            - args.tables.lgamma_total_plus_alpha_beta[state, total_slot]
        )
        log_baf = coeff + num - args.bb.log_beta[state]
    if args.clones.phases[clone, segment] == 0:
        return log_baf
    return baf_switch_logpmf_tabled(log_baf, b_count, total, state, args)


@njit(cache=True, inline="always")
def rdr_logpmf_at_count(count, spot, segment, clone, args):
    """Return `rdr_logpmf_tabled` at a count, the baseline the sized product or read from the dense table.

    Args:
        count: the read count.
        spot: the spot index.
        segment: the segment index.
        clone: the clone index.
        args: a `FieldArgs`.

    Returns:
        The log-pmf.
    """
    if args.dense.rdr_normal.size:
        baseline = args.dense.spots_coverage[spot] * args.dense.rdr_normal[segment]
    else:
        baseline = args.dense.baseline[spot, segment]
    return rdr_logpmf_tabled(count, baseline, segment, clone, args)


@njit(cache=True, inline="always")
def rdr_logpmf_at_stored_baseline(baseline, count, spot, segment, clone, args):
    """Return `rdr_logpmf_tabled` at a stored baseline and its count: the sparse-baseline arm's element.

    Args:
        baseline: the stored baseline.
        count: the read count.
        spot: the spot index.
        segment: the segment index.
        clone: the clone index.
        args: a `FieldArgs`.

    Returns:
        The log-pmf.
    """
    return rdr_logpmf_tabled(count, baseline, segment, clone, args)


@njit(cache=True, inline="always")
def baf_logpmf_at_stored_total(total, b_count, spot, segment, clone, args):
    """Return `baf_logpmf_tabled` at a stored depth and its B count: the sparse-depth arm's element.

    Args:
        total: the stored allele depth.
        b_count: the B allele count.
        spot: the spot index.
        segment: the segment index.
        clone: the clone index.
        args: a `FieldArgs`.

    Returns:
        The log-pmf.
    """
    return baf_logpmf_tabled(b_count, total, segment, clone, args)


@njit(cache=True, inline="always")
def baf_logpmf_at_dense_total(total, spot, segment, clone, args):
    """Return `baf_logpmf_tabled` at a depth, the B count read from the dense table.

    Args:
        total: the allele depth.
        spot: the spot index.
        segment: the segment index.
        clone: the clone index.
        args: a `FieldArgs`.

    Returns:
        The log-pmf.
    """
    return baf_logpmf_tabled(args.dense.b_counts[spot, segment], total, segment, clone, args)


@njit(parallel=True, cache=True)
def rdr_field_dense(X_nb, n_clones, clones, nb, bb, tables, dense):
    """Return the read-depth log-likelihood per spot and clone, dense counts and baseline.

    Args:
        X_nb: (n_spots, n_obs) read counts.
        n_clones: the number of clones.
        clones, nb, bb, tables, dense: the `FieldArgs` fields, apart: a parfor takes no nested tuple.

    Returns:
        The (n_spots, n_clones) field.
    """
    field = np.empty((X_nb.shape[0], n_clones))
    for spot in prange(X_nb.shape[0]):
        args = FieldArgs(clones, nb, bb, tables, dense)
        for clone in range(n_clones):
            field[spot, clone] = sum_over_row_dense(rdr_logpmf_at_count, X_nb, spot, clone, args)
    return field


@njit(parallel=True, cache=True)
def rdr_field_csr(n_obs, indptr, indices, data, n_clones, clones, nb, bb, tables, dense):
    """Return the read-depth log-likelihood per spot and clone, CSR counts and a dense baseline.

    Args:
        n_obs: the number of segments.
        indptr: the counts' CSR row pointers.
        indices: the counts' CSR column indices.
        data: the counts' CSR values.
        n_clones: the number of clones.
        clones, nb, bb, tables, dense: the `FieldArgs` fields, apart: a parfor takes no nested tuple.

    Returns:
        The (n_spots, n_clones) field.
    """
    field = np.empty((len(indptr) - 1, n_clones))
    for spot in prange(len(indptr) - 1):
        args = FieldArgs(clones, nb, bb, tables, dense)
        for clone in range(n_clones):
            field[spot, clone] = sum_over_row_csr(rdr_logpmf_at_count, n_obs, indptr, indices, data, spot, clone, args)
    return field


@njit(parallel=True, cache=True)
def rdr_field_sized_csr(indptr, indices, data, coverage_class, coverages, n_clones, clones, nb, bb, tables, dense):
    """Return `rdr_field_csr` on the sized baseline, its zero counts summed once per spot UMI total: each total's sum
    of the zero count's log-pmf over every segment, then each stored count's log-pmf less the zero count's.

    The same terms as `rdr_field_csr`, summed in another order: a stated difference, to rounding.

    Args:
        indptr: the counts' CSR row pointers.
        indices: the counts' CSR column indices.
        data: the counts' CSR values.
        coverage_class: (n_spots,) each spot's index into `coverages`.
        coverages: the distinct `spots_coverage`.
        n_clones: the number of clones.
        clones, nb, bb, tables, dense: the `FieldArgs` fields, apart: a parfor takes no nested tuple.

    Returns:
        The (n_spots, n_clones) field.
    """
    zero = np.zeros(1, data.dtype)[0]
    rdr_normal = dense.rdr_normal
    zeros = np.empty((coverages.size, n_clones))
    for unique in prange(coverages.size):
        args = FieldArgs(clones, nb, bb, tables, dense)
        for clone in range(n_clones):
            total = 0.0
            for segment in range(rdr_normal.size):
                total += rdr_logpmf_tabled(zero, coverages[unique] * rdr_normal[segment], segment, clone, args)
            zeros[unique, clone] = total
    field = np.empty((len(indptr) - 1, n_clones))
    for spot in prange(len(indptr) - 1):
        args = FieldArgs(clones, nb, bb, tables, dense)
        for clone in range(n_clones):
            total = zeros[coverage_class[spot], clone]
            for entry in range(indptr[spot], indptr[spot + 1]):
                segment = indices[entry]
                count_term = rdr_logpmf_at_count(data[entry], spot, segment, clone, args)
                total += count_term - rdr_logpmf_at_count(zero, spot, segment, clone, args)
            field[spot, clone] = total
    return field


@njit(parallel=True, cache=True)
def rdr_field_stored_baseline(indptr, indices, baseline, counts, n_clones, clones, nb, bb, tables, dense):
    """Return the read-depth log-likelihood per spot and clone, summed over the stored baseline.

    Args:
        indptr: the baseline's CSR row pointers.
        indices: the baseline's CSR column indices.
        baseline: the stored baselines.
        counts: the read counts at the stored entries.
        n_clones: the number of clones.
        clones, nb, bb, tables, dense: the `FieldArgs` fields, apart: a parfor takes no nested tuple.

    Returns:
        The (n_spots, n_clones) field.
    """
    field = np.empty((len(indptr) - 1, n_clones))
    for spot in prange(len(indptr) - 1):
        args = FieldArgs(clones, nb, bb, tables, dense)
        for clone in range(n_clones):
            field[spot, clone] = sum_over_stored_entries(
                rdr_logpmf_at_stored_baseline, indptr, indices, baseline, counts, spot, clone, args
            )
    return field


@njit(parallel=True, cache=True)
def baf_field_stored_total(
    indptr, indices, total, counts, switched_indptr, switched_segments, n_clones, clones, nb, bb, tables, dense
):
    """Return the allele log-likelihood per spot and clone, summed over the stored depths and the switched segments.

    Args:
        indptr: the depth's CSR row pointers.
        indices: the depth's CSR column indices.
        total: the stored allele depths.
        counts: the B allele counts at the stored entries.
        switched_indptr: (n_clones + 1,) row pointers of each clone's phase 1 segments.
        switched_segments: each clone's phase 1 segments, sorted.
        n_clones: the number of clones.
        clones, nb, bb, tables, dense: the `FieldArgs` fields, apart: a parfor takes no nested tuple.

    Returns:
        The (n_spots, n_clones) field.
    """
    field = np.empty((len(indptr) - 1, n_clones))
    for spot in prange(len(indptr) - 1):
        args = FieldArgs(clones, nb, bb, tables, dense)
        for clone in range(n_clones):
            switched = switched_segments[switched_indptr[clone] : switched_indptr[clone + 1]]
            field[spot, clone] = sum_over_stored_entries_and_columns(
                baf_logpmf_at_stored_total, indptr, indices, total, counts, switched, spot, clone, args
            )
    return field


@njit(parallel=True, cache=True)
def baf_field_dense_total(total, n_clones, clones, nb, bb, tables, dense):
    """Return the allele log-likelihood per spot and clone, dense depths and B counts.

    Args:
        total: (n_spots, n_obs) allele depths.
        n_clones: the number of clones.
        clones, nb, bb, tables, dense: the `FieldArgs` fields, apart: a parfor takes no nested tuple.

    Returns:
        The (n_spots, n_clones) field.
    """
    field = np.empty((total.shape[0], n_clones))
    for spot in prange(total.shape[0]):
        args = FieldArgs(clones, nb, bb, tables, dense)
        for clone in range(n_clones):
            field[spot, clone] = sum_over_row_dense(baf_logpmf_at_dense_total, total, spot, clone, args)
    return field


@njit(cache=True)
def get_state_parameters(log_mu, alphas, p_binom, taus, EPS):
    """Return the negative binomial and beta-binomial parameters per state.

    Args:
        log_mu: log mean per state.
        alphas: NB dispersion per state, floored at 1e-10.
        p_binom: B allele frequency per state.
        taus: beta-binomial concentration per state.
        EPS: the beta-binomial parameters' floor.

    Returns:
        The `NBParameters` and `BBParameters`.
    """
    n_states = log_mu.shape[0]
    mu, alpha, r, lgamma_r = np.empty(n_states), np.empty(n_states), np.empty(n_states), np.empty(n_states)
    bb_alpha, bb_beta, log_beta = np.empty(n_states), np.empty(n_states), np.empty(n_states)
    for state in range(n_states):
        mu[state] = exp(log_mu[state])
        alpha[state] = max(alphas[state], 1.0e-10)
        r[state] = 1.0 / alpha[state]
        lgamma_r[state] = lgamma(r[state])
        bb_alpha[state] = max(p_binom[state] * taus[state], EPS)
        bb_beta[state] = max((1.0 - p_binom[state]) * taus[state], EPS)
        log_beta[state] = lgamma(bb_alpha[state]) + lgamma(bb_beta[state]) - lgamma(bb_alpha[state] + bb_beta[state])
    return NBParameters(mu, alpha, r, lgamma_r), BBParameters(bb_alpha, bb_beta, log_beta)


@njit(parallel=True, cache=True)
def get_count_tables(counts, r, bb_alpha, bb_beta):
    """Tabulate the `lgamma` arguments of `bb_logpmf_element` and `nb_logpmf_element` at each tabled count j.

    Args:
        counts: the tabled integer counts, one per column.
        r: NB 1 / dispersion per state.
        bb_alpha: beta-binomial B concentration per state.
        bb_beta: beta-binomial A concentration per state.

    Returns:
        lgamma(j + 1), lgamma(j + r_s), lgamma(j + alpha_s), lgamma(j + beta_s), lgamma((j + alpha_s) + beta_s).
    """
    n_states, n_slots = r.shape[0], counts.shape[0]
    log_factorial, lgamma_count_plus_r = np.empty(n_slots), np.empty((n_states, n_slots))
    lgamma_count_plus_alpha = np.empty((n_states, n_slots))
    lgamma_count_plus_beta = np.empty((n_states, n_slots))
    lgamma_total_plus_alpha_beta = np.empty((n_states, n_slots))
    for slot in prange(n_slots):
        count = float(counts[slot])
        log_factorial[slot] = lgamma(count + 1.0)
        for state in range(n_states):
            lgamma_count_plus_r[state, slot] = lgamma(count + r[state])
            lgamma_count_plus_alpha[state, slot] = lgamma(count + bb_alpha[state])
            lgamma_count_plus_beta[state, slot] = lgamma(count + bb_beta[state])
            lgamma_total_plus_alpha_beta[state, slot] = lgamma(count + bb_alpha[state] + bb_beta[state])
    return (
        log_factorial,
        lgamma_count_plus_r,
        lgamma_count_plus_alpha,
        lgamma_count_plus_beta,
        lgamma_total_plus_alpha_beta,
    )


def get_count_values(counts: SpatioGenomicCounts) -> Any:
    """Return the values `count_table_size` scans: a CSR's stored entries, or the dense array.

    Args:
        counts: dense or CSR counts.

    Returns:
        The values array.
    """
    return counts.values.data if counts.sparse else counts.values


def get_switched_segments(phases: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return each clone's phase 1 segments as CSR rows: `(indptr, segments)`, sorted within a clone.

    Args:
        phases: (n_clones, n_obs) phase per segment.

    Returns:
        The (n_clones + 1,) row pointers and the segments.
    """
    indptr = np.concatenate(([0], np.cumsum(np.count_nonzero(phases, axis=1))))
    return indptr.astype(np.int64), np.nonzero(phases)[1].astype(np.int64)


def get_labelling_unary_counts(
    X_nb: SpatioGenomicCounts,
    base: SpatioGenomicCounts | NormalBaselineSized,
    X_bb: SpatioGenomicCounts,
    total: SpatioGenomicCounts,
) -> PseudobulkCounts:
    """The counts the labelling unary reads: the read counts' values, and the B counts and depths at the depth's
    stored entries where it is CSR, else every entry.

    Args:
        X_nb: (n_spots, n_obs) read counts.
        base: (n_spots, n_obs) baseline.
        X_bb: (n_spots, n_obs) B allele counts.
        total: (n_spots, n_obs) allele depths.

    Returns:
        The `PseudobulkCounts` of those values, no encoding.
    """
    if total.sparse:
        totals, b_counts = total.values.data, total.get_held_values_at_stored_entries(X_bb)
    else:
        totals, b_counts = total.values, X_bb.toarray()
    return PseudobulkCounts(get_count_values(X_nb), base, b_counts, totals, None, {})


def get_labelling_unary_encoding(X_nb, base, X_bb, total) -> CountsEncoding:
    """Return the `labelling_unary` encoding of the counts the labelling unary reads (`get_labelling_unary_counts`):
    the count tables' classes and unique counts, built once per counts and handed to every field call.

    Args:
        X_nb: (n_spots, n_obs) read counts.
        base: (n_spots, n_obs) baseline, dense, CSR or `NormalBaselineSized`.
        X_bb: (n_spots, n_obs) B allele counts.
        total: (n_spots, n_obs) allele depths.

    Returns:
        The `CountsEncoding`.
    """
    X_nb, X_bb, total = (SpatioGenomicCounts.of(a) for a in (X_nb, X_bb, total))
    read = get_labelling_unary_counts(X_nb, base, X_bb, total)
    key = "labelling_unary"
    return read.update_encoding(key, get_encoding_channels()[key], common_zero_depth=False).get_encoding(key)


def get_clone_decoded_fields(
    X_nb, base, X_bb, total, paths, log_mu, alphas, p_binom, taus, scales, EPS=1e-10, *, encoding
):
    """Return each spot's read-depth and B-allele log-likelihoods under each clone's decoded path, bitwise the dense
    two-step unary; to rounding on the sized baseline with CSR read counts (`rdr_field_sized_csr`).

    Args:
        X_nb: (n_spots, n_obs) read counts.
        base: (n_spots, n_obs) baseline, dense, CSR or `NormalBaselineSized`.
        X_bb: (n_spots, n_obs) B allele counts.
        total: (n_spots, n_obs) allele depths.
        paths: (n_clones, n_obs) decoded pair p per segment: state p at phase 0 below n_states, else p - n_states at 1.
        log_mu: log mean per state.
        alphas: NB dispersion per state.
        p_binom: B allele frequency per state.
        taus: beta-binomial concentration per state.
        scales: (n_clones,) read-depth scale.
        EPS: the beta-binomial parameters' floor.
        encoding: these counts' `labelling_unary` encoding (`get_labelling_unary_encoding`), built once per counts.

    Returns:
        The (n_spots, n_clones) read-depth and B-allele log-likelihoods.
    """
    X_nb, X_bb, total = (SpatioGenomicCounts.of(a) for a in (X_nb, X_bb, total))
    sized = isinstance(base, NormalBaselineSized)
    base = base if sized else SpatioGenomicCounts.of(base)
    n_obs, n_clones, n_states = X_nb.shape[1], paths.shape[0], log_mu.shape[0]
    if paths.size and (paths.min() < 0 or paths.max() >= 2 * n_states):
        raise ValueError(f"paths span [{paths.min()}, {paths.max()}], not the 2 x {n_states} (state, phase) pairs.")
    clones = ClonePaths(paths % n_states, paths // n_states, scales)
    switched_indptr, switched_segments = get_switched_segments(clones.phases)
    # NB contiguous and writable, whatever the caller held: one compiled kernel for every stage and replay.
    state = (np.require(v, dtype=np.float64, requirements=["C", "W"]) for v in (log_mu, alphas, p_binom, taus))
    nb, bb = get_state_parameters(*state, EPS)

    read = get_labelling_unary_counts(X_nb, base, X_bb, total)
    totals, b_counts = read.total_snp_umis, read.alt_snp_umis

    count_table_extent = get_count_table_extent(get_global_config())
    extents = tuple(count_table_size(get_count_values(a), count_table_extent) for a in (X_nb, X_bb, total))
    extent = 0 if min(extents) == 0 else max(extents)
    # NB the tables do not reproduce bb_logpmf_element's rising factorials at or above the threshold.
    if np.any(bb.alpha + bb.beta >= rising_factorial_threshold):
        extent = 0
    classes, counts = np.full(1, -1, dtype=np.int32), np.zeros(0, dtype=np.int64)
    if extent > 0:
        classes, counts = encoding.equivalence_class[0], encoding.unique_counts[0]
    tables = CountTables(extent > 0, classes, *get_count_tables(counts, nb.r, bb.alpha, bb.beta))

    unread = np.zeros(0)
    dense = DenseOperands(
        baseline=np.zeros((0, 0)) if sized or base.sparse else np.asarray(base.values, dtype=np.float64),
        b_counts=b_counts if not total.sparse else np.zeros((0, 0), dtype=X_bb.dtype),
        spots_coverage=base.spots_coverage if sized else unread,
        rdr_normal=base.rdr_normal if sized else unread,
    )
    args = FieldArgs(clones=clones, nb=nb, bb=bb, tables=tables, dense=dense)

    if not sized and base.sparse:
        stored_baseline = base.values
        rdr = rdr_field_stored_baseline(
            stored_baseline.indptr,
            stored_baseline.indices,
            stored_baseline.data.astype(np.float64),
            values_at_stored_entries(stored_baseline, X_nb.values),
            n_clones,
            *args,
        )
    elif X_nb.sparse and sized:
        counts_csr = X_nb.values
        coverages, coverage_class = np.unique(base.spots_coverage, return_inverse=True)
        rdr = rdr_field_sized_csr(
            counts_csr.indptr,
            counts_csr.indices,
            counts_csr.data,
            coverage_class.astype(np.int64),
            coverages,
            n_clones,
            *args,
        )
    elif X_nb.sparse:
        counts_csr = X_nb.values
        rdr = rdr_field_csr(n_obs, counts_csr.indptr, counts_csr.indices, counts_csr.data, n_clones, *args)
    else:
        rdr = rdr_field_dense(X_nb.values, n_clones, *args)

    if total.sparse:
        stored_total = total.values
        baf = baf_field_stored_total(
            stored_total.indptr,
            stored_total.indices,
            totals,
            b_counts,
            switched_indptr,
            switched_segments,
            n_clones,
            *args,
        )
    else:
        baf = baf_field_dense_total(total.values, n_clones, *args)

    return rdr, baf
