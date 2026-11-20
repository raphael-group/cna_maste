"""Frozen copies of `hmrf.run_core_inference`'s `normal_lambda` and `emission.get_log_mu_shift` before T- #100 PR 1;
of `hmrf.pin_neutral_state`, `reindex_clones` and `merge_by_minspots` before T- #102 PR 3; of
`hmrf.clone_decoded_unary_of_fit` before T- #100 PR 5; of `emission.get_logsumexp_by_clone` before T- #157."""

import copy
from typing import Any

import numpy as np
from numba import njit

from cnamaste.config import start_time
from cnamaste.emission import CnaEmissionConfounders, get_log_emission
from cnamaste.hmm_nophasing import get_hmm_baf_eps, get_neutral_state, get_path_baf_profiles
from cnamaste.hmm_utils import get_normal_clone
from cnamaste.hmrf import get_nested_copy
from cnamaste.logger import get_logger
from cnamaste.spatio_genomic_counts import PseudobulkCounts
from legacy.spatio_genomic_counts import SpatioGenomicCounts

logger = get_logger(__name__, start_time=start_time)


@njit(nogil=True, cache=True, error_model="numpy")
def get_logsumexp_by_clone(
    log_mu: np.ndarray, log_state_weights: np.ndarray, log_lambda: np.ndarray, clone_lengths: np.ndarray
) -> np.ndarray:
    """`scipy.special.logsumexp(log sum_s w_gs mu_s + log_lambda)` over each clone's bins, guarded against `-inf`.

    Each bin's depth `log sum_s w_gs mu_s` first: one-hot weights (`get_hard_state_posteriors`) take it as the path's
    state's `log_mu` exactly; posteriors take its expectation.

    Args:
        log_mu: `(n_states,)` each state's log depth.
        log_state_weights: `(n_obs, n_states)` `log w_gs`, each bin's weight per state, its rows summing to one.
        log_lambda: `(n_obs,)` `log lambda_g`, each clone's baseline over its sum.
        clone_lengths: `(n_clones,)` bins per clone, clone c's the c-th block.

    Returns:
        The `(n_clones,)` sums.
    """
    n_obs, n_states = log_state_weights.shape
    log_depth = np.empty(n_obs, dtype=np.float64)
    for g in range(n_obs):
        top = -np.inf
        for s in range(n_states):
            top = max(top, log_state_weights[g, s] + log_mu[s])
        if np.isinf(top):
            log_depth[g] = top
        else:
            total = 0.0
            for s in range(n_states):
                total += np.exp(log_state_weights[g, s] + log_mu[s] - top)
            log_depth[g] = top + np.log(total)
    log_mu_shift = np.empty(len(clone_lengths), dtype=np.float64)
    start = 0
    for c in range(len(clone_lengths)):
        stop = start + clone_lengths[c]
        top = -np.inf
        for g in range(start, stop):
            top = max(top, log_depth[g] + log_lambda[g])
        if np.isinf(top):
            log_mu_shift[c] = top
        else:
            total = 0.0
            for g in range(start, stop):
                total += np.exp(log_depth[g] + log_lambda[g] - top)
            log_mu_shift[c] = top + np.log(total)
        start = stop
    return log_mu_shift


def get_normal_lambda(single_base_nb_mean: Any) -> Any:
    """`run_core_inference`'s `normal_lambda`, verbatim."""
    has_normal_lambda = single_base_nb_mean.count_positive()
    normal_lambda = None

    # NB baseline expression by summing over all clones; should be zero for BAF only.
    if not has_normal_lambda:
        pass
    else:
        # NB expect the normal baseline, scaled by the total number of transcripts per spot.
        with np.errstate(divide="ignore", invalid="ignore"):
            # TBC sum over all spots, lamba x total sample transcripts.
            normal_lambda = single_base_nb_mean.sum(axis=1)

            # NB lambda.
            normal_lambda /= single_base_nb_mean.sum()
    return normal_lambda


def get_log_mu_shift(
    parameters: Any, log_state_weights: Any, log_lambda: Any, clone_lengths: Any, normal_fraction: Any
) -> np.ndarray:
    """`emission.get_log_mu_shift`, verbatim, unmixed (`normal_fraction` None)."""
    assert normal_fraction is None
    log_mu = np.asarray(parameters.log_mu, dtype=np.float64).reshape(-1)
    return get_logsumexp_by_clone(
        log_mu,
        np.asarray(log_state_weights, dtype=np.float64),
        np.asarray(log_lambda, dtype=np.float64),
        np.asarray(clone_lengths, dtype=np.int64),
    )


def pin_neutral_state(res, fitted_baf=True):
    """Fix `log_mu`'s scale in place under `hmrf.apply_logmu_shift`; a no-op where every `log Z_c` is 0.

    Sets `hmm_nophasing.get_neutral_state` (the normal clone's most used balanced state) to `mu = 1` and moves each
    `log Z_c` by the same amount: the likelihood sees only `mu / Z_c`, so no likelihood or path moves.
    Without a shift there is no such scale; `ab_decode.decode` reads `mu` against the state nearest 1.

    Args:
        res: the fit, updated in place.
        fitted_baf: whether the neutral state reads the fitted `p_binom`.
    """
    shift = np.asarray(res.confounders.log_mu_shift, dtype=np.float64)
    if not np.any(shift):
        return
    paths = np.asarray(res.profile.pred_cnv).reshape(shift.size, -1).T
    neutral = get_neutral_state(paths, res.parameters.p_binom if fitted_baf else None)
    pin = float(np.asarray(res.parameters.log_mu).reshape(-1)[neutral])
    res.parameters = res.parameters._replace(log_mu=np.asarray(res.parameters.log_mu) - pin)
    res.confounders = res.confounders._replace(log_mu_shift=shift - pin)
    logger.info(f"logmu_shift: state {neutral} pinned to mu = 1, from log mu = {pin:.6f}.")


def reindex_clones(res_combine, posterior=None):
    """Renumber the clones: the normal clone 0, the rest by spot count, fewest first.

    The normal clone is `get_normal_clone`'s, the one whose BAF strays least from 0.5 outside the deadband. The
    permutation follows everything carried per clone -- the assignment, `pred_cnv` and `log_gamma` (concatenated
    along the genome or one column per clone), the confounders and `posterior` -- and nothing carried per state,
    which every clone shares. The result is a copy whose nested containers are copies, so the caller's is untouched.

    `run_cnamaste` calls it once, after the last merge, which is why the written tables name the normal clone 0.

    Args:
        res_combine: the fit.
        posterior: (n_spots, n_clones), or None.

    Returns:
        The renumbered result and the renumbered posterior, `posterior` itself where it has one column.
    """

    new_res_combine = get_nested_copy(res_combine)

    assignments = res_combine["new_assignment"]
    clone_labels = np.unique(assignments)
    n_clones = len(clone_labels)

    pred_cnv = res_combine["pred_cnv"]

    is_concatenated = pred_cnv.ndim == 1

    if is_concatenated:
        n_obs = len(pred_cnv) // n_clones
    else:
        n_obs = pred_cnv.shape[0]

    # NB one parameter per state, shared by every clone: nothing per-clone to permute.
    assert all(np.ndim(v) == 1 for v in res_combine.parameters[1:])

    # NB normal clone minimizes deviation from 0.5 (outside the EPS_BAF deadband); its balanced state is pinned.
    paths = pred_cnv.reshape(n_clones, n_obs).T if is_concatenated else pred_cnv
    cid_normal = get_normal_clone(get_path_baf_profiles(paths, res_combine.parameters.p_binom), get_hmm_baf_eps())

    unique_clones, spot_counts = np.unique(assignments, return_counts=True)

    mask_rest = unique_clones != cid_normal
    cid_rest = unique_clones[mask_rest]
    counts_rest = spot_counts[mask_rest]

    cid_rest_sorted = cid_rest[np.argsort(counts_rest)]

    reidx = np.concatenate(([cid_normal], cid_rest_sorted)).astype(int)

    logger.info(f"Remapping clone index: {cid_normal} (normal) to 0, otherwise sorted by spot count.")

    max_id = np.max(unique_clones)
    palette = np.zeros(max_id + 1, dtype=int)

    for new_idx, old_idx in enumerate(reidx):
        palette[old_idx] = new_idx

    new_res_combine["new_assignment"] = palette[assignments]

    if is_concatenated:
        concat_idx = np.concatenate([np.arange(c * n_obs, c * n_obs + n_obs) for c in reidx])

        new_res_combine["pred_cnv"] = pred_cnv[concat_idx]

        if "log_gamma" in res_combine.keys():
            new_res_combine["log_gamma"] = res_combine["log_gamma"][:, concat_idx]

    else:
        if pred_cnv.shape[1] > 1:
            new_res_combine["pred_cnv"] = pred_cnv[:, reidx]

        if "log_gamma" in res_combine.keys():
            log_gamma = res_combine["log_gamma"]
            if log_gamma.ndim == 3 and log_gamma.shape[2] > 1:
                new_res_combine["log_gamma"] = log_gamma[:, :, reidx]

    # NB each clone's log Z_c follows its clone.
    new_res_combine.confounders = CnaEmissionConfounders(*(np.asarray(v)[reidx] for v in res_combine.confounders))

    if posterior is not None and posterior.shape[1] > 1:
        new_posterior = copy.copy(posterior)[:, reidx]
    else:
        new_posterior = posterior

    return new_res_combine, new_posterior


# TODO FINAL
def merge_by_minspots(
    assignment,
    res,
    single_total_bb_RD,
    min_spots_thresholds=50,
    min_umicount_thresholds=0,
):
    n_clones = len(np.unique(assignment))
    if n_clones == 1:
        merged_groups = [[assignment[0]]]
        return merged_groups, res

    # NB genomic axis is concatenated across clones.
    n_obs = int(len(res["pred_cnv"]) / n_clones)
    new_assignment = copy.copy(assignment)

    unique_assignment = np.unique(new_assignment)

    # NB clones failing min_spots_thresholds or min_umicount_thresholds.
    insufficient_spots_clones = [c for c in unique_assignment if np.sum(new_assignment == c) < min_spots_thresholds]

    # NB find clones failing min_umicount_thresholds
    insufficient_umi_clones = [
        c
        for c in unique_assignment
        if SpatioGenomicCounts.of(single_total_bb_RD).spots(new_assignment == c).sum() < min_umicount_thresholds
    ]

    # NB log each condition separately
    logger.info(
        f"Found {len(insufficient_spots_clones)} clones with < {min_spots_thresholds} spots: {insufficient_spots_clones}"
    )
    logger.info(
        f"Found {len(insufficient_umi_clones)} clones with < {min_umicount_thresholds:_} SNP UMIs: {insufficient_umi_clones}"
    )

    failed_clones = [c for c in unique_assignment if c in insufficient_spots_clones or c in insufficient_umi_clones]
    logger.info(f"Found {len(failed_clones)} new clones failing thresholds on min. spots or min. SNP umis.")

    # NB find the remaining unique_assigment that satisfies both thresholds
    successful_clones = [c for c in unique_assignment if not c in failed_clones]

    if len(successful_clones) == 0:
        logger.error("All clones failed min. spots or min. SNP UMIs thresholds; cannot proceed with merging.")
        raise RuntimeError()

    # NB initial merging groups: each successful clone is its own group
    merging_groups = [[i] for i in successful_clones]

    if len(failed_clones) > 0:
        for c in failed_clones:
            # NB assigns failed clone to that with largest snp umis.
            idx_max = np.argmax(
                [
                    SpatioGenomicCounts.of(single_total_bb_RD).spots(new_assignment == c_prime).sum()
                    for c_prime in successful_clones
                ]
            )
            logger.warning(
                f"Assigning failed clone {c} to clone {[successful_clones[idx_max]]} (with largest SNP UMIs)."
            )

            merging_groups[idx_max].append(c)

    # NB re-map new_assignment according to merging_groups.
    map_clone_id = {}
    for i, x in enumerate(merging_groups):
        for z in x:
            map_clone_id[z] = i
    new_assignment = np.array([map_clone_id[x] for x in new_assignment])

    merged_res = get_nested_copy(res)
    merged_res["new_assignment"] = new_assignment
    merged_res["total_llf"] = np.nan

    # Extract the representative clone IDs to keep
    rep_clones = [c[0] for c in merging_groups]

    # NB expect an array of clones concatenated along the genomic axis,
    if res["pred_cnv"].ndim == 1:
        n_obs = len(res["pred_cnv"]) // n_clones
        merged_res["pred_cnv"] = np.concatenate(
            [res["pred_cnv"][(c * n_obs) : (c * n_obs + n_obs)] for c in rep_clones]
        )
        merged_res["log_gamma"] = np.hstack(
            [res["log_gamma"][:, (c * n_obs) : (c * n_obs + n_obs)] for c in rep_clones]
        )
    # NB expect an independent clone axis.
    else:
        merged_res["pred_cnv"] = res["pred_cnv"][:, rep_clones]
        merged_res["log_gamma"] = res["log_gamma"][:, :, rep_clones]

    # NB a merged clone keeps its representative's log Z_c, as it keeps its path.
    merged_res.confounders = CnaEmissionConfounders(*(np.asarray(v)[rep_clones] for v in res.confounders))

    return merging_groups, merged_res


def clone_decoded_unary_of_fit(counts, parameters, confounders, pred, *, phased):
    """`hmrf.clone_decoded_unary_of_fit` before T- #100 PR 5, verbatim.

    Args:
        counts: a `SpatioGenomicLabelledMultiCounts`, its baseline `normal_baseline_sized`.
        parameters: the fit's `CnaHmmParameters`.
        confounders: the fit's `CnaEmissionConfounders`, clone c's depth mean at base mu / Z_c.
        pred: `(n_clones, n_obs)` decoded states, `(state, phase)` pairs where `phased`.
        phased: whether `pred` carries the phase.

    Returns:
        The (n_spots, n_clones) log-likelihood field.
    """
    N = counts.n_spots
    paths = np.ascontiguousarray(pred, dtype=np.int64)

    # NB the weight of `sandbox.spot_assignment.compute_loglike_spot_assignment_given_emission`: valid allele over
    #    valid read-depth bins where both.
    single_base_nb_mean, single_total_bb_RD = counts.normal_baseline_sized, counts.total_snp_umis
    # NB axis 1: each spot's segments in the spot-major counts.
    nb_valid, bb_valid = single_base_nb_mean.count_positive(axis=1), single_total_bb_RD.count_positive(axis=1)
    weights = np.ones(N)
    both = (nb_valid > 0) & (bb_valid > 0)
    weights[both] = bb_valid[both] / nb_valid[both]

    by_spot = PseudobulkCounts(counts.umis, single_base_nb_mean, counts.alt_snp_umis, single_total_bb_RD, None, {})
    rdr, baf = get_log_emission(
        by_spot,
        parameters,
        confounders,
        layout="field",
        phased=phased,
        data_terms=True,
        path=paths,
        spot_weights=weights,
    )
    return rdr + baf
