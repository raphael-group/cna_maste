"""Frozen copies of the forward-backward kernels before T- #102 PR 2, verbatim: `hmm_nophasing.forward_recursion`,
`backward_recursion`, `get_lengths_start_indices`; `hmm_phased.update_phased_transmat`, `phased_forward_lattice`,
`phased_backward_lattice`; `hmm.compute_copy_state_posterior`."""

import numpy as np
import scipy.special
from numba import njit, prange

from cnamaste.hmm_utils import logsumexp


@njit(parallel=True, cache=True)
def forward_recursion(log_emission, log_startprob, log_transmat, starts, lengths):
    """`log_alpha[j, t] = log P(o_1 ... o_t, q_t = j)`, one sequence per thread.

    Args: `log_emission` (n_states, n_obs); `log_startprob` (n_states,); `log_transmat`; `starts`, `lengths`.
    Returns: the `(n_states, n_obs)` forward log probabilities.
    """
    n_states, n_obs = log_emission.shape
    log_alpha = np.zeros((n_states, n_obs))

    for k in prange(lengths.size):
        first, le = starts[k], lengths[k]
        buf = np.zeros(n_states)
        log_alpha[:, first] = log_startprob + log_emission[:, first]

        for t in range(1, le):
            for j in range(n_states):
                for i in range(n_states):
                    buf[i] = log_alpha[i, first + t - 1] + log_transmat[i, j]

                log_alpha[j, first + t] = logsumexp(buf) + log_emission[j, first + t]

    return log_alpha


@njit(parallel=True, cache=True)
def backward_recursion(log_emission, log_transmat, starts, lengths):
    """`log_beta[i, t] = log P(o_{t+1} ... o_T | q_t = i)`, one sequence per thread.

    Args: `log_emission` (n_states, n_obs); `log_transmat`; `starts`, `lengths` per sequence.
    Returns: the `(n_states, n_obs)` backward log probabilities.
    """
    n_states, n_obs = log_emission.shape
    log_beta = np.zeros((n_states, n_obs))

    for k in prange(lengths.size):
        first, le = starts[k], lengths[k]
        buf = np.zeros(n_states)
        log_beta[:, first + le - 1] = 0.0

        for t in range(le - 2, -1, -1):
            for i in range(n_states):
                for j in range(n_states):
                    buf[j] = log_beta[j, first + t + 1] + log_transmat[i, j] + log_emission[j, first + t + 1]

                log_beta[i, first + t] = logsumexp(buf)

    return log_beta


def get_lengths_start_indices(lengths):
    """Each sequence's first column, as the recursions index them.

    Args: `lengths` per sequence.
    Returns: the start indices.
    """
    starts = np.zeros(lengths.size + 1, dtype=np.int64)

    for k in range(lengths.size):
        starts[k + 1] = starts[k] + lengths[k]

    return starts


@njit(nogil=True, cache=True, error_model="numpy")
def update_phased_transmat(
    out_transmat,
    n_states,
    log_transmat,
    self_trans,
    switch_trans,
    penalize_phase_only_on_same_cnv,
    log_half,
):
    if penalize_phase_only_on_same_cnv:
        # NB base case: All transitions split equally between phases (0 penalty)
        out_transmat[:n_states, :n_states] = log_half + log_transmat
        out_transmat[:n_states, n_states:] = log_half + log_transmat
        out_transmat[n_states:, :n_states] = log_half + log_transmat
        out_transmat[n_states:, n_states:] = log_half + log_transmat

        # NB overwrite __only__ the diagonals of the cnv blocks (where cnv state is conserved)
        for i in range(n_states):
            # Phase 0 -> Phase 0
            out_transmat[i, i] = self_trans + log_transmat[i, i]
            # Phase 0 -> Phase 1
            out_transmat[i, i + n_states] = switch_trans + log_transmat[i, i]
            # Phase 1 -> Phase 0
            out_transmat[i + n_states, i] = switch_trans + log_transmat[i, i]
            # Phase 1 -> Phase 1
            out_transmat[i + n_states, i + n_states] = self_trans + log_transmat[i, i]
    else:
        # NB original behavior:  apply sitewise phase matrices globally
        out_transmat[:n_states, :n_states] = self_trans + log_transmat
        out_transmat[:n_states, n_states:] = switch_trans + log_transmat
        out_transmat[n_states:, :n_states] = switch_trans + log_transmat
        out_transmat[n_states:, n_states:] = self_trans + log_transmat


@njit(nogil=True, cache=True, error_model="numpy")
def phased_forward_lattice(
    lengths,
    log_transmat,
    log_startprob,
    log_emission,
    log_sitewise_transmat,
    penalize_phase_only_on_same_cnv,
):
    n_paired_states, n_obs, _ = log_emission.shape
    n_states = int(np.ceil(n_paired_states / 2))

    log_sitewise_self_transmat = np.log(1.0 - np.exp(log_sitewise_transmat))
    log_alpha = np.zeros((n_paired_states, n_obs))
    buf = np.zeros(n_paired_states)

    log_half = np.log(0.5)
    combined_log_startprob = log_half + np.append(log_startprob, log_startprob)
    combined_transmat = np.empty((n_paired_states, n_paired_states))

    cumlen = 0
    for le in lengths:
        for j in range(n_paired_states):
            log_alpha[j, cumlen] = combined_log_startprob[j] + np.sum(log_emission[j, cumlen, :])

        for t in range(1, le):
            idx = cumlen + t - 1

            update_phased_transmat(
                combined_transmat,
                n_states,
                log_transmat,
                log_sitewise_self_transmat[idx],
                log_sitewise_transmat[idx],
                penalize_phase_only_on_same_cnv,
                log_half,
            )

            for j in range(n_paired_states):
                for i in range(n_paired_states):
                    buf[i] = log_alpha[i, idx] + combined_transmat[i, j]

                log_alpha[j, cumlen + t] = logsumexp(buf) + np.sum(log_emission[j, cumlen + t, :])

        cumlen += le

    return log_alpha


@njit(nogil=True, cache=True, error_model="numpy")
def phased_backward_lattice(
    lengths,
    log_transmat,
    log_startprob,
    log_emission,
    log_sitewise_transmat,
    penalize_phase_only_on_same_cnv,
):
    n_paired_states, n_obs, _ = log_emission.shape
    n_states = int(np.ceil(n_paired_states / 2))
    # NB beta does not depend on the start probabilities; checked against the states, as hmm_nophasing's.
    assert len(log_startprob) == n_states, "Length of startprob_ must be the number of copy states"

    log_sitewise_self_transmat = np.log(1.0 - np.exp(log_sitewise_transmat))
    log_beta = np.zeros((n_paired_states, n_obs))
    buf = np.zeros(n_paired_states)

    log_half = np.log(0.5)
    combined_transmat = np.empty((n_paired_states, n_paired_states))

    cumlen = 0
    for le in lengths:
        log_beta[:, cumlen + le - 1] = 0.0

        for t in range(le - 2, -1, -1):
            idx = cumlen + t

            update_phased_transmat(
                combined_transmat,
                n_states,
                log_transmat,
                log_sitewise_self_transmat[idx],
                log_sitewise_transmat[idx],
                penalize_phase_only_on_same_cnv,
                log_half,
            )

            for i in range(n_paired_states):
                for j in range(n_paired_states):
                    buf[j] = (
                        log_beta[j, cumlen + t + 1]
                        + combined_transmat[i, j]
                        + np.sum(log_emission[j, cumlen + t + 1, :])
                    )

                log_beta[i, cumlen + t] = logsumexp(buf)

        cumlen += le

    return log_beta


def compute_copy_state_posterior(
    log_alpha,
    log_beta,
):
    # NB log_gamma (n_states * n_observations), potentially concatenated by clone.
    log_gamma = log_alpha + log_beta

    # NB an observation no state can emit: every entry of its column -inf, which normalising would make NaN;
    #    the check read sum(log_gamma) == 0, a sum of logs.
    if np.any(np.all(np.isneginf(log_gamma), axis=0)):
        raise RuntimeError()

    # NB normalize across states for each observation.
    log_gamma -= scipy.special.logsumexp(log_gamma, axis=0)

    return log_gamma
