"""The beta-binomial allele log-pmf, `eqn:bb_emission`: one element, its phase switch, a state's vector, an array
and the dense `(n_states, n_obs, n_spots)` field."""

from math import lgamma

import numpy as np
from numba import njit, prange

from cnamaste.hmm_utils import log_rising, rising_factorial_threshold


@njit(nogil=True, cache=True, inline="always", fastmath=False, error_model="numpy")
def bb_logpmf_element(k, n, alpha, beta, data_terms=True):
    """The beta-binomial log-pmf; `data_terms` keeps log C(n, k), the term with no parameter in it.

    Args:
        k: count.
        n: total.
        alpha: shape parameters.
        beta: shape parameters.
        data_terms: include log C(n, k).

    Returns:
        The log-pmf.
    """
    if alpha <= 0.0 or beta <= 0.0 or n < 0 or k < 0 or k > n:
        return 0.0

    if alpha + beta < rising_factorial_threshold:
        log_beta_num = lgamma(k + alpha) + lgamma(n - k + beta) - lgamma(n + alpha + beta)
        log_beta_denom = lgamma(alpha) + lgamma(beta) - lgamma(alpha + beta)
    else:
        # NB at large concentration the lgamma differences cancel (5e-5 nats at tau 1e10, 0.41 at 1e14):
        #    each rising factorial as its sum of logs, which tends to the binomial's k log p + (n - k) log(1 - p).
        log_beta_num = log_rising(alpha, k) + log_rising(beta, n - k) - log_rising(alpha + beta, n)
        log_beta_denom = 0.0

    if data_terms:
        log_binom_coeff = lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)
        return log_binom_coeff + log_beta_num - log_beta_denom
    else:
        return log_beta_num - log_beta_denom


@njit(nogil=True, cache=True, error_model="numpy")
def bb_logpmf_vector(obs, total, p_binom, tau, out, EPS=1e-10, data_terms=True):
    """`bb_logpmf_element` per count for one state, written in place to `out`.

    Args:
        obs: (n,) B-allele counts.
        total: (n,) allele totals.
        p_binom: the state's B-allele fraction.
        tau: the state's concentration.
        out: (n,) output.
        EPS: shape floor.
        data_terms: include log C(n, k).
    """
    alpha = max(p_binom * tau, EPS)
    beta = max((1.0 - p_binom) * tau, EPS)

    for i in range(len(obs)):
        out[i] = bb_logpmf_element(obs[i], total[i], alpha, beta, data_terms)


@njit(nogil=True, cache=True, error_model="numpy")
def bb_logpmf_vector_by_element(obs, total, alpha, beta, out, data_terms, EPS=1e-10):
    """`bb_logpmf_element` per count, each with its own shape parameters floored at `EPS`, written in place to `out`.

    Args:
        obs: (n,).
        total: (n,).
        alpha: (n,).
        beta: (n,).
        out: (n,) output.
        data_terms: include log C(n, k).
        EPS: shape floor.
    """
    for i in range(len(obs)):
        out[i] = bb_logpmf_element(obs[i], total[i], max(alpha[i], EPS), max(beta[i], EPS), data_terms)


@njit(nogil=True, cache=True, inline="always", error_model="numpy")
def bb_switch_logpmf_element(log_baf_nophase, k, n, alpha, beta):
    """One count's beta-binomial log-pmf with the alleles switched, from its unphased `bb_logpmf_element`.

    Args:
        log_baf_nophase: the unphased log-pmf.
        k: count.
        n: total.
        alpha: floored shape parameters.
        beta: floored shape parameters.

    Returns:
        The switched log-pmf.
    """
    if alpha + beta < rising_factorial_threshold:
        return log_baf_nophase + lgamma(beta + k) - lgamma(alpha + k) + lgamma(alpha + n - k) - lgamma(beta + n - k)
    # NB the same difference as rising factorials, which cancel no large lgammas.
    return (
        log_baf_nophase
        + log_rising(beta, k)
        - log_rising(alpha, k)
        + log_rising(alpha, n - k)
        - log_rising(beta, n - k)
    )


@njit(nogil=True, cache=True, error_model="numpy")
def bb_switch_logpmf_vector(log_baf_nophase, obs, total, p_binom_s, tau_s, EPS=1e-10):
    """Apply the phase switch to unphased beta-binomial log emissions over compressed 1-D arrays.

    Args:
        log_baf_nophase: (n_states, n_unique).
        obs: (n_unique,).
        total: (n_unique,).
        p_binom_s: (n_states,) one spot.
        tau_s: (n_states,) one spot.

    Returns:
        The (n_states, n_unique) log emission with the alleles switched.
    """
    n_states, n_unique = log_baf_nophase.shape
    out = np.empty_like(log_baf_nophase)

    for i in range(n_states):
        alpha = max(p_binom_s[i] * tau_s[i], EPS)
        beta = max((1.0 - p_binom_s[i]) * tau_s[i], EPS)
        for j in range(n_unique):
            out[i, j] = bb_switch_logpmf_element(log_baf_nophase[i, j], obs[j], total[j], alpha, beta)
    return out


def bb_logpmf_array(obs, total, p_binom, tau, *, data_terms):
    """`bb_logpmf_element` elementwise over `obs`, `total` and `p_binom` broadcast, at one concentration.

    Args:
        obs: B-allele counts.
        total: allele totals.
        p_binom: mean B fraction.
        tau: concentration.
        data_terms: include log C(n, k).

    Returns:
        The log-pmf, the broadcast shape.
    """
    obs, total, p_binom = (np.ascontiguousarray(v, dtype=np.float64) for v in np.broadcast_arrays(obs, total, p_binom))
    out = np.empty(obs.shape)
    alpha, beta = (p_binom * float(tau)).reshape(-1), ((1.0 - p_binom) * float(tau)).reshape(-1)
    bb_logpmf_vector_by_element(obs.reshape(-1), total.reshape(-1), alpha, beta, out.reshape(-1), data_terms)
    return out


@njit(nogil=True, cache=True, parallel=True, error_model="numpy")
def bb_logpmf_dense(X_bb, total_bb_RD, p_binom, taus, EPS=1e-10):
    """`bb_logpmf_vector` per state and spot.

    Args:
        X_bb: (n_obs, n_spots).
        total_bb_RD: (n_obs, n_spots).
        p_binom: (n_states, 1).
        taus: (n_states, 1).
        EPS: shape floor.

    Returns:
        The `(n_states, n_obs, n_spots)` log-pmf.
    """
    n_states = p_binom.shape[0]
    n_obs, n_spots = X_bb.shape

    out = np.zeros((n_states, n_obs, n_spots), dtype=np.float64)

    # NB threads over (state, spot) pairs.
    for job in prange(n_states * n_spots):
        i, s = job // n_spots, job % n_spots
        bb_logpmf_vector(X_bb[:, s], total_bb_RD[:, s], p_binom[i, 0], taus[i, 0], out[i, :, s], EPS)
    return out
