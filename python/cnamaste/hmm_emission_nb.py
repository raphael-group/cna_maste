"""The negative-binomial read-depth log-pmf, `eqn:nb_emission`: one element, a state's vector, an array and the dense
`(n_states, n_obs, n_spots)` field."""

from math import exp, lgamma, log, log1p

import numpy as np
from numba import njit, prange

from cnamaste.hmm_utils import log_factorial


@njit(nogil=True, cache=True, error_model="numpy")
def nb_logpmf_vector(obs, exposure, mu, alpha, out, log_k_factorial=None):
    """The NB log-pmf per count for one state, written in place to `out`.

    `log_k_factorial` (`log_factorial(obs)`) lets a loop over states share the data term, and `lgamma(r)` is taken
    once per call; bitwise the three-call form, the same operations in the same order.

    Args:
        obs: (n,).
        exposure: (n,).
        mu: mean rate.
        alpha: dispersion.
        out: (n,) output.
        log_k_factorial: optional.
    """
    # NB the floored dispersion sets both r and p, which then agree on the mean; log p = -log1p(alpha lambda) and
    #    log(1 - p) = log(alpha lambda) - log1p(alpha lambda) keep their precision where alpha lambda is tiny, where
    #    1 / (1 + alpha lambda) rounds to 1 and the pmf read 1 at every count.
    if log_k_factorial is None:
        data_term = log_factorial(obs)
    else:
        data_term = log_k_factorial

    alpha = max(alpha, 1.0e-10)
    r = 1.0 / alpha
    lgamma_r = lgamma(r)

    for i in range(len(obs)):
        out[i] = nb_logpmf_element(obs[i], exposure[i] * mu, alpha, r, lgamma_r, data_term[i])


@njit(nogil=True, cache=True, inline="always", error_model="numpy")
def nb_logpmf_element(k, lambda_i, alpha, r, lgamma_r, data_term):
    """One count's NB log-pmf: `nb_logpmf_vector`'s element, which the fused field also scores.

    Args:
        k: count.
        lambda_i: mean.
        alpha: floored dispersion.
        r: `1 / alpha`.
        lgamma_r: `lgamma(r)`.
        data_term: `log k!`.

    Returns:
        The log-pmf.
    """
    if lambda_i <= 0.0 or k < 0:
        return 0.0
    x = alpha * lambda_i
    log1px = log1p(x)
    return lgamma(k + r) - lgamma_r - data_term - r * log1px + k * (log(x) - log1px)


def nb_logpmf_array(obs, mean, alpha, *, data_terms):
    """`nb_logpmf_element` elementwise over `obs` and `mean` broadcast, at one dispersion; a bin with no mean scores 0.

    Args:
        obs: counts.
        mean: NB means.
        alpha: dispersion.
        data_terms: include `-lgamma(k + 1)`.

    Returns:
        The log-pmf, the broadcast shape.
    """
    obs, mean = (np.ascontiguousarray(v, dtype=np.float64) for v in np.broadcast_arrays(obs, mean))
    out = np.empty(obs.shape)
    flat = obs.reshape(-1)
    log_k_factorial = log_factorial(flat) if data_terms else np.zeros(flat.size)
    nb_logpmf_vector(flat, mean.reshape(-1), 1.0, float(alpha), out.reshape(-1), log_k_factorial)
    return out


@njit(nogil=True, cache=True, parallel=True, error_model="numpy")
def nb_logpmf_dense(X_nb, base_nb_mean, log_mu, alphas):
    """`nb_logpmf_vector` per state and spot.

    Args:
        X_nb: (n_obs, n_spots).
        base_nb_mean: (n_obs, n_spots).
        log_mu: (n_states, 1).
        alphas: (n_states, 1).

    Returns:
        The `(n_states, n_obs, n_spots)` log-pmf.
    """
    n_states = log_mu.shape[0]
    n_obs, n_spots = X_nb.shape

    out = np.zeros((n_states, n_obs, n_spots), dtype=np.float64)

    # NB each spot's data term once for every state; threads over (state, spot) pairs, so a clone stack's
    #    one spot still spreads over the states.
    log_k_factorial = np.empty((n_spots, n_obs))
    for s in prange(n_spots):
        log_k_factorial[s] = log_factorial(X_nb[:, s])
    for job in prange(n_states * n_spots):
        i, s = job // n_spots, job % n_spots
        nb_logpmf_vector(
            X_nb[:, s], base_nb_mean[:, s], exp(log_mu[i, 0]), alphas[i, 0], out[i, :, s], log_k_factorial[s]
        )

    return out
