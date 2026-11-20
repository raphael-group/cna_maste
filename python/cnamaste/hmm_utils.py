from math import lgamma, log, log1p

import numpy as np
from numba import njit

from cnamaste.config import get_global_config, start_time
from cnamaste.external.digamma import digamma_element
from cnamaste.logger import get_logger

logger = get_logger(__name__, start_time=start_time)


# NB transition for better behaved beta-binomial limit @ large tau=alpha+beta,
rising_factorial_threshold = 1.0e7
"""Below this concentration `a + b` the beta-binomial takes its lgamma differences; at and above it,
`log_rising`'s sums."""


@njit(nogil=True, cache=True, error_model="numpy")
def log_rising(x, m):
    """`lgamma(x + m) - lgamma(x)` for a count `m`, as `m log x + sum_{i < m} log1p(i / x)`: no large lgammas to cancel.

    Args:
        x: positive real.
        m: count.

    Returns:
        The log rising factorial.
    """
    total = 0.0
    for i in range(int(m)):
        total += log1p(i / x)
    return m * log(x) + total


@njit(nogil=True, cache=True, error_model="numpy")
def log_factorial(obs):
    """`lgamma(k + 1)` per count: the NB log-pmf's data term, shared by every state.

    Args:
        obs: counts.

    Returns:
        The per-count log factorial.
    """
    out = np.empty(len(obs))
    for i in range(len(obs)):
        out[i] = lgamma(obs[i] + 1.0)
    return out


@njit(cache=True)
def logsumexp(a):
    """`scipy.special.logsumexp` of a 1-D array, callable from `njit` code.

    The one implementation of four: `a_max` is guarded against `-inf`, so an all `-inf` row
    returns `-inf`, as scipy does, rather than NaN.

    Args:
        a: 1-D array of log values.

    Returns:
        `log(sum(exp(a)))`, a float.
    """
    a_max = np.max(a)

    if np.isinf(a_max):
        return a_max

    return a_max + np.log(np.sum(np.exp(a - a_max)))


@njit(cache=True, inline="always", error_model="numpy")
def digamma_difference_element(x: float, k: float) -> float:
    """`digamma(x + k) - digamma(x)` of one `(x, k)`, asymptotic above `x = 100`, callable from `njit` code.

    Args:
        x: a positive real.
        k: a count.

    Returns:
        The difference; above `x = 100` its libm `log1p` differs from numpy's SIMD one in the last bit.
    """
    if x > 100.0:
        return log1p(k / x) + k / (2 * x * (x + k)) + k * (2 * x + k) / (12 * x**2 * (x + k) ** 2)
    return digamma_element(x + k) - digamma_element(x)


def get_normal_clone(baf_profiles: np.ndarray, eps: float) -> int:
    """The clone whose BAF strays least from 0.5 outside the `eps` deadband, summed over its bins; ties to the lower
    index.

    Args:
        baf_profiles: `(n_obs, n_clones)` each clone's BAF per bin, NaN where a bin is not observed.
        eps: the deadband about 0.5.

    Returns:
        The normal clone's index.
    """
    stray = np.nansum(np.maximum(np.abs(baf_profiles - 0.5) - eps, 0.0), axis=0)
    return int(np.argmin(stray))


def get_log_transmat(n_states: int, t: float) -> np.ndarray:
    """The log transition matrix that stays with probability `t` and moves to each other state alike.

    Args:
        n_states: the copy-number states.
        t: the stay probability.

    Returns:
        The `(n_states, n_states)` log matrix; `[[0]]` for one state.
    """
    if n_states > 1:
        transmat = np.ones((n_states, n_states)) * (1.0 - t) / (n_states - 1)
        np.fill_diagonal(transmat, t)
        log_transmat = np.log(transmat)
    else:
        log_transmat = np.zeros((1, 1))

    return log_transmat


def get_em_solver_params() -> dict[str, float]:
    """The normal spots' beta-binomial fit's L-BFGS-B options, `hmm.em_maxiter` and `hmm.em_ftol`.

    Returns:
        The options as floats, keyed without the `em_` prefix.
    """
    config = get_global_config()
    return {k: float(getattr(config.hmm, f"em_{k}")) for k in ("maxiter", "ftol")}
