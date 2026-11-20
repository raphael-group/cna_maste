"""The `(A, B)` lattice, `sec:ab_decoder`: the integer allele copies a decode, the HMM's start and the tracks take."""

from __future__ import annotations

from typing import Any

import numpy as np

from cnamaste.hmm_params import CnaHmmParameters


def get_ab_lattice(*, params: str, fold: bool, max_allele_copy: int, max_total_copy: int) -> np.ndarray:
    """`sec:ab_decoder`'s lattice: every `(A, B)` with `0 < A + B <= max_total_copy` and each allele at most
    `max_allele_copy`.

    Without read depth (`"m" not in params`) pairs of one B fraction are one, the lowest total; unfolded, each pair
    `A != B` also enters as `(B, A)`.

    Args:
        params: the channels scored.
        fold: keep `A >= B` only, a departure from the paper's phased `(A, B)` and `(B, A)`.
        max_allele_copy: the cap on either allele.
        max_total_copy: the cap on `A + B`.

    Returns:
        The `(K, 2)` pairs.
    """
    ab_lattice = np.array(
        [(a, b) for a in range(max_allele_copy + 1) for b in range(a + 1) if 0 < a + b <= max_total_copy]
    )
    if "m" not in params:
        fraction = ab_lattice[:, 1] / ab_lattice.sum(axis=1)
        _, first = np.unique(np.round(fraction, 9), return_index=True)
        ab_lattice = ab_lattice[np.sort(first)]
    if not fold:
        flipped = ab_lattice[ab_lattice[:, 0] != ab_lattice[:, 1]][:, ::-1]
        ab_lattice = np.vstack([ab_lattice, flipped])
    return ab_lattice


def get_ab_is_ploidy_neutral(ab_lattice: np.ndarray, ploidy: int) -> np.ndarray:
    """Each pair's `(A, B) == (P / 2, P / 2)`, the pair `eqn:ab_hmm`'s `eta_s` leaves free; none at odd P.

    Args:
        ab_lattice: `(K, 2)` the pairs.
        ploidy: P.

    Returns:
        `(K,)` whether each pair is the neutral pair.
    """
    return (ab_lattice[:, 0] == ab_lattice[:, 1]) & (2 * ab_lattice[:, 0] == ploidy)


def get_ab_is_nonbalanced(ab_lattice: np.ndarray, min_baf_dist: float) -> np.ndarray:
    """Each pair's B-allele fraction `B / (A + B)` at least `min_baf_dist` from 1/2, (0, 0) excluded.

    Args:
        ab_lattice: `(K, 2)` the pairs.
        min_baf_dist: the least distance of the B-allele fraction from 1/2.

    Returns:
        `(K,)` whether each pair is nonbalanced.
    """
    total = ab_lattice.sum(axis=1)
    baf = np.divide(ab_lattice[:, 1], total, out=np.full(len(total), 0.5), where=total > 0)
    dist = np.abs(baf - 0.5)
    # NB inclusive: (2, 1) sits at 1/6 to rounding.
    return (total > 0) & ((dist >= min_baf_dist) | np.isclose(dist, min_baf_dist, rtol=0.0, atol=1e-12))


def get_lattice_parameters(pairs: np.ndarray, alpha: Any, tau: Any) -> CnaHmmParameters:
    """Integer `(A, B)` states as `CnaHmmParameters`: `log_mu = log((A + B) / 2)` and `p_binom = B / (A + B)`.

    Args:
        pairs: `(..., 2)` integer copies `(A, B)`, `A + B > 0`.
        alpha: NB dispersion, or None.
        tau: BB concentration, or None.

    Returns:
        The `CnaHmmParameters`, `log_startprob` None, `log_mu` and `p_binom` of shape `pairs.shape[:-1]`.
    """
    pairs = np.asarray(pairs)
    total = pairs.sum(axis=-1)
    return CnaHmmParameters(
        log_startprob=None, log_mu=np.log(total / 2.0), p_binom=pairs[..., 1] / total, alphas=alpha, taus=tau
    )
