"""The CalicoST emission comparisons' shared input and referee call, for `test_hmm_nophasing` and `test_hmm_phased`."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from calicost_cache import Referee
from cnamaste.emission import get_log_emission, get_state_major, get_unconfounded
from cnamaste.hmm.forward_backward import get_dense_encoded_emission, get_forward_backward, get_log_gamma
from cnamaste.hmm_params import CnaHmmParameters
from cnamaste.spatio_genomic_counts import PseudobulkCounts

EMISSION_ATOL = 1e-12


def get_emission_atol() -> float:
    """Return the emission atol, 1e-12."""
    return EMISSION_ATOL


class HMMInputs(NamedTuple):
    X: np.ndarray
    base_nb_mean: np.ndarray
    total_bb_RD: np.ndarray
    log_mu: np.ndarray
    alphas: np.ndarray
    p_binom: np.ndarray
    taus: np.ndarray
    lengths: np.ndarray
    log_startprob: np.ndarray
    log_transmat: np.ndarray
    log_sitewise_transmat: np.ndarray


def get_hmm_inputs(n_spots: int = 4) -> HMMInputs:
    """Seeded counts from three (mu, p) states, per-state parameters `(n_states, 1)`, a sticky transition matrix.

    Returns: the `HMMInputs`; bin 3 has zero baseline in spot 1, bin 5 zero coverage in every spot.
    """
    rng = np.random.default_rng(0)
    n_obs, n_states = 60, 3
    state = np.repeat([0, 1, 2, 1], 15)
    log_mu = np.log([[0.5], [1.0], [1.5]])
    p_binom = np.array([[0.1], [0.5], [0.3]])
    base = rng.uniform(5.0, 50.0, (n_obs, n_spots))
    base[3, 1] = 0.0
    total = rng.integers(0, 40, (n_obs, n_spots)).astype(float)
    total[5] = 0.0
    X = np.zeros((n_obs, 2, n_spots))
    X[:, 0] = rng.poisson(base * np.exp(log_mu[state]))
    X[:, 1] = rng.binomial(total.astype(int), p_binom[state])
    transmat = np.full((n_states, n_states), 0.01 / (n_states - 1)) + np.eye(n_states) * (0.99 - 0.01 / (n_states - 1))
    return HMMInputs(
        X,
        base,
        total,
        log_mu,
        np.full((n_states, 1), 0.05),
        p_binom,
        np.full((n_states, 1), 30.0),
        np.array([25, 35]),
        np.log(np.full(n_states, 1.0 / n_states)),
        np.log(transmat),
        np.log(rng.uniform(1e-4, 0.2, n_obs)),
    )


def get_calicost_emission(hmmclass: type, h: HMMInputs, referee: Referee) -> tuple[np.ndarray, np.ndarray]:
    """`hmmclass`'s read depth and allele log emissions, each parameter broadcast to `(n_states, n_spots)`, recorded
    whole: they are compared within `EMISSION_ATOL` and feed the lattices."""
    n_spots = h.X.shape[2]
    log_mu, alphas, p_binom, taus = (
        np.broadcast_to(a, (len(a), n_spots)) for a in (h.log_mu, h.alphas, h.p_binom, h.taus)
    )

    def call() -> tuple[np.ndarray, np.ndarray]:
        return tuple(
            hmmclass.compute_emission_probability_nb_betabinom(
                h.X, h.base_nb_mean, log_mu, alphas, h.total_bb_RD, p_binom, taus
            )
        )

    rdr, baf = referee(f"hmm.emission.{hmmclass.__name__}", tuple(h), call, kept=(0, 1))
    return rdr, baf


def get_dense_emission(h: HMMInputs, *, phased: bool) -> tuple[np.ndarray, np.ndarray]:
    """`emission.get_log_emission`'s dense lattice on `h`, `2 n_states` (state, phase) pairs where `phased`."""
    counts = PseudobulkCounts(h.X[:, 0].T, h.base_nb_mean.T, h.X[:, 1].T, h.total_bb_RD.T, None, {})
    parameters = CnaHmmParameters(None, h.log_mu, h.p_binom, h.alphas, h.taus)
    found = get_log_emission(counts, parameters, get_unconfounded(1), layout="lattice", phased=phased, data_terms=True)
    return get_state_major(found.rdr), get_state_major(found.baf)


def get_state_major_lattices(
    lengths: np.ndarray,
    log_transmat: np.ndarray,
    log_startprob: np.ndarray,
    log_emission: np.ndarray,
    log_sitewise_transmat: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`hmm.forward_backward` as CalicoST's lattices take it: an `(S, n_obs, n_spots)` emission summed over spots, one
    clone, unpenalized.

    Args:
        lengths: bins per contig.
        log_transmat: `(K, K)`.
        log_startprob: `(K,)`.
        log_emission: `(S, n_obs, n_spots)`.
        log_sitewise_transmat: `(n_obs,)` log phase switch, or None where unphased.

    Returns:
        `(log_alpha, log_beta, log_gamma)`, `(S, n_obs)` each.
    """
    encoded = get_dense_encoded_emission(log_emission.sum(axis=2).T[None])
    lattices = get_forward_backward(encoded, log_startprob, log_transmat, lengths, log_sitewise_transmat, False)
    return tuple(np.ascontiguousarray(v[0].T) for v in (*lattices, get_log_gamma(*lattices)))
