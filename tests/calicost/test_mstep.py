"""The M step against CalicoST `c1abcae`'s `utils_hmm.update_emission_params_{nb,bb}_sitewise_uniqvalues` and the
`Weighted_NegativeBinomial` and `Weighted_BetaBinom` likelihoods they maximize.

The referee is CalicoST's own output on the same input, computed once and recorded (`calicost_cache`); cnamaste's side
is `hmm_mstep.get_cost_and_gradient`, minimized by `mstep.get_mstep_minimum` (BFGS). Inputs: one clone of 150 bins on
a planted path through 3 states, 50 bins each, read depth negative binomial at baseline 200 and B-allele counts
binomial of 80, seeded; one-hot phased posteriors on the path; shared dispersions, as CalicoST's `main()` runs them.
CalicoST's M step reads `csr_matrix.A`, which scipy removed: a property returning `toarray()` stands in for it.

Tolerances: the cost 1e-12 relative (both score scipy's log-pmfs); the minimum, cnamaste's cost at its parameters no
more than at CalicoST's, and the parameters 1e-3 relative (CalicoST's Nelder-Mead stops at `xtol` 1e-4; measured
2.5e-4 on the dispersion, 2e-5 on the rest).
"""

from __future__ import annotations

import functools
from types import SimpleNamespace
from typing import NamedTuple
from unittest import mock

import numpy as np
import scipy.sparse
from calicost.utils_distribution_fitting import Weighted_BetaBinom, Weighted_NegativeBinomial
from calicost.utils_hmm import (
    construct_unique_matrix,
    update_emission_params_bb_sitewise_uniqvalues,
    update_emission_params_nb_sitewise_uniqvalues,
)

import cnamaste.calicost  # noqa: F401  (imports CalicoST with the environment held, as cnamaste does)
from calicost_cache import Referee, departs
from cnamaste.emission import get_encoded
from cnamaste.hmm_mstep import ClassWeights, get_class_weights, get_cost_and_gradient
from cnamaste.hmm_nophasing import hmm_nophasing
from cnamaste.hmm_params import CnaHmmParameters
from cnamaste.mstep import get_mstep_minimum
from cnamaste.spatio_genomic_counts import PseudobulkCounts

N_STATES, PER_STATE = 3, 50
PLANTED_MU = np.array([0.5, 1.0, 1.5])
STRUCTURE = {"n_states": N_STATES, "params": "mp", "optimize_nb": True, "use_logit": True}
MODEL = hmm_nophasing(params="mp")


class PlantedMstep(NamedTuple):
    reads: np.ndarray
    baseline: np.ndarray
    allele: np.ndarray
    total: np.ndarray
    gamma: np.ndarray
    """`(2 K, n_obs)`, one-hot on the path in the first phase."""
    start: CnaHmmParameters
    weights: ClassWeights


def get_planted_mstep(p_binom: tuple[float, ...]) -> PlantedMstep:
    """Counts drawn on the planted path at `PLANTED_MU` and `p_binom`, and a start 20% off the planted depth ratios."""
    rng = np.random.default_rng(0)
    path = np.repeat(np.arange(N_STATES), PER_STATE)
    baseline, total = np.full(path.size, 200.0), np.full(path.size, 80.0)
    reads = rng.negative_binomial(100.0, 100.0 / (100.0 + baseline * PLANTED_MU[path])).astype(float)
    allele = rng.binomial(80, np.asarray(p_binom)[path]).astype(float)
    gamma = np.zeros((2 * N_STATES, path.size))
    gamma[path, np.arange(path.size)] = 1.0
    start = CnaHmmParameters(
        np.log(np.full(N_STATES, 1.0 / N_STATES)),
        np.log(PLANTED_MU * 1.2),
        np.minimum(np.asarray(p_binom) + 0.05, 0.9),
        np.full(N_STATES, 0.3),
        np.full(N_STATES, 100.0),
    )
    counts = PseudobulkCounts(reads[None], baseline[None], allele[None], total[None], np.array([path.size]), {})
    encoded = get_encoded(counts, common_zero_depth=False)
    weights = get_class_weights(encoded, gamma, None, counts, n_states=N_STATES, data_terms=True)
    return PlantedMstep(reads, baseline, allele, total, gamma, start, weights)


def get_cost(planted: PlantedMstep, theta: np.ndarray) -> float:
    """cnamaste's M-step cost at `theta`, data terms kept."""
    return float(get_cost_and_gradient(theta, planted.start, planted.weights, **STRUCTURE)[0])


def get_theta(log_mu: np.ndarray, p_binom: np.ndarray, alphas: np.ndarray, taus: np.ndarray) -> np.ndarray:
    """The optimization vector of canonical parameters."""
    log_startprob = np.log(np.full(N_STATES, 1.0 / N_STATES))
    return MODEL.pack_params(log_startprob, log_mu, p_binom, alphas, taus, optimize_nb=True, use_logit=True)


def call_calicost_nloglikeobs(planted: PlantedMstep, parameters: CnaHmmParameters) -> float:
    """CalicoST's per-state `Weighted_NegativeBinomial` and `Weighted_BetaBinom` negative log-likelihoods at
    `parameters`, summed over states."""
    total = 0.0
    for counts, exposure, model in ((planted.reads, planted.baseline, "nb"), (planted.allele, planted.total, "bb")):
        unique_values, mapping_matrices = construct_unique_matrix(counts[:, None], exposure[:, None])
        # NB copied from utils_hmm.py:401-408 and 619-624 (c1abcae), `s` the one clone.
        tmp = (scipy.sparse.csr_matrix(planted.gamma) @ mapping_matrices[0]).toarray()
        idx_nonzero = np.where(unique_values[0][:, 1] > 0)[0]
        for i in range(N_STATES):
            if model == "nb":
                likelihood = Weighted_NegativeBinomial(
                    unique_values[0][idx_nonzero, 0],
                    np.ones(len(idx_nonzero)).reshape(-1, 1),
                    weights=tmp[i, idx_nonzero] + tmp[i + N_STATES, idx_nonzero],
                    exposure=unique_values[0][idx_nonzero, 1],
                    penalty=0,
                )
                params = np.array([parameters.log_mu[i], parameters.alphas[i]])
            else:
                likelihood = Weighted_BetaBinom(
                    np.append(
                        unique_values[0][idx_nonzero, 0],
                        unique_values[0][idx_nonzero, 1] - unique_values[0][idx_nonzero, 0],
                    ),
                    np.ones(2 * len(idx_nonzero)).reshape(-1, 1),
                    weights=np.append(tmp[i, idx_nonzero], tmp[i + N_STATES, idx_nonzero]),
                    exposure=np.append(unique_values[0][idx_nonzero, 1], unique_values[0][idx_nonzero, 1]),
                )
                params = np.array([parameters.p_binom[i], parameters.taus[i]])
            total += float(likelihood.nloglikeobs(params))
    return total


def call_calicost_mstep(planted: PlantedMstep) -> tuple[np.ndarray, ...]:
    """CalicoST's M step from `planted.start`, shared dispersions, as `run_baum_welch_nb_bb` calls it.

    Returns: `new_log_mu`, `new_alphas`, `new_p_binom`, `new_taus`, one column.
    """
    log_gamma = np.log(planted.gamma, out=np.full_like(planted.gamma, -np.inf), where=planted.gamma > 0)
    start = planted.start
    with mock.patch.object(scipy.sparse.csr_matrix, "A", property(lambda m: m.toarray()), create=True):
        values, matrices = construct_unique_matrix(planted.reads[:, None], planted.baseline[:, None])
        log_mu, alphas = update_emission_params_nb_sitewise_uniqvalues(
            values,
            matrices,
            log_gamma,
            planted.baseline[:, None],
            start.alphas[:, None].copy(),
            start_log_mu=start.log_mu[:, None].copy(),
            shared_NB_dispersion=True,
        )
        values, matrices = construct_unique_matrix(planted.allele[:, None], planted.total[:, None])
        p_binom, taus = update_emission_params_bb_sitewise_uniqvalues(
            values,
            matrices,
            log_gamma,
            planted.total[:, None],
            start.taus[:, None].copy(),
            start_p_binom=start.p_binom[:, None].copy(),
            shared_BB_dispersion=True,
        )
    return log_mu, alphas, p_binom, taus


def get_calicost_mstep(referee: Referee, name: str, planted: PlantedMstep) -> tuple[np.ndarray, ...]:
    """`call_calicost_mstep`, from the cache, each output flattened."""
    inputs = (planted.reads, planted.baseline, planted.allele, planted.total, planted.gamma, *planted.start[1:])
    found = referee(f"mstep.{name}", inputs, functools.partial(call_calicost_mstep, planted), kept=(0, 1, 2, 3))
    return tuple(np.ravel(v) for v in found)


def get_mstep(planted: PlantedMstep) -> tuple[np.ndarray, ...]:
    """cnamaste's M step from `planted.start`, BFGS.

    Returns: `log_mu`, `alphas`, `p_binom`, `taus`.
    """
    found = get_mstep_minimum(
        lambda theta: tuple(
            np.asarray(v) for v in get_cost_and_gradient(theta, planted.start, planted.weights, **STRUCTURE)
        ),
        get_theta(*planted.start[1:]),
        SimpleNamespace(mstep=SimpleNamespace(solver="scipy_bfgs", cost_rtol=1.0e-12)),
        max_iter=200,
        bounds=None,
        callback=None,
    )
    _, log_mu, p_binom, alphas, taus = MODEL.unpack_params(found.x, N_STATES, *planted.start)
    return log_mu, alphas, p_binom, taus


def test_the_mstep_cost_is_calicost_s_negative_log_likelihood(referee: Referee) -> None:
    """At the start, cnamaste's cost with its data terms is CalicoST's per-state NB and BB negative log-likelihoods,
    summed, within 1e-12 relative. Fails where a class weight, the NB or BB parametrization, or the switched phase's
    `1 - p` drifts."""
    planted = get_planted_mstep((0.1, 0.3, 0.5))
    call = functools.partial(call_calicost_nloglikeobs, planted, planted.start)
    inputs = (planted.reads, planted.baseline, planted.allele, planted.total, planted.gamma, *planted.start[1:])
    (theirs,) = referee("mstep.nloglikeobs", inputs, call, kept=(0,))
    np.testing.assert_allclose(get_cost(planted, get_theta(*planted.start[1:])), theirs, rtol=1e-12, atol=0.0)


def test_the_mstep_minimum_is_calicost_s(referee: Referee) -> None:
    """From the same start, cnamaste's minimum costs no more than CalicoST's parameters, and its depth ratios, B
    fractions and dispersions are CalicoST's within 1e-3 relative. Fails where the objective's minimum moves."""
    planted = get_planted_mstep((0.1, 0.3, 0.5))
    log_mu, alphas, p_binom, taus = get_calicost_mstep(referee, "planted", planted)
    ours = get_mstep(planted)
    assert get_cost(planted, get_theta(ours[0], ours[2], ours[1], ours[3])) <= get_cost(
        planted, get_theta(log_mu, p_binom, alphas, taus)
    )
    found = (np.exp(ours[0]), *ours[1:])
    for value, theirs in zip(found, (np.exp(log_mu), alphas, p_binom, taus), strict=True):
        np.testing.assert_allclose(value, theirs, rtol=1e-3, atol=0.0)


@departs("CalicoST clips each fitted B fraction to [0.01, 0.99] and log depth ratio to [-2, 2]; cnamaste does not")
def test_the_mstep_minimum_is_calicost_s_at_a_b_fraction_under_0_01(referee: Referee) -> None:
    """A state planted at B fraction 0.004: CalicoST returns 0.01, its floor; cnamaste 0.00375, 63% under."""
    planted = get_planted_mstep((0.004, 0.3, 0.5))
    _, _, p_binom, _ = get_calicost_mstep(referee, "low_b_fraction", planted)
    np.testing.assert_allclose(get_mstep(planted)[2], p_binom, rtol=1e-3, atol=0.0)
