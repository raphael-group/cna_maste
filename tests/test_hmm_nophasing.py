"""`hmm_nophasing`'s NB and beta-binomial evaluators, every shape, against scipy.

Referees: `scipy.stats.nbinom.logpmf(k, 1 / alpha, 1 / (1 + alpha lambda))`, `scipy.stats.betabinom.logpmf`, and their
limits `scipy.stats.poisson` and `scipy.stats.binom`. Tolerances: 1e-12 relative or absolute where the log-pmf's
lgamma differences cancel less than 1e3-fold (alpha >= 1e-2, tau <= 60); the limits as measured, stated per row.
"""

from __future__ import annotations

from math import lgamma

import numpy as np
import pytest
import scipy.special
import scipy.stats

from cnamaste.emission import get_decoded_emission, get_encoded, get_log_emission, get_unconfounded
from cnamaste.hmm_emission_bb import (
    bb_logpmf_array,
    bb_logpmf_dense,
    bb_logpmf_element,
    bb_logpmf_vector,
    bb_logpmf_vector_by_element,
    bb_switch_logpmf_element,
    bb_switch_logpmf_vector,
)
from cnamaste.hmm_emission_nb import nb_logpmf_array, nb_logpmf_dense, nb_logpmf_element, nb_logpmf_vector
from cnamaste.hmm_params import CnaHmmParameters
from cnamaste.hmm_utils import rising_factorial_threshold
from cnamaste.spatio_genomic_counts import PseudobulkCounts

COUNTS = np.array([0.0, 1.0, 3.0, 12.0, 40.0, 100.0])
MEANS = np.array([5.0, 0.5, 4.0, 20.0, 30.0, 80.0])
TOTALS = np.array([0.0, 4.0, 10.0, 25.0, 60.0, 100.0])
B_COUNTS = np.array([0.0, 0.0, 3.0, 25.0, 30.0, 5.0])


def get_nb_by_shape(alpha: float) -> dict[str, np.ndarray]:
    """The NB log-pmf at `(COUNTS, MEANS)` from every evaluator shape, one state and one dispersion.

    Args:
        alpha: the dispersion.

    Returns:
        The log-pmf by evaluator.
    """
    floored = max(alpha, 1.0e-10)
    log_k = np.array([lgamma(k + 1.0) for k in COUNTS])
    vector = np.empty(COUNTS.size)
    nb_logpmf_vector(COUNTS, MEANS, 1.0, alpha, vector)
    dense = nb_logpmf_dense(COUNTS[:, None], MEANS[:, None], np.zeros((1, 1)), np.full((1, 1), alpha))[0, :, 0]
    encoded = get_encoded_rows(COUNTS, MEANS, B_COUNTS, TOTALS, alpha, 0.5, 60.0)[0]
    element = [
        nb_logpmf_element(k, mean, floored, 1.0 / floored, lgamma(1.0 / floored), lk)
        for k, mean, lk in zip(COUNTS, MEANS, log_k)
    ]
    return {
        "element": np.array(element),
        "vector": vector,
        "dense": dense,
        "encoded": encoded,
        "array": nb_logpmf_array(COUNTS, MEANS, alpha, data_terms=True),
    }


def get_bb_by_shape(p_binom: float, tau: float) -> dict[str, np.ndarray]:
    """The beta-binomial log-pmf at `(B_COUNTS, TOTALS)` from every evaluator shape, one state.

    Args:
        p_binom: the B allele frequency.
        tau: the concentration.

    Returns:
        The log-pmf by evaluator.
    """
    alpha, beta = max(p_binom * tau, 1e-10), max((1.0 - p_binom) * tau, 1e-10)
    vector, by_element = np.empty(TOTALS.size), np.empty(TOTALS.size)
    bb_logpmf_vector(B_COUNTS, TOTALS, p_binom, tau, vector)
    bb_logpmf_vector_by_element(
        B_COUNTS, TOTALS, np.full(6, p_binom * tau), np.full(6, (1.0 - p_binom) * tau), by_element, True
    )
    dense = bb_logpmf_dense(B_COUNTS[:, None], TOTALS[:, None], np.full((1, 1), p_binom), np.full((1, 1), tau))[0, :, 0]
    encoded = get_encoded_rows(COUNTS, MEANS, B_COUNTS, TOTALS, 0.1, p_binom, tau)[1]
    element = np.array([bb_logpmf_element(k, n, alpha, beta) for k, n in zip(B_COUNTS, TOTALS)])
    array = bb_logpmf_array(B_COUNTS, TOTALS, p_binom, tau, data_terms=True)
    return {
        "element": element,
        "vector": vector,
        "by_element": by_element,
        "dense": dense,
        "encoded": encoded,
        "array": array,
    }


def get_encoded_rows(
    counts: np.ndarray, means: np.ndarray, b_counts: np.ndarray, totals: np.ndarray, alpha: float, p: float, tau: float
) -> tuple[np.ndarray, np.ndarray]:
    """The encoded lattice's read-depth and allele rows, one state and one spot.

    Args:
        counts, means: read counts and NB means.
        b_counts, totals: B allele counts and depths.
        alpha, p, tau: the state's dispersion, B allele frequency and concentration.

    Returns:
        The read-depth and allele log-pmf, (n,) each.
    """
    column = PseudobulkCounts(counts[None], means[None], b_counts[None], totals[None], None, {})
    parameters = CnaHmmParameters(None, np.zeros(1), np.full(1, p), np.full(1, alpha), np.full(1, tau))
    encoded = get_encoded(column, common_zero_depth=False)
    emission = get_log_emission(
        encoded, parameters, get_unconfounded(1), layout="lattice", encoding="coded", phased=False, data_terms=True
    )
    rdr, baf = get_decoded_emission(emission)
    return rdr[0, :, 0], baf[0, :, 0]


@pytest.mark.parametrize("alpha", [1e-2, 0.5, 10.0])
def test_every_nb_evaluator_is_scipy_s(alpha: float) -> None:
    """Each shape is scipy's NB log-pmf, 1e-12 relative or absolute (measured 1.3e-13 at alpha 1e-2, k = 100)."""
    expected = scipy.stats.nbinom.logpmf(COUNTS, 1.0 / alpha, 1.0 / (1.0 + alpha * MEANS))
    for name, got in get_nb_by_shape(alpha).items():
        np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12, err_msg=name)


def test_the_nb_evaluators_score_zero_at_no_mean_and_tend_to_poisson_at_the_floor() -> None:
    """lambda = 0 scores 0.0; alpha 1e-12 is floored at 1e-10, the Poisson log-pmf to 1e-4: lgamma(k + 1e10) -
    lgamma(1e10) cancels 1e11-fold (measured 2.0e-5 at k = 40)."""
    assert nb_logpmf_array(np.array([0.0, 7.0]), np.array([0.0, 0.0]), 0.5, data_terms=True).tolist() == [0.0, 0.0]
    for name, got in get_nb_by_shape(1e-12).items():
        np.testing.assert_allclose(got, scipy.stats.poisson.logpmf(COUNTS, MEANS), rtol=0.0, atol=1e-4, err_msg=name)


@pytest.mark.parametrize(("p_binom", "tau"), [(0.3, 1.0), (0.5, 60.0), (0.05, 60.0), (0.0, 60.0)])
def test_every_bb_evaluator_is_scipy_s(p_binom: float, tau: float) -> None:
    """Each shape is scipy's beta-binomial at the floored shapes (p = 0: alpha = EPS = 1e-10), 1e-12; n = 0 scores
    0.0, scipy's log 1; `data_terms=False`, the fit's form, is scipy's less log C(n, k). Fails where a shape drops the
    floor or a data term."""
    alpha, beta = max(p_binom * tau, 1e-10), max((1.0 - p_binom) * tau, 1e-10)
    expected = scipy.stats.betabinom.logpmf(B_COUNTS, TOTALS, alpha, beta)
    for name, got in get_bb_by_shape(p_binom, tau).items():
        np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12, err_msg=name)
        assert got[0] == 0.0, name
    log_binom_coeff = scipy.special.gammaln(TOTALS + 1) - scipy.special.gammaln(B_COUNTS + 1)
    log_binom_coeff -= scipy.special.gammaln(TOTALS - B_COUNTS + 1)
    no_data = [bb_logpmf_element(k, n, alpha, beta, False) for k, n in zip(B_COUNTS, TOTALS)]
    np.testing.assert_allclose(no_data, expected - log_binom_coeff, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(("tau", "atol"), [(1e10, 1e-6), (1e14, 1e-10)])
def test_every_bb_evaluator_tends_to_the_binomial_by_rising_factorials(tau: float, atol: float) -> None:
    """Above `rising_factorial_threshold` the beta-binomial is the binomial to n^2 / (p tau), the model's own gap
    (measured 1.5e-7 at tau 1e10, n = 25), within 1e-10 at 1e14, where lgamma differences are 0.41 nats off."""
    assert tau > rising_factorial_threshold
    expected = scipy.stats.binom.logpmf(B_COUNTS, TOTALS, 0.3)
    for name, got in get_bb_by_shape(0.3, tau).items():
        np.testing.assert_allclose(got, expected, rtol=0.0, atol=atol, err_msg=name)


def test_a_count_above_its_total_scores_zero() -> None:
    """k > n is outside the support: every BB evaluator's convention is 0.0, not scipy's -inf."""
    assert bb_logpmf_element(3.0, 2.0, 1.0, 1.0) == 0.0
    assert bb_logpmf_array(np.array([3.0]), np.array([2.0]), 0.5, 2.0, data_terms=True).tolist() == [0.0]


@pytest.mark.parametrize("tau", [60.0, 2.0 * rising_factorial_threshold], ids=["lgamma", "rising"])
def test_the_switch_is_the_swapped_count_log_pmf_less_the_original(tau: float) -> None:
    """`bb_switch_logpmf_vector` adds log BB(n - k) - log BB(k): scipy's below the threshold, to 1e-10; above,
    `bb_logpmf_element`'s rising factorials (scipy's lgamma differences are 8e-8 off at 2e7), to 1e-10;
    `bb_switch_logpmf_element` elementwise, bitwise."""
    p = np.array([0.1, 0.5, 0.8])
    a, b = p[:, None] * tau, (1.0 - p[:, None]) * tau
    if tau < rising_factorial_threshold:
        swapped = scipy.stats.betabinom.logpmf(TOTALS - B_COUNTS, TOTALS, a, b)
        original = scipy.stats.betabinom.logpmf(B_COUNTS, TOTALS, a, b)
    else:
        logpmf = np.vectorize(bb_logpmf_element)
        swapped, original = logpmf(TOTALS - B_COUNTS, TOTALS, a, b), logpmf(B_COUNTS, TOTALS, a, b)
    got = bb_switch_logpmf_vector(np.zeros((3, TOTALS.size)), B_COUNTS, TOTALS, p, np.full(3, tau))
    np.testing.assert_allclose(got, swapped - original, rtol=0.0, atol=1e-10)
    assert np.array_equal(got, np.vectorize(bb_switch_logpmf_element)(0.0, B_COUNTS, TOTALS, a, b))
