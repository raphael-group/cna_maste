"""`hmm_emission.bb_negative_loglike`, the normal-spot fit's objective, against scipy's beta-binomial.

Referees: `-scipy.stats.betabinom.logpmf` at each row's state's `(p tau, (1 - p) tau)`, summed with the weights, to
1e-12 relative (tau <= 60, where the lgamma differences cancel less than 1e3-fold); `get_bb_shape_params` against
the shapes written out, bitwise.
"""

from __future__ import annotations

import numpy as np
import scipy.stats

from cnamaste.hmm_emission import bb_negative_loglike, get_bb_shape_params


def test_the_negative_loglike_is_scipy_s_per_row_and_weighted() -> None:
    """Two states, rows assigned by a one-hot design: per row and weighted, scipy's; fails where a data term drops."""
    total = np.array([4.0, 10.0, 25.0, 60.0, 100.0, 1.0])
    endog = np.array([0.0, 3.0, 25.0, 30.0, 5.0, 1.0])
    exog = np.array([[1.0, 0.0], [0.0, 1.0]] * 3)
    params = np.array([0.3, 0.6, 5.0, 60.0])
    weights = np.linspace(0.5, 2.0, 6)
    p, tau = exog @ params[:2], exog @ params[2:]
    expected = -scipy.stats.betabinom.logpmf(endog, total, p * tau, (1.0 - p) * tau)
    np.testing.assert_allclose(
        bb_negative_loglike(endog, exog, weights, total, params, reduce=False), expected, rtol=1e-12
    )
    np.testing.assert_allclose(
        bb_negative_loglike(endog, exog, weights, total, params, reduce=True), expected @ weights, rtol=1e-12
    )


def test_the_shape_params_are_p_tau_and_one_minus_p_tau_at_each_row_s_state() -> None:
    """Three states, a one-hot design: `a = p tau`, `b = (1 - p) tau` at each row's state, bitwise. Fails where the
    p and tau halves of `params` are crossed or a row reads another state."""
    rng = np.random.default_rng(2)
    states = rng.integers(0, 3, 8)
    exog = np.eye(3)[states]
    p, tau = np.array([0.1, 0.37, 0.8]), np.array([7.7, 60.0, 300.0])
    a, b = get_bb_shape_params(exog, np.concatenate([p, tau]))
    assert np.array_equal(a, p[states] * tau[states]) and np.array_equal(b, (1.0 - p[states]) * tau[states])
