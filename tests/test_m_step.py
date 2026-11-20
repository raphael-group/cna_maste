"""The M step: `hmm_mstep.get_cost_and_gradient`, the fit's cost and its `jax.value_and_grad` on class weights, and
`mstep.get_mstep_minimum`, its minimizer, on a planted M step.

Referees: the fit's cost over every bin, scored by the numba
kernels on the coded emission, to 1e-12, and its central difference (step 1e-6) to 1e-5 of the largest component, the
difference's own error; scipy's log-pmfs, and `hmm_utils.log_rising`'s sum of logs above `rising_factorial_threshold`.
Settings: each `params` the run reaches, unshifted, on a path's and on soft `log Z_c` weights, and phased.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import jax
import numpy as np
import pytest
import scipy.optimize
import scipy.special
import scipy.stats

from cnamaste import hmm_mstep
from cnamaste.emission import (
    get_decoded_emission,
    get_encoded,
    get_hard_state_posteriors,
    get_log_emission,
    get_state_major,
    get_unconfounded,
)
from cnamaste.hmm_mstep import (
    get_bb_logpmf,
    get_class_weights,
    get_cost,
    get_cost_and_gradient,
    get_log_rising,
    get_nb_logpmf,
)
from cnamaste.hmm_nophasing import get_fit_confounders, hmm_nophasing
from cnamaste.hmm_params import CnaHmmParameters
from cnamaste.hmm_utils import log_rising
from cnamaste.mstep import MstepResult, get_mstep_minimum
from cnamaste.spatio_genomic_counts import PseudobulkCounts

N_STATES, N_CLONES, PER_CLONE = 4, 2, 50
N_OBS = N_CLONES * PER_CLONE

SETTINGS: list[tuple[str, dict[str, Any]]] = [
    ("smp", {"params": "smp", "optimize_nb": True}),
    ("mp", {"params": "mp", "optimize_nb": True}),
    ("smp, no nb", {"params": "smp", "optimize_nb": False}),
    ("sp", {"params": "sp", "optimize_nb": True}),
    ("m", {"params": "m", "optimize_nb": True}),
]
"""Each `(label, settings)` the run can reach: `params` as the stage sets it, `optimize_nb` as the baseline decides
it."""


def case(seed: int = 17) -> dict[str, Any]:
    """One synthetic M step: counts, baselines, parameters and posteriors, with a zero-baseline bin.

    Args: `seed` the generator's.
    Returns: the arrays `pack_params`, the cost and `get_cost_and_gradient` take.
    """
    rng = np.random.default_rng(seed)
    base = rng.uniform(80, 120, size=(N_OBS, 1))
    base[:2] = 0.0
    total = rng.integers(40, 80, size=(N_OBS, 1)).astype(float)
    counts = rng.poisson(np.maximum(base, 1.0)).astype(float)
    allele = rng.binomial(total.astype(int), 0.42).astype(float)
    posterior = np.random.default_rng(seed + 1).dirichlet(np.ones(N_STATES), N_OBS)

    return {
        "counts": PseudobulkCounts(
            counts.T, base.T, allele.T, total.T, np.full(N_CLONES, PER_CLONE, dtype=np.int64), {}
        ),
        "base": base,
        "total": total,
        "log_mu": rng.normal(0, 0.25, (N_STATES, 1)),
        "p_binom": rng.uniform(0.2, 0.8, (N_STATES, 1)),
        "alphas": np.full((N_STATES, 1), 0.5),
        "taus": np.full((N_STATES, 1), 300.0),
        "log_startprob": np.log(np.full(N_STATES, 1.0 / N_STATES)),
        "shifted": get_hard_state_posteriors(rng.integers(0, N_STATES, N_OBS), N_STATES),
        "posterior": np.log(posterior),
        "rng": rng,
    }


def packed(given: dict[str, Any], settings: dict[str, Any], model: Any) -> np.ndarray:
    """`model.pack_params` on `case`'s parameters, at `settings`."""
    return model.pack_params(
        given["log_startprob"],
        given["log_mu"],
        given["p_binom"],
        given["alphas"],
        given["taus"],
        optimize_nb=settings["optimize_nb"],
        use_logit=True,
    )


@pytest.mark.parametrize(
    ("label", "settings", "weighting", "phased"),
    [(label, settings, None, False) for label, settings in SETTINGS]
    + [("smp, path", SETTINGS[0][1], "shifted", False), ("smp, posterior", SETTINGS[0][1], "posterior", False)]
    + [("smp, phased", SETTINGS[0][1], None, True)],
)
def test_the_cost_and_gradient_are_the_fit_s_cost(
    label: str, settings: dict[str, Any], weighting: str | None, phased: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`get_cost_and_gradient` is `-sum(gamma log P)` over every bin of the coded emission, to 1e-12, and its gradient
    that cost's central difference; scored in blocks of 8 rows, the padded ones skipped, the same to 1e-12. Fails where
    a class weight, a held block, `log Z_c`'s dependence on `log_mu`, the switched phase or a valid block is lost."""
    given = case()
    gamma = given["rng"].random(((2 if phased else 1) * N_STATES, N_OBS))
    gamma /= gamma.sum(axis=0)
    log_state_weights = None if weighting is None else given[weighting]
    base = given["base"] if settings["optimize_nb"] else np.zeros_like(given["base"])
    counts = given["counts"]._replace(normal_baseline_sized=base.T)
    model = hmm_nophasing(params=settings["params"])
    theta = packed(given, settings, model)
    canonical = CnaHmmParameters(
        *(np.reshape(given[k], -1) for k in ("log_startprob", "log_mu", "p_binom", "alphas", "taus"))
    )
    encoded = get_encoded(counts, common_zero_depth=False)

    def cost(vector: np.ndarray) -> float:
        parameters = CnaHmmParameters(
            None, *model.unpack_params(vector, N_STATES, *canonical, settings["optimize_nb"])[1:]
        )
        confounders = get_fit_confounders(parameters, log_state_weights, counts, get_unconfounded(N_CLONES))
        rdr, baf = get_decoded_emission(
            get_log_emission(
                encoded, parameters, confounders, layout="lattice", encoding="coded", phased=phased, data_terms=True
            )
        )
        return float(-np.sum(gamma * get_state_major(rdr + baf)[..., 0]))

    weights = get_class_weights(encoded, gamma, log_state_weights, counts, n_states=N_STATES, data_terms=True)
    found, gradient = get_cost_and_gradient(
        theta,
        canonical,
        weights,
        n_states=N_STATES,
        params=settings["params"],
        optimize_nb=settings["optimize_nb"],
        use_logit=True,
    )
    step, numeric = 1e-6, np.zeros_like(theta)
    for i in range(theta.size):
        ahead, behind = theta.copy(), theta.copy()
        ahead[i] += step
        behind[i] -= step
        numeric[i] = (cost(ahead) - cost(behind)) / (2 * step)

    np.testing.assert_allclose(float(found), cost(theta), rtol=1e-12, atol=0, err_msg=label)
    scale = max(float(np.abs(numeric).max()), 1.0)
    np.testing.assert_allclose(np.asarray(gradient), numeric, rtol=0.0, atol=1e-5 * scale, err_msg=label)

    monkeypatch.setattr(hmm_mstep, "BLOCK_SIZE", 8)
    padded = (weights.rdr_weights.shape[0] - weights.n_rdr_valid, weights.baf_weights.shape[0] - weights.n_baf_valid)
    assert min(weights.n_rdr_valid, weights.n_baf_valid) > 8 and max(padded) >= 8, label
    blocked, blocked_gradient = jax.value_and_grad(get_cost)(
        theta,
        canonical,
        weights,
        n_states=N_STATES,
        params=settings["params"],
        optimize_nb=settings["optimize_nb"],
        use_logit=True,
    )
    np.testing.assert_allclose(float(blocked), float(found), rtol=1e-12, atol=0, err_msg=label)
    np.testing.assert_allclose(np.asarray(blocked_gradient), np.asarray(gradient), rtol=0, atol=1e-12 * scale)


def test_the_log_pmfs_are_scipy_s() -> None:
    """`get_nb_logpmf` and `get_bb_logpmf`, with the data terms added back, are scipy's to 1e-12, 0 at no exposure;
    `get_log_rising` above `rising_factorial_threshold` is the sum of logs to 1e-12, and its derivative the digamma
    difference. Fails where a branch of the log rising factorial, or its derivative, is wrong."""
    k, exposure, alpha = np.array([0.0, 3.0, 40.0, 5.0]), np.array([1.0, 2.0, 30.0, 0.0]), 0.2
    mean = exposure[:3] * np.exp(0.4)
    nb = np.asarray(get_nb_logpmf(k, exposure, 0.4, alpha)) - scipy.special.gammaln(k + 1.0) * (exposure > 0)
    np.testing.assert_allclose(nb[:3], scipy.stats.nbinom.logpmf(k[:3], 1 / alpha, 1 / (1 + alpha * mean)), rtol=1e-12)
    assert nb[3] == 0.0

    b, n = np.array([0.0, 3.0, 25.0, 30.0]), np.array([4.0, 10.0, 25.0, 60.0])
    log_binomial = scipy.special.gammaln(n + 1) - scipy.special.gammaln(b + 1) - scipy.special.gammaln(n - b + 1)
    for p, tau in ((0.3, 5.0), (0.9, 1e3)):
        bb = np.asarray(get_bb_logpmf(b, n, p, tau)) + log_binomial
        np.testing.assert_allclose(bb, scipy.stats.betabinom.logpmf(b, n, p * tau, (1 - p) * tau), rtol=1e-12)

    for x in (3.0, 250.0, 2e7, 1e12):
        found = np.asarray(get_log_rising(x, n))
        np.testing.assert_allclose(found, [log_rising(x, m) for m in n], rtol=1e-12, err_msg=str(x))
        derivative = np.asarray(jax.vmap(jax.grad(get_log_rising), in_axes=(None, 0))(x, n))
        np.testing.assert_allclose(derivative, [np.sum(1.0 / (x + np.arange(m))) for m in n], rtol=1e-9, err_msg=str(x))


PLANTED_MU, PLANTED_P = np.array([0.5, 1.0, 1.5, 2.0]), np.array([0.1, 0.3, 0.5, 0.7])
"""Each planted state's depth ratio and BAF, `N_STATES` of them, `PER_CLONE // 2` bins each per clone."""


def get_planted_mstep() -> dict[str, Any]:
    """An M step on counts drawn from `PLANTED_MU` and `PLANTED_P` at their own posteriors, started off them.

    Returns:
        The cost, its data, the start and the model.
    """
    rng = np.random.default_rng(3)
    path = np.tile(np.repeat(np.arange(N_STATES), PER_CLONE // N_STATES + 1)[:PER_CLONE], N_CLONES)
    base = np.full((N_OBS, 1), 200.0)
    total = np.full((N_OBS, 1), 80.0)
    reads = rng.negative_binomial(100.0, 100.0 / (100.0 + base[:, 0] * PLANTED_MU[path]))[:, None].astype(float)
    allele = rng.binomial(80, PLANTED_P[path])[:, None].astype(float)
    counts = PseudobulkCounts(reads.T, base.T, allele.T, total.T, np.full(N_CLONES, PER_CLONE, dtype=np.int64), {})
    gamma = np.eye(N_STATES)[path].T
    model = hmm_nophasing(params="smp")
    start = CnaHmmParameters(
        np.log(np.full(N_STATES, 1.0 / N_STATES)),
        np.log(PLANTED_MU * 1.2),
        PLANTED_P + 0.05,
        np.full(N_STATES, 0.3),
        np.full(N_STATES, 100.0),
    )
    x0 = model.pack_params(*start, optimize_nb=True, use_logit=True)
    weights = get_class_weights(
        get_encoded(counts, common_zero_depth=False), gamma, None, counts, n_states=N_STATES, data_terms=False
    )
    structure = {"n_states": N_STATES, "params": "smp", "optimize_nb": True, "use_logit": True}

    def cost_and_gradient(theta: np.ndarray) -> tuple[float, np.ndarray]:
        value, gradient = get_cost_and_gradient(theta, start, weights, **structure)
        return float(value), np.asarray(gradient)

    return {
        "cost_and_gradient": cost_and_gradient,
        "data": (start, weights),
        "x0": x0,
        "model": model,
        "start": start,
    }


@pytest.fixture(scope="module")
def planted_mstep() -> dict[str, Any]:
    return get_planted_mstep()


def get_planted_minimum(planted: dict[str, Any], solver: str, cost_rtol: float = 0.0) -> MstepResult:
    """`get_mstep_minimum` of `planted` by `solver`, `mstep.max_iter` 100, unbounded."""
    return get_mstep_minimum(
        planted["cost_and_gradient"],
        planted["x0"],
        SimpleNamespace(mstep=SimpleNamespace(solver=solver, cost_rtol=cost_rtol)),
        max_iter=100,
        bounds=None,
        callback=None,
    )


@pytest.mark.parametrize(("solver", "method"), [("scipy_bfgs", "BFGS"), ("scipy_lbfgsb", "L-BFGS-B")])
def test_the_scipy_mstep_is_scipy_s_minimize(planted_mstep: dict[str, Any], solver: str, method: str) -> None:
    """`get_mstep_minimum` by scipy at `mstep.cost_rtol` 0 is the run's `scipy.optimize.minimize` call: x, fun and nit
    to 1e-12, and it recovers each planted state's mu within 5% and p within 0.02; at 1e-12 it ends no later, at fun
    to 1e-12. Fails where the frontend changes the call or the stop ends short of the minimum."""
    found = get_planted_minimum(planted_mstep, solver)
    options = {"maxiter": 100, "gtol": 1e-5} | ({"disp": False} if method == "BFGS" else {})
    expected = scipy.optimize.minimize(
        planted_mstep["cost_and_gradient"], planted_mstep["x0"], method=method, jac=True, options=options
    )
    np.testing.assert_allclose(found.x, expected.x, rtol=1e-12, atol=0)
    np.testing.assert_allclose(found.fun, expected.fun, rtol=1e-12, atol=0)
    assert (found.nit, found.nfev, found.success, found.message) == (
        expected.nit,
        expected.nfev,
        expected.success,
        expected.message,
    )
    assert found.seconds > 0
    _, log_mu, p_binom, *_ = planted_mstep["model"].unpack_params(found.x, N_STATES, *planted_mstep["start"])
    np.testing.assert_allclose(np.exp(log_mu), PLANTED_MU, rtol=0.05)
    np.testing.assert_allclose(p_binom, PLANTED_P, atol=0.02)
    stopped = get_planted_minimum(planted_mstep, solver, cost_rtol=1e-12)
    assert stopped.success and stopped.nfev <= expected.nfev
    np.testing.assert_allclose(stopped.fun, expected.fun, rtol=1e-12, atol=0)
