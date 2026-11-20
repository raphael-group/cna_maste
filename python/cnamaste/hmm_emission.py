import time
import warnings
from dataclasses import dataclass

import numpy as np
import scipy.optimize
from numba import njit

from cnamaste.config import get_global_config, start_time
from cnamaste.hmm_emission_bb import bb_logpmf_vector_by_element
from cnamaste.logger import get_logger

logger = get_logger(__name__, start_time=start_time)

# TODO
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")


@dataclass
class OptimizationResult:
    optimizer: str
    params: np.ndarray
    llf: float
    converged: bool
    iterations: int
    fcalls: int

    def __post_init__(self):
        if not hasattr(self, "mle_retvals"):
            self.mle_retvals = {
                "converged": self.converged,
                "iterations": self.iterations,
                "fcalls": self.fcalls,
            }
        if not hasattr(self, "mle_settings"):
            self.mle_settings = {"optimizer": self.optimizer}


def get_betabinom_start_params(legacy=False, exog=None):
    config = get_global_config()

    if legacy:
        return (0.5 / exog.shape[1]) * np.ones(config.hmm.n_states), 1.0

    ps = [float(xx) for xx in config.betabinom.start_params.split(",")]

    return ps, float(config.betabinom.start_disp)


@njit(nogil=True, cache=True, error_model="numpy")
def get_bb_shape_params(exog, params):
    """The beta-binomial shape parameters per row, `a = p tau`, `b = (1 - p) tau`, `p` and `tau` the row's state's.

    Args:
        exog: (n, n_states) state design.
        params: the `n_states` p, then the `n_states` tau.

    Returns:
        `(a, b)`, (n,) each.
    """
    num_states = exog.shape[-1]

    p = np.dot(exog, params[:num_states])
    t = np.dot(exog, params[num_states:])

    a = p * t
    b = (1.0 - p) * t

    return a, b


def bb_negative_loglike(endog, exog, weights, exposure, params, reduce):
    """The negative beta-binomial log-likelihood per row, `bb_logpmf_element` with its data terms, or its weighted sum.

    Args:
        endog: B allele counts.
        exog: state design.
        weights: per row.
        exposure: totals.
        params: as `get_bb_shape_params`.
        reduce: sums with `weights`.

    Returns:
        The (n,) negative log-pmf, or its weighted sum.

    Raises:
        RuntimeError: where the weighted sum is NaN.
    """
    a, b = get_bb_shape_params(exog, params)
    logpmf = np.empty(len(endog))
    bb_logpmf_vector_by_element(endog, exposure, a, b, logpmf, True)
    result = -logpmf

    if not reduce:
        return result

    reduced_result = result.dot(weights)

    if np.isnan(reduced_result):
        logger.info(
            f"Detected invalid ln. likelihood={reduced_result} for:\n{params}\n"
            f"  NaN weights: {np.count_nonzero(np.isnan(weights))}, NaN log-pmf: {np.count_nonzero(np.isnan(result))}"
        )
        raise RuntimeError()

    return reduced_result


class WeightedBBFit:
    """The weighted beta-binomial MLE of `p` and `tau` per state, the dispersion shared unless asked otherwise."""

    def __init__(
        self,
        endog,
        exog,
        weights,
        exposure,
        fixed_dispersion=False,
        shared_dispersion=True,
    ):
        exog = exog.copy()

        # NB corresponds to a single (potentially unknown) state.
        if exog.ndim == 1:
            exog = np.atleast_2d(exog).T

        # NB EM-based posterior weights.
        self.endog = np.asarray(endog, dtype=np.float64)
        self.exog = np.asarray(exog, dtype=np.float64)
        self.weights = np.asarray(weights, dtype=np.float64)
        self.exposure = np.asarray(exposure, dtype=np.float64)
        self.fixed_dispersion = fixed_dispersion
        self.shared_dispersion = shared_dispersion

        self.num_states = self.exog.shape[-1]

    def nloglikeobs(self, params, *args):
        params = np.array(params)

        if self.fixed_dispersion:
            params = np.concatenate([params, np.full(self.num_states, args[0])])
        elif self.shared_dispersion:
            params = np.concatenate([params[:-1], np.full(self.num_states, params[-1])])
        else:
            assert len(params) == self.num_states * 2

        return bb_negative_loglike(self.endog, self.exog, self.weights, self.exposure, params, reduce=True)

    def get_default_params(self, legacy=False):
        ps, single_dispersion = get_betabinom_start_params(legacy=legacy, exog=self.exog)

        if self.fixed_dispersion or self.shared_dispersion:
            disp = [single_dispersion]
        else:
            disp = [single_dispersion] * self.num_states

        return np.array(ps[: self.num_states] + disp)

    def get_bounds(self):
        EPSILON = 1.0e-6
        bounds = []

        for _ in range(self.num_states):
            bounds.append((EPSILON, 1.0 - EPSILON))

        if self.fixed_dispersion:
            pass
        elif self.shared_dispersion:
            bounds.append((EPSILON, 1e6))
        else:
            for _ in range(self.num_states):
                bounds.append((EPSILON, 1e6))

        return bounds

    def fit(self, start_params=None, maxiter=10_000, maxfun=5_000, legacy=False, **kwargs):
        if using_default_params := (start_params is None):
            start_params = self.get_default_params(legacy=legacy)

        start_time = time.time()

        logger.info(
            f"WeightedBBFit (num_states={self.num_states}, endog.shape={self.endog.shape}), initial nloglike={self.nloglikeobs(start_params):.6e} @ start_params:\n{[xx for xx in start_params]}"
        )

        bounds = self.get_bounds()
        options = {"maxiter": maxiter, "maxfun": maxfun} | {k: kwargs[k] for k in ("ftol", "disp") if k in kwargs}

        if self.fixed_dispersion:
            args = (start_params[-1],)
            start_params = start_params[:-1]
        else:
            args = ()

        # NB the dispersions are searched on log scale: on its own scale tau's gradient at the start (1,000) is below
        #    the solver's tolerance, so the fit reported convergence at the start against an MLE near 50.
        n_p = self.num_states
        to_model = lambda x: np.concatenate([x[:n_p], np.exp(x[n_p:])])
        start = np.concatenate([start_params[:n_p], np.log(start_params[n_p:])])
        searched = bounds[:n_p] + [(np.log(lo), np.log(hi)) for lo, hi in bounds[n_p:]]

        result = scipy.optimize.minimize(
            lambda x, *a: self.nloglikeobs(to_model(x), *a),
            start,
            method="L-BFGS-B",
            bounds=searched,
            options=options,
            args=args,
        )
        result.x = to_model(result.x)

        optimize_result = OptimizationResult(
            optimizer="L-BFGS-B",
            params=result.x,
            llf=-result.fun,
            converged=result.success,
            iterations=result.get("nit", None),
            fcalls=result.get("nfev", None),
        )

        end_time = time.time()
        runtime = end_time - start_time

        logger.info(
            f"WeightedBBFit done: {runtime:.2f}s with L-BFGS-B\nendog_shape={self.endog.shape},\n"
            f"{len(start_params)} params ({'with default start' if using_default_params else 'with custom start'}),\n"
            f"{optimize_result.mle_retvals.get('iterations', 'N/A')} iter,\n"
            f"{optimize_result.mle_retvals.get('fcalls', 'N/A')} fcalls,\n"
            f"optimizer: {optimize_result.mle_settings.get('optimizer', 'Unknown')},\n"
            f"converged: {optimize_result.mle_retvals.get('converged', 'N/A')},\n"
            f"llf: {optimize_result.llf:.6e}\n"
            f"params:\n{[xx for xx in optimize_result.params]}"
        )

        return optimize_result
