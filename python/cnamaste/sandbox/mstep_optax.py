"""optax's L-BFGS as an M-step solver, `optax_lbfgs`. Not shipped: out of the package path since the M-step study
(T- #100), where it lowered clone ARI on easy (`2d4ce9a9`), hard (`8797710b`) and harder (`a9a68913`) against scipy's
BFGS, at 2-11x the wall time."""

from __future__ import annotations

import functools
import time
from collections.abc import Callable
from typing import Any
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import optax
import scipy.special

from cnamaste import hmm_mstep
from cnamaste.hmm_nophasing import get_logit_max
from cnamaste.mstep import MstepResult, get_gtol, record_mstep


@functools.cache
def get_lbfgs() -> optax.GradientTransformationExtraArgs:
    """`optax.lbfgs` with its zoom line search, built once."""
    return optax.lbfgs()


@functools.partial(jax.jit, static_argnames=("cost",))
def get_lbfgs_step(
    theta: Any, state: Any, data: Any, lower: Any, upper: Any, *, cost: Callable[[Any, Any], Any]
) -> tuple[Any, Any, Any, Any]:
    """One `get_lbfgs` step on `cost(theta, data)`, the update projected into `[lower, upper]`.

    Args:
        theta: the unconstrained vector.
        state: the solver's state.
        data: what `cost` reads besides `theta`.
        lower: each entry's lower bound, `-inf` for none.
        upper: each entry's upper bound, `inf` for none.
        cost: a scalar function of `(theta, data)`.

    Returns:
        The next vector, the solver's state, and the cost and its gradient at `theta`.
    """
    value_fn = functools.partial(cost, data=data)
    value, gradient = jax.value_and_grad(value_fn)(theta)
    updates, state = get_lbfgs().update(gradient, state, theta, value=value, grad=gradient, value_fn=value_fn)
    return jnp.clip(optax.apply_updates(theta, updates), lower, upper), state, value, gradient


def get_optax_minimum(
    x0: np.ndarray,
    get_data: Callable[[], Any],
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    cost: Callable[[Any, Any], Any],
    max_iter: int,
    callback: Callable[[np.ndarray], None] | None,
) -> MstepResult:
    """`get_lbfgs_step` from `x0` until the gradient's largest entry is under `get_gtol`, the line search makes no
    step, or `max_iter`.

    Args:
        x0: the start.
        get_data: what `cost` reads besides `theta`.
        lower: each entry's lower bound.
        upper: each entry's upper bound.
        cost: a scalar function of `(theta, data)`.
        max_iter: the iterations.
        callback: called with each iterate the next iteration starts from, or None.

    Returns:
        The `MstepResult`, `seconds` 0.
    """
    theta = jnp.asarray(x0)
    state = get_lbfgs().init(theta)
    for iteration in range(max_iter):
        if callback is not None and iteration > 0:
            callback(np.asarray(theta))
        data = get_data()
        found, state, value, gradient = get_lbfgs_step(theta, state, data, lower, upper, cost=cost)
        if float(jnp.max(jnp.abs(gradient))) < get_gtol():
            message, success = "gradient under gtol", True
        elif bool(jnp.all(found == theta)):
            message, success = "no step", False
        else:
            theta = found
            continue
        return MstepResult(np.asarray(theta), float(value), iteration, iteration + 1, success, message, 0.0)
    return MstepResult(np.asarray(theta), float(cost(theta, data)), max_iter, max_iter + 1, False, "max_iter", 0.0)


def get_optax_bounds(bounds: list[tuple[float | None, float | None]] | None, size: int) -> tuple[Any, Any]:
    """`get_optax_minimum`'s `lower` and `upper` from scipy's `(min, max)` per entry, None for none.

    Args:
        bounds: scipy's bounds, or None for an unbounded fit.
        size: the vector's length.

    Returns:
        `(lower, upper)`, `(size,)` each, `-inf` and `inf` where unbounded.
    """
    pairs = [(None, None)] * size if bounds is None else bounds
    lower = np.array([-np.inf if lo is None else lo for lo, _ in pairs], dtype=np.float64)
    upper = np.array([np.inf if hi is None else hi for _, hi in pairs], dtype=np.float64)
    return lower, upper


@functools.cache
def get_fit_cost(n_states: int, params: str, optimize_nb: bool, use_logit: bool) -> Callable[[Any, Any], Any]:
    """`hmm_mstep.get_cost` of `(theta, (held, weights))` for a fit's structure, one function per structure, so
    `get_lbfgs_step` compiles once per structure and bucket.

    Args:
        n_states: `K`.
        params: the fit's channels.
        optimize_nb: whether any bin has a read-depth baseline.
        use_logit: `theta`'s BAF block is a logit.

    Returns:
        The cost.
    """
    structure = {"n_states": n_states, "params": params, "optimize_nb": optimize_nb, "use_logit": use_logit}
    return lambda theta, data: hmm_mstep.get_cost(theta, *data, **structure)


def get_fit_inputs(
    cost_and_gradient: Callable[[np.ndarray], tuple[float, np.ndarray]], theta: np.ndarray
) -> tuple[Callable[[Any, Any], Any], Any]:
    """The `get_fit_cost` and data a `hmm_nophasing` M-step cost reads at `theta`, by its `get_cost_and_gradient` call.

    Args:
        cost_and_gradient: `hmm_nophasing`'s M-step cost.
        theta: the vector.

    Returns:
        `(cost, data)`, `data` the `(held, weights)` the cost reads at `theta`.
    """
    inputs: list[tuple[Any, dict[str, Any]]] = []

    def record(theta: Any, *data: Any, **structure: Any) -> tuple[float, np.ndarray]:
        inputs.append((data, structure))
        return 0.0, np.zeros_like(theta)

    with mock.patch.object(hmm_mstep, "get_cost_and_gradient", record):
        cost_and_gradient(theta)
    data, structure = inputs[0]
    return get_fit_cost(**structure), data


def get_optax_mstep_minimum(
    cost_and_gradient: Callable[[np.ndarray], tuple[float, np.ndarray]],
    x0: np.ndarray,
    config: Any,
    *,
    max_iter: int,
    bounds: list[tuple[float | None, float | None]] | None,
    callback: Callable[[np.ndarray], None] | None,
) -> MstepResult:
    """`mstep.get_mstep_minimum` by `get_optax_minimum`, for a `hmm_nophasing` M step, logged and counted.

    Args:
        cost_and_gradient: `hmm_nophasing`'s M-step cost.
        x0: the start; an infinite BAF logit, which optax's inner products turn to NaN, at `get_logit_max`.
        config: the run's; unread.
        max_iter: the solver's iterations.
        bounds: scipy's `(min, max)` per entry, or None.
        callback: called with each iterate the next iteration starts from, or None.

    Returns:
        The `MstepResult`.
    """
    x0 = np.where(np.isinf(x0), np.sign(x0) * get_logit_max(), x0)
    iterate = [x0]

    def on_iterate(x: np.ndarray) -> None:
        iterate[0] = x
        if callback is not None:
            callback(x)

    cost, _ = get_fit_inputs(cost_and_gradient, x0)
    start = time.perf_counter()
    result = get_optax_minimum(
        x0,
        lambda: get_fit_inputs(cost_and_gradient, iterate[0])[1],
        *get_optax_bounds(bounds, len(x0)),
        cost=cost,
        max_iter=max_iter,
        callback=on_iterate,
    )
    result = result._replace(seconds=time.perf_counter() - start)
    record_mstep("optax_lbfgs", result)
    return result


def get_bb_cost(theta: Any, data: Any) -> Any:
    """The beta-binomial negative log-likelihood of `(b, n)` at `p = expit(theta[0])`, `tau = exp(theta[1])`, less
    `log C(n, b)`.

    Args:
        theta: `(logit p, log tau)`.
        data: `(b, n)`.

    Returns:
        The cost.
    """
    counts, totals = data
    return -jnp.sum(hmm_mstep.get_bb_logpmf(counts, totals, jax.nn.sigmoid(theta[0]), jnp.exp(theta[1])))


def get_bb_fit(counts: np.ndarray, totals: np.ndarray, start: tuple[float, float], max_iter: int) -> np.ndarray:
    """The beta-binomial MLE of one `(p, tau)` by `get_optax_minimum` on `(logit p, log tau)`, `p` in
    `[1e-6, 1 - 1e-6]` and `tau` in `[1e-6, 1e6]`.

    Args:
        counts: `(n,)` B-allele counts.
        totals: `(n,)` allele totals.
        start: `(p, tau)`.
        max_iter: the solver's iterations.

    Returns:
        `(p, tau)`.
    """
    data = (jnp.asarray(counts, dtype=jnp.float64), jnp.asarray(totals, dtype=jnp.float64))
    lower = np.array([scipy.special.logit(1e-6), np.log(1e-6)])
    upper = np.array([scipy.special.logit(1 - 1e-6), np.log(1e6)])
    start_theta = np.array([scipy.special.logit(start[0]), np.log(start[1])])
    theta = get_optax_minimum(
        start_theta, lambda: data, lower, upper, cost=get_bb_cost, max_iter=max_iter, callback=None
    ).x
    return np.array([scipy.special.expit(theta[0]), np.exp(theta[1])])
