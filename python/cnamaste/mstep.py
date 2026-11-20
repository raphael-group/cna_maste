"""The M step's minimizer, one frontend over `mstep.solver`: scipy's BFGS or L-BFGS-B."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, NamedTuple

import numpy as np
import scipy.optimize

from cnamaste.config import get_mstep_solvers, start_time
from cnamaste.fit_checks import record_fit_count
from cnamaste.logger import get_logger

logger = get_logger(__name__, start_time=start_time)

SCIPY_METHODS = {"scipy_bfgs": "BFGS", "scipy_lbfgsb": "L-BFGS-B"}
"""`mstep.solver` -> its `scipy.optimize.minimize` method."""

LINE_SEARCH_ENDINGS = ("precision loss", "ABNORMAL")
"""A message ending the solve where the line search made no step: BFGS's and L-BFGS-B's."""


class MstepResult(NamedTuple):
    """One M step's solve.

    Attributes:
        x: the minimizing vector.
        fun: the cost at `x`.
        nit: the solver's iterations.
        nfev: the cost's evaluations.
        success: the solver's convergence.
        message: why the solver stopped.
        seconds: the solve's wall time.
    """

    x: np.ndarray
    fun: float
    nit: int
    nfev: int
    success: bool
    message: str
    seconds: float


def get_scipy_methods() -> dict[str, str]:
    """Return `mstep.solver` -> scipy method, {"scipy_bfgs": "BFGS", "scipy_lbfgsb": "L-BFGS-B"}."""
    return SCIPY_METHODS


def get_line_search_endings() -> tuple[str, ...]:
    """Return the messages of a solve the line search ended, ("precision loss", "ABNORMAL")."""
    return LINE_SEARCH_ENDINGS


def get_gtol() -> float:
    """The solvers' stop on the gradient's largest entry, scipy's BFGS `gtol`.

    Returns:
        1e-5.
    """
    return 1e-5


def get_relative_cost_stop(cost_rtol: float) -> Callable[[scipy.optimize.OptimizeResult], None]:
    """A scipy callback ending the solve at the first iteration that lowers the cost by at most `cost_rtol |cost|`.

    Args:
        cost_rtol: the relative decrease; 0 for none.

    Returns:
        The callback, which raises `StopIteration`.
    """
    last_cost = [np.inf]

    def stop(intermediate_result: scipy.optimize.OptimizeResult) -> None:
        cost = float(intermediate_result.fun)
        if last_cost[0] - cost <= cost_rtol * abs(cost):
            raise StopIteration
        last_cost[0] = cost

    return stop


def get_mstep_minimum(
    cost_and_gradient: Callable[[np.ndarray], tuple[float, np.ndarray]],
    x0: np.ndarray,
    config: Any,
    *,
    max_iter: int,
    bounds: list[tuple[float | None, float | None]] | None,
    callback: Callable[[np.ndarray], None] | None,
) -> MstepResult:
    """The minimum of an M step's cost by `config.mstep.solver`, logged and counted into the run's fit counts.

    Args:
        cost_and_gradient: the cost and its gradient at a vector.
        x0: the start.
        config: the run's; `mstep.solver` and `mstep.cost_rtol` read.
        max_iter: the solver's iterations.
        bounds: scipy's `(min, max)` per entry, or None; BFGS under bounds is L-BFGS-B.
        callback: called with each iterate the next iteration starts from, or None.

    Returns:
        The `MstepResult`.

    Raises:
        ValueError: for a solver not one of `get_mstep_solvers`.
    """
    solver = str(config.mstep.solver)
    if solver not in get_mstep_solvers():
        raise ValueError(f"mstep.solver {solver!r} is not one of {get_mstep_solvers()}.")
    start = time.perf_counter()
    result = get_scipy_minimum(
        cost_and_gradient,
        x0,
        get_scipy_methods()[solver],
        max_iter=max_iter,
        bounds=bounds,
        callback=callback,
        cost_rtol=float(config.mstep.cost_rtol),
    )
    result = result._replace(seconds=time.perf_counter() - start)
    record_mstep(solver, result)
    return result


def record_mstep(solver: str, result: MstepResult) -> None:
    """Log one M step's line and add it to the `mstep_*` fit counts.

    Args:
        solver: `mstep.solver`.
        result: the step's `MstepResult`.
    """
    line_search = any(ending in result.message for ending in get_line_search_endings())
    logger.info(
        f"M step: {solver} | nit={result.nit} | success={result.success} | message={result.message} | "
        f"seconds={result.seconds:.2f}"
    )
    record_fit_count("mstep_calls", 1)
    record_fit_count("mstep_unconverged", int(not result.success))
    record_fit_count("mstep_precision_loss", int(line_search))
    record_fit_count("mstep_seconds", result.seconds)


def get_scipy_minimum(
    cost_and_gradient: Callable[[np.ndarray], tuple[float, np.ndarray]],
    x0: np.ndarray,
    method: str,
    *,
    max_iter: int,
    bounds: list[tuple[float | None, float | None]] | None,
    callback: Callable[[np.ndarray], None] | None,
    cost_rtol: float,
) -> MstepResult:
    """`scipy.optimize.minimize` of `cost_and_gradient` by `method`, L-BFGS-B where bounded.

    Args:
        cost_and_gradient: the cost and its gradient at a vector.
        x0: the start.
        method: `BFGS` or `L-BFGS-B`.
        max_iter: the iterations.
        bounds: `(min, max)` per entry, or None.
        callback: called with the iterate after each iteration, or None.
        cost_rtol: end at an iteration lowering the cost by at most `cost_rtol |cost|`; 0 for none.

    Returns:
        The `MstepResult`, `seconds` 0.
    """
    method = "L-BFGS-B" if bounds is not None else method
    # NB `disp` is BFGS's alone; L-BFGS-B takes no `ftol` here: at 1e-6 it ended 18 of 23 easy (2d4ce9a9) fits
    #    after 1 iteration, 4.6e5 nats short.
    options = {"maxiter": max_iter, "gtol": get_gtol()} | ({"disp": False} if method == "BFGS" else {})
    stop = get_relative_cost_stop(cost_rtol) if cost_rtol > 0.0 else None

    def on_iteration(intermediate_result: scipy.optimize.OptimizeResult) -> None:
        if callback is not None:
            callback(intermediate_result.x)
        if stop is not None:
            stop(intermediate_result)

    res = scipy.optimize.minimize(
        cost_and_gradient,
        x0,
        method=method,
        jac=True,
        bounds=bounds,
        callback=None if callback is None and stop is None else on_iteration,
        options=options,
    )
    if stop is not None and "StopIteration" in str(res.message):
        res.success, res.message = True, f"cost decrease <= {cost_rtol:g} |cost|"
    return MstepResult(
        np.asarray(res.x), float(res.fun), int(res.nit), int(res.nfev), bool(res.success), str(res.message), 0.0
    )
