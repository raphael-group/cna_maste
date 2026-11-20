"""`sandbox.mstep_optax`, optax's L-BFGS as the M step's solver: on the planted M step against scipy's BFGS, and on
the planted HMRF in place of `mstep.get_mstep_minimum`."""

from __future__ import annotations

import pytest

import test_hmrf
import test_m_step
from cnamaste import mstep
from cnamaste.sandbox.mstep_optax import (
    get_fit_cost,
    get_optax_bounds,
    get_optax_minimum,
    get_optax_mstep_minimum,
)

OPTAX = pytest.mark.xdist_group("run_hmrf_optax")


def test_the_optax_mstep_reaches_the_scipy_minimum() -> None:
    """`get_optax_minimum` ends within 1e-6 nats of scipy BFGS's minimum on the planted M step. Fails where the optax
    backend stops short of the optimum."""
    planted = test_m_step.get_planted_mstep()
    found = get_optax_minimum(
        planted["x0"],
        lambda: planted["data"],
        *get_optax_bounds(None, len(planted["x0"])),
        cost=get_fit_cost(test_m_step.N_STATES, "smp", True, True),
        max_iter=100,
        callback=None,
    )
    expected = test_m_step.get_planted_minimum(planted, "scipy_bfgs")
    assert found.nit > 0
    assert found.fun - expected.fun <= 1e-6, (found.message, found.fun - expected.fun)


@pytest.fixture(scope="module")
def cna_planted() -> test_hmrf.Planted:
    return test_hmrf.get_cna_planted()


@pytest.fixture(scope="module", params=["sp", "smp", "smp-posterior"])
def optax_inferred(request: pytest.FixtureRequest, cna_planted: test_hmrf.Planted) -> test_hmrf.Inference:
    params, _, logmu_shift_type = request.param.partition("-")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(mstep, "get_mstep_minimum", get_optax_mstep_minimum)
        return test_hmrf.get_inference(
            cna_planted, params, "opengm_expansion", logmu_shift_type=logmu_shift_type or "argmax"
        )


@OPTAX
def test_optax_run_hmrf_recovers_the_planted_clones(
    optax_inferred: test_hmrf.Inference, cna_planted: test_hmrf.Planted
) -> None:
    """`test_hmrf.test_run_hmrf_recovers_the_planted_clones` under optax_lbfgs."""
    test_hmrf.test_run_hmrf_recovers_the_planted_clones(optax_inferred, cna_planted)


@OPTAX
def test_optax_run_hmrf_recovers_the_planted_states(
    optax_inferred: test_hmrf.Inference, cna_planted: test_hmrf.Planted, request: pytest.FixtureRequest
) -> None:
    """`test_hmrf.test_run_hmrf_recovers_the_planted_states` under optax_lbfgs; fails at `smp`."""
    if optax_inferred.params == "smp":
        reason = "optax_lbfgs, untuned: planted-state BAF error 0.043 vs BFGS 0.009 (T- #100)"
        request.applymarker(pytest.mark.xfail(strict=True, reason=reason))
    test_hmrf.test_run_hmrf_recovers_the_planted_states(optax_inferred, cna_planted)


@OPTAX
def test_optax_the_outer_loop_stops_at_the_ari_tolerance(optax_inferred: test_hmrf.Inference) -> None:
    """`test_hmrf.test_the_outer_loop_stops_at_the_ari_tolerance` under optax_lbfgs."""
    test_hmrf.test_the_outer_loop_stops_at_the_ari_tolerance(optax_inferred)
