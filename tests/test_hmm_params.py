"""`hmm_params`: the HMM's parameters as JAX pytrees, and the one vector the fit, its bounds and its gradient read.

Referees: the blocks in the order `pack_params` laid them out before `hmm_params`, written out; and the round trip
through the vector, `jax.flatten_util.ravel_pytree`. It costs 1 s.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.flatten_util
import numpy as np
import pytest
import scipy.special

from cnamaste.hmm_nophasing import hmm_nophasing
from cnamaste.hmm_params import (
    CnaHmmParameters,
    Theta,
    from_theta,
    get_theta_bounds,
    get_theta_slices,
    get_unravel_theta,
    to_theta,
)

N_STATES = 4

FITS = [("sp", True), ("smp", True), ("stmp", True), ("smp", False), ("m", True)]
"""`(params, optimize_nb)`: the phasing fit, the RDR-BAF fit, with the start, without a baseline, and depth alone."""


def canonical(seed: int = 0) -> CnaHmmParameters:
    rng = np.random.default_rng(seed)
    return CnaHmmParameters(
        np.log(rng.dirichlet(np.ones(N_STATES))),
        rng.normal(0.0, 0.3, N_STATES),
        rng.uniform(0.05, 0.5, N_STATES),
        np.full(N_STATES, 0.02),
        np.full(N_STATES, 80.0),
    )


def written_out(parameters: CnaHmmParameters, params: str, optimize_nb: bool) -> np.ndarray:
    """`pack_params`' vector as it was written before `hmm_params`: start, rates, logits, then the two dispersions."""
    fits_rates = optimize_nb and "m" in params
    blocks = [parameters.log_startprob.flatten()] if "s" in params else []
    blocks += [parameters.log_mu.flatten()] if fits_rates else []
    blocks += [scipy.special.logit(parameters.p_binom.flatten())] if "p" in params else []
    blocks += [np.array([np.log(parameters.alphas.flatten()[0])])] if fits_rates else []
    blocks += [np.array([np.log(parameters.taus.flatten()[0])])] if "p" in params else []
    return np.concatenate(blocks)


@pytest.mark.parametrize(("params", "optimize_nb"), FITS)
def test_the_vector_is_the_pytree_ravelled(params: str, optimize_nb: bool) -> None:
    """`pack_params` is `ravel_pytree` of the `Theta`, float64, the blocks in their written-out order, exactly; held
    blocks are `None`, empty subtrees. Fails where a block moves, is dropped, or loses precision."""
    parameters = canonical()
    flat = hmm_nophasing(params=params).pack_params(*parameters, optimize_nb=optimize_nb)
    assert flat.dtype == np.float64 and np.array_equal(flat, written_out(parameters, params, optimize_nb))


@pytest.mark.parametrize(("params", "optimize_nb"), FITS)
def test_the_vector_round_trips_and_held_blocks_are_kept(params: str, optimize_nb: bool) -> None:
    """`from_theta(get_unravel_theta(...)(ravel(to_theta(x))))` is `x` on every fitted block, to the last bit for the rates and the
    start and to 1e-15 through the logit and the log; a held block is the value given. Fails where a block is read
    from another's place."""
    parameters, held = canonical(0), canonical(1)
    flat = jax.flatten_util.ravel_pytree(to_theta(parameters, params, optimize_nb))[0]
    back = from_theta(get_unravel_theta(N_STATES, params, optimize_nb)(flat), held, N_STATES)
    fitted = get_theta_slices(N_STATES, params, optimize_nb)
    for name, field in (("log_mu", "log_mu"), ("logit_p", "p_binom"), ("log_alpha", "alphas"), ("log_tau", "taus")):
        expected = getattr(parameters if name in fitted else held, field)
        np.testing.assert_allclose(getattr(back, field), expected, rtol=1e-15, atol=0)
    start = parameters.log_startprob if "startprob" in fitted else held.log_startprob
    np.testing.assert_allclose(back.log_startprob, start, rtol=0, atol=1e-15)


@pytest.mark.parametrize(("params", "optimize_nb"), FITS)
def test_the_bounds_and_slices_follow_the_vector(params: str, optimize_nb: bool) -> None:
    """One `(min, max)` per entry, the dispersions' log ranges at their slices and nothing else bounded; the slices
    tile the vector in `Theta`'s field order. Fails where the bounds and the vector drift apart."""
    flat = jax.flatten_util.ravel_pytree(to_theta(canonical(), params, optimize_nb))[0]
    slices = get_theta_slices(N_STATES, params, optimize_nb)
    bounds = get_theta_bounds(N_STATES, params, optimize_nb)
    assert len(bounds) == flat.size == sum(s.stop - s.start for s in slices.values())
    assert list(slices) == [name for name in Theta._fields if name in slices]
    for name, place in slices.items():
        expected = {"log_alpha": (np.log(1e-6), np.log(1e3)), "log_tau": (np.log(1e-4), np.log(5e3))}.get(name)
        for entry in bounds[place]:
            assert entry == ((None, None) if expected is None else tuple(float(v) for v in expected)), name
    assert hmm_nophasing(params=params).get_bounds(N_STATES, optimize_nb=optimize_nb) == bounds


def test_a_theta_is_a_pytree() -> None:
    """`jax.tree_util.tree_map` acts on the fitted blocks alone and keeps the held ones `None`. Fails where `Theta`
    stops being a pytree, as an optimizer over pytrees (`optax`) would read it."""
    theta = to_theta(canonical(), "sp", True)
    doubled: Any = jax.tree_util.tree_map(lambda block: 2.0 * block, theta)
    assert doubled.log_mu is None and doubled.log_alpha is None
    assert np.array_equal(doubled.logit_p, 2.0 * theta.logit_p)
    assert len(jax.tree_util.tree_leaves(theta)) == 3
