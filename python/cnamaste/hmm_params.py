"""The HMM's parameters as JAX pytrees: canonical (`CnaHmmParameters`) and the optimizer's unconstrained `Theta`."""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.flatten_util
import jax.numpy as jnp
import numpy as np
import scipy.special


class CnaHmmParameters(NamedTuple):
    """The start and emission parameters, `(n_states,)` each; a JAX pytree."""

    log_startprob: Any
    log_mu: Any
    p_binom: Any
    alphas: Any
    taus: Any


class Theta(NamedTuple):
    """The optimizer's coordinates, in the vector's order; `None` for a block the fit holds."""

    startprob: Any = None
    log_mu: Any = None
    logit_p: Any = None
    log_alpha: Any = None
    log_tau: Any = None


def get_fitted_blocks(params: str, optimize_nb: bool) -> dict[str, bool]:
    """Which `Theta` blocks a fit carries.

    Args:
        params: the fit's channels.
        optimize_nb: whether any bin has a read-depth baseline.

    Returns:
        Block name -> carried.
    """
    fits_rates = optimize_nb and "m" in params
    return {
        "startprob": "s" in params,
        "log_mu": fits_rates,
        "logit_p": "p" in params,
        "log_alpha": fits_rates,
        "log_tau": "p" in params,
    }


def get_block_widths(n_states: int) -> dict[str, int]:
    """Each block's length: one per state, one for each shared dispersion."""
    return {"startprob": n_states, "log_mu": n_states, "logit_p": n_states, "log_alpha": 1, "log_tau": 1}


def to_theta(parameters: CnaHmmParameters, params: str, optimize_nb: bool, use_logit: bool = True) -> Theta:
    """The fitted blocks of `parameters`, unconstrained.

    Args:
        parameters: canonical.
        params: as `get_fitted_blocks`.
        optimize_nb: as `get_fitted_blocks`.
        use_logit: the frequencies' logits.

    Returns:
        The `Theta`.
    """
    fitted = get_fitted_blocks(params, optimize_nb)
    p_binom = np.asarray(parameters.p_binom, dtype=np.float64).flatten()
    blocks = {
        "startprob": lambda: np.asarray(parameters.log_startprob).flatten(),
        "log_mu": lambda: np.asarray(parameters.log_mu).flatten(),
        "logit_p": lambda: scipy.special.logit(p_binom) if use_logit else p_binom,
        "log_alpha": lambda: np.array([np.log(np.asarray(parameters.alphas).flatten()[0])]),
        "log_tau": lambda: np.array([np.log(np.asarray(parameters.taus).flatten()[0])]),
    }
    return Theta(**{name: make() for name, make in blocks.items() if fitted[name]})


def from_theta(theta: Theta, held: CnaHmmParameters, n_states: int, use_logit: bool = True) -> CnaHmmParameters:
    """`theta` back to canonical parameters, `(n_states,)` each, in `jax.numpy`; a block `theta` does not carry is
    `held`'s.

    Args:
        theta: the `Theta`.
        held: the values of the blocks the fit holds.
        n_states: the states.
        use_logit: as `to_theta`.

    Returns:
        The `CnaHmmParameters`.
    """
    log_startprob, log_mu, p_binom, alphas, taus = held
    if theta.startprob is not None:
        log_startprob = theta.startprob - jax.scipy.special.logsumexp(theta.startprob)
    if theta.log_mu is not None:
        log_mu = jnp.reshape(theta.log_mu, n_states)
    if theta.logit_p is not None:
        block = jnp.reshape(theta.logit_p, n_states)
        p_binom = jax.nn.sigmoid(block) if use_logit else jnp.clip(block, 1e-6, 1 - 1e-6)
    if theta.log_alpha is not None:
        alphas = jnp.full(n_states, jnp.exp(theta.log_alpha[0]))
    if theta.log_tau is not None:
        taus = jnp.full(n_states, jnp.exp(theta.log_tau[0]))
    return CnaHmmParameters(log_startprob, log_mu, p_binom, alphas, taus)


def get_theta_template(n_states: int, params: str, optimize_nb: bool) -> Theta:
    """A `Theta` of zeros carrying the fit's blocks: the vector's structure, for `get_unravel_theta`."""
    fitted, widths = get_fitted_blocks(params, optimize_nb), get_block_widths(n_states)
    return Theta(**{name: np.zeros(widths[name]) for name, carried in fitted.items() if carried})


@functools.cache
def get_unravel_theta(n_states: int, params: str, optimize_nb: bool) -> Callable[[Any], Theta]:
    """`jax.flatten_util.ravel_pytree`'s `unravel` for a fit's structure, built once per structure.

    Args:
        n_states: the states.
        params: the fit's channels.
        optimize_nb: whether any bin has a read-depth baseline.

    Returns:
        The vector's `unravel`.
    """
    return jax.flatten_util.ravel_pytree(get_theta_template(n_states, params, optimize_nb))[1]


def get_theta_slices(n_states: int, params: str, optimize_nb: bool) -> dict[str, slice]:
    """Each fitted block's place in the vector, by name, in the vector's order."""
    template = get_theta_template(n_states, params, optimize_nb)
    names = [name for name in Theta._fields if getattr(template, name) is not None]
    widths = get_block_widths(n_states)
    ends = np.cumsum([widths[name] for name in names], dtype=np.int64)
    return {name: slice(int(end - widths[name]), int(end)) for name, end in zip(names, ends)}


def get_theta_bounds(
    n_states: int,
    params: str,
    optimize_nb: bool,
    use_logit: bool = True,
    alpha_range: tuple[float, float] = (1e-6, 1_000.0),
    tau_range: tuple[float, float] = (1e-4, 5_000.0),
) -> list[tuple[float | None, float | None]]:
    """The `(min, max)` of each vector entry, in the vector's order.

    Args: as `get_theta_template`; `use_logit`; `alpha_range`, `tau_range` the dispersions'.

    Returns:
        One `(min, max)` per entry.
    """
    per_entry = {
        "startprob": (None, None),
        "log_mu": (None, None),
        "logit_p": (None, None) if use_logit else (1e-6, 1.0 - 1e-6),
        "log_alpha": (float(np.log(alpha_range[0])), float(np.log(alpha_range[1]))),
        "log_tau": (float(np.log(tau_range[0])), float(np.log(tau_range[1]))),
    }
    slices = get_theta_slices(n_states, params, optimize_nb)
    return [per_entry[name] for name, place in slices.items() for _ in range(place.stop - place.start)]
