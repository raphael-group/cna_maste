"""The M step's objective in `jax.numpy`: `eqn:nb_emission` and `eqn:bb_emission` scored once per equivalence class,
against the E step's posteriors summed per class; `get_cost_and_gradient` is its `jax.value_and_grad`.

A read-depth class is (clone, y, baseline), each a clone's own, a B-allele class (b, n); a clone's `log Z_c` follows
`log_mu`.
"""

from __future__ import annotations

import functools
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import scipy.special

from cnamaste.emission import get_class_index, get_log_state_sums
from cnamaste.hmm_params import CnaHmmParameters, from_theta, get_fitted_blocks, get_unravel_theta
from cnamaste.hmm_utils import rising_factorial_threshold
from cnamaste.spatio_genomic_counts import PseudobulkCounts

BLOCK_SIZE = 1024
"""The rows `get_cost` scores at once; a block past the valid rows is skipped."""


class ClassWeights(NamedTuple):
    """The objective's data, padded to `get_bucket` sizes with weight 0, its valid rows counted.

    Each read-depth class's clone, count and exposure, `(n_rdr,)`, and weight per state, `(n_rdr, K)`, the first
    `n_rdr_valid` rows valid; the distinct B counts `b`, the distinct `n - b` and the distinct totals `n`, each B-allele
    class's index into the three, `(n_baf, 3)`, and its weight per phase and state, `(n_baf, P, K)`, the first
    `n_baf_valid` rows valid; `log Z_c`'s state sums `log S_cs`, `(n_clones, K)` (`emission.get_log_state_sums`), None
    for `Z_c = 1`; and the weighted terms no parameter enters, 0 unless `data_terms`."""

    rdr_clones: Any
    rdr_counts: Any
    rdr_exposures: Any
    rdr_weights: Any
    n_rdr_valid: Any
    baf_counts: Any
    baf_rests: Any
    baf_totals: Any
    baf_index: Any
    baf_weights: Any
    n_baf_valid: Any
    log_state_sums: Any
    data_term: Any


def get_block_size() -> int:
    """Return the rows `get_cost` scores at once, 1024."""
    return BLOCK_SIZE


def get_bucket(size: int) -> int:
    """The padded size of an axis: the next power of two, at least 8, so a run compiles few shapes.

    Args:
        size: the axis' length.

    Returns:
        The bucket.
    """
    return max(8, 1 << max(int(size) - 1, 0).bit_length())


def get_padded(values: np.ndarray, size: int, axis: int, fill: float) -> np.ndarray:
    """`values` padded with `fill` to `size` along `axis`.

    Args:
        values: the array.
        size: the padded length.
        axis: the axis.
        fill: the padding's value.

    Returns:
        The padded array.
    """
    widths = [(0, 0)] * values.ndim
    widths[axis] = (0, size - values.shape[axis])
    return np.pad(values, widths, constant_values=fill)


def get_class_weights(
    encoded: PseudobulkCounts,
    state_posteriors: np.ndarray,
    log_state_weights: np.ndarray | None,
    counts: PseudobulkCounts,
    *,
    n_states: int,
    data_terms: bool,
) -> ClassWeights:
    """The E step's posteriors summed per class: a segment-sum of `gamma` over the bins sharing a class.

    Args:
        encoded: the fit's stacked `PseudobulkCounts`, one column, its `rdr` and `baf` encodings built.
        state_posteriors: `(P K, n_obs)` `gamma`, the second `K` rows the switched phase where `P` is 2.
        log_state_weights: `(n_obs, K)` `log Z_c`'s state weights, or None for `Z_c = 1`.
        counts: the stacked `PseudobulkCounts` whose `get_log_lambda` `log Z_c` reads.
        n_states: `K`.
        data_terms: keep `-lgamma(k + 1)` and `log C(n, k)`, which no parameter enters.

    Returns:
        The `ClassWeights`.
    """
    clone_lengths = encoded.get_clone_lengths()
    n_clones = clone_lengths.size
    gamma = np.asarray(state_posteriors, dtype=np.float64).reshape(-1, n_states, clone_lengths.sum())
    unique_counts, unique_exposures = (np.asarray(v[0], dtype=np.float64) for v in encoded.get_unique("rdr"))
    baf_counts, baf_totals = (np.asarray(v[0], dtype=np.float64) for v in encoded.get_unique("baf"))
    n_baf = baf_counts.size

    # NB a read-depth class is (clone, y, baseline): each clone reads its own log Z_c.
    by_bin = np.repeat(np.arange(n_clones), clone_lengths) * unique_counts.size + get_class_index(encoded, "rdr")[0]
    pairs, rdr_class = np.unique(by_bin, return_inverse=True)
    rdr_clones, unique = np.divmod(pairs, unique_counts.size)
    rdr_counts, rdr_exposures = unique_counts[unique], unique_exposures[unique]
    by_rdr = gamma.sum(axis=0)
    rdr_weights = np.stack([np.bincount(rdr_class, by_rdr[s], pairs.size) for s in range(n_states)], axis=-1)
    baf_class = get_class_index(encoded, "baf")[0]
    baf_weights = np.stack(
        [np.stack([np.bincount(baf_class, phase[s], n_baf) for s in range(n_states)], axis=-1) for phase in gamma],
        axis=1,
    )

    data_term = 0.0
    if data_terms:
        by_class = rdr_weights.sum(axis=1)
        data_term -= np.sum(np.where(rdr_exposures > 0.0, by_class * scipy.special.gammaln(rdr_counts + 1.0), 0.0))
        log_binomial = (
            scipy.special.gammaln(baf_totals + 1.0)
            - scipy.special.gammaln(baf_counts + 1.0)
            - scipy.special.gammaln(baf_totals - baf_counts + 1.0)
        )
        data_term += np.sum(baf_weights.sum(axis=(1, 2)) * log_binomial)

    # NB the state weights are held for the M step: log S_cs once here, not per cost.
    log_state_sums = None if log_state_weights is None else get_log_state_sums(log_state_weights, counts)

    # NB each log rising factorial of the BB term reads one of b, n - b and n: scored once per distinct value.
    distinct = [np.unique(v, return_inverse=True) for v in (baf_counts, baf_totals - baf_counts, baf_totals)]
    baf_index = np.stack([index for _, index in distinct], axis=1)
    distinct_counts, distinct_rests, distinct_totals = (get_padded(v, get_bucket(v.size), 0, 0.0) for v, _ in distinct)

    n_rdr_padded, n_baf_padded = get_bucket(pairs.size), get_bucket(n_baf)
    return ClassWeights(
        get_padded(rdr_clones, n_rdr_padded, 0, 0),
        get_padded(rdr_counts, n_rdr_padded, 0, 0.0),
        get_padded(rdr_exposures, n_rdr_padded, 0, 0.0),
        get_padded(rdr_weights, n_rdr_padded, 0, 0.0),
        np.int64(pairs.size),
        distinct_counts,
        distinct_rests,
        distinct_totals,
        get_padded(baf_index, n_baf_padded, 0, 0),
        get_padded(baf_weights, n_baf_padded, 0, 0.0),
        np.int64(n_baf),
        log_state_sums,
        data_term,
    )


def get_digamma_difference(x: Any, k: Any) -> Any:
    """`digamma(x + k) - digamma(x)`, as `hmm_utils.digamma_difference_element`: asymptotic above `x = 100`.

    Args:
        x: positive reals.
        k: counts.

    Returns:
        The difference.
    """
    big = x > 100.0
    xs, ks = jnp.where(big, x, 1.0), jnp.where(big, k, 0.0)
    asymptotic = jnp.log1p(ks / xs) + ks / (2 * xs * (xs + ks)) + ks * (2 * xs + ks) / (12 * xs**2 * (xs + ks) ** 2)
    xd, kd = jnp.where(big, 1.0, x), jnp.where(big, 0.0, k)
    return jnp.where(big, asymptotic, jax.scipy.special.digamma(xd + kd) - jax.scipy.special.digamma(xd))


@jax.custom_jvp
def get_log_rising(x: Any, k: Any) -> Any:
    """`lgamma(x + k) - lgamma(x)`, by Stirling's series from `rising_factorial_threshold`, where the lgammas cancel.

    Args:
        x: positive reals.
        k: counts, without a derivative.

    Returns:
        The log rising factorial.
    """
    big = x >= rising_factorial_threshold
    xs = jnp.where(big, x, 1.0)
    stirling = (xs - 0.5) * jnp.log1p(k / xs) + k * jnp.log(xs + k) - k + 1.0 / (12.0 * (xs + k)) - 1.0 / (12.0 * xs)
    xd = jnp.where(big, 1.0, x)
    return jnp.where(big, stirling, jax.scipy.special.gammaln(xd + k) - jax.scipy.special.gammaln(xd))


get_log_rising.defjvps(lambda x_dot, _, x, k: x_dot * get_digamma_difference(x, k), None)


def get_nb_logpmf(counts: Any, exposures: Any, log_rates: Any, alphas: Any) -> Any:
    """`hmm_emission_nb.nb_logpmf_element` without `-lgamma(k + 1)`; 0 at no exposure.

    Args:
        counts: `y`.
        exposures: the baseline, broadcast against `counts`.
        log_rates: `log_mu - log Z_c`, broadcast.
        alphas: the dispersions, broadcast.

    Returns:
        The log-pmf, the broadcast shape.
    """
    alphas = jnp.maximum(alphas, 1.0e-10)
    observed = exposures > 0.0
    log_x = jnp.log(alphas) + jnp.log(jnp.where(observed, exposures, 1.0)) + log_rates
    log1px = jnp.logaddexp(0.0, log_x)
    value = get_log_rising(1.0 / alphas, counts) - log1px / alphas + counts * (log_x - log1px)
    return jnp.where(observed, value, 0.0)


def get_bb_log_risings(counts: Any, rests: Any, totals: Any, p_binom: Any, taus: Any) -> tuple[Any, Any, Any]:
    """The BB log-pmf's three log rising factorials, each shape floored at 1e-10; each count broadcasts on its own.

    Args:
        counts: `b`.
        rests: `n - b`.
        totals: `n`.
        p_binom: the B fractions, broadcast.
        taus: the concentrations, broadcast.

    Returns:
        `log (a)_b`, `log (a')_{n - b}` and `log (a + a')_n`, `a = p tau`, `a' = (1 - p) tau`.
    """
    a, b = jnp.maximum(p_binom * taus, 1.0e-10), jnp.maximum((1.0 - p_binom) * taus, 1.0e-10)
    return get_log_rising(a, counts), get_log_rising(b, rests), get_log_rising(a + b, totals)


def get_bb_logpmf(counts: Any, totals: Any, p_binom: Any, taus: Any) -> Any:
    """`hmm_emission_bb.bb_logpmf_element` without `log C(n, k)`, each shape floored at 1e-10.

    Args:
        counts: `b`.
        totals: `n`, broadcast against `counts`.
        p_binom: the B fractions, broadcast.
        taus: the concentrations, broadcast.

    Returns:
        The log-pmf, the broadcast shape.
    """
    rise_b, rise_rest, rise_n = get_bb_log_risings(counts, totals - counts, totals, p_binom, taus)
    return rise_b + rise_rest - rise_n


def get_block_sum(term: Any, n_valid: Any, rows: tuple[Any, ...]) -> Any:
    """`sum(term(*rows))` over blocks of `get_block_size` rows, a block from `n_valid` on skipped.

    Args:
        term: the scalar sum over a block's rows.
        n_valid: the valid rows, the rest padding.
        rows: arrays sharing their first axis, a multiple of the block.

    Returns:
        The sum.
    """
    size = rows[0].shape[0]
    block = min(get_block_size(), size)
    blocks = tuple(jnp.reshape(v, (size // block, block, *v.shape[1:])) for v in rows)

    def add_block(total: Any, step: tuple[Any, tuple[Any, ...]]) -> tuple[Any, None]:
        start, chunk = step
        value = jax.lax.cond(start < n_valid, lambda c: term(*c), lambda c: jnp.zeros((), jnp.float64), chunk)
        return total + value, None

    starts = jnp.arange(0, size, block)
    return jax.lax.scan(add_block, jnp.zeros((), jnp.float64), (starts, blocks))[0]


def get_log_normalizers(log_mu: Any, log_state_sums: Any) -> Any:
    """`emission.get_log_normalizers` in `jax.numpy`: `log Z_c = logsumexp_s(log mu_s + log S_cs)`, `(n_clones,)`.

    Args:
        log_mu: `(K,)`.
        log_state_sums: `(n_clones, K)`.

    Returns:
        The log normalizers.
    """
    return jax.scipy.special.logsumexp(log_mu[None, :] + log_state_sums, axis=1)


def get_cost(
    theta: Any,
    held: CnaHmmParameters,
    weights: ClassWeights,
    *,
    n_states: int,
    params: str,
    optimize_nb: bool,
    use_logit: bool,
) -> Any:
    """The EM cost at fixed posteriors, `-sum_{c,k,s} W_cks log P(class k | state s, clone c)`.

    Args:
        theta: the optimizer's vector, `hmm_params.Theta` ravelled.
        held: the values of the blocks `theta` does not carry.
        weights: the `ClassWeights`.
        n_states: `K`.
        params: the fit's channels.
        optimize_nb: whether any bin has a read-depth baseline.
        use_logit: `theta`'s BAF block is a logit.

    Returns:
        The cost, a scalar.
    """
    parameters = from_theta(get_unravel_theta(n_states, params, optimize_nb)(theta), held, n_states, use_logit)
    log_mu = jnp.reshape(parameters.log_mu, -1)
    log_z = None if weights.log_state_sums is None else get_log_normalizers(log_mu, weights.log_state_sums)
    # NB a fitted dispersion is one for every state (`hmm_params.Theta`): its lgamma once per class, not per state.
    alphas = jnp.reshape(parameters.alphas, -1)
    alphas = alphas[:1] if get_fitted_blocks(params, optimize_nb)["log_alpha"] else alphas

    def get_rdr_block(clones: Any, counts: Any, exposures: Any, class_weights: Any) -> Any:
        log_rates = log_mu[None, :] if log_z is None else log_mu[None, :] - log_z[clones][:, None]
        return jnp.sum(class_weights * get_nb_logpmf(counts[:, None], exposures[:, None], log_rates, alphas))

    # NB the second phase scores p -> 1 - p, as `bb_switch_logpmf_vector`.
    p_binom = jnp.reshape(parameters.p_binom, -1)
    phases = jnp.stack([p_binom, 1.0 - p_binom])[: weights.baf_weights.shape[1]]
    rise_b, rise_rest, rise_n = get_bb_log_risings(
        weights.baf_counts[:, None, None],
        weights.baf_rests[:, None, None],
        weights.baf_totals[:, None, None],
        phases[None],
        jnp.reshape(parameters.taus, -1)[None, None, :],
    )

    def get_baf_block(index: Any, class_weights: Any) -> Any:
        return jnp.sum(class_weights * (rise_b[index[:, 0]] + rise_rest[index[:, 1]] - rise_n[index[:, 2]]))

    rdr = get_block_sum(
        get_rdr_block,
        weights.n_rdr_valid,
        (weights.rdr_clones, weights.rdr_counts, weights.rdr_exposures, weights.rdr_weights),
    )
    baf = get_block_sum(get_baf_block, weights.n_baf_valid, (weights.baf_index, weights.baf_weights))
    return -(rdr + baf + weights.data_term)


@functools.partial(jax.jit, static_argnames=("n_states", "params", "optimize_nb", "use_logit"))
def get_cost_and_gradient(
    theta: Any,
    held: CnaHmmParameters,
    weights: ClassWeights,
    *,
    n_states: int,
    params: str,
    optimize_nb: bool,
    use_logit: bool,
) -> tuple[Any, Any]:
    """`get_cost` and its gradient in `theta`, `jax.value_and_grad`, compiled once per bucketed shape.

    Args:
        theta: the optimizer's vector.
        held: the values of the blocks `theta` does not carry.
        weights: the `ClassWeights`.
        n_states: `K`.
        params: the fit's channels.
        optimize_nb: whether any bin has a read-depth baseline.
        use_logit: `theta`'s BAF block is a logit.

    Returns:
        The cost and its gradient.
    """
    return jax.value_and_grad(get_cost)(
        theta, held, weights, n_states=n_states, params=params, optimize_nb=optimize_nb, use_logit=use_logit
    )
