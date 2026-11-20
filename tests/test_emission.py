"""`cnamaste.emission` against the paper's expressions written out by hand: the admixture of `eqn:ab_admixture`, the
library normalizer `log Z_c` and the NB mean of `eqn:nb_emission`, the minor count, and `get_log_emission`'s
layouts against each other.

Three states, a gain, the normal and a copy-neutral LOH of total 3, at theta 0, 0.25 and 1. Below 1 ms.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import scipy.special
from scipy import stats

from cnamaste.ab_decode.lattice import get_lattice_parameters
from cnamaste.emission import (
    CnaEmissionConfounders,
    get_admixed_moments,
    get_decoded_emission,
    get_encoded,
    get_hard_state_posteriors,
    get_log_emission,
    get_log_mu_shift,
    get_minor_allele_count,
    get_nb_mean,
    get_state_major,
)
from cnamaste.hmm_params import CnaHmmParameters
from cnamaste.spatio_genomic_counts import PseudobulkCounts

COPIES = np.array([[2, 1], [1, 1], [3, 0]])
"""`(A, B)`: mu = (3/2, 1, 3/2), p = (1/3, 1/2, 0)."""


@pytest.mark.parametrize("theta", [0.0, 0.25, 1.0])
def test_the_admixed_moments_are_the_paper_s(theta: float) -> None:
    """Depth `theta + (1 - theta)(A + B) / 2` and B fraction `(theta + (1 - theta) B) / (2 theta + (1 - theta)(A + B))`
    to 1e-15 from the lattice's `log_mu` and `p_binom`; the normal's `(1, 1/2)` at 1, and per clone along a trailing
    axis. Fails where theta mixes the wrong moment or the clone axis is misplaced."""
    parameters = get_lattice_parameters(COPIES, None, None)
    depth, fraction = get_admixed_moments(parameters, theta)
    f = theta
    total = COPIES.sum(axis=1)
    np.testing.assert_allclose(depth, f + (1 - f) * total / 2.0, rtol=1e-15, atol=0)
    np.testing.assert_allclose(fraction, (f + (1 - f) * COPIES[:, 1]) / (2 * f + (1 - f) * total), rtol=1e-15, atol=0)
    if theta == 1.0:
        assert np.array_equal(depth, np.ones(3)) and np.array_equal(fraction, np.full(3, 0.5))
    by_clone = get_admixed_moments(parameters, np.array([f, 1.0]))
    np.testing.assert_array_equal(by_clone[0][:, 1], np.ones(3))
    np.testing.assert_allclose(by_clone[1][:, 0], fraction, rtol=1e-15, atol=0)


def test_each_clone_s_log_mu_shift_is_its_bins_logsumexp() -> None:
    """On a path's one-hot weights, `log Z_c = logsumexp(log_mu[s_g] + log lambda_g)` over clone c's bins, `lambda_g`
    each clone's baseline over its sum; on soft weights, `log sum_g lambda_g sum_s w_gs mu_s`; both to 1e-12. A bin
    of no baseline adds nothing. Fails where a clone's bins are read from another clone, or a bin's states are weighed
    other than by `w_gs`."""
    rng = np.random.default_rng(4)
    log_mu, clone_lengths = np.log([0.5, 1.0, 1.5]), np.array([4, 3, 2])
    path = rng.integers(0, 3, clone_lengths.sum())
    base = rng.uniform(1.0, 2.0, (clone_lengths.sum(), 1))
    base[0] = 0.0
    counts = PseudobulkCounts(None, base.T, None, None, clone_lengths, {})
    log_lambda = np.concatenate([np.log(block / block.sum()) for block in np.split(base[:, 0], [4, 7])])
    parameters = CnaHmmParameters(None, log_mu, None, None, None)
    found = get_log_mu_shift(parameters, get_hard_state_posteriors(path, 3), counts)[:, 0]
    terms = log_mu[path] + log_lambda
    expected = [scipy.special.logsumexp(block) for block in np.split(terms, [4, 7])]
    np.testing.assert_allclose(found, expected, rtol=0, atol=1e-12)
    weights = rng.dirichlet(np.ones(3), clone_lengths.sum())
    soft = get_log_mu_shift(parameters, np.log(weights), counts)[:, 0]
    depth = np.exp(log_lambda) * (weights @ np.exp(log_mu))
    np.testing.assert_allclose(soft, np.log([block.sum() for block in np.split(depth, [4, 7])]), rtol=0, atol=1e-12)


def test_the_nb_mean_and_the_minor_allele_count() -> None:
    """The NB mean is `base mu / Z_c`, `base mu` at `log Z_c` 0; the minor count is `min(b, n - b)`. Fails where the mean
    drops `Z_c` or reads it linear, or the fold takes the wrong allele."""
    base, mu = np.array([1.0, 3.0]), np.array([1.5, 0.5])
    np.testing.assert_allclose(get_nb_mean(base, mu, np.log(0.75)), base * mu / 0.75, rtol=1e-15, atol=0)
    np.testing.assert_array_equal(get_nb_mean(base, mu, 0.0), base * mu)
    np.testing.assert_array_equal(get_minor_allele_count(np.array([0, 3, 7]), np.array([5, 5, 10])), [0, 2, 3])


def test_the_path_is_the_lattice_at_the_path_clone_by_clone() -> None:
    """Two clones stacked along bins, each its own `theta_c` and `log Z_c`: the `path` layout is the `lattice` at each
    bin's state, to 1e-12, the second clone's block is its own lattice alone, and the clones as columns, dense and
    encoded, are the stacked lattice per column, to 1e-12; `data_terms=False` drops
    `-lgamma(y + 1)` and log C(n, k) alone, to 1e-12. Fails where the confounders do not expand along
    `clone_lengths`, the layouts score different moments, or a data term stays or a parameter's term is dropped."""
    rng = np.random.default_rng(5)
    n_bins = 8
    total = rng.integers(1, 40, (2 * n_bins, 1)).astype(float)
    X_nb, base = rng.integers(0, 200, total.shape).astype(float), rng.uniform(50, 150, total.shape)
    X_bb = get_minor_allele_count(np.floor(total * rng.uniform(0, 1, total.shape)), total)
    counts = PseudobulkCounts(X_nb.T, base.T, X_bb.T, total.T, np.array([n_bins, n_bins]), {})
    parameters = get_lattice_parameters(COPIES, 0.05, 80.0)
    confounders = CnaEmissionConfounders(np.array([[0.1], [0.4]]), np.log([[0.9], [1.2]]))
    kwargs: dict[str, Any] = {"phased": False, "data_terms": True}
    lattice = [
        get_state_major(v) for v in get_log_emission(counts, parameters, confounders, layout="lattice", **kwargs)
    ]
    path = rng.integers(0, 3, 2 * n_bins)
    along = get_log_emission(counts, parameters, confounders, layout="path", path=path, **kwargs)
    bins = np.arange(2 * n_bins)
    for found, full in zip(along, lattice, strict=True):
        np.testing.assert_allclose(found[:, 0], full[path, bins, 0], rtol=1e-12, atol=0)
    bare = get_log_emission(
        counts, parameters, confounders, layout="path", path=path, **{**kwargs, "data_terms": False}
    )
    minor, gammaln = X_bb, scipy.special.gammaln
    log_binom_coeff = gammaln(total + 1) - gammaln(minor + 1) - gammaln(total - minor + 1)
    np.testing.assert_allclose(bare.rdr, along.rdr + gammaln(X_nb + 1), rtol=1e-12, atol=0)
    np.testing.assert_allclose(bare.baf, along.baf - log_binom_coeff, rtol=1e-12, atol=1e-12)
    second = PseudobulkCounts(X_nb[n_bins:].T, base[n_bins:].T, X_bb[n_bins:].T, total[n_bins:].T, None, {})
    alone = get_log_emission(
        second,
        parameters,
        CnaEmissionConfounders(np.full((1, 1), 0.4), np.full((1, 1), np.log(1.2))),
        layout="lattice",
        **kwargs,
    )
    for found, full in zip(alone, lattice, strict=True):
        np.testing.assert_array_equal(get_state_major(found), full[:, n_bins:])
    by_column = PseudobulkCounts(*(v.reshape(2, n_bins) for v in (X_nb, base, X_bb, total)), None, {})
    encoded = get_encoded(by_column, common_zero_depth=False)
    for found in (
        get_log_emission(by_column, parameters, confounders, layout="lattice", **kwargs),
        get_decoded_emission(
            get_log_emission(encoded, parameters, confounders, layout="lattice", encoding="coded", **kwargs)
        ),
    ):
        for channel, full in zip(found, lattice, strict=True):
            stacked = full[:, :, 0].reshape(-1, 2, n_bins).transpose(1, 2, 0)
            np.testing.assert_allclose(channel, stacked, rtol=1e-12)


def test_the_field_is_the_path_summed_over_bins() -> None:
    """Six spots by spot, two clones' paths, the second at its own `log Z_c`: the `field` is each spot's `path`
    layout summed over bins, the read depth weighted per spot, to 1e-12. Fails where the field reads another clone's
    path or normalizer, or weights the B allele."""
    rng = np.random.default_rng(6)
    n_bins, n_spots = 12, 6
    total = rng.integers(0, 30, (n_bins, n_spots)).astype(float)
    X_nb, base = rng.integers(0, 50, total.shape).astype(float), rng.uniform(5, 30, total.shape)
    X_bb = np.floor(total * rng.uniform(0, 1, total.shape))
    by_spot = PseudobulkCounts(*(v.T for v in (X_nb, base, X_bb, total)), None, {})
    parameters = get_lattice_parameters(COPIES, np.full(3, 0.05), np.full(3, 80.0))
    paths, shifts, weights = rng.integers(0, 3, (2, n_bins)), np.array([0.0, np.log(1.3)]), rng.uniform(0.5, 2, n_spots)
    kwargs: dict[str, Any] = {"phased": False, "data_terms": True}
    rdr, baf = get_log_emission(
        by_spot,
        parameters,
        CnaEmissionConfounders(np.zeros((2, 1)), shifts[:, None]),
        layout="field",
        path=paths,
        spot_weights=weights,
        **kwargs,
    )
    for c in range(2):
        along = get_log_emission(
            by_spot,
            parameters,
            CnaEmissionConfounders(np.zeros((1, 1)), np.full((1, 1), shifts[c])),
            layout="path",
            path=paths[c],
            **kwargs,
        )
        np.testing.assert_allclose(rdr[:, c], weights * along.rdr.sum(axis=0), rtol=1e-12, atol=0)
        np.testing.assert_allclose(baf[:, c], along.baf.sum(axis=0), rtol=1e-12, atol=0)


def test_the_field_refuses_a_normal_fraction() -> None:
    """The field scores unmixed counts: a normal fraction raises `NotImplementedError` naming it."""
    counts = PseudobulkCounts(*(np.ones((2, 1)) for _ in range(4)), None, {})
    confounders = CnaEmissionConfounders(np.full((1, 1), 0.2), np.zeros((1, 1)))
    with pytest.raises(NotImplementedError, match="normal_fraction"):
        get_log_emission(
            counts,
            get_lattice_parameters(COPIES, 0.05, 80.0),
            confounders,
            layout="field",
            phased=False,
            data_terms=True,
            path=np.zeros((1, 2), dtype=np.int64),
            spot_weights=np.ones(1),
        )


@pytest.mark.parametrize("tau", [60.0, 1.0e4])
@pytest.mark.parametrize("encoded", [False, True])
def test_the_admixed_lattice_is_scipy_s_nb_and_beta_binomial(tau: float, encoded: bool) -> None:
    """The decoder's lattice per pair and bin, dense and on encoded counts: `scipy.stats.nbinom` at the admixed depth
    over `Z_c` and `betabinom` at the admixed B fraction, `eqn:ab_admixture` written out, to 1e-10."""
    pairs = np.array([[1, 0], [1, 1], [2, 0], [2, 1], [3, 1], [0, 1], [1, 2]])
    rng = np.random.default_rng(3)
    n = rng.integers(1, 60, (40, 1)).astype(float)
    y, b = rng.integers(0, 300, (40, 1)).astype(float), np.floor(n * rng.uniform(0, 1, (40, 1)))
    base = rng.uniform(80.0, 120.0, (40, 1))
    counts = PseudobulkCounts(y.T, base.T, b.T, n.T, None, {})
    counts = get_encoded(counts, common_zero_depth=False) if encoded else counts
    f, alpha, shift = 0.2, 0.02, 1.07
    totals = pairs.sum(axis=1)
    depth, fraction = f + (1 - f) * totals / 2.0, (f + (1 - f) * pairs[:, 1]) / (2 * f + (1 - f) * totals)
    mean, r = base[None, :, 0] * depth[:, None] / shift, 1.0 / alpha
    expected = stats.nbinom.logpmf(y[None, :, 0], r, r / (r + mean))
    shapes = np.maximum(fraction[:, None] * tau, 1e-10), np.maximum((1.0 - fraction[:, None]) * tau, 1e-10)
    expected += stats.betabinom.logpmf(b[None, :, 0], n[None, :, 0], *shapes)
    parameters = get_lattice_parameters(pairs, alpha, tau)
    confounders = CnaEmissionConfounders(np.full((1, 1), f), np.full((1, 1), np.log(shift)))
    kwargs: dict[str, Any] = {"layout": "lattice", "phased": False, "data_terms": True}
    if encoded:
        rdr, baf = get_decoded_emission(get_log_emission(counts, parameters, confounders, encoding="coded", **kwargs))
    else:
        rdr, baf = get_log_emission(counts, parameters, confounders, **kwargs)
    np.testing.assert_allclose(get_state_major(rdr + baf)[:, :, 0], expected, rtol=1e-10, atol=1e-10)


def test_a_channel_drops_where_its_data_is_absent_and_meets_its_limits() -> None:
    """A bin with no baseline scores its B count alone (`betabinom`), a bin of no allele trials its read count alone
    (`nbinom`); alpha 0 is `poisson` to 1e-4 and tau 1e6 is `binom` to 3e-4 relative, n / tau absolute."""
    pairs = np.array([[1, 0], [1, 1], [2, 1], [3, 1]])
    rng = np.random.default_rng(6)
    n = rng.integers(1, 60, (40, 1)).astype(float)
    n[1] = 0.0
    base = rng.uniform(80.0, 120.0, (40, 1))
    y = rng.poisson(base).astype(float)
    zeroed = base.copy()
    zeroed[0] = 0.0
    b = np.floor(n * rng.uniform(0, 1, (40, 1)))
    f, alpha, tau = 0.2, 0.02, 60.0
    depth, fraction = get_admixed_moments(get_lattice_parameters(pairs, None, None), f)

    def emission(counts: PseudobulkCounts, alpha: float, tau: float) -> Any:
        rdr, baf = get_log_emission(
            counts,
            get_lattice_parameters(pairs, alpha, tau),
            CnaEmissionConfounders(np.full((1, 1), f), np.zeros((1, 1))),
            layout="lattice",
            phased=False,
            data_terms=True,
        )
        return get_state_major(rdr + baf)[:, :, 0]

    found, r = emission(PseudobulkCounts(y.T, zeroed.T, b.T, n.T, None, {}), alpha, tau), 1.0 / alpha
    shapes = np.maximum(fraction * tau, 1e-10), np.maximum((1.0 - fraction) * tau, 1e-10)
    np.testing.assert_allclose(found[:, 0], stats.betabinom.logpmf(b[0, 0], n[0, 0], *shapes), rtol=1e-10)
    np.testing.assert_allclose(found[:, 1], stats.nbinom.logpmf(y[1, 0], r, r / (r + base[1, 0] * depth)), rtol=1e-10)
    poisson = stats.poisson.logpmf(y[None, :, 0], base[None, :, 0] * depth[:, None])
    np.testing.assert_allclose(
        emission(PseudobulkCounts(y.T, base.T, None, None, None, {}), 0.0, tau), poisson, atol=1e-4
    )
    binom = stats.binom.logpmf(b[None, :, 0], n[None, :, 0], fraction[:, None])
    found = emission(PseudobulkCounts(None, None, b.T, n.T, None, {}), alpha, 1.0e6)
    np.testing.assert_allclose(found, binom, rtol=3e-4, atol=float(n.max() / 1.0e6))


def test_z_c_with_admixture_is_the_baseline_weighted_admixed_depth() -> None:
    """`log Z_c = log sum_g lambda_g mu~_{s_g}` at `theta` 0.2 on a 5-bin path, `eqn:ab_admixture`'s depth by hand,
    to 1e-14. Fails where the admixture is left out of `Z_c`."""
    pairs = np.array([[1, 1], [2, 1], [1, 0]])
    path, base = np.array([0, 1, 1, 2, 0]), np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    lattice = get_lattice_parameters(pairs, None, None)
    admixed = lattice._replace(log_mu=np.log(get_admixed_moments(lattice, 0.2)[0]))
    counts = PseudobulkCounts(None, base[None, :], None, None, None, {})
    found = get_log_mu_shift(admixed, get_hard_state_posteriors(path, len(pairs)), counts)[:, 0]
    depth = 0.2 + 0.8 * pairs[path].sum(axis=1) / 2.0
    np.testing.assert_allclose(found, [np.log(base @ depth / base.sum())], rtol=1e-14)
