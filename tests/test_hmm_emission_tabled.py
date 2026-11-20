"""`get_clone_decoded_fields`' tabled kernels, `rdr_logpmf_tabled` and `baf_logpmf_tabled`, against scipy and the dense
phased field.

Referees: scipy's NB and beta-binomial log-pmfs summed over each clone's decoded (state, phase) path, to 1e-12
relative, on the dense, CSR and stored-entry arms: counts up to `hmm.count_table_extent` read the lgamma tables, a
count above it the element kernels, so both arms are reached. The phased field: the phased dense emission reduced
by `sandbox.spot_assignment`'s `compute_loglike_spot_assignment_given_emission`, bitwise on dense counts, to 1e-12 on
CSR. The element kernels: `hmm_nophasing`'s, bitwise; the tables and state parameters: `math.lgamma` and scipy's
closed forms; the `labelling_unary` encoding: `np.unique`.
"""

from __future__ import annotations

import math
from typing import Any, NamedTuple

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.special
import scipy.stats

from cnamaste.config import YAMLConfig, set_global_config
from cnamaste.emission import CnaEmissionConfounders, get_log_emission, get_state_major, get_unconfounded
from cnamaste.hmm_emission_bb import bb_logpmf_element, bb_switch_logpmf_element
from cnamaste.hmm_emission_nb import nb_logpmf_element
from cnamaste.hmm_emission_tabled import (
    ClonePaths,
    CountTables,
    DenseOperands,
    FieldArgs,
    baf_logpmf_at_dense_total,
    baf_logpmf_at_stored_total,
    baf_logpmf_tabled,
    baf_switch_logpmf_tabled,
    count_table_size,
    get_clone_decoded_fields,
    get_count_tables,
    get_labelling_unary_encoding,
    get_state_parameters,
    rdr_logpmf_at_count,
    rdr_logpmf_at_stored_baseline,
    rdr_logpmf_tabled,
)
from cnamaste.hmm_params import CnaHmmParameters
from cnamaste.hmm_utils import log_factorial
from cnamaste.hmrf import get_labelling_unary
from cnamaste.sandbox.spot_assignment import compute_loglike_spot_assignment_given_emission
from cnamaste.sparse import values_at_stored_entries
from cnamaste.spatio_genomic_counts import PseudobulkCounts, SpatioGenomicCounts, get_encoding_channels
from labelled_counts import get_labelled_counts

ARMS = {"dense": (), "csr": ("counts",), "stored": ("counts", "base", "b_counts", "totals")}
"""The arrays given as CSR per arm: dense reaches `rdr_field_dense` and `baf_field_dense_total`, csr `rdr_field_csr`,
stored `rdr_field_stored_baseline` and `baf_field_stored_total`."""


@pytest.mark.parametrize("arm", sorted(ARMS))
@pytest.mark.parametrize("extent", [1000, 10], ids=["tabled", "element"])
def test_the_unary_is_scipy_s_log_pmf_summed_over_each_clone_s_path(extent: int, arm: str) -> None:
    """Two spots, two clones on two states, both phases; a zero baseline scores 0, as the NB kernels do; phase 1 is
    the beta-binomial of the A count, including at a depth the CSR does not store (spot 1, segment 1)."""
    set_global_config(YAMLConfig({"hmm": {"count_table_extent": extent}}))
    arrays = {
        "counts": np.array([[0.0, 3.0, 12.0, 40.0], [1.0, 0.0, 7.0, 25.0]]),
        "base": np.array([[5.0, 2.0, 0.0, 30.0], [4.0, 1.0, 9.0, 20.0]]),
        "b_counts": np.array([[0.0, 2.0, 5.0, 10.0], [1.0, 0.0, 3.0, 12.0]]),
        "totals": np.array([[0.0, 4.0, 10.0, 25.0], [3.0, 0.0, 8.0, 30.0]]),
    }
    paths = np.array([[0, 2, 1, 3], [1, 0, 3, 0]])
    log_mu, alphas = np.log(np.array([1.0, 1.5])), np.array([0.1, 0.4])
    p_binom, taus = np.array([0.5, 0.2]), np.array([60.0, 20.0])
    given = {name: sp.csr_matrix(a) if name in ARMS[arm] else a for name, a in arrays.items()}
    encoding = get_labelling_unary_encoding(*given.values())
    rdr, baf = get_clone_decoded_fields(
        *given.values(), paths, log_mu, alphas, p_binom, taus, np.ones(2), encoding=encoding
    )
    got = rdr + baf

    counts, base, b_counts, totals = arrays.values()
    for clone in range(2):
        state, phase = paths[clone] % 2, paths[clone] // 2
        mean = base * np.exp(log_mu[state])
        with np.errstate(divide="ignore", invalid="ignore"):
            rdr = scipy.stats.nbinom.logpmf(counts, 1.0 / alphas[state], 1.0 / (1.0 + alphas[state] * mean))
        rdr = np.where(base > 0.0, rdr, 0.0)
        a, b = p_binom[state] * taus[state], (1.0 - p_binom[state]) * taus[state]
        baf = scipy.stats.betabinom.logpmf(np.where(phase == 1, totals - b_counts, b_counts), totals, a, b)
        np.testing.assert_allclose(got[:, clone], (rdr + baf).sum(axis=1), rtol=1e-12)


def get_field_args(tabled: bool, extent: int = 64) -> tuple[FieldArgs, np.ndarray]:
    """Three states, two clones on four segments, both phases, tables over every count below `extent`.

    Args:
        tabled: whether the kernels read the tables.
        extent: 1 + the largest tabled count.

    Returns:
        The `FieldArgs` and the per-state `(alpha, beta)` of the beta-binomial, (2, n_states).
    """
    rng = np.random.default_rng(13)
    nb, bb = get_state_parameters(
        np.log([0.5, 1.0, 1.7]),
        np.array([0.05, 0.3, 2.0]),
        np.array([0.37, 0.5, 0.0]),
        np.array([7.7, 60.0, 300.0]),
        1e-10,
    )
    counts = np.arange(extent)
    tables = CountTables(tabled, counts.astype(np.int32), *get_count_tables(counts, nb.r, bb.alpha, bb.beta))
    clones = ClonePaths(rng.integers(0, 3, (2, 4)), np.array([[0, 1, 0, 1], [1, 1, 0, 0]]), np.array([1.0, 1.3]))
    dense = DenseOperands(
        rng.random((3, 4)) * 20.0, rng.integers(0, 30, (3, 4)).astype(np.float64), np.zeros(0), np.zeros(0)
    )
    return FieldArgs(clones, nb, bb, tables, dense), np.stack([bb.alpha, bb.beta])


@pytest.mark.parametrize("tabled", [True, False], ids=["tabled", "element"])
def test_the_tabled_element_kernels_are_hmm_nophasing_s_bitwise(tabled: bool) -> None:
    """At 200 random (count, baseline, B count, depth, spot, segment, clone), B counts above the depth included:
    `rdr_logpmf_tabled` and its two readers are `nb_logpmf_element`; `baf_logpmf_tabled` and its two readers
    `bb_logpmf_element`, then `bb_switch_logpmf_element` at phase 1; bitwise, tabled or not (the data term
    `log_factorial`'s, numba's lgamma). Fails where a table column is misread or a phase is not switched."""
    args, (bb_alpha, bb_beta) = get_field_args(tabled)
    rng = np.random.default_rng(17)
    for _ in range(200):
        spot, segment, clone = rng.integers(0, 3), rng.integers(0, 4), rng.integers(0, 2)
        count, total, baseline = float(rng.integers(0, 64)), float(rng.integers(0, 40)), rng.random() * 20.0
        b_count = float(rng.integers(0, 50))
        state, phase = args.clones.paths[clone, segment], args.clones.phases[clone, segment]
        scale, alpha = args.clones.scales[clone], args.nb.alpha[state]
        nb_args = (alpha, args.nb.r[state], args.nb.lgamma_r[state], log_factorial(np.array([count]))[0])
        rdr = nb_logpmf_element(count, baseline * scale * args.nb.mu[state], *nb_args)
        assert rdr_logpmf_tabled(count, baseline, segment, clone, args) == rdr
        assert rdr_logpmf_at_stored_baseline(baseline, count, spot, segment, clone, args) == rdr
        at_dense = nb_logpmf_element(count, args.dense.baseline[spot, segment] * scale * args.nb.mu[state], *nb_args)
        assert rdr_logpmf_at_count(count, spot, segment, clone, args) == at_dense

        a, b = bb_alpha[state], bb_beta[state]
        baf = bb_logpmf_element(b_count, total, a, b)
        if phase == 1:
            baf = bb_switch_logpmf_element(baf, b_count, total, a, b)
            assert baf_switch_logpmf_tabled(bb_logpmf_element(b_count, total, a, b), b_count, total, state, args) == baf
        assert baf_logpmf_tabled(b_count, total, segment, clone, args) == baf
        assert baf_logpmf_at_stored_total(total, b_count, spot, segment, clone, args) == baf
        dense_b = args.dense.b_counts[spot, segment]
        at_dense = bb_logpmf_element(dense_b, total, a, b)
        at_dense = bb_switch_logpmf_element(at_dense, dense_b, total, a, b) if phase == 1 else at_dense
        assert baf_logpmf_at_dense_total(total, spot, segment, clone, args) == at_dense


def test_the_count_tables_are_lgamma() -> None:
    """`get_count_tables` at counts 0 to 999: each table `math.lgamma` of its argument, to 2e-15 relative and 1e-15
    absolute where lgamma is near 0 (measured 1.7e-15 compiled, bitwise in Python: numba's libm and CPython's lgamma
    differ by ulps). Fails where a table is shifted by a count or reads another state's parameter."""
    r, a, b = np.array([20.0, 0.5]), np.array([2.849, 1e-10]), np.array([4.851, 300.0])
    counts = np.arange(1000)
    tables = get_count_tables(counts, r, a, b)
    lgamma = np.vectorize(math.lgamma)
    j = counts.astype(np.float64)
    expected = (
        lgamma(j + 1.0),
        lgamma(j + r[:, None]),
        lgamma(j + a[:, None]),
        lgamma(j + b[:, None]),
        lgamma(j + a[:, None] + b[:, None]),
    )
    for got, want in zip(tables, expected):
        np.testing.assert_allclose(got, want, rtol=2e-15, atol=1e-15)


def test_the_state_parameters_are_the_closed_forms() -> None:
    """`exp(log mu)`, alpha floored at 1e-10, `r = 1 / alpha`, the shapes `p tau`, `(1 - p) tau` floored at EPS,
    bitwise; `lgamma(r)` scipy's `gammaln` to 1e-15 relative (measured 1.9e-16 compiled, 7.8e-16 in Python);
    `log B(a, b)` scipy's `betaln` to 1e-13 relative (measured 1.2e-15 compiled, 9.9e-15 in Python)."""
    log_mu, alphas = np.log([0.5, 1.0, 1.7]), np.array([0.05, 1e-12, 2.0])
    p_binom, taus = np.array([0.37, 0.5, 0.0]), np.array([7.7, 60.0, 300.0])
    nb, bb = get_state_parameters(log_mu, alphas, p_binom, taus, 1e-10)
    alpha = np.maximum(alphas, 1e-10)
    assert np.array_equal(nb.mu, [math.exp(v) for v in log_mu]) and np.array_equal(nb.alpha, alpha)
    assert np.array_equal(nb.r, 1.0 / alpha)
    np.testing.assert_allclose(nb.lgamma_r, scipy.special.gammaln(1.0 / alpha), rtol=1e-15, atol=0.0)
    a, b = np.maximum(p_binom * taus, 1e-10), np.maximum((1.0 - p_binom) * taus, 1e-10)
    assert np.array_equal(bb.alpha, a) and np.array_equal(bb.beta, b)
    np.testing.assert_allclose(bb.log_beta, scipy.special.betaln(a, b), rtol=1e-13, atol=0.0)


@pytest.mark.parametrize(
    ("counts", "expected"),
    [([[0.0, 3.0], [7.0, 2.0]], 8), ([0.0], 1), ([100.0], 101), ([-1.0, 3.0], 0), ([2.5, 3.0], 0), ([101.0], 0)],
    ids=["2-D", "zero", "at extent", "negative", "non-integer", "over extent"],
)
def test_the_count_table_size_is_one_plus_the_largest_count_or_zero(counts: Any, expected: int) -> None:
    """At extent 100: 1 + max for non-negative integers up to 100, else 0. Fails where an untabulable count is read."""
    assert count_table_size(np.array(counts), 100) == expected


def test_the_labelling_unary_encoding_is_np_unique_s() -> None:
    """The `labelling_unary` encoding's counts are `np.unique` of 0, the read counts, B counts, depths and `n - b` over
    b <= n, and each count's class its index there, -1 elsewhere. Fails where a count is missed, or a pair with b > n
    marks a difference."""
    rng = np.random.default_rng(23)
    umis = rng.integers(100, 130, (5, 7)).astype(np.float64)
    totals = rng.integers(1, 40, (5, 7)).astype(np.float64)
    b_counts = rng.integers(0, 45, (5, 7)).astype(np.float64)
    assert np.any(b_counts > totals)
    key = "labelling_unary"
    read = PseudobulkCounts(umis, np.ones_like(umis), b_counts, totals, None, {})
    read = read.update_encoding(key, get_encoding_channels()[key], common_zero_depth=False)
    kept = b_counts <= totals
    expected = np.unique(
        np.concatenate([[0], umis.ravel(), b_counts.ravel(), totals.ravel(), totals[kept] - b_counts[kept]])
    )
    (counts,), (totals_read,) = read.get_unique(key)
    classes = read.get_equivalence_class(key)[0]
    assert np.array_equal(counts, expected) and np.array_equal(totals_read, expected)
    assert np.array_equal(classes[counts], np.arange(counts.size))
    assert np.count_nonzero(classes >= 0) == counts.size == classes.size - np.count_nonzero(classes == -1)


def get_dense_phased_field(
    single_X: np.ndarray,
    base: np.ndarray,
    total: np.ndarray,
    res: dict[str, np.ndarray],
    pred: np.ndarray,
) -> np.ndarray:
    """The dense phased field: `get_log_emission`'s phased lattice at clone c's baseline, reduced by
    `compute_loglike_spot_assignment_given_emission`."""
    n_obs, _, n_spots = single_X.shape
    n_clones = pred.shape[1]
    parameters = CnaHmmParameters(None, *(res[f"new_{k}"] for k in ("log_mu", "p_binom", "alphas", "taus")))
    nb_valid, bb_valid = (base > 0).sum(axis=0), (total > 0).sum(axis=0)
    field = np.empty((n_spots, n_clones))
    for clone in range(n_clones):
        exposure = base * np.exp(-res["new_log_mu_shift"][clone])
        counts = PseudobulkCounts(single_X[:, 0].T, exposure.T, single_X[:, 1].T, total.T, None, {})
        rdr, baf = get_log_emission(
            counts, parameters, get_unconfounded(1), layout="lattice", phased=True, data_terms=True
        )
        rdr, baf = get_state_major(rdr), get_state_major(baf)
        reduced = compute_loglike_spot_assignment_given_emission(
            n_spots, nb_valid, bb_valid, rdr, baf, pred, n_obs, n_clones
        )
        field[:, clone] = reduced[:, clone]
    return field


class PhasedCase(NamedTuple):
    """One phased field input: the concentrations, and the read counts' offset and scale."""

    taus: list[float]
    depth_shift: float
    depth_scale: int


CASES = {
    # NB p_binom 0.37 at tau 7.7 puts alpha = 2.849 below n, where (alpha + n) - k rounds off (n - k) + alpha.
    "tabled": PhasedCase([60.0, 7.7, 300.0], 0.0, 1),
    "rising": PhasedCase([60.0, 2.0e7, 300.0], 0.0, 1),
    "non-integer": PhasedCase([60.0, 7.7, 300.0], 0.5, 1),
    "realized": PhasedCase([60.0, 7.7, 300.0], 0.0, 30_000),
}


@pytest.mark.parametrize("sparse", [False, True], ids=["dense", "csr"])
@pytest.mark.parametrize("case", sorted(CASES))
def test_the_phased_field_is_the_dense_phased_field_bitwise(case: str, sparse: bool) -> None:
    """`get_labelling_unary`, phased and tabled, == the dense emission reduced, bitwise on dense counts and to 1e-12
    relative on CSR, whose read depth sums the zero counts once per spot total (`rdr_field_sized_csr`): both phases on
    every path, a library shift per clone, a NaN tumour proportion; a rising-factorial state and non-integer counts
    take the element kernels, counts to 9e5 with few distinct values the tables of those counts alone."""
    set_global_config(YAMLConfig({"hmm": {"count_table_extent": 1_000_000}}))
    rng = np.random.default_rng(7)
    n_obs, n_spots, n_states = 12, 5, 3
    total = rng.integers(0, 40, (n_obs, n_spots)) * (rng.random((n_obs, n_spots)) < 0.7)
    b_counts = rng.binomial(total, 0.3).astype(np.float64)
    depth = rng.integers(0, 30, (n_obs, n_spots)) * CASES[case].depth_scale + CASES[case].depth_shift
    normal = np.arange(n_spots) < 3
    single_X = np.stack([depth.astype(np.float64), b_counts], axis=1)
    total = total.astype(np.float64)
    res = {
        "new_log_mu": np.log(np.array([1.0, 1.5, 0.5])),
        "new_alphas": np.array([0.1, 0.4, 0.05]),
        "new_p_binom": np.array([0.5, 0.37, 0.3]),
        "new_taus": np.array(CASES[case].taus),
        "new_log_mu_shift": np.array([0.0, 0.3]),
    }
    pred = rng.integers(0, 2 * n_states, (n_obs, 2))
    assert ((pred < n_states).any(axis=0) & (pred >= n_states).any(axis=0)).all()

    backing = sp.csr_matrix if sparse else np.asarray
    counts = get_labelled_counts(*(backing(c) for c in (single_X[:, 0], single_X[:, 1], total)), normal_spot=normal)
    base = counts.normal_baseline_sized.toarray().T
    expected = get_dense_phased_field(single_X, base, total, res, pred)
    parameters = CnaHmmParameters(None, *(res[f"new_{k}"] for k in ("log_mu", "p_binom", "alphas", "taus")))
    confounders = CnaEmissionConfounders(np.zeros((2, 1)), res["new_log_mu_shift"][:, None])
    got = get_labelling_unary(counts, parameters, confounders, pred.T, phased=True)
    if sparse:
        np.testing.assert_allclose(got, expected, rtol=1e-12, atol=0.0)
    else:
        np.testing.assert_array_equal(got, expected)


def get_sized_case(seed: int) -> tuple[Any, CnaHmmParameters, CnaEmissionConfounders, np.ndarray]:
    """400 spots by 60 segments, CSR counts with 70% zeros, at most 25 distinct spot UMI totals, a segment no normal
    spot covers (a zero baseline), three clones on three states with a library shift each.

    Returns:
        The labelled counts (sized baseline), parameters, confounders and `(n_clones, n_obs)` paths.
    """
    rng = np.random.default_rng(seed)
    n_obs, n_spots = 60, 400
    depth = rng.integers(1, 12, (n_obs, n_spots)) * (rng.random((n_obs, n_spots)) < 0.3)
    # NB 25 spot totals, each spot's counts rescaled to one: the zero sums are shared between spots.
    depth[:, 25:] = depth[:, rng.integers(0, 25, n_spots - 25)]
    depth[7] = 0
    total = rng.integers(0, 40, (n_obs, n_spots)) * (rng.random((n_obs, n_spots)) < 0.1)
    b_counts = rng.binomial(total, 0.3)
    normal = np.arange(n_spots) < 80
    counts = get_labelled_counts(
        *(sp.csr_matrix(c.astype(np.float64)) for c in (depth, b_counts, total)), normal_spot=normal
    )
    parameters = CnaHmmParameters(
        None, np.log(np.array([1.0, 1.5, 0.5])), np.array([0.5, 0.37, 0.3]), np.array([0.1, 0.4, 0.05]),
        np.array([60.0, 7.7, 300.0]),
    )  # fmt: skip
    confounders = CnaEmissionConfounders(np.zeros((3, 1)), np.array([[0.0], [0.3], [-0.2]]))
    return counts, parameters, confounders, rng.integers(0, 3, (3, n_obs))


@pytest.mark.parametrize("extent", [1_000_000, 4], ids=["tabled", "element"])
def test_the_coverage_summed_field_is_the_csr_field_to_rounding(extent: int) -> None:
    """`rdr_field_sized_csr`: on 400 spots with at most 25 distinct UMI totals, the field on the sized baseline is the
    field on that baseline as a dense array, `rdr_field_csr`'s per-segment sum (refereed by scipy above), to 1e-12
    relative, a zero-baseline segment included, tabled and element kernels. Fails where a stored count's zero term is
    not removed, a spot reads another total's zero sum, or a zero baseline scores."""
    set_global_config(YAMLConfig({"hmm": {"count_table_extent": extent}}))
    counts, parameters, confounders, paths = get_sized_case(seed=19)
    sized = counts.normal_baseline_sized
    assert np.unique(sized.spots_coverage).size <= 25
    channels = (counts.umis, counts.alt_snp_umis, counts.total_snp_umis)
    scales = np.exp(-np.asarray(confounders.log_mu_shift).reshape(-1))
    rest = (paths, parameters.log_mu, parameters.alphas, parameters.p_binom, parameters.taus, scales)
    fields = {}
    for name, base in (("sized", sized), ("dense", sized.toarray())):
        encoding = get_labelling_unary_encoding(channels[0], base, *channels[1:])
        fields[name] = get_clone_decoded_fields(channels[0], base, *channels[1:], *rest, encoding=encoding)
    for got, want in zip(fields["sized"], fields["dense"], strict=True):
        np.testing.assert_allclose(got, want, rtol=1e-12, atol=0.0)


def test_the_held_b_counts_are_the_b_counts_read_again() -> None:
    """The depth holds the B counts at its stored entries for the last B counts read: the field from the held read is
    the field read afresh, bitwise, and other B counts are read again, not the held. Fails where a held array is read
    for counts it was not built from."""
    set_global_config(YAMLConfig({"hmm": {"count_table_extent": 1_000_000}}))
    counts, parameters, _, paths = get_sized_case(seed=23)
    total, b_counts = counts.total_snp_umis, counts.alt_snp_umis
    held = total.get_held_values_at_stored_entries(b_counts)
    assert held is total.get_held_values_at_stored_entries(b_counts)
    np.testing.assert_array_equal(held, values_at_stored_entries(total.values, b_counts.values))
    other = SpatioGenomicCounts(b_counts.values.multiply(2))
    np.testing.assert_array_equal(
        total.get_held_values_at_stored_entries(other), values_at_stored_entries(total.values, other.values)
    )
    channels = (counts.umis, counts.normal_baseline_sized, b_counts, total)
    args = (*channels, paths, parameters.log_mu, parameters.alphas, parameters.p_binom, parameters.taus, np.ones(3))
    encoding = get_labelling_unary_encoding(*channels)
    fresh = (SpatioGenomicCounts(total.values.copy()),)
    for got, want in zip(
        get_clone_decoded_fields(*args, encoding=encoding),
        get_clone_decoded_fields(*channels[:3], *fresh, *args[4:], encoding=encoding),
        strict=True,
    ):
        np.testing.assert_array_equal(got, want)
