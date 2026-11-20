"""`ab_decode.decode`'s core, one test per function, and one end-to-end decode of a planted genome over the settings a
run and the HMM's start reach.

Referees: enumeration of every copy path (`clone_viterbi`, the ploidy-constrained Viterbi,
`all_clone_combinatorial_viterbi`, the solvers); the exact joint decode against the sweeps; the Potts energy against
the summed loglikes (`eqn:ab_coupling`); a 0.001 grid (`fit_normal_fraction`); central differences (the dispersion
objective's gradient); the hard EM's `log Z_c` order; and the planted truth (copies, breakpoints,
`theta_c`, `log Z_c`, `DecodeStatus`). The helpers' own tests are `test_ab_decode_model.py`'s; the emission's
`test_emission.py`'s.
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import pytest

import cnamaste.ab_decode.decode as hd
from cnamaste.ab_decode.decode import (
    clone_viterbi,
    clone_viterbi_mean_ploidy_constraint,
    decode_at_ploidy,
    decode_integer_copies,
    fit_normal_fraction,
    get_clone_chains_potts_problem,
    get_clone_viterbi,
    hmm_ab_decoder_dispersion_objective,
    solve_clone_chains,
)
from cnamaste.ab_decode.lattice import get_ab_is_ploidy_neutral, get_ab_lattice, get_lattice_parameters
from cnamaste.ab_decode.model import (
    AbDecodeResult,
    CloneChain,
    DecodeStatus,
    get_all_clone_agreement_prior,
    get_clone_rows,
    get_contig_start_prior,
    get_copy_path_emission_loglike,
    get_copy_path_loglike,
    get_log_mu_shift_start,
    get_ploidy_band,
    in_ploidy_band,
)
from cnamaste.config import YAMLConfig, config_defaults
from cnamaste.emission import (
    CnaEmissionConfounders,
    get_admixed_moments,
    get_hard_state_posteriors,
    get_log_mu_shift,
    get_unconfounded,
)
from cnamaste.hmm.baum_welch import HmmTransitions
from cnamaste.hmm.forward_backward import get_dense_encoded_emission
from cnamaste.hmm_params import CnaHmmParameters
from cnamaste.hmm_utils import get_log_transmat
from cnamaste.potts import pairwise_potts_energy
from cnamaste.sandbox.combinatorial_viterbi import all_clone_combinatorial_viterbi
from cnamaste.spatio_genomic_counts import PseudobulkCounts

PRICE_STEPS = int(config_defaults["ab_decode"]["ploidy_price_steps"])
"""The configuration's ploidy price bisections."""

SOLVERS = ("icm", "expansion")
CAPS = {"max_total_copy": 4, "max_allele_copy": 3}
"""A lattice small enough for the joint Viterbi's `K ** C` to be decoded in this file."""


def get_path_log_z(pairs: np.ndarray, path: np.ndarray, f: float, base: np.ndarray, clone: int) -> float:
    """`get_log_mu_shift` of `clone`'s copy path on `pairs` at `theta_c = f`, `lambda` from the clones'
    `(n_clones, n_obs)` base."""
    lattice = get_lattice_parameters(pairs, None, None)
    admixed = lattice._replace(log_mu=np.log(get_admixed_moments(lattice, f)[0]))
    weights = get_hard_state_posteriors(path, np.size(lattice.log_mu))
    counts = PseudobulkCounts(None, np.asarray(base, dtype=np.float64), None, None, None, {})
    return float(get_log_mu_shift(admixed, weights, counts)[clone, 0])


def config_of(ab_decode: dict[str, Any], max_rdr: float = 10.0) -> YAMLConfig:
    """The configuration a run decodes at, `ab_decode` overriding the defaults."""
    return YAMLConfig({"ab_decode": {"max_rdr": max_rdr, **ab_decode}})


def chain_of(pairs: np.ndarray, lengths: list[int], clone_coupling: float, t: float = 0.99) -> CloneChain:
    """The chain on `pairs` at a uniform contig start."""
    K = pairs.shape[0]
    transitions = HmmTransitions(get_log_transmat(K, t), None, np.array(lengths))
    return CloneChain(transitions, np.full(K, -np.log(K)), clone_coupling)


def solve(chain: CloneChain, loglikes: dict[int, np.ndarray], band: Any, solver: str) -> Any:
    """`solve_clone_chains` at the configuration's min cut."""
    return solve_clone_chains(
        chain,
        loglikes,
        band,
        solver=solver,
        mincut="boykov_kolmogorov",
        icm_sweeps=int(config_defaults["ab_decode"]["icm_sweeps"]),
        ploidy_price_steps=PRICE_STEPS,
    )


def enumerated(chain: CloneChain, loglikes: dict[int, np.ndarray]) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Every copy path of `chain`'s lengths and each clone's loglike of each, written out independently."""
    n = int(chain.lengths.sum())
    paths = np.array(list(itertools.product(range(chain.K), repeat=n)))
    first = np.concatenate([[0], np.cumsum(chain.lengths)[:-1]])
    inner = np.setdiff1d(np.arange(1, n), first)
    steps = np.where(paths[:, inner] == paths[:, inner - 1], chain.log_stay, chain.log_switch).sum(axis=1)
    starts = chain.log_start[paths[:, first]].sum(axis=1)
    return paths, {c: s[paths, np.arange(n)].sum(axis=1) + starts + steps for c, s in loglikes.items()}


def tiny() -> tuple[CloneChain, dict[int, np.ndarray], np.ndarray]:
    """K = 3 pairs, totals 2, 3 and 1, segments of 4 and 2 bins, two clones at random, anchored at (1, 1)."""
    pairs = np.array([[1, 1], [2, 1], [1, 0]])
    transitions = HmmTransitions(get_log_transmat(3, 0.8), None, np.array([4, 2]))
    chain = CloneChain(transitions, get_contig_start_prior(pairs, 0.8, 2, "anchored"), 0.0)
    rng = np.random.default_rng(4)
    return chain, {c: rng.normal(0.0, 1.5, (3, 6)) for c in (1, 2)}, pairs.sum(axis=1)


def test_clone_viterbi_is_the_argmax_over_every_path_and_refuses_another_lattice() -> None:
    """Referee: all 3^6 paths. Each clone's path and loglike are the best's; a score over another K raises
    `ValueError`. Fails where the recursion, the start or a segment boundary is scored wrong."""
    chain, loglikes, _ = tiny()
    paths, value = enumerated(chain, loglikes)
    for c in loglikes:
        path, found = clone_viterbi(chain, c, loglikes[c], None, ploidy_price_steps=PRICE_STEPS)
        assert np.array_equal(path, paths[value[c].argmax()]) and np.isclose(found, value[c].max(), rtol=1e-12)
    with pytest.raises(ValueError):
        clone_viterbi(chain, 1, np.zeros((4, 6)), None, ploidy_price_steps=PRICE_STEPS)


def test_the_ploidy_constrained_viterbi_holds_the_band_or_scores_minus_inf() -> None:
    """Referee: all 3^6 paths. A score whose free path is heavy is priced into the band, its loglike at most the best
    in-band path's (the price is a Lagrangian bound) and equal to `get_copy_path_loglike` on the unpriced score; a
    one-bin lattice of totals 1 and 3 holds no mean within 0.5 of 2: the free path at -inf."""
    chain, loglikes, totals = tiny()
    paths, value = enumerated(chain, loglikes)
    heavy = {c: loglikes[c] + 2.0 * (totals[:, None] == 3) for c in loglikes}
    inside = np.abs(totals[paths].mean(axis=1) - 2) <= get_ploidy_band()
    for c in loglikes:
        free = clone_viterbi(chain, c, heavy[c], None, ploidy_price_steps=PRICE_STEPS)[0]
        path, found, price, _ = clone_viterbi_mean_ploidy_constraint(
            heavy[c], chain, totals, 2, ploidy_price_steps=PRICE_STEPS, name=""
        )
        assert in_ploidy_band(path, totals, 2) and (price < 0.0 or in_ploidy_band(free, totals, 2))
        assert found <= (value[c] + 2.0 * (totals[paths] == 3).sum(axis=1))[inside].max() + 1e-12
        assert found == get_copy_path_loglike(heavy[c], chain, path)
    two = chain_of(np.array([[1, 0], [2, 1]]), [1], 0.0)
    path, found, _, _ = clone_viterbi_mean_ploidy_constraint(
        np.array([[0.0], [-1.0]]), two, np.array([1, 3]), 2, ploidy_price_steps=PRICE_STEPS, name=""
    )
    assert found == -np.inf and path.tolist() == [0]
    light = chain_of(np.array([[1, 0], [0, 1]]), [2], 0.0)
    _, found, price, detail = clone_viterbi_mean_ploidy_constraint(
        np.zeros((2, 2)), light, np.array([1, 1]), 2, ploidy_price_steps=PRICE_STEPS, name=""
    )
    assert found == -np.inf and price > 1.0e6 and detail == "no price reaches the band"


def get_joint_loglikes(emission: np.ndarray, chain: CloneChain) -> tuple[np.ndarray, np.ndarray]:
    """Every joint path of two clones and its loglike, added in `all_clone_combinatorial_viterbi`'s order."""
    _, K, n = emission.shape
    joint = np.array(np.unravel_index(np.arange((K * K) ** n), (K * K,) * n)).T
    paths = np.stack([joint % K, joint // K], axis=1)
    total, first = np.zeros(len(joint)), 0
    for le in chain.lengths:
        for g in range(first, first + le):
            unary = np.where(paths[:, 0, g] != paths[:, 1, g], 0.0 - chain.clone_coupling, 0.0)
            for c in range(2):
                unary = unary + emission[c, paths[:, c, g], g]
                if g == first:
                    unary = unary + chain.log_start[paths[:, c, g]]
            if g == first:
                score = unary
                continue
            for c in range(2):
                score = score + np.where(paths[:, c, g] == paths[:, c, g - 1], chain.log_stay, chain.log_switch)
            score = score + unary
        total, first = total + score, first + le
    return paths, total


@pytest.mark.parametrize("stay", [0.7, 0.1], ids=["stay", "switch"])
@pytest.mark.parametrize("clone_coupling", [0.0, 2.0])
def test_all_clone_combinatorial_viterbi_is_the_argmax_over_every_joint_path(
    clone_coupling: float, stay: float
) -> None:
    """Two clones, K = 4, 4 bins in segments of 2 and 2: the paths and loglike are the best of all 4^8 joint paths',
    bitwise; at coupling 0 each clone's path is its own Viterbi. Fails where the agreement prior is charged per
    agreeing pair, a clone's step crosses axes, or a segment does not restart."""
    rng = np.random.default_rng(9)
    emission = rng.normal(-2.0, 1.5, (2, 4, 4))
    emission[1] = 0.3 * emission[0] + rng.normal(-2.0, 1.5, (4, 4))
    transitions = HmmTransitions(get_log_transmat(4, stay), None, np.array([2, 2]))
    chain = CloneChain(transitions, np.log(np.array([0.4, 0.3, 0.2, 0.1])), clone_coupling)
    paths, loglikes = get_joint_loglikes(emission, chain)
    best = np.argmax(loglikes)
    found, value = all_clone_combinatorial_viterbi(
        emission, chain.log_stay, chain.log_switch, chain.log_start, chain.lengths, clone_coupling
    )
    assert np.array_equal(found, paths[best]) and value == loglikes[best]
    own = get_clone_viterbi(get_dense_encoded_emission(np.transpose(emission, (0, 2, 1))), chain)[0]
    assert all(np.array_equal(found[c], own[c]) for c in range(2)) is (clone_coupling == 0.0)


@pytest.mark.parametrize("solver", SOLVERS)
def test_each_solver_at_no_coupling_is_each_clone_s_own_viterbi(solver: str) -> None:
    """At coupling 0 the joint loglike separates: each clone's own Viterbi and their summed loglike, free and (bar
    `expansion`) under the band. Fails where a solver charges an agreement prior at 0."""
    pairs = get_ab_lattice(params="mp", fold=False, **CAPS)
    chain = chain_of(pairs, [25, 15], 0.0)
    rng = np.random.default_rng(11)
    loglikes = {c: rng.normal(0.0, 2.0, (pairs.shape[0], 40)) for c in (1, 2)}
    for band in (None, (pairs.sum(axis=1), 2)):
        own = {c: clone_viterbi(chain, c, loglikes[c], band, ploidy_price_steps=PRICE_STEPS) for c in loglikes}
        found = solve(chain, loglikes, band, solver)
        if band is not None and solver == "expansion":
            assert all(in_ploidy_band(found[0][c], *band) for c in loglikes)
            continue
        assert all(np.array_equal(own[c][0], found[0][c]) for c in loglikes)
        assert np.isclose(sum(v for _, v in own.values()), found[3], rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("solver", SOLVERS)
def test_coupling_merges_monotonically_keeps_a_paid_for_private_cna_and_is_not_beaten_by_the_exact_decode(
    solver: str,
) -> None:
    """Referee: `eqn:ab_coupling`'s exact optimum. Disagreeing bins never rise with the coupling and none remain at
    50 nats; a private CNA on 20 bins at 1.5 nats each survives coupling 0.5 and is erased at 2.0; no solver scores
    above `all_clone_combinatorial_viterbi` (`sandbox`). Fails where the prior is charged per clone, has the wrong sign, or a sweep beats the
    exact decode."""
    pairs = get_ab_lattice(params="mp", fold=False, **CAPS)
    rng = np.random.default_rng(9)
    loglikes = {c: rng.normal(0.0, 1.0, (pairs.shape[0], 30)) for c in (1, 2)}
    found = [solve(chain_of(pairs, [30], k), loglikes, None, solver) for k in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 50.0)]
    apart = [int((p[1] != p[2]).sum()) for p, *_ in found]
    assert apart == sorted(apart, reverse=True) and apart[-1] == 0 and apart[0] > 0, apart
    chain = chain_of(pairs, [30], 0.5)
    exact = all_clone_combinatorial_viterbi(
        np.stack([loglikes[1], loglikes[2]]),
        chain.log_stay,
        chain.log_switch,
        chain.log_start,
        chain.lengths,
        chain.clone_coupling,
    )[1]
    assert found[2][3] <= exact + 1e-9

    private = {c: np.where(np.arange(pairs.shape[0])[:, None] == 0, 0.0, -3.0) * np.ones((1, 40)) for c in (1, 2)}
    private[2][1, 10:30] = 1.5
    for coupling, survives in ((0.5, True), (2.0, False)):
        paths = solve(chain_of(pairs, [40], coupling), private, None, solver)[0]
        assert bool((paths[2][10:30] == 1).all()) is survives and (paths[1] == 0).all(), coupling


def test_the_ploidy_band_sets_each_decode_status_and_an_unknown_solver_is_refused() -> None:
    """A lattice of totals 1 and 3 holds no path at P = 2: `NO_PATH_IN_PLOIDY_BAND` and a joint -inf under every
    solver; an unknown solver raises `ValueError`."""
    two = np.array([[1, 0], [2, 1]])
    loglikes = {1: np.array([[0.0], [-1.0]]), 2: np.array([[-1.0], [0.0]])}
    for solver in SOLVERS:
        status, joint = solve(chain_of(two, [1], 0.5), loglikes, (two.sum(axis=1), 2), solver)[2:]
        assert joint == -np.inf and set(status.values()) == {DecodeStatus.NO_PATH_IN_PLOIDY_BAND}, solver
    with pytest.raises(ValueError):
        solve(chain_of(two, [1], 0.5), loglikes, None, "viterbi")


def test_the_potts_problem_energy_is_the_negated_joint_loglike() -> None:
    """Three clones on 12 bins in segments of 7 and 5, random copy paths: the Potts energy is minus the clones' own
    loglikes and the agreement prior, up to the constant the stay rewards and pair rewards carry. Fails where a unary,
    an edge or the start is posed off the chain."""
    pairs = get_ab_lattice(params="mp", fold=True, **CAPS)
    chain = chain_of(pairs, [7, 5], 0.3)
    rng = np.random.default_rng(2)
    loglikes = {c: rng.normal(0.0, 1.0, (pairs.shape[0], 12)) for c in (0, 1, 2)}
    problem = get_clone_chains_potts_problem(chain, loglikes, [0, 1, 2])
    constants = []
    for _ in range(3):
        paths = {c: rng.integers(0, pairs.shape[0], 12) for c in loglikes}
        loglike = sum(get_copy_path_loglike(loglikes[c], chain, paths[c]) for c in loglikes)
        loglike += get_all_clone_agreement_prior(paths, chain.clone_coupling)
        constants.append(pairwise_potts_energy(problem, np.concatenate([paths[c] for c in (0, 1, 2)])) + loglike)
    steps, pair_bins = 3 * (12 - 2), 3 * 12
    np.testing.assert_allclose(constants, steps * chain.log_switch - 0.3 * pair_bins, rtol=1e-12)


PLANTED_THETA = (1.0, 0.2, 0.0)
"""The planted clones' normal fractions: the normal clone, then two tumour clones."""


def planted(P: int, seed: int = 0, n_obs: int = 60, alpha: float = 0.01, tau: float = 1.0e3) -> dict[str, Any]:
    """Counts drawn from `eqn:nb_emission` and `eqn:bb_emission` over a planted genome at ploidy P, 2 contigs of 30.

    Each tumour clone sits at the neutral pair `(k, k)`, `k = P / 2`, with a shared gain `(k + 1, k)` on bins 5-14, a
    copy-neutral LOH `(P, 0)` on bins 35-44, a mirrored gain on bins 48-55, `(k + 1, k)` in clone 1 and `(k, k + 1)` in
    clone 2, and clone 2's private loss `(k, k - 1)` on bins 20-27; read depth over the clone's own `Z_c`.

    Returns: the `PseudobulkCounts`, the planted `(n_obs, 3, 2)` copies, and the dispersions.
    """
    k = P // 2
    truth = np.tile(np.array([k, k]), (n_obs, 3, 1))
    truth[:, 0] = (1, 1)
    truth[5:15, 1:] = (k + 1, k)
    truth[20:28, 2] = (k, k - 1)
    truth[35:45, 1:] = (P, 0)
    truth[48:56, 1], truth[48:56, 2] = (k + 1, k), (k, k + 1)
    rng = np.random.default_rng(seed)
    base = rng.uniform(600.0, 900.0, (n_obs, 3))
    total = rng.integers(600, 900, (n_obs, 3)).astype(float)
    y, b = np.empty((n_obs, 3)), np.empty((n_obs, 3))
    for c, theta in enumerate(PLANTED_THETA):
        depth, fraction = get_admixed_moments(get_lattice_parameters(truth[:, c], None, None), theta)
        mean = base[:, c] * depth / (base[:, c] @ depth / base[:, c].sum())
        y[:, c] = rng.negative_binomial(1.0 / alpha, 1.0 / (1.0 + alpha * mean))
        p = rng.beta(tau * np.clip(fraction, 1e-9, 1.0), tau * np.clip(1.0 - fraction, 1e-9, 1.0))
        b[:, c] = rng.binomial(total[:, c].astype(int), p)
    channels = (np.ascontiguousarray(v.T) for v in (y, base, b, total))
    return {"counts": PseudobulkCounts(*channels, None, {}), "truth": truth, "alpha": alpha, "tau": tau}


def get_parameters(given: dict[str, Any]) -> CnaHmmParameters:
    """The planted dispersions as the fit's."""
    return CnaHmmParameters(None, None, None, np.array([given["alpha"]]), np.array([given["tau"]]))


def breakpoints(copies: np.ndarray) -> list[int]:
    """The bins after which `copies` changes pair."""
    return np.flatnonzero((np.diff(copies, axis=0) != 0).any(axis=1)).tolist()


@pytest.mark.parametrize("baf_only", [False, True])
def test_fit_normal_fraction_is_the_dense_grid_s_best_and_holds_its_edges(baf_only: bool) -> None:
    """Referee: a 0.001 grid over [0, 1) of the loglike on the path's `A != B` bins at the held `log Z_c`, as on main,
    or of their B-allele fraction alone under `baf_only`.
    On the planted 0.2 clone's path, held at that path's `log Z_c` at 0.2, the fit scores at least the grid's best less
    1e-6 nats, within 0.001 of it and 0.02 of 0.2; on the pure clone, 0; on an all-`A == B` path, 0. Fails where Brent
    stops short, misses the bound, or a flat score moves it."""
    given = planted(2)
    pairs = get_ab_lattice(params="mp", fold=False, max_total_copy=6, max_allele_copy=5)
    index = {tuple(pair): k for k, pair in enumerate(pairs.tolist())}
    parameters = get_lattice_parameters(pairs, given["alpha"], given["tau"])
    imbalanced = pairs[:, 0] != pairs[:, 1]
    counts = given["counts"]

    def path_of(c: int) -> np.ndarray:
        return np.array([index[tuple(pair)] for pair in given["truth"][:, c].tolist()])

    path = path_of(1)
    log_z = get_path_log_z(pairs, path, 0.2, counts.normal_baseline_sized, 1)
    bins = imbalanced[path]
    kept = get_clone_rows(counts, np.s_[1:2, bins])
    if baf_only:
        kept = kept._replace(umis=None, normal_baseline_sized=None)

    def score(f: float) -> float:
        return get_copy_path_emission_loglike(
            kept,
            parameters,
            CnaEmissionConfounders(np.full((1, 1), f), np.full((1, 1), log_z)),
            path[bins],
            fold=False,
            data_terms=False,
        )

    theta = fit_normal_fraction(counts, 1, parameters, path, imbalanced, log_z, fold=False, baf_only=baf_only)
    grid = np.arange(0.0, 1.0, 0.001)
    on_grid = np.array([score(f) for f in grid])
    assert score(theta) >= on_grid.max() - 1e-6
    assert abs(theta - grid[on_grid.argmax()]) <= 0.001 and abs(theta - 0.2) <= 0.02
    assert fit_normal_fraction(counts, 2, parameters, path_of(2), imbalanced, 0.0, fold=False, baf_only=baf_only) == 0.0
    flat = np.full(path.size, index[(1, 1)])
    assert fit_normal_fraction(counts, 1, parameters, flat, imbalanced, log_z, fold=False, baf_only=baf_only) == 0.0


def test_the_dispersion_objective_gradient_is_central_differences() -> None:
    """Referee: central differences at h = 1e-4 in `(log alpha, log tau)`, to 1e-6 relative, on the planted clones'
    paths; the value is `get_copy_path_emission_loglike` without data terms summed over clones, to 1e-12. Fails where
    the derivatives are read at the wrong mean or B count, or a bin's term is lost."""
    given = planted(2)
    pairs = get_ab_lattice(params="mp", fold=False, max_total_copy=6, max_allele_copy=5)
    index = {tuple(pair): k for k, pair in enumerate(pairs.tolist())}
    counts = given["counts"]
    paths = {c: np.array([index[tuple(pair)] for pair in given["truth"][:, c].tolist()]) for c in (1, 2)}
    confounders = CnaEmissionConfounders(np.array([[1.0], [0.2], [0.0]]), np.array([[0.0], [0.05], [0.02]]))
    x = np.log([0.02, 500.0])
    value, grad = hmm_ab_decoder_dispersion_objective(x, counts, (1, 2), pairs, paths, confounders, fold=False)
    parameters = get_lattice_parameters(pairs, *np.exp(x))
    expected = sum(
        get_copy_path_emission_loglike(
            get_clone_rows(counts, slice(c, c + 1)),
            parameters,
            CnaEmissionConfounders(*(v[c : c + 1] for v in confounders)),
            paths[c],
            fold=False,
            data_terms=False,
        )
        for c in (1, 2)
    )
    np.testing.assert_allclose(value, -expected, rtol=1e-12)
    h = 1e-4
    for i in range(2):
        step = np.eye(2)[i] * h
        up, down = (
            hmm_ab_decoder_dispersion_objective(x + s, counts, (1, 2), pairs, paths, confounders, fold=False)[0]
            for s in (step, -step)
        )
        np.testing.assert_allclose(grad[i], (up - down) / (2 * h), rtol=1e-6)


def test_each_round_decodes_at_the_previous_path_s_log_z() -> None:
    """On planted P = 2 each decode after the first scores a clone at the `log Z_c` of its previous path at the
    `theta_c` it decodes at; the returned `log Z_c` is the returned path's. Fails where `Z_c` lags its path. The joint
    loglike is not asserted to rise: the `theta` step, on the `A != B` bins at the held `log Z_c` as on main, is not an
    ascent of it."""
    given = planted(2)
    seen: list[dict[int, tuple[float, float]]] = []
    decoded: list[tuple[dict[int, np.ndarray], float]] = []
    real_emission, real_decode = hd.get_clone_emission_loglikes, hd.solve_clone_chains

    def recorded_emission(counts: Any, parameters: Any, confounders: Any, *args: Any, **kwargs: Any) -> Any:
        rows = enumerate(kwargs["clones"])
        seen.append({c: (confounders.normal_fraction[i, 0], confounders.log_mu_shift[i, 0]) for i, c in rows})
        return real_emission(counts, parameters, confounders, *args, **kwargs)

    def recorded_decode(*args: Any, **kwargs: Any) -> Any:
        out = real_decode(*args, **kwargs)
        decoded.append((dict(out[0]), out[3]))
        return out

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(hd, "get_clone_emission_loglikes", recorded_emission)
        patch.setattr(hd, "solve_clone_chains", recorded_decode)
        found = decode_at_ploidy(
            given["counts"],
            get_parameters(given),
            CnaEmissionConfounders(np.array([[1.0], [0.1], [0.1]]), np.zeros((3, 1))),
            np.array([30, 30]),
            config_of({}),
            ploidy=2,
        )
    pairs = get_ab_lattice(params="mp", fold=False, max_total_copy=6, max_allele_copy=5)
    base = given["counts"].normal_baseline_sized
    assert len(decoded) >= 2
    for (paths, _), used in zip(decoded[:-1], seen[1:], strict=True):
        for c, (f, log_z) in used.items():
            assert log_z == get_path_log_z(pairs, paths[c], f, base, c)
    index = {tuple(pair): k for k, pair in enumerate(pairs.tolist())}
    for c in (1, 2):
        path = np.array([index[tuple(pair)] for pair in found.copies[:, c].tolist()])
        own = get_path_log_z(pairs, path, found.normal_fractions[c], base, c)
        assert np.isclose(found.log_mu_shift[c], own, rtol=0.0, atol=1e-9)


@pytest.mark.parametrize("assumed", ["triploid", "diploid,triploid"])
def test_odd_ploidy_is_not_implemented(assumed: str) -> None:
    """Odd ploidy has no neutral pair: both entry points raise `NotImplementedError`."""
    given = planted(2)
    args = (
        given["counts"],
        get_parameters(given),
        get_unconfounded(3),
        np.array([30, 30]),
        config_of({"assumed_ploidy": assumed}),
    )
    with pytest.raises(NotImplementedError, match="odd ploidy"):
        decode_integer_copies(*args)
    with pytest.raises(NotImplementedError, match="odd ploidy"):
        decode_at_ploidy(*args, ploidy=3)


@pytest.mark.parametrize(
    "refused",
    [
        {"ploidy": 0},
        {"min_alpha": -1.0},
        {"max_tau": 0.5},
        {"clone_coupling": -0.1},
        {"solver": "viterbi"},
        {"contig_start": "free"},
        {"assumed_ploidy": "pentaploid"},
    ],
)
def test_out_of_range_settings_are_refused(refused: dict[str, Any]) -> None:
    """Each setting out of its range raises `ValueError` rather than decoding."""
    given = planted(2)
    refused = dict(refused)
    ploidy = refused.pop("ploidy", 2)
    args = (given["counts"], get_parameters(given), get_unconfounded(3), np.array([30, 30]), config_of(refused))
    with pytest.raises(ValueError):
        if "assumed_ploidy" in refused:
            decode_integer_copies(*args)
        else:
            decode_at_ploidy(*args, ploidy=ploidy)


def tied(*args: Any, ploidy: int) -> AbDecodeResult:
    """A decode at `ploidy` scoring clone 1 -5 whatever the ploidy, its copies `(ploidy, 0)`: a tie."""
    return AbDecodeResult(
        copies=np.full((3, 2, 2), ploidy),
        normal_fractions=np.array([1.0, 0.1 * ploidy]),
        log_mu_shift=np.zeros(2),
        stay_prob=0.99,
        score=-5.0,
        clone_scores={1: -5.0},
        ploidy={1: str(ploidy)},
        status={1: DecodeStatus.OK},
    )


def test_a_ploidy_tie_takes_the_lower_ploidy() -> None:
    """Diploid and tetraploid score clone 1 the same: the diploid decode's copies and `theta_c` are returned."""
    given = planted(2)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(hd, "decode_at_ploidy", tied)
        found = decode_integer_copies(
            given["counts"],
            get_parameters(given),
            get_unconfounded(3),
            np.array([30, 30]),
            config_of({"assumed_ploidy": "diploid,tetraploid"}),
        )
    assert found.ploidy == {1: "diploid"} and (found.copies[:, 1] == 2).all() and found.normal_fractions[1] == 0.2


DECODES: dict[str, dict[str, Any]] = {
    "mp icm": {},
    "mp folded icm": {"fold": True, **CAPS},
    "mp expansion": {"solver": "expansion"},
    "p folded icm": {"rdr": False, "fold": True},
    "p icm, Z held, dispersions held": {
        "rdr": False,
        **CAPS,
        "recompute_logmu_shift": False,
        "fit_dispersions": False,
    },
    "pure clones": {"fit_normal": False},
    "a bin over max_rdr": {"max_rdr": True},
    "normal clone by B fraction": {"normal_clone": None},
    "the fit's log Z carried": {"shifted": True},
    "diploid and tetraploid": {"assumed_ploidy": "diploid,tetraploid"},
    "tetraploid": {"ploidy": 4},
}
"""The covering set of the decoder's settings: each channel and fold, each solver, the `log Z_c` and dispersion
refits off, pure clones, a dropped read-depth bin, the normal clone found by its B fraction, and the ploidy search."""


@pytest.mark.parametrize(
    "label",
    [
        pytest.param(
            label,
            marks=pytest.mark.xfail(
                strict=True, reason="the one start, theta 0.10, stops clone 1 at a local optimum at P = 4"
            ),
        )
        if label == "tetraploid"
        else label
        for label in DECODES
    ],
)
def test_planted_decode_recovers_every_planted_feature(label: str) -> None:
    """Referee: the planted truth. Every copy is on the lattice and equals the planted pair (folded where `fold`;
    its B fraction alone where no read depth), with the planted breakpoints; each tumour `theta_c` is within 0.02 of
    0.2 and 0 (0 held for pure clones); `log Z_c` is `get_log_mu_shift` on the decoded path where recomputed, the
    start where held; every status `OK`; the normal clone (1, 1) at 1; each contig starts at the neutral pair; the
    same call twice is the same decode. Fails where a channel, the fold, a solver, the admixture or `Z_c` drifts."""
    settings = dict(DECODES[label])
    P = settings.pop("ploidy", 2)
    normal_clone = settings.pop("normal_clone", 0)
    shift = np.array([0.0, 0.05, -0.05]) if settings.pop("shifted", False) else None
    given = planted(P)
    counts = given["counts"]
    if settings.pop("max_rdr", False):
        counts = counts._replace(umis=counts.umis.copy())
        counts.umis[1, 3] = 40.0 * counts.normal_baseline_sized[1, 3]
    starts = None if normal_clone is None else np.array([1.0, 0.1, 0.1])
    caps = {"max_total_copy": 6, "max_allele_copy": 5}
    config = config_of({**caps, **settings})
    args = (
        counts,
        get_parameters(given),
        CnaEmissionConfounders(
            np.zeros((3, 1)) if starts is None else starts[:, None],
            np.zeros((3, 1)) if shift is None else shift[:, None],
        ),
        np.array([30, 30]),
        config,
    )
    found = decode_integer_copies(*args) if "assumed_ploidy" in settings else decode_at_ploidy(*args, ploidy=P)
    fold, has_rdr = settings.get("fold", False), settings.get("rdr", True)
    pairs = get_ab_lattice(
        params="mp" if has_rdr else "p",
        fold=fold,
        max_total_copy=settings.get("max_total_copy", 6),
        max_allele_copy=settings.get("max_allele_copy", 5),
    )
    truth = np.sort(given["truth"], axis=2)[..., ::-1] if fold else given["truth"]
    on_lattice = (found.copies[:, 1:, None, :] == pairs[None, None, :, :]).all(axis=3).any(axis=2)
    assert on_lattice.all()
    assert (found.copies[:, 0] == 1).all() and found.normal_fractions[0] == 1.0
    # NB pure clones hold clone 1's theta 0.2 at 0: its pairs are not the planted ones.
    for c in (1, 2) if settings.get("fit_normal", True) else (2,):
        if has_rdr:
            assert np.array_equal(found.copies[:, c], truth[:, c]), (label, c)
        else:
            fraction = found.copies[:, c, 1] / found.copies[:, c].sum(axis=1)
            assert np.allclose(fraction, truth[:, c, 1] / truth[:, c].sum(axis=1)), (label, c)
        assert breakpoints(found.copies[:, c]) == breakpoints(truth[:, c]), (label, c)
        assert get_ab_is_ploidy_neutral(found.copies[[0, 30], c], P).all() if has_rdr else True
    if not settings.get("fit_normal", True):
        assert found.normal_fractions[1:].tolist() == [0.0, 0.0]
    elif has_rdr:
        np.testing.assert_allclose(found.normal_fractions[1:], PLANTED_THETA[1:], atol=0.02)
    lattice_pairs = get_ab_lattice(params="mp", fold=fold, **caps)
    index = {tuple(pair): k for k, pair in enumerate(lattice_pairs.tolist())}
    base = (
        np.where(counts.umis > 10.0 * counts.normal_baseline_sized, 0.0, counts.normal_baseline_sized)
        if has_rdr
        else None
    )
    for c in (1, 2):
        if settings.get("recompute_logmu_shift", True) and base is not None:
            path = np.array([index[tuple(pair)] for pair in found.copies[:, c].tolist()])
            own = get_path_log_z(lattice_pairs, path, found.normal_fractions[c], base, c)
            assert np.isclose(found.log_mu_shift[c], own, rtol=0.0, atol=1e-9), (label, c)
        else:
            held = get_log_mu_shift_start(None if shift is None else shift[c], 0.1, P)
            assert found.log_mu_shift[c] == held, (label, c)
    assert set(found.status.values()) == {DecodeStatus.OK}
    if label == "mp icm":
        again = decode_at_ploidy(*args, ploidy=P)
        assert np.array_equal(found.copies, again.copies) and found.score == again.score
        assert np.array_equal(found.log_mu_shift, again.log_mu_shift)


def test_a_lone_normal_clone_has_nothing_to_decode() -> None:
    """One clone held normal leaves no tumour clone: (1, 1) at theta 1, loglike 0."""
    given = planted(2)
    counts = PseudobulkCounts(*(v[:1] for v in given["counts"][:4]), None, {})
    found = decode_at_ploidy(
        counts,
        get_parameters(given),
        CnaEmissionConfounders(np.ones((1, 1)), np.zeros((1, 1))),
        np.array([30, 30]),
        config_of({}),
        ploidy=2,
    )
    assert (found.copies == 1).all() and found.normal_fractions.tolist() == [1.0] and found.score == 0.0


def test_a_chain_restarts_at_each_contig() -> None:
    """A flat score over contigs of 5, 1 and 4 bins decodes every bin to the neutral pair at P = 2 and 4, the one-bin
    contig included."""
    pairs = get_ab_lattice(params="mp", fold=False, max_total_copy=6, max_allele_copy=5)
    for P in (2, 4):
        chain = chain_of(pairs, [5, 1, 4], 0.0)._replace(log_start=get_contig_start_prior(pairs, 0.99, P, "anchored"))
        path, _ = clone_viterbi(chain, 1, np.zeros((pairs.shape[0], 10)), None, ploidy_price_steps=PRICE_STEPS)
        assert get_ab_is_ploidy_neutral(pairs, P)[path].all()


def test_one_stacked_column_is_decoded_whole_and_settles_its_log_z() -> None:
    """A start's clones stacked as one column hold no normal clone: every bin is decoded, after one round its
    `log Z_c` settled on the decoded path's own. Fails where the stacked column is held normal or `Z_c` lags."""
    given = planted(2)
    counts = PseudobulkCounts(*(v[1:2] for v in given["counts"][:4]), None, {})
    found = decode_integer_copies(
        counts, get_parameters(given), get_unconfounded(1), np.array([30, 30]), config_of({"max_iter": 1})
    )
    pairs = get_ab_lattice(params="mp", fold=False, max_total_copy=6, max_allele_copy=5)
    index = {tuple(pair): k for k, pair in enumerate(pairs.tolist())}
    path = np.array([index[tuple(pair)] for pair in found.copies[:, 0].tolist()])
    own = get_path_log_z(pairs, path, found.normal_fractions[0], counts.normal_baseline_sized, 0)
    assert found.normal_fractions[0] < 1.0 and np.isclose(found.log_mu_shift[0], own, rtol=0.0, atol=1e-9)


def test_mirrored_b_counts_decode_to_mirrored_pairs() -> None:
    """Referee: the lattice's reflection symmetry. Unfolded, `n - b` for `b` decodes every pair `(A, B)` to `(B, A)` at
    the same normal fractions and loglike, to 1e-9; folded, the two are one decode. Fails where the lattice, prior or
    contig start favours one phase."""
    given = planted(2)
    counts = given["counts"]
    mirrored = counts._replace(alt_snp_umis=counts.total_snp_umis - counts.alt_snp_umis)
    for fold in (False, True):
        first, second = (
            decode_at_ploidy(
                c,
                get_parameters(given),
                CnaEmissionConfounders(np.array([[1.0], [0.1], [0.1]]), np.zeros((3, 1))),
                np.array([30, 30]),
                config_of({"fold": fold}),
                ploidy=2,
            )
            for c in (counts, mirrored)
        )
        assert np.array_equal(first.copies, second.copies if fold else second.copies[..., ::-1]), fold
        np.testing.assert_allclose(first.normal_fractions, second.normal_fractions, rtol=0.0, atol=1e-9)
        assert np.isclose(first.score, second.score, rtol=1e-9, atol=0.0)
