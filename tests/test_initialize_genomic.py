"""`get_lattice_states` at the three stages a run starts an HMM, and `get_initial_parameters`' routing to it.

A run starts its HMM three times, each through `get_initial_parameters` and the stage's `get_stage_channels`:
`phasing` (`params="sp"`, folded, unphased counts), `baf` (`"sp"`, unfolded, phased counts) and `rdr` (`"smp"`,
unfolded). `hmm.baum_welch` starts none: it refuses a missing start.

Referees: copy states planted per contig at high depth, so the decoded start is the planted states exactly; and
`ab_lattice.get_ab_lattice`, the states the decoder itself returns. It costs 2 s, numba cache warm.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import scipy.special

from cnamaste.ab_decode.lattice import get_ab_lattice
from cnamaste.config import YAMLConfig
from cnamaste.emission import get_unconfounded
from cnamaste.hmm.baum_welch import HmmTransitions, baum_welch
from cnamaste.hmm_params import CnaHmmParameters
from cnamaste.hmm_utils import get_log_transmat
from cnamaste.initialize.genomic import (
    get_default_parameters,
    get_initial_parameters,
    get_lattice_states,
    get_stage_channels,
)
from cnamaste.qa.sims import get_project
from cnamaste.spatio_genomic_counts import PseudobulkCounts
from labelled_counts import get_labelled_counts

BINS = 40
"""Bins per planted contig."""


def planted(pairs: list[tuple[int, int]], flip: bool = False, seed: int = 0) -> dict[str, Any]:
    """One clone stack, one contig of `BINS` bins per `(A, B)` of `pairs`, at base 300 and allele depth 200.

    Args: `pairs` the planted copy states; `flip` swap A and B on a random half of the bins, as unphased counts are;
        `seed` the generator's.
    Returns: the counts' arrays and `lengths`.
    """
    rng = np.random.default_rng(seed)
    copies = np.repeat(np.array(pairs, dtype=float), BINS, axis=0)
    total_copies = copies.sum(axis=1)
    b_fraction = copies[:, 1] / total_copies
    if flip:
        b_fraction = np.where(rng.random(len(b_fraction)) < 0.5, 1.0 - b_fraction, b_fraction)
    base = np.full((len(copies), 1), 300.0)
    depth = np.full((len(copies), 1), 200.0)
    reads = rng.poisson(base[:, 0] * total_copies / 2.0)
    b_counts = rng.binomial(depth[:, 0].astype(int), b_fraction)
    return {
        "X": np.stack([reads, b_counts], axis=1)[:, :, None].astype(float),
        "base_nb_mean": base,
        "total_bb_RD": depth,
        "lengths": np.full(len(pairs), BINS),
    }


def get_config(n_states: int, initializer: str = "lattice") -> YAMLConfig:
    """The shipped CalicoST configuration at `n_states` and `initializer`."""
    config = YAMLConfig.from_file(get_project() / "zenodo_sim_config.yaml")
    config.over_ride([f"hmm.n_states={n_states}", f"hmm.initializer={initializer}"])
    return config


def start(case: dict[str, Any], n_states: int, params: str, only_minor: bool) -> tuple[Any, Any]:
    """`get_lattice_states` on `case`: its `(log_mu, p_binom)`."""
    X = case["X"]
    counts = PseudobulkCounts(X[:, 0, :].T, case["base_nb_mean"].T, X[:, 1, :].T, case["total_bb_RD"].T, None, {})
    return get_lattice_states(counts, case["lengths"], get_config(n_states), params, only_minor)


def test_phasing_folds_unphased_counts_to_the_minor_allele() -> None:
    """The `phasing` start (`"sp"`, folded) on unphased counts: every `p_binom` at most 1/2, and the planted minor
    fractions 0, 1/3 and 1/2 among them exactly; no read-depth start. Fails where the fold is lost and a flipped
    bin starts a state above 1/2."""
    assert get_stage_channels("phasing") == ("sp", True)
    log_mu, p_binom = start(planted([(1, 1), (2, 0), (2, 1)], flip=True), 3, "sp", only_minor=True)
    assert log_mu is None
    assert np.all(p_binom <= 0.5)
    assert {0.0, 1 / 3, 0.5} <= set(p_binom.ravel().tolist())


def test_core_inference_keeps_a_mirrored_pair_as_two_states() -> None:
    """The `baf` start (`"sp"`, unfolded) on phased counts: `(2, 0)` and its mirror `(0, 2)` start two
    states, p 0 and 1. Folded, as phasing is, the same counts start one, p 0. Fails where `baf` folds."""
    assert get_stage_channels("baf") == ("sp", False)
    case = planted([(2, 0), (0, 2), (1, 1)])
    _, unfolded = start(case, 3, "sp", only_minor=False)
    assert {0.0, 1.0, 0.5} == set(unfolded.ravel().tolist())
    _, folded = start(case, 3, "sp", only_minor=True)
    assert 1.0 not in folded.ravel() and 0.0 in folded.ravel()


def test_the_rdr_stage_starts_each_planted_depth_and_b_fraction() -> None:
    """The `rdr` start (`"smp"`, unfolded): `log_mu = log((A + B) / 2)` and `p_binom = B / (A + B)` of each
    planted pair, `(1, 0)` and `(2, 0)` two states at one p. Fails where the depth channel merges them."""
    pairs = [(1, 0), (2, 0), (2, 1)]
    log_mu, p_binom = start(planted(pairs), 3, "smp", only_minor=False)
    found = set(zip(np.round(log_mu.ravel(), 12).tolist(), p_binom.ravel().tolist()))
    assert found == {(round(float(np.log((a + b) / 2.0)), 12), b / (a + b)) for a, b in pairs}


@pytest.mark.parametrize(("params", "only_minor"), [("sp", True), ("sp", False), ("smp", True), ("smp", False)])
def test_every_start_is_a_lattice_state_padded_to_n_states(params: str, only_minor: bool) -> None:
    """Six states asked of three planted: six rows, none repeated, each a pair of `get_ab_lattice(params, fold)` --
    a state `ab_decode.decode` can return. Fails where padding repeats a state or leaves the lattice."""
    log_mu, p_binom = start(planted([(1, 1), (2, 0), (2, 1)]), 6, params, only_minor=only_minor)
    pairs = get_ab_lattice(params=params, fold=only_minor, max_allele_copy=5, max_total_copy=6)
    allowed = {(round(float(np.log(a + b) - np.log(2.0)), 12), b / (a + b)) for a, b in pairs}
    rows = [
        (round(float(mu), 12) if log_mu is not None else None, float(p))
        for mu, p in zip(log_mu.ravel() if log_mu is not None else [None] * 6, p_binom.ravel())
    ]
    assert len(rows) == 6 and len(set(rows)) == 6
    if log_mu is None:
        assert {p for _, p in rows} <= {p for _, p in allowed}
    else:
        assert set(rows) <= allowed


def test_the_start_is_the_configured_one_and_unknown_names_are_refused() -> None:
    """`get_initial_parameters` on two planted clones: `lattice` starts the planted p, `calicost` another; both keep
    the defaults' `log_mu` at `baf`; an unknown `hmm.initializer` or stage raises `ValueError`. Fails where a stage or a
    name silently takes another start."""
    case = planted([(1, 1), (2, 0), (2, 1)])
    X = np.concatenate([case["X"]] * 2, axis=2)
    counts = get_labelled_counts(
        X[:, 0, :], X[:, 1, :], np.concatenate([case["total_bb_RD"]] * 2, axis=1), case["lengths"]
    )
    labelling = np.array([0, 1])
    lattice = get_initial_parameters(counts, labelling, get_config(3), stage="baf")
    calicost = get_initial_parameters(counts, labelling, get_config(3, "calicost"), stage="baf")
    assert {0.0, 0.5} <= set(lattice.p_binom.tolist()) and not np.array_equal(lattice.p_binom, calicost.p_binom)
    for found in (lattice, calicost):
        assert np.array_equal(found.log_mu, get_default_parameters(3).log_mu)
    config = get_config(3)
    config.hmm.initializer = "gmm"
    with pytest.raises(ValueError, match="hmm.initializer 'gmm'"):
        get_initial_parameters(counts, labelling, config, stage="baf")
    with pytest.raises(ValueError, match="'decoder' is not one of"):
        get_stage_channels("decoder")


def test_the_default_start_is_spread_and_uniform() -> None:
    """`get_default_parameters`: `log_mu` from -0.1 to 0.1 and `p` from 0.05 to 0.45 per state, a uniform start,
    alpha 0.5 and tau 1,000; the fit's transitions stay with `t`, each row summing to 1."""
    log_startprob, log_mu, p_binom, alphas, taus = get_default_parameters(3)
    assert np.allclose(log_mu, [-0.1, 0.0, 0.1]) and np.allclose(p_binom, [0.05, 0.25, 0.45])
    assert np.array_equal(alphas, np.full(3, 0.5)) and np.array_equal(taus, np.full(3, 1_000.0))
    assert np.allclose(np.exp(log_startprob), 1 / 3)
    log_transmat = get_log_transmat(3, 0.99)
    np.testing.assert_allclose(scipy.special.logsumexp(log_transmat, axis=1), 0.0, atol=1e-12)


def test_baum_welch_starts_no_hmm_itself() -> None:
    """A channel `params` fits with no start: `ValueError` naming it, before any fit. Fails where the deprecated
    fallback starts the HMM behind the stage's back."""
    counts = PseudobulkCounts(np.zeros((1, 4)), np.ones((1, 4)), np.zeros((1, 4)), np.ones((1, 4)), None, {})
    with pytest.raises(ValueError, match="without log_mu, p_binom"):
        baum_welch(
            counts,
            CnaHmmParameters(None, None, None, None, None),
            get_unconfounded(1),
            HmmTransitions(get_log_transmat(3, 0.99), None, np.array([4])),
            params="smp",
            tol=1e-4,
            previous_log_gamma=None,
            config=None,
        )
