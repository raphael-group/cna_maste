"""`hmrf.opengm_patch` (T- #195): `expansion` with OpenGM's moves (false) or with the auxiliary node patched (true,
the default), so that a pair labelled apart, neither end alpha, counts in the move.

Referees: enumeration of every labelling (the two-node minimum, local minima under every alpha move and the 2x bound
on planted 3 x 3 and 2 x 4 problems); OpenGM's TRW-S, its bound certifying a minimum, on 100 planted 10 x 10 problems
where the two move sets part; the planted truth on 60 x 60 lattices. OpenGM's `opengm_expansion` is not changed.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from cnamaste import expansion, external
from cnamaste.config import YAMLConfig
from cnamaste.config.yaml_config import LabellingOptions, config_defaults, get_labelling_options
from cnamaste.opengm import solve_labelling
from cnamaste.potts import PottsProblem, pairwise_potts_energy
from potts_lattice import get_planted_lattice

OPENGM = LabellingOptions(False, False, "index", False)
PATCHED = LabellingOptions(False, False, "index", True)


def solve(
    problem: PottsProblem, solver: str, start: np.ndarray, options: LabellingOptions = OPENGM
) -> external.OpenGMResult:
    return solve_labelling(problem, solver, start, 1000, 1e-9, "boykov_kolmogorov", options)


def get_enumerated_energies(problem: PottsProblem) -> tuple[np.ndarray, np.ndarray]:
    """Every labelling of a small problem and its Potts energy, vectorized."""
    n, k = problem.unary.shape
    labellings = np.array(list(itertools.product(range(k), repeat=n)))
    energies = problem.unary[np.arange(n), labellings].sum(axis=1)
    energies -= ((labellings[:, problem.first] == labellings[:, problem.second]) * problem.coupling).sum(axis=1)
    return labellings, energies


def is_certified(run: external.OpenGMResult) -> bool:
    return abs(run.energy - run.bound) <= 1e-9 * abs(run.bound)


def test_the_flag_defaults_to_opengm_s_moves_and_passes_through() -> None:
    """`hmrf.label_solver` defaults to expansion and `hmrf.opengm_patch` to true; `solve_labelling` hands the flag to
    `expansion.solve`, bitwise, on 10 planted problems the two move sets part on. Fails where a default changes or the
    option does not reach the library."""
    assert config_defaults["hmrf"]["label_solver"] == "expansion"
    assert config_defaults["hmrf"]["opengm_patch"] is True
    assert get_labelling_options(YAMLConfig({})).opengm_patch is True
    assert get_labelling_options(YAMLConfig({"hmrf": {"opengm_patch": False}})).opengm_patch is False
    parted = 0
    for seed in range(10):
        problem = get_planted_lattice(10, 10, 6, 0.5, 1.0, seed=seed).problem
        start = np.argmin(problem.unary, axis=1)
        found = {}
        for options in (OPENGM, PATCHED):
            found[options] = solve(problem, "expansion", start, options)
            direct = expansion.solve(problem, start, 1000, opengm_patch=options.opengm_patch)
            np.testing.assert_array_equal(found[options].labels, direct.labels)
            assert found[options].energy == direct.energy
        parted += found[OPENGM].energy != found[PATCHED].energy
    assert parted > 0, "no problem tells the two move sets apart"


def test_a_pair_labelled_apart_off_alpha_counts_in_the_patched_move() -> None:
    """Two nodes at labels 0 and 1, coupling 1, label 2 costing each 0.4 more: the patched moves reach (2, 2), the
    enumerated minimum and TRW-S's, Potts energy -0.2; OpenGM's expansion, and `expansion` unpatched, stay at
    (0, 1), energy 0 (T- #195's reproduction). Fails where the auxiliary node's arcs are OpenGM's under the flag."""
    if not external.available():
        pytest.skip(str(external.unavailable()))
    problem = PottsProblem(np.array([[0.0, 9.0, 0.4], [9.0, 0.0, 0.4]]), np.array([0]), np.array([1]), np.array([1.0]))
    labellings, energies = get_enumerated_energies(problem)
    start = np.array([0, 1])
    patched = solve(problem, "expansion", start, PATCHED)
    np.testing.assert_array_equal(patched.labels, labellings[np.argmin(energies)])
    np.testing.assert_array_equal(patched.labels, solve(problem, "opengm_trws", start).labels)
    assert patched.energy == pytest.approx(-0.2, abs=1e-15)
    for solver in ("expansion", "opengm_expansion"):
        np.testing.assert_array_equal(solve(problem, solver, start).labels, start)


@pytest.mark.parametrize(("rows", "cols", "n_clones", "mislabelled"), [(3, 3, 3, 0.4), (2, 4, 4, 0.5)])
def test_the_patched_result_is_a_local_minimum_within_twice_the_minimum(
    rows: int, cols: int, n_clones: int, mislabelled: float
) -> None:
    """On 40 planted problems each, the patched moves end where no alpha move lowers the energy, by enumeration, and
    the metric energy is within twice the enumerated minimum's (Boykov, Veksler and Zabih 2001, the field shifted to be
    non-negative). Fails where a patched move is not the optimal alpha move."""
    for seed in range(40):
        problem = get_planted_lattice(rows, cols, n_clones, mislabelled, spatial_weight=1.0, seed=seed).problem
        problem = problem._replace(unary=problem.unary - problem.unary.min())
        labellings, energies = get_enumerated_energies(problem)
        found = solve(problem, "expansion", np.argmin(problem.unary, axis=1), PATCHED)
        energy = pairwise_potts_energy(problem, found.labels)
        for alpha in range(n_clones):
            move = np.all((labellings == found.labels) | (labellings == alpha), axis=1)
            assert energies[move].min() >= energy - 1e-12, f"seed {seed}: expanding {alpha} lowers the energy"
        total = problem.coupling.sum()
        assert energy + total <= 2.0 * (energies.min() + total) + 1e-12


def test_the_patched_moves_reach_trws_s_certified_minimum_more_often() -> None:
    """100 planted 10 x 10 problems, 6 clones, argmax error 0.5, coupling 1.0, from the field's argmax: on those whose
    TRW-S bound equals its energy (94), the patched moves end at that minimum in 71 and OpenGM's in 47 (measured);
    no labelling falls below a bound. Fails where the patched moves lose a minimum they reached, or a labelling
    beats a lower bound."""
    if not external.available():
        pytest.skip(str(external.unavailable()))
    certified, reached = 0, {"opengm": 0, "patched": 0}
    for seed in range(100):
        problem = get_planted_lattice(10, 10, 6, 0.5, 1.0, seed=seed).problem
        start = np.argmin(problem.unary, axis=1)
        trws = solve(problem, "opengm_trws", start)
        runs = {
            "opengm": solve(problem, "expansion", start),
            "patched": solve(problem, "expansion", start, PATCHED),
        }
        for name, run in runs.items():
            assert run.energy >= trws.bound - 1e-9 * abs(trws.bound), f"seed {seed}: {name} below the bound"
        if is_certified(trws):
            certified += 1
            for name, run in runs.items():
                reached[name] += abs(run.energy - trws.bound) <= 1e-9 * abs(trws.bound)
    assert certified == 94
    assert reached["patched"] >= 71 and reached["opengm"] <= 47


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_patched_moves_recover_the_planted_truth(seed: int) -> None:
    """60 x 60 planted lattices, 6 clones, argmax error 0.4, coupling 1.5: the patched labelling is TRW-S's, at or
    below the planted truth's energy, and labels at least 99% of spots as planted (measured 0.9914-0.9928). Fails where
    the patched moves stop short of what the truth and TRW-S reach."""
    if not external.available():
        pytest.skip(str(external.unavailable()))
    planted = get_planted_lattice(60, 60, 6, 0.4, 1.5, seed=seed)
    problem, start = planted.problem, np.argmin(planted.problem.unary, axis=1)
    patched = solve(problem, "expansion", start, PATCHED)
    trws = solve(problem, "opengm_trws", start)
    assert patched.energy <= pairwise_potts_energy(problem, planted.truth)
    assert patched.energy == pytest.approx(trws.energy, rel=1e-12)
    assert np.mean(patched.labels == planted.truth) >= 0.99
