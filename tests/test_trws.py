"""cnamaste's TRW-S (`hmrf.label_solver: trws`, `cnamaste.trws`, T- #198), a Rust port of OpenGM's `TRWSi`.

Referees: OpenGM's TRW-S (`opengm_trws`) through `solve_labelling`, bitwise in labels, energy, bound and bound trace,
on `test_expansion.py`'s planted lattices and the Visium HD 3,000 x 4 labelling, with and without the reductions;
enumeration (the two-node minimum, T- #195's case); TRW-S's bound certifying minima on 100 planted 10 x 10 problems,
and the planted truth on 60 x 60 lattices.
"""

from __future__ import annotations

import numpy as np
import pytest

from cnamaste import expansion, external, trws
from cnamaste.config.yaml_config import LabellingOptions
from cnamaste.opengm import solve_labelling
from cnamaste.potts import PottsProblem, pairwise_potts_energy
from potts_lattice import get_planted_lattice
from test_benchmarks import LABELLINGS, get_labelling

DEFAULT = LabellingOptions(False, False, "index")
REDUCED = LabellingOptions(True, True, "index")

# NB test_expansion.py's lattices: (rows, cols, clones, mislabelled, spatial_weight, rounded to halves).
LATTICES = {
    "fixture-4": (30, 40, 4, 0.3, 1.1, False),
    "8-clones": (50, 60, 8, 0.3, 1.1, False),
    "reductions-fire": (40, 40, 6, 0.02, 0.2, False),
    "integer-ties": (40, 40, 6, 0.3, 1.0, True),
}


def solve(problem: PottsProblem, solver: str, options: LabellingOptions = DEFAULT) -> external.OpenGMResult:
    start = np.zeros(problem.unary.shape[0], dtype=np.int64)
    return solve_labelling(problem, solver, start, 1000, 1e-9, "boykov_kolmogorov", options)


def get_problem(name: str) -> PottsProblem:
    if name in LATTICES:
        rows, cols, n_clones, mislabelled, spatial_weight, rounded = LATTICES[name]
        problem = get_planted_lattice(rows, cols, n_clones, mislabelled, spatial_weight, seed=3).problem
        return problem._replace(unary=np.round(2.0 * problem.unary) / 2.0) if rounded else problem
    return get_labelling(*LABELLINGS[name]).problem


def require_both() -> None:
    if not external.available():
        pytest.skip(str(external.unavailable()))
    if not expansion.available():
        pytest.skip(str(expansion.unavailable()))


@pytest.mark.parametrize("options", [DEFAULT, REDUCED], ids=["default", "reduced"])
@pytest.mark.parametrize("name", [*LATTICES, "hd16um-3000x4"])
def test_trws_is_opengm_s_trws(name: str, options: LabellingOptions) -> None:
    """Labels, Potts energy, bound, iterations and the bound trace are `opengm_trws`'s bitwise, with no reduction and
    with both. Fails where a chain, a message, the averaging, the rounding or a stop departs from OpenGM's."""
    require_both()
    problem = get_problem(name)
    ours, theirs = solve(problem, "trws", options), solve(problem, "opengm_trws", options)
    np.testing.assert_array_equal(ours.labels, theirs.labels)
    assert ours.energy == theirs.energy and ours.bound == theirs.bound and ours.iterations == theirs.iterations
    np.testing.assert_array_equal(ours.trace, theirs.trace)


def test_two_nodes_reach_the_enumerated_minimum() -> None:
    """Two nodes, unary rows (0, 9, 0.4) and (9, 0, 0.4), coupling 1: TRW-S labels (2, 2), Potts energy -0.2, the
    minimum by enumeration, its bound equal (T- #195's case, where both expansions stop at (0, 1))."""
    if not expansion.available():
        pytest.skip(str(expansion.unavailable()))
    problem = PottsProblem(np.array([[0.0, 9.0, 0.4], [9.0, 0.0, 0.4]]), np.array([0]), np.array([1]), np.array([1.0]))
    energies = {(a, b): pairwise_potts_energy(problem, np.array([a, b])) for a in range(3) for b in range(3)}
    found = trws.solve(problem)
    assert tuple(found.labels) == min(energies, key=energies.get) == (2, 2)
    assert found.energy == found.bound == pytest.approx(-0.2, abs=1e-15)


def test_trws_certifies_the_same_minima_as_opengm() -> None:
    """On `test_expansion_opengm_patch.py`'s 100 planted 10 x 10 problems, 6 clones, argmax error 0.5, coupling 1.0:
    TRW-S's energy and bound are OpenGM's bitwise on every one, and the bound meets the energy on 94 (measured), each
    a minimum. Fails where any problem's result departs from OpenGM's."""
    require_both()
    certified = 0
    for seed in range(100):
        problem = get_planted_lattice(10, 10, 6, 0.5, 1.0, seed=seed).problem
        ours, theirs = solve(problem, "trws"), solve(problem, "opengm_trws")
        assert ours.energy == theirs.energy and ours.bound == theirs.bound, f"seed {seed}"
        certified += abs(ours.energy - ours.bound) <= 1e-9 * abs(ours.bound)
    assert certified == 94


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_trws_recovers_the_planted_truth(seed: int) -> None:
    """60 x 60 planted lattices, 6 clones, argmax error 0.4, coupling 1.5: the labelling is at or below the planted
    truth's energy, its gap to the bound within 1e-9, and labels at least 99% of spots as planted (0.9914-0.9928 under
    OpenGM's TRW-S). Fails where TRW-S stops short of the truth's energy."""
    if not expansion.available():
        pytest.skip(str(expansion.unavailable()))
    planted = get_planted_lattice(60, 60, 6, 0.4, 1.5, seed=seed)
    found = trws.solve(planted.problem)
    assert found.energy <= pairwise_potts_energy(planted.problem, planted.truth)
    assert abs(found.energy - found.bound) <= 1e-9 * abs(found.bound)
    assert np.mean(found.labels == planted.truth) >= 0.99


def test_trws_needs_the_expansion_library(monkeypatch: pytest.MonkeyPatch) -> None:
    """`hmrf.label_solver: trws` stops a run where `$CNAMASTE_EXPANSION_HOME` holds no library, as `expansion` does.
    Fails where a run would start and fail at its first labelling."""
    monkeypatch.delenv(expansion.get_home_variable(), raising=False)
    with pytest.raises(RuntimeError, match="not built"):
        expansion.require("trws")
    expansion.require("opengm_trws")
