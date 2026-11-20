"""cnamaste's alpha expansion (`cnamaste.expansion`) and the reductions before a labelling (`potts.reduce_potts_problem`).

Referees: enumeration of every s-t cut and every labelling of small problems; OpenGM's expansion with Boost Graph's
Boykov-Kolmogorov cut on planted square lattices (`potts_lattice`), bitwise on a real field. On a field of a few values
a move's minimum cuts tie, and Boost's takes as sink a node whose source and sink capacities are equal, which can leave
it short of the minimum (a stated difference): there the two expansions are held to the same final energy.
"""

from __future__ import annotations

import ctypes
import itertools
import os

import numpy as np
import pytest

from cnamaste import expansion, external
from cnamaste.config.yaml_config import LabellingOptions
from cnamaste.opengm import solve_labelling
from cnamaste.potts import PottsProblem, pairwise_potts_energy, reduce_potts_problem
from potts_lattice import get_planted_lattice

DEFAULT = LabellingOptions(False, False, "index")


def get_enumerated_cut(
    n_nodes: int, tail: np.ndarray, head: np.ndarray, capacity: np.ndarray
) -> tuple[np.ndarray, float]:
    """The minimum s-t cut by enumeration, and the smallest sink side among the minimum cuts.

    Args:
        n_nodes: the nodes, node 0 the source and 1 the sink.
        tail: each directed edge's tail.
        head: each directed edge's head.
        capacity: each directed edge's capacity.

    Returns:
        The sink side per node and the minimum.
    """
    best, sides = np.inf, []
    for bits in itertools.product((0, 1), repeat=n_nodes - 2):
        side = np.array((0, 1) + bits)
        value = capacity[(side[tail] == 0) & (side[head] == 1)].sum()
        if value < best - 1e-12:
            best, sides = value, [side]
        elif value <= best + 1e-12:
            sides.append(side)
    return np.min(sides, axis=0), best


def get_boost_cut(n_nodes: int, tail: np.ndarray, head: np.ndarray, capacity: np.ndarray) -> tuple[np.ndarray, float]:
    """Boost Graph's Boykov-Kolmogorov cut through `cnamaste_opengm.cxx`'s `cnamaste_min_cut`, as OpenGM's moves use it.

    Args:
        n_nodes: the nodes, node 0 the source and 1 the sink.
        tail: each directed edge's tail.
        head: each directed edge's head.
        capacity: each directed edge's capacity.

    Returns:
        The sink side per node and the cut's capacity.
    """
    lib = ctypes.CDLL(str(external.library()))
    side, value, error = np.zeros(n_nodes, dtype=np.int64), ctypes.c_double(), ctypes.create_string_buffer(256)
    as_int = [np.ascontiguousarray(a, dtype=np.int64) for a in (tail, head)]
    capacity = np.ascontiguousarray(capacity, dtype=np.float64)
    status = lib.cnamaste_min_cut(
        ctypes.c_int(1),
        ctypes.c_int64(n_nodes),
        ctypes.c_int64(capacity.size),
        *(a.ctypes.data_as(ctypes.c_void_p) for a in as_int),
        capacity.ctypes.data_as(ctypes.c_void_p),
        side.ctypes.data_as(ctypes.c_void_p),
        ctypes.byref(value),
        error,
        ctypes.c_int64(256),
    )
    assert status == 0, error.value
    return side, value.value


def test_the_expansion_home_is_set() -> None:
    """`CNAMASTE_EXPANSION_HOME` holds the built library. Fails where `infra/expansion/build_expansion.sh` has not
    built into it, which a run with `hmrf.label_solver: expansion` refuses."""
    assert os.environ.get(expansion.get_home_variable()), f"set {expansion.get_home_variable()}: see OPENGM.md"
    assert expansion.available(), str(expansion.unavailable())


@pytest.mark.parametrize("integer", [False, True], ids=["real", "tied"])
def test_min_cut_is_the_enumerated_smallest_sink_side(integer: bool) -> None:
    """On 40 random graphs of 10 nodes, the cut's capacity is the enumerated minimum (to 1e-12) and its sink side the
    smallest among the minimum cuts, as Boost Graph's is; integer capacities tie many cuts. Fails where the max flow
    is short of the maximum or the cut leaves a tie on the sink's side."""
    rng = np.random.default_rng(1)
    for _ in range(40):
        n, m = 10, 30
        tail, head = rng.integers(0, n, m), rng.integers(0, n, m)
        keep = tail != head
        tail, head = tail[keep], head[keep]
        capacity = rng.integers(0, 4, tail.size).astype(float) if integer else rng.exponential(1.0, tail.size)
        expected, best = get_enumerated_cut(n, tail, head, capacity)
        side, value = expansion.min_cut(n, tail, head, capacity)
        assert abs(value - best) <= 1e-12 * max(1.0, best)
        np.testing.assert_array_equal(side, expected)
        if external.available():
            np.testing.assert_array_equal(side, get_boost_cut(n, tail, head, capacity)[0])


@pytest.mark.parametrize(
    ("shape", "mislabelled", "spatial_weight", "rounded"),
    [
        ((30, 40, 4), 0.3, 1.1, False),
        ((50, 60, 8), 0.3, 1.1, False),
        ((40, 40, 6), 0.02, 0.2, False),
        ((40, 40, 6), 0.3, 1.0, True),
    ],
    ids=["fixture-4", "8-clones", "reductions-fire", "integer-ties"],
)
@pytest.mark.parametrize(
    "options", [DEFAULT, LabellingOptions(True, True, "field_share")], ids=["default", "reduced-ordered"]
)
def test_expansion_is_opengm_expansion(
    shape: tuple[int, int, int], mislabelled: float, spatial_weight: float, rounded: bool, options: LabellingOptions
) -> None:
    """From the field's argmax, cnamaste's expansion returns OpenGM's expansion (Boykov-Kolmogorov) labels, energy and
    move count bitwise, by default and with every reduction and the field-share order; on a field rounded to halves,
    the same energy to 1e-9 relative. Fails where a capacity, the label order or the stop departs from OpenGM's."""
    if not external.available():
        pytest.skip(str(external.unavailable()))
    rows, cols, n_clones = shape
    problem = get_planted_lattice(rows, cols, n_clones, mislabelled, spatial_weight, seed=3).problem
    if rounded:
        problem = problem._replace(unary=np.round(2.0 * problem.unary) / 2.0)
    start = np.argmin(problem.unary, axis=1)
    ours = solve_labelling(problem, "expansion", start, 1000, 1e-9, "boykov_kolmogorov", options)
    theirs = solve_labelling(problem, "opengm_expansion", start, 1000, 1e-9, "boykov_kolmogorov", options)
    if rounded:
        assert abs(ours.energy - theirs.energy) <= 1e-9 * abs(theirs.energy)
        return
    np.testing.assert_array_equal(ours.labels, theirs.labels)
    assert ours.energy == theirs.energy and ours.iterations == theirs.iterations


def get_enumerated_energies(problem: PottsProblem) -> tuple[np.ndarray, np.ndarray]:
    """Every labelling of a small problem and its energy, vectorized.

    Args:
        problem: a small `PottsProblem`.

    Returns:
        `(labellings, energies)`.
    """
    n, k = problem.unary.shape
    labellings = np.array(list(itertools.product(range(k), repeat=n)))
    energies = problem.unary[np.arange(n), labellings].sum(axis=1)
    energies -= ((labellings[:, problem.first] == labellings[:, problem.second]) * problem.coupling).sum(axis=1)
    return labellings, energies


def test_expansion_is_a_local_minimum_within_twice_the_optimum() -> None:
    """On 20 planted 3 x 3 problems with 3 clones, no expansion move lowers the returned labelling's energy, and its
    metric energy is at most twice the enumerated minimum's (Boykov, Veksler and Zabih 2001, the field shifted to be
    non-negative). Fails where the stop is not convergence or a move is not the optimal one."""
    for seed in range(20):
        problem = get_planted_lattice(3, 3, 3, mislabelled=0.4, spatial_weight=1.0, seed=seed).problem
        problem = problem._replace(unary=problem.unary - problem.unary.min())
        labellings, energies = get_enumerated_energies(problem)
        found = expansion.solve(problem, start=np.argmin(problem.unary, axis=1))
        energy = pairwise_potts_energy(problem, found.labels)
        for alpha in range(3):
            # NB an alpha move keeps each spot's label or takes alpha.
            move = np.all((labellings == found.labels) | (labellings == alpha), axis=1)
            assert energies[move].min() >= energy - 1e-12, f"seed {seed}: expanding {alpha} lowers the energy"
        total = problem.coupling.sum()
        assert energy + total <= 2.0 * (energies.min() + total) + 1e-12


@pytest.mark.slow
def test_reductions_are_exact() -> None:
    """On 20 planted 3 x 3 problems with 3 clones and a fourth no spot prefers, each reduction alone and both: every
    fixed spot takes its label in the enumerated minimum, every dropped clone is absent from it, and the reduced energy
    differs from the full one by a constant across 50 random reduced labellings. Fails where a fixing or a pruning is
    not implied by the energy, or an edge is folded wrongly."""
    rng = np.random.default_rng(0)
    fixed_spots = pruned = 0
    for seed, (fix, prune) in itertools.product(range(20), [(True, False), (False, True), (True, True)]):
        planted = get_planted_lattice(3, 3, 3, mislabelled=0.2, spatial_weight=0.3, seed=seed)
        unary = np.c_[planted.problem.unary, planted.problem.unary.max(axis=1) + 5.0]
        problem = planted.problem._replace(unary=unary)
        labellings, energies = get_enumerated_energies(problem)
        best = labellings[np.argmin(energies)]
        reduced = reduce_potts_problem(problem, fix_dominant_spots=fix, prune_dominated_labels=prune)
        at = reduced.fixed >= 0
        np.testing.assert_array_equal(reduced.fixed[at], best[at])
        assert set(best[~at]) <= set(reduced.clones)
        fixed_spots, pruned = fixed_spots + at.sum(), pruned + 4 - reduced.clones.size
        if reduced.free.size:
            k = reduced.clones.size
            offsets = [
                pairwise_potts_energy(problem, reduced.full_labels(y)) - pairwise_potts_energy(reduced.problem, y)
                for y in rng.integers(0, k, (50, reduced.free.size))
            ]
            np.testing.assert_allclose(offsets, offsets[0], rtol=0, atol=1e-9)
    assert fixed_spots > 0 and pruned >= 20, "the regime fixes spots and drops the fourth clone"


def test_a_fully_fixed_problem_is_its_field_argmax() -> None:
    """With no coupling every spot is fixed at its field argmax and no solver runs. Fails where an empty reduced
    problem reaches a solver."""
    problem = PottsProblem(np.array([[0.0, 1.0], [2.0, 0.5]]), np.array([0]), np.array([1]), np.array([0.0]))
    found = solve_labelling(
        problem,
        "expansion",
        np.zeros(2, dtype=int),
        10,
        1e-9,
        "boykov_kolmogorov",
        LabellingOptions(True, False, "index"),
    )
    np.testing.assert_array_equal(found.labels, [0, 1])
    assert found.iterations == 0 and found.energy == 0.5
