"""Every `hmrf.label_solver` on a square lattice with planted clones, every branch: 1,024 spots (32 x 32) in the
standard tier, and the recovery on 16,384 (128 x 128) as a `benchmark`.

Input, seeded: the run's 8-nearest adjacency; four planted quadrants, quadrant 3 split in two clones of one genotype
(labels 3 and 5), and a small clone (label 4) under the floor: 4 x 4 under 25 at 32, 10 x 10 under 400 at 128; field
`-1 [genotype differs] + N(0, sigma^2)`, sigma 0.7 at 32 and 1.5 at 128, `spatial_weight` 1. In this regime the solvers
differ: expansion ends 1.8 points of ARI below TRW-S at 32. The referee is the planted genotype (ARI), the energy
(`potts.potts_energy`) and the single-spot move.

Cases: `icm` (`icm_sweep_deque`), and OpenGM's `icm`, `expansion` (Boykov-Kolmogorov and Dinic) and `trws` through
`opengm.external_assignment`; each with the floor off and on, OpenGM's with `floor_by_solver` False and True; each
merged after (`icm.merge_clones`). The 14 recoveries take 2.7 s serial at 32 x 32, 100 s at 128 x 128.
"""

from __future__ import annotations

import functools
import time
from typing import NamedTuple

import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.metrics import adjusted_rand_score

from cnamaste import opengm
from cnamaste.config import YAMLConfig, set_global_config
from cnamaste.icm import icm_sweep_deque, merge_clones
from cnamaste.potts import potts_energy, potts_problem
from cnamaste.spatial import construct_lattice_adjacency
from potts_reference import get_grid_coords

pytestmark = pytest.mark.xdist_group("label_solvers")

N_CLONES, SPATIAL_WEIGHT = 6, 1.0
GENOTYPE = np.array([0, 1, 2, 3, 4, 3])
"""Each label's genotype: labels 3 and 5 share one, so a merge folds them."""


class Lattice(NamedTuple):
    side: int
    box: int
    floor: int
    sigma: float


SMALL = Lattice(side=32, box=4, floor=25, sigma=0.7)
LARGE = Lattice(side=128, box=10, floor=400, sigma=1.5)
"""Each lattice's side, its small clone's side, the floor above that clone, and the field's noise."""


class Planted(NamedTuple):
    field: np.ndarray
    adjacency: sp.csr_matrix
    genotype: np.ndarray
    start: np.ndarray


@functools.cache
def get_planted(lattice: Lattice) -> Planted:
    """The field, the adjacency, each spot's planted genotype and the field's argmax, the start."""
    side = lattice.side
    coords = get_grid_coords(side, side)
    planted = (coords[:, 0] >= side // 2).astype(np.int64) * 2 + (coords[:, 1] >= side // 2)
    planted[(planted == 3) & (coords[:, 1] >= 3 * side // 4)] = 5
    lo, hi = side * 20 // 128, side * 20 // 128 + lattice.box
    planted[(coords[:, 0] >= lo) & (coords[:, 0] < hi) & (coords[:, 1] >= lo) & (coords[:, 1] < hi)] = 4
    rng = np.random.default_rng(0)
    field = -1.0 * (GENOTYPE[None, :] != GENOTYPE[planted][:, None]) + rng.normal(
        0.0, lattice.sigma, (len(coords), N_CLONES)
    )
    adjacency = construct_lattice_adjacency(coords, coordination_num=8)
    return Planted(field, adjacency, GENOTYPE[planted], np.argmax(field, axis=1))


class Case(NamedTuple):
    lattice: Lattice
    solver: str
    mincut: str
    floor: int
    floor_by_solver: bool


class Solved(NamedTuple):
    labels: np.ndarray
    seconds: float
    merged: np.ndarray
    merges: list


def get_cases(lattice: Lattice, marks: tuple[pytest.MarkDecorator, ...] = ()) -> list:
    """Every solver, mincut, floor and `floor_by_solver` case on `lattice`, each row carrying `marks`."""
    bk, dinic, floor = "boykov_kolmogorov", "dinic", lattice.floor
    cases = [Case(lattice, "icm", "-", 0, False), Case(lattice, "icm", "-", floor, False)]
    cases += [Case(lattice, "opengm_icm", bk, f, by) for f, by in ((0, False), (floor, False), (floor, True))]
    cases += [
        Case(lattice, "opengm_expansion", m, f, by)
        for m in (bk, dinic)
        for f, by in ((0, False), (floor, False), (floor, True))
    ]
    cases += [Case(lattice, "opengm_trws", bk, f, by) for f, by in ((0, False), (floor, False), (floor, True))]
    return [
        pytest.param(
            c,
            marks=marks,
            id=f"{lattice.side}-{c.solver}-{c.mincut}-floor{c.floor}{'-by_solver' if c.floor_by_solver else ''}",
        )
        for c in cases
    ]


CASES = get_cases(SMALL)
LARGE_CASES = get_cases(LARGE, (pytest.mark.benchmark, pytest.mark.slow, pytest.mark.very_slow))
"""The 128 x 128 recoveries, a `benchmark`: 100 s serial, TRW-S 12-18 s a case."""

ARI_FLOOR = {"icm": 0.84, "opengm_icm": 0.90, "opengm_expansion": 0.93, "opengm_trws": 0.97}
"""Each solver's ARI against the planted genotype after the merge, pinned below its lowest over cases, on both lattices:
0.962, 0.954, 0.959 and 0.978 at 32; 0.853, 0.906, 0.938 and 0.973 at 128, where expansion keeps 492 spots of
quadrant 3 apart, which no merge lowers."""

MERGING = ("icm", "opengm_icm")
"""The solvers that leave quadrant 3 in two clones, which the merge folds; the others leave none."""

SOLVED: dict[Case, Solved] = {}


def get_solved(c: Case) -> Solved:
    """`c` run once per worker, from the field's argmax, then merged."""
    if c not in SOLVED:
        p = get_planted(c.lattice)
        a, labels = p.adjacency, p.start.copy()
        began = time.perf_counter()
        if c.solver == "icm":
            labels = icm_sweep_deque(
                p.field, a.indptr, a.indices, a.data, labels, SPATIAL_WEIGHT, None, min_clone_spots=c.floor
            )[2]
        else:
            set_global_config(YAMLConfig({"hmrf": {"opengm_mincut_solver": c.mincut}}))
            opengm.external_assignment(
                p.field,
                a,
                labels,
                SPATIAL_WEIGHT,
                c.solver,
                min_clone_spots=c.floor,
                floor_by_solver=c.floor_by_solver,
            )
        seconds = time.perf_counter() - began
        merged, _, merges = merge_clones(p.field, a.indptr, a.indices, a.data, labels, SPATIAL_WEIGHT)
        SOLVED[c] = Solved(labels, seconds, merged, merges)
    return SOLVED[c]


def get_energy(labels: np.ndarray) -> float:
    """`potts_energy` of `labels` on `SMALL`."""
    p = get_planted(SMALL)
    return potts_energy(p.field, p.adjacency, labels, SPATIAL_WEIGHT)


def get_best_move_gain(labels: np.ndarray) -> float:
    """The most any single spot's move to an occupied clone lowers the energy on `SMALL`: its field plus
    `spatial_weight` times its neighbours' `(A + A^T) / 2` weight at each label, against its own. A clone the floor
    emptied is not a move."""
    p = get_planted(SMALL)
    half = ((p.adjacency + p.adjacency.T) / 2.0).tocsr()
    score = p.field + SPATIAL_WEIGHT * (half @ np.eye(N_CLONES)[labels])
    score[:, np.bincount(labels, minlength=N_CLONES) == 0] = -np.inf
    return float((score.max(axis=1) - score[np.arange(len(labels)), labels]).max())


@pytest.mark.parametrize("c", CASES + LARGE_CASES)
def test_each_solver_recovers_the_planted_clones(c: Case) -> None:
    """After the merge, every case's ARI against the planted genotype is at or above its solver's pinned floor. Fails
    where a solver, its floor or the merge loses the quadrants."""
    genotype = get_planted(c.lattice).genotype
    assert adjusted_rand_score(genotype, get_solved(c).merged) >= ARI_FLOOR[c.solver]


@pytest.mark.parametrize("c", CASES)
def test_every_merge_lowers_the_energy(c: Case) -> None:
    """Each merge `merge_clones` takes, replayed in order, lowers `potts_energy`; under ICM, the shared genotype is
    folded. Fails where the merge gain's sign, the boundary or the unary trade is wrong."""
    solved = get_solved(c)
    assert bool(solved.merges) == (c.solver in MERGING)
    labels, energy = solved.labels.copy(), get_energy(solved.labels)
    for u, v, _ in solved.merges:
        labels[labels == u] = v
        assert get_energy(labels) < energy
        energy = get_energy(labels)
    assert np.array_equal(labels, solved.merged)


@pytest.mark.parametrize("c", [p for p in CASES if p.values[0].floor > 0])
def test_the_floor_leaves_no_clone_under_it(c: Case) -> None:
    """With `min_clone_spots` 25, which the planted 16-spot clone is under: no occupied clone below 25, before the
    merge. Fails where the floor, ICM's or OpenGM's re-solve, leaves one."""
    counts = np.bincount(get_solved(c).labels, minlength=N_CLONES)
    assert counts[counts > 0].min() >= SMALL.floor


@pytest.mark.parametrize("c", CASES)
def test_every_case_ends_at_a_single_spot_fixed_point(c: Case) -> None:
    """No single spot's move to an occupied clone lowers the energy, to 1e-9, also after the floor moves spots at
    random. Fails where a sweep stops with a spot still to move, or the floor's moved spots are not revisited (T- #22)."""
    assert get_best_move_gain(get_solved(c).labels) <= 1e-9


def test_icm_sweeps_never_raise_the_energy() -> None:
    """`icm_sweep_deque`'s energy after each sweep, floor off, never rises and falls at least once. Fails where a move
    or the asymmetric coupling's half weights raise the energy."""
    p, trace = get_planted(SMALL), []
    a = p.adjacency
    icm_sweep_deque(
        p.field, a.indptr, a.indices, a.data, p.start, SPATIAL_WEIGHT, None, min_clone_spots=0, energy_trace=trace
    )
    assert len(trace) > 1 and np.all(np.diff(trace) <= 0.0) and trace[-1] < get_energy(p.start)


def test_icm_is_deterministic_and_reads_a_by_its_symmetrization() -> None:
    """`icm_sweep_deque` twice at `random_state` 0 gives the same labels, and on the asymmetric 8-nearest adjacency
    (280 one-way entries at the perimeter) gives those of `(A + A^T) / 2`. Fails where the order draws from an
    unseeded generator or the asymmetric read drifts from its symmetrization."""
    p = get_planted(SMALL)
    a, half = p.adjacency, ((p.adjacency + p.adjacency.T) / 2.0).tocsr()
    assert (a != a.T).nnz > 0
    first, again, symmetric = (
        icm_sweep_deque(p.field, m.indptr, m.indices, m.data, p.start, SPATIAL_WEIGHT, None, min_clone_spots=0)[2]
        for m in (a, a, half)
    )
    assert np.array_equal(first, again) and np.array_equal(first, symmetric)


def test_expansion_ends_below_icm_and_trws_bounds_every_energy() -> None:
    """Floor off: expansion's energy is below `icm_sweep_deque`'s and OpenGM ICM's; TRW-S's lower bound is below every
    case's energy, and within 0.5% of the best. Fails where expansion stops early or the bound is not a bound."""
    off = {c.values[0].solver: get_energy(get_solved(c.values[0]).labels) for c in CASES if c.values[0].floor == 0}
    assert off["opengm_expansion"] < min(off["icm"], off["opengm_icm"])
    p = get_planted(SMALL)
    bound = opengm.opengm_session().solve(potts_problem(p.field, p.adjacency, SPATIAL_WEIGHT), "trws").bound
    best = min(off.values())
    assert bound <= best and abs(best - bound) <= 0.005 * abs(best)
