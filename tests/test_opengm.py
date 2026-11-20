"""OpenGM, the default label solver's library, against exact minima (`potts_reference`).

- two labels, 16 x 16 bins: the max-flow minimum cut, which alpha expansion must reach, by either min-cut;
- a 256-spot chain, 4 labels: the dynamic programme, which TRW-S must reach, its bound at the minimum;
- 3 x 3 bins at 3 labels, 4 x 4 at 2: enumeration, which expansion must reach.
Every other algorithm is at or above each minimum. 256 spots are not enumerated: 2^256 labellings.
Tolerance: the min cut's field and couplings are integers, so equal there means equal; elsewhere 1e-9, summation
order. The file runs in 2 s, the worker's start among it.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import numpy as np
import pytest

from cnamaste import external
from cnamaste.potts import pairwise_potts_energy, potts_problem
from potts_reference import (
    PlantedField,
    get_chain,
    get_chain_minimum,
    get_enumerable,
    get_enumerable_grid,
    get_enumerated_minimum,
    get_min_cut_labels,
    get_two_label_lattice,
)

ATOL = 1e-9


def test_the_opengm_home_is_set() -> None:
    """`CNAMASTE_OPENGM_HOME` is set and holds the built library. Fails where the variable is unset, or names a
    directory `infra/opengm/build_opengm.sh` has not built into, which the default run refuses."""
    assert os.environ.get(external.get_home_variable()), f"set {external.get_home_variable()}: see OPENGM.md"
    assert external.available(), str(external.unavailable())


@pytest.fixture(scope="module")
def session() -> Iterator[external.Session]:
    with external.Session() as opened:
        yield opened


def get_solved(session: external.Session, field: PlantedField, mincut: str = "boykov_kolmogorov") -> dict:
    """ICM and expansion from the field's argmax, TRW-S from none: each `OpenGMRun`, its energy checked as its labels'."""
    problem = potts_problem(field.single_llf, field.adjacency, field.spatial_weight)
    start = np.argmax(field.single_llf, axis=1)
    runs = {
        algorithm: session.solve(problem, algorithm, start=None if algorithm == "trws" else start, mincut=mincut)
        for algorithm in ("icm", "expansion", "trws")
    }
    for run in runs.values():
        assert np.isclose(run.energy, pairwise_potts_energy(problem, run.labels), rtol=0.0, atol=ATOL)
    return runs


@pytest.mark.parametrize("mincut", sorted(external.get_mincut_solvers()))
def test_expansion_reaches_the_two_label_min_cut(session: external.Session, mincut: str) -> None:
    """16 x 16 bins, two labels: expansion's energy is the max-flow minimum, -1872, by Boykov-Kolmogorov and by Dinic;
    ICM and TRW-S are at or above it, ICM 9 above. Fails where a cut, the coupling's fold or the transport drifts."""
    field = get_two_label_lattice()
    problem = potts_problem(field.single_llf, field.adjacency, field.spatial_weight)
    minimum = pairwise_potts_energy(problem, get_min_cut_labels(problem))
    runs = get_solved(session, field, mincut)
    assert minimum == -1872.0 and runs["expansion"].energy == minimum
    assert runs["icm"].energy >= minimum and runs["trws"].energy >= minimum


def test_trws_reaches_the_chain_minimum(session: external.Session) -> None:
    """A 256-spot chain, 4 labels: TRW-S's energy and its bound are the dynamic programme's minimum, as TRW-S is exact
    on a tree; ICM and expansion are above it. Fails where TRW-S's messages, its bound or the field drift."""
    field = get_chain()
    problem = potts_problem(field.single_llf, field.adjacency, field.spatial_weight)
    minimum = pairwise_potts_energy(problem, get_chain_minimum(problem))
    runs = get_solved(session, field)
    assert np.isclose(runs["trws"].energy, minimum, rtol=0.0, atol=ATOL)
    assert minimum - ATOL <= runs["trws"].bound <= minimum + ATOL
    assert min(runs["icm"].energy, runs["expansion"].energy) > minimum + 1.0


@pytest.mark.parametrize("name", get_enumerable())
def test_expansion_reaches_the_enumerated_minimum(session: external.Session, name: str) -> None:
    """Every labelling of 3 x 3 bins at 3 labels and 4 x 4 at 2: expansion's energy is the minimum, which the field's
    argmax misses; ICM and TRW-S are at or above it. Fails where any algorithm returns below the true minimum."""
    field = get_enumerable_grid(name)
    problem = potts_problem(field.single_llf, field.adjacency, field.spatial_weight)
    labels, minimum = get_enumerated_minimum(problem)
    assert not np.array_equal(labels, np.argmax(field.single_llf, axis=1))
    runs = get_solved(session, field)
    assert np.isclose(runs["expansion"].energy, minimum, rtol=0.0, atol=ATOL)
    assert all(run.energy >= minimum - ATOL for run in runs.values())
