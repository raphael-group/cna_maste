"""`icm`'s sweep and merges: `icm_sweep_deque` against the exact minima of `potts_reference`, its refusals, and the
merge tallies against a recount.

ICM is a local search, so against an exact minimum it is at or above it, never below; the planted 16,384-spot lattice
(`test_label_solvers.py`) judges where it ends. The file runs in 1 s.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from cnamaste.icm import calc_assignment_cost, fold_clone, icm_sweep_deque, merge_tallies
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


def get_icm_labels(field: PlantedField, start: np.ndarray | None = None, **kwargs) -> np.ndarray:
    a = field.adjacency
    start = np.argmax(field.single_llf, axis=1) if start is None else start
    kwargs.setdefault("min_clone_spots", 0)
    return icm_sweep_deque(field.single_llf, a.indptr, a.indices, a.data, start, field.spatial_weight, None, **kwargs)[
        2
    ]


@pytest.mark.parametrize(
    ("get_field", "get_minimum"),
    [
        (get_two_label_lattice, lambda p: get_min_cut_labels(p)),
        (get_chain, lambda p: get_chain_minimum(p)),
        *[(lambda n=name: get_enumerable_grid(n), lambda p: get_enumerated_minimum(p)[0]) for name in get_enumerable()],
    ],
    ids=["min_cut", "chain", *get_enumerable()],
)
def test_icm_ends_at_or_above_the_exact_minimum(get_field, get_minimum) -> None:
    """From the field's argmax: `icm_sweep_deque`'s energy is at or above the min cut, the chain's dynamic programme and
    the enumerated minima, and no higher than its start. Fails where a move raises the energy or ICM reports below a
    minimum no labelling beats."""
    field = get_field()
    problem = potts_problem(field.single_llf, field.adjacency, field.spatial_weight)
    minimum = pairwise_potts_energy(problem, get_minimum(problem))
    energy = pairwise_potts_energy(problem, get_icm_labels(field))
    assert minimum - 1e-9 <= energy <= pairwise_potts_energy(problem, np.argmax(field.single_llf, axis=1))


@pytest.mark.xfail(
    strict=True, reason="a tie goes to the lowest-numbered clone (strict > over c), not the current label"
)
def test_tie_keeps_current_label() -> None:
    """Two unconnected spots, each with the same field at both clones, start at clone 1: ICM leaves them there, as no
    move lowers the energy. `icm_sweep_deque` moves both to clone 0."""
    field = PlantedField(np.zeros((2, 2)), sp.csr_matrix((2, 2)), 1.0, np.ones(2, dtype=np.int64))
    assert np.array_equal(get_icm_labels(field, start=np.ones(2, dtype=np.int64)), [1, 1])


def test_a_self_loop_is_refused() -> None:
    """`icm_sweep_deque` and `calc_assignment_cost` refuse an adjacency that joins a spot to itself, as `potts` does:
    a self-loop is aligned in every labelling. Fails where either accepts one."""
    adjacency = sp.csr_matrix(np.array([[1.0, 1.0], [1.0, 0.0]]))
    field = PlantedField(np.zeros((2, 2)), adjacency, 1.0, np.zeros(2, dtype=np.int64))
    with pytest.raises(ValueError, match="self-loops"):
        get_icm_labels(field)
    a = adjacency.tocoo()
    with pytest.raises(ValueError, match="self-loops"):
        calc_assignment_cost(field.single_llf, a.row, a.col, a.data, field.planted, 1.0)


def test_folded_tallies_are_a_recount() -> None:
    """On the chain, ICM's labels, each clone pair: `fold_clone` of `merge_tallies` is `merge_tallies` of the merged
    labels within 1e-12, unary sums, boundaries and aligned coupling alike. Fails where a fold drops a boundary, keeps
    the folded clone's row, or misses the coupling the merge aligns."""
    field = get_chain()
    a = field.adjacency
    labels = get_icm_labels(field)
    args = (a.indptr, a.indices, a.data)
    for u in range(4):
        for v in range(4):
            if u == v:
                continue
            unary_sum, boundary_gain, aligned = merge_tallies(field.single_llf, *args, labels, field.spatial_weight)
            aligned = fold_clone(unary_sum, boundary_gain, aligned, u, v)
            recount = merge_tallies(field.single_llf, *args, np.where(labels == u, v, labels), field.spatial_weight)
            for folded, counted in zip((unary_sum, boundary_gain, aligned), recount, strict=True):
                np.testing.assert_allclose(folded, counted, rtol=0.0, atol=1e-12)
