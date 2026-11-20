"""`potts`'s energy in its two representations, the run's stored adjacency and the solver's pairwise form.

Referee: each form against the other, `potts_problem` folding the edges independently of `icm.calc_assignment_cost`'s
sum over them. Input: 80 spots, 600 random directed edges, duplicates and one-way edges among them, weights in
[0.1, 2], 4 clones. Tolerance 1e-12 absolute, as the two sum the same terms in different orders (measured 2.1e-14).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from cnamaste.icm import calc_assignment_cost
from cnamaste.potts import pairwise_potts_energy, potts_energy, potts_problem

N_SPOTS, N_CLONES, SPATIAL_WEIGHT = 80, 4, 0.7


def get_random_graph() -> tuple[np.ndarray, sp.coo_matrix, np.ndarray]:
    """A field, an asymmetric COO adjacency with duplicate entries kept unsummed, and 20 labellings."""
    rng = np.random.default_rng(3)
    rows, cols = rng.integers(0, N_SPOTS, 600), rng.integers(0, N_SPOTS, 600)
    keep = rows != cols
    rows, cols = rows[keep], cols[keep]
    adjacency = sp.coo_matrix((rng.uniform(0.1, 2.0, rows.size), (rows, cols)), shape=(N_SPOTS, N_SPOTS))
    return rng.normal(0.0, 2.0, (N_SPOTS, N_CLONES)), adjacency, rng.integers(0, N_CLONES, (20, N_SPOTS))


def test_the_pairwise_form_is_the_stored_form_s_energy() -> None:
    """On 20 labellings of an asymmetric graph with duplicate edges: `pairwise_potts_energy(potts_problem(...))` is
    `potts_energy` within 1e-12, and `potts_energy` is `-calc_assignment_cost` bitwise. Fails where the fold to
    unordered pairs, the duplicate sum or the `w / 2` per directed edge drifts."""
    field, adjacency, labellings = get_random_graph()
    assert (adjacency != adjacency.T).nnz > 0 and len(set(zip(adjacency.row, adjacency.col))) < adjacency.nnz
    problem = potts_problem(field, adjacency, SPATIAL_WEIGHT)
    for labels in labellings:
        energy = potts_energy(field, adjacency, labels, SPATIAL_WEIGHT)
        cost = calc_assignment_cost(field, adjacency.row, adjacency.col, adjacency.data, labels, SPATIAL_WEIGHT)
        assert energy == -cost
        np.testing.assert_allclose(pairwise_potts_energy(problem, labels), energy, rtol=0.0, atol=1e-12)


def test_relabelling_the_clones_leaves_the_energy() -> None:
    """Permuting the clone labels, field columns and labels alike, leaves `potts_energy` bitwise. Fails where the
    energy reads a label's value rather than whether two labels agree."""
    field, adjacency, labellings = get_random_graph()
    permutation = np.array([2, 0, 3, 1])
    relabelled = np.empty_like(field)
    relabelled[:, permutation] = field
    for labels in labellings:
        assert potts_energy(relabelled, adjacency, permutation[labels], SPATIAL_WEIGHT) == potts_energy(
            field, adjacency, labels, SPATIAL_WEIGHT
        )
