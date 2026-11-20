"""Exact minima of `potts.PottsProblem`s, each by a method independent of the solvers they referee.

- `get_min_cut_labels`: two labels, by a max flow (`scipy.sparse.csgraph.maximum_flow`); two-label Potts with
  `J >= 0` is submodular, so the minimum cut is the global minimum. Integer costs, as the max flow takes integers.
- `get_chain_minimum`: a chain, by dynamic programming over its spots in order.
- `get_enumerated_minimum`: every labelling, for `K^n` up to ~1e5.

Enumeration does not reach 256 spots, 2^256 labellings: the first two stand in at that size.
"""

from __future__ import annotations

import itertools
from typing import NamedTuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import breadth_first_order, maximum_flow

from cnamaste.potts import PottsProblem
from cnamaste.spatial import construct_lattice_adjacency


def get_grid_coords(rows: int, cols: int) -> np.ndarray:
    return np.array([(i, j) for i in range(rows) for j in range(cols)], dtype=float)


def get_four_neighbour_adjacency(rows: int, cols: int) -> sp.csr_matrix:
    """A grid's 4-neighbour adjacency, symmetric, unit weight."""
    four = construct_lattice_adjacency(get_grid_coords(rows, cols), coordination_num=4)
    return ((four + four.T) > 0).astype(float).tocsr()


def get_min_cut_labels(problem: PottsProblem) -> np.ndarray:
    """The two-label minimum by a max flow: source side label 0, sink side label 1.

    Spot i pays `unary[i, 1]` on the sink side and `unary[i, 0]` on the source side, each less their minimum; a pair
    pays `J` where cut, as `-J [l_i == l_j] = -J + J [l_i != l_j]`.

    Args: `problem` with two labels, integer unary costs and couplings, `J >= 0`.
    Returns: the labels.
    """
    n = problem.unary.shape[0]
    source, sink = n, n + 1
    cost = problem.unary - problem.unary.min(axis=1, keepdims=True)
    rows = np.concatenate([np.full(n, source), np.arange(n), problem.first, problem.second])
    cols = np.concatenate([np.arange(n), np.full(n, sink), problem.second, problem.first])
    capacity = np.concatenate([cost[:, 1], cost[:, 0], problem.coupling, problem.coupling])
    if not np.array_equal(capacity, np.round(capacity)) or capacity.min() < 0:
        raise ValueError("the max flow takes non-negative integer costs and couplings")
    graph = sp.csr_matrix((capacity.astype(np.int32), (rows, cols)), shape=(n + 2, n + 2))
    flow = maximum_flow(graph, source, sink).flow
    residual = (graph - flow).tocsr()
    residual.data[residual.data < 0] = 0
    residual.eliminate_zeros()
    reached = breadth_first_order(residual, source, directed=True, return_predecessors=False)
    labels = np.ones(n, dtype=np.int64)
    labels[reached[reached < n]] = 0
    return labels


def get_chain_minimum(problem: PottsProblem) -> np.ndarray:
    """The minimum of a chain, pairs `(i, i + 1)`, by dynamic programming (min-sum Viterbi).

    Returns: the labels.
    """
    n, k = problem.unary.shape
    assert np.array_equal(problem.first, np.arange(n - 1)) and np.array_equal(problem.second, np.arange(1, n))
    best = problem.unary[0].copy()
    back = np.zeros((n, k), dtype=np.int64)
    for i in range(1, n):
        # NB step[a, b]: the chain to spot i - 1 at a, then spot i at b.
        step = best[:, None] - problem.coupling[i - 1] * np.eye(k)
        back[i] = np.argmin(step, axis=0)
        best = step[back[i], np.arange(k)] + problem.unary[i]
    labels = np.zeros(n, dtype=np.int64)
    labels[-1] = np.argmin(best)
    for i in range(n - 1, 0, -1):
        labels[i - 1] = back[i, labels[i]]
    return labels


def get_enumerated_minimum(problem: PottsProblem) -> tuple[np.ndarray, float]:
    """The minimum over all `K^n` labellings, vectorized.

    Returns: the minimizing labels and their energy.
    """
    n, k = problem.unary.shape
    labellings = np.array(list(itertools.product(range(k), repeat=n)), dtype=np.int64)
    energies = problem.unary[np.arange(n), labellings].sum(axis=1)
    energies -= (problem.coupling * (labellings[:, problem.first] == labellings[:, problem.second])).sum(axis=1)
    best = int(np.argmin(energies))
    return labellings[best], float(energies[best])


class PlantedField(NamedTuple):
    """A field, its adjacency and coupling, and the labels it was planted from."""

    single_llf: np.ndarray
    adjacency: sp.csr_matrix
    spatial_weight: float
    planted: np.ndarray


def get_two_label_lattice() -> PlantedField:
    """16 x 16 bins, the run's 8 nearest, two planted halves; integer field, `spatial_weight` 2, so every `J` is 1 or 2.

    The field alone is 444 above the minimum, ICM 9 above it.
    """
    rng = np.random.default_rng(0)
    coords = get_grid_coords(16, 16)
    planted = (coords.sum(axis=1) >= 16).astype(np.int64)
    field = -(4.0 * (np.arange(2)[None, :] != planted[:, None]) + rng.integers(-6, 7, (len(coords), 2)))
    return PlantedField(field, construct_lattice_adjacency(coords, coordination_num=8), 2.0, planted)


def get_chain() -> PlantedField:
    """256 spots in a chain, 4 planted runs of 64, Gaussian noise (sd 1.5) over a gap of 1; `spatial_weight` 3."""
    rng = np.random.default_rng(1)
    n = 256
    planted = np.repeat(np.arange(4), n // 4)
    field = -(1.0 * (np.arange(4)[None, :] != planted[:, None]) + rng.normal(0.0, 1.5, (n, 4)))
    adjacency = sp.coo_matrix((np.ones(n - 1), (np.arange(n - 1), np.arange(1, n))), shape=(n, n)).tocsr()
    return PlantedField(field, adjacency, 3.0, planted)


ENUMERABLE = {"3x3_k3": (3, 3, 3), "4x4_k2": (4, 4, 2)}
"""Grids small enough to enumerate: 3^9 = 19,683 and 2^16 = 65,536 labellings."""


def get_enumerable() -> dict[str, tuple[int, ...]]:
    """Return the enumerable, {"3x3_k3": (3, 3, 3), "4x4_k2": (4, 4, 2)}."""
    return ENUMERABLE


def get_enumerable_grid(name: str) -> PlantedField:
    """`ENUMERABLE[name]`'s 4-neighbour grid, a standard normal field, `spatial_weight` 0.3, no planted labels."""
    rows, cols, k = ENUMERABLE[name]
    field = np.random.default_rng(2).normal(0.0, 1.0, (rows * cols, k))
    return PlantedField(field, get_four_neighbour_adjacency(rows, cols), 0.3, np.zeros(rows * cols, dtype=np.int64))
