"""Synthetic Potts problems on the square lattice: planted contiguous clones under a noisy field.

The truth is a Voronoi partition of a `rows x cols` lattice from `n_clones` seeds, so every clone is contiguous. The
field is `signal` at each spot's true clone plus Gaussian noise of unit scale; `signal` is solved for so the field's
argmax mislabels `mislabelled` of the spots. The adjacency is `hmrf.lattice_type: square`'s, each spot's 4 edges and 4
corners at weight 1, stored both ways as the run stores it.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import scipy.sparse as sp
from scipy.special import ndtr

from cnamaste.potts import PottsProblem, potts_problem


class PlantedLattice(NamedTuple):
    """A planted problem: the field `(n_spots, n_clones)`, the adjacency, the true labels and the `PottsProblem`.

    Args:
        single_llf: the field.
        adjacency: the CSR adjacency.
        truth: the planted labels.
        problem: the `PottsProblem`.
    """

    single_llf: np.ndarray
    adjacency: sp.csr_matrix
    truth: np.ndarray
    problem: PottsProblem


def get_square_adjacency(rows: int, cols: int) -> sp.csr_matrix:
    """The 8-neighbour square lattice, weight 1, both directions stored.

    Args:
        rows: the lattice's rows.
        cols: the lattice's columns; spot `r * cols + c`.

    Returns:
        The `(rows * cols, rows * cols)` CSR adjacency.
    """
    index = np.arange(rows * cols).reshape(rows, cols)
    tails, heads = [], []
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        r0, r1 = 0, rows - dr
        c0, c1 = max(0, -dc), cols - max(0, dc)
        tails.append(index[r0:r1, c0:c1].ravel())
        heads.append(index[r0 + dr : r1 + dr, c0 + dc : c1 + dc].ravel())
    tail, head = np.concatenate(tails), np.concatenate(heads)
    n = rows * cols
    a = sp.coo_matrix((np.ones(2 * tail.size), (np.r_[tail, head], np.r_[head, tail])), shape=(n, n))
    return a.tocsr()


def get_signal_for_mislabelled(n_clones: int, mislabelled: float) -> float:
    """The field's margin at which a spot's argmax is wrong with probability `mislabelled`, by bisection.

    Args:
        n_clones: the labels.
        mislabelled: the target share.

    Returns:
        The signal, in units of the noise's scale.
    """
    grid = np.linspace(-8.0, 16.0, 4001)
    density = np.exp(-0.5 * grid**2) / np.sqrt(2.0 * np.pi)
    lo, hi = 0.0, 16.0
    for _ in range(60):
        signal = 0.5 * (lo + hi)
        correct = np.trapezoid(density * ndtr(grid + signal) ** (n_clones - 1), grid)
        lo, hi = (signal, hi) if 1.0 - correct > mislabelled else (lo, signal)
    return 0.5 * (lo + hi)


def get_planted_lattice(
    rows: int, cols: int, n_clones: int, mislabelled: float = 0.3, spatial_weight: float = 1.1, seed: int = 0
) -> PlantedLattice:
    """A planted problem on the square lattice.

    Args:
        rows: the lattice's rows.
        cols: the lattice's columns.
        n_clones: the labels and seeds.
        mislabelled: the field argmax's error rate.
        spatial_weight: the coupling.
        seed: the generator's.

    Returns:
        The `PlantedLattice`.
    """
    rng = np.random.default_rng(seed)
    r, c = np.divmod(np.arange(rows * cols), cols)
    seeds = rng.uniform((0.0, 0.0), (rows, cols), size=(n_clones, 2))
    truth = np.argmin((r[:, None] - seeds[:, 0]) ** 2 + (c[:, None] - seeds[:, 1]) ** 2, axis=1).astype(np.int64)
    single_llf = rng.standard_normal((rows * cols, n_clones))
    single_llf[np.arange(truth.size), truth] += get_signal_for_mislabelled(n_clones, mislabelled)
    adjacency = get_square_adjacency(rows, cols)
    return PlantedLattice(single_llf, adjacency, truth, potts_problem(single_llf, adjacency, spatial_weight))
