from __future__ import annotations

from typing import NamedTuple

import numpy as np
import scipy.sparse as sp

from cnamaste.icm import calc_assignment_cost, refuse_self_loops


class PottsProblem(NamedTuple):
    """Unary costs `(n_spots, n_clones)`, each pair's ends `first < second`, and its coupling `J`.

    Args:
        unary: the arrays.
        first: the arrays.
        second: the arrays.
        coupling: the arrays.
    """

    unary: np.ndarray
    first: np.ndarray
    second: np.ndarray
    coupling: np.ndarray


def noloop_adjacency_coo_cast(adjacency):
    """Return `adjacency` as a COO matrix, refused if any stored edge joins a spot to itself.

    Args:
        adjacency: sparse or dense.

    Returns:
        The COO matrix.

    Raises:
        ValueError: from `icm.refuse_self_loops`.
    """
    a = sp.coo_matrix(adjacency)
    refuse_self_loops(a.row, a.col, a.data)
    return a


def potts_energy(single_llf, adjacency, labels, spatial_weight):
    """The Potts energy of `labels` on the spot adjacency as the run stores it: each spot's field plus
    `spatial_weight * w / 2` per agreeing stored edge, directed, not symmetrized, negated: `-icm.calc_assignment_cost`.

    `pairwise_potts_energy` is the same energy on `potts_problem`'s pairwise form, the one an external solver takes.

    Args: `single_llf` (n_spots, n_clones); `adjacency` sparse, as stored, no self-loop; `labels`; `spatial_weight`.

    Returns:
        The energy.
    """
    a = sp.coo_matrix(adjacency)
    return -calc_assignment_cost(single_llf, a.row, a.col, a.data, labels, spatial_weight)


def potts_problem(single_llf, adjacency, spatial_weight):
    """Pose the energy as unary costs and couplings: `E(l) = sum unary[i, l_i] - sum J [l_first == l_second]`.

    Args: `single_llf` (n_spots, n_clones); `adjacency` sparse, no self-loop; `spatial_weight`.

    Returns:
        The `PottsProblem`.
    """
    a = noloop_adjacency_coo_cast(adjacency)
    lo, hi = np.minimum(a.row, a.col), np.maximum(a.row, a.col)
    pairs = sp.coo_matrix((a.data, (lo, hi)), shape=a.shape).tocsr()
    pairs.sum_duplicates()
    pairs = pairs.tocoo()
    return PottsProblem(
        unary=np.ascontiguousarray(-np.asarray(single_llf, dtype=np.float64)),
        first=np.ascontiguousarray(pairs.row, dtype=np.int64),
        second=np.ascontiguousarray(pairs.col, dtype=np.int64),
        coupling=np.ascontiguousarray(spatial_weight * pairs.data / 2.0, dtype=np.float64),
    )


def pairwise_potts_energy(problem, labels):
    """The Potts energy of `labels` on `potts_problem`'s pairwise form: unary costs and one coupling `J` per
    unordered pair, `E(l) = sum unary[i, l_i] - sum J [l_first == l_second]`.

    Its intention: the energy in the representation an external solver (OpenGM) minimizes, so its labels, energy and
    bound are scored in the solver's own terms. It equals `potts_energy` on the adjacency the problem was
    posed from, for every labelling.

    Args:
        problem: a `PottsProblem`.
        labels: per spot.

    Returns:
        The energy.
    """
    labels = np.asarray(labels)
    unary = problem.unary[np.arange(labels.size), labels].sum()
    return float(unary - np.sum(problem.coupling * (labels[problem.first] == labels[problem.second])))


class ReducedPotts(NamedTuple):
    """A `PottsProblem` reduced to the spots and clones a solve still decides, and the map back.

    Args:
        problem: the reduced problem.
        fixed: the full problem's labels, -1 at each spot still free.
        free: the full problem's index of each reduced spot.
        clones: the full problem's clone of each reduced label.
    """

    problem: PottsProblem
    fixed: np.ndarray
    free: np.ndarray
    clones: np.ndarray

    def full_labels(self, labels):
        """The full problem's labels from the reduced problem's.

        Args:
            labels: per reduced spot.

        Returns:
            The labels per spot.
        """
        full = self.fixed.copy()
        full[self.free] = self.clones[np.asarray(labels)]
        return full

    def reduced_labels(self, labels):
        """The reduced problem's labels from the full problem's: a clone the reduction dropped is the spot's best
        remaining, by its reduced unary.

        Args:
            labels: per spot.

        Returns:
            The labels per reduced spot.
        """
        labels = np.asarray(labels)[self.free]
        found = np.minimum(np.searchsorted(self.clones, labels), self.clones.size - 1)
        dropped = self.clones[found] != labels
        found[dropped] = np.argmin(self.problem.unary[dropped], axis=1)
        return found


def get_coupling_sums(problem):
    """Each spot's summed coupling, the most its labelling can gain or lose by its edges.

    Args:
        problem: a `PottsProblem`.

    Returns:
        `(n_spots,)` sums.
    """
    n = problem.unary.shape[0]
    return np.bincount(problem.first, problem.coupling, n) + np.bincount(problem.second, problem.coupling, n)


def get_label_margins(unary):
    """Each spot's best label by its unary cost, and the cost of its second best less its best's.

    Args:
        unary: `(n_spots, n_clones)`.

    Returns:
        The labels and the margins, `inf` with one clone.
    """
    best = np.argmin(unary, axis=1)
    if unary.shape[1] == 1:
        return best, np.full(unary.shape[0], np.inf)
    two = np.partition(unary, 1, axis=1)
    return best, two[:, 1] - two[:, 0]


def reduce_potts_problem(problem, fix_dominant_spots=False, prune_dominated_labels=False):
    """Fix the spots, and drop the clones, that every minimum of the energy decides, and pose what remains.

    A spot whose unary margin exceeds its summed coupling takes its best label in every minimum: any other label
    loses more by its unary than its edges can return. Fixing it folds its edges into its neighbours' unary, which
    can fix more, so fixing repeats until none is fixed. A clone whose unary cost exceeds every remaining spot's best
    by more than that spot's summed coupling is in no minimum, and is dropped. Both are exact for a minimum; a solver
    that stops at a local one, as expansion does, can stop elsewhere on the reduced problem.

    Args:
        problem: a `PottsProblem`.
        fix_dominant_spots: fix each spot whose unary margin exceeds its summed coupling.
        prune_dominated_labels: drop each clone in no minimum.

    Returns:
        The `ReducedPotts`, the problem itself where neither is taken.
    """
    n, k = problem.unary.shape
    fixed = np.full(n, -1, dtype=np.int64)
    unary = np.array(problem.unary, dtype=np.float64, copy=True)
    first, second, coupling = problem.first, problem.second, problem.coupling
    while fix_dominant_spots:
        free = fixed < 0
        inner = free[first] & free[second]
        sums = np.bincount(first[inner], coupling[inner], n) + np.bincount(second[inner], coupling[inner], n)
        best, margin = get_label_margins(unary)
        new = free & (margin > sums)
        if not new.any():
            break
        fixed[new] = best[new]
        # NB an edge from a newly fixed spot to a free one is that free spot's unary, at the fixed spot's label.
        for ends, other in ((first, second), (second, first)):
            fold = new[ends] & free[other] & ~new[other]
            np.subtract.at(unary, (other[fold], fixed[ends[fold]]), coupling[fold])
    free = np.flatnonzero(fixed < 0)
    index = np.full(n, -1, dtype=np.int64)
    index[free] = np.arange(free.size)
    inner = (fixed[first] < 0) & (fixed[second] < 0)
    reduced = PottsProblem(
        np.ascontiguousarray(unary[free]),
        np.ascontiguousarray(index[first[inner]]),
        np.ascontiguousarray(index[second[inner]]),
        np.ascontiguousarray(coupling[inner]),
    )
    clones = np.arange(k)
    if prune_dominated_labels and free.size:
        excess = reduced.unary - reduced.unary.min(axis=1, keepdims=True)
        clones = np.flatnonzero(~np.all(excess > get_coupling_sums(reduced)[:, None], axis=0))
        reduced = reduced._replace(unary=np.ascontiguousarray(reduced.unary[:, clones]))
    return ReducedPotts(reduced, fixed, free, clones)


def get_field_share_order(unary):
    """The clones by how many spots' unary each is best for, most first, ties by clone.

    Args:
        unary: `(n_spots, n_clones)`.

    Returns:
        The clones in order.
    """
    counts = np.bincount(np.argmin(unary, axis=1), minlength=unary.shape[1])
    return np.argsort(-counts, kind="stable")
