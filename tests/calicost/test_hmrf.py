"""The HMRF labelling against CalicoST `c1abcae`'s `hmrf` reassignments and merges, and `utils_hmrf`'s adjacencies,
pseudobulk and reorder.

The referee is CalicoST's own output on the same input, computed once and recorded (`calicost_cache`); cnamaste's side
is `icm.calc_assignment_cost`, the energy the run's solvers maximize (`hmrf.get_labelling_unary` its unary),
`icm.icm_sweep_deque`, `icm.merge_clones`, `get_pseudobulk`, `hmrf.get_merged_fit` and `hmrf.get_reindexed_fit`.
Inputs: 64 spots on an 8 x 8 square lattice, three planted clones in bands (24, 24 and 16 spots), 40 bins at depth 2,
seeded; its 4-neighbour adjacency, symmetric and without self-loops. CalicoST's `hmrf` imports `turtle`, which needs
`tkinter`, for a name it never calls: a stub stands in for it at import. The file runs in 3 s.

Equivalences, under their conditions:
- an identity `smooth_mat` (no pooling): CalicoST's unary, `nodepotential: max` and the shipped `weighted_sum` on a
  one-hot posterior, is `get_labelling_unary`'s, to 1e-11 nats (measured 1.6e-12 on 40 bins of ~1e2 nats; the
  emissions differ by 1e-13 per term, `test_hmm_nophasing`);
- a symmetric adjacency without self-loops, the same unary and order: CalicoST's update is a best single-spot move of
  cnamaste's energy, label for label;
- interior square bins: CalicoST's `KNN` pooled and HMRF neighbours together, at `maxspots_pooling = 2` and less its
  self-loops, are cnamaste's 8 nearest, on one slice and on two. No `maxspots_pooling` reaches the triangular 6:
  CalicoST takes 6 + it, >= 7;
- the pseudobulk, the spot-floor merge and the normal clone, bitwise; the Neyman-Pearson merge of clones on one path.

Drift, each a strict xfail naming the difference: `KNN`'s self-loops and pooled neighbours, its row-only reading of an
asymmetric adjacency, `total_llf`, one pass, the visiting order, `hexagon`'s weights, the per-sample weights, the soft
posterior, the order of the clones after the normal one and the Neyman-Pearson merge's event length.
"""

from __future__ import annotations

import functools
import sys
import types
from typing import Any, NamedTuple
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from calicost.hmm_NB_BB_nophasing import hmm_nophasing as calicost_hmm_nophasing
from calicost.hmm_NB_BB_nophasing_v2 import hmm_nophasing_v2
from calicost.hmm_NB_BB_phaseswitch import similarity_components_rdrbaf_neymanpearson
from calicost.utils_hmrf import (
    choose_adjacency_by_KNN,
    choose_adjacency_by_readcounts,
    multislice_adjacency as calicost_multislice_adjacency,
)

import cnamaste.calicost  # noqa: F401  (imports CalicoST with the environment held, as cnamaste does)
from calicost_cache import Referee, departs, get_digest
from cnamaste.emission import get_unconfounded
from cnamaste.hmm_params import CnaHmmParameters
from cnamaste.hmrf import get_labelling_unary, get_merged_fit, get_reindexed_fit
from cnamaste.hmrf_fit import CloneAssignment, HMMProfile, HmrfFit
from cnamaste.icm import calc_assignment_cost, icm_sweep_deque, merge_clones
from cnamaste.labelling import get_merged_labelling
from cnamaste.potts import noloop_adjacency_coo_cast
from cnamaste.spatial import construct_lattice_adjacency, construct_multislice_lattice_adjacency
from cnamaste.spatio_genomic_counts import SpatioGenomicLabelledMultiCounts
from labelled_counts import get_labelled_counts

# NB `from turtle import reset` at the top of CalicoST's `hmrf`, never called; `turtle` needs `tkinter`.
with mock.patch.dict(sys.modules, {"turtle": types.SimpleNamespace(reset=None)}):
    from calicost import hmrf as calicost_hmrf

FIELD_ATOL = 1e-11
SIDE, N_CLONES, N_OBS = 8, 3, 40
SPATIAL_WEIGHT = 10.0
"""The coupling at which one pass is not a fixed point on these inputs (7 spots move on a second)."""

PATHS = np.stack([np.repeat([0, 1, 2, 1], 10), np.repeat([1, 1, 0, 2], 10), np.repeat([2, 0, 1, 1], 10)], axis=1)
"""Each clone's decoded state per bin, `pred` (n_obs, n_clones)."""

LOG_MU, P_BINOM = np.log([[0.5], [1.0], [1.5]]), np.array([[0.1], [0.5], [0.3]])
PARAMETERS = CnaHmmParameters(None, LOG_MU, P_BINOM, np.full((3, 1), 0.05), np.full((3, 1), 30.0))


class PlantedInputs(NamedTuple):
    counts: SpatioGenomicLabelledMultiCounts
    """`X`'s, every spot labelled normal, so `base_nb_mean` is its `normal_baseline_sized`."""
    X: np.ndarray
    base_nb_mean: np.ndarray
    total_bb_RD: np.ndarray
    coords: np.ndarray
    planted: np.ndarray
    start: np.ndarray
    adjacency: sp.csr_matrix


@pytest.fixture(scope="module")
def planted() -> PlantedInputs:
    """Counts drawn from each spot's planted clone's path, a random start, and the 4-neighbour lattice."""
    rng = np.random.default_rng(0)
    coords = np.array([(i, j) for i in range(SIDE) for j in range(SIDE)], dtype=float)
    labels = np.minimum(coords[:, 0].astype(int) * N_CLONES // SIDE, N_CLONES - 1)
    state = PATHS[:, labels]
    base = rng.uniform(0.5, 2.0, (N_OBS, len(coords)))
    total = rng.integers(0, 2, (N_OBS, len(coords))).astype(float)
    X = np.zeros((N_OBS, 2, len(coords)))
    X[:, 0] = rng.poisson(base * np.exp(LOG_MU[state, 0]))
    X[:, 1] = rng.binomial(total.astype(int), P_BINOM[state, 0])
    four = construct_lattice_adjacency(coords, coordination_num=4)
    adjacency = ((four + four.T) > 0).astype(float).tocsr()
    counts = get_labelled_counts(X[:, 0], X[:, 1], total, normal_spot=np.ones(len(coords)))
    base = counts.normal_baseline_sized.toarray().T
    return PlantedInputs(counts, X, base, total, coords, labels, rng.integers(0, N_CLONES, len(coords)), adjacency)


def get_calicost_res(paths: np.ndarray, labelling: np.ndarray) -> dict[str, np.ndarray]:
    """CalicoST's `res` at `PARAMETERS`, each shared by the clones: `paths` (n_obs, n_clones) concatenated along bins,
    a one-hot `log_gamma` on them, and `labelling`."""
    n_clones = paths.shape[1]
    return {
        "new_log_mu": np.repeat(LOG_MU, n_clones, 1),
        "new_alphas": np.full((3, n_clones), 0.05),
        "new_p_binom": np.repeat(P_BINOM, n_clones, 1),
        "new_taus": np.full((3, n_clones), 30.0),
        "pred_cnv": paths.T.reshape(-1),
        "log_gamma": np.where(np.arange(3)[:, None] == paths.T.reshape(-1), 0.0, -np.inf),
        "new_assignment": labelling,
    }


def get_hmrf_fit(paths: np.ndarray, labelling: np.ndarray) -> HmrfFit:
    """`get_calicost_res` as cnamaste's `HmrfFit`, unconfounded, clone-major."""
    log_gamma = np.where(paths.T[..., None] == np.arange(3), 0.0, -np.inf)
    profile = HMMProfile(log_gamma, np.ascontiguousarray(paths.T))
    assignment = CloneAssignment(new_assignment=labelling)
    return HmrfFit(PARAMETERS, get_unconfounded(paths.shape[1]), np.zeros((3, 3)), profile, np.nan, 3, assignment)


def call_calicost_pass(
    p: PlantedInputs, start: np.ndarray, adjacency: Any, weights: np.ndarray, sample_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """One `aggr_hmrf_reassignment` pass, identity `smooth_mat`, each parameter shared by the clones.

    Returns: CalicoST's labels, unary and `total_llf`.
    """
    n_spots = len(start)
    return tuple(
        calicost_hmrf.aggr_hmrf_reassignment(
            p.X,
            p.base_nb_mean,
            p.total_bb_RD,
            get_calicost_res(PATHS, start),
            PATHS,
            sp.identity(n_spots, format="csr"),
            adjacency,
            start.copy(),
            sample_ids,
            weights,
            SPATIAL_WEIGHT,
            hmmclass=calicost_hmm_nophasing,
        )
    )


def get_calicost_pass(
    referee: Referee,
    name: str,
    p: PlantedInputs,
    start: np.ndarray,
    adjacency: Any,
    log_persample_weights: np.ndarray | None = None,
    sample_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """`call_calicost_pass`, from the cache; one sample, unweighted, where `sample_ids` is None.

    Returns: CalicoST's labels, unary and `total_llf`, recorded whole.
    """
    weights = np.zeros((N_CLONES, 1)) if log_persample_weights is None else log_persample_weights
    ids = np.zeros(len(start), dtype=int) if sample_ids is None else sample_ids
    call = functools.partial(call_calicost_pass, p, start, adjacency, weights, ids)
    inputs = (p.X, p.base_nb_mean, p.total_bb_RD, start, adjacency, weights, ids)
    labels, unary, total_llf = referee(f"hmrf.pass.{name}", inputs, call, kept=(0, 1, 2))
    return labels, unary, total_llf


def get_cost(single_llf: np.ndarray, adjacency, labels: np.ndarray) -> float:
    """`calc_assignment_cost`, cnamaste's maximized energy, on `adjacency` as stored."""
    a = sp.coo_matrix(adjacency)
    return calc_assignment_cost(single_llf, a.row, a.col, a.data.astype(float), labels, SPATIAL_WEIGHT)


def get_best_move_pass(single_llf: np.ndarray, adjacency, start: np.ndarray) -> np.ndarray:
    """Each spot in index order to the label that maximizes `calc_assignment_cost`, the rest held."""
    labels = start.copy()
    for i in range(len(labels)):
        labels[i] = np.argmax(
            [get_cost(single_llf, adjacency, np.where(np.arange(len(labels)) == i, c, labels)) for c in range(N_CLONES)]
        )
    return labels


def get_calicost_knn(referee: Referee, coords: np.ndarray, maxspots_pooling: int) -> tuple[np.ndarray, np.ndarray]:
    """CalicoST's `KNN` `smooth_mat` and HMRF adjacency, dense, on spatial distance alone (`w = 1`), from the cache.

    Seeded expression counts stand in for `exp_counts`: at `w = 1` they rescale the coordinates, not their order.
    """
    counts = pd.DataFrame(np.random.default_rng(1).poisson(5, (len(coords), 30)))

    def call() -> tuple[np.ndarray, np.ndarray]:
        smooth, adjacency = choose_adjacency_by_KNN(coords, counts, w=1.0, maxspots_pooling=maxspots_pooling)
        return smooth.toarray(), adjacency.toarray()

    name = f"hmrf.knn.{get_digest(coords)[:8]}[{maxspots_pooling}]"
    smooth, adjacency = referee(name, (coords, counts, maxspots_pooling), call, kept=(0, 1))
    return smooth, adjacency


def interior(coords: np.ndarray, margin: int = 1) -> np.ndarray:
    lo, hi = coords.min(axis=0) + margin, coords.max(axis=0) - margin
    return np.flatnonzero(np.all((coords >= lo) & (coords <= hi), axis=1))


def test_an_identity_smooth_mat_makes_calicost_s_unary_get_labelling_unary_s(
    planted: PlantedInputs, referee: Referee
) -> None:
    """With no pooling, CalicoST's per-spot, per-clone unary is `get_labelling_unary`'s within 1e-11 nats. Fails where the
    RDR/BAF valid-bin weight, the path lookup or the emission drifts from CalicoST's."""
    _, theirs, _ = get_calicost_pass(referee, "start", planted, planted.start, planted.adjacency)
    ours = get_labelling_unary(planted.counts, PARAMETERS, get_unconfounded(N_CLONES), PATHS.T, phased=False)
    np.testing.assert_allclose(ours, theirs, rtol=0.0, atol=FIELD_ATOL)


def test_calicost_s_update_is_a_best_single_spot_move(planted: PlantedInputs, referee: Referee) -> None:
    """Symmetric, loop-free adjacency, same unary and index order: CalicoST's pass is each spot's best move of
    `calc_assignment_cost`, label for label, with 33 of 64 spots moved. Fails where the coupling's scale (`w / 2`
    per directed edge, both directions), its sign or the neighbour count drifts."""
    labels, single_llf, _ = get_calicost_pass(referee, "start", planted, planted.start, planted.adjacency)
    assert np.array_equal(get_best_move_pass(single_llf, planted.adjacency, planted.start), labels)
    assert np.count_nonzero(labels != planted.start) > len(labels) // 3


def test_calicost_s_knn_neighbourhood_is_the_square_8_nearest_on_interior_bins(referee: Referee) -> None:
    """Interior bins of a 9 x 9 grid: CalicoST's pooled and HMRF neighbours at `maxspots_pooling = 2`, less self-loops,
    are `construct_lattice_adjacency`'s 8 nearest, 0/1. Fails where either kNN, its ties or its unit weights drift."""
    coords = np.array([(i, j) for i in range(9) for j in range(9)], dtype=float)
    smooth, adjacency = get_calicost_knn(referee, coords, maxspots_pooling=2)
    theirs = ((smooth + adjacency) > 0).astype(float)
    np.fill_diagonal(theirs, 0.0)
    ours = construct_lattice_adjacency(coords, coordination_num=8).toarray()
    rows = interior(coords)
    assert np.array_equal(theirs[rows], ours[rows])


@departs("CalicoST's KNN sets every spot its own neighbour", raises=ValueError)
def test_calicost_s_knn_adjacency_has_no_self_loops(referee: Referee) -> None:
    """`noloop_adjacency_coo_cast` takes CalicoST's `KNN` HMRF adjacency. CalicoST's `setdiag(1)` makes a spot its
    own neighbour, which adds `spatial_weight` to its current label: an inertia cnamaste's energy refuses."""
    coords = np.array([(i, j) for i in range(9) for j in range(9)], dtype=float)
    noloop_adjacency_coo_cast(get_calicost_knn(referee, coords, maxspots_pooling=2)[1])


@departs("CalicoST's KNN HMRF neighbours exclude the pooled ones: ranks 1 + p to 6 + p")
@pytest.mark.parametrize("lattice", ["square", "triangular"])
def test_calicost_s_knn_adjacency_is_cnamaste_s_knn(lattice: str, referee: Referee) -> None:
    """Interior spots: CalicoST's `KNN` HMRF adjacency at `maxspots_pooling = 1`, less self-loops, is cnamaste's k
    nearest, 8 square or 6 triangular. CalicoST gives every spot 6, its pooled nearest removed."""
    if lattice == "square":
        coords, k = np.array([(i, j) for i in range(9) for j in range(9)], dtype=float), 8
    else:
        coords, k = np.array([(j + 0.5 * (i % 2), i * np.sqrt(3) / 2) for i in range(9) for j in range(9)]), 6
    adjacency = get_calicost_knn(referee, coords, maxspots_pooling=1)[1]
    np.fill_diagonal(adjacency, 0.0)
    rows = interior(coords)
    assert np.array_equal(adjacency[rows], construct_lattice_adjacency(coords, coordination_num=k).toarray()[rows])


@departs("CalicoST reads an asymmetric adjacency by row; cnamaste by (A + A^T) / 2")
def test_calicost_s_update_on_an_asymmetric_adjacency_is_a_best_single_spot_move(
    planted: PlantedInputs, referee: Referee
) -> None:
    """The 8-nearest adjacency, asymmetric at the perimeter (88 one-way entries): CalicoST's pass is each spot's best
    move of `calc_assignment_cost`. CalicoST counts spot i's out-edges alone; the energy, and `icm_sweep_deque`, half
    of each out- and in-edge, so 11 of 64 spots differ."""
    knn = construct_lattice_adjacency(planted.coords, coordination_num=8)
    labels, single_llf, _ = get_calicost_pass(referee, "knn", planted, planted.start, knn)
    assert np.array_equal(get_best_move_pass(single_llf, knn, planted.start), labels)


@departs("CalicoST's total_llf counts each agreeing directed edge in full, not w / 2")
def test_calicost_s_total_llf_is_the_energy(planted: PlantedInputs, referee: Referee) -> None:
    """CalicoST's `total_llf` is `calc_assignment_cost` of its labels, the negated `potts` energy. CalicoST adds
    `spatial_weight` per agreeing directed neighbour, twice the energy's coupling."""
    labels, single_llf, total_llf = get_calicost_pass(referee, "start", planted, planted.start, planted.adjacency)
    assert np.isclose(total_llf, get_cost(single_llf, planted.adjacency, labels), rtol=0.0, atol=1e-9)


@departs("CalicoST runs one pass per outer iteration; cnamaste's ICM to a fixed point")
def test_calicost_s_pass_is_a_fixed_point(planted: PlantedInputs, referee: Referee) -> None:
    """A second CalicoST pass from its first moves no spot, as `icm_sweep_deque`'s stop does. It moves 7."""
    once, _, _ = get_calicost_pass(referee, "start", planted, planted.start, planted.adjacency)
    twice, _, _ = get_calicost_pass(referee, "second", planted, once, planted.adjacency)
    assert np.array_equal(twice, once)


@departs("CalicoST visits spots in index order; icm_sweep_deque in a seeded random one")
def test_calicost_s_fixed_point_is_icm_sweep_deque_s(planted: PlantedInputs, referee: Referee) -> None:
    """CalicoST's passes, repeated to their fixed point, and `icm_sweep_deque` (no floor) from the same start reach
    the same labels. Same unary, coupling and adjacency, so the order alone separates them: 18 spots differ."""
    single, ids = np.zeros((N_CLONES, 1)), np.zeros(len(planted.start), dtype=int)

    def call() -> tuple[np.ndarray, np.ndarray]:
        labels, single_llf, _ = call_calicost_pass(planted, planted.start, planted.adjacency, single, ids)
        for _ in range(50):
            again, _, _ = call_calicost_pass(planted, labels, planted.adjacency, single, ids)
            if np.array_equal(again, labels):
                break
            labels = again
        return labels, single_llf

    inputs = (planted.X, planted.base_nb_mean, planted.total_bb_RD, planted.start, planted.adjacency, single, ids)
    labels, single_llf = referee("hmrf.fixed_point", inputs, call, kept=(0, 1))
    a = planted.adjacency
    _, _, ours = icm_sweep_deque(
        single_llf, a.indptr, a.indices, a.data, planted.start, SPATIAL_WEIGHT, None, min_clone_spots=0
    )
    assert np.array_equal(ours, labels)


@departs("CalicoST's hexagon adjacency is Gaussian, exp(-(d^2 / bw)^5); cnamaste's unit")
def test_calicost_s_hexagon_adjacency_is_the_unit_triangular_lattice(referee: Referee) -> None:
    """Interior spots of a Visium array, indices `(array_row, array_col)`: CalicoST's `hexagon` HMRF adjacency at
    `maxspots_pooling = 1` (an identity `smooth_mat`) is cnamaste's 6 nearest at unit weight. CalicoST's bandwidth
    search gives 18 neighbours weighted 0.9959, 0.3679 and 0.0148."""
    rows, cols = np.meshgrid(np.arange(12), np.arange(24), indexing="ij")
    on = (rows + cols) % 2 == 0
    array = np.column_stack([rows[on], cols[on]]).astype(float)

    def call() -> tuple[np.ndarray, np.ndarray]:
        smooth, adjacency = choose_adjacency_by_readcounts(array, np.ones((5, len(array))), maxspots_pooling=1)
        return smooth.toarray(), adjacency.toarray()

    smooth, adjacency = referee("hmrf.hexagon", (array,), call, kept=(0, 1))
    assert np.array_equal(smooth, np.eye(len(array)))
    # NB CalicoST's distance, 9 drow^2 + 3 dcol^2, is Euclidean on (3 row, sqrt(3) col).
    ours = construct_lattice_adjacency(array * [3.0, np.sqrt(3.0)], coordination_num=6).toarray()
    inner = interior(array, margin=3)
    assert np.array_equal(adjacency[inner], ours[inner])


@departs("CalicoST adds log_persample_weights to its unary; cnamaste does not (deprecated)")
def test_calicost_s_update_ignores_per_sample_weights(planted: PlantedInputs, referee: Referee) -> None:
    """Two slices of 32 spots, slice 0 weighted toward clone 0 and slice 1 toward clone 2 (log 0.8 against log 0.1):
    CalicoST's pass is the one without the weights, as cnamaste's unary carries no per-sample term."""
    without, _, _ = get_calicost_pass(referee, "start", planted, planted.start, planted.adjacency)
    weights = np.log([[0.8, 0.1], [0.1, 0.1], [0.1, 0.8]])
    sample_ids = (np.arange(len(planted.start)) >= len(planted.start) // 2).astype(int)
    weighted, _, _ = get_calicost_pass(
        referee, "weighted", planted, planted.start, planted.adjacency, weights, sample_ids
    )
    assert np.array_equal(weighted, without)


def get_calicost_pseudobulk(referee: Referee, p: PlantedInputs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CalicoST's `merge_pseudobulk_by_index` over the planted clones, recorded whole: `X`, `base_nb_mean`,
    `total_bb_RD`, one column per clone."""
    clone_index = [np.flatnonzero(p.planted == c) for c in range(N_CLONES)]
    call = functools.partial(calicost_hmrf.merge_pseudobulk_by_index, p.X, p.base_nb_mean, p.total_bb_RD, clone_index)
    X, base, total = referee("hmrf.pseudobulk", (p.X, p.base_nb_mean, p.total_bb_RD, p.planted), call, kept=(0, 1, 2))
    return X, base, total


def test_merge_pseudobulk_by_index_is_get_pseudobulk(planted: PlantedInputs, referee: Referee) -> None:
    """The planted clones: CalicoST's `merge_pseudobulk_by_index` (utils_hmrf.py:254) UMIs, baseline, B-allele and
    total SNP UMIs are `get_pseudobulk`'s, bitwise. Fails where a channel, the baseline's sum order or the clone
    order drifts."""
    X, base, total = get_calicost_pseudobulk(referee, planted)
    ours = planted.counts.with_labelling("clone", planted.planted).get_pseudobulk("clone")
    for theirs, found in zip((X[:, 0], base, X[:, 1], total), ours[:4], strict=True):
        assert np.array_equal(np.asarray(found).T, theirs)


MIN_SPOTS = 20
"""The spot floor (`hmrf.min_spots_per_clone`): over clone 2's 16 planted spots, under clones 0 and 1's 24."""


def test_merge_by_minspots_is_get_merged_fit(planted: PlantedInputs, referee: Referee) -> None:
    """Clone 2 under the floor: CalicoST's `merge_by_minspots` (hmrf.py:248) groups are `get_merged_labelling`'s,
    its labels and kept paths `get_merged_fit`'s, bitwise. Fails where the floor's comparison, the receiving clone
    (most SNP UMIs) or the path a group keeps drifts."""
    res = get_calicost_res(PATHS, planted.planted)

    def call() -> tuple[list[list[int]], np.ndarray, np.ndarray]:
        with mock.patch.object(np, "NAN", np.nan, create=True):
            groups, merged = calicost_hmrf.merge_by_minspots(planted.planted, res, planted.total_bb_RD, MIN_SPOTS, 0)
        return [[int(c) for c in group] for group in groups], merged["new_assignment"], merged["pred_cnv"]

    inputs = (planted.planted, PATHS, planted.total_bb_RD, MIN_SPOTS)
    groups, labels, pred_cnv = referee("hmrf.merge_by_minspots.floor", inputs, call, kept=(0, 1, 2))
    _, ours_groups = get_merged_labelling(planted.planted, planted.total_bb_RD.T, MIN_SPOTS, 0)
    assert groups == [[int(c) for c in group] for group in ours_groups]
    assert any(2 in group and len(group) == 2 for group in groups), "clone 2 joins a passing clone"
    ours = get_merged_fit(get_hmrf_fit(PATHS, planted.planted), planted.total_bb_RD.T, MIN_SPOTS, 0)
    assert np.array_equal(ours["new_assignment"], labels)
    assert np.array_equal(ours["pred_cnv"].reshape(-1), pred_cnv)


def get_reindexed_labels(planted: PlantedInputs, referee: Referee) -> tuple[np.ndarray, np.ndarray]:
    """Clone 1 given the neutral path (state 1, BAF 0.5): `get_reindexed_fit`'s labels and CalicoST's
    `reorder_results`', from the cache."""
    paths = PATHS.copy()
    paths[:, 1] = 1
    res = get_calicost_res(paths, planted.planted)
    log_gamma = res["log_gamma"].reshape(3, N_CLONES, N_OBS).transpose(0, 2, 1)

    def call() -> np.ndarray:
        posterior = np.zeros((len(planted.planted), N_CLONES))
        reordered, _ = calicost_hmrf.reorder_results(res | {"pred_cnv": paths, "log_gamma": log_gamma}, posterior, None)
        return reordered["new_assignment"]

    (theirs,) = referee("hmrf.reorder", (planted.planted, paths), call, kept=(0,))
    return get_reindexed_fit(get_hmrf_fit(paths, planted.planted))["new_assignment"], theirs


def test_reorder_results_normal_clone_is_get_reindexed_fit_s(planted: PlantedInputs, referee: Referee) -> None:
    """CalicoST's `reorder_results` (utils_hmrf.py:345) and `get_reindexed_fit` make the planted clone 1, the one
    on the neutral path, clone 0, spot for spot. Fails where the BAF deadband (0.05) or the stray sum drifts."""
    ours, theirs = get_reindexed_labels(planted, referee)
    assert np.array_equal(theirs == 0, planted.planted == 1)
    assert np.array_equal(ours == 0, theirs == 0)


@departs("CalicoST keeps the clones after the normal one in index order; cnamaste sorts them by spot count")
def test_reorder_results_is_get_reindexed_fit(planted: PlantedInputs, referee: Referee) -> None:
    """Every clone's new index, CalicoST's against `get_reindexed_fit`'s. After the normal clone CalicoST keeps 0
    then 2; cnamaste puts 2 (16 spots) before 0 (24, labelling.py:80)."""
    ours, theirs = get_reindexed_labels(planted, referee)
    assert np.array_equal(ours, theirs)


def get_calicost_posterior_unary(referee: Referee, p: PlantedInputs, variant: str, path_mass: float) -> np.ndarray:
    """CalicoST's `weighted_sum` unary, `hmrf_reassignment_<variant>`, identity `smooth_mat`: each clone's `log_gamma`
    `path_mass` on its `PATHS` state, the rest shared by the other two, concatenated along bins where `variant` is
    `posterior_concatenate`. Recorded whole."""
    with np.errstate(divide="ignore"):
        log_gamma = np.where(np.arange(3)[:, None, None] == PATHS, np.log(path_mass), np.log((1.0 - path_mass) / 2))
    if variant == "posterior_concatenate":
        log_gamma = log_gamma.transpose(0, 2, 1).reshape(3, -1)
    res = get_calicost_res(PATHS, p.start) | {"log_gamma": log_gamma}
    reassign = getattr(calicost_hmrf, f"hmrf_reassignment_{variant}")
    args = (sp.identity(len(p.start), format="csr"), p.adjacency, p.start.copy(), np.zeros(len(p.start), dtype=int))
    weights = np.zeros((N_CLONES, 1))

    def call() -> np.ndarray:
        found = reassign(
            p.X, p.base_nb_mean, p.total_bb_RD, res, *args, weights, SPATIAL_WEIGHT, calicost_hmm_nophasing
        )
        return found[1]

    inputs = (p.X, p.base_nb_mean, p.total_bb_RD, p.start, p.adjacency, log_gamma)
    (single_llf,) = referee(f"hmrf.{variant}[{path_mass}]", inputs, call, kept=(0,))
    return single_llf


@pytest.mark.parametrize("variant", ["posterior_concatenate", "posterior"])
def test_the_weighted_sum_unary_on_a_one_hot_posterior_is_get_labelling_unary_s(
    variant: str, planted: PlantedInputs, referee: Referee
) -> None:
    """The shipped `nodepotential: weighted_sum`, `hmrf_reassignment_posterior_concatenate` (hmrf.py:118) and the final
    `hmrf_reassignment_posterior` (hmrf.py:29), identity `smooth_mat`, `log_gamma` one-hot on `PATHS`: CalicoST's
    unary is `get_labelling_unary`'s within 1e-11 nats. Fails where the valid-bin weight or the path term drifts."""
    theirs = get_calicost_posterior_unary(referee, planted, variant, path_mass=1.0)
    ours = get_labelling_unary(planted.counts, PARAMETERS, get_unconfounded(N_CLONES), PATHS.T, phased=False)
    np.testing.assert_allclose(ours, theirs, rtol=0.0, atol=FIELD_ATOL)


@departs("CalicoST's weighted_sum scores the posterior's expectation over states; cnamaste the decoded path")
def test_the_weighted_sum_unary_on_a_soft_posterior_is_get_labelling_unary_s(
    planted: PlantedInputs, referee: Referee
) -> None:
    """`hmrf_reassignment_posterior_concatenate`, `log_gamma` 0.9 on `PATHS`, 0.05 on each other state: CalicoST's
    unary, a logsumexp over states per bin, is `get_labelling_unary`'s path unary within 1e-11 nats."""
    theirs = get_calicost_posterior_unary(referee, planted, "posterior_concatenate", path_mass=0.9)
    ours = get_labelling_unary(planted.counts, PARAMETERS, get_unconfounded(N_CLONES), PATHS.T, phased=False)
    np.testing.assert_allclose(ours, theirs, rtol=0.0, atol=FIELD_ATOL)


MERGE_SPATIAL_WEIGHT = 3.0
"""The merge rows' coupling: the energy folds clones 0 and 1 alone on a shared path (at 10 clone 1 joins 2), and none
on 5 differing bins (at 5 it folds 0 and 1)."""


def get_merges(referee: Referee, p: PlantedInputs, shared_from: int) -> tuple[np.ndarray, np.ndarray]:
    """Clone 1 given clone 0's path from bin `shared_from` on: `merge_clones`' labels at `MERGE_SPATIAL_WEIGHT` and
    CalicoST's `similarity_components_rdrbaf_neymanpearson` labels, at its shipped `np_threshold` 1 and
    `np_eventminlen` 10, on `merge_pseudobulk_by_index`'s clones, from the cache."""
    paths = PATHS.copy()
    paths[shared_from:, 1] = paths[shared_from:, 0]
    field = get_labelling_unary(p.counts, PARAMETERS, get_unconfounded(N_CLONES), paths.T, phased=False)
    a = p.adjacency
    ours, _, _ = merge_clones(field, a.indptr, a.indices, a.data, p.planted, MERGE_SPATIAL_WEIGHT)
    X, base, total = get_calicost_pseudobulk(referee, p)
    res = get_calicost_res(paths, p.planted)

    def call() -> np.ndarray:
        # NB as calicost_main.py:227 (c1abcae) calls it, `tumor_prop` None; CalicoST writes `np.NAN`.
        with mock.patch.object(np, "NAN", np.nan, create=True):
            _, merged = similarity_components_rdrbaf_neymanpearson(
                X, base, total, res, threshold=1.0, minlength=10, params="smp", hmmclass=hmm_nophasing_v2
            )
        return merged["new_assignment"]

    (theirs,) = referee(f"hmrf.neymanpearson[{shared_from}]", (X, base, total, p.planted, paths), call, kept=(0,))
    return ours, theirs


def get_partition(labels: np.ndarray) -> np.ndarray:
    """`labels` renumbered in order of first appearance, so equal partitions compare equal."""
    _, first, inverse = np.unique(labels, return_index=True, return_inverse=True)
    return np.argsort(np.argsort(first))[inverse]


def test_the_neyman_pearson_merge_on_a_shared_path_is_merge_clones(planted: PlantedInputs, referee: Referee) -> None:
    """Clones 0 and 1 on one path: CalicoST's Neyman-Pearson merge, with no differing bin to test, and the energy merge
    both fold clones 0 and 1 and keep clone 2, spot for spot. Fails where either stops folding identical clones or
    folds a distinct one."""
    ours, theirs = get_merges(referee, planted, shared_from=0)
    assert np.array_equal(get_partition(theirs), get_partition(np.where(planted.planted == 2, 1, 0)))
    assert np.array_equal(get_partition(ours), get_partition(theirs))


@departs("CalicoST's Neyman-Pearson merge ignores events under np_eventminlen bins; the energy counts every bin")
def test_the_neyman_pearson_merge_on_a_short_event_is_merge_clones(planted: PlantedInputs, referee: Referee) -> None:
    """Clones 0 and 1 apart on 5 bins: CalicoST folds them, as no event reaches 10 bins; the energy keeps all three."""
    ours, theirs = get_merges(referee, planted, shared_from=5)
    assert np.array_equal(get_partition(ours), get_partition(theirs))


def test_calicost_s_multislice_knn_neighbourhood_is_construct_multislice_lattice_adjacency_s(referee: Referee) -> None:
    """Two 9 x 9 square slices on one grid: CalicoST's `multislice_adjacency`, `KNN` at `maxspots_pooling = 2`, pooled
    and HMRF neighbours together less self-loops, is `construct_multislice_lattice_adjacency`'s 8 nearest on each
    slice's interior bins, and neither joins the slices. Fails where a slice's spots, its block or a cross-slice entry
    drift."""
    grid = np.array([(i, j) for i in range(9) for j in range(9)], dtype=float)
    coords, sample_ids = np.vstack([grid, grid]), np.repeat([0, 1], len(grid))
    counts = pd.DataFrame(np.random.default_rng(1).poisson(5, (len(coords), 30)))

    def call() -> tuple[np.ndarray, np.ndarray]:
        # NB CalicoST reads `csr_matrix.A`, which scipy removed.
        with mock.patch.object(sp.csr_matrix, "A", property(lambda m: m.toarray()), create=True):
            adjacency, smooth = calicost_multislice_adjacency(
                sample_ids, ["a", "b"], coords, None, counts, None, "KNN", 2, 1.0
            )
        return smooth.toarray(), adjacency.toarray()

    smooth, adjacency = referee("hmrf.multislice_knn", (coords, sample_ids, counts), call, kept=(0, 1))
    theirs = ((smooth + adjacency) > 0).astype(float)
    np.fill_diagonal(theirs, 0.0)
    ours = construct_multislice_lattice_adjacency(sample_ids, ["a", "b"], coords, None, "square", 8).toarray()
    across = sample_ids[:, None] != sample_ids[None, :]
    assert not theirs[across].any() and not ours[across].any()
    rows = np.concatenate([interior(grid), len(grid) + interior(grid)])
    assert np.array_equal(theirs[rows], ours[rows])
