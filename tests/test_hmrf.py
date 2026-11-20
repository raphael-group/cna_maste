"""`hmrf.get_hmrf_labelling` on synthetic per-clone emissions with planted lattice labels,
`hmrf.run_hmrf` on counts drawn at planted clones and copy states, and `hmrf.get_merged_fit` against its
thresholds.

`get_hmrf_labelling`'s input, seeded: 32 x 32 bins, the run's 8 nearest; quadrants planted with clones 0-3,
quadrant 3's right half clone 4 on clone 3's path (one genotype, two labels), and a 5 x 5 clone 5 under the floor of
50. 40 bins, read depth and allele counts drawn from each spot's clone's path, the start the planted
labels with 30% of spots redrawn.

`run_hmrf`'s input, seeded: 24 x 24 bins, the run's 8 nearest; quadrants planted with clones 0-3 and a 6 x 6
clone 4 under the floor. 60 bins on 2 contigs, 5 planted copy states (`CNA_RDR`, `CNA_BAF`); read depth NB at each
spot's library of 600-1,200 UMIs, its clone's `Z_c` normalized, allele counts BB of Poisson(1.5) totals (`5447dde0`, the
SHA-256 of `X`, totals and baseline). The start is the quadrants shifted 3 bins, start clone 3 split in two by rows,
and clone 4: six clones. Each stage as the run calls it (run_cnamaste.py:595, 972),
at the shipped `config.yaml` with `opengm_expansion` and the floor at 50.

The referees are the planted labels and states (ARI, tolerances), and `potts.potts_energy`. The file runs in 8 s
serially (-n 0), 5.4 s of it `run_hmrf`'s 9 rows.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.metrics import adjusted_rand_score

from cnamaste import hmrf, mstep
from cnamaste.config import YAMLConfig, get_global_config, get_label_starts, set_global_config
from cnamaste.emission import get_unconfounded
from cnamaste.fit_checks import get_fit_check_counts, reset_fit_check_counts
from cnamaste.hmm_params import CnaHmmParameters
from cnamaste.hmm_utils import get_log_transmat
from cnamaste.hmrf import (
    field_argmax,
    get_hmrf_labelling,
    get_labelling_turnover,
    get_merged_fit,
    get_reindexed_history,
    run_hmrf,
)
from cnamaste.hmrf_fit import CloneAssignment, HMMProfile, HmrfFit
from cnamaste.initialize.genomic import get_initial_parameters
from cnamaste.labelling import get_merged_labelling
from cnamaste.potts import potts_energy
from cnamaste.spatial import construct_lattice_adjacency
from cnamaste.spatio_genomic_counts import SpatioGenomicLabelledMultiCounts
from labelled_counts import get_labelled_counts
from potts_reference import get_grid_coords

SIDE, N_OBS, FLOOR, SPATIAL_WEIGHT = 32, 40, 50, 1.0
GENOTYPE = np.array([0, 1, 2, 3, 3, 4])
"""Each label's genotype: clones 3 and 4 share clone 3's path."""

PATHS = np.stack([np.repeat(p, 10) for p in ([0, 1, 2, 1], [1, 1, 0, 2], [2, 0, 1, 1], [1, 2, 2, 0], [0, 0, 2, 2])])
"""Each genotype's decoded state per bin."""

PARAMETERS = CnaHmmParameters(
    None, np.log([0.5, 1.0, 1.5]), np.array([0.1, 0.5, 0.3]), np.full(3, 0.05), np.full(3, 30.0)
)


def get_result(n_clones: int, pred_cnv: np.ndarray, log_gamma: np.ndarray) -> HmrfFit:
    """A fit at `PARAMETERS`, unconfounded, with its decoded profile."""
    return HmrfFit(
        PARAMETERS,
        get_unconfounded(n_clones),
        get_log_transmat(3, 1 - 1e-4),
        HMMProfile(log_gamma, pred_cnv),
        np.nan,
        3,
        CloneAssignment(),
    )


RES = get_result(len(GENOTYPE), PATHS[GENOTYPE], np.zeros((len(GENOTYPE), N_OBS, 3)))


class Planted(NamedTuple):
    counts: SpatioGenomicLabelledMultiCounts
    """The RDR stage's, its baseline sized by the normal spots."""
    adjacency: sp.csr_matrix
    planted: np.ndarray
    start: np.ndarray


@pytest.fixture(scope="module")
def planted() -> Planted:
    rng = np.random.default_rng(0)
    coords = get_grid_coords(SIDE, SIDE)
    labels = (coords[:, 0] >= SIDE // 2).astype(np.int64) * 2 + (coords[:, 1] >= SIDE // 2)
    labels[(labels == 3) & (coords[:, 1] >= 3 * SIDE // 4)] = 4
    labels[(coords[:, 0] >= 5) & (coords[:, 0] < 10) & (coords[:, 1] >= 5) & (coords[:, 1] < 10)] = 5
    state = PATHS[GENOTYPE[labels]].T
    base = rng.uniform(1.0, 3.0, (N_OBS, len(coords)))
    total = rng.integers(0, 4, (N_OBS, len(coords))).astype(float)
    X = np.zeros((N_OBS, 2, len(coords)))
    X[:, 0] = rng.poisson(base * np.exp(PARAMETERS.log_mu[state]))
    X[:, 1] = rng.binomial(total.astype(int), PARAMETERS.p_binom[state])
    start = np.where(rng.random(len(coords)) < 0.3, rng.integers(0, len(GENOTYPE), len(coords)), labels)
    counts = get_labelled_counts(X[:, 0], X[:, 1], total, normal_spot=np.ones(len(coords)))
    return Planted(counts, construct_lattice_adjacency(coords, coordination_num=8), labels, start)


@pytest.fixture(params=["icm", "opengm_expansion"])
def solver(request: pytest.FixtureRequest) -> Iterator[str]:
    set_global_config(get_labelling_config(request.param))
    yield request.param


def get_labelling_config(solver: str) -> YAMLConfig:
    """The labelling's configuration: `solver`, the floor at `FLOOR` and `SPATIAL_WEIGHT`."""
    hmrf_config = {"label_solver": solver, "min_spots_per_clone": FLOOR, "fixed_assignment": False}
    return YAMLConfig({"hmrf": hmrf_config | {"spatial_weight": SPATIAL_WEIGHT}})


def get_assigned(p: Planted, merge: bool) -> tuple[np.ndarray, np.ndarray, float]:
    """`get_hmrf_labelling`'s labels, field and reported log-likelihood."""
    return get_hmrf_labelling(p.counts, RES, p.adjacency, p.start, get_global_config(), merge=merge, index=0)


def test_the_pipeline_recovers_the_planted_genotypes(planted: Planted, solver: str) -> None:
    """With the merge, outside the 25-spot clone: ARI 1 against the planted genotype, clones 3 and 4 one label. Fails
    where the field, the solver dispatch or the merge drifts."""
    merged, _, _ = get_assigned(planted, merge=True)
    genotype = GENOTYPE[planted.planted]
    kept = genotype != 4
    assert adjusted_rand_score(genotype[kept], merged[kept]) == 1.0


def test_the_merge_folds_one_genotype_and_lowers_the_energy(planted: Planted) -> None:
    """ICM holds clones 3 and 4, of one path, apart (142 and 118 spots); the merge folds them, and `potts_energy` on the
    field the pipeline returns falls. Expansion folds them itself, so ICM alone. Fails where the merge misses a pair
    that lowers the energy, or takes one that raises it."""
    set_global_config(get_labelling_config("icm"))
    unmerged, field, _ = get_assigned(planted, merge=False)
    merged, _, _ = get_assigned(planted, merge=True)
    assert np.count_nonzero(np.bincount(unmerged)[3:5]) == 2 and np.count_nonzero(np.bincount(merged)[3:5]) == 1
    energy = [potts_energy(field, planted.adjacency, labels, SPATIAL_WEIGHT) for labels in (merged, unmerged)]
    assert energy[0] < energy[1]


def test_the_floor_leaves_no_clone_under_it(planted: Planted, solver: str, request: pytest.FixtureRequest) -> None:
    """At `min_spots_per_clone` 50 the planted 25-spot clone 5 is gone, and no occupied clone is under 50, also
    under expansion, which empties clone 3. Fails where an empty clone switches the floor off (T- #22)."""
    counts = np.bincount(get_assigned(planted, merge=False)[0], minlength=len(GENOTYPE))
    assert counts[5] == 0 and counts[counts > 0].min() >= FLOOR


def test_each_solver_starts_from_hmrf_label_start(
    planted: Planted, solver: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The solver is handed `labelling` at `hmrf.label_start` prev_assignment (the default) and `field_argmax` of the
    field at field_argmax, which differ on the planted start. Fails where a solver keeps the previous labelling at
    field_argmax, or the default start moves."""
    starts = []

    def record_icm(**kwargs: Any) -> tuple[int, float, np.ndarray]:
        starts.append(kwargs["new_assignment"].copy())
        return 0, 0.0, kwargs["new_assignment"]

    def record_external(
        field: np.ndarray, adjacency: Any, labels: np.ndarray, *args: Any, **kwargs: Any
    ) -> tuple[int, float]:
        starts.append(labels.copy())
        return 0, 0.0

    monkeypatch.setattr(hmrf, "icm_sweep_deque", record_icm)
    monkeypatch.setattr(hmrf, "external_assignment", record_external)
    config = get_labelling_config(solver)
    assert config.hmrf.label_start == get_label_starts()[0]
    for start in get_label_starts():
        config.hmrf.label_start = start
        set_global_config(config)
        field = get_assigned(planted, merge=False)[1]
    assert np.array_equal(starts[0], planted.start) and np.array_equal(starts[1], field_argmax(field))
    assert not np.array_equal(starts[0], starts[1])


@pytest.mark.xfail(strict=True, reason="T- #21: the reported llf counts upper-triangle pairs, unweighted, without /2")
def test_the_reported_llf_is_the_negated_energy(planted: Planted, solver: str) -> None:
    """On the 8 nearest of the bins jittered by up to 0.2 pitch (648 one-way entries), the log-likelihood
    `get_hmrf_labelling` reports is `-potts_energy` of its labels. It adds `spatial_weight` per aligned stored
    pair with row < col, unweighted, where the energy adds `w / 2` per aligned stored edge, both directions."""
    coords = get_grid_coords(SIDE, SIDE) + np.random.default_rng(5).uniform(-0.2, 0.2, (SIDE * SIDE, 2))
    adjacency = construct_lattice_adjacency(coords, coordination_num=8)
    labels, field, reported = get_hmrf_labelling(
        planted.counts, RES, adjacency, planted.start, get_global_config(), merge=False, index=0
    )
    assert np.isclose(reported, -potts_energy(field, adjacency, labels, SPATIAL_WEIGHT), rtol=0.0, atol=1e-9)


def test_the_merged_fit_folds_clones_under_either_threshold() -> None:
    """Four clones of 60, 60, 30 and 60 spots, clone 3 with 10 SNP UMIs per spot against 5 elsewhere: at 50 spots and
    500 UMIs, clone 2 (too few spots) and clone 1 (300 UMIs) fold into clone 3, the one with the most UMIs, and clone
    0 (300 UMIs) too; at 50 spots and 0 UMIs, clone 2 alone folds, into clone 3. Fails where a threshold's comparison,
    the UMI sum or the destination drifts."""
    assignment = np.repeat([0, 1, 2, 3], [60, 60, 30, 60])
    total = np.ones((assignment.size, 5))
    total[assignment == 3] = 2.0
    n_obs = 4
    hmrf_result = get_result(4, np.arange(4 * n_obs).reshape(4, n_obs), np.zeros((4, n_obs, 3)))
    hmrf_result["new_assignment"] = assignment
    assert get_merged_labelling(assignment, total, 50, 500)[1] == [[3, 0, 1, 2]]
    merged = get_merged_fit(hmrf_result, total, 50, 500)
    assert np.array_equal(merged["new_assignment"], np.zeros(assignment.size)) and np.isnan(merged["total_llf"])
    assert get_merged_labelling(assignment, total, 50, 0)[1] == [[0], [1], [3, 2]]
    merged = get_merged_fit(hmrf_result, total, 50, 0)
    assert np.array_equal(merged["pred_cnv"], hmrf_result["pred_cnv"][[0, 1, 3]])


CNA_SIDE, CNA_OBS = 24, 60
CNA_RDR, CNA_BAF = np.array([1.0, 0.5, 1.5, 1.5, 2.0]), np.array([0.5, 0.05, 1 / 3, 0.5, 0.25])
"""Each planted copy state's read depth ratio and BAF: neutral, a loss, two gains of one ratio and a gain of
two. The fit takes `hmm.n_states` (7) states, as the run does."""

CNA_PATHS = np.zeros((5, CNA_OBS), dtype=np.int64)
for clone, segments in enumerate(
    [[], [(0, 20, 1), (40, 50, 3)], [(10, 30, 2), (50, 60, 1)], [(0, 10, 3), (20, 40, 4)], [(50, 60, 2)]]
):
    for first, last, cna in segments:
        CNA_PATHS[clone, first:last] = cna
"""Each planted clone's copy state per bin; clone 0 is normal, clone 4 the one under the floor."""

ARI_FLOOR, RDR_RTOL, BAF_ATOL = 0.99, 0.055, 0.03
"""Measured here: ARI 1.0 at both stages; depth ratio within 0.053 (argmax) and 0.049 (posterior), BAF within 0.017;
the baseline is the normal clone's pooled profile, about 2.4% noise per bin."""


class Step(NamedTuple):
    merge: bool
    prev: np.ndarray
    labels: np.ndarray
    field: np.ndarray
    unmerged: np.ndarray
    """The labels without the merge: those `get_hmrf_labelling` returns at `merge=False` on the same input."""


class Inference(NamedTuple):
    params: str
    logmu_shift_type: str
    """`hmrf.logmu_shift_type`."""
    res: Any
    steps: list[Step]
    llf: list[float]
    """`baum_welch`'s llf per outer iteration, then the final fit's."""
    m_steps: list[tuple[float, float, float]]
    """Each minimize without a callback E step: its cost at the start, before and after, and at the end."""


def get_cna_planted() -> Planted:
    """The planted problem; the counts the RDR stage's, whose umis and baseline the BAF stage zeroes."""
    rng = np.random.default_rng(1)
    coords = get_grid_coords(CNA_SIDE, CNA_SIDE)
    half, row, col = CNA_SIDE // 2, coords[:, 0], coords[:, 1]
    small = (row >= 3) & (row < 9) & (col >= 3) & (col < 9)
    labels = np.where(small, 4, (row >= half).astype(np.int64) * 2 + (col >= half))
    start = (row >= half + 3).astype(np.int64) * 2 + (col >= half - 3)
    start[(start == 3) & (row >= 3 * CNA_SIDE // 4)] = 4
    start[small] = 5
    state = CNA_PATHS[labels].T
    library = rng.dirichlet(np.full(CNA_OBS, 50.0))
    base = library[:, None] * rng.integers(600, 1200, len(coords))[None, :]
    # NB each clone's depth at its library normalizer Z_c, as `apply_logmu_shift` models it.
    z_c = (library[:, None] * CNA_RDR[CNA_PATHS.T]).sum(axis=0)
    mean, alpha = base * CNA_RDR[state] / z_c[labels], 0.02
    X = np.zeros((CNA_OBS, 2, len(coords)))
    X[:, 0] = rng.negative_binomial(1 / alpha, 1 / (1 + alpha * mean))
    total = rng.poisson(1.5, (CNA_OBS, len(coords))).astype(float)
    tau = 50.0
    X[:, 1] = rng.binomial(total.astype(int), rng.beta(CNA_BAF[state] * tau, (1 - CNA_BAF[state]) * tau))
    counts = get_labelled_counts(X[:, 0], X[:, 1], total, np.array([30, 30]), normal_spot=labels == 0)
    return Planted(counts, construct_lattice_adjacency(coords, coordination_num=8), labels, start)


def get_inference(
    p: Planted,
    params: str,
    solver: str,
    max_iter_outer: int = 0,
    logmu_shift_type: str = "argmax",
    max_iter_em: int = 0,
) -> Inference:
    """`run_hmrf` as the run calls it at `params`, each outer iteration's labelling recorded.

    Args: `max_iter_outer` 0 for the shipped configuration's (25); `logmu_shift_type` `hmrf.logmu_shift_type`;
        `max_iter_em` 0 for the shipped configuration's (25).
    """
    config = YAMLConfig.from_file(Path(__file__).parents[1] / "config.yaml")
    config.hmrf.label_solver, config.hmrf.min_spots_per_clone = solver, FLOOR
    config.hmrf.spatial_weight, config.hmrf.merge_clones_baf, config.hmrf.merge_clones_rdr = SPATIAL_WEIGHT, True, True
    config.hmrf.max_iter_outer = max_iter_outer or config.hmrf.max_iter_outer
    config.hmrf.logmu_shift_type = logmu_shift_type
    config.hmm.max_iter_em = max_iter_em or config.hmm.max_iter_em
    set_global_config(config)
    steps, llf = [], []
    m_steps: list[tuple[float, float, float]] = []

    def assign(*args: Any, **kwargs: Any) -> tuple[np.ndarray, np.ndarray, float]:
        unmerged = get_hmrf_labelling(*args, **{**kwargs, "merge": False})[0]
        out = get_hmrf_labelling(*args, **kwargs)
        steps.append(Step(kwargs["merge"], args[3].copy(), out[0].copy(), out[1], unmerged))
        return out

    def fit(*args: Any, **kwargs: Any) -> Any:
        hmm_fit_result = hmm_fit(*args, **kwargs)
        llf.append(float(hmm_fit_result.llf))
        return hmm_fit_result

    def minimize(fun: Any, x0: np.ndarray, config: Any, **kwargs: Any) -> Any:
        if kwargs.get("callback") is not None:
            return mstep_minimum(fun, x0, config, **kwargs)
        # NB the M step's cost returns its gradient with it (`jac=True`).
        start = float(fun(x0)[0])
        res = mstep_minimum(fun, x0, config, **kwargs)
        m_steps.append((start, float(fun(x0)[0]), float(res.fun)))
        return res

    hmm_fit, mstep_minimum = hmrf.baum_welch, mstep.get_mstep_minimum
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(hmrf, "get_hmrf_labelling", assign)
        patch.setattr(hmrf, "baum_welch", fit)
        patch.setattr(mstep, "get_mstep_minimum", minimize)
        res = get_core_inference(p, params)
    return Inference(params, logmu_shift_type, res, steps, llf, m_steps)


def get_core_inference(p: Planted, params: str) -> Any:
    """`run_hmrf` as run_cnamaste calls it at `sp` (read depth zeroed, stage `baf`) or `smp` (stage `rdr`), at the
    global config."""
    config = get_global_config()
    stage = "baf" if params == "sp" else "rdr"
    counts = p.counts
    if stage == "baf":
        counts = counts.with_umis(counts.umis.zeros(counts.umis.dtype))._replace(labellings={})
    initial_parameters = get_initial_parameters(counts, p.start, config, stage=stage)
    return run_hmrf(counts, p.start, p.adjacency, config, parameters=initial_parameters, stage=stage)


@pytest.fixture(scope="module")
def cna_planted() -> Planted:
    return get_cna_planted()


@pytest.fixture(
    scope="module",
    params=["sp", "smp", "smp-posterior"],
)
def inferred(request: pytest.FixtureRequest, cna_planted: Planted) -> Inference:
    params, _, logmu_shift_type = request.param.partition("-")
    return get_inference(cna_planted, params, "opengm_expansion", logmu_shift_type=logmu_shift_type or "argmax")


CORE = pytest.mark.xdist_group("run_hmrf")


@CORE
def test_run_hmrf_recovers_the_planted_clones(inferred: Inference, cna_planted: Planted) -> None:
    """ARI >= 0.99 against the planted clones outside the 36-spot clone 4, at `sp` and `smp`. Fails where the fit, the
    field or the labelling drifts from the planted clones."""
    kept = cna_planted.planted != 4
    assert adjusted_rand_score(cna_planted.planted[kept], inferred.res["new_assignment"][kept]) >= ARI_FLOOR


@CORE
def test_run_hmrf_recovers_the_planted_states(inferred: Inference, cna_planted: Planted) -> None:
    """Per clone, each bin its planted spots share a state at: decoded BAF within 0.03 of the planted; at `smp` also its
    depth ratio to the clone's planted-neutral bins within 5%. Clone 4's 36 spots, under the floor, pool into clone 0
    and admix its bins 50-59, which are not asserted. Not the states' partition: 7 states fit 5, and a clone's neutral bins take two of
    near equal parameters. Fails where a segment is lost, two states merge, or `Z_c` mis-scales a clone."""
    res, labels = inferred.res, inferred.res["new_assignment"]
    pred = np.asarray(res["pred_cnv"])
    mu, baf = np.exp(res.parameters.log_mu), res.parameters.p_binom
    for c in np.unique(labels):
        paths = CNA_PATHS[np.unique(cna_planted.planted[labels == c])]
        shared, path = (paths == paths[0]).all(axis=0), paths[0]
        decoded = pred[c]
        assert np.abs(baf[decoded] - CNA_BAF[path])[shared].max() <= BAF_ATOL
        if inferred.params == "sp":
            continue
        ratio = mu[decoded] / np.median(mu[decoded][shared & (path == 0)])
        assert np.abs(ratio / CNA_RDR[path] - 1.0)[shared].max() <= RDR_RTOL


@CORE
def test_the_outer_loop_stops_at_the_ari_tolerance(inferred: Inference) -> None:
    """At `max_iter_outer` 25, the first iteration whose ARI to the last labelling reaches
    `hmrf.assignment_ari_tolerance` (1.0) is the last unmerged one, and two merging iterations follow: the GOTO to
    `max_iter_outer - 1`. Fails where the tolerance does not stop the loop, or the merge rounds are skipped or
    repeated."""
    tolerance = get_global_config().hmrf.assignment_ari_tolerance
    ari = [adjusted_rand_score(s.prev, s.labels) for s in inferred.steps]
    merges = [s.merge for s in inferred.steps]
    stop = next(i for i, a in enumerate(ari) if a >= tolerance)
    assert merges == [False] * (stop + 1) + [True, True] and len(merges) < 25 - 2
    assert len(inferred.llf) == len(merges) + 1


@CORE
def test_the_outer_loop_stops_at_max_iter_outer(cna_planted: Planted) -> None:
    """At `max_iter_outer` 3 and `sp`, iteration 0 is under the tolerance (ARI 0.40 to the start) and is the last
    unmerged one, at `max_iter_outer - 2`; two merging iterations follow, three in all. Fails where the cap is not
    `max_iter_outer` iterations, or the tolerance is taken for it."""
    run = get_inference(cna_planted, "sp", "opengm_expansion", max_iter_outer=3)
    assert [s.merge for s in run.steps] == [False, True, True]
    assert (
        adjusted_rand_score(run.steps[0].prev, run.steps[0].labels) < get_global_config().hmrf.assignment_ari_tolerance
    )


def test_the_fit_checks_count_the_emptied_start_clone_alone(cna_planted: Planted) -> None:
    """At `max_iter_outer` 3 and `sp`, the start's clone 4 (36 spots, under the floor of 50) empties on iteration 0:
    `emptied_clone` counts it, no labelling holds a clone under the floor and every llf is finite. Fails where a
    check misses the emptied clone or fires on a labelling the floor holds."""
    reset_fit_check_counts()
    get_inference(cna_planted, "sp", "opengm_expansion", max_iter_outer=3)
    counts = get_fit_check_counts()
    assert counts["emptied_clone"] >= 1
    assert counts["labelling_below_min_spots"] == 0 and counts["non_finite_llf"] == 0


def test_the_turnover_counts_a_return_to_an_earlier_label_across_a_reindex() -> None:
    """Clone 1 empties and clone 2 becomes 1. Spot 0 goes 0 -> 2 -> 0 and returns; spot 1 goes 2 -> 0 and spot 2
    leaves the emptied clone, neither returns; spot 3 holds clone 2. Fails where the re-index breaks a return or a
    hold."""
    history = get_reindexed_history(np.array([[0, 2, 1, 2], [2, 2, 1, 2]]), np.array([0, 2]))
    turnover = get_labelling_turnover(history, np.array([0, 0, 0, 1]))
    assert turnover.changed == 3 and turnover.returned.tolist() == [True, False, False, False]
    assert turnover.stable == 1


@CORE
def test_each_labelling_lowers_the_energy_each_refit_raises_the_llf(inferred: Inference, cna_planted: Planted) -> None:
    """Each iteration's labels have no higher `potts_energy` than the last labels on that iteration's field, and the
    first falls; `baum_welch`'s llf is no lower where the labels, so the pseudobulk, did not change. Fails
    where a half step of the alternation moves uphill.

    Across a refit the energy is on a new field, which the HMM does not maximize: seeds 0 and 3 at `sp` rise by 0.4
    and 0.7 of 2.1e4 with the labels fixed. Neither the docstrings nor run_cnamaste state it falls, so no row asserts
    it.
    """
    energy = [
        [potts_energy(s.field, cna_planted.adjacency, x, SPATIAL_WEIGHT) for x in (s.labels, s.prev)]
        for s in inferred.steps
    ]
    assert all(e <= prev for e, prev in energy) and energy[0][0] < energy[0][1]
    fixed = [np.array_equal(s.prev, s.labels) for s in inferred.steps]
    llf = np.array(inferred.llf)
    assert fixed.count(True) >= 2 and all(llf[i + 1] >= llf[i] for i, kept in enumerate(fixed) if kept)


@CORE
@pytest.mark.parametrize("params, logmu_shift_type", [("sp", "argmax"), ("smp", "posterior")])
def test_each_fixed_m_step_lowers_its_own_cost(cna_planted: Planted, params: str, logmu_shift_type: str) -> None:
    """At `hmm.max_iter_em` 100, after `hmm.callback_warm_start`'s M step with callback E steps, each fit's M steps
    run at fixed posteriors: each minimize's cost at its start reads the same after it, and its end is no higher.
    Fails where an E step or `log Z_c`'s weights move within such an M step."""
    run = get_inference(cna_planted, params, "opengm_expansion", logmu_shift_type=logmu_shift_type, max_iter_em=100)
    assert len(run.m_steps) > len(run.llf)
    assert all(start == after and end <= start for start, after, end in run.m_steps)


@CORE
def test_the_merge_folds_the_split_clone_and_never_raises_the_energy(cna_planted: Planted) -> None:
    """Under ICM at `sp`, start clone 3's two halves (one genotype) stay apart until the merge, which folds them; at
    every merging iteration the merged labels' `potts_energy` is no higher than the unmerged, on the same field, and
    lower once. Fails where the merge takes a pair that raises the energy, or misses the one that lowers it."""
    run = get_inference(cna_planted, "sp", "icm")
    merged = [s for s in run.steps if s.merge]
    energy = [
        [potts_energy(s.field, cna_planted.adjacency, x, SPATIAL_WEIGHT) for x in (s.labels, s.unmerged)]
        for s in merged
    ]
    assert all(m <= u for m, u in energy) and any(m < u for m, u in energy)
    assert len(np.unique(merged[0].unmerged)) == len(np.unique(merged[0].labels)) + 1


@CORE
def test_no_clone_ends_under_the_floor(inferred: Inference, cna_planted: Planted) -> None:
    """The 36-spot clone 4 is gone and every clone holds >= 50 spots, the floor; at `sp`, `get_merged_labelling` at the
    run's thresholds (50 spots, 10 UMIs per bin; run_cnamaste.py:689) then folds none. Fails where the floor or the reindex leaves a small clone."""
    labels = inferred.res["new_assignment"]
    assert np.bincount(labels).min() >= FLOOR and len(np.unique(labels)) == 4
    if inferred.params == "sp":
        _, groups = get_merged_labelling(labels, cna_planted.counts.total_snp_umis, FLOOR, CNA_OBS * 10)
        assert groups == [[c] for c in range(4)]


@CORE
@pytest.mark.parametrize(
    "inferred", ["sp", pytest.param("smp", marks=pytest.mark.slow)], indirect=True, scope="module"
)  # NB `smp` measured 6.0 and 6.1 s serial, so `slow`
def test_run_hmrf_is_deterministic_under_the_seed(inferred: Inference, cna_planted: Planted) -> None:
    """A second run, unrecorded, returns bitwise the labels, parameters and posteriors. Fails where a solver, the
    initializer or a parallel reduction draws outside the seed, or the recording perturbs the run."""
    get_global_config().hmrf.logmu_shift_type = inferred.logmu_shift_type
    again = get_core_inference(cna_planted, inferred.params)
    assert all(np.array_equal(a, b) for a, b in zip(again.parameters[1:], inferred.res.parameters[1:], strict=True))
    for key in ("new_assignment", "log_gamma"):
        assert np.array_equal(again[key], inferred.res[key]), key
