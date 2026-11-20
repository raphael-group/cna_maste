"""Wall time of each NB/BB evaluator on its recorded call in the captures of CalicoST easy (`2d4ce9a9`) and hard
(`8797710b`), and of each `hmrf.label_solver` on seeded synthetic Visium HD labellings (`test_label_solver_wall_time`).

An evaluator's call is the one `tests/scripts/run_capture_sim.py` recorded, its first in the run (`/config`'s
`evaluators`), decoded by `qa.capture.Capture.decode`. Each runs twice before timing, so numba compilation is not
timed. Off by default: `pytest tests/test_benchmarks.py -m benchmark -p no:xdist -o addopts=--strict-markers
--benchmark-json=<file>`, two files compared by `tests/scripts/compare_benchmarks.py`.
"""

from __future__ import annotations

import json
import shutil
import time
from collections.abc import Iterator
from functools import cache
from typing import Any, NamedTuple

import h5py
import numpy as np
import pytest
import scipy.sparse as sp
from _pytest.mark import ParameterSet
from scipy.spatial import cKDTree
from sklearn.metrics import adjusted_rand_score

from cnamaste import expansion, external, trws
from cnamaste.config import YAMLConfig, config_defaults, set_global_config
from cnamaste.config.yaml_config import LabellingOptions
from cnamaste.icm import icm_sweep_deque
from cnamaste.initialize.spatial import get_rectangle_labelling
from cnamaste.opengm import get_opengm_start_algorithms, solve_labelling
from cnamaste.potts import PottsProblem, pairwise_potts_energy, potts_problem
from cnamaste.qa.capture import Capture
from cnamaste.qa.sims import get_dev_sims, get_project, get_supported_sims
from cnamaste.spatial import construct_multislice_lattice_adjacency

TIMED_SECONDS = 2.0
"""An evaluator's timed rounds: as many as fill this, from its second untimed call, between 3 and 100."""


def get_evaluator_calls() -> list[ParameterSet]:
    """Each dev-tier capture's recorded evaluator calls, as `(capture_hash, call)`.

    Returns:
        The parameters, `<hash>-<call>` by id.
    """
    params = []
    for sim_hash in get_dev_sims():
        with h5py.File(get_project() / get_supported_sims()[sim_hash], "r") as h5:
            calls = json.loads(h5["config"].attrs["evaluators"]).values()
        params += [pytest.param(sim_hash, call, id=f"{sim_hash}-{call}") for call in calls]
    return params


@pytest.fixture(autouse=True)
def node_id_for_pytest_benchmark(request: pytest.FixtureRequest) -> None:
    """Set `_nodeid`, which pytest-benchmark 5.3.0 reads and pytest 9.2 no longer sets."""
    request.node._nodeid = request.node.nodeid


@pytest.mark.benchmark
@pytest.mark.parametrize(("capture_hash", "call"), get_evaluator_calls())
def test_evaluator_wall_time(benchmark: Any, capture_hash: str, call: str) -> None:
    """Time one evaluator on its recorded call, after two untimed calls, the second sizing the rounds."""
    capture = Capture(get_project() / get_supported_sims()[capture_hash])
    try:
        function, args, kwargs = capture.decode(call)
    finally:
        capture.close()
    function(*args, **kwargs)
    start = time.perf_counter()
    function(*args, **kwargs)
    rounds = int(np.clip(TIMED_SECONDS / max(time.perf_counter() - start, 1e-6), 3, 100))
    benchmark.pedantic(function, args=tuple(args), kwargs=kwargs, rounds=rounds, warmup_rounds=2)


class Labelling(NamedTuple):
    """A planted labelling problem: the field `single_llf` (n_bins, n_labels), its CSR adjacency and pairwise form,
    the start labels and the planted truth.

    Args: `single_llf`, `adjacency`, `problem`, `start`, `truth` as above.
    """

    single_llf: np.ndarray
    adjacency: sp.csr_matrix
    problem: PottsProblem
    start: np.ndarray
    truth: np.ndarray


# NB (bins, labels, signal): signal is the planted clone's field lead in units of the unit-normal per-bin noise, so the
#    field's argmax alone mislabels 36-45% of bins; 87,010 bins is a Visium HD 16 um sample's.
LABELLINGS = {
    "hd16um-87010x16": (87_010, 16, 2.0),
    "hd16um-87010x12": (87_010, 12, 2.0),
    "hd16um-20000x16": (20_000, 16, 2.0),
    "hd16um-3000x4": (3_000, 4, 1.0),
}

# NB `<hmrf.label_solver>`, `opengm_expansion-<hmrf.opengm_mincut_solver>`, or `expansion-reduced`: both reductions
#    and the field_share order.
label_solvers = (
    "icm",
    "opengm_icm",
    "opengm_expansion-boykov_kolmogorov",
    "opengm_expansion-dinic",
    "opengm_trws",
    "expansion",
    "expansion-reduced",
    "trws",
)

BIN_PITCH_UM = 16.0

# NB the shipped config.yaml's hmrf.spatial_weight.
SPATIAL_WEIGHT = 0.75

MIN_CLONE_SPOTS = int(config_defaults["hmrf"]["min_spots_per_clone"])

# NB opengm_assignment's max_iterations and tolerance.
OPENGM_MAX_ITERATIONS = 1000
OPENGM_TOLERANCE = 1e-9

# NB timed rounds by bins, after one untimed call: numba's compilation and the reported labelling.
LABELLING_ROUNDS = {87_010: 1, 20_000: 3, 3_000: 5}


def get_tissue_coords(n_bins: int, rng: np.random.Generator) -> np.ndarray:
    """The `n_bins` grid bins nearest the centre of a three-lobed disc, at `BIN_PITCH_UM`.

    Args:
        n_bins: the bins kept.
        rng: the generator the lobes' phase is drawn from.

    Returns:
        (n_bins, 2) positions in um.
    """
    side = int(np.ceil(2.6 * np.sqrt(n_bins / np.pi)))
    grid = np.stack(np.meshgrid(np.arange(side), np.arange(side), indexing="ij"), axis=-1).reshape(-1, 2) - side / 2
    radius = np.hypot(grid[:, 0], grid[:, 1])
    angle = np.arctan2(grid[:, 1], grid[:, 0])
    lobed = radius / (1.0 + 0.1 * np.sin(3.0 * angle + rng.uniform(0.0, 2.0 * np.pi)))
    return BIN_PITCH_UM * grid[np.argsort(lobed, kind="stable")[:n_bins]]


@cache
def get_labelling(n_bins: int, n_labels: int, signal: float, seed: int = 0) -> Labelling:
    """A seeded labelling: Voronoi clones of `n_labels` random bins, field `signal` on the planted clone plus unit
    normal noise, square-lattice adjacency (k=8) and the run's `get_rectangle_labelling` start.

    Args:
        n_bins: the bins.
        n_labels: the clones.
        signal: the planted clone's field lead.
        seed: the generator's seed.

    Returns:
        The `Labelling`.
    """
    rng = np.random.default_rng(seed)
    coords = get_tissue_coords(n_bins, rng)
    adjacency = construct_multislice_lattice_adjacency(np.zeros(n_bins, dtype=int), ["slice"], coords, None, "square")
    truth = cKDTree(coords[rng.choice(n_bins, n_labels, replace=False)]).query(coords)[1]
    single_llf = rng.standard_normal((n_bins, n_labels))
    single_llf[np.arange(n_bins), truth] += signal
    x_part = next(part for part in range(int(np.sqrt(n_labels)), 0, -1) if n_labels % part == 0)
    start = get_rectangle_labelling(coords, x_part, n_labels // x_part, None)
    problem = potts_problem(single_llf, adjacency, SPATIAL_WEIGHT)
    return Labelling(single_llf, adjacency, problem, start, truth)


def get_icm_finish(labelling: Labelling, labels: np.ndarray) -> tuple[int, np.ndarray]:
    """`icm_sweep_deque` from `labels` with the run's floor: the run's ICM, and the finish of each OpenGM labelling.

    Args:
        labelling: the problem.
        labels: the labels it starts from, not changed.

    Returns:
        The sweeps and the labels.
    """
    sweeps, _, finished = icm_sweep_deque(
        single_llf=labelling.single_llf,
        adj_indptr=labelling.adjacency.indptr,
        adj_indices=labelling.adjacency.indices,
        adj_weights=labelling.adjacency.data,
        new_assignment=labels.copy(),
        spatial_weight=SPATIAL_WEIGHT,
        posterior=None,
        min_clone_spots=MIN_CLONE_SPOTS,
    )
    return sweeps, finished


def get_solved_labelling(
    solver: str, labelling: Labelling, session: external.Session | None
) -> tuple[np.ndarray, float, int]:
    """`solver`'s labelling of `labelling` from its start, the call `hmrf.get_hmrf_labelling` makes.

    OpenGM's is `Session.solve` alone, and cnamaste's expansion `opengm.solve_labelling` alone, on the posed
    `potts_problem`, without `external_assignment`'s ICM finish.

    Args:
        solver: one of `label_solvers`.
        labelling: the problem.
        session: the OpenGM session; None for `icm`, `expansion` and `trws`.

    Returns:
        The labels, the lower bound (`-inf` where none) and the iterations.
    """
    if solver == "icm":
        sweeps, labels = get_icm_finish(labelling, labelling.start)
        return labels, -np.inf, sweeps
    if solver == "trws":
        run = trws.solve(labelling.problem, OPENGM_MAX_ITERATIONS, OPENGM_TOLERANCE)
        return run.labels, run.bound, run.iterations
    if solver.startswith("expansion"):
        reduced = solver == "expansion-reduced"
        options = LabellingOptions(reduced, reduced, "field_share" if reduced else "index")
        run = solve_labelling(
            labelling.problem,
            "expansion",
            labelling.start,
            OPENGM_MAX_ITERATIONS,
            OPENGM_TOLERANCE,
            config_defaults["hmrf"]["opengm_mincut_solver"],
            options,
        )
        return run.labels, run.bound, run.iterations
    algorithm, _, mincut = solver.removeprefix("opengm_").partition("-")
    run = session.solve(
        labelling.problem,
        algorithm,
        max_iterations=OPENGM_MAX_ITERATIONS,
        tolerance=OPENGM_TOLERANCE,
        start=labelling.start if algorithm in get_opengm_start_algorithms() else None,
        mincut=mincut or config_defaults["hmrf"]["opengm_mincut_solver"],
    )
    return run.labels, run.bound, run.iterations


@pytest.fixture(scope="module")
def opengm_session() -> Iterator[external.Session]:
    """One OpenGM worker for the module, its blocks removed at teardown; skips where the library is not built."""
    if not external.available():
        pytest.skip(f"OpenGM is not built: set ${external.get_home_variable()}")
    set_global_config(YAMLConfig({}))
    with external.session() as session:
        yield session
    shutil.rmtree(session.directory, ignore_errors=True)


@pytest.mark.benchmark
@pytest.mark.parametrize("size", LABELLINGS)
@pytest.mark.parametrize("solver", label_solvers)
def test_label_solver_wall_time(benchmark: Any, request: pytest.FixtureRequest, solver: str, size: str) -> None:
    """Time one label solver; its energy, bound, iterations and ARI to the planted clones as `extra_info`, below the
    field argmax's energy and above its ARI."""
    set_global_config(YAMLConfig({}))
    n_bins, n_labels, signal = LABELLINGS[size]
    labelling = get_labelling(n_bins, n_labels, signal)
    if (solver.startswith("expansion") or solver == "trws") and not expansion.available():
        pytest.skip(str(expansion.unavailable()))
    session = request.getfixturevalue("opengm_session") if solver.startswith("opengm_") else None
    labels, bound, iterations = get_solved_labelling(solver, labelling, session)
    argmax = labelling.single_llf.argmax(axis=1)
    _, finished = get_icm_finish(labelling, labels)
    energy = pairwise_potts_energy(labelling.problem, labels)
    ari = float(adjusted_rand_score(labelling.truth, labels))
    benchmark.extra_info.update(
        n_pairs=int(labelling.problem.first.size),
        energy=energy,
        bound=float(bound),
        iterations=int(iterations),
        ari=ari,
        icm_finish_energy=pairwise_potts_energy(labelling.problem, finished),
        icm_finish_ari=float(adjusted_rand_score(labelling.truth, finished)),
        start_energy=pairwise_potts_energy(labelling.problem, labelling.start),
        argmax_energy=pairwise_potts_energy(labelling.problem, argmax),
        argmax_mislabelled=float(np.mean(argmax != labelling.truth)),
        argmax_ari=float(adjusted_rand_score(labelling.truth, argmax)),
    )
    benchmark.pedantic(
        get_solved_labelling, args=(solver, labelling, session), rounds=LABELLING_ROUNDS[n_bins], iterations=1
    )
    assert energy < benchmark.extra_info["argmax_energy"]
    assert ari > benchmark.extra_info["argmax_ari"]
