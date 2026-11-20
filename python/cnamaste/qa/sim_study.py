"""A study streamed one sample at a time: draw and write it, run each method, score each run, keep a record.

The root is `batch_zenodo_run.sh`'s (`qa.sim_runs`): the sample under `ROOT/simulated_data_related/<sample_id>/`,
cnamaste's run under `ROOT/nomixing_cnamaste_related/<sample_id>/<solver>/run_<id>/`, CalicoST's under
`ROOT/nomixing_calicost_related/<sample_id>/`, and a record per sample, `ROOT/qa/stream/<sample_id>.json`: per method
its wall, peak MB and scores. The sample's truth pages are written to its `qa/`; once every method has scored it, its
counts are deleted (`drop_sample_counts`);
its truth, the runs' tables, pages and logs, and the record are kept. Nothing is replaced.
"""

from __future__ import annotations

import json
import math
import resource
import time
from pathlib import Path
from typing import Any

from cnamaste.calicost.read import (
    first_run,
    read_estimate as read_calicost_estimate,
    read_neighbour_counts,
    results_dir,
)
from cnamaste.calicost.run import get_calicost_config, run_calicost
from cnamaste.qa.metrics import get_run_scores, metrics
from cnamaste.qa.sim_runs import run_dir_of, solver_dir, solver_over_rides
from cnamaste.scripts.run_cnamaste import run_cnamaste
from cnamaste.sim.manifest import Simulation
from cnamaste.sim.plot import write_truth_pages
from cnamaste.sim.study import draw_study_sample
from cnamaste.sim.write import get_made_dir
from cnamaste.truth import read_sim_truth, read_truth

METHODS = ("cnamaste", "calicost")
"""The methods a study runs, in order."""


def get_methods() -> tuple[str, ...]:
    """Return the methods a study runs, ("cnamaste", "calicost")."""
    return METHODS


def get_nested_dir(root: Path, names: tuple[str, ...]) -> Path:
    """`root/names[0]/names[1]/...`, each level made where missing (`sim.write.get_made_dir`).

    Args:
        root: an existing directory.
        names: the levels.

    Returns:
        The innermost directory.
    """
    for name in names:
        root = get_made_dir(root, name, exist_ok=True)
    return root


def run_study_cnamaste(sample: Path, root: Path, solver: str) -> dict[str, Any]:
    """`run_cnamaste` on the sample's `config.yaml` with `solver`'s overrides, its run under `solver_dir`, scored.

    Args:
        sample: the drawn sample.
        root: the study directory.
        solver: `<label_solver>-<initializer>-<decoder>`.

    Returns:
        `seconds`, `peak_mb` (this process's high-water mark after the run), the run's directory, `get_run_scores`
            and `score_seconds`.
    """
    output_dir = get_nested_dir(root, ("nomixing_cnamaste_related", sample.name, solver))
    start = time.perf_counter()
    run_cnamaste(str(sample / "config.yaml"), [f"paths.output_dir={output_dir}", *solver_over_rides(solver)])  # type: ignore[no-untyped-call]
    seconds = time.perf_counter() - start
    run_dir = run_dir_of(solver_dir(root, sample.name, solver))
    record = {"seconds": seconds, "peak_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}
    start = time.perf_counter()
    scores = get_run_scores(run_dir, sample)
    return record | {"run_dir": str(run_dir)} | scores | {"score_seconds": time.perf_counter() - start}


def run_study_calicost(sample: Path, root: Path, stop_at_first_inference: bool) -> dict[str, Any]:
    """CalicoST on the sample, its run under `calicost.read.results_dir`, scored where it ran to its end.

    Args:
        sample: the drawn sample.
        root: the study directory.
        stop_at_first_inference: stop it once its first BAF-only inference starts (`calicost.run.run_calicost`).

    Returns:
        `seconds`, `peak_mb`, `returncode`, `started_inference`, its neighbour counts (`read_neighbour_counts`), and
            `metrics` of its first rectangle where it has one.
    """
    output_dir = get_nested_dir(root, ("nomixing_calicost_related", sample.name))
    config = output_dir / "calicost_config.txt"
    with config.open("x") as stream:
        stream.write(get_calicost_config(sample, output_dir))
    run = run_calicost(config, output_dir, output_dir / "calicost.log", stop_at_first_inference)
    record: dict[str, Any] = run._asdict() | read_neighbour_counts(output_dir)
    if not stop_at_first_inference and run.returncode == 0:
        _, run_dir, _ = first_run(results_dir(root, sample.name))
        record |= {"run_dir": str(run_dir)} | metrics(read_calicost_estimate(run_dir), read_truth(sample))
    return record


def stream_study_sample(
    simulation: Simulation, k: int, root: Path, methods: tuple[str, ...], solver: str, stop_calicost: bool
) -> dict[str, Any]:
    """One study sample through the stream: drawn, written, each of `methods` run and scored, its truth pages
    written, its counts dropped, a record written.

    Args:
        simulation: a study simulation.
        k: its realization.
        root: the study directory, existing.
        methods: some of `get_methods()`.
        solver: cnamaste's `<label_solver>-<initializer>-<decoder>`.
        stop_calicost: stop CalicoST at its first BAF-only inference.

    Returns:
        The record: `sample_id`, the draw's seconds, a record per method, the truth pages (`sim.plot`, in the
            sample's `qa/`) and their seconds, and the count files dropped, none unless every method scored the sample.
    """
    start = time.perf_counter()
    _, sample = draw_study_sample(simulation, k, get_nested_dir(root, ("simulated_data_related",)))
    record: dict[str, Any] = {"sample_id": sample.name, "draw_seconds": time.perf_counter() - start}
    if "cnamaste" in methods:
        record["cnamaste"] = run_study_cnamaste(sample, root, solver) | {"solver": solver}
    if "calicost" in methods:
        record["calicost"] = run_study_calicost(sample, root, stop_calicost)
    start = time.perf_counter()
    record["truth_pages"] = [str(p.relative_to(sample)) for p in write_truth_pages(read_sim_truth(sample), None)]
    record["truth_pages_seconds"] = time.perf_counter() - start
    scored = set(methods) == set(get_methods()) and "clone_ari" in record["calicost"]
    record["dropped"] = [str(p.relative_to(sample)) for p in drop_sample_counts(sample)] if scored else []
    path = get_nested_dir(root, ("qa", "stream")) / f"{sample.name}.json"
    with path.open("x") as stream:
        stream.write(json.dumps(record, indent=1, default=get_json_number) + "\n")
    return record


def get_count_files(sample: Path) -> list[Path]:
    """A drawn sample's count files: each slice's `filtered_feature_bc_matrix.h5ad`, `snp/*.npz`, and CalicoST's
    copies (`spatial/tissue_positions.csv`, plain `snp/barcodes.txt` and `snp/unique_snp_ids.npy`), those present.

    Args:
        sample: the drawn sample.

    Returns:
        The files.
    """
    slices = [*sample.glob("*/filtered_feature_bc_matrix.h5ad"), *sample.glob("*/spatial/tissue_positions.csv")]
    snp = [*sample.glob("snp/*.npz"), sample / "snp/barcodes.txt", sample / "snp/unique_snp_ids.npy"]
    return [p for p in sorted([*slices, *snp]) if p.is_file()]


def drop_sample_counts(sample: Path) -> list[Path]:
    """Delete `get_count_files(sample)`; its truth, `snp/*.gz`, positions and runs are kept.

    Args:
        sample: the drawn sample, scored by every method.

    Returns:
        The files deleted.
    """
    found = get_count_files(sample)
    for path in found:
        path.unlink()
    return found


def get_json_number(value: Any) -> float | None:
    """A NumPy scalar as JSON states it; NaN as null.

    Args:
        value: e.g. `np.float64`.

    Returns:
        The float, or None for NaN.
    """
    number = float(value)
    return None if math.isnan(number) else number
