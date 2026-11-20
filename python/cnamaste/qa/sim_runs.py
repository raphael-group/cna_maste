"""Find a simulation study's runs: the samples with a cnamaste run per solver and CalicoST's rectangle, read one at
a time for `qa.metrics.calc_metrics`.

A study is `batch_zenodo_run.sh`'s tree:

- `ROOT/simulated_data_related/<sample_id>/`: the planted truth (`truth.read_truth`).
- `ROOT/nomixing_cnamaste_related/<sample_id>/<solver>/run_<id>/`: a cnamaste run per configuration (`results.run_id`), its
  `cnv_<ploidy>_seglevel.tsv`, `clone_labels.tsv[.gz]` and `clone_labels_int.tsv[.gz]`; a solver is
  `<label_solver>-<initializer>-<decoder>`. Without a configuration a solver's directory holds one run; with one, the
  run it names (`solver_run_ids`).
- `ROOT/nomixing_calicost_related/<sample_id>/clone*_rectangle*_w*/`: CalicoST's rectangles, each
  `cnv<_ploidy>_seglevel.tsv` and a final fit `rdrbaf_final_nstates*_smp.npz`.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cnamaste.calicost.read import (
    best_run,
    first_run,
    fitted_runs,
    get_default_calicost_ploidy,
    read_estimate as read_calicost_estimate,
    results_dir,
)
from cnamaste.config import YAMLConfig, start_time
from cnamaste.logger import get_logger
from cnamaste.qa.metrics import RunEstimate, read_estimate
from cnamaste.results import run_id

logger = get_logger("cnamaste.qa.sim_runs", start_time=start_time)

default_solver = "opengm_expansion-lattice-lattice"
"""`run_sim_sample_qa`'s `--solver` default: `zenodo_sim_config.yaml`'s `hmrf.label_solver`, `hmm.initializer` and
`ab_decode.decoder` (`lattice` for `hmm_ab`), hyphen-joined as `batch_zenodo_run.sh` names a run's directory."""

Calicost = tuple[int, Path, float]
"""CalicoST's rectangle for a sample: `(rectangle, run directory, total_llf)`."""


@dataclass(frozen=True)
class SimSample:
    """A sample `find_scoreable_samples` found: its truth directory, the solvers it has a run with, CalicoST's
    rectangles to score: the best or the first, one; every one with a final fit (`all_shot`); none where it has no run."""

    path: Path
    solvers: list[str]
    calicost: tuple[Calicost, ...]

    @property
    def name(self) -> str:
        """The sample id."""
        return self.path.name


def solver_dir(root: Path, sample_id: str, solver: str) -> Path:
    """The directory holding a sample's cnamaste runs with one solver.

    Args:
        root: the study directory.
        sample_id: the sample.
        solver: `<label_solver>-<initializer>-<decoder>`.

    Returns:
        `root / "nomixing_cnamaste_related" / sample_id / solver`.
    """
    return Path(root) / "nomixing_cnamaste_related" / sample_id / solver


RUN_GLOB = "run_*"
"""A cnamaste run's directory under its solver's: `run_<id>` (`results.get_run_dir`)."""

DECODER_KEYS = {"lattice": "hmm_ab", "calicost": "calicost"}
"""A solver's `<decoder>` field -> its `ab_decode.decoder`, as `batch_zenodo_run.sh`'s `run_single_job` sets it."""


def solver_over_rides(solver: str) -> list[str]:
    """The `-o` overrides `batch_zenodo_run.sh` gives a job of `solver`.

    Args:
        solver: `<label_solver>-<initializer>-<decoder>`.

    Returns:
        `hmrf.label_solver`, `hmm.initializer` and `ab_decode.decoder`, as `key=value`.

    Raises:
        ValueError: where `solver` is not three fields, or its decoder is not one of `DECODER_KEYS`.
    """
    fields = solver.split("-")
    if len(fields) != 3 or fields[2] not in DECODER_KEYS:
        raise ValueError(
            f"{solver!r} is not <label_solver>-<initializer>-<decoder>, decoder one of {list(DECODER_KEYS)}"
        )
    label_solver, initializer, decoder = fields
    return [
        f"hmrf.label_solver={label_solver}",
        f"hmm.initializer={initializer}",
        f"ab_decode.decoder={DECODER_KEYS[decoder]}",
    ]


def solver_run_ids(config_path: Path, over_rides: list[str], solvers: list[str]) -> dict[str, str]:
    """Each solver's run id: `config_path` with `over_rides`, then the solver's own (`solver_over_rides`), as the batch
    runs it.

    Args:
        config_path: the batch's configuration.
        over_rides: the overrides every job shares, e.g. the seed.
        solvers: the solvers.

    Returns:
        Solver -> `results.run_id`.
    """
    ids = {}
    for solver in solvers:
        config = YAMLConfig.from_file(config_path)
        config.over_ride([*over_rides, *solver_over_rides(solver)])
        ids[solver] = run_id(config)
    return ids


def has_run(directory: Path, identity: str | None = None) -> bool:
    """Whether `directory` holds a run: `run_<identity>`, or any `run_*` where `identity` is None."""
    pattern = RUN_GLOB if identity is None else f"run_{identity}"
    return any(d.is_dir() for d in Path(directory).glob(pattern))


def run_dir_of(directory: Path, identity: str | None = None) -> Path:
    """The run directory under a solver's directory: `run_<identity>`, or the one `run_*` where `identity` is None.

    Args:
        directory: a `solver_dir`.
        identity: the run's `results.run_id`, or None.

    Returns:
        The run directory.

    Raises:
        FileNotFoundError: where there is none.
        ValueError: where `identity` is None and there are several (several configurations).
    """
    found = sorted(d for d in Path(directory).glob(RUN_GLOB if identity is None else f"run_{identity}") if d.is_dir())
    if not found:
        raise FileNotFoundError(f"no {RUN_GLOB if identity is None else f'run_{identity}'} under {directory}")
    if len(found) > 1:
        raise ValueError(
            f"{len(found)} {RUN_GLOB} runs under {directory}, expected one: {[d.name for d in found]}; "
            "pass the configuration to name one"
        )
    return found[0]


def calicost_best(root: Path, sample_id: str, one_shot: bool = False) -> Calicost | None:
    """CalicoST's rectangle for a sample, logged at debug, or None, logged, where it has no run with a final fit.

    Args:
        root: the study directory.
        sample_id: the sample.
        one_shot: take the first rectangle (`first_run`) rather than the best (`best_run`).

    Returns:
        `(rectangle, run directory, total_llf)` or None.
    """
    directory = results_dir(root, sample_id)
    if not directory.is_dir():
        logger.info(f"CalicoST: absent, no {directory}")
        return None
    try:
        rectangle, run_dir, total_llf = first_run(directory) if one_shot else best_run(directory)
    except FileNotFoundError as error:
        logger.info(f"CalicoST: absent, {error}")
        return None
    logger.debug(
        f"{sample_id} CalicoST {'first' if one_shot else 'best'}: rectangle {rectangle}, total_llf {total_llf}: {run_dir}"
    )
    return rectangle, run_dir, total_llf


def calicost_all(root: Path, sample_id: str) -> tuple[Calicost, ...]:
    """Every CalicoST rectangle for a sample with a final fit, by rectangle index, or none, logged, where it has none.

    Args:
        root: the study directory.
        sample_id: the sample.

    Returns:
        `(rectangle, run directory, total_llf)` per rectangle.
    """
    directory = results_dir(root, sample_id)
    if not directory.is_dir():
        logger.info(f"CalicoST: absent, no {directory}")
        return ()
    try:
        found = fitted_runs(directory)
    except FileNotFoundError as error:
        logger.info(f"CalicoST: absent, {error}")
        return ()
    logger.debug(f"{sample_id} CalicoST all: rectangles {[r[0] for r in found]}")
    return tuple(sorted(found, key=lambda r: (r[0], r[1].name)))


def runs(
    root: Path,
    sample_id: str,
    requested_solvers: list[str],
    calicost: Calicost | None,
    calicost_ploidy: str = get_default_calicost_ploidy(),
    ids: Mapping[str, str] | None = None,
) -> Iterator[tuple[dict[str, Any], RunEstimate]]:
    """A sample's runs, read one at a time: its cnamaste run with each solver, then CalicoST's rectangle where given.

    Args:
        root: the study directory.
        sample_id: the sample.
        requested_solvers: the cnamaste solvers.
        calicost: `calicost_best`'s rectangle, or None.
        calicost_ploidy: CalicoST's copies, one of `calicost.read.CALICOST_PLOIDIES`.
        ids: solver -> its run's id (`solver_run_ids`), or None for each solver's one run.

    Yields:
        The run's `qa.metrics.run_columns` identity (`sample_id`, `method`, then `solver`, or `rectangle`, `total_llf`
            and `calicost_ploidy`), and its estimate.

    Raises: as `run_dir_of`, where a solver's cnamaste run is not exactly one; `FileNotFoundError` where CalicoST's run
    has no table at `calicost_ploidy`.
    """
    for solver in requested_solvers:
        run = {"sample_id": sample_id, "method": "cnamaste", "solver": solver}
        yield run, read_estimate(run_dir_of(solver_dir(root, sample_id, solver), (ids or {}).get(solver)))
    if calicost is not None:
        rectangle, run_dir, total_llf = calicost
        run = {
            "sample_id": sample_id,
            "method": "calicost",
            "rectangle": rectangle,
            "total_llf": total_llf,
            "calicost_ploidy": calicost_ploidy,
        }
        yield run, read_calicost_estimate(run_dir, calicost_ploidy)


def find_scoreable_samples(
    root: Path,
    requested_solvers: list[str],
    one_shot: bool = False,
    ids: Mapping[str, str] | None = None,
    all_shot: bool = False,
) -> tuple[list[SimSample], list[str]]:
    """Every sample under `root/simulated_data_related/` with a cnamaste run with any of `requested_solvers`, by file
    alone.

    A sample with no run is logged and skipped. A run that cannot be scored as found -- several in one solver's
    directory, or a CalicoST directory that cannot be read -- is refused, naming it and why. A run missing its tables,
    as a batch in progress leaves one, is found here and refused when `read_estimate` cannot read it.

    Args:
        root: the study directory.
        requested_solvers: the cnamaste solvers.
        one_shot: CalicoST's first rectangle in place of its best.
        ids: solver -> its run's id, or None for each solver's one run.
        all_shot: every CalicoST rectangle (`calicost_all`) in place of one.

    Returns:
        `(samples, refused)`: the samples, sorted by id, each with its solvers and CalicoST's rectangles; and the
            refusals, `<sample> [<solver>]: <why>`.
    """
    samples, refused, ids = [], [], ids or {}
    for path in sorted(d for d in (Path(root) / "simulated_data_related").iterdir() if d.is_dir()):
        found = [s for s in requested_solvers if has_run(solver_dir(root, path.name, s), ids.get(s))]
        if not found:
            logger.info(f"{path.name}: skipped, no cnamaste run with {', '.join(requested_solvers)}")
            continue
        ready = []
        for solver in found:
            try:
                run_dir_of(solver_dir(root, path.name, solver), ids.get(solver))
            except ValueError as error:
                refused.append(f"{path.name} {solver}: {error}; remove the stale run")
                continue
            ready.append(solver)
        try:
            if all_shot:
                calicost = calicost_all(root, path.name)
            else:
                best = calicost_best(root, path.name, one_shot)
                calicost = () if best is None else (best,)
            samples.append(SimSample(path, ready, calicost))
        except Exception as error:  # noqa: BLE001 -- every sample is checked, so the stop names them all
            refused.append(f"{path.name}: {type(error).__name__}: {error}")
    return samples, refused
