"""
Score every cnamaste solver's runs, and CalicoST's, on each simulated sample against its planted truth.

Writes `ROOT/qa/all_sample_metrics[_one_shot|_all_shot].tsv`, one row per run, and a validation figure per solver.
CalicoST is scored at its best rectangle by `total_llf`; `--one_shot` at its first; `--all_shot` at every rectangle with
a final fit, one row each, the baseline's boxes pooling them. With
`--config`, the runs that configuration names (`results.run_id`), each solver's overrides applied as the batch applies
them, scored into `ROOT/qa/<id>/` beside `config_<id>.yaml`, `<id>` the configuration's own.

Usage:
    uv run run_sim_qa --root /Users/mw9568/Work/ragr/sim/ --baseline_solver calicost --one_shot
    uv run run_sim_qa --root /Users/mw9568/Work/ragr/sim/ --config zenodo_sim_config.yaml -o hmrf.random_seed=0 ...

Reads `ROOT` as `batch_zenodo_run.sh` writes it (`qa.sim_runs`):

    ROOT/
        simulated_data_related/<sample_id>/              truth_clone_labels.tsv, truth_acn_profile.tsv
        nomixing_cnamaste_related/<sample_id>/<solver>/
            run_<id>/                                    cnv_<ploidy>_seglevel.tsv, clone_labels.tsv[.gz],
                                                         clone_labels_int.tsv[.gz]
        nomixing_calicost_related/<sample_id>/
            clone*_rectangle*_w*/                        cnv<_ploidy>_seglevel.tsv, rdrbaf_final_nstates*_smp.npz
        qa/[<id>/]                                       written here

`<sample_id>` is `--study`'s: zenodo `numcnas<a.b>_cnasize<s>_ploidy<p>_random<r>`, or hd_tree
`hd_tree_cnas<n>_length<L>mb_r<k>_<hash>` (`run_sim_study`); `<solver>` `<label_solver>-<initializer>-<decoder>`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from cnamaste.calicost.read import get_calicost_ploidies, get_default_calicost_ploidy
from cnamaste.config import YAMLConfig, start_time
from cnamaste.logger import get_logger
from cnamaste.qa.metrics import calc_metrics, metrics as panel_metrics, panel_columns

# NB `METHOD_LABELS` how the figure's key writes each method; `SAMPLE_IDS` each study's sample id, the figure's groups.
from cnamaste.qa.plot import get_method_labels, get_sample_ids, plot_validation
from cnamaste.qa.sim_runs import SimSample, find_scoreable_samples, has_run, runs, solver_dir, solver_run_ids
from cnamaste.results import run_config_name, run_id, write_fig

# NB `read_truth` reads a sample's truth: its planted clone per spot, spot coordinates and allele-specific copies.
from cnamaste.truth import read_truth

logger = get_logger("cnamaste.scripts.run_sim_qa", start_time=start_time)

BASELINE_CALICOST = "calicost"

Figures = dict[str, tuple[str | None, Path]]
"""solver -> (the series compared beside it, or None; its figure's path)."""


def solvers_with_runs(root: Path) -> list[str]:
    """Every solver for which a sample has a valid cnamaste run, sorted: the solvers scored.

    Args:
        root: the study directory.

    Returns:
        The solvers' directory names under `root/nomixing_cnamaste_related/<sample_id>/`.
    """
    runs = Path(root) / "nomixing_cnamaste_related"
    return sorted({d.name for d in runs.glob("*/*") if d.is_dir() and has_run(d)})


def solver_label(solver: str) -> str:
    """`solver` as it appears on a figure.

    Args:
        solver: `<label_solver>-<initializer>-<decoder>`.

    Returns:
        E.g. `expansion-lattice-lattice` for `opengm_expansion-lattice-lattice`.
    """
    return solver.removeprefix("opengm_")


def has_baseline(sample: SimSample, baseline: str) -> bool:
    """Whether `sample` has a run of `baseline`: CalicoST's rectangle, or a cnamaste solver's run."""
    return bool(sample.calicost) if baseline == BASELINE_CALICOST else baseline in sample.solvers


def baseline_rows(table: pd.DataFrame, baseline: str | None) -> tuple[pd.DataFrame, str] | None:
    """`table`'s rows of `baseline` and their method: CalicoST's, or a cnamaste solver's; None without a baseline,
    each solver then drawn alone."""
    if baseline == BASELINE_CALICOST:
        return table[table["method"] == "calicost"], "calicost"
    if baseline is not None:
        return table[(table["method"] == "cnamaste") & (table["solver"] == baseline)], "cnamaste"
    return None


SHOTS = {"best": "", "one_shot": "_one_shot", "all_shot": "_all_shot"}
"""Which CalicoST rectangles are scored -> the suffix of the table and figures: its best by `total_llf`, its first
(`--one_shot`), or every one (`--all_shot`)."""


def scoreable_samples(
    root: Path, solvers: list[str], shot: str, study: str, ids: dict[str, str] | None = None
) -> tuple[list[SimSample], list[str]]:
    """`find_scoreable_samples`, a sample whose id the figure cannot group refused with the rest.

    Args:
        root: the study directory.
        solvers: the cnamaste solvers.
        shot: one of `SHOTS`.
        study: `--study`, one of `get_sample_ids()`, the ids the figure groups.
        ids: solver -> its run's id (`--config`), or None.

    Returns:
        `(samples, refused)`, as `find_scoreable_samples`.
    """
    samples, refused = find_scoreable_samples(root, solvers, shot == "one_shot", ids, shot == "all_shot")
    pattern = get_sample_ids()[study]
    refused += [
        f"{s.name}: not a {study} sample id, which the figure groups by" for s in samples if not pattern.match(s.name)
    ]
    return [s for s in samples if pattern.match(s.name)], refused


def qa_dir(root: Path, config_id: str | None = None) -> Path:
    """`ROOT/qa/`, or `ROOT/qa/<config_id>/` for the runs one configuration names, `config_id` its `results.run_id`."""
    return Path(root) / "qa" if config_id is None else Path(root) / "qa" / config_id


def get_all_sample_metrics_output_path(root: Path, shot: str, config_id: str | None = None) -> Path:
    """`qa_dir`'s `all_sample_metrics[_one_shot|_all_shot].tsv`: one row per run scored, `panel_columns`.

    Args:
        root: the study directory.
        shot: one of `SHOTS`.
        config_id: `--config`'s id, or None.

    Returns:
        The table's path.
    """
    return qa_dir(root, config_id) / f"all_sample_metrics{SHOTS[shot]}.tsv"


def figure_paths(
    root: Path,
    samples: list[SimSample],
    baseline: str | None,
    shot: str,
    config_id: str | None = None,
    solver: str | None = None,
) -> Figures:
    """The validation figure per solver that scoring `samples` writes, under `qa_dir`.

    Args:
        root: the study directory.
        samples: `scoreable_samples`'.
        baseline: `--baseline_solver`.
        shot: one of `SHOTS`.
        config_id: `--config`'s id, or None.
        solver: `--solver`: that solver's figure alone, or None for every solver's.

    Returns:
        The `Figures`; a cnamaste `baseline` has no figure of its own, its rows drawn beside each other solver's.
    """
    suffix = SHOTS[shot]
    qa = qa_dir(root, config_id)
    figures = {}
    # NB `sample.solvers` are the solvers the sample has a finished run with, the baseline's included.
    # NB a cnamaste baseline is drawn beside every other solver's boxes; a figure of its own repeats them.
    drawn = {s for sample in samples for s in sample.solvers} - {baseline}
    for drawn_solver in sorted(drawn if solver is None else drawn & {solver}):
        # NB the baseline is drawn beside a solver only where it ran one of the solver's samples; otherwise the solver
        #    is drawn alone, its figure named without `_baseline_`.
        ran = [s for s in samples if drawn_solver in s.solvers]
        compared = baseline if baseline is not None and any(has_baseline(s, baseline) for s in ran) else None
        name = f"cnamaste_{drawn_solver}_baseline_{baseline}" if compared else f"cnamaste_{drawn_solver}"
        figures[drawn_solver] = (compared, qa / f"{name}{suffix}_validation.pdf")
    return figures


def score_samples(
    root: Path, samples: list[SimSample], calicost_ploidy: str, ids: dict[str, str] | None = None
) -> tuple[list[dict[str, Any]], list[SimSample], list[str]]:
    """`calc_metrics` of the panels for every run, each run tried alone, so one that cannot be scored leaves out itself
    and no other.

    Args:
        root: the study directory.
        samples: `scoreable_samples`'.
        calicost_ploidy: `--calicost_ploidy`.
        ids: solver -> its run's id (`--config`), or None.

    Returns:
        `(rows, scored, refused)`: the rows, in `runs`' order per sample; each sample with the runs scored alone, for
            `figure_paths`; and `<sample> [<solver> | calicost rectangle <r>]: <error>` for each run not scored.
    """
    rows, scored, refused = [], [], []
    for sample in samples:
        try:
            truth = read_truth(sample.path)
        except Exception as error:  # noqa: BLE001 -- every run is tried, so the warning names them all
            refused.append(f"{sample.name}: {type(error).__name__}: {error}")
            continue
        solvers, calicost = [], []
        for solver in sample.solvers:
            try:
                rows += calc_metrics(
                    runs(root, sample.name, [solver], None, calicost_ploidy, ids), truth, panel_metrics
                )
                solvers.append(solver)
            except Exception as error:  # noqa: BLE001 -- as above
                refused.append(f"{sample.name} {solver}: {type(error).__name__}: {error}")
        for rectangle in sample.calicost:
            try:
                rows += calc_metrics(runs(root, sample.name, [], rectangle, calicost_ploidy), truth, panel_metrics)
                calicost.append(rectangle)
            except Exception as error:  # noqa: BLE001 -- as above
                refused.append(f"{sample.name} calicost rectangle {rectangle[0]}: {type(error).__name__}: {error}")
        if solvers or calicost:
            scored.append(SimSample(sample.path, solvers, tuple(calicost)))
    return rows, scored, refused


def write_validation_figs(table: pd.DataFrame, figures: Figures, baseline: str | None, study: str) -> None:
    """`plot_validation` of each solver's cnamaste rows, with `baseline`'s beside them over the solver's samples, to
    `figure_paths`' figure.
    """
    cnamaste = table[table["method"] == "cnamaste"]
    found = baseline_rows(table, baseline)
    kind = found[1] if found is not None else None
    for solver, (compared, path) in figures.items():
        solver_rows = cnamaste[cnamaste["solver"] == solver].assign(series=solver)
        names = (
            {solver: get_method_labels()["cnamaste"]}
            if kind != "cnamaste"
            else {
                solver: f"{get_method_labels()['cnamaste']} {solver_label(solver)}",
                baseline: f"{get_method_labels()['cnamaste']} {solver_label(baseline)}",
            }
        )
        if found is None or compared is None:
            figure = plot_validation(solver_rows, study, solver, compared=None, kinds={solver: "cnamaste"}, names=names)
        else:
            # NB the baseline's rows over the samples this solver ran, so both series' boxes count the same samples.
            shown = found[0][found[0]["sample_id"].isin(solver_rows["sample_id"])]
            figure = plot_validation(
                pd.concat([solver_rows, shown.assign(series=baseline)]),
                study,
                solver,
                compared=compared,
                kinds={solver: "cnamaste", baseline: kind},
                names=names,
            )
        write_fig(str(path.parent), path.name, figure)


def write_all_metrics(
    path: Path, figures: Figures, rows: list[dict[str, Any]], baseline: str | None, study: str
) -> None:
    """The table of `rows` to `path`, then `write_validation_figs`.

    Args:
        path: `get_all_sample_metrics_output_path`'s.
        figures: `figure_paths`'.
        rows: `score_samples`'.
        baseline: `--baseline_solver`.
        study: `--study`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows, columns=list(panel_columns)).astype({"rectangle": "Int64"})
    table.to_csv(path, sep="\t", index=False)
    write_validation_figs(table, figures, baseline, study)


def parser() -> argparse.ArgumentParser:
    found = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    found.add_argument(
        "--root",
        type=Path,
        required=True,
        help="holds simulated_data_related/ and nomixing_<method>_related/",
    )
    found.add_argument(
        "--study",
        choices=sorted(get_sample_ids()),
        default="zenodo",
        help="the samples' id pattern, the figure's groups: zenodo, CalicoST's numcnas<a.b>_cnasize<s>_ploidy<p>_random<r>; "
        "hd_tree, hd_tree_cnas<n>_length<L>mb_r<k>_<hash>",
    )
    found.add_argument(
        "--baseline_solver",
        default=None,
        help=f"draw this series' boxes beside each solver's: {BASELINE_CALICOST!r}, CalicoST's runs, or a cnamaste "
        "solver <label_solver>-<initializer>-<decoder>, e.g. opengm_expansion-lattice-lattice",
    )
    found.add_argument(
        "--solver",
        default=None,
        help="draw this solver's figure alone, <label_solver>-<initializer>-<decoder>; the table scores every solver",
    )
    shots = found.add_mutually_exclusive_group()
    shots.add_argument(
        "--one_shot",
        action="store_true",
        help="score CalicoST's first rectangle initialization rather than its best",
    )
    shots.add_argument(
        "--all_shot",
        action="store_true",
        help="score every CalicoST rectangle initialization with a final fit, one row each, rather than its best",
    )
    found.add_argument(
        "--calicost_ploidy",
        choices=get_calicost_ploidies(),
        default=get_default_calicost_ploidy(),
        help=f"CalicoST's copies to score: cnv<_ploidy>_seglevel.tsv (default {get_default_calicost_ploidy()})",
    )
    found.add_argument(
        "--overwrite",
        action="store_true",
        help="replace the table and figures already on disk",
    )
    found.add_argument(
        "--config",
        type=Path,
        default=None,
        help="score the runs this configuration names (results.run_id), each solver's overrides applied, into qa/<id>/",
    )
    found.add_argument(
        "--over_rides",
        "-o",
        action="append",
        default=[],
        help="key=value on --config, as the batch gives every job, e.g. -o hmrf.random_seed=0; repeated",
    )
    return found


def configured(
    config_path: Path | None, over_rides: list[str], solvers: list[str]
) -> tuple[str | None, dict[str, str] | None]:
    """`--config`'s id and each solver's run id, or `(None, None)` without it.

    Args:
        config_path: `--config`.
        over_rides: `-o`.
        solvers: the solvers with any run.

    Returns:
        `(config_id, ids)`: `config_id` the configuration's own `results.run_id`, naming `qa/<config_id>/`; `ids` solver -> its
            run's id.
    """
    if config_path is None:
        return None, None
    config = YAMLConfig.from_file(config_path)
    config.over_ride(over_rides)
    return run_id(config), solver_run_ids(config_path, over_rides, solvers)


def main(argv: list[str] | None = None) -> int:
    """Score every solver with a run in the study, then write the table and figures: find, check, score, write.

    Stops before writing on argparse's usage error where the study has no run, `--baseline_solver` names no run, or an
    output is on disk without `--overwrite`; and on `SystemExit` where no run can be scored. A run that cannot be
    scored is left out, the rest written, and a warning at the end names each.
    """
    command = parser()
    args = command.parse_args(argv)
    if args.over_rides and args.config is None:
        command.error("-o overrides --config: pass the configuration")
    solvers = solvers_with_runs(args.root)
    config_id, ids = configured(args.config, args.over_rides, solvers)
    if ids is not None:
        # NB a solver counts where some sample has the run the configuration names for it.
        runs_dir = Path(args.root) / "nomixing_cnamaste_related"
        solvers = [
            s
            for s in solvers
            if any(has_run(solver_dir(args.root, d.name, s), ids[s]) for d in runs_dir.iterdir() if d.is_dir())
        ]
    if not solvers:
        named = f" named by {args.config}" if args.config is not None else ""
        command.error(f"no cnamaste run{named} under {Path(args.root) / 'nomixing_cnamaste_related'}")
    baseline = args.baseline_solver
    if baseline not in (None, BASELINE_CALICOST) and baseline not in solvers:
        command.error(
            f"--baseline_solver {baseline!r} is {BASELINE_CALICOST!r} or a solver some sample has a run "
            f"with: {', '.join(solvers)}"
        )
    if args.solver is not None and (args.solver not in solvers or args.solver == baseline):
        command.error(
            f"--solver {args.solver!r} is a solver some sample has a run with, other than the baseline: "
            f"{', '.join(solvers)}"
        )

    # NB refused: several runs in one solver's directory,
    #             an unreadable CalicoST directory,
    #             a sample id the figure cannot group (`scoreable_samples`),
    #             a run whose tables cannot be read or scored,
    #             a run in progress among them (`score_samples`).
    shot = "one_shot" if args.one_shot else "all_shot" if args.all_shot else "best"
    samples, refused = scoreable_samples(args.root, solvers, shot, args.study, ids)
    table = get_all_sample_metrics_output_path(args.root, shot, config_id)
    # NB every output the samples found could write, checked before the scoring that takes the time.
    figures = figure_paths(args.root, samples, baseline, shot, config_id, args.solver)
    existing = [p for p in (table, *(path for _, path in figures.values())) if p.exists()]
    if existing and not args.overwrite:
        command.error(f"would overwrite {', '.join(map(str, existing))}; pass --overwrite to replace them")

    rows, scored, failed = score_samples(args.root, samples, args.calicost_ploidy, ids)
    refused += failed
    listed = "\n  ".join(refused)
    if not rows:
        left = ", ".join(map(str, existing)) or "none exist"
        raise SystemExit(
            f"run_sim_qa: stopped, nothing written; no run can be scored, {len(refused)} refused:\n"
            f"  {listed}\nexisting outputs left unchanged: {left}"
        )

    write_all_metrics(
        table, figure_paths(args.root, scored, baseline, shot, config_id, args.solver), rows, baseline, args.study
    )
    if config_id is not None:
        # NB the configuration the table scored, resolved, beside it: what `qa/<id>/` names.
        config = YAMLConfig.from_file(args.config)
        config.over_ride(args.over_rides)
        (table.parent / run_config_name(config_id)).write_text(yaml.safe_dump(config.to_dict(), sort_keys=False))
    logger.info(f"{len(rows)} rows to {table}")
    if refused:
        logger.warning(
            f"run_sim_qa: {len(refused)} runs cannot be scored, left out of {table} and its figures:\n  {listed}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
