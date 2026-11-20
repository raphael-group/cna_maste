"""Redraw a finished run's final figures from its written files: `run_cnamaste_plot CONFIG [-o key=value ...]`.

The run is staged as `run_cnamaste_decode` stages a decode (`get_staged_run`): the inferences `RUN_DIR` holds are read,
the decode and its tables are made again, and only the `finalize` stage's figures are drawn (`DRAWN_IN`). Every table
the staged run writes is checked equal to `RUN_DIR`'s before anything moves, so the figures are the run's; then the
figures alone, under `plots/`, are moved into `RUN_DIR`, with the log `cnamaste_plot_<id>.log` beside the run's own.

`RUN_DIR` is `run_<id>`, `<id>` `CONFIG`'s `results.run_id` before `-o`. `-o` takes a key of `PLOT_KEYS`, which changes
figures alone, and `paths.output_dir=<dir>` to write the figures under `<dir>/run_<id>/` rather than into `RUN_DIR`.
The figures of the stages before the decode (the phasing, BAF-only and RDR/BAF pages, `he_clones.png`,
`spot_umis.pdf`) need fits the run does not write and are left as they are.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from cnamaste.config import start_time
from cnamaste.logger import get_logger
from cnamaste.results import run_log_name
from cnamaste.scripts.run_cnamaste_decode import get_staged_run, moved_out, rename_staged_log, unchanged

logger = get_logger("cnamaste.scripts.run_cnamaste_plot", start_time=start_time)

PLOT_KEYS = ("run.plot_field_argmax",)
"""The keys `-o` may set: each changes figures alone, not a table the run writes."""


def get_plot_keys() -> tuple[str, ...]:
    """Return the keys `run_cnamaste_plot -o` may set, ("run.plot_field_argmax",)."""
    return PLOT_KEYS


def get_tables(run_dir: Path) -> tuple[str, ...]:
    """The text tables a run directory holds, `.tsv` or `.tsv.gz`, by name.

    Args:
        run_dir: a run's directory.

    Returns:
        Their names, sorted.
    """
    return tuple(sorted(p.name for p in Path(run_dir).iterdir() if p.name.endswith((".tsv", ".tsv.gz"))))


def plot(
    config_path: Path, run_dir: Path | None = None, outputdir: Path | None = None, over_rides: list[str] | None = None
) -> Path:
    """Redraw `RUN_DIR`'s `finalize` figures and move them, with the log, into its run directory under `outputdir`.

    Args:
        config_path: the run's configuration.
        run_dir: the finished run's directory, the configuration's where None.
        outputdir: where the figures' run directory is, `paths.output_dir` where None.
        over_rides: keys of `get_plot_keys()`, `<key>=<value>` each.

    Returns:
        The run directory the figures were moved into.

    Raises:
        SystemExit: for an override outside `get_plot_keys()`, a `RUN_DIR` without what the decode reads, or a table
            that differs from `RUN_DIR`'s; `RUN_DIR` is then untouched, the staging kept.
    """
    over_rides = list(over_rides or [])
    outside = [o for o in over_rides if o.split("=", 1)[0] not in get_plot_keys()]
    if outside:
        raise SystemExit(
            f"run_cnamaste_decode: -o takes one of {list(get_plot_keys())} or paths.output_dir=<dir>, not {outside}; "
            f"ab_decode.<key> is run_cnamaste_decode's"
        )
    run_dir, staged, outputdir = get_staged_run(config_path, run_dir, outputdir, over_rides, "plot")
    tables = get_tables(staged)
    differ = unchanged(run_dir, staged, tables)
    if differ:
        raise SystemExit(
            f"run_cnamaste_decode: {', '.join(differ)} differ from {run_dir}'s: CONFIG or the inputs are not the "
            f"run's; the figures it refused are {staged}"
        )
    target = outputdir / run_dir.name
    identity = run_dir.name.removeprefix("run_")
    rename_staged_log(staged, identity, "cnamaste_plot")
    # NB the staged run draws DRAWN_IN's figures alone, so its plots/ holds exactly the figures redrawn.
    redrawn = sorted(p.name for p in (staged / "plots").rglob("*") if p.is_file())
    moved_out(staged / "plots", target / "plots")
    log = staged / run_log_name(identity, prefix="cnamaste_plot")
    if log.is_file():
        os.replace(log, target / log.name)
    # NB one path per line, so each copies whole.
    logger.info(
        f"Plotted:\n"
        f"    run: {run_dir}\n"
        f"    figures: {target / 'plots'}\n"
        f"    redrawn: {', '.join(redrawn)}\n"
        f"    unchanged: {len(tables)} tables"
    )
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("config_path", type=Path, help="the run's configuration")
    parser.add_argument(
        "--over_rides",
        "-o",
        action="append",
        default=[],
        help=f"one of {list(get_plot_keys())} as <key>=<value>, or paths.output_dir=<dir> to write the figures there",
    )
    args = parser.parse_args(argv)
    # NB paths.output_dir moves the figures, not the run they are read from: that is CONFIG's own.
    output = [o.split("=", 1)[1] for o in args.over_rides if o.startswith("paths.output_dir=")]
    rest = [o for o in args.over_rides if not o.startswith("paths.output_dir=")]
    try:
        plot(args.config_path, outputdir=Path(output[-1]) if output else None, over_rides=rest)
    except SystemExit as refused:
        # NB a refusal is a line of the run's log, in cnamaste's format, not a bare message.
        if isinstance(refused.code, str):
            logger.error(refused.code.removeprefix("run_cnamaste_decode: "))
            return 1
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
