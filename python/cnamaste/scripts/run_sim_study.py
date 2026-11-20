"""
Stream a study's samples, in its seeded order, through cnamaste and CalicoST: draw, write, run, score, keep.

Usage:
    uv run pytest tests/sim/test_sim_study.py -m cache -p no:xdist
    uv run run_sim_study hd_tree --root ROOT [--methods cnamaste,calicost] [--max-samples N]

`pytest -m cache` first fills numba's and JAX's compile caches, so the first sample's wall holds no compilation.

Writes `ROOT` as `batch_zenodo_run.sh` does (`qa.sim_study`), so `run_sim_qa --root ROOT --study hd_tree` scores it.
A sample already under `ROOT/simulated_data_related/` is skipped.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cnamaste.config import start_time
from cnamaste.logger import get_logger
from cnamaste.qa.sim_runs import default_solver
from cnamaste.qa.sim_study import get_methods, stream_study_sample
from cnamaste.sim.study import get_sample_name, get_study_order

logger = get_logger("cnamaste.scripts.run_sim_study", start_time=start_time)  # type: ignore[no-untyped-call]


def parser() -> argparse.ArgumentParser:
    found = argparse.ArgumentParser(description=(__doc__ or "").strip().splitlines()[0])
    found.add_argument("study", help="a [tool.cnamaste.study.<name>] table, e.g. hd_tree")
    found.add_argument("--root", type=Path, required=True, help="the study directory, existing")
    found.add_argument("--methods", default=",".join(get_methods()), help="comma-separated, of cnamaste,calicost")
    found.add_argument("--max-samples", type=int, default=None, help="the first N samples in study order")
    found.add_argument("--solver", default=default_solver, help="cnamaste's <label_solver>-<initializer>-<decoder>")
    return found


def main(argv: list[str] | None = None) -> int:
    """Stream each sample not yet drawn under `ROOT`, in study order, through `--methods`.

    Returns:
        0; argparse's usage error for a method outside `get_methods()`.
    """
    command = parser()
    args = command.parse_args(argv)
    methods = tuple(args.methods.split(","))
    if not set(methods) <= set(get_methods()):
        command.error(f"--methods {args.methods}: of {','.join(get_methods())}")
    drawn = args.root / "simulated_data_related"
    for simulation, k in get_study_order(args.study)[: args.max_samples]:
        name = get_sample_name(simulation, k)
        if any(drawn.glob(f"{name}_*")):
            logger.info(f"{name}: under {drawn}, skipped")
            continue
        record = stream_study_sample(simulation, k, args.root, methods, args.solver, stop_calicost=False)
        logger.info(f"{record['sample_id']}: streamed through {', '.join(methods)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
