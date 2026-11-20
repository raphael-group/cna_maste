"""
Fit the normal baseline and coverage laws on CalicoST samples' normal spots (`cnamaste.sim.normal`).

Usage:
    uv run run_sim_normal sim/easy_2d4ce9a9 sim/hard_8797710b --into DIR

`DIR` exists and holds neither file; compare them with `sim/reference/`.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from cnamaste.qa.sims import get_resources, grch38
from cnamaste.sim.normal import (
    fit_baseline,
    fit_coverage,
    fit_entry_laws,
    get_baseline_text,
    get_coverage_text,
    get_pooled_counts,
    get_source_name,
)
from cnamaste.sim.write import write_reference


def parser() -> argparse.ArgumentParser:
    found = argparse.ArgumentParser(description=(__doc__ or "").strip().splitlines()[0])
    found.add_argument("samples", nargs="+", type=Path, help="CalicoST sample directories, <name>_<hash>")
    found.add_argument("--into", type=Path, required=True, help="an existing directory for both files")
    return found


def main(argv: list[str] | None = None) -> int:
    """Fit and write both files, printing their paths.

    Returns:
        0; argparse's usage error where no GRCh38 resources are found.
    """
    command = parser()
    args = command.parse_args(argv)
    references = grch38()
    if references is None:
        command.error("no GRCh38 resources: ref/ or $CNAMASTE_GRCH38")
    started = time.perf_counter()
    gene_table = references / get_resources()["hgtable_file"]
    pooled = get_pooled_counts(args.samples)
    baseline = fit_baseline(pooled, gene_table)
    laws = fit_coverage(pooled)
    laws |= fit_entry_laws(pooled, baseline, laws)
    source = get_source_name(args.samples)
    written = write_reference(
        get_baseline_text(baseline, source, gene_table), get_coverage_text(laws, source), args.into
    )
    print(f"wrote {', '.join(map(str, written))} in {time.perf_counter() - started:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
