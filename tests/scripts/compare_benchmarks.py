"""Compare `pytest --benchmark-json` runs by median: exit 1 where any evaluator's median exceeds `ratio` x baseline.

Usage: `python tests/scripts/compare_benchmarks.py <base.json[,base2.json]> <head.json[,head2.json]> [ratio, 1.05]`.
Several runs per side, comma separated, take each evaluator's least median: the VM's load only adds time.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def get_medians(paths: list[Path]) -> dict[str, float]:
    """Each benchmark's least median wall time in seconds over `paths`, by its test parameters.

    Args:
        paths: `--benchmark-json` files.

    Returns:
        The medians, keyed `evaluator-fixture`.
    """
    medians: dict[str, float] = {}
    for path in paths:
        for run in json.loads(path.read_text())["benchmarks"]:
            median = float(run["stats"]["median"])
            medians[run["param"]] = min(median, medians.get(run["param"], median))
    return medians


def main(baseline: list[Path], head: list[Path], ratio: float) -> int:
    """Print base, head and head / base per evaluator; return 1 where a ratio exceeds `ratio` or a row is missing.

    Args:
        baseline: the baseline's json files.
        head: the head's json files.
        ratio: the largest accepted head / base median.

    Returns:
        The exit status.
    """
    base, new = get_medians(baseline), get_medians(head)
    failed = sorted(set(base) ^ set(new))
    print(f"{'evaluator-fixture':40s} {'base ms':>9s} {'head ms':>9s} {'head/base':>9s}")
    for name in sorted(set(base) & set(new)):
        print(f"{name:40s} {1e3 * base[name]:9.3f} {1e3 * new[name]:9.3f} {new[name] / base[name]:9.3f}")
        if new[name] > ratio * base[name]:
            failed.append(name)
    if failed:
        print(f"over {ratio} x baseline or missing: {failed}")
    return int(bool(failed))


if __name__ == "__main__":
    base, head = ([Path(each) for each in arg.split(",")] for arg in sys.argv[1:3])
    sys.exit(main(base, head, float(sys.argv[3]) if len(sys.argv) > 3 else 1.05))
