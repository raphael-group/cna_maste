"""
Draw a `cnamaste.sim` draw's truth pages, `truth_combined.pdf` and `truth_combined_spatial.pdf` (`cnamaste.sim.plot`).

Usage:
    uv run run_sim_plot sim/generated/dev_tree_easy_e4bbe03f [--into DIR]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from cnamaste.sim.plot import write_truth_pages
from cnamaste.truth import read_sim_truth


def parser() -> argparse.ArgumentParser:
    found = argparse.ArgumentParser(description=(__doc__ or "").strip().splitlines()[0])
    found.add_argument("sample", type=Path, help="a draw's directory, <name>_<hash>, as cnamaste.sim writes it")
    found.add_argument("--into", type=Path, default=None, help="the pages' directory; the draw's qa/ where omitted")
    return found


def main(argv: list[str] | None = None) -> int:
    """Write both pages, printing their paths.

    Returns:
        0.
    """
    args = parser().parse_args(argv)
    for page in write_truth_pages(read_sim_truth(args.sample), args.into):
        print(page)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
