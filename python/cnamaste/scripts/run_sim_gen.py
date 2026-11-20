"""
Draw a perfect sample from a fixture's planted truth, named by its hash: `OUT/<name>_<hash>/` (`cnamaste.sim.perfect`).

Usage:
    uv run run_sim_gen {perfect_easy,perfect_hard} [--depth-factor 100] [--seed 0] [--out DIR]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from cnamaste.config import YAMLConfig, set_global_config
from cnamaste.qa.sims import fixture, grch38
from cnamaste.sim.perfect import draw_sample, get_commit, get_config, get_perfect, get_source, write_sample


def parser() -> argparse.ArgumentParser:
    found = argparse.ArgumentParser(description=(__doc__ or "").strip().splitlines()[0])
    found.add_argument("name", choices=sorted(get_perfect()), help="the perfect sample, named by its source fixture")
    found.add_argument("--depth-factor", type=float, default=100.0, help="multiple of the source's mean depths")
    found.add_argument("--seed", type=int, default=0, help="the draw's seed")
    found.add_argument("--out", type=Path, default=Path("."), help="existing parent directory of <name>_<hash>/")
    return found


def main(argv: list[str] | None = None) -> int:
    """Draw and write the sample, printing its directory and manifest.

    Returns:
        0; argparse's usage error where no GRCh38 resources are found.
    """
    command = parser()
    args = command.parse_args(argv)
    references = grch38()
    if references is None:
        command.error("no GRCh38 resources: ref/ or $CNAMASTE_GRCH38")

    started = time.perf_counter()
    config = YAMLConfig.from_file(get_config())
    # NB recomb.compute_numbat_phase_switch_prob reads min_prob from the global config.
    set_global_config(config)
    source = get_source(fixture(get_perfect()[args.name].source))
    sample = draw_sample(source, args.depth_factor, args.seed, references, config)
    manifest = {
        "name": args.name,
        "source": get_perfect()[args.name].source,
        "depth_factor": args.depth_factor,
        "seed": args.seed,
        "mu0": sample.mu0,
        "snp_depth": sample.snp_depth,
        "switch_model": "recomb.get_sitewise_transmat, one block per SNP",
        "phasing": {k: getattr(config.phasing, k) for k in ("nu", "logphase_shift", "min_prob")},
        "expected_switches": float(sample.switch_probs[:-1].sum()),
        "commit": get_commit(),
    }
    directory = write_sample(source, sample, args.out, args.name, manifest)
    print(directory)
    print((directory / "manifest.json").read_text(), end="")
    print(f"{time.perf_counter() - started:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
