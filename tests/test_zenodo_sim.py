"""`run_cnamaste` end to end on CalicoST easy (`2d4ce9a9`) and hard (`8797710b`), as a fresh clone runs it: the
repository-root `zenodo_sim_config.yaml` with `test_zenodo_sim_config.yaml` laid over it, each fixture's own
`sample_sheet.tsv`, from the repository root.

Referee: the planted truth (`sim/<name>_<hash>/truth_clone_labels.tsv.gz`, `truth_acn_profile.tsv.gz`); the clone
ARI, integer clone ARI, CNA sensitivity, specificity and exact-pair recall (`cnamaste.qa.metrics`) equal the ones recorded under the
default label solver, to 4 decimals (`RECORDED_METRICS`); the phasing's segment count and total phase switches equal
the recorded ones (`RECORDED_PHASING`), as drift pins. Departure: the references are
`ref/` where it holds the four files, else `sims.grch38` (`$CNAMASTE_GRCH38`). About 100 s per fixture on 4 shared
cores (28-40 s measured alone); a real `run_cnamaste`, so the rows carry `end2end`.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from cnamaste import external
from cnamaste.phasing import cumulative_phase_switches
from cnamaste.qa.metrics import metrics, read_estimate
from cnamaste.qa.sim_runs import run_dir_of
from cnamaste.qa.sims import get_project, get_resources, grch38
from cnamaste.scripts.run_cnamaste_decode import block_phase, written
from cnamaste.segments import get_dropped
from cnamaste.truth import read_truth

OVERRIDE = Path(__file__).parent / "test_zenodo_sim_config.yaml"


REDRAWN = (
    "clones_genomic.pdf",
    "clones_spatial.pdf",
    "combined.pdf",
    "copy_number_profile.pdf",
    "real_clones_genomic.pdf",
    "real_combined.pdf",
)
"""The figures `run_cnamaste_plot` redraws: the `finalize` stage's."""


def get_override() -> Path:
    """Return the override, `OVERRIDE`."""
    return OVERRIDE


# NB CalicoST easy (`2d4ce9a9`) and hard (`8797710b`), label solver `opengm_expansion`, measured on the hmm_ab
#    decoder's alignment to `sec:ab_decoder` merged with `cleanup`: hard moved by T- #22's floor fix, easy not traced;
#    then on the emission stack (`a02ed43`): hard moved by the M-step gradient's log Z_c, easy not.
#    Then both by bb_logpmf_gradient on hmm_utils.digamma_difference (`claude/mstep-digamma-difference`).
#    Then both by the M step's JAX objective on class weights (T- #100, PR 3).
RECORDED_METRICS = {
    "easy_2d4ce9a9": {
        "clone_ari": 0.9860,
        "int_clone_ari": 0.9860,
        "sensitivity": 0.8216,
        "specificity": 0.9992,
        "exact_altered": 0.5833,
    },
    "hard_8797710b": {
        "clone_ari": 0.9741,
        "int_clone_ari": 0.9741,
        "sensitivity": 0.9229,
        "specificity": 0.9986,
        "exact_altered": 0.5076,
    },
}

# NB (refined segments, total phase switches), label solver `opengm_expansion`, measured on `7c0136d` (2026-10-30).
RECORDED_PHASING = {"easy_2d4ce9a9": (30, 11), "hard_8797710b": (39, 20)}


def get_phasing_summary(run_dir: Path) -> tuple[int, int]:
    """The phasing's segment count and total phase switches, from the run's `segmentation.tsv.gz`.

    Args: `run_dir` the run's directory.
    Returns: `(len(refined_lengths), switches)`, switches summed over contigs of `cumulative_phase_switches`' last bin.
    """
    segmentation = written(run_dir, "segmentation.tsv", "phase")
    phase_indicator, refined_lengths = block_phase(segmentation)
    rows = segmentation.rename(columns={"initial_blocks": "block_id", "phased_bins": "bin_id"})
    rows = rows[["CHR", "block_id", "bin_id"]].replace(get_dropped(), np.nan)
    switches = cumulative_phase_switches(rows, phase_indicator)
    contig = rows.dropna().groupby("bin_id")["CHR"].first().to_numpy()
    return len(refined_lengths), int(pd.Series(switches).groupby(contig, sort=False).last().sum())


def merged(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    """`base` with `over`'s keys set, section by section."""
    return {**base, **{k: {**base.get(k, {}), **v} if isinstance(v, dict) else v for k, v in over.items()}}


def references() -> dict[str, str] | None:
    """`references.*`: the override's `ref/` paths where `ref/` holds every file, else `sims.grch38`'s, else None."""
    if all((get_project() / "ref" / f).is_file() for f in get_resources().values()):
        return {}
    found = grch38()
    return None if found is None else {k: str(found / f) for k, f in get_resources().items()}


@pytest.mark.end2end
@pytest.mark.parametrize("fixture", sorted(RECORDED_METRICS))
def test_a_fixture_runs_from_the_repository_root_and_recovers_its_clones(tmp_path: Path, fixture: str) -> None:
    """`run_cnamaste` on the merged configuration, its output in `tmp_path`: the run exits 0 and its clones score
    `RECORDED_METRICS` against the planted truth, and its phasing `RECORDED_PHASING`. Fails where a sheet path, a
    reference or an override does not resolve from the repository root, or the run's clones or phasing move from the
    recorded ones."""
    if not external.available():
        pytest.skip(str(external.unavailable()))
    found = references()
    if found is None:
        pytest.skip("GRCh38 resources: neither ref/ nor $CNAMASTE_GRCH38 holds them")
    config = merged(
        yaml.safe_load((get_project() / "zenodo_sim_config.yaml").read_text()), yaml.safe_load(OVERRIDE.read_text())
    )
    config["references"].update(found)
    config["paths"]["sample_sheet"] = f"sim/{fixture}/sample_sheet.tsv"
    (tmp_path / "output").mkdir()
    config["paths"]["output_dir"] = str(tmp_path / "output")
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    start = time.perf_counter()
    done = subprocess.run(
        [sys.executable, "-m", "cnamaste.scripts.run_cnamaste", str(tmp_path / "config.yaml")],
        cwd=get_project(),
        capture_output=True,
        text=True,
        env=os.environ,
        check=False,
    )
    seconds = time.perf_counter() - start
    assert done.returncode == 0, done.stderr[-3000:]
    run_dir = run_dir_of(tmp_path / "output")
    found = metrics(read_estimate(run_dir), read_truth(get_project() / "sim" / fixture))
    phasing = get_phasing_summary(run_dir)
    print(
        f"{fixture}: {', '.join(f'{k} {v:.4f}' for k, v in found.items())}, (segments, phase switches) {phasing} "
        f"in {seconds:.1f} s"
    )
    # NB run_cnamaste_plot on the finished run: the finalize figures redrawn in place, every other file as it was.
    kept = {p: p.read_bytes() for p in run_dir.rglob("*") if p.is_file() and p.parent.name != "plots"}
    drawn = {p: p.stat().st_mtime_ns for p in (run_dir / "plots").iterdir()}
    replot = subprocess.run(
        [sys.executable, "-m", "cnamaste.scripts.run_cnamaste_plot", str(tmp_path / "config.yaml")],
        cwd=get_project(),
        capture_output=True,
        text=True,
        env=os.environ,
        check=False,
    )
    assert replot.returncode == 0, replot.stderr[-3000:]
    redrawn = {p.name for p, mtime in drawn.items() if p.stat().st_mtime_ns != mtime}
    assert redrawn == set(REDRAWN)
    assert all(p.read_bytes() == data for p, data in kept.items())
    assert (run_dir / f"cnamaste_plot_{run_dir.name.removeprefix('run_')}.log").is_file()
    assert {k: round(float(found[k]), 4) for k in RECORDED_METRICS[fixture]} == RECORDED_METRICS[fixture]
    assert phasing == RECORDED_PHASING[fixture]
