"""`run_cnamaste` end to end on tiled CalicoST easy (`2d4ce9a9`): `side x side` copies of its spots, each copy's counts
redrawn (`sim.tile`), timed per stage with the run's peak memory, its clone ARI held above a recorded floor.

The samples are drawn on demand into `sim/.stress/` (gitignored), named by hash and checked against `STRESS_HASHES`.
Each row writes `sim/.stress/results/<name>.json` (spots, wall, per-stage seconds, peak MB, clone ARI), which
`tests/scripts/estimate_runtime.py` fits. Off by default: `pytest -m cache` first, so no compile is timed, then
`pytest tests/test_stress.py -m stress -p no:xdist`.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

from cnamaste import external
from cnamaste.qa.metrics import metrics, read_estimate
from cnamaste.qa.sim_runs import run_dir_of
from cnamaste.qa.sims import get_fixture_hash, get_project
from cnamaste.results import run_log_name
from cnamaste.sim.tile import write_tiled_sample
from cnamaste.sim.write import get_made_dir
from cnamaste.truth import read_truth
from test_zenodo_sim import get_override, merged, references

STRESS_SOURCE = "easy_2d4ce9a9"
"""The fixture tiled."""

STRESS_SEED = 0
"""The tiles' draw seed."""

STRESS_HASHES = {1: "6bb8b91a", 2: "e210e853", 4: "fe822d5b"}
"""Tiles per side -> the tiled sample's hash: 3,000, 12,000 and 48,000 spots."""

STRESS_CLONE_ARI_FLOOR = {1: 0.95, 2: 0.95, 4: 0.97}
"""Tiles per side -> the clone ARI floor: the first measured, rounded down to 2 decimals."""

STAGE_LINE = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) - [\d.]+m - \w+ \(([^)]*)\)")
"""A run log line's timestamp and stage, `logger.runtime_phase`."""


def get_stress_dir() -> Path:
    """Return the tiled samples' directory, `sim/.stress/`, made where missing."""
    return get_made_dir(get_project() / "sim", ".stress", exist_ok=True)


def get_tiled_sample(side: int) -> Path:
    """The tiled sample at `side`, drawn on first use and reused while its hash holds.

    Args:
        side: tiles per side.

    Returns:
        The sample's directory.
    """
    name = f"easy_x{side**2}"
    for found in sorted(get_stress_dir().glob(f"{name}_*")):
        if found.name.endswith("_staging"):
            continue
        if found.name == f"{name}_{get_fixture_hash(found)}":
            return found
    return write_tiled_sample(get_project() / "sim" / STRESS_SOURCE, side, STRESS_SEED, get_stress_dir(), name)


def get_stage_seconds(log: Path) -> dict[str, float]:
    """Seconds per stage, from the first line of each stage to the first line of the next.

    Args:
        log: the run's log.

    Returns:
        Stage -> seconds, in run order.
    """
    firsts: list[tuple[str, float]] = []
    for line in log.read_text().splitlines():
        found = STAGE_LINE.match(line)
        if found is None:
            continue
        stamp = time.mktime(time.strptime(found.group(1), "%Y-%m-%d %H:%M:%S"))
        if not firsts or firsts[-1][0] != found.group(2):
            firsts.append((found.group(2), stamp))
        last = stamp
    ends = [t for _, t in firsts[1:]] + [last]
    return {stage: end - start for (stage, start), end in zip(firsts, ends, strict=True)}


@pytest.mark.stress
@pytest.mark.parametrize("side", sorted(STRESS_HASHES))
def test_a_tiled_easy_run_is_timed_and_recovers_its_clones(tmp_path: Path, side: int) -> None:
    """`run_cnamaste` on tiled easy, the shipped configuration: exits 0, the sample's hash is `STRESS_HASHES`', and
    its clone ARI against the tiled truth is at least `STRESS_CLONE_ARI_FLOOR`'s; wall, stages and peak memory are
    written to `sim/.stress/results/`."""
    if not external.available():
        pytest.skip(str(external.unavailable()))
    found = references()
    if found is None:
        pytest.skip("GRCh38 resources: neither ref/ nor $CNAMASTE_GRCH38 holds them")
    sample = get_tiled_sample(side)
    config = merged(
        yaml.safe_load((get_project() / "zenodo_sim_config.yaml").read_text()),
        yaml.safe_load(get_override().read_text()),
    )
    config["references"].update(found)
    config["paths"]["sample_sheet"] = str(sample / "sample_sheet.tsv")
    config["paths"]["output_dir"] = str(tmp_path / "output")
    (tmp_path / "output").mkdir()
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))

    start = time.perf_counter()
    child = subprocess.Popen(
        [sys.executable, "-m", "cnamaste.scripts.run_cnamaste", str(tmp_path / "config.yaml")],
        cwd=get_project(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    _, status, usage = os.wait4(child.pid, 0)
    wall = time.perf_counter() - start
    assert os.waitstatus_to_exitcode(status) == 0, child.stderr.read()[-3000:] if child.stderr else ""

    run_dir = run_dir_of(tmp_path / "output")
    clone_ari = float(metrics(read_estimate(run_dir), read_truth(sample))["clone_ari"])
    record = {
        "sample": sample.name,
        "spots": json.loads((sample / "manifest.json").read_text())["spots"],
        "wall_seconds": round(wall, 1),
        "stage_seconds": get_stage_seconds(run_dir / run_log_name(run_dir.name.removeprefix("run_"))),
        "peak_mb": round(usage.ru_maxrss / 1024),
        "clone_ari": round(clone_ari, 4),
    }
    (get_stress_dir() / "results").mkdir(exist_ok=True)
    (get_stress_dir() / "results" / f"{sample.name}.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record))

    assert sample.name.rsplit("_", 1)[1] == STRESS_HASHES[side] or STRESS_HASHES[side] is None
    assert STRESS_CLONE_ARI_FLOOR[side] is None or clone_ari >= STRESS_CLONE_ARI_FLOOR[side]
