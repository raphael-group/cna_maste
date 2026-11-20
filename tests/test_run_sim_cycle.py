"""study15's cycle: realizations drawn by `cnamaste.sim.draw.stream` (`vary = "truth"`, each its own clone count,
layout and tree after r0), `run_cnamaste` on each with the `config.yaml` its `[config]` writes, scored against its
truth by `cnamaste.qa.metrics.get_run_scores`.

Referee: each realization's planted truth. Writes `study15_cycle.tsv` (one row per realization: the scores, the
clones inferred and planted, normal counted in both, wall seconds) under `tmp_path` and prints it; 306 s for 3
realizations on this host. In the standard tier as `slow` and `very_slow`; skipped without OpenGM or GRCh38:
`pytest tests/test_run_sim_cycle.py -n 0 -s`.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import pytest

from cnamaste import external
from cnamaste.qa.metrics import get_run_scores
from cnamaste.qa.sim_runs import run_dir_of
from cnamaste.qa.sims import get_project, grch38
from cnamaste.sim.draw import stream
from cnamaste.sim.manifest import get_simulation

# NB clone ARI per realization under the default label solver, expansion with hmrf.opengm_patch (T- #195), rounded
#    down to 2 decimals: 0.8486, 0.9316, 0.4695 (`f701ebb`). OpenGM's moves gave 0.9005, 0.9347, 0.4614 (`1dc81ce`).
CLONE_ARI_FLOOR = {0: 0.84, 1: 0.93, 2: 0.46}

CYCLE_REALIZATIONS = 3
"""The study15 realizations run, those `CLONE_ARI_FLOOR` records."""


@pytest.mark.slow
@pytest.mark.very_slow
def test_study15_realizations_run_and_recover_their_clones(tmp_path: Path) -> None:
    """Each realization's run exits 0 and its clone ARI holds `CLONE_ARI_FLOOR`, where recorded."""
    if not external.available():
        pytest.skip(str(external.unavailable()))
    if grch38() is None:
        pytest.skip("no GRCh38 resources: ref/ or $CNAMASTE_GRCH38")
    simulation = get_simulation("study15")._replace(realizations=CYCLE_REALIZATIONS)
    rows = []
    for realization, sample in stream(simulation, None, tmp_path):
        assert sample is not None
        start = time.perf_counter()
        done = subprocess.run(
            [sys.executable, "-m", "cnamaste.scripts.run_cnamaste", str(sample / "config.yaml")],
            cwd=get_project(),
            capture_output=True,
            text=True,
            env=os.environ,
            check=False,
        )
        seconds = time.perf_counter() - start
        assert done.returncode == 0, done.stderr[-3000:]
        scores = get_run_scores(run_dir_of(sample / "output"), sample)
        assert scores["n_clones_true"] == len(realization.planted.clones)
        rows.append({"realization": realization.k, "sample": sample.name, **scores, "seconds": seconds})
    table = pd.DataFrame(rows)
    table.to_csv(tmp_path / "study15_cycle.tsv", sep="\t", index=False, float_format="%.4f")
    print(f"\n{table.to_string(index=False, float_format='%.4f')}\n{tmp_path / 'study15_cycle.tsv'}")
    for k, clone_ari in zip(table["realization"], table["clone_ari"], strict=True):
        assert k not in CLONE_ARI_FLOOR or clone_ari >= CLONE_ARI_FLOOR[k]
