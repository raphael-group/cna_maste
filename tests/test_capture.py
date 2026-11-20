"""Each recorded call of CalicoST easy (`2d4ce9a9`) and hard (`8797710b`) replayed from its capture against its stored
output, floats to `qa.capture.FLOAT_RTOL` and all else exactly; and, under `end2end`, a fresh run's metrics against the
capture's `/metrics`.

`stage` is every call with a stored `out` (conftest.py `get_stage_params`): `run_hmrf` replays under
`end2end` only, no replay is over 5 s serial (3.4 s at most). A replay runs where the capture ran, in a scratch
directory linking the checkout's `sim/` and `ref/`. `pytest -m cache` runs every replay, `run_hmrf`'s included, to fill
the compile caches a run reads.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import cnamaste.scripts.run_cnamaste as rc
from cnamaste.fit_checks import get_fit_check_counts, get_fit_times
from cnamaste.qa.capture import (
    Capture,
    assert_same,
    get_entry_point_seed,
    get_output_dir,
    get_overrides,
    get_replay_file,
    get_template,
)
from cnamaste.qa.metrics import metrics, read_estimate
from cnamaste.qa.sim_runs import run_dir_of
from cnamaste.qa.sims import get_fixtures, get_project, get_supported_sims
from cnamaste.results import write_tsv
from cnamaste.truth import read_truth

METRIC_DECIMALS = 4
"""A fresh run's metrics agree with `/metrics` to this many decimal places."""


@pytest.mark.cache
def test_call_replays_its_output(sim_hash: str, stage: str, scratch: Path) -> None:
    """The call, from its recorded arguments, global config and generator states, returns its recorded output."""
    capture = Capture(get_project() / get_supported_sims()[sim_hash])
    try:
        function, args, kwargs = capture.decode(stage)
        if function is write_tsv:
            Path(args[0]).mkdir(parents=True, exist_ok=True)
        found = function(*args, **kwargs)
        with get_replay_file(found) as replayed:
            assert_same(capture.entry(stage)["out"], replayed["out"])
    finally:
        capture.close()


@pytest.mark.end2end
@pytest.mark.parametrize("fixture_hash", list(get_supported_sims()))
def test_fresh_run_metrics_are_the_captured(fixture_hash: str, scratch: Path) -> None:
    """A fresh `run_cnamaste` of the fixture scores the capture's `/metrics` to `METRIC_DECIMALS` places, wall times
    (`fit_checks.get_fit_times`) present and unscored."""
    capture = Capture(get_project() / get_supported_sims()[fixture_hash])
    try:
        recorded = {key: float(value) for key, value in capture.h5["metrics"].attrs.items()}
    finally:
        capture.close()
    np.random.seed(get_entry_point_seed())
    rc.run_cnamaste(str(get_template()), over_rides=get_overrides(get_fixtures()[fixture_hash]))
    sample = get_project() / "sim" / get_fixtures()[fixture_hash]
    found = metrics(read_estimate(run_dir_of(scratch / get_output_dir())), read_truth(sample)) | get_fit_check_counts()
    assert sorted(found) == sorted(recorded)
    for key, value in found.items():
        if key in get_fit_times():
            continue
        np.testing.assert_almost_equal(value, recorded[key], decimal=METRIC_DECIMALS, err_msg=key)
