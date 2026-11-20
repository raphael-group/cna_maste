"""The fixtures `--sims` selects: `sim_hash` (with `stage`) and `staged_hash`; the `fixture_stage` marker; the CalicoST
referee's flags; and a warning where a row outgrows its `slow` / `very_slow` marker.

Every test that takes `sim_hash` (directly or through a fixture that takes it) runs once per selected
sample. The default is the dev tier, `DEV_SIMS`: CalicoST easy (`2d4ce9a9`) and hard (`8797710b`). `--sims=a,b`
selects others by hash, and `--sims=all` every staged fixture; a `TRUTH_SIMS` fixture is read by `test_sim_fixtures.py`
alone. Every test a staged fixture parametrizes, or a sample drawn from one, carries `fixture_stage`.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path

import h5py
import pytest
from _pytest.mark import ParameterSet

from cnamaste.qa.capture import stage_links
from cnamaste.qa.sims import get_fixtures, get_project, get_supported_sims, selected_sims
from cnamaste.sim.perfect import get_perfect

STAGED = {*get_fixtures(), *get_fixtures().values(), *get_perfect()}
"""Every name a staged fixture, or a perfect sample drawn from one, is parametrized by: its hash, its directory."""

SLOW_SECONDS = {"slow": 5.0, "very_slow": 20.0}
"""Each marker's threshold: a row over 5 s is `slow`, over 20 s also `very_slow`."""


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--sims",
        default="dev",
        help="'dev' (CalicoST easy and hard, the default), 'all', or fixture hashes, comma separated",
    )
    parser.addoption("--calicost-refresh", action="store_true", help="recompute and rewrite tests/calicost/.cache")
    parser.addoption("--calicost-live", action="store_true", help="call CalicoST, failing where the cache differs")


def asked(config: pytest.Config) -> list[str]:
    """Every fixture `--sims` names, `SUPPORTED_SIMS` and `TRUTH_SIMS` alike."""
    try:
        return selected_sims(config.getoption("--sims"))
    except ValueError as refused:
        raise pytest.UsageError(str(refused)) from None


def selected(config: pytest.Config) -> list[str]:
    """The `SUPPORTED_SIMS` fixtures `--sims` names: those `sim_hash` runs on."""
    return [h for h in asked(config) if h in get_supported_sims()]


def get_stage_params(sim_hash: str) -> list[ParameterSet]:
    """`sim_hash`'s recorded calls with a stored `out`, as `(sim_hash, stage)`, `run_hmrf` marked `end2end`.

    Args:
        sim_hash: the fixture hash.

    Returns:
        The parameters.
    """
    params = []
    with h5py.File(get_project() / get_supported_sims()[sim_hash], "r") as h5:
        for call in json.loads(h5["config"].attrs["stages"]):
            entry = h5[call]
            if "out" not in entry:
                continue
            marks = [pytest.mark.end2end] if entry.attrs["function"].endswith(":run_hmrf") else []
            params.append(pytest.param(sim_hash, call, marks=marks, id=f"{sim_hash}-{call}"))
    return params


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    hashes = selected(metafunc.config)
    if "stage" in metafunc.fixturenames:
        params = [p for h in hashes for p in get_stage_params(h)]
        metafunc.parametrize(("sim_hash", "stage"), params, scope="session")
    elif "sim_hash" in metafunc.fixturenames:
        metafunc.parametrize("sim_hash", hashes, scope="session")
    if "staged_hash" in metafunc.fixturenames:
        # NB `test_sim_fixtures.py`'s fixtures: every one `--sims` names, `TRUTH_SIMS` too, as it reads the staged file alone.
        metafunc.parametrize("staged_hash", asked(metafunc.config), scope="module")


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Every test a staged fixture parametrizes, by `--sims` or by name (`STAGED`), carries `fixture_stage`."""
    for item in items:
        params = item.callspec.params if hasattr(item, "callspec") else {}
        if (
            "sim_hash" in params
            or "staged_hash" in params
            or any(isinstance(v, str) and v in STAGED for v in params.values())
        ):
            item.add_marker(pytest.mark.fixture_stage)


OUTGROWN: list[str] = []
"""The rows `pytest_runtest_logreport` found over their marker's threshold, reported at the session's end."""


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    """Records a row whose call outgrows its marker: over 5 s unmarked, or over 20 s `slow` but not `very_slow`."""
    tier = "very_slow" if "slow" in report.keywords else "slow"
    if report.when == "call" and "very_slow" not in report.keywords and report.duration > SLOW_SECONDS[tier]:
        OUTGROWN.append(f"{report.nodeid}: {report.duration:.1f} s, over {SLOW_SECONDS[tier]:.0f} s; mark it {tier}")


def pytest_terminal_summary(terminalreporter: pytest.TerminalReporter) -> None:
    """Warns, without failing, of each row `pytest_runtest_logreport` recorded."""
    if OUTGROWN:
        terminalreporter.write_sep("=", "warning: rows over their slow marker's threshold", yellow=True)
        for line in OUTGROWN:
            terminalreporter.write_line(line)


@pytest.fixture(scope="session", autouse=True)
def _working_directory(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    """A scratch working directory, so nothing a replay writes relative to it lands in the checkout."""
    before = Path.cwd()
    os.chdir(tmp_path_factory.mktemp("cwd"))
    yield
    os.chdir(before)


@pytest.fixture
def scratch(tmp_path: Path) -> Iterator[Path]:
    """A scratch working directory as `stage_links` lays it out, the capture's own (`test_capture.py`,
    `test_oracle.py`)."""
    before = Path.cwd()
    stage_links(tmp_path)
    os.chdir(tmp_path)
    yield tmp_path
    os.chdir(before)
