"""Marks each row under `tests/calicost` that takes a `referee` `calicost_departure` where it is a stated departure (a
strict xfail, `departs`), else `calicost`; its referee is CalicoST `c1abcae`'s output on the same input, recorded once
(`calicost_cache`). A row without one compares nothing with CalicoST and stays in the default tier."""

from __future__ import annotations

import functools
from pathlib import Path

import pytest

from calicost_cache import Referee, get_referee

HERE = Path(__file__).parent


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if HERE in item.path.parents and "referee" in getattr(item, "fixturenames", ()):
            strict = any(mark.kwargs.get("strict") for mark in item.iter_markers("xfail"))
            item.add_marker(pytest.mark.calicost_departure if strict else pytest.mark.calicost)


@pytest.fixture(scope="session")
def referee(pytestconfig: pytest.Config) -> Referee:
    """`get_referee` in the mode `--calicost-refresh` or `--calicost-live` sets, else from the cache."""
    refresh, live = pytestconfig.getoption("--calicost-refresh"), pytestconfig.getoption("--calicost-live")
    if refresh and live:
        raise pytest.UsageError("--calicost-refresh and --calicost-live are exclusive")
    return functools.partial(get_referee, mode="refresh" if refresh else "live" if live else "cached")
