"""`computed_once` under xdist: rows on different workers read one computation (T- #42).

Referee: a log every computation appends its worker and pid to, beside the result. Fails where a second worker
recomputes (two lines), or a reader returns before the computation's result is written.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from computed_once import computed_once, get_shared_dir

NAME = "computed_once_guard"


def record_computer(out: Path) -> None:
    """Write the computing worker and pid to `out/computed_by` and append them to the shared log, after 1 s so a
    second worker arrives while the first computes."""
    computer = f"{os.environ.get('PYTEST_XDIST_WORKER', 'main')} {os.getpid()}"
    with open(out.parent / f"{NAME}.log", "a") as log:
        log.write(computer + "\n")
    time.sleep(1.0)
    (out / "computed_by").write_text(computer)


@pytest.mark.parametrize("reader", ["first", "second"])
def test_every_worker_reads_the_one_computation(tmp_path_factory: pytest.TempPathFactory, reader: str) -> None:
    """Each reader sees `computed_by`, and the log holds exactly that one computation."""
    out = computed_once(tmp_path_factory, NAME, record_computer)
    computed_by = (out / "computed_by").read_text()
    log = (get_shared_dir(tmp_path_factory) / f"{NAME}.log").read_text().splitlines()
    assert log == [computed_by], f"{reader} on {os.environ.get('PYTEST_XDIST_WORKER', 'main')}"
