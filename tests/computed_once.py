"""A result computed once per pytest session and read by every xdist worker (T- #42).

`xdist_group` cannot pin rows to one worker: xdist 3.8 renames `item._nodeid`, which pytest 9.2's `nodeid` no longer
reads. The first worker to take a file lock in the session's shared directory computes; the rest wait, then read.
"""

from __future__ import annotations

import fcntl
import os
from collections.abc import Callable
from pathlib import Path

import pytest


def get_shared_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """This session's directory every xdist worker sees: the workers' common parent, else the session's own."""
    base = tmp_path_factory.getbasetemp()
    return base.parent if os.environ.get("PYTEST_XDIST_WORKER") else base


def computed_once(tmp_path_factory: pytest.TempPathFactory, name: str, compute: Callable[[Path], object]) -> Path:
    """The directory `<shared>/<name>`, filled by `compute` on the first worker to take `<name>.lock`.

    Returns: the directory, once `compute` has returned; a computation that raised fails every later reader.
    """
    shared = get_shared_dir(tmp_path_factory)
    out, done = shared / name, shared / f"{name}.done"
    with open(shared / f"{name}.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not done.exists():
            assert not out.exists(), f"an earlier worker's computation failed: {out}"
            out.mkdir()
            compute(out)
            done.touch()
    return out
