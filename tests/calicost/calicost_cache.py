"""CalicoST's output on each comparison's input, computed once and recorded in `tests/calicost/.cache/`, one JSON
entry per call.

An entry is keyed by CalicoST's revision and the sha256 of every input, the call's arguments among them. A hit returns
the recorded output and calls nothing of CalicoST; a miss, or a key that differs, calls CalicoST and writes the entry.
An output compared exactly is recorded as the sha256 of its canonical bytes (`get_digest`); one the test reads, or
compares within a tolerance, is recorded whole, zlib-compressed. Each entry records its revision, input hashes and date.

`--calicost-refresh` recomputes and rewrites every entry, keeping an entry's date where its content is unchanged, so
a refresh on a deterministic referee leaves the cache byte for byte. `--calicost-live` calls CalicoST on every row and
fails where its output is not the one recorded.
"""

from __future__ import annotations

import base64
import builtins
import datetime
import functools
import hashlib
import json
import os
import re
import tempfile
import zlib
from collections.abc import Callable, Collection
from importlib.metadata import distribution
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

CACHE = Path(__file__).parent / ".cache"
REVISION: str = json.loads(distribution("calicost").read_text("direct_url.json") or "{}")["vcs_info"]["commit_id"]
"""CalicoST's installed revision, the commit `uv.lock` pins."""

MODES = ("cached", "refresh", "live")

Referee = Callable[..., tuple[Any, ...]]


@functools.cache
def get_file_digest(path: Path) -> str:
    """The sha256 of `path`'s bytes, hex, read once per process (the GRCh38 map is 34 MB)."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def update_digest(digest: Any, value: Any) -> None:
    """Feeds `value`'s canonical bytes to `digest`: an array's dtype, shape and C-order bytes; a frame's columns and
    each column's values; a file's bytes; a sequence's length and items; a mapping's sorted items; a scalar's `repr`.
    Anything with `toarray` is its array; an object with no `repr` of its own raises `TypeError`."""
    if isinstance(value, pd.DataFrame):
        digest.update(json.dumps([str(c) for c in value.columns]).encode())
        for column in value.columns:
            update_digest(digest, value[column].to_numpy())
    elif sp.issparse(value) or hasattr(value, "toarray"):
        update_digest(digest, value.toarray())
    elif isinstance(value, dict):
        digest.update(f"dict{len(value)}".encode())
        for key in sorted(value):
            update_digest(digest, key)
            update_digest(digest, value[key])
    elif isinstance(value, np.ndarray) and value.dtype == object:
        digest.update(f"object{value.shape}".encode() + json.dumps(value.tolist(), default=str).encode())
    elif isinstance(value, np.ndarray):
        digest.update(f"{value.dtype.str}{value.shape}".encode() + np.ascontiguousarray(value).tobytes())
    elif isinstance(value, Path):
        digest.update(b"file" + get_file_digest(value).encode())
    elif isinstance(value, list | tuple):
        digest.update(f"{type(value).__name__}{len(value)}".encode())
        for item in value:
            update_digest(digest, item)
    elif type(value).__repr__ is object.__repr__:
        raise TypeError(f"{type(value).__name__} has no canonical bytes: its repr is its address")
    else:
        digest.update(repr(value.item() if isinstance(value, np.generic) else value).encode())


def get_digest(value: Any) -> str:
    """The sha256 of `value`'s canonical bytes (`update_digest`), hex."""
    digest = hashlib.sha256()
    update_digest(digest, value)
    return digest.hexdigest()


def encode(value: Any) -> Any:
    """`value` as JSON: an array as dtype, shape and zlib-compressed bytes, a tuple tagged, a list as a list."""
    if isinstance(value, np.ndarray):
        raw = zlib.compress(np.ascontiguousarray(value).tobytes(), 9)
        return {"dtype": value.dtype.str, "shape": list(value.shape), "zlib": base64.b64encode(raw).decode()}
    if isinstance(value, tuple):
        return {"tuple": [encode(v) for v in value]}
    if isinstance(value, list):
        return [encode(v) for v in value]
    return value.item() if isinstance(value, np.generic) else value


def decode(value: Any) -> Any:
    """`encode`'s inverse."""
    if isinstance(value, dict) and "zlib" in value:
        raw = zlib.decompress(base64.b64decode(value["zlib"]))
        return np.frombuffer(raw, dtype=value["dtype"]).reshape(value["shape"]).copy()
    if isinstance(value, dict) and "tuple" in value:
        return tuple(decode(v) for v in value["tuple"])
    if isinstance(value, list):
        return [decode(v) for v in value]
    return value


def get_entry(call: Callable[[], Any], inputs: list[str], kept: Collection[int]) -> dict[str, Any]:
    """CalicoST's `call`, its outputs digested but for `kept`, or the builtin exception it raised."""
    entry: dict[str, Any] = {"revision": REVISION, "inputs": inputs}
    try:
        called = call()
        outputs = called if isinstance(called, tuple) else (called,)
    except Exception as raised:
        if getattr(builtins, type(raised).__name__, None) is not type(raised):
            raise
        entry["raised"] = [type(raised).__name__, str(raised.args[0]) if raised.args else ""]
    else:
        entry["outputs"] = [encode(v) if i in kept else get_digest(v) for i, v in enumerate(outputs)]
    return entry


def write_entry(path: Path, entry: dict[str, Any]) -> None:
    """`entry` at `path`, atomically, so workers that miss together write one whole file."""
    path.parent.mkdir(exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, suffix=".tmp") as handle:
        handle.write(json.dumps(entry, indent=1) + "\n")
    os.chmod(handle.name, 0o644)
    os.replace(handle.name, path)


def get_referee(
    name: str, inputs: tuple[Any, ...], call: Callable[[], Any], kept: Collection[int] = (), *, mode: str
) -> tuple[Any, ...]:
    """CalicoST's outputs of `call` on `inputs`, from the cache where its key holds, else computed and recorded.

    Args:
        name: the entry's name, unique per call and parametrization.
        inputs: everything `call` reads, its arguments among them; a `Path` stands for its file's bytes.
        call: CalicoST's call, returning a tuple of outputs, or one output.
        kept: the indices of the outputs recorded whole; the others are recorded and returned as `get_digest`s.
        mode: one of `MODES`.

    Returns: the outputs, each whole or as its digest. Raises the builtin exception CalicoST raised, as recorded.
    """
    path = CACHE / f"{re.sub(r'[^A-Za-z0-9._=-]+', '_', name)}.json"
    key = [get_digest(v) for v in inputs]
    recorded: dict[str, Any] = json.loads(path.read_text()) if path.exists() else {}
    hit = recorded.get("revision") == REVISION and recorded.get("inputs") == key
    if hit and mode == "cached":
        entry = recorded
    else:
        entry = get_entry(call, key, set(kept))
        unchanged = hit and {k: v for k, v in recorded.items() if k != "date"} == entry
        if mode == "live" and hit and not unchanged:
            pytest.fail(f"{path.name}: CalicoST's output is not the one recorded; --calicost-refresh rewrites it")
        if unchanged:
            entry = recorded
        elif mode != "live":
            write_entry(path, {**entry, "date": datetime.datetime.now(datetime.UTC).date().isoformat()})
    if "raised" in entry:
        kind, message = entry["raised"]
        raise getattr(builtins, kind)(message)
    return tuple(decode(v) for v in entry["outputs"])


def departs(reason: str, raises: type[Exception] = AssertionError) -> pytest.MarkDecorator:
    """A stated departure from CalicoST: the equality fails with `raises`, strictly, and the row is
    `calicost_departure` (conftest.py); any other exception is an error."""
    return pytest.mark.xfail(strict=True, raises=raises, reason=reason)
