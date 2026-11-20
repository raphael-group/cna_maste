"""cnamaste's alpha expansion of the Potts energy, each move a Boykov-Kolmogorov min cut: `infra/expansion`'s library,
built by `infra/expansion/build_expansion.sh` under `$CNAMASTE_EXPANSION_HOME` and called in process through ctypes.

It makes OpenGM's moves (`external`'s `expansion` with `boykov_kolmogorov`) on flat arrays: the same labels in the same
order, the same capacities and the same cut, and the same stop. A library that is not built raises `RuntimeError`,
never a fallback to another solver.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Any

import numpy as np

from cnamaste.external import OpenGMResult

HOME_VARIABLE = "CNAMASTE_EXPANSION_HOME"


def get_home_variable() -> str:
    """Return the environment variable naming the library's directory, "CNAMASTE_EXPANSION_HOME"."""
    return HOME_VARIABLE


LIBRARY = "libcnamaste_expansion.so"

LOADED: list[ctypes.CDLL] = []
"""The library, loaded on the first solve."""

INT64_POINTER = ctypes.POINTER(ctypes.c_int64)
DOUBLE_POINTER = ctypes.POINTER(ctypes.c_double)


def library() -> Path | None:
    """The built library under `$CNAMASTE_EXPANSION_HOME`, or None where it is unset or holds no build.

    Returns:
        The library's path, or None.
    """
    found = os.environ.get(get_home_variable())
    path = Path(found) / LIBRARY if found else None
    return path if path is not None and path.is_file() else None


def available() -> bool:
    """Whether the library is built under `$CNAMASTE_EXPANSION_HOME`."""
    return library() is not None


def unavailable() -> RuntimeError:
    """The error for a library that is not built."""
    return RuntimeError(
        f"cnamaste's expansion is not built: set ${HOME_VARIABLE} to a directory and run infra/expansion/build_expansion.sh"
    )


def require(solver: str) -> None:
    """Stop where `solver` is `expansion` or `trws` (the same library) and the library is not built: called when a run
    starts, as `external.require`.

    Args:
        solver: `hmrf.label_solver`.

    Raises:
        RuntimeError: from `unavailable`.
    """
    if solver in ("expansion", "trws") and not available():
        raise unavailable()


def get_library() -> ctypes.CDLL:
    """The loaded library, its three entry points typed.

    Returns:
        The `CDLL`.

    Raises:
        RuntimeError: from `unavailable`.
    """
    if not LOADED:
        built = library()
        if built is None:
            raise unavailable()
        lib = ctypes.CDLL(str(built))
        lib.cnamaste_expansion_solve.restype = ctypes.c_int
        lib.cnamaste_expansion_solve.argtypes = [
            ctypes.c_int64,
            ctypes.c_int64,
            DOUBLE_POINTER,
            ctypes.c_int64,
            INT64_POINTER,
            INT64_POINTER,
            DOUBLE_POINTER,
            INT64_POINTER,
            ctypes.c_int64,
            ctypes.c_int64,
            INT64_POINTER,
            DOUBLE_POINTER,
            INT64_POINTER,
            DOUBLE_POINTER,
            ctypes.c_char_p,
            ctypes.c_int64,
        ]
        lib.cnamaste_trws_solve.restype = ctypes.c_int
        lib.cnamaste_trws_solve.argtypes = [
            ctypes.c_int64,
            ctypes.c_int64,
            DOUBLE_POINTER,
            ctypes.c_int64,
            INT64_POINTER,
            INT64_POINTER,
            DOUBLE_POINTER,
            ctypes.c_int64,
            ctypes.c_double,
            INT64_POINTER,
            DOUBLE_POINTER,
            DOUBLE_POINTER,
            DOUBLE_POINTER,
            INT64_POINTER,
            DOUBLE_POINTER,
            ctypes.c_char_p,
            ctypes.c_int64,
        ]
        lib.cnamaste_expansion_min_cut.restype = ctypes.c_int
        lib.cnamaste_expansion_min_cut.argtypes = [
            ctypes.c_int64,
            ctypes.c_int64,
            INT64_POINTER,
            INT64_POINTER,
            DOUBLE_POINTER,
            INT64_POINTER,
            DOUBLE_POINTER,
            ctypes.c_char_p,
            ctypes.c_int64,
        ]
        LOADED.append(lib)
    loaded: ctypes.CDLL = LOADED[0]
    return loaded


def pointer(values: np.ndarray, kind: Any) -> Any:
    """Return `values`' data as a ctypes pointer of `kind`."""
    return values.ctypes.data_as(kind)


def solve(
    problem: Any, start: np.ndarray | None = None, max_iterations: int = 1000, opengm_patch: bool = False
) -> OpenGMResult:
    """Expand `problem` (a `cnamaste.potts.PottsProblem`) from `start`, every node at 0 where None.

    Args:
        problem: a `PottsProblem`.
        start: labels.
        max_iterations: the moves at most, as OpenGM's expansion counts them.
        opengm_patch: the moves with T- #195's auxiliary node, else OpenGM's (`hmrf.opengm_patch`).

    Returns:
        The `OpenGMResult`, its bound `-inf` and its trace empty: expansion keeps no bound.

    Raises:
        RuntimeError: where the library is not built or refuses the problem.
    """
    lib = get_library()
    n, k = problem.unary.shape
    unary = np.ascontiguousarray(problem.unary, dtype=np.float64)
    first = np.ascontiguousarray(problem.first, dtype=np.int64)
    second = np.ascontiguousarray(problem.second, dtype=np.int64)
    coupling = np.ascontiguousarray(problem.coupling, dtype=np.float64)
    begin = None if start is None else np.ascontiguousarray(start, dtype=np.int64)
    labels = np.zeros(n, dtype=np.int64)
    value, steps, seconds = ctypes.c_double(), ctypes.c_int64(), ctypes.c_double()
    error = ctypes.create_string_buffer(1024)
    status = lib.cnamaste_expansion_solve(
        n,
        k,
        pointer(unary, DOUBLE_POINTER),
        first.size,
        pointer(first, INT64_POINTER),
        pointer(second, INT64_POINTER),
        pointer(coupling, DOUBLE_POINTER),
        None if begin is None else pointer(begin, INT64_POINTER),
        max(int(max_iterations), 1),
        int(bool(opengm_patch)),
        pointer(labels, INT64_POINTER),
        ctypes.byref(value),
        ctypes.byref(steps),
        ctypes.byref(seconds),
        error,
        len(error),
    )
    if status != 0:
        raise RuntimeError(f"cnamaste's expansion refused the problem: {error.value.decode()}")
    return OpenGMResult(labels, value.value, -np.inf, np.zeros(0), int(steps.value))


def min_cut(n_nodes: int, tail: np.ndarray, head: np.ndarray, capacity: np.ndarray) -> tuple[np.ndarray, float]:
    """One min s-t cut, node 0 the source and node 1 the sink, as `cnamaste_opengm.cxx`'s `cnamaste_min_cut` poses it.

    Args:
        n_nodes: the nodes, both terminals among them.
        tail: each directed edge's tail.
        head: each directed edge's head.
        capacity: each directed edge's capacity.

    Returns:
        `sink_side` per node (1 where it reaches the sink in the residual graph) and the cut's capacity.
    """
    lib = get_library()
    tail, head = np.ascontiguousarray(tail, dtype=np.int64), np.ascontiguousarray(head, dtype=np.int64)
    capacity = np.ascontiguousarray(capacity, dtype=np.float64)
    side, value = np.zeros(n_nodes, dtype=np.int64), ctypes.c_double()
    error = ctypes.create_string_buffer(1024)
    status = lib.cnamaste_expansion_min_cut(
        n_nodes,
        tail.size,
        pointer(tail, INT64_POINTER),
        pointer(head, INT64_POINTER),
        pointer(capacity, DOUBLE_POINTER),
        pointer(side, INT64_POINTER),
        ctypes.byref(value),
        error,
        len(error),
    )
    if status != 0:
        raise RuntimeError(f"cnamaste's min cut refused the graph: {error.value.decode()}")
    return side, value.value
