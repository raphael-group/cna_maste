"""cnamaste's TRW-S of the Potts energy (T- #198): `infra/expansion`'s library, `trws.rs`, a port of OpenGM's `TRWSi` as
`external`'s `trws` runs it, called in process through ctypes. Built and found as `cnamaste.expansion`'s library.
"""

from __future__ import annotations

import ctypes
from typing import Any

import numpy as np

from cnamaste.expansion import DOUBLE_POINTER, INT64_POINTER, get_library, pointer
from cnamaste.external import OpenGMResult


def solve(problem: Any, max_iterations: int = 1000, tolerance: float = 1e-9) -> OpenGMResult:
    """TRW-S on `problem` (a `cnamaste.potts.PottsProblem`); it takes no start.

    Args:
        problem: a `PottsProblem`.
        max_iterations: the passes after the forward one at most, as OpenGM's `TRWSi` counts them.
        tolerance: the relative gap, and the relative rise of the bound in a pass, it stops at.

    Returns:
        The `OpenGMResult`: the best labelling, its energy in the Potts form, the dual bound, the bound after the
        forward pass and each pass, and the entries of that trace, as `external`'s `trws` returns them.

    Raises:
        RuntimeError: where the library is not built or refuses the problem.
    """
    lib = get_library()
    n, k = problem.unary.shape
    unary = np.ascontiguousarray(problem.unary, dtype=np.float64)
    first = np.ascontiguousarray(problem.first, dtype=np.int64)
    second = np.ascontiguousarray(problem.second, dtype=np.int64)
    coupling = np.ascontiguousarray(problem.coupling, dtype=np.float64)
    steps = max(int(max_iterations), 1)
    labels, trace = np.zeros(n, dtype=np.int64), np.zeros(steps, dtype=np.float64)
    value, bound, taken, seconds = ctypes.c_double(), ctypes.c_double(), ctypes.c_int64(), ctypes.c_double()
    error = ctypes.create_string_buffer(1024)
    status = lib.cnamaste_trws_solve(
        n,
        k,
        pointer(unary, DOUBLE_POINTER),
        first.size,
        pointer(first, INT64_POINTER),
        pointer(second, INT64_POINTER),
        pointer(coupling, DOUBLE_POINTER),
        steps,
        float(tolerance),
        pointer(labels, INT64_POINTER),
        ctypes.byref(value),
        ctypes.byref(bound),
        pointer(trace, DOUBLE_POINTER),
        ctypes.byref(taken),
        ctypes.byref(seconds),
        error,
        len(error),
    )
    if status != 0:
        raise RuntimeError(f"cnamaste's TRW-S refused the problem: {error.value.decode()}")
    return OpenGMResult(labels, value.value, bound.value, trace[: taken.value].copy(), int(taken.value))
