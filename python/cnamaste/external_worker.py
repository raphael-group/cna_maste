"""The worker `cnamaste.external.Session` starts: `python -m cnamaste.external_worker <library> <directory>`.

It loads the library once, writes `{"ready": true}`, then serves one JSON request per line on stdin: it maps the
request's blocks from `<directory>`, calls `cnamaste_opengm_solve`, flushes the outputs and replies with the energy,
bound and iterations. Replies go to a duplicate of stdout, and stdout to stderr, so the library's own output cannot
corrupt a reply. A library error is a reply (`status` 1); anything else ends the worker. It creates and removes no file.
"""

from __future__ import annotations

import ctypes
import json
import os
import sys

import numpy as np

ERROR_BYTES = 1024


def get_pointer(array: np.ndarray) -> ctypes.c_void_p:
    """Return `array`'s data pointer, as the library takes it."""
    return array.ctypes.data_as(ctypes.c_void_p)


def main() -> None:
    """Serve requests on stdin until it closes; `sys.argv` is the library and the blocks' directory."""
    solve = ctypes.CDLL(sys.argv[1]).cnamaste_opengm_solve
    directory = sys.argv[2]
    solve.restype = ctypes.c_int
    reply = os.fdopen(os.dup(1), "w")
    os.dup2(2, 1)
    reply.write(json.dumps({"ready": True}) + "\n")
    reply.flush()
    for line in sys.stdin:
        request = json.loads(line)
        arrays = {
            name: np.memmap(os.path.join(directory, named), dtype=np.dtype(dtype), mode="r+", shape=tuple(shape))
            for name, (named, dtype, shape) in request["blocks"].items()
        }
        value, bound, seconds, taken = ctypes.c_double(), ctypes.c_double(), ctypes.c_double(), ctypes.c_int64()
        error = ctypes.create_string_buffer(ERROR_BYTES)
        status = solve(
            ctypes.c_int(request["algorithm"]),
            ctypes.c_int(request["mincut"]),
            ctypes.c_int64(request["n_nodes"]),
            ctypes.c_int64(request["n_states"]),
            get_pointer(arrays["unary"]),
            ctypes.c_int64(request["n_edges"]),
            get_pointer(arrays["first"]),
            get_pointer(arrays["second"]),
            get_pointer(arrays["coupling"]),
            get_pointer(arrays["start"]) if "start" in arrays else None,
            ctypes.c_int64(request["max_iterations"]),
            ctypes.c_double(request["tolerance"]),
            get_pointer(arrays["labels"]),
            ctypes.byref(value),
            ctypes.byref(bound),
            get_pointer(arrays["trace"]),
            ctypes.byref(taken),
            ctypes.byref(seconds),
            error,
            ctypes.c_int64(ERROR_BYTES),
        )
        arrays["seconds"][0] = seconds.value
        for name in ("labels", "trace", "seconds"):
            arrays[name].flush()
        del arrays
        if status != 0:
            out = {"status": 1, "error": error.value.decode(errors="replace")}
        else:
            out = {"status": 0, "value": value.value, "bound": bound.value, "taken": taken.value}
        reply.write(json.dumps(out) + "\n")
        reply.flush()


if __name__ == "__main__":
    main()
