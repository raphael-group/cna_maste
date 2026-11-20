"""OpenGM's Potts solvers, run in a subprocess one session at a time.

`Session` starts `cnamaste.external_worker` on the library `infra/opengm/build_opengm.sh` builds under
`$CNAMASTE_OPENGM_HOME`, and passes each problem's arrays as memory-mapped files. A library that is not built raises
`RuntimeError` before any subprocess, never a fallback to another solver.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import NamedTuple

import numpy as np

from cnamaste.config import config_defaults, get_global_config, start_time
from cnamaste.logger import get_logger

logger = get_logger(__name__, start_time=start_time)

OPENGM_ALGORITHMS = {"icm": 0, "lbp": 1, "astar": 2, "trws": 3, "dd": 4, "expansion": 5, "swap": 6}
"""Each algorithm's index in `cnamaste_opengm.cxx`'s `Algorithm`."""

MINCUT_SOLVERS = {"dinic": 0, "boykov_kolmogorov": 1}
"""Each min s-t cut's index in `cnamaste_opengm.cxx`'s `Mincut`: what expansion and the swap make each move with."""


def get_mincut_solvers() -> dict[str, int]:
    """Return the mincut solvers, {"dinic": 0, "boykov_kolmogorov": 1}."""
    return MINCUT_SOLVERS


HOME_VARIABLE = "CNAMASTE_OPENGM_HOME"


def get_home_variable() -> str:
    """Return the home variable, "CNAMASTE_OPENGM_HOME"."""
    return HOME_VARIABLE


LIBRARY = "libcnamaste_opengm.so"


class OpenGMResult(NamedTuple):
    """One solve: labelling, energy, lower bound (`-inf` where none), the bound per iteration, the iterations reported.

    Args:
        labels: as above.
        energy: as above.
        bound: as above.
        trace: as above.
        iterations: as above.
    """

    labels: np.ndarray
    energy: float
    bound: float
    trace: np.ndarray
    iterations: int


def home() -> Path | None:
    """`$CNAMASTE_OPENGM_HOME`, where `build_opengm.sh` writes the library; `None` where it is unset.

    There is no default: the build script reads the same variable, so the two cannot point at different places.

    Returns:
        The directory, or None.
    """
    found = os.environ.get(HOME_VARIABLE)
    return Path(found) if found else None


def library() -> Path | None:
    """The built library, or None where `$CNAMASTE_OPENGM_HOME` is unset or holds no build.

    Returns:
        The library's path, or None.
    """
    where = home()

    if where is None:
        return None

    path = where / LIBRARY
    return path if path.is_file() else None


def available() -> bool:
    return library() is not None


def unavailable() -> RuntimeError:
    """The error for a library that is not built: `$CNAMASTE_OPENGM_HOME` unset, or set and holding no build."""
    where = home()
    return RuntimeError(
        f"${HOME_VARIABLE} is unset: set it to OpenGM's directory and run infra/opengm/build_opengm.sh"
        if where is None
        else f"OpenGM is not built under {where}: run infra/opengm/build_opengm.sh"
    )


def require(solver: str) -> None:
    """Stop where `solver` is one of OpenGM's (`opengm_*`) and the library is not built.

    Called when a run starts, so a configuration that names an OpenGM solver stops before its first stage rather
    than at its first labelling; no other solver stands in.

    Args:
        solver: `hmrf.label_solver`.

    Raises:
        RuntimeError: from `unavailable`.
    """
    if solver.startswith("opengm_") and not available():
        raise unavailable()


def staging_root() -> Path | None:
    """`tmp` under the configured `paths.output_dir`, where a session writes, made where it is not there.

    The blocks are the run's, not the machine's: they belong beside what the run writes rather than under `$TMPDIR`,
    where a session that outlives its files is invisible.

    Returns:
        The directory, or None with no configuration, which is `tempfile`'s own.
    """
    where = getattr(getattr(get_global_config(), "paths", None), "output_dir", None)
    if where is None:
        return None
    root = Path(where) / "tmp"
    root.mkdir(parents=True, exist_ok=True)
    return root


class Session:
    """One worker for many solves; a context manager that shuts the worker down on exit, keeping its files.

    Its directory is `<paths.output_dir>/tmp/opengm-*`, or `tempfile`'s own with no configuration (`staging_root`).

    Raises:
        RuntimeError: where the library is not built.
    """

    def __init__(self) -> None:
        built = library()
        if built is None:
            raise unavailable()
        self.directory = Path(tempfile.mkdtemp(prefix="opengm-", dir=staging_root()))
        self.process: subprocess.Popen | None = subprocess.Popen(
            [sys.executable, "-m", "cnamaste.external_worker", str(built), str(self.directory)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self._calls = 0
        self._reply()

    def __enter__(self) -> Session:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _reply(self) -> dict:
        assert self.process is not None and self.process.stdout is not None
        line = self.process.stdout.readline()
        if not line:
            stderr = self.process.stderr.read() if self.process.stderr is not None else ""
            self.close()
            raise RuntimeError(f"the OpenGM worker died: {stderr.strip()}")
        return json.loads(line)

    def _block(self, name: str, values: np.ndarray) -> list:
        """Write `values`, at least one element, to a file the worker maps, named within the session's directory.

        Args:
            name: the block's name.
            values: the array.

        Returns:
            The spec that crosses the pipe.
        """
        values = np.ascontiguousarray(values)
        if values.size == 0:
            values = np.zeros(1, dtype=values.dtype)
        named = f"{self._calls}-{name}"
        block = np.memmap(self.directory / named, dtype=values.dtype, mode="w+", shape=values.shape)
        block[...] = values
        block.flush()
        del block
        return [named, values.dtype.str, list(values.shape)]

    def solve(
        self,
        problem,
        algorithm: str,
        max_iterations: int = 1000,
        tolerance: float = 1e-9,
        start=None,
        mincut: str = config_defaults["hmrf"]["opengm_mincut_solver"],
    ) -> OpenGMResult:
        """Run `algorithm` on `problem` (a `cnamaste.potts.PottsProblem`), from `start` where the algorithm takes one.

        Args: `problem`; `algorithm` in `OPENGM_ALGORITHMS`; `max_iterations`, `tolerance` stop rules; `start` labels;
            `mincut` in `MINCUT_SOLVERS`, the cut expansion and the swap move by, which no other algorithm reads.

        Returns:
            The `OpenGMResult`, energy and bound in `problem`'s terms.
        """
        if self.process is None:
            raise RuntimeError("the session is closed")
        if algorithm not in OPENGM_ALGORITHMS:
            raise ValueError(f"algorithm {algorithm!r} is not one of {sorted(OPENGM_ALGORITHMS)}")
        if mincut not in MINCUT_SOLVERS:
            raise ValueError(f"mincut {mincut!r} is not one of {sorted(MINCUT_SOLVERS)}")
        n, k = problem.unary.shape
        steps = max(int(max_iterations), 1)
        self._calls += 1
        blocks = {
            "unary": self._block("unary", np.asarray(problem.unary, dtype=np.float64)),
            "first": self._block("first", np.asarray(problem.first, dtype=np.int64)),
            "second": self._block("second", np.asarray(problem.second, dtype=np.int64)),
            "coupling": self._block("coupling", np.asarray(problem.coupling, dtype=np.float64)),
            "labels": self._block("labels", np.zeros(n, dtype=np.int64)),
            "trace": self._block("trace", np.zeros(steps, dtype=np.float64)),
            "seconds": self._block("seconds", np.zeros(1, dtype=np.float64)),
        }
        if start is not None:
            blocks["start"] = self._block("start", np.asarray(start, dtype=np.int64))
        request = {
            "algorithm": OPENGM_ALGORITHMS[algorithm],
            "mincut": MINCUT_SOLVERS[mincut],
            "n_nodes": n,
            "n_states": k,
            "n_edges": int(problem.first.size),
            "max_iterations": steps,
            "tolerance": float(tolerance),
            "blocks": blocks,
        }
        assert self.process.stdin is not None
        try:
            self.process.stdin.write(json.dumps(request) + "\n")
            self.process.stdin.flush()
        except BrokenPipeError:
            pass  # NB a dead worker: its empty reply below raises RuntimeError with its standard error
        reply = self._reply()
        if reply["status"] != 0:
            raise RuntimeError(f"OpenGM refused the problem: {reply['error']}")
        labels = np.array(np.memmap(self.directory / blocks["labels"][0], dtype=np.int64, mode="r", shape=(n,)))
        trace = np.array(
            np.memmap(self.directory / blocks["trace"][0], dtype=np.float64, mode="r", shape=(steps,))[: reply["taken"]]
        )
        seconds = float(np.memmap(self.directory / blocks["seconds"][0], dtype=np.float64, mode="r", shape=(1,))[0])
        # NB the call's blocks stay in the session's directory, named `<call>-<block>`: nothing here removes a file,
        #    so a session's directory holds every block it wrote and is the caller's to remove.
        logger.info(
            f"OpenGM {algorithm}{f' ({mincut})' if algorithm in ('expansion', 'swap') else ''}: {n} nodes, {k} labels, {problem.first.size} pairs; energy {reply['value']:.6e}, "
            f"bound {reply['bound']:.6e}, {reply['taken']} iterations, {seconds:.3f}s inference."
        )
        return OpenGMResult(labels, reply["value"], reply["bound"], trace, int(reply["taken"]))

    def close(self) -> None:
        """Shut the worker down, keeping the session's directory and its blocks; idempotent.

        Nothing here removes a file or a directory: `self.directory` outlives the session, named on the log, for the
        caller to remove.
        """
        if self.process is not None:
            process, self.process = self.process, None
            if process.stdin is not None:
                try:
                    process.stdin.close()
                except BrokenPipeError:
                    pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            for stream in (process.stdout, process.stderr):
                if stream is not None:
                    stream.close()
            logger.info(f"OpenGM session closed; {self.directory} kept, with the blocks of its calls")


def session() -> Session:
    """A `Session`: `with external.session() as s: s.solve(problem, "trws")`.

    Returns:
        A new `Session`.
    """
    return Session()


def solve(
    problem,
    algorithm: str,
    max_iterations: int = 1000,
    tolerance: float = 1e-9,
    start=None,
    mincut: str = config_defaults["hmrf"]["opengm_mincut_solver"],
) -> OpenGMResult:
    """One solve in its own session: a one-shot call, which pays the worker's start-up.

    Args: as `Session.solve`.

    Returns:
        The `OpenGMResult`.
    """
    with Session() as s:
        return s.solve(problem, algorithm, max_iterations, tolerance, start, mincut)
