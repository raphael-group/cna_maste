"""Write OpenGM's TRW-S (`TRWSi`, as `opengm_trws` runs it) on fixed Potts problems: the oracle
`verify/tests/opengm_trws_oracle.rs` reads (T- #198).

Each case is one file `<out>/<case>.bin`, little-endian: `CNXT`, then int64 `version, n_nodes, n_states, n_pairs,
max_iterations, n_trace`, float64 `tolerance`, then `unary` float64 `(n_nodes * n_states)`, `first` and `second` int64
`(n_pairs)`, `coupling` float64 `(n_pairs)`, and OpenGM's answer: `labels` int64 `(n_nodes)`, `energy` and `bound`
float64 (the Potts form), and `trace` float64 `(n_trace)`, the bound after the forward pass and each pass.

The cases are `write_rust_expansion_oracle.py`'s: `tests/test_expansion.py`'s planted lattices and
`tests/test_benchmarks.py`'s Visium HD 16 um labellings; `--stress` adds the 87,010 x 16 one. TRW-S takes no start.
Nothing is committed: run this before `verify/`'s `cargo test --test opengm_trws_oracle -- --include-ignored`.

    CNAMASTE_OPENGM_HOME=... uv run python infra/expansion/write_rust_trws_oracle.py [--stress] [--out DIR]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "tests"))

from write_rust_expansion_oracle import LATTICES

from cnamaste.opengm import opengm_session
from cnamaste.potts import PottsProblem
from potts_lattice import get_planted_lattice
from test_benchmarks import LABELLINGS, get_labelling

VERSION = 1
MAX_ITERATIONS = 1000
TOLERANCE = 1e-9


def write_case(path: Path, problem: PottsProblem) -> None:
    """Solve `problem` by OpenGM's TRW-S and write the case."""
    found = opengm_session().solve(problem, "trws", max_iterations=MAX_ITERATIONS, tolerance=TOLERANCE)
    n, k = problem.unary.shape
    header = np.array([VERSION, n, k, problem.first.size, MAX_ITERATIONS, found.trace.size], dtype="<i8")
    with path.open("wb") as f:
        f.write(b"CNXT")
        for array, dtype in (
            (header, "<i8"),
            (np.array([TOLERANCE]), "<f8"),
            (problem.unary.ravel(), "<f8"),
            (problem.first, "<i8"),
            (problem.second, "<i8"),
            (problem.coupling, "<f8"),
            (found.labels, "<i8"),
            (np.array([found.energy, found.bound]), "<f8"),
            (found.trace, "<f8"),
        ):
            f.write(np.ascontiguousarray(array, dtype=dtype).tobytes())
    gap = found.energy - found.bound
    print(f"{path.name}: {n} nodes, {k} states, {problem.first.size} pairs; energy {found.energy:.6e}, gap {gap:.1e}, "
          f"{found.trace.size} passes")  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=HERE / "oracle" / "trws")
    parser.add_argument("--stress", action="store_true", help="add the 87,010 x 16 labelling")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for name, (rows, cols, n_clones, mislabelled, spatial_weight, rounded) in LATTICES.items():
        problem = get_planted_lattice(rows, cols, n_clones, mislabelled, spatial_weight, seed=3).problem
        if rounded:
            problem = problem._replace(unary=np.round(2.0 * problem.unary) / 2.0)
        write_case(args.out / f"{name}.bin", problem)
    sizes = ["hd16um-3000x4", "hd16um-20000x16"] + (["hd16um-87010x16"] if args.stress else [])
    for size in sizes:
        write_case(args.out / f"{size}.bin", get_labelling(*LABELLINGS[size]).problem)


if __name__ == "__main__":
    main()
