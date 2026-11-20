"""Write OpenGM's expansion (Boykov-Kolmogorov) on fixed Potts problems: the oracle
`verify/tests/opengm_oracle.rs` reads.

Each case is one file `<out>/<case>.bin`, little-endian: `CNXO`, then int64 `version, n_nodes, n_states, n_pairs,
max_steps, energy_only`, then `unary` float64 `(n_nodes * n_states)`, `first` and `second` int64 `(n_pairs)`,
`coupling` float64 `(n_pairs)`, `start` int64 `(n_nodes)`, and OpenGM's answer: `labels` int64 `(n_nodes)`, `energy`
float64 (the Potts form) and `steps` int64. `energy_only` marks a field of few values, where OpenGM's cut takes a tied
node to the sink and the two expansions are held to the same energy (`tests/test_expansion.py`'s stated difference).

The cases are `tests/test_expansion.py`'s planted lattices, started at the field's argmax, and
`tests/test_benchmarks.py`'s Visium HD 16 um labellings, started at the run's rectangles; `--stress` adds the
87,010 x 16 one, about a minute of OpenGM. Nothing is committed: run this before `verify/`'s
`cargo test -- --include-ignored`.

    CNAMASTE_OPENGM_HOME=... uv run python infra/expansion/write_rust_expansion_oracle.py [--stress] [--out DIR]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "tests"))

from cnamaste.config.yaml_config import LabellingOptions
from cnamaste.opengm import solve_labelling
from cnamaste.potts import PottsProblem
from potts_lattice import get_planted_lattice
from test_benchmarks import LABELLINGS, get_labelling

VERSION = 1
MAX_STEPS = 1000
DEFAULT = LabellingOptions(False, False, "index")

# NB tests/test_expansion.py's: (rows, cols, clones, mislabelled, spatial_weight, rounded to halves).
LATTICES = {
    "fixture-4": (30, 40, 4, 0.3, 1.1, False),
    "8-clones": (50, 60, 8, 0.3, 1.1, False),
    "reductions-fire": (40, 40, 6, 0.02, 0.2, False),
    "integer-ties": (40, 40, 6, 0.3, 1.0, True),
}


def write_case(path: Path, problem: PottsProblem, start: np.ndarray, energy_only: bool) -> None:
    """Solve `problem` by OpenGM's expansion from `start` and write the case."""
    found = solve_labelling(problem, "opengm_expansion", start, MAX_STEPS, 1e-9, "boykov_kolmogorov", DEFAULT)
    n, k = problem.unary.shape
    header = np.array([VERSION, n, k, problem.first.size, MAX_STEPS, int(energy_only)], dtype="<i8")
    with path.open("wb") as f:
        f.write(b"CNXO")
        for array, dtype in (
            (header, "<i8"),
            (problem.unary.ravel(), "<f8"),
            (problem.first, "<i8"),
            (problem.second, "<i8"),
            (problem.coupling, "<f8"),
            (start, "<i8"),
            (found.labels, "<i8"),
            (np.array([found.energy]), "<f8"),
            (np.array([found.iterations]), "<i8"),
        ):
            f.write(np.ascontiguousarray(array, dtype=dtype).tobytes())
    summary = f"{n} nodes, {k} states, {problem.first.size} pairs; energy {found.energy:.6e}, {found.iterations} moves"
    print(f"{path.name}: {summary}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=HERE / "oracle")
    parser.add_argument("--stress", action="store_true", help="add the 87,010 x 16 labelling")
    args = parser.parse_args()
    args.out.mkdir(exist_ok=True)
    for name, (rows, cols, n_clones, mislabelled, spatial_weight, rounded) in LATTICES.items():
        problem = get_planted_lattice(rows, cols, n_clones, mislabelled, spatial_weight, seed=3).problem
        if rounded:
            problem = problem._replace(unary=np.round(2.0 * problem.unary) / 2.0)
        write_case(args.out / f"{name}.bin", problem, np.argmin(problem.unary, axis=1), rounded)
    sizes = ["hd16um-3000x4", "hd16um-20000x16"] + (["hd16um-87010x16"] if args.stress else [])
    for size in sizes:
        labelling = get_labelling(*LABELLINGS[size])
        write_case(args.out / f"{size}.bin", labelling.problem, np.asarray(labelling.start), False)


if __name__ == "__main__":
    main()
