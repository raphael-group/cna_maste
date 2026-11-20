"""Fit `t = a * N ** b` per stage, and for the wall time and peak memory, to the stress tier's records, and predict at N.

    PYTHONPATH=python python tests/scripts/estimate_runtime.py --spots 100000

Reads `sim/.stress/results/*.json` (`tests/test_stress.py`), at least two sizes; prints each fit's exponent `b` and the
prediction at `--spots`.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from cnamaste.qa.sims import get_project


def get_power_law(spots: np.ndarray, values: np.ndarray) -> tuple[float, float]:
    """The least-squares `(a, b)` of `log v = log a + b log N`, over the positive values.

    Args:
        spots: (n,) sizes.
        values: (n,) measured values.

    Returns:
        `(a, b)`.
    """
    kept = values > 0
    b, log_a = np.polyfit(np.log(spots[kept]), np.log(values[kept]), 1)
    return float(np.exp(log_a)), float(b)


def main(argv: list[str] | None = None) -> int:
    """Print each fit and its prediction.

    Returns:
        0; 1 where fewer than two sizes are recorded.
    """
    parser = argparse.ArgumentParser(description=(__doc__ or "").strip().splitlines()[0])
    parser.add_argument("--spots", type=int, default=100_000, help="the size predicted")
    args = parser.parse_args(argv)
    records = [json.loads(p.read_text()) for p in sorted((get_project() / "sim/.stress/results").glob("*.json"))]
    if len({r["spots"] for r in records}) < 2:
        print("fewer than two sizes recorded: run pytest tests/test_stress.py -m stress")
        return 1
    spots = np.array([r["spots"] for r in records], dtype=np.float64)
    series = {"wall seconds": [r["wall_seconds"] for r in records], "peak MB": [r["peak_mb"] for r in records]}
    for stage in records[-1]["stage_seconds"]:
        series[f"{stage} s"] = [r["stage_seconds"].get(stage, 0.0) for r in records]
    print(f"{'series':<48} {'b':>6} {'at ' + format(args.spots, ','):>14}")
    for name, values in series.items():
        a, b = get_power_law(spots, np.asarray(values, dtype=np.float64))
        print(f"{name:<48} {b:6.2f} {a * args.spots**b:14,.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
