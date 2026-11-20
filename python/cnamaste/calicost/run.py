"""CalicoST `c1abcae` on a drawn sample: its `key : value` configuration and `calicost_main` as a subprocess, its
results under `ROOT/nomixing_calicost_related/<sample_id>/clone*_rectangle*_w*/` as `calicost.read` reads them.

CalicoST calls scipy sparse matrices' `.A`, which scipy 1.14 removed, numpy's `NAN`, which numpy 2.0 removed, and
imports `turtle`, which needs `tkinter`, for a name it never calls; `main`, the subprocess, restores `.A` as `toarray()`
and `NAN` as `nan`, and stubs `turtle` before importing
CalicoST, which is not modified (as tests/calicost stub `turtle`). A run may stop at CalicoST's first BAF-only inference,
the first rectangle's initial assignment (`allspots_nstates<n>_sp.npz`), written once its inputs are read, its
adjacency built and its phasing done.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import types
from pathlib import Path
from typing import NamedTuple
from unittest import mock

import numpy as np
import pandas as pd
import scipy.sparse as sp

from cnamaste.qa.sims import get_resources, grch38

CALICOST_SETTINGS = {
    "n_clones": 3,
    "n_states": 7,
    "spatial_weight": 1.0,
    "num_hmrf_initialization_start": 0,
    "num_hmrf_initialization_end": 1,
}
"""CalicoST's settings beyond its defaults: the zenodo results' `clone3_rectangle0_w1.0`, one rectangle, as
`run_sim_qa --one_shot` scores; `n_states` cnamaste's `hmm.n_states`."""


def get_calicost_settings() -> dict[str, float]:
    """Return CalicoST's settings beyond its defaults: n_clones 3, n_states 7, spatial_weight 1.0, rectangle 0."""
    return CALICOST_SETTINGS


POLL_SECONDS = 5.0
"""Seconds between checks of a run stopped at its first BAF-only inference."""


def get_poll_seconds() -> float:
    """Return the seconds between checks of a run stopped at its first inference, 5.0."""
    return POLL_SECONDS


class CalicostRun(NamedTuple):
    """A CalicoST subprocess: its exit status, whether it reached its first BAF-only inference, wall and peak MB."""

    returncode: int
    started_inference: bool
    seconds: float
    peak_mb: float


def get_calicost_config(sample: Path, output_dir: Path) -> str:
    """CalicoST's configuration for a one-slice drawn sample, `key : value` per line (`arg_parse`).

    Args:
        sample: the drawn sample's directory.
        output_dir: CalicoST's `output_dir`, `calicost.read.results_dir`.

    Returns:
        The text.

    Raises:
        ValueError: where the sample has other than one slice, or no GRCh38 resources are found.
    """
    sheet = pd.read_csv(sample / "sample_sheet.tsv", sep="\t")
    references = grch38()
    if len(sheet) != 1 or references is None:
        raise ValueError(f"{sample}: CalicoST runs one slice with GRCh38 resources ({len(sheet)} slices)")
    values = {
        "spaceranger_dir": sheet["spaceranger_dir"].iloc[0],
        "snp_dir": sheet["snp_dir"].iloc[0],
        "output_dir": str(output_dir),
        **{key: str(references / name) for key, name in get_resources().items()},
        **get_calicost_settings(),
    }
    return "".join(f"{key} : {value}\n" for key, value in values.items())


def get_first_inference(output_dir: Path) -> Path | None:
    """The first rectangle's initial assignment, which CalicoST writes as its BAF-only inference starts.

    Args:
        output_dir: CalicoST's `output_dir`.

    Returns:
        `clone*_rectangle0_w*/allspots_nstates*_sp.npz`, or None before it is written.
    """
    return next(iter(sorted(output_dir.glob("clone*_rectangle0_w*/allspots_nstates*_sp.npz"))), None)


def run_calicost(config: Path, output_dir: Path, log: Path, stop_at_first_inference: bool) -> CalicostRun:
    """`main` on `config` as a subprocess, its output to `log`; stopped once its first BAF-only inference starts
    where asked.

    Args:
        config: the configuration, `get_calicost_config`'s.
        output_dir: its `output_dir`.
        log: the subprocess's stdout and stderr.
        stop_at_first_inference: terminate the run once `get_first_inference` finds the assignment.

    Returns:
        The `CalicostRun`; a stopped run's `returncode` is the termination's.
    """
    start = time.perf_counter()
    with log.open("x") as stream:
        child = subprocess.Popen(
            [sys.executable, "-m", "cnamaste.calicost.run", str(config)], stdout=stream, stderr=subprocess.STDOUT
        )
        started = False
        pid, status, usage = os.wait4(child.pid, os.WNOHANG)
        while pid == 0 and stop_at_first_inference and not started:
            time.sleep(get_poll_seconds())
            started = get_first_inference(output_dir) is not None
            pid, status, usage = os.wait4(child.pid, os.WNOHANG)
        if pid == 0:
            if started:
                child.terminate()
            _, status, usage = os.wait4(child.pid, 0)
    started = started or get_first_inference(output_dir) is not None
    return CalicostRun(os.waitstatus_to_exitcode(status), started, time.perf_counter() - start, usage.ru_maxrss / 1024)


def main(argv: list[str] | None = None) -> int:
    """`calicost_main.main` on a configuration, scipy sparse matrices' `.A` restored as `toarray()`.

    Args:
        argv: the configuration's path; `sys.argv[1:]` where None.

    Returns:
        0.
    """
    (config,) = sys.argv[1:] if argv is None else argv
    # NB scipy >= 1.14 has no spmatrix.A; CalicoST c1abcae calls it 39 times.
    sp.spmatrix.A = property(lambda matrix: matrix.toarray())
    # NB pandas >= 1.0 has no Series.nonzero; scipy calls it on the boolean Series CalicoST indexes rows by.
    pd.Series.nonzero = lambda series: series.to_numpy().nonzero()
    # NB numpy >= 2.0 has no NAN; CalicoST c1abcae sets a merged fit's total_llf to np.NAN.
    np.NAN = np.nan
    # NB `from turtle import reset` in CalicoST's `hmrf` and `phasing`, never called; `turtle` needs `tkinter`.
    with mock.patch.dict(sys.modules, {"turtle": types.SimpleNamespace(reset=None)}):
        from calicost.calicost_main import main as calicost_main

    calicost_main(config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
