"""`results.write_tsv` and `results.write_npz` against the readers that take their files back: `io.read_text_table` and
`run_cnamaste_decode.read_fit` (the final fit, `rdrbaf_final_nstates<n>_smp.npz`); `write_fig` and `write_page`
written twice.

Input, seeded: a fit of 4 states, 10 bins, 3 clones and 20 spots as `get_reindexed_fit` leaves it. The referee is the input: every reloaded array equals it bitwise, dtype included. The file
runs in under 1 s.
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from cnamaste.emission import CnaEmissionConfounders
from cnamaste.hmm_params import CnaHmmParameters
from cnamaste.hmrf_fit import CloneAssignment, HMMProfile, HmrfFit
from cnamaste.io import read_text_table
from cnamaste.results import write_fig, write_npz, write_page, write_tsv
from cnamaste.scripts.run_cnamaste_decode import read_fit

N_STATES, N_OBS, N_CLONES, N_SPOTS = 4, 10, 3, 20


def get_fit(seed: int, with_shift: bool) -> HmrfFit:
    """A final fit with every field `run_cnamaste` writes; `with_shift` False as a fit with no per-clone depth shift."""
    rng = np.random.default_rng(seed)
    return HmrfFit(
        parameters=CnaHmmParameters(
            log_mu=rng.normal(size=N_STATES),
            alphas=rng.uniform(0.01, 0.1, N_STATES),
            p_binom=rng.uniform(0.1, 0.5, N_STATES),
            taus=rng.uniform(10.0, 50.0, N_STATES),
            log_startprob=np.log(np.full(N_STATES, 1.0 / N_STATES)),
        ),
        confounders=CnaEmissionConfounders(
            np.zeros((N_CLONES, 1)), rng.normal(size=(N_CLONES, 1)) if with_shift else np.zeros((N_CLONES, 1))
        ),
        log_transmat=rng.normal(size=(N_STATES, N_STATES)),
        profile=HMMProfile(
            log_gamma=rng.normal(size=(N_STATES, N_OBS, N_CLONES)),
            pred_cnv=rng.integers(0, N_STATES, (N_OBS, N_CLONES)),
        ),
        llf=float(rng.normal()),
        n_states=N_STATES,
        assignment=CloneAssignment(
            assignment_before_reindex={2: 0, 0: 1, 1: 2},
            prev_assignment=rng.integers(0, N_CLONES, N_SPOTS),
            new_assignment=np.r_[np.arange(N_CLONES), rng.integers(0, N_CLONES, N_SPOTS - N_CLONES)],
            total_llf=float(rng.normal()),
        ),
    )


def test_a_gz_table_reloads_exactly_and_writes_the_same_bytes_twice(tmp_path: Path) -> None:
    """`clone_labels.tsv.gz`'s writer: `read_text_table` on the plain name finds the `.gz` and reloads the frame,
    index and dtypes; two writes are byte-identical (gzip `mtime` 0). Fails where a column, dtype or the gzip header's
    timestamp changes between writes."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {"x": rng.integers(0, 9, 5), "y": rng.normal(size=5), "clone_label": rng.integers(0, 3, 5)},
        index=pd.Index([f"spot_{i}" for i in range(5)], name="barcode"),
    )
    write_tsv(tmp_path, "a.tsv.gz", df, header=True, index=True, index_label="barcode")
    first = (tmp_path / "a.tsv.gz").read_bytes()
    write_tsv(tmp_path, "a.tsv.gz", df, header=True, index=True, index_label="barcode")
    assert (tmp_path / "a.tsv.gz").read_bytes() == first
    found = read_text_table(tmp_path / "a.tsv", sep="\t", index_col=0, float_precision="round_trip")
    pd.testing.assert_frame_equal(found, df, check_exact=True)


@pytest.mark.parametrize("with_shift", [True, False], ids=["shift", "no-shift"])
def test_the_final_fit_reads_back_as_the_fit_written(tmp_path: Path, with_shift: bool) -> None:
    """`write_npz(**get_npz_fields())` then `read_fit`: every key of the fit bitwise, dtype included, the scalars and
    `assignment_before_reindex` as written, and `new_log_mu_shift` None for a fit with every `log Z_c` 0 (T- #32).
    Fails where a key is dropped, recast or read from another key."""
    fields = get_fit(0, with_shift=with_shift).get_npz_fields()
    write_npz(tmp_path, "rdrbaf_final_nstates4_smp.npz", **fields)
    found = read_fit(tmp_path / "rdrbaf_final_nstates4_smp.npz").get_npz_fields()
    assert list(found) == list(fields)
    for key, expected in fields.items():
        got = found[key]
        if isinstance(expected, np.ndarray):
            assert got.dtype == expected.dtype and np.array_equal(got, expected), key
        else:
            assert type(got) is type(expected) and got == expected, key


def test_a_figure_writes_the_same_bytes_twice_with_no_creation_date(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`write_fig` and `write_page` save a PDF with no `/CreationDate`, so the same figure written a second apart,
    to two directories, is the same bytes. Fails where a date, or any other run-dependent field, enters the file."""
    pages = []
    for k, now in enumerate((1_700_000_000.0, 1_700_000_001.0)):
        monkeypatch.setattr(time, "time", lambda now=now: now)
        directory = tmp_path / str(k)
        directory.mkdir()
        for write in (write_fig, write_page):
            figure, ax = plt.subplots()
            ax.plot([0, 1], [1, 0])
            write(str(directory), f"{write.__name__}.pdf", figure)
        pages.append([(directory / f"{w.__name__}.pdf").read_bytes() for w in (write_fig, write_page)])
    assert pages[0] == pages[1]
    assert all(b"/CreationDate" not in page for page in pages[0])
