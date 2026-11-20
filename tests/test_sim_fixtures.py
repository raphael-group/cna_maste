"""`sim/<name>_<hash>/` holds each staged fixture's sample: the inputs and planted truth its capture read.

The referee is each capture (`capture/<name>_<hash>_capture.hdf5`): `config.attrs["inputs"]` names every input file
the captured run read, and `truth/<name>` stores each planted truth file gzipped. What the inputs load to is held by
`tests/test_capture.py`'s replay of `00_inputs/load_input_data`; a capture written before `qa.capture` also records each
input's sha256, as stored, which is checked here.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import h5py
import pytest

from cnamaste.qa.sims import fixture, get_names, get_project, get_samples, input_files, staged


def committed(staged_hash: str) -> Path:
    """`staged_hash`'s committed sample, whatever `CNAMASTE_SIMS` says the run reads; a skip where it has none.

    Each row runs on the fixtures `--sims` names (`staged_hash`, conftest.py): CalicoST easy and hard by default,
    every staged fixture under `--sims all`, so a committed sample outside the dev tier costs a bare run nothing."""
    if staged_hash not in get_samples():
        pytest.skip(f"{get_names()[staged_hash]} ({staged_hash}): no committed sample")
    return fixture(staged_hash)


def test_each_input_is_the_one_its_capture_read(staged_hash: str) -> None:
    """Every input the capture recorded is here, at the path it recorded (and with the sha256, where recorded), and they
    are the inputs the sample's layout names (`sims.input_files`: CalicoST's six, or a drawn sample's two per slice and
    `snp/`'s four). Fails where a file was moved, swapped between fixtures, or left out."""
    directory = committed(staged_hash)
    with h5py.File(get_project() / staged(staged_hash), "r") as h5:
        assert h5["config"].attrs["fixture_hash"] == staged_hash
        recorded = json.loads(h5["config"].attrs["inputs"])
    found = input_files(directory)
    assert sorted(recorded) == sorted(found)
    for input_name, entry in recorded.items():
        path = found[input_name]
        if isinstance(entry, str):
            assert path == get_project() / entry, f"{staged_hash}: {input_name}"
        else:
            assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"], f"{staged_hash}: {input_name}"


def test_the_planted_truth_is_its_capture_s(staged_hash: str) -> None:
    """Each truth file decompresses to the bytes the capture stores. Fails where the planted clones or profile differ
    from what the fixture's metrics were scored against."""
    directory = committed(staged_hash)
    with h5py.File(get_project() / staged(staged_hash), "r") as h5:
        truth = {key: gzip.decompress(ds[()].tobytes()) for key, ds in h5["truth"].items()}
    assert sorted(truth) == ["truth_acn_profile.tsv", "truth_clone_labels.tsv"]
    for key, data in truth.items():
        assert gzip.decompress((directory / f"{key}.gz").read_bytes()) == data, f"{staged_hash}: {key}"
