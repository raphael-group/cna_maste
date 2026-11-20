"""`io.get_assignment_frame` as `run_cnamaste` writes it, `clone_labels.tsv.gz` and `clone_labels_int.tsv.gz`, read back
by `qa.metrics.read_estimate`, with the integer clones (`ab_decode.decode.get_merged_ab_decoded_clones`) and the field's own
labelling (`hmrf.field_argmax` of `hmrf.get_labelling_unary`) composed as `run_cnamaste` composes them.

Input, seeded: 30 spots on a 6 x 5 grid, barcodes `spot_<n>` as CalicoST easy (`2d4ce9a9`) names them, 3 clones on 8
bins, clones 0 and 2 one `(A, B)` profile. Counts are drawn from each spot's planted clone's path (the
negative binomial and beta-binomial) at 200 reads per bin, so the field recovers the plant. The referees are the
planted labels, the inputs per barcode, and `get_labelling_unary` itself, which `tests/calicost/test_hmrf.py` pins to
CalicoST's unary. The file runs in 2.6 s with numba's cache warm, 8.1 s cold.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import pytest

from cnamaste.ab_decode.model import get_merged_ab_decoded_clones
from cnamaste.emission import get_unconfounded
from cnamaste.hmm_params import CnaHmmParameters
from cnamaste.hmrf import field_argmax, get_labelling_unary
from cnamaste.io import get_assignment_frame, read_text_table
from cnamaste.qa.metrics import read_estimate
from cnamaste.results import write_tsv
from labelled_counts import get_labelled_counts

N_SPOTS, N_OBS, N_CLONES = 30, 8, 3
PATHS = np.array([[0, 1, 1, 2, 1, 1, 0, 1], [1, 1, 2, 2, 2, 1, 1, 0], [2, 0, 1, 1, 0, 2, 1, 1]]).T
"""Each clone's decoded state per bin, `pred_cnv` (n_obs, n_clones)."""

PARAMETERS = CnaHmmParameters(
    None, np.log([0.5, 1.0, 1.5]), np.array([0.1, 0.5, 0.3]), np.full(3, 0.01), np.full(3, 50.0)
)
"""The fit's states: depth 0.5, 1 and 1.5 at B fraction 0.1, 0.5 and 0.3."""

COPIES = np.array([[1, 1], [2, 1], [1, 1]])[:, None, :].repeat(N_OBS, axis=1)
"""Each clone's `(A, B)` per bin: clones 0 and 2 agree on every bin, so share integer clone 0."""


class Planted(NamedTuple):
    barcodes: list[str]
    coords: np.ndarray
    labels: np.ndarray
    normal_candidate: np.ndarray
    field: np.ndarray


@pytest.fixture(scope="module")
def planted() -> Planted:
    rng = np.random.default_rng(0)
    coords = np.stack(np.meshgrid(np.arange(6), np.arange(5), indexing="ij"), axis=-1).reshape(-1, 2)
    labels = rng.integers(0, N_CLONES, N_SPOTS)
    state = PATHS[:, labels]
    base = np.full((N_OBS, N_SPOTS), 200.0)
    total = np.full((N_OBS, N_SPOTS), 200.0)
    single_X = np.zeros((N_OBS, 2, N_SPOTS))
    single_X[:, 0] = rng.poisson(base * np.exp(PARAMETERS.log_mu[state]))
    single_X[:, 1] = rng.binomial(total.astype(int), PARAMETERS.p_binom[state])
    counts = get_labelled_counts(single_X[:, 0], single_X[:, 1], total, normal_spot=np.ones(N_SPOTS))
    field = get_labelling_unary(counts, PARAMETERS, get_unconfounded(N_CLONES), PATHS.T, phased=False)
    barcodes = [f"spot_{i}" for i in range(N_SPOTS)]
    return Planted(barcodes, coords, labels, rng.random(N_SPOTS) < 0.2, field)


def get_seglevel() -> pd.DataFrame:
    columns = {f"clone{c} {a}": COPIES[c, :, i] for c in range(N_CLONES) for i, a in enumerate("AB")}
    return pd.DataFrame({"CHR": 1, "START": np.arange(N_OBS) * 1_000, "END": np.arange(1, N_OBS + 1) * 1_000} | columns)


@pytest.fixture(scope="module")
def run_dir(planted: Planted, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The three files `read_estimate` reads, written by `run_cnamaste`'s calls."""
    directory = tmp_path_factory.mktemp("run")
    p = planted
    seglevel = get_seglevel()
    int_groups = get_merged_ab_decoded_clones(seglevel, 0.99)
    args = (p.barcodes, p.coords)
    write_tsv(directory, "cnv_diploid_seglevel.tsv", seglevel, header=True, index=False)
    write_tsv(
        directory,
        "clone_labels.tsv.gz",
        get_assignment_frame(*args, p.labels, p.normal_candidate),
        header=True,
        index=True,
        index_label="barcode",
    )
    labelled = get_assignment_frame(*args, int_groups[p.labels], p.normal_candidate, int_groups[field_argmax(p.field)])
    write_tsv(directory, "clone_labels_int.tsv.gz", labelled, header=True, index=True, index_label="barcode")
    return directory


def test_the_field_alone_recovers_every_planted_clone(planted: Planted) -> None:
    """At 200 reads per bin each spot's best clone under `get_labelling_unary` is its planted one, and `field_argmax` takes
    the row maximum. Fails where the field scores a spot under another clone's path or the argmax reads columns
    out of order."""
    labels = field_argmax(planted.field)
    assert np.array_equal(labels, planted.labels)
    assert np.array_equal(planted.field[np.arange(N_SPOTS), labels], planted.field.max(axis=1))


def test_field_argmax_breaks_a_tie_to_the_lowest_clone() -> None:
    """Fails where a tie goes to another clone than the lowest, as its docstring states."""
    assert field_argmax(np.array([[-1.0, -1.0, -2.0], [-3.0, -1.0, -1.0]])).tolist() == [0, 1]


def test_read_estimate_reads_each_barcodes_labels_as_written(planted: Planted, run_dir: Path) -> None:
    """`read_estimate` on the run's directory: per barcode, `clones` the labels passed and `int_clones` their integer
    clone, clones 0 and 2 merged; `seglevel` the table written. Fails where the writer's sort by sample and position
    separates a label from its barcode, or the integer groups are not the shared profile's."""
    estimate = read_estimate(run_dir)
    assert get_merged_ab_decoded_clones(get_seglevel(), 0.99).tolist() == [0, 1, 0]
    assert np.array_equal(estimate.clones.reindex(planted.barcodes), planted.labels)
    assert np.array_equal(estimate.int_clones.reindex(planted.barcodes), np.array([0, 1, 0])[planted.labels])
    pd.testing.assert_frame_equal(estimate.seglevel, get_seglevel(), check_exact=True)


def test_clone_labels_int_carries_each_spots_position_normal_candidate_and_field_label(
    planted: Planted, run_dir: Path
) -> None:
    """`clone_labels_int.tsv.gz` per barcode: `x`, `y`, `normal_candidate` (0 or 1) as passed and
    `field_argmax` the integer clone of the field's own label, columns in the documented order. Fails where a column
    moves, is recast or is another spot's."""
    p = planted
    found = read_text_table(run_dir / "clone_labels_int.tsv", sep="\t", index_col=0, float_precision="round_trip")
    columns = ["sample_id", "x", "y", "clone_label", "normal_candidate", "field_argmax"]
    assert list(found.columns) == columns
    found = found.reindex(p.barcodes)
    assert np.array_equal(found[["x", "y"]], p.coords)
    assert found["normal_candidate"].tolist() == p.normal_candidate.astype(int).tolist()
    assert np.array_equal(found["field_argmax"], np.array([0, 1, 0])[p.labels])


@pytest.mark.xfail(strict=True, reason="T- #31: sample_id is the barcode's last `_` field")
def test_a_one_sample_runs_clone_labels_name_one_sample(planted: Planted) -> None:
    """CalicoST easy's barcodes `spot_<n>` from its one sample: `sample_id` should be one value. Fails today with 30,
    each spot's own index, which also orders the rows as text (`spot_0`, `spot_1`, `spot_10`)."""
    frame = get_assignment_frame(planted.barcodes, planted.coords, planted.labels)
    assert frame["sample_id"].nunique() == 1
