"""`qa.metrics`' panels on a planted run, each value computed by hand from the cell table below.

Truth: 7 spots `s0`-`s6`, true clones normal (`s0`, `s1`), `clone_0` (`s2`, `s3`, `s6`), `clone_1` (`s4`, `s5`), on
3 chr1 segments [0, 100), [100, 200), [200, 300). The run: 2 fitted clones on 4 segments, [0, 90), [90, 210),
[210, 300) and chr2 [0, 100), which no true segment overlaps; `s6` dropped.

| segment | normal | clone_0 | clone_1 | fitted clone0 | fitted clone1 |
|---|---|---|---|---|---|
| [0, 90)    | (1, 1) | (1, 1) | (1, 1) | (1, 1) | (1, 1) |
| [90, 210)  | (1, 1) | (2, 1) | (1, 2) | (1, 1) | (2, 1) |
| [210, 300) | (1, 1) | (0, 1) | (1, 3) | (2, 2) | (1, 0) |

Fitted clone per spot: `s0`, `s1`, `s5` clone0; `s2`, `s3`, `s4` clone1; 6 kept spots x 3 scored segments = 18 cells.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cnamaste.qa.metrics import (
    RunEstimate,
    clone_ari,
    exact_altered,
    get_cell_confusion,
    int_clone_ari,
    metrics,
    sensitivity,
    specificity,
)
from cnamaste.truth import Truth

BARCODES = np.array([f"s{n}" for n in range(7)])
TRUE_COPIES = {"normal": [(1, 1)] * 3, "clone_0": [(1, 1), (2, 1), (0, 1)], "clone_1": [(1, 1), (1, 2), (1, 3)]}
CALLED_COPIES = [[(1, 1), (1, 1), (2, 2), (1, 1)], [(1, 1), (2, 1), (1, 0), (1, 1)]]
"""Each fitted clone's `(A, B)` per run segment, the chr2 segment last."""


def get_truth(labels: list[int], copies: dict[str, list[tuple[int, int]]]) -> Truth:
    """A truth on the 3 chr1 segments.

    Args:
        labels: true clone per spot, `s0` first.
        copies: `(A, B)` per segment, by clone name in label order.

    Returns:
        The `Truth`.
    """
    profile = pd.DataFrame({"chr": ["chr1"] * 3, "start": [0, 100, 200], "end": [100, 200, 300]})
    for clone, pairs in copies.items():
        profile[f"{clone}_A_copy"] = [a for a, _ in pairs]
        profile[f"{clone}_B_copy"] = [b for _, b in pairs]
    n = len(labels)
    return Truth(BARCODES[:n], np.zeros((n, 2)), np.array(labels), tuple(copies), profile)


def get_estimate(
    clones: dict[str, int], int_clones: dict[str, int], called: list[list[tuple[int, int]]]
) -> RunEstimate:
    """A run on the 4 run segments.

    Args:
        clones: fitted clone per kept barcode.
        int_clones: integer clone per barcode.
        called: each fitted clone's `(A, B)` per run segment.

    Returns:
        The `RunEstimate`.
    """
    seglevel = pd.DataFrame({"CHR": ["1", "1", "1", "2"], "START": [0, 90, 210, 0], "END": [90, 210, 300, 100]})
    for k, pairs in enumerate(called):
        seglevel[f"clone{k} A"] = [a for a, _ in pairs]
        seglevel[f"clone{k} B"] = [b for _, b in pairs]
    return RunEstimate(pd.Series(clones), pd.Series(int_clones), seglevel)


TRUTH = get_truth([0, 0, 1, 1, 2, 2, 1], TRUE_COPIES)
FITTED = {"s0": 0, "s1": 0, "s2": 1, "s3": 1, "s4": 1, "s5": 0}
MERGED = {"s0": 0, "s1": 0, "s2": 1, "s3": 1, "s4": 1, "s5": 1, "s6": 0}
"""Integer clones; `s6` carries one although the clone labels dropped it."""


@pytest.fixture(scope="module")
def estimate() -> RunEstimate:
    return get_estimate(FITTED, MERGED, CALLED_COPIES)


def test_each_cell_is_its_true_clones_pair_against_its_fitted_clones_pair_phase_free(estimate: RunEstimate) -> None:
    """[90, 210) overlaps [100, 200) most, (2, 1) and (1, 2) code alike, (1, 0) reads as (0, 1), chr2 is unscored."""
    assert get_cell_confusion(estimate, TRUTH) == {
        # NB s0, s1 on [0, 90), [90, 210); s2, s3, s4, s5 on [0, 90)
        ((1, 1), (1, 1)): 8,
        # NB s0, s1 on [210, 300)
        ((1, 1), (2, 2)): 2,
        # NB s2, s3 on [210, 300)
        ((0, 1), (0, 1)): 2,
        # NB s5 on [90, 210)
        ((1, 2), (1, 1)): 1,
        # NB s2, s3, s4 on [90, 210)
        ((1, 2), (1, 2)): 3,
        # NB s4 on [210, 300)
        ((1, 3), (0, 1)): 1,
        # NB s5 on [210, 300)
        ((1, 3), (2, 2)): 1,
    }


def test_the_panels_are_the_hand_computed_values(estimate: RunEstimate) -> None:
    """Sensitivity, specificity, exact altered and both ARIs, each from the planted cells and labels."""
    expected = {
        # NB 8 altered cells, 1 called (1, 1): 1 - 1 / 8
        "sensitivity": 7 / 8,
        # NB 10 neutral cells, 2 called (2, 2): 1 - 2 / 10
        "specificity": 4 / 5,
        # NB (0, 1) 2 + (1, 2) 3 of 8 altered cells
        "exact_altered": 5 / 8,
        # NB true [0, 0, 1, 1, 2, 2] vs fitted [0, 0, 1, 1, 1, 0]: pairs 15, index 2, rows 3, cols 3 + 3 = 6;
        #    expected 3 * 6 / 15 = 1.2, max 4.5, ARI (2 - 1.2) / (4.5 - 1.2) = 8 / 33
        "clone_ari": 8 / 33,
        # NB s6 left out; vs integer [0, 0, 1, 1, 1, 1]: index 3, rows 3, cols 1 + 6 = 7; expected 1.4, max 5,
        #    ARI (3 - 1.4) / (5 - 1.4) = 4 / 9
        "int_clone_ari": 4 / 9,
    }
    found = metrics(estimate, TRUTH)
    assert found.keys() == expected.keys()
    for name, value in expected.items():
        assert found[name] == pytest.approx(value, rel=1e-12, abs=0.0), name


def test_a_spot_labelled_minus_one_is_scored_as_dropped(estimate: RunEstimate) -> None:
    """`s6` written with clone label -1 scores as `s6` absent from the clone labels."""
    labelled = get_estimate(FITTED | {"s6": -1}, MERGED, CALLED_COPIES)
    assert metrics(labelled, TRUTH) == metrics(estimate, TRUTH)


def test_a_run_calling_no_cna_has_sensitivity_zero_and_specificity_one() -> None:
    """Every segment called (1, 1): no altered cell is detected, no neutral cell is called altered."""
    neutral = get_estimate(FITTED, MERGED, [[(1, 1)] * 4] * 2)
    assert sensitivity(neutral, TRUTH) == 0.0
    assert exact_altered(neutral, TRUTH) == 0.0
    assert specificity(neutral, TRUTH) == 1.0


def test_a_run_with_no_segments_scores_nan() -> None:
    """An empty `seglevel` has no cells, so each copy panel is 0 / 0."""
    full = get_estimate(FITTED, MERGED, CALLED_COPIES)
    empty = RunEstimate(full.clones, full.int_clones, full.seglevel.iloc[:0])
    assert get_cell_confusion(empty, TRUTH) == {}
    assert np.isnan([sensitivity(empty, TRUTH), specificity(empty, TRUTH), exact_altered(empty, TRUTH)]).all()


def test_one_true_clone_split_in_two_has_ari_zero_and_no_altered_cells() -> None:
    """4 normal spots fitted [0, 0, 1, 1]: index 2, rows 6, cols 2; expected 6 * 2 / 6 = 2, so ARI (2 - 2) / 2 = 0.
    The truth has no altered cell: sensitivity and exact altered are NaN, specificity 1 - 0 / 12."""
    truth = get_truth([0, 0, 0, 0], {"normal": [(1, 1)] * 3})
    split = get_estimate({"s0": 0, "s1": 0, "s2": 1, "s3": 1}, {"s0": 0, "s1": 0, "s2": 0, "s3": 0}, [[(1, 1)] * 4] * 2)
    assert clone_ari(split, truth) == 0.0
    # NB both labellings one cluster: sklearn's convention, a perfect match
    assert int_clone_ari(split, truth) == 1.0
    assert np.isnan(sensitivity(split, truth)) and np.isnan(exact_altered(split, truth))
    assert specificity(split, truth) == 1.0
