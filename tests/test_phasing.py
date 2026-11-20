"""`phasing`: the phase vote, the length refinement, the phasing of a planted partition, and the phase-switch count.

Referees: hand-written cases for `get_phase_indicator`, `get_refined_lengths` and `cumulative_phase_switches`, each
expected value written out; for `initial_phase_given_partition`, the phases, balanced segments and BAF steps planted
in a 3-clone, 2-contig, 200-block simulation. The tie -> 1 and unvoted -> 0 conventions are pinned as they stand
(T- #27). It costs 0.9 s in calls, 2.9 s with imports, numba cache warm.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
import pytest

from cnamaste.config import YAMLConfig, get_global_config, set_global_config
from cnamaste.hmrf_utils import get_clone_assignment
from cnamaste.initialize.genomic import get_initial_parameters
from cnamaste.phasing import (
    cumulative_phase_switches,
    get_model_baf_profiles,
    get_phase_indicator,
    get_refined_lengths,
    initial_phase_given_partition,
)
from cnamaste.qa.sims import get_project
from labelled_counts import get_labelled_counts

EPS_BAF = 0.1
"""`initial_phase_given_partition`'s no-vote half-width about 1/2."""

P_BINOM = np.array([[0.1], [0.45], [0.3]])
"""Three states: 0.45 lies within `EPS_BAF` of 1/2, so neither it nor its mirror 0.55 votes."""

UP, DOWN, MUTE = 0, 3, 1
"""`pred` codes for `P_BINOM`'s state 0 unflipped (phase 1), flipped (phase 0), and state 1 (no vote)."""


@pytest.mark.parametrize(
    ("clones", "known_normal", "expected"),
    [
        ((UP, UP, UP), False, 1),
        ((DOWN, DOWN, DOWN), False, 0),
        ((UP, DOWN, DOWN), False, 0),
        ((MUTE, MUTE, MUTE), False, 0),
        ((MUTE + 3, MUTE, MUTE + 3), False, 0),
        ((UP, DOWN, MUTE), False, 1),
        ((UP, DOWN, MUTE), True, 0),
        ((DOWN, UP, DOWN), True, 1),
        ((UP, MUTE, MUTE), True, 0),
    ],
)
def test_the_phase_is_the_clones_majority_vote(clones: tuple[int, ...], known_normal: bool, expected: int) -> None:
    """Phase 1 is `pred < n_states`, the state as fit; a clone within `EPS_BAF` of 1/2 casts no vote; a tie gives 1
    and a block with no vote 0 (T- #27); `known_normal` drops clone 0's vote and no other's. Fails where a convention
    moves, or a muted or dropped clone votes."""
    pred = np.array(clones).reshape(-1, 1)
    found = get_phase_indicator(pred, P_BINOM, len(P_BINOM), known_normal, EPS_BAF)
    assert found.tolist() == [expected]


def test_the_model_baf_is_the_state_or_its_mirror() -> None:
    """`get_model_baf_profiles`: `p_binom[pred]` below `n_states`, `1 - p_binom[pred - n_states]` above. Fails where
    the phase reads the mirror as the state."""
    pred = np.array([[0, 1, 2, 3, 4, 5]])
    np.testing.assert_array_equal(get_model_baf_profiles(pred, P_BINOM, 3), [[0.1, 0.45, 0.3, 0.9, 0.55, 0.7]])


STEP = 0.0625
"""A minor-BAF step exactly representable in binary, used as the threshold: `>=` is tested at equality."""


def test_the_lengths_split_at_a_step_past_the_minimum_segment() -> None:
    """Contig A (30 blocks): clone 0 steps at 10 (= s + 10, no split), clone 1 at 11 (splits), clone 0 at 15 (closer
    than 10, no split) and at 22 by exactly the threshold (splits). Contig B (25): a step across the contig boundary
    and one below the threshold, neither splits. Contig C (8): a step, shorter than 10, never splits. Fails where
    either inequality changes, one clone's step stops splitting, or a split crosses a contig."""
    lengths = np.array([30, 25, 8])
    minor = np.full((2, lengths.sum()), 0.25)
    minor[0, 10:] = 0.375
    minor[1, 11:] = 0.125
    minor[0, 15:] = 0.25
    minor[0, 22:30] = 0.25 + STEP
    minor[:, 30:] = 0.5
    minor[0, 50:] = 0.5 - STEP / 2
    minor[:, 60:] = 0.0
    assert get_refined_lengths(minor, lengths, STEP, 10).tolist() == [11, 11, 8, 25, 8]


def test_the_refined_lengths_tile_each_contig() -> None:
    """A minor BAF stepping at every block: the refined lengths sum to `sum(lengths)`, each contig ends at its own
    length, and every segment but a contig's last holds `min_new_segment_size + 1` blocks. Fails where a segment
    crosses a contig or blocks are lost."""
    lengths = np.array([40, 7, 23, 11])
    minor = np.random.default_rng(0).uniform(0.0, 0.4, (3, lengths.sum()))
    minor[:, 1::2] = 0.5
    found = get_refined_lengths(minor, lengths, 0.05, 10)
    ends = np.cumsum(found)
    assert found.sum() == lengths.sum()
    assert set(np.cumsum(lengths)) <= set(ends)
    last = np.isin(ends, np.cumsum(lengths))
    assert set(found[~last]) == {11}


class Planted(NamedTuple):
    """A simulation and its truth: the counts `initial_phase_given_partition` reads, the per-block phase and BAF."""

    X: np.ndarray
    total_bb_RD: np.ndarray
    clone_index: list[np.ndarray]
    phase: np.ndarray
    baf: np.ndarray


LENGTHS = np.array([100, 100])
SEGMENTS = [
    (0, 0, 40, (0.1, 0.3, 0.5)),
    (0, 40, 70, (0.5, 0.5, 0.5)),
    (0, 70, 100, (0.3, 0.1, 0.1)),
    (1, 0, 50, (0.5, 0.5, 0.3)),
    (1, 50, 100, (0.1, 0.1, 0.1)),
]
"""(contig, first block, end block, BAF per clone): a balanced segment, and steps at blocks 40, 70 and 50."""
SWITCH = 0.1
"""Planted phase-switch probability per block, the HMM's `log_sitewise_transmat`."""
SPOTS, DEPTH = 4, 30


def get_planted(seed: int) -> Planted:
    """3 clones of `SPOTS` spots, each at `DEPTH` reads per block; phase 1 makes the B allele the minor one."""
    rng = np.random.default_rng(seed)
    offset = np.r_[0, np.cumsum(LENGTHS)]
    baf = np.empty((LENGTHS.sum(), 3))
    for contig, start, end, clone_baf in SEGMENTS:
        baf[offset[contig] + start : offset[contig] + end] = clone_baf
    switches = rng.random(LENGTHS.sum()) < SWITCH
    switches[offset[:-1]] = rng.integers(2, size=len(LENGTHS)).astype(bool)
    phase = np.concatenate([np.cumsum(switches[s:e]) % 2 for s, e in itertools.pairwise(offset)])
    b_allele = np.repeat(np.where(phase[:, None] == 1, baf, 1.0 - baf), SPOTS, axis=1)
    total_bb_RD = np.full(b_allele.shape, float(DEPTH))
    X = np.stack([np.full(b_allele.shape, 100.0), rng.binomial(DEPTH, b_allele).astype(float)], axis=1)
    clone_index = [np.arange(k * SPOTS, (k + 1) * SPOTS) for k in range(3)]
    return Planted(X, total_bb_RD, clone_index, phase, baf)


def phase_planted(planted: Planted, normal_spot: np.ndarray | None = None) -> Any:
    """`initial_phase_given_partition` on `planted`, at `hmm.n_states` and `mstep.max_iter` of the shipped config."""
    config = get_global_config()
    counts = get_labelled_counts(planted.X[:, 0], planted.X[:, 1], planted.total_bb_RD, LENGTHS, normal_spot)
    labelling = get_clone_assignment(counts.n_spots, planted.clone_index)
    return initial_phase_given_partition(
        counts,
        planted.clone_index,
        get_initial_parameters(counts, labelling, config, stage="phasing"),
        config.hmm.n_states,
        np.full(LENGTHS.sum(), np.log(SWITCH)),
        "sp",
        config.hmm.t_phaseing,
        tol=1.0e-3,
    )


@pytest.fixture(scope="module")
def shipped() -> Iterator[None]:
    """The repository-root `zenodo_sim_config.yaml` as the global config, the previous one restored after."""
    previous = get_global_config()
    set_global_config(YAMLConfig.from_file(get_project() / "zenodo_sim_config.yaml"))
    yield
    set_global_config(previous)


@pytest.fixture(scope="module")
def phased(shipped: None) -> tuple[Planted, Any]:
    """`get_planted(0)` and its phasing, shared by the recovery and determinism tests."""
    planted = get_planted(0)
    return planted, phase_planted(planted)


def test_a_planted_partition_recovers_its_phases_and_steps(phased: tuple[Planted, Any]) -> None:
    """At least 95% of informative blocks take their planted phase, up to one flip per contig; every clone of a
    balanced block lies within `EPS_BAF` of 1/2, so the block takes phase 0; each refined boundary lies within 1
    block of a planted step, and each step has one. Fails where the vote, the mirror convention or the split
    moves off the truth."""
    planted, (res, phase_indicator, refined_lengths) = phased
    informative = (np.abs(planted.baf - 0.5) > 0.0).any(axis=1)
    offset = np.r_[0, np.cumsum(LENGTHS)]
    for start, end in itertools.pairwise(offset):
        kept = informative[start:end]
        agree = np.mean(phase_indicator[start:end][kept] == planted.phase[start:end][kept])
        assert max(agree, 1.0 - agree) >= 0.95

    n_clones = len(planted.clone_index)
    pred = np.argmax(res["log_gamma"].reshape(n_clones, -1, 2 * len(res.parameters.p_binom)), axis=-1)
    model_baf = get_model_baf_profiles(pred, res.parameters.p_binom, len(res.parameters.p_binom))
    assert np.all(np.abs(model_baf[:, ~informative] - 0.5) < EPS_BAF)
    assert not phase_indicator[~informative].any()

    found = set(np.cumsum(refined_lengths)[:-1]) - set(offset[1:-1])
    planted_steps = {offset[contig] + start for contig, start, _, _ in SEGMENTS if start > 0}
    assert all(min(abs(b - s) for s in planted_steps) <= 1 for b in found)
    assert all(min(abs(b - s) for b in found) <= 1 for s in planted_steps)


def test_phasing_is_deterministic_and_refuses_a_baseline(phased: tuple[Planted, Any]) -> None:
    """The same call twice is bitwise identical, `res`, phases and lengths; a labelled `normal_spot`, so a nonzero baseline, raises.
    Fails where the fit reads global state or a random stream it does not own, or the refusal is dropped."""
    planted, (res, phase_indicator, refined_lengths) = phased
    again, again_phase, again_lengths = phase_planted(planted)
    fields, again_fields = res.get_npz_fields(), again.get_npz_fields()
    assert list(fields) == list(again_fields)
    for key, value in fields.items():
        np.testing.assert_array_equal(value, again_fields[key])
    np.testing.assert_array_equal(phase_indicator, again_phase)
    np.testing.assert_array_equal(refined_lengths, again_lengths)
    with pytest.raises(AssertionError):
        phase_planted(planted, normal_spot=np.ones(planted.X.shape[2]))


def test_phase_switches_count_along_each_contig() -> None:
    """Blocks 0-3 on contig 1, 4-6 on contig 2, bins 20, 35, 50, 60 with 35 holding blocks 2 and 3; rows shuffled,
    a row without a block and block 7 without a bin. Switches 0-1, 1-2, 2-3 and 4-5 count; 3-4 crosses the contig and
    does not. Fails where a contig's count leaks into the next, a dropped row counts, or bins leave rank order."""
    rows = pd.DataFrame(
        {
            "CHR": [1, 1, 1, 1, 1, 2, 2, 2, 2, 1, 2],
            "block_id": [0, 1, 1, 2, 3, 4, 5, 6, 7, np.nan, 6],
            "bin_id": [20, 20, 20, 35, 35, 50, 50, 60, np.nan, 20, 60],
        }
    ).sample(frac=1.0, random_state=0)
    phase_indicator = np.array([0, 1, 0, 1, 0, 1, 1, 0])
    assert cumulative_phase_switches(rows, phase_indicator).tolist() == [1, 3, 1, 1]
