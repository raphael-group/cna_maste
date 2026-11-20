"""`phasing.get_refined_lengths` and `phasing.get_phase_indicator` against CalicoST `c1abcae`'s
`phasing.initial_phase_given_partition`, after its per-clone fits.

The referee is a verbatim copy of CalicoST's lines that read the fits (`call_calicost_phasing`): the function fits
inside its loop, so no call of it takes the fits as given. Its outputs are recorded (`calicost_cache`). Inputs: 3
clones of 10, 10 and 40 spots, each decoded on 2 contigs of 30 and 25 blocks through 3 states, minor BAF 0.1, 0.3 and
0.5, one shared `p_binom` (cnamaste fits one; CalicoST one per clone), each block's phase drawn per clone, seeded; no
BAF step equals the 0.1 threshold. The file runs in under 1 s.
"""

from __future__ import annotations

import numpy as np
import pytest

from calicost_cache import Referee, departs
from cnamaste.phasing import get_model_baf_profiles, get_phase_indicator, get_refined_lengths

N_STATES = 3
P_BINOM = np.array([0.1, 0.3, 0.5])
LENGTHS = np.array([30, 25])
CLONE_SIZES = (10, 10, 40)


@pytest.fixture(scope="module")
def pred() -> np.ndarray:
    """Each clone's (phase, state) per block, `(n_clones, n_obs)`: states in runs of 5 to 15 blocks, phases drawn."""
    rng = np.random.default_rng(0)
    n_obs = int(LENGTHS.sum())
    states = np.stack([np.repeat(rng.integers(0, N_STATES, 12), rng.integers(5, 16, 12))[:n_obs] for _ in CLONE_SIZES])
    return states + N_STATES * rng.integers(0, 2, states.shape)


def call_calicost_phasing(pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """CalicoST's phase indicator and refined lengths, given each clone's decoded `pred` at `P_BINOM`."""
    lengths, n_states = LENGTHS, N_STATES
    initial_clone_index = [np.arange(n) + sum(CLONE_SIZES[:i]) for i, n in enumerate(CLONE_SIZES)]
    res = {"new_p_binom": P_BINOM[:, None]}
    # NB copied from phasing.py:51, 58-59, 66-71 and 73-95 (c1abcae), each clone's fit `res` and `pred` given.
    EPS_BAF = 0.05
    baf_profiles = np.zeros((pred.shape[0], pred.shape[1]))
    for i in range(pred.shape[0]):
        this_baf_profiles = np.where(
            pred[i] < n_states, res["new_p_binom"][pred[i] % n_states, 0], 1 - res["new_p_binom"][pred[i] % n_states, 0]
        )
        this_baf_profiles[np.abs(this_baf_profiles - 0.5) < EPS_BAF] = 0.5
        baf_profiles[i, :] = this_baf_profiles
    n_total_spots = np.sum([len(x) for x in initial_clone_index])
    population_baf = np.array([1.0 * len(x) / n_total_spots for x in initial_clone_index]) @ baf_profiles
    adj_baf_profiles = np.where(baf_profiles < 0.5, baf_profiles, 1 - baf_profiles)
    phase_indicator = population_baf < 0.5
    refined_lengths = []
    cumlen = 0
    for le in lengths:
        s = 0
        for i in range(le):
            if i > s + 10 and np.any(
                np.abs(adj_baf_profiles[:, i + cumlen] - adj_baf_profiles[:, i + cumlen - 1]) > 0.1
            ):
                refined_lengths.append(i - s)
                s = i
        refined_lengths.append(le - s)
        cumlen += le
    refined_lengths = np.array(refined_lengths)
    return phase_indicator, refined_lengths


def get_calicost_phasing(referee: Referee, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`call_calicost_phasing`, from the cache, recorded whole."""
    phase_indicator, refined_lengths = referee(
        "phasing.given_partition", (pred,), lambda: call_calicost_phasing(pred), kept=(0, 1)
    )
    return phase_indicator, refined_lengths


def test_get_refined_lengths_is_calicost_s(pred: np.ndarray, referee: Referee) -> None:
    """At `baf_change_threshold` 0.1 and `min_new_segment_size` 10, cnamaste's refined lengths are CalicoST's, bitwise,
    and split the 55 blocks into more than the 2 contigs. Fails where the minor BAF, the step's threshold or the
    minimum segment drifts."""
    _, theirs = get_calicost_phasing(referee, pred)
    model_baf_profiles = get_model_baf_profiles(pred, P_BINOM, N_STATES)
    minor_baf_profiles = np.where(model_baf_profiles < 0.5, model_baf_profiles, 1.0 - model_baf_profiles)
    ours = get_refined_lengths(minor_baf_profiles, LENGTHS, 0.1, 10)
    assert np.array_equal(ours, theirs)
    assert len(theirs) > len(LENGTHS) and theirs.sum() == LENGTHS.sum()


@departs("CalicoST phases a block by the spot-weighted population BAF; cnamaste by the clones' majority vote")
def test_get_phase_indicator_is_calicost_s(pred: np.ndarray, referee: Referee) -> None:
    """Each block's phase, CalicoST's against `get_phase_indicator`'s at `eps_baf` 0.05. Where the 40-spot clone's
    phase opposes the two 10-spot clones', CalicoST follows it, cnamaste the two: 12 of 55 blocks differ."""
    theirs, _ = get_calicost_phasing(referee, pred)
    ours = get_phase_indicator(pred, P_BINOM, N_STATES, known_normal=False, eps_baf=0.05)
    assert np.array_equal(ours, theirs.astype(int))
