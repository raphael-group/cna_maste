"""`hmm.baum_welch`'s forward-backward pass against every (state, phase) path enumerated, and its encoded emission
against the dense one; its max-product form, `get_viterbi_paths`, against the argmax over every path.

Referee: the joint log probability of each path over 6 bins, 2 copy states (x 2 phases where phased), written from
the model's definition: start `startprob[s]` (halved per phase); a step `transmat[s, s']` times the phase's stay
`1 - r_t` or switch `r_t`, or, with `penalize_phase_only_on_same_cnv`, times 1/2 off the copy diagonal. Two clones,
each its own chain per contig (4, 2). It costs 1 s, numba cache warm.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from scipy.special import logsumexp

from cnamaste.hmm.forward_backward import (
    EncodedEmission,
    get_contig_offsets,
    get_dense_encoded_emission,
    get_llf,
    get_log_alpha,
    get_log_beta,
    get_log_gamma,
    get_phase_steps,
    get_viterbi,
    get_viterbi_paths,
)
from cnamaste.hmm_utils import get_log_transmat

N_COPY_STATES, LENGTHS = 2, np.array([4, 2])
"""The enumeration's copy states and contigs."""


def get_path_log_probability(
    path: tuple[int, ...], log_startprob: np.ndarray, transmat: np.ndarray, switch: np.ndarray | None, penalize: bool
) -> float:
    """The log prior of one path, `phase * N_COPY_STATES + state` per bin; unphased where `switch` is None."""
    state, phase = np.array(path) % N_COPY_STATES, np.array(path) // N_COPY_STATES
    probability = np.exp(log_startprob[state[0]]) * (1.0 if switch is None else 0.5)
    for t in range(1, len(path)):
        phase_step = 1.0
        if switch is not None:
            phase_step = 1.0 - switch[t - 1] if phase[t] == phase[t - 1] else switch[t - 1]
            if penalize and state[t] != state[t - 1]:
                phase_step = 0.5
        probability *= transmat[state[t - 1], state[t]] * phase_step
    return float(np.log(probability))


@pytest.mark.parametrize(("phased", "penalize"), [(False, False), (True, False), (True, True)])
def test_posteriors_and_llf_are_the_enumerated_paths(phased: bool, penalize: bool) -> None:
    """Posteriors and llf to rtol 1e-12 of the enumeration, per clone and contig; each chain restarts at its contig.
    Fails where a step, the phase mix, a restart or the clone-major layout is misread."""
    rng = np.random.default_rng(102)
    n_clones, n_bins, n_paired = 2, int(LENGTHS.sum()), N_COPY_STATES * (2 if phased else 1)
    transmat = rng.dirichlet(np.ones(N_COPY_STATES), N_COPY_STATES)
    log_startprob = np.log(rng.dirichlet(np.ones(N_COPY_STATES)))
    switch = rng.uniform(0.05, 0.4, n_bins) if phased else None
    log_emission = rng.normal(-3.0, 1.0, (n_clones, n_bins, n_paired))

    starts = get_contig_offsets(LENGTHS)
    stay, log_switch = get_phase_steps(None if switch is None else np.log(switch), n_bins)
    encoded = get_dense_encoded_emission(log_emission)
    log_alpha = get_log_alpha(*encoded, log_startprob, np.log(transmat), starts, stay, log_switch, penalize)
    log_beta = get_log_beta(*encoded, np.log(transmat), starts, stay, log_switch, penalize)

    expected_llf, expected_posterior = 0.0, np.zeros((n_clones, n_bins, n_paired))
    for clone, (first, last) in itertools.product(range(n_clones), itertools.pairwise(starts)):
        paths = list(itertools.product(range(n_paired), repeat=last - first))
        steps = None if switch is None else switch[first:last]
        joint = np.array(
            [
                get_path_log_probability(p, log_startprob, transmat, steps, penalize)
                + sum(log_emission[clone, first + t, s] for t, s in enumerate(p))
                for p in paths
            ]
        )
        evidence = logsumexp(joint)
        expected_llf += evidence
        for p, weight in zip(paths, np.exp(joint - evidence), strict=True):
            expected_posterior[clone, np.arange(first, last), list(p)] += weight

    np.testing.assert_allclose(get_llf(log_alpha, starts), expected_llf, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(np.exp(get_log_gamma(log_alpha, log_beta)), expected_posterior, rtol=1e-12, atol=0.0)


def test_a_coded_emission_is_its_dense_gather_bitwise() -> None:
    """Class tables per clone and each bin's class give the forward and backward of their dense gather
    `rdr[c, rdr_index[c, g]] + baf[c, baf_index[c, g]]`, bitwise, phased. Fails where a bin reads another's class
    or clone."""
    rng = np.random.default_rng(0)
    n_clones, n_bins, n_states = 3, 12, 3
    rdr, baf = rng.normal(-3.0, 1.0, (n_clones, 4, 2 * n_states)), rng.normal(-2.0, 1.0, (n_clones, 5, 2 * n_states))
    rdr_index, baf_index = rng.integers(0, 4, (n_clones, n_bins)), rng.integers(0, 5, (n_clones, n_bins))
    clones = np.arange(n_clones)[:, None]
    dense = rdr[clones, rdr_index] + baf[clones, baf_index]

    args = (
        np.log(np.full(n_states, 1.0 / n_states)),
        np.log(rng.dirichlet(np.ones(n_states), n_states)),
        get_contig_offsets(np.array([7, 5])),
        *get_phase_steps(np.log(np.full(n_bins, 0.1)), n_bins),
        False,
    )
    encoded = EncodedEmission(rdr, rdr_index, baf, baf_index)
    assert np.array_equal(get_log_alpha(*encoded, *args), get_log_alpha(*get_dense_encoded_emission(dense), *args))
    assert np.array_equal(
        get_log_beta(*encoded, *args[1:]), get_log_beta(*get_dense_encoded_emission(dense), *args[1:])
    )


@pytest.mark.parametrize("stay", [0.7, 0.1, None], ids=["stay", "switch", "dense"])
def test_viterbi_is_the_argmax_over_every_path(stay: float | None) -> None:
    """2 clones, 3 states, 6 bins in contigs of 4 and 2, at a uniform switch or a random transition matrix: each
    clone's path and score are the best of all 729 paths', bitwise, summed in the kernel's order, by the K x K and the
    O(K) uniform-switch step alike. Fails where a step
    picks a wrong predecessor, a stay loses a tie, or a contig does not restart at the start prior."""
    rng = np.random.default_rng(5)
    emission = rng.normal(-2.0, 1.5, (2, 6, 3))
    if stay is None:
        log_transmat = np.log(rng.dirichlet(np.ones(3), 3))
    else:
        log_transmat = get_log_transmat(3, stay)
    log_startprob, lengths = np.log(np.array([0.5, 0.3, 0.2])), [4, 2]
    paths = np.array(list(itertools.product(range(3), repeat=6)))
    found, scores = get_viterbi_paths(
        get_dense_encoded_emission(emission), log_startprob, log_transmat, np.array(lengths)
    )
    for clone in range(2):
        total, first = np.zeros(len(paths)), 0
        for le in lengths:
            score = log_startprob[paths[:, first]] + emission[clone, first, paths[:, first]]
            for g in range(first + 1, first + le):
                score = (score + log_transmat[paths[:, g - 1], paths[:, g]]) + emission[clone, g, paths[:, g]]
            total, first = total + score, first + le
        best = np.argmax(total)
        assert np.sort(total)[-2] < total[best]
        assert np.array_equal(found[clone], paths[best]) and scores[clone] == total[best]
    encoded, starts = get_dense_encoded_emission(emission), get_contig_offsets(np.array(lengths))
    for uniform in (False, True) if stay is not None else (False,):
        dense = get_viterbi(*encoded, log_startprob, log_transmat, starts, uniform)
        assert np.array_equal(dense[0], found) and np.array_equal(np.cumsum(dense[1], axis=1)[:, -1], scores)
