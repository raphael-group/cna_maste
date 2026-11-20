"""The phased emission and forward-backward against CalicoST `c1abcae`'s `hmm_NB_BB_phaseswitch.hmm_sitewise`.

The referee is CalicoST's own output on the same input, computed once and recorded whole (`calicost_cache`):
`compute_emission_probability_nb_betabinom`, `forward_lattice`, `backward_lattice`, and `posterior_nb_bb_sitewise` for
the posterior from counts. Inputs: `calicost_hmm_inputs.get_hmm_inputs`, with parameters that differ per spot, and its
per-bin phase-switch probabilities. CalicoST's lattices are `njit` without a cache: a cache miss pays their
compilation, 4 s; a hit compiles nothing of CalicoST's.

Tolerance:
- emission, 1e-12 nats absolute: as `test_hmm_nophasing`'s; cnamaste also takes the switched allele term as an
  lgamma difference from the unswitched one, where CalicoST evaluates `scipy.stats.betabinom` again. At zero
  coverage that difference is the rounding of 0 (7e-15 nats), where CalicoST's guard writes 0.
- lattices, 1e-13 relative: CalicoST builds each step's transitions as `log(kron(exp(switch), exp(transmat)))`,
  cnamaste as `switch + transmat`; one rounding per entry, carried through each logsumexp (bitwise on this input,
  1.9e-14 at most over 200 seeded cases).
- posterior from counts, 1e-12 absolute in probability: the two differences above, normalized (measured 8.4e-14).
"""

from __future__ import annotations

import numpy as np
from calicost.hmm_NB_BB_phaseswitch import hmm_sitewise, posterior_nb_bb_sitewise

import cnamaste.calicost  # noqa: F401  (imports CalicoST with the environment held, as cnamaste does)
from calicost_cache import Referee
from calicost_hmm_inputs import (
    HMMInputs,
    get_calicost_emission,
    get_dense_emission,
    get_emission_atol,
    get_hmm_inputs,
    get_state_major_lattices,
)

LATTICE_RTOL = 1e-13
POSTERIOR_ATOL = 1e-12
SPOT_SHIFT = np.array([[0.0, 0.1, -0.1, 0.2]])
"""Each spot's offset to `log_mu` and `p_binom`, so a parameter read from another spot's column is caught."""


def get_phased_inputs() -> HMMInputs:
    """`get_hmm_inputs` with `SPOT_SHIFT` applied: `log_mu` and `p_binom` `(n_states, n_spots)`."""
    h = get_hmm_inputs()
    return h._replace(log_mu=h.log_mu + SPOT_SHIFT, p_binom=h.p_binom + SPOT_SHIFT / 4)


def test_the_emission_is_calicost_s(referee: Referee) -> None:
    """Read depth and allele log emissions over (state, phase) pairs, per-spot parameters: CalicoST's within 1e-12
    nats, the read depth repeated for both phases, and 0 in both where baseline or coverage is, but for the switched
    phase at zero coverage, which is the lgamma difference's rounding of 0 (7e-15). Fails where the switched allele
    term, a spot's parameter column or the zero-depth guards drift."""
    h = get_phased_inputs()
    ours = get_dense_emission(h, phased=True)
    theirs = get_calicost_emission(hmm_sitewise, h, referee)
    for our, their in zip(ours, theirs):
        assert our.shape == their.shape == (6, 60, 4)
        np.testing.assert_allclose(our, their, rtol=0.0, atol=get_emission_atol())
    assert np.array_equal(ours[0] == 0.0, theirs[0] == 0.0) and np.array_equal(ours[0][:3], ours[0][3:])
    assert np.array_equal(ours[1][:3] == 0.0, theirs[1][:3] == 0.0)
    assert np.abs(ours[1][3:, 5, :]).max() <= get_emission_atol() and (theirs[1][3:, 5, :] == 0.0).all()


def test_forward_backward_and_posterior_are_calicost_s(referee: Referee) -> None:
    """On CalicoST's emission and per-bin switch probabilities: its forward and backward lattices within 1e-13
    relative; from counts, cnamaste's emission and posterior against CalicoST's `posterior_nb_bb_sitewise`, within
    1e-12. Fails where the phase-switch transitions, the paired start probabilities, the sequence starts or the
    normalization drift."""
    h = get_phased_inputs()
    rdr, baf = get_calicost_emission(hmm_sitewise, h, referee)
    args = (h.lengths, h.log_transmat, h.log_startprob, rdr + baf, h.log_sitewise_transmat)

    def call() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        posterior = posterior_nb_bb_sitewise(
            h.X,
            h.lengths,
            h.base_nb_mean,
            h.log_mu,
            np.broadcast_to(h.alphas, h.log_mu.shape),
            h.total_bb_RD,
            h.p_binom,
            np.broadcast_to(h.taus, h.log_mu.shape),
            h.log_startprob,
            h.log_transmat,
            h.log_sitewise_transmat,
        )
        return hmm_sitewise.forward_lattice(*args), hmm_sitewise.backward_lattice(*args), posterior

    log_alpha, log_beta, theirs = referee("hmm.lattices.hmm_sitewise", (*args, *h), call, kept=(0, 1, 2))
    ours_alpha, ours_beta, _ = get_state_major_lattices(*args)
    np.testing.assert_allclose(ours_alpha, log_alpha, rtol=LATTICE_RTOL, atol=0.0)
    np.testing.assert_allclose(ours_beta, log_beta, rtol=LATTICE_RTOL, atol=0.0)

    ours_emission = sum(get_dense_emission(h, phased=True))
    ours = get_state_major_lattices(*args[:3], ours_emission, h.log_sitewise_transmat)[2]
    np.testing.assert_allclose(np.exp(ours), np.exp(theirs), rtol=0.0, atol=POSTERIOR_ATOL)
