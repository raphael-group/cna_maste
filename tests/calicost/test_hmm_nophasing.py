"""`hmm_nophasing`'s emission and forward-backward against CalicoST `c1abcae`'s `hmm_NB_BB_nophasing.hmm_nophasing`.

The referee is CalicoST's own output on the same input, computed once and recorded (`calicost_cache`):
`compute_emission_probability_nb_betabinom`, `forward_lattice`, `backward_lattice`, and `utils_hmm.compute_posterior_obs`
for the posterior. Inputs: 3 states, 60 bins in two sequences (25, 35), 4 spots, seeded counts with one bin of zero
baseline and one of zero coverage. CalicoST's lattices are `njit` without a cache: a cache miss pays their compilation,
4 s; a hit compiles nothing of CalicoST's.

Tolerance:
- emission, 1e-12 nats absolute: CalicoST takes `scipy.stats.nbinom` on (n, p) from the mean and variance and
  `scipy.stats.betabinom`; cnamaste the same log-pmfs as lgamma and log1p sums. Terms reach lgamma(200) ~ 860 nats,
  so rounding differs by ~1e-13 (measured 1.5e-13). Bins of zero baseline or coverage are 0 in both, exactly.
- lattices and posterior: none, as the sha256 of CalicoST's. The same recursion, logsumexp and spot sum, in the same
  order, up to 7 spots.

Departure, stated (a choice): cnamaste's dense emission takes one parameter per state, `(n_states, 1)`, shared by
every spot, as the fit carries one; CalicoST's takes `(n_states, n_spots)`. Compared on parameters shared by spots.
"""

from __future__ import annotations

import numpy as np
from calicost.hmm_NB_BB_nophasing import hmm_nophasing as calicost_hmm_nophasing
from calicost.utils_hmm import compute_posterior_obs

import cnamaste.calicost  # noqa: F401  (imports CalicoST with the environment held, as cnamaste does)
from calicost_cache import Referee, get_digest
from calicost_hmm_inputs import (
    get_calicost_emission,
    get_dense_emission,
    get_emission_atol,
    get_hmm_inputs,
    get_state_major_lattices,
)


def test_the_emission_is_calicost_s(referee: Referee) -> None:
    """Read depth and allele log emissions: CalicoST's within 1e-12 nats, and 0 in both where the baseline or the
    coverage is. Fails where the NB mean, dispersion, the beta-binomial shapes or a zero-depth guard drift."""
    h = get_hmm_inputs()
    ours = get_dense_emission(h, phased=False)
    theirs = get_calicost_emission(calicost_hmm_nophasing, h, referee)
    for our, their in zip(ours, theirs):
        np.testing.assert_allclose(our, their, rtol=0.0, atol=get_emission_atol())
        assert np.array_equal(our == 0.0, their == 0.0)
    assert (ours[0][:, 3, 1] == 0.0).all() and (ours[1][:, 5, :] == 0.0).all()


def test_forward_backward_and_posterior_are_calicost_s(referee: Referee) -> None:
    """On CalicoST's emission over two sequences: its forward and backward lattices and its log posterior, bitwise.
    Fails where the recursion, the sequence starts, the spot sum or the normalization drifts."""
    h = get_hmm_inputs()
    rdr, baf = get_calicost_emission(calicost_hmm_nophasing, h, referee)
    args = (h.lengths, h.log_transmat, h.log_startprob, rdr + baf, None)

    def call() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        log_alpha = calicost_hmm_nophasing.forward_lattice(*args)
        log_beta = calicost_hmm_nophasing.backward_lattice(*args)
        return log_alpha, log_beta, compute_posterior_obs(log_alpha, log_beta)

    ours = get_state_major_lattices(*args)
    assert tuple(map(get_digest, ours)) == referee("hmm.lattices.hmm_nophasing", args, call)
