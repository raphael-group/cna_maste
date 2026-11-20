"""`sandbox.spot_assignment.compute_loglike_spot_assignment_given_emission`, the dense field, against a Python sum.

Referee: each spot's indexed RDR and BAF emissions summed over bins in Python, the RDR weighted by the spot's valid
BAF over valid RDR segment counts, to 1e-12 relative.
"""

from __future__ import annotations

import numpy as np
import pytest

from cnamaste.sandbox.spot_assignment import compute_loglike_spot_assignment_given_emission


@pytest.mark.parametrize("weighted", [True, False])
@pytest.mark.parametrize("pred_ndim", [1, 2])
def test_the_field_is_the_python_sum_of_the_indexed_emissions(pred_ndim: int, weighted: bool) -> None:
    """Five spots, three clones, seven bins, four states; pred concatenated by clone (1-D) or (n_obs, n_clones).
    Weighted: `bb_valid / nb_valid` per spot, 1 where either is 0.
    Fails where a clone reads another's states, the 1-D and 2-D layouts disagree, or a weight is misapplied."""
    rng = np.random.default_rng(31)
    n_states, n_obs, n_spots, n_clones = 4, 7, 5, 3
    rdr, baf = rng.normal(-3.0, 1.0, (2, n_states, n_obs, n_spots))
    states = rng.integers(0, n_states, (n_obs, n_clones))
    nb_valid, bb_valid = np.array([7, 0, 5, 3, 6]), np.array([4, 2, 0, 6, 5])
    pred = states.T.ravel() if pred_ndim == 1 else states
    got = compute_loglike_spot_assignment_given_emission(
        n_spots, nb_valid, bb_valid, rdr, baf, pred, n_obs, n_clones, weighted
    )

    expected = np.empty((n_spots, n_clones))
    for spot in range(n_spots):
        weight = 1.0
        if weighted and nb_valid[spot] > 0 and bb_valid[spot] > 0:
            weight = bb_valid[spot] / nb_valid[spot]
        for clone in range(n_clones):
            rdr_sum = sum(rdr[states[o, clone], o, spot] for o in range(n_obs))
            baf_sum = sum(baf[states[o, clone], o, spot] for o in range(n_obs))
            expected[spot, clone] = weight * rdr_sum + baf_sum
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=0.0)
