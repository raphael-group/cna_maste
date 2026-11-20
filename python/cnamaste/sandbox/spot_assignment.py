"""`compute_loglike_spot_assignment_given_emission`, each spot's log-likelihood under every clone from a dense
emission. No caller in `cnamaste`; `hmrf.get_labelling_unary` computes the field without the emission,
and `tests/test_hmm_emission_tabled.py` holds it to this reduction, bitwise. Not shipped."""

import numpy as np
from numba import njit, prange


@njit(parallel=True, cache=True)
def compute_loglike_spot_assignment_given_emission(
    n_spots,
    num_valid_nb_spotwise,
    num_valid_bb_spotwise,
    log_emission_rdr,
    log_emission_baf,
    pred,
    n_obs,
    n_clones,
    apply_rel_weight_rdr_baf=True,  # NB if False, rdr out-weighs the baf signal, which has many zero read_depth segments.
):
    """Compute each spot's emission log-likelihood under every clone.

    Optionally weights the RDR and BAF terms by each spot's number of valid segments per emission.

    Args:
        n_spots: the spots.
        num_valid_nb_spotwise: (n_spots,) bins with a read-depth baseline.
        num_valid_bb_spotwise: (n_spots,) bins with SNP UMIs.
        log_emission_rdr: (n_states, n_obs, n_spots) read-depth emission.
        log_emission_baf: (n_states, n_obs, n_spots) allele emission.
        pred: decoded states, concatenated by clone or (n_obs, n_clones).
        n_obs: the bins.
        n_clones: the clones.
        apply_rel_weight_rdr_baf: weight the read-depth term by valid allele over valid read-depth bins.

    Returns:
        The (n_spots, n_clones) log-likelihood field.
    """
    # NB compute the log likelihood for each spot, for all clones.
    loglike_spot_clone_assignment = np.zeros((n_spots, n_clones))
    rel_valid_emision_weight = np.ones(n_spots, dtype=np.float64)

    if apply_rel_weight_rdr_baf:
        for i in prange(n_spots):
            # NB both normal and baf signals available.
            if num_valid_nb_spotwise[i] > 0 and num_valid_bb_spotwise[i] > 0:
                rel_valid_emision_weight[i] = num_valid_bb_spotwise[i] / num_valid_nb_spotwise[i]

    # NB numba evaluates .ndim at compile_time.
    is_1d_pred = pred.ndim == 1

    for spot in prange(n_spots):
        for c in range(n_clones):
            spot_log_like_rdr, spot_log_like_baf = 0.0, 0.0

            for o in range(n_obs):
                copy_state = pred[c * n_obs + o] if is_1d_pred else pred[o, c]

                spot_log_like_rdr += log_emission_rdr[copy_state, o, spot]
                spot_log_like_baf += log_emission_baf[copy_state, o, spot]

            loglike_spot_clone_assignment[spot, c] = (
                rel_valid_emision_weight[spot] * spot_log_like_rdr + spot_log_like_baf
            )

    return loglike_spot_clone_assignment
