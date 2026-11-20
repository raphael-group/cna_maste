"""CalicoST's calls, adapted to cnamaste's: each as a CalicoST entry point makes it, on cnamaste's arguments.

- `calicost_fixdiploid` -- the diploid decoder for one clone (`ab_decode.decoder: calicost`), as `calicost_main`
  calls it.
- `calicost_diploid_decoder` -- `calicost_fixdiploid` over every clone, as `calicost_main` loops it.
"""

from __future__ import annotations

import numpy as np

from cnamaste.calicost import (
    fixdiploid_copy_numbers as hill_climbing_integer_copynumber_fixdiploid,
)
from cnamaste.config import start_time
from cnamaste.logger import get_logger

logger = get_logger(__name__, start_time=start_time)


def calicost_fixdiploid(log_mu, base_nb_mean, p_binom, pred_cnv, nondiploid_rdrdist):
    """CalicoST's diploid decoder for one clone (`ab_decode.decoder: calicost`), as `calicost_main` calls it.

    c1abcae, calicost_main.py:329-337: `log_mu` rescaled so sum_bin lambda * mu is 1, then
    `hill_climbing_integer_copynumber_fixdiploid` at its defaults, retried at `min_prop_threshold=0.02`.
    Departures: a second failure raises, where CalicoST skips the clone; the ploidy returned is NaN.

    Args: `log_mu`, `base_nb_mean`, `p_binom`, `pred_cnv` the clone's fit; `nondiploid_rdrdist`. `nonbalance_bafdist`
    takes CalicoST's None, as the 1.0 it held never binds on `|p - 1/2| <= 1/2`.

    Returns:
        `(copies (n_states, 2), loss, ploidy)`.
    """
    # NB names the decoder in the run's log by its caller, `ab_decode.decoder: calicost`.
    logger.info(f"Decoding integer (A, B) by CalicoST's diploid hill climb over {len(pred_cnv)} bins.")
    lambd = base_nb_mean / np.sum(base_nb_mean)
    adjusted = np.log(np.exp(log_mu) / np.sum(np.exp(log_mu[pred_cnv]) * lambd))
    kwargs = {"nondiploid_rdrdist": nondiploid_rdrdist}
    try:
        copies, loss = hill_climbing_integer_copynumber_fixdiploid(adjusted, base_nb_mean, p_binom, pred_cnv, **kwargs)
    except Exception:  # noqa: BLE001  NB CalicoST's bare `except:` retry, narrowed to Exception
        copies, loss = hill_climbing_integer_copynumber_fixdiploid(
            adjusted, base_nb_mean, p_binom, pred_cnv, min_prop_threshold=0.02, **kwargs
        )
    return np.asarray(copies, dtype=int), float(loss), np.nan


def calicost_diploid_decoder(log_mu, base_nb_mean, p_binom, pred_cnv, nondiploid_rdrdist):
    """CalicoST's diploid decoder over every clone (`ab_decode.decoder: calicost`), `calicost_fixdiploid` per clone.

    c1abcae, calicost_main.py:329-367 loops the clones. Departure: a clone with no read depth keeps (1, 1) in every
    state, where CalicoST divides by its zero total; `run_cnamaste` writes no column for it.

    Args: `log_mu`, `p_binom` the fit's per-state values; `base_nb_mean`, `pred_cnv` (bins, clones);
        `nondiploid_rdrdist`.

    Returns:
        Each clone's integer `(A, B)` per state, `(n_states, n_clones, 2)`.
    """
    base_nb_mean, pred_cnv = np.asarray(base_nb_mean), np.asarray(pred_cnv)
    copies = np.ones((len(log_mu), pred_cnv.shape[1], 2), dtype=int)
    for s in range(pred_cnv.shape[1]):
        if np.sum(base_nb_mean[:, s]) == 0:
            continue
        copies[:, s], loss, _ = calicost_fixdiploid(
            log_mu, base_nb_mean[:, s], p_binom, pred_cnv[:, s], nondiploid_rdrdist
        )
        logger.info(f"calicost: clone {s} integer copy number loss {loss:.4e}.")
    return copies
