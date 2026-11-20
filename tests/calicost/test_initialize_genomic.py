"""`initialize.genomic.get_calicost_states` against CalicoST `c1abcae`'s `utils_hmm.initialization_by_gmm`, as pinned.

The referee is CalicoST's own output (`cnamaste.calicost.gmm_start`) on the same counts and seed, computed once and
recorded (`calicost_cache`). Inputs: 200
seeded bins over 2 spots in four planted (mu, p) states. No capture is read.

Tolerance: none. Bitwise on float64, as the sha256 of CalicoST's output: on usable bins both run CalicoST's arithmetic
on the same arrays, at the run's arguments (`in_log_space=False`).

Departure, stated (a choice): `get_calicost_states` drops bins with zero baseline (`base_nb_mean`) or zero coverage
(`total_bb_RD`) in any spot before CalicoST's fit, where CalicoST forward-fills a 0/0 and refuses an x/0 as infinite.
A zero count at a positive baseline is kept, a ratio of 0 at `in_log_space=False`.

Drift, a strict xfail: CalicoST's phasing path starts its GMM in log space.
"""

from __future__ import annotations

import functools

import numpy as np
import pytest

from calicost_cache import Referee, departs, get_digest
from cnamaste.calicost import gmm_start
from cnamaste.config import YAMLConfig
from cnamaste.initialize.genomic import get_calicost_states
from cnamaste.spatio_genomic_counts import PseudobulkCounts

N_STATES, N_OBS, N_SPOTS = 4, 200, 2
CLIP = (0.15, 0.85)
"""`hmm.gmm_min_binom_prob`, `hmm.gmm_max_binom_prob`: off CalicoST's 0.1, 0.9 default, so the configured clip is
what reaches CalicoST."""


def get_counts() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Four states of 50 bins each, mu 0.5-2 and p 0.1-0.5, Poisson read depth and binomial allele counts.

    Returns: `X` (n_obs, 2, n_spots), `base_nb_mean` and `total_bb_RD` (n_obs, n_spots).
    """
    rng = np.random.default_rng(0)
    base = rng.uniform(20.0, 80.0, (N_OBS, N_SPOTS))
    total = rng.integers(10, 60, (N_OBS, N_SPOTS)).astype(float)
    mu = np.repeat([0.5, 1.0, 1.5, 2.0], N_OBS // N_STATES)[:, None]
    p = np.repeat([0.5, 0.3, 0.1, 0.4], N_OBS // N_STATES)[:, None]
    X = np.stack([rng.poisson(base * mu), rng.binomial(total.astype(int), p)], axis=1).astype(float)
    return X, base, total


def get_states(X: np.ndarray, base: np.ndarray, total: np.ndarray, params: str, only_minor: bool) -> tuple:
    """`get_calicost_states` on the counts at `CLIP`, `N_STATES` and seed 0."""
    config = YAMLConfig(
        {"hmm": {"n_states": N_STATES, "random_state": 0, "gmm_min_binom_prob": CLIP[0], "gmm_max_binom_prob": CLIP[1]}}
    )
    counts = PseudobulkCounts(X[:, 0, :].T, base.T, X[:, 1, :].T, total.T, None, {})
    return get_calicost_states(counts, config, params, only_minor)


def gmm_kwargs(only_minor: bool) -> dict:
    return {
        "random_state": 0,
        "in_log_space": False,
        "only_minor": only_minor,
        "min_binom_prob": CLIP[0],
        "max_binom_prob": CLIP[1],
    }


@pytest.mark.parametrize("params", ["smp", "sm", "sp"])
@pytest.mark.parametrize("only_minor", [True, False])
def test_calicost_states_are_calicost_s_on_usable_bins(params: str, only_minor: bool, referee: Referee) -> None:
    """Finite bins, seed 0, each channel set: CalicoST's `log_mu` and `p_binom`, bitwise, `None` where `params` omits
    a channel, and the four planted read-depth states recovered. Fails where the clip, the seed, the log space or
    `only_minor` stops reaching CalicoST, or a bin is dropped from usable input."""
    X, base, total = get_counts()
    log_mu, p_binom = get_states(X, base, total, params, only_minor)
    call = functools.partial(gmm_start, N_STATES, X, base, total, params, **gmm_kwargs(only_minor))
    theirs = referee(f"initialize.gmm[{params}-{only_minor}]", (X, base, total, params, gmm_kwargs(only_minor)), call)
    assert (get_digest(log_mu), get_digest(p_binom)) == theirs
    if "m" in params:
        assert np.allclose(np.sort(np.exp(log_mu[:, 0])), [0.5, 1.0, 1.5, 2.0], rtol=0.1)


def test_calicost_states_drop_bins_without_depth(referee: Referee) -> None:
    """Departure, stated (a choice): bin 3 has a count over zero baseline (x/0, inf), bin 50 none over none (0/0) and
    bin 7 no allele coverage in spot 1. `get_calicost_states` is CalicoST's fit on the 197 other bins, bitwise, where
    CalicoST on all 200 refuses the inf. A zero count at positive baseline (bin 9) is kept, bitwise CalicoST's on the
    197. Fails where another bin is dropped or a dropped one kept."""
    X, base, total = get_counts()
    base[[3, 50], 0] = 0.0
    X[50, 0, 0] = 0.0
    total[7, 1], X[7, 1, 1] = 0.0, 0.0
    usable = ~np.isin(np.arange(N_OBS), [3, 7, 50])
    ours = get_states(X, base, total, "smp", True)

    def get_gmm(name: str, X: np.ndarray, base: np.ndarray, total: np.ndarray) -> tuple[str, ...]:
        call = functools.partial(gmm_start, N_STATES, X, base, total, "smp", **gmm_kwargs(True))
        return referee(f"initialize.gmm.{name}", (X, base, total, "smp", gmm_kwargs(True)), call)

    assert (get_digest(ours[0]), get_digest(ours[1])) == get_gmm("usable", X[usable], base[usable], total[usable])
    with pytest.raises(ValueError, match="infinity"):
        get_gmm("all", X, base, total)
    X[9, 0, 0] = 0.0
    ours = get_states(X, base, total, "smp", True)
    assert (get_digest(ours[0]), get_digest(ours[1])) == get_gmm("zero_count", X[usable], base[usable], total[usable])


@departs("CalicoST's phasing fit starts its GMM on log depth ratios (in_log_space=True); cnamaste on the ratios")
def test_calicost_states_are_calicost_s_in_log_space(referee: Referee) -> None:
    """`params: smp`, seed 0: `get_calicost_states` is CalicoST's GMM start at `in_log_space=True`, the default that
    `pipeline_baum_welch` passes on the phasing path, bitwise."""
    X, base, total = get_counts()
    log_mu, p_binom = get_states(X, base, total, "smp", True)
    kwargs = gmm_kwargs(True) | {"in_log_space": True}
    call = functools.partial(gmm_start, N_STATES, X, base, total, "smp", **kwargs)
    assert (get_digest(log_mu), get_digest(p_binom)) == referee(
        "initialize.gmm.log_space", (X, base, total, kwargs), call
    )
