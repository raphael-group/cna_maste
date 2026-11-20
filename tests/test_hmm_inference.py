"""The HMM's fit, `hmm.baum_welch`, unphased and phased, and the run's helpers.

Referees: states planted at high depth for the fit, and the definitions written out for the helpers. The lattices and
posteriors are `test_hmm_baum_welch`'s, against every path. It costs 4 s, numba cache warm.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from cnamaste import hmm_nophasing
from cnamaste.emission import get_unconfounded, reshape_params_by_spot
from cnamaste.hmm.baum_welch import HmmTransitions, baum_welch, get_paths
from cnamaste.hmm.forward_backward import get_log_gamma
from cnamaste.hmm_nophasing import get_max_rdr_zeroed_baseline, get_neutral_state, get_path_baf_profiles
from cnamaste.hmm_params import CnaHmmParameters
from cnamaste.hmm_utils import get_log_transmat, get_normal_clone
from cnamaste.initialize.genomic import get_default_parameters
from cnamaste.spatio_genomic_counts import PseudobulkCounts

CONFIG = SimpleNamespace(
    hmm=SimpleNamespace(
        penalize_phase_only_on_same_cnv=False, max_iter_em=100, mstep_callback_cadence=2, callback_warm_start=True
    ),
    mstep=SimpleNamespace(solver="scipy_bfgs", max_iter=200, cost_rtol=1.0e-12),
)
"""The keys `baum_welch` reads."""


def test_a_bin_no_state_emits_is_refused() -> None:
    """A bin of -inf in every state: `RuntimeError`, not a NaN posterior. Fails where the bin is normalized."""
    log_alpha = np.random.default_rng(0).normal(-2.0, 1.0, (2, 7, 3))
    log_alpha[1, 2] = -np.inf
    with pytest.raises(RuntimeError):
        get_log_gamma(log_alpha, np.zeros_like(log_alpha))


def test_the_copy_state_is_the_most_probable_folded_over_phase() -> None:
    """`get_paths`: the argmax over states, over (phase, state) pairs folded to the state where phased. Fails where a
    phased state is read as another copy state."""
    log_gamma = np.log(np.array([[[0.1, 0.2, 0.6, 0.1], [0.6, 0.1, 0.1, 0.2]]]))
    assert get_paths(log_gamma).tolist() == [[2, 0]]
    assert get_paths(log_gamma.reshape(1, 2, 2, 2)).tolist() == [[0, 0]]


def planted(states: list[tuple[float, float]], bins: int = 60, flip: bool = False, seed: int = 1) -> dict:
    """Counts from `(mu, p)` states, `bins` bins each in turn, base 200 and depth 150; `flip` swaps the alleles on a
    contiguous half of each state's bins, as an unknown phase does."""
    rng = np.random.default_rng(seed)
    mu = np.repeat([s[0] for s in states], bins)
    p = np.repeat([s[1] for s in states], bins)
    if flip:
        p = np.where((np.arange(p.size) // (bins // 2)) % 2 == 1, 1.0 - p, p)
    base, total = np.full((p.size, 1), 200.0), np.full((p.size, 1), 150.0)
    reads = rng.poisson(base[:, 0] * mu)
    b_counts = rng.binomial(total[:, 0].astype(int), p)
    return {
        "X": np.stack([reads, b_counts], axis=1)[:, :, None].astype(float),
        "base": base,
        "total": total,
        "lengths": np.array([p.size]),
    }


def get_planted_counts(case: dict[str, np.ndarray]) -> PseudobulkCounts:
    """A planted case's counts as one clone's `PseudobulkCounts`."""
    X = case["X"]
    channels = (X[:, 0, :], case["base"], X[:, 1, :], case["total"])
    return PseudobulkCounts(*(np.ascontiguousarray(v.T) for v in channels), None, {})


def fit(
    case: dict[str, np.ndarray],
    init: CnaHmmParameters,
    params: str,
    log_sitewise_transmat: Any,
    previous_log_gamma: np.ndarray | None = None,
    **hmm: Any,
) -> Any:
    """`baum_welch` on a planted case, one clone, unconfounded; `hmm` overrides `CONFIG.hmm`'s keys."""
    transitions = HmmTransitions(get_log_transmat(init.p_binom.size, 1 - 1e-4), log_sitewise_transmat, case["lengths"])
    return baum_welch(
        get_planted_counts(case),
        init,
        get_unconfounded(1),
        transitions,
        params=params,
        tol=1e-4,
        previous_log_gamma=previous_log_gamma,
        config=SimpleNamespace(hmm=SimpleNamespace(**{**vars(CONFIG.hmm), **hmm}), mstep=CONFIG.mstep),
    )


def test_the_fit_recovers_planted_depth_and_baf_states() -> None:
    """`baum_welch` (`smp`) on three planted states, started a step off: each fitted state's `mu` within 5% and `p`
    within 0.03 of its planted one, and the decoded path the planted states on 97% of bins. Fails where the E or M
    step, the shared dispersion or the encoded emission drifts."""
    states = [(1.0, 0.5), (0.5, 0.1), (1.5, 0.33)]
    case = planted(states)
    init = CnaHmmParameters(
        None, np.log([1.1, 0.6, 1.3]), np.array([0.45, 0.15, 0.3]), np.full(3, 0.05), np.full(3, 100.0)
    )
    found = fit(case, init, "smp", None)
    np.testing.assert_allclose(np.exp(found.parameters.log_mu), [s[0] for s in states], rtol=0.05)
    np.testing.assert_allclose(found.parameters.p_binom, [s[1] for s in states], atol=0.03)
    assert np.mean(get_paths(found.log_gamma)[0] == np.repeat([0, 1, 2], 60)) >= 0.97
    assert np.isfinite(found.llf)


# NB dev 860962d's posterior log Z_c fit, its outer loop at max_iter 100, on the callback fit's posteriors.
POSTERIOR_FIT = (
    -1281.1173794540318,
    [1.0601927051035887, 0.5187129588766728, 1.5601824136329994],
    [0.49833316569699915, 0.09978765076917236, 0.3352180374258666],
)


def test_the_posterior_fit_without_callback_warm_start_is_dev_s(monkeypatch: pytest.MonkeyPatch) -> None:
    """A callback fit at `max_iter_em` 1, then `hmrf.logmu_shift_type` posterior on its posteriors at
    `callback_warm_start` false and `max_iter_em` 100: llf to 1e-12, `mu` to 1e-7 and `p` to 1e-7 of dev 860962d's, the
    M step's blocked sums and `mstep.cost_rtol` apart. Fails where the EM loop departs from dev's posterior outer loop."""
    states = [(1.0, 0.5), (0.5, 0.1), (1.5, 0.33)]
    case = planted(states)
    init = CnaHmmParameters(
        None, np.log([1.1, 0.6, 1.3]), np.array([0.45, 0.15, 0.3]), np.full(3, 0.05), np.full(3, 100.0)
    )
    first = fit(case, init, "smp", None, max_iter_em=1, callback_warm_start=True)
    monkeypatch.setattr(hmm_nophasing, "get_logmu_shift_type", lambda: "posterior")
    found = fit(case, init, "smp", None, first.log_gamma, max_iter_em=100, callback_warm_start=False)
    llf, mu, p_binom = POSTERIOR_FIT
    np.testing.assert_allclose(float(found.llf), llf, rtol=1e-12, atol=0)
    np.testing.assert_allclose(np.exp(found.parameters.log_mu), mu, rtol=1e-7, atol=0)
    np.testing.assert_allclose(found.parameters.p_binom, p_binom, rtol=0, atol=1e-7)


def test_the_phased_fit_recovers_the_minor_fraction_and_the_switches() -> None:
    """Phased `baum_welch` (`sp`) on two planted BAF states, each with its alleles swapped on half its bins: the
    fitted `p` within 0.03 of the planted minor fractions, and the decoded phase the planted one on 95% of bins (up to
    a global swap). Fails where the switched half of the emission or the phase transitions are misread."""
    case = planted([(1.0, 0.1), (1.0, 0.3)], flip=True)
    init = get_default_parameters(2)._replace(p_binom=np.array([0.15, 0.35]), taus=np.full(2, 100.0))
    found = fit(case, init, "sp", np.full(case["lengths"][0], np.log(1e-3)))
    np.testing.assert_allclose(np.sort(found.parameters.p_binom), [0.1, 0.3], atol=0.03)
    phase = np.argmax(found.log_gamma[0].reshape(120, 4), axis=-1) >= 2
    planted_phase = (np.arange(120) // 30) % 2 == 1
    assert max(np.mean(phase == planted_phase), np.mean(phase != planted_phase)) >= 0.95


# -- the run's helpers ---------------------------------------------------------------------------------------------


def test_a_per_state_parameter_is_repeated_over_spots_unless_given_per_spot() -> None:
    """`(n,)` and `(n, 1)` are repeated to `(n, n_spots)`; `(n, n_spots)` is returned as given."""
    per_spot = np.arange(6.0).reshape(3, 2)
    assert np.array_equal(reshape_params_by_spot(np.array([1.0, 2.0, 3.0]), 2), [[1, 1], [2, 2], [3, 3]])
    assert np.array_equal(reshape_params_by_spot(np.array([[1.0], [2.0], [3.0]]), 2), [[1, 1], [2, 2], [3, 3]])
    assert np.array_equal(reshape_params_by_spot(per_spot, 2), per_spot)


def test_bins_above_max_rdr_lose_their_baseline() -> None:
    """A bin whose count exceeds `max_rdr` times its baseline is zeroed; a zero baseline stays zero; the input is not
    written. Fails where the ceiling is read the other way or the caller's baseline is modified."""
    counts, base = np.array([[10.0], [60.0], [5.0]]), np.array([[10.0], [10.0], [0.0]])
    before = base.copy()
    assert get_max_rdr_zeroed_baseline(counts, base, max_rdr=5.0)[:, 0].tolist() == [10.0, 0.0, 0.0]
    assert np.array_equal(base, before)


def test_the_normal_clone_and_neutral_state_are_read_off_the_balanced_baf() -> None:
    """Three clones over four states: clone 1, at p = 0.48 and 0.5, strays least outside the 0.05 deadband and is the
    normal clone; its most used balanced state, 3 (three bins to state 2's one), is the neutral state, though state 0 is
    more used overall. Without
    `p_binom`, the most used state of every clone. Fails where the deadband, the clone or the usage is misread."""
    p_binom = np.array([0.1, 0.3, 0.48, 0.5])
    paths = np.array([[0, 2, 1], [0, 3, 1], [0, 3, 0], [1, 3, 0]])
    assert get_normal_clone(get_path_baf_profiles(paths, p_binom), 0.05) == 1
    assert get_neutral_state(paths, p_binom, eps_baf=0.05) == 3
    assert get_neutral_state(paths) == 0
