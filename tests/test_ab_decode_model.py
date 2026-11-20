"""`ab_decode.model`' helpers, each against a case written out by hand: the contig start prior, the chain, the copy path's emission and path loglikes (`eqn:ab_hmm`), the agreement priors
(`eqn:ab_coupling`), the ploidy band, the first `log Z_c` and the merged integer clones. Below 0.5 s.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.special
from scipy import stats

from cnamaste.ab_decode.lattice import get_ab_is_ploidy_neutral, get_ab_lattice, get_lattice_parameters
from cnamaste.ab_decode.model import (
    get_all_clone_agreement_prior,
    get_clone_agreement_prior,
    get_clone_chain,
    get_contig_start_prior,
    get_copy_path_emission_loglike,
    get_copy_path_loglike,
    get_log_mu_shift_start,
    get_merged_ab_decoded_clones,
    get_ploidy_band,
    in_ploidy_band,
)
from cnamaste.config import YAMLConfig
from cnamaste.emission import CnaEmissionConfounders
from cnamaste.spatio_genomic_counts import PseudobulkCounts

CAPS = {"max_total_copy": 4, "max_allele_copy": 3}


def as_set(pairs: np.ndarray) -> set[tuple[int, int]]:
    """The pairs as a set of tuples."""
    return {(int(a), int(b)) for a, b in pairs}


@pytest.mark.parametrize("start_prior_type", ["anchored", "uniform"])
def test_each_contig_start_prior_is_normalized_and_anchored_favours_the_neutral_pair(start_prior_type: str) -> None:
    """Each prior sums to 1; `anchored` gives the neutral pair `stay_prob` and every other `(1 - stay_prob) / (K - 1)`;
    an unknown type raises `ValueError`. Fails where the start is not a distribution or favours another pair."""
    pairs = get_ab_lattice(params="mp", fold=False, **CAPS)
    prior = get_contig_start_prior(pairs, 0.9, 2, start_prior_type)
    assert np.isclose(scipy.special.logsumexp(prior), 0.0, atol=1e-12)
    if start_prior_type == "anchored":
        neutral = get_ab_is_ploidy_neutral(pairs, 2)
        np.testing.assert_allclose(prior[neutral], np.log(0.9), rtol=1e-15)
        np.testing.assert_allclose(prior[~neutral], np.log(0.1 / (len(pairs) - 1)), rtol=1e-15)
    with pytest.raises(ValueError):
        get_contig_start_prior(pairs, 0.9, 2, "free")


def test_the_chain_is_the_stay_probability() -> None:
    """`log_stay = log t`, `log_switch = log((1 - t) / (K - 1))`, `K` the lattice's pairs, the start the anchored
    prior, at the configuration's `ab_decode.t`. Fails where a setting is read from elsewhere."""
    config = YAMLConfig({"ab_decode": {"t": 0.95, "clone_coupling": 0.3}})
    pairs = get_ab_lattice(params="mp", fold=False, **CAPS)
    chain = get_clone_chain(pairs, np.array([3, 2]), config, ploidy=2)
    assert chain.K == len(pairs) and chain.clone_coupling == 0.3
    assert chain.log_stay == np.log(0.95) and chain.log_switch == np.log((1.0 - 0.95) / (len(pairs) - 1))
    np.testing.assert_array_equal(chain.log_start, get_contig_start_prior(pairs, 0.95, 2, "anchored"))


@pytest.mark.parametrize("theta", [None, 0.3])
def test_the_copy_path_emission_loglike_is_scipy_s_nb_plus_beta_binomial(theta: float | None) -> None:
    """On a 4-bin copy path at `log Z_c` 0.1: `scipy.stats.nbinom` at `base mu~ / Z_c` plus `betabinom` at `p~`,
    `eqn:ab_admixture` written out, to 1e-12. Fails where the admixture, `Z_c` or a channel departs."""
    pairs = np.array([[1, 1], [2, 1], [1, 0], [0, 2]])
    copy_path = np.array([0, 1, 2, 3])
    y, base = np.array([[90.0], [130.0], [55.0], [80.0]]), np.array([[100.0], [100.0], [100.0], [90.0]])
    b, n = np.array([[20.0], [12.0], [3.0], [30.0]]), np.array([[40.0], [40.0], [40.0], [35.0]])
    alpha, tau, log_z = 0.05, 50.0, 0.1
    found = get_copy_path_emission_loglike(
        PseudobulkCounts(y.T, base.T, b.T, n.T, None, {}),
        get_lattice_parameters(pairs, alpha, tau),
        CnaEmissionConfounders(np.full((1, 1), 0.0 if theta is None else theta), np.full((1, 1), log_z)),
        copy_path,
        fold=False,
        data_terms=True,
    )
    f = 0.0 if theta is None else theta
    total = pairs[copy_path].sum(axis=1)
    depth = f + (1 - f) * total / 2.0
    fraction = (f / 2.0 + (1 - f) * pairs[copy_path, 1] / 2.0) / depth
    mean, r = base[:, 0] * depth / np.exp(log_z), 1.0 / alpha
    expected = stats.nbinom.logpmf(y[:, 0], r, r / (r + mean)).sum()
    # NB the kernels floor each shape at 1e-10, as at the LOH bin, p~ = 0 at theta 0.
    shapes = np.maximum(fraction * tau, 1e-10), np.maximum((1 - fraction) * tau, 1e-10)
    expected += stats.betabinom.logpmf(b[:, 0], n[:, 0], *shapes).sum()
    np.testing.assert_allclose(found, expected, rtol=1e-12)


def test_the_copy_path_loglike_is_emission_start_and_transitions() -> None:
    """3 bins, segments of 2 and 1: the emission along the path, the start at bins 0 and 2, one switch. Fails where a
    step crosses the segment boundary or the start is charged once."""
    pairs = np.array([[1, 1], [2, 1], [1, 0]])
    chain = get_clone_chain(pairs, np.array([2, 1]), YAMLConfig({"ab_decode": {"t": 0.8}}), ploidy=2)
    emission = np.arange(9.0).reshape(3, 3) / -10.0
    copy_path = np.array([0, 1, 1])
    expected = emission[0, 0] + emission[1, 1] + emission[1, 2] + chain.log_start[0] + chain.log_start[1]
    expected += chain.log_switch
    assert get_copy_path_loglike(emission, chain, copy_path) == pytest.approx(expected, rel=1e-15)


def test_the_agreement_priors_charge_each_disagreeing_pair_once() -> None:
    """3 clones x 4 bins: the per-clone prior summed over each clone's own path is twice the all-clone prior, which
    is `-clone_coupling` per disagreeing pair and bin; 0 at coupling 0. Fails where a pair is charged per clone
    rather than per pair, or with the wrong sign."""
    paths = {0: np.array([0, 1, 1, 2]), 1: np.array([0, 1, 2, 2]), 2: np.array([1, 1, 2, 0])}
    coupling = 0.4
    own = sum(get_clone_agreement_prior(paths, c, 3, 4, coupling)[paths[c], np.arange(4)].sum() for c in paths)
    disagree = sum(int((paths[c] != paths[d]).sum()) for c in paths for d in paths if c < d)
    assert get_all_clone_agreement_prior(paths, coupling) == pytest.approx(-coupling * disagree, rel=1e-15)
    assert own == pytest.approx(2 * get_all_clone_agreement_prior(paths, coupling), rel=1e-15)
    assert get_all_clone_agreement_prior(paths, 0.0) == 0.0 and not get_clone_agreement_prior(paths, 0, 3, 4, 0.0).any()


def test_the_ploidy_band_holds_its_edges() -> None:
    """A mean `A + B` of exactly P +/- `PLOIDY_BAND` is in the band; 0.05 further is not."""
    totals = np.array([1, 2, 3, 4])
    for mean in (2.0 - get_ploidy_band(), 2.0 + get_ploidy_band()):
        path = np.array([1, 1, 2, 2] if mean > 2 else [0, 0, 1, 1])
        assert np.mean(totals[path]) == mean and in_ploidy_band(path, totals, 2)
    assert np.mean(totals[[1] * 9 + [2] * 11]) == 2.55
    assert not in_ploidy_band(np.array([1] * 9 + [2] * 11), totals, 2)


def test_the_first_log_z_carries_the_fit_s_to_the_ploidy_or_takes_the_ploidy_s_own() -> None:
    """At diploid the fit's, 0 where none; at P = 4 the fit's plus `log(k / (1 + (k - 1) theta))`, or
    `log(theta + (1 - theta) k)` where the fit carries none, `k = 2`."""
    assert get_log_mu_shift_start(0.3, 0.2, 2) == 0.3 and get_log_mu_shift_start(None, 0.2, 2) == 0.0
    assert get_log_mu_shift_start(0.3, 0.2, 4) == pytest.approx(0.3 + np.log(2.0 / 1.2), rel=1e-15)
    assert get_log_mu_shift_start(None, 0.2, 4) == pytest.approx(np.log(0.2 + 0.8 * 2.0), rel=1e-15)


def get_planted_seglevel() -> pd.DataFrame:
    """Five clones on 20 bins, as `cnv_<ploidy>_seglevel.tsv` holds them: 1 departs from 0 on 10 bins, 2 agrees with
    1 on 15, 3 is 0 and 4 is 2."""
    profiles = np.ones((5, 20, 2), dtype=np.int64)
    profiles[1, :10] = (2, 1)
    profiles[2] = profiles[1]
    profiles[2, 15:] = (0, 1)
    profiles[3], profiles[4] = profiles[0], profiles[2]
    return pd.DataFrame({f"clone{c} {a}": profiles[c, :, i] for c in range(5) for i, a in enumerate("AB")})


@pytest.mark.parametrize(("agreement", "expected"), [(0.75, [0, 1, 1, 0, 1]), (0.76, [0, 1, 2, 0, 2])])
def test_clones_at_or_above_the_agreement_share_the_first_group_they_agree_with(
    agreement: float, expected: list[int]
) -> None:
    """At 15/20 = 0.75 clone 2 joins clone 1's group, and clone 4 with it; just above, clone 2 names its own and
    clone 4 joins it. Fails where the bound is strict or a clone is compared with a member, not the named clone."""
    assert get_merged_ab_decoded_clones(get_planted_seglevel(), agreement).tolist() == expected


@pytest.mark.xfail(strict=True, raises=KeyError, reason="T- #33: a clone left out of seglevel")
def test_a_clone_left_out_of_seglevel_leaves_the_others_grouped() -> None:
    """`results.get_integer_copy_tables` leaves out a clone with no transcripts: `clone0`, `clone2` remain, one
    profile, whose groups should be one. Fails today: the merge counts 2 clones and asks for `clone1`."""
    seglevel = get_planted_seglevel()
    seglevel = seglevel[["clone0 A", "clone0 B", "clone3 A", "clone3 B"]].set_axis(
        ["clone0 A", "clone0 B", "clone2 A", "clone2 B"], axis=1
    )
    groups = get_merged_ab_decoded_clones(seglevel, 0.99)
    assert groups[0] == groups[2]
