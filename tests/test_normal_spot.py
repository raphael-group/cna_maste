"""`normal_spot`'s three `run_cnamaste` stages: the normal candidates, the BAF bin filter and the normal baseline.

Referees: the planted truth of seeded simulations for `determine_normal_candidates` and `normal_baf_bin_filter`;
`scipy.stats.betabinom.ppf`, CalicoST `c1abcae`'s `bin_selection_basedon_normal` cut, for the filter's band; and the
baseline's formula written out in NumPy for `determine_normal_baseline`. Config: `zenodo_sim_config.yaml`.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
import scipy.sparse
import scipy.stats

from cnamaste.config import YAMLConfig, get_global_config, set_global_config
from cnamaste.normal_spot import determine_normal_baseline, determine_normal_candidates, normal_baf_bin_filter
from cnamaste.qa.sims import get_project
from labelled_counts import get_labelled_counts


@pytest.fixture(scope="module")
def config() -> Iterator[YAMLConfig]:
    """`zenodo_sim_config.yaml` as the global config, the previous one restored after."""
    previous = get_global_config()
    shipped = YAMLConfig.from_file(get_project() / "zenodo_sim_config.yaml")
    set_global_config(shipped)
    yield shipped
    set_global_config(previous)


# -- determine_normal_candidates ---------------------------------------------------------------------------------

N_BINS = 120
CLONE_SIZES = (60, 75, 65)
"""BAF clones 0, 1, 2; clone 1, the near-normal one, holds all 60 planted normal spots and 15 tumour spots."""
N_NORMAL = 60


def get_candidate_problem(seed: int) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """Seeded read depth for 200 spots over 120 bins, 2,000 UMIs per spot; tumour spots carry a copy-number 1 run
    (bins 0-39) and a 3 (bins 40-79). Clone 1's model BAF strays 0.1 from 1/2 on 10 bins, clones 0 and 2 0.3 and
    0.17 on 40.

    Returns: the BAF-only `res`, `baf_profiles` (3, 120), `single_X_rdr` (120, 200) and the planted normal mask.
    """
    rng = np.random.default_rng(seed)
    assignment = np.repeat(np.arange(3), CLONE_SIZES)
    normal = np.zeros(assignment.size, dtype=bool)
    normal[CLONE_SIZES[0] : CLONE_SIZES[0] + N_NORMAL] = True
    copy_number = np.full(N_BINS, 2.0)
    copy_number[:40], copy_number[40:80] = 1.0, 3.0
    weight = rng.gamma(50.0, 1.0 / 50.0, N_BINS)
    rate = np.where(normal, 2.0, copy_number[:, None]) * weight[:, None]
    single_X_rdr = rng.poisson(2000.0 * rate / rate.sum(axis=0)).astype(float)
    baf_profiles = np.full((3, N_BINS), 0.5)
    baf_profiles[0, :40], baf_profiles[1, 100:110], baf_profiles[2, 40:80] = 0.2, 0.4, 0.33
    return {"new_assignment": assignment}, baf_profiles, single_X_rdr, normal


def find_candidates(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`determine_normal_candidates` on `get_candidate_problem(seed)`, no input files, as `run_cnamaste` calls it.

    Returns: the candidates, the planted normal mask and each spot's std of `log1p` read depth.
    """
    res, baf_profiles, single_X_rdr, normal = get_candidate_problem(seed)
    files = SimpleNamespace(
        preprocessing=SimpleNamespace(normalidx_file=None), quality=SimpleNamespace(baf=SimpleNamespace(eps=0.05))
    )
    candidate = determine_normal_candidates(files, res, baf_profiles, single_X_rdr.T)
    return candidate, normal, np.std(np.log1p(single_X_rdr), axis=0)


@pytest.mark.parametrize("seed", [0, 1])
def test_candidates_are_the_near_normal_clone_but_its_noisiest_spot(seed: int) -> None:
    """The clone least off 1/2 in BAF is clone 1, not the first; every candidate is in it, recall 60/60 on both seeds
    (floor 0.98), precision 60/74 (floor 0.79): the 15 planted tumour spots in clone 1 are the stated impurity.
    The search runs to 100% (T- #28): every clone-1 spot but the one of highest std. A given `normalidx_file` returns
    None. Fails where the clone is misread, a spot outside it is drawn, or the percentile search changes."""
    candidate, normal, stds = find_candidates(seed)
    clone = np.repeat(np.arange(3), CLONE_SIZES) == 1
    assert not candidate[~clone].any()
    assert np.count_nonzero(candidate) == clone.sum() - 1 and not candidate[np.flatnonzero(clone)[stds[clone].argmax()]]
    assert np.count_nonzero(candidate & normal) / N_NORMAL >= 0.98
    assert np.count_nonzero(candidate & normal) / np.count_nonzero(candidate) >= 0.79

    files = SimpleNamespace(preprocessing=SimpleNamespace(normalidx_file="normal.tsv"))
    res, baf_profiles, single_X_rdr, _ = get_candidate_problem(seed)
    assert determine_normal_candidates(files, res, baf_profiles, single_X_rdr) is None


@pytest.mark.xfail(strict=True, reason="T- #28: the percentile search's stop, stdthreshold > 1.5 * inf, never holds")
def test_candidates_exclude_the_clone_s_tumour_spots() -> None:
    """Clone 1's tumour spots carry two CNA runs, so their `log1p` std exceeds every normal spot's: a search that
    stops at 40%, as CalicoST's does here (30 spots, 60,000 UMIs > 200 per bin), draws normal spots alone. Precision
    floor 0.95. Fails, as it stands, because the search always reaches 100% (precision 0.81)."""
    candidate, normal, stds = find_candidates(seed=0)
    clone = np.repeat(np.arange(3), CLONE_SIZES) == 1
    assert stds[clone & normal].max() < stds[clone & ~normal].min()
    assert np.count_nonzero(candidate & normal) / np.count_nonzero(candidate) >= 0.95


# -- normal_baf_bin_filter ---------------------------------------------------------------------------------------

N_FILTER_BINS, N_FILTER_SPOTS = 80, 50
NORMAL_SPOTS = np.arange(40)
"""The normal spots among 50; spots 40-49 are tumour spots, at BAF 0.1 on every bin, and must not enter the fit."""


def get_filter_problem(seed: int, tau: float | None, off_balance: int | None) -> dict[str, Any]:
    """Seeded allele counts on 80 bins over 2 contigs (50 and 30 bins), a gene row per bin.

    Each bin's normal-spot BAF is drawn from Beta(tau/2, tau/2), or is exactly 1/2 with `tau` None, so the pooled
    B count is BetaBinom(n, tau/2, tau/2); bin `off_balance` sits at 0.15.

    Returns: the `normal_baf_bin_filter` inputs and the pooled normal B and total counts.
    """
    rng = np.random.default_rng(seed)
    total = rng.poisson(10.0, (N_FILTER_BINS, N_FILTER_SPOTS)).astype(float)
    p = np.full(N_FILTER_BINS, 0.5) if tau is None else rng.beta(tau / 2, tau / 2, N_FILTER_BINS)
    if off_balance is not None:
        p[off_balance] = 0.15
    b_allele = rng.binomial(total.astype(int), p[:, None]).astype(float)
    if tau is None:
        b_allele[:, NORMAL_SPOTS] = np.floor(total[:, NORMAL_SPOTS] / 2)
        b_allele[off_balance, NORMAL_SPOTS] = rng.binomial(total[off_balance, NORMAL_SPOTS].astype(int), 0.15)
    b_allele[:, 40:] = rng.binomial(total[:, 40:].astype(int), 0.1)
    single_X = np.stack([rng.poisson(50.0, total.shape).astype(float), b_allele], axis=1)
    df = pd.DataFrame({"CHR": np.repeat([1, 2], [50, 30]), "bin_id": np.arange(N_FILTER_BINS)})
    return {
        "df_gene_snp": df,
        "single_X": single_X,
        "single_total_bb_RD": total,
        "pooled": (b_allele[:, NORMAL_SPOTS].sum(axis=1), total[:, NORMAL_SPOTS].sum(axis=1)),
    }


def filter_bins(problem: dict[str, Any], confidence_interval: tuple[float, float] | None = None) -> Any:
    """`normal_baf_bin_filter` as `run_cnamaste` calls it, `confidence_interval` None reading the config's."""
    return normal_baf_bin_filter(
        problem["df_gene_snp"].copy(),
        get_labelled_counts(
            problem["single_X"][:, 0], problem["single_X"][:, 1], problem["single_total_bb_RD"], [50, 30]
        ),
        NORMAL_SPOTS,
        confidence_interval=confidence_interval,
    )


def test_an_off_balance_bin_is_removed_and_balanced_bins_kept(config: YAMLConfig) -> None:
    """Normal-spot BAF exactly 1/2 on 79 bins and 0.15 on bin 57 (contig 2), at the config's (0.01, 0.99): bin 57
    alone is removed, the rest renumbered 0-78 in order, its contig's length 29, and each kept row the input's.
    Fails where a balanced bin is cut, the off-balance bin kept, the tumour spots pooled, or rows misaligned."""
    problem = get_filter_problem(seed=0, tau=None, off_balance=57)
    df, counts = filter_bins(problem)
    kept = np.delete(np.arange(N_FILTER_BINS), 57)
    assert df.bin_id.isna().tolist() == (np.arange(N_FILTER_BINS) == 57).tolist()
    assert df.bin_id.dropna().tolist() == list(range(N_FILTER_BINS - 1))
    assert counts.lengths.tolist() == [50, 29]
    np.testing.assert_array_equal(counts.umis.toarray().T, problem["single_X"][kept, 0])
    np.testing.assert_array_equal(counts.alt_snp_umis.toarray().T, problem["single_X"][kept, 1])
    np.testing.assert_array_equal(counts.total_snp_umis.toarray().T, problem["single_total_bb_RD"][kept])


@pytest.mark.parametrize("confidence_interval", [None, (0.1, 0.9), (0.25, 0.75)])
def test_the_band_is_the_betabinomial_ppf_at_the_tau_floor(
    config: YAMLConfig, confidence_interval: tuple[float, float] | None
) -> None:
    """Normal-spot BAF drawn at tau = 10, below `min_betabinom_tau` 30, so the fit is patched to p = 1/2, tau = 30: a
    bin is kept iff its pooled B count lies in [ppf(lo), ppf(hi)] of BetaBinom(n, 15, 15), `scipy.stats`
    directly, CalicoST's cut. Exact set equality, at the config's (0.01, 0.99) and two narrower bands, each removing
    at least one bin. Fails where the cdf form departs from the ppf one at a boundary, the floor or p patch moves,
    or `quality.normal_allele_specific_confidence` is not read."""
    problem = get_filter_problem(seed=1, tau=10.0, off_balance=None)
    df, _ = filter_bins(problem, confidence_interval)
    lo, hi = confidence_interval or ast.literal_eval(config.quality.normal_allele_specific_confidence)
    b_allele, total = problem["pooled"]
    lower, upper = (scipy.stats.betabinom.ppf(q, total, 15.0, 15.0) for q in (lo, hi))
    expected = (b_allele >= lower) & (b_allele <= upper)
    assert df.bin_id.notna().tolist() == expected.tolist()
    assert not expected.all()


# -- determine_normal_baseline -----------------------------------------------------------------------------------


def get_baseline_problem(seed: int, threshold: int) -> tuple[np.ndarray, np.ndarray]:
    """Seeded int64 read depth, as `summarize_counts_for_bins` returns it, 30 bins by 40 spots, 20 normal; bins 3 and
    17 sum to `threshold - 1` over the normal spots, bin 9 to `threshold` exactly."""
    rng = np.random.default_rng(seed)
    single_X_rdr = rng.poisson(20.0, (30, 40))
    normal = np.arange(40) < 20
    for b, target in ((3, threshold - 1), (17, threshold - 1), (9, threshold)):
        single_X_rdr[b, normal] = rng.multinomial(target, np.full(20, 1 / 20))
    return single_X_rdr, normal


@pytest.mark.parametrize("backing", ["dense", "csr"])
def test_the_baseline_is_the_normal_profile_times_each_spot_s_depth(config: YAMLConfig, backing: str) -> None:
    """At `quality.min_normal_count_perbin` 100: bins 3 and 17 (normal sum 99) are zeroed in the profile and in the
    read depth, bin 9 (100) kept; the profile is the normal spots' per-bin sums over their total, the baseline its
    outer product with each spot's depth after zeroing. Against NumPy written out, to 1e-12 relative; the input is
    not written. Fails where the gate's inequality, the normalization, the zeroing or the depth's order moves."""
    threshold = config.quality.min_normal_count_perbin
    single_X_rdr, normal = get_baseline_problem(seed=0, threshold=threshold)
    given = single_X_rdr if backing == "dense" else scipy.sparse.csr_matrix(single_X_rdr)
    counts = determine_normal_baseline(get_labelled_counts(given, given, given), normal, config)

    profile = single_X_rdr[:, normal].sum(axis=1)
    gated = profile < threshold
    assert np.flatnonzero(gated).tolist() == [3, 17]
    profile = np.where(gated, 0.0, profile) / profile[~gated].sum()
    depth = np.where(gated[:, None], 0.0, single_X_rdr)
    np.testing.assert_array_equal(counts.umis.toarray().T, depth)
    np.testing.assert_array_equal(counts.labellings["normal_spot"], normal)
    base_nb_mean = counts.normal_baseline_sized.toarray().T
    np.testing.assert_allclose(base_nb_mean, np.outer(profile, depth.sum(axis=0)), rtol=1e-12, atol=0.0)
    after = given if backing == "dense" else given.toarray()
    assert after[3, normal].sum() == threshold - 1
