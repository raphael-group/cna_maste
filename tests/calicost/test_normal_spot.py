"""`normal_spot` against CalicoST `c1abcae`'s normal-spot steps: `utils_IO.filter_de_genes_tri` and
`bin_selection_basedon_normal`, and `calicost_main.py`'s inline normal candidates and RDR baseline.

The referee is CalicoST's own output on the same input, computed once and recorded (`calicost_cache`); for the inline
steps, a verbatim copy of `main()`'s lines, cited where copied. Inputs: for the gene filter, two seeded slices of 120
spots by 60 genes, 40 normal candidates, 20 normal-looking spots outside them (CalicoST's `unsure`) and 60 tumour
spots, with six genes planted as differentially expressed; for the rest, 3 clones of 20 spots by 60 bins, clone 1
normal-like, seeded. No capture is read; the file runs in 1.5 s.

Tolerance: none on sets, labels, bins and counts; 1e-12 relative on the baseline, a product of sums; 1e-3 on the
normal spots' fitted tau.

Departures: each reads its own pipeline's bin table, `INCLUDED_GENES` joined by "," in cnamaste
(`omics.binned_gene_snp`), by " " in CalicoST, stated in `test_omics`; cnamaste takes the whole normal-like clone as
normal candidates, a strict xfail.
"""

from __future__ import annotations

import functools
import gzip
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from calicost.utils_distribution_fitting import Weighted_BetaBinom
from calicost.utils_IO import bin_selection_basedon_normal

from calicost_cache import Referee, departs, get_digest
from cnamaste.calicost import filter_diffexp
from cnamaste.config import YAMLConfig, set_global_config
from cnamaste.normal_spot import (
    determine_normal_baseline,
    determine_normal_candidates,
    filter_normal_diffexp,
    fit_normal_spot_bb,
    normal_baf_bin_filter,
)
from labelled_counts import get_labelled_counts

N_SPOTS, N_GENES = 120, 60
GENES = np.array([f"g{i}" for i in range(N_GENES)])
NORMAL, UNSURE, TUMOR = slice(0, 40), slice(40, 60), slice(60, N_SPOTS)
PLANTED = {0: ([0, 1], [3]), 1: ([2], [4, 5])}
"""Per slice, the genes depleted 100-fold in the tumour spots (|log2 fc| 6.6 > 4) and enriched 8-fold in the unsure
spots (log2 fc 3 > 2), all at 30 UMI per spot, above the 80th percentile of gene totals."""


def get_planted_counts(seed: int) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Two slices of Poisson counts with `PLANTED`'s genes; tumour spots carry 3 copies of genes 10-29 to cluster.

    Returns: `exp_counts` (spots, genes), the normal-candidate mask and each spot's slice.
    """
    rng = np.random.default_rng(seed)
    rate = rng.gamma(2.0, 3.0, N_GENES)
    rate[:6] = 30.0
    slices = []
    for depleted, enriched in PLANTED.values():
        fold = np.ones((N_SPOTS, N_GENES))
        fold[TUMOR, depleted] = 0.01
        fold[UNSURE, enriched] = 8.0
        fold[TUMOR, 10:30] *= 3.0
        slices.append(rng.poisson(rate * fold))
    normal = np.tile(np.arange(N_SPOTS) < NORMAL.stop, 2)
    return pd.DataFrame(np.vstack(slices), columns=GENES), normal, np.repeat([0, 1], N_SPOTS)


def get_bin_table(sep: str) -> pd.DataFrame:
    """Ten bins of six consecutive genes, `INCLUDED_GENES` joined by `sep`."""
    return pd.DataFrame({"INCLUDED_GENES": [sep.join(GENES[i : i + 6]) for i in range(0, N_GENES, 6)]})


def test_filter_normal_diffexp_is_calicost_s(referee: Referee) -> None:
    """Two slices with six planted genes: both flag exactly the planted six, the union over slices, and return the
    same bin counts, bitwise. Fails where the gene filter, normalization, PCA and k-means labels, fold changes, UMI
    threshold or the bin sums drift from CalicoST's, or either misses a planted gene."""
    exp_counts, normal, sample_ids = get_planted_counts(seed=0)
    kwargs = {"sample_list": ["a", "b"], "sample_ids": sample_ids}

    def call() -> tuple[np.ndarray, list[str]]:
        counts, flagged = filter_diffexp(exp_counts, get_bin_table(" "), normal, **kwargs)
        return counts, sorted(flagged)

    inputs = (exp_counts, get_bin_table(" "), normal, ["a", "b"], sample_ids)
    theirs, theirs_out = referee("normal_spot.filter_diffexp", inputs, call, kept=(1,))
    ours, ours_out = filter_normal_diffexp(exp_counts, get_bin_table(","), normal, **kwargs)
    planted = {f"g{g}" for depleted, enriched in PLANTED.values() for g in depleted + enriched}
    assert ours_out == set(theirs_out) == planted
    assert get_digest(ours.toarray().T) == theirs
    retained = ~np.isin(GENES, sorted(planted))
    assert ours.toarray().sum() == exp_counts.to_numpy()[:, retained].sum()


N_OBS = 60
CLONE_SPOTS = 20
"""Spots per clone, 3 clones: clone 1 the normal-like one."""

BAF_PROFILES = np.stack([np.full(N_OBS, 0.2), np.full(N_OBS, 0.5), np.repeat([0.3, 0.8], N_OBS // 2)])
"""Each clone's decoded BAF per bin, `merged_baf_profiles` (n_clones, n_obs)."""


@pytest.fixture(scope="module")
def umis() -> np.ndarray:
    """`(n_obs, n_spots)` read depth, Poisson at a per-spot depth 10-30, seeded: 1,200 UMIs per spot on average."""
    rng = np.random.default_rng(2)
    return rng.poisson(rng.uniform(10, 30, 3 * CLONE_SPOTS), (N_OBS, 3 * CLONE_SPOTS)).astype(float)


def call_calicost_normal_candidate(copy_single_X_rdr: np.ndarray, assignment: np.ndarray) -> np.ndarray:
    """CalicoST's normal candidates, an identity `smooth_mat`."""
    smooth_mat = np.eye(copy_single_X_rdr.shape[1])
    merged_res, merged_baf_profiles, single_X = {"new_assignment": assignment}, BAF_PROFILES, copy_single_X_rdr[:, None]
    # NB copied from calicost_main.py:115-125 (c1abcae).
    EPS_BAF = 0.05
    PERCENT_NORMAL = 40
    vec_stds = np.std(np.log1p(copy_single_X_rdr @ smooth_mat), axis=0)
    id_nearnormal_clone = np.argmin(np.sum(np.maximum(np.abs(merged_baf_profiles - 0.5) - EPS_BAF, 0), axis=1))
    while True:
        stdthreshold = np.percentile(vec_stds[merged_res["new_assignment"] == id_nearnormal_clone], PERCENT_NORMAL)
        normal_candidate = (vec_stds < stdthreshold) & (merged_res["new_assignment"] == id_nearnormal_clone)
        if np.sum(copy_single_X_rdr[:, (normal_candidate == True)]) > single_X.shape[0] * 200 or PERCENT_NORMAL == 100:
            break
        PERCENT_NORMAL += 10
    return normal_candidate


@departs(
    "CalicoST stops at the first percentile whose candidates hold n_obs x 200 UMIs; cnamaste takes the whole clone"
)
def test_determine_normal_candidates_is_calicost_s(umis: np.ndarray, referee: Referee) -> None:
    """Clone 1 (BAF 0.5, 20 spots) is the normal-like clone on both sides. CalicoST takes its 10 spots under the
    50th percentile of log read-depth spread, the first to hold over 12,000 UMIs (14,721); cnamaste all but the
    widest, 19, as intended: every spot of the normal-like clone is a normal candidate."""
    assignment = np.repeat([0, 1, 2], CLONE_SPOTS)
    call = functools.partial(call_calicost_normal_candidate, umis, assignment)
    (theirs,) = referee("normal_spot.candidates", (umis, assignment, BAF_PROFILES), call, kept=(0,))
    config = SimpleNamespace(
        preprocessing=SimpleNamespace(normalidx_file=None), quality=SimpleNamespace(baf=SimpleNamespace(eps=0.05))
    )
    ours = determine_normal_candidates(config, {"new_assignment": assignment}, BAF_PROFILES, umis.T)
    assert np.array_equal(ours, theirs)


def call_calicost_baseline(copy_single_X_rdr: np.ndarray, normal_candidate: np.ndarray) -> tuple[np.ndarray, ...]:
    """CalicoST's read depth and normal baseline, `(n_obs, n_spots)`, after its spot filter."""
    copy_single_X_rdr = copy_single_X_rdr.copy()
    # NB copied from calicost_main.py:146-152 (c1abcae).
    MIN_NORMAL_COUNT_PERBIN = 20
    bidx_inconfident = np.where(
        np.sum(copy_single_X_rdr[:, (normal_candidate == True)], axis=1) < MIN_NORMAL_COUNT_PERBIN
    )[0]
    rdr_normal = np.sum(copy_single_X_rdr[:, (normal_candidate == True)], axis=1)
    rdr_normal[bidx_inconfident] = 0
    rdr_normal = rdr_normal / np.sum(rdr_normal)
    copy_single_X_rdr[bidx_inconfident, :] = 0
    copy_single_base_nb_mean = rdr_normal.reshape(-1, 1) @ np.sum(copy_single_X_rdr, axis=0).reshape(1, -1)
    return copy_single_X_rdr, copy_single_base_nb_mean


def test_the_normal_baseline_is_calicost_s(umis: np.ndarray, referee: Referee) -> None:
    """Clone 1's 20 spots as the normal ones, 6 bins with no normal UMI and one with 20, at `min_normal_count_perbin`
    20: `determine_normal_baseline` zeroes CalicoST's bins, bitwise, and `normal_baseline_sized` is CalicoST's
    baseline within 1e-12 relative. Fails where the floor, its comparison, the zeroed bins or the baseline's
    normalization drifts."""
    umis = umis.copy()
    normal = np.repeat([False, True, False], CLONE_SPOTS)
    umis[:6, normal] = 0.0
    umis[6, normal] = 1.0
    call = functools.partial(call_calicost_baseline, umis, normal)
    theirs_umis, theirs_baseline = referee("normal_spot.baseline", (umis, normal), call, kept=(0, 1))
    config = SimpleNamespace(quality=SimpleNamespace(min_normal_count_perbin=20))
    counts = determine_normal_baseline(get_labelled_counts(umis, umis, umis), normal, config)
    assert np.array_equal(counts.umis.toarray().T, theirs_umis)
    assert np.count_nonzero(~theirs_umis.any(axis=1)) == 6
    np.testing.assert_allclose(counts.normal_baseline_sized.toarray().T, theirs_baseline, rtol=1e-12, atol=0.0)


PLANTED_BINS = [7, 23, 41]
"""Bins whose normal spots carry B fraction 0.8: allele-specific expression the filter removes."""


@pytest.fixture(scope="module")
def allele_counts() -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """`(n_obs, n_spots)` B-allele and total SNP UMIs, each bin's B fraction drawn Beta(100, 100) (tau 200) but
    `PLANTED_BINS`', and `df_gene_snp`, two rows per bin on contig 1, seeded."""
    rng = np.random.default_rng(3)
    p = rng.beta(100.0, 100.0, N_OBS)
    p[PLANTED_BINS] = 0.8
    total = rng.integers(2, 8, (N_OBS, 3 * CLONE_SPOTS)).astype(float)
    alt = rng.binomial(total.astype(int), p[:, None]).astype(float)
    start = 1_000_000 + 100_000 * np.arange(2 * N_OBS)
    table = pd.DataFrame({"CHR": 1, "START": start, "END": start + 50_000, "bin_id": np.repeat(np.arange(N_OBS), 2)})
    return alt, total, table


def call_calicost_bin_selection(
    alt: np.ndarray, total: np.ndarray, table: pd.DataFrame, index_normal: np.ndarray, geneticmap_file: str
) -> tuple[np.ndarray, np.ndarray, float]:
    """CalicoST's `bin_selection_basedon_normal` at (0.05, 0.95), and its normal spots' beta-binomial tau.

    Returns: the remapped `bin_id`, -1 where removed; the kept bins' B-allele UMIs; the fitted tau.
    """
    single_X = np.stack([np.zeros_like(alt), alt], axis=1)
    found = bin_selection_basedon_normal(
        table.copy(), single_X, np.zeros_like(alt), total, 1.0, 1.0, index_normal, geneticmap_file
    )
    single_total_bb_RD = total
    # NB copied from utils_IO.py:975-977 (c1abcae), the fit `bin_selection_basedon_normal` does not return.
    tmpX = np.sum(single_X[:, 1, index_normal], axis=1)
    tmptotal_bb_RD = np.sum(single_total_bb_RD[:, index_normal], axis=1)
    model = Weighted_BetaBinom(tmpX, np.ones(len(tmpX)), weights=np.ones(len(tmpX)), exposure=tmptotal_bb_RD)
    tmpres = model.fit(disp=0)
    return found[5].bin_id.fillna(-1).to_numpy(dtype=int), found[1][:, 1, :], float(tmpres.params[-1])


def test_normal_baf_bin_filter_is_bin_selection_basedon_normal(
    allele_counts: tuple[np.ndarray, np.ndarray, pd.DataFrame], tmp_path: Path, referee: Referee
) -> None:
    """Clone 1's 20 spots as the normal ones, at CalicoST's (0.05, 0.95): `normal_baf_bin_filter` removes CalicoST's
    4 bins, the 3 planted among them, and remaps `bin_id` and keeps the B-allele UMIs as CalicoST, bitwise; the normal
    spots' fitted tau, over the floor of 30, is CalicoST's within 1e-3 relative (measured 9.7e-4 at 42.2: cnamaste's
    L-BFGS-B stops at `hmm.em_ftol` 1e-4). Fails where the pooled counts, the beta-binomial fit, the band's cut or
    the remap drifts."""
    alt, total, table = allele_counts
    index_normal = np.flatnonzero(np.repeat([False, True, False], CLONE_SPOTS))
    geneticmap = tmp_path / "map.tab.gz"
    with gzip.open(geneticmap, "wt") as handle:
        handle.write("chrom\tpos\trecomb_rate\tpos_cm\nchr1\t100\t0\t0.0\nchr1\t20000000\t0\t20.0\n")
    call = functools.partial(call_calicost_bin_selection, alt, total, table, index_normal, str(geneticmap))
    inputs = (alt, total, table, index_normal)
    bin_id, kept_alt, tau = referee("normal_spot.bin_selection", inputs, call, kept=(0, 1, 2))
    counts = get_labelled_counts(np.zeros_like(alt), alt, total)
    # NB the fit reads `hmm.em_maxiter` and `hmm.em_ftol`, the defaults.
    set_global_config(YAMLConfig({}))
    ours_table, kept = normal_baf_bin_filter(table.copy(), counts, index_normal, (0.05, 0.95), 30)
    assert np.array_equal(ours_table.bin_id.fillna(-1).to_numpy(dtype=int), bin_id)
    assert np.isin(PLANTED_BINS, table.bin_id[bin_id == -1]).all()
    assert np.array_equal(kept.alt_snp_umis.toarray().T, kept_alt)
    ours_tau = fit_normal_spot_bb(alt[:, index_normal].sum(axis=1), total[:, index_normal].sum(axis=1))[-1]
    assert min(ours_tau, tau) > 30
    np.testing.assert_allclose(ours_tau, tau, rtol=1e-3, atol=0.0)
