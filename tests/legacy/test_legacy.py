"""Each frozen copy in `tests/legacy/` against the code that replaced it, byte for byte on CalicoST easy (`2d4ce9a9`),
or within a stated tolerance where the replacement sums in another order: both run on one recorded call's arguments
from the capture, the legacy copy on the channels as it took them, and their returns, each count container as its
arrays, encode (`qa.capture.get_replay_file`) to the same attributes and bytes.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import h5py
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import scipy.special
import yaml

from cnamaste import annotation, emission, hmm_mstep, normal_spot, omics, results
from cnamaste.ab_decode.decode import get_clone_chains_potts_problem, get_clone_emission_loglikes, get_clone_viterbi
from cnamaste.ab_decode.lattice import get_ab_is_ploidy_neutral, get_ab_lattice, get_lattice_parameters
from cnamaste.ab_decode.model import get_clone_chain, get_coded_rows
from cnamaste.config import YAMLConfig, get_global_config
from cnamaste.emission import (
    CnaEmissionConfounders,
    get_decoded_emission,
    get_hard_state_posteriors,
    get_log_emission,
    get_log_mu_shift,
    get_state_major,
    get_unconfounded,
)
from cnamaste.hmm.forward_backward import (
    get_contig_offsets,
    get_dense_encoded_emission,
    get_llf,
    get_log_alpha,
    get_log_beta,
    get_log_gamma,
    get_phase_steps,
)
from cnamaste.hmm_emission_tabled import count_table_size, get_count_values, get_labelling_unary_counts
from cnamaste.hmm_nophasing import get_clone_columns, get_fit_confounders, get_max_rdr_zeroed_baseline, hmm_nophasing
from cnamaste.hmm_params import CnaHmmParameters
from cnamaste.hmm_utils import get_log_transmat
from cnamaste.hmrf import get_labelling_unary, get_merged_fit, get_nested_copy, get_reindexed_fit
from cnamaste.hmrf_utils import get_clone_assignment
from cnamaste.initialize.genomic import get_initial_parameters
from cnamaste.initialize.spatial import get_initial_labelling
from cnamaste.io import get_labelled_counts
from cnamaste.labelling import get_neutral_pinned
from cnamaste.opengm import opengm_session
from cnamaste.qa.capture import Capture, get_overrides, get_replay_file, get_template
from cnamaste.qa.sims import get_fixtures, get_project, get_supported_sims
from cnamaste.segments import Segmentation
from cnamaste.spatio_genomic_counts import (
    PseudobulkCounts,
    SpatioGenomicCounts,
    SpatioGenomicLabelledMultiCounts,
    get_encoding_channels,
    get_segmented,
)
from legacy import (
    ab_decode as legacy_ab_decode,
    annotation as legacy_annotation,
    emission as legacy_emission,
    hmm as legacy_hmm,
    hmm_emission_tabled as legacy_tabled,
    hmm_initialize as legacy_hmm_initialize,
    hmm_mstep_grad as legacy_mstep,
    hmrf as legacy_hmrf,
    normal_spot as legacy_normal_spot,
    omics as legacy_omics,
    pseudobulk as legacy_pseudobulk,
    spatial as legacy_spatial,
    spatio_genomic_counts as legacy_sgc,
    utils as legacy_utils,
)
from legacy.count_encoder import CountEncoder
from legacy.spatio_genomic_counts import BinnedCounts, SpatioGenomicPairCounts

EASY = "2d4ce9a9"

BASELINE = "07_rebin/determine_normal_baseline"

DECODE = "09_outputs/decode_integer_copies"
"""The `(A, B)` decode's call: its pseudobulk and confounders are the decoder's inputs."""

ENCODING = "03_phasing/PseudobulkCounts.update_encoding"
"""The first recorded encoding: its pseudobulk is the legacy `CountEncoder`'s input."""
"""The call whose return holds the labelled counts the RDR stage starts from."""


def assert_byte_equal(expected: h5py.Group | h5py.Dataset, found: h5py.Group | h5py.Dataset) -> None:
    """Assert two encoded values have the same attributes, children and bytes.

    Args:
        expected: the legacy return's node.
        found: the new return's node.

    Raises:
        AssertionError: naming the first node that differs.
    """
    assert dict(expected.attrs) == dict(found.attrs), expected.name
    if isinstance(expected, h5py.Group):
        assert sorted(expected) == sorted(found), expected.name
        for name in expected:
            assert_byte_equal(expected[name], found[name])
        return
    assert expected.dtype == found.dtype and expected.shape == found.shape, expected.name
    assert np.asarray(expected[()]).tobytes() == np.asarray(found[()]).tobytes(), expected.name


def get_arrays(value: Any) -> Any:
    """A return with each count container as its arrays: `(lengths, umis, alt_snp_umis, baseline, total_snp_umis)`.

    Args:
        value: a return, or part of one.

    Returns:
        The same value, its containers and `SpatioGenomicCounts` as arrays, dense or CSR.
    """
    if isinstance(value, SpatioGenomicLabelledMultiCounts):
        channels = (value.umis, value.alt_snp_umis, value.normal_baseline_sized, value.total_snp_umis)
        return (value.lengths, *(c.get_bin_major() for c in channels))
    if isinstance(value, BinnedCounts):
        lengths, X, base_nb_mean, total_bb_RD = value
        return (lengths, X.rdr.values, X.baf.values, base_nb_mean.values, total_bb_RD.values)
    if isinstance(value, SpatioGenomicCounts):
        return value.get_bin_major()
    if isinstance(value, legacy_sgc.SpatioGenomicCounts):
        return value.values
    if isinstance(value, tuple):
        return tuple(get_arrays(v) for v in value)
    return value


def get_legacy_counts(counts: SpatioGenomicCounts) -> Any:
    """The counts as the legacy copies took them, bin-major (`legacy.spatio_genomic_counts.SpatioGenomicCounts`)."""
    return legacy_sgc.SpatioGenomicCounts(counts.get_bin_major())


def get_legacy_channels(counts: SpatioGenomicLabelledMultiCounts) -> tuple[SpatioGenomicPairCounts, Any, Any]:
    """The counts as the legacy copies took them: `single_X`, `single_base_nb_mean` and `single_total_bb_RD`."""
    single_X = SpatioGenomicPairCounts(get_legacy_counts(counts.umis), get_legacy_counts(counts.alt_snp_umis))
    return single_X, get_legacy_counts(counts.normal_baseline_sized), get_legacy_counts(counts.total_snp_umis)


def get_call(call: str) -> tuple[list[Any], dict[str, Any]]:
    """The recorded call's arguments, decoded afresh (a call may mutate them)."""
    capture = Capture(get_project() / get_supported_sims()[EASY])
    try:
        _, args, kwargs = capture.decode(call)
    finally:
        capture.close()
    return args, kwargs


def get_baseline_counts() -> tuple[SpatioGenomicLabelledMultiCounts, list[np.ndarray]]:
    """The RDR stage's counts, with its normal spots labelled, and its initial clones."""
    args, kwargs = get_call(BASELINE)
    rdr_args, _ = get_call("08_rdr/run_hmrf")
    labelling = rdr_args[1]
    return normal_spot.determine_normal_baseline(*args, **kwargs), [
        np.flatnonzero(labelling == c) for c in range(labelling.max() + 1)
    ]


def assert_same_returns(expected: Any, found: Any) -> None:
    """Assert both returns, as arrays, encode to the same bytes."""
    with get_replay_file(get_arrays(expected)) as stored, get_replay_file(get_arrays(found)) as replayed:
        assert_byte_equal(stored["out"], replayed["out"])


def legacy_bins(df_gene_snp: Any, adata: Any, counts: Any, phase_indicator: Any) -> Any:
    single_X, _, single_total_bb_RD = get_legacy_channels(counts)
    return legacy_omics.summarize_counts_for_bins(df_gene_snp, adata, single_X, single_total_bb_RD, phase_indicator)


def legacy_bin_filter(df_gene_snp: Any, counts: Any, index_normal: Any, **kwargs: Any) -> Any:
    return legacy_normal_spot.normal_baf_bin_filter(df_gene_snp, *get_legacy_channels(counts), index_normal, **kwargs)


@pytest.mark.parametrize(
    ("call", "legacy", "new"),
    [
        (
            "02_blocks/summarize_counts_for_blocks",
            legacy_omics.summarize_counts_for_blocks,
            omics.summarize_counts_for_blocks,
        ),
        ("04_bins/summarize_counts_for_bins", legacy_bins, omics.summarize_counts_for_bins),
        ("06_normal/normal_baf_bin_filter", legacy_bin_filter, normal_spot.normal_baf_bin_filter),
    ],
    ids=["blocks", "bins", "normal-bin-filter"],
)
def test_the_new_code_is_its_legacy_copy_byte_for_byte(
    call: str, legacy: Callable[..., Any], new: Callable[..., Any]
) -> None:
    """The replacing code returns the frozen copy's bytes on easy's recorded call."""
    args, kwargs = get_call(call)
    expected = legacy(*args, **kwargs)
    args, kwargs = get_call(call)
    assert_same_returns(expected, new(*args, **kwargs))


def test_normal_baseline_sized_is_the_legacy_baseline_byte_for_byte(tmp_path: Path) -> None:
    """`normal_baseline_sized` on easy's RDR-stage counts is the legacy `determine_normal_baseline`'s baseline and
    zeroed read depth, the legacy pseudobulk merge's clone sums, the legacy `load_clone_labels`' baseline (its clone 0
    the normal spots), byte for byte. Fails where the baseline's order of sums or product, or the bins zeroed, move."""
    args, kwargs = get_call(BASELINE)
    _, zeroed, base_nb_mean = legacy_normal_spot.determine_normal_baseline(
        get_legacy_counts(args[0].umis), *args[1:], **kwargs
    )
    counts, clone_index = get_baseline_counts()
    assert_same_returns((zeroed, base_nb_mean), (counts.umis, counts.normal_baseline_sized))

    single_X, single_base_nb_mean, single_total_bb_RD = get_legacy_channels(counts)
    expected = legacy_pseudobulk.merge_pseudobulk_by_index_mix(
        single_X, single_base_nb_mean, single_total_bb_RD, clone_index, None
    )
    found = counts.with_labelling("clone", get_clone_assignment(counts.n_spots, clone_index)).get_pseudobulk("clone")
    assert_same_returns(expected, (found.X, found.normal_baseline_sized.T, found.total_snp_umis.T, None))

    labels = np.full(counts.n_spots, "normal", dtype=object)
    for k, index in enumerate(clone_index[1:], start=1):
        labels[index] = f"clone_{k - 1}"
    config = SimpleNamespace(annotation=SimpleNamespace(clone_label=tmp_path / "clone_label.tsv"))
    pd.DataFrame({"labels": labels}).to_csv(config.annotation.clone_label, sep="\t")
    unlabelled = counts._replace(labellings={})
    assert_same_returns(
        legacy_annotation.load_clone_labels(get_legacy_channels(counts)[0], config),
        annotation.load_clone_labels(unlabelled, config),
    )


def test_a_segmentation_sums_as_its_pairs_byte_for_byte() -> None:
    """On easy's phased bins, the blocks' SNP UMIs and the genes' UMIs summed through a `Segmentation` are the legacy
    pairs form's bytes. Fails where a row is dropped, moved or counted twice."""
    (df_gene_snp, adata, counts, _), _ = get_call("04_bins/summarize_counts_for_bins")
    kept = df_gene_snp.dropna(subset=["block_id", "bin_id"])
    bins = np.unique(kept.bin_id.to_numpy())
    blocks = kept[["block_id", "bin_id"]].drop_duplicates()
    block_ids = blocks.block_id.to_numpy(np.int64)
    block_bins = np.searchsorted(bins, blocks.bin_id.to_numpy())
    genes = kept[kept.is_interval == True][["gene", "bin_id"]].drop_duplicates()
    columns = {g: i for i, g in enumerate(adata.var.index)}
    genes = genes[genes.gene.isin(list(columns))]
    gene_cols = genes.gene.map(columns).to_numpy(np.int64)
    gene_bins = np.searchsorted(bins, genes.bin_id.to_numpy())
    n_bins = bins.size

    found = get_segmented(
        counts.total_snp_umis,
        Segmentation.from_terms(counts.n_segments, block_ids, block_bins, n_bins, name="bin_of_block"),
    )
    expected = legacy_sgc.get_segmented(get_legacy_counts(counts.total_snp_umis), block_bins, block_ids, n_bins)
    assert_same_returns(expected, found)
    segmentation = Segmentation.from_terms(adata.n_vars, gene_cols, gene_bins, n_bins, name="bin_of_gene")
    found = SpatioGenomicCounts.from_grouped_columns(adata.layers["count"], segmentation, np.int64)
    expected = legacy_sgc.from_grouped_columns(
        legacy_sgc.SpatioGenomicCounts, adata.layers["count"], gene_cols, gene_bins, n_bins, np.int64
    )
    assert_same_returns(expected, found)


@pytest.mark.parametrize("key", ["rdr", "baf"])
def test_the_encodings_are_the_legacy_count_encoder_byte_for_byte(key: str) -> None:
    """Each encoding of easy's first recorded pseudobulk holds the legacy `CountEncoder`'s unique pairs, its inverse
    offset by the columns before it, and its decode, byte for byte. Fails where `np.unique`'s order, the classes or
    the gather move."""
    args, kwargs = get_call(ENCODING)
    count, total = get_encoding_channels()[key]
    legacy = CountEncoder(getattr(args[0], count).T, getattr(args[0], total).T, kwargs["common_zero_depth"])
    found = args[0].update_encoding(key, (count, total), **kwargs)
    offsets = np.cumsum([0] + [len(u) for u in legacy.unique_obs])
    inverse = np.stack([v.astype(np.int64) + offset for v, offset in zip(legacy.inverse, offsets, strict=False)], 1)
    values = np.random.default_rng(0).random((3, int(offsets[-1])))
    decoded = [legacy.decode(values[:, offsets[s] : offsets[s + 1]], s, axis=1) for s in range(len(legacy.inverse))]
    assert_same_returns(
        (legacy.unique_obs, legacy.unique_total, inverse, np.stack(decoded, axis=2), legacy.compression_rate),
        (
            *found.get_unique(key),
            found.get_equivalence_class(key).T,
            found.decode(values, 1, key).transpose(0, 2, 1),
            found.compression_rate(key),
        ),
    )


def test_log_lambda_and_log_mu_shift_are_the_legacy_to_rtol_1e_12() -> None:
    """On easy's RDR-stage clones, stacked: `get_log_lambda`, each clone's baseline over its own sum, is the legacy
    pooled `normal_lambda`, logged and tiled per clone, and `get_log_mu_shift` on it, at a path's and at soft state
    weights, the legacy normalizers, to rtol 1e-12 (T- #118). Fails where a clone reads another's bins."""
    counts, clone_index = get_baseline_counts()
    pseudobulk = counts.with_labelling("clone", get_clone_assignment(counts.n_spots, clone_index))
    pseudobulk = pseudobulk.get_pseudobulk("clone").get_stacked()
    n_clones = len(pseudobulk.clone_lengths)
    log_lambda = np.tile(
        np.log(legacy_hmrf.get_normal_lambda(get_legacy_counts(counts.normal_baseline_sized))), n_clones
    )
    np.testing.assert_allclose(pseudobulk.get_log_lambda().reshape(-1), log_lambda, rtol=1e-12, atol=0.0)
    rng = np.random.default_rng(0)
    parameters = CnaHmmParameters(None, rng.normal(0.0, 0.3, 7), None, None, None)
    for weights in (
        get_hard_state_posteriors(rng.integers(0, 7, log_lambda.size), 7),
        np.log(rng.dirichlet(np.ones(7), log_lambda.size)),
    ):
        expected = legacy_hmrf.get_log_mu_shift(parameters, weights, log_lambda, pseudobulk.clone_lengths, None)
        found = get_log_mu_shift(parameters, weights, pseudobulk).reshape(-1)
        np.testing.assert_allclose(found, expected, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("group", ["03_phasing", "08_rdr"])
def test_the_initial_labelling_is_the_legacy_clone_start_byte_for_byte(group: str) -> None:
    """`get_initial_labelling` on easy's recorded call is the labelling of the legacy `get_phasing_clones` (phasing)
    or `initialize_rdr_clone_refinement` (rdr), byte for byte. Fails where a cut, a draw or a label offset moves."""
    args, kwargs = get_call(f"{group}/get_initial_labelling")
    found = get_initial_labelling(*args, **kwargs)
    counts, (coords, physical, sample_ids), config = args
    state = np.random.get_state()
    try:
        if kwargs["stage"] == "phasing":
            clone_index = legacy_spatial.get_phasing_clones(coords, physical, sample_ids, config)
            expected = get_clone_assignment(counts.n_spots, clone_index)
        else:
            expected, _, _ = legacy_spatial.initialize_rdr_clone_refinement(
                counts.labellings["clone"],
                coords,
                physical,
                get_legacy_counts(counts.total_snp_umis),
                counts.n_segments,
                config,
            )
    finally:
        np.random.set_state(state)
    assert_same_returns(expected, found)


@pytest.mark.parametrize(
    ("group", "params", "legacy_stage"),
    [("03_phasing", "sp", "phasing"), ("05_baf", "sp", "core_inference"), ("08_rdr", "smp", "core_inference")],
)
def test_the_initial_parameters_are_the_legacy_start_byte_for_byte(group: str, params: str, legacy_stage: str) -> None:
    """`get_initial_parameters` on easy's recorded call is the legacy `configured_initializer`'s start, completed as
    the legacy `get_initial_params` completed it, byte for byte. Fails where a state, its order or a default moves."""
    args, kwargs = get_call(f"{group}/get_initial_parameters")
    found = get_initial_parameters(*args, **kwargs)
    counts, labelling, config = args
    pseudobulk = counts.with_labelling("clone", labelling).get_pseudobulk("clone").get_stacked()
    log_mu, p_binom, _, _ = legacy_hmm_initialize.configured_initializer()(
        config.hmm.n_states,
        pseudobulk.X,
        pseudobulk.normal_baseline_sized.T,
        pseudobulk.total_snp_umis.T,
        params,
        np.tile(counts.lengths, len(pseudobulk.clone_lengths)),
        None,
        None,
        random_state=config.hmm.random_state,
        in_log_space=False,
        only_minor=legacy_hmm_initialize.minor_allele_only_at_stage(legacy_stage),
    )
    log_mu = log_mu if "m" in params else None
    *blocks, log_startprob, _ = legacy_hmm_initialize.get_initial_params(
        config.hmm.t, config.hmm.n_states, 1, log_mu, p_binom
    )
    expected = (log_startprob, *(np.asarray(block).reshape(-1) for block in blocks))
    assert_same_returns(expected, tuple(found))


def assert_legacy_forward_backward(
    coding: Any,
    log_startprob: np.ndarray,
    log_transmat: np.ndarray,
    lengths: np.ndarray,
    log_switch: np.ndarray | None,
    penalize: bool,
    posterior_atol: float,
) -> None:
    """`get_log_alpha`, `get_log_beta`, `get_log_gamma` and `get_llf` on clone columns against the legacy kernels on
    the clones stacked along bins, posteriors as probabilities: bitwise unphased; phased rtol 1e-12, posteriors also
    within `posterior_atol`."""
    n_clones, n_bins = coding.rdr_index.shape
    rdr, baf = (np.reshape(table, (n_clones, table.shape[1], -1)) for table in (coding.rdr, coding.baf))
    coding = coding._replace(rdr=rdr, baf=baf)
    starts = get_contig_offsets(lengths)
    steps = get_phase_steps(log_switch, n_bins)
    log_alpha = get_log_alpha(*coding, log_startprob, log_transmat, starts, *steps, penalize)
    log_beta = get_log_beta(*coding, log_transmat, starts, *steps, penalize)
    found = (log_alpha, log_beta, np.exp(get_log_gamma(log_alpha, log_beta)))

    n_paired = log_alpha.shape[-1]
    stacked_lengths = np.tile(lengths, n_clones)
    rdr, baf = get_decoded_emission(coding)
    stacked = (rdr + baf).reshape(n_clones * n_bins, n_paired).T.copy()
    if log_switch is not None:
        args = (stacked_lengths, log_transmat, log_startprob, stacked[:, :, None], np.tile(log_switch, n_clones))
        alpha = legacy_hmm.phased_forward_lattice(*args, penalize)
        beta = legacy_hmm.phased_backward_lattice(*args, penalize)
    else:
        starts_stacked = legacy_hmm.get_lengths_start_indices(stacked_lengths)
        alpha = legacy_hmm.forward_recursion(stacked, log_startprob, log_transmat, starts_stacked, stacked_lengths)
        beta = legacy_hmm.backward_recursion(stacked, log_transmat, starts_stacked, stacked_lengths)
    gamma = np.exp(legacy_hmm.compute_copy_state_posterior(alpha.copy(), beta))
    llf = np.sum(scipy.special.logsumexp(alpha[:, np.cumsum(stacked_lengths) - 1], axis=0))
    expected = tuple(v.T.reshape(n_clones, n_bins, n_paired) for v in (alpha, beta, gamma))

    if log_switch is not None:
        for f, e, atol in zip(found, expected, (0.0, 0.0, posterior_atol), strict=True):
            np.testing.assert_allclose(f, e, rtol=1e-12, atol=atol)
        np.testing.assert_allclose(get_llf(log_alpha, starts), llf, rtol=1e-12, atol=0.0)
    else:
        assert all(np.array_equal(f, e) for f, e in zip(found, expected, strict=True))
        assert get_llf(log_alpha, starts) == llf


def get_fit_columns(res: Any) -> tuple[Any, ...]:
    """The per-clone and labelling entries of a fit: labels, paths, posteriors, confounders and `total_llf`."""
    return (res["new_assignment"], res["pred_cnv"], res["log_gamma"], *res.confounders, res["total_llf"])


def get_transposed_fit(res: Any) -> Any:
    """`res` with `pred_cnv` and `log_gamma` between clone-major and the legacy `(n_bins, n_clones)` layout.

    Args:
        res: the `HmrfFit`, not modified.

    Returns:
        The copy, `pred_cnv` transposed and `log_gamma`'s clone and state axes swapped.
    """
    transposed = get_nested_copy(res)
    transposed["pred_cnv"] = np.asarray(res["pred_cnv"]).T
    transposed["log_gamma"] = np.asarray(res["log_gamma"]).transpose(2, 1, 0)
    return transposed


@pytest.mark.parametrize("call", ["05_baf/get_merged_fit", "08_rdr/get_merged_fit", "08_rdr/get_reindexed_fit"])
def test_the_merged_and_reindexed_fits_are_the_legacy_byte_for_byte(call: str) -> None:
    """`get_merged_fit` and `get_reindexed_fit` on easy's recorded calls are the legacy `merge_by_minspots` and
    `reindex_clones`, byte for byte, the legacy fits transposed to clone-major. Fails where a group, its
    representative or the normal-first order moves."""
    args, kwargs = get_call(call)
    legacy_res = get_transposed_fit(args[0])
    if call.endswith("get_merged_fit"):
        _, total_snp_umis, min_spots, min_snp_umis = args
        assignment = legacy_res["new_assignment"]
        _, legacy = legacy_hmrf.merge_by_minspots(
            assignment, legacy_res, get_legacy_counts(total_snp_umis), min_spots, min_snp_umis
        )
    else:
        legacy, _ = legacy_hmrf.reindex_clones(legacy_res)
    expected = get_transposed_fit(legacy)
    args, kwargs = get_call(call)
    found = (get_merged_fit if call.endswith("get_merged_fit") else get_reindexed_fit)(*args, **kwargs)
    assert_same_returns(get_fit_columns(expected), get_fit_columns(found))


def test_the_neutral_pin_is_the_legacy_pin_byte_for_byte() -> None:
    """`get_neutral_pinned` on easy's BAF fit, given each clone a nonzero `log Z_c`, is the legacy
    `pin_neutral_state`, byte for byte. Fails where the neutral state or the shift moves."""
    (res, *_), _ = get_call("05_baf/get_merged_fit")
    shift = np.random.default_rng(0).normal(0.0, 0.3, np.shape(res.confounders.log_mu_shift))
    res.confounders = res.confounders._replace(log_mu_shift=shift)
    paths = np.asarray(res.profile.pred_cnv).T
    found = get_neutral_pinned(res.parameters, res.confounders, paths, True)
    legacy_hmrf.pin_neutral_state(res, fitted_baf=True)
    assert_same_returns((res.parameters.log_mu, *res.confounders), (found[0].log_mu, *found[1]))


@pytest.mark.parametrize(("phased", "penalize"), [(False, False), (True, False), (True, True)])
def test_baum_welch_is_the_legacy_forward_backward(phased: bool, penalize: bool) -> None:
    """3 clone columns of a random dense emission: `assert_legacy_forward_backward`, the phased factored step summing
    in another order. Fails where a clone, contig restart or phase step departs from the legacy lattice."""
    rng = np.random.default_rng(102)
    n_clones, n_states, lengths = 3, 5, np.array([40, 25, 61])
    n_bins, n_paired = int(lengths.sum()), n_states * (2 if phased else 1)
    log_transmat = np.log(rng.dirichlet(np.ones(n_states) * 0.3, n_states))
    log_startprob = np.log(rng.dirichlet(np.ones(n_states)))
    log_switch = np.log(rng.uniform(1e-4, 0.3, n_bins))
    log_emission = rng.normal(-30.0, 10.0, (n_clones, n_bins, n_paired))
    coding = get_dense_encoded_emission(log_emission)
    assert_legacy_forward_backward(
        coding, log_startprob, log_transmat, lengths, log_switch if phased else None, penalize, 0.0
    )


@pytest.mark.parametrize("phased", [False, True])
def test_baum_welch_on_easy_s_emission_is_the_legacy_forward_backward(phased: bool) -> None:
    """Easy's recorded phasing-stage coded emission (4 clones stacked), as recorded and `phased`, as clone columns,
    with its contigs and phase switches at `hmm.t_phaseing`: `assert_legacy_forward_backward`, phased posteriors within
    1e-11 (measured 2.7e-12; lattices 5.8e-15 relative, llf 2e-15). Fails where the coded
    clone columns or the contig restarts depart from the legacy lattice on the run's data."""
    capture = Capture(get_project() / get_supported_sims()[EASY])
    try:
        function, args, kwargs = capture.decode("03_phasing/get_log_emission")
        coded = function(*args, **{**kwargs, "phased": phased})
        _, phasing_args, _ = capture.decode("03_phasing/initial_phase_given_partition")
    finally:
        capture.close()
    counts, initial_clone_index, log_switch = phasing_args[0], phasing_args[1], phasing_args[4]
    coding = get_clone_columns(coded, len(initial_clone_index))
    n_states = coding.rdr.shape[-1]
    log_transmat = get_log_transmat(n_states, get_global_config().hmm.t_phaseing)
    log_startprob = np.full(n_states, -np.log(n_states))
    assert_legacy_forward_backward(
        coding, log_startprob, log_transmat, counts.lengths, log_switch if phased else None, False, 1e-11
    )


@pytest.mark.parametrize(
    ("encoded", "phased", "data_terms"),
    [(False, False, True), (True, False, False), (True, True, True), (False, True, True)],
    ids=["dense", "coded", "coded-phased", "dense-phased"],
)
def test_the_lattice_is_the_legacy_get_log_emission_byte_for_byte(
    encoded: bool, phased: bool, data_terms: bool
) -> None:
    """On easy's RDR-stage clones, stacked, each its own `log Z_c` and one `theta`: the lattice, dense or decoded from
    the class tables, is the legacy `get_log_emission`'s, byte for byte; and per clone as columns. Fails where a class
    reads another clone's `log Z_c` or the sum order of a channel changes."""
    counts, clone_index = get_baseline_counts()
    pseudobulk = counts.with_labelling("clone", get_clone_assignment(counts.n_spots, clone_index))
    columns = pseudobulk.get_pseudobulk("clone")
    stacked = columns.get_stacked()
    n_clones = stacked.get_clone_lengths().size
    rng = np.random.default_rng(1)
    log_mu, p_binom = rng.normal(0.0, 0.3, 7), rng.uniform(0.1, 0.5, 7)
    parameters = CnaHmmParameters(None, log_mu, p_binom, np.full(7, 0.05), np.full(7, 40.0))
    confounders = CnaEmissionConfounders(np.full((n_clones, 1), 0.1), rng.normal(0.0, 0.2, (n_clones, 1)))
    for found, confounded in ((stacked, confounders), (columns, get_unconfounded(1))):
        kwargs: dict[str, Any] = {"layout": "lattice", "phased": phased, "data_terms": data_terms}
        if encoded:
            found = emission.get_encoded(found, common_zero_depth=False)
            new = get_decoded_emission(get_log_emission(found, parameters, confounded, encoding="coded", **kwargs))
        else:
            new = get_log_emission(found, parameters, confounded, **kwargs)
        expected = legacy_emission.get_log_emission(found, parameters, confounded, fold=False, **kwargs)
        for want, got in zip(expected, new, strict=True):
            assert get_state_major(got).tobytes() == np.asarray(want).tobytes()


@pytest.mark.parametrize("sparse", [False, True], ids=["dense", "csr"])
def test_the_labelling_unary_encoding_is_the_legacy_realized_count_slots_byte_for_byte(sparse: bool) -> None:
    """On easy's RDR-stage counts as the field reads them, dense and CSR: the `labelling_unary` encoding's counts are
    the legacy `mark_realized_*` counts, byte for byte, and each count's class tables the count its legacy
    `get_count_slots` slot does. Fails where a read count is missed, a pair with b > n is marked, or a class reads
    another count's column."""
    counts, _ = get_baseline_counts()
    channels = (counts.umis, counts.normal_baseline_sized, counts.alt_snp_umis, counts.total_snp_umis)
    by_spot = [SpatioGenomicCounts.of(sp.csr_matrix(c.toarray()) if sparse else c.toarray()) for c in channels]
    read = get_labelling_unary_counts(*by_spot)
    extent = max(count_table_size(get_count_values(by_spot[k]), 1_000_000) for k in (0, 2, 3))
    realized = legacy_tabled.get_realized(read.umis, read.alt_snp_umis, read.total_snp_umis, extent)
    slots, legacy_counts = legacy_tabled.get_count_slots(realized)
    key = "labelling_unary"
    read = read.update_encoding(key, get_encoding_channels()[key], common_zero_depth=False)
    (found,), classes = read.get_unique(key)[0], read.get_equivalence_class(key)[0]
    assert np.flatnonzero(realized).tobytes() == found.tobytes()
    assert legacy_counts[slots[found]].tobytes() == found[classes[found]].tobytes() == found.tobytes()
    assert np.count_nonzero(classes >= 0) == found.size


@pytest.mark.parametrize("phased", [False, True], ids=["unphased", "phased"])
def test_the_clone_field_is_the_legacy_clone_decoded_unary_of_fit_byte_for_byte(phased: bool) -> None:
    """On easy's RDR-stage counts, 4 clones' seeded paths, each its own `log Z_c`: `get_labelling_unary` is the legacy
    `clone_decoded_unary_of_fit`, byte for byte. Fails where `get_spot_weights` or a clone's
    path moves."""
    counts, _ = get_baseline_counts()
    rng = np.random.default_rng(2)
    n_clones, n_states = 4, 7
    paths = rng.integers(0, (2 if phased else 1) * n_states, (n_clones, counts.n_segments))
    parameters = CnaHmmParameters(
        None,
        rng.normal(0.0, 0.3, n_states),
        rng.uniform(0.1, 0.5, n_states),
        np.full(n_states, 0.05),
        np.full(n_states, 40.0),
    )
    confounders = CnaEmissionConfounders(np.zeros((n_clones, 1)), rng.normal(0.0, 0.2, (n_clones, 1)))
    expected = legacy_hmrf.clone_decoded_unary_of_fit(counts, parameters, confounders, paths, phased=phased)
    found = get_labelling_unary(counts, parameters, confounders, paths, phased=phased)
    assert np.any(counts.get_spot_weights() != 1.0)
    assert found.tobytes() == expected.tobytes()


@pytest.mark.parametrize(("params", "fold"), [("mp", False), ("p", True)])
def test_the_coded_tumour_rows_are_the_legacy_clone_columns_byte_for_byte(params: str, fold: bool) -> None:
    """On easy's recorded decode inputs, each tumour clone at its own `theta_c` (one 0) and `log Z_c`: the emission of
    the coded tumour rows (`get_coded_rows`) is the legacy per-clone columns', byte for byte. Fails where a clone reads
    another clone's classes, `theta_c` or `log Z_c`, or the fold or a channel moves."""
    (counts, parameters, confounders, *_), _ = get_call(DECODE)
    counts = PseudobulkCounts(*(np.asarray(v, dtype=np.float64) for v in counts[:4]), None, {})
    n_clones = counts.umis.shape[0]
    tumour = [c for c in range(n_clones) if confounders.normal_fraction[c, 0] != 1.0]
    rng = np.random.default_rng(124)
    thetas, log_z = rng.uniform(0.0, 0.5, n_clones), rng.normal(0.0, 0.2, n_clones)
    thetas[tumour[0]] = 0.0
    pairs = get_ab_lattice(params=params, fold=fold, max_allele_copy=5, max_total_copy=6)
    lattice = get_lattice_parameters(pairs, np.ravel(parameters.alphas)[0], np.ravel(parameters.taus)[0])
    prior = 0.1 * ~get_ab_is_ploidy_neutral(pairs, 2)
    columns = {c: legacy_ab_decode.get_clone_column(counts, c, params=params, fold=fold) for c in tumour}
    confounded = CnaEmissionConfounders(thetas[:, None], log_z[:, None])
    expected = legacy_ab_decode.get_clone_emission_loglikes(columns, lattice, confounded, prior, params=params)
    rows = np.asarray(tumour)
    confounded = CnaEmissionConfounders(thetas[rows, None], log_z[rows, None])
    found = get_clone_emission_loglikes(
        get_coded_rows(counts, tumour, fold=fold), lattice, confounded, prior, params=params, clones=tumour
    )
    assert sorted(found) == sorted(expected) == tumour
    for c in tumour:
        assert found[c].tobytes() == expected[c].tobytes(), c


@pytest.mark.parametrize("price", [0.0, 0.5, -2.0])
def test_the_clone_viterbi_is_the_legacy_uniform_switch_viterbi_byte_for_byte(price: float) -> None:
    """On easy's recorded decode, each tumour clone's emission at its start and the run's chain, priced
    `price * (A + B - 2)`: `get_clone_viterbi` on the priced class tables is the legacy `viterbi_uniform_switch`, path
    and score byte for byte. Fails where a step picks another predecessor, a tie leaves the stay, or a contig restarts
    elsewhere."""
    (counts, parameters, confounders, lengths, config), _ = get_call(DECODE)
    counts = PseudobulkCounts(*(np.asarray(v, dtype=np.float64) for v in counts[:4]), None, {})
    tumour = [c for c in range(counts.umis.shape[0]) if confounders.normal_fraction[c, 0] != 1.0]
    rows = np.asarray(tumour)
    pairs = get_ab_lattice(params="mp", fold=False, max_allele_copy=5, max_total_copy=6)
    lattice = get_lattice_parameters(pairs, np.ravel(parameters.alphas)[0], np.ravel(parameters.taus)[0])
    confounded = CnaEmissionConfounders(*(np.asarray(v)[rows] for v in confounders))
    prior = 0.1 * ~get_ab_is_ploidy_neutral(pairs, 2)
    emission = get_clone_emission_loglikes(
        get_coded_rows(counts, tumour, fold=False), lattice, confounded, prior, params="mp", clones=tumour
    )
    chain = get_clone_chain(pairs, lengths, config, ploidy=2)
    term = price * (pairs.sum(axis=1) - 2)
    for c in tumour:
        coded = get_dense_encoded_emission(emission[c].T[None])
        found, score = get_clone_viterbi(coded._replace(rdr=coded.rdr + term), chain)
        args = (chain.log_stay, chain.log_switch, chain.log_start, chain.lengths)
        path, value = legacy_ab_decode.viterbi_uniform_switch(emission[c] + term[:, None], *args)
        assert found[0].tobytes() == path.tobytes() and score[0] == value, c


def test_the_clone_chains_potts_problem_is_the_legacy_one_byte_for_byte() -> None:
    """On easy's recorded decode, the tumour clones' emissions at their start and the run's chain: the shared
    `potts.potts_problem` posing has the legacy builder's unary and edges, each `(first, second, coupling)` byte for
    byte, edges in CSR order; OpenGM's expansion from each clone's Viterbi labels both alike. Fails where an edge, its
    coupling or a segment's start prior moves."""
    (counts, parameters, confounders, lengths, config), _ = get_call(DECODE)
    counts = PseudobulkCounts(*(np.asarray(v, dtype=np.float64) for v in counts[:4]), None, {})
    tumour = [c for c in range(counts.umis.shape[0]) if confounders.normal_fraction[c, 0] != 1.0]
    rows = np.asarray(tumour)
    pairs = get_ab_lattice(params="mp", fold=False, max_allele_copy=5, max_total_copy=6)
    lattice = get_lattice_parameters(pairs, np.ravel(parameters.alphas)[0], np.ravel(parameters.taus)[0])
    confounded = CnaEmissionConfounders(*(np.asarray(v)[rows] for v in confounders))
    prior = 0.1 * ~get_ab_is_ploidy_neutral(pairs, 2)
    emission = get_clone_emission_loglikes(
        get_coded_rows(counts, tumour, fold=False), lattice, confounded, prior, params="mp", clones=tumour
    )
    chain = get_clone_chain(pairs, lengths, config, ploidy=2)
    expected = legacy_ab_decode.hmm_ab_decoder_potts_problem(chain, emission, tumour)
    found = get_clone_chains_potts_problem(chain, emission, tumour)
    order = np.lexsort((expected.second, expected.first))
    assert found.unary.tobytes() == expected.unary.tobytes()
    for name in ("first", "second", "coupling"):
        assert getattr(found, name).tobytes() == getattr(expected, name)[order].tobytes(), name
    start = np.concatenate(
        [get_clone_viterbi(get_dense_encoded_emission(emission[c].T[None]), chain)[0][0] for c in tumour]
    )
    session = opengm_session()
    labels = [session.solve(p, "expansion", start=start, mincut="boykov_kolmogorov").labels for p in (expected, found)]
    assert labels[0].tobytes() == labels[1].tobytes()


def test_labelled_counts_read_back_are_the_run_s_binned_data_byte_for_byte(tmp_path: Path) -> None:
    """`io.get_labelled_counts` on easy's inputs and its written `segmentation.tsv.gz`, `snp_segmentation.tsv.gz`
    and `clone_labels.tsv.gz` writes the run's `binned_data.npz` (the legacy writer on the RDR stage's counts), entry
    by entry, the spots named by index. Fails where a gene or SNP is summed into another bin, a phase flip is lost, or a zeroed bin moves."""
    capture = Capture(get_project() / get_supported_sims()[EASY])
    try:
        for name in ("segmentation.tsv.gz", "snp_segmentation.tsv.gz", "clone_labels.tsv.gz"):
            (tmp_path / name).write_bytes(capture.h5["files"][name][()].tobytes())
    finally:
        capture.close()
    config = YAMLConfig.from_file(get_template())
    config.over_ride(get_overrides(get_fixtures()[EASY]))
    with open(tmp_path / results.run_config_name(results.run_id(config)), "w") as out:
        yaml.safe_dump(config.to_dict(), out, sort_keys=False)

    for directory, counts in (
        ("expected", get_baseline_counts()[0]),
        ("found", get_labelled_counts(get_project(), tmp_path)),
    ):
        (tmp_path / directory).mkdir()
        barcodes = [f"spot_{i}" for i in range(counts.n_spots)]
        single_X, single_base_nb_mean, single_total_bb_RD = get_legacy_channels(counts)
        legacy_utils.write_binned_data(
            tmp_path / directory,
            counts.lengths,
            barcodes,
            single_X=single_X,
            single_base_nb_mean=single_base_nb_mean.values,
            single_total_bb_RD=single_total_bb_RD,
        )
    with (
        np.load(tmp_path / "expected" / "binned_data.npz") as expected,
        np.load(tmp_path / "found" / "binned_data.npz") as found,
    ):
        assert expected.files == found.files
        for name in expected.files:
            assert expected[name].dtype == found[name].dtype, name
            assert expected[name].tobytes() == found[name].tobytes(), name


@pytest.mark.parametrize("weighting", ["unshifted", "argmax", "posterior", "phased"])
def test_the_m_step_cost_and_gradient_are_the_legacy_analytic_ones(weighting: str) -> None:
    """On easy's RDR-stage clones, stacked, at seeded parameters and posteriors: `hmm_mstep.get_cost_and_gradient` is
    the fit's cost, `-sum(gamma log P)` over every bin, to rtol 1e-12, and its gradient the frozen analytic
    `em_gradient`, to rtol 1e-8; `log Z_c` on a path's or soft state weights. Fails where a class weight is lost, a
    clone reads another's `log Z_c`, or a derivative is wrong."""
    counts, clone_index = get_baseline_counts()
    labelled = counts.with_labelling("clone", get_clone_assignment(counts.n_spots, clone_index))
    stacked = labelled.get_pseudobulk("clone").get_stacked()
    base = get_max_rdr_zeroed_baseline(stacked.umis.T, stacked.normal_baseline_sized.T, 5.0)
    encoded = emission.get_encoded(stacked._replace(normal_baseline_sized=base.T), common_zero_depth=False)
    n_states, n_obs, phased = 7, base.shape[0], weighting == "phased"
    rng = np.random.default_rng(3)
    gamma = rng.dirichlet(np.ones((2 if phased else 1) * n_states), n_obs).T
    log_state_weights = {
        "argmax": get_hard_state_posteriors(rng.integers(0, n_states, n_obs), n_states),
        "posterior": np.log(rng.dirichlet(np.ones(n_states), n_obs)),
    }.get(weighting)
    start = (np.log(np.full(n_states, 1 / n_states)), rng.normal(0, 0.3, n_states), rng.uniform(0.1, 0.5, n_states))
    canonical = CnaHmmParameters(*start, np.full(n_states, 0.05), np.full(n_states, 40.0))
    model = hmm_nophasing(params="smp")
    theta = model.pack_params(*canonical)
    weights = hmm_mstep.get_class_weights(
        encoded, gamma, log_state_weights, stacked, n_states=n_states, data_terms=False
    )
    cost, gradient = hmm_mstep.get_cost_and_gradient(
        theta, canonical, weights, n_states=n_states, params="smp", optimize_nb=True, use_logit=True
    )

    parameters = CnaHmmParameters(None, *model.unpack_params(theta, n_states, *canonical)[1:])
    confounders = get_fit_confounders(
        parameters, log_state_weights, stacked, get_unconfounded(stacked.get_clone_lengths().size)
    )
    rdr, baf = get_decoded_emission(
        get_log_emission(
            encoded, parameters, confounders, layout="lattice", encoding="coded", phased=phased, data_terms=False
        )
    )
    model.state_posteriors = gamma
    held = CnaHmmParameters(canonical.log_startprob, None, None, canonical.alphas, canonical.taus)
    expected = legacy_mstep.em_gradient(
        model, n_states, stacked, base, held, optimize_nb=True, use_logit=True, log_state_weights=log_state_weights
    )(theta)
    np.testing.assert_allclose(float(cost), -np.sum(gamma * get_state_major(rdr + baf)[..., 0]), rtol=1e-12, atol=0)
    np.testing.assert_allclose(np.asarray(gradient), expected, rtol=1e-8, atol=0)
