import time

import numpy as np
import pandas as pd

from cnamaste.config import get_global_config, start_time
from cnamaste.emission import get_unconfounded
from cnamaste.hmm.baum_welch import HmmTransitions, baum_welch
from cnamaste.hmm_utils import get_log_transmat
from cnamaste.hmrf_fit import get_hmrf_fit
from cnamaste.hmrf_utils import get_clone_assignment
from cnamaste.logger import get_logger

logger = get_logger(__name__, start_time=start_time)


def initial_phase_given_partition(
    counts,
    initial_clone_index,
    initial_parameters,
    n_states,
    log_sitewise_transmat,
    params,
    t,
    tol,
    known_normal=False,
):
    """Phase blocks by per-clone HMMs on the given clone partition, and refine `lengths`.

    Aggregates by `initial_clone_index`, fits dispersions on minor BAF, then a phased Baum-Welch per clone; the phase is
    the majority vote across clones (`get_phase_indicator`), ignoring normal-like segments (BAF ~ 0.5). Splits segments
    where minor BAF steps (`get_refined_lengths`).

    Args:
        counts: a `SpatioGenomicLabelledMultiCounts`, no normal spots labelled.
        initial_clone_index: spots per clone.
        initial_parameters: the start, `initialize.genomic.get_initial_parameters` at `phasing`.
        n_states: the copy states.
        log_sitewise_transmat: the phase switches.
        params: the parameters fit.
        t: the stay probability.
        tol: the Baum-Welch tolerance.
        known_normal: drop clone 0's phase vote.

    Returns:
        The fit, the phase indicator and the refined lengths.
    """
    assert counts.normal_baseline_sized.is_zero()
    lengths = counts.lengths

    # NB TODO attractor to 0.5 if sufficiently close, independent of coverage.
    start_time = time.time()

    # NB on input phase_indicator is 0s by construction, up to phase switch errors TBD.
    logger.info(f"Starting phasing assuming {len(initial_clone_index)} clones.")

    # NB the initial partition's pseudobulk, one column per clone.
    labels = get_clone_assignment(counts.n_spots, initial_clone_index)
    pseudobulk = counts.with_labelling("clone", labels).get_pseudobulk("clone")
    n_clones = len(initial_clone_index)
    transitions = HmmTransitions(get_log_transmat(n_states, t), log_sitewise_transmat, lengths)
    fit = baum_welch(
        pseudobulk,
        initial_parameters,
        get_unconfounded(n_clones),
        transitions,
        params=params,
        tol=tol,
        previous_log_gamma=None,
        config=get_global_config(),
    )
    phasing_fit = get_hmrf_fit(fit, transitions.log_transmat)

    # NB the (phase, state) argmax, (n_clones, n_obs): a state of the first n_states is phase 1.
    pred = np.argmax(fit.log_gamma.reshape(n_clones, -1, 2 * n_states), axis=-1)
    phase_indicator = get_phase_indicator(
        pred, phasing_fit.parameters.p_binom, n_states, known_normal, get_global_config().phasing.baf.eps
    )

    # TODO HACK < -> <= to reduce flips for phasing.baf.eps.
    config = get_global_config()
    model_baf_profiles = get_model_baf_profiles(pred, phasing_fit.parameters.p_binom, n_states)
    minor_baf_profiles = np.where(model_baf_profiles < 0.5, model_baf_profiles, 1.0 - model_baf_profiles)
    refined_lengths = get_refined_lengths(
        minor_baf_profiles, lengths, config.phasing.baf_change_threshold, config.phasing.min_new_segment_size
    )

    end_time = time.time()

    # NB total number of blocks is conserved, but redistributed between "contigs".
    logger.info(
        f"Solved for {len(refined_lengths)} phase-refined lengths () given {len(lengths)} input lengths with sum={sum(lengths)} in {(end_time - start_time):.2f} seconds."
    )

    return phasing_fit, phase_indicator, refined_lengths


def get_model_baf_profiles(pred, p_binom, n_states):
    """The B-allele frequency of each clone's decoded (state, phase), phase 1 the state as fit, phase 0 its mirror.

    Args:
        pred: (n_clones, n_obs) argmax over `2 * n_states` phased states.
        p_binom: (n_states,) the fit's BAF.
        n_states: the states.

    Returns:
        (n_clones, n_obs) BAF, `p_binom[pred]` for `pred < n_states`, else `1 - p_binom[pred - n_states]`.
    """
    base_bafs = np.asarray(p_binom).reshape(-1)[pred % n_states]
    return np.where(pred < n_states, base_bafs, 1.0 - base_bafs)


def get_phase_indicator(pred, p_binom, n_states, known_normal, eps_baf):
    """The phase per block: the majority vote of the clones' decoded phases, a tie voting 1.

    A clone whose model BAF lies within `eps_baf` of 1/2 casts no vote on that block; a block with no vote takes
    phase 0 (T- #27). Phase 1 is `pred < n_states`, the state unflipped.

    Args:
        pred: (n_clones, n_obs) argmax over `2 * n_states` phased states.
        p_binom: (n_states,) the fit's BAF.
        n_states: the states.
        known_normal: drop clone 0's vote.
        eps_baf: the half-width of the no-vote band about 1/2.

    Returns:
        (n_obs,) int, 0 or 1.
    """
    model_baf_profiles = get_model_baf_profiles(pred, p_binom, n_states)
    phase_profiles = (pred < n_states).astype(np.int8)

    # NB do not define phase for normal-like segments (for this clone).
    phase_profiles[np.abs(model_baf_profiles - 0.5) < eps_baf] = -1
    phase_votes = phase_profiles[1:, :] if known_normal else phase_profiles

    valid_mask = phase_votes != -1
    valid_counts = np.sum(valid_mask, axis=0)
    valid_sums = np.sum(np.where(valid_mask, phase_votes, 0), axis=0)

    phase_indicator = np.zeros(pred.shape[1], dtype=int)
    has_votes = valid_counts > 0
    phase_indicator[has_votes] = (valid_sums[has_votes] / valid_counts[has_votes]) >= 0.5
    return phase_indicator


def get_refined_lengths(minor_baf_profiles, lengths, baf_change_threshold, min_new_segment_size):
    """Split each contig's blocks where any clone's minor BAF steps by at least `baf_change_threshold`.

    A split at block `i` needs `i > s + min_new_segment_size`, `s` the contig's last split; no segment crosses a
    contig, so `sum` is conserved.

    Args: `minor_baf_profiles` (n_clones, n_obs); `lengths` blocks per contig; the two thresholds,
    `config.phasing`'s `baf_change_threshold` and `min_new_segment_size`.

    Returns:
        (n_segments,) blocks per phasing segment, `n_segments >= len(lengths)`.
    """
    refined_lengths = []
    cumlen = 0
    for ii, le in enumerate(lengths):
        s = 0
        for i in range(le):
            if i <= s + min_new_segment_size:
                continue
            step = np.abs(minor_baf_profiles[:, i + cumlen] - minor_baf_profiles[:, i + cumlen - 1])
            if np.any(step >= baf_change_threshold):
                logger.warning(
                    f"Forced a block boundary at contig {1 + ii} pos {i} despite conserved phase (minor baf state), given a baf switch of {step.max():.4f}."
                )
                refined_lengths.append(i - s)
                s = i
        # NB force a stop at contig end.
        refined_lengths.append(le - s)
        cumlen += le
    return np.array(refined_lengths)


def cumulative_phase_switches(df_gene_snp, phase_indicator):
    """The phase switches the run inferred, counted along each contig: at each bin, the number of consecutive block
    pairs on its contig, up to its last block, whose `phase_indicator` differ; the count restarts at 0 on each contig.

    Args:
        df_gene_snp: with `CHR`, `block_id` and `bin_id`.
        phase_indicator: one 0/1 per block.

    Returns:
        `(n_bins,)` counts, rows by sorted `bin_id` as `summarize_counts_for_bins` orders its bins.
    """
    kept = df_gene_snp.dropna(subset=["block_id", "bin_id"])
    blocks = kept.groupby("block_id", sort=True).agg(chrom=("CHR", "first"), bin=("bin_id", "first"))
    phase = np.asarray(phase_indicator)[blocks.index.to_numpy(dtype=np.int64)]
    chrom = blocks["chrom"].to_numpy()
    switch = np.r_[0, (phase[1:] != phase[:-1]) & (chrom[1:] == chrom[:-1])]
    rows = blocks["bin"].rank(method="dense").to_numpy(dtype=np.int64) - 1
    per_bin = np.bincount(rows, weights=switch).astype(np.int64)
    contig = pd.Series(chrom).groupby(rows).first().to_numpy()
    return pd.Series(per_bin).groupby(contig, sort=False).cumsum().to_numpy(dtype=np.int64)
