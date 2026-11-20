"""The oracle: one inference step run alone, every other step's answer the planted truth (`tests/test_oracle.py`).

- truth labelling:  each spot's planted clone in the run's spot order, normal 0 (`Truth.labels`).
- truth bin copies:  each bin's planted AB pair per true clone, phase-free (`qa.metrics.segment_true_ab_pairs`).
- oracle fit:  an `HmrfFit` whose states are the planted AB pairs on the lattice (`ab_decode.lattice`), each clone's
  path its planted pairs, dispersions and `log Z_c` a fit's; its field is `hmrf.get_labelling_unary`'s.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cnamaste.ab_decode.lattice import get_lattice_parameters
from cnamaste.hmrf import get_labelling_unary, get_nested_copy
from cnamaste.hmrf_fit import HmrfFit
from cnamaste.qa.metrics import RunEstimate, overlap, segment_true_ab_pairs
from cnamaste.spatio_genomic_counts import PseudobulkCounts
from cnamaste.truth import Truth


def get_truth_labelling(truth: Truth, barcodes: Any) -> np.ndarray:
    """Each spot's planted clone, normal 0, in the run's spot order.

    Args:
        truth: the planted truth.
        barcodes: the run's spots.

    Returns:
        `(n_spots,)` int64 labels.

    Raises:
        KeyError: where a spot has no planted clone.
    """
    return pd.Series(truth.labels, index=truth.barcodes).loc[np.asarray(barcodes).astype(str)].to_numpy(np.int64)


def get_bin_frame(counts: Any) -> pd.DataFrame:
    """The bins as `seglevel` names them: `CHR`, `START`, `END`, one row per bin.

    Args:
        counts: a `SpatioGenomicLabelledMultiCounts` or `PseudobulkCounts` with `contig`, `start`, `end`.

    Returns:
        The frame.
    """
    return pd.DataFrame({"CHR": counts.contig, "START": counts.start, "END": counts.end})


def get_truth_bin_copies(truth: Truth, counts: Any) -> np.ndarray:
    """Each bin's planted AB pair per true clone, phase-free: that of the true segment overlapping it most.

    Args:
        truth: the planted truth.
        counts: the run's counts, its bins.

    Returns:
        `(n_bins, n_clones)` `A * 1_000 + B` codes as `(minor, major)`, -1 on a bin no true segment overlaps.
    """
    return segment_true_ab_pairs(get_bin_frame(counts), truth)


def get_oracle_fit(fit: HmrfFit, codes: np.ndarray, pseudobulk: PseudobulkCounts) -> HmrfFit:
    """`fit` with its states the planted AB pairs and each clone's path its planted pairs: `log_mu` the lattice's,
    `p_binom` each pair's pooled B-allele fraction, dispersions and `log Z_c` `fit`'s.

    Args:
        fit: the clones' fit, one clone per true clone, its dispersions tied across states.
        codes: `(n_bins, n_clones)` `get_truth_bin_copies`, every bin covered.
        pseudobulk: the true clones' `PseudobulkCounts`; a clone's B-allele fraction under 1/2 orients its pair B
            minor, and each pair's pooled fraction is its `p_binom`.

    Returns:
        The copy; its `log_gamma` `fit`'s, unread by the field.
    """
    alt, total = np.asarray(pseudobulk.alt_snp_umis), np.asarray(pseudobulk.total_snp_umis)
    minor, major = codes.T // 1_000, codes.T % 1_000
    # NB a bin with no SNP UMI orients B major; its pair's fraction does not read it.
    b_minor = alt < total / 2.0
    pairs = np.stack([np.where(b_minor, major, minor), np.where(b_minor, minor, major)], axis=-1)
    states, path = np.unique(pairs.reshape(-1, 2), axis=0, return_inverse=True)
    # NB the lattice's p_binom is 0 at a lost allele, where a mis-phased B read costs the field without bound.
    b_fraction = np.bincount(path, weights=alt.ravel()) / np.bincount(path, weights=total.ravel())
    n_states = len(states)
    lattice = get_lattice_parameters(
        states, np.full(n_states, fit.parameters.alphas[0]), np.full(n_states, fit.parameters.taus[0])
    )
    oracle = get_nested_copy(fit)
    oracle.parameters = lattice._replace(log_startprob=np.full(n_states, -np.log(n_states)), p_binom=b_fraction)
    oracle["pred_cnv"] = path.reshape(pairs.shape[:2]).astype(np.int64)
    return oracle


def get_oracle_field(counts: Any, fit: HmrfFit) -> np.ndarray:
    """The labelling's field under `fit`'s paths, `hmrf.get_labelling_unary`.

    Args:
        counts: the run's `SpatioGenomicLabelledMultiCounts`.
        fit: an `HmrfFit`, `get_oracle_fit`'s.

    Returns:
        The `(n_spots, n_clones)` log-likelihood field.
    """
    return get_labelling_unary(counts, fit.parameters, fit.confounders, fit["pred_cnv"], phased=False)


def get_seglevel(counts: Any, copies: np.ndarray) -> pd.DataFrame:
    """Decoded copies as a `cnv_<ploidy>_seglevel` table, one segment per bin, as `qa.metrics` scores it.

    Args:
        counts: the run's counts, its bins.
        copies: `(n_bins, n_clones, 2)` `(A, B)`.

    Returns:
        `CHR`, `START`, `END`, then `clone<k> A`, `clone<k> B` per clone.
    """
    frame = get_bin_frame(counts)
    for k in range(copies.shape[1]):
        frame[f"clone{k} A"], frame[f"clone{k} B"] = copies[:, k, 0], copies[:, k, 1]
    return frame


def get_clone_estimate(truth: Truth, seglevel: pd.DataFrame, clone: int) -> RunEstimate:
    """A `RunEstimate` of the planted clones holding `clone`'s spots alone, so `qa.metrics`' cell panels score it.

    Args:
        truth: the planted truth.
        seglevel: the decoded copies, one column pair per true clone.
        clone: the true clone.

    Returns:
        The estimate.
    """
    kept = truth.labels == clone
    labels = pd.Series(truth.labels[kept], index=truth.barcodes[kept])
    return RunEstimate(clones=labels, int_clones=labels, seglevel=seglevel)


def get_start_on_truth(start: np.ndarray, labelling: np.ndarray, n_clones: int) -> np.ndarray:
    """A start labelling on the true clones: each start clone as the true clone most of its spots carry.

    Args:
        start: `(n_spots,)` the run's start clones.
        labelling: `(n_spots,)` `get_truth_labelling`.
        n_clones: the true clones.

    Returns:
        `(n_spots,)` labels in `[0, n_clones)`.
    """
    return overlap(start, labelling, int(start.max()) + 1, n_clones).argmax(axis=1)[start]


def get_breakpoint_recall(codes: np.ndarray, path: np.ndarray, contig: np.ndarray, tolerance: int) -> float:
    """The planted breakpoints with a decoded one within `tolerance` bins, on one contig.

    Args:
        codes: `(n_bins,)` one clone's planted codes.
        path: `(n_bins,)` its decoded states.
        contig: `(n_bins,)` each bin's contig.
        tolerance: bins.

    Returns:
        The recall; NaN where the clone has no planted breakpoint.
    """
    same = contig[1:] == contig[:-1]
    planted = np.flatnonzero((codes[1:] != codes[:-1]) & same) + 1
    decoded = np.flatnonzero((path[1:] != path[:-1]) & same) + 1
    if planted.size == 0:
        return float("nan")
    return float(np.mean([np.any(np.abs(decoded - b) <= tolerance) for b in planted]))


def get_clone_label_frame(truth: Truth, barcodes: Any) -> pd.DataFrame:
    """The planted clones as `annotation.clone_label` reads them: a `labels` column of clone names, in spot order.

    Args:
        truth: the planted truth.
        barcodes: the run's spots, the order `annotation.load_clone_labels` reads.

    Returns:
        The frame, indexed by barcode.
    """
    names = np.asarray(truth.clones)[get_truth_labelling(truth, barcodes)]
    return pd.DataFrame({"labels": names}, index=pd.Index(np.asarray(barcodes).astype(str), name="barcode"))
