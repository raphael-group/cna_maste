"""The validation figure's panels, scored per spot on each run's own segments, and the table `run_sim_qa` writes.

- segment:  one row of the caller's `cnv_<ploidy>_seglevel.tsv`; each counts once, irrespective of length.
- ab_pair:  a segment's allele copies `(A, B)` in one clone, coded `A * 1_000 + B`, and compared phase-free.
- cell:  one kept spot on one segment that a true segment overlaps; its true AB pair is its true clone's there, its
  called AB pair its fitted clone's, so every kept spot is scored and no clone is matched to another.
- true AB pair:  per true clone, that of the true segment overlapping the segment most (`segment_true_ab_pairs`).
- Cells are counted, never realized: per `(true, fitted)` clone pair, its spots times its segments (`confusion`).
- Panels: CNA sensitivity, normal specificity, exact altered (N-CNA recall), and clone ARI, `int_clone_ari` for
  cnamaste and `clone_ari` for CalicoST; each `(RunEstimate, Truth) -> float`.
- `spot_metrics` scores the rest of a run; `cnamaste.sandbox.metrics` the clone-bins `run_sim_qa` scored before.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from cnamaste.config import start_time
from cnamaste.io import read_text_table
from cnamaste.logger import get_logger
from cnamaste.results import get_output_path

# NB `Truth` holds a sample's truth: `barcodes`, `coords`, `labels` (each spot's true clone, 0 normal),
#    `clones` (their names), `profile` (`truth_acn_profile.tsv`, each clone's `(A, B)` per segment) and `n_clones`.
from cnamaste.truth import Truth, read_truth

logger = get_logger("cnamaste.qa.metrics", start_time=start_time)

ABPair = tuple[int, int]

NEUTRAL = 1_001
"""The AB pair `(1, 1)`, as `A * 1_000 + B`."""


def overlap(true_labels: np.ndarray, fitted: np.ndarray, n_true: int, n_fitted: int) -> np.ndarray:
    """The contingency table of two labellings: `counts[i, j]`, the kept spots in true clone `i` and fitted clone `j`.

    Args:
        true_labels: labels per spot.
        fitted: labels per spot.
        n_true: the label counts.
        n_fitted: the label counts.

    Returns:
        `(n_true, n_fitted)` counts.
    """
    counts = np.zeros((n_true, n_fitted), dtype=np.int64)

    # NB for each spot s, counts[true_labels[s], fitted[s]] += 1. `np.add.at` is unbuffered: an index pair that
    #    repeats across spots adds once per spot, where `counts[true_labels, fitted] += 1` would add once in all.
    np.add.at(counts, (true_labels, fitted), 1)
    return counts


# NB the run's tables and the truth hold phased `(A, B)`: A and B name haplotypes, and which is which follows the
#    phasing. `phase_free` codes `(A, B)` and `(B, A)` alike, as `(minor, major)`; every comparison in a panel is of
#    phase-free codes, and `decode_ab_pair` reads a code back as the AB pair it holds.
def phase_free(codes: np.ndarray) -> np.ndarray:
    """`A * 1000 + B` codes as `(minor, major)`: i.e. `(A, B)` and `(B, A)` coded alike.

    Args:
        codes: AB pair codes.

    Returns:
        Phase-free codes.
    """
    major, minor = codes // 1_000, codes % 1_000
    return np.asarray(np.minimum(major, minor) * 1_000 + np.maximum(major, minor))


def decode_ab_pair(code: int) -> ABPair:
    """`A * 1_000 + B` as `(A, B)`: `phase_free`'s input read back, as `(minor, major)` where `phase_free` coded it."""
    return int(code // 1_000), int(code % 1_000)


@dataclass(frozen=True)
class RunEstimate:
    """A run's fitted clones and copies.

    Args:
        clones: each documented at its attribute.
        int_clones: each documented at its attribute.
        seglevel: each documented at its attribute.
    """

    # NB clone label per barcode (indexed by barcode).
    clones: pd.Series

    # NB integer clone per barcode: the run's `clone_labels_int.tsv`, clones whose `(A, B)` agree merged.
    int_clones: pd.Series

    # NB the run's `cnv_<ploidy>_seglevel.tsv`.
    seglevel: pd.DataFrame


def read_labels(path: Path) -> pd.Series:
    """A run's per-spot label table as label per barcode.

    Args:
        path: a `clone_labels*.tsv`, read plain or `.gz`.

    Returns:
        `clone_label` indexed by `barcode`.

    Raises:
        FileNotFoundError: where neither exists.
    """
    found = read_text_table(path, sep="\t", comment="#")
    return pd.Series(found["clone_label"].to_numpy(), index=found["barcode"].astype(str))


def read_estimate(run_dir: Path) -> RunEstimate:
    """The run's `clone_labels.tsv`, `clone_labels_int.tsv` and `cnv_<ploidy>_seglevel.tsv` (`results.get_output_path`; per-spot
    tables plain or `.gz`).

    Args:
        run_dir: the run's directory.

    Returns:
        The `RunEstimate`.

    Raises:
        FileNotFoundError: where any of the three is missing, a run from before `clone_labels_int.tsv` among them.
    """
    run_dir = Path(run_dir)
    clones = read_labels(run_dir / "clone_labels.tsv")
    int_clones = read_labels(run_dir / "clone_labels_int.tsv")
    seglevel = pd.read_csv(get_output_path(run_dir, "seglevel"), sep="\t")
    return RunEstimate(clones=clones, int_clones=int_clones, seglevel=seglevel)


def fitted_at_barcode(estimate: RunEstimate, truth: Truth) -> np.ndarray:
    """The estimate's clones at the truth's barcodes; -1 where the run dropped a spot."""
    return estimate.clones.reindex(truth.barcodes, fill_value=-1).to_numpy()


def merged_at_barcode(estimate: RunEstimate, truth: Truth) -> np.ndarray:
    """The estimate's integer clones at the truth's barcodes; -1 where the run dropped a spot.
    Each names its group by a member clone, whose copies it calls."""
    return estimate.int_clones.reindex(truth.barcodes, fill_value=-1).to_numpy()


def segment_true_ab_pairs(seglevel: Any, truth: Truth) -> np.ndarray:
    """Each segment's true AB pair per true clone: that of the true segment overlapping it most.

    Args:
        seglevel: the run's `cnv_<ploidy>_seglevel` table.
        truth: the truth.

    Returns:
        `(segments, true clones)` phase-free codes, -1 on a segment no true segment overlaps.
    """
    profile = truth.profile
    chrom = profile["chr"].astype(str).str.removeprefix("chr").to_numpy()
    starts, ends = profile["start"].to_numpy(), profile["end"].to_numpy()
    contigs = seglevel["CHR"].astype(str).str.removeprefix("chr").to_numpy()
    lo, hi = seglevel["START"].to_numpy(), seglevel["END"].to_numpy()
    best = np.full(len(seglevel), -1)
    for i in range(len(seglevel)):
        same_contig = np.flatnonzero(chrom == contigs[i])

        # NB the length of each true segment's intersection with segment i, negative where they are disjoint: the
        #    true segment with the longest positive one gives segment i its AB pair.
        shared = np.minimum(ends[same_contig], hi[i]) - np.maximum(starts[same_contig], lo[i])
        if same_contig.size and shared.max() > 0:
            # NB ties to the first in the profile's order, the leftmost on a sorted profile
            best[i] = same_contig[int(np.argmax(shared))]
    codes = np.full((len(seglevel), truth.n_clones), -1, dtype=np.int64)
    found = best >= 0
    for c, clone in enumerate(truth.clones):
        a, b = profile[f"{clone}_A_copy"].to_numpy(), profile[f"{clone}_B_copy"].to_numpy()
        codes[found, c] = phase_free(a[best[found]] * 1_000 + b[best[found]])
    return codes


def segment_called_ab_pairs(seglevel: Any) -> np.ndarray:
    """Each segment's called AB pair per fitted clone.

    Args:
        seglevel: the run's `cnv_<ploidy>_seglevel` table, `clone<k> A`, `clone<k> B` per fitted clone `k`.

    Returns:
        `(segments, fitted clones)` phase-free codes.
    """
    n = sum(1 for c in seglevel.columns if str(c).startswith("clone") and str(c).endswith(" A"))
    return np.stack(
        [
            phase_free(
                seglevel[f"clone{k} A"].to_numpy(dtype=np.int64) * 1_000
                + seglevel[f"clone{k} B"].to_numpy(dtype=np.int64)
            )
            for k in range(n)
        ],
        axis=1,
    )


def scored_segments(seglevel: Any, truth: Truth) -> tuple[np.ndarray, np.ndarray]:
    """The scored segments' true and called AB pairs: the segments some true segment overlaps.

    Args:
        seglevel: the run's table.
        truth: the truth.

    Returns:
        `(segments, true clones)` and `(segments, fitted clones)` phase-free codes, the segments both score.
    """
    true_ab, called_ab = segment_true_ab_pairs(seglevel, truth), segment_called_ab_pairs(seglevel)
    scored = (true_ab[:, 0] >= 0) & (called_ab[:, 0] >= 0)
    return true_ab[scored], called_ab[scored]


def confusion(seglevel: Any, truth: Truth, fitted: np.ndarray) -> dict[tuple[ABPair, ABPair], int]:
    """Cells by `(true, called)` phase-free AB pair, the diagonal included.

    Args:
        seglevel: the run's table.
        truth: the truth.
        fitted: each truth barcode's fitted label, a `clone<k>` id of `seglevel`, -1 where the run dropped the spot.

    Returns:
        `(true AB pair, called AB pair)` -> cells, sorted by true then called AB pair.
    """
    return cells_by_ab_pair(*scored_segments(seglevel, truth), truth, fitted)


def cells_by_ab_pair(
    true_ab: np.ndarray, called_ab: np.ndarray, truth: Truth, fitted: np.ndarray
) -> dict[tuple[ABPair, ABPair], int]:
    """Cells by `(true, called)` phase-free AB pair over the scored segments, each spot on its clones' segments.

    Args:
        true_ab: `(segments, true clones)` and `called_ab` `(segments, fitted clones)` phase-free codes, as
            `scored_segments` gives them.
        truth: the truth.
        fitted: as `confusion`'s.

    Returns:
        As `confusion`.
    """
    kept = fitted >= 0
    spots = overlap(truth.labels[kept], fitted[kept], truth.n_clones, called_ab.shape[1])
    counts: dict[tuple[int, int], int] = {}
    for c, k in zip(*np.nonzero(spots), strict=True):
        codes, n = np.unique(np.stack([true_ab[:, c], called_ab[:, k]], axis=1), axis=0, return_counts=True)
        for (t, f), segments in zip(codes.tolist(), n.tolist(), strict=True):
            counts[(t, f)] = counts.get((t, f), 0) + int(spots[c, k]) * segments
    return {(decode_ab_pair(t), decode_ab_pair(f)): n for (t, f), n in sorted(counts.items())}


def get_cell_confusion(estimate: RunEstimate, truth: Truth) -> dict[tuple[ABPair, ABPair], int]:
    """`confusion` of the run's cells, each spot called by its fitted clone."""
    return confusion(estimate.seglevel, truth, fitted_at_barcode(estimate, truth))


def sensitivity(estimate: RunEstimate, truth: Truth) -> float:
    """Truly altered cells decoded altered, as any AB pair but (1, 1): `1 - false_negative / altered`;
    NaN where the truth has none."""
    counts = get_cell_confusion(estimate, truth)
    altered = sum(n for (t, _), n in counts.items() if t != (1, 1))
    missed = sum(n for (t, f), n in counts.items() if t != (1, 1) and f == (1, 1))
    return 1.0 - missed / altered if altered else float("nan")


def specificity(estimate: RunEstimate, truth: Truth) -> float:
    """Truly (1, 1) cells decoded (1, 1): `1 - false_positive / neutral`; NaN where the truth has none."""
    counts = get_cell_confusion(estimate, truth)
    neutral = sum(n for (t, _), n in counts.items() if t == (1, 1))
    called = sum(n for (t, f), n in counts.items() if t == (1, 1) and f != (1, 1))
    return 1.0 - called / neutral if neutral else float("nan")


def exact_altered(estimate: RunEstimate, truth: Truth) -> float:
    """Truly altered cells decoded as their true AB pair, phase-free: N-CNA recall; NaN where the truth has none."""
    counts = get_cell_confusion(estimate, truth)
    altered = sum(n for (t, _), n in counts.items() if t != (1, 1))
    return sum(n for (t, f), n in counts.items() if t != (1, 1) and t == f) / altered if altered else float("nan")


def clone_ari(estimate: RunEstimate, truth: Truth) -> float:
    """Adjusted Rand index of the fitted clones against the true ones, over the kept spots: CalicoST's ARI panel."""
    fitted = fitted_at_barcode(estimate, truth)
    kept = fitted >= 0
    return float(adjusted_rand_score(truth.labels[kept], fitted[kept]))


def int_clone_ari(estimate: RunEstimate, truth: Truth) -> float:
    """Adjusted Rand index of the integer clones against the true ones, over the spots the clone labels kept:
    cnamaste's ARI panel."""
    kept = fitted_at_barcode(estimate, truth) >= 0
    return float(adjusted_rand_score(truth.labels[kept], merged_at_barcode(estimate, truth)[kept]))


PANEL_METRICS: dict[str, Callable[[RunEstimate, Truth], float]] = {
    "clone_ari": clone_ari,
    "int_clone_ari": int_clone_ari,
    "exact_altered": exact_altered,
    "sensitivity": sensitivity,
    "specificity": specificity,
}
"""The four panels' columns by name, in `all_sample_metrics.tsv`'s order: the ARI panel is two, one per method."""


def metrics(estimate: RunEstimate, truth: Truth) -> dict[str, float]:
    """Every entry of `PANEL_METRICS` on `estimate` against `truth`.

    Args:
        estimate: the run.
        truth: the truth.

    Returns:
        Metric name -> value.
    """
    return {name: metric(estimate, truth) for name, metric in PANEL_METRICS.items()}


def get_run_scores(run_dir: Path, sample: Path) -> dict[str, float]:
    """`metrics` of a run against its sample's truth, with the clones it inferred and the clones planted, normal
    counted in both.

    Args:
        run_dir: the run's directory.
        sample: the sample it ran on.

    Returns:
        Metric name -> value, then `n_clones` and `n_clones_true`.
    """
    estimate, truth = read_estimate(run_dir), read_truth(sample)
    counts = {"n_clones": float(estimate.clones.nunique()), "n_clones_true": float(truth.n_clones)}
    return metrics(estimate, truth) | counts


# --- the table: one row per run ----------------------------------------------

run_columns = ("sample_id", "method", "solver", "rectangle", "total_llf", "calicost_ploidy")
"""A scored run's identity: `solver` cnamaste's; `rectangle`, `total_llf` and `calicost_ploidy`, the copies' table,
CalicoST's."""

panel_columns = (*run_columns, *PANEL_METRICS)
"""`all_sample_metrics.tsv`'s columns: the run, then `PANEL_METRICS`, the validation figure's panels."""


def calc_metrics(
    sample_runs: Iterable[tuple[dict[str, Any], RunEstimate]],
    truth: Truth,
    measure: Callable[[RunEstimate, Truth], dict[str, Any]],
) -> list[dict[str, Any]]:
    """Score a sample's runs by `measure`, logging `<sample> cnamaste <solver>: Success` once a cnamaste run is read
    and scored.

    Args:
        sample_runs: `qa.sim_runs.runs`' `(identity, estimate)` pairs.
        truth: the sample's truth.
        measure: a row of scalars per run: `metrics`, the panels `run_sim_qa` tables, or `spot_metrics.table_row`.

    Returns:
        One row per run, its identity and `measure`'s scores, in `sample_runs`' order.

    Raises: as `sample_runs`.
    """
    rows = []
    for run, estimate in sample_runs:
        found = measure(estimate, truth)
        if run["method"] == "cnamaste":
            logger.info(f"{run['sample_id']} cnamaste {run['solver']}: Success")
        rows.append(run | found)
    return rows
