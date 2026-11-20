"""The initial labelling of each stage: `phasing`, the clones phasing starts from, and `rdr`, each BAF clone split
into RDR clones; by `phasing.clone_init_type` and `hmrf.rdr_clone_init_type`, `rectangle` or `median_bisection`."""

import time
from typing import Any, NamedTuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph
from scipy.spatial import cKDTree

from cnamaste.config import start_time
from cnamaste.logger import get_logger
from cnamaste.platforms.visium import bin_pitch
from cnamaste.spatio_genomic_counts import get_grouped_sums, get_incidence

logger = get_logger(__name__, start_time=start_time)

BISECTION_MAX_DRAWS = 100
"""The draws of (theta, quantile) `get_bisection_labelling` makes for a cut that leaves both sides contiguous: a
half-plane can cut a lattice in a sliver whose spots are not neighbours."""


START_MIN_SNP_UMIS_PER_SEGMENT = 20
"""A start clone's SNP UMIs per segment below which it scores no BAF distinction (`get_baf_distinction`), as
`get_rdr_labelling` keeps a BAF clone whole below 20 x n_segments."""


def get_start_min_snp_umis_per_segment() -> int:
    """Return a scored start clone's least SNP UMIs per segment, 20."""
    return START_MIN_SNP_UMIS_PER_SEGMENT


class SpotPositions(NamedTuple):
    """Each spot's positions: `coords` (n_spots, 2) pixel, `physical` (n_spots, 2) um, `sample_ids` its slice."""

    coords: np.ndarray
    physical: np.ndarray
    sample_ids: np.ndarray


def get_initial_labelling(
    counts: Any, positions: SpotPositions, config: Any, *, stage: str, rng: np.random.Generator
) -> np.ndarray:
    """The initial labelling of `stage`.

    Args:
        counts: a `SpatioGenomicLabelledMultiCounts`; at `rdr`, its `clone` labelling the BAF clones 0..N-1.
        positions: the spots' positions.
        config: the run's.
        stage: `phasing` or `rdr`.
        rng: the generator the bisection draws from, seeded by `hmrf.random_seed`.

    Returns:
        (n_spots,) each spot's label; at `phasing` a `rectangle` cell may hold no spot.

    Raises:
        ValueError: for an unknown stage.
    """
    if stage == "phasing":
        return get_phasing_labelling(counts, positions, config, rng)
    if stage == "rdr":
        return get_rdr_labelling(counts, positions, config, rng)
    raise ValueError(f"{stage!r} is not one of ('phasing', 'rdr').")


def get_phasing_labelling(counts: Any, positions: SpotPositions, config: Any, rng: np.random.Generator) -> np.ndarray:
    """The phasing start by `phasing.clone_init_type`: `rectangle`, each slice's npart_phasing^2 grid on its pixel
    `coords`; or `median_bisection`, npart_phasing^2 x n_slices clones on the `physical` positions, the one of
    `phasing.n_start_candidates` of the strongest BAF distinction (`get_best_start`).

    Args:
        counts: a `SpatioGenomicLabelledMultiCounts`, its SNP UMIs per segment.
        positions: the spots' positions.
        config: the run's.
        rng: the generator each slice's stream is spawned from.

    Returns:
        (n_spots,) each spot's clone, a slice's npart_phasing^2 labels after the previous slice's.

    Raises:
        RuntimeError: where a slice holds no spot.
    """
    coords, physical, sample_ids = positions
    n_part = config.phasing.npart_phasing
    n_slices = 1 + int(np.max(sample_ids))
    if config.phasing.clone_init_type != "rectangle" and config.phasing.n_start_candidates == 1:
        return get_slice_bisection_labelling(
            physical, sample_ids, n_part**2 * n_slices, rng, config.hmrf.bisection_quantile_jitter
        )
    if config.phasing.clone_init_type != "rectangle":
        candidates = [
            get_slice_bisection_labelling(
                physical, sample_ids, n_part**2 * n_slices, stream, config.hmrf.bisection_quantile_jitter
            )
            for stream in rng.spawn(config.phasing.n_start_candidates)
        ]
        return get_best_start(counts, candidates, config.phasing.baf.eps)
    labelling = np.full(len(sample_ids), -1, dtype=int)
    for slice_id in range(n_slices):
        index = np.where(sample_ids == slice_id)[0]
        if len(index) == 0:
            raise RuntimeError(f"Expected at least one spot in slice {slice_id}.")
        labelling[index] = slice_id * n_part**2 + get_rectangle_labelling(coords[index, :], n_part, n_part, None)
    logger.info(f"Initialized {n_part**2 * n_slices} clones given x_part,y_part={n_part},{n_part}.")
    return labelling


def get_baf_distinction(alt: np.ndarray, total: np.ndarray, min_snp_umis: float, eps: float) -> float:
    """How far a labelling's clones are apart in BAF, phase-free and blind to normal: per clone and segment the folded
    BAF of the clone's pooled SNP UMIs beyond the deadband, `max(0, |alt / total - 1/2| - eps)`, continuous in the
    BAF so a clone at the band's edge does not jump; per segment its SNP-UMI-weighted variance between clones; summed
    over segments. A clone under `min_snp_umis` in all, or a segment it does not cover, weighs nothing.

    Args:
        alt: (n_clones, n_segments) pooled haplotype-A SNP UMIs.
        total: (n_clones, n_segments) pooled SNP UMIs.
        min_snp_umis: a clone's least SNP UMIs to be scored.
        eps: the deadband about 1/2, `phasing.baf.eps`, the band in which a clone casts no phase vote.

    Returns:
        The distinction, 0 where fewer than two clones are scored or every clone is within the band.
    """
    weight = np.where(total.sum(axis=1, keepdims=True) >= min_snp_umis, total, 0.0).astype(np.float64)
    baf = np.divide(alt, total, out=np.full(total.shape, 0.5), where=total > 0)
    folded = np.maximum(np.abs(baf - 0.5) - eps, 0.0)
    held = weight.sum(axis=0)
    mean = np.divide((weight * folded).sum(axis=0), held, out=np.zeros(held.shape), where=held > 0)
    spread = np.divide((weight * (folded - mean) ** 2).sum(axis=0), held, out=np.zeros(held.shape), where=held > 0)
    return float(spread.sum())


def get_best_start(counts: Any, candidates: list[np.ndarray], eps: float) -> np.ndarray:
    """The candidate labelling of the largest `get_baf_distinction` on `counts`' SNP UMIs per segment, the first on
    ties; each candidate's score, the kept one and the scoring's seconds logged.

    Args:
        counts: a `SpatioGenomicLabelledMultiCounts`.
        candidates: (n_spots,) labellings, clones 0..n - 1.
        eps: the BAF deadband, `phasing.baf.eps`.

    Returns:
        The kept labelling.
    """
    start = time.perf_counter()
    alt, total = (sp.csr_matrix(c.values).T.tocsr() for c in (counts.alt_snp_umis, counts.total_snp_umis))
    floor = get_start_min_snp_umis_per_segment() * total.shape[0]
    scores = []
    for labelling in candidates:
        n_clones = int(labelling.max()) + 1
        incidence = get_incidence(np.arange(labelling.size), labelling, (labelling.size, n_clones), np.int64)
        pooled_alt, pooled_total = (get_grouped_sums(x, incidence).toarray().T for x in (alt, total))
        scores.append(get_baf_distinction(pooled_alt, pooled_total, floor, eps))
    best = int(np.argmax(scores))
    logger.info(
        f"Phasing start: candidate {best} of {len(candidates)} kept, BAF distinction {scores[best]:.6g}; every "
        f"candidate's: {[round(s, 6) for s in scores]}; scored in {time.perf_counter() - start:.2f} s."
    )
    return candidates[best]


def get_rdr_labelling(counts: Any, positions: SpotPositions, config: Any, rng: np.random.Generator) -> np.ndarray:
    """Split each BAF clone into `hmrf.n_clones_rdr` RDR clones, or keep it whole where it holds fewer than 20 x
    n_segments SNP UMIs, by `hmrf.rdr_clone_init_type`: `rectangle`, `get_block_labelling` on the pixel `coords`
    seeded by `hmrf.random_seed`; or `median_bisection` on the `physical` positions, one stream per BAF clone spawned
    from `rng`, itself seeded by `hmrf.random_seed`.

    Args:
        counts: a `SpatioGenomicLabelledMultiCounts`, its `clone` labelling the BAF clones 0..N-1.
        positions: the spots' positions.
        config: the run's.
        rng: the generator each BAF clone's stream is spawned from.

    Returns:
        (n_spots,) each spot's RDR clone, a BAF clone's labels after the previous one's.

    Raises:
        ValueError: where the BAF clone labels are not 0..N-1.
    """
    baf_labelling = counts.labellings["clone"]
    n_baf_clones = len(np.unique(baf_labelling))
    if not np.array_equal(np.unique(baf_labelling), np.arange(n_baf_clones)):
        raise ValueError(f"BAF clone labels are not 0..{n_baf_clones - 1}: {np.unique(baf_labelling)}.")

    splits_per_baf = []
    for bafc in range(n_baf_clones):
        idx_spots = np.flatnonzero(baf_labelling == bafc)
        sufficient_snp_umi = counts.total_snp_umis.spots(idx_spots).sum() >= 20 * counts.n_segments  # MAGIC
        splits_per_baf.append(config.hmrf.n_clones_rdr if sufficient_snp_umi else 1)

    logger.info(
        f"Global hmrf will optimize {int(sum(splits_per_baf))} rdr-refined clones given {n_baf_clones} baf-identified "
        "clones."
    )

    labelling = np.zeros(len(baf_labelling), dtype=np.int32)
    streams = rng.spawn(n_baf_clones)
    offset = 0
    for bafc, n_splits in enumerate(splits_per_baf):
        idx_spots = np.flatnonzero(baf_labelling == bafc)
        if config.hmrf.rdr_clone_init_type == "rectangle":
            parts = get_block_labelling(positions.coords[idx_spots], n_splits, config.hmrf.random_seed)
        else:
            parts = get_bisection_labelling(
                positions.physical[idx_spots], n_splits, streams[bafc], config.hmrf.bisection_quantile_jitter
            )
        labelling[idx_spots] = offset + parts
        offset += n_splits
    return labelling


def get_rectangle_labelling(
    coords: np.ndarray, x_part: int, y_part: int, rng: np.random.Generator | None
) -> np.ndarray:
    """Cut spots on an `x_part` by `y_part` grid over their x and y ranges, CalicoST's `rectangle_partition`.

    Args:
        coords: (n_spots, 2) positions.
        x_part: columns of the grid.
        y_part: rows of the grid.
        rng: None for equal-width cells, or the generator of uniform random cell edges.

    Returns:
        (n_spots,) each spot's cell, column-major; a cell may hold no spot.
    """
    if rng is not None:
        x_edges = np.sort(rng.uniform(0, 1, x_part))
        x_edges[-1] = 1.01
    else:
        x_edges = np.linspace(0, 1, 1 + x_part)
        x_edges[-1] += 0.01
        x_edges = x_edges[1:]

    x_range = [np.min(coords[:, 0]), np.max(coords[:, 0])]
    x_digit = np.digitize(coords[:, 0], x_range[0] + (x_range[1] - x_range[0]) * x_edges, right=True)

    if rng is not None:
        y_edges = np.sort(rng.uniform(0, 1, y_part))
        y_edges[-1] = 1.01
    else:
        y_edges = np.linspace(0, 1, y_part + 1)
        y_edges[-1] += 0.01
        y_edges = y_edges[1:]

    y_range = [np.min(coords[:, 1]), np.max(coords[:, 1])]
    y_digit = np.digitize(coords[:, 1], y_range[0] + (y_range[1] - y_range[0]) * y_edges, right=True)

    labelling = np.full(coords.shape[0], -1)
    for x_id in range(x_part):
        for y_id in range(y_part):
            labelling[(x_digit == x_id) & (y_digit == y_id)] = x_id * y_part + y_id
    return labelling


def admits_assignment(block_sizes: Any, n_clones: int, floor: float) -> bool:
    """Whether some assignment of blocks to clones gives every clone more than `floor` spots.

    `get_block_labelling`'s loop reaches every surjective map, so it returns exactly when this is True. A depth-first
    search, largest block first; equal-total clones are interchangeable.

    Args:
        block_sizes: spots per block.
        n_clones: the clones.
        floor: the spot floor.

    Returns:
        Whether an assignment exists.
    """
    sizes = sorted((int(size) for size in block_sizes), reverse=True)
    need = int(np.floor(floor)) + 1
    remaining = [*np.cumsum(sizes[::-1])[::-1].tolist(), 0]
    totals = [0] * n_clones

    def place(block: int) -> bool:
        deficits = [need - total for total in totals if total < need]
        if not deficits:
            return True
        if block == len(sizes) or len(deficits) > len(sizes) - block:
            return False
        if sum(deficits) > remaining[block]:
            return False
        tried = set()
        for clone in range(n_clones):
            if totals[clone] in tried:
                continue
            tried.add(totals[clone])
            totals[clone] += sizes[block]
            if place(block + 1):
                return True
            totals[clone] -= sizes[block]
        return False

    return place(0)


def get_banded_labelling(coords: np.ndarray, n_clones: int) -> np.ndarray:
    """Split spots into `n_clones` equal-count bands along the axis with the most distinct values.

    Ordered along that axis, ties by the other then by index; every band passes the 20% test whenever
    n_spots >= n_clones.

    Args:
        coords: (n_spots, 2) positions.
        n_clones: the bands.

    Returns:
        (n_spots,) each spot's band.
    """
    n_spots = len(coords)
    axis = int(np.argmax([np.unique(coords[:, a]).size for a in (0, 1)]))
    order = np.lexsort((np.arange(n_spots), coords[:, 1 - axis], coords[:, axis]))
    labelling = np.empty(n_spots, dtype=int)
    labelling[order] = np.arange(n_spots) * n_clones // n_spots
    return labelling


def get_block_labelling(coords: np.ndarray, n_clones: int, random_state: int) -> np.ndarray:
    """Split spots into `n_clones` by a ceil(sqrt(n_clones))^2 grid of Dirichlet(10)-sized blocks over the 5-95%
    range, the blocks drawn to clones until each clone holds more than 20% of an equal share; equal-count bands
    (`get_banded_labelling`) where no assignment of the blocks can (`admits_assignment`).

    Args:
        coords: (n_spots, 2) positions.
        n_clones: the clones.
        random_state: the seed of CalicoST's `RandomState` draws.

    Returns:
        (n_spots,) each spot's clone.
    """
    # NB CalicoST's legacy draws, a second source beside the bisection's Generator: NumPy's global stream is untouched.
    draws = np.random.RandomState(random_state)

    logger.info(f"Solving for non-contiguous clone initialization for {n_clones} clones.")
    if n_clones == 1:
        return np.zeros(len(coords), dtype=int)

    grid = int(np.ceil(np.sqrt(n_clones)))
    digits = []
    for axis in (0, 1):
        widths = draws.dirichlet(np.ones(grid) * 10)
        widths[-1] += 1e-4
        axis_range = [np.percentile(coords[:, axis], 5), np.percentile(coords[:, axis], 95)]
        boundary = axis_range[0] + (axis_range[1] - axis_range[0]) * np.cumsum(widths)
        boundary[-1] = np.max(coords[:, axis]) + 1
        digits.append(np.digitize(coords[:, axis], boundary, right=True))
    block_id = digits[0] * grid + digits[1]

    # NB on dev (`07b82e92`) blocks of [194, 3, 77, 23] spots admit no 4 clones above the floor: banded instead.
    floor = 0.2 * coords.shape[0] / n_clones  # MAGIC
    if not admits_assignment(np.bincount(block_id, minlength=grid**2), n_clones, floor):
        logger.info(f"Rectangular blocks admit no {n_clones} clones above {floor:.2f} spots; banded.")
        return get_banded_labelling(coords, n_clones)

    while True:
        block_clone = draws.randint(low=0, high=n_clones, size=grid**2)
        # NB a clone given no block takes the first block of the clone given most.
        while len(np.unique(block_clone)) < n_clones:
            counts = np.bincount(block_clone, minlength=n_clones)
            block_clone[np.where(block_clone == np.argmax(counts))[0][0]] = np.where(counts == 0)[0][0]
        labelling = block_clone[block_id]
        if np.min(np.bincount(labelling, minlength=n_clones)) > floor:
            break

    logger.info(f"Solved for clone initialization for {n_clones} clones.")
    return labelling


def count_components(neighbours: Any, index: np.ndarray) -> int:
    """The connected components of the spots `index` in the `neighbours` graph restricted to them."""
    return int(scipy.sparse.csgraph.connected_components(neighbours[index][:, index], directed=False)[0])


def get_bisection_labelling(
    coords: np.ndarray, n_parts: int, rng: np.random.Generator, quantile_jitter: float
) -> np.ndarray:
    """Cut spots into `n_parts` convex, contiguous parts of near-equal count by rotated median bisection.

    The piece with the most spots still to be divided, of k parts, is cut along a direction theta ~ U[0, pi) at the
    quantile k1 / k + U[-delta, delta] of its spots' projection on it, k1 = k // 2, delta `quantile_jitter`
    (`hmrf.bisection_quantile_jitter`), until every piece is one part. A draw whose sides hold more components than
    the piece and one is redrawn, up to `BISECTION_MAX_DRAWS`; neighbours are spots within 1.5 pitches
    (`platforms.visium.bin_pitch`), a Visium spot's 6 or a Visium HD bin's 8.

    Args:
        coords: (n_spots, 2) positions.
        n_parts: the parts, at most n_spots.
        rng: the generator drawn from.
        quantile_jitter: delta.

    Returns:
        (n_spots,) each spot's part, 0..n_parts - 1, in the order the cuts leave them.

    Raises:
        ValueError: where there are fewer spots than parts.
    """
    coords = np.asarray(coords, dtype=np.float64)
    if not 1 <= n_parts <= len(coords):
        raise ValueError(f"{len(coords)} spots cannot be cut into {n_parts} parts.")
    neighbours = None
    if n_parts > 1:
        neighbours = (
            cKDTree(coords)
            .sparse_distance_matrix(cKDTree(coords), 1.5 * bin_pitch(coords), output_type="coo_matrix")
            .tocsr()
        )  # MAGIC 1.5 pitches: between the square's sqrt(2) and the hexagon's sqrt(3).
    pieces = [(np.arange(len(coords)), n_parts)]
    while any(k > 1 for _, k in pieces):
        at = max((i for i, (_, k) in enumerate(pieces) if k > 1), key=lambda i: len(pieces[i][0]))
        index, k = pieces.pop(at)
        k_lower = k // 2
        allowed = count_components(neighbours, index) + 1
        best: tuple[int, tuple[np.ndarray, np.ndarray]] | None = None
        for _ in range(BISECTION_MAX_DRAWS):
            theta = rng.uniform(0.0, np.pi)
            quantile = k_lower / k + rng.uniform(-quantile_jitter, quantile_jitter)
            # NB a cut at a rank, not a value, so lattice ties along theta do not move the counts.
            order = np.argsort(coords[index] @ np.array([np.cos(theta), np.sin(theta)]), kind="stable")
            n_lower = int(np.clip(np.rint(quantile * len(index)), k_lower, len(index) - (k - k_lower)))
            sides = (index[order[:n_lower]], index[order[n_lower:]])
            components = count_components(neighbours, sides[0]) + count_components(neighbours, sides[1])
            if best is None or components < best[0]:
                best = (components, sides)
            if components <= allowed:
                break
        assert best is not None
        if best[0] > allowed:
            logger.warning(f"No cut of {len(index)} spots in {BISECTION_MAX_DRAWS} draws keeps both sides contiguous.")
        pieces[at:at] = [(best[1][0], k_lower), (best[1][1], k - k_lower)]
    labelling = np.empty(len(coords), dtype=np.int64)
    for part, (index, _) in enumerate(pieces):
        labelling[index] = part
    return labelling


def get_parts_per_slice(n_spots_per_slice: Any, n_parts: int) -> np.ndarray:
    """`n_parts` allocated to slices in proportion to their spot counts by largest remainder, at least 1 each.

    Args:
        n_spots_per_slice: (n_slices,) spots per slice.
        n_parts: the parts, at least n_slices.

    Returns:
        (n_slices,) parts per slice, summing to `n_parts`.

    Raises:
        ValueError: where there are fewer parts than slices.
    """
    n_spots_per_slice = np.asarray(n_spots_per_slice, dtype=np.float64)
    if n_parts < len(n_spots_per_slice):
        raise ValueError(f"{n_parts} parts cannot give each of {len(n_spots_per_slice)} slices one.")
    quota = n_parts * n_spots_per_slice / n_spots_per_slice.sum()
    parts = np.maximum(np.floor(quota).astype(np.int64), 1)
    # NB stable sorts, so a tie in remainder goes to the earlier slice.
    while parts.sum() < n_parts:
        parts[np.argsort(-(quota - parts), kind="stable")[0]] += 1
    while parts.sum() > n_parts:
        parts[np.argsort(-np.where(parts > 1, parts - quota, -np.inf), kind="stable")[0]] -= 1
    return parts


def get_slice_bisection_labelling(
    coords: np.ndarray, sample_ids: np.ndarray, n_clones: int, rng: np.random.Generator, quantile_jitter: float
) -> np.ndarray:
    """Each slice cut by `get_bisection_labelling` into its share of `n_clones` (`get_parts_per_slice`), with one
    stream per slice spawned from `rng`: a slice's parts depend on its own spots alone.

    Args:
        coords: (n_spots, 2) positions.
        sample_ids: each spot's slice, 0..n_slices - 1.
        n_clones: the clones.
        rng: the generator each slice's stream is spawned from.
        quantile_jitter: `get_bisection_labelling`'s delta.

    Returns:
        (n_spots,) each spot's clone, 0..n_clones - 1, slice by slice.
    """
    sample_ids = np.asarray(sample_ids)
    n_spots_per_slice = np.bincount(sample_ids)
    parts_per_slice = get_parts_per_slice(n_spots_per_slice, n_clones)
    streams = rng.spawn(len(n_spots_per_slice))
    labelling = np.empty(len(sample_ids), dtype=np.int64)
    offset = 0
    for slice_id, n_parts in enumerate(parts_per_slice):
        index = np.flatnonzero(sample_ids == slice_id)
        labelling[index] = offset + get_bisection_labelling(coords[index], n_parts, streams[slice_id], quantile_jitter)
        offset += n_parts
    logger.info(
        f"Cut {len(parts_per_slice)} slices into {parts_per_slice.tolist()} clones by rotated median bisection."
    )
    return labelling
