"""Frozen copies of `spatial`'s clone starts before T- #102 PR 1, verbatim: `get_phasing_clones`,
`initialize_rdr_clone_refinement` and what they call."""

from typing import Any

import numpy as np
import scipy.sparse.csgraph
from scipy.spatial import cKDTree

from cnamaste.config import start_time
from cnamaste.logger import get_logger
from cnamaste.platforms.visium import bin_pitch
from legacy.spatio_genomic_counts import SpatioGenomicCounts

logger = get_logger(__name__, start_time=start_time)


def rectangle_partition(
    coords: np.ndarray, x_part: int, y_part: int, random_state: int | None
) -> tuple[list[np.ndarray], np.ndarray]:
    """Cut spots on an `x_part` by `y_part` grid over their x and y ranges, CalicoST's `rectangle_partition`.

    Args:
        coords: (n_spots, 2) positions.
        x_part: columns of the grid.
        y_part: rows of the grid.
        random_state: None for equal-width cells, or the seed of uniform random cell edges.

    Returns:
        The spot indices of each cell, column-major and possibly empty, and each spot's cell.
    """
    if random_state is not None:
        rng = np.random.default_rng(random_state)
        x_edges = np.sort(rng.uniform(0, 1, x_part))
        x_edges[-1] = 1.01
    else:
        x_edges = np.linspace(0, 1, 1 + x_part)
        x_edges[-1] += 0.01
        x_edges = x_edges[1:]

    x_range = [np.min(coords[:, 0]), np.max(coords[:, 0])]
    x_digit = np.digitize(coords[:, 0], x_range[0] + (x_range[1] - x_range[0]) * x_edges, right=True)

    if random_state is not None:
        y_edges = np.sort(rng.uniform(0, 1, y_part))
        y_edges[-1] = 1.01
    else:
        y_edges = np.linspace(0, 1, y_part + 1)
        y_edges[-1] += 0.01
        y_edges = y_edges[1:]

    y_range = [np.min(coords[:, 1]), np.max(coords[:, 1])]
    y_digit = np.digitize(coords[:, 1], y_range[0] + (y_range[1] - y_range[0]) * y_edges, right=True)

    cell_index = []
    cell_assignment = np.full(coords.shape[0], -1)
    for x_id in range(x_part):
        for y_id in range(y_part):
            index = np.where((x_digit == x_id) & (y_digit == y_id))[0]
            cell_assignment[index] = len(cell_index)
            cell_index.append(index)
    return cell_index, cell_assignment


def rectangle_partition_slices(
    coords: np.ndarray, sample_ids: np.ndarray, x_part: int, y_part: int
) -> list[np.ndarray]:
    """Cut each slice's spots on its own equal-width `x_part` by `y_part` grid (`rectangle_partition`).

    Args:
        coords: (n_spots, 2) positions, all slices.
        sample_ids: each spot's slice, 0..n_slices - 1.
        x_part: columns of each slice's grid.
        y_part: rows of each slice's grid.

    Returns:
        The spot indices of each clone, x_part x y_part per slice, slice by slice, possibly empty.

    Raises:
        RuntimeError: where a slice holds no spot.
    """
    clone_index = []
    for slice_id in range(1 + np.max(sample_ids)):
        index = np.where(sample_ids == slice_id)[0]
        if len(index) == 0:
            raise RuntimeError(f"Expected at least one spot in slice {slice_id}.")
        cell_index, _ = rectangle_partition(coords[index, :], x_part, y_part, random_state=None)
        clone_index.extend(index[cell] for cell in cell_index)
    logger.info(f"Initialized {len(clone_index)} clones given x_part,y_part={x_part},{y_part}.")
    return clone_index


def admits_assignment(block_sizes: Any, n_clones: int, floor: float) -> bool:
    """Whether some assignment of blocks to clones gives every clone more than `floor` spots.

    `initialize_rectangular_clones`' loop reaches every surjective map, so it returns exactly when this is True. A
    depth-first search, largest block first; equal-total clones are interchangeable.

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


def banded(coords: np.ndarray, n_clones: int) -> tuple[list[np.ndarray], np.ndarray]:
    """Split spots into `n_clones` equal-count bands along the axis with the most distinct values.

    Ordered along that axis, ties by the other then by index; every band passes the 20% test whenever
    n_spots >= n_clones.

    Args:
        coords: (n_spots, 2) positions.
        n_clones: the bands.

    Returns:
        The spot indices per band and each spot's band.
    """
    n_spots = len(coords)
    axis = int(np.argmax([np.unique(coords[:, a]).size for a in (0, 1)]))
    order = np.lexsort((np.arange(n_spots), coords[:, 1 - axis], coords[:, axis]))
    clone_id = np.empty(n_spots, dtype=int)
    clone_id[order] = np.arange(n_spots) * n_clones // n_spots
    return [np.where(clone_id == i)[0] for i in range(n_clones)], clone_id


def initialize_rectangular_clones(
    coords: np.ndarray, n_clones: int, random_state: int
) -> tuple[list[np.ndarray], np.ndarray]:
    """Split spots into `n_clones` by a ceil(sqrt(n_clones))^2 grid of Dirichlet(10)-sized blocks over the 5-95%
    range, the blocks drawn to clones until each clone holds more than 20% of an equal share; equal-count bands
    (`banded`) where no assignment of the blocks can (`admits_assignment`).

    Args:
        coords: (n_spots, 2) positions.
        n_clones: the clones.
        random_state: the seed of NumPy's global generator.

    Returns:
        The spot indices per clone and each spot's clone.
    """
    # NB reseeds NumPy's global stream, which every later global draw of the run continues.
    np.random.seed(random_state)

    logger.info(f"Solving for non-contiguous clone initialization for {n_clones} clones.")
    if n_clones == 1:
        clone_id = np.zeros(len(coords), dtype=int)
        return [np.where(clone_id == i)[0] for i in range(n_clones)], clone_id

    grid = int(np.ceil(np.sqrt(n_clones)))
    digits = []
    for axis in (0, 1):
        widths = np.random.dirichlet(np.ones(grid) * 10)
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
        return banded(coords, n_clones)

    while True:
        block_clone = np.random.randint(low=0, high=n_clones, size=grid**2)
        # NB a clone given no block takes the first block of the clone given most.
        while len(np.unique(block_clone)) < n_clones:
            counts = np.bincount(block_clone, minlength=n_clones)
            block_clone[np.where(block_clone == np.argmax(counts))[0][0]] = np.where(counts == 0)[0][0]
        clone_id = block_clone[block_id]
        clone_index = [np.where(clone_id == i)[0] for i in range(n_clones)]
        if np.min([len(index) for index in clone_index]) > floor:
            break

    logger.info(f"Solved for clone initialization for {n_clones} clones.")
    return clone_index, clone_id


def get_phasing_clones(
    coords: np.ndarray, physical: np.ndarray, sample_ids: np.ndarray, config: Any
) -> list[np.ndarray]:
    """The phasing start by `phasing.clone_init_type`: `rectangle`, each slice's npart_phasing^2 grid on its
    pixel `coords` (`rectangle_partition_slices`); or `median_bisection`, npart_phasing^2 x n_slices clones on the
    `physical` positions (`partition_slices`).

    Args:
        coords: (n_spots, 2) pixel positions.
        physical: (n_spots, 2) physical positions, um.
        sample_ids: each spot's slice, 0..n_slices - 1.
        config: the run's.

    Returns:
        The spot indices of each clone, possibly empty for `rectangle`.
    """
    n_part = config.phasing.npart_phasing
    if config.phasing.clone_init_type == "rectangle":
        return rectangle_partition_slices(coords, sample_ids, n_part, n_part)
    n_clones = n_part**2 * (1 + int(np.max(sample_ids)))
    clones = partition_slices(
        physical, sample_ids, n_clones, config.hmrf.random_seed, config.hmrf.bisection_quantile_jitter
    )
    return [np.flatnonzero(clones == clone) for clone in range(n_clones)]


def initialize_rdr_clone_refinement(
    merged_baf_assignment: np.ndarray,
    coords: np.ndarray,
    physical: np.ndarray,
    single_total_bb_RD: Any,
    n_obs: int,
    config: Any,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Split each BAF clone into `hmrf.n_clones_rdr` RDR clones, or keep it whole where it holds fewer than 20 x
    `n_obs` SNP UMIs, by `hmrf.rdr_clone_init_type`: `rectangle`, `initialize_rectangular_clones` on the pixel
    `coords` seeded by `hmm.random_state`; or `median_bisection` on the `physical` positions, one stream per BAF clone
    spawned from `hmrf.random_seed`.

    Args:
        merged_baf_assignment: (n_spots,) BAF clone labels 0..N-1.
        coords: (n_spots, 2) pixel positions.
        physical: (n_spots, 2) physical positions, um.
        single_total_bb_RD: SNP UMIs.
        n_obs: the segments.
        config: the run's.

    Returns:
        Each spot's RDR clone, the (n_spots, n_rdr_clones) mask of the RDR clones each spot's BAF clone allows, and
        the number of RDR clones.

    Raises:
        ValueError: where the BAF clone labels are not 0..N-1.
    """
    n_spots = len(merged_baf_assignment)
    n_baf_clones = len(np.unique(merged_baf_assignment))
    if not np.array_equal(np.unique(merged_baf_assignment), np.arange(n_baf_clones)):
        raise ValueError(f"BAF clone labels are not 0..{n_baf_clones - 1}: {np.unique(merged_baf_assignment)}.")

    splits_per_baf = []
    for bafc in range(n_baf_clones):
        idx_spots = np.flatnonzero(merged_baf_assignment == bafc)
        sufficient_snp_umi = SpatioGenomicCounts.of(single_total_bb_RD).spots(idx_spots).sum() >= 20 * n_obs  # MAGIC
        splits_per_baf.append(config.hmrf.n_clones_rdr if sufficient_snp_umi else 1)
    total_rdr_clones = int(sum(splits_per_baf))

    logger.info(
        f"Global hmrf will optimize {total_rdr_clones} rdr-refined clones given {n_baf_clones} baf-identified clones."
    )

    global_initial_assignment = np.zeros(n_spots, dtype=np.int32)
    # NB one-hot allowed (rdr-refined) clones.
    allowed_clones = np.zeros((n_spots, total_rdr_clones), dtype=bool)
    streams = np.random.SeedSequence(int(config.hmrf.random_seed)).spawn(n_baf_clones)
    global_clone_offset = 0

    for bafc, n_splits in enumerate(splits_per_baf):
        idx_spots = np.flatnonzero(merged_baf_assignment == bafc)
        if config.hmrf.rdr_clone_init_type == "rectangle":
            _, parts = initialize_rectangular_clones(coords[idx_spots], n_splits, config.hmm.random_state)
        else:
            parts = rotated_median_bisection(
                physical[idx_spots],
                n_splits,
                np.random.default_rng(streams[bafc]),
                config.hmrf.bisection_quantile_jitter,
            )
        global_initial_assignment[idx_spots] = global_clone_offset + parts
        allowed_clones[idx_spots, global_clone_offset : global_clone_offset + n_splits] = True
        global_clone_offset += n_splits

    return global_initial_assignment, allowed_clones, total_rdr_clones


BISECTION_MAX_DRAWS = 100
"""The draws of (theta, quantile) `rotated_median_bisection` makes for a cut that leaves both sides contiguous: a
half-plane can cut a lattice in a sliver whose spots are not neighbours."""


def count_components(neighbours: Any, index: np.ndarray) -> int:
    """The connected components of the spots `index` in the `neighbours` graph restricted to them."""
    return int(scipy.sparse.csgraph.connected_components(neighbours[index][:, index], directed=False)[0])


def rotated_median_bisection(
    coords: np.ndarray, n_parts: int, rng: np.random.Generator, quantile_jitter: float
) -> np.ndarray:
    """Cut spots into `n_parts` convex, contiguous parts of near-equal count: the piece with the most spots still to be
    divided, of k parts, is cut along a direction theta ~ U[0, pi) at the quantile k1 / k + U[-delta, delta] of its
    spots' projection on it, k1 = k // 2, delta `quantile_jitter` (`hmrf.bisection_quantile_jitter`), until every piece
    is one part. A draw whose sides hold more components than the piece and one is redrawn, up to `BISECTION_MAX_DRAWS`; neighbours are
    spots within 1.5 pitches (`platforms.visium.bin_pitch`), a Visium spot's 6 or a Visium HD bin's 8.

    Args: `coords` (n_spots, 2); `n_parts` at most n_spots; `rng` the `np.random.Generator` drawn from;
    `quantile_jitter` delta.
    Returns: (n_spots,) each spot's part, 0..n_parts - 1, in the order the cuts leave them.
    Raises: `ValueError` where there are fewer spots than parts.
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
    parts = np.empty(len(coords), dtype=np.int64)
    for part, (index, _) in enumerate(pieces):
        parts[index] = part
    return parts


def get_parts_per_slice(n_spots_per_slice: Any, n_parts: int) -> np.ndarray:
    """`n_parts` allocated to slices in proportion to their spot counts by largest remainder, at least 1 each.

    Args: `n_spots_per_slice` (n_slices,); `n_parts` at least n_slices.
    Returns: (n_slices,) parts per slice, summing to `n_parts`.
    Raises: `ValueError` where there are fewer parts than slices.
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


def partition_slices(
    coords: np.ndarray, sample_ids: np.ndarray, n_clones: int, seed: int, quantile_jitter: float
) -> np.ndarray:
    """Each slice cut by `rotated_median_bisection` into its share of `n_clones` (`get_parts_per_slice`), with one
    stream per slice spawned from `np.random.SeedSequence(seed)`: a slice's parts depend on its own spots alone.

    Args: `coords` (n_spots, 2); `sample_ids` each spot's slice, 0..n_slices - 1; `n_clones`; `seed` the int seed;
    `quantile_jitter` `rotated_median_bisection`'s delta.
    Returns: (n_spots,) each spot's clone, 0..n_clones - 1, slice by slice.
    """
    sample_ids = np.asarray(sample_ids)
    n_spots_per_slice = np.bincount(sample_ids)
    parts_per_slice = get_parts_per_slice(n_spots_per_slice, n_clones)
    streams = np.random.SeedSequence(int(seed)).spawn(len(n_spots_per_slice))
    clones = np.empty(len(sample_ids), dtype=np.int64)
    offset = 0
    for slice_id, n_parts in enumerate(parts_per_slice):
        index = np.flatnonzero(sample_ids == slice_id)
        clones[index] = offset + rotated_median_bisection(
            coords[index], n_parts, np.random.default_rng(streams[slice_id]), quantile_jitter
        )
        offset += n_parts
    logger.info(
        f"Cut {len(parts_per_slice)} slices into {parts_per_slice.tolist()} clones by rotated median bisection."
    )
    return clones
