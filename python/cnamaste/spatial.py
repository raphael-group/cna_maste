from typing import Any

import numpy as np
import scipy.linalg
import scipy.sparse
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

from cnamaste.config import get_lattice_types, start_time
from cnamaste.logger import get_logger
from cnamaste.platforms.visium import (
    validate_visium_adjacency,
    validate_visiumhd_adjacency,
)
from cnamaste.potts import noloop_adjacency_coo_cast

logger = get_logger(__name__, start_time=start_time)


def log_sparse_matrix_stats(matrix, name):
    # NB see https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.getnnz.html
    num_neighbors = matrix.getnnz(axis=1)

    unique_vals, counts = np.unique(num_neighbors, return_counts=True)
    fractions = counts / len(num_neighbors)
    sort_idx = np.argsort(fractions)[::-1]

    fraction_strs = [f"{unique_vals[i]} neighbors\t{fractions[i]:.1%}" for i in sort_idx]

    if matrix.nnz > 0:
        median_weight = np.median(matrix.data)
    else:
        median_weight = 0.0

    logger.info(f"{name} | Median edge weight: {median_weight:.2f} | Neighbor fractions:\n{'\n'.join(fraction_strs)}")


def construct_lattice_adjacency(coords, coordination_num=8):
    logger.info(f"Assigning lattice adjacency matrix with coordination_num={coordination_num}.")

    n_spots = coords.shape[0]

    scaled_coords = np.asarray(coords, dtype=float)

    logger.info(f"Building kd-tree for efficient nearest neighbor search")

    tree = cKDTree(scaled_coords)

    # NB query (k+1) nearest neighbors, as includes self.
    _, indices = tree.query(scaled_coords, k=1 + coordination_num)

    indices = indices[:, 1:]

    logger.info(f"Constructed nearest neighbor indices via KD-tree")

    logger.info(f"Constructing adjacency matrix.")

    rows = np.repeat(np.arange(n_spots), coordination_num)
    cols = indices.flatten()

    # NB uniform (unit) edge weight, we will apply global spatial_weight in hmrf.
    data = np.ones(len(rows), dtype=np.float64)

    adjacency_mat = csr_matrix((data, (rows, cols)), shape=(n_spots, n_spots))

    log_sparse_matrix_stats(adjacency_mat, "adjacency_mat")

    return adjacency_mat


def validate_expected_adjacency(positions, adjacency, lattice, coordination_num=None):
    """Refuse an adjacency whose spots are mostly not joined to their lattice's ring, by platform.

    Args:
        positions: (n_spots, 2) um, `physical_coords`.
        adjacency: the slice's kNN adjacency.
        lattice: `hmrf.lattice_type`.
        coordination_num: `hmrf.coordination_num`, the lattice's default where None.

    Returns:
        The share of spots on their ring.

    Raises:
        ValueError: for a lattice not in `LATTICE_TYPES`, or where the share is not a majority.
    """
    match lattice:
        case "triangular":
            return validate_visium_adjacency(positions, adjacency)
        case "square":
            return validate_visiumhd_adjacency(
                positions, adjacency, coordination_num or get_lattice_types()["square"][0]
            )
        case _:
            raise ValueError(f"hmrf.lattice_type {lattice!r} is not one of {tuple(get_lattice_types())}.")


def construct_multislice_lattice_adjacency(
    sample_ids,
    sample_list,
    physical_coords,
    across_slice_adjacency_mat,
    lattice_type,
    coordination_num=None,
):
    """Each slice's k nearest neighbours on the spots' physical positions, block diagonal across slices.

    `lattice_type`, `hmrf.lattice_type`, reads `physical_coords`, `(x, y)` in um (`platforms.visium.physical_coords`):
    - `triangular`: Visium's array (`visium_physical`). Each spot is joined to its 6 nearest, and a majority must be
      their hexagonal ring at the pitch, `VISIUM_PITCH_UM` (`validate_visium_adjacency`).
    - `square`: Visium HD's bins (`pixel_physical`). Each bin is joined to its 8 nearest, and a majority must be its
      4 edges and 4 corners at the pitch, `bin_pitch` (`validate_visiumhd_adjacency`); at `coordination_num` 4, its
      4 nearest, its 4 edges.
    The kNN reads distances alone, so neither y's sense nor a rotation of the slide moves it. An unset lattice, the
    8 nearest in the reader's coordinates as given, is deprecated and refused.

    Args:
        sample_ids: each spot's slice.
        sample_list: the slices.
        physical_coords: (n_spots, 2) um.
        across_slice_adjacency_mat: added where given.
        lattice_type: a key of `config.get_lattice_types()`.
        coordination_num: `hmrf.coordination_num`, one of the lattice's, its first where None.

    Returns:
        The (n_spots, n_spots) CSR adjacency.

    Raises:
        ValueError: for a lattice not in `config.get_lattice_types()`, a `coordination_num` it does not take, positions
            that are not the lattice's, or an adjacency `check_multislice_adjacency` refuses.
    """
    logger.info("Solving for multi-slice adjacency matrix.")
    if lattice_type not in get_lattice_types():
        raise ValueError(
            f"hmrf.lattice_type {lattice_type!r} is not one of {tuple(get_lattice_types())}: an unset lattice is deprecated."
        )
    coordination_num = coordination_num or get_lattice_types()[lattice_type][0]
    if coordination_num not in get_lattice_types()[lattice_type]:
        raise ValueError(
            f"hmrf.coordination_num {coordination_num!r} is not one of {get_lattice_types()[lattice_type]} at "
            f"hmrf.lattice_type {lattice_type}."
        )

    adjacency_mat = []

    for i, _ in enumerate(sample_list):
        # NB spots per slice.
        index = np.where(sample_ids == i)[0]

        positions = np.asarray(physical_coords[index, :], dtype=np.float64)
        if not np.all(np.isfinite(positions)):
            raise ValueError(f"hmrf.lattice_type {lattice_type}: a spot's physical coordinates are not finite.")
        tmpadjacency_mat = construct_lattice_adjacency(positions, coordination_num=coordination_num)
        validate_expected_adjacency(positions, tmpadjacency_mat, lattice_type, coordination_num)

        adjacency_mat.append(scipy.sparse.csr_matrix(tmpadjacency_mat))

    # NB sets block diagonals corresponding to inter-slice: zero between slices. Sparse, where the dense (spots, spots)
    #    block diagonal took 83 GB at 102,000 spots; the same entries, dtype and canonical CSR as `csr_matrix` of it.
    adjacency_mat = scipy.sparse.block_diag(
        adjacency_mat, format="csr", dtype=np.result_type(*(m.dtype for m in adjacency_mat))
    )
    adjacency_mat.eliminate_zeros()
    adjacency_mat.sort_indices()

    # NB add intra-slice adjacency.
    if across_slice_adjacency_mat is not None:
        adjacency_mat += across_slice_adjacency_mat

    logger.info("Solving for multi-slice adjacency matrix.")

    check_multislice_adjacency(adjacency_mat, sample_ids)

    return adjacency_mat


def check_multislice_adjacency(adjacency_mat: Any, sample_ids: np.ndarray) -> None:
    """Refuse a multi-slice adjacency the Potts energy does not read as built: a self-loop, a CSR matrix that is not
    canonical (unsorted indices, an explicit zero or a duplicate), a within-slice weight other than 1.0, or an
    across-slice weight (`across_slice_adjacency_mat`'s) outside (0, 1].

    Args:
        adjacency_mat: (n_spots, n_spots) CSR.
        sample_ids: each spot's slice.

    Raises:
        ValueError: naming the failure.
    """
    # NB a spot is not its own neighbour: a self-loop is aligned in every labelling, a constant in the Potts energy.
    noloop_adjacency_coo_cast(adjacency_mat)
    if not adjacency_mat.has_sorted_indices:
        raise ValueError("the adjacency's CSR indices are not sorted: it is not canonical.")
    if not adjacency_mat.has_canonical_format:
        raise ValueError("the adjacency's CSR holds duplicate entries: it is not canonical.")
    if np.any(adjacency_mat.data == 0):
        raise ValueError(f"the adjacency's CSR stores {np.count_nonzero(adjacency_mat.data == 0)} explicit zeros.")
    rows = np.repeat(np.arange(adjacency_mat.shape[0]), np.diff(adjacency_mat.indptr))
    within = np.asarray(sample_ids)[rows] == np.asarray(sample_ids)[adjacency_mat.indices]
    if np.any(adjacency_mat.data[within] != 1.0):
        raise ValueError(
            f"{np.count_nonzero(adjacency_mat.data[within] != 1.0)} within-slice adjacency weights are not 1.0."
        )
    across = adjacency_mat.data[~within]
    if np.any((across <= 0.0) | (across > 1.0)):
        raise ValueError(
            f"{np.count_nonzero((across <= 0.0) | (across > 1.0))} across-slice adjacency weights are not in (0, 1]."
        )
