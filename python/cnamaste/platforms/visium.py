"""Visium and Visium HD: array indices and bin pixels as physical positions, and which a Space Ranger directory holds."""

import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from cnamaste.config import get_lattice_types, start_time
from cnamaste.logger import get_logger
from cnamaste.utils import find_plain_or_gz

logger = get_logger(__name__, start_time=start_time)

VISIUM_PITCH_UM = 100.0
"""Visium's spot pitch: the centre-to-centre distance of hexagonal neighbours, in um."""


def get_visium_pitch_um() -> float:
    """Return the visium pitch um, 100.0."""
    return VISIUM_PITCH_UM


LATTICE_RINGS = {
    ("triangular", 6): (1.0,) * 6,
    ("square", 8): (1.0,) * 4 + (np.sqrt(2.0),) * 4,
    ("square", 4): (1.0,) * 4,
}
"""`(hmrf.lattice_type, hmrf.coordination_num)` -> an interior spot's neighbours' distances, in pitches, sorted."""

RING_RTOL = {"triangular": 1e-9, "square": 0.02}
"""`hmrf.lattice_type` -> an edge's relative tolerance to its ring's length; Visium HD's pixels are a fitted transform."""


def bin_pitch(positions):
    """Visium HD's bin pitch in the positions' own units, um: the median distance from a bin to its nearest.

    Args:
        positions: (n_spots, 2).

    Returns:
        The pitch.
    """
    return float(np.median(cKDTree(positions).query(positions, k=2)[0][:, 1]))


def validate_visium_adjacency(positions, adjacency):
    """Visium: a majority of spots joined to their hexagonal ring, all 6 at `VISIUM_PITCH_UM`."""
    return validate_ring_majority(positions, adjacency, "triangular", VISIUM_PITCH_UM, "hexagonal ring", "Visium array")


def validate_visiumhd_adjacency(positions, adjacency, coordination_num=8):
    """Visium HD: a majority of bins joined to their 4 edges, and at `coordination_num` 8 their 4 corners."""
    ring_name = "4 edges and 4 corners" if coordination_num == 8 else "4 edges"
    return validate_ring_majority(
        positions, adjacency, "square", bin_pitch(positions), ring_name, "Visium HD bin grid", coordination_num
    )


def validate_ring_majority(positions, adjacency, lattice, pitch, ring_name, grid_name, coordination_num=None):
    """Refuse an adjacency in which half the spots or fewer are joined to exactly their lattice's ring, to `RING_RTOL`.

    Args: `positions` (n_spots, 2) um; `adjacency` the kNN adjacency; `lattice` `hmrf.lattice_type`; `pitch`;
    `ring_name`, `grid_name` for the messages; `coordination_num` the k, the lattice's default where None.

    Returns:
        The share of spots on their ring.

    Raises:
        ValueError: where it is not a majority.
    """
    ring = np.asarray(LATTICE_RINGS[lattice, coordination_num or get_lattice_types()[lattice][0]]) * pitch
    a = adjacency.tocsr()
    a.sort_indices()
    counts = np.diff(a.indptr)
    lengths = np.linalg.norm(positions[np.repeat(np.arange(len(positions)), counts)] - positions[a.indices], axis=1)
    # NB the kNN gives every spot k; a spot with any other count is off its ring.
    on_ring = counts == len(ring)
    if on_ring.any():
        held = np.isin(np.repeat(np.arange(len(positions)), counts), np.flatnonzero(on_ring))
        found = np.sort(lengths[held].reshape(-1, len(ring)), axis=1)
        on_ring[on_ring] = np.all(np.isclose(found, ring, rtol=RING_RTOL[lattice], atol=0.0), axis=1)
    share = float(np.mean(on_ring))
    if share <= 0.5:
        raise ValueError(
            f"hmrf.lattice_type {lattice}: {share:.1%} of spots have their {ring_name} as their {len(ring)} nearest, "
            f"not a majority; the coordinates are not a {grid_name}'s."
        )
    logger.info(
        f"{lattice.capitalize()} adjacency: {share:.1%} of spots have their {ring_name}; the rest are at edges or holes."
    )
    return share


def visium_physical(coords):
    """Visium array indices `(array_row, array_col)` as physical positions `(x, y)` in um, by the array's lattice vectors.

    Args:
        coords: (n_spots, 2) array indices, `(array_row, array_col)`.

    Returns:
        (n_spots, 2) `(x, y)` in um.
    """
    indices = np.asarray(coords, dtype=np.float64)
    return np.column_stack(
        [indices[:, 1] * VISIUM_PITCH_UM / 2.0, -indices[:, 0] * VISIUM_PITCH_UM * np.sqrt(3.0) / 2.0]
    )


def pixel_physical(coords, microns_per_pixel):
    """Visium HD's bin pixels `(pxl_col, pxl_row)` as physical positions `(x, y)` in um.

    Args:
        coords: (n_spots, 2) full-resolution pixels.
        microns_per_pixel: Space Ranger's scale factor.

    Returns:
        (n_spots, 2) `(x, y)` in um.
    """
    pixels = np.asarray(coords, dtype=np.float64)
    return np.column_stack([pixels[:, 0], -pixels[:, 1]]) * float(microns_per_pixel)


def check_visium_indices(coords):
    """Refuse coordinates that are not Visium array indices: integers, unique, `row + col` of one parity.

    Args:
        coords: (n_spots, 2).

    Raises:
        ValueError: naming what was found: pixels, or a square grid.
    """
    found = np.asarray(coords, dtype=np.float64)
    if not np.all(np.isfinite(found)) or not np.array_equal(found, np.round(found)):
        raise ValueError(
            "hmrf.lattice_type triangular reads Visium array indices (array_row, array_col); found non-integer "
            "coordinates, as pixels are."
        )
    indices = np.round(found).astype(np.int64)
    if len(np.unique(indices, axis=0)) < len(indices):
        raise ValueError("hmrf.lattice_type triangular: two spots share an array index.")
    if np.unique((indices[:, 0] + indices[:, 1]) % 2).size > 1:
        raise ValueError(
            "hmrf.lattice_type triangular reads Visium's hexagonal array, its row + col of one parity; found both "
            "parities, as on a square grid."
        )


def reads_visium_hd(spaceranger_dir):
    """Whether `get_spatial_positions` reads `spaceranger_dir` as Visium HD, from `spatial/tissue_positions.parquet`.

    Args:
        spaceranger_dir: a slice's Space Ranger directory.

    Returns:
        True for Visium HD, False for Visium.
    """
    spatial = Path(f"{spaceranger_dir}/spatial")
    if not (spatial / "tissue_positions.parquet").exists():
        return False
    scalefactors = spatial / "scalefactors_json.json"
    if scalefactors.is_file() and "bin_size_um" in json.loads(scalefactors.read_text()):
        return True
    return not any(
        find_plain_or_gz(f"{spatial}/{name}") for name in ("tissue_positions.csv", "tissue_positions_list.csv")
    )


def validate_platform_lattice(spaceranger_dir, lattice):
    """Refuse a lattice other than square for Visium HD, or square for Visium.

    Args:
        spaceranger_dir: a slice's Space Ranger directory.
        lattice: `hmrf.lattice_type`.

    Raises:
        ValueError: where the lattice does not match the platform.
    """
    if reads_visium_hd(spaceranger_dir):
        if lattice != "square":
            raise ValueError(
                f"{spaceranger_dir} is read as Visium HD (spatial/tissue_positions.parquet), a square lattice of bins: "
                f"hmrf.lattice_type must be square, not {lattice!r}."
            )
    elif lattice == "square":
        raise ValueError(
            f"{spaceranger_dir} is read as Visium (spatial/tissue_positions[_list].csv), a triangular array: "
            "hmrf.lattice_type must be triangular, not 'square'."
        )


def get_visiumhd_microns_per_pixel(spaceranger_dir):
    """Visium HD's full-resolution pixel size: `spatial/scalefactors_json.json`'s `microns_per_pixel`.

    Args:
        spaceranger_dir: a slice's Space Ranger directory.

    Returns:
        Um per pixel.

    Raises:
        ValueError: where the file or the key is missing: a bin's pixels then have no size in um.
    """
    path = Path(f"{spaceranger_dir}/spatial/scalefactors_json.json")
    found = json.loads(path.read_text()).get("microns_per_pixel") if path.is_file() else None
    if found is None:
        raise ValueError(f"{path} gives no microns_per_pixel: Visium HD's bin pixels have no size in um.")
    return float(found)


def physical_coords(coords, sample_ids, spaceranger_dirs, check_indices=True, lattice="triangular"):
    found = np.full((len(coords), 2), np.nan)
    for s, spaceranger_dir in enumerate(spaceranger_dirs):
        index = np.flatnonzero(np.asarray(sample_ids) == s)
        if reads_visium_hd(spaceranger_dir):
            found[index] = pixel_physical(coords[index], get_visiumhd_microns_per_pixel(spaceranger_dir))
        elif lattice == "triangular":
            if check_indices:
                check_visium_indices(coords[index])
            found[index] = visium_physical(coords[index])
        else:
            raise ValueError(
                f"{spaceranger_dir} is read as Visium (spatial/tissue_positions[_list].csv), a triangular array: "
                f"hmrf.lattice_type must be triangular, not {lattice!r}."
            )
    return found
