"""`hmrf.lattice_type` on the positions the run reads, through `platforms.visium.physical_coords`: the CalicoST simulations' Visium
array, a Visium HD slice written here, the square-grid draw stored as Visium HD (dev_tree easy square,
`bdff091f`), the two-slice draws joined block diagonal (dev_tree easy `e4bbe03f`, triangular, and dev_tree easy square
`bdff091f`), and the guards.

Referees: the hexagonal ring written down by hand in Visium's array indices, `(0, +-2)` and `(+-1, +-1)`; a Visium HD
bin's 4 edges and 4 corners in its bin indices, `(+-1, 0)`, `(0, +-1)`, `(+-1, +-1)`; and each lattice's geometry, every
ring neighbour at the pitch. The Visium samples are CalicoST easy (`2d4ce9a9`) and hard (`8797710b`), their
`spatial/tissue_positions_list.csv` read by `io.get_spatial_positions` as the run reads it; each skips where the sample
is not beside the checkout. The Visium HD slice is an 8 um bin grid on its full-resolution pixels, `array_row` down
the slide as `pxl_row_in_fullres`, turned by 0.4 degrees as Space Ranger's fitted transform leaves it, with holes in the
tissue, and `microns_per_pixel` 0.2737.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest
import scipy.sparse

from cnamaste.io import get_spatial_positions
from cnamaste.platforms.visium import (
    bin_pitch,
    get_visium_pitch_um,
    physical_coords,
    reads_visium_hd,
    validate_platform_lattice,
)
from cnamaste.plot.plot_clones_spatial import plot_clones_spatial
from cnamaste.qa.sims import fixture, get_project
from cnamaste.spatial import (
    check_multislice_adjacency,
    construct_multislice_lattice_adjacency,
    validate_expected_adjacency,
)

SIMS = {"2d4ce9a9": "numcnas1.2_cnasize5e7_ploidy2_random0", "8797710b": "numcnas6.3_cnasize1e7_ploidy2_random0"}
"""Fixture hash -> the simulated sample's directory under `sim/`, beside the checkout: CalicoST easy and hard."""

RING = {(0, -2), (0, 2), (-1, -1), (-1, 1), (1, -1), (1, 1)}
"""A spot's hexagonal neighbours in Visium's array indices `(array_row, array_col)`."""

BOX = {(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)}
"""A bin's edges and corners in Visium HD's bin indices `(array_row, array_col)`."""

EDGES = {(-1, 0), (1, 0), (0, -1), (0, 1)}
"""A bin's edges alone, `hmrf.coordination_num` 4."""

MPP = 0.2737
"""The Visium HD slice's `microns_per_pixel`."""

BIN_UM = 8.0
"""The Visium HD slice's bin size, um."""


def sample_dir(fixture: str) -> Path:
    """The simulated sample's Space Ranger directory, or a skip where it is absent."""
    path = get_project().parent / "sim" / SIMS[fixture]
    if not path.is_dir():
        pytest.skip(f"{path} is not beside the checkout")
    return path


def read(directory: Path) -> tuple[np.ndarray, np.ndarray]:
    """A slice's `coords` as the run reads them, and its `physical_coords`."""
    coords = get_spatial_positions(directory)[["x", "y"]].to_numpy(np.float64)
    return coords, physical_coords(coords, np.zeros(len(coords), dtype=int), [directory])


def adjacency_of(physical: np.ndarray, lattice: str | None, coordination_num: int | None = None) -> object:
    """One slice's adjacency at `lattice`, and `coordination_num` where given, on its `physical` positions."""
    sample_ids = np.zeros(len(physical), dtype=int)
    return construct_multislice_lattice_adjacency(sample_ids, ["s"], physical, None, lattice, coordination_num)


def neighbour_offsets(adjacency: object, index: np.ndarray, spot: int) -> set[tuple[int, int]]:
    """`spot`'s neighbours as offsets in the lattice's indices, `index` (n_spots, 2) integers."""
    a = adjacency.tocsr()
    return {tuple(int(v) for v in index[j] - index[spot]) for j in a.indices[a.indptr[spot] : a.indptr[spot + 1]]}


def write_hd(root: Path, rows: int = 40, cols: int = 50, turn: float = 0.4, mpp: float | None = MPP) -> pl.DataFrame:
    """A Visium HD slice under `root`: `spatial/tissue_positions.parquet` for a `rows` x `cols` grid of `BIN_UM` bins
    on full-resolution pixels, turned `turn` degrees, a disc of tissue less two holes, and
    `spatial/scalefactors_json.json` with `mpp` where given. Returns the positions written."""
    r, c = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    r, c = r.ravel(), c.ravel()
    pitch, theta = BIN_UM / MPP, np.radians(turn)
    pxl_col = 1_200.0 + pitch * (np.cos(theta) * c - np.sin(theta) * r)
    pxl_row = 5_600.0 + pitch * (np.sin(theta) * c + np.cos(theta) * r)
    centre = np.array([rows / 2.0, cols / 2.0])
    tissue = np.hypot(r - centre[0], (c - centre[1]) * rows / cols) < rows / 2.0
    for hole in ((10, 15), (25, 30)):
        tissue &= np.hypot(r - hole[0], c - hole[1]) > 2.5
    frame = pl.DataFrame(
        {
            "barcode": [f"s_{i:06d}" for i in range(len(r))],
            "in_tissue": tissue.astype(np.int64),
            "array_row": r,
            "array_col": c,
            "pxl_row_in_fullres": pxl_row,
            "pxl_col_in_fullres": pxl_col,
        }
    )
    (root / "spatial").mkdir(parents=True, exist_ok=True)
    frame.write_parquet(root / "spatial" / "tissue_positions.parquet")
    if mpp is not None:
        (root / "spatial" / "scalefactors_json.json").write_text(json.dumps({"microns_per_pixel": mpp}))
    return frame.filter(pl.col("in_tissue") == 1)


# -- Visium: the CalicoST simulations ---------------------------------------------------------------------------


@pytest.mark.parametrize("fixture", list(SIMS))
def test_the_sims_adjacency_is_the_hexagonal_ring(fixture: str) -> None:
    """`triangular` on the sample's physical coordinates: every spot has 6 neighbours, none nearer than the pitch, and
    every spot whose 6 ring positions all hold a spot is joined to exactly them. Easy and hard are 100 rows of 30
    spots, so 2,744 of 3,000 spots are interior. Fails where the array's rows and columns are not taken to their
    true spacing, as the 6 nearest of the indices themselves cut the ring (`(+-2, 0)` ties with `(0, +-2)`)."""
    coords, physical = read(sample_dir(fixture))
    adjacency = adjacency_of(physical, "triangular").tocsr()

    assert np.all(np.diff(adjacency.indptr) == 6)
    coo = adjacency.tocoo()
    lengths = np.linalg.norm(physical[coo.row] - physical[coo.col], axis=1)
    assert lengths.min() == pytest.approx(get_visium_pitch_um())

    index = coords.astype(int)
    at = {tuple(p) for p in index.tolist()}
    interior = [i for i, (r, c) in enumerate(index.tolist()) if all((r + dr, c + dc) in at for dr, dc in RING)]
    for i in interior:
        assert neighbour_offsets(adjacency, index, i) == RING, (fixture, coords[i].tolist())
    assert len(interior) > len(coords) // 2


@pytest.mark.parametrize("fixture", list(SIMS))
def test_visium_physical_coordinates_are_um_x_right_y_up(fixture: str) -> None:
    """The sample's physical coordinates: x is the array's column at half the pitch, y the row at sqrt(3) / 2 of it,
    negated, so the first row, the top of the slide, is the highest and the first column the leftmost; the clone map
    draws them as given. Fails where a row and a column are crossed, y keeps the image's sense, or a page negates it."""
    coords, physical = read(sample_dir(fixture))
    assert np.allclose(physical[:, 0], coords[:, 1] * get_visium_pitch_um() / 2.0)
    assert np.allclose(physical[:, 1], -coords[:, 0] * get_visium_pitch_um() * np.sqrt(3.0) / 2.0)
    first_row = coords[:, 0] == coords[:, 0].min()
    assert physical[first_row, 1].min() == physical[:, 1].max()
    figure = plot_clones_spatial(physical, pd.Series(["clone 0"] * len(physical)))
    drawn = np.vstack([c.get_offsets() for c in figure.axes[0].collections])
    assert np.allclose(drawn, physical)


@pytest.mark.parametrize("fixture", list(SIMS))
def test_an_unset_lattice_is_refused(fixture: str) -> None:
    """An unset `hmrf.lattice_type`, the 8 nearest in the reader's coordinates as given, is deprecated: the adjacency
    refuses it. Fails where a configuration without the key builds an adjacency nobody chose."""
    _, physical = read(sample_dir(fixture))
    with pytest.raises(ValueError, match="unset lattice is deprecated"):
        adjacency_of(physical, None)


def test_pixels_are_refused_as_visium_indices() -> None:
    """Coordinates that are not integers, as pixels are, are refused where Visium's are taken to um. Fails where a
    positions file in pixels is read as array indices."""
    coords, _ = read(sample_dir("2d4ce9a9"))
    with pytest.raises(ValueError, match="non-integer"):
        physical_coords(coords * 137.9 + 2546.3, np.zeros(len(coords), dtype=int), [sample_dir("2d4ce9a9")])


def test_a_square_grid_is_refused_as_visium_indices() -> None:
    """A square grid's indices, `row + col` of both parities, are refused where Visium's are taken to um. Fails where
    Visium HD's bins are read as a hexagonal array."""
    grid = np.array([(r, c) for r in range(20) for c in range(20)], dtype=float)
    with pytest.raises(ValueError, match="both parities"):
        physical_coords(grid, np.zeros(len(grid), dtype=int), [sample_dir("2d4ce9a9")])


SQUARE = fixture("bdff091f")
"""dev_tree_easy_square r0 (`bdff091f`): a square-grid draw, each slice stored as Visium HD positions."""


@pytest.mark.parametrize("slice_id", ["8a2f1e43", "ec4b49a0"])
def test_the_square_draw_is_read_as_visium_hd_bins(slice_id: str) -> None:
    """dev_tree easy square (`bdff091f`) as committed: each slice reads as Visium HD, refuses `triangular`, its bins 8
    um apart, and `square` joins every interior bin to its 4 edges and 4 corners in the draw's `array_row`,
    `array_col`. Fails where the square draw is stored as Visium positions, or its pixels are not the bin pitch."""
    directory = SQUARE / slice_id
    with pytest.raises(ValueError, match="must be square"):
        validate_platform_lattice(directory, "triangular")
    validate_platform_lattice(directory, "square")
    coords, physical = read(directory)
    assert bin_pitch(physical) == pytest.approx(BIN_UM, rel=1e-12)
    index = (coords[:, ::-1] / BIN_UM).astype(np.int64)
    adjacency = adjacency_of(physical, "square")
    lo, hi = index.min(axis=0), index.max(axis=0)
    interior = np.flatnonzero(np.all((index > lo) & (index < hi), axis=1))
    assert len(interior) > len(index) // 2
    for i in interior:
        assert neighbour_offsets(adjacency, index, i) == BOX, index[i].tolist()


def test_an_array_with_no_rings_is_refused() -> None:
    """Easy's even rows alone pass the index guard (integers, one parity) but leave no spot its ring: the rows sit
    2 x sqrt(3) / 2 pitches apart, and each spot's 6 nearest reach past them. The majority guard refuses it. Fails
    where an adjacency whose spots are not on their ring is handed to the solver."""
    coords, physical = read(sample_dir("2d4ce9a9"))
    even = coords[:, 0] % 2 == 0
    with pytest.raises(ValueError, match="not a majority"):
        adjacency_of(physical[even], "triangular")


def test_square_on_a_visium_array_is_refused() -> None:
    """`square` on easy's physical coordinates: the 8 nearest of a hexagonal spot are its 6 at the pitch and 2 at
    sqrt(3) of it, not a bin's 4 edges and 4 corners, and the majority guard refuses it. Fails where Visium's array
    is built as Visium HD's bins."""
    _, physical = read(sample_dir("2d4ce9a9"))
    with pytest.raises(ValueError, match="not a majority"):
        adjacency_of(physical, "square")
    validate_platform_lattice(sample_dir("2d4ce9a9"), "triangular")
    with pytest.raises(ValueError, match="must be triangular"):
        validate_platform_lattice(sample_dir("2d4ce9a9"), "square")


# -- Visium HD: a slice written here ----------------------------------------------------------------------------


def test_visium_hd_is_read_as_pixels_x_the_column_y_the_row(tmp_path: Path) -> None:
    """`get_spatial_positions` reads Visium HD's in-tissue bins with `x` its `pxl_col_in_fullres`, increasing left to
    right, and `y` its `pxl_row_in_fullres`, increasing top to bottom, the image's sense. Fails where a pixel axis is
    crossed or flipped on reading."""
    written = write_hd(tmp_path)
    read_back = get_spatial_positions(tmp_path).set_index("barcode").loc[written["barcode"].to_list()]
    assert np.allclose(read_back["x"], written["pxl_col_in_fullres"])
    assert np.allclose(read_back["y"], written["pxl_row_in_fullres"])


def test_visium_hd_physical_coordinates_are_um_x_right_y_up(tmp_path: Path) -> None:
    """Visium HD's physical coordinates are its pixels times `microns_per_pixel`, x the column and y the row, negated:
    the bins of the first row, at the top of the slide, are the highest; the bin pitch is 8 um to 1e-9. A slice without
    `microns_per_pixel` is refused. Fails where y keeps the image's sense, or a bin's size is taken in pixels."""
    written = write_hd(tmp_path)
    coords, physical = read(tmp_path)
    assert np.allclose(physical, np.column_stack([coords[:, 0], -coords[:, 1]]) * MPP)
    rows = written["array_row"].to_numpy()
    assert physical[rows == rows.min(), 1].min() > physical[rows == rows.max(), 1].max()
    assert bin_pitch(physical) == pytest.approx(BIN_UM, rel=1e-9)
    write_hd(tmp_path / "bare", mpp=None)
    with pytest.raises(ValueError, match="microns_per_pixel"):
        read(tmp_path / "bare")


def test_visium_hd_adjacency_is_each_bin_s_edges_and_corners(tmp_path: Path) -> None:
    """`square` on Visium HD's physical coordinates: every bin has 8 neighbours, and every bin whose 8 neighbours all
    lie in the tissue is joined to exactly them, turned 0.4 degrees and around the holes alike; on the pixels, on y
    flipped, and on um, every bin without a tie at its 8th distance has the same 8, since the kNN reads distances
    alone. Fails where the kNN takes another ring, or a sense or a scale of the axes moves it."""
    written = write_hd(tmp_path)
    coords, physical = read(tmp_path)
    index = written.select(["array_row", "array_col"]).to_numpy()
    adjacency = adjacency_of(physical, "square").tocsr()
    assert np.all(np.diff(adjacency.indptr) == 8)
    at = {tuple(p) for p in index.tolist()}
    interior = [i for i, (r, c) in enumerate(index.tolist()) if all((r + dr, c + dc) in at for dr, dc in BOX)]
    assert len(interior) > len(index) // 2
    for i in interior:
        assert neighbour_offsets(adjacency, index, i) == BOX, index[i].tolist()
    # NB a bin with candidates tied at its 8th distance (at the tissue's edge or a hole) takes the tie in the order the
    #    tree meets them, which a flip or turn of the axes moves; every other bin's 8 nearest are the same set.
    distance = np.sort(np.linalg.norm(physical[:, None, :] - physical[None, :, :], axis=2), axis=1)
    untied = ~np.isclose(distance[:, 8], distance[:, 9], rtol=1e-6)
    assert untied[interior].all()
    for other in (coords, coords * np.array([1.0, -1.0])):
        moved = adjacency_of(other, "square").tocsr()
        for i in np.flatnonzero(untied):
            assert neighbour_offsets(moved, index, i) == neighbour_offsets(adjacency, index, i), index[i].tolist()


def test_visium_hd_adjacency_at_4_is_each_bin_s_edges(tmp_path: Path) -> None:
    """`square` at `hmrf.coordination_num` 4: every bin has 4 neighbours, every bin whose 4 edges lie in the tissue is
    joined to exactly them, and to the edges of its 8-neighbour adjacency, the corners left out; the majority guard
    passes on the 4-edge ring and refuses 4 nearest held to the 8-ring, and 6 is refused at `square`. Fails where the
    kNN takes a corner, the guard reads the 8-ring at 4, or a k the lattice does not take is built."""
    written = write_hd(tmp_path)
    _, physical = read(tmp_path)
    index = written.select(["array_row", "array_col"]).to_numpy()
    four = adjacency_of(physical, "square", 4).tocsr()
    eight = adjacency_of(physical, "square", 8).tocsr()
    assert np.all(np.diff(four.indptr) == 4)
    at = {tuple(p) for p in index.tolist()}
    interior = [i for i, (r, c) in enumerate(index.tolist()) if all((r + dr, c + dc) in at for dr, dc in BOX)]
    assert len(interior) > len(index) // 2
    for i in interior:
        assert neighbour_offsets(four, index, i) == EDGES, index[i].tolist()
        assert neighbour_offsets(eight, index, i) - neighbour_offsets(four, index, i) == BOX - EDGES
    assert validate_expected_adjacency(physical, four, "square", 4) > 0.5
    with pytest.raises(ValueError, match="not a majority"):
        validate_expected_adjacency(physical, four, "square", 8)
    with pytest.raises(ValueError, match="coordination_num 6 is not one of"):
        adjacency_of(physical, "square", 6)


def test_visium_hd_positions_require_square(tmp_path: Path) -> None:
    """A Space Ranger directory read as Visium HD (`spatial/tissue_positions.parquet`, no Visium CSV) requires
    `square`: `triangular` and the unset key are refused, `square` passes; and `triangular` on its physical
    coordinates is refused by the majority guard. Fails where Visium HD's bins could be built as another lattice."""
    write_hd(tmp_path)
    for lattice in ("triangular", None):
        with pytest.raises(ValueError, match="must be square"):
            validate_platform_lattice(tmp_path, lattice)
    validate_platform_lattice(tmp_path, "square")
    coords, physical = read(tmp_path)
    with pytest.raises(ValueError, match="not a majority"):
        adjacency_of(physical, "triangular")


def test_a_binned_output_with_a_visium_csv_is_read_as_visium_hd(tmp_path: Path) -> None:
    """A Visium HD binned output may carry `tissue_positions.csv` beside its parquet: with `bin_size_um` stated it is
    read as Visium HD, its bins from the parquet, and requires `square`; without it the CSV is Visium's, as before."""
    positions = write_hd(tmp_path)
    positions.to_pandas().to_csv(tmp_path / "spatial" / "tissue_positions.csv", index=False)
    assert not reads_visium_hd(tmp_path)
    (tmp_path / "spatial" / "scalefactors_json.json").write_text(
        json.dumps({"microns_per_pixel": MPP, "bin_size_um": BIN_UM})
    )
    assert reads_visium_hd(tmp_path)
    with pytest.raises(ValueError, match="must be square"):
        validate_platform_lattice(tmp_path, "triangular")
    validate_platform_lattice(tmp_path, "square")
    coords, _ = read(tmp_path)
    assert np.array_equal(coords, positions.select(["pxl_col_in_fullres", "pxl_row_in_fullres"]).to_numpy())


# -- Two slices: the dev_tree draws, and the guards ----------------------------------------------------------------

TWO_SLICES = {"e4bbe03f": "triangular", "bdff091f": "square"}
"""Fixture hash -> its `hmrf.lattice_type`: dev_tree easy r0 (Visium arrays) and dev_tree easy square r0 (Visium HD)."""


def get_two_slices() -> dict[str, str]:
    """Return the two slices, {"e4bbe03f": "triangular", "bdff091f": "square"}."""
    return TWO_SLICES


def two_slice_positions(draw: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A two-slice draw's `coords`, `physical_coords` and `sample_ids`, its slices in its sample sheet's order."""
    sheet = pd.read_csv(fixture(draw) / "sample_sheet.tsv", sep="\t")
    directories = [get_project() / d for d in sheet["spaceranger_dir"]]
    slices = [read(d)[0] for d in directories]
    coords = np.vstack(slices)
    sample_ids = np.repeat(np.arange(len(slices)), [len(c) for c in slices])
    return coords, physical_coords(coords, sample_ids, directories, lattice=TWO_SLICES[draw]), sample_ids


@pytest.mark.parametrize("draw", list(TWO_SLICES))
def test_two_slices_are_joined_block_diagonal(draw: str) -> None:
    """Each slice's block is its own adjacency built alone, nothing joins the slices, and an across-slice weight of 0.5
    is taken. Fails where a slice's kNN reaches into the other slice, or a guard refuses a valid across-slice edge."""
    _, physical, sample_ids = two_slice_positions(draw)
    lattice = TWO_SLICES[draw]
    joined = construct_multislice_lattice_adjacency(sample_ids, ["a", "b"], physical, None, lattice)
    for s in (0, 1):
        index = np.flatnonzero(sample_ids == s)
        alone = adjacency_of(physical[index], lattice)
        assert (joined[index][:, index] != alone).nnz == 0
        assert joined[index][:, np.flatnonzero(sample_ids != s)].nnz == 0
    first, second = np.flatnonzero(sample_ids == 0)[0], np.flatnonzero(sample_ids == 1)[0]
    across = scipy.sparse.csr_matrix(([0.5, 0.5], ([first, second], [second, first])), shape=joined.shape)
    with_across = construct_multislice_lattice_adjacency(sample_ids, ["a", "b"], physical, across, lattice)
    assert with_across[first, second] == 0.5 and with_across.nnz == joined.nnz + 2


@pytest.mark.parametrize("guard", ["self-loop", "within 2.0", "across 1.5", "unsorted", "explicit zero"])
def test_each_adjacency_guard_refuses(guard: str) -> None:
    """dev_tree easy (`e4bbe03f`): an across-slice matrix that adds a self-loop, doubles a within-slice edge to 2.0 or
    joins the slices at 1.5 is refused by `construct_multislice_lattice_adjacency`; a joined CSR with one row's indices
    reversed, or one explicit zero, by `check_multislice_adjacency`. Fails where the Potts energy is handed an adjacency
    other than the one built."""
    _, physical, sample_ids = two_slice_positions("e4bbe03f")
    joined = construct_multislice_lattice_adjacency(sample_ids, ["a", "b"], physical, None, "triangular")
    first, second = 0, int(np.flatnonzero(sample_ids == 1)[0])
    neighbour = int(joined.indices[0])
    added = {
        "self-loop": (first, first, 1.0),
        "within 2.0": (first, neighbour, 1.0),
        "across 1.5": (first, second, 1.5),
    }
    refusal = {
        "self-loop": "self-loops",
        "within 2.0": "within-slice adjacency weights are not 1.0",
        "across 1.5": r"across-slice adjacency weights are not in \(0, 1\]",
        "unsorted": "not sorted",
        "explicit zero": "explicit zeros",
    }[guard]
    if guard in added:
        row, col, weight = added[guard]
        across = scipy.sparse.csr_matrix(([weight], ([row], [col])), shape=joined.shape)
        with pytest.raises(ValueError, match=refusal):
            construct_multislice_lattice_adjacency(sample_ids, ["a", "b"], physical, across, "triangular")
        return
    broken = joined.copy()
    if guard == "unsorted":
        broken.indices[broken.indptr[0] : broken.indptr[1]] = broken.indices[broken.indptr[0] : broken.indptr[1]][::-1]
        broken = scipy.sparse.csr_matrix((broken.data, broken.indices, broken.indptr), shape=broken.shape)
    else:
        broken.data[0] = 0.0
    with pytest.raises(ValueError, match=refusal):
        check_multislice_adjacency(broken, sample_ids)
