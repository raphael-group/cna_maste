"""Pages for run_cnamaste's figures: a fixed page, panels placed in inches, one genomic axis.

Pages a third, half or full text block tall.
`plot_combined_page`: (a) H&E slide and clones, (b) copy-number profile, (c) each clone's RDR and BAF tracks.
`plot_he_page`: (a) H&E slide and (b) each spot's H&E class.
`plot_spot_umi_page`: each spot's (a) UMIs and (b) SNP-UMIs, as histograms side by side.
Genomic panels share one `GenomicAxis`, spatial panels one extent; every text is `[tool.cnamaste.figures]`'s
`font_size`, clones named as the run names them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D, Transform, offset_copy
from scipy.spatial import cKDTree

from cnamaste.figure_style import (
    get_box,
    get_box_alpha,
    get_clone_palette,
    get_compared_box,
    get_compared_whisker,
    get_dpi,
    get_he_palette,
    get_whisker,
    page_size,
    page_style,
    rasterize,
    stagger,
    style,
    uniform,
)
from cnamaste.plot.plot_clones_spatial import get_clone_colours, plot_clones_spatial
from cnamaste.plot.plot_copy_number_profile import plot_state_key, plot_state_profile, used_states
from cnamaste.plot.plot_genomic import get_rdr_floor, get_rdr_ylim, plot_clones_genomic

MARGIN = {"left": 0.62, "right": 0.05, "top": 0.05, "bottom": 0.35}
"""Inches around the panels: tick labels left, staggered chromosome names below."""

LETTER = 0.16
"""Inches of a panel letter's line."""

TICK_LINE = 0.25
"""Inches under the spatial panels for their tick labels."""

GAP = 0.1
"""Inches between panels."""

PANEL_GAP = 0.3
"""Inches between spatial panels side by side: each labels its x ticks, so one's last label and the next's first
(`100`, `0`) need the room of a tick label between them."""

ROW = 1.9
"""Inches, a spatial panel's height at most, so the tracks keep their share of the page."""

KEY_COLUMN = 1.35
"""Inches right of the spatial panels kept for their key at least; `spatial_row` widens it to the key as drawn."""

PROFILE_KEY = 0.3
"""Inches of the profile's key, above it."""

PROFILE_ROW = 0.165
"""Inches of profile per clone. The page's height is fixed, so what the profile does not take goes to the clone
tracks below it."""

NAME_LINE = 0.14
"""Inches above a clone's tracks for its name and state key."""

TRACK_GAP = 0.05
"""Inches between a clone's RDR and BAF tracks, so their end ticks do not meet."""

NORMAL_RDR_YLIM = 3.0
"""The top of the normal clone's RDR track on `plot_combined_page`, in place of `RDR_YLIM`: its track is shorter by
the same ratio of y ranges, so RDR takes the same inches per unit on every track."""


def get_margin() -> dict[str, float]:
    """Return the margin, `MARGIN`: inches left 0.62, right 0.05, top 0.05, bottom 0.35."""
    return dict(MARGIN)


def get_letter() -> float:
    """Return the letter, 0.16 inches."""
    return LETTER


def get_tick_line() -> float:
    """Return the tick line, 0.25 inches."""
    return TICK_LINE


def get_gap() -> float:
    """Return the gap, 0.1 inches."""
    return GAP


def get_row() -> float:
    """Return the row, 1.9 inches."""
    return ROW


def get_name_line() -> float:
    """Return the name line, 0.14 inches."""
    return NAME_LINE


def get_track_gap() -> float:
    """Return the track gap, 0.05 inches."""
    return TRACK_GAP


@dataclass(frozen=True)
class GenomicAxis:
    """Bins in genome order, one run per chromosome: the x axis every genomic panel draws on.

    Args:
        lengths: bins per chromosome.
        names: the chromosomes' names.
    """

    lengths: np.ndarray
    names: tuple[str, ...]

    @classmethod
    def from_bins(cls, chromosomes: Any) -> GenomicAxis:
        """Build the axis from each bin's chromosome, in bin order (`cnv_<ploidy>_seglevel.tsv`'s `CHR`).

        Args:
            chromosomes: per-bin chromosome names.

        Returns:
            A `GenomicAxis`.

        Raises:
            ValueError: where a chromosome's bins are not contiguous.
        """
        chromosomes = np.asarray(chromosomes)
        starts = np.flatnonzero(np.r_[True, chromosomes[1:] != chromosomes[:-1]])
        lengths = np.diff(np.r_[starts, chromosomes.size])
        names = tuple(str(c) for c in chromosomes[starts])
        if len(set(names)) != len(names):
            raise ValueError("a chromosome's bins are not contiguous")
        return cls(lengths, names)

    @property
    def n_bins(self) -> int:
        return int(np.sum(self.lengths))

    @property
    def starts(self) -> np.ndarray:
        return np.r_[0, np.cumsum(self.lengths)[:-1]]

    def frame(self, ax: Any, labels: bool = False) -> None:
        """Set `ax` on bins `[0, n_bins)` with a line at each chromosome's start.

        With `labels`, names each chromosome under its middle, a clashing name on a second line, and `chr` at the left.

        Args:
            ax: the axes.
            labels: whether to name chromosomes.
        """
        ax.set_xlim(0, self.n_bins)
        ax.set_xticks([])
        for x in self.starts[1:]:
            ax.axvline(x, color="black", linewidth=0.5, zorder=3)
        if not labels:
            return
        size = style()["font_size"]
        points = ax.get_position().width * ax.figure.get_size_inches()[0] * 72.0 / self.n_bins
        middles = [(start + length / 2.0) * points for start, length in zip(self.starts, self.lengths)]
        for start, length, name, line in zip(self.starts, self.lengths, self.names, stagger(middles, self.names, size)):
            at = offset_copy(ax.get_xaxis_transform(), ax.figure, y=-2.0 - line * 1.1 * size, units="points")
            ax.text(start + length / 2.0, 0.0, name, transform=at, ha="center", va="top")
        at = offset_copy(ax.transAxes, ax.figure, x=-4.0, y=-2.0, units="points")
        ax.text(0.0, 0.0, "chr", transform=at, ha="right", va="top")


def add_axes(figure: Any, x0: float, top: float, width: float, height: float) -> Any:
    """Add an axes `width` by `height` inches, its left edge `x0` and its top `top` inches from the page's left and top.

    Args:
        figure: the page.
        x0: in inches.
        top: in inches.
        width: in inches.
        height: in inches.

    Returns:
        The axes.
    """
    w, h = figure.get_size_inches()
    return figure.add_axes((x0 / w, 1.0 - (top + height) / h, width / w, height / h))


def place(ax: Any, x0: float, top: float, width: float, height: float) -> None:
    """Move `ax` to `width` by `height` inches at `x0`, `top` inches down.

    Args:
        ax: the axes.
        x0: in inches.
        top: in inches.
        width: in inches.
        height: in inches.
    """
    w, h = ax.figure.get_size_inches()
    ax.set_position((x0 / w, 1.0 - (top + height) / h, width / w, height / h))


def letter(figure: Any, top: float, label: str, x0: float = 0.02) -> None:
    """Place a panel letter `(label)` `x0` inches from the page's left edge, `top` inches from its top.

    Args:
        figure: the page.
        top: in inches.
        x0: in inches.
        label: the letter.
    """
    w, h = figure.get_size_inches()
    figure.text(x0 / w, 1.0 - top / h, f"({label})", va="top", ha="left")


def spot_size(n_spots: int) -> float:
    """Points^2 per spot on a slide-sized panel, where the spots are drawn as points."""
    return float(np.clip(12_000.0 / n_spots, 0.1, 25.0))


def slide_grid(he_frame: pd.DataFrame, physical_coords: np.ndarray) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray]:
    """The slide's pixels against the spots: where they are drawn, the spot nearest each, and which are on the tissue.

    A pixel is on the tissue where a spot lies within one spot spacing (the spots' median nearest-neighbour distance)
    of it; the slide's own frame places each pixel at its `x, y`, in the spots' physical coordinates
    (`he.slide_image_and_meta`, taken to um by `he.physical_affine`).

    Args:
        he_frame: the per-pixel frame, `x, y` um.
        physical_coords: (n_spots, 2) um, y up.

    Returns:
        The transform taking pixel `(column, row)` to the spots' `(x, y)` as drawn; each pixel's `(row, column)`; its
            nearest spot; whether it is on the tissue.
    """
    rows, cols = he_frame["array_row"].to_numpy(), he_frame["array_col"].to_numpy()
    xy = he_frame[["x", "y"]].to_numpy(np.float64)
    # NB exact on any three pixels not in a line; a sample of them keeps the fit cheap on a large image.
    every = max(1, len(rows) // 10_000)
    design = np.column_stack([cols[::every], rows[::every], np.ones(len(rows[::every]))]).astype(np.float64)
    fit = np.linalg.lstsq(design, xy[::every], rcond=None)[0].T
    to_axes = Affine2D(np.vstack([fit[0], fit[1], [0.0, 0.0, 1.0]]))
    tree = cKDTree(physical_coords)
    spacing = float(np.median(tree.query(physical_coords, k=2)[0][:, 1]))
    distance, nearest = tree.query(xy)
    return to_axes, np.column_stack([rows, cols]), nearest, distance <= spacing


def draw_image(ax: Any, to_axes: Any, at: np.ndarray, colours: np.ndarray, shape: tuple[int, int]) -> None:
    """Draw `colours` at pixels `at` on a white image of `shape`, through `to_axes` onto the spots' coordinates."""
    image = np.ones((*shape, 3))
    image[at[:, 0], at[:, 1]] = colours
    height, width = shape
    ax.imshow(
        image, extent=(-0.5, width - 0.5, height - 0.5, -0.5), interpolation="none", transform=to_axes + ax.transData
    )


def draw_slide(ax: Any, he_frame: pd.DataFrame, physical_coords: np.ndarray) -> None:
    """Draw the H&E slide `he.slide_image_and_meta` returns under the spots, drawn at their physical `(x, y)`, white
    off the tissue.

    Args:
        ax: the axes.
        he_frame: the per-pixel frame, `x, y` um.
        physical_coords: (n_spots, 2) um, y up.
    """
    to_axes, at, _, on = slide_grid(he_frame, physical_coords)
    rgb = he_frame[["red", "green", "blue"]].to_numpy(np.float64)
    rgb = np.clip(rgb / (255.0 if rgb.max() > 1.0 else 1.0), 0.0, 1.0)
    shape = (int(at[:, 0].max()) + 1, int(at[:, 1].max()) + 1)
    draw_image(ax, to_axes, at[on], rgb[on], shape)


def square_handle(colour: Any) -> Line2D:
    return Line2D([0], [0], marker="s", color="w", markerfacecolor=colour, markeredgewidth=0, markersize=6)


def he_colours(he_labels: np.ndarray) -> list[str]:
    """One `HE_PALETTE` colour per H&E class, 0 to the largest in `he_labels`, as `draw_classes` and
    `he_clones_genomic.pdf` draw them."""
    return sns.color_palette(get_he_palette(), int(np.max(he_labels)) + 1).as_hex()


def draw_classes(
    ax: Any, physical_coords: np.ndarray, he_labels: np.ndarray, he_frame: pd.DataFrame | None = None
) -> tuple[list[Any], list[str]]:
    """Draw each spot's H&E class (`get_he_image`'s `label`, 0 darkest) in `HE_PALETTE`.

    With the slide's frame, on its pixels: each tissue pixel in its nearest spot's class, white off the tissue, at the
    slide's resolution and scale (`draw_slide`); without one, a point per spot.

    Args:
        ax: the axes.
        physical_coords: (n_spots, 2) um, y up.
        he_labels: per-spot classes.
        he_frame: the slide's frame, `x, y` um, or None.

    Returns:
        The key's handles and labels.
    """
    coords = np.asarray(physical_coords)
    he_labels = np.asarray(he_labels).astype(int)
    classes = np.arange(he_labels.max() + 1)
    colours = he_colours(he_labels)
    if he_frame is None:
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            s=spot_size(len(coords)),
            c=np.array(colours)[he_labels],
            linewidth=0,
            rasterized=rasterize(len(coords)),
        )
    else:
        to_axes, at, nearest, on = slide_grid(he_frame, coords)
        rgb = np.array([mcolors.to_rgb(c) for c in colours])[he_labels[nearest]]
        shape = (int(at[:, 0].max()) + 1, int(at[:, 1].max()) + 1)
        draw_image(ax, to_axes, at[on], rgb[on], shape)
        # NB the spots' extent frames the panel (`spatial_row`), as the slide's beside it.
        ax.dataLim.update_from_data_xy(coords, ignore=True)
    return [square_handle(colours[k]) for k in classes], [f"H&E {k}" for k in classes]


LATTICE_TOLERANCE = (1e-6, 8.0, 0.25)
"""`get_lattice_tolerance`'s floor, its multiple of the places' noise and its ceiling, in cells."""


def get_lattice_tolerance(residual: np.ndarray, xy: np.ndarray) -> float:
    """The offset, in cells, within which spots are one lattice: 8 times the places' noise, estimated from neighbouring
    spots' offsets (their wrapped differences' median absolute value, as a standard deviation), held to [1e-6, 0.25].
    Exact places keep 1e-6; measured ones, as Visium HD's from microscope pixels, take their own noise, so a slice is one
    lattice; slices placed side by side are not neighbours, so their offsets stay apart.

    Args:
        residual: `(n_spots, 2)` each spot's offset from the lattice, in cells, in [-1/2, 1/2].
        xy: `(n_spots, 2)` the spots' places.

    Returns:
        The tolerance.
    """
    floor, multiple, ceiling = LATTICE_TOLERANCE
    distance, nearest = cKDTree(xy).query(xy, k=min(5, len(xy)))
    near = distance[:, 1:] <= 1.5 * np.median(distance[:, 1])
    apart = residual[nearest[:, 1:]] - residual[:, None]
    apart = np.abs(apart - np.round(apart))[near]
    noise = 1.4826 * float(np.median(apart)) if apart.size else 0.0
    return float(np.clip(multiple * noise, floor, ceiling))


def get_lattice(xy: np.ndarray) -> np.ndarray:
    """The basis of the spots' lattice, any two-dimensional lattice (square, rectangular, hexagonal, oblique): the
    commonest shortest step between neighbours, then the commonest shortest step not along it, each the median of the
    steps that round to it at a hundredth of the pitch.

    Args:
        xy: `(n_spots, 2)` the spots' places, at least 3 not in a line.

    Returns:
        `(2, 2)` the basis vectors as columns, in `xy`'s units.
    """
    distance, nearest = cKDTree(xy).query(xy, k=min(9, len(xy)))
    steps = (xy[nearest[:, 1:]] - xy[:, None]).reshape(-1, 2)
    # NB one sign per direction, so a step and its reverse are one vector.
    steps = np.where((steps[:, :1] < 0) | ((steps[:, :1] == 0) & (steps[:, 1:] < 0)), -steps, steps)
    quantum = float(np.median(distance[:, 1])) / 100.0
    vectors, counts = np.unique(np.round(steps / quantum), axis=0, return_counts=True)
    lengths = np.hypot(*vectors.T)
    first = vectors[np.lexsort((lengths, -counts))[0]]
    across = np.abs(first[0] * vectors[:, 1] - first[1] * vectors[:, 0]) > 0.1 * np.hypot(*first) * lengths
    second = vectors[across][np.lexsort((lengths[across], -counts[across]))[0]]
    # NB each the median of the steps it rounds: steps within a slice are its lattice's, so slices sit off nothing.
    rounded = np.round(steps / quantum)
    return np.column_stack([np.median(steps[(rounded == v).all(axis=1)], axis=0) for v in (first, second)])


def draw_spot_image(ax: Any, xy: np.ndarray, rgba: np.ndarray) -> None:
    """Draw each spot as one cell of an image on the spots' lattice (`get_lattice`), the cell the parallelogram its
    basis spans, centred on the spot, the cells with no spot clear. Cells tile the plane without overlap on any lattice,
    so no spot covers a neighbour, and a PDF embeds the image unsampled (`interpolation="none"`), one cell per spot,
    sharp at any zoom. Spots off one another's lattice by more than `get_lattice_tolerance` of a cell, as slices placed side by side, are
    drawn one image per lattice.

    Args:
        ax: the axes.
        xy: `(n_spots, 2)` the spots' places.
        rgba: `(n_spots, 4)` their colours.

    Raises:
        ValueError: two spots fall in one cell.
    """
    basis = get_lattice(xy)
    uv = np.linalg.solve(basis, (xy - xy[0]).T).T
    residual = uv - np.round(uv)
    tolerance = get_lattice_tolerance(residual, xy)
    unplaced = np.ones(len(xy), dtype=bool)
    while unplaced.any():
        key = residual[np.argmax(unplaced)]
        # NB one lattice per offset to `tolerance` of a cell, wrapped at a whole cell.
        apart = residual - key
        on = unplaced & (np.abs(apart - np.round(apart)) <= tolerance).all(axis=1)
        unplaced &= ~on
        # NB the lattice's own offset, the mean of its spots' about `key`.
        offset = np.mean(uv[on] - np.round(uv[on] - key), axis=0)
        cells = np.round(uv[on] - offset).astype(int)
        start = cells.min(axis=0)
        cells -= start
        if len(np.unique(cells, axis=0)) != on.sum():
            raise ValueError("spots share a lattice cell")
        image = np.zeros((cells[:, 1].max() + 1, cells[:, 0].max() + 1, 4))
        image[cells[:, 1], cells[:, 0]] = rgba[on]
        # NB image column c, row r is centred at xy[0] + basis @ (start + offset + (c, r)).
        origin = xy[0] + basis @ (start + offset)
        to_axes = Affine2D(np.vstack([np.column_stack([basis, origin]), [0.0, 0.0, 1.0]]))
        height, width = image.shape[:2]
        ax.imshow(
            image,
            extent=(-0.5, width - 0.5, height - 0.5, -0.5),
            interpolation="none",
            transform=to_axes + ax.transData,
        )


def image_spots(ax: Any) -> None:
    """Redraw a placed, framed spatial panel's spot collections as one image (`draw_spot_image`), its frame kept.

    Args:
        ax: the panel, its collections its spots.
    """
    collections = [c for c in ax.collections if len(c.get_offsets())]
    xy = np.vstack([np.ma.getdata(c.get_offsets()) for c in collections])
    rgba = np.vstack([np.broadcast_to(c.get_facecolors(), (len(c.get_offsets()), 4)) for c in collections])
    limits = ax.get_xlim(), ax.get_ylim()
    for collection in collections:
        collection.remove()
    draw_spot_image(ax, xy, rgba)
    # NB imshow autoscales; the frame is spatial_row's.
    ax.set_xlim(limits[0])
    ax.set_ylim(limits[1])


def clone_key(ax: Any) -> tuple[list[Any], list[str]]:
    """Take `plot_clones_spatial`'s key as square handles, removing its own from the panel.

    Args:
        ax: the clones panel.

    Returns:
        The handles and labels.
    """
    legend = ax.get_legend()
    handles = [square_handle(h.get_markerfacecolor()) for h in legend.legend_handles]
    labels = [t.get_text() for t in legend.get_texts()]
    legend.remove()
    return handles, labels


def extent(ax: Any) -> tuple[float, float, float, float]:
    """Return the spots' extent on `ax` as drawn, `(x0, x1, y0, y1)` on their physical `(x, y)`.

    Args:
        ax: a spatial panel.

    Returns:
        The extent.
    """
    (x0, x1), (y0, y1) = ax.dataLim.intervalx, ax.dataLim.intervaly
    return x0, x1, y0, y1


def coordinates(values: Any, thousands: bool | None = None) -> list[str]:
    """Format spatial ticks: in thousands where `thousands` or any reaches 1,000, else one decimal.

    In thousands the unit is written once, on the outermost tick (`0, 10, 20, 30K`); the inner ticks are bare numbers
    on the same scale. Adds a decimal to every tick while two would read alike.

    Args:
        values: the tick values.
        thousands: to force thousands.

    Returns:
        The tick labels.
    """
    values = [float(v) for v in values]
    if not (max(abs(v) for v in values) >= 1_000 if thousands is None else thousands):
        # NB whole numbers plain: `frame_extent` puts every tick on a round step, so a decimal says nothing.
        return [f"{v:.0f}" if all(float(x).is_integer() for x in values) else f"{v:.1f}" for v in values]
    outer = int(np.argmax([abs(v) for v in values]))
    for digits in range(4):
        out = ["0" if v == 0 else f"{v / 1_000:.{digits if abs(v) >= 1_000 else max(digits, 1)}f}" for v in values]
        if len(set(out)) == len(set(values)):
            break
    out[outer] += "K" if values[outer] != 0 else ""
    return out


def rounded_end(reach: float) -> tuple[float, float]:
    """The tick step and the end of a spatial axis reaching `reach`: a round number at or above it, from zero.

    1K up to 10K, 5K up to 50K, 10K beyond, so a panel carries three or four ticks whatever the slide's units:
    1,000 ends at 1K, 26K at 30K, 48K at 50K.

    Below 1,000 the coordinates are a lattice's own rather than a slide's pixels -- a simulation's grid runs to tens
    or hundreds -- and the step is the smallest of 1, 2 and 5 times a power of ten that ends the axis within four
    steps: 59 ends at 60 by 20, 140 at 150 by 50, 500 at 600 by 200. A 1K step there draws the spots into a corner.

    Args:
        reach: the larger of the panel's two extents, in the coordinates' own units.

    Returns:
        The step and the end.
    """
    if reach < 1_000.0:
        reach = max(reach, 1e-9)
        power = 10.0 ** np.floor(np.log10(reach / 4.0))
        step = next(float(m * power) for m in (1.0, 2.0, 5.0, 10.0) if np.ceil(reach / (m * power)) <= 4)
    elif reach <= 10_000.0:
        step = 1_000.0
    else:
        step = 5_000.0 if reach <= 50_000.0 else 10_000.0
    return step, float(np.ceil(reach / step) * step)


def framed(box: tuple[float, float, float, float]) -> tuple[float, float, float, float, float]:
    """The frame `frame_extent` draws on `box`: each axis from the round step at or below its least coordinate to the
    one at or above its greatest, one step for both, from the smaller span.

    Args:
        box: the spots' physical `(x0, x1, y0, y1)`.

    Returns:
        `(start_x, end_x, start_y, end_y, step)`.
    """
    x0, x1, y0, y1 = box
    # NB one step for both axes, from the smaller span: the panel is drawn at equal aspect, so two steps would put two
    #    different grids on one square, and a step from the larger span ends the shorter axis far past its spots.
    step, _ = rounded_end(min(x1 - x0, y1 - y0))
    ends = (np.floor(x0 / step), np.ceil(x1 / step), np.floor(y0 / step), np.ceil(y1 / step))
    return (*(float(e * step) for e in ends), step)


def frame_extent(ax: Any, box: tuple[float, float, float, float], labels: bool, pad: float = 0.02) -> None:
    """Frame `ax` on `box` padded by `pad` per side, aspect equal, ticks on round steps from each axis's start to end.

    `box` is the spots' physical `(x0, x1, y0, y1)` as drawn, y up (`platforms.visium.physical_coords`); each axis runs from the
    round step at or below its least coordinate to the one at or above its greatest, and its ticks read the
    coordinates themselves.

    Args:
        ax: the axes.
        box: `(x0, x1, y0, y1)`.
        labels: whether to label the y ticks, which panels side by side share (every panel labels its x ticks).
        pad: the fraction of each side.
    """
    start_x, end_x, start_y, end_y, step = framed(box)
    dx, dy = pad * (end_x - start_x), pad * (end_y - start_y)
    ax.set_xlim(start_x - dx, end_x + dx)
    ax.set_ylim(start_y - dy, end_y + dy)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("on")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_bounds(start_x, end_x)
    ax.spines["left"].set_bounds(start_y, end_y)
    xs = np.arange(start_x, end_x + step / 2.0, step)
    ys = np.arange(start_y, end_y + step / 2.0, step)
    ax.set_xticks(xs)
    ax.set_yticks(ys)
    thousands = max(abs(start_x), abs(end_x), abs(start_y), abs(end_y)) >= 1_000
    ax.set_xticklabels(coordinates(xs, thousands))
    ax.set_yticklabels(coordinates(ys, thousands) if labels else [])


def spatial_row(
    figure: Any, top: float, panels: list[Any], keys: list[tuple[list[Any], list[str]]], most: float
) -> float:
    """Place `panels` side by side `top` inches down on the last one's extent, `keys` in a column to their right.

    Each panel is as wide as the row allows and at most `most` inches tall.

    Args:
        figure: the page.
        top: inches.
        panels: drawn axes.
        keys: handles and labels.
        most: the height cap.

    Returns:
        The row's height in inches.
    """
    box = extent(panels[-1])
    # NB the ratio of the framed panel, not of the spots: `frame_extent` runs each axis between round steps
    #    (`framed`), and a panel placed at the spots' ratio would be drawn at an aspect its own limits then fight.
    start_x, end_x, start_y, end_y, _ = framed(box)
    ratio = (end_y - start_y) / (end_x - start_x)
    # NB one key, each panel's entries in turn, a blank entry between them.
    handles, labels = [], []
    for k, (h, l) in enumerate(keys):
        if k:
            handles.append(Line2D([], [], linestyle="none"))
            labels.append("")
        handles, labels = handles + list(h), labels + list(l)
    options = {
        "loc": "lower left",
        "frameon": False,
        "handlelength": 0.8,
        "handletextpad": 0.4,
        "labelspacing": 0.3,
        "borderaxespad": 0.0,
    }
    # NB the key as drawn sets its column: a clone's label is its name and two percentages, and a panel that fills
    #    the row (a lattice wider than tall) leaves the key no slack past a fixed column.
    key = panels[-1].legend(handles, labels, **options)
    drawn = key.get_window_extent(figure.canvas.get_renderer()).width / figure.dpi
    column = max(KEY_COLUMN, GAP + drawn)
    room = figure.get_size_inches()[0] - MARGIN["left"] - MARGIN["right"] - column - (len(panels) - 1) * PANEL_GAP
    width = min(room / len(panels), most / ratio)
    for k, ax in enumerate(panels):
        frame_extent(ax, box, labels=k == 0)
        place(ax, MARGIN["left"] + k * (width + PANEL_GAP), top, width, width * ratio)
    key.set_bbox_to_anchor((1.0 + GAP / width, 0.0), transform=panels[-1].transAxes)
    return width * ratio


def name_line(ax: Any) -> None:
    """Put the clone's name alone above its tracks, its state key on the same line at the right.

    Args:
        ax: the clone's first track.
    """
    names = [t for t in ax.texts if t.get_rotation() == 90.0]
    stats = [t for t in ax.texts if t.get_rotation() == 0.0 and t.get_position() == (0.0, 1.02)]
    if names and stats:
        stats[0].set_text(names[0].get_text())
        for t in names:
            t.remove()
    legend = ax.get_legend()
    if legend is not None:
        handles, labels = legend.legend_handles, [t.get_text() for t in legend.get_texts()]
        for handle in handles:
            handle.set_markersize(3)
        # NB the key sits right of the name on one line; a clone with many states (six on a dense genome) runs it
        #    into the name. Narrower column spacing first, then two rows, so the two never meet.
        renderer = ax.figure.canvas.get_renderer()
        name = stats[0].get_window_extent(renderer).x1 if names and stats else ax.get_window_extent(renderer).x0
        for spacing, rows in ((1.0, 1), (0.6, 1), (0.3, 1), (1.0, 2)):
            key = ax.legend(
                handles,
                labels,
                loc="lower right",
                bbox_to_anchor=(1.0, 1.0),
                ncol=-(-len(labels) // rows),
                frameon=False,
                handlelength=0.6,
                handletextpad=0.3,
                columnspacing=spacing,
                borderaxespad=0.0,
                borderpad=0.0,
            )
            if key.get_window_extent(renderer).x0 > name:
                break


def track_furniture(ax: Any, name: str) -> None:
    """Keep only the end y ticks, and name the track `name` beside them.

    The bottom label's digits rest on its tick (baseline-aligned: digits have no descender) and the top label's
    hang from its tick (baseline lowered by the digits' height, read from the glyphs), so both stay within the
    track's height and one track's 0.0 does not meet the top label of the track below.

    Args:
        ax: the track.
        name: its label.
    """
    low, high = ax.get_yticks()[[0, -1]]
    ax.set_yticks([low, high])
    bottom, top = ax.set_yticklabels([f"{low:.1f}", f"{high:.1f}"])
    for label in (bottom, top):
        label.set_verticalalignment("baseline")
    top.set_transform(top.get_transform() + HangFromTick(top))
    ax.set_ylabel(name)


def draw_profile(figure: Any, top: float, df_cnv: pd.DataFrame, genome: GenomicAxis, order: Any) -> float:
    """Draw the copy-number profile `top` inches down, across the page's margins: the state key over one row per clone
    of `order`, each named from the page's left edge, as the panel letters.

    Args:
        figure: the page.
        top: inches.
        df_cnv: copy numbers, `clone<id> A` and `clone<id> B` per bin of `genome`.
        genome: the genomic axis.
        order: clone ids, top to bottom.

    Returns:
        The inches down to the panel below.
    """
    width = figure.get_size_inches()[0] - MARGIN["left"] - MARGIN["right"]
    plot_state_key(add_axes(figure, MARGIN["left"], top, width, PROFILE_KEY), used_states(df_cnv, order))
    top += PROFILE_KEY + GAP
    profile_ax = add_axes(figure, MARGIN["left"], top, width, PROFILE_ROW * len(order))
    plot_state_profile(df_cnv, profile_ax, genome, order=order)
    for label in profile_ax.get_yticklabels():
        label.set_horizontalalignment("left")
    profile_ax.tick_params(axis="y", pad=72.0 * (MARGIN["left"] - 0.02))
    return top + PROFILE_ROW * len(order) + GAP


def frame_tracks(axes: list[Any], per: int) -> None:
    """Frame `plot_clones_genomic`'s tracks, `per` a clone (RDR then BAF, or BAF alone): end ticks, each named, and
    the clone's name over its first track (`track_furniture`, `name_line`).

    Args:
        axes: the tracks, clone by clone.
        per: tracks per clone, 1 or 2.
    """
    for k, ax in enumerate(axes):
        track_furniture(ax, ("RDR", "BAF")[k % per] if per == 2 else "BAF")
        if k % per == 0:
            name_line(ax)


class HangFromTick(Transform):
    """Lower `label` by its digits' height, so they hang from its anchor; read at every draw, as the page's fonts are
    set after layout (`uniform`) and again on the way out (`restyle`).

    Args:
        label: the text.
    """

    input_dims = output_dims = 2
    is_affine = False

    def __init__(self, label: Any) -> None:
        super().__init__()
        self.label = label

    def transform_non_affine(self, values: Any) -> Any:
        height = TextPath((0, 0), self.label.get_text(), prop=self.label.get_fontproperties()).get_extents().y1
        return np.asarray(values, dtype=float) - [0.0, height * self.label.figure.dpi / 72.0]


DECODED = "decoded"
fitted_states_real = "real"
PAGES = (DECODED, fitted_states_real)
"""`plot_combined_page`'s two pages: the integer copies the decoder found, or the fit's own states before it."""


def plot_combined_page(
    counts: Any,
    df_cnv: pd.DataFrame,
    res_combine: dict[str, Any],
    physical_coords: np.ndarray,
    assignment: pd.Series,
    sample_list: list[str] | None = None,
    sample_ids: np.ndarray | None = None,
    he_frame: pd.DataFrame | None = None,
    int_groups: np.ndarray | None = None,
    normal_fractions: dict[Any, float] | None = None,
    states: str = DECODED,
    field_assignment: pd.Series | None = None,
) -> Any:
    """Draw the full page: (a) H&E slide, clones map and the field's own labelling, (b) the copy-number profile, (c) each clone's tracks.

    `states` says which page: `DECODED`, the integer copies the decoder found, profiled in (b), with the tracks
    carrying their states; or `fitted_states_real`, the RDR and BAF fit before any decode -- no copy-number profile at all, and
    the tracks carrying the data alone. `fitted_states_real` draws the inferred clones whatever `int_groups` says, because
    the integer clones are a grouping the decode made.

    Given `int_groups`, the clones map, the profile and the tracks draw the integer clones (`clone_labels_int.tsv`),
    each named by and drawn at its first clone, the tracks pooling its spots; else the inferred clones. Tracks lie on
    the profile's genomic axis. Arguments as `plot_clones_genomic` and `plot_clones_spatial` take them in run_cnamaste.

    `field_assignment` (`run.plot_field_argmax`) adds a third panel to the top row: the same spots labelled by the
    field alone, with the Potts coupling dropped, in the clones map's own colours, so what the spatial prior did is
    the difference between the two. It is the integer clones' labelling, so `fitted_states_real` carries none.

    The spots and the slide are drawn at the spots' physical `(x, y)`, um, y up (`platforms.visium.physical_coords`): the slide's
    orientation, on Visium and Visium HD alike. Each spot is one cell of an image on its lattice (`image_spots`).

    Args:
        counts: a `SpatioGenomicLabelledMultiCounts`.
        df_cnv: the segment-level copies.
        res_combine: the fit.
        physical_coords: (n_spots, 2) um.
        assignment: the clone per spot.
        sample_list: the slices' names.
        sample_ids: the slice per spot.
        he_frame: the slide (`he.slide_image_and_meta`), or None.
        int_groups: each inferred clone's integer clone, its first clone's id (`get_merged_ab_decoded_clones`), or None.
        normal_fractions: clone label -> the decoder's normal fraction, which the key carries as `(XX%)`.
        states: `DECODED` or `fitted_states_real`.
        field_assignment: the field's own label per spot, or None.

    Returns:
        The figure.
    """
    if states not in PAGES:
        raise ValueError(f"states {states!r} is not one of {PAGES}")
    int_groups = None if states == fitted_states_real else int_groups
    labels = (
        np.asarray(res_combine["new_assignment"]) if int_groups is None else int_groups[res_combine["new_assignment"]]
    )
    int_assignment = None if int_groups is None else pd.Series([f"clone {x}" for x in labels])
    # NB the field's labelling is the integer clones', as `clone_labels_int.tsv` writes it, so the page before the
    #    decode carries none.
    field_assignment = None if states == fitted_states_real else field_assignment
    genome = GenomicAxis.from_bins(df_cnv["CHR"])
    n_clones = len(np.unique(labels))
    # TODO move to spatio_genomic_counts: whether a run carries RDR; plot_genomic.plot_clones_genomic decides it again.
    has_rdr = counts.normal_baseline_sized.count_positive() > 0
    per = 2 if has_rdr else 1

    with page_style() as size:
        figure = plt.figure(figsize=page_size("full"), dpi=get_dpi(), facecolor="white")
        page_width, page_height = figure.get_size_inches()
        width = page_width - MARGIN["left"] - MARGIN["right"]

        top = MARGIN["top"]
        letter(figure, top, "a")
        top += LETTER
        # NB the slide's panel only where there is a slide; then one clone map, the integer clones where given, each
        #    group in its first clone's colour, so the tracks below read in the same colours; then the field's own
        #    labelling where a caller passes one.
        wanted = (1 if he_frame is None else 2) + (0 if field_assignment is None else 1)
        panels = [add_axes(figure, MARGIN["left"], top, 1.0, 1.0) for _ in range(wanted)]
        clones = panels[0] if he_frame is None else panels[1]
        ids = list(np.unique(assignment[~assignment.isnull()].values))
        colours = get_clone_colours(ids, get_clone_palette())
        if int_assignment is None:
            plot_clones_spatial(
                physical_coords,
                assignment,
                sample_ids=sample_ids,
                sample_list=sample_list,
                ax=clones,
                normal_fractions=normal_fractions,
            )
        else:
            groups = np.unique(int_assignment[~int_assignment.isnull()].values)
            plot_clones_spatial(
                physical_coords,
                int_assignment,
                sample_ids=sample_ids,
                sample_list=sample_list,
                ax=clones,
                normal_fractions=normal_fractions,
                colorlist=[colours[ids.index(g)] for g in groups],
            )
        keys = [clone_key(clones)]
        for text in list(clones.texts):
            text.remove()
        if field_assignment is not None:
            # NB the clones map's colours, by name, so a clone is one colour across the row whatever share it holds
            #    under each labelling; the key is the clones map's alone, which those shares belong to.
            named = list(np.unique(field_assignment[~field_assignment.isnull()].values))
            plot_clones_spatial(
                physical_coords,
                field_assignment,
                sample_ids=sample_ids,
                sample_list=sample_list,
                ax=panels[-1],
                colorlist=[colours[ids.index(g)] for g in named],
            )
            panels[-1].get_legend().remove()
            for text in list(panels[-1].texts):
                text.remove()
        if he_frame is not None:
            draw_slide(panels[0], he_frame, physical_coords)
        top += spatial_row(figure, top, panels, keys, ROW) + TICK_LINE
        for panel in panels[1 if he_frame is not None else 0 :]:
            image_spots(panel)

        # NB fitted_states_real carries no copy-number profile: it is the decode's own panel, and the page is the fit before it.
        if states == DECODED:
            letter(figure, top, "b")
            # NB the tracks' order, clone index from the normal clone down, on the profile's rows too.
            top = draw_profile(figure, top, df_cnv, genome, np.unique(labels))

        letter(figure, top, "c" if states == DECODED else "b")
        # NB the normal clone, first, draws RDR to NORMAL_RDR_YLIM on a track shortened in proportion; the other
        #    tracks share what it frees.
        shrink = (NORMAL_RDR_YLIM - get_rdr_floor()) / (get_rdr_ylim() - get_rdr_floor()) if has_rdr else 1.0
        fixed = n_clones * (NAME_LINE + GAP / 2.0 + (per - 1) * TRACK_GAP)
        track = (page_height - top - MARGIN["bottom"] - fixed) / (n_clones * per - 1 + shrink)
        axes, at = [], top
        for c in range(n_clones):
            at += NAME_LINE
            for t in range(per):
                height = track * shrink if c == 0 and t == 0 else track
                axes.append(add_axes(figure, MARGIN["left"], at, width, height))
                at += height + (TRACK_GAP if t < per - 1 else GAP / 2.0)
        # NB fitted_states_real plots the data alone: `df_cnv` None is `plot_clones_genomic`'s own switch for no integer states.
        plot_clones_genomic(
            counts,
            df_cnv=None if states == fitted_states_real else df_cnv,
            res_combine=res_combine,
            sample_list=sample_list,
            axes=axes,
            genome=genome,
            normal_fractions=None if states == fitted_states_real else normal_fractions,
            clone_labels=labels,
        )
        if has_rdr:
            axes[0].set_ylim(get_rdr_floor(), NORMAL_RDR_YLIM)
            axes[0].set_yticks(np.arange(0.0, NORMAL_RDR_YLIM + 1.0))
        frame_tracks(axes, per)

        uniform(figure, size)
    return figure


def plot_he_page(physical_coords: np.ndarray, he_labels: np.ndarray, he_frame: pd.DataFrame | None = None) -> Any:
    """Draw a third of a page: (a) the H&E slide, a blank frame where none, (b) each spot's H&E class, keyed right;
    with a slide, both on its pixels, white off the tissue (`draw_slide`, `draw_classes`).

    Args:
        physical_coords: (n_spots, 2) um, y up.
        he_labels: per-spot classes.
        he_frame: the slide (`he.slide_image_and_meta`, `x, y` um) or None.

    Returns:
        The figure.
    """
    with page_style() as size:
        figure = plt.figure(figsize=page_size("third"), dpi=get_dpi(), facecolor="white")
        top = MARGIN["top"] + LETTER
        slide_ax, classes_ax = (add_axes(figure, MARGIN["left"], top, 1.0, 1.0) for _ in range(2))
        if he_frame is not None:
            draw_slide(slide_ax, he_frame, physical_coords)
        he_key = draw_classes(classes_ax, physical_coords, he_labels, he_frame)
        most = figure.get_size_inches()[1] - top - 0.3
        spatial_row(figure, top, [slide_ax, classes_ax], [he_key], most)
        letter(figure, MARGIN["top"], "a")
        letter(figure, MARGIN["top"], "b", x0=classes_ax.get_position().x0 * figure.get_size_inches()[0])
        uniform(figure, size)
    return figure


AXIS_LABEL = 0.2
"""Inches under a panel's tick labels for its axis label."""

TICK_OVERHANG = 0.15
"""Inches right of a panel for half of its last x tick label, which the page's right margin does not hold."""

UMI_RANGE = (10.0, 1.0e4)
"""UMIs per spot, the x range of `plot_spot_umi_page`'s panels, on a log axis."""

UMI_BINS = 60
"""Log-spaced bins over `UMI_RANGE`."""


def plot_spot_umi_page(spot_umis: np.ndarray, snp_umis: np.ndarray) -> Any:
    """Draw a third of a page: histograms of each retained spot's (a) UMIs and (b) SNP-UMIs, side by side, on
    `UMI_BINS` log-spaced bins over `UMI_RANGE`, in the validation figures' colours: UMIs in `figure_style.BOX` and
    `WHISKER`, SNP-UMIs in `COMPARED_BOX` and `COMPARED_WHISKER`, each at `BOX_ALPHA`.

    Args:
        spot_umis: (n_spots,) each retained spot's UMIs.
        snp_umis: (n_spots,) each retained spot's SNP-covering UMIs.

    Returns:
        The figure.
    """
    with page_style() as size:
        figure = plt.figure(figsize=page_size("third"), dpi=get_dpi(), facecolor="white")
        page_width, page_height = figure.get_size_inches()
        top = MARGIN["top"] + LETTER
        width = (page_width - MARGIN["left"] - MARGIN["right"] - TICK_OVERHANG - PANEL_GAP) / 2
        height = page_height - top - MARGIN["bottom"] - AXIS_LABEL
        panels = (
            (spot_umis, "UMIs per spot", get_box(), get_whisker()),
            (snp_umis, "SNP-UMIs per spot", get_compared_box(), get_compared_whisker()),
        )
        for k, (counts, label, face, edge) in enumerate(panels):
            x0 = MARGIN["left"] + k * (width + PANEL_GAP)
            ax = add_axes(figure, x0, top, width, height)
            ax.hist(
                np.asarray(counts, dtype=np.float64),
                bins=np.geomspace(*UMI_RANGE, UMI_BINS + 1),
                facecolor=face,
                alpha=get_box_alpha(),
                edgecolor=edge,
                linewidth=0.5,
            )
            ax.set_xscale("log")
            ax.set_xlim(*UMI_RANGE)
            ax.set_xlabel(label)
            ax.set_ylabel("# spots")
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            letter(figure, MARGIN["top"], "ab"[k], x0=x0 - MARGIN["left"] + 0.02)
        uniform(figure, size)
    return figure
