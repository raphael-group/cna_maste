"""`plot_annotation_page`: one Visium HD sample's (a) H&E slide, cell types and each method's clones, (b) where given, a
method's per-clone pseudobulk RDR and BAF and their expectation, and (c) each method's clone profile, on cnplot's
`GenomeAxis`; the profiles are (b) where no tracks are given.
"""

from __future__ import annotations

from typing import Any

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D, offset_copy
from scipy.spatial import cKDTree

from cnamaste.config import start_time
from cnamaste.figure_style import CLONE_PALETTE, DPI, page_size, page_style, uniform
from cnamaste.logger import get_logger
from cnamaste.plot.layout import (
    GAP,
    LETTER,
    MARGIN,
    NAME_LINE,
    TICK_LINE,
    TRACK_GAP,
    add_axes,
    draw_slide,
    frame_extent,
    framed,
    letter,
    place,
    square_handle,
    track_furniture,
)
from cnamaste.utils import cast_clone_label

# NB cnplot writes its own rcParams on import (`cnplot_style.apply_style`); the context restores cnamaste's.
with mpl.rc_context():
    from cnplot import (
        BAF_LIM,
        NORMAL_CLONE,
        GenomeAxis,
        chrname_tiers,
        decorate_genome_axis,
        genome_label_room,
        get_cn_cmap,
        plot_cnv_profile,
        plot_heatmap,
        plot_scatter_1d,
        resolve_ylim_scaled,
    )

logger = get_logger(__name__, start_time=start_time)


def get_lattice(xy: np.ndarray) -> tuple[float, float]:
    """The spots' square lattice: its pitch, the median distance to a spot's nearest neighbour (the spacing `slide_grid`
    reads a tissue pixel by), and its turn, the mean direction of those neighbours, modulo 90 degrees.

    Args: `xy` (n_spots, 2) the spots' places.
    Returns: the pitch, in `xy`'s units, and the turn in degrees.
    """
    distance, nearest = cKDTree(xy).query(xy, k=2)
    pitch = float(np.median(distance[:, 1]))
    step = xy[nearest[:, 1]] - xy
    # NB the mean direction modulo 90 degrees, on the circle: 4 theta, so 0 and 90 degrees are one direction.
    # NB rounded before the modulo, so a direction a rounding error below 0 reads 0 rather than 90.
    turn = float(
        np.round(np.degrees(np.angle(np.mean(np.exp(4j * np.arctan2(step[:, 1], step[:, 0]))))) / 4.0, 6) % 90.0
    )
    return pitch, turn


TITLE_LINE = 0.16
"""Inches above a panel of `plot_annotation_page` for its title."""

AXIS_LABEL_LINE = 0.16
"""Inches outside a spatial grid's tick labels for its axis label."""

CLONE_ROW = 0.2
"""Inches of profile per tumour clone on `plot_annotation_page`."""

CHR_OVERHANG = 0.2
"""Inches right of the profiles, where cnplot nudges the last, crowded chromosome names past the genome's end."""

CN_KEY = 0.75
"""Inches below the profiles for cnplot's copy-number key: its axes, `CN_KEY_AXES`, and the rows it wraps below them."""

CN_KEY_AXES = 0.3
"""Inches of cnplot's copy-number key axes, its first row of swatches."""

STATE_KEY = 0.35
"""Inches below the profiles for a key of categorical states (`draw_state_key`)."""

PSEUDOBULK_TRACK = 0.55
"""Inches of one pseudobulk track, RDR or BAF, on `plot_annotation_page`."""

PSEUDOBULK_POINT = 3.0
"""Points squared of a pseudobulk bin's dot: `plot_clones_genomic`'s `pointsize`, the combined page's tracks at this
width."""

PSEUDOBULK_LINE = 0.5
"""Points: the width of a pseudobulk track's expected line, `plot_clones_genomic`'s."""


def get_genome_axis(region_bed: str, chrom_sizes: str) -> GenomeAxis:
    """cnplot's genome axis on `region_bed`'s regions of `chrom_sizes`' chromosomes, in its order, the stretches between
    regions collapsed, X, Y and M left out (cnplot's `DEFAULT_EXCLUDED_CHROMS`).

    Args: `region_bed` the regions drawn, e.g. `python/cnamaste/sandbox/luo/benchmark/data/hg38.regions.bed`, a chromosome's arms; `chrom_sizes`
        a headerless `<chromosome> <length>` file, e.g. `python/cnamaste/sandbox/luo/benchmark/data/hg38.chrom.sizes`.
    Returns: the axis.
    """
    return GenomeAxis(region_bed, chrom_sizes)


def get_cnplot_bins(table: pd.DataFrame) -> pd.DataFrame:
    """`table` with its chromosomes named as cnplot names them, `#CHR`, `chr`-prefixed, in place of `CHR`.

    Args: `table` `CHR` and any columns.
    Returns: `table`'s other columns, then `#CHR`.
    """
    return table.drop(columns="CHR").assign(**{"#CHR": "chr" + table["CHR"].astype(str).str.removeprefix("chr")})


def get_cnplot_segments(table: pd.DataFrame, genome_axis: GenomeAxis) -> pd.DataFrame:
    """`table`'s segments on `genome_axis`: `#CHR` named as its chromosomes are, `chr`-prefixed, and each segment cut at
    its regions, one row per piece, the rest of the segment's columns kept.

    A segment through a centromere is so drawn on both arms; cnplot would drop it (`contain=True`) or keep one arm.

    Args: `table` `CHR`, `START`, `END` per segment, and any columns; `genome_axis` cnplot's axis (`get_genome_axis`).
    Returns: `#CHR`, `START`, `END` per piece, then `table`'s other columns.
    """
    regions = genome_axis.regions[["#CHR", "START", "END"]].rename(
        columns={"START": "region_start", "END": "region_end"}
    )
    pieces = get_cnplot_bins(table).merge(regions, on="#CHR")
    pieces = pieces[(pieces["START"] < pieces["region_end"]) & (pieces["END"] > pieces["region_start"])]
    pieces = pieces.assign(
        START=np.maximum(pieces["START"], pieces["region_start"]), END=np.minimum(pieces["END"], pieces["region_end"])
    )
    columns = ["#CHR", "START", "END"]
    return pieces[columns + [c for c in table.columns if c not in ("CHR", "START", "END")]].reset_index(drop=True)


def get_seg_ucn(segments: pd.DataFrame) -> pd.DataFrame:
    """`segments` as cnplot's seg.ucn profile: `#CHR`, `START`, `END` and `cn_<clone>`, `"A|B"` per clone.

    A clone is named as `cast_clone_label` names it (`Clone I`), but for clone 0 where it is (1, 1) on every segment:
    that is `NORMAL_CLONE`, which `cnplot.plot_cnv_profile` leaves out of its stack.

    Args: `segments` `get_cnplot_segments` of a `cnv_<ploidy>_seglevel.tsv`, `clone<k> A`, `clone<k> B` per clone.
    Returns: the seg.ucn frame.
    Raises: `ValueError` where a clone's copy number is missing.
    """
    ucn = segments[["#CHR", "START", "END"]].astype({"START": int, "END": int})
    for k in sorted(int(c.split()[0].removeprefix("clone")) for c in segments.columns if c.endswith(" A")):
        a, b = segments[f"clone{k} A"], segments[f"clone{k} B"]
        if a.isna().any() or b.isna().any():
            raise ValueError(f"clone {k} has segments with no copy number")
        a, b = a.astype(int), b.astype(int)
        name = NORMAL_CLONE if k == 0 and ((a == 1) & (b == 1)).all() else cast_clone_label(f"clone {k}")
        ucn[f"cn_{name}"] = a.astype(str) + "|" + b.astype(str)
    return ucn


def get_cn_states(a: Any, b: Any) -> tuple[np.ndarray, dict[str, Any]]:
    """Each bin's `A|B` and the colour cnplot's profile gives that pair (`get_cn_cmap`, a pair and its mirror alike),
    its default colour where the map has none.

    Args: `a`, `b` the integer copies per bin.
    Returns: the `A|B` label per bin, and label -> colour.
    """
    style_map, _ = get_cn_cmap()
    pairs = list(zip(np.asarray(a).astype(int).tolist(), np.asarray(b).astype(int).tolist()))
    labels = np.array([f"{x}|{y}" for x, y in pairs])
    return labels, {f"{x}|{y}": style_map.get((x, y), style_map["default"]) for x, y in set(pairs)}


def get_expected_segments(bins: pd.DataFrame) -> pd.DataFrame:
    """`bins`' expectation as segments: each run of consecutive bins of one chromosome reading one `exp_RDR` and one
    `exp_BAF` joined, from its first bin's start to its last's end, as `plot_clones_genomic` draws a line over each
    run of one `(A, B)`; cnplot joins only bins that abut, and the bins leave the stretches between genes out.

    Args: `bins` one clone's rows of `pseudobulk.get_clone_pseudobulk`, in genomic order.
    Returns: `CHR`, `START`, `END`, `exp_RDR`, `exp_BAF` per segment.
    """
    key = bins[["CHR", "exp_RDR", "exp_BAF"]].reset_index(drop=True)
    run = (key != key.shift()).any(axis=1).cumsum()
    return (
        bins.reset_index(drop=True)
        .groupby(run, sort=False)
        .agg(
            CHR=("CHR", "first"),
            START=("START", "first"),
            END=("END", "last"),
            exp_RDR=("exp_RDR", "first"),
            exp_BAF=("exp_BAF", "first"),
        )
        .reset_index(drop=True)
    )


def mirror_expected_baf(ax: Any) -> None:
    """Add the phase flip `1 - p` of the expected BAF cnplot drew on `ax`, in its colour, width and style, as
    `plot_clones_genomic` draws both.

    cnplot's expected overlay is the one `LineCollection` of horizontal segments its `plot_scatter_1d` adds; its
    chromosome boundaries are vertical.

    Args: `ax` a BAF track `plot_scatter_1d` drew with `expected_df`.
    Raises: `ValueError` where `ax` carries no such collection, or several.
    """
    found = [
        c
        for c in ax.collections
        if isinstance(c, LineCollection)
        and c.get_segments()
        and all(np.isclose(s[0][1], s[-1][1]) for s in c.get_segments())
    ]
    if len(found) != 1:
        raise ValueError(f"{len(found)} expected-BAF overlays on the track, not one")
    expected = found[0]
    flipped = [np.column_stack([s[:, 0], 1.0 - s[:, 1]]) for s in expected.get_segments()]
    ax.add_collection(LineCollection(flipped, colors=expected.get_colors(), linewidths=expected.get_linewidths()))


def draw_clone_tracks(rdr_ax: Any, baf_ax: Any, bins: pd.DataFrame, genome_axis: GenomeAxis) -> None:
    """Draw one clone's pseudobulk RDR and BAF per bin on `genome_axis` by cnplot's `plot_scatter_1d`, each bin in its
    decoded `(A, B)`'s colour on cnplot's profile (`get_cn_states`), a dotted line at RDR 1 and BAF 1/2, and the
    clone's expected RDR and BAF over them, black, as cnplot draws an expectation, a line per run of one expectation
    cut at the regions (`get_expected_segments`, `get_cnplot_segments`), and the expected BAF's phase flip `1 - p`
    (`mirror_expected_baf`).

    RDR's range is cnplot's for a ratio (`resolve_ylim_scaled`), the expectation kept in it, BAF's cnplot's `BAF_LIM`;
    each track ticked at its ends, 0 and the largest whole RDR in range or 1 BAF, as the combined page's
    (`track_furniture`); centromeres dotted (`dot_centromeres`), no chromosome names.

    Args: `rdr_ax`, `baf_ax` the tracks' axes; `bins` one clone's rows of `pseudobulk.get_clone_pseudobulk`, its
        `exp_RDR`, `exp_BAF` the expectation; `genome_axis` cnplot's axis.
    """
    labels, palette = get_cn_states(bins["A"], bins["B"])
    observed = get_cnplot_bins(bins)
    expected = get_cnplot_segments(get_expected_segments(bins), genome_axis)
    rdr_ylim = resolve_ylim_scaled(bins["RDR"].to_numpy(), expected=bins["exp_RDR"].to_numpy())
    for ax, ycol, ylim, href, ticks in (
        (rdr_ax, "RDR", rdr_ylim, 1.0, [0.0, float(np.floor(rdr_ylim[1]))]),
        (baf_ax, "BAF", BAF_LIM, 0.5, [0.0, 1.0]),
    ):
        plot_scatter_1d(ax, observed, genome_axis, ycol, expected_df=expected, hue=labels, palette=palette,
                        markersize=PSEUDOBULK_POINT, ylim=ylim, ylabel=ycol, href=href, plot_chrname=False,
                        exp_linewidth=PSEUDOBULK_LINE)  # fmt: skip
        ax.set_yticks(ticks)
        track_furniture(ax, ycol)
        dot_centromeres(ax, genome_axis)
    mirror_expected_baf(baf_ax)


def draw_spot_image(ax: Any, xy: np.ndarray, rgba: np.ndarray) -> None:
    """Draw each spot as one cell of an image on the spots' square lattice (`get_lattice`), in its colour, the cells
    with no spot clear.

    A cell is one pitch a side and cells tile without overlap, so no spot covers a neighbour, and the order the spots
    are given in draws nothing; squares drawn one by one at a panel's scale, a bin less than a device pixel across,
    cover their neighbours and leave the last drawn on top.

    Args: `ax` the axes; `xy` (n_spots, 2) the spots' places; `rgba` (n_spots, 4) their colours.
    Raises: `ValueError` where two spots fall in one cell, the spots on no one square lattice.
    """
    pitch, turn = get_lattice(xy)
    to_lattice = Affine2D().rotate_deg(-turn).scale(1.0 / pitch)
    uv = to_lattice.transform(xy)
    start = uv.min(axis=0)
    cells = np.round(uv - start).astype(int)
    if len(np.unique(cells, axis=0)) != len(cells):
        raise ValueError("spots share a lattice cell: they are on no one square lattice")
    image = np.zeros((cells[:, 1].max() + 1, cells[:, 0].max() + 1, 4))
    image[cells[:, 1], cells[:, 0]] = rgba
    to_axes = Affine2D().translate(*start).scale(pitch).rotate_deg(turn)
    height, width = image.shape[:2]
    ax.imshow(image, extent=(-0.5, width - 0.5, height - 0.5, -0.5), interpolation="none",
              transform=to_axes + ax.transData)  # fmt: skip


def draw_labels(ax: Any, physical: pd.DataFrame, labels: pd.Series, colours: dict[Any, str]) -> None:
    """Draw each spot of `labels` at its `physical` place in its label's colour (`draw_spot_image`).

    Args: `ax` the axes; `physical` `x, y` um per barcode; `labels` a label per barcode; `colours` label -> colour.
    """
    rgba = {label: mcolors.to_rgba(colour) for label, colour in colours.items()}
    xy = physical.loc[labels.index, ["x", "y"]].to_numpy(float)
    draw_spot_image(ax, xy, np.array([rgba[label] for label in labels.to_numpy()]))


def dot_centromeres(ax: Any, genome_axis: GenomeAxis) -> None:
    """Dot the lines cnplot dashes where a chromosome's regions part (`draw_segment_boundaries`): on
    `python/cnamaste/sandbox/luo/benchmark/data/hg38.regions.bed`'s arms, its centromere.

    Args: `ax` a profile cnplot drew on `genome_axis`; `genome_axis` its axis.
    """
    at = {gap.start for gap in genome_axis.interior_gaps}
    for line in ax.lines:
        xs = np.unique(line.get_xdata())
        if len(xs) == 1 and float(xs[0]) in at:
            line.set(linestyle=":", linewidth=0.8, alpha=1.0)


def draw_clone_states(
    ax: Any, segments: pd.DataFrame, genome_axis: GenomeAxis, colours: dict[str, str], clones: list[int]
) -> None:
    """Draw `clones`' categorical states on `genome_axis`, one row each, the first on top, each state in its colour, by
    cnplot's `plot_heatmap`; rows named as `cast_clone_label` names them.

    Args: `ax` the axes; `segments` `get_cnplot_segments` of a state table, `clone<k>` the state per clone;
        `genome_axis` cnplot's axis; `colours` state -> colour, every state drawn; `clones` the clones drawn, top to
        bottom.
    """
    order = list(colours)
    codes = np.array([[order.index(s) for s in segments[f"clone{k}"]] for k in reversed(clones)])
    names = [cast_clone_label(f"clone {k}") for k in reversed(clones)]
    plot_heatmap(ax, codes, segments[["#CHR", "START", "END"]], genome_axis, row_labels=np.array(names),
                 plot_chrname=False, cmap=mcolors.ListedColormap([colours[s] for s in order]),
                 norm=mcolors.BoundaryNorm(np.arange(len(order) + 1) - 0.5, len(order)),
                 show_block_labels=False)  # fmt: skip
    ax.set_yticks((np.arange(len(clones)) + 0.5) / len(clones), names)
    ax.tick_params(axis="y", left=True, length=4, pad=4)
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_state_key(ax: Any, colours: dict[str, str], name: str, swatch: tuple[float, float], indent: float) -> None:
    """Draw a key of `colours` on `ax`: one swatch per state, its name under it, and `name` left of the first.

    Args: `ax` the key's axes; `colours` state -> colour, in key order; `name` the key's label; `swatch` a swatch's
        width and height in inches; `indent` inches from the axes' left edge to the first swatch.
    """
    width, height = (
        ax.get_position().width * ax.figure.get_figwidth(),
        ax.get_position().height * ax.figure.get_figheight(),
    )
    ax.set_xlim(0.0, width)
    ax.set_ylim(0.0, height)
    ax.axis("off")
    for i, (state, colour) in enumerate(colours.items()):
        x = indent + i * (swatch[0] + GAP)
        ax.add_patch(Rectangle((x, height - swatch[1]), swatch[0], swatch[1], facecolor=colour, edgecolor="black"))
        ax.text(x + swatch[0] / 2.0, height - swatch[1] - 0.03, state, ha="center", va="top")
    ax.text(indent - GAP, height - swatch[1] / 2.0, name, ha="right", va="center")


def get_clone_names(labels: pd.Series, clone_colours: dict[int, str]) -> tuple[pd.Series, dict[str, str]]:
    """Each spot's clone by its name (`cast_clone_label`), and the colour of each name `labels` holds, in clone order.

    Args: `labels` a clone per barcode; `clone_colours` clone -> colour.
    Returns: the name per barcode, and name -> colour.
    """
    found = sorted(set(labels.astype(int)))
    names = {k: cast_clone_label(f"clone {k}") for k in found}
    return labels.astype(int).map(names), {names[k]: clone_colours[k] for k in found}


def plot_annotation_page(
    physical: pd.DataFrame,
    epithelial: pd.Series,
    clones: dict[str, pd.Series],
    profiles: dict[str, pd.DataFrame],
    genome_axis: GenomeAxis,
    state_colours: dict[str, str] | None = None,
    he_frame: pd.DataFrame | None = None,
    pseudobulk: dict[str, pd.DataFrame] | None = None,
) -> Any:
    """Draw a page the text column wide: (a) a grid of spatial panels two wide, the H&E slide, each spot's cell type,
    then each method's clones, each keyed right; where `pseudobulk` is given, (b) each of its methods' clones'
    pseudobulk RDR and BAF; then each method's clone profile, keyed below, (c) after the tracks, else (b).

    Spots are cells of an image on their lattice, one pitch a side, at their `physical` place (`draw_spot_image`),
    every panel framed on the extent of them all. A panel is as wide as the column leaves beside its column of keys,
    at the spots' framed aspect, and the page as tall as its panels need, so may run past `figure_style.TEXT_HEIGHT`.
    A clone is one colour in every panel, clone 0 light grey as in `plot_clones_spatial`; each panel's key names its
    own clones, each with its spots in the panel. A cell type is epithelial, in clone I's colour, or not, in clone 0's.

    A profile stacks the tumour clones, `CLONE_ROW` each, on `genome_axis`, segments cut at its regions
    (`get_cnplot_segments`), centromeres dotted (`dot_centromeres`): integer copies by cnplot's `plot_cnv_profile`
    (`get_seg_ucn`), its key carrying cnplot's mirrored swatch where the last of them hatched a mirrored bin; or, for a
    table of states, each segment in its clone's state's colour (`draw_clone_states`), its clone 0, the normal clone,
    left out, keyed by the states drawn. Every colour is opaque.

    The pseudobulk tracks stack every clone, clone 0 among them, named as (a)'s keys name them, an RDR and a BAF track
    each, on `genome_axis` as the profiles, each bin in its decoded `(A, B)`'s colour on cnplot's profile, so the
    profiles' key keys them, its expected RDR and BAF over them (`draw_clone_tracks`); the chromosomes named under the
    last.

    Args: `physical` `x, y` um, y up, per barcode, every spot drawn; `epithelial` whether a spot's cell type is
        epithelial, per barcode, the spots drawn; `clones` title -> clone per barcode, in panel order; `profiles` title
        -> `CHR`, `START`, `END` and either `clone<k> A`, `clone<k> B` or `clone<k>` state, in row order; `genome_axis`
        cnplot's axis (`get_genome_axis`); `state_colours` state -> colour, the key writing each state as named;
        `he_frame` the slide, `x, y` um (`draw_slide`), or None; `pseudobulk` title -> a method's
        `pseudobulk.get_clone_pseudobulk`, or None for no tracks.
    Returns: the figure.
    """
    segments = {title: get_cnplot_segments(profile, genome_axis) for title, profile in profiles.items()}
    categorical = {title: s for title, s in segments.items() if not any(c.endswith(" A") for c in s.columns)}
    integer = {title: get_seg_ucn(s) for title, s in segments.items() if title not in categorical}
    tumour: dict[str, list[Any]] = {
        title: sorted(int(c.removeprefix("clone")) for c in s.columns if c.startswith("clone") and c != "clone0")
        if title in categorical
        else [c[3:] for c in integer[title].columns if c.startswith("cn_") and c != f"cn_{NORMAL_CLONE}"]
        for title, s in segments.items()
    }
    drawn = {state for title, s in categorical.items() for k in tumour[title] for state in s[f"clone{k}"]}
    key_colours = {state: colour for state, colour in (state_colours or {}).items() if state in drawn}

    ids = sorted(set().union({0, 1}, *(set(c.astype(int)) for c in clones.values())))
    palette = sns.color_palette(CLONE_PALETTE, len([k for k in ids if k != 0])).as_hex()
    clone_colours = {k: "lightgrey" if k == 0 else palette[[i for i in ids if i != 0].index(k)] for k in ids}
    kinds = {True: "Epithelial", False: "Non-epithelial"}
    grid = [("H&E", None, {}, None), ("Cell type", epithelial.astype(bool).map(kinds),
            {kinds[True]: clone_colours[1], kinds[False]: clone_colours[0]}, "Cell type")]  # fmt: skip
    grid += [(title, *get_clone_names(labels, clone_colours), "Clone") for title, labels in clones.items()]
    box = (physical["x"].min(), physical["x"].max(), physical["y"].min(), physical["y"].max())
    start_x, end_x, start_y, end_y, _ = framed(box)
    ratio = (end_y - start_y) / (end_x - start_x)

    with page_style() as size:
        figure = plt.figure(figsize=page_size("full"), dpi=DPI, facecolor="white")
        page_width = figure.get_figwidth()
        width = page_width - MARGIN["left"] - MARGIN["right"]

        panels = [add_axes(figure, MARGIN["left"], MARGIN["top"], 1.0, 1.0) for _ in grid]
        if he_frame is not None:
            draw_slide(panels[0], he_frame, physical[["x", "y"]].to_numpy())
        options = {"loc": "upper left", "frameon": False, "handlelength": 0.8, "handletextpad": 0.4,
                   "labelspacing": 0.3, "borderaxespad": 0.0, "alignment": "left"}  # fmt: skip
        for ax, (_, labels, colours, name) in zip(panels, grid):
            if labels is not None:
                draw_labels(ax, physical, labels, colours)
                counts = [f"{label} ({int((labels == label).sum()):,})" for label in colours]
                ax.legend([square_handle(c) for c in colours.values()], counts, title=name, **options)
        renderer = figure.canvas.get_renderer()
        # NB each column of keys as wide as its widest.
        columns = [
            GAP + max((ax.get_legend().get_window_extent(renderer).width / figure.dpi for ax in panels[col::2]
                       if ax.get_legend() is not None), default=0.0)
            for col in (0, 1)
        ]  # fmt: skip

        # NB a panel as wide as the column leaves beside its column of keys, at the spots' framed aspect; the page as
        #    tall as (a) and (b) then need.
        panel_w = (width - sum(columns) - GAP) / 2.0
        panel_h = panel_w * ratio
        rows = -(-len(grid) // 2)
        height_a = LETTER + rows * (TITLE_LINE + panel_h) + (rows - 1) * GAP + TICK_LINE + AXIS_LABEL_LINE + GAP
        # NB the chromosome names' room at the page's size: cnplot sizes its own at FS_TICK, which `uniform` undoes.
        chr_line = genome_label_room(
            fontsize=size, tiers=chrname_tiers(genome_axis, width - CHR_OVERHANG, fontsize=size)
        )
        rows_profiles = chr_line + sum(TITLE_LINE + CLONE_ROW * len(t) + GAP for t in tumour.values())
        height_profiles = LETTER + rows_profiles + (CN_KEY if integer else 0.0) + (STATE_KEY if categorical else 0.0)
        tracks = pseudobulk or {}
        n_tracks = sum(table["clone"].nunique() for table in tracks.values())
        clone_tracks = NAME_LINE + 2 * PSEUDOBULK_TRACK + TRACK_GAP + GAP
        height_tracks = LETTER + len(tracks) * TITLE_LINE + n_tracks * clone_tracks + chr_line if tracks else 0.0
        page_height = MARGIN["top"] + height_a + height_tracks + height_profiles + MARGIN["top"]
        figure.set_size_inches(page_width, page_height)
        logger.info(
            f"Annotation page {page_width:.2f} x {page_height:.2f} in, spatial panels {panel_w:.2f} x {panel_h:.2f} in."
        )

        top = MARGIN["top"]
        letter(figure, top, "a")
        top += LETTER
        for k, (ax, (title, _, _, _)) in enumerate(zip(panels, grid)):
            row, col = divmod(k, 2)
            frame_extent(ax, box, labels=col == 0)
            left = MARGIN["left"] + col * (panel_w + columns[0] + GAP)
            place(ax, left, top + row * (TITLE_LINE + panel_h + GAP) + TITLE_LINE, panel_w, panel_h)
            if ax.get_legend() is not None:
                ax.get_legend().set_bbox_to_anchor((1.0 + GAP / panel_w, 1.0), transform=ax.transAxes)
            # NB every other tick labelled, counting from the outermost, which carries the unit: `frame_extent`'s round
            #    steps run their labels together on a panel this size.
            for axis in (ax.xaxis, ax.yaxis):
                found = [t.get_text() for t in axis.get_ticklabels()]
                axis.set_ticklabels([t if (len(found) - 1 - i) % 2 == 0 else "" for i, t in enumerate(found)])
            ax.text(0.0, 1.0, title, transform=offset_copy(ax.transAxes, figure, y=2.0, units="points"), va="bottom")
            # NB x labelled on each column's last panel, which an odd count leaves a row short on the right.
            if k + 2 < len(grid):
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel(r"$x$ ($\mu$m)")
            if col == 0:
                ax.set_ylabel(r"$y$ ($\mu$m)")
        top += height_a - LETTER

        if tracks:
            letter(figure, top, "b")
            top += LETTER
            last_track = None
            for title, table in tracks.items():
                figure.text(MARGIN["left"] / page_width, 1.0 - top / page_height, title, va="top", ha="left")
                top += TITLE_LINE
                for k, bins in table.groupby("clone", sort=True):
                    top += NAME_LINE
                    rdr_ax = add_axes(figure, MARGIN["left"], top, width - CHR_OVERHANG, PSEUDOBULK_TRACK)
                    top += PSEUDOBULK_TRACK + TRACK_GAP
                    last_track = add_axes(figure, MARGIN["left"], top, width - CHR_OVERHANG, PSEUDOBULK_TRACK)
                    top += PSEUDOBULK_TRACK + GAP
                    draw_clone_tracks(rdr_ax, last_track, bins, genome_axis)
                    rdr_ax.text(0.0, 1.0, cast_clone_label(f"clone {k}"), va="bottom",
                                transform=offset_copy(rdr_ax.transAxes, figure, y=2.0, units="points"))  # fmt: skip
            decorate_genome_axis(last_track, genome_axis, label_pos="bottom", fontsize=size)
            top = MARGIN["top"] + height_a + height_tracks

        letter(figure, top, "c" if tracks else "b")
        top += LETTER
        cn_key = add_axes(figure, MARGIN["left"], top + rows_profiles, width, CN_KEY_AXES) if integer else None
        last = list(integer)[-1] if integer else None
        for k, title in enumerate(tumour):
            figure.text(MARGIN["left"] / page_width, 1.0 - top / page_height, title, va="top", ha="left")
            top += TITLE_LINE + (chr_line if k == 0 else 0.0)
            height = CLONE_ROW * len(tumour[title])
            ax = add_axes(figure, MARGIN["left"], top, width - CHR_OVERHANG, height)
            if title in integer:
                plot_cnv_profile(ax, integer[title], genome_axis, ax_leg=cn_key if title == last else None,
                                 clones=tumour[title], plot_chrname=False, show_pi_viol=False,
                                 legend_kwargs={"fontsize": size})  # fmt: skip
            else:
                draw_clone_states(ax, categorical[title], genome_axis, state_colours or {}, tumour[title])
            if k == 0:
                decorate_genome_axis(ax, genome_axis, label_pos="top", hide_spines=True, fontsize=size)
            dot_centromeres(ax, genome_axis)
            top += height + GAP
        if categorical:
            # NB the states' swatches under cnplot's, as wide and as far in, where it drew one.
            first = cn_key.patches[0].get_window_extent(renderer) if cn_key is not None and cn_key.patches else None
            swatch = (0.3, 0.08) if first is None else (first.width / figure.dpi, first.height / figure.dpi)
            indent = 0.4 if first is None else first.x0 / figure.dpi - MARGIN["left"]
            state_key = add_axes(figure, MARGIN["left"], top + (CN_KEY if integer else 0.0), width, STATE_KEY)
            draw_state_key(state_key, key_colours, ", ".join(categorical), swatch, indent)

        uniform(figure, size)
    return figure
