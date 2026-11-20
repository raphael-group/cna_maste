from typing import Any

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from matplotlib.transforms import offset_copy

from cnamaste.figure_style import column_size, get_dpi, stagger, style
from cnamaste.palette import get_copy_states, get_full_palette, state_colour
from cnamaste.utils import cast_clone_label

NORMAL_OPACITY = 0.25

BAR_SHARE = 0.8
"""The share of a clone's row its bars take, centred in it; the rest is the gap to the next clone. One value for
both profiles, `plot_copy_number_profile`'s pair of allele bars and `plot_hatched_profile`'s single hatched bar, so
the two draw a clone's row alike."""

PROFILE_HEIGHT = 0.9
"""Inches of the standalone figure per clone (`plot_copy_number_profile`), as `layout.PROFILE_ROW` is the combined
page's inches per clone."""

PROFILE_FLOOR = 2.25
"""Inches of the standalone figure at its shortest: a profile of one or two clones still carries its axis."""


# TODO move to segments: runs of equal state in a sequence; duplicates utils.get_intervals, which plot_genomic uses.
def get_intervals(pred_cnv: Any) -> tuple[list[tuple[int, int]], list[Any]]:
    """Find contiguous intervals of equal state in `pred_cnv`, vectorized.

    Args:
        pred_cnv: the state labels.

    Returns:
        The (start, end) index intervals and each interval's state label.
    """
    pred_cnv = np.asarray(pred_cnv)
    if len(pred_cnv) == 0:
        return [], []

    changes = np.where(pred_cnv[:-1] != pred_cnv[1:])[0] + 1
    splits = np.concatenate(([0], changes, [len(pred_cnv)]))

    intervals = [(splits[i], splits[i + 1]) for i in range(len(splits) - 1)]
    labs = pred_cnv[splits[:-1]].tolist()

    return intervals, labs


def draw_mirror_chevrons(ax: Any, x0: float, y_b: float, w: float, h_sub: float, direction: int) -> None:
    """Shade a mirrored event, e.g. (A, B) -> (B, A) between clones, with vertical chevrons pointing by `direction`.

    Args:
        ax: the axes.
        x0: the box.
        y_b: the box.
        w: the box.
        h_sub: the box.
        direction: the mirror direction.
    """
    n_chev = 2
    chev_unit = w * 0.48
    gap = w * 0.04
    total_w = n_chev * chev_unit + (n_chev - 1) * gap
    x_start = x0 + (w - total_w) / 2.0

    y_high = y_b + 2 * h_sub
    y_low = y_b

    for i in range(n_chev):
        cx_left = x_start + i * (chev_unit + gap)
        cx_right = cx_left + chev_unit
        cx_mid = (cx_left + cx_right) / 2.0

        xs = [cx_left, cx_mid, cx_right]
        ys = [y_low, y_high, y_low] if direction > 0 else [y_high, y_low, y_high]

        ax.plot(
            xs,
            ys,
            color="black",
            linewidth=1.2,
            alpha=0.5,
            solid_capstyle="round",
        )


def plot_ascn_legend(
    ax: Any,
    box_w: float = 0.8,
    box_h: float = 0.8,
    tick_len: float = 0.08,
    palette_name: str = "chisel_single",
) -> Any:
    """Draw a horizontal colour bar legend for single-allele copy numbers.

    Args: `ax` the axes; `box_w`, `box_h`, `tick_len` the geometry; `label_fontsize`; `palette_name`.

    Returns:
        The axes.
    """
    state_style, ordered_acn = get_full_palette(palette_name)
    boxes = list(ordered_acn)

    ax.axis("off")

    swatch_w = box_w * 0.5
    ax.add_patch(Rectangle((0.0, 0.0), swatch_w, box_h, facecolor="white", edgecolor="black"))

    draw_mirror_chevrons(ax, 0.0, 0.0, swatch_w, box_h / 2, direction=1)

    ax.text(
        swatch_w / 2.0,
        -tick_len - 0.04,
        "Mirror",
        ha="center",
        va="top",
    )

    x0 = swatch_w + 0.5

    for i, label in enumerate(boxes):
        color = state_style.get(label)
        rect = Rectangle(
            (x0 + i * box_w, 0.0),
            box_w,
            box_h,
            facecolor=mcolors.to_rgba(color, alpha=(NORMAL_OPACITY if label == 1 else 1.0)),
            edgecolor="black",
        )

        ax.add_patch(rect)

        xc = x0 + i * box_w + box_w / 2.0
        ax.plot([xc, xc], [-tick_len, 0.0], color="black", linewidth=0.8)
        ax.text(
            xc,
            -tick_len - 0.04,
            str(label),
            ha="center",
            va="top",
        )

    total_boxes_w = len(boxes) * box_w

    ax.text(
        x0 + total_boxes_w + 0.2,
        box_h / 2.0,
        r"$\mathbb{N}$" + "-CNA",
        ha="left",
        va="center",
    )

    # Adjust x limit to encapsulate the entire updated legend span
    ax.set_xlim(0.0, x0 + total_boxes_w + 2.0)
    ax.set_ylim(-0.5, box_h + 0.2)
    ax.set_aspect("auto")

    return ax


def plot_copy_number_profile(
    df_cnv: pd.DataFrame,
    ax: Any = None,
    height: float = 1.0,
    title: str | None = None,
    show_clone_name: bool = True,
    plot_chrname: bool = True,
    figsize: tuple[float, float] | None = None,
    palette_name: str = "chisel_single",
) -> Any:
    """Plot per-clone (allele-specific) copy-number profiles, width proportional to the number of segments.

    Args:
        df_cnv: the per-segment copy numbers.
        ax: an axes or None.
        palette_name: the palette.

    Returns:
        The figure.
    """

    state_style, _ = get_full_palette(palette_name)

    clone_ids = [c.split(" ")[0][5:] for c in df_cnv.columns if c.endswith(" A")]

    A_full = df_cnv[[f"clone{cid} A" for cid in clone_ids]].fillna(1).to_numpy()
    B_full = df_cnv[[f"clone{cid} B" for cid in clone_ids]].fillna(1).to_numpy()

    # TODO move to qa.metrics: ranks clones by aberration, as clone_order does again below.
    deviations = []
    for k in range(len(clone_ids)):
        a_col, b_col = A_full[:, k], B_full[:, k]

        # Combine into a single state array to identify contiguous blocks
        encoded = a_col * 1_000 + b_col
        intervals, _ = get_intervals(encoded)

        # Extract the start indices of each interval
        starts = [s for s, e in intervals]

        a_rle = a_col[starts]
        b_rle = b_col[starts]

        # Sum the deviation of just the segments (unweighted by genomic length)
        deviations.append(np.sum(np.abs(a_rle - 1) + np.abs(b_rle - 1)))

    clone_ids = [clone_ids[i] for i in np.argsort(deviations)]

    num_clones = len(clone_ids)

    if ax is None:
        figsize = figsize or column_size(max(PROFILE_FLOOR, PROFILE_HEIGHT * num_clones))
        fig, ax = plt.subplots(figsize=figsize, dpi=get_dpi(), facecolor="white")

        fig.subplots_adjust(bottom=0.15)
    else:
        fig = ax.figure

    h = height / num_clones
    h_pair = BAR_SHARE * h
    clone_gap = h - h_pair
    h_sub = h_pair / 2
    y_gap = clone_gap / 2

    ch_offset = 0
    ch_coords = []
    chs = []

    for ch, df_ch in df_cnv.groupby("CHR", sort=False):
        chs.append(ch)
        ch_coords.append(ch_offset)
        n_rows = len(df_ch)

        A_mat = df_ch[[f"clone{cid} A" for cid in clone_ids]].to_numpy()
        B_mat = df_ch[[f"clone{cid} B" for cid in clone_ids]].to_numpy()

        # Keep directions to dictate the orientation of the chevron (up/down)
        dirs_mat = np.zeros_like(A_mat, dtype=int)
        dirs_mat[A_mat > B_mat] = 1
        dirs_mat[A_mat < B_mat] = -1

        # TODO move to ab_decode: calls a mirrored event, (A, B) in one clone and (B, A) in another.
        has_mirror = np.zeros(n_rows, dtype=bool)

        for c in range(num_clones):
            exact_swap = (A_mat == B_mat[:, [c]]) & (B_mat == A_mat[:, [c]]) & (A_mat != B_mat)
            has_mirror |= np.any(exact_swap, axis=1)

        for k, _ in enumerate(clone_ids):
            a_states = A_mat[:, k]
            b_states = B_mat[:, k]

            dirs = dirs_mat[:, k]

            encoded_states = a_states * 1_000 + b_states * 10 + has_mirror.astype(int)
            intervals, _ = get_intervals(encoded_states)

            k_plot = num_clones - k - 1
            y_b = y_gap + h * k_plot
            y_a = h_sub + y_b

            for s, e in intervals:
                x0 = ch_offset + s
                w = e - s

                cna, cnb = a_states[s], b_states[s]
                is_mirror, direction = has_mirror[s], dirs[s]

                ax.add_patch(
                    Rectangle(
                        (x0, y_b),
                        w,
                        h_sub,
                        facecolor=state_style.get(cnb, state_style.get("default", "lightgray")),
                        edgecolor="none",
                        linewidth=0,
                        alpha=(NORMAL_OPACITY if cnb == 1 else 1.0),
                    )
                )

                ax.add_patch(
                    Rectangle(
                        (x0, y_a),
                        w,
                        h_sub,
                        facecolor=state_style.get(cna, state_style.get("default", "lightgray")),
                        edgecolor="none",
                        linewidth=0,
                        alpha=(NORMAL_OPACITY if cna == 1 else 1.0),
                    )
                )

                if is_mirror and direction != 0:
                    draw_mirror_chevrons(ax, x0, y_b, w, h_sub, direction)

        ch_offset += n_rows

    ch_coords.append(ch_offset)

    for k in range(num_clones):
        y_b_k = k * h + y_gap
        ax.vlines(
            ch_coords,
            ymin=y_b_k,
            ymax=y_b_k + h_pair,
            linewidth=1,
            colors="black",
            clip_on=False,
        )

    for k in range(num_clones):
        y_b_k = k * h + y_gap
        for y0 in (y_b_k, y_b_k + h_sub):
            ax.add_patch(
                Rectangle(
                    (0, y0),
                    ch_offset,
                    h_sub,
                    facecolor="none",
                    edgecolor="black",
                    linewidth=1,
                )
            )

    ax.grid(False)
    ax.set_xlim(0, ch_offset)
    ax.set_xlabel("")

    for spine in ax.spines.values():
        spine.set_visible(False)

    if plot_chrname and chs:
        midpoints = [ch_coords[i] for i in range(len(ch_coords) - 1)]
        ax.set_xticks(midpoints)

        chr_labels = [f"chr{ch}" if str(ch).isdigit() else str(ch) for ch in chs]

        # NB each name level under its chromosome's middle, on a second line where it would meet its neighbour's:
        #    combined.pdf's chromosome axis. Placed rather than set as tick labels, so a name on the second line
        #    sits on its own line rather than carrying a blank one above it (T- #46).
        size = style()["font_size"]
        span = ax.get_position().width * ax.figure.get_size_inches()[0] * 72.0
        left, right = ax.get_xlim()
        points = [(m - left) * span / max(right - left, 1e-9) for m in midpoints]

        ax.set_xticklabels([])

        # NB three lines, not combined.pdf's two: easy's last chromosomes are short enough that two leave
        #    chr20 on chr22 at this width.
        for middle, name, line in zip(midpoints, chr_labels, stagger(points, chr_labels, size, lines=3)):
            at = offset_copy(ax.get_xaxis_transform(), ax.figure, y=-2.0 - line * 1.1 * size, units="points")
            ax.text(middle, 0.0, name, transform=at, ha="center", va="top")
        ax.tick_params(
            axis="x",
            labeltop=False,
            labelbottom=True,
            top=False,
            bottom=False,
            pad=-5,
        )
    else:
        ax.set_xticks([])

    ax.set_yticks([h * (i + 0.5) for i in range(num_clones)])
    ylabels = [f"Clone {cid}" if show_clone_name else str(cid) for cid in reversed(clone_ids)]
    # NB cast to Roman.
    ylabels = [cast_clone_label(cid) for cid in ylabels]

    ax.set_yticklabels(ylabels, va="center", ha="left", rotation=90)

    minor_positions, minor_labels = [], []
    for k in range(num_clones):
        minor_positions.extend([k * h + y_gap + h_sub * 0.5, k * h + y_gap + h_sub * 1.5])
        minor_labels.extend(["B", "A"])

    ax.set_yticks(minor_positions, minor=True)
    ax.set_yticklabels(minor_labels, minor=True)
    ax.tick_params(axis="y", which="minor", left=False, right=False, pad=2)

    ax.set_ylim(0, num_clones * h)
    ax.tick_params(axis="y", which="major", left=True, right=False, length=4, pad=20)

    # TODO BUG ch_coords[0] aligned.
    legend_ax = fig.add_axes([0.15, 0.025, 0.7, 0.05])

    plot_ascn_legend(legend_ax, palette_name=palette_name)

    if title:
        ax.set_title(title)

    return fig


# --- hatched profile ---

OUTLINE = 0.5
"""Points, for row outlines and an aberration's ends."""


# TODO move to qa.metrics: ranks clones by aberration, missing copies read (1, 1), summed over runs of equal (A, B).
def clone_order(df_cnv: pd.DataFrame) -> list[str]:
    """Order the clone ids of `df_cnv`, least aberrant first, by segment deviation.

    Args:
        df_cnv: the per-segment copy numbers.

    Returns:
        The clone ids.
    """
    clone_ids = [c.split(" ")[0][5:] for c in df_cnv.columns if c.endswith(" A")]
    deviations = []
    for cid in clone_ids:
        a, b = df_cnv[f"clone{cid} A"].fillna(1).to_numpy(), df_cnv[f"clone{cid} B"].fillna(1).to_numpy()
        starts = [s for s, _ in get_intervals(a * 1_000 + b)[0]]
        deviations.append(np.sum(np.abs(a[starts] - 1) + np.abs(b[starts] - 1)))
    return [clone_ids[i] for i in np.argsort(deviations)]


def used_states(df_cnv: pd.DataFrame, order: Any) -> list[tuple[int, int]]:
    """The phased `(A, B)` the clones of `order` carry, in `COPY_STATES`' order, a pair before its flip.

    Args:
        df_cnv: copy numbers.
        order: clone ids.

    Returns:
        The `(A, B)` tuples.
    """
    seen = set()
    for cid in order:
        a = df_cnv[f"clone{cid} A"].fillna(1).to_numpy().astype(int)
        b = df_cnv[f"clone{cid} B"].fillna(1).to_numpy().astype(int)
        seen.update(zip(a.tolist(), b.tolist()))
    rank = {state: i for i, state in enumerate(get_copy_states())}
    return sorted(seen, key=lambda s: (rank.get((max(s), min(s)), len(rank)), s[1] > s[0]))


def state_fill(a: int, b: int) -> tuple[float, ...]:
    """`(a, b)`'s profile fill, opaque over white: `palette.state_colour` composited, so a flip is a lighter fill."""
    r, g, bl, alpha = state_colour(a, b)
    return tuple(float(alpha * v + 1.0 - alpha) for v in (r, g, bl))


def plot_state_profile(df_cnv: pd.DataFrame, ax: Any, genome: Any, order: Any = None) -> list[str]:
    """Draw one row per clone on `genome`'s axis, each segment one `state_fill`, an aberration's ends outlined.

    Rows run `order`'s ids top to bottom (`None`: least aberrant first), named as `df_cnv` names them.

    Args:
        df_cnv: copy numbers.
        ax: the axes.
        genome: a `cnamaste.plot.layout.GenomicAxis`.
        order: clone ids or None.

    Returns:
        The clone order drawn.

    Raises:
        ValueError: where `genome` and `df_cnv` differ in bins.
    """
    order = clone_order(df_cnv) if order is None else [str(cid) for cid in order]
    contig = np.repeat(np.arange(len(genome.lengths)), genome.lengths)
    if contig.size != len(df_cnv):
        raise ValueError(f"the genomic axis has {contig.size} bins, the profile {len(df_cnv)}")
    h = 1.0 / len(order)
    for k, cid in enumerate(order):
        height = BAR_SHARE * h
        y0 = 1.0 - (k + 1) * h + (h - height) / 2.0
        a = df_cnv[f"clone{cid} A"].fillna(1).to_numpy().astype(int)
        b = df_cnv[f"clone{cid} B"].fillna(1).to_numpy().astype(int)
        for s, e in get_intervals((contig * 1_000 + a) * 1_000 + b)[0]:
            ax.add_patch(
                Rectangle((s, y0), e - s, height, facecolor=state_fill(a[s], b[s]), edgecolor="none", linewidth=0)
            )
            if (a[s], b[s]) != (1, 1):
                ax.vlines([s, e], ymin=y0, ymax=y0 + height, linewidth=OUTLINE, colors="black", zorder=3)
        ax.add_patch(
            Rectangle((0, y0), genome.n_bins, height, facecolor="none", edgecolor="black", linewidth=OUTLINE, zorder=3)
        )
    genome.frame(ax, labels=False)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([1.0 - (k + 0.5) * h for k in range(len(order))])
    ax.set_yticklabels([cast_clone_label(f"clone {cid}") for cid in order])
    ax.tick_params(axis="y", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return order


def plot_state_key(ax: Any, states: list[tuple[int, int]], box: float = 0.18) -> None:
    """Draw the profile's key, in inches on `ax`: one swatch per state of `states`, labelled `(A,B)`.

    Args:
        ax: the axes.
        states: the `(A, B)` drawn, as `used_states` orders them.
        box: a swatch's width in inches.
    """
    width, height = (
        ax.get_position().width * ax.figure.get_size_inches()[0],
        ax.get_position().height * ax.figure.get_size_inches()[1],
    )
    ax.set_xlim(0.0, width)
    ax.set_ylim(0.0, height)
    ax.axis("off")
    step = 2.4 * box
    for i, (a, b) in enumerate(states):
        ax.add_patch(
            Rectangle(
                (i * step, height / 2.0),
                box,
                height / 2.0,
                facecolor=state_fill(a, b),
                edgecolor="black",
                linewidth=OUTLINE,
            )
        )
        ax.text(i * step + box / 2.0, height / 2.0 - 0.02, f"({a},{b})", ha="center", va="top")
