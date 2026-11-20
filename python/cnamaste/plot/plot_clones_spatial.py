"""`plot_clones_spatial`: the spots drawn at their coordinates, coloured by the clone each carries, and the key
that names them. One module for the one figure every stage of `run_cnamaste` draws, and the labels it writes.
"""

import copy
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

from cnamaste.config import start_time
from cnamaste.figure_style import column_size, em_width, get_clone_palette, get_dpi, get_paper_width, padded, rasterize
from cnamaste.logger import get_logger
from cnamaste.utils import cast_clone_label

logger = get_logger("cnamaste.plot.plot_clones_spatial", start_time=start_time)


def clone_names(
    ids: Any, spot_fractions: dict[Any, float] | None = None, normal_fractions: dict[Any, float] | None = None
) -> list[str]:
    """Return one key label per clone in `ids`, as a column: the names padded alike, the percentages right-aligned.

    The face is proportional and carries no figure space, so the column is held by ordinary spaces: a name is
    padded to the widest (`figure_style.padded`), and a percentage to the widest in its own field by the spaces
    that stand for one digit -- two, where a digit is 0.5 em against a space's 0.25.

    Args:
        ids: the clones, in the order the key lists them.
        spot_fractions: clone -> fraction in [0, 1], or None for either.
        normal_fractions: clone -> fraction in [0, 1], or None for either.

    Returns:
        The labels.
    """
    rows = [
        (cast_clone_label(c), get_whole_percent(spot_fractions, c), get_whole_percent(normal_fractions, c)) for c in ids
    ]
    width = max(em_width(name) for name, *_ in rows)
    digit = " " * max(round(em_width("0") / em_width(" ")), 1)
    fields = [max((len(r[i]) for r in rows if r[i] is not None), default=0) for i in (1, 2)]
    out = []
    for name, *values in rows:
        found = [digit * (fields[i] - len(v)) + v + "%" for i, v in enumerate(values) if v is not None]
        out.append(f"{padded(name, width)} ({', '.join(found)})" if found else name)
    return out


def get_clone_colours(ids: Any, palette: str) -> list[str]:
    """One colour per clone of `ids`: `clone 0`, the normal clone, light grey, the others in `palette` in turn.

    Args:
        ids: the clones' labels, `clone <id>`, in the order the key lists them.
        palette: a seaborn palette.

    Returns:
        The hex colours.
    """
    ids = list(ids)
    if "clone 0" in ids:
        return ["lightgrey", *sns.color_palette(palette, len(ids) - 1).as_hex()]
    return list(sns.color_palette(palette, len(ids)).as_hex())


def get_whole_percent(fractions: dict[Any, float] | None, cid: Any) -> str | None:
    """`fractions[cid]` as a whole percent without its sign, or None where there is none."""
    found = None if fractions is None else fractions.get(cid)
    return None if found is None else f"{100.0 * float(found):.0f}"


def clone_name(cid: Any, spot_fraction: float | None = None, normal_fraction: float | None = None) -> str:
    """Return `cid`'s key label: `Clone I (34%, 9%)`, its share of the labelled spots and its normal fraction.

    Each is a whole percent, because a key is read at a glance and the fractions themselves are in the run's log.
    The shares are over the labelled spots alone, so they sum to 100%. A clone with no normal fraction reads
    `Clone I (34%)`, and one with neither keeps its name.

    Args:
        cid: the clone's label, as the assignment carries it.
        spot_fraction: in [0, 1], or None for either.
        normal_fraction: in [0, 1], or None for either.

    Returns:
        The label.
    """
    name = cast_clone_label(cid)
    parts = [f"{100.0 * float(f):.0f}%" for f in (spot_fraction, normal_fraction) if f is not None]
    return f"{name} ({', '.join(parts)})" if parts else name


def plot_clones_spatial(
    physical_coords: np.ndarray,
    assignment: pd.Series,
    sample_list: list[str] | None = None,
    sample_ids: np.ndarray | None = None,
    base_width: float = 4,
    base_height: float = 4,
    palette: str = get_clone_palette(),
    ax: Any = None,
    colorlist: list[Any] | None = None,
    normal_fractions: dict[Any, float] | None = None,
) -> Any:
    """Plot the spatial distribution of assigned clones for each slice.

    Given `ax`, draws there, on that axes' figure, rather than on a new one. The key carries each clone's share of
    the labelled spots, and its inferred normal fraction where `normal_fractions` names one (`clone_name`).

    Args:
        physical_coords: (n_spots, 2) um, x left to right and y up (`platforms.visium.physical_coords`), drawn as given.
        assignment: labels.
        sample_list: the slices.
        sample_ids: the slices.
        ax: None.
        normal_fractions: clone label -> fraction in [0, 1], or None.

    Returns:
        The figure.
    """
    logger.info("Plotting inferred positions for all clones.")

    # NB shift coordinates across samples
    shifted_coords = copy.copy(physical_coords)

    if sample_ids is not None:
        x_offset = 0

        for s, _ in enumerate(sample_list):
            index = np.where(sample_ids == s)[0]
            shifted_coords[index, 0] = shifted_coords[index, 0] + x_offset
            x_offset += np.max(physical_coords[index, 0]) + 10

    # NB number of clones and samples
    final_clone_ids = np.unique(assignment[~assignment.isnull()].values)
    n_final_clones = len(final_clone_ids)

    # NB heuristic for marker size: 120000.0 is roughly appropriate for s=0.1 with ~100k spots.
    #    If we have fewer spots, we want larger markers.
    #    Clip to a reasonable range [0.1, 20].
    n_points = physical_coords.shape[0]
    marker_size = np.clip(12_000.0 / n_points, 0.1, 25.0)

    if ax is None:
        fig, ax = plt.subplots(
            1, 1, figsize=column_size(base_height * get_paper_width() / base_width), dpi=get_dpi(), facecolor="white"
        )
    else:
        fig = ax.figure

    # NB a caller may give the colours, one per sorted label, so a second map of the same clones keeps them.
    if colorlist is not None:
        colorlist = list(colorlist)
    else:
        colorlist = get_clone_colours(final_clone_ids, palette)

    for c, cid in enumerate(final_clone_ids):
        idx = np.where(assignment.values == cid)[0]

        ax.scatter(
            x=shifted_coords[idx, 0],
            y=shifted_coords[idx, 1],
            s=marker_size,
            color=colorlist[c],
            linewidth=0,
            rasterized=rasterize(len(shifted_coords)),
        )

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=colorlist[c],
            label=cid,
            markersize=6,
        )
        for c, cid in enumerate(final_clone_ids)
    ]

    # NB the share of the spots a clone holds, over the labelled spots alone, so the key's shares sum to 100%.
    labelled = assignment[~assignment.isnull()].values
    shares = {cid: float(np.mean(labelled == cid)) for cid in final_clone_ids}
    ax.legend(
        legend_elements,
        clone_names(final_clone_ids, shares, normal_fractions),
        handlelength=0.1,
        loc="upper left",
        bbox_to_anchor=(0.05, 0.02),
        ncol=n_final_clones,
        frameon=False,
        borderaxespad=0.0,
    )

    if sample_list is not None:
        ax.text(
            0.05,
            0.99,
            ", ".join(sample_list),
            transform=ax.transAxes,
            ha="left",
            va="bottom",
        )

    ax.axis("off")

    return fig
