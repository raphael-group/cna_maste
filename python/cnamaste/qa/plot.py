"""The cross-sample validation figure: `all_sample_metrics.tsv`'s scores per simulation group, one method at a time.

`plot_validation` draws cnaster's `plot_validation_stats.plot_metrics` from `run_sim_qa`'s table: a 2 x 2 grid, one
metric per panel, a box per simulated class (samples sharing a CNA number and size) with each sample overlaid. A panel whose
metric the table does not carry for the method reads "not computed". `sim_groups` parses the sample ids into groups.

The figure is `combined.pdf`'s format (`figure_style`): half a page, the text column wide, every text in the stated
serif at its one size, panels lettered (a)-(d).

Departures from cnaster's: matplotlib only (no seaborn), so the boxes and points are drawn by hand; every panel keeps
one x position per group present for the method, so the bottom row's tick labels name every panel's boxes; the
panels are CNA sensitivity and normal specificity over the phase-free N-CNA recall and (N-) Clone ARI, with no log-likelihood
or third row; specificity's axis starts at 0.8 (`FLOORS`); there is no title; the size is written `50M` rather than
`5e7`; groups tied on size and number of CNAs are ordered by name.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from cnamaste.config import start_time
from cnamaste.figure_style import (
    get_box,
    get_box_alpha,
    get_compared_box,
    get_compared_whisker,
    get_whisker,
    page_size,
    page_style,
    uniform,
)
from cnamaste.logger import get_logger

logger = get_logger(__name__, start_time=start_time)

SAMPLE_IDS = {
    "zenodo": re.compile(
        r"^numcnas(?P<numcnas>\d+(?:\.\d+)*)_cnasize(?P<cnasize>[\d.]+(?:[eE][+-]?\d+)?)"
        r"_ploidy(?P<ploidy>\d+(?:\.\d+)?)(?P<realization>_random(?P<random>\d+))$"
    ),
    "hd_tree": re.compile(
        r"^hd_tree_cnas(?P<numcnas>\d+)_length(?P<length_mb>\d+)mb(?P<realization>_r(?P<random>\d+)_[0-9a-f]{8})$"
    ),
}
"""A study -> its sample ids: CalicoST's zenodo `numcnas<a.b>_cnasize<s>_ploidy<p>_random<r>`, and `cnamaste.sim.study`'s
`hd_tree_cnas<n>_length<L>mb_r<k>_<hash>`; the group is the id less its `realization`."""


def get_sample_ids() -> dict[str, re.Pattern[str]]:
    """Return each study's sample id pattern, `SAMPLE_IDS`: zenodo, hd_tree."""
    return SAMPLE_IDS


PANELS = (
    ("sensitivity", "CNA sensitivity"),
    ("specificity", "Normal specificity"),
    ("exact_altered", r"$\mathbb{N}$-CNA recall"),
    ({"cnamaste": "int_clone_ari", "calicost": "clone_ari"}, r"($\mathbb{N}$-) Clone ARI"),
)
"""The four panels, row by row, as `(column, label)`: a column for every method, or a column per method
(`panel_column`). Sensitivity: a planted altered clone-bin decoded as any pair but (1, 1); specificity: a planted
(1, 1) decoded (1, 1); the recall is phase-free. ARI is cnamaste's integer clones' and CalicoST's fitted clones', which
take no integer merge: hence `(N-)`."""

RATES = {"sensitivity", "specificity", "exact_altered", "clone_ari", "int_clone_ari"}
"""Columns in `[0, 1]`, drawn on `[-0.05, 1.05]` as cnaster draws them, but for `FLOORS`'."""

FLOORS = {"specificity": 0.8}
"""A rate's axis floor where `[0, 1]` would flatten its spread: specificity is 0.991 or above on the six fixtures'
current-`main` runs, so the panel is drawn on `[0.8, 1.01]`, the top margin 5% of the range as the others'. A
sample below the floor is off the axis, and logged."""

METHOD_LABELS = {
    "cnamaste": r"$\mathtt{cn}\alpha$-$\mathtt{ma}\mathit{st}\mathtt{e}$",
    "calicost": r"$\mathtt{CalicoST}$",
}
"""How the key writes a method; a method not here is written as named."""


def get_method_labels() -> dict[str, str]:
    """Return the method labels, `METHOD_LABELS`."""
    return METHOD_LABELS


NOT_COMPUTED = "not computed"

PAIR_OFFSET = 0.2
"""Group positions either side of a group's centre where two methods' boxes sit, each `PAIR_WIDTH` wide."""

PAIR_WIDTH = 0.35

JITTER = 0.1
"""Half-width of the points' horizontal jitter, in group positions, as cnaster's strip."""


def compact_size(size: float) -> str:
    """A CNA size in bases as `10M`, `50M`, `1.5M`, `500K`.

    Args:
        size: in bases.

    Returns:
        The size with a K, M or G suffix where at least a thousand.
    """
    for scale, unit in ((1e9, "G"), (1e6, "M"), (1e3, "K")):
        if size >= scale:
            return f"{size / scale:g}{unit}"
    return f"{size:g}"


def sim_groups(sample_ids: Iterable[str], study: str) -> pd.DataFrame:
    """Parse simulated sample ids into their parameters and group; an id that does not parse is dropped, logged.

    Args:
        sample_ids: ids such as `numcnas1.2_cnasize5e7_ploidy2_random0`, repeats allowed.
        study: one of `get_sample_ids()`, the ids' pattern.

    Returns:
        One row per parsed id, in input order: `sample_id`; `numcnas` the string (`"1.2"`); `cnasize` in bases and
            `ploidy` floats, `ploidy` NaN where unstated; `random` int; `group` the id less its realization; `label`
            `(1,2, 50M)`.
    """
    rows = []
    for sample_id in sample_ids:
        found = SAMPLE_IDS[study].match(str(sample_id))
        if found is None:
            logger.warning(f"{sample_id}: not a {study} sample id, left out of the figure")
            continue
        parsed = found.groupdict()
        numcnas = parsed["numcnas"]
        cnasize = float(parsed["cnasize"]) if "cnasize" in parsed else 1e6 * float(parsed["length_mb"])
        rows.append(
            {
                "sample_id": str(sample_id),
                "numcnas": numcnas,
                "cnasize": cnasize,
                "ploidy": float(parsed.get("ploidy") or "nan"),
                "random": int(parsed["random"]),
                "group": str(sample_id)[: found.start("realization")],
                "label": f"({numcnas.replace('.', ',')}, {compact_size(cnasize)})",
            }
        )
    return pd.DataFrame(rows, columns=["sample_id", "numcnas", "cnasize", "ploidy", "random", "group", "label"])


def group_order(groups: pd.DataFrame) -> list[str]:
    """The groups in drawing order: `cnasize` largest first, then the sum of `numcnas`' components largest first, then
    by name.

    Args:
        groups: `sim_groups`' result.

    Returns:
        Each group once.
    """
    unique = groups.drop_duplicates("group").assign(
        total=lambda d: d["numcnas"].map(lambda n: sum(int(p) for p in n.split(".")))
    )
    return list(unique.sort_values(["cnasize", "total", "group"], ascending=[False, False, True])["group"])


def not_computed(ax: Any) -> None:
    """Write `NOT_COMPUTED` across the middle of `ax`."""
    ax.text(0.5, 0.5, NOT_COMPUTED, transform=ax.transAxes, ha="center", va="center", color="0.4")
    ax.set_yticks([])


def draw_groups(
    ax: Any,
    values: list[np.ndarray],
    seed: int,
    offset: float = 0.0,
    width: float = 0.6,
    box: str = get_box(),
    whisker: str = get_whisker(),
) -> None:
    """A box per non-empty group at its position `0, 1, ...` plus `offset`, with its samples as jittered points.

    Args:
        ax: the panel.
        values: each group's samples, NaN dropped.
        seed: the jitter's generator seed.
        offset: in group positions.
        width: in group positions.
        box: the colours.
        whisker: the colours.
    """
    rng = np.random.default_rng(seed)
    drawn = [k for k, v in enumerate(values) if v.size]
    if not drawn:
        return
    ax.boxplot(
        [values[k] for k in drawn],
        positions=[k + offset for k in drawn],
        widths=width,
        showfliers=False,
        patch_artist=True,
        boxprops={"facecolor": box, "alpha": get_box_alpha(), "edgecolor": whisker, "linewidth": 0.5},
        whiskerprops={"color": whisker, "linewidth": 0.5},
        capprops={"color": whisker, "linewidth": 0.5},
        medianprops={"color": "black", "linewidth": 0.8},
        manage_ticks=False,
    )
    for k in drawn:
        x = k + offset + rng.uniform(-JITTER * width / 0.6, JITTER * width / 0.6, size=values[k].size)
        ax.scatter(x, values[k], s=3, color="black", alpha=0.7, edgecolors="black", linewidths=0.2, zorder=3)


def panel_column(column: str | dict[str, str] | None, method: str) -> str | None:
    """A `PANELS` entry's column for `method`: the column, or the method's where it is given per method.

    Args:
        column: a `PANELS` column.
        method: e.g. `cnamaste`.

    Returns:
        The table column, or None where the panel has none for `method`.
    """
    return column.get(method) if isinstance(column, dict) else column


def plot_validation(
    table: pd.DataFrame,
    study: str,
    method: str,
    compared: str | None = None,
    kinds: dict[str, str] | None = None,
    names: dict[str, str] | None = None,
) -> Any:
    """Draw one method's validation figure from `all_sample_metrics.tsv`'s rows, with a second method's beside it.

    Each panel of `PANELS` holds a box per group of `method`'s samples (`sim_groups`, ordered by `group_order`), at
    one x position per group, with the samples overlaid; a panel whose column is absent or all NaN for `method`, or
    has no counterpart, reads `NOT_COMPUTED`. Rows whose sample id does not parse are left out, logged. Given
    `compared`, its boxes sit right of `method`'s in each group (`PAIR_OFFSET`), in `COMPARED_BOX`, over the groups
    either method has, and a key names both (`METHOD_LABELS`) in the lower left of the specificity panel alone.

    Rows are told apart by `series` where `table` has the column, else by `method`, so two cnamaste solvers can be
    drawn as two series; `kinds` names each series' method for `panel_column` (a series not in it is its own method),
    and `names` writes a series in the key in place of `METHOD_LABELS`.

    Args:
        table: `run_sim_qa`'s table, `sample_id`, `method` and the metric columns, optionally `series`.
        study: one of `get_sample_ids()`, the sample ids' pattern.
        method: e.g. `cnamaste`.
        compared: e.g. `calicost`, or None.
        kinds: series -> method.
        names: series -> key label.

    Returns:
        The figure, `page_size("half")`, untitled.
    """
    methods = [method] if compared is None else [method, compared]
    kinds, names = kinds or {}, names or {}
    table = table.assign(series=table["series"] if "series" in table else table["method"])
    every = table[table["series"].isin(methods)].reset_index(drop=True)
    groups = sim_groups(every["sample_id"], study)
    every = every[every["sample_id"].astype(str).isin(set(groups["sample_id"]))].merge(
        groups.drop_duplicates("sample_id")[["sample_id", "group"]], on="sample_id"
    )
    order = group_order(groups)
    labels = dict(zip(groups["group"], groups["label"]))
    styles = {method: (get_box(), get_whisker()), compared: (get_compared_box(), get_compared_whisker())}
    offsets = {method: 0.0} if compared is None else {method: -PAIR_OFFSET, compared: PAIR_OFFSET}
    width = 0.6 if compared is None else PAIR_WIDTH
    with page_style() as size:
        figure, axes = plt.subplots(2, 2, figsize=page_size("half"), sharex=True)
        for k, (ax, (spec, label)) in enumerate(zip(axes.flat, PANELS)):
            ax.set_ylabel(label)
            ax.set_xlim(-0.5, max(len(order), 1) - 0.5)
            drawn = False
            for m in methods:
                rows, column = every[every["series"] == m], panel_column(spec, kinds.get(m, m))
                values = (
                    []
                    if column is None or column not in rows
                    else [
                        pd.to_numeric(rows.loc[rows["group"] == g, column], errors="coerce").dropna().to_numpy(float)
                        for g in order
                    ]
                )
                if any(v.size for v in values):
                    draw_groups(
                        ax, values, seed=k, offset=offsets[m], width=width, box=styles[m][0], whisker=styles[m][1]
                    )
                    drawn = True
                    if column in FLOORS:
                        floor = FLOORS[column]
                        ax.set_ylim(floor, 1.0 + 0.05 * (1.0 - floor))
                        below = sum(int((v < floor).sum()) for v in values)
                        if below:
                            logger.warning(f"{m} {column}: {below} samples below the axis floor {floor}, not shown")
                    elif column in RATES:
                        ax.set_ylim(-0.05, 1.05)
            if not drawn:
                not_computed(ax)
            if compared is not None and panel_column(spec, kinds.get(method, method)) == "specificity":
                ax.legend(
                    handles=[
                        Patch(
                            facecolor=styles[m][0],
                            alpha=get_box_alpha(),
                            edgecolor=styles[m][1],
                            linewidth=0.5,
                            label=names.get(m, METHOD_LABELS.get(m, m)),
                        )
                        for m in methods
                    ],
                    loc="lower left",
                    frameon=True,
                    framealpha=0.85,
                    facecolor="white",
                    edgecolor="none",
                )
        for ax in axes[-1]:
            ax.set_xticks(range(len(order)), [labels[g] for g in order], rotation=45, ha="right")
            ax.set_xlabel("Simulated class")
        uniform(figure, size)
        figure.tight_layout()
        # NB each panel's letter at its top left, left of its tick labels: combined.pdf letters its panels so.
        renderer = figure.canvas.get_renderer()
        width = figure.get_size_inches()[0] * figure.dpi
        for ax, name in zip(axes.flat, "abcd"):
            x0 = ax.get_tightbbox(renderer).x0 / width
            figure.text(x0, ax.get_position().y1, f"({name})", va="bottom", ha="left", fontsize=size)
    return figure
