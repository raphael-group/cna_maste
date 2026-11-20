"""What a `cnamaste.sim` draw planted, on `combined.pdf`'s page (`plot.layout`): `truth_combined.pdf` and
`truth_combined_spatial.pdf`, written to the draw's `qa/`.

(a) The clone tree, each event on the edge it arose on, each clone named with its event barcode. (b) Each clone's
planted `(A, B)` on 1 Mb bins (`layout.draw_profile`). (c) Each tumour clone's RDR and BAF from the pseudobulk by true
clone, the B counts of the planted haplotype, under the planted `(A, B)` at the clone's normal fraction
(`plot_clones_genomic`); the normal clone's tracks give way to the phase switches per Mb. With `spatial`, (c) is each
slice's true clones in the slices' shared frame, and the phase track sits under (b).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cnamaste.ab_decode.lattice import get_lattice_parameters
from cnamaste.emission import CnaEmissionConfounders, get_admixed_moments, get_hard_state_posteriors, get_log_mu_shift
from cnamaste.figure_style import get_clone_palette, get_dpi, page_size, page_style, uniform
from cnamaste.phasing import cumulative_phase_switches
from cnamaste.platforms.visium import visium_physical
from cnamaste.plot.layout import (
    GenomicAxis,
    add_axes,
    draw_profile,
    frame_tracks,
    get_gap,
    get_letter,
    get_margin,
    get_name_line,
    get_row,
    get_tick_line,
    get_track_gap,
    image_spots,
    letter,
    spatial_row,
    square_handle,
    track_furniture,
)
from cnamaste.plot.plot_clones_spatial import clone_names, get_clone_colours, plot_clones_spatial
from cnamaste.plot.plot_genomic import draw_track_axis, plot_clones_genomic
from cnamaste.qa.sims import get_project
from cnamaste.results import write_page
from cnamaste.sim.perfect import get_snp_loci
from cnamaste.sim.tree import CloneTree, get_clone_tree, get_path
from cnamaste.sim.write import get_made_dir
from cnamaste.spatio_genomic_counts import (
    PseudobulkCounts,
    SpatioGenomicCounts,
    SpatioGenomicLabelledMultiCounts,
    SpatioGenomicMultiCounts,
    get_grouped_sums,
    get_incidence,
)
from cnamaste.truth import Truth

BIN_SIZE = 1_000_000
"""Bases per bin: the planted profile and the counts are drawn on 1 Mb bins."""


def get_bin_size() -> int:
    """Return the bin size, 1,000,000 bases."""
    return BIN_SIZE


MANY_EVENTS = 10
"""Events above which (a) marks each event on its edge without its label."""


def get_many_events() -> int:
    """Return the many events, 10."""
    return MANY_EVENTS


TREE_ROW = 0.36
"""Inches of (a) per leaf: an event's two-line label fits between two leaves."""


def get_tree_row() -> float:
    """Return the tree row, 0.36 inches."""
    return TREE_ROW


TREE_MARGINS = (1.2, 1.4)
"""Inches left of the root and right of the leaves in (a), for their names and barcodes."""


def get_tree_margins() -> tuple[float, float]:
    """Return the tree margins, 1.2 inches left of the root and 1.4 right of the leaves."""
    return TREE_MARGINS


PHASE_TRACK = 0.45
"""Inches of the spatial page's phase track under (b)."""


def get_phase_track() -> float:
    """Return the phase track, 0.45 inches."""
    return PHASE_TRACK


ROOT = "root"
"""(a)'s unlabelled root, above normal and the tumour clones' common ancestor."""

TRUTH_PAGES = {False: "truth_combined.pdf", True: "truth_combined_spatial.pdf"}
"""Each page's file, by `spatial`."""


def get_truth_pages() -> dict[bool, str]:
    """Return the truth pages, `TRUTH_PAGES`."""
    return dict(TRUTH_PAGES)


class PlantedFit(NamedTuple):
    """What `plot_clones_genomic` reads of a fit to draw each clone's expectation: its confounders, `log Z_c`."""

    confounders: CnaEmissionConfounders


def get_bins(chromosome_lengths: list[int]) -> pd.DataFrame:
    """The genome on `BIN_SIZE` bins, `CHR START END`, chromosomes `1, 2, ...` in order.

    Args:
        chromosome_lengths: bases per chromosome.

    Returns:
        One row per bin.
    """
    size = get_bin_size()
    rows = []
    for chromosome, length in enumerate(chromosome_lengths, start=1):
        starts = np.arange(0, length, size)
        rows.append(pd.DataFrame({"CHR": chromosome, "START": starts, "END": np.minimum(starts + size, length)}))
    return pd.concat(rows, ignore_index=True)


def get_bin_index(bins: pd.DataFrame, chromosome: np.ndarray, position: np.ndarray) -> np.ndarray:
    """Each locus's bin; -1 off the binned chromosomes.

    Args:
        bins: `get_bins`'.
        chromosome: (n,) names, `7` or `chr7`.
        position: (n,) bases.

    Returns:
        (n,) bin indices.
    """
    names = pd.Series(np.asarray(chromosome).astype(str)).str.removeprefix("chr")
    first = bins.groupby("CHR").head(1)
    offset = dict(zip(first["CHR"].astype(str), first.index))
    count = bins["CHR"].astype(str).value_counts().to_dict()
    start = names.map(offset).to_numpy(dtype=np.float64)
    step = np.asarray(position) // get_bin_size()
    inside = np.isfinite(start) & (step < names.map(count).fillna(0).to_numpy())
    return np.where(inside, np.nan_to_num(start) + step, -1).astype(np.int64)


def get_binned_profile(truth: Truth, bins: pd.DataFrame) -> pd.DataFrame:
    """The planted `(A, B)` per bin at its middle, as `cnv_<ploidy>_seglevel.tsv` holds copies: `clone<c> A` and
    `clone<c> B` per clone, `c` the truth's label.

    Args:
        truth: the draw's truth.
        bins: `get_bins`'.

    Returns:
        `bins` with the copies.
    """
    copies = truth.copies_at(bins["CHR"].astype(str).to_numpy(), ((bins["START"] + bins["END"]) // 2).to_numpy())
    columns = {f"clone{c} {x}": copies[:, c, k] for c in range(truth.n_clones) for k, x in enumerate("AB")}
    return pd.concat([bins, pd.DataFrame(columns, index=bins.index)], axis=1)


def get_binned_counts(truth: Truth, bins: pd.DataFrame) -> SpatioGenomicLabelledMultiCounts:
    """Each spot's counts per bin: gene UMIs by the gene's middle, and the SNP UMIs of the planted haplotypes, its
    BAF channel the planted B (`Truth.get_snp_counts`); the normal clone's spots the `normal_spot` labelling.

    Args:
        truth: the draw's truth.
        bins: `get_bins`'.

    Returns:
        The counts, a spot per truth barcode.

    Raises:
        ValueError: where the baseline's genes are not the counts'.
    """
    # NB a recorded path is the project's where relative (`sim.write.get_record`).
    baseline = pd.read_csv(get_project() / truth.get_simulation()["baseline"], sep="\t", comment="#")
    umis = truth.get_umis()
    if umis.shape[1] != len(baseline):
        raise ValueError(f"{umis.shape[1]} genes counted, {len(baseline)} in the baseline")
    gene_bin = get_bin_index(bins, baseline["chrom"].to_numpy(), ((baseline["cdsStart"] + baseline["cdsEnd"]) // 2))
    snp_bin = get_bin_index(bins, *get_snp_loci(truth.get_snp_ids()))
    a_counts, b_counts = truth.get_snp_counts()

    def binned(counts: Any, at: np.ndarray) -> SpatioGenomicCounts:
        kept = np.flatnonzero(at >= 0)
        incidence = get_incidence(kept, at[kept], (at.size, len(bins)), np.int64)
        return SpatioGenomicCounts(get_grouped_sums(counts.astype(np.int64), incidence))

    multi = SpatioGenomicMultiCounts(
        binned(umis, gene_bin), binned(b_counts, snp_bin), binned(a_counts + b_counts, snp_bin)
    )
    labelled = multi.get_labelled(bins["CHR"], bins["START"], bins["END"])
    return labelled.with_labelling("normal_spot", truth.labels == 0)


def get_normal_fractions(truth: Truth) -> np.ndarray:
    """Each clone's planted normal fraction: 0 for the normal clone, the model's `normal_frac` for a tumour clone.

    Args:
        truth: the draw's truth.

    Returns:
        (n_clones,) fractions.
    """
    fraction = float(truth.get_simulation()["model"]["normal_frac"])
    return np.array([0.0, *[fraction] * (truth.n_clones - 1)])


def get_planted_tracks(
    profile: pd.DataFrame, pseudobulk: PseudobulkCounts, normal_fractions: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Each clone's expected RDR and BAF per bin: its planted `(A, B)` admixed at its normal fraction
    (`eqn:ab_admixture`), the depth over the clone's `Z_c` on its pseudobulk's baseline (`get_log_mu_shift`).

    Args:
        profile: `get_binned_profile`'s.
        pseudobulk: by true clone, a row per clone.
        normal_fractions: (n_clones,) each clone's.

    Returns:
        `(n_clones, n_bins)` RDR and BAF, and `(n_clones, 1)` `log Z_c`.
    """
    rdr, baf, log_mu_shift = [], [], []
    for c, fraction in enumerate(normal_fractions):
        pairs = profile[[f"clone{c} A", f"clone{c} B"]].to_numpy()
        states, path = np.unique(pairs, axis=0, return_inverse=True)
        lattice = get_lattice_parameters(states, None, None)
        depth, share = get_admixed_moments(lattice, float(fraction))
        umis, base, alt, total = (np.asarray(v)[c : c + 1] for v in pseudobulk[:4])
        row = PseudobulkCounts(umis, base, alt, total, None, {})
        weights = get_hard_state_posteriors(path.reshape(-1), len(states))
        log_z = get_log_mu_shift(lattice._replace(log_mu=np.log(depth)), weights, row)
        rdr.append(depth[path.reshape(-1)] / np.exp(log_z[0, 0]))
        baf.append(share[path.reshape(-1)])
        log_mu_shift.append(log_z[0])
    return np.array(rdr), np.array(baf), np.array(log_mu_shift)


def get_switches_per_mb(truth: Truth, bins: pd.DataFrame) -> np.ndarray:
    """The planted phase switches counted along each chromosome to each bin (`phasing.cumulative_phase_switches`, a
    SNP a block), over the chromosome's Mb: each chromosome's track ends at its switch rate.

    Args:
        truth: the draw's truth.
        bins: `get_bins`'.

    Returns:
        (n_bins,) switches per Mb.
    """
    chromosome, position = get_snp_loci(truth.get_snp_ids())
    snp_bin = get_bin_index(bins, chromosome, position)
    blocks = pd.DataFrame({"CHR": chromosome, "block_id": np.arange(chromosome.size), "bin_id": snp_bin})
    assert truth.phase is not None
    counted = cumulative_phase_switches(blocks, truth.phase.astype(np.int64))  # type: ignore[no-untyped-call]
    # NB a bin with no SNP carries the count of the bin before it on its chromosome.
    found = pd.Series(np.nan, index=bins.index)
    found.iloc[np.unique(snp_bin)] = counted
    filled = found.groupby(bins["CHR"]).ffill().fillna(0.0).to_numpy()
    megabases = bins.groupby("CHR")["END"].transform("max").to_numpy() / 1e6
    return np.asarray(filled / megabases)


def draw_phase_track(ax: Any, switches: np.ndarray, genome: GenomicAxis, labels: bool) -> None:
    """The phase switches per Mb (`get_switches_per_mb`) as a track on `genome`, framed as the clones' tracks.

    Args:
        ax: the track.
        switches: (n_bins,) per Mb.
        genome: the page's genomic axis.
        labels: whether to name the chromosomes under it.
    """
    ax.plot(np.arange(switches.size), switches, drawstyle="steps-mid", color="black", linewidth=0.5)
    high = max(float(np.max(switches)), 0.1)
    draw_track_axis(ax, "", (-0.05 * high, 1.05 * high), [0.0, high], True, switches.size)
    genome.frame(ax, labels=labels)
    track_furniture(ax, "Switches\nper Mb")


def get_tree_layout(tree: CloneTree) -> tuple[dict[str, float], dict[str, float]]:
    """Each node's place in (a): x in events from the root, an edge as long as its events and at least 1, every leaf
    at the deepest leaf's; y the leaves `n - 1, ..., 0` top down in the tree's walk order, a node at its children's
    mean.

    Args:
        tree: the tree.

    Returns:
        x and y per node.
    """
    children: dict[str, list[str]] = {node: [] for node in tree.parent}
    for node, up in tree.parent.items():
        if up is not None:
            children[up].append(node)
    x: dict[str, float] = {}
    for node, up in tree.parent.items():
        x[node] = 0.0 if up is None else x[up] + max(len(tree.edge_events[node]), 1)
    leaves = [node for node in tree.parent if not children[node]]
    width = max(x[leaf] for leaf in leaves)
    y: dict[str, float] = {}
    order: list[str] = []

    def walk(node: str) -> None:
        for child in children[node]:
            walk(child)
        if not children[node]:
            order.append(node)

    walk(next(node for node, up in tree.parent.items() if up is None))
    for k, leaf in enumerate(order):
        x[leaf], y[leaf] = width, float(len(order) - 1 - k)
    for node in reversed(list(tree.parent)):
        if children[node]:
            y[node] = float(np.mean([y[child] for child in children[node]]))
    return x, y


def get_drawn_tree(tree: CloneTree) -> CloneTree:
    """`tree` as (a) draws it: an unlabelled root, `ROOT`, above normal and the node below normal, normal a leaf
    whose edge carries no CNAs.

    Args:
        tree: the tree, rooted at normal.

    Returns:
        The tree rooted at `ROOT`.
    """
    parent: dict[str, str | None] = {ROOT: None, "normal": ROOT}
    parent |= {node: ROOT if up == "normal" else up for node, up in tree.parent.items() if up is not None}
    return CloneTree(parent, {ROOT: ()} | dict(tree.edge_events))


def get_event_barcodes(tree: CloneTree) -> dict[str, str]:
    """Each node's event barcode: one bit per event, in the order the events arose (by events from the root, then the
    tree's walk), set where the event is on the node's path.

    Args:
        tree: the tree.

    Returns:
        The barcode per node.
    """
    arisen = []
    for node, up in tree.parent.items():
        depth = sum(len(tree.edge_events[n]) for n in get_path(tree, up)) if up is not None else 0
        arisen += [(depth + k, node) for k in range(len(tree.edge_events[node]))]
    events = [node for _, node in sorted(arisen, key=lambda pair: pair[0])]
    return {node: "".join("1" if e in get_path(tree, node) else "0" for e in events) for node in tree.parent}


def draw_clone_tree(ax: Any, tree: CloneTree, names: dict[str, str], colours: dict[str, str]) -> None:
    """(a): the tree from an unlabelled root to normal and to the tumour clones' common ancestor
    (`get_drawn_tree`), along events, each edge's events marked by ticks and listed above it as `chr<c> (A,B) <Mb>`,
    each observed clone coloured and named with its event barcode, an unobserved ancestor drawn open.

    Args:
        ax: the panel.
        tree: the tree.
        names: each observed clone's name.
        colours: each observed clone's colour.
    """
    tree = get_drawn_tree(tree)
    x, y = get_tree_layout(tree)
    barcodes = get_event_barcodes(tree)
    labelled = sum(len(events) for events in tree.edge_events.values()) <= get_many_events()
    for node, up in tree.parent.items():
        if up is None:
            continue
        ax.plot([x[up], x[up], x[node]], [y[up], y[node], y[node]], color="grey", linewidth=0.8)
        events = tree.edge_events[node]
        for k in range(len(events)):
            at = x[up] + k + 0.5
            ax.plot([at, at], [y[node] - 0.12, y[node] + 0.12], color="black", linewidth=0.8)
        if labelled and events:
            # NB an edge's labels stacked in one block over its ticks, one line per CNA, so none collide.
            lines = [f"chr{e.chromosome} ({e.a},{e.b}) {max(round((e.end - e.start) / 1e6), 1)} Mb" for e in events]
            middle = x[up] + 0.5 * len(events)
            ax.text(middle, y[node] + 0.15, "\n".join(lines), ha="center", va="bottom", linespacing=1.0)
    for node, up in tree.parent.items():
        observed = node in names
        ax.scatter(x[node], y[node], s=18.0, color=colours[node] if observed else "white", edgecolors="grey", zorder=3)
        if observed:
            # NB the root's name to its left, as its edge leaves to the right.
            leaf = up is not None
            label = f"{names[node]}  {barcodes[node]}" if leaf else f"{barcodes[node]}  {names[node]}"
            ax.annotate(
                label,
                (x[node], y[node]),
                xytext=(5.0 if leaf else -5.0, 0.0),
                textcoords="offset points",
                ha="left" if leaf else "right",
                va="center",
            )
    # NB inches either side of the tree for the root's and the leaves' names and barcodes.
    left, right = get_tree_margins()
    inches = ax.get_position().width * ax.figure.get_size_inches()[0]
    width = max(x.values())
    unit = width / (inches - left - right)
    ax.set_xlim(-left * unit, width + right * unit)
    ax.set_ylim(min(y.values()) - 0.5, max(y.values()) + 0.6)
    ax.axis("off")


def get_slice_coords(truth: Truth) -> np.ndarray:
    """Each spot's place in the slices' shared frame, um and y up as `platforms.visium.physical_coords` places a
    Visium spot: each slice moved by its offset in shares of a slice's extent, as `sim.layout` places it.

    Args:
        truth: the draw's truth.

    Returns:
        (n_barcodes, 2) `(x, y)`.
    """
    simulation = truth.get_simulation()
    row, col = truth.coords[:, 0], truth.coords[:, 1]
    # NB a square array is Visium HD's grid at unit spacing.
    hexagonal = visium_physical(truth.coords)  # type: ignore[no-untyped-call]
    xy = hexagonal if simulation["array"][0] == "hex" else np.column_stack([col, -row])
    assert truth.sample_ids is not None
    slices = list(dict.fromkeys(truth.sample_ids))
    first = xy[truth.sample_ids == slices[0]]
    extent = first.max(axis=0) - first.min(axis=0)
    for sample_id, piece in zip(slices, simulation["slices"], strict=True):
        xy[truth.sample_ids == sample_id] += np.asarray(piece["offset"]) * extent * [1.0, -1.0]
    return np.asarray(xy)


def draw_slices(figure: Any, top: float, truth: Truth, colours: dict[str, str]) -> None:
    """(c) of the spatial page: each slice's true clones on its own panel, every panel framed on the slices' shared
    frame (`layout.spatial_row`), one key of every clone and its share of the spots.

    Args:
        figure: the page.
        top: inches.
        truth: the draw's truth.
        colours: each clone's colour, by `clone <c>`.
    """
    assert truth.sample_ids is not None
    margin = get_margin()
    xy = get_slice_coords(truth)
    assignment = np.array([f"clone {c}" for c in truth.labels])
    panels = []
    for sample_id in dict.fromkeys(truth.sample_ids):
        on = truth.sample_ids == sample_id
        ax = add_axes(figure, margin["left"], top, 1.0, 1.0)
        found = pd.Series(assignment[on])
        plot_clones_spatial(
            xy[on], found, sample_list=[sample_id], ax=ax, colorlist=[colours[c] for c in np.unique(found)]
        )
        ax.get_legend().remove()
        panels.append(ax)
    # NB the last panel's extent frames every panel: the slices' shared frame.
    panels[-1].update_datalim(np.array([xy.min(axis=0), xy.max(axis=0)]))
    ids = list(np.unique(assignment))
    key = (
        [square_handle(colours[c]) for c in ids],
        clone_names(ids, {c: float(np.mean(assignment == c)) for c in ids}),
    )
    spatial_row(figure, top, panels, [key], get_row())
    for ax in panels:
        image_spots(ax)


def get_truth_combined_figure(truth: Truth, *, spatial: bool) -> Any:
    """The truth page: (a) the clone tree, (b) the planted profile, (c) each tumour clone's RDR and BAF over the phase
    track, or with `spatial` each slice's true clones, the phase track under (b).

    Args:
        truth: a draw's truth (`truth.read_sim_truth`).
        spatial: which page.

    Returns:
        The figure.
    """
    assert truth.tree is not None
    tree = get_clone_tree(truth.tree)
    bins = get_bins(truth.get_simulation()["chromosome_lengths"])
    profile = get_binned_profile(truth, bins)
    genome = GenomicAxis.from_bins(bins["CHR"].astype(str))
    ids = [f"clone {c}" for c in range(truth.n_clones)]
    colour_of = dict(zip(np.unique(ids), get_clone_colours(np.unique(ids), get_clone_palette())))
    names = {clone: str(name) for clone, name in zip(truth.clones, clone_names(ids), strict=True)}
    switches = get_switches_per_mb(truth, bins)

    margin = get_margin()
    with page_style() as size:
        figure = plt.figure(figsize=page_size("full"), dpi=get_dpi(), facecolor="white")
        page_width, page_height = figure.get_size_inches()
        width = page_width - margin["left"] - margin["right"]
        top = margin["top"]
        letter(figure, top, "a")
        top += get_letter()
        drawn = get_drawn_tree(tree).parent
        leaves = sum(1 for node in drawn if node not in drawn.values())
        # NB a row per leaf, and a row more per two CNAs past two on the edge holding the most.
        stacked = max(len(events) for events in tree.edge_events.values())
        tree_height = get_tree_row() * (leaves + 1 + max(stacked - 2, 0) // 2)
        colours = {clone: colour_of[c] for clone, c in zip(truth.clones, ids, strict=True)}
        draw_clone_tree(add_axes(figure, margin["left"], top, width, tree_height), tree, names, colours)
        top += tree_height + get_gap()

        letter(figure, top, "b")
        top = draw_profile(figure, top, profile, genome, list(range(truth.n_clones)))

        if spatial:
            draw_phase_track(add_axes(figure, margin["left"], top, width, get_phase_track()), switches, genome, True)
            top += get_phase_track() + get_tick_line() + get_gap()
            letter(figure, top, "c")
            draw_slices(figure, top + get_letter(), truth, colour_of)
        else:
            letter(figure, top, "c")
            top += get_letter()
            counts = get_binned_counts(truth, bins).with_labelling("clone", truth.labels)
            fractions = get_normal_fractions(truth)
            _, _, log_mu_shift = get_planted_tracks(profile, counts.get_pseudobulk("clone"), fractions)
            n_tumour = truth.n_clones - 1
            fixed = n_tumour * (get_name_line() + get_track_gap() + get_gap() / 2.0) + get_gap() / 2.0
            track = (page_height - top - margin["bottom"] - fixed) / (2 * n_tumour + 1)
            draw_phase_track(add_axes(figure, margin["left"], top, width, track), switches, genome, False)
            at = top + track + get_gap() / 2.0
            # NB the normal clone's tracks are drawn off the page and removed: the phase track takes their place.
            axes = [add_axes(figure, 0.0, page_height, 1.0, 0.1) for _ in range(2)]
            for _ in range(n_tumour):
                at += get_name_line()
                for t in range(2):
                    axes.append(add_axes(figure, margin["left"], at, width, track))
                    at += track + (get_track_gap() if t == 0 else get_gap() / 2.0)
            plot_clones_genomic(
                counts,
                df_cnv=profile,
                # NB plot_clones_genomic reads a fit's `confounders`; its annotation says a dict.
                res_combine=PlantedFit(CnaEmissionConfounders(fractions[:, None], log_mu_shift)),  # type: ignore[arg-type]
                axes=axes,
                genome=genome,
                normal_fractions=dict(zip(ids, fractions.tolist())),
                clone_labels=truth.labels,
            )
            for ax in axes[:2]:
                ax.remove()
            frame_tracks(axes[2:], 2)
        uniform(figure, size)
    return figure


def write_truth_pages(truth: Truth, into: Path | None) -> list[Path]:
    """Both truth pages (`get_truth_pages`), each at its page size with no creation date (`results.write_page`), so a
    page's bytes are what it draws.

    Args:
        truth: a draw's truth.
        into: the directory, made within its existing parent where missing; None for the draw's `qa/`.

    Returns:
        The pages written.
    """
    assert truth.sample is not None
    out, name = (truth.sample, "qa") if into is None else (into.parent, into.name)
    directory = get_made_dir(out, name, exist_ok=True)
    for spatial, name in get_truth_pages().items():
        write_page(str(directory), name, get_truth_combined_figure(truth, spatial=spatial))
    return [directory / name for name in get_truth_pages().values()]
