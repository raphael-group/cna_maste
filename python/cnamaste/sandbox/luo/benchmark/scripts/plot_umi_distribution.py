"""One sample's per-spot UMI distributions: total UMI from its AnnData, SNP-covering UMI from its allele matrices.

Author: Runpeng Luo (2026-10-30)

Dependencies: cnamaste (`figure_style`, `logger`), anndata, scipy, matplotlib.

Usage:
    uv run python python/cnamaste/sandbox/luo/benchmark/scripts/plot_umi_distribution.py --sample NAME --side_um UM [--h5ad FILE] \
        [--aallele FILE --ballele FILE] [--spot_axis 1] [--bins 100] [--outdir DIR] \
        [--total_umi_pdf NAME.pdf] [--snp_umi_pdf NAME.pdf]

    From the repository root. `--h5ad` draws the total UMI; `--aallele` with `--ballele` draws the SNP-covering UMI.
    `python/cnamaste/sandbox/luo/benchmark/README.md` gives a sample's run.

Inputs:
    h5ad: an AnnData of (spots, genes) UMI counts in `X`.
    aallele, ballele: `scipy.sparse.save_npz` matrices of A- and B-allele UMI counts, one axis spots.
Parameters:
    sample, side_um: the sample's name and its bins' side in um, the title's `<sample> (<side_um> um)`.
    spot_axis: the allele matrices' spot axis: 1 for UGP's (bins, spots) `bb.{A,B}allele.npz`, 0 for CalicoST's
        (spots, SNPs) `cell_snp_{A,B}allele.npz`.
    bins: the histogram's bin count.
    outdir: the directory the PDFs are written under, `python/cnamaste/sandbox/luo/benchmark/output/` by default.
Outputs:
    total_umi_pdf: the per-spot total UMI histogram, default `spot_total_umi_histplot.pdf`.
    snp_umi_pdf: the per-spot SNP-covering UMI histogram, default `spot_snp_umi_histplot.pdf`.
Notes/References:
    The title's in-tissue spots are the input's spots: UGP's `bb/<assay>.h5ad` and allele matrices hold the barcodes of
    Space Ranger's filtered matrix, the bins under tissue, and UGP filters genes only
    (Universal-Genotyping-Pipeline `workflow/scripts/process_rna_anndata.py`). The total UMI is summed over the genes
    the AnnData keeps.
    The SNP-covering UMI of a spot is A + B summed over its SNPs or bins: the count `cnamaste.io.load_input_data`
    floors at `quality.spot_min_snp_umis`, and CalicoST's `load_data` at `min_snpumi_perspot`
    (https://github.com/raphael-group/CalicoST/blob/c1abcae/src/calicost/utils_IO.py#L72).
    Space Ranger's filtered matrix: https://www.10xgenomics.com/support/software/space-ranger/latest/analysis/outputs/output-overview
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import anndata
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from matplotlib.figure import Figure

from cnamaste.config import start_time
from cnamaste.figure_style import PAPER_WIDTH, page_style, uniform
from cnamaste.logger import get_logger
from cnamaste.plot.layout import add_axes
from cnamaste.utils import write_fig

logger = get_logger("plot_umi_distribution", start_time=start_time)

OUTPUT = Path(__file__).resolve().parents[1] / "output"
"""`python/cnamaste/sandbox/luo/benchmark/output/`, gitignored: the default output directory, found from this script whatever the working
directory."""

BAR_COLOUR = "#2a78d6"
"""The bars' fill: the dataviz palette's first categorical slot."""

INK = "#52514e"
"""The median line and its label: secondary ink."""

GRID = "#e1e0d9"
"""The gridlines: hairline grey."""

PAGE = (8.5, 11.0)
"""Inches: a US letter page, each figure's canvas; `figure_style.PAPER_WIDTH` is its text column."""

PAGE_MARGIN = (PAGE[0] - PAPER_WIDTH) / 2.0
"""Inches from the page's left and top edges to its text column."""

LABEL_ROOM = (0.6, 0.4)
"""Inches inside the text column left of the axes, for the y label and ticks, and above it, for the two-line title."""

XLABEL_ROOM = 0.45
"""Inches under the axes for the x ticks and label."""

HISTOGRAM_HEIGHT = 2.0
"""Inches: a histogram's axes."""


def get_spot_sums(counts: Any, spot_axis: int = 0) -> np.ndarray:
    """Sum `counts` over every axis but the spots'.

    Args:
        counts: a 2-D count matrix, sparse or dense.
        spot_axis: the axis indexing spots.

    Returns:
        The per-spot sums, int64 for integer counts.
    """
    sums = np.asarray(counts.sum(axis=1 - spot_axis)).ravel()
    return sums.astype(np.int64) if np.all(np.mod(sums, 1) == 0) else sums


def plot_umi_histogram(
    umi: np.ndarray, xlabel: str, sample: str, side_um: int, bins: int = 100, ax: Any = None
) -> Figure:
    """Draw a histogram of per-spot UMI, its median marked, titled by sample, bin side and spots.

    Without `ax`, the figure is a whole letter `PAGE`, the axes the text column wide at its top, so a multi-page PDF
    holds one figure per page; a PDF written trimmed (`write_figure`) is the drawn extent alone.

    Args:
        umi: one value per in-tissue spot.
        xlabel: the quantity drawn, the x axis' label.
        sample: the sample's name.
        side_um: the bins' side in um.
        bins: the histogram's bin count.
        ax: the axes to draw on, or None for a page of its own.

    Returns:
        The figure `ax` is on, in the page's face (`figure_style.page_style`).
    """
    median = float(np.median(umi))
    with page_style() as size:
        if ax is None:
            figure = plt.figure(figsize=PAGE, facecolor="white")
            ax = add_axes(
                figure,
                PAGE_MARGIN + LABEL_ROOM[0],
                PAGE_MARGIN + LABEL_ROOM[1],
                PAPER_WIDTH - LABEL_ROOM[0],
                HISTOGRAM_HEIGHT,
            )
        figure, axis = ax.figure, ax
        axis.hist(umi, bins=bins, color=BAR_COLOUR, edgecolor="white", linewidth=0.3)
        axis.axvline(median, color=INK, linestyle="--", linewidth=1.0)
        axis.annotate(
            f"median {median:,.0f}",
            xy=(median, 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(3, -3),
            textcoords="offset points",
            va="top",
            color=INK,
        )
        axis.set_xlabel(xlabel)
        axis.set_ylabel("spots")
        axis.set_title(rf"{sample} ({side_um} $\mu$m)" + f"\n#in-tissue spots={umi.size:,}")
        axis.grid(axis="y", color=GRID, linewidth=0.5)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_xlim(left=0)
        uniform(figure, size)
    logger.info(
        f"Drew {xlabel}, {sample} {side_um} um: {umi.size:_} spots, {int(np.sum(umi == 0)):_} at 0, "
        f"median {median:_.1f}."
    )
    return figure


def write_figure(figure: Figure, output: str | Path | None) -> None:
    """Write `figure` to `output` as a PDF in the page's face (`utils.write_fig`), its directory created.

    Args:
        figure: the figure; closed once written, and still saveable.
        output: the PDF's path, or None to write nothing.
    """
    if output is None:
        return
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_fig(str(output.parent), output.name, figure)


def plot_spot_total_umi(
    h5ad_file: str | Path,
    sample: str,
    side_um: int,
    output: str | Path | None = OUTPUT / "spot_total_umi_histplot.pdf",
    bins: int = 100,
    ax: Any = None,
) -> Figure:
    """Draw the distribution of per-spot total UMI of an AnnData, and write it as a PDF unless `output` is None.

    Args:
        h5ad_file: an AnnData of (spots, genes) UMI counts in `X`.
        sample: the sample's name.
        side_um: the bins' side in um.
        output: the PDF to write, or None to write nothing.
        bins: the histogram's bin count.
        ax: the axes to draw on, or None for a page of its own (`plot_umi_histogram`).

    Returns:
        The figure.
    """
    logger.info(f"Reading {h5ad_file}.")
    adata = anndata.read_h5ad(h5ad_file)
    umi = get_spot_sums(adata.X, spot_axis=0)
    figure = plot_umi_histogram(umi, "total UMI per spot", sample, side_um, bins, ax=ax)
    write_figure(figure, output)
    return figure


def plot_spot_snp_umi(
    a_allele: Any,
    b_allele: Any,
    sample: str,
    side_um: int,
    output: str | Path | None = OUTPUT / "spot_snp_umi_histplot.pdf",
    spot_axis: int = 1,
    bins: int = 100,
    ax: Any = None,
) -> Figure:
    """Draw the distribution of per-spot SNP-covering UMI, A + B, and write it as a PDF unless `output` is None.

    Args:
        a_allele: A-allele UMI counts, sparse or dense, one axis spots.
        b_allele: B-allele UMI counts, the shape of `a_allele`.
        sample: the sample's name.
        side_um: the bins' side in um.
        output: the PDF to write, or None to write nothing.
        spot_axis: the axis indexing spots, 1 for UGP's (bins, spots), 0 for CalicoST's (spots, SNPs).
        bins: the histogram's bin count.
        ax: the axes to draw on, or None for a page of its own (`plot_umi_histogram`).

    Returns:
        The figure.

    Raises:
        ValueError: where the two matrices' shapes differ.
    """
    if a_allele.shape != b_allele.shape:
        raise ValueError(f"A and B allele shapes differ: {a_allele.shape} != {b_allele.shape}")
    umi = get_spot_sums(a_allele, spot_axis) + get_spot_sums(b_allele, spot_axis)
    figure = plot_umi_histogram(umi, "SNP-covering UMI per spot", sample, side_um, bins, ax=ax)
    write_figure(figure, output)
    return figure


def parser() -> argparse.ArgumentParser:
    """The command's arguments."""
    command = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    command.add_argument("--sample", required=True)
    command.add_argument("--side_um", type=int, required=True)
    command.add_argument("--h5ad", type=Path)
    command.add_argument("--aallele", type=Path)
    command.add_argument("--ballele", type=Path)
    command.add_argument("--spot_axis", type=int, choices=(0, 1), default=1)
    command.add_argument("--bins", type=int, default=100)
    command.add_argument("--outdir", type=Path, default=OUTPUT)
    command.add_argument("--total_umi_pdf", type=Path, default=Path("spot_total_umi_histplot.pdf"))
    command.add_argument("--snp_umi_pdf", type=Path, default=Path("spot_snp_umi_histplot.pdf"))
    return command


def main(argv: list[str] | None = None) -> int:
    """Draw the total UMI where `--h5ad` is given, the SNP-covering UMI where both allele matrices are."""
    command = parser()
    args = command.parse_args(argv)
    if args.h5ad is None and (args.aallele is None or args.ballele is None):
        command.error("give --h5ad, or --aallele with --ballele, or all three")
    if (args.aallele is None) != (args.ballele is None):
        command.error("--aallele and --ballele are given together")
    if args.h5ad is not None:
        plot_spot_total_umi(args.h5ad, args.sample, args.side_um, args.outdir / args.total_umi_pdf, args.bins)
    if args.aallele is not None:
        logger.info(f"Reading {args.aallele} and {args.ballele}.")
        plot_spot_snp_umi(
            sp.load_npz(args.aallele),
            sp.load_npz(args.ballele),
            args.sample,
            args.side_um,
            args.outdir / args.snp_umi_pdf,
            args.spot_axis,
            args.bins,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
