"""The benchmark's supplementary figures of one sample, one figure per letter page, in one PDF.

Author: Runpeng Luo (2026-10-30)

Dependencies: cnamaste (`figure_style`, `logger`, `plot.layout`), scipy, matplotlib; `plot_umi_distribution`, beside
this script.

Usage:
    uv run python python/cnamaste/sandbox/luo/benchmark/scripts/plot_supplement.py --sample NAME --bb_dir SIDE_UM=DIR [--bb_dir SIDE_UM=DIR ...] \
        [--assay VISIUM3prime] [--bins 100] [--outdir DIR] [-o supplement.pdf]

    From the repository root; `python/cnamaste/sandbox/luo/benchmark/README.md` gives a sample's run.

Inputs:
    bb_dir: a UGP `bb` directory and its bins' side in um, `<side_um>=<dir>`, once per side: its `<assay>.h5ad` and
        `multi_snp/<assay>/bb.{A,B}allele.npz`, (bins, spots).
Parameters:
    sample: the sample's name, in each histogram's title.
    assay: UGP's assay type, the stem of the AnnData and the allele matrices' directory.
    bins: each histogram's bin count.
    outdir: the directory the supplement is written under, `python/cnamaste/sandbox/luo/benchmark/output/` by default.
Outputs:
    output: the supplement, `supplement.pdf` by default. Page 1: one row per `--bb_dir`, in the order given and
        lettered (a), (b), ..., its per-spot total UMI left and SNP-covering UMI right (`plot_umi_distribution`).
Notes/References:
    matplotlib's multi-page PDF: https://matplotlib.org/stable/api/backend_pdf_api.html#matplotlib.backends.backend_pdf.PdfPages
"""

from __future__ import annotations

import argparse
import string
from pathlib import Path

import matplotlib.pyplot as plt
import scipy.sparse as sp
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from cnamaste.config import start_time
from cnamaste.figure_style import PAPER_WIDTH, page_style, restyle
from cnamaste.logger import get_logger
from cnamaste.plot.layout import GAP, LETTER, add_axes, letter
from cnamaste.sandbox.luo.benchmark.scripts.plot_umi_distribution import (
    HISTOGRAM_HEIGHT,
    LABEL_ROOM,
    OUTPUT,
    PAGE,
    PAGE_MARGIN,
    XLABEL_ROOM,
    plot_spot_snp_umi,
    plot_spot_total_umi,
)

logger = get_logger("plot_supplement", start_time=start_time)


def get_bb_dir(text: str) -> tuple[int, Path]:
    """Read a `--bb_dir` value, `<side_um>=<dir>`.

    Args:
        text: the value.

    Returns:
        The bins' side in um and the directory.

    Raises:
        argparse.ArgumentTypeError: where `text` is not `<int>=<dir>`.
    """
    side, sep, directory = text.partition("=")
    if not sep or not side.isdigit() or not directory:
        raise argparse.ArgumentTypeError(f"expected <side_um>=<dir>, got {text!r}")
    return int(side), Path(directory)


def plot_umi_page(sample: str, bb_dirs: list[tuple[int, Path]], assay: str, bins: int = 100) -> Figure:
    """Draw a letter page of per-spot UMI histograms: a row per bin side, lettered (a), (b), ..., total UMI left and
    SNP-covering UMI right, the rows in the text column from its top.

    Args:
        sample: the sample's name.
        bb_dirs: each bin side in um and its UGP `bb` directory, one row each, in order.
        assay: UGP's assay type.
        bins: each histogram's bin count.

    Returns:
        The page.

    Raises:
        ValueError: where the rows run past the text column's bottom.
    """
    column = PAPER_WIDTH / 2.0
    row = LETTER + LABEL_ROOM[1] + HISTOGRAM_HEIGHT + XLABEL_ROOM
    if PAGE_MARGIN + len(bb_dirs) * row > PAGE[1] - PAGE_MARGIN:
        raise ValueError(f"{len(bb_dirs)} rows of {row:.2f} in run past a letter page's text column")
    with page_style():
        figure = plt.figure(figsize=PAGE, facecolor="white")
        top = PAGE_MARGIN
        for r, (side_um, bb_dir) in enumerate(bb_dirs):
            letter(figure, top, string.ascii_lowercase[r], x0=PAGE_MARGIN)
            total_ax, snp_ax = (
                add_axes(
                    figure,
                    PAGE_MARGIN + c * column + LABEL_ROOM[0],
                    top + LETTER + LABEL_ROOM[1],
                    column - LABEL_ROOM[0] - GAP,
                    HISTOGRAM_HEIGHT,
                )
                for c in range(2)
            )
            snp_dir = bb_dir / "multi_snp" / assay
            plot_spot_total_umi(bb_dir / f"{assay}.h5ad", sample, side_um, output=None, bins=bins, ax=total_ax)
            logger.info(f"Reading {snp_dir}'s A and B allele matrices.")
            plot_spot_snp_umi(
                sp.load_npz(snp_dir / "bb.Aallele.npz"),
                sp.load_npz(snp_dir / "bb.Ballele.npz"),
                sample,
                side_um,
                output=None,
                bins=bins,
                ax=snp_ax,
            )
            top += row
    return figure


def write_pages(figures: list[Figure], output: str | Path) -> Path:
    """Write `figures` to one PDF, one per page in order, each at its own size, untrimmed, in the page's face.

    A figure drawn on a whole letter page fills one page; trimming it would leave a page the figure's extent alone.

    Args:
        figures: the figures; each is closed once written.
        output: the PDF's path; its directory is created.

    Returns:
        The written path.
    """
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with page_style(), PdfPages(output) as pdf:
        for figure in figures:
            restyle(figure)
            pdf.savefig(figure, facecolor="white")
            plt.close(figure)
    logger.info(f"Wrote {output}: {len(figures)} pages.")
    return output


def parser() -> argparse.ArgumentParser:
    """The command's arguments."""
    command = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    command.add_argument("--sample", required=True)
    command.add_argument("--bb_dir", type=get_bb_dir, action="append", required=True)
    command.add_argument("--assay", default="VISIUM3prime")
    command.add_argument("--bins", type=int, default=100)
    command.add_argument("--outdir", type=Path, default=OUTPUT)
    command.add_argument("-o", "--output", type=Path, default=Path("supplement.pdf"))
    return command


def main(argv: list[str] | None = None) -> int:
    """Draw the supplement's pages and write them as one PDF."""
    args = parser().parse_args(argv)
    pages = [plot_umi_page(args.sample, args.bb_dir, args.assay, args.bins)]
    write_pages(pages, args.outdir / args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
