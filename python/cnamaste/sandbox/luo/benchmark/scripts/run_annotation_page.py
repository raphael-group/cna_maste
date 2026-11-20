"""One Visium HD sample's spot labels and clone profiles, method by method, on one page (`plot_annotation_page`).

Author: Runpeng Luo (2026-10-30)

Dependencies: cnplot (`[tool.uv.sources]`), for the profiles and tracks; `read_numbat`, beside this script, for
Numbat's run.

Usage:
    uv run python python/cnamaste/sandbox/luo/benchmark/scripts/run_annotation_page.py --spaceranger_dir DIR --celltype FILE \
        --cnamaste_labels FILE --cnamaste_seglevel FILE [--cnamaste_binned FILE] \
        [--cnamaste_normal_fraction CLONE=F ...] --calicost_dir DIR --numbat_dir DIR [--outdir DIR] [-o combined.pdf]

    From the repository root; `python/cnamaste/sandbox/luo/benchmark/README.md` gives a sample's run.

Inputs:
    spaceranger_dir: a Space Ranger directory of the sample, any bin side: `spatial/tissue_positions.parquet`, which
        places every bin (`sandbox.luo.platforms.visium.get_visiumhd_bin_pixels`), `scalefactors_json.json` and the H&E image.
    celltype: per-spot cell types, `barcode` and `cell_type` (`annotation.load_celltype`), `<broad> (<fine>)`; drawn
        as epithelial or not by the broad class, `LOW_CONFIDENCE` spots left out.
    cnamaste_labels: a run's `clone_labels_int.tsv[.gz]`; cnamaste_seglevel: its `cnv_<ploidy>_seglevel.tsv`.
    cnamaste_binned: the run's `binned_data.npz` (`run.write_binned_data`), with `clone_labels.tsv[.gz]` beside it, the
        clones `cnamaste_seglevel` decodes, and `rdrbaf_final_nstates<n>_smp.npz`, the fit, its `new_log_mu_shift`
        each clone's `log Z_c`: (b)'s per-clone pseudobulk RDR and BAF and their expectation
        (`pseudobulk.get_clone_pseudobulk`). Absent, the page has no tracks and the profiles are (b).
    calicost_dir: a CalicoST sample directory, read at its best rectangle (`calicost.read.best_run`): its
        `clone_labels.tsv` and `cnv<_ploidy>_seglevel.tsv` at `calicost_ploidy`.
    numbat_dir: a Numbat output directory: its clones (`read_numbat.read_labels`) and each clone's CNV state per
        segment, its `bulk_clones_final.tsv.gz` (`read_numbat.read_clone_profile`), in Numbat's colours and names
        (`STATE_COLOURS`, `STATE_LABELS`).
Parameters:
    side_um: the side of the bins drawn; coarser bins' labels are given to each such bin inside them.
    cnamaste_normal_fraction: `<clone>=<f>`, once per clone, the decoder's normal fraction f_c: `run_cnamaste` writes
        it to no table but logs it (`<decoder> normal fractions per clone`) and keys it on its `combined.png`,
        `Clone I (<spots>%, <f_c>%)`, exact there, f_c being on a 0.01 grid (`hmm_ab_decoder.NORMAL_FRACTIONS`). A clone
        given none is expected at its fitted states, as `plot_clones_genomic` draws it.
    calicost_ploidy: CalicoST's table, `unconstrained` its `cnv_seglevel.tsv`, which its `plots/acn_genome.pdf` draws.
    region_bed, chrom_sizes: the regions and chromosome lengths the tracks' and profiles' genome axis is drawn on, by
        default `python/cnamaste/sandbox/luo/benchmark/data/`'s copies of UGP's resources/data/hg38.regions.bed and hg38.chrom.sizes: a
        chromosome's arms, its centromere between them.
    outdir: the directory the page is written under, `python/cnamaste/sandbox/luo/benchmark/output/` by default.
Outputs:
    output: the page, `.pdf` or `.png` (`layout.write_page`), its path under `outdir`, `combined.pdf` by default.
Notes/References:
    Numbat is titled by the side of the bins it ran on, read from its barcodes (`sandbox.luo.platforms.visium.get_visiumhd_bins`).
    Numbat's outputs: https://kharchenkolab.github.io/numbat/articles/descriptions.html
    cnplot: https://github.com/raphael-group/cnplot
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cnamaste.annotation import load_celltype
from cnamaste.calicost.read import CALICOST_PLOIDIES, best_run, read_estimate as read_calicost_estimate
from cnamaste.config import start_time
from cnamaste.he import slide_image_and_meta
from cnamaste.io import get_spatial_positions
from cnamaste.logger import get_logger
from cnamaste.platforms.visium import get_visiumhd_microns_per_pixel, pixel_physical
from cnamaste.plot.layout import write_page
from cnamaste.qa.metrics import read_labels
from cnamaste.qa.plot import NAMES as VALIDATION_NAMES
from cnamaste.sandbox.luo.benchmark.scripts.read_numbat import (
    STATE_COLOURS,
    STATE_LABELS,
    read_clone_profile,
    read_labels as read_numbat_labels,
)
from cnamaste.sandbox.luo.platforms.visium import get_visiumhd_bin_pixels, get_visiumhd_bins, get_visiumhd_finer_labels
from cnamaste.sandbox.luo.plot.layout import get_genome_axis, plot_annotation_page
from cnamaste.sandbox.luo.pseudobulk import get_clone_pseudobulk

logger = get_logger("run_annotation_page", start_time=start_time)

DATA = Path(__file__).resolve().parents[1] / "data"
"""`python/cnamaste/sandbox/luo/benchmark/data/`: the genome axis' default files, found from this script whatever the working directory."""

OUTPUT = Path(__file__).resolve().parents[1] / "output"
"""`python/cnamaste/sandbox/luo/benchmark/output/`, gitignored: the default output directory, found from this script whatever the working
directory."""

EPITHELIAL = "Epithelial"
"""The broad cell type drawn as epithelial, a `cell_type`'s class before its `(<fine>)`."""

NAMES = {**VALIDATION_NAMES, "numbat": r"$\mathtt{Numbat}$"}
"""How the page titles a method: as the validation figure's key does (`qa.plot.NAMES`), and Numbat, whose title
also carries its bins' side (`main`)."""

LOW_CONFIDENCE = "Low confidence"
"""The broad class of a spot the cell typing could not call, left out of the page."""


def get_normal_fraction(text: str) -> tuple[int, float]:
    """Read a `--cnamaste_normal_fraction` value, `<clone>=<f>`.

    Args: `text` the value.
    Returns: the clone and its normal fraction.
    Raises: `argparse.ArgumentTypeError` where `text` is not `<int>=<f>`, f in [0, 1].
    """
    clone, sep, fraction = text.partition("=")
    try:
        found = int(clone), float(fraction)
    except ValueError:
        found = None
    if not sep or found is None or not 0.0 <= found[1] <= 1.0:
        raise argparse.ArgumentTypeError(f"expected <clone>=<f>, f in [0, 1], got {text!r}")
    return found


def get_fit_log_mu_shift(run_dir: Path) -> np.ndarray | None:
    """The fit's `new_log_mu_shift`, each clone's `log Z_c`, from `run_dir`'s one `rdrbaf_final_nstates<n>_smp.npz`.

    Args: `run_dir` the cnamaste run's directory.
    Returns: one value per clone, in ascending clone order, or None where the fit carries none.
    Raises: `FileNotFoundError` where `run_dir` holds no such file, or several.
    """
    found = sorted(run_dir.glob("rdrbaf_final_nstates*_smp.npz"))
    if len(found) != 1:
        raise FileNotFoundError(f"{len(found)} rdrbaf_final_nstates*_smp.npz in {run_dir}, not one")
    with np.load(found[0], allow_pickle=True) as fit:
        return np.asarray(fit["new_log_mu_shift"]) if "new_log_mu_shift" in fit.files else None


def parser() -> argparse.ArgumentParser:
    """The command line: every input, the drawn bins' side, CalicoST's table and the genome axis' files."""
    command = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    command.add_argument("--spaceranger_dir", type=Path, required=True)
    command.add_argument("--celltype", type=Path, required=True)
    command.add_argument("--cnamaste_labels", type=Path, required=True)
    command.add_argument("--cnamaste_seglevel", type=Path, required=True)
    command.add_argument("--cnamaste_binned", type=Path)
    command.add_argument("--cnamaste_normal_fraction", type=get_normal_fraction, action="append", default=[])
    command.add_argument("--calicost_dir", type=Path, required=True)
    command.add_argument("--numbat_dir", type=Path, required=True)
    command.add_argument("--side_um", type=int, default=16)
    command.add_argument("--calicost_ploidy", choices=CALICOST_PLOIDIES, default="unconstrained")
    command.add_argument("--region_bed", type=Path, default=DATA / "hg38.regions.bed")
    command.add_argument("--chrom_sizes", type=Path, default=DATA / "hg38.chrom.sizes")
    command.add_argument("--outdir", type=Path, default=OUTPUT)
    command.add_argument("-o", "--output", type=Path, default=Path("combined.pdf"))
    return command


def main(argv: list[str] | None = None) -> int:
    """Read every method's labels and copies, place the spots, draw the page and write it."""
    args = parser().parse_args(argv)

    celltype = load_celltype(args.celltype)
    broad = pd.Series(celltype["cell_type"].str.split(" (", regex=False).str[0].to_numpy(), index=celltype["barcode"])
    epithelial = broad[broad != LOW_CONFIDENCE] == EPITHELIAL
    logger.info(f"Cell types: {int(epithelial.sum()):_} epithelial, {int((~epithelial).sum()):_} not, "
                f"{int((broad == LOW_CONFIDENCE).sum()):_} {LOW_CONFIDENCE.lower()} left out.")  # fmt: skip
    _, calicost_run, _ = best_run(args.calicost_dir)
    logger.info(f"Reading CalicoST at its best rectangle, {calicost_run}, its {args.calicost_ploidy} copies.")
    calicost = read_calicost_estimate(calicost_run, args.calicost_ploidy)
    numbat_labels = read_numbat_labels(args.numbat_dir)
    numbat_sides = np.unique(get_visiumhd_bins(numbat_labels.index)[:, 0])
    numbat_title = NAMES["numbat"] + "".join(rf" ({side} $\mu$m)" for side in numbat_sides)
    clones = {
        numbat_title: get_visiumhd_finer_labels(numbat_labels, args.side_um),
        NAMES["calicost"]: calicost.int_clones,
        NAMES["cnamaste"]: read_labels(args.cnamaste_labels),
    }
    numbat = read_clone_profile(args.numbat_dir)
    numbat_clones = [c for c in numbat.columns if c.startswith("clone")]
    numbat[numbat_clones] = numbat[numbat_clones].apply(lambda states: states.map(STATE_LABELS))
    cnamaste_seglevel = pd.read_csv(args.cnamaste_seglevel, sep="\t")
    profiles = {
        numbat_title: numbat,
        NAMES["calicost"]: calicost.seglevel,
        NAMES["cnamaste"]: cnamaste_seglevel,
    }
    pseudobulk = None
    if args.cnamaste_binned is not None:
        logger.info(f"Reading {args.cnamaste_binned} for cnamaste's per-clone pseudobulk.")
        run_dir = args.cnamaste_binned.parent
        fractions = dict(args.cnamaste_normal_fraction)
        logger.info(f"cnamaste normal fractions given: {fractions or 'none, the fitted states expected'}.")
        with np.load(args.cnamaste_binned) as binned:
            pseudobulk = {
                NAMES["cnamaste"]: get_clone_pseudobulk(
                    binned,
                    cnamaste_seglevel,
                    read_labels(run_dir / "clone_labels.tsv"),
                    log_mu_shift=get_fit_log_mu_shift(run_dir),
                    normal_fractions=fractions,
                )
            }
    state_colours = {STATE_LABELS[s]: colour for s, colour in STATE_COLOURS.items()}

    barcodes = pd.Index(epithelial.index).append([pd.Index(c.index) for c in clones.values()]).unique()
    positions = get_spatial_positions(str(args.spaceranger_dir))
    microns = get_visiumhd_microns_per_pixel(args.spaceranger_dir)
    xy = pixel_physical(get_visiumhd_bin_pixels(barcodes, positions), microns)
    physical = pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1]}, index=barcodes)
    logger.info(f"Placed {len(physical):_} bins of {args.side_um} um from {args.spaceranger_dir}.")

    he_frame = slide_image_and_meta([args.spaceranger_dir])
    if he_frame is not None:
        he_xy = pixel_physical(he_frame[["x", "y"]].to_numpy(), microns)
        he_frame = he_frame.assign(x=he_xy[:, 0], y=he_xy[:, 1])

    genome_axis = get_genome_axis(str(args.region_bed), str(args.chrom_sizes))
    figure = plot_annotation_page(
        physical,
        epithelial,
        clones,
        profiles,
        genome_axis,
        state_colours=state_colours,
        he_frame=he_frame,
        pseudobulk=pseudobulk,
    )
    page = args.outdir / args.output
    page.parent.mkdir(parents=True, exist_ok=True)
    write_page(str(page.parent), page.name, figure)
    logger.info(f"Wrote {page}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
