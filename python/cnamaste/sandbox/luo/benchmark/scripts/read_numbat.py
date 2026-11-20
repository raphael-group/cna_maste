"""Readers of a Numbat run directory, as Numbat (kharchenkolab/numbat) writes it.

- `final_iteration` -- the directory's last phylogeny iteration, the `{i}` its files are post-fixed with;
- `read_labels` -- the run's clone per barcode;
- `read_clone_profile` -- each clone's CNV state per consensus segment, its clone-level pseudobulk call.

Files and columns as https://kharchenkolab.github.io/numbat/articles/descriptions.html describes them.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from cnamaste.config import start_time
from cnamaste.logger import get_logger

logger = get_logger(__name__, start_time=start_time)

STATE_COLOURS = {
    "neu": "#FFFFFF",
    "bamp": "#FA8072",
    "bdel": "#87CEEB",
    "amp": "#CD4F39",
    "loh": "#698B22",
    "del": "#4169E1",
}
"""Each CNV state's colour on Numbat's clone profile, `cnv_heatmap`'s fill (R/vis.R): `neu` white, the rest its
`cnv_colors`, salmon, skyblue, tomato3, olivedrab4 and royalblue, as R gives them in hex, in `cnv_colors`' order."""

STATE_LABELS = {"neu": "NEU", "bamp": "BAMP", "bdel": "BDEL", "amp": "AMP", "loh": "CNLoH", "del": "DEL"}
"""Each CNV state's name in Numbat's key, `cnv_labels` (R/vis.R): upper case, `loh` written CNLoH."""


def final_iteration(numbat_dir: Path) -> int:
    """The last iteration of phylogeny optimization `numbat_dir` holds: "The file names are post-fixed with the `i`th
    iteration of phylogeny optimization" (descriptions.html).

    Args: `numbat_dir` a Numbat output directory.
    Returns: the largest `i` of its `clone_post_{i}.tsv`.
    Raises: `FileNotFoundError` where it holds none.
    """
    found = [
        int(m.group(1)) for p in Path(numbat_dir).iterdir() if (m := re.fullmatch(r"clone_post_(\d+)\.tsv", p.name))
    ]
    if not found:
        raise FileNotFoundError(f"no clone_post_<i>.tsv in {numbat_dir}")
    return max(found)


def read_labels(numbat_dir: Path) -> pd.Series:
    """The run's clone per barcode at its `final_iteration` `i`: `clone_post_{i}.tsv`'s `clone_opt`, "Maximum a
    posteriori clone assignment", less one, by `cell`. Numbat's "clone 1 is always the normal cells" (results.html), so
    the normal clone is clone 0, as cnamaste's is.

    Args: `numbat_dir` a Numbat output directory.
    Returns: the clone per barcode, Numbat's barcodes.
    """
    i = final_iteration(numbat_dir)
    post = pd.read_csv(Path(numbat_dir) / f"clone_post_{i}.tsv", sep="\t")
    labels = pd.Series(post["clone_opt"].astype(int).to_numpy() - 1, index=post["cell"].astype(str).to_numpy())
    logger.info(f"Read Numbat iteration {i} labels: {len(labels):_} barcodes, clones {sorted(set(labels))}.")
    return labels


def read_clone_profile(numbat_dir: Path) -> pd.DataFrame:
    """The run's clone profile: `bulk_clones_final.tsv.gz`, "Clone-level pseudobulk profiles based on final cell lineage
    tree" (descriptions.html), each clone's `cnv_state` per consensus segment (`seg_cons`), a clone its `sample`,
    numbered as `read_labels` numbers them.

    Args: `numbat_dir` a Numbat output directory.
    Returns: `CHR`, `START`, `END` per segment, in genome order, then `clone<k>` per clone, its state.
    Raises: `ValueError` where a clone carries several states on a segment, or a state has no `STATE_COLOURS`.
    """
    columns = ["sample", "CHROM", "seg_cons", "seg_start", "seg_end", "cnv_state"]
    bulk = pd.read_csv(Path(numbat_dir) / "bulk_clones_final.tsv.gz", sep="\t", usecols=columns, low_memory=False)
    segment = ["CHROM", "seg_cons", "seg_start", "seg_end"]
    if (bulk.groupby(["sample", *segment])["cnv_state"].nunique() > 1).any():
        raise ValueError("bulk_clones_final.tsv.gz: a clone carries several states on one segment")
    unknown = set(bulk["cnv_state"]) - set(STATE_COLOURS)
    if unknown:
        raise ValueError(f"bulk_clones_final.tsv.gz: states {sorted(unknown)} not in STATE_COLOURS")

    states = bulk.drop_duplicates(["sample", *segment]).pivot(index=segment, columns="sample", values="cnv_state")
    states = states.rename(columns=lambda k: f"clone{int(k) - 1}").rename_axis(columns=None).reset_index()
    states = states.sort_values(["CHROM", "seg_start"]).drop(columns="seg_cons")
    logger.info(f"Read Numbat clone profile: {len(states)} segments, {states.shape[1] - 3} clones.")
    return states.rename(columns={"CHROM": "CHR", "seg_start": "START", "seg_end": "END"}).reset_index(drop=True)
