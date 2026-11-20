"""Each plot function on CalicoST easy (`2d4ce9a9`) from its capture, written as `run_cnamaste` writes it, is the
run's PDF byte for byte: its SHA-256 is pinned in `PDF_DIGESTS`, taken from an easy run (PDFs carry no creation date,
`figure_style.PDF_METADATA`).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
import pytest

from cnamaste.plot.layout import fitted_states_real, plot_combined_page, plot_spot_umi_page
from cnamaste.plot.plot_clones_spatial import plot_clones_spatial
from cnamaste.plot.plot_copy_number_profile import plot_copy_number_profile
from cnamaste.plot.plot_genomic import plot_clones_genomic
from cnamaste.qa.capture import Capture
from cnamaste.qa.sims import get_capture_path
from cnamaste.results import write_fig, write_page

PDF_DIGESTS = {
    "spot_umis.pdf": "63201720b8c281301650c7b7b7f1dedb4f4d1b631cdeac308c0ff840f04b240a",
    "clones_spatial.pdf": "9cae4d63aa67a641c250edd3180ccd1487785e60838361e947002b0b791ed1d5",
    "copy_number_profile.pdf": "411898bd34ca09fa27c664fbc50a09709123006e4dcbc7d1474aa19227c0ccb1",
    "real_clones_genomic.pdf": "1fb3be91b39e36dfb560f351db8c5485751b53390a32c59af9a91cb13dbd1cf8",
    "real_combined.pdf": "8fc71f021561ec935483716136cab56f4fc7116caab69f6695ea9fb92dad9a89",
}
"""Each PDF's SHA-256 in an easy run's `plots/`."""


class EasyOutputs(NamedTuple):
    """What easy's last stages hand the plots: the spots, the final counts, fit and integer copies."""

    physical: Any
    spot_umis: Any
    snp_umis: Any
    sample_list: Any
    sample_ids: Any
    counts: Any
    res_combine: Any
    seglevel: pd.DataFrame
    normal_fractions: dict[str, float]


@pytest.fixture(scope="module")
def easy() -> EasyOutputs:
    """Easy's recorded stage outputs, as `run_cnamaste` holds them when it draws."""
    capture = Capture(get_capture_path("2d4ce9a9"))
    try:
        _, adata, alt, ref, _ = capture.stored("02_blocks/summarize_counts_for_blocks/in")["args"]
        sample_ids, sample_list, physical, *_ = capture.stored("04_bins/construct_multislice_lattice_adjacency/in")[
            "args"
        ]
        counts = capture.stored("07_rebin/determine_normal_baseline/out")
        res_combine = capture.stored("08_rdr/get_reindexed_fit/out")
        tables = {}
        for k in range(7):
            args = capture.stored(f"09_outputs/write_tsv_{k}/in")["args"]
            tables[args[1]] = args[2]
        decoded = capture.stored("09_outputs/decode_integer_copies/out")
    finally:
        capture.close()
    final_clone_ids = sorted(set(res_combine["new_assignment"]))
    normal_fractions = {f"clone {c}": float(f) for c, f in zip(final_clone_ids, decoded.normal_fractions, strict=True)}
    snp_umis = (alt.sum(axis=1).A1 if hasattr(alt, "toarray") else alt.sum(axis=1)) + (
        ref.sum(axis=1).A1 if hasattr(ref, "toarray") else ref.sum(axis=1)
    )
    spot_umis = adata.layers["count"].sum(axis=1)
    spot_umis = spot_umis.A1 if hasattr(spot_umis, "A1") else spot_umis
    return EasyOutputs(
        physical,
        spot_umis,
        snp_umis,
        sample_list,
        sample_ids,
        counts,
        res_combine,
        tables["cnv_diploid_seglevel.tsv"],
        normal_fractions,
    )


def get_assignment(easy: EasyOutputs) -> pd.Series:
    return pd.Series([f"clone {x}" for x in easy.res_combine["new_assignment"]])


def get_figures(easy: EasyOutputs) -> dict[str, tuple[Any, Any]]:
    """Each PDF's figure and its writer, as `run_cnamaste` calls them."""
    assignment = get_assignment(easy)
    return {
        "spot_umis.pdf": (lambda: plot_spot_umi_page(easy.spot_umis, easy.snp_umis), write_page),
        "clones_spatial.pdf": (
            lambda: plot_clones_spatial(
                easy.physical, assignment, sample_list=easy.sample_list, sample_ids=easy.sample_ids
            ),
            write_fig,
        ),
        "copy_number_profile.pdf": (lambda: plot_copy_number_profile(easy.seglevel), write_fig),
        "real_clones_genomic.pdf": (
            lambda: plot_clones_genomic(
                easy.counts, res_combine=easy.res_combine, sample_list=easy.sample_list, chrtext_shift=-0.3
            ),
            write_fig,
        ),
        "real_combined.pdf": (
            lambda: plot_combined_page(
                easy.counts,
                easy.seglevel,
                easy.res_combine,
                easy.physical,
                assignment,
                sample_list=easy.sample_list,
                sample_ids=easy.sample_ids,
                he_frame=None,
                normal_fractions=easy.normal_fractions,
                states=fitted_states_real,
            ),
            write_page,
        ),
    }


@pytest.mark.parametrize("name", list(PDF_DIGESTS))
def test_the_plot_is_the_run_s_pdf_byte_for_byte(name: str, easy: EasyOutputs, tmp_path: Path) -> None:
    """The figure, drawn from easy's capture and written as the run writes it, hashes to the run's PDF. Fails where
    a plot's input, layout, style or the PDF writer moves."""
    draw, write = get_figures(easy)[name]
    write(str(tmp_path), name, draw())
    assert hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == PDF_DIGESTS[name]


def test_the_combined_page_draws_the_integer_clones_in_every_panel(easy: EasyOutputs) -> None:
    """Easy with clone 3 merged into clone 1: the profile's rows and the tracks' pairs are the 3 integer clones, each
    named by its first clone, as the clones map's key names them. Fails where (b) or (c) draws the inferred clones."""
    figure = plot_combined_page(
        easy.counts,
        easy.seglevel,
        easy.res_combine,
        easy.physical,
        get_assignment(easy),
        sample_list=easy.sample_list,
        sample_ids=easy.sample_ids,
        he_frame=None,
        int_groups=np.array([0, 1, 2, 1]),
        normal_fractions=easy.normal_fractions,
    )
    rows = [label.get_text() for ax in figure.axes for label in ax.get_yticklabels() if "lone" in label.get_text()]
    assert rows == ["Clone 0", "Clone I", "Clone II"]
    assert sum(ax.get_ylabel() == "BAF" for ax in figure.axes) == 3
