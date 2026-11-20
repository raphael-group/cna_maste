"""`run_cnamaste_decode.substituted`: before the decode, every figure is skipped, a PNG as a PDF; tables are written.
`run_cnamaste_plot.plot` takes figure keys alone."""

from __future__ import annotations

from pathlib import Path

import pytest

from cnamaste import inference
from cnamaste.results import OUTPUT_FILES, write_fig, write_page
from cnamaste.scripts.run_cnamaste_decode import substituted
from cnamaste.scripts.run_cnamaste_plot import plot


def test_every_figure_before_the_decode_is_skipped_and_every_table_written(tmp_path: Path) -> None:
    """Outside the decode's `finalize`, each `OUTPUT_FILES` figure's writer, `he_clones.png` with a slide's run as the
    PDFs, takes the `None` its gated `plot_` returns and writes nothing; every other writer is the run's own. Fails where
    a figure writer is left ungated, as `he_clones.png` was (`AttributeError: 'NoneType' object has no attribute
    'savefig'` on Visium HD N3KXRGQ), or a table's writer is gated."""
    assert "he_clones.png" in OUTPUT_FILES
    with substituted(tmp_path):
        for name, output_file in OUTPUT_FILES.items():
            written = inference.get_output_file_meta(name).writer
            if output_file.writer in (write_fig, write_page):
                assert written(str(tmp_path), name, None) is None
            else:
                assert written is output_file.writer
    assert not any(tmp_path.iterdir())


def test_a_plot_takes_figure_keys_alone_and_refuses_before_finding_a_run(tmp_path: Path) -> None:
    """`run_cnamaste_plot.plot` refuses an `ab_decode` or `hmrf` override, naming `run_cnamaste_decode` for the first,
    before it reads the configuration or makes a directory. Fails where a key that changes a table could reach the
    figures, which would then not be the run's."""
    for over_ride in ("ab_decode.t=0.999", "hmrf.spatial_weight=0.8"):
        with pytest.raises(SystemExit, match=r"-o takes one of \['run.plot_field_argmax'\]") as refused:
            plot(tmp_path / "absent.yaml", over_rides=[over_ride])
        assert "ab_decode.<key> is run_cnamaste_decode's" in str(refused.value)
    assert not any(tmp_path.iterdir())
