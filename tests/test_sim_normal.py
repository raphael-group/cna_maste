"""`cnamaste.sim.normal` refits `sim/reference/` byte for byte from CalicoST easy (`2d4ce9a9`) and hard (`8797710b`)."""

from __future__ import annotations

from pathlib import Path

import pytest

from cnamaste.qa.sims import fixture, get_project, get_resources, grch38
from cnamaste.scripts.run_sim_normal import main

REFERENCES = grch38()


@pytest.mark.slow
@pytest.mark.very_slow
@pytest.mark.skipif(REFERENCES is None, reason="no GRCh38 resources: ref/ or $CNAMASTE_GRCH38")
def test_the_normal_fit_is_the_committed_reference(tmp_path: Path) -> None:
    """The baseline (35,291 genes) and the seven coverage laws, as committed: every KS score, scan and digit."""
    assert REFERENCES is not None and (REFERENCES / get_resources()["hgtable_file"]).exists()
    assert main([str(fixture("2d4ce9a9")), str(fixture("8797710b")), "--into", str(tmp_path)]) == 0
    for name in ("normal_baseline.txt.gz", "normal_coverage.toml"):
        assert (tmp_path / name).read_bytes() == (get_project() / "sim/reference" / name).read_bytes()
