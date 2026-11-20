"""`sim.write`'s directories: one level at a time within an existing directory; a written sample or reference file
never replaced."""

from __future__ import annotations

from pathlib import Path

import pytest

from cnamaste.qa.sims import get_fixture_hash, get_project
from cnamaste.sim.tile import write_tiled_sample
from cnamaste.sim.write import get_made_dir, write_reference


def test_a_directory_is_made_only_directly_within_an_existing_one(tmp_path: Path) -> None:
    """A missing parent, a missing intermediate, a name outside the parent and an existing name with `exist_ok`
    False raise; otherwise exactly one directory is made within the parent."""
    get_made_dir(tmp_path, "out", exist_ok=False)
    for out, name in (
        (tmp_path / "missing", "a"),
        (tmp_path / "out", "missing/a"),
        (tmp_path / "out", ".."),
        (tmp_path / "out", str(tmp_path / "a")),
    ):
        with pytest.raises(RuntimeError):
            get_made_dir(out, name, exist_ok=True)
    assert get_made_dir(tmp_path / "out", "a", exist_ok=False) == tmp_path / "out" / "a"
    with pytest.raises(RuntimeError, match="exists"):
        get_made_dir(tmp_path / "out", "a", exist_ok=False)
    assert get_made_dir(tmp_path / "out", "a", exist_ok=True) == tmp_path / "out" / "a"
    assert sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*")) == ["out", "out/a"]


def test_a_written_sample_is_neither_replaced_nor_written_into_a_missing_directory(tmp_path: Path) -> None:
    """Stress x1 (`6bb8b91a`) is written once; a second write raises naming it and leaves it whole; a write into a
    missing directory raises and makes nothing."""
    source = get_project() / "sim" / "easy_2d4ce9a9"
    written = write_tiled_sample(source, 1, 0, tmp_path, "easy_x1")
    assert written.name == f"easy_x1_{get_fixture_hash(written)}" == "easy_x1_6bb8b91a"
    with pytest.raises(RuntimeError, match=str(written)):
        write_tiled_sample(source, 1, 0, tmp_path, "easy_x1")
    assert get_fixture_hash(written) == "6bb8b91a"
    with pytest.raises(RuntimeError, match="no output directory"):
        write_tiled_sample(source, 1, 0, tmp_path / "missing", "easy_x1")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["easy_x1_6bb8b91a", "easy_x1_staging"]


def test_the_reference_files_are_written_once_into_an_existing_directory(tmp_path: Path) -> None:
    """Both files are written once; a second write raises and leaves them as they were; a missing directory raises."""
    written = write_reference("baseline\n", "coverage\n", tmp_path)
    before = [path.read_bytes() for path in written]
    with pytest.raises(RuntimeError, match="exists"):
        write_reference("other\n", "other\n", tmp_path)
    assert [path.read_bytes() for path in written] == before
    with pytest.raises(RuntimeError, match="not an existing directory"):
        write_reference("baseline\n", "coverage\n", tmp_path / "missing")
    assert sorted(path.name for path in tmp_path.iterdir()) == ["normal_baseline.txt.gz", "normal_coverage.toml"]
