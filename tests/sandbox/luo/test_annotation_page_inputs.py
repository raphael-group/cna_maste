"""The inputs `sandbox/luo/benchmark/scripts/run_annotation_page.py` converts: Visium HD bins given to their finer
bins and placed on the image (`sandbox.luo.platforms.visium`), a Numbat run's clones and clone profile
(`sandbox/luo/benchmark/scripts/read_numbat.py`), spots drawn as lattice cells (`layout.draw_spot_image`), and segments
on cnplot's arms (`layout.get_cnplot_segments`, `layout.get_seg_ucn`).

Referees: a 32 um bin's 16 um bins are the 2 x 2 block at twice its indices, their centres a quarter of its side
either way of its own; the grid ends at 3,350 2 um squares, so a bin on its far edge holds one; Numbat's documented
outputs (https://kharchenkolab.github.io/numbat/articles/descriptions.html), its clone 1 the normal cells, and
its clone-level pseudobulk profiles.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from cnamaste.sandbox.luo.benchmark.scripts.read_numbat import read_clone_profile, read_labels
from cnamaste.sandbox.luo.platforms.visium import get_visiumhd_bin_pixels, get_visiumhd_finer_labels


def test_finer_bins_tile_their_bin_and_end_with_the_grid() -> None:
    """Three 32 um bins to 16 um: each label on its own block, each 16 um centre a quarter side from its bin's, under a
    turned, scaled and shifted image as Space Ranger fits one; the bin at (209, 209), past which the 16 um grid stops
    at index 418, keeps one."""
    labels = pd.Series([1, 2, 0], index=["s_032um_00000_00000-1", "s_032um_00005_00007-1", "s_032um_00209_00209-1"])
    fine = get_visiumhd_finer_labels(labels, 16)

    assert fine.groupby(fine).size().to_dict() == {0: 1, 1: 4, 2: 4}
    assert set(fine[fine == 2].index) == {f"s_016um_{r:05d}_{c:05d}-1" for r in (10, 11) for c in (14, 15)}

    turn = np.radians(0.4)
    image = 3.65 * np.array([[np.cos(turn), -np.sin(turn)], [np.sin(turn), np.cos(turn)]])
    grid = [f"s_032um_{r:05d}_{c:05d}-1" for r in range(0, 200, 37) for c in range(0, 200, 41)]
    um = np.array([[(int(b[14:19]) + 0.5) * 32, (int(b[8:13]) + 0.5) * 32] for b in grid])
    xy = um @ image.T + [1_200.0, 5_600.0]
    positions = pd.DataFrame({"barcode": grid, "x": xy[:, 0], "y": xy[:, 1]})

    coarse = get_visiumhd_bin_pixels(labels.index[:2], positions)
    within = get_visiumhd_bin_pixels(fine[fine != 0].index, positions).reshape(2, 4, 2)
    assert np.allclose(within.mean(axis=1), coarse)
    assert np.allclose(np.abs(np.linalg.solve(image, (within - coarse[:, None]).reshape(-1, 2).T)), 8.0)


def test_numbat_clone_profile_is_each_clones_pseudobulk_state(tmp_path: Path) -> None:
    """A run of two iterations: labels read at the last, clones shifted to cnamaste's numbering, the normal clone 0; the
    profile each clone's `cnv_state` per consensus segment of `bulk_clones_final.tsv.gz`, one row per SNP in the file,
    one per segment out, in genome order."""
    pd.DataFrame({"cell": ["a"], "clone_opt": [1]}).to_csv(tmp_path / "clone_post_1.tsv", sep="\t")
    pd.DataFrame({"cell": ["a", "b", "c", "d"], "clone_opt": [1, 2, 3, 3]}).to_csv(
        tmp_path / "clone_post_2.tsv", sep="\t", index=False
    )
    states = {1: ["neu", "neu"], 2: ["neu", "amp"], 3: ["loh", "amp"]}
    pd.DataFrame(
        [
            {"sample": k, "CHROM": chrom, "seg_cons": seg, "seg_start": 100, "seg_end": 200, "cnv_state": s}
            for k, row in states.items()
            for (chrom, seg), s in zip(((2, "2a"), (1, "1a")), row)
            for _ in range(3)
        ]
    ).to_csv(tmp_path / "bulk_clones_final.tsv.gz", sep="\t", index=False)

    assert read_labels(tmp_path).to_dict() == {"a": 0, "b": 1, "c": 2, "d": 2}
    profile = read_clone_profile(tmp_path)
    assert profile.to_dict("list") == {
        "CHR": [1, 2], "START": [100, 100], "END": [200, 200],
        "clone0": ["neu", "neu"], "clone1": ["amp", "neu"], "clone2": ["amp", "loh"],
    }  # fmt: skip


def test_segments_are_cut_at_the_arms(tmp_path: Path) -> None:
    """A segment through chr1's centromere, on two arm regions, is drawn either side of it and nowhere in it; cnplot's
    profile names clone 0 `normal` where it is (1, 1) throughout."""
    pytest.importorskip("cnplot", reason="cnplot is in the luo dependency group: uv sync --group luo")
    from cnamaste.sandbox.luo.plot.layout import get_cnplot_segments, get_genome_axis, get_seg_ucn

    (tmp_path / "regions.bed").write_text("chr1\t0\t100\tchr1p\nchr1\t150\t300\tchr1q\n")
    (tmp_path / "sizes").write_text("chr1\t300\n")
    axis = get_genome_axis(str(tmp_path / "regions.bed"), str(tmp_path / "sizes"))
    seglevel = pd.DataFrame(
        {"CHR": [1, 1], "START": [20, 200], "END": [180, 280], "clone0 A": [1, 1], "clone0 B": [1, 1],
         "clone1 A": [2, 1], "clone1 B": [1, 0]}
    )  # fmt: skip

    ucn = get_seg_ucn(get_cnplot_segments(seglevel, axis))

    assert ucn[["START", "END"]].to_numpy().tolist() == [[20, 100], [150, 180], [200, 280]]
    assert list(ucn.columns) == ["#CHR", "START", "END", "cn_normal", "cn_Clone I"]
    assert ucn["cn_Clone I"].tolist() == ["2|1", "2|1", "1|0"]


def test_spots_are_one_cell_each_whatever_their_order() -> None:
    """A 3 x 4 lattice turned 0.4 degrees, 16 um pitch, two spots missing: each spot its own cell in its own colour, the
    missing cells clear, and the image the same for the spots in reverse order."""
    pytest.importorskip("cnplot", reason="cnplot is in the luo dependency group: uv sync --group luo")
    from cnamaste.sandbox.luo.plot.layout import draw_spot_image

    turn = np.radians(0.4)
    rotation = np.array([[np.cos(turn), -np.sin(turn)], [np.sin(turn), np.cos(turn)]])
    grid = np.array([(c, r) for r in range(3) for c in range(4) if (c, r) not in ((1, 1), (3, 2))])
    xy = (16.0 * grid) @ rotation.T + [500.0, -2_000.0]
    rgba = np.random.default_rng(0).uniform(size=(len(grid), 4))

    images = []
    for order in (np.arange(len(grid)), np.arange(len(grid))[::-1]):
        figure, ax = plt.subplots()
        draw_spot_image(ax, xy[order], rgba[order])
        images.append(np.asarray(ax.images[0].get_array()))
        plt.close(figure)

    assert images[0].shape == (3, 4, 4)
    assert np.array_equal(images[0], images[1])
    assert np.array_equal(images[0][grid[:, 1], grid[:, 0]], rgba)
    assert np.all(images[0][[1, 2], [1, 3], 3] == 0.0)
