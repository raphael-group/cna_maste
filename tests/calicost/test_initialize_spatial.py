"""`initialize.spatial`'s `rectangle` starts against CalicoST `c1abcae`'s `utils_hmrf.rectangle_initialize_initial_clone`
and `phasing.perform_partition`, bitwise.

The referee is CalicoST's own output on the same input, computed once and recorded (`calicost_cache`) as the sha256 of
its labels, each spot's clone the position of the index list that holds it. Inputs: the pixel coordinates of CalicoST easy's
(`2d4ce9a9`) 3,000 spots, and of dev_tree_easy's (`e4bbe03f`) two slices, the plan's one multi-slice referee.
CalicoST's `phasing` imports `turtle`, which needs `tkinter`, for a name it never calls: a stub stands in for it at
import. The file runs in under 1 s.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd
import pytest
from calicost.utils_hmrf import rectangle_initialize_initial_clone

import cnamaste.calicost  # noqa: F401  (imports CalicoST with the environment held, as cnamaste does)
from calicost_cache import Referee, get_digest
from cnamaste.initialize.spatial import SpotPositions, get_block_labelling, get_phasing_labelling
from cnamaste.qa.sims import fixture

# NB `from turtle import reset` at the top of CalicoST's `phasing`, never called; `turtle` needs `tkinter`.
with mock.patch.dict(sys.modules, {"turtle": types.SimpleNamespace(reset=None)}):
    from calicost.phasing import perform_partition


def get_pixel_coords(spaceranger_dir: Path) -> np.ndarray:
    """The in-tissue spots' `(pxl_row_in_fullres, pxl_col_in_fullres)`, from `spatial/tissue_positions_list.csv.gz`."""
    positions = pd.read_csv(spaceranger_dir / "spatial/tissue_positions_list.csv.gz", header=None)
    return positions.loc[positions[1] == 1, [4, 5]].to_numpy(dtype=float)


def get_labels(clone_index: list[np.ndarray], n_spots: int) -> np.ndarray:
    """Each spot's clone from CalicoST's per-clone spot indices, -1 where no clone holds it."""
    labels = np.full(n_spots, -1)
    for clone, index in enumerate(clone_index):
        labels[index] = clone
    return labels


@pytest.mark.parametrize("random_state", [0, 1, 2])
@pytest.mark.parametrize("n_clones", [2, 3, 4])
def test_get_block_labelling_is_rectangle_initialize_initial_clone(
    n_clones: int, random_state: int, referee: Referee
) -> None:
    """Easy's spots, the RDR split's `rectangle` start: `get_block_labelling` is CalicoST's, bitwise, for each clone
    count and seed. Fails where the Dirichlet edges, the 5-95% range, the block-to-clone draw, its redraw under the
    20% floor or the seeded stream drift."""
    coords = get_pixel_coords(fixture("2d4ce9a9"))

    def call() -> np.ndarray:
        return get_labels(rectangle_initialize_initial_clone(coords, n_clones, random_state), len(coords))

    name = f"initialize.rectangle[{n_clones}-{random_state}]"
    assert (get_digest(get_block_labelling(coords, n_clones, random_state)),) == referee(
        name, (coords, n_clones, random_state), call
    )


def test_get_phasing_labelling_is_perform_partition_on_two_slices(referee: Referee) -> None:
    """Dev_tree_easy's two slices at `npart_phasing: 3`, `clone_init_type: rectangle`: each slice's 3 x 3 grid, the
    second slice's 9 labels after the first's, is CalicoST's `perform_partition`, bitwise. Fails where a slice's
    grid, its cell order or the per-slice offset drifts."""
    slices = [get_pixel_coords(fixture("e4bbe03f") / name) for name in ("8a2f1e43", "ec4b49a0")]
    coords, sample_ids = np.vstack(slices), np.repeat([0, 1], [len(s) for s in slices])

    def call() -> np.ndarray:
        return get_labels(perform_partition(coords, sample_ids, 3, 3, None, 0.5), len(coords))

    config = SimpleNamespace(phasing=SimpleNamespace(npart_phasing=3, clone_init_type="rectangle"))
    ours = get_phasing_labelling(None, SpotPositions(coords, None, sample_ids), config, None)
    assert len(np.unique(ours)) == 18
    assert (get_digest(ours),) == referee("initialize.perform_partition", (coords, sample_ids), call)
