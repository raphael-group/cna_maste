"""`cnamaste.sim.layout`: `[layout.size]` sizes a clone in spots, grown or clipped at the array's edge, and
`[[slice]] regions` place a clone's polygon where they state. Six and five layouts of a 20 x 20 hex array, in 1 s."""

from __future__ import annotations

import numpy as np
import pytest

from cnamaste.sim.laws import draw_clone_size
from cnamaste.sim.layout import draw_clone_labels, get_array
from cnamaste.sim.manifest import Region, Slice, StatedLaw, get_simulation

ROWS = COLUMNS = 20
SIZE = StatedLaw("loguniform", {"minimum": 30.0, "maximum": 80.0})


def get_edge_spots() -> np.ndarray:
    """(spots,) True on the array's first or last row or column."""
    row, _, _ = get_array("hex", ROWS, COLUMNS)
    column = np.arange(ROWS * COLUMNS) % COLUMNS
    return np.asarray((row == 0) | (row == ROWS - 1) | (column == 0) | (column == COLUMNS - 1))


@pytest.mark.parametrize("edge", ["grow", "clip"])
def test_a_sized_clone_holds_its_drawn_spots_grown_or_clipped_at_the_edge(edge: str) -> None:
    """Sizes are drawn first, in clone order. `grow`: every clone holds its drawn count, at the edge too. `clip`: a
    clone off the edge holds its count, one meeting it fewer, sized on the array continued past the edge."""
    simulation = get_simulation("calicost_grch38")
    simulation = simulation._replace(
        array=("hex", ROWS, COLUMNS), layout=simulation.layout._replace(size=SIZE, edge=edge)
    )
    _, _, points = get_array("hex", ROWS, COLUMNS)
    on_edge = get_edge_spots()
    met = 0
    for seed in range(6):
        labels = draw_clone_labels(simulation, points, np.random.default_rng(seed))[0][0]
        rng = np.random.default_rng(seed)
        for clone, drawn in enumerate([draw_clone_size(SIZE, rng) for _ in range(3)]):
            held, meets = int(np.sum(labels == clone)), bool(np.any(on_edge[labels == clone]))
            met += meets
            assert held == drawn if edge == "grow" or not meets else held < drawn
    assert met >= 6


def test_a_stated_region_holds_its_polygon_on_its_slice() -> None:
    """A regular hexagon (jitter 0) at its stated centre and radius on its slice: its clone's spots lie within the
    circumradius, every spot within the inradius is its clone's, and it holds no spot of the other slice; the drawn
    clone_1 is placed beside it."""
    simulation = get_simulation("calicost_grch38")
    regions = (Region("clone_2", (0.3, 0.6), 0.25, 6, 0.0), Region("clone_0", (0.6, 0.4), 0.3, 6, 0.0))
    slices = (Slice(("clone_1",), (0.0, 0.0), regions[:1]), Slice((), (1.5, 0.0), regions[1:]))
    simulation = simulation._replace(array=("hex", ROWS, COLUMNS), slices=slices)
    _, _, points = get_array("hex", ROWS, COLUMNS)
    local = points - points.min(axis=0)
    extent = np.ptp(points, axis=0)
    for seed in range(5):
        labels = draw_clone_labels(simulation, points, np.random.default_rng(seed))[0]
        assert np.any(labels[0] == 1)
        for piece, region in enumerate(regions):
            clone = int(region.clone.removeprefix("clone_"))
            assert region.center is not None
            distance = np.linalg.norm(local - np.asarray(region.center) * extent, axis=1)
            radius = region.radius * extent.min()
            assert np.any(labels[piece] == clone) and np.all(distance[labels[piece] == clone] <= radius)
            assert np.all(labels[piece][distance < radius * np.cos(np.pi / 6)] == clone)
            assert not np.any(labels[1 - piece] == clone)
