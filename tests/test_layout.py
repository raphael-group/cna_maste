"""`layout.draw_spot_image` on any lattice: each spot one cell, centred on it, in its colour, no cell shared."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from cnamaste.plot.layout import draw_spot_image, get_lattice

LATTICES = {
    "square, turned 30 degrees": ((16.0, 0.0), (0.0, 16.0), 30.0),
    "rectangular 2:1": ((20.0, 0.0), (0.0, 40.0), 0.0),
    "hexagonal, Visium's 100 um": ((100.0, 0.0), (50.0, 50.0 * np.sqrt(3.0)), 0.0),
    "hexagonal, turned 15 degrees": ((10.0, 0.0), (5.0, 5.0 * np.sqrt(3.0)), 15.0),
    "oblique": ((10.0, 0.0), (4.0, 13.0), 10.0),
}


def get_spots(first: tuple[float, float], second: tuple[float, float], turn: float, slices: int) -> np.ndarray:
    """A 12 x 8 patch of the lattice `first`, `second` turned by `turn` degrees, `slices` of them placed side by side
    10 units apart, as `plot_clones_spatial` places slices."""
    i, j = np.meshgrid(np.arange(12), np.arange(8))
    patch = i.reshape(-1, 1) * np.array(first) + j.reshape(-1, 1) * np.array(second)
    t = np.radians(turn)
    patch = patch @ np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]])
    shift = np.ptp(patch[:, 0]) + 10.0
    return np.vstack([patch + [k * shift, 0.0] for k in range(slices)])


@pytest.mark.parametrize("slices", [1, 2])
@pytest.mark.parametrize("name", list(LATTICES))
def test_each_spot_is_one_cell_centred_on_it_on_any_lattice(name: str, slices: int) -> None:
    """The basis spans the planted lattice (every spot at integer coordinates in it, one spot per cell of its area);
    every spot's colour fills exactly one image cell, whose centre maps back to the spot to 1e-9. Fails where a lattice
    is misread, a cell is shared, or a cell is drawn off its spot."""
    first, second, turn = LATTICES[name]
    xy = get_spots(first, second, turn, slices)
    basis = get_lattice(xy)
    assert np.isclose(abs(np.linalg.det(basis)), abs(first[0] * second[1] - first[1] * second[0]), rtol=1e-9)
    rgba = np.column_stack([np.linspace(0.0, 1.0, len(xy)), np.zeros((len(xy), 2)), np.ones(len(xy))])
    figure, ax = plt.subplots()
    draw_spot_image(ax, xy, rgba)
    found = 0
    for image in ax.get_images():
        array = np.asarray(image.get_array())
        to_data = image.get_transform() - ax.transData
        rows, columns = np.nonzero(array[..., 3] > 0)
        centres = to_data.transform(np.column_stack([columns, rows]).astype(float))
        for centre, colour in zip(centres, array[rows, columns], strict=True):
            spot = np.flatnonzero(np.all(np.isclose(xy, centre, rtol=0.0, atol=1e-9), axis=1))
            assert len(spot) == 1 and np.array_equal(colour, rgba[spot[0]])
        found += len(rows)
    plt.close(figure)
    assert found == len(xy)


@pytest.mark.parametrize("name", list(LATTICES))
def test_measured_places_are_one_lattice_per_slice(name: str) -> None:
    """Two slices of the lattice placed side by side, each place off it by Gaussian noise of 1e-3 of the pitch, as
    Visium HD's microscope pixels are: one image per slice, every spot's colour in exactly one cell, centred on the spot
    to 6 noise widths. Fails where measured places split a slice into one image per offset (one per spot at 1e-6 of a
    cell, minutes at 100K spots), merge the two slices, or draw a cell off its spot."""
    first, second, turn = LATTICES[name]
    exact = get_spots(first, second, turn, 2)
    noise = 1e-3 * min(np.hypot(*first), np.hypot(*second))
    xy = exact + np.random.default_rng(0).normal(0.0, noise, exact.shape)
    rgba = np.column_stack([np.linspace(0.0, 1.0, len(xy)), np.zeros((len(xy), 2)), np.ones(len(xy))])
    figure, ax = plt.subplots()
    draw_spot_image(ax, xy, rgba)
    images = ax.get_images()
    drawn = []
    for image in images:
        array = np.asarray(image.get_array())
        rows, columns = np.nonzero(array[..., 3] > 0)
        centres = (image.get_transform() - ax.transData).transform(np.column_stack([columns, rows]).astype(float))
        for centre, colour in zip(centres, array[rows, columns], strict=True):
            spot = int(np.argmin(np.hypot(*(xy - centre).T)))
            assert np.hypot(*(xy[spot] - centre)) <= 6.0 * noise and np.array_equal(colour, rgba[spot])
            drawn.append(spot)
    plt.close(figure)
    assert len(images) == 2 and sorted(drawn) == list(range(len(xy)))
