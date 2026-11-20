"""The arrays, the clones' placement on them, and the barcodes.

The slices lie in one frame, each shifted by its offset in shares of the array's extent, so overlapping slices image
a common piece of tissue. Each clone is a jittered polygon placed once in that frame, its centre uniform over the part
every slice listing it covers, so a clone on two slices sits where they overlap; with `overlap = false` a polygon is
redrawn until it claims no taken spot. A polygon is sized by its radius, stated or drawn, or to a drawn count of
spots (`[layout.size]`); `[[slice]] regions` place a clone's polygon where they state. `rectangles` cuts the frame
into a grid of blocks instead.
"""

from __future__ import annotations

import numpy as np
from matplotlib.path import Path as MplPath

from cnamaste.sim.laws import draw_clone_radius, draw_clone_size
from cnamaste.sim.manifest import Region, Simulation, StatedLaw, get_all, get_listing_slices, get_tumour_clones


def get_array(kind: str, rows: int, columns: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """An array's `array_row`, `array_col` and points in spot spacings: `hex`, Visium's packing (`array_col = 2 col +
    row % 2`, points `(array_col / 2, row sqrt(3) / 2)`); `square`, Visium HD's grid at unit spacing.

    Args:
        kind: hex or square.
        rows: the rows.
        columns: the spots per row.

    Returns:
        `(array_row, array_col, points)`.
    """
    index = np.arange(rows * columns)
    row = index // columns
    if kind == "hex":
        col = 2 * (index % columns) + row % 2
        return row, col, np.column_stack([col / 2.0, row * np.sqrt(3.0) / 2.0])
    col = index % columns
    return row, col, np.column_stack([col, row]).astype(np.float64)


def draw_polygon(
    center: np.ndarray, radius: float, vertices: int, jitter: float, rng: np.random.Generator
) -> np.ndarray:
    """A polygon: evenly spaced angles from a uniform phase, each vertex's radius `radius (1 + jitter U(-1, 1))`.

    Args:
        center: (2,) its centre.
        radius: its radius, in the frame's units.
        vertices: its vertices.
        jitter: the radii's relative spread.
        rng: the layout's generator.

    Returns:
        (vertices, 2) its vertices.
    """
    angles = 2 * np.pi * np.arange(vertices) / vertices + rng.uniform(0, 2 * np.pi)
    radii = radius * (1 + jitter * rng.uniform(-1, 1, angles.size))
    return center + np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])


def draw_sized_polygon(
    center: np.ndarray,
    scale: float,
    vertices: int,
    jitter: float,
    target: int,
    frame: np.ndarray,
    sized_on: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """A unit polygon's shape drawn once, its radius bisected (40 steps) to the smallest claiming `target` spots of
    `sized_on`; scaling a star-shaped polygon nests what it claims, so the count is monotone in the radius.

    Args:
        center: (2,) its centre.
        scale: the frame's unit, the array's shorter extent.
        vertices: its vertices.
        jitter: the radii's relative spread.
        target: the spots it claims.
        frame: (spots, 2) the slices' points.
        sized_on: the points it is sized on, `frame` or `get_continued_frame`'s.
        rng: the layout's generator.

    Returns:
        `(vertices, claimed)`, claimed of `frame`.
    """
    unit = draw_polygon(np.zeros(2), 1.0, vertices, jitter, rng)
    low, high = 0.0, float(np.ptp(sized_on, axis=0).max()) / scale
    for _ in range(40):
        middle = 0.5 * (low + high)
        if int(MplPath(center + unit * middle * scale).contains_points(sized_on).sum()) >= target:
            high = middle
        else:
            low = middle
    polygon = center + unit * high * scale
    return polygon, np.asarray(MplPath(polygon).contains_points(frame))


def get_continued_frame(points: np.ndarray) -> np.ndarray:
    """An array's points continued one period past each edge, 3 x 3 copies, so a clone sized on it is clipped by the
    edge rather than grown (`[layout.size] edge = "clip"`).

    Args:
        points: (spots, 2) one slice's points in the frame.

    Returns:
        (9 spots, 2) points.
    """
    ys = np.unique(points[:, 1])
    row = np.sort(points[points[:, 1] == ys[0], 0])
    period = np.array([row.size * np.diff(row).min(), ys.size * np.diff(ys).min()])
    return np.concatenate([points + np.array([i, j]) * period for i in (-1, 0, 1) for j in (-1, 0, 1)])


def draw_rectangle_labels(
    normal_share: float, frame: np.ndarray, listed: list[str], tumour: tuple[str, ...], rng: np.random.Generator
) -> np.ndarray:
    """The frame cut into a `p x p` grid, `p = ceil(sqrt(n + 1))` for normal and `n` clones, block `b` holding group
    `b % (n + 1)`, group 0 normal, whose first row and column take `sqrt(normal_share)` of each side; the clones
    assigned to groups by a permutation (CalicoST's `rectangle_initialize_initial_clone` layout).

    Args:
        normal_share: the normal block's share of the frame's area.
        frame: (spots, 2) the slices' points.
        listed: the clones placed.
        tumour: every tumour clone.
        rng: the layout's generator.

    Returns:
        (spots,) `-1` for normal, `k` for `clone_k`.
    """
    groups = len(listed) + 1
    p = int(np.ceil(np.sqrt(groups)))
    lower, upper = frame.min(axis=0), frame.max(axis=0)
    first = np.sqrt(normal_share) if groups > 1 else 1.0 / p
    rest = np.linspace(first, 1.0, p)[1:-1] if p > 2 else np.array([])
    edges = np.concatenate(([first], rest))[: p - 1] if p > 1 else np.array([])
    x_cuts, y_cuts = (lower[axis] + edges * (upper[axis] - lower[axis]) for axis in (0, 1))
    block = np.searchsorted(y_cuts, frame[:, 1], side="right") * p + np.searchsorted(x_cuts, frame[:, 0], side="right")
    order = [listed[k] for k in rng.permutation(len(listed))]
    labels = np.full(frame.shape[0], -1, dtype=np.int64)
    for k, clone in enumerate(order, start=1):
        labels[block % groups == k] = tumour.index(clone)
    return labels


def draw_clone_labels(
    simulation: Simulation, points: np.ndarray, rng: np.random.Generator
) -> tuple[list[np.ndarray], tuple[str, ...]]:
    """Each spot's clone per slice, `-1` for normal, `k` for `clone_k`; and the clones placed.

    Stated regions are placed first, on their slice; the other clones in turn, each centred uniformly where every
    slice naming it covers (anywhere a slice covers, for `"all"`). Sizes, then drawn radii, are drawn per clone in
    name order before any placement. A clone no placement clears refuses the draw, stops it, or is dropped.

    Args:
        simulation: the simulation, its clone count resolved.
        points: (spots, 2) one slice's array.
        rng: the layout's generator.

    Returns:
        One (spots,) array per slice; the tumour clones placed, in order.

    Raises:
        ValueError: where a clone's slices do not overlap, a stated region overlaps another without `overlap`, or
            `unplaced = "refuse"` and no placement in `max_placements` clears the others.
    """
    layout, tumour = simulation.layout, get_tumour_clones(simulation)
    lower = points.min(axis=0)
    extent = points.max(axis=0) - lower
    scale = float(extent.min())
    origins = [lower + np.asarray(piece.offset) * extent for piece in simulation.slices]
    frames = [points - lower + origin for origin in origins]
    frame = np.concatenate(frames)
    stated = {r.clone: (r, o) for piece, o in zip(simulation.slices, origins, strict=True) for r in piece.regions}
    listed = [c for c in tumour if any(get_listing_slices(simulation, c))]
    listed += [c for c in stated if c not in listed]
    if layout.shape == "rectangles":
        assert layout.normal_share is not None
        labels = draw_rectangle_labels(layout.normal_share, frame, listed, tumour, rng)
        return list(np.split(labels, len(origins))), tumour
    targets = {} if layout.size is None else {c: draw_clone_size(layout.size, rng) for c in sorted(listed)}
    radius = layout.radius
    radii = (
        {c: draw_clone_radius(radius, rng) for c in sorted(listed)}
        if isinstance(radius, StatedLaw)
        else dict.fromkeys(listed, float(radius))
    )
    sized_on = np.concatenate([get_continued_frame(f) for f in frames]) if layout.edge == "clip" else frame
    labels = np.full(frame.shape[0], -1, dtype=np.int64)
    order = sorted(listed, key=lambda c: c not in stated)
    unplaced: set[str] = set()
    for index, clone in enumerate(order):
        if clone in stated:
            region, origin = stated[clone]
            center = None if region.center is None else origin + np.asarray(region.center) * extent
            low, high = origin, origin + extent
        else:
            region = Region(clone, None, radii[clone], layout.vertices, layout.jitter)
            flags = get_listing_slices(simulation, clone)
            covering = [o for on, o in zip(flags, origins, strict=True) if on]
            every = all(piece.clones == (get_all(),) for on, piece in zip(flags, simulation.slices, strict=True) if on)
            low, high = (
                (np.min(covering, axis=0), np.max(covering, axis=0) + extent)
                if every
                else (np.max(covering, axis=0), np.min(covering, axis=0) + extent)
            )
            center = None
            if np.any(low > high):
                raise ValueError(f"{clone} is listed on slices whose arrays do not overlap")
        for _ in range(layout.max_placements):
            middle = rng.uniform(low, high) if center is None else center
            if center is None and layout.size is not None:
                polygon, claimed = draw_sized_polygon(
                    middle, scale, region.vertices, region.jitter, targets[clone], frame, sized_on, rng
                )
            else:
                polygon = draw_polygon(middle, region.radius * scale, region.vertices, region.jitter, rng)
                claimed = np.asarray(MplPath(polygon).contains_points(frame))
            if layout.overlap or not np.any(labels[claimed] >= 0):
                break
            if center is not None:
                raise ValueError(f"the region of {clone} overlaps another and [layout] overlap = false")
        else:
            if layout.unplaced == "refuse":
                raise ValueError(f"no placement of {clone} in {layout.max_placements} clears the others")
            unplaced |= set(order[index:]) if layout.unplaced == "stop" else {clone}
            if layout.unplaced == "stop":
                break
            continue
        labels[claimed] = tumour.index(clone)
    return list(np.split(labels, len(origins))), tuple(c for c in tumour if c not in unplaced)


def draw_barcodes(n_spots: int, length: int, suffix: str, rng: np.random.Generator) -> np.ndarray:
    """`n_spots` distinct `length`-mers over `ACGT`, then `suffix`: one whitelist, shared by every slice.

    Args:
        n_spots: the spots per slice.
        length: the k-mer length.
        suffix: e.g. `-1`.
        rng: the barcodes' generator.

    Returns:
        The barcodes.
    """
    drawn: dict[str, None] = {}
    while len(drawn) < n_spots:
        for row in rng.integers(0, 4, (n_spots, length)):
            drawn.setdefault("".join("ACGT"[i] for i in row) + suffix)
            if len(drawn) == n_spots:
                break
    return np.array(list(drawn))


def is_number(text: str) -> bool:
    """Whether `text` parses as a float.

    Args:
        text: the text.

    Returns:
        Whether it does.
    """
    try:
        float(text)
    except ValueError:
        return False
    return True


def draw_sample_ids(n_slices: int, n_bytes: int, rng: np.random.Generator) -> list[str]:
    """Distinct hexadecimal slice ids of `2 n_bytes` digits, none of them a number (a sample sheet would read
    `823468e7` as a float).

    Args:
        n_slices: the slices.
        n_bytes: the bytes per id.
        rng: the barcodes' generator.

    Returns:
        The ids.
    """
    ids: list[str] = []
    while len(ids) < n_slices:
        candidate = rng.bytes(n_bytes).hex()
        if candidate not in ids and not is_number(candidate):
            ids.append(candidate)
    return ids
