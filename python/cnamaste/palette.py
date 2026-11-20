from typing import Any

import matplotlib.colors as mcolors
import seaborn as sns


def get_ordered_acn(mode: str = "joint") -> tuple[Any, ...]:
    """Return the ordered allele copy-number states as an immutable tuple.

    Args:
        mode: "joint" for (A, B) tuples, "single" for single integers.

    Returns:
        The tuple of states.
    """
    if mode == "single":
        return (0, 1, 2, 3, 4, 5, 6, "7+")

    # NB fallback to joint as default
    return (
        (0, 0),
        (1, 0),
        (1, 1),
        (2, 0),
        (2, 1),
        (3, 0),
        (2, 2),
        (3, 1),
        (4, 0),
        (3, 2),
        (4, 1),
        (5, 0),
        (3, 3),
        (4, 2),
        (5, 1),
        (6, 0),
    )


COPY_STATES = {
    (1, 1): "#EEEEEE",
    (2, 2): "#FC9272",
    (3, 3): "#CB181D",
    (1, 0): "#66D9E8",
    (2, 0): "#22B5D2",
    (3, 0): "#1A86C2",
    (4, 0): "#0D56A0",
    (5, 0): "#08306B",
    (6, 0): "#041A3A",
    (2, 1): "#A1D99B",
    (4, 2): "#31A354",
    (3, 1): "#FEC44F",
    (4, 1): "#6A51A3",
    (5, 1): "#AE017E",
    (3, 2): "#F16913",
    (0, 0): "#000000",
}
"""The `(A, B)` states with `A >= B`, in the profile key's order: hue by minor BAF class -- balanced reds from a
light-gray (1, 1), LOH from cyan to navy, 1/3 greens, 1/4 yellow, 1/5 purple, 1/6 magenta, 2/5 orange -- and darker for
more copies. (6, 0) and (0, 0) lie outside the decoder's lattice and are coloured for a fit's states alone."""


def get_copy_states() -> dict[tuple[int, ...], str]:
    """Return the copy states, `COPY_STATES`."""
    return COPY_STATES


FLIP_OPACITY = 0.5
"""A `B > A` state's opacity, of its `(A, B)` partner's colour: the same hue, so phase reads as a lighter fill."""


def get_flip_opacity() -> float:
    """Return the flip opacity, 0.5."""
    return FLIP_OPACITY


TRACK_NORMAL = "steelblue"
"""(1, 1)'s colour on the tracks, where `COPY_STATES`' light gray would leave a normal bin's dots unseen on white."""

TRACKS = {**COPY_STATES, (1, 1): TRACK_NORMAL}
"""The `tracks` palette: `COPY_STATES`, (1, 1) as `TRACK_NORMAL`."""


def state_colour(a: int, b: int, normal: str = COPY_STATES[(1, 1)]) -> tuple[float, float, float, float]:
    """`(a, b)`'s RGBA: `COPY_STATES`' `(max, min)`, at `FLIP_OPACITY` where `b > a`; `normal` for (1, 1).

    Args:
        a: integer copies.
        b: integer copies.
        normal: (1, 1)'s colour.

    Returns:
        The RGBA tuple, black where the pair is outside `COPY_STATES`.
    """
    key = (max(a, b), min(a, b))
    colour = normal if key == (1, 1) else COPY_STATES.get(key, "#000000")
    return mcolors.to_rgba(colour, FLIP_OPACITY if b > a else 1.0)


def get_full_palette(palette_name: str = "chisel_joint") -> tuple[dict[Any, str], tuple[Any, ...]]:
    """Map copy states to colours under a named palette.

    "chisel_joint" maps (A, B) tuples to colours; "chisel_independent" maps integer copies, i.e. A.

    Args:
        palette_name: the palette.

    Returns:
        The state-to-colour dict and the ordered states.
    """
    palette: dict[Any, str] = {}

    # NB default joint encoding.
    if palette_name in ("chisel", "chisel_joint"):
        ordered_acn = get_ordered_acn(mode="joint")

        palette.update({(0, 0): "lightblue"})  # TODO never used??
        palette.update({(1, 0): "darkblue"})  # NB swapped with (0,0)
        palette.update({(1, 1): "steelblue"})  # NB chisel is lightgray
        palette.update({(2, 0): "dimgray"})
        palette.update({(2, 1): "gold"})  # lightgoldenrodyellow
        palette.update({(3, 0): "goldenrod"})  # gold
        palette.update({(2, 2): "navajowhite"})
        palette.update({(3, 1): "orange"})
        palette.update({(4, 0): "darkorange"})
        palette.update({(3, 2): "salmon"})
        palette.update({(4, 1): "red"})
        palette.update({(5, 0): "darkred"})
        palette.update({(3, 3): "plum"})
        palette.update({(4, 2): "orchid"})
        palette.update({(5, 1): "purple"})
        palette.update({(6, 0): "indigo"})

    elif palette_name == "tracks":
        # NB combined.pdf's track dots and key: `COPY_STATES`, (1, 1) as `TRACK_NORMAL`.
        ordered_acn = get_ordered_acn(mode="joint")
        palette.update(TRACKS)

    elif palette_name == "chisel_single":
        ordered_acn = get_ordered_acn(mode="single")

        palette.update(
            {
                0: "white",  # Negative space for complete loss
                1: "steelblue",
                2: "khaki",
                3: "#feb24c",  # Orange
                4: "#fd8d3c",  # Dark orange
                5: "#f03b20",  # Red
                6: "#bd0026",  # Dark red
                "7+": "#660013",  # Deep burgundy
            }
        )

        # NB default to color for 1
        palette["default"] = palette[1]

    else:
        ordered_acn = get_ordered_acn(mode="joint")
        colors = sns.color_palette(palette_name, len(ordered_acn)).as_hex()
        palette = dict(zip(ordered_acn, colors))

    if "default" not in palette:
        palette["default"] = "steelblue"

    return palette, ordered_acn
