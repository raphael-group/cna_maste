"""One style for every figure cnamaste writes: the face, the text size, the page's width, the palettes.

`combined.pdf` (`layout.plot_combined_page`) is where these were settled, and this module is them taken out of it so
a figure written on its own is the same figure: the stated serif face at `[tool.cnamaste.figures]`'s `font_size`,
every text at that one size, a page the text column wide, at `DPI`, and clones in `CLONE_PALETTE` with H&E classes
in `HE_PALETTE`; a spatial scatter above `RASTERIZE_ABOVE` spots drawn as one raster (`rasterize`).

Style only. Nothing here decides what is drawn -- which bins, which segments, which states -- and a plotter that
adopts it draws exactly what it drew before, at the page's size and in the page's face.
"""

from __future__ import annotations

import contextlib
import functools
import tomllib
from pathlib import Path
from typing import Any, Iterator

import matplotlib as mpl
from fontTools.ttLib import TTFont
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.text import Text

PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"

DEFAULT = {"family": "serif", "font": "STIXGeneral", "mathtext": "stix", "font_size": 8.0}
"""`[tool.cnamaste.figures]` as shipped; used where no `pyproject.toml` is found."""

PAPER_WIDTH = 8.5 - 2 * 1.01
"""Inches: a text column, US letter less 1.01 in margins, as port's `figure_style.PAPER_WIDTH`."""


def get_paper_width() -> float:
    """Return the paper width, `PAPER_WIDTH`."""
    return PAPER_WIDTH


TEXT_HEIGHT = 11.0 - 2 * 1.01 - 58.0 / 72.27
"""Inches: the text block less its running head and folio, as port's `figure_style.TEXT_HEIGHT`."""

CAPTION_ROOM = 1.0
"""Inches of the text block a page leaves for its caption."""

PAGES = {"third": 1 / 3, "half": 1 / 2, "full": 1.0}
"""The supported page heights, each a share of `TEXT_HEIGHT - CAPTION_ROOM`."""

DPI = 300


def get_dpi() -> int:
    """Return the dpi, 300."""
    return DPI


PDF_METADATA = {"CreationDate": None}
"""A PDF's metadata: no creation date, so a figure's bytes depend on what is drawn alone."""


def get_pdf_metadata() -> dict[str, None]:
    """Return a PDF's metadata, `PDF_METADATA`.

    Returns:
        A copy of the metadata `savefig` takes.
    """
    return dict(PDF_METADATA)


CLONE_PALETTE = "rocket"
"""The clones' palette, wherever clones are coloured."""


def get_clone_palette() -> str:
    """Return the clone palette, "rocket"."""
    return CLONE_PALETTE


HE_PALETTE = "mako"
"""The H&E classes' palette, distinct from the clones'."""


def get_he_palette() -> str:
    """Return the he palette, "mako"."""
    return HE_PALETTE


BOX = "#4A90E2"


def get_box() -> str:
    """Return the box, "#4A90E2"."""
    return BOX


WHISKER = "#2E5A8F"
"""cnaster's box and whisker colours: the first method's (`qa.plot`), and the face and edge of a histogram's bars."""


def get_whisker() -> str:
    """Return the whisker, "#2E5A8F"."""
    return WHISKER


COMPARED_BOX = "#E2904A"


def get_compared_box() -> str:
    """Return the compared box, "#E2904A"."""
    return COMPARED_BOX


COMPARED_WHISKER = "#8F5A2E"
"""The compared method's box and whisker colours, beside the first's: cnaster's hue turned to orange."""


def get_compared_whisker() -> str:
    """Return the compared whisker, "#8F5A2E"."""
    return COMPARED_WHISKER


BOX_ALPHA = 0.4
"""The opacity a box, or a histogram's bars, is filled at."""


def get_box_alpha() -> float:
    """Return the box alpha, 0.4."""
    return BOX_ALPHA


def style() -> dict[str, Any]:
    """Read `[tool.cnamaste.figures]` from cnamaste's `pyproject.toml`, else `DEFAULT`.

    Returns:
        The figure style mapping.
    """
    if not PYPROJECT.exists():
        return dict(DEFAULT)
    stated = tomllib.loads(PYPROJECT.read_text()).get("tool", {}).get("cnamaste", {}).get("figures")
    return dict(DEFAULT) if stated is None else dict(DEFAULT) | stated


@contextlib.contextmanager
def page_style() -> Iterator[float]:
    """Set the stated face for the block, over matplotlib's DejaVu Serif.

    Yields:
        The font size.
    """
    stated = style()
    family = stated["family"]
    with mpl.rc_context(
        {"font.family": family, f"font.{family}": [stated["font"]], "mathtext.fontset": stated["mathtext"]}
    ):
        yield float(stated["font_size"])


def page_size(page: str) -> tuple[float, float]:
    """Return a page's size in inches, width by height.

    Args:
        page: "third", "half" or "full".

    Returns:
        `(width, height)`.
    """
    return PAPER_WIDTH, PAGES[page] * (TEXT_HEIGHT - CAPTION_ROOM)


def column_size(height: float) -> tuple[float, float]:
    """Inches, a figure the text column wide and `height` tall: what a standalone figure is drawn at.

    Args:
        height: the figure's own height in inches.

    Returns:
        `(PAPER_WIDTH, height)`.

    A figure sized for a slide, 15 or 20 inches across, is a figure whose text is unreadable at the width it is
    printed. Its height is the plotter's own, so the rows it tuned keep their inches and only the width moves:
    scaling the height by the same factor as the width squeezes a row below what its own labels need, which
    matplotlib reports as `tight_layout cannot make Axes height small enough`.

    Panels side by side are the exception a caller states: there the height is what keeps their aspect, so the
    caller passes `PAPER_WIDTH / <the old ratio>` rather than the old height.
    """
    return PAPER_WIDTH, float(height)


def stagger(middles: Any, names: Any, size: float, gap: float = 1.0, lines: int = 2) -> list[int]:
    """Which line each label takes so a name does not meet its neighbour: the first line it fits on.

    `combined.pdf`'s rule (`layout.GenomicAxis.frame`), here so every genomic axis staggers the same way, and with
    the second line tracked as well as the first: the rule as it was let two names that had both been pushed down
    meet each other, which is the collision it exists to prevent. A chromosome name is about `0.6 * size` points
    per character, so a name is half that each side of its middle.

    Args:
        middles: each label's centre in points along the axis.
        names: the labels.
        size: the text size in points.
        gap: the points to leave between two names on one line.
        lines: the lines available.

    Returns:
        One line index per label, `0 .. lines - 1`.
    """
    right = [-float("inf")] * lines
    taken = []
    for middle, name in zip(middles, names):
        half = 0.3 * size * len(str(name))
        fits = [k for k in range(lines) if middle - half >= right[k] + gap]
        # NB the first line it fits on; where it fits on none, the emptiest, which is the least it can overlap.
        line = fits[0] if fits else min(range(lines), key=lambda k: right[k])
        right[line] = middle + half
        taken.append(line)
    return taken


def uniform(figure: Any, size: float) -> None:
    """Set every text on the figure to `size` points.

    Args:
        figure: the page.
        size: in points.
    """
    for text in figure.findobj(Text):
        text.set_fontsize(size)


def restyle(figure: Any) -> None:
    """Set every text on the figure to the stated size, in the stated family; `write_fig` calls it on the way out.

    Args:
        figure: the figure being written.

    The family is set per text, and generically ("serif"), rather than left to `rcParams`: a `Text` resolves its
    font when it is made, so a figure built while `rcParams` said DejaVu Serif keeps DejaVu Serif however the
    figure is saved. Generic, because the face itself comes from the `page_style` the save runs under, which is
    also what makes the mathtext match.
    """
    stated = style()
    for text in figure.findobj(Text):
        text.set_fontsize(stated["font_size"])
        text.set_fontfamily(stated["family"])
        # NB mathtext picks its fontset per text as well, and the same way: a $\log \mu$ laid out under the
        #    default set stays in DejaVu Sans beside STIX prose, which is the pair that reads as a mistake.
        text.set_math_fontfamily(stated["mathtext"])


@functools.cache
def get_advances() -> dict[str, float]:
    """The page face's advance width per character, in em; `{}` where the face cannot be read."""
    try:
        font = TTFont(findfont(FontProperties(family=style()["font"])))
        upem, metrics, cmap = font["head"].unitsPerEm, font["hmtx"], font.getBestCmap()
        return {chr(c): metrics[g][0] / upem for c, g in cmap.items() if c < 0x3000 and g in metrics.metrics}
    except Exception:  # NB a face without metrics is not a reason to fail a figure; the padding falls back to none
        return {}


def em_width(text: str) -> float:
    """The width of `text` in the page face, in em; 0.5 per character the face does not carry.

    Args:
        text: the string.

    Returns:
        The width in em.
    """
    found = get_advances()
    return sum(found.get(c, 0.5) for c in text)


def padded(text: str, width: float) -> str:
    """`text` followed by the spaces that bring it closest to `width` em in the page face.

    A proportional face has no figure space (STIXGeneral carries neither U+2007 nor U+2009), so a column is held
    by ordinary spaces: the face's space is 0.25 em against a digit's 0.5, so two spaces stand for one digit
    exactly, and a name is padded to within half a space of the column.

    Args:
        text: the string.
        width: the column's width in em.

    Returns:
        The padded string.
    """
    space = get_advances().get(" ")
    if not space:
        return text
    return text + " " * max(round((width - em_width(text)) / space), 0)


RASTERIZE_ABOVE = 30_000
"""Spots above which a spatial scatter is drawn as one raster in a vector figure. Measured with `plot_clones_spatial`
at 300 dpi: vector 15 KB at 500 spots, 314 KB at 20,000 and 1,566 KB at 102,000 (1.4 s to save); rasterized 82, 454 and
430 KB (0.3 s). Axes, ticks and key stay vector either way."""


def rasterize(n_spots: int) -> bool:
    """Whether a spatial scatter of `n_spots` is drawn as one raster."""
    return n_spots > RASTERIZE_ABOVE
