"""`recomb`'s three functions against the CalicoST code they stand in for, CalicoST `c1abcae` as pinned, bitwise.

The referee is CalicoST's own output on the same input, computed once and recorded (`calicost_cache`), never a copy
of its code:

- `recomb.compute_numbat_phase_switch_prob` against `utils_phase_switch.compute_phase_switch_probability_position`;
- `recomb.assign_centiMorgans` against `utils_phase_switch.get_position_cM_table`;
- `recomb.get_sitewise_transmat` against the two composed as `utils_IO.summarize_counts_for_blocks` and `_for_bins`
  compose them (`calicost_transmat`).

Inputs: CalicoST easy's (`2d4ce9a9`) 13,482 SNP positions (`unique_snp_ids.npy`, committed; hard `8797710b` holds the
same 13,482) on CalicoST's shipped GRCh38 genetic map, and synthetic edge cases. No staged capture is read; they skip
where the GRCh38 resources are not installed.

Tolerance: none. Every comparison is bitwise on float64, as the sha256 of CalicoST's output: both sides run the same
arithmetic in the same order.

Departures, each a strict xfail (`test_cnamaste_is_calicost_where_it_departs`) and a choice rather than a defect:
- the floor: CalicoST fills an entry with no switch to compute (a contig's last SNP, a NaN neighbour, the last SNP)
  with 1e-20 before flooring at `min_prob`; cnamaste fills it with `min_prob`. Equal at `min_prob >= 1e-20`; below it,
  22 of easy's 13,482 entries differ (21 contig changes and the last SNP). `get_sitewise_transmat` floors at the
  configuration's `phasing.min_prob` where CalicoST's composition takes the default, 1e-20: equal at the shipped 1e-20.
- the order: CalicoST sorts the caller's list in place and returns cM in sorted order; cnamaste refuses pairs out of
  logical order (T- #20) and leaves the list as given, and refuses segment keys out of genomic order. Equal on ordered
  input, which is all a run passes.
- contig names: CalicoST also reads `chr`-prefixed contigs; cnamaste's loader yields integer contigs and refuses others.
"""

from __future__ import annotations

import functools
import gzip
import io
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import pandas as pd
import pytest
from calicost import utils_phase_switch
from calicost.utils_phase_switch import compute_phase_switch_probability_position, get_position_cM_table

import cnamaste.calicost  # noqa: F401  (imports CalicoST with the environment held, as cnamaste does)
from calicost_cache import Referee, departs, get_digest
from cnamaste.config import YAMLConfig, set_global_config
from cnamaste.qa.sims import fixture, get_resources, grch38
from cnamaste.recomb import assign_centiMorgans, compute_numbat_phase_switch_prob, get_sitewise_transmat
from cnamaste.reference import get_reference_recomb_rates

CALICOST_FLOOR = 1e-20
"""What CalicoST fills an entry with before flooring: `np.ones(n) * 1e-20`, whatever `min_prob`."""


def calicost_transmat(df: pd.DataFrame, key: str, geneticmap_file: str, nu: float, shift: float) -> np.ndarray:
    first = df.groupby(key).agg({"CHR": "first", "START": "first"})
    last = df.groupby(key).agg({"CHR": "last", "END": "last"})
    chr_pos = [
        p
        for pair in zip(zip(first.CHR.values, first.START.values), zip(last.CHR.values, last.END.values))
        for p in pair
    ]
    position_cm = get_position_cM_table(list(chr_pos), geneticmap_file)
    prob = compute_phase_switch_probability_position(position_cm, chr_pos, nu)
    return np.minimum(np.log(0.5), np.log(prob) - shift)[np.arange(1, len(prob), 2)]


def snp_positions(sim_hash: str) -> list[tuple[int, int]]:
    """The sample's SNPs as (contig, position), in the file's order: `<contig>_<position>_<ref>_<alt>`."""
    raw = gzip.decompress((fixture(sim_hash) / "unique_snp_ids.npy.gz").read_bytes())
    ids = np.load(io.BytesIO(raw), allow_pickle=True)
    return [(int(s.split("_")[0]), int(s.split("_")[1])) for s in ids]


@pytest.fixture(scope="module")
def geneticmap() -> str:
    resources = grch38()
    if resources is None:
        pytest.skip("CalicoST's GRCh38_resources not found; set $CNAMASTE_GRCH38")
    return str(resources / get_resources()["geneticmap_file"])


@pytest.fixture(scope="module")
def shipped_map(geneticmap: str) -> Callable[[], Any]:
    """A context in which `utils_phase_switch` is handed a copy of CalicoST's own `pd.read_csv` of the shipped map,
    parsed on the first call that misses the cache (6 s) and never on a hit: the parse is CalicoST's call, the body
    after it untouched. Only for calls on the shipped map."""
    read = functools.cache(lambda: pd.read_csv(geneticmap, header=0, sep="\t"))
    stand_in = types.SimpleNamespace(read_csv=lambda *args, **kwargs: read().copy())
    return lambda: mock.patch.object(utils_phase_switch, "pd", stand_in)


@pytest.fixture(scope="module")
def reference(geneticmap: str) -> pd.DataFrame:
    return get_reference_recomb_rates(geneticmap)


@pytest.fixture(scope="module")
def easy() -> list[tuple[int, int]]:
    positions = snp_positions("2d4ce9a9")
    assert positions == sorted(positions) and len(set(positions)) == len(positions) == 13_482
    return positions


@pytest.fixture(scope="module")
def easy_cm(easy: list, reference: pd.DataFrame) -> np.ndarray:
    """cnamaste's cM of easy's SNPs, CalicoST's bitwise (`test_assign_centimorgans_is_calicost_s_on_easy`): the
    switch probabilities' input on both sides."""
    return assign_centiMorgans(easy, reference)


def get_calicost_transmat(
    referee: Referee, name: str, frame: pd.DataFrame, geneticmap: str, shipped_map: Any, shift: float
) -> str:
    """The digest of `calicost_transmat` on `frame` at `nu` 1, from the cache."""

    def call() -> tuple[np.ndarray]:
        with shipped_map():
            return (calicost_transmat(frame, "block_id", geneticmap, 1.0, shift),)

    (theirs,) = referee(f"recomb.transmat.{name}", (frame, Path(geneticmap), shift), call)
    return str(theirs)


def test_the_samples_hold_one_snp_set() -> None:
    """Easy and hard carry the same SNP positions, so easy's stand for both. Fails where a fixture's SNPs change."""
    assert snp_positions("2d4ce9a9") == snp_positions("8797710b")


# -- assign_centiMorgans -------------------------------------------------------------------------------------------


def test_assign_centimorgans_is_calicost_s_on_easy(
    easy: list, easy_cm: np.ndarray, geneticmap: str, shipped_map: Any, referee: Referee
) -> None:
    """Easy's 13,482 SNPs on the shipped map: CalicoST's cM, bitwise, none NaN. Fails where the interpolation, the
    extrapolation from a contig's start or past its last row, or the map's loading drifts."""

    def call() -> tuple[np.ndarray]:
        with shipped_map():
            return (get_position_cM_table(list(easy), geneticmap),)

    (theirs,) = referee("recomb.cm.easy", (easy, Path(geneticmap)), call)
    assert get_digest(easy_cm) == theirs
    assert np.isfinite(easy_cm).all()


EDGE_MAP = [
    ("chr1", 100, 0.5),
    ("chr1", 200, 1.0),
    ("chr1", 400, 1.0),
    ("chr1", 800, 3.0),
    ("chr2", 1000, 0.2),
    ("chr2", 2000, 0.6),
    ("chrX", 50, 0.1),
    ("chr4", 10, 0.0),
    ("chr4", 20, 0.0),
]
"""A map reaching every branch: a flat stretch (chr1 200-400), chrX (dropped by both), a contig of zero cM (chr4),
and contig 3 unmapped."""

EDGE_QUERIES = [
    (0, 5),
    (1, 0),
    (1, 50),
    (1, 100),
    (1, 150),
    (1, 150),
    (1, 200),
    (1, 300),
    (1, 800),
    (1, 900),
    (2, 1),
    (2, 1500),
    (2, 5000),
    (3, 10),
    (4, 15),
    (4, 30),
    (5, 1),
]
"""Before the map (contig 0), at position 0, before a contig's first row, on rows, a repeated query, between rows, past
a contig's last row, on an unmapped contig and past the map's end."""


def write_map(path: Path, rows: list[tuple[str, int, float]]) -> str:
    with gzip.open(path, "wt") as handle:
        handle.write("chrom\tpos\trecomb_rate\tpos_cm\n" + "".join(f"{c}\t{p}\t0\t{m}\n" for c, p, m in rows))
    return str(path)


def test_assign_centimorgans_is_calicost_s_on_every_branch(tmp_path: Path, referee: Referee) -> None:
    """`EDGE_QUERIES` on `EDGE_MAP`: CalicoST's cM, bitwise. Fails where a branch reads another row than CalicoST's
    sweep (row -1 before the map, the contig's last row past it) or other arithmetic."""
    path = write_map(tmp_path / "map.tab.gz", EDGE_MAP)
    ours = assign_centiMorgans(list(EDGE_QUERIES), get_reference_recomb_rates(path))
    call = functools.partial(get_position_cM_table, list(EDGE_QUERIES), path)
    assert (get_digest(ours),) == referee("recomb.cm.edges", (EDGE_QUERIES, EDGE_MAP), call)
    assert ours[[1, 4, 5, 7, 11]].tolist() == [0.0, 0.75, 0.75, 1.0, 0.4]


# -- compute_numbat_phase_switch_prob ------------------------------------------------------------------------------


@pytest.mark.parametrize("nu", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("min_prob", [CALICOST_FLOOR, 1e-10, 0.01])
def test_the_switch_probability_is_calicost_s_on_easy(
    easy: list, easy_cm: np.ndarray, nu: float, min_prob: float, referee: Referee
) -> None:
    """On easy's cM (CalicoST's), at each `nu` and a floor at or above CalicoST's: its switch probabilities, bitwise.
    Fails where the law `(1 - exp(-2 nu d)) / 2`, the contig change, or the floor drifts."""
    ours = compute_numbat_phase_switch_prob(easy_cm, easy, nu=nu, min_prob=min_prob)
    call = functools.partial(compute_phase_switch_probability_position, easy_cm, easy, nu=nu, min_prob=min_prob)
    inputs = (easy_cm, easy, nu, min_prob)
    assert (get_digest(ours),) == referee(f"recomb.switch.easy[{nu}-{min_prob}]", inputs, call)


EDGE_SWITCHES = {
    "empty": ([], []),
    "one SNP": ([1.0], [(1, 5)]),
    "NaN neighbours": ([1.0, np.nan, 2.0, 3.0], [(1, 1), (1, 2), (1, 3), (1, 4)]),
    "contig change": ([0.0, 1.0, 0.0, 3.0], [(1, 1), (1, 2), (2, 1), (2, 2)]),
    "no distance": ([1.0, 1.0, 1.0], [(1, 1), (1, 2), (1, 3)]),
    "saturated": ([0.0, 1e4, 2e4], [(1, 1), (1, 2), (1, 3)]),
    "below the floor": ([0.0, 1e-25, 1e-15], [(1, 1), (1, 2), (1, 3)]),
    "text contigs": ([0.0, 1.0], [("1", 1), ("1", 2)]),
}


@pytest.mark.parametrize("case", list(EDGE_SWITCHES))
@pytest.mark.parametrize("nu", [0.0, 1.0])
def test_the_switch_probability_is_calicost_s_on_every_branch(case: str, nu: float, referee: Referee) -> None:
    """Each of `EDGE_SWITCHES` at `nu` 0 and 1: CalicoST's probabilities, bitwise. Fails where an entry CalicoST skips
    (NaN, a contig change, the last) is computed, or one it computes is skipped."""
    cm, chr_pos = EDGE_SWITCHES[case]
    cm = np.asarray(cm, dtype=float)
    ours = compute_numbat_phase_switch_prob(cm, chr_pos, nu=nu, min_prob=CALICOST_FLOOR)
    call = functools.partial(compute_phase_switch_probability_position, cm, chr_pos, nu=nu)
    assert (get_digest(ours),) == referee(f"recomb.switch.edges[{nu}-{case}]", (cm, chr_pos, nu), call)


def test_both_refuse_decreasing_cm(referee: Referee) -> None:
    """cM falling within a contig: both raise. Fails where either accepts a negative distance."""
    cm, chr_pos = np.array([2.0, 1.0]), [(1, 1), (1, 2)]
    with pytest.raises(AssertionError):
        compute_numbat_phase_switch_prob(cm, chr_pos, nu=1.0, min_prob=CALICOST_FLOOR)
    call = functools.partial(compute_phase_switch_probability_position, cm, chr_pos, nu=1.0)
    with pytest.raises(AssertionError):
        referee("recomb.switch.decreasing", (cm, chr_pos, 1.0), call)


# -- get_sitewise_transmat -----------------------------------------------------------------------------------------


def segments(positions: list[tuple[int, int]], size: int) -> pd.DataFrame:
    """`df_gene_snp` for `positions` cut into segments of `size` consecutive SNPs within a contig, as blocks and bins
    are: one row per SNP, `START == END` its position, `block_id` its segment."""
    frame = pd.DataFrame(positions, columns=["CHR", "START"])
    frame["END"] = frame.START
    frame["block_id"] = frame.groupby("CHR").cumcount() // size + 1_000_000 * frame.CHR
    return frame


def configured(min_prob: float) -> None:
    set_global_config(YAMLConfig({"phasing": {"min_prob": min_prob}}))


EDGE_SNPS = [
    (1, 1),
    (1, 10_000),
    (1, 248_900_000),
    (1, 248_950_000),
    (2, 5_000_000),
    (21, 46_700_000),
    (22, 1),
    (22, 50_810_000),
]
"""Before contig 1's first map row and past its last (248.9 Mb, 248.95 Mb), one-SNP contigs, contig 22's first base and
past the map's end."""


@pytest.mark.parametrize(("size", "shift"), [(1, 0.0), (7, 1.5)])
def test_get_sitewise_transmat_is_calicost_s_on_easy(
    easy: list, geneticmap: str, shipped_map: Any, size: int, shift: float, referee: Referee
) -> None:
    """Easy's SNPs in segments of 1 at `logphase_shift` 0, and of 7 at 1.5, `min_prob` 1e-20 as shipped: CalicoST's log
    switch probabilities, bitwise. Fails where the segment ends, the cM, the floor, the shift or the end-of-segment
    selection drifts."""
    configured(CALICOST_FLOOR)
    frame = segments(easy, size)
    ours = get_sitewise_transmat("block_id", frame, geneticmap, 1.0, shift)
    assert len(ours) == frame.block_id.nunique()
    theirs = get_calicost_transmat(referee, f"easy[{size}-{shift}]", frame, geneticmap, shipped_map, shift)
    assert get_digest(ours) == theirs


def test_get_sitewise_transmat_is_calicost_s_at_the_map_s_edges(
    geneticmap: str, shipped_map: Any, referee: Referee
) -> None:
    """`EDGE_SNPS`, each its own segment: CalicoST's, bitwise. Fails where a segment beyond the map or alone on its
    contig reads another row than CalicoST's."""
    configured(CALICOST_FLOOR)
    frame = segments(EDGE_SNPS, 1)
    ours = get_sitewise_transmat("block_id", frame, geneticmap, 1.0, 0.0)
    assert get_digest(ours) == get_calicost_transmat(referee, "edges", frame, geneticmap, shipped_map, 0.0)


# -- the departures, as equality with CalicoST ---------------------------------------------------------------------


def departure(
    name: str, easy: list, easy_cm: np.ndarray, geneticmap: str, shipped_map: Any, path: str, referee: Referee
) -> bool:
    """Whether cnamaste equals CalicoST, bitwise, on the input of departure `name`: what each departure row above
    shows does not hold; an xfail here passes only while the departure stands. cnamaste runs first, as a refusal
    precedes CalicoST's call."""
    reference = get_reference_recomb_rates(path)
    if name in ("reorders", "chr contigs"):
        given = [(1, 300), (1, 100)] if name == "reorders" else [("chr1", 150), ("chr2", 1500)]
        ours = assign_centiMorgans(list(given), reference)
        call = functools.partial(get_position_cM_table, list(given), path)
        return (get_digest(ours),) == referee(f"recomb.departure.{name}", (given, EDGE_MAP), call)
    if name == "floor below 1e-20":
        ours = compute_numbat_phase_switch_prob(easy_cm, easy, nu=1.0, min_prob=0.0)
        call = functools.partial(compute_phase_switch_probability_position, easy_cm, easy, nu=1.0, min_prob=0.0)
        return (get_digest(ours),) == referee(f"recomb.departure.{name}", (easy_cm, easy, 0.0), call)
    configured(0.01 if name == "configured floor" else CALICOST_FLOOR)
    frame = segments(easy, 7) if name == "configured floor" else segments(EDGE_SNPS, 1)
    if name == "segments out of order":
        frame["block_id"] = -frame.block_id
    ours = get_sitewise_transmat("block_id", frame, geneticmap, 1.0, 0.0)
    return get_digest(ours) == get_calicost_transmat(referee, f"departure.{name}", frame, geneticmap, shipped_map, 0.0)


DEPARTURES = {
    "reorders": "CalicoST sorts pairs out of order in place; cnamaste refuses them (T- #20)",
    "chr contigs": "CalicoST reads chr-prefixed contigs; cnamaste's loader yields integer contigs and refuses others",
    "floor below 1e-20": "an entry with no switch takes min_prob in cnamaste, 1e-20 in CalicoST",
    "configured floor": "cnamaste floors at phasing.min_prob, CalicoST's composition at its default 1e-20",
    "segments out of order": "cnamaste refuses segment keys out of genomic order; CalicoST sorts the ends",
}


REFUSED = {"chr contigs": ValueError}
"""The exception each departure's equality fails with, where not an `AssertionError`: anything else is an error."""


@pytest.mark.parametrize(
    "name",
    [
        pytest.param(name, marks=departs(why, raises=REFUSED.get(name, AssertionError)))
        for name, why in DEPARTURES.items()
    ],
)
def test_cnamaste_is_calicost_where_it_departs(
    name: str, easy: list, easy_cm: np.ndarray, geneticmap: str, shipped_map: Any, tmp_path: Path, referee: Referee
) -> None:
    """Each stated departure as plain equality with CalicoST: strict xfails, so a departure that disappears fails
    the run. Fails (xpasses) where cnamaste comes to equal CalicoST on the departure's input."""
    path = write_map(tmp_path / "map.tab.gz", EDGE_MAP)
    assert departure(name, easy, easy_cm, geneticmap, shipped_map, path, referee)
