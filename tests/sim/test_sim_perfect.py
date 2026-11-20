"""`run_cnamaste` end to end on each perfect sample (`cnamaste.sim.perfect.PERFECT`, seed 0, depth factor 100):
perfect_easy (`f11658d6`) from CalicoST easy (`2d4ce9a9`)'s truth, perfect_hard (`bd4c8102`) from CalicoST hard
(`8797710b`)'s; each row a recovery the planted truth makes exact.

Referee: the sample's planted truth (`truth_clone_labels`, `truth_acn_profile`, `truth_phase`). A sample is drawn
into `$CNAMASTE_SIM_CACHE`, else pytest's cache, else the run's tmp dir, and reused where its hash is its pin; the
run is `tests/test_zenodo_sim.py`'s, once per sample and session, the samples one after the other. Per sample about
110 s and 4.1 GB to draw, 350 s and 11.1 GB to run; a real `run_cnamaste`, so the module is `end2end`.
A row short of perfect is a strict xfail on `Shortfall`, its measured value recorded: a drift from it fails outright.
"""

from __future__ import annotations

import fcntl
import gzip
import io
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import pytest
import yaml

from cnamaste import external
from cnamaste.io import read_numpy, read_text_table
from cnamaste.qa.metrics import RunEstimate, clone_ari, fitted_at_barcode, phase_free, read_estimate
from cnamaste.qa.sim_runs import run_dir_of
from cnamaste.qa.sims import get_fixture_hash, get_project
from cnamaste.results import get_output_path
from cnamaste.sim.perfect import get_perfect, get_snp_loci
from cnamaste.truth import Truth, read_truth
from test_zenodo_sim import get_override, merged, references

pytestmark = pytest.mark.end2end

SEED, DEPTH_FACTOR = 0, 100.0
CACHE_VARIABLE = "CNAMASTE_SIM_CACHE"
PLOIDY_NAMES = {2: "diploid", 3: "triploid", 4: "tetraploid"}


class Recorded(NamedTuple):
    """A sample's measured recovery, the values its rows assert.

    Args: `exact_ab` (exact, observable) planted segment and true clone pairs; `ab_misses` per missed pair (chr, start,
    true clone, planted (A, B) phase-free as (minor, major), missed bins); `unpooled_bins` (chr, start, end) of each
    missed bin not pooled across a planted switch; `normal_candidates` (candidates, planted) normal spots;
    `phase_agreement` (agreeing, resolvable) informative SNPs.
    """

    exact_ab: tuple[int, int]
    ab_misses: list[tuple[int, int, str, tuple[int, int], int]]
    unpooled_bins: list[tuple[int, int, int]]
    normal_candidates: tuple[int, int]
    phase_agreement: tuple[int, int]


# NB measured with `zenodo_sim_config.yaml` at `ced3f57` (2026-10-30).
RECORDED = {
    "perfect_easy": Recorded(
        exact_ab=(134, 140),
        ab_misses=[
            (1, 172929859, "clone_0", (1, 3), 2),
            (20, 12888623, "clone_0", (0, 2), 7),
            (20, 12888623, "clone_1", (0, 2), 6),
            (20, 52701626, "clone_0", (0, 2), 3),
            (20, 52701626, "clone_1", (0, 2), 3),
            (20, 52701626, "clone_2", (0, 2), 3),
        ],
        unpooled_bins=[],
        normal_candidates=(669, 670),
        phase_agreement=(925, 1010),
    ),
    "perfect_hard": Recorded(
        exact_ab=(187, 196),
        ab_misses=[
            (5, 102990579, "clone_1", (0, 2), 1),
            (6, 80102583, "clone_2", (0, 1), 1),
            (8, 62515875, "clone_2", (1, 3), 1),
            (12, 60969723, "clone_0", (0, 2), 1),
            (12, 60969723, "clone_1", (0, 2), 1),
            (12, 60969723, "clone_2", (0, 2), 1),
            (16, 74753033, "clone_0", (0, 2), 4),
            (16, 74753033, "clone_1", (0, 2), 4),
            (16, 74753033, "clone_2", (0, 2), 4),
        ],
        unpooled_bins=[(6, 81813286, 89675783)],
        normal_candidates=(669, 670),
        phase_agreement=(472, 472),
    ),
}
SHORTFALLS = {
    ("perfect_easy", "normal_candidates"): "T- #28: the 100% mask drops the clone's highest-std spot",
    ("perfect_easy", "phase"): "T- #38: chr13's (1, 2) segment is left at phase 0",
    ("perfect_hard", "normal_candidates"): "T- #28: the 100% mask drops the clone's highest-std spot",
    ("perfect_hard", "ab"): "T- #46: chr6 clone_2's (0, 1) is decoded as copy-neutral LOH (0, 2)",
}
"""(sample, row) -> the ticket of a row's recorded shortfall."""


class Shortfall(Exception):
    """A row's recorded shortfall from perfect recovery: the `raises` of its strict xfail."""


def get_samples(row: str) -> list[object]:
    """Each perfect sample as a `recovery` parameter, a strict xfail where `SHORTFALLS` holds its row."""
    return [
        pytest.param(
            name,
            id=name,
            marks=[pytest.mark.xfail(strict=True, raises=Shortfall, reason=SHORTFALLS[name, row])]
            if (name, row) in SHORTFALLS
            else [],
        )
        for name in get_perfect()
    ]


def on_samples(row: str) -> pytest.MarkDecorator:
    """The row parametrized over the perfect samples, one `recovery` per sample and module."""
    return pytest.mark.parametrize("recovery", get_samples(row), indirect=True, scope="module")


class Recovery(NamedTuple):
    """The drawn sample, its truth and the run's estimate.

    Args: `name` the sample's short name, a key of `PERFECT`; `sample` its directory; `run_dir` the run's; `truth` the planted truth; `estimate` the run's.
    """

    name: str
    sample: Path
    run_dir: Path
    truth: Truth
    estimate: RunEstimate


class Bins(NamedTuple):
    """The run's bins (`cnv_<ploidy>_seglevel.tsv` rows) against the planted profile.

    Args: `chrom`, `start`, `end` (bins,); `segment` each bin's planted segment (row of `truth_acn_profile`), the
    first it overlaps; `straddles` a bin over a planted breakpoint; `called`, `planted` (bins, true clones)
    phase-free `A * 1000 + B`, the called through each true clone's fitted clone.
    """

    chrom: np.ndarray
    start: np.ndarray
    end: np.ndarray
    segment: np.ndarray
    straddles: np.ndarray
    called: np.ndarray
    planted: np.ndarray


class Snps(NamedTuple):
    """Each SNP's planted and recovered phase.

    Args: `chrom`, `pos` (SNPs,); `planted` `truth_phase`; `block` its initial block, -1 outside every block's span;
    `recovered` the block's phase (`segmentation.tsv.gz`); `segment` its planted segment; `informative` some clone's
    planted A != B; `switch_held` its block holds a planted switch.
    """

    chrom: np.ndarray
    pos: np.ndarray
    planted: np.ndarray
    block: np.ndarray
    recovered: np.ndarray
    segment: np.ndarray
    informative: np.ndarray
    switch_held: np.ndarray


def run_measured(command: list[str], log: Path) -> tuple[float, float]:
    """`python <command>` from the repository root, its output in `log`; asserts it succeeds.

    Returns: its wall seconds and peak resident set, GB (`os.wait4`'s own child, Linux `ru_maxrss` in KB).
    """
    started = time.perf_counter()
    with open(log, "wb") as written:
        child = subprocess.Popen(
            [sys.executable, *command], cwd=get_project(), stdout=written, stderr=subprocess.STDOUT
        )
        _, status, usage = os.wait4(child.pid, 0)
    # NB reaped here, so `Popen` is told rather than left to wait again.
    child.returncode = os.waitstatus_to_exitcode(status)
    assert child.returncode == 0, log.read_text()[-3000:]
    return time.perf_counter() - started, usage.ru_maxrss / 1e6


def get_sample(name: str, cache: Path, out: Path) -> Path:
    """`cache/<name>_<hash>`, drawn by `run_sim_gen` (logged in `out/draw.log`) unless its hash is its pin."""
    pinned = get_perfect()[name].sample
    directory = cache / f"{name}_{pinned}"
    if not (directory.is_dir() and get_fixture_hash(directory) == pinned):
        command = ["-m", "cnamaste.scripts.run_sim_gen", name, "--seed", str(SEED), "--depth-factor", str(DEPTH_FACTOR)]
        seconds, peak = run_measured([*command, "--out", str(cache)], out / "draw.log")
        print(f"\n{name}_{pinned}: drawn {seconds:.1f} s, peak {peak:.2f} GB")
    assert get_fixture_hash(directory) == pinned
    return directory


def get_shared_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """This session's directory every xdist worker sees: the workers' common parent, else the session's own."""
    base = tmp_path_factory.getbasetemp()
    return base.parent if os.environ.get("PYTEST_XDIST_WORKER") else base


def run_sample(name: str, sample: Path, out: Path, references: dict[str, str]) -> None:
    """`run_cnamaste` on `sample` from the repository root, `tests/test_zenodo_sim.py`'s merged configuration with
    its output in `out/output` and its log in `out/run.log`."""
    located = str(sample.resolve())
    sheet = pd.DataFrame({"bam": ["NULL"], "sample_id": [name], "spaceranger_dir": [located], "snp_dir": [located]})
    sheet.to_csv(out / "sample_sheet.tsv", sep="\t", index=False)
    config = merged(
        yaml.safe_load((get_project() / "zenodo_sim_config.yaml").read_text()),
        yaml.safe_load(get_override().read_text()),
    )
    config["references"].update(references)
    config["paths"]["sample_sheet"] = str(out / "sample_sheet.tsv")
    (out / "output").mkdir()
    config["paths"]["output_dir"] = str(out / "output")
    (out / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    command = ["-m", "cnamaste.scripts.run_cnamaste", str(out / "config.yaml")]
    seconds, peak = run_measured(command, out / "run.log")
    print(f"\n{name}_{get_perfect()[name].sample}: run {seconds:.1f} s, peak {peak:.2f} GB")


@pytest.fixture(scope="module")
def recovery(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> Recovery:
    """`request.param`'s sample drawn (or reused) and run once per session with the shipped configuration."""
    if not external.available():
        pytest.skip(str(external.unavailable()))
    found = references()
    if found is None:
        pytest.skip("GRCh38 resources: neither ref/ nor $CNAMASTE_GRCH38 holds them")
    name = request.param
    named = os.environ.get(CACHE_VARIABLE)
    cache = Path(named) if named else request.config.cache.mkdir("sim") if request.config.cache else None
    shared = get_shared_dir(tmp_path_factory)
    out = shared / f"{name}_{get_perfect()[name].sample}_run"
    # NB one lock for every sample: the first worker to take it draws and runs, the rest read its run, and no two
    #    samples draw or run at once. `xdist_group` cannot pin the module to one worker (T- #42).
    with open(shared / "perfect.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not (out / "ran").exists():
            assert not out.exists(), f"an earlier worker's draw or run failed: {out}"
            out.mkdir()
            sample = get_sample(name, cache or out, out)
            run_sample(name, sample, out, found)
            (out / "ran").write_text(str(sample))
    sample, run_dir = Path((out / "ran").read_text()), run_dir_of(out / "output")
    return Recovery(name, sample, run_dir, read_truth(sample), read_estimate(run_dir))


def get_clone_of_truth(recovery: Recovery) -> list[int]:
    """Each true clone's fitted clone: the one most of its spots carry."""
    fitted = fitted_at_barcode(recovery.estimate, recovery.truth)
    labels = recovery.truth.labels
    return [int(pd.Series(fitted[labels == t]).mode()[0]) for t in range(recovery.truth.n_clones)]


def get_planted_codes(truth: Truth) -> np.ndarray:
    """(segments, true clones) phase-free planted `A * 1000 + B`, `truth_acn_profile`'s rows."""
    profile = truth.profile
    return np.stack(
        [
            phase_free(profile[f"{c}_A_copy"].to_numpy() * 1_000 + profile[f"{c}_B_copy"].to_numpy())
            for c in truth.clones
        ],
        axis=1,
    )


@pytest.fixture(scope="module")
def bins(recovery: Recovery) -> Bins:
    seglevel, profile = recovery.estimate.seglevel, recovery.truth.profile
    chrom, start, end = (seglevel[c].to_numpy(np.int64) for c in ("CHR", "START", "END"))
    on = (profile["chr"].to_numpy(np.int64)[None, :] == chrom[:, None]) & (
        (profile["start"].to_numpy()[None, :] < end[:, None]) & (profile["end"].to_numpy()[None, :] > start[:, None])
    )
    # NB every bin lies in a planted segment: the profile tiles each contig.
    assert on.any(axis=1).all()
    called = np.stack(
        [
            phase_free(seglevel[f"clone{k} A"].to_numpy(np.int64) * 1_000 + seglevel[f"clone{k} B"].to_numpy(np.int64))
            for k in get_clone_of_truth(recovery)
        ],
        axis=1,
    )
    segment = on.argmax(axis=1)
    planted = get_planted_codes(recovery.truth)[segment]
    return Bins(chrom, start, end, segment, on.sum(axis=1) > 1, called, planted)


@pytest.fixture(scope="module")
def snps(recovery: Recovery) -> Snps:
    sample, truth = recovery.sample, recovery.truth
    planted = np.load(io.BytesIO(gzip.decompress((sample / "truth_phase.npy.gz").read_bytes()))).astype(np.int64)
    chrom, pos = get_snp_loci(read_numpy(sample / "unique_snp_ids.npy", allow_pickle=True))
    genes = read_text_table(recovery.run_dir / "segmentation.tsv", sep="\t")
    blocks = (
        genes[genes["initial_blocks"] >= 0]
        .groupby("initial_blocks")
        .agg(CHR=("CHR", "first"), START=("START", "min"), END=("END", "max"), phase=("phase", "first"))
    )
    # NB a block's span is its genes'; spans are disjoint, as blocks are runs of the sorted gene table.
    block = np.full(len(pos), -1)
    for contig in np.unique(chrom):
        spans, on = blocks[blocks["CHR"] == contig].sort_values("START"), np.flatnonzero(chrom == contig)
        k = np.searchsorted(spans["START"].to_numpy(), pos[on], "right") - 1
        inside = (k >= 0) & (pos[on] < spans["END"].to_numpy()[np.maximum(k, 0)])
        block[on[inside]] = spans.index.to_numpy()[k[inside]]
    recovered = np.where(block >= 0, blocks["phase"].reindex(block).to_numpy(), -1).astype(np.int64)
    copies = truth.copies_at(chrom, pos)
    informative = (copies[..., 0] != copies[..., 1]).any(axis=1) & (copies >= 0).all(axis=(1, 2))
    segment = np.full(len(pos), -1)
    profile = truth.profile
    for row, (c, lo, hi) in enumerate(profile[["chr", "start", "end"]].itertuples(index=False)):
        segment[(chrom == int(c)) & (pos >= lo) & (pos < hi)] = row
    phases = pd.Series(planted).groupby(block).nunique()
    switch_held = np.isin(block, phases.index[(phases > 1) & (phases.index >= 0)])
    return Snps(chrom, pos, planted, block, recovered, segment, informative, switch_held)


def get_observed_segments(bins: Bins, n_segments: int) -> np.ndarray:
    """(segments,) planted segments holding a bin inside them, i.e. one over no planted breakpoint."""
    return np.isin(np.arange(n_segments), bins.segment[~bins.straddles])


def get_missed_bins(bins: Bins) -> np.ndarray:
    """(bins, true clones) where a bin over no planted breakpoint is called off its planted AB pair."""
    return (bins.called != bins.planted) & ~bins.straddles[:, None]


def get_pooled_bins(bins: Bins, snps: Snps) -> np.ndarray:
    """(bins,) bins whose informative SNPs, at least one, all lie in a block holding a planted switch."""
    pooled = np.zeros(len(bins.chrom), dtype=bool)
    for i in range(len(bins.chrom)):
        on = (snps.chrom == bins.chrom[i]) & (snps.pos >= bins.start[i]) & (snps.pos < bins.end[i]) & snps.informative
        pooled[i] = on.any() and snps.switch_held[on].all()
    return pooled


@on_samples("clones")
def test_the_clones_are_the_planted_clones(recovery: Recovery) -> None:
    """Clone ARI 1 against `truth_clone_labels`, over every kept spot; fails on one spot in the wrong clone."""
    found = clone_ari(recovery.estimate, recovery.truth)
    print(f"{recovery.name} clone ARI {found:.4f}")
    assert found == 1.0


@on_samples("ab")
def test_every_ab_miss_is_a_bin_pooled_across_a_switch(recovery: Recovery, bins: Bins, snps: Snps) -> None:
    """Exact phase-free (A, B) per planted segment and true clone, over the bins inside the segment: `RECORDED`'s
    `exact_ab`, perfect_easy 134 of 140.

    Resolution limits, asserted: a bin over a planted breakpoint is left out, and with it a segment holding no other
    bin (perfect_hard's chr9 (0, 2) at 40.29 Mb, inside one 31 Mb pericentromeric bin); every missed bin is one whose
    informative SNPs all lie in initial blocks holding a planted switch, where one block phase pools both haplotypes
    and the BAF reads 1/2 (copy-neutral LOH (0, 2) as (1, 1), (3, 1) as (2, 2)). Fails where a miss moves or a bin
    with a resolvable SNP is missed; a recorded such bin is a `Shortfall` (T- #46)."""
    missed = get_missed_bins(bins)
    profile, clones = recovery.truth.profile, recovery.truth.clones
    observed = get_observed_segments(bins, len(profile))
    misses = []
    for segment, clone in np.ndindex(len(profile), len(clones)):
        n = int(missed[bins.segment == segment, clone].sum())
        if n:
            planted = get_planted_codes(recovery.truth)[segment, clone]
            row = (int(profile["chr"].iloc[segment]), int(profile["start"].iloc[segment]))
            misses.append((*row, clones[clone], (int(planted // 1_000), int(planted % 1_000)), n))
    pairs = int(observed.sum()) * len(clones)
    unpooled = [
        (int(bins.chrom[i]), int(bins.start[i]), int(bins.end[i]))
        for i in np.flatnonzero(missed.any(axis=1) & ~get_pooled_bins(bins, snps))
    ]
    print(
        f"{recovery.name} exact (A, B) {pairs - len(misses)} of {pairs} observable; misses {misses}; "
        f"unpooled bins {unpooled}"
    )
    recorded = RECORDED[recovery.name]
    assert (pairs - len(misses), pairs) == recorded.exact_ab
    assert misses == recorded.ab_misses
    assert unpooled == recorded.unpooled_bins
    if unpooled:
        raise Shortfall(f"{len(unpooled)} missed bins hold a resolvable informative SNP")


@on_samples("breakpoints")
def test_every_planted_breakpoint_is_recovered_within_one_bin(recovery: Recovery, bins: Bins) -> None:
    """A planted breakpoint (a profile row start where some clone's phase-free (A, B) changes) has a recovered one
    (a bin boundary where some true clone's called (A, B) changes) within one bin, and a recovered one a planted one
    or a missed bin's edge, the previous row's limit. Resolution limit: a planted segment holding no bin is not
    observable and is left out, its neighbours' breakpoint the change between them (perfect_easy's chr13 0-2.83 Mb,
    before its first bin; perfect_hard's chr9 40.29-50.29 Mb, inside one bin). Fails on a breakpoint missed by two bins
    or a spurious one elsewhere."""
    profile, planted_codes = recovery.truth.profile, get_planted_codes(recovery.truth)
    observed = get_observed_segments(bins, len(profile))
    missed = get_missed_bins(bins).any(axis=1)
    unmatched, spurious = [], []
    for contig in np.unique(bins.chrom):
        on = np.flatnonzero(bins.chrom == contig)
        rows = np.flatnonzero((profile["chr"].to_numpy(np.int64) == contig) & observed)
        changes = (np.diff(planted_codes[rows], axis=0) != 0).any(axis=1)
        planted = np.searchsorted((bins.start[on] + bins.end[on]) // 2, profile["start"].to_numpy()[rows][1:][changes])
        planted = planted[(planted > 0) & (planted < len(on))]
        recovered = np.flatnonzero((np.diff(bins.called[on], axis=0) != 0).any(axis=1)) + 1
        unmatched += [(int(contig), int(b)) for b in planted if not (np.abs(recovered - b) <= 1).any()]
        spurious += [
            (int(contig), int(j))
            for j in recovered
            if not (np.abs(planted - j) <= 1).any() and not (missed[on[j - 1]] or missed[on[j]])
        ]
    print(f"{recovery.name} unmatched {unmatched}, spurious {spurious} (contig, bin)")
    assert unmatched == [] and spurious == []


@on_samples("normal_candidates")
def test_the_normal_candidates_are_the_planted_normal_spots(recovery: Recovery) -> None:
    """`clone_labels.tsv`'s `normal_candidate` spots are the planted normal spots, none of them tumour: `RECORDED`'s
    `normal_candidates`, 669 of 670 on each, spot_1553 (easy) or spot_323 (hard) out, the highest-std spot, which
    `determine_normal_candidates`' `vec_stds < percentile(100)` drops (T- #28)."""
    truth = recovery.truth
    labels = read_text_table(recovery.run_dir / "clone_labels.tsv", sep="\t", comment="#").set_index("barcode")
    candidate = labels["normal_candidate"].reindex(truth.barcodes, fill_value=0).to_numpy() == 1
    normal = truth.labels == 0
    found = (int((candidate & normal).sum()), int(normal.sum()))
    print(f"{recovery.name} normal candidates {found[0]} of {found[1]}, {int((candidate & ~normal).sum())} tumour")
    assert found == RECORDED[recovery.name].normal_candidates and not (candidate & ~normal).any()
    if found[0] != found[1]:
        raise Shortfall(f"{found[0]} of {found[1]} planted normal spots are candidates")


@on_samples("phase")
def test_the_block_phase_follows_the_planted_phase(recovery: Recovery, snps: Snps) -> None:
    """On informative SNPs (some clone's planted A != B) in a block, `d = truth_phase XOR block phase` is the planted
    haplotype the run's corrected B count reads; perfect phasing holds `d` constant over each planted segment, i.e.
    the block phase flips at each planted switch and only there, one flip of the whole segment free (which haplotype
    is B is arbitrary). Agreement: the share of SNPs whose `d` is their segment's majority, `RECORDED`'s
    `phase_agreement`, perfect_easy 925 of 1,010 (0.9158).

    Resolution limit: a block holding a planted switch has one phase for both sides and is left out. Fails where a
    block is phased against its segment."""
    used = snps.informative & (snps.block >= 0) & ~snps.switch_held
    d = snps.planted ^ snps.recovered
    agree = 0
    for segment in np.unique(snps.segment[used]):
        on = used & (snps.segment == segment)
        agree += int(max((d[on] == 0).sum(), (d[on] == 1).sum()))
    found = (agree, int(used.sum()))
    print(f"{recovery.name} phase agreement {found[0]} of {found[1]} ({found[0] / found[1]:.4f})")
    assert found == RECORDED[recovery.name].phase_agreement
    if found[0] != found[1]:
        raise Shortfall(f"phase agreement {found[0]} of {found[1]}")


@on_samples("ploidy")
def test_each_clone_s_ploidy_is_the_planted_integer_ploidy(recovery: Recovery, bins: Bins) -> None:
    """Each true clone's length-weighted mean A + B rounds to the planted one, 2 for both samples, and the run's
    tables name that ploidy (`cnv_diploid_*`). Fails where a clone's decoded genome moves by over 1/2 a copy, or the
    run picks another ploidy."""
    truth, profile = recovery.truth, recovery.truth.profile
    length = (profile["end"] - profile["start"]).to_numpy()
    planted = [
        float(((profile[f"{c}_A_copy"] + profile[f"{c}_B_copy"]).to_numpy() * length).sum() / length.sum())
        for c in truth.clones
    ]
    if len({round(p) for p in planted}) != 1:
        pytest.skip(f"the truth implies no one integer ploidy: {planted}")
    width = bins.end - bins.start
    total = bins.called // 1_000 + bins.called % 1_000
    called = [float((total[:, t] * width).sum() / width.sum()) for t in range(truth.n_clones)]
    print(f"{recovery.name} mean A + B planted {np.round(planted, 4)}, called {np.round(called, 4)}")
    assert [round(c) for c in called] == [round(p) for p in planted]
    assert get_output_path(recovery.run_dir, "seglevel").name == f"cnv_{PLOIDY_NAMES[round(planted[0])]}_seglevel.tsv"
