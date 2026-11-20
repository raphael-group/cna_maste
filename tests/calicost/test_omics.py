"""`omics` against the CalicoST `c1abcae` functions it stands in for, bitwise; the referee is CalicoST's own output on
the same input, computed once and recorded (`calicost_cache`).

- `form_gene_snp_table` / `utils_IO.combine_gene_snps`; `assign_initial_blocks` / `create_haplotype_block_ranges`;
- `summarize_counts_for_blocks` and `summarize_counts_for_bins` / `utils_IO`'s, the counts (CalicoST's phase-switch
  transitions are `test_recomb.py`'s);
- `create_bin_ranges` / `utils_IO.create_bin_ranges`; `binned_gene_snp` / `parse_input.genesnp_to_bininfo`;
- `io.load_input_data` / `utils_IO.load_data`, on easy whole, its inputs uncompressed to a temporary directory.

Inputs: CalicoST easy (`2d4ce9a9`) as `run_cnamaste` loads it, under `zenodo_sim_config.yaml` and `ref/`'s GRCh38
resources, its first `N_SPOTS` spots dense (CalicoST sums dense arrays; the equality holds on any subset of spots); and
synthetic tables for the departures. Each stated departure is an equality row, `xfail(strict=True)`. Skipped where the
resources are absent. Each comparison is the sha256 of CalicoST's output, against cnamaste's.
"""

from __future__ import annotations

import gzip
import importlib
import os
import sys
from pathlib import Path
from typing import Any
from unittest import mock

import anndata
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from calicost import utils_IO

import cnamaste.calicost  # noqa: F401  (imports CalicoST with the environment held, as cnamaste does)
from calicost_cache import Referee, departs, get_digest
from cnamaste.config import YAMLConfig, get_global_config, set_global_config
from cnamaste.io import load_input_data
from cnamaste.omics import (
    assign_initial_blocks,
    binned_gene_snp,
    create_bin_ranges,
    form_gene_snp_table,
    summarize_counts_for_bins,
    summarize_counts_for_blocks,
)
from cnamaste.qa.sims import fixture, get_project, get_resources, grch38
from labelled_counts import get_labelled_counts


def import_calicost(name: str) -> Any:
    """`calicost.<name>`, imported with `calicost.phasing` stood in by a mock for the import alone.

    `calicost.parse_input` imports `calicost.phasing`, whose module-level imports need a desktop GUI toolkit that a
    headless host lacks; nothing compared here calls into it.
    """
    with mock.patch.dict(sys.modules, {"calicost.phasing": mock.MagicMock()}):
        return importlib.import_module(f"calicost.{name}")


parse_input = import_calicost("parse_input")

N_SPOTS = 300
"""Easy's first spots, dense: 32 MB per allele."""

COLUMNS = ["CHR", "START", "END", "snp_id", "gene", "is_interval"]


def dense(a: Any) -> np.ndarray:
    return a.toarray() if sp.issparse(a) else np.asarray(a)


def get_adata_inputs(adata: anndata.AnnData) -> tuple[Any, ...]:
    """What CalicoST reads of `adata`, for the cache's key: its spots, genes and counts."""
    count = adata.layers.get("count")
    return list(adata.obs_names), list(adata.var_names), dense(adata.X), None if count is None else dense(count)


def get_easy_inputs(easy: dict[str, Any], blocks: pd.DataFrame | None = None) -> tuple[Any, ...]:
    """Easy's inputs to a CalicoST call, for the cache's key: the counts, SNP ids, gene table and map, and `blocks`."""
    references = (Path(easy["hgtable"]), Path(easy["map"]), easy["floor"])
    return *get_adata_inputs(easy["adata"]), easy["A"], easy["B"], easy["ids"], *references, blocks


@pytest.fixture(scope="module")
def loaded() -> tuple[YAMLConfig, Any]:
    """Easy as `run_cnamaste` loads it, under `zenodo_sim_config.yaml` and `ref/`'s GRCh38 resources: the config and
    `load_input_data`'s outputs."""
    resources = grch38()
    if resources is None:
        pytest.skip("CalicoST's GRCh38_resources not found; set $CNAMASTE_GRCH38")
    config = YAMLConfig.from_file(get_project() / "zenodo_sim_config.yaml")
    config.over_ride(
        [f"paths.sample_sheet={fixture('2d4ce9a9') / 'sample_sheet.tsv'}", f"paths.output_dir={get_project()}"]
        + [f"references.{key}={resources / name}" for key, name in get_resources().items()]
    )
    # NB the sample sheet's relative paths resolve from the run's directory, the repository root.
    previous, cwd = get_global_config(), os.getcwd()
    set_global_config(config)
    os.chdir(get_project())
    try:
        found = load_input_data(
            config,
            filter_gene_file=config.references.filtergenelist_file,
            filter_range_file=config.references.filterregion_file,
            min_snp_umis=config.quality.spot_min_snp_umis,
            min_percent_expressed_spots=config.quality.min_percent_expressed_spots,
        )
    finally:
        set_global_config(previous)
        os.chdir(cwd)
    return config, found


@pytest.fixture(scope="module")
def easy(loaded: tuple[YAMLConfig, Any]) -> dict[str, Any]:
    """Easy's AnnData, A and B counts and SNP ids, as `run_cnamaste` loads them; its first `N_SPOTS` spots, dense."""
    config, found = loaded
    adata = found[2][:N_SPOTS].copy()
    adata.layers["count"] = dense(adata.layers["count"])
    return {
        "adata": adata,
        "A": dense(found[4][:N_SPOTS]),
        "B": dense(found[5][:N_SPOTS]),
        "ids": found[6],
        "hgtable": str(config.references.hgtable_file),
        "map": str(config.references.geneticmap_file),
        "floor": int(config.quality.phasing_min_snp_umis),
    }


@pytest.fixture(scope="module")
def table(easy: dict[str, Any]) -> pd.DataFrame:
    """`form_gene_snp_table` at CalicoST's search of 50 rows."""
    return form_gene_snp_table(easy["ids"], easy["hgtable"], easy["adata"], num_preceeding_rows=50)


@pytest.fixture(scope="module")
def blocks(easy: dict[str, Any], table: pd.DataFrame) -> pd.DataFrame:
    return assign_initial_blocks(table.copy(), easy["adata"], easy["A"], easy["B"], easy["ids"], easy["floor"])


@pytest.fixture(scope="module")
def block_counts(easy: dict[str, Any], blocks: pd.DataFrame) -> Any:
    return summarize_counts_for_blocks(blocks, easy["adata"], easy["A"], easy["B"], easy["ids"])


def test_the_gene_snp_table_is_calicost_s(easy: dict[str, Any], table: pd.DataFrame, referee: Referee) -> None:
    """Easy's 15,175 gene and SNP rows: CalicoST's, row for row, each SNP's gene included. Fails where a SNP takes
    another gene, a row is kept or dropped otherwise, or the order differs."""

    def call() -> pd.DataFrame:
        return utils_IO.combine_gene_snps(easy["ids"], easy["hgtable"], easy["adata"])[COLUMNS].reset_index(drop=True)

    (theirs,) = referee("omics.gene_snp_table", get_easy_inputs(easy), call)
    assert get_digest(table[COLUMNS].reset_index(drop=True)) == theirs


def test_the_initial_blocks_are_calicost_s(
    easy: dict[str, Any], table: pd.DataFrame, blocks: pd.DataFrame, referee: Referee
) -> None:
    """Easy at `quality.phasing_min_snp_umis`: CalicoST's `block_id` on every row (1,041 blocks). Fails where gene
    overlap, the SNP-UMI floor or a contig end segments otherwise."""

    def call() -> np.ndarray:
        return utils_IO.create_haplotype_block_ranges(
            table.copy(), easy["adata"], easy["A"], easy["B"], easy["ids"], easy["floor"]
        ).block_id.to_numpy()

    (theirs,) = referee("omics.initial_blocks", get_easy_inputs(easy, table), call)
    assert get_digest(blocks.block_id.to_numpy()) == theirs


def test_the_block_counts_are_calicost_s(
    easy: dict[str, Any], blocks: pd.DataFrame, block_counts: Any, referee: Referee
) -> None:
    """Per block and spot, read counts, B counts and allele depth, and blocks per contig: CalicoST's, exactly."""

    def call() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        theirs = utils_IO.summarize_counts_for_blocks(
            blocks, easy["adata"], easy["A"], easy["B"], easy["ids"], 1.0, 0.0, easy["map"]
        )
        return theirs[0], theirs[1], theirs[3]

    ours = (block_counts.lengths, get_stacked(block_counts), block_counts.total_snp_umis.toarray().T)
    assert tuple(map(get_digest, ours)) == referee("omics.block_counts", get_easy_inputs(easy, blocks), call)


def bins_of(
    blocks: pd.DataFrame, block_counts: Any, floor: int, referee: Referee, max_binlength: float = 5e6
) -> tuple[pd.DataFrame, str]:
    """`create_bin_ranges` with CalicoST's one floor (SNP UMIs), refined segments the contigs: ours, and the digest
    of CalicoST's `bin_id`."""
    ours = create_bin_ranges(
        blocks.copy(),
        None,
        None,
        None,
        None,
        block_counts,
        block_counts.lengths,
        0,
        floor,
        0,
        max_binlength=max_binlength,
    )
    total = block_counts.total_snp_umis.toarray().T

    def call() -> np.ndarray:
        return utils_IO.create_bin_ranges(
            blocks.copy(), total, block_counts.lengths, floor, max_binlength
        ).bin_id.to_numpy()

    inputs = (blocks, total, block_counts.lengths, floor, max_binlength)
    (theirs,) = referee(f"omics.bins[{floor}-{max_binlength}]", inputs, call)
    return ours, str(theirs)


def test_the_bin_breakpoints_are_calicost_s(blocks: pd.DataFrame, block_counts: Any, referee: Referee) -> None:
    """Floor 0, so each run is cut only by the refined segments and the blocks over `max_binlength`: CalicoST's bins.
    Fails where a breakpoint is drawn otherwise."""
    ours, theirs = bins_of(blocks, block_counts, 0, referee)
    assert get_digest(ours.bin_id.to_numpy()) == theirs


def test_the_bin_counts_are_calicost_s(
    easy: dict[str, Any], blocks: pd.DataFrame, block_counts: Any, referee: Referee
) -> None:
    """Easy's blocks in bins, phased by a seeded indicator: per bin, read counts, phased B counts and depth, and bins
    per contig, CalicoST's exactly. Fails where a block is phased or a gene counted otherwise."""
    binned, _ = bins_of(blocks, block_counts, 200, referee)
    phase = np.random.default_rng(0).random(block_counts.n_segments) < 0.5
    ours = summarize_counts_for_bins(binned, easy["adata"], block_counts, phase)
    x, total = get_stacked(block_counts), block_counts.total_snp_umis.toarray().T

    def call() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        theirs = utils_IO.summarize_counts_for_bins(
            binned.copy(), easy["adata"], x, total, phase, 1.0, 0.0, easy["map"]
        )
        return theirs[0], theirs[1], theirs[3]

    inputs = (*get_easy_inputs(easy, binned), x, total, phase)
    theirs = referee("omics.bin_counts", inputs, call)
    assert tuple(map(get_digest, (ours.lengths, get_stacked(ours), ours.total_snp_umis.toarray().T))) == theirs


def get_stacked(counts: Any) -> np.ndarray:
    """CalicoST's `single_X`, `(n_obs, 2, n_spots)`: `umis` and `alt_snp_umis` stacked."""
    return np.stack([counts.umis.toarray().T, counts.alt_snp_umis.toarray().T], axis=1)


BIN_INFO_KEYS = ("bin_id", "CHR", "START", "END", "N_SNPS")


def get_bin_contents(info: pd.DataFrame, sep: str) -> list[list[list[str]]]:
    """Each bin's genes and SNP ids, sorted, from `info`'s `sep`-joined columns."""
    return [[sorted(set(x.split(sep)) - {""}) for x in info[key]] for key in ("INCLUDED_GENES", "INCLUDED_SNP_IDS")]


def test_the_bin_info_shares_calicost_s_columns(blocks: pd.DataFrame, block_counts: Any, referee: Referee) -> None:
    """Per bin, CHR, START, END, `N_SNPS`, and the genes and SNPs it holds: CalicoST's. Fails where a bin's span or
    contents differ."""
    binned, _ = bins_of(blocks, block_counts, 200, referee)
    ours = binned_gene_snp(binned)

    def call() -> tuple[Any, ...]:
        theirs = parse_input.genesnp_to_bininfo(binned)
        return *(theirs[key].to_numpy() for key in BIN_INFO_KEYS), get_bin_contents(theirs, " ")

    theirs = referee("omics.bin_info", (binned,), call)
    assert tuple(get_digest(ours[key].to_numpy()) for key in BIN_INFO_KEYS) == theirs[:-1]
    assert get_digest(get_bin_contents(ours, ",")) == theirs[-1]


# -- the departures, as equality with CalicoST ---------------------------------------------------------------------


def frame(rows: list[tuple[int, int, int, Any, Any, bool]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=COLUMNS)


def blocks_frame(lengths: list[int]) -> pd.DataFrame:
    """One block per row on contig 1, abutting, each `lengths[i]` long."""
    ends = np.cumsum(lengths)
    return pd.DataFrame({"CHR": 1, "START": ends - np.array(lengths), "END": ends, "block_id": range(len(lengths))})


def binned_both(lengths: list[int], umi: list[int], floor: int, max_binlength: float, referee: Referee) -> bool:
    """Whether `create_bin_ranges` is CalicoST's on one-spot blocks of `lengths` and SNP UMIs `umi`."""
    total = np.array(umi)[:, None]
    counts = get_labelled_counts(total, 0 * total, total)
    ours = create_bin_ranges(
        blocks_frame(lengths),
        None,
        None,
        None,
        None,
        counts,
        np.array([len(umi)]),
        0,
        floor,
        0,
        max_binlength=max_binlength,
    )

    def call() -> np.ndarray:
        theirs = utils_IO.create_bin_ranges(blocks_frame(lengths), total, np.array([len(umi)]), floor, max_binlength)
        return theirs.bin_id.to_numpy()

    inputs = (lengths, umi, floor, max_binlength)
    return (get_digest(ours.bin_id.to_numpy()),) == referee(f"omics.departure.bins{inputs}", inputs, call)


@departs(
    "a tail under the floor joins the previous bin in cnamaste; in CalicoST only "
    "under half the floor and half max_binlength"
)
def test_a_tail_under_the_floor_bins_as_calicost_s(referee: Referee) -> None:
    """UMIs 10, 10, 6 at floor 10: cnamaste's tail joins (two bins), CalicoST's stands (three)."""
    assert binned_both([1, 1, 1], [10, 10, 6], 10, 1e9, referee)


@departs("cnamaste caps a bin at max_binlength only once its floors are met; CalicoST caps every bin")
def test_max_binlength_caps_an_under_floor_run_as_calicost_s(referee: Referee) -> None:
    """Four blocks of length 3, max 5, the floor met only by the last: cnamaste one bin of 12, CalicoST four."""
    assert binned_both([3, 3, 3, 3], [1, 1, 1, 30], 10, 5, referee)


@departs("the tail and length rules above, on easy: 424 bins against CalicoST's 537 at SNP-UMI floor 200")
def test_easy_s_bins_are_calicost_s(blocks: pd.DataFrame, block_counts: Any, referee: Referee) -> None:
    """Easy's blocks at SNP-UMI floor 200, no read-UMI or normal floor (CalicoST has neither): CalicoST's bins."""
    ours, theirs = bins_of(blocks, block_counts, 200, referee)
    assert get_digest(ours.bin_id.to_numpy()) == theirs


@departs(
    "cnamaste searches num_preceeding_rows=100 rows back for a SNP's gene, "
    "CalicoST 50; equal on easy, where no gene holds more than 50 SNPs"
)
def test_a_gene_s_51st_snp_takes_it_as_calicost_s(tmp_path: Any, referee: Referee) -> None:
    """One gene and 60 SNPs inside it: CalicoST's rows, row for row. cnamaste's default keeps the gene and its 60
    SNPs, 61 rows; CalicoST the first 50, 51 rows."""
    hgtable = tmp_path / "hg.tsv"
    hgtable.write_text("#bin\tname2\tchrom\tcdsStart\tcdsEnd\n0\tg\tchr1\t100\t1000\n")
    ids = np.array([f"1_{200 + k}_A_C" for k in range(60)])
    adata = anndata.AnnData(X=np.zeros((1, 1)), var=pd.DataFrame(index=["g"]))
    ours = form_gene_snp_table(ids, str(hgtable), adata)

    def call() -> pd.DataFrame:
        return utils_IO.combine_gene_snps(ids, str(hgtable), adata)[COLUMNS].reset_index(drop=True)

    (theirs,) = referee("omics.departure.gene_51st_snp", (ids, hgtable, *get_adata_inputs(adata)), call)
    assert get_digest(ours[COLUMNS].reset_index(drop=True)) == theirs


@departs(
    "a SNP before its contig's first gene: CalicoST counts its UMIs in block 0; cnamaste refuses it (ValueError)",
    raises=ValueError,
)
def test_a_row_before_the_first_gene_blocks_as_calicost_s(referee: Referee) -> None:
    """A SNP before gene A, A and B each with a SNP: CalicoST adds that SNP's UMIs to block 0 and closes A alone (two
    blocks); cnamaste refuses the table rather than place UMIs in a block that does not hold the SNP. Unreached by a
    run: `form_gene_snp_table` drops a SNP in no gene."""
    table = frame(
        [
            (1, 10, 11, "1_10_A_C", None, False),
            (1, 100, 200, None, "gA", True),
            (1, 150, 151, "1_150_A_C", "gA", False),
            (1, 1000, 1100, None, "gB", True),
            (1, 1050, 1051, "1_1050_A_C", "gB", False),
        ]
    )
    ids = np.array(["1_10_A_C", "1_150_A_C", "1_1050_A_C"])
    a_counts, b_counts = np.array([[6, 5, 50]]), np.zeros((1, 3), dtype=int)
    adata = anndata.AnnData(X=np.zeros((1, 2)), var=pd.DataFrame(index=["gA", "gB"]))
    adata.layers["count"] = np.ones((1, 2), dtype=int)
    ours = assign_initial_blocks(table.copy(), adata, a_counts, b_counts, ids, 10)

    def call() -> np.ndarray:
        theirs = utils_IO.create_haplotype_block_ranges(table.copy(), adata, a_counts, b_counts, ids, 10)
        return theirs.block_id.to_numpy()

    inputs = (table, *get_adata_inputs(adata), a_counts, b_counts, ids, 10)
    assert (get_digest(ours.block_id.to_numpy()),) == referee("omics.departure.row_before_first_gene", inputs, call)


def get_sorted_contents(info: pd.DataFrame, sep: str) -> pd.DataFrame:
    """`info` with each bin's genes and SNP ids in sorted order, still `sep`-joined: both sides join a set, in an
    order that varies with the interpreter's hash seed."""
    contents = get_bin_contents(info, sep)
    return info.assign(
        INCLUDED_GENES=[sep.join(x) for x in contents[0]], INCLUDED_SNP_IDS=[sep.join(x) for x in contents[1]]
    )


@departs("binned_gene_snp joins genes and SNP ids with ',', genesnp_to_bininfo with ' ', and has no ARM column")
def test_the_bin_info_is_calicost_s_frame(blocks: pd.DataFrame, block_counts: Any, referee: Referee) -> None:
    """The whole frame, CalicoST's, each bin's contents in sorted order."""
    binned, _ = bins_of(blocks, block_counts, 200, referee)

    def call() -> pd.DataFrame:
        return get_sorted_contents(parse_input.genesnp_to_bininfo(binned), " ")

    ours = get_sorted_contents(binned_gene_snp(binned), ",")
    assert (get_digest(ours),) == referee("omics.departure.bin_info_frame", (binned,), call)


GZIPPED_INPUTS = ("barcodes.txt", "unique_snp_ids.npy", "spatial/tissue_positions_list.csv")
PLAIN_INPUTS = ("filtered_feature_bc_matrix.h5ad", "cell_snp_Aallele.npz", "cell_snp_Ballele.npz")
"""Easy's inputs to `load_data`, committed gzipped and plain."""


def get_plain_inputs(directory: Path) -> Path:
    """Easy's inputs in `directory`, uncompressed under the names CalicoST's `load_data` reads."""
    source = fixture("2d4ce9a9")
    (directory / "spatial").mkdir()
    for name in GZIPPED_INPUTS:
        (directory / name).write_bytes(gzip.decompress((source / f"{name}.gz").read_bytes()))
    for name in PLAIN_INPUTS:
        (directory / name).write_bytes((source / name).read_bytes())
    return directory


def get_loaded_inputs(adata: anndata.AnnData, a_allele: Any, b_allele: Any, snp_ids: Any) -> tuple[Any, ...]:
    """What a load holds: spots, the counts layer and the A and B counts as int64, each SNP's (contig, position), then
    the names, genes and SNP ids."""
    counts = [np.asarray(dense(v), dtype=np.int64) for v in (adata.layers["count"], a_allele, b_allele)]
    loci = [tuple(int(x) for x in str(snp_id).split("_")[:2]) for snp_id in snp_ids]
    return list(adata.obs_names), *counts, loci, list(adata.var_names), [str(x) for x in snp_ids]


@pytest.fixture(scope="module")
def load_digests(loaded: tuple[YAMLConfig, Any], tmp_path_factory: pytest.TempPathFactory, referee: Referee) -> Any:
    """`get_loaded_inputs`' digests, cnamaste's `load_input_data`'s and CalicoST's `load_data`'s on easy, from the
    cache. On a miss CalicoST holds both allele matrices dense, ~1 GB, 26 s; on a hit cnamaste's load alone, 10 s."""
    config, found = loaded
    references = (config.references.filtergenelist_file, config.references.filterregion_file)
    floors = (config.quality.spot_min_snp_umis, config.quality.min_percent_expressed_spots)

    def call() -> tuple[Any, ...]:
        directory = get_plain_inputs(tmp_path_factory.mktemp("load_data"))
        # NB CalicoST reads `csr_matrix.A`, which scipy removed, and `Series.nonzero` through scipy, which pandas removed.
        with (
            mock.patch.object(sp.csr_matrix, "A", property(lambda m: m.toarray()), create=True),
            mock.patch.object(pd.Series, "nonzero", lambda s: s.to_numpy().nonzero(), create=True),
        ):
            adata, a_allele, b_allele, snp_ids = utils_IO.load_data(
                str(directory), str(directory), *references, None, *floors
            )
        return get_loaded_inputs(adata, a_allele, b_allele, snp_ids)

    source = fixture("2d4ce9a9")
    files = [source / f"{name}.gz" for name in GZIPPED_INPUTS] + [source / name for name in PLAIN_INPUTS]
    theirs = referee("omics.load_data", (*files, *map(Path, references), *floors), call)
    ours = tuple(get_digest(v) for v in get_loaded_inputs(found[2], found[4], found[5], found[6]))
    return ours, theirs


@pytest.mark.slow
def test_load_input_data_is_calicost_s_load_data(load_digests: Any) -> None:
    """Easy's 3,000 spots, under the run's filters (gene list, HLA regions, 50 SNP UMIs per spot, genes in 0.5% of
    spots, the local-outlier gene filter): `load_input_data`'s spots, counts (3,000 x 7,232), A and B counts and SNP
    loci (13,333) are CalicoST's `load_data`'s, bitwise. Fails where a spot or gene filter, the barcode order or a
    count drifts."""
    ours, theirs = load_digests
    assert ours[:5] == theirs[:5]


@pytest.mark.slow
@departs("cnamaste makes duplicate gene names unique and renames SNP ids contig_position_enum; CalicoST keeps both")
def test_load_input_data_names_genes_and_snps_as_calicost_s_load_data(load_digests: Any) -> None:
    """The gene names (`LINC01505` twice in CalicoST's, `LINC01505-1` the second in cnamaste's) and the SNP ids
    (`6_8428968_N_N` against `6_8428968_0`)."""
    ours, theirs = load_digests
    assert ours[5:] == theirs[5:]
