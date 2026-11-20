"""The `.tsv.gz` tables `run_cnamaste` writes through `results.write_tsv`, from CalicoST easy (`2d4ce9a9`)'s capture.

Each table is its recorded `09_outputs/write_tsv_<k>` call replayed into a temporary directory and read back by
`io.read_text_table`, as `qa.metrics.read_estimate` reads it. Referees: the recorded in-memory frames, and the
relations the run states between them: one barcode set, one integer clone per fitted clone, and each segmentation
level a coarsening of its parent. The plain `.tsv` copy tables are `test_results.py`'s.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cnamaste.io import read_text_table
from cnamaste.qa.capture import Capture
from cnamaste.qa.sims import get_project, get_supported_sims
from cnamaste.results import get_output_file_meta, write_tsv
from cnamaste.segments import Segmentation, get_levels

COARSENS = (
    ("reference_genes", "initial_blocks"),
    ("initial_blocks", "phasing_segments"),
    ("initial_blocks", "phased_bins"),
    ("phased_bins", "phasing_segments"),
    ("phased_bins", "normal_filtered_bins"),
    ("normal_filtered_bins", "normal_rebinned_bins"),
)
"""`(finer, coarser)` levels: phasing segments group blocks, bins are drawn within phasing segments
(`refined_lengths`), the normal filter drops bins and the rebinning groups the kept ones."""

CALLS = {
    "baf_clone_labels.tsv.gz": 0,
    "clone_labels.tsv.gz": 1,
    "clone_labels_int.tsv.gz": 5,
    "segmentation.tsv.gz": 6,
    "snp_segmentation.tsv.gz": 7,
}
"""Each table's `09_outputs/write_tsv_<k>` call in the capture."""


@pytest.fixture(scope="module")
def written(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, tuple[pd.DataFrame, pd.DataFrame]]]:
    """Each table's recorded frame, index named as written, and the frame read back from the replayed write."""
    directory = tmp_path_factory.mktemp("outputs")
    capture = Capture(get_project() / get_supported_sims()["2d4ce9a9"])
    tables = {}
    try:
        for name, k in CALLS.items():
            function, args, kwargs = capture.decode(f"09_outputs/write_tsv_{k}")
            assert function is write_tsv and args[1] == name
            write_tsv(directory, *args[1:], **kwargs)
            frame = args[2].rename_axis(kwargs.get("index_label")) if kwargs["index"] else args[2]
            index_col = 0 if kwargs["index"] else None
            path = Path(directory) / name.removesuffix(".gz")
            tables[name] = (frame, read_text_table(path, sep="\t", index_col=index_col, float_precision="round_trip"))
    finally:
        capture.close()
    yield tables


@pytest.mark.parametrize("name", list(CALLS))
def test_each_table_reads_back_as_written_but_sample_id(
    written: dict[str, tuple[pd.DataFrame, pd.DataFrame]], name: str
) -> None:
    """The recorded frame has `get_output_file_meta`'s columns and dtypes, in order; read back, every column but
    `sample_id` equals it exactly, the `barcode` index included."""
    frame, found = written[name]
    columns = get_output_file_meta(name).columns
    assert dict(frame.dtypes.astype(str)) == columns
    assert list(found.columns) == list(columns)
    kept = [c for c in frame.columns if c != "sample_id"]
    pd.testing.assert_frame_equal(found[kept], frame[kept], check_exact=True)


@pytest.mark.xfail(strict=True, reason="T- #31: sample_id written as str ('0', '1', ...) reads back as int64")
def test_sample_id_reads_back_as_written(written: dict[str, tuple[pd.DataFrame, pd.DataFrame]]) -> None:
    """`sample_id` is a sample's name, a string, written and read alike."""
    frame, found = written["clone_labels.tsv.gz"]
    pd.testing.assert_series_equal(found["sample_id"], frame["sample_id"], check_exact=True)


def test_the_label_tables_carry_one_spot_set_and_one_integer_clone_per_clone(
    written: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
) -> None:
    """The 3,000 barcodes, `x`, `y` and `normal_candidate` (0 or 1) agree across the three label tables, as read;
    each fitted clone of `clone_labels` maps to one integer clone of `clone_labels_int`."""
    baf, clones, merged = (
        written[name][1] for name in ("baf_clone_labels.tsv.gz", "clone_labels.tsv.gz", "clone_labels_int.tsv.gz")
    )
    assert len(clones) == 3_000 and clones.index.is_unique
    shared = ["x", "y", "normal_candidate"]
    for other in (baf, merged):
        pd.testing.assert_frame_equal(other[shared], clones[shared], check_exact=True)
    assert set(clones["normal_candidate"]) <= {0, 1}
    assert (merged.groupby(clones["clone_label"])["clone_label"].nunique() == 1).all()


def test_each_segmentation_level_coarsens_its_parent(written: dict[str, tuple[pd.DataFrame, pd.DataFrame]]) -> None:
    """As read: genes in genome order, `START < END`, each its own reference segment; each level's labels run 0, 1,
    ... in that order, -1 on dropped genes; for each of `COARSENS`, a kept finer segment lies in one coarser segment,
    and a gene the finer level dropped the coarser drops."""
    found = written["segmentation.tsv.gz"][1]
    assert (found["START"] < found["END"]).all()
    assert found.sort_values(["CHR", "START"], kind="stable").index.equals(found.index)
    assert (found["reference_genes"].to_numpy() == np.arange(len(found))).all()
    for level in get_levels():
        kept = found[level].to_numpy()[found[level].to_numpy() >= 0]
        assert kept[0] == 0 and set(np.diff(kept)) <= {0, 1}, level
    for finer, coarser in COARSENS:
        within = found[found[finer] >= 0]
        assert (within.groupby(finer)[coarser].nunique() == 1).all(), (finer, coarser)
        assert (found.loc[found[finer] < 0, coarser] < 0).all(), (finer, coarser)


def test_each_segmentation_row_names_its_gene_and_each_bin_its_included_genes(
    written: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
) -> None:
    """As read: each gene row's `gene` has the row's contig, start and end in `ref/hgTables_hg38_gencode.txt`; each
    SNP lies in its `gene`, `START <= POS < END`, and takes that gene's `reference_genes` row; each
    `normal_rebinned_bins` bin's genes, in genome order, are its `INCLUDED_GENES` from `binned_gene_snp`
    (`07_rebin`). Fails where a name is shifted to a neighbouring row or bin, or a SNP is named for another gene."""
    genes, snps = written["segmentation.tsv.gz"][1], written["snp_segmentation.tsv.gz"][1]
    hgtable = pd.read_csv(get_project() / "ref" / "hgTables_hg38_gencode.txt", sep="\t", index_col=0)
    annotated = set(
        zip(hgtable["name2"], hgtable["chrom"].str.removeprefix("chr"), hgtable["cdsStart"], hgtable["cdsEnd"])
    )
    assert set(zip(genes["gene"], genes["CHR"].astype(str), genes["START"], genes["END"])) <= annotated
    assert snps["gene"].notna().all() and (snps["reference_genes"] >= 0).all()
    gene_of_snp = genes.iloc[snps["reference_genes"].to_numpy()]
    assert (gene_of_snp["gene"].to_numpy() == snps["gene"].to_numpy()).all()
    assert (gene_of_snp["CHR"].to_numpy() == snps["CHR"].to_numpy()).all()
    assert ((gene_of_snp["START"].to_numpy() <= snps["POS"]) & (snps["POS"] < gene_of_snp["END"].to_numpy())).all()

    capture = Capture(get_project() / get_supported_sims()["2d4ce9a9"])
    try:
        bins = capture.stored("07_rebin/binned_gene_snp/out")
    finally:
        capture.close()
    binned = genes[genes["normal_rebinned_bins"] >= 0].groupby("normal_rebinned_bins")["gene"]
    included = [",".join(dict.fromkeys(names)) for _, names in binned]
    assert included == bins["INCLUDED_GENES"].tolist()


def test_easy_s_rows_are_labelled_with_segmentation_tsv_s_spans_and_dev_s_lengths(
    written: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
) -> None:
    """Easy's blocks from `summarize_counts_for_blocks` replayed, then carried (`get_segment_labels`) to the
    `normal_rebinned_bins` the written table gives each block: at both levels, `contig`, `start` and `end` equal
    `segmentation.tsv.gz`'s `CHR`, first `START` and last `END` row for row, and `lengths` its rows per `CHR`. Fails
    where a row's span is read from another row or contig, or a contig's rows are miscounted."""
    genes = written["segmentation.tsv.gz"][1]
    capture = Capture(get_project() / get_supported_sims()["2d4ce9a9"])
    try:
        function, args, kwargs = capture.decode("02_blocks/summarize_counts_for_blocks")
    finally:
        capture.close()
    blocks = function(*args, **kwargs)
    kept = genes[(genes["initial_blocks"] >= 0) & (genes["normal_rebinned_bins"] >= 0)]
    pairs = kept[["initial_blocks", "normal_rebinned_bins"]].drop_duplicates().to_numpy().T
    rebinned = Segmentation.from_terms(blocks.n_segments, *pairs, int(pairs[1].max()) + 1, name="normal_rebinned_bins")
    bins = blocks.multi_counts.get_segmented(rebinned).get_labelled(*blocks.get_segment_labels(rebinned))
    for level, counts in zip(("initial_blocks", "normal_rebinned_bins"), (blocks, bins)):
        rows = genes[genes[level] >= 0].groupby(level)
        spans = np.stack([rows["CHR"].first(), rows["START"].first(), rows["END"].last()], axis=1)
        assert np.array_equal(np.stack([counts.contig, counts.start, counts.end], axis=1), spans), level
        contigs = spans[:, 0]
        lengths = np.diff(np.flatnonzero(np.r_[True, contigs[1:] != contigs[:-1], True]))
        assert np.array_equal(counts.lengths, lengths), level
