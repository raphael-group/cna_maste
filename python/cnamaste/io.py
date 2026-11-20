import contextlib
import copy
import gzip
import logging
import os
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import anndata
import numpy as np
import pandas as pd
import polars as pl
import scanpy as sc
import scipy.sparse as sp
from sklearn.neighbors import LocalOutlierFactor

from cnamaste.config import YAMLConfig, get_global_config, he_components, lattice_type, set_global_config, start_time
from cnamaste.he import get_he_image
from cnamaste.logger import get_logger
from cnamaste.normal_spot import determine_normal_baseline
from cnamaste.omics import assign_initial_blocks, form_gene_snp_table
from cnamaste.platforms.visium import physical_coords, reads_visium_hd, validate_platform_lattice
from cnamaste.quality_cuts import downsample_genes, get_filter_genes, get_filter_ranges, zero_genes
from cnamaste.reference import exp_cancer_gene
from cnamaste.results import run_config_name
from cnamaste.segments import Genes, Segmentation, get_levels
from cnamaste.spatio_genomic_counts import SpatioGenomicCounts, SpatioGenomicMultiCounts
from cnamaste.utils import find_plain_or_gz

logger = get_logger(__name__, start_time=start_time)

pl.Config.set_tbl_cols(-1)


def read_text_table(path, **kwargs):
    """A delimited text table (`.csv`, `.tsv`, `.txt`) at `path` or `path.gz`, by `pandas.read_csv`.

    pandas infers gzip from the `.gz` suffix, so the compressed file is read as the plain one.

    Args:
        path: the plain file's name.
        kwargs: to `pandas.read_csv`.

    Returns:
        The frame.

    Raises:
        FileNotFoundError: where neither exists.
    """
    found = find_plain_or_gz(path)
    if found is None:
        raise FileNotFoundError(f"{path} (or {path}.gz)")
    return pd.read_csv(found, **kwargs)


def read_numpy(path, **kwargs):
    """A NumPy array at `path` (`.npy`) or `path.gz`, by `numpy.load`, from a `gzip` stream where compressed.

    Args:
        path: the plain file's name.
        kwargs: to `numpy.load`.

    Returns:
        The array.

    Raises:
        FileNotFoundError: where neither exists.
    """
    found = find_plain_or_gz(path)
    if found is None:
        raise FileNotFoundError(f"{path} (or {path}.gz)")
    if found.suffix != ".gz":
        return np.load(found, **kwargs)
    with gzip.open(found, "rb") as stream:
        return np.load(stream, **kwargs)


def read_anndata_counts(path):
    """A count matrix as AnnData: 10x `.h5` by `scanpy.read_10x_h5`, `.h5ad` by `anndata.read_h5ad`.

    Neither is read gzipped: HDF5 needs random access, and both compress their own datasets.

    Args:
        path: the `.h5` or `.h5ad` file.

    Returns:
        The AnnData.
    """
    path = Path(path)
    return sc.read_10x_h5(path) if path.suffix == ".h5" else anndata.read_h5ad(path)


def get_assignment_frame(barcodes, coords, assignment, normal_candidate=None, field_argmax=None):
    """Construct a data frame with the assigned clone label for all spots.

    Args:
        barcodes: per spot.
        coords: (n_spots, 2).
        assignment: clone per spot.
        normal_candidate: per spot, `determine_normal_candidates`' call, or None.
        field_argmax: the label the field alone gives each spot (`hmrf.field_argmax`), or None.

    Returns:
        Frame of sample_id, x, y, clone_label, then normal_candidate (0 or 1) and field_argmax where given.
    """
    df_clone_label = pd.DataFrame(
        {
            "sample_id": [barcode.split("_")[-1] for barcode in barcodes],
            "x": coords[:, 0],
            "y": coords[:, 1],
            "clone_label": assignment,
        },
        index=barcodes,
    )

    if normal_candidate is not None:
        df_clone_label["normal_candidate"] = np.asarray(normal_candidate).astype(np.int64)

    # NB last, and only where a caller has it: the labelling with no spatial prior, beside the one that was solved.
    if field_argmax is not None:
        df_clone_label["field_argmax"] = np.asarray(field_argmax).astype(np.int64)

    df_clone_label = df_clone_label.sort_values(["sample_id", "x", "y"])

    return df_clone_label


def get_sample_sheet(sample_sheet_path):
    """Read the sample sheet, which gives `bam`, `sample_id`, `spaceranger_dir` and `snp_dir` per sample.

    Args:
        sample_sheet_path: the sample sheet file.

    Returns:
        The sample metadata frame.
    """
    df_meta = pd.read_csv(sample_sheet_path, sep=r"\s+", comment="#")

    required_columns = {"bam", "sample_id", "spaceranger_dir", "snp_dir"}

    # TODO
    assert required_columns.issubset(df_meta.columns), (
        f"sample_sheet has columns {df_meta.columns} which missese the required: {required_columns - set(df_meta.columns)}:\n{df_meta}"
    )

    logger.info(f"Input sample_sheet_path={sample_sheet_path} contains {len(df_meta)} samples:\n{df_meta}")

    return df_meta


def get_barcodes(barcode_file):
    found_file = None

    for ext in (".tsv.gz", ".tsv", ".txt", ".txt.gz"):
        candidate = barcode_file.replace(".txt", ext)

        if os.path.exists(candidate):
            found_file = candidate
            break

    if found_file is None:
        raise RuntimeError(
            f"Failed to retrieves barcodes.txt (or its known alternative extensions: ().tsv.gz, .tsv, .txt.gz) from {barcode_file})"
        )

    # NB spot bar-codes combined across slices (TBC).
    if found_file.endswith(".tsv.gz"):
        df_barcodes = pd.read_csv(
            found_file,
            sep="\t",
            header=None,
            names=["combined_barcode"],
            compression="gzip",
        )
    else:
        # NB default for .txt
        df_barcodes = pd.read_csv(found_file, header=None, names=["combined_barcode"])

    return df_barcodes


# TODO check (e.g. sample sheet with john):
#             AAACAAGTATCTCCCA-1_HT112C1-U1 == {spot}-1_{sample_id}-{slice}.
def get_aggregated_barcodes(barcode_file, known_sample_id=None):
    df_barcodes = get_barcodes(barcode_file)

    sample_id_defined = df_barcodes.combined_barcode.str.contains("_").all()

    # NB Visium 10x provided spot barcode for each slice.
    df_barcodes["barcode"] = [x.split("_")[0] for x in df_barcodes.combined_barcode.to_numpy()]

    # NB {sample_id}-{slice}
    if sample_id_defined:
        df_barcodes["sample_id"] = [x.split("_")[-1] for x in df_barcodes.combined_barcode.to_numpy()]
    else:
        logger.warning(
            f"Unable to resolve sample_ids from aggregated barcodes.  Assuming known sample_id={known_sample_id}."
        )
        df_barcodes["sample_id"] = known_sample_id or "UNKNOWN"

    # TODO HACK assumes {sample_id}-{slice} is not provided, i.e. single slice.
    df_barcodes["barcode"] = df_barcodes.combined_barcode
    df_barcodes["sample_id"] = known_sample_id

    # TODO sample ids currently slice, e.g. U1;
    logger.info(
        f"Input aggregated barcode file {barcode_file} with {df_barcodes.shape[0]:_} barcodes for all samples/bams, e.g.\n{df_barcodes.head()}\n"
    )

    return df_barcodes


def get_spatial_positions(spaceranger_dir, filter_in_tissue=True):
    # TODO x,y vs row,col?  sub-pixel position?
    names = ("barcode", "in_tissue", "x", "y", "pixel_row", "pixel_col")

    # NB Visium HD first: a binned output may carry a Visium-style CSV beside its parquet (`reads_visium_hd`).
    if reads_visium_hd(spaceranger_dir):
        # NB 11,222,500 rows vs 4,992 rows for visium HD.
        #    see https://www.10xgenomics.com/support/software/space-ranger/latest/analysis/outputs/spatial-outputs
        #
        #
        #    native columns :  barcode, in_tissue, array_row, array_col, pxl_row_in_fullres, pxl_col_in_fullres.
        #
        #    visium HD : x2 6.5mm capture areas, array_row = row coordinate of the spot in the array from 0 to 77.
        #                                        array_col = given orange crate [sic], i.e. hexagonal, for 6.55 this uses even numbers from 0 to 126 for even rows
        #                                                   and odd numbers from 1 to 127 for odd rows.
        #                                        pxl_row_in_fullres: The row pixel coordinate of the center of the spot in the full resolution image
        #                                        pxl_col_in_fullres: The column pixel coordinate of the center of the spot in the full resolution image. This is the x coordinate in pixel space.
        """
        df_this_pos = (
            pl.scan_parquet(f"{spaceranger_dir}/spatial/tissue_positions.parquet")
            .rename({"pxl_row_in_fullres": "y", "pxl_col_in_fullres": "x"})
            .select(["barcode", "in_tissue", "x", "y"])
            # .filter(pl.col("in_tissue") == True)
            .collect()
            .with_columns(pl.col("barcode").alias("square_002um"))
        )

        # NB native columns: square_002um, square_008um, square_016um, cell_id, in_nucleus, in_cell
        # TODO CHECK in_cell
        df_this_pos = (
            pl.scan_parquet(f"{spaceranger_dir}/spatial/barcode_mappings.parquet")
            .filter(pl.col("in_cell") == True)
            .collect()
            .join(
                df_this_pos.select(["square_002um", "x", "y", "in_tissue"]),
                on="square_002um",
                how="left",
            )
            .select(["square_002um", "cell_id", "x", "y", "in_tissue"])
            .group_by("cell_id")
            .agg(
                [
                    pl.col("x").mean().alias("x"),
                    pl.col("y").mean().alias("y"),
                    pl.col("square_002um").n_unique().alias("num_square_002um"),
                    pl.col("in_tissue").cast(pl.Boolean).any().alias("in_tissue"),
                ]
            )
            .with_columns(pl.col("cell_id").alias("barcode"))
        )
        """
        # NB x is pxl_col_in_fullres, increasing left to right, and y pxl_row_in_fullres, increasing top to bottom: the
        #    image's sense, opposite to a plot's y. Kept as read, in pixels, as `coords`; `physical_coords` turns them
        #    to um with y up (`platforms.visium.pixel_physical`), and the kNN and every spatial page read those.
        df_this_pos = (
            pl.scan_parquet(f"{spaceranger_dir}/spatial/tissue_positions.parquet")
            .rename({"pxl_row_in_fullres": "y", "pxl_col_in_fullres": "x"})
            .filter(pl.col("in_tissue") == True)
            .collect()
        )

        logger.info(f"Read {spaceranger_dir}/spatial/tissue_positions.parquet:\n{df_this_pos}")

        df_this_pos = df_this_pos.to_pandas()
    elif find_plain_or_gz(f"{spaceranger_dir}/spatial/tissue_positions.csv"):
        df_this_pos = read_text_table(
            f"{spaceranger_dir}/spatial/tissue_positions.csv",
            sep=",",
            header=0,
            names=names,
        )

        logger.info(f"Reading {spaceranger_dir}/spatial/tissue_positions.csv")

    elif find_plain_or_gz(f"{spaceranger_dir}/spatial/tissue_positions_list.csv"):
        df_this_pos = read_text_table(
            f"{spaceranger_dir}/spatial/tissue_positions_list.csv",
            sep=",",
            header=None,
            names=names,
        )

        logger.info(f"Reading {spaceranger_dir}/spatial/tissue_positions_list.csv")

    else:
        logger.error(f"No spatial coordinate file @ {spaceranger_dir}.")
        raise RuntimeError()

    # TODO alignment defined for in_tissue == True only?
    if filter_in_tissue:
        logger.warning(
            f"Filtering spatial positions to in_tissue == True (retained {100.0 * np.mean(df_this_pos.in_tissue)}%)."
        )

        result = df_this_pos[df_this_pos.in_tissue == True]
    else:
        result = df_this_pos

    # NB x,y positions for each barcode in this sample.
    return result


def get_spaceranger_counts(spaceranger_dir, *, sparse=False):
    """Read the spaceranger (spot, gene) UMIs, with an integer `count` layer.

    `sparse` keeps the layer in the format of `X`, the stored values cast as the dense layer's are; `load_input_data`
    filters before densifying. A dense `X` raises `AttributeError` either way.

    Args:
        spaceranger_dir: the spaceranger output.
        sparse: keeps the `count` layer sparse.

    Returns:
        The AnnData.
    """
    config = get_global_config()
    filtered_feature_name = config.visium.filtered_feature_name

    supported_types = ["filtered_feature_bc_matrix", "filtered_feature_cell_matrix"]

    if filtered_feature_name not in supported_types:
        logger.error(f"{filtered_feature_name} is not supported; expected one  of {supported_types}")
        raise ValueError()

    # NB see https://scanpy.readthedocs.io/en/stable/generated/scanpy.read_10x_h5.html
    if Path(f"{spaceranger_dir}/{filtered_feature_name}.h5").exists():
        adatatmp = read_anndata_counts(f"{spaceranger_dir}/{filtered_feature_name}.h5")
        logger.info(f"Reading {spaceranger_dir}/{filtered_feature_name}.h5")

    elif Path(f"{spaceranger_dir}/{filtered_feature_name}.h5ad").exists():
        adatatmp = read_anndata_counts(f"{spaceranger_dir}/{filtered_feature_name}.h5ad")
        logger.info(f"Reading {spaceranger_dir}/{filtered_feature_name}.h5ad")

    else:
        logging.error(f"{spaceranger_dir} directory does not have a {filtered_feature_name}.h5(ad)!")
        raise RuntimeError()

    # TODO comment on adatatmp.x (nobs x nvars for space ranger, i.e. barcodes x gene transcripts).
    if sparse:
        # NB NaN is never implicit, so the stored values carry every NaN.
        counts = adatatmp.X.copy() if sp.issparse(adatatmp.X) else adatatmp.X.toarray()
        is_nan = np.isnan(counts.data)

        logger.info(f"Found {100.0 * np.sum(is_nan) / np.prod(counts.shape):.3f}% NaN counts in anndata.")

        counts.data = np.where(is_nan, 0, counts.data).astype(int)
        adatatmp.layers["count"] = counts
    else:
        adatatmp.layers["count"] = adatatmp.X.toarray()

        is_nan = np.isnan(adatatmp.layers["count"])

        logger.info(f"Found {100.0 * np.mean(is_nan):.3f}% NaN counts in anndata.")

        # NB replace nan with 0 and cast to int.
        if np.any(is_nan):
            adatatmp.layers["count"][is_nan] = 0

        # TODO CHECK
        adatatmp.layers["count"] = adatatmp.layers["count"].astype(int)

    logger.info(f"Read transcript counts of shape {adatatmp.shape}, i.e. (barcodes, genes) from {spaceranger_dir}")

    logger.info(f"Example names for {len(adatatmp.obs_names):_} barcodes:\n{adatatmp.obs_names[:5]}")
    logger.info(f"Example names for {len(adatatmp.var_names):_} genes:\n{adatatmp.var_names[:5]}")

    # NB var names made unique by appending an index string,
    #    see https://anndata.readthedocs.io/en/latest/generated/anndata.AnnData.var_names_make_unique.html
    adatatmp.var_names_make_unique()

    # NB data matrix X (ndarray/csr matrix, dask ...): observations/cells are named by their barcode and variables/genes by gene name.
    return adatatmp


def map_unique_snps_enum(unique_snp_ids):
    """Map each `{contig}_{pos}_{r}_{a}` SNP id (r = a = 'N' marks unknown) to a unique `{contig}_{pos}_{enum}`.

    `enum` allows for erroneous repeats, and is 0 in almost all cases.

    Args:
        unique_snp_ids: the SNP ids.

    Returns:
        The enumerated ids.
    """
    # NB log the number of unique snps and warn on any repeats
    bonafide_unique_snps, cnts = np.unique(unique_snp_ids, return_counts=True)
    logger.info(
        f"Detected {len(bonafide_unique_snps)} unique snps from {len(unique_snp_ids)} input snp ids with dtype={unique_snp_ids.dtype}."
    )

    repeats = dict()

    if len(bonafide_unique_snps) != len(unique_snp_ids):
        for snp_id, count in zip(bonafide_unique_snps[cnts > 1], cnts[cnts > 1]):
            contig, pos, _, _ = snp_id.split("_")
            logger.warning(f"Detected repeated snp_id @ chr{contig}:{pos} with count={count}.")
            repeats[snp_id] = 0

    result = []

    for snp_id in unique_snp_ids:
        contig, pos, *_ = snp_id.split("_")

        if snp_id in repeats:
            enum = repeats[snp_id]
            repeats[snp_id] += 1
        else:
            enum = 0

        # TODO HACK JOHN
        contig = contig.replace("chr", "")

        new_snp_id = f"{contig}_{pos}_{enum}"
        result.append(new_snp_id)

    result = np.array(result, dtype=unique_snp_ids.dtype)

    logger.info(f"Mapped input snp ids to enum:\n{result[:5]}")

    return result


def load_input_data(
    config,
    filter_gene_file=None,
    filter_range_file=None,
    normal_idx_file=None,
    min_snp_umis=50,
    min_percent_expressed_spots=5.0e-3,  # BUG actually a fraction.
):
    """Read a sample's SNP and transcript counts, aligned by spot and filtered.

    The counts layer and allele matrices are dense. For cost, counts
    stay sparse until spots and genes are filtered, `adata` is actual rather than a view, and the range filter is a
    `searchsorted` over SNPs in genomic order.

    Args: `config`; the optional `*_file` filters; `min_snp_umis`, `min_percent_expressed_spots` (a fraction) floors.

    Returns:
        `ProcessedData`, `physical_coords` last: the spots' `(x, y)` in um under `hmrf.lattice_type`.
    """
    df_meta = get_sample_sheet(config.paths.sample_sheet)

    # NB assumes snps derived from aggregation of all provided samples,
    assert np.all(df_meta["snp_dir"] == df_meta["snp_dir"].iloc[0])

    # NB (phased) SNPs are determined for the pseudobulk of all spots.
    snp_dir = df_meta["snp_dir"].iloc[0]

    # TODO sample_id not defined?  barcodes uniquely identify each spot per slice,
    #      aggregated across slices/bams.
    known_sample_id = df_meta.sample_id[0] if len(df_meta) == 1 else None
    df_agg_barcode = get_aggregated_barcodes(f"{snp_dir}/barcodes.txt", known_sample_id)

    # TODO duplicate of df_agg_barcode
    # NB dataframe of combined barcodes, i.e. Visium barcode + slice 'sample_id'.
    snp_barcodes = get_barcodes(f"{snp_dir}/barcodes.txt").rename(
        columns={"combined_barcode": "barcodes"},
        errors="raise",
    )

    # TODO HACK JOHN
    snp_barcodes["barcodes"] = snp_barcodes["barcodes"].map(lambda xx: xx.replace("_U1", ""))

    unique_snp_ids = read_numpy(f"{snp_dir}/unique_snp_ids.npy", allow_pickle=True)
    unique_snp_ids = map_unique_snps_enum(unique_snp_ids)

    # NB read (phased) counts for H0/H1 for (spots, snps), inflated on two threads.
    with ThreadPoolExecutor(max_workers=2) as pool:
        cell_snp_Aallele, cell_snp_Ballele = pool.map(
            lambda allele: sp.load_npz(f"{snp_dir}/cell_snp_{allele}allele.npz").tocsr(),
            ("A", "B"),
        )

    # TODO HACK JOHN
    # cell_snp_Aallele = cell_snp_Aallele.T
    # cell_snp_Ballele = cell_snp_Ballele.T

    assert cell_snp_Aallele.shape == cell_snp_Ballele.shape

    # NB per-spot snp-umis, retained for the spot filter below.
    snp_umis = np.asarray(cell_snp_Aallele.sum(axis=1)).ravel() + np.asarray(cell_snp_Ballele.sum(axis=1)).ravel()

    logger.info(
        f"Read cell-snp A,B matrices of shape={cell_snp_Aallele.shape} with min={snp_umis.min()}, max={snp_umis.max()}, median={np.median(snp_umis)} snp-umis per cell.  Found {np.sum(snp_umis):_} snp-umis total."
    )

    # NB read Visium transcripts/UMIs anndata & spot spatial coordinate.
    adata = None

    # NB df_meta provides the sample_ids, one per bam.
    for i, sname in enumerate(df_meta.sample_id.to_numpy()):
        logger.info(f"Reading (spot, gene) UMIs for spaceranger sample={sname}.")

        index = np.where(df_agg_barcode["sample_id"] == sname)[0]

        logger.info(f"Found {len(index)}/{len(df_agg_barcode)} matches by sample_id.")

        # NB indexed spot barcodes for this sample/slice.
        df_this_barcode = copy.copy(df_agg_barcode.iloc[index, :])
        df_this_barcode.index = df_this_barcode.barcode

        # NB (x,y) positions for each barcode (one per row).  limited to "in tissue" by default.
        validate_platform_lattice(df_meta["spaceranger_dir"].iloc[i], lattice_type(config))
        df_this_pos = get_spatial_positions(df_meta["spaceranger_dir"].iloc[i])

        # NB adds H&E derived features if available, in he.n_components classes.
        df_this_pos = get_he_image(
            df_meta["spaceranger_dir"].iloc[i], pos=df_this_pos, num_labels=he_components(config)
        )

        # NB read filtered_feature_bc_matrix.h5(ad) from spaceranger_dir for this sample - UMIs (spot barcode, gene).
        #    sparse until spots and genes are filtered.
        adatatmp = get_spaceranger_counts(df_meta["spaceranger_dir"].iloc[i], sparse=True)

        # NB re-order anndata spots to have the order of "df_this_barcode" (with enum).
        idx_argsort = pd.Categorical(
            adatatmp.obs.index, categories=list(df_this_barcode.barcode), ordered=True
        ).argsort()

        if not np.array_equal(idx_argsort, np.arange(len(idx_argsort))):
            logger.info("Sorting ST data by barcode.")
            adatatmp = adatatmp[idx_argsort, :].copy()

        # NB only keep shared barcodes between (IN_TISSUE) visium barcodes and filtered_feature_bc_matrix.
        pos_barcodes = set(list(df_this_pos.barcode))
        count_barcodes = set(list(adatatmp.obs.index))

        shared_barcodes = pos_barcodes & count_barcodes

        isin = adatatmp.obs.index.isin(shared_barcodes)

        logger.info(f"Retaining {100.0 * np.mean(isin):.3f}% of spots based on (in-tissue) position and UMIs.")

        # TODO visium hd.
        if not isin.all():
            adatatmp = adatatmp[isin, :].copy()

        df_this_pos = df_this_pos[df_this_pos.barcode.isin(shared_barcodes)]

        # NB re-order positions to have order of df_this_barcode barcodes.
        df_this_pos.barcode = pd.Categorical(df_this_pos.barcode, categories=list(adatatmp.obs.index), ordered=True)

        df_this_pos.sort_values(by="barcode", inplace=True)

        # NEW
        adatatmp.obsm["X_pos"] = np.vstack([df_this_pos.x, df_this_pos.y]).T

        if "gray" in df_this_pos.columns:
            adatatmp.obsm["he_gray"] = df_this_pos.gray.to_numpy()

        # TODO he_label
        if "label" in df_this_pos.columns:
            adatatmp.obsm["he_label"] = df_this_pos.label.to_numpy()

        adatatmp.obs["sample"] = sname

        # TODO HACK
        adatatmp.obs.index = [f"{x}" for x in adatatmp.obs.index]

        # NB concatenate across samples.
        adata = adatatmp if adata is None else anndata.concat([adata, adatatmp], join="outer")

    # NB filter by spots:  shared barcodes between adata and SNPs; e.g. drop spots with SNP counts but no transcripts.
    shared_barcodes = set(list(snp_barcodes.barcodes)) & set(list(adata.obs.index))

    isin = snp_barcodes.barcodes.isin(shared_barcodes).to_numpy()

    logger.info(f"Retaining {100.0 * np.mean(isin):.3f}% of barcodes with snp calls (shared between umis and snps).")

    # TODO barcode inconsistent between snps and umis.
    assert np.any(isin), (
        f"Found inconsistent barcodes between SNPs and UMIs, e.g. \n{list(snp_barcodes.barcodes)[:5]}\nvs\n{list(adata.obs.index)[:5]}"
    )

    # NB barcode (row) selection.
    if not isin.all():
        cell_snp_Aallele = cell_snp_Aallele[isin, :]
        cell_snp_Ballele = cell_snp_Ballele[isin, :]

        snp_barcodes = snp_barcodes[isin]
        snp_umis = snp_umis[isin]

    isin = adata.obs.index.isin(shared_barcodes)

    logger.info(f"Retaining {100.0 * np.mean(isin):.3f}% of umi barcodes (shared between umis and snps).")

    if not isin.all():
        adata = adata[isin, :].copy()

    idx_argsort = pd.Categorical(adata.obs.index, categories=list(snp_barcodes.barcodes), ordered=True).argsort()

    if not np.array_equal(idx_argsort, np.arange(len(idx_argsort))):
        logger.info("Sorting data by barcode.")
        adata = adata[idx_argsort, :]

    # DEPRECATE across-slice alignments: slices are joined by no alignment, so their spots are adjacent only within
    #    a slice.
    across_slice_adjacency_mat = None

    # NB filter out spots with too small number of UMIs (genome wide);
    # TODO differentiate min_snpumis; why before genomic binning?
    spot_umis = np.asarray(adata.layers["count"].sum(axis=1)).ravel()
    indicator = spot_umis >= min_snp_umis

    logger.info(f"Retaining {100.0 * np.mean(indicator):.3f}% of spots with sufficient umis (>= {min_snp_umis}).")

    # NB retain barcodes with sufficient SNP covering UMIs per spot.
    indicator &= snp_umis >= min_snp_umis

    logger.info(f"Retaining {100.0 * np.mean(indicator):.3f}% of spots with sufficient snp-umis (>= {min_snp_umis}).")

    adata = adata[indicator, :]

    cell_snp_Aallele = cell_snp_Aallele[indicator, :]
    cell_snp_Ballele = cell_snp_Ballele[indicator, :]

    spot_umis = spot_umis[indicator]

    # TODO HACK
    logger.info(f"Found total umi = {np.sum(spot_umis):_} for input.")

    percentiles = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
    perc_vals = np.percentile(spot_umis, percentiles)

    pairs = "\n".join(f"{p:.3f} [%]\t{v:_.0f}" for p, v in zip(perc_vals, percentiles))

    logger.info(f"Per-spot umi percentiles:\n{pairs}")

    # NB a view's X is sliced from its parent on each access.
    X = adata.X

    # NB number of barcodes expressing a particular gene; num. spots.  the stored
    #    entries alone, where each is positive and none repeats.
    if X.has_canonical_format and (X.data > 0).all():
        expressed = X.getnnz(axis=0)
    else:
        expressed = np.asarray((X > 0).sum(axis=0)).ravel()

    # NB filter out genes that are expressed in < min_percent_expressed_spots spots.
    indicator = expressed >= min_percent_expressed_spots * adata.shape[0]

    # NB ratio of total UMIs across all spots for gene selection vs all.
    gene_x = np.asarray(X.sum(axis=0)).ravel()
    ratio = np.sum(gene_x[indicator]) / np.sum(gene_x)

    # TODO gencode gene list is not all sampled by (3') visium umis?
    # TODO excludes 50% of genes, but retains 99.97% of UMIs; resolves gene definition to house-keeping?
    #
    # NB removes genes expressed in too few spots.
    logger.info(
        f"Retaining {100.0 * np.mean(indicator):.3f}% of genes ({100.0 * ratio:.2f}% of total umis) with sufficient expression across spots @ {min_percent_expressed_spots} fraction of spots."
    )

    adata = adata[:, indicator]

    logger.info(
        f"Median spot umi after filtering genes based on num. spots expressed = {np.median(np.asarray(adata.layers['count'].sum(axis=1))):_.3f}"
    )

    if filter_gene_file is not None:
        genes_to_filter = get_filter_genes(filter_gene_file).iloc[:, 0].to_numpy()
        indicator_filter = ~np.isin(adata.var.index, genes_to_filter)

        logger.info(f"Removing {len(filter_gene_file)} genes based on input file={filter_gene_file}.")

        adata = adata[:, indicator_filter]

        logger.info(
            f"Median spot umi after filtering genes = {np.median(np.asarray(adata.layers['count'].sum(axis=1))):_.3f}"
        )

        # TODO?
        # apply ranges cut to cell_snp_Aallele, cell_snp_Ballele, unique_snp_ids?

    # NB the counts layer as CSR, built once spots and genes are filtered: 2.1% nonzero on CalicoST easy (2d4ce9a9), 174
    #    MB dense against 3.4 MB. Every consumer sums integer counts over spots or genes, which CSR does exactly. An
    #    actual AnnData, as a view re-slices its layer from the parent on each access.
    counts = sp.csr_matrix(adata.layers["count"])

    adata = adata.copy()
    adata.layers["count"] = counts

    if filter_range_file is not None:
        ranges = get_filter_ranges(filter_range_file)
        num_ranges = ranges.shape[0]

        this_chr = np.array([int(x.split("_")[0]) for x in unique_snp_ids], dtype=np.int64)
        this_pos = np.array([int(x.split("_")[1]) for x in unique_snp_ids], dtype=np.int64)

        chrs = ranges.Chr.to_numpy().astype(np.int64)
        starts = ranges.Start.to_numpy().astype(np.int64)
        ends = ranges.End.to_numpy().astype(np.int64)

        # NB (Chr, End) and (chr, pos) as one integer each, ordered as the pairs.
        scale = 1 + int(max(ends.max(initial=0), this_pos.max(initial=0)))
        snp_key = this_chr * scale + this_pos
        range_key = chrs * scale + ends

        # NB j, the first range not before the snp, fast forwarded in genomic order:
        #    the first running maximum of range_key above the snp's key.
        if np.all(snp_key[1:] >= snp_key[:-1]):
            j = np.searchsorted(np.maximum.accumulate(range_key), snp_key, side="right")
        else:
            j = np.empty(len(snp_key), dtype=np.int64)
            k = 0

            for i, key in enumerate(snp_key):
                while k < num_ranges and range_key[k] <= key:
                    k += 1

                j[i] = k

        at = np.minimum(j, num_ranges - 1)

        # NB defaults to retain all SNP counts, excluded based on filter_range_file.
        indicator_filter = ~(
            (j < num_ranges) & (chrs[at] == this_chr) & (starts[at] <= this_pos) & (ends[at] > this_pos)
        )

        logger.info(f"Retaining {100.0 * np.mean(indicator_filter):.2f}% of snps based on input filter ranges.")

        cell_snp_Aallele = cell_snp_Aallele[:, indicator_filter]
        cell_snp_Ballele = cell_snp_Ballele[:, indicator_filter]

        unique_snp_ids = unique_snp_ids[indicator_filter]

    if config.quality.local_outlier_filter:
        # NB  k-NN defined density estimates used to filter local outliers given density wrt neighbors.
        #     see https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html
        #         https://en.wikipedia.org/wiki/Local_outlier_factor
        clf = LocalOutlierFactor(n_neighbors=200)

        # NB  prediction on barcode-summed transcripts for each gene; i.e. outlier gene detection.
        label = clf.fit_predict(np.asarray(adata.layers["count"].sum(axis=0)).reshape(-1, 1))

        to_zero = np.where(label == -1)[0]

        # NB ratio of total UMIs across all spots for genes-to-be-nulled vs all.
        ratio = adata.layers["count"][:, to_zero].sum() / adata.layers["count"].sum()

        # NB removed 235 outlier genes (51.310% of UMIs)!!  for both "normal" and "tumor" spots.
        logger.info(
            f"Removed {len(to_zero)} outlier genes ({100.0 * ratio:.3f}% of umis) based on {clf.__class__.__name__}."
        )

        if len(to_zero) > 0:
            # NB barcode summed counts per gene.
            gene_umi_counts = np.asarray(adata.layers["count"].sum(axis=0)).ravel()
            total_umis = adata.layers["count"].sum()

            outlier_genes_info = []

            for gene_idx in to_zero:
                gene_name = adata.var.index[gene_idx]
                gene_umis = gene_umi_counts[gene_idx]
                gene_pct = 100.0 * gene_umis / total_umis
                outlier_genes_info.append((gene_name, gene_umis, gene_pct))

            outlier_genes_info.sort(key=lambda x: x[2], reverse=True)

            # TODO log chr, start, end.
            logger.info("Top 25 outlier genes removed:")

            for i, (gene_name, gene_umis, gene_pct) in enumerate(outlier_genes_info[:25]):
                # NB altered mitochondrial metabolism (Warburg effect) in cancer;
                warning = "WARNING known to be cancerous" if exp_cancer_gene(gene_name) else ""

                logger.info(f"  {i + 1:2d}. {gene_name:<20} {gene_pct:6.3f}% UMIs {warning}")

        # TODO copy? ... ImplicitModificationWarning: Trying to modify attribute `.layers` of view, initializing view as actual.
        # NB  zero count of outlier genes (!)  Should retain snp-umi counts ...
        adata.layers["count"] = zero_genes(adata.layers["count"], np.isin(np.arange(adata.shape[1]), to_zero))

    # DEPRECATE quality.normalize_gene_outliers: the arm no configuration reaches. Both shipped
    #    configurations set it False, and config.yaml sets local_outlier_filter True, so the branch above wins
    #    whatever it holds; the scaling it applied was never measured on a run. Refused rather than ported.

    # NB quality.max_umi_per_gene: every gene whose total read-depth count exceeds it thinned to it, binomially per
    #    spot, the SNP counts untouched; None, the default, keeps every count. Off by default: 6,500 in place of the
    #    local outlier filter lifts copy ARI on easy (2d4ce9a9) and dense (33e3471e) and merges harder (a9a68913)'s
    #    three tumour clones into one, clone ARI 0.9781 to 0.3528.
    max_umi_per_gene = config.quality.max_umi_per_gene
    if str(max_umi_per_gene).lower() != "none":
        cap = float(max_umi_per_gene)
        totals = np.asarray(adata.layers["count"].sum(axis=0)).ravel()
        adata.layers["count"] = downsample_genes(adata.layers["count"], cap)
        logger.info(
            f"quality.max_umi_per_gene={cap:.0f}: {int((totals > cap).sum())} of {totals.size} genes thinned to it per "
            f"spot, {100.0 * float(adata.layers['count'].sum()) / totals.sum():.3f}% of read-depth UMIs kept."
        )

    if normal_idx_file is not None:
        normal_barcodes = pd.read_csv(normal_idx_file, header=None).iloc[:, 0].to_numpy()

        # NB column with tumor/normal designation.
        adata.obs["tumor_annotation"] = "tumor"
        adata.obs["tumor_annotation"][adata.obs.index.isin(normal_barcodes)] = "normal"

        logger.info("Applied tumor annotation: {adata.obs['tumor_annotation'].value_counts()}")

    logger.info(f"Realized AnnData:\n{adata}")

    # NB barcode consistency
    assert adata.layers["count"].shape[0] == cell_snp_Aallele.shape[0]
    assert cell_snp_Aallele.shape[0] == cell_snp_Ballele.shape[0]

    # NB SNP consistency; 17_797 anndata genes vs 16_681 SNPs.
    assert len(unique_snp_ids) == cell_snp_Aallele.shape[1]
    assert cell_snp_Aallele.shape[1] == cell_snp_Ballele.shape[1]

    # DEPRECATE?
    ProcessedData = namedtuple(
        "ProcessedData",
        [
            "coords",
            "barcodes",
            "adata",
            "exp_counts",
            "cell_snp_Aallele",
            "cell_snp_Ballele",
            "unique_snp_ids",
            "across_slice_adjacency_mat",
            "physical_coords",
        ],
    )

    # NB (x,y) per spot.
    coords = adata.obsm["X_pos"]

    # NB the spots' physical positions, um, x left to right and y up (`physical_coords`): what the adjacency's kNN and
    #    every spatial page read. `coords` keep the reader's meaning, array indices or Visium HD's pixels.
    sample_list, sample_ids = get_sample_list(adata)
    slice_dirs = df_meta.set_index("sample_id")["spaceranger_dir"]
    physical = physical_coords(
        coords, sample_ids, [slice_dirs.loc[s] for s in sample_list], lattice=lattice_type(config)
    )

    # NB e.g. 'AAACAAGTATCTCCCA-1_HT112C1-U1' currently.
    barcodes = adata.obs.index

    # NB sparse transcript counts (spot, gene).
    exp_counts = pd.DataFrame.sparse.from_spmatrix(
        sp.csc_matrix(adata.layers["count"]),
        index=adata.obs.index,
        columns=adata.var.index,
    )

    # NB the allele counts stay CSR (1.6% nonzero on CalicoST easy, 320 MB each dense): every consumer sums them over
    #    spots or sites, which CSR does exactly.
    result = ProcessedData(
        coords,
        barcodes,
        adata,
        exp_counts,
        cell_snp_Aallele,
        cell_snp_Ballele,
        unique_snp_ids,
        across_slice_adjacency_mat,
        physical,
    )

    return result


def get_sample_list(adata):
    sample_list = [adata.obs["sample"].iloc[0]]

    # NB loop through rows (barcodes x samples) and collect sample names;
    #    assumes sorted by sample and is unique in this case.
    for i in range(1, adata.shape[0]):
        if adata.obs["sample"].iloc[i] != sample_list[-1]:
            logger.warning(f"Appending sample_id={adata.obs['sample'].iloc[i]} to sample list.")
            sample_list.append(adata.obs["sample"].iloc[i])

    # NB e.g. HT112C1-U1.
    logger.info(f"Found {len(sample_list)} unique samples:\n{sample_list}")

    # NB array: assigns to each transcript row (barcode x sample) unique index according to sample names.
    sample_ids = -np.ones(adata.shape[0], dtype=int)

    for s, sname in enumerate(sample_list):
        index = np.where(adata.obs["sample"] == sname)[0]
        sample_ids[index] = s

    assert np.all(sample_ids >= 0), "Failed to assign unique integer to all samples in list. Bug?"

    return sample_list, sample_ids


def get_labelled_counts(input_dir: str | Path, output_dir: str | Path) -> Any:
    """Return a run's final counts, read back from its inputs and its written files.

    `umis` are the genes' UMIs summed by `segmentation.tsv.gz`'s last level; `alt_snp_umis` and `total_snp_umis`
    the SNPs' allele counts summed by `snp_segmentation.tsv.gz`'s, the alternate allele flipped where `phase` is 0.
    The bins under `quality.min_normal_count_perbin` are zeroed and the spots labelled `normal_spot` and `clone` from
    `clone_labels.tsv.gz`, as `run_cnamaste` holds them once its normal baseline is set.

    Args:
        input_dir: the directory the run's relative paths resolve against.
        output_dir: the run's `run_<id>` directory.

    Returns:
        The `SpatioGenomicLabelledMultiCounts`.

    Raises:
        ValueError: where the run zeroed differentially expressed genes (`quality.filter_normal_diffexp`), which its
            files do not name, or its genes are not `segmentation.tsv.gz`'s.
    """
    output_dir = Path(output_dir)
    (config_path,) = output_dir.glob(run_config_name("*"))
    config = YAMLConfig.from_file(config_path)
    if config.quality.filter_normal_diffexp:
        raise ValueError(f"{output_dir}: quality.filter_normal_diffexp zeroed genes its files do not name.")
    level = get_levels()[-1]
    segmentation = pd.read_csv(output_dir / "segmentation.tsv.gz", sep="\t")
    snp_segmentation = pd.read_csv(output_dir / "snp_segmentation.tsv.gz", sep="\t")
    clone_labels = pd.read_csv(output_dir / "clone_labels.tsv.gz", sep="\t", index_col="barcode")

    previous = get_global_config()
    with contextlib.chdir(input_dir):
        try:
            set_global_config(config)
            _, barcodes, adata, _, alt, other, unique_snp_ids, *_ = load_input_data(
                config,
                filter_gene_file=config.references.filtergenelist_file,
                filter_range_file=config.references.filterregion_file,
                min_snp_umis=config.quality.spot_min_snp_umis,
                min_percent_expressed_spots=config.quality.min_percent_expressed_spots,
            )
            table = form_gene_snp_table(unique_snp_ids, config.references.hgtable_file, adata)
            table = assign_initial_blocks(
                table, adata, alt, other, unique_snp_ids, initial_min_umi=config.quality.phasing_min_snp_umis
            )
        finally:
            set_global_config(previous)

    genes = table[table["is_interval"].astype(bool)]
    if not all((genes[c].to_numpy() == segmentation[c].to_numpy()).all() for c in ("CHR", "START", "END")):
        raise ValueError(f"{output_dir}: the inputs' genes are not segmentation.tsv.gz's.")

    n_bins = int(segmentation[level].max()) + 1
    kept = segmentation[level].to_numpy() >= 0
    pairs = pd.DataFrame(
        {"column": adata.var.index.get_indexer(genes["gene"].to_numpy()[kept]), "bin": segmentation[level][kept]}
    ).drop_duplicates()
    bin_of_gene = Segmentation.from_terms(adata.n_vars, pairs["column"], pairs["bin"], n_bins, name="bin_of_gene")

    snps = snp_segmentation[snp_segmentation[level] >= 0]
    columns = pd.Index(unique_snp_ids).get_indexer(snps["snp_id"].to_numpy())
    kept_phase = sp.diags(snps["phase"].to_numpy(np.int64), format="csr", dtype=np.int64)
    alt, other = sp.csr_matrix(alt)[:, columns], sp.csr_matrix(other)[:, columns]
    phased = alt @ kept_phase + other @ (sp.identity(len(columns), dtype=np.int64, format="csr") - kept_phase)
    bin_of_snp = Segmentation.from_terms(
        len(columns), np.arange(len(columns)), snps[level].to_numpy(np.int64), n_bins, name="bin_of_snp"
    )
    counts = SpatioGenomicMultiCounts(
        umis=SpatioGenomicCounts.from_grouped_columns(adata.layers["count"], bin_of_gene, np.int64),
        alt_snp_umis=SpatioGenomicCounts.from_grouped_columns(phased, bin_of_snp, np.int64),
        total_snp_umis=SpatioGenomicCounts.from_grouped_columns(alt + other, bin_of_snp, np.int64),
    )
    bins = Segmentation.of(Genes.from_table(segmentation.assign(is_interval=True)), segmentation[level], name=level)

    labels = clone_labels.reindex(np.asarray(barcodes))
    normal_spot = labels["normal_candidate"].to_numpy().astype(bool)
    counts = determine_normal_baseline(counts.get_labelled(bins.contig, bins.start, bins.end), normal_spot, config)
    return counts.with_labelling("clone", labels["clone_label"].to_numpy())
