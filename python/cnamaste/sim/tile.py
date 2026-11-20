"""A tiled sample: each of a fixture's spots tiled by a `side x side` block of spots, each block member's counts
redrawn independently.

Copy `k` of a spot sits at offset `get_tile_offsets(side)[k]` inside its block, so the planted clones keep their shape at
`side` times the scale, as `phasing.clone_init_type: rectangle` expects; copies of the whole slide side by side would
put one copy in each rectangle cell, every cell the same clone mixture. A copy's UMIs are Poisson at the source's
counts, its allele depth Poisson at the source's A + B, its B count binomial at the source's B fraction, so zeros stay
zero and every copy shares the source's germline phase and planted truth. The stress tier (`tests/test_stress.py`)
draws these on demand to time `run_cnamaste` beyond the fixtures' 3,000 spots.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import NamedTuple, cast

import anndata
import numpy as np
import pandas as pd
import scipy.sparse as sp

from cnamaste.sim.write import get_gzipped_bytes, get_made_dir, get_named_dir, write_sim_npz


class TiledCounts(NamedTuple):
    """One tile's redrawn counts.

    Args:
        umis: (spots, genes) UMIs.
        a_counts: (spots, SNPs) A-allele counts.
        b_counts: (spots, SNPs) B-allele counts.
    """

    umis: sp.csr_matrix
    a_counts: sp.csr_matrix
    b_counts: sp.csr_matrix


def draw_tile_counts(
    umis: sp.csr_matrix, a_counts: sp.csr_matrix, b_counts: sp.csr_matrix, rng: np.random.Generator
) -> TiledCounts:
    """One tile's counts, redrawn at the source's: UMIs Poisson, allele depth Poisson, B binomial at the B fraction.

    Args:
        umis: (spots, genes) the source's UMIs.
        a_counts: (spots, SNPs) the source's A-allele counts.
        b_counts: (spots, SNPs) the source's B-allele counts.
        rng: the tile's generator.

    Returns:
        The `TiledCounts`.
    """
    drawn = umis.copy()
    drawn.data = rng.poisson(umis.data).astype(np.int64)
    drawn.eliminate_zeros()

    depth = (a_counts + b_counts).tocsr()
    depth.sort_indices()
    rows = np.repeat(np.arange(depth.shape[0]), np.diff(depth.indptr))
    b_at_depth = np.asarray(b_counts.tocsr()[rows, depth.indices]).ravel()
    redrawn = rng.poisson(depth.data)
    b_data = rng.binomial(redrawn, b_at_depth / depth.data)
    # NB copy=True: A and B otherwise share index arrays, which eliminate_zeros compacts in place.
    a = sp.csr_matrix(((redrawn - b_data).astype(np.int64), depth.indices, depth.indptr), shape=depth.shape, copy=True)
    b = sp.csr_matrix((b_data.astype(np.int64), depth.indices, depth.indptr), shape=depth.shape, copy=True)
    a.eliminate_zeros()
    b.eliminate_zeros()
    return TiledCounts(drawn, a, b)


def get_clone_donors(clones: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Each spot's donor: a permutation of the spots of its planted clone, so neighbouring copies of a spot draw
    independent noise and each clone's spots are each drawn once.

    Args:
        clones: (spots,) each spot's planted clone.
        rng: the copy's generator.

    Returns:
        (spots,) each spot's donor index.
    """
    donors = np.arange(len(clones))
    for clone in np.unique(clones):
        members = np.flatnonzero(clones == clone)
        donors[members] = rng.permutation(members)
    return donors


def get_tile_offsets(side: int) -> list[tuple[int, int]]:
    """Each copy's `(row, column)` lattice offset inside its spot's block, row-major: the block of the source spot at
    `(r, c)` starts at `(side r, side c)`, and odd rows shift one column so `row + column` stays even, the hexagonal
    lattice's parity.

    Args:
        side: copies per side.

    Returns:
        The offsets.
    """
    return [(i, 2 * j + i % 2) for i in range(side) for j in range(side)]


def write_tiled_sample(source: Path, side: int, seed: int, out: Path, name: str) -> Path:
    """The tiled sample of `source` under `out/<name>_<hash>/`, with its truth, a `sample_sheet.tsv` (absolute paths)
    and a `manifest.json`, staged then renamed (`sim.write.get_named_dir`).

    Args:
        source: the fixture's sample directory, e.g. `sim/easy_2d4ce9a9`.
        side: copies per side, `side ** 2` copies of each spot.
        seed: the draw's; copy `k` takes the `k`-th stream spawned from it.
        out: the parent directory, existing.
        name: the sample's short name.

    Returns:
        The sample's directory.

    Raises:
        RuntimeError: where `out` is missing, `out/<name>_staging/` or the sample's directory exists.
    """
    adata = anndata.read_h5ad(source / "filtered_feature_bc_matrix.h5ad")
    a_counts = sp.load_npz(source / "cell_snp_Aallele.npz").tocsr()
    b_counts = sp.load_npz(source / "cell_snp_Ballele.npz").tocsr()
    barcodes = pd.read_csv(source / "barcodes.txt.gz", header=None)[0].astype(str).to_numpy()
    positions = pd.read_csv(source / "spatial/tissue_positions_list.csv.gz", header=None)
    positions.columns = ["barcode", "in_tissue", "array_row", "array_col", "pxl_row", "pxl_col"]
    truth = pd.read_csv(source / "truth_clone_labels.tsv.gz", sep="\t", index_col=0)

    streams = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(side**2)]
    tiles, names, placed, planted = [], [], [], []
    umis = sp.csr_matrix(adata.X)
    clones = truth["labels"].reindex(barcodes).to_numpy()
    for k, ((row, col), rng) in enumerate(zip(get_tile_offsets(side), streams, strict=True)):
        donors = get_clone_donors(clones, rng) if k else np.arange(len(barcodes))
        tiles.append(draw_tile_counts(umis[donors], a_counts[donors], b_counts[donors], rng))
        suffix = f"-t{k}"
        names.append(barcodes + suffix)
        moved = positions.assign(barcode=positions["barcode"] + suffix)
        # NB pixel coordinates move with the lattice's, as the fixtures' pixels are their lattice coordinates.
        moved[["array_row", "pxl_row"]] = moved[["array_row", "pxl_row"]] * side + row
        moved[["array_col", "pxl_col"]] = moved[["array_col", "pxl_col"]] * side + col
        placed.append(moved)
        planted.append(
            truth.set_axis(truth.index + suffix).assign(
                x=truth["x"].to_numpy() * side + row, y=truth["y"].to_numpy() * side + col
            )
        )

    staging = get_made_dir(out, f"{name}_staging", exist_ok=False)
    get_made_dir(staging, "spatial", exist_ok=False)
    shutil.copyfile(source / "unique_snp_ids.npy.gz", staging / "unique_snp_ids.npy.gz")
    shutil.copyfile(source / "truth_acn_profile.tsv.gz", staging / "truth_acn_profile.tsv.gz")
    all_barcodes = np.concatenate(names)
    (staging / "barcodes.txt.gz").write_bytes(get_gzipped_bytes(("\n".join(all_barcodes) + "\n").encode()))
    (staging / "spatial/tissue_positions_list.csv.gz").write_bytes(
        get_gzipped_bytes(pd.concat(placed).to_csv(header=False, index=False).encode())
    )
    (staging / "truth_clone_labels.tsv.gz").write_bytes(get_gzipped_bytes(pd.concat(planted).to_csv(sep="\t").encode()))
    obs = pd.concat([cast(pd.DataFrame, adata.obs).set_axis(adata.obs_names + f"-t{k}") for k in range(side**2)])
    tiled = anndata.AnnData(X=sp.vstack([t.umis for t in tiles], format="csr"), obs=obs, var=adata.var)
    tiled.write_h5ad(staging / "filtered_feature_bc_matrix.h5ad", compression="gzip")
    write_sim_npz(staging / "cell_snp_Aallele.npz", sp.vstack([t.a_counts for t in tiles], format="csr"), None, False)
    write_sim_npz(staging / "cell_snp_Ballele.npz", sp.vstack([t.b_counts for t in tiles], format="csr"), None, False)

    directory, fixture_hash = get_named_dir(staging, name)
    located = str(directory.resolve())
    sheet = pd.DataFrame({"bam": ["NULL"], "sample_id": [name], "spaceranger_dir": [located], "snp_dir": [located]})
    sheet.to_csv(directory / "sample_sheet.tsv", sep="\t", index=False)
    manifest = {"name": name, "source": source.name, "side": side, "seed": seed, "spots": len(all_barcodes)}
    (directory / "manifest.json").write_text(json.dumps({**manifest, "hash": fixture_hash}, indent=2) + "\n")
    return directory
