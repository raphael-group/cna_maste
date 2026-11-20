"""A realization written as a sample `run_cnamaste` reads: one directory per slice (its `.h5ad` and positions, Visium's
or, for a square array, Visium HD's with CalicoST's inputs beside them), a shared `snp/`, the planted truth, a
`sample_sheet.tsv` and a `manifest.json`, named `<name>_<fixture hash>`.

Text and `.npy` files are gzipped as the fixtures store them (`get_gzipped_bytes`); the counts are deflated at
`COMPRESSION_LEVEL`. Every directory is made one level at a time by `get_made_dir`, within an existing directory;
a sample is written to `<name>_staging/` and renamed by `get_named_dir`; nothing is deleted. Files are created in
`write_realization` and in `write_reference`, which writes `sim.normal`'s fit.
"""

from __future__ import annotations

import gzip
import io
import json
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import anndata
import numpy as np
import pandas as pd
import scipy.sparse as sp
import yaml

from cnamaste.qa.sims import get_fixture_hash, get_project, get_resources, grch38
from cnamaste.sim.manifest import Simulation
from cnamaste.sim.tree import get_tree_table

if TYPE_CHECKING:
    from cnamaste.sim.draw import Realization

COMPRESSION_LEVEL = 1
"""zlib level of a drawn realization's `.npz` and `.h5ad`."""


BIN_SIZE_UM = 16.0
"""A Visium HD bin's side, um: the 16 um binning."""


def get_bin_size_um() -> float:
    """Return a Visium HD bin's side in um, 16.0."""
    return BIN_SIZE_UM


MICRONS_PER_PIXEL = 0.2737061758251425
"""A Visium HD slide's full-resolution pixel, um (`he.py`'s stated example)."""


def get_microns_per_pixel() -> float:
    """Return a Visium HD full-resolution pixel's side in um, 0.2737061758251425."""
    return MICRONS_PER_PIXEL


def get_gzipped_bytes(data: bytes) -> bytes:
    """`data` gzipped as the fixtures store it: level 9, no name, `mtime` 0, OS byte 255 (unknown).

    Args:
        data: the plain bytes.

    Returns:
        The stored bytes.
    """
    packed = gzip.compress(data, compresslevel=9, mtime=0)
    # NB Python 3.12 writes OS byte 3, 3.13 writes 255; the committed fixtures hold 255.
    return packed[:9] + b"\xff" + packed[10:]


def get_npy(array: np.ndarray) -> bytes:
    """`array` as `.npy` bytes; an object array pickled.

    Args:
        array: the array.

    Returns:
        The bytes.
    """
    stream = io.BytesIO()
    np.save(stream, array, allow_pickle=True)
    return stream.getvalue()


def write_sim_npz(path: Path, matrix: sp.csr_matrix, compresslevel: int | None, force_zip64: bool) -> None:
    """`matrix` as `scipy.sparse.save_npz` stores it, its members dated 1980-01-01 so the bytes depend on the content.

    Args:
        path: the `.npz`.
        matrix: the CSR matrix.
        compresslevel: the zlib level; None is zlib's default.
        force_zip64: whether each member's local header carries zip64 sizes.
    """
    members = {
        "indices": matrix.indices,
        "indptr": matrix.indptr,
        "format": np.array(matrix.format.encode("ascii")),
        "shape": np.array(matrix.shape),
        "data": matrix.data,
    }
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=compresslevel) as packed:
        for name, array in members.items():
            with packed.open(f"{name}.npy", "w", force_zip64=force_zip64) as stream:
                stream.write(get_npy(array))


def get_record(value: Any) -> Any:
    """`value` as JSON states it: a NamedTuple as a table, a tuple as a list, a path as text, relative to the project
    where it lies in it, so a committed `manifest.json` reads the same from any checkout.

    Args:
        value: e.g. a `Simulation`.

    Returns:
        The record.
    """
    if hasattr(value, "_asdict"):
        return {k: get_record(v) for k, v in value._asdict().items()}
    if isinstance(value, dict):
        return {str(k): get_record(v) for k, v in value.items()}
    if isinstance(value, tuple | list):
        return [get_record(v) for v in value]
    if isinstance(value, Path):
        return str(value.relative_to(get_project()) if value.is_relative_to(get_project()) else value)
    return value


def get_run_config(simulation: Simulation, sample: Path) -> dict[str, Any]:
    """A realization's `run_cnamaste` configuration: `[config] base` with each `[config.<section>]` laid over its
    section, its paths at `sample` and its references at CalicoST's GRCh38 resources.

    Args:
        simulation: the simulation.
        sample: the realization's directory, absolute.

    Returns:
        The configuration.

    Raises:
        ValueError: where no GRCh38 resources are found.
    """
    references = grch38()
    if references is None:
        raise ValueError("no GRCh38 resources: ref/ or $CNAMASTE_GRCH38")
    config: dict[str, Any] = yaml.safe_load(simulation.config_base.read_text())
    for section, values in simulation.config.items():
        config.setdefault(section, {}).update(values)
    config["paths"] = {"sample_sheet": str(sample / "sample_sheet.tsv"), "output_dir": str(sample / "output")}
    config["references"] |= {key: str(references / name) for key, name in get_resources().items()}
    return config


def write_realization(realization: Realization, simulation: Simulation, out: Path) -> Path:
    """`realization` under `out/<name>_<fixture hash>/`, staged then renamed (`get_named_dir`).

    Args:
        realization: the draw.
        simulation: its simulation.
        out: the parent directory, existing.

    Returns:
        The sample's directory.

    Raises:
        RuntimeError: where `out` is missing, `out/<name>_staging/` or the sample's directory exists.
    """
    planted = realization.planted
    clones = np.asarray(planted.clones)
    staging = get_made_dir(out, f"{simulation.name}_staging", exist_ok=False)
    get_made_dir(staging, "snp", exist_ok=False)
    for umis, labels, sample_id, names in zip(
        realization.umis, planted.labels, planted.sample_ids, planted.barcodes, strict=True
    ):
        get_made_dir(get_made_dir(staging, sample_id, exist_ok=False), "spatial", exist_ok=False)
        counts = anndata.AnnData(
            X=umis, obs=pd.DataFrame({"labels": clones[labels]}, index=names), var=pd.DataFrame(index=planted.genes)
        )
        counts.write_h5ad(
            staging / sample_id / "filtered_feature_bc_matrix.h5ad",
            compression="gzip",
            compression_opts=COMPRESSION_LEVEL,
        )
        if simulation.array[0] == "square":
            write_visium_hd_positions(staging / sample_id / "spatial", names, planted.array_row, planted.array_col)
        else:
            row, col = planted.array_row, planted.array_col
            positions = pd.DataFrame({0: names, 1: 1, 2: row, 3: col, 4: row, 5: col})
            (staging / sample_id / "spatial" / "tissue_positions_list.csv.gz").write_bytes(
                get_gzipped_bytes(positions.to_csv(header=False, index=False).encode())
            )

    texts = {
        "snp/barcodes.txt.gz": "\n".join(np.concatenate(planted.barcodes)) + "\n",
        "truth_clone_labels.tsv.gz": pd.concat(
            pd.DataFrame(
                {"labels": clones[labels], "x": planted.array_row, "y": planted.array_col, "sample_id": sample_id},
                index=pd.Index(names, name="barcode"),
            )
            for labels, sample_id, names in zip(planted.labels, planted.sample_ids, planted.barcodes, strict=True)
        ).to_csv(sep="\t"),
        "truth_acn_profile.tsv.gz": planted.profile.to_csv(sep="\t", index=False),
        "truth_tree.tsv.gz": get_tree_table(planted.tree).to_csv(sep="\t", index=False),
    }
    for name, text in texts.items():
        (staging / name).write_bytes(get_gzipped_bytes(text.encode()))
    (staging / "snp/unique_snp_ids.npy.gz").write_bytes(get_gzipped_bytes(get_npy(planted.snp_ids.astype(object))))
    (staging / "truth_phase.npy.gz").write_bytes(get_gzipped_bytes(get_npy(realization.switched)))
    if simulation.array[0] == "square":
        # NB CalicoST's load_data reads these two uncompressed; the fixture hash keys each once with its .gz copy.
        (staging / "snp/barcodes.txt").write_text(texts["snp/barcodes.txt.gz"])
        (staging / "snp/unique_snp_ids.npy").write_bytes(get_npy(planted.snp_ids.astype(object)))
    write_sim_npz(staging / "snp/cell_snp_Aallele.npz", realization.a_counts, COMPRESSION_LEVEL, True)
    write_sim_npz(staging / "snp/cell_snp_Ballele.npz", realization.b_counts, COMPRESSION_LEVEL, True)

    directory, fixture_hash = get_named_dir(staging, simulation.name)

    # NB absolute, so a run reads the sample from any working directory.
    located = directory.resolve()
    pd.DataFrame(
        {
            "bam": ["unused.bam"] * len(planted.sample_ids),
            "sample_id": planted.sample_ids,
            "spaceranger_dir": [str(located / i) for i in planted.sample_ids],
            "snp_dir": [str(located / "snp")] * len(planted.sample_ids),
        }
    ).to_csv(directory / "sample_sheet.tsv", sep="\t", index=False)
    manifest = {"name": simulation.name, "realization": realization.k, "hash": fixture_hash}
    manifest |= {"clones": list(planted.clones), "simulation": get_record(simulation)}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    (directory / "config.yaml").write_text(yaml.safe_dump(get_run_config(simulation, located), sort_keys=False))
    # NB run_cnamaste requires its paths.output_dir; the fixture hash leaves output/ out.
    get_made_dir(directory, "output", exist_ok=False)
    return directory


def write_visium_hd_positions(spatial: Path, names: np.ndarray, row: np.ndarray, col: np.ndarray) -> None:
    """A square array's slice as Visium HD's bins: `tissue_positions.parquet` (`io.get_spatial_positions`) and
    `scalefactors_json.json`, `bin_size_um` 16 and `MICRONS_PER_PIXEL`; and the same table as CalicoST's
    `tissue_positions.csv`, which `load_data` reads, its `x`, `y` the array row and column.

    Args:
        spatial: the slice's existing `spatial/`.
        names: (bins,) barcodes.
        row: (bins,) array rows.
        col: (bins,) array columns.
    """
    pixels = get_bin_size_um() / get_microns_per_pixel()
    positions = pd.DataFrame(
        {
            "barcode": names,
            "in_tissue": 1,
            "array_row": row,
            "array_col": col,
            "pxl_row_in_fullres": row * pixels,
            "pxl_col_in_fullres": col * pixels,
        }
    )
    positions.to_parquet(spatial / "tissue_positions.parquet", index=False)
    positions.to_csv(spatial / "tissue_positions.csv", index=False)
    scalefactors = {"bin_size_um": get_bin_size_um(), "microns_per_pixel": get_microns_per_pixel()}
    (spatial / "scalefactors_json.json").write_text(json.dumps(scalefactors) + "\n")


def write_reference(baseline_text: str, coverage_text: str, into: Path) -> list[Path]:
    """`sim.normal`'s fit as `into/normal_baseline.txt.gz` and `into/normal_coverage.toml`, neither replaced.

    Args:
        baseline_text: `normal.get_baseline_text`'s.
        coverage_text: `normal.get_coverage_text`'s.
        into: an existing directory.

    Returns:
        The two files.

    Raises:
        RuntimeError: where `into` is missing or either file exists; nothing is written.
    """
    if not into.is_dir():
        raise RuntimeError(f"{into}: not an existing directory")
    contents = {
        into / "normal_baseline.txt.gz": get_gzipped_bytes(baseline_text.encode()),
        into / "normal_coverage.toml": coverage_text.encode(),
    }
    found = [str(path) for path in contents if path.exists()]
    if found:
        raise RuntimeError(f"{', '.join(found)}: exists, not replaced")
    for path, content in contents.items():
        # NB "x" creates the file and refuses an existing one.
        with path.open("xb") as stream:
            stream.write(content)
    return list(contents)


def get_made_dir(out: Path, name: str, exist_ok: bool) -> Path:
    """`out/name`, made directly within `out`, e.g. a draw's `qa/` for its truth pages (`sim.plot`).

    Args:
        out: the parent directory, existing.
        name: the new directory's name, one level.
        exist_ok: whether an existing `out/name` is reused.

    Returns:
        `out/name`.

    Raises:
        RuntimeError: where `out` is missing, `out/name` resolves outside `out` or exists and `exist_ok` is False.
    """
    if not out.is_dir():
        raise RuntimeError(f"no output directory: {out}")
    directory = out / name
    if directory.resolve().parent != out.resolve():
        raise RuntimeError(f"{name} is not directly within {out}")
    if directory.exists() and not exist_ok:
        raise RuntimeError(f"{directory} exists")
    directory.mkdir(exist_ok=exist_ok)
    return directory


def get_named_dir(staging: Path, name: str) -> tuple[Path, str]:
    """`staging` renamed to `<name>_<fixture hash>/` beside it; an existing one is not replaced.

    Args:
        staging: the written sample.
        name: the sample's short name.

    Returns:
        The sample's directory and its fixture hash.

    Raises:
        RuntimeError: where the sample's directory exists; `staging` is left in place.
    """
    fixture_hash = get_fixture_hash(staging)
    directory = staging.parent / f"{name}_{fixture_hash}"
    if directory.exists():
        raise RuntimeError(f"{directory} exists; {staging} is left in place")
    staging.rename(directory)
    return directory, fixture_hash
