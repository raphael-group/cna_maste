"""The staged fixtures: each one's sample and staged file, where the samples live, the ones a bare run selects, the
inputs a sample holds and CalicoST's GRCh38 resources."""

from __future__ import annotations

import gzip
import hashlib
import os
from pathlib import Path

import pandas as pd

from cnamaste.platforms.visium import reads_visium_hd

PROJECT = Path(__file__).resolve().parents[3]
"""The cnamaste checkout the package runs from (an editable install, `cnamaste/python/`)."""


def get_project() -> Path:
    """Return the project, `PROJECT`."""
    return PROJECT


SAMPLES_VARIABLE = "CNAMASTE_SIMS"
"""The one name that moves the samples elsewhere; unset, a checkout reads the committed ones."""

SIM = PROJECT / "sim"
"""The committed fixtures: one directory per fixture, `<name>_<hash>`, holding its sample and `capture/`."""


def fixture(sim_hash: str) -> Path:
    """`sim_hash`'s committed directory, `sim/<name>_<hash>`.

    Args:
        sim_hash: the fixture hash, one `FIXTURES` names.

    Returns:
        The directory.
    """
    return SIM / FIXTURES[sim_hash]


def sample(sim_hash: str) -> Path:
    """`sim_hash`'s sample directory: `CNAMASTE_SIMS`/`SAMPLES[sim_hash]` where it is set, else `fixture(sim_hash)`.

    A committed sample is the six inputs `load_input_data` reads and the two planted truth files, byte for byte the
    ones its capture read, so a row that falls back to the sample runs on a bare checkout.

    Args:
        sim_hash: the fixture hash, one `SAMPLES` names.

    Returns:
        The directory, which need not exist.
    """
    named = os.environ.get(SAMPLES_VARIABLE)
    return Path(named) / SAMPLES[sim_hash] if named else fixture(sim_hash)


def simulation(sample: Path) -> str:
    """`sample`'s simulation name, the `sample_id` a run of it reads: `SAMPLES`' for a committed fixture's directory,
    else the directory's own name.

    Args:
        sample: a sample directory.

    Returns:
        The name.
    """
    named = {FIXTURES[h]: name for h, name in SAMPLES.items()}
    return named.get(sample.name, sample.name)


def capture_file(sim_hash: str) -> Path:
    """`sim_hash`'s capture, `sim/<name>_<hash>/capture/<name>_<hash>_capture.hdf5`, which need not exist.

    Args:
        sim_hash: the fixture hash.

    Returns:
        The path.
    """
    return fixture(sim_hash) / "capture" / f"{FIXTURES[sim_hash]}_capture.hdf5"


FIXTURES: dict[str, str] = {
    "2d4ce9a9": "easy_2d4ce9a9",
    "8797710b": "hard_8797710b",
    "a9a68913": "harder_a9a68913",
    "81ebaea6": "random4_81ebaea6",
    "bdff091f": "dev_tree_easy_square_bdff091f",
    "e4bbe03f": "dev_tree_easy_e4bbe03f",
    "33e3471e": "dev_tree_1s_dense_33e3471e",
}
"""Fixture hash -> its directory under `sim/`: its short name and its hash."""


def get_fixtures() -> dict[str, str]:
    """Return the fixtures, `FIXTURES`."""
    return FIXTURES


SAMPLES: dict[str, str] = {
    "2d4ce9a9": "numcnas1.2_cnasize5e7_ploidy2_random0",
    "8797710b": "numcnas6.3_cnasize1e7_ploidy2_random0",
    "a9a68913": "numcnas3.3_cnasize1e7_ploidy2_random6",
    "81ebaea6": "numcnas3.3_cnasize5e7_ploidy2_random4",
    "bdff091f": "dev_tree_easy_square",
    "e4bbe03f": "dev_tree_easy",
    "33e3471e": "dev_tree_1s_dense",
}
"""Fixture hash -> its simulation's name, the sample's name under `CNAMASTE_SIMS`; committed, the sample is
`fixture(hash)`, held to the fixture's capture. Which fixtures a run selects is `SUPPORTED_SIMS`
and `TRUTH_SIMS`; this says only where a sample is. A sample is matched to its capture by `truth_acn_profile`,
which differs per simulation where the clone labels do not.

The hash is `port.sim.fixtures.realization_hash` of the sample as committed with its `filtered_feature_bc_matrix.h5ad`
as first written: `dev_tree_easy` and `dev_tree_1s_dense` realizations 0 their manifests' `r0_hash` (`port`
`sim/manifests/<name>.toml`), `dev_tree_easy_square` realization 0's with its square grid stored as Visium HD
positions (`sandbox.store_sample.hd_positions`; its draw's `r0_hash` is `23acef50`), the CalicoST samples their
committed layout's. Each fixture's `README.md` says where it came from."""


def get_samples() -> dict[str, str]:
    """Return the samples, `SAMPLES`."""
    return SAMPLES


SUPPORTED_SIMS: dict[str, str] = {
    "2d4ce9a9": "sim/easy_2d4ce9a9/capture/easy_2d4ce9a9_capture.hdf5",
    "8797710b": "sim/hard_8797710b/capture/hard_8797710b_capture.hdf5",
}
"""Fixture hash -> its staged file, relative to `cnamaste/`: the fixtures every file can select (`--sims`)."""


def get_supported_sims() -> dict[str, str]:
    """Return the supported sims, `SUPPORTED_SIMS`."""
    return SUPPORTED_SIMS


DEV_SIMS = ("2d4ce9a9", "8797710b")
"""The dev tier: CalicoST easy and hard. A bare `pytest`, every pull request's evidence and `recapture` run on these
two; every other fixture (`SUPPORTED_SIMS` beyond them, `TRUTH_SIMS`) runs on request, `--sims all` or by hash."""


def get_dev_sims() -> tuple[str, ...]:
    """Return the dev sims, ("2d4ce9a9", "8797710b")."""
    return DEV_SIMS


DEFAULT_SIMS = DEV_SIMS

TRUTH_SIMS: dict[str, str] = {
    "e4bbe03f": "sim/dev_tree_easy_e4bbe03f/capture/dev_tree_easy_e4bbe03f_capture.hdf5",
    "33e3471e": "sim/dev_tree_1s_dense_33e3471e/capture/dev_tree_1s_dense_33e3471e_capture.hdf5",
    "a9a68913": "sim/harder_a9a68913/capture/harder_a9a68913_capture.hdf5",
    "81ebaea6": "sim/random4_81ebaea6/capture/random4_81ebaea6_capture.hdf5",
}
"""Fixture hash -> its staged file (with `/truth`), scored against its truth and held to its
sample and template by the guards alone, on request: `dev_tree_easy` r0 and `dev_tree_1s_dense` r0, drawn
by `port`; CalicoST harder, whose four open rows are not here: two
CalicoST-reference rows on a gene name, the lattice decoder against the MILP, and the lookups' decode agreement;
and CalicoST `numcnas3.3_cnasize5e7_ploidy2_random4`. Every sample is committed. `dev_tree_easy_square` r0
(`bdff091f`) is not here until its Visium HD sample is captured: the capture of its Visium form (`23acef50`) went with
that form. A fixture moves to `SUPPORTED_SIMS`
with the rows that score it, never to `DEV_SIMS` unasked: each costs a default run nothing. Only shrinks."""

NAMES: dict[str, str] = {
    "2d4ce9a9": "CalicoST easy",
    "8797710b": "CalicoST hard",
    "bdff091f": "dev_tree_easy_square r0",
    "e4bbe03f": "dev_tree_easy r0",
    "33e3471e": "dev_tree_1s_dense r0",
    "a9a68913": "CalicoST harder",
    "81ebaea6": "CalicoST random4",
}
"""Fixture hash -> its readable name, as the ledger writes it."""


def get_names() -> dict[str, str]:
    """Return the names, `NAMES`."""
    return NAMES


def selected_sims(asked: str) -> list[str]:
    """The fixtures `--sims` names: `dev` (`DEV_SIMS`, the default), `all` (every staged fixture, `SUPPORTED_SIMS` then
    `TRUTH_SIMS`), or hashes, comma separated.

    Args:
        asked: the option's text.

    Returns:
        The hashes, in order.

    Raises:
        ValueError: naming a hash with no staged file.
    """
    if asked in ("", "dev"):
        return list(DEV_SIMS)
    if asked == "all":
        return [*SUPPORTED_SIMS, *TRUTH_SIMS]
    hashes = [h for h in asked.split(",") if h]
    unknown = sorted(set(hashes) - set(SUPPORTED_SIMS) - set(TRUTH_SIMS))
    if unknown:
        raise ValueError(f"--sims: no staged file for {unknown}")
    return hashes


def staged(sim_hash: str) -> str:
    """`sim_hash`'s staged file, relative to `cnamaste/`, from `SUPPORTED_SIMS` or `TRUTH_SIMS`.

    Args:
        sim_hash: the fixture hash.

    Returns:
        The relative path.
    """
    return SUPPORTED_SIMS[sim_hash] if sim_hash in SUPPORTED_SIMS else TRUTH_SIMS[sim_hash]


INPUTS = (
    "barcodes.txt",
    "unique_snp_ids.npy",
    "cell_snp_Aallele.npz",
    "cell_snp_Ballele.npz",
    "filtered_feature_bc_matrix.h5ad",
    "spatial/tissue_positions_list.csv",
)
"""What `cnamaste.io.load_input_data` reads from a CalicoST sample, committed as `<name>.gz` where not compressed already."""

RESOURCES = {
    "geneticmap_file": "genetic_map_GRCh38_merged.tab.gz",
    "hgtable_file": "hgTables_hg38_gencode.txt",
    "filtergenelist_file": "ig_gene_list.txt",
    "filterregion_file": "HLA_regions.bed",
}
"""`references.*` -> CalicoST's `GRCh38_resources` file, as the staged configuration names them."""


def get_resources() -> dict[str, str]:
    """Return the resources, `RESOURCES`."""
    return RESOURCES


def grch38() -> Path | None:
    """CalicoST's `GRCh38_resources`: `$CNAMASTE_GRCH38`, else the checkout's `ref/`, else uv's git checkout.

    Returns:
        The directory, or None where none holds the four files.
    """
    candidates = [os.environ.get("CNAMASTE_GRCH38", ""), str(PROJECT / "ref")]
    candidates += sorted(str(p) for p in (Path.home() / ".cache/uv/git-v0/checkouts").glob("*/*/GRCh38_resources"))
    for c in candidates:
        if c and all((Path(c) / f).exists() for f in RESOURCES.values()):
            return Path(c)
    return None


SLICE_INPUTS = ("filtered_feature_bc_matrix.h5ad", "spatial/tissue_positions_list.csv")
"""What `load_input_data` reads from each Visium slice of a drawn sample (one directory per slice)."""

HD_SLICE_INPUTS = (
    "filtered_feature_bc_matrix.h5ad",
    "spatial/tissue_positions.parquet",
    "spatial/scalefactors_json.json",
)
"""What `load_input_data` reads from each Visium HD slice of a drawn sample: a square-grid draw's
(`store_sample.hd_positions`)."""

SNP_INPUTS = ("barcodes.txt", "unique_snp_ids.npy", "cell_snp_Aallele.npz", "cell_snp_Ballele.npz")
"""What `load_input_data` reads from a drawn sample's shared `snp/` directory."""


def drawn(sample: Path) -> bool:
    """A drawn sample: one directory per slice beside a shared `snp/`, its `sample_sheet.tsv` naming the slices and
    its `manifest.json` (`sim.write.write_realization`).

    Args:
        sample: the sample directory.

    Returns:
        Whether `sample` is drawn.
    """
    return (sample / "snp").is_dir()


def slices(sample: Path) -> list[str]:
    """A drawn sample's slices, in its sample sheet's order.

    Args:
        sample: the drawn sample directory.

    Returns:
        The slice ids.
    """
    return [str(s) for s in pd.read_csv(sample / "sample_sheet.tsv", sep="\t")["sample_id"]]


def slice_inputs(directory: Path) -> tuple[str, ...]:
    """What `load_input_data` reads from one slice of a drawn sample: Visium HD's where the slice reads as Visium HD
    (`platforms.visium.reads_visium_hd`), else Visium's.

    Args:
        directory: the slice's directory.

    Returns:
        The names, relative to `directory`.
    """
    return HD_SLICE_INPUTS if reads_visium_hd(directory) else SLICE_INPUTS


def input_files(sample: Path) -> dict[str, Path]:
    """Each input `load_input_data` reads, by name (relative to `sample`), at its path (`.gz` where compressed).

    Args:
        sample: the sample directory.

    Returns:
        Name -> path.
    """
    names = (
        [f"{s}/{n}" for s in slices(sample) for n in slice_inputs(sample / s)] + [f"snp/{n}" for n in SNP_INPUTS]
        if drawn(sample)
        else list(INPUTS)
    )
    found = {}
    for name in names:
        plain, packed = sample / name, sample / (name + ".gz")
        found[name] = plain if plain.exists() else packed
    return found


HASH_SKIPPED = ("config.yaml", "manifest.json", "sample_sheet.tsv")
"""Files a fixture hash leaves out: each records a path or a run's parameters rather than the sample."""


def get_fixture_hash(sample: Path) -> str:
    """A sample's fixture hash: the first 8 hex of SHA-256 over each file's name and decoded bytes, in sorted order.

    A `.gz` file is keyed by its name without `.gz` and hashed decompressed, so how it is stored cannot move the hash;
    other files, the `.h5ad` and `.npz` included, are hashed as stored. `HASH_SKIPPED` and the `output/` and `qa/`
    directories a run writes are left out. Reproduces CalicoST easy's `2d4ce9a9` from its eight sample files.

    Args:
        sample: the sample directory.

    Returns:
        The hash.

    Raises:
        ValueError: where a plain and a `.gz` copy of one name decode differently.
    """
    digest = hashlib.sha256()
    for directory, names, files in os.walk(sample):
        names[:] = sorted(n for n in names if n not in {"output", "qa"})
        decoded: dict[str, bytes] = {}
        for name in files:
            stored = Path(directory, name)
            key = stored.relative_to(sample).as_posix().removesuffix(".gz")
            data = gzip.decompress(stored.read_bytes()) if name.endswith(".gz") else stored.read_bytes()
            if decoded.get(key, data) != data:
                raise ValueError(f"{stored}: plain and .gz copies of {key} decode differently")
            decoded[key] = data
        for key in sorted(k for k in decoded if k not in HASH_SKIPPED):
            digest.update(key.encode())
            digest.update(decoded[key])
    return digest.hexdigest()[:8]


def get_capture_path(fixture_hash: str) -> Path:
    """Return a captured fixture's HDF5 file.

    Args:
        fixture_hash: the fixture's hash, a key of `SUPPORTED_SIMS`.

    Returns:
        The capture's path in this checkout.
    """
    return PROJECT / SUPPORTED_SIMS[fixture_hash]
