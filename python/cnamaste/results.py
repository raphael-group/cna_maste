"""Where a run's results are written: its name (`run_id`), its files (`get_output_file_meta`) and their writers, and
the integer-copy tables'."""

from __future__ import annotations

import copy
import functools
import hashlib
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.text import Text

from cnamaste.config import YAMLConfig, get_global_config, start_time
from cnamaste.figure_style import get_dpi, get_pdf_metadata, page_style, restyle
from cnamaste.logger import get_logger
from cnamaste.segments import get_levels

logger = get_logger(__name__, start_time=start_time)


def get_output_path_prefix(names):
    """The run's integer-copy tables' prefix, `<prefix>_<level>.tsv`: `cnv_<ploidy>` at one assumed ploidy, else
    `cnv_best_ploidy`, each clone at its best (`config.assumed_ploidy`). CalicoST names its own `cnv_<ploidy>` by its
    ploidy constraint (`calicost.read.CALICOST_PLOIDY_SUFFIXES`).

    Args:
        names: the assumed ploidy's names.

    Returns:
        The prefix.
    """
    return f"cnv_{names[0] if len(names) == 1 else 'best_ploidy'}"


def get_output_path(run_dir, level):
    """The run's one integer-copy table at `level` in `run_dir`, whichever ploidy names it (`get_output_path_prefix`).

    Args:
        run_dir: a `run_cnamaste` output directory.
        level: `seglevel`, `genelevel` or `perstate`.

    Returns:
        The path.

    Raises:
        FileNotFoundError: where `run_dir` holds none, or more than one.
    """
    found = sorted(Path(run_dir).glob(f"cnv_*_{level}.tsv"))
    if len(found) != 1:
        raise FileNotFoundError(f"one cnv_<ploidy>_{level}.tsv in {run_dir}, found {[p.name for p in found]}")
    return found[0]


class IntegerCopyTables(NamedTuple):
    """A run's integer copies, `<prefix>_genelevel.tsv`, `_seglevel.tsv` and `_perstate.tsv` as written."""

    genelevel: pd.DataFrame
    seglevel: pd.DataFrame
    perstate: pd.DataFrame


def get_integer_copy_tables(
    df_bin_info,
    df_gene_snp,
    res_combine,
    bin_copies,
    final_clone_ids,
    base_nb_mean,
    n_states,
    ploidies,
    clone_ploidy,
):
    """The integer-copy tables of the final clones, a clone with no assigned transcripts left out.

    Args: `df_bin_info` CHR, START, END per bin; `df_gene_snp` genes (`is_interval`) with `bin_id`, null where no bin;
    `res_combine` the final fit; `bin_copies` (n_obs, n_clones, 2) decoded (A, B); `final_clone_ids` 0 .. N-1;
    `base_nb_mean` (n_obs, n_clones) pseudobulk; `n_states`; `ploidies` assumed; `clone_ploidy` clone index ->
    ploidy decoded at.

    Returns:
        The `IntegerCopyTables`.

    Raises:
        ValueError: where no final clone has assigned transcripts.
    """
    n_obs = bin_copies.shape[0]
    allele_specific_copy, state_cnv, df_genelevel_cnv = [], [], None

    for s, cid in enumerate(final_clone_ids):
        if np.sum(base_nb_mean[:, s]) == 0:
            logger.warning(f"Final clone {cid} has no assigned transcripts.")
            continue

        this_pred_cnv = res_combine["pred_cnv"][s]

        us, cnts = np.unique(this_pred_cnv, return_counts=True)

        logger.debug(f"Found state usage for clone {cid}:\n{pd.DataFrame({'state': us, 'counts': cnts})}")

        clone_copies = bin_copies[:, s]

        for name, data in zip(
            ("k", "logmu", "p", "A", "B"),
            [
                this_pred_cnv,  # NB the HMM state k of each bin, for each clone.
                res_combine.parameters.log_mu[this_pred_cnv],  # NB best model read depth for each clone.
                res_combine.parameters.p_binom[this_pred_cnv],  # NB best model baf for each clone.
                clone_copies[:, 0],  # NB best integer A-copies for each clone.
                clone_copies[:, 1],  # NB best integer B-copies for each clone.
            ],
        ):
            allele_specific_copy.append(
                pd.DataFrame(
                    data.reshape(1, -1),
                    index=[f"clone{cid} {name}"],
                    columns=np.arange(n_obs),
                )
            )

        for name, data in zip(
            ("logmu", "p"),
            [
                res_combine.parameters.log_mu,  # NB best per-state read depth for each clone.
                res_combine.parameters.p_binom,  # NB best per-state baf for each clone.
            ],
        ):
            state_cnv.append(
                pd.DataFrame(
                    data.reshape(-1, 1),
                    columns=[f"clone{cid} {name}"],
                    index=np.arange(n_states),
                )
            )
        # NB the ploidy each clone was decoded at, where several were assumed; the normal clone is (1, 1).
        if len(ploidies) > 1:
            state_cnv.append(
                pd.DataFrame(
                    {f"clone{cid} ploidy": [clone_ploidy.get(s, "diploid")] * n_states},
                    index=np.arange(n_states),
                )
            )

        # NB a gene whose bin normal_baf_bin_filter removed has a null bin_id and no call:
        #    NaN here, dropped by the isnull filter below.
        df_genes = df_gene_snp[df_gene_snp.is_interval]
        called = df_genes["bin_id"].notna().to_numpy()

        copies = np.full((len(df_genes), 2), np.nan)
        copies[called] = clone_copies[df_genes["bin_id"][called].to_numpy(dtype=int)]

        tmpdf = pd.DataFrame(
            {
                "gene": df_genes.gene,
                f"clone{s} A": copies[:, 0],
                f"clone{s} B": copies[:, 1],
            }
        ).set_index("gene")

        # NB join the temporary dataframe with the existing gene-level copy number dataframe, i.e. if a subsequent clone.
        if df_genelevel_cnv is None:
            df_genelevel_cnv = copy.copy(tmpdf[~tmpdf[f"clone{s} A"].isnull()].astype(int))
        else:
            df_genelevel_cnv = df_genelevel_cnv.join(tmpdf[~tmpdf[f"clone{s} A"].isnull()].astype(int))

    if len(state_cnv) == 0:
        raise ValueError(f"No final clone of {list(final_clone_ids)} has assigned transcripts: no integer copies.")

    df_seglevel_cnv = df_bin_info[["CHR", "START", "END"]].join(pd.concat(allele_specific_copy).T)

    state_cnv = functools.reduce(
        lambda left, right: pd.merge(left, right, left_index=True, right_index=True, how="inner"),
        state_cnv,
    )

    return IntegerCopyTables(df_genelevel_cnv, df_seglevel_cnv, state_cnv)


RUN_ID_EXCLUDED = ("paths", "references", "run.pause", "hmrf.n_colors")
"""The keys `run_id` leaves out: where a run reads and writes (`paths`, the `references` files), a key that changes
nothing it computes (`run.pause`), and one the run derives rather than is given (`hmrf.n_colors`, set by its initial
partition). Every other key, defaults included, names the run."""

RUN_ID_DIGITS = 8
"""The hex digits of the configuration's SHA-256 a run is named by: `run_<id>`."""


def canonical_run_id_value(value: Any) -> Any:
    """`value` as `run_id` hashes it: a number as a float and a numeric text as its number, so `1`, `1.0` and `-o`'s
    `"1.0"` agree; `"none"` as None, as `YAMLConfig` reads it; nested dicts likewise."""
    if isinstance(value, dict):
        return {k: canonical_run_id_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [canonical_run_id_value(v) for v in value]
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        if value.strip().lower() == "none":
            return None
        try:
            return float(value)
        except ValueError:
            return value
    return str(value)


def run_id(config: YAMLConfig) -> str:
    """The run's name: the first `RUN_ID_DIGITS` hex digits of the SHA-256 of `config` as resolved -- the file,
    `config_defaults` and `-o` overrides -- less `RUN_ID_EXCLUDED`, as sorted JSON.

    Two configurations that compute alike share an id, whatever their key order or how a value was written; any key
    outside `RUN_ID_EXCLUDED` set otherwise names another run.

    Args:
        config: the run's `YAMLConfig`.

    Returns:
        The id, e.g. `3f9c2a1e`.
    """
    resolved = config.to_dict()
    for key in RUN_ID_EXCLUDED:
        *sections, last = key.split(".")
        section = resolved
        for name in sections:
            section = section.get(name, {}) if isinstance(section, dict) else {}
        if isinstance(section, dict):
            section.pop(last, None)
    text = json.dumps(canonical_run_id_value(resolved), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()[:RUN_ID_DIGITS]


def run_config_name(identity: str) -> str:
    """The resolved configuration's file in its run's directory: `config_<id>.yaml`."""
    return f"config_{identity}.yaml"


def run_log_name(identity: str, prefix: str = "cnamaste") -> str:
    """The run's log in its directory: `cnamaste_<id>.log`; a decode's, `prefix` `cnamaste_decode`."""
    return f"{prefix}_{identity}.log"


def write_tsv(directory, name, df=None, header=True, index=False, index_label=None, prefix=""):
    """`df` as a TSV to `directory/name`, gzipped where `name` ends `.gz`; nothing where `directory` is None."""
    if directory is None:
        return

    opath = f"{directory}/{name}"

    if df is None:
        df = pd.DataFrame()

    logger.info(f"Writing {prefix} to {opath},\n{df.head()}")

    # NB a `.gz` path is gzipped with a zero timestamp, so the same table writes the same bytes.
    compression = {"method": "gzip", "mtime": 0} if str(opath).endswith(".gz") else None
    df.to_csv(opath, sep="\t", header=header, index=index, index_label=index_label, compression=compression)


def write_fig(directory, name, fig=None, transparent=True, bbox_inches="tight", dpi: int = get_dpi(), page=True):
    """Write a figure to `directory/name`. With `page`, in the page's face at the page's text size (`figure_style`), so
    a figure written on its own reads as the one on `combined.pdf`; what is drawn is untouched.

    Args:
        directory: the output directory, or None to write nothing, the figure closed.
        name: the file's name.
        fig: the figure, or None for an empty one.
        transparent: `savefig`'s.
        bbox_inches: `savefig`'s.
        dpi: the resolution of a PDF's images.
        page: True to restyle the figure as the page's.
    """
    if directory is None:
        plt.close(fig)
        return

    opath = f"{directory}/{name}"

    if fig is None:
        fig = plt.figure()
        fig.add_subplot(111)

    logger.info(f"Writing figure to:\n{opath}")

    with page_style():
        if page:
            restyle(fig)
        fig.savefig(
            opath,
            format="pdf",
            metadata=get_pdf_metadata(),
            transparent=transparent,
            bbox_inches=bbox_inches,
            dpi=dpi,
        )
    plt.close(fig)

    # NB the renderer each `Text` cached holds the PDF's rasters: a run's figures reached 11.8 GiB on
    #    CalicoST easy (`2d4ce9a9`) and the run was killed at this host's 14.3 GB; 4.2 GiB after.
    for text in fig.findobj(Text):
        text._renderer = None


def write_page(directory: str | None, name: str, figure: Any, dpi: int = get_dpi()) -> None:
    """Write `figure` at its page size, on white, nothing trimmed, in the format `name`'s suffix names: `.pdf` keeps
    every artist vector but images; `.png` rasterizes the page at `dpi`.

    Args:
        directory: the output directory, or None to write nothing.
        name: the file's name.
        figure: the page.
        dpi: the resolution of a PNG, and of a PDF's images.
    """
    if directory is None:
        plt.close(figure)
        return
    form = Path(name).suffix.lstrip(".").lower()
    metadata = get_pdf_metadata() if form == "pdf" else None
    figure.savefig(f"{directory}/{name}", format=form, dpi=dpi, facecolor="white", metadata=metadata)
    plt.close(figure)


def write_npz(directory, name, compress=True, **arrays):
    """`arrays` as an `.npz` to `directory/name`; nothing where `directory` is None.

    `compress` False stores the members rather than deflating them, which is what CSR entries want: a dense array's
    zeros deflate almost for free, and what is left once they are gone is what zlib cannot compress.
    """
    if directory is not None:
        saver = np.savez_compressed if compress else np.savez
        saver(f"{directory}/{name}", **arrays)


class OutputFile(NamedTuple):
    """A file `run_cnamaste` writes.

    Attributes:
        name: the file's name, any `{field}` filled by the caller.
        stage: the run's `runtime_phase` that writes it.
        writer: the function that writes it, `writer(directory, name, value, ...)`.
        columns: column -> dtype as written, the label tables indexed by `barcode`; None where the columns vary by run.
    """

    name: str
    stage: str
    writer: Callable[..., None]
    columns: dict[str, str] | None


LABEL_COLUMNS = {"sample_id": "object", "x": "int64", "y": "int64", "clone_label": "int64", "normal_candidate": "int64"}
"""A clone label table's columns and dtypes, indexed by `barcode`."""

SEGMENTATION_COLUMNS = {"CHR": "int64", "START": "int64", "END": "int64", "gene": "object"} | dict.fromkeys(
    get_levels(), "int64"
)
"""`segmentation.tsv.gz`'s columns and dtypes but `phase`: a gene's coordinates and name and its label at each level."""

SNP_SEGMENTATION_COLUMNS = {"CHR": "int64", "POS": "int64", "snp_id": "object", "gene": "object"} | dict.fromkeys(
    get_levels(), "int64"
)
"""`snp_segmentation.tsv.gz`'s columns and dtypes but `phase`: a SNP's coordinate, id and gene and its label at each
level."""

PHASING, SEGMENTATION, BAF = "phasing", "phased genomic segmentation", "baf-only clone & copy state inference"
RDR, INTEGER, FINAL = "baf/rdr clone & copy state inference", "integer copy number determination", "finalize"

OUTPUT_FILES = {
    file.name: file
    for file in (
        OutputFile("spot_umis.pdf", "prep.", write_page, None),
        OutputFile("pseudobulk_clones_genomic.pdf", PHASING, write_fig, None),
        OutputFile("he_clones.png", PHASING, write_page, None),
        OutputFile("prephasing_clones_spatial.pdf", PHASING, write_fig, None),
        OutputFile("prephasing_clones_genomic.pdf", PHASING, write_fig, None),
        OutputFile("postphasing_clones_genomic.pdf", PHASING, write_fig, None),
        OutputFile("postphasing_pseudobulk_clones_genomic.pdf", SEGMENTATION, write_fig, None),
        OutputFile("he_clones_genomic.pdf", SEGMENTATION, write_fig, None),
        OutputFile("initial_baf_spatial.pdf", BAF, write_fig, None),
        OutputFile("bafonly_clones_spatial.pdf", BAF, write_fig, None),
        OutputFile("bafonly_clones_genomic.pdf", BAF, write_fig, None),
        OutputFile("merged_bafonly_clones_spatial.pdf", BAF, write_fig, None),
        OutputFile("merged_bafonly_clones_genomic.pdf", BAF, write_fig, None),
        OutputFile("baf_clone_labels.tsv.gz", "normal candidate determination", write_tsv, LABEL_COLUMNS),
        OutputFile("rdr_baf_clones_spatial.pdf", RDR, write_fig, None),
        OutputFile("rdr_baf_clones_genomic.pdf", RDR, write_fig, None),
        OutputFile("merged_rdr_baf_clones_spatial.pdf", RDR, write_fig, None),
        OutputFile("merged_rdr_baf_clones_genomic.pdf", RDR, write_fig, None),
        OutputFile("clone_labels.tsv.gz", RDR, write_tsv, LABEL_COLUMNS),
        OutputFile("rdrbaf_final_nstates{n_states}_smp.npz", RDR, write_npz, None),
        OutputFile("{prefix}_genelevel.tsv", INTEGER, write_tsv, None),
        OutputFile("{prefix}_seglevel.tsv", INTEGER, write_tsv, None),
        OutputFile("{prefix}_perstate.tsv", INTEGER, write_tsv, None),
        OutputFile("clone_labels_int.tsv.gz", FINAL, write_tsv, LABEL_COLUMNS | {"field_argmax": "int64"}),
        OutputFile("segmentation.tsv.gz", FINAL, write_tsv, SEGMENTATION_COLUMNS | {"phase": "int64"}),
        OutputFile("snp_segmentation.tsv.gz", FINAL, write_tsv, SNP_SEGMENTATION_COLUMNS | {"phase": "int64"}),
        OutputFile("real_clones_genomic.pdf", FINAL, write_fig, None),
        OutputFile("clones_genomic.pdf", FINAL, write_fig, None),
        OutputFile("clones_spatial.pdf", FINAL, write_fig, None),
        OutputFile("copy_number_profile.pdf", FINAL, write_fig, None),
        OutputFile("combined.pdf", FINAL, write_page, None),
        OutputFile("real_combined.pdf", FINAL, write_page, None),
    )
}
"""Each file `run_cnamaste` writes by its name, in the order the run writes it; reached by `get_output_file_meta`."""


def get_output_file_meta(name: str) -> OutputFile:
    """Return the file `run_cnamaste` writes as `name`.

    Args:
        name: the file's name, as `OUTPUT_FILES` keys it.

    Returns:
        Its `OutputFile`: stage, writer and columns.
    """
    return OUTPUT_FILES[name]


def get_run_dir(config: Any = None) -> str:
    """`run_<id>/` under `paths.output_dir`, `<id>` the configuration's `results.run_id`: a name that states no
    parameter, and that no two configurations computing differently share."""
    if config is None:
        config = get_global_config()

    return f"{config.paths.output_dir}/run_{run_id(config)}/"


USE_EXISTING_HOURS = 6
"""`--use_existing`: a run whose final fit was written within this many hours is kept rather than run again, the
window `batch_zenodo_run.sh`'s `USE_EXISTING` has always taken. The id holds the configuration, not the code, so an
older fit is taken to be another commit's."""


def get_use_existing_hours() -> int:
    """Return the use existing hours, 6."""
    return USE_EXISTING_HOURS


def existing_fit(config: Any) -> Path | None:
    """The final fit of the run `config` names, written within `USE_EXISTING_HOURS`, or None.

    Args:
        config: the run's resolved `YAMLConfig`.

    Returns:
        The `rdrbaf_final_nstates*_smp.npz` path, or None where there is none that recent.
    """
    cutoff = time.time() - USE_EXISTING_HOURS * 3600
    fits = sorted(Path(get_run_dir(config)).glob("rdrbaf_final_nstates*_smp.npz"))
    return next((fit for fit in fits if fit.stat().st_mtime >= cutoff), None)


def configure_output_dir(config: Any = None) -> tuple[str, str]:
    """The run's directory and its `plots/`, made under `paths.output_dir`, which `YAMLConfig.check_paths` requires."""
    output_dir = get_run_dir(config)

    if not (poutput_dir := Path(output_dir)).exists():
        logger.info(f"Creating {output_dir}")

        poutput_dir.parent.mkdir(exist_ok=True)
        poutput_dir.mkdir(exist_ok=True)

    plots_dir = f"{output_dir}/plots/"

    if not (pplots_dir := Path(plots_dir)).exists():
        logger.info(f"Creating {plots_dir}")
        pplots_dir.mkdir(exist_ok=True)

    return output_dir, plots_dir
