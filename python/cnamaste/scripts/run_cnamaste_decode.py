"""Decode a finished run again, from its written files: `run_cnamaste_decode CONFIG [-o key=value ...]`.

`run_cnamaste` itself runs on `CONFIG`, its inputs read as it reads them. The inferences whose results `RUN_DIR`
holds are read from it rather than run (`READ`): the phase per block (`segmentation.tsv.gz`), the BAF-only clones
(`baf_clone_labels.tsv.gz`), the normal spots (`clone_labels.tsv.gz`) and the final fit
(`rdrbaf_final_nstates<n>_smp.npz`). Every other step is the run's own code, so the bins, the decode, the tables and
the pages are as `run_cnamaste` makes them; the figures before the decode are not drawn. `RUN_DIR` is the run
`CONFIG` wrote under its `paths.output_dir`, and the decode overwrites its files there.

`RUN_DIR` is `run_<id>`, `<id>` `CONFIG`'s `results.run_id` before the decode's `-o`: the run `CONFIG` names. An
`ab_decode` override is another configuration, so its decode is written to its own `run_<id>`, and only a decode
with none overwrites the run.

`-o` overrides as `run_cnamaste -o` does: `ab_decode.<key>=<value>` for the decoder, and
`paths.output_dir=<dir>` to write the decode elsewhere rather than over the run. The tables a decode does not change are
checked equal to `RUN_DIR`'s (`UNCHANGED`) before anything is written, so a `CONFIG` or inputs other than the run's
stop the decode and leave the run as it was.

The decode writes under `<paths.output_dir>/tmp/decode_*` and is moved out of it on success, file by file. Nothing
here removes a directory, so `tmp` keeps a staging per decode -- emptied where the decode landed, whole where a
refusal stopped it -- until it is removed by hand.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
import tempfile
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cnamaste import inference
from cnamaste.config import YAMLConfig, start_time
from cnamaste.emission import CnaEmissionConfounders, get_unconfounded
from cnamaste.hmm_params import CnaHmmParameters
from cnamaste.hmrf_fit import CloneAssignment, HMMProfile, HmrfFit
from cnamaste.io import read_text_table
from cnamaste.logger import get_logger
from cnamaste.results import run_id, run_log_name, write_fig, write_page
from cnamaste.scripts import run_cnamaste as rc
from cnamaste.segments import get_dropped

logger = get_logger("cnamaste.scripts.run_cnamaste_decode", start_time=start_time)

READ = (
    "initial_phase_given_partition",
    "run_hmrf",
    "get_merged_fit",
    "determine_normal_candidates",
    "get_initial_labelling",
    "get_initial_parameters",
    "get_reindexed_fit",
)
"""`inference`'s names a decode replaces with what `RUN_DIR` wrote: the phasing, both clone searches and their
merges, the normal spots, the RDR labelling, the HMM starts and the clones' reordering, which the written fit already
carries."""

DRAWN_IN = "finalize"
"""The `runtime_phase` whose figures a decode draws: the pages after the decode; every figure before it is skipped."""

UNCHANGED = ("segmentation.tsv", "baf_clone_labels.tsv", "clone_labels.tsv")
"""The tables a decode writes again unchanged: the bins and phase, the BAF-only and the final clones with the normal
spots. Each is checked equal to `RUN_DIR`'s."""


def fit_file(run_dir: Path) -> Path:
    """`RUN_DIR`'s `rdrbaf_final_nstates<n>_smp.npz`.

    Raises:
        SystemExit: where there is not exactly one.
    """
    found = sorted(Path(run_dir).glob("rdrbaf_final_nstates*_smp.npz"))
    if len(found) != 1:
        raise SystemExit(f"run_cnamaste_decode: {run_dir} holds {len(found)} rdrbaf_final_nstates*_smp.npz, one wanted")
    return found[0]


def read_fit(path: Path) -> HmrfFit:
    """The final fit `run_cnamaste` wrote, after `get_reindexed_fit`, as the `HmrfFit` it wrote.

    Args:
        path: the `.npz`; its `assignment_before_reindex` is a pickled object, read as the run wrote it.

    Returns:
        The result.
    """
    with np.load(path, allow_pickle=True) as found:
        get = {k: found[k] for k in found.files}
    before = get.get("assignment_before_reindex")
    before = before.item() if before is not None and before.dtype == object and before.ndim == 0 else before
    n_clones = np.unique(get["new_assignment"]).size
    shift = get.get("new_log_mu_shift")
    shift = np.zeros(n_clones) if shift is None or shift.dtype == object else shift
    return HmrfFit(
        parameters=CnaHmmParameters(
            *(get[f"new_{k}"] for k in ("log_startprob", "log_mu", "p_binom", "alphas", "taus"))
        ),
        confounders=CnaEmissionConfounders(np.zeros((n_clones, 1)), np.asarray(shift).reshape(-1, 1)),
        log_transmat=get["new_log_transmat"],
        profile=HMMProfile(log_gamma=get["log_gamma"], pred_cnv=get["pred_cnv"]),
        llf=float(get["llf"]),
        n_states=int(get["n_states"]),
        assignment=CloneAssignment(
            assignment_before_reindex=before,
            prev_assignment=get["prev_assignment"],
            new_assignment=get["new_assignment"],
            total_llf=float(get["total_llf"]),
        ),
    )


def written(run_dir: Path, name: str, column: str) -> pd.DataFrame:
    """`RUN_DIR`'s table `name` (plain or `.gz`), refused where it lacks `column`: a run written before it was.

    Raises:
        SystemExit: naming the table and the column.
    """
    try:
        found = read_text_table(Path(run_dir) / name, sep="\t", comment="#")
    except FileNotFoundError as error:
        raise SystemExit(f"run_cnamaste_decode: no {error}") from error
    if column not in found:
        raise SystemExit(
            f"run_cnamaste_decode: {run_dir}/{name} has no `{column}` column: the run predates it; rerun it"
        )
    return found


def block_phase(segmentation: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """The phase indicator per initial block and the blocks per phasing segment, as `initial_phase_given_partition`
    returned them, from the written segmentation.

    Args:
        segmentation: `segmentation.tsv.gz`, a row per gene.

    Returns:
        `(phase_indicator, refined_lengths)`.

    Raises:
        SystemExit: where a block holds two phases.
    """
    kept = segmentation[segmentation["initial_blocks"] != get_dropped()]
    per_block = kept.groupby("initial_blocks").agg(
        phase=("phase", "first"), phases=("phase", "nunique"), segment=("phasing_segments", "first")
    )
    if (per_block["phases"] != 1).any() or not np.array_equal(per_block.index, np.arange(len(per_block))):
        raise SystemExit("run_cnamaste_decode: segmentation.tsv.gz does not give one phase to each block 0, 1, ...")
    return per_block["phase"].to_numpy(np.int64), np.bincount(per_block["segment"].to_numpy(np.int64))


@contextlib.contextmanager
def substituted(run_dir: Path | Callable[[], Path]) -> Iterator[dict[str, Any]]:
    """`run_cnamaste`'s module with `READ`'s names reading `RUN_DIR` and figures drawn in `DRAWN_IN` alone, restored
    on exit.

    Args:
        run_dir: the run's directory, or a function giving it, called when the first of `READ`'s names is.

    Yields:
        What the decode saw, by name: `barcodes`, the spot order `load_input_data` gave; `run_dir`, the directory read.
    """
    seen: dict[str, Any] = {}
    read: dict[str, Any] = {}

    def files() -> dict[str, Any]:
        if not read:
            found = Path(run_dir() if callable(run_dir) else run_dir)
            segmentation = written(found, "segmentation.tsv", "phase")
            read.update(
                baf=written(found, "baf_clone_labels.tsv", "normal_candidate").set_index("barcode"),
                final=written(found, "clone_labels.tsv", "normal_candidate").set_index("barcode"),
                fit=read_fit(fit_file(found)),
                phase=block_phase(segmentation),
            )
            seen["run_dir"] = found
        return read

    def by_barcode(frame: pd.DataFrame, column: str) -> np.ndarray:
        return frame[column].reindex(seen["barcodes"]).to_numpy()

    def load_input_data(*args: Any, **kwargs: Any) -> Any:
        out = original["load_input_data"](*args, **kwargs)
        seen["barcodes"] = np.asarray(out[1]).astype(str)
        return out

    def initial_phase_given_partition(*args: Any, **kwargs: Any) -> Any:
        phase, refined_lengths = files()["phase"]
        return {}, phase, refined_lengths

    def run_hmrf(counts: Any, *args: Any, stage: str, **kwargs: Any) -> Any:
        fit = files()["fit"]
        if stage != "baf":
            return fit
        # NB the BAF-only search's result as far as run_cnamaste reads it before the normal call, which is read too:
        #    its clones, and a model whose BAF profiles nothing then reads.
        labels = by_barcode(files()["baf"], "clone_label").astype(np.int64)
        n_clones = np.unique(labels).size
        return HmrfFit(
            parameters=fit.parameters._replace(p_binom=np.zeros(fit.n_states)),
            confounders=get_unconfounded(n_clones),
            log_transmat=fit.log_transmat,
            profile=HMMProfile(
                log_gamma=np.zeros((n_clones, counts.n_segments, fit.n_states)),
                pred_cnv=np.zeros((n_clones, counts.n_segments), dtype=np.int64),
            ),
            llf=fit.llf,
            n_states=fit.n_states,
            assignment=CloneAssignment(new_assignment=labels),
        )

    def determine_normal_candidates(*args: Any, **kwargs: Any) -> np.ndarray:
        return by_barcode(files()["final"], "normal_candidate").astype(bool)

    def get_initial_labelling(counts: Any, *args: Any, stage: str, **kwargs: Any) -> Any:
        if stage == "phasing":
            return original["get_initial_labelling"](counts, *args, stage=stage, **kwargs)
        return np.zeros(counts.n_spots, dtype=np.int64)

    def gated(draw: Any) -> Any:
        def drawn(*args: Any, **kwargs: Any) -> Any:
            return draw(*args, **kwargs) if getattr(inference.logger, "runtime_phase", None) == DRAWN_IN else None

        return drawn

    def get_output_file_meta(name: str) -> Any:
        output_file = original["get_output_file_meta"](name)
        # NB every figure, PDF or PNG (he_clones.png, drawn with a slide), as its plot_ returns None outside DRAWN_IN.
        drawn = output_file.writer in (write_fig, write_page)
        return output_file._replace(writer=gated(output_file.writer)) if drawn else output_file

    replaced = {
        "load_input_data": load_input_data,
        "initial_phase_given_partition": initial_phase_given_partition,
        "run_hmrf": run_hmrf,
        "get_merged_fit": lambda res, *args, **kwargs: res,
        "determine_normal_candidates": determine_normal_candidates,
        "get_initial_labelling": get_initial_labelling,
        "get_initial_parameters": lambda *args, **kwargs: None,
        "get_reindexed_fit": lambda res: res,
        "get_output_file_meta": get_output_file_meta,
    }
    replaced |= {name: gated(getattr(inference, name)) for name in dir(inference) if name.startswith("plot_")}
    original = {name: getattr(inference, name) for name in replaced}
    try:
        for name, value in replaced.items():
            setattr(inference, name, value)
        yield seen
    finally:
        for name, value in original.items():
            setattr(inference, name, value)


def unchanged(run_dir: Path, decoded_dir: Path, names: tuple[str, ...] = UNCHANGED) -> list[str]:
    """The tables `names`, `UNCHANGED`'s by default, where the decode's differ from `RUN_DIR`'s, compared as frames.

    Returns:
        The names that differ, empty where every one is equal.
    """
    return [
        name
        for name in names
        if not read_text_table(Path(run_dir) / name, sep="\t", comment="#").equals(
            read_text_table(Path(decoded_dir) / name, sep="\t", comment="#")
        )
    ]


def moved_out(source: Path, target: Path) -> int:
    """Move every file under `source` onto `target`, keeping what only `target` holds.

    Both are under one output directory, hence one filesystem, so each file is renamed rather than copied: a decode
    is moved out of its staging, which is left empty of it.

    Args:
        source: the decode's directory.
        target: where the run's files are.

    Returns:
        The files moved.
    """
    files = [path for path in sorted(source.rglob("*")) if path.is_file()]
    for path in files:
        destination = target / path.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(path, destination)
    return len(files)


def run_dir_of(config: Any, output_dir: Path) -> Path:
    """The run directory `config` wrote under `output_dir`: `run_<id>`, `<id>` its `results.run_id`.

    Raises:
        SystemExit: where there is none.
    """
    found = Path(output_dir) / f"run_{run_id(config)}"
    if not found.is_dir():
        raise SystemExit(f"run_cnamaste_decode: no {found.name} under {output_dir}, the run this configuration names")
    return found


def get_staged_run(
    config_path: Path, run_dir: Path | None, outputdir: Path | None, over_rides: list[str], label: str
) -> tuple[Path, Path, Path]:
    """Run `run_cnamaste` on `config_path` with `RUN_DIR`'s inferences read (`substituted`) into a staging directory,
    `<outputdir>/tmp/<label>_*`, which nothing here removes.

    Args:
        config_path: the run's configuration.
        run_dir: the finished run's directory, `run_dir_of` the configuration's where None.
        outputdir: where the result is to be written, `paths.output_dir` where None.
        over_rides: `-o` overrides for the staged run, checked by the caller.
        label: names the staging and its log line, e.g. `decode`.

    Returns:
        `RUN_DIR`, the staged run's directory and `outputdir`, resolved.

    Raises:
        SystemExit: where the configuration names no run, or `RUN_DIR` lacks what `READ` reads.
    """
    config = YAMLConfig.from_file(str(config_path))
    outputdir = Path(outputdir if outputdir is not None else config.paths.output_dir).resolve()
    written_under = Path(config.paths.output_dir).resolve()
    # NB the run is found before anything is made, so a configuration that names none stops with nothing written.
    source = Path(run_dir).resolve() if run_dir is not None else run_dir_of(config, written_under)
    outputdir.mkdir(parents=True, exist_ok=True)
    root = outputdir / "tmp"
    root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f"{label}_", dir=root))
    logger.info(f"staging the {label} in {staging}; nothing here removes a directory, so {root} is yours to remove")
    with substituted(source) as seen:
        rc.run_cnamaste(str(config_path), over_rides=[f"paths.output_dir={staging}", *over_rides])
    staged = next(d for d in staging.glob("run_*") if d.is_dir())
    return seen["run_dir"], staged, outputdir


def rename_staged_log(staged: Path, identity: str, prefix: str) -> None:
    """Rename the staged run's log `<prefix>_<identity>.log`, so a result moved in place keeps the run's own beside it.

    Args:
        staged: the staged run's directory.
        identity: the id the log is renamed to.
        prefix: e.g. `cnamaste_decode`.
    """
    log = staged / run_log_name(staged.name.removeprefix("run_"))
    if log.is_file():
        os.replace(log, staged / run_log_name(identity, prefix=prefix))


def decode(
    config_path: Path, run_dir: Path | None = None, outputdir: Path | None = None, over_rides: list[str] | None = None
) -> Path:
    """Decode `RUN_DIR` again (`get_staged_run`), then write its run directory under `outputdir`.

    The decode is staged under `<outputdir>/tmp/decode_*` and moved into place only once `UNCHANGED`'s tables are
    found equal to `RUN_DIR`'s, so a decode that stops overwrites nothing. By default `RUN_DIR` is the run the
    configuration wrote and `outputdir` its `paths.output_dir`: the decode overwrites the run's files in place. Its
    log is `cnamaste_decode_<id>.log` in its run directory, beside the run's own.

    Args:
        config_path: the run's configuration.
        run_dir: the finished run's directory, `run_dir_of` the configuration's where None.
        outputdir: where the decode's run directory is written, `paths.output_dir` where None.
        over_rides: `ab_decode.<key>=<value>` each.

    Returns:
        The decode's run directory.

    Raises:
        SystemExit: for an override outside `ab_decode`, a `RUN_DIR` without what `READ` reads, or a table `UNCHANGED`
            names that differs.
        RUN_DIR: is then untouched, the staging kept.
    """
    over_rides = list(over_rides or [])
    outside = [o for o in over_rides if not o.startswith("ab_decode.")]
    if outside:
        raise SystemExit(
            f"run_cnamaste_decode: -o takes ab_decode.<key>=<value> or paths.output_dir=<dir>, not {outside}"
        )
    run_dir, decoded, outputdir = get_staged_run(config_path, run_dir, outputdir, over_rides, "decode")
    moved = unchanged(run_dir, decoded)
    if moved:
        raise SystemExit(
            f"run_cnamaste_decode: {', '.join(moved)} differ from {run_dir}'s: CONFIG or the inputs are not the "
            f"run's; the decode it refused is {decoded}"
        )
    rename_staged_log(decoded, decoded.name.removeprefix("run_"), "cnamaste_decode")
    target = outputdir / decoded.name
    files = moved_out(decoded, target)
    # NB one path per line, so each copies whole.
    logger.info(
        f"Decoded:\n"
        f"    run: {run_dir}\n"
        f"    decode: {target}\n"
        f"    moved: {files} files\n"
        f"    unchanged: {', '.join(UNCHANGED)}"
    )
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("config_path", type=Path, help="the run's configuration")
    parser.add_argument(
        "--over_rides",
        "-o",
        action="append",
        default=[],
        help="ab_decode.<key>=<value>, or paths.output_dir=<dir> to write the decode there; repeated",
    )
    args = parser.parse_args(argv)
    # NB paths.output_dir moves the decode's output, not the run it reads: that is CONFIG's own.
    output = [o.split("=", 1)[1] for o in args.over_rides if o.startswith("paths.output_dir=")]
    rest = [o for o in args.over_rides if not o.startswith("paths.output_dir=")]
    try:
        decode(args.config_path, outputdir=Path(output[-1]) if output else None, over_rides=rest)
    except SystemExit as refused:
        # NB a refusal is a line of the run's log, in cnamaste's format, not a bare message.
        if isinstance(refused.code, str):
            logger.error(refused.code.removeprefix("run_cnamaste_decode: "))
            return 1
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
