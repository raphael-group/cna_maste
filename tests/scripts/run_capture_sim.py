"""Capture one `run_cnamaste` run of a committed fixture into `sim/<name>_<hash>/capture/<name>_<hash>_capture.hdf5`.

    PYTHONPATH=python python tests/scripts/run_capture_sim.py easy_2d4ce9a9

The run is `zenodo_sim_config.yaml` with `get_overrides`, from a scratch directory holding links to the checkout's
`sim/` and `ref/` (`stage_links`), so every recorded path is relative and a replay resolves it the same way. numpy's
legacy generator is seeded with `ENTRY_POINT_SEED`; plots are stubbed.

Every call of a stage function (`SCHEDULE`) is recorded under its group, `/<NN_group>/<function>[_k]/{in,out,rng}`;
each evaluator (`EVALUATORS`) is recorded on its first call in the run, under the group of the stage last entered, its
`out` left out where `RECOMPUTED` names it. `/config` keeps the template, the calls in order, the inputs, the commit
and the timings; `/truth` the planted truth, `/files` each table `write_tsv` wrote and `/metrics` the run's
`qa.metrics.metrics`. The capture exits 1 where the file exceeds `[tool.cnamaste.capture] max_file_bytes`.
"""

from __future__ import annotations

import gzip
import importlib
import inspect
import json
import os
import subprocess
import sys
import tempfile
import time
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import h5py
import numpy as np

import cnamaste.inference as stages
import cnamaste.scripts.run_cnamaste as rc
from cnamaste.config import get_global_config
from cnamaste.fit_checks import get_fit_check_counts
from cnamaste.qa.capture import (
    Writer,
    get_entry_point_seed,
    get_gzip,
    get_output_dir,
    get_overrides,
    get_path_of,
    get_rng_state,
    get_template,
    put_rng,
    stage_links,
)
from cnamaste.qa.metrics import metrics, read_estimate
from cnamaste.qa.sim_runs import run_dir_of
from cnamaste.qa.sims import get_fixtures, get_project, input_files
from cnamaste.results import write_tsv
from cnamaste.truth import read_truth

MAX_FILE_BYTES: int = tomllib.loads((get_project() / "pyproject.toml").read_text())["tool"]["cnamaste"]["capture"][
    "max_file_bytes"
]


SCHEDULE: dict[str, list[str]] = {
    "load_input_data": ["00_inputs"],
    "get_sample_list": ["00_inputs"],
    "form_gene_snp_table": ["01_genes"],
    "assign_initial_blocks": ["02_blocks"],
    "summarize_counts_for_blocks": ["02_blocks"],
    "get_sitewise_transmat": ["02_blocks"],
    "get_initial_labelling": ["03_phasing", "08_rdr"],
    "get_initial_parameters": ["03_phasing", "05_baf", "08_rdr"],
    "initial_phase_given_partition": ["03_phasing"],
    "create_bin_ranges": ["04_bins", "07_rebin"],
    "summarize_counts_for_bins": ["04_bins", "07_rebin"],
    "construct_multislice_lattice_adjacency": ["04_bins"],
    "run_hmrf": ["05_baf", "08_rdr"],
    "get_merged_fit": ["05_baf", "08_rdr"],
    "get_assignment_frame": ["05_baf", "09_outputs", "09_outputs"],
    "determine_normal_candidates": ["06_normal"],
    "normal_baf_bin_filter": ["06_normal"],
    "binned_gene_snp": ["06_normal", "07_rebin"],
    "filter_normal_diffexp": ["06_normal"],
    "determine_normal_baseline": ["07_rebin"],
    "get_reindexed_fit": ["08_rdr"],
    "decode_integer_copies": ["09_outputs"],
    "calicost_diploid_decoder": ["09_outputs"],
    "get_merged_ab_decoded_clones": ["09_outputs"],
    "write_tsv": ["09_outputs"] * 8,
}
"""Each stage function `run_cnamaste` calls -> the group of each of its calls, in call order."""

EVALUATORS = (
    "cnamaste.emission:get_log_emission",
    "cnamaste.hmm_emission_tabled:get_clone_decoded_fields",
    "cnamaste.hmm_mstep:get_cost_and_gradient",
    "cnamaste.emission:get_log_mu_shift",
    "cnamaste.ab_decode.decode:get_clone_emission_loglikes",
    "cnamaste.ab_decode.decode:hmm_ab_decoder_dispersion_objective",
    "cnamaste.normal_spot:fit_normal_spot_bb",
    "cnamaste.spatio_genomic_counts:PseudobulkCounts.update_encoding",
)
"""The NB/BB evaluators, recorded on their first call in the run, which `tests/test_benchmarks.py` times."""

RECOMPUTED = (
    "cnamaste.emission:get_log_emission",
    "cnamaste.hmm_emission_tabled:get_clone_decoded_fields",
    "cnamaste.hmm_mstep:get_cost_and_gradient",
    "cnamaste.ab_decode.decode:get_clone_emission_loglikes",
)
"""The evaluators whose return is recomputed on replay, not stored: a dense `(n_states, n_obs)` matrix, or a JAX array."""


PLOTS = (
    "plot_clones_genomic",
    "plot_clones_spatial",
    "plot_combined_page",
    "plot_copy_number_profile",
    "plot_he_page",
    "plot_spot_umi_page",
)


def get_entry_names(function: str) -> list[str]:
    """Each call's entry name: `function`, or `function_k` where `SCHEDULE` puts several of its calls in one group.

    Args:
        function: a `SCHEDULE` key.

    Returns:
        The names, in call order.
    """
    groups = SCHEDULE[function]
    seen: dict[str, int] = {}
    names = []
    for group in groups:
        k = seen.get(group, 0)
        seen[group] = k + 1
        names.append(f"{group}/{function}_{k}" if groups.count(group) > 1 else f"{group}/{function}")
    return names


class Recorder:
    """Records the wrapped calls into one open file.

    Args:
        h5: the file.
    """

    def __init__(self, h5: h5py.File) -> None:
        self.h5 = h5
        self.writer = Writer(h5, compress=True)
        self.calls: dict[str, int] = {}
        self.stages: list[str] = []
        self.configs: list[str] = []
        self.group = ""
        self.files: dict[str, str] = {}
        self.evaluators: dict[str, str] = {}
        """Each recorded evaluator's `module:qualname` -> its call, `NN_group/qualname`."""
        self.overhead = 0.0

    def get_config_index(self) -> int:
        """The global config's index in `configs`, appended where new."""
        config = json.dumps(get_global_config().to_dict())
        if config not in self.configs:
            self.configs.append(config)
        return self.configs.index(config)

    def record(self, name: str, function: Callable[..., Any], path: str, args: Any, kwargs: Any) -> Any:
        """Call `function`, writing its arguments on entry and its return on exit under `name`, the return left out
        where `RECOMPUTED` names `path`.

        Args:
            name: the entry, `NN_group/function[_k]`.
            function: the original.
            path: what a replay calls, `module:qualname`.
            args: positional arguments, as the replay passes them.
            kwargs: keyword arguments.

        Returns:
            The return.
        """
        entered = time.perf_counter()
        group = self.h5.create_group(name)
        group.attrs.update({"function": path, "config": self.get_config_index()})
        put_rng(self.writer, group, get_rng_state())
        self.writer.put(group, "in", {"args": list(args), "kwargs": dict(kwargs)})
        self.stages.append(name)
        start = time.perf_counter()
        out = function(*args, **kwargs)
        seconds = time.perf_counter() - start
        if path not in RECOMPUTED:
            self.writer.put(group, "out", out)
        group.attrs["seconds"] = seconds
        self.overhead += time.perf_counter() - entered - seconds
        print(f"{name} {seconds:.2f} s", flush=True)
        return out

    def wrap_stage(self, function: str) -> None:
        """Rebind `inference.<function>` to record every call under its `SCHEDULE` entry.

        Args:
            function: a `SCHEDULE` key.
        """
        setattr(stages, function, self.get_recorded(function, getattr(stages, function)))

    def wrap_output_files(self) -> None:
        """Rebind `inference.get_output_file_meta`: each table `write_tsv` writes is recorded under `SCHEDULE`'s
        `write_tsv` entry, and no figure is written."""
        original, recorded = stages.get_output_file_meta, self.get_recorded("write_tsv", write_tsv)

        def get_output_file_meta(name: str) -> Any:
            output_file = original(name)
            if output_file.writer is write_tsv:
                return output_file._replace(writer=recorded)
            return output_file._replace(writer=lambda *_, **__: None) if name.endswith(".pdf") else output_file

        stages.get_output_file_meta = get_output_file_meta

    def get_recorded(self, function: str, original: Callable[..., Any]) -> Callable[..., Any]:
        """`original` recording every call under `function`'s `SCHEDULE` entry.

        Args:
            function: a `SCHEDULE` key.
            original: the function.

        Returns:
            The wrapper.
        """
        names = get_entry_names(function)

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            k = self.calls.get(function, 0)
            self.calls[function] = k + 1
            if k >= len(names):
                raise RuntimeError(f"{function}: call {k + 1}, where SCHEDULE names {len(names)}")
            self.group = names[k].split("/")[0]
            out = self.record(names[k], original, get_path_of(original), args, kwargs)
            if function == "write_tsv":
                self.files[Path(args[1]).name] = names[k]
            return out

        return wrapped

    def get_first_call(self, path: str, original: Callable[..., Any]) -> Callable[..., Any]:
        """`original` recording its first call in the run as `<group>/<qualname>`, the group of the stage last entered.

        Args:
            path: `module:qualname`, what a replay calls.
            original: the function.

        Returns:
            The wrapper.
        """
        qualname = path.partition(":")[2]

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            if not self.group or path in self.evaluators:
                return original(*args, **kwargs)
            self.evaluators[path] = f"{self.group}/{qualname}"
            return self.record(self.evaluators[path], original, path, args, kwargs)

        return wrapped

    def wrap_evaluator(self, path: str) -> None:
        """Rebind the evaluator at `path` wherever `cnamaste` looks it up: a class's method in its class, a
        function in every module holding it.

        Args:
            path: `module:qualname`.
        """
        module, _, qualname = path.partition(":")
        owner: Any = importlib.import_module(module)
        *outer, name = qualname.split(".")
        for part in outer:
            owner = getattr(owner, part)
        raw = inspect.getattr_static(owner, name)
        if isinstance(owner, type):
            function = raw.__func__ if isinstance(raw, (staticmethod, classmethod)) else raw
            wrapped = self.get_first_call(path, function)
            setattr(owner, name, type(raw)(wrapped) if isinstance(raw, (staticmethod, classmethod)) else wrapped)
            return
        wrapped = self.get_first_call(path, raw)
        for found in list(sys.modules.values()):
            if (found.__name__ or "").startswith("cnamaste") and getattr(found, name, None) is raw:
                setattr(found, name, wrapped)


def get_commit() -> str:
    """The checkout's commit, `-dirty` where `python/` has uncommitted edits."""
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=get_project(), capture_output=True, text=True, check=True)
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "python"], cwd=get_project(), capture_output=True, text=True, check=True
    )
    return head.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")


def put_tables(h5: h5py.File, run_dir: Path) -> None:
    """Store each table the run wrote under `/files`, byte for byte: every file in `run_dir` but its log, profile and
    configuration.

    Args:
        h5: the file.
        run_dir: the run's directory.
    """
    for path in sorted(run_dir.iterdir()):
        if path.is_file() and path.suffix not in (".log", ".perf", ".yaml"):
            data = np.frombuffer(path.read_bytes(), dtype=np.uint8)
            h5.require_group("files").create_dataset(path.name, data=data, chunks=data.shape, **get_gzip())


def put_truth(h5: h5py.File, sample: Path) -> None:
    """Store the planted truth gzipped under `/truth`.

    Args:
        h5: the file.
        sample: the fixture's directory.
    """
    for name in ("truth_acn_profile.tsv", "truth_clone_labels.tsv"):
        data = gzip.decompress((sample / f"{name}.gz").read_bytes())
        h5.require_group("truth").create_dataset(name, data=np.frombuffer(gzip.compress(data, mtime=0), dtype=np.uint8))


def main(fixture: str) -> int:
    """Capture `fixture`'s run.

    Args:
        fixture: the fixture's directory, `<name>_<hash>`.

    Returns:
        The exit status: 1 where the file is over `MAX_FILE_BYTES`.
    """
    sim_hash = fixture.rpartition("_")[2]
    assert get_fixtures()[sim_hash] == fixture, fixture
    sample = get_project() / "sim" / fixture
    out = sample / "capture" / f"{fixture}_capture.hdf5"
    overrides = get_overrides(fixture)
    with tempfile.TemporaryDirectory() as work:
        os.chdir(work)
        stage_links(Path(work))
        partial = Path(work) / out.name
        with h5py.File(partial, "w", libver="latest") as h5:
            recorder = Recorder(h5)
            wrapped = [name for name in SCHEDULE if hasattr(stages, name)]
            for name in wrapped:
                recorder.wrap_stage(name)
            recorder.wrap_output_files()
            for path in EVALUATORS:
                recorder.wrap_evaluator(path)
            for name in PLOTS:
                setattr(stages, name, lambda *_, **__: None)
            np.random.seed(get_entry_point_seed())
            start = time.perf_counter()
            rc.run_cnamaste(str(get_template()), over_rides=overrides)
            wall = time.perf_counter() - start
            run_dir = run_dir_of(Path(work) / get_output_dir())
            put_tables(h5, run_dir)
            found = metrics(read_estimate(run_dir), read_truth(sample)) | get_fit_check_counts()
            h5.create_group("metrics").attrs.update({key: float(value) for key, value in found.items()})
            put_truth(h5, sample)
            inputs = {name: str(path.relative_to(get_project())) for name, path in input_files(sample).items()}
            h5.create_group("config").attrs.update(
                {
                    "yaml": get_template().read_text(),
                    "overrides": json.dumps(overrides),
                    "fixture_hash": sim_hash,
                    "entry_point_seed": get_entry_point_seed(),
                    "inputs": json.dumps(inputs),
                    "commit": get_commit(),
                    "wrapped": json.dumps([*wrapped, *EVALUATORS]),
                    "evaluators": json.dumps(recorder.evaluators),
                    "recomputed": json.dumps(RECOMPUTED),
                    "stages": json.dumps(recorder.stages),
                    "configs": json.dumps(recorder.configs),
                    "files": json.dumps(recorder.files),
                    "timings": json.dumps({"wall_seconds": wall, "capture_seconds": recorder.overhead}),
                }
            )
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(partial.read_bytes())
    size = out.stat().st_size
    print(f"{out}: {size:_} bytes, {len(recorder.stages)} calls, run {wall:.1f} s ({recorder.overhead:.1f} s capture)")
    if size > MAX_FILE_BYTES:
        print(f"over [tool.cnamaste.capture] max_file_bytes, {MAX_FILE_BYTES:_}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
