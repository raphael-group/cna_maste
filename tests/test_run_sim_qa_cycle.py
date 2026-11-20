"""`batch_zenodo_run.sh` and the production cycle it feeds: the batch -> `run_cnamaste` -> its run directory ->
`run_sim_qa`, against CalicoST's own results.

**The batch's mechanics.** The shipped script runs by bash on a scratch study (`zenodo_batch.stage`); `uv` is a stub
that records its argv and computes nothing, `gshuf` GNU `shuf` where absent. The referees are the files on disk, the
argv the stub recorded, and the command `run_single_job` is documented to build (`zenodo_batch.batch_command`). Tests
that need more than the shipped solvers and seed run a copy with `SOLVERS` / `RANDOM_SEEDS` replaced, the only edit.
`QUIET` is `test_batch_quiet.py`'s.

**The cycle.** The shipped batch, its `SOLVERS` and `RANDOM_SEEDS` as shipped, runs `run_cnamaste` for real on the
`CYCLE_SIMS` fixtures `--sims` selects: CalicoST easy (`2d4ce9a9`) and CalicoST hard (`8797710b`), the dev tier every
other fixture row runs on, from their committed samples; `--sims` names outside `CYCLE_SIMS` are ignored; `uv` is
a stub that execs this environment's `run_cnamaste` with the argv the batch gave it. CalicoST's own results on the same
samples (`BENCHMARKED`: easy and hard, `tests/data/benchmarks/<sample>/calicost.tar.xz`, CalicoST `c1abcae` at its shipped
configuration) are laid out as its run directories, and `run_sim_qa` scores both against the planted truth, CalicoST
as the baseline. CalicoST harder (`a9a68913`) is not captured and is not a cycle fixture.

Referees: the overrides the batch passes, read back from the configuration each run writes beside its outputs
(`config_<id>.yaml`); the run id `run_sim_qa --config` computes, against the one each run named its directory; the
callers each run's log names for its initializer, label solver and decoder; `sim_runs`' paths; and the clone ARI
CalicoST's archive records (`run.json`), which the baseline rows must reproduce.

Departures, stated: the shipped configuration's `references` name its author's host, so the staged copy names this
host's GRCh38 resources (`sims.grch38`) and nothing else; CalicoST's archive keeps only its unconstrained table,
`cnv_seglevel.tsv`, so the baseline is read at `--calicost_ploidy unconstrained`. Cost: eight full runs, about 50 s on
easy and 90 s on hard each, so its rows carry `end2end`, off by default (`-m end2end` runs them); skipped where OpenGM or
the GRCh38 resources are absent.
"""

from __future__ import annotations

import itertools
import json
import os
import re
import shutil
import sys
import tarfile
from pathlib import Path

import pandas as pd
import pytest
import yaml

import cnamaste
from cnamaste import external
from cnamaste.qa.metrics import read_estimate
from cnamaste.qa.sim_runs import default_solver, run_dir_of, solver_dir, solver_run_ids
from cnamaste.qa.sims import get_resources, get_samples, grch38, sample as sample_of, selected_sims
from cnamaste.results import run_config_name, run_log_name
from cnamaste.scripts import run_sim_qa as qa
from computed_once import computed_once
from zenodo_batch import (
    Study,
    batch_command,
    batch_solvers,
    get_header,
    get_ids,
    get_pdf,
    get_random_seeds,
    get_template_id,
    get_template_sample,
    qa_command,
    run,
    stage,
    study_at,
    table_of,
    template,
)

SOLVERS = batch_solvers()
TWO = ("opengm_expansion-lattice-lattice", "icm-calicost-calicost")
QA_SOLVER = "opengm_expansion-lattice-lattice"
"""`batch_zenodo_run.sh`'s `QA_SOLVER`, stated here."""
"""Two solvers for a copy of the script: both decoder fields, an OpenGM and a native label solver."""


# -- the batch's mechanics: `uv` records its argv ---------------------------------------------------------------


def jobs(argvs: list[list[str]]) -> list[tuple[str, int, str]]:
    """`(sample, seed, solver)` of each recorded command, read back from its `-o` values."""
    found = []
    for argv in argvs:
        o = dict(a.split("=", 1) for a in argv[4::2])
        solver = Path(o["paths.output_dir"]).name
        found.append((Path(o["paths.output_dir"]).parent.name, int(o["hmrf.random_seed"]), solver))
    return found


def test_one_sheet_per_sample_with_the_template_s_tokens_substituted(tmp_path: Path) -> None:
    """Three samples: `zenodo_sample_sheets/` holds exactly `zenodo_<sample>_sheet.tsv` for each, its text the
    template's with `TEMPLATE_SAMPLE` and `TEMPLATE_ID` replaced by the sample, its two directories the sample's own.
    Checked against `str.replace` on the template and the study's directories; fails where a sheet keeps a token,
    names another sample's inputs, or one is missing."""
    study = stage(tmp_path, list(get_ids()[:3]))
    done = run(study)
    assert done.returncode == 0, done.stderr
    sheets = study.work / "zenodo_sample_sheets"
    assert sorted(p.name for p in sheets.iterdir()) == sorted(f"zenodo_{s}_sheet.tsv" for s in get_ids()[:3])
    for sample in get_ids()[:3]:
        text = (study.work / study.sheet_of(sample)).read_text()
        assert text == template(study.root).replace(get_template_sample(), sample).replace(get_template_id(), sample)
        row = text.splitlines()[1].split("\t")
        assert row[1] == sample and {Path(row[2]), Path(row[3])} == {study.root / "simulated_data_related" / sample}


@pytest.mark.parametrize("token", ["sample", "id"])
def test_a_template_without_its_token_is_refused_before_any_job(tmp_path: Path, token: str) -> None:
    """A template sheet missing `TEMPLATE_SAMPLE` or `TEMPLATE_ID`: exit 1 naming the token, no `uv` call, no output
    directory. Checked against the stub's record and the file system; fails where every sample would run the
    template's inputs, or the sheet's id, under its own name."""
    sheet = template(tmp_path / "sim")
    missing = get_template_sample() if token == "sample" else get_template_id()
    sheet = sheet.replace(missing, "something_else")
    study = stage(tmp_path, list(get_ids()[:2]), sheet=sheet)
    done = run(study)
    assert done.returncode == 1 and f"does not contain {missing}" in done.stderr
    assert study.recorded() == [] and not (study.root / "nomixing_cnamaste_related").exists()


@pytest.mark.parametrize("fault", ["another root", "no sample directory"])
def test_a_sheet_naming_no_input_directory_is_refused_before_any_job(tmp_path: Path, fault: str) -> None:
    """A template whose directories are under another root (as a sheet written for the other host), or do not name
    the sample at all: exit 1 naming the sheet and the fault, no `uv` call, no output directory. Fails where a job is
    started on inputs that do not exist."""
    if fault == "another root":
        sheet, refused = template(tmp_path / "elsewhere"), "which does not exist."
    else:
        sheet = f"{get_header()}NULL\t{get_template_id()}\t{tmp_path}/{get_template_sample()}_inputs\t{tmp_path}/\n"
        refused = "names no directory for"
    study = stage(tmp_path, list(get_ids()[:2]), sheet=sheet)
    done = run(study)
    assert done.returncode == 1 and refused in done.stderr and "zenodo_sample_sheets/zenodo_" in done.stderr
    assert study.recorded() == [] and not (study.root / "nomixing_cnamaste_related").exists()


def test_each_job_runs_run_cnamaste_with_run_single_job_s_six_overrides(tmp_path: Path) -> None:
    """The shipped script on two samples: one `uv` call per sample, seed and shipped solver, each exactly
    `run run_cnamaste zenodo_sim_config.yaml` and the six `-o` `run_single_job` builds, `paths.output_dir`
    `ROOT/nomixing_cnamaste_related/<sample>/<solver>`, made before the call; exit 0 and `All <n> ... completed`.
    Checked against `batch_command`; fails where an override is dropped, reordered or mis-mapped, or a run is sent
    to a directory `run_sim_qa` does not read."""
    study = stage(tmp_path, list(get_ids()[:2]))
    done = run(study)
    assert done.returncode == 0, done.stderr
    expected = [batch_command(study, s, r, v) for s in get_ids()[:2] for r in get_random_seeds() for v in SOLVERS]
    assert sorted(study.recorded()) == sorted(expected) and study.scorings() == []
    for sample, solver in itertools.product(get_ids()[:2], SOLVERS):
        assert (study.root / "nomixing_cnamaste_related" / sample / solver).is_dir()
    assert f"All {len(expected)} cna-maste jobs completed." in done.stdout


def test_the_jobs_are_samples_by_seeds_by_solvers(tmp_path: Path) -> None:
    """A copy with two solvers and seeds `(0, 3)` on two samples: eight calls, one per `(sample, seed, solver)` of the
    product, each command `batch_command`'s; `8 jobs.` and `8 of 8 runs complete 100%`. Fails where a solver, seed or
    sample is skipped or repeated, or a job's fields are crossed with another's."""
    study = stage(tmp_path, list(get_ids()[:2]))
    done = run(study, solvers=TWO, seeds=(0, 3))
    assert done.returncode == 0, done.stderr
    product = list(itertools.product(get_ids()[:2], (0, 3), TWO))
    assert sorted(jobs(study.jobs())) == sorted(product)
    assert sorted(study.jobs()) == sorted(batch_command(study, *job) for job in product)
    assert "8 jobs." in done.stdout and "8 of 8 runs complete 100%" in done.stdout


def test_max_sample_caps_the_samples_and_the_shuffle_is_seeded(tmp_path: Path) -> None:
    """Four samples, `MAX_SAMPLE=2`: two distinct samples run, sheets for those two alone, `Found 2 sample ids`; a
    second batch picks the same two. Checked against the samples staged; fails where the cap is ignored, applied after
    the sheets, or the selection is not reproducible from `SEED`."""
    picked = []
    for attempt in ("a", "b"):
        study = stage(tmp_path / attempt, list(get_ids()))
        done = run(study, MAX_SAMPLE="2")
        assert done.returncode == 0, done.stderr
        samples = sorted({job[0] for job in jobs(study.jobs())})
        assert len(samples) == 2 and set(samples) <= set(get_ids()) and "Found 2 sample ids" in done.stdout
        assert sorted(p.name for p in (study.work / "zenodo_sample_sheets").iterdir()) == sorted(
            f"zenodo_{s}_sheet.tsv" for s in samples
        )
        picked.append(samples)
    assert picked[0] == picked[1]


@pytest.mark.parametrize("use_existing", [True, False])
def test_use_existing_reaches_run_cnamaste(tmp_path: Path, use_existing: bool) -> None:
    """A copy with one solver: `USE_EXISTING=true` adds `--use_existing` to the job's command and `false` (the
    default) does not; the job runs either way, `run_cnamaste` deciding from the run its configuration names
    (`test_run_id.py`). Checked against `batch_command`; fails where the flag is lost or given unasked."""
    study = stage(tmp_path, [get_ids()[0]])
    solver, seed = SOLVERS[0], get_random_seeds()[0]
    done = run(study, solvers=(solver,), USE_EXISTING=str(use_existing).lower())
    assert done.returncode == 0, done.stderr
    assert study.recorded() == [batch_command(study, get_ids()[0], seed, solver, use_existing)]


@pytest.mark.parametrize("fail", [False, True])
def test_qa_scores_each_seed_once_the_jobs_end(tmp_path: Path, fail: bool) -> None:
    """`QA=true` on a copy with seeds `(0, 3)`: after every job, one `run_sim_qa --config` per seed, each
    `qa_command`'s; a failing job still exits 1 after the scoring, which names the runs it could not score.
    `QA=false` scores nothing. Fails where the scoring runs before a job, misses a seed, or masks a failed job."""
    study = stage(tmp_path, list(get_ids()[:2]))
    done = run(
        study, solvers=(SOLVERS[0],), seeds=(0, 3), QA="true", **({"STUB_FAIL": f"/{get_ids()[1]}/"} if fail else {})
    )
    assert done.returncode == (1 if fail else 0), done.stderr
    recorded = study.recorded()
    assert recorded[-2:] == [qa_command(study, 0, QA_SOLVER), qa_command(study, 3, QA_SOLVER)]
    assert study.scorings() == recorded[-2:] and len(study.jobs()) == 4
    quiet = stage(tmp_path / "off", list(get_ids()[:2]))
    assert run(quiet, solvers=(SOLVERS[0],)).returncode == 0 and quiet.scorings() == []


def test_qa_scores_qa_solver_against_calicost_and_each_other_solver(tmp_path: Path) -> None:
    """`QA=true` on a copy with three solvers, `QA_SOLVER` (`opengm_expansion-lattice-lattice`) last, and seeds
    `(0, 3)`: per seed, three `run_sim_qa` calls, `QA_SOLVER` against `calicost`, then against the other two in
    `SOLVERS`' order. Fails where the scored solver follows `SOLVERS`' order rather than `QA_SOLVER`, a solver is
    missed, or `QA_SOLVER` is its own baseline."""
    three = (TWO[1], SOLVERS[1], QA_SOLVER)
    study = stage(tmp_path, list(get_ids()[:1]))
    done = run(study, solvers=three, seeds=(0, 3), QA="true")
    assert done.returncode == 0, done.stderr
    expected = [
        qa_command(study, seed, QA_SOLVER, baseline) for seed in (0, 3) for baseline in ("calicost", *three[:2])
    ]
    assert study.scorings() == expected


def test_qa_without_qa_solver_among_solvers_is_refused_before_any_job(tmp_path: Path) -> None:
    """`QA=true` on a copy whose `SOLVERS` lacks `QA_SOLVER`: exit 1 naming it, before any `uv` call; `QA=false` runs.
    Fails where a batch spends its jobs and then has nothing to score."""
    study = stage(tmp_path, [get_ids()[0]])
    done = run(study, solvers=(TWO[1],), QA="true")
    assert done.returncode == 1 and f"QA_SOLVER {QA_SOLVER}" in done.stderr and study.recorded() == []
    assert run(study, solvers=(TWO[1],)).returncode == 0


def test_a_failing_job_is_summarised_and_the_batch_exits_1(tmp_path: Path) -> None:
    """Three samples on a copy with one solver, the stub failing (status 3) on the second's commands: every job still runs, the summary on
    stderr lists `<sample> <seed> <label> <init> <decoder> (rc=3)` for that one alone, `1 of 3 cna-maste jobs
    failed`, exit 1, and no `completed` line. Fails where a failure stops the batch, is lost, or names another job."""
    study = stage(tmp_path, list(get_ids()[:3]))
    failing = get_ids()[1]
    done = run(study, solvers=(SOLVERS[0],), STUB_FAIL=f"/{failing}/")
    assert done.returncode == 1
    assert sorted(job[0] for job in jobs(study.jobs())) == sorted(get_ids()[:3])
    fields = " ".join(SOLVERS[0].split("-"))
    assert f"1 of 3 cna-maste jobs failed:\n  {failing} {get_random_seeds()[0]} {fields} (rc=3)\n" in done.stderr
    assert done.stderr.count("(rc=3)") == 2, "the job's FAILED line and its summary row"
    assert "completed." not in done.stdout and "3 of 3 runs complete 100%" in done.stdout


@pytest.mark.parametrize(
    ("solver", "refused"),
    [
        ("opengm_icm-lattice", "is not <label_solver>-<initializer>-<decoder>"),
        ("opengm_icm-lattice-lattice-x", "is not <label_solver>-<initializer>-<decoder>"),
        ("simplex-lattice-lattice", "label solver simplex is not one of"),
        ("icm-kmeans-lattice", "initializer kmeans is not one of"),
        ("icm-lattice-hmm_ab", "decoder hmm_ab is not one of"),
    ],
)
def test_a_solver_the_run_does_not_take_is_refused_before_any_job(tmp_path: Path, solver: str, refused: str) -> None:
    """A copy whose `SOLVERS` holds one malformed or unknown solver: exit 1 naming it, before any sheet or `uv` call.
    Fails where a batch spends its first jobs before reaching a solver `run_cnamaste` refuses."""
    study = stage(tmp_path, [get_ids()[0]])
    done = run(study, solvers=(SOLVERS[0], solver))
    assert done.returncode == 1 and f"solver {solver}" in done.stderr and refused in done.stderr
    assert study.recorded() == [] and not (study.work / "zenodo_sample_sheets").exists()


# -- the cycle: `run_cnamaste` for real on CalicoST easy and hard, CalicoST's results as the baseline --------------

SPEC = Path(__file__).parent
BENCHMARKS = SPEC / "data" / "benchmarks"

CYCLE_SIMS = ("2d4ce9a9", "8797710b")
"""The fixtures the cycle can run: CalicoST easy and hard, each from its committed sample; harder is not captured."""

BENCHMARKED = {"2d4ce9a9": "easy", "8797710b": "hard"}
"""Fixture hash -> its CalicoST archive's name under `BENCHMARKS`: the cycle samples scored against CalicoST."""


@pytest.fixture(scope="module")
def samples(pytestconfig: pytest.Config) -> dict[str, str | None]:
    """The cycle's samples, id -> CalicoST archive name or None: the `CYCLE_SIMS` fixtures `--sims` selects, as every
    other fixture row takes them (`conftest.selected`). CalicoST easy and hard; `--sims` cannot add harder."""
    chosen = [h for h in selected_sims(pytestconfig.getoption("--sims")) if h in CYCLE_SIMS]
    if not chosen:
        pytest.skip(f"--sims selects no cycle fixture ({list(CYCLE_SIMS)})")
    return {get_samples()[h]: BENCHMARKED.get(h) for h in chosen}


def benchmarked(samples: dict[str, str | None]) -> dict[str, str]:
    """`samples` with a CalicoST archive, id -> archive name."""
    return {s: name for s, name in samples.items() if name is not None}


CALICOST_RUN = "clone3_rectangle0_w1.0"
"""CalicoST's run directory name for its archived run: `n_clones 3`, rectangle 0, `spatial_weight 1.0`."""

CALICOST_PLOIDY = "unconstrained"
"""The table CalicoST's archive keeps, `cnv_seglevel.tsv`."""

UV_RUN = f"""#!/bin/bash
# NB `uv run run_cnamaste ...` and `uv run run_sim_qa ...` as this environment's console scripts, with the argv the
#    batch gave them; then a run's OpenGM session directory, `<paths.output_dir>/tmp/`, removed: no row reads it, and
#    eight fill a session's disk.
[[ "$1" == run && ( "$2" == run_cnamaste || "$2" == run_sim_qa ) ]] || {{ echo "stub uv: $*" >&2; exit 9; }}
script=$2
shift 2
{Path(sys.executable).parent}/$script "$@"
rc=$?
for arg in "$@"; do
    [[ "$arg" == paths.output_dir=* ]] && rm -rf "${{arg#paths.output_dir=}}/tmp"
done
exit $rc
"""

INITIALIZER_CALLER = {
    "lattice": "cnamaste.initialize.genomic.get_lattice_states",
    "calicost": "cnamaste.initialize.genomic.get_calicost_states",
}
"""The function each `hmm.initializer` the batch sets is logged from."""

SOLVER_ALGORITHM = {"opengm_expansion": "expansion", "opengm_icm": "icm"}
"""The OpenGM algorithm `cnamaste.external.solve` logs for each label solver the batch ships."""

DECODER_CALLER = {
    "lattice": "cnamaste.ab_decode.decode.decode_integer_copies",
    "calicost": "cnamaste.calicost.adapters.calicost_fixdiploid",
}
"""The function each decoder field's `ab_decode.decoder` is logged from."""


def caller(log: str, name: str) -> bool:
    """Whether `log` holds a line logged from `name` (`<module>.<function>:<line> - `)."""
    return re.search(rf" - {re.escape(name)}:\d+ - ", log) is not None


@pytest.fixture(scope="module")
def cycle(tmp_path_factory: pytest.TempPathFactory, samples: dict[str, str | None]) -> Study:
    """The shipped batch over `samples`, `run_cnamaste` for real on each committed sample (`qa.sims.sample`, or
    `CNAMASTE_SIMS`'); CalicoST's runs laid out beside. Run once per session, every worker reading it."""
    if not external.available():
        pytest.skip(str(external.unavailable()))
    resources = grch38()
    if resources is None:
        pytest.skip("CalicoST's GRCh38_resources not found; set $CNAMASTE_GRCH38")
    return study_at(computed_once(tmp_path_factory, "run_sim_qa_cycle", lambda out: run_cycle(out, samples, resources)))


def run_cycle(out: Path, samples: dict[str, str | None], resources: Path) -> None:
    """The shipped batch staged in `out` over `samples`, run with QA on."""
    study = stage(out, list(samples), uv=UV_RUN)
    hashes = {get_samples()[h]: h for h in CYCLE_SIMS}
    for sample, name in samples.items():
        directory = study.root / "simulated_data_related" / sample
        directory.rmdir()
        directory.symlink_to(sample_of(hashes[sample]).resolve(), target_is_directory=True)
        if name is None:
            continue
        calicost = study.root / "nomixing_calicost_related" / sample / CALICOST_RUN
        calicost.mkdir(parents=True)
        with tarfile.open(BENCHMARKS / name / "calicost.tar.xz") as archive:
            archive.extractall(calicost, filter="data")
    config = yaml.safe_load((study.work / "zenodo_sim_config.yaml").read_text())
    config["references"].update({k: str(resources / f) for k, f in get_resources().items()})
    config["references"]["annotation_file"] = str(study.work / "unused.gtf.gz")
    (study.work / "zenodo_sim_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    # NB the batch runs in `study.work`, where a relative `PYTHONPATH` names no checkout: this one's, absolute.
    package_root = str(Path(cnamaste.__file__).resolve().parents[1])
    done = run(study, QA="true", CALICOST_PLOIDY=CALICOST_PLOIDY, PYTHONPATH=package_root)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    assert not list((study.root / "nomixing_cnamaste_related").glob("*/*/tmp")), "UV_RUN removes each session"


def run_ids(cycle: Study) -> dict[str, str]:
    """Each solver's run id as `run_sim_qa --config` computes it from the batch's configuration and seed."""
    return solver_run_ids(cycle.work / "zenodo_sim_config.yaml", [f"hmrf.random_seed={get_random_seeds()[0]}"], SOLVERS)


@pytest.fixture
def study_root(cycle: Study, tmp_path: Path) -> Path:
    """A copy of the cycle's `ROOT` without the batch's `qa/`, for one row's `run_sim_qa` outputs, its files hard links: no row writes into a run's
    files, only beside them, and eight runs copied per row filled a 3 GB session's disk."""
    root = tmp_path / "sim"
    shutil.copytree(
        cycle.root, root, symlinks=True, ignore=shutil.ignore_patterns("plots", "qa"), copy_function=os.link
    )
    return root


def scored(root: Path, *flags: str) -> int:
    """`run_sim_qa` over `root`, CalicoST at its archived table, with `flags`."""
    return qa.main(["--root", str(root), "--calicost_ploidy", CALICOST_PLOIDY, *flags])


@pytest.mark.end2end
def test_each_run_writes_the_configuration_the_batch_asked_for(cycle: Study, samples: dict[str, str | None]) -> None:
    """Each `(sample, solver)`'s run writes `config_<id>.yaml` beside its outputs, and it carries the six overrides
    `run_single_job` passes: the sample's sheet, `paths.output_dir`, the seed, the label solver, initializer and
    decoder key, each as its YAML value (the seed an int). Checked against `batch_command`'s `-o` values; fails where an
    override is lost on the way into the run, resolved to a default or as text, or the run writes no configuration."""
    for sample, solver in itertools.product(samples, SOLVERS):
        command = batch_command(cycle, sample, get_random_seeds()[0], solver)
        overrides = dict(token.split("=", 1) for token in command[4::2])
        identity = run_ids(cycle)[solver]
        written = yaml.safe_load(
            (run_dir_of(solver_dir(cycle.root, sample, solver), identity) / run_config_name(identity)).read_text()
        )
        for key, value in overrides.items():
            section, name = key.split(".")
            assert written[section][name] == yaml.safe_load(value), f"{sample} {solver}: {key}"


@pytest.mark.end2end
def test_each_run_s_log_names_the_initializer_label_solver_and_decoder_it_called(
    cycle: Study, samples: dict[str, str | None]
) -> None:
    """Each run's `cnamaste_<id>.log`, in its run directory, holds lines from its initializer's function and its
    decoder's, and from no other shipped one; every OpenGM call it logs is its label solver's algorithm; the batch's
    `cnamaste_batch_<id>.log`, moved into the run's directory, ends the run (`Done in`). Checked against
    `INITIALIZER_CALLER`, `SOLVER_ALGORITHM` and `DECODER_CALLER`; fails where a solver field is configured but
    another method runs, or the batch's log is not filed with its run."""
    for sample, solver in itertools.product(samples, SOLVERS):
        label, initializer, decoder = solver.split("-")
        out = solver_dir(cycle.root, sample, solver)
        identity = run_ids(cycle)[solver]
        log = (run_dir_of(out, identity) / run_log_name(identity)).read_text()
        for field, callers in ((initializer, INITIALIZER_CALLER), (decoder, DECODER_CALLER)):
            assert caller(log, callers[field]), f"{sample} {solver}: no line from {callers[field]}"
            assert not any(caller(log, c) for f, c in callers.items() if f != field), f"{sample} {solver}: {field}"
        algorithms = set(re.findall(r" - cnamaste\.external\.solve:\d+ - OpenGM (\w+)", log))
        assert algorithms == {SOLVER_ALGORITHM[label]}, f"{sample} {solver}: OpenGM {sorted(algorithms)}"
        run_dir = run_dir_of(out, identity)
        assert "Done in" in (run_dir / f"cnamaste_batch_{identity}.log").read_text()
        assert not (out / "cnamaste_batch.log.tmp").exists(), "the batch's log moved into the run's directory"


@pytest.mark.end2end
def test_each_run_lands_where_run_sim_qa_reads_it(cycle: Study, samples: dict[str, str | None]) -> None:
    """For every `(sample, solver)`, `run_dir_of(solver_dir(ROOT, sample, solver))` -- `run_sim_qa`'s lookup -- is the
    one finished `run_<id>` directory, `<id>` the one `run_sim_qa --config` computes from the batch's configuration and
    seed, and its tables read; `solvers_with_runs` finds exactly the batch's solvers. Fails where the batch's
    `OUT_BASE` and the reader's `solver_dir` part, or the id the run took differs from the one the reader computes."""
    ids = run_ids(cycle)
    assert len(set(ids.values())) == len(SOLVERS), ids
    for sample, solver in itertools.product(samples, SOLVERS):
        run_dir = run_dir_of(solver_dir(cycle.root, sample, solver))
        assert run_dir.name == f"run_{ids[solver]}" and re.fullmatch(r"run_[0-9a-f]{8}", run_dir.name)
        read_estimate(run_dir)
    assert qa.solvers_with_runs(cycle.root) == sorted(SOLVERS)


@pytest.mark.end2end
def test_the_batch_scores_qa_solver_against_calicost_and_each_other_solver(
    cycle: Study, samples: dict[str, str | None]
) -> None:
    """The batch's own QA, `QA=true` at `CALICOST_PLOIDY` `unconstrained`, into `qa/<id>/`, `<id>` the configuration's
    run id: exactly the table, `config_<id>.yaml`, and one figure per baseline, `QA_SOLVER` against `calicost` and
    against each other shipped solver, each a PDF; the table one row per sample and solver and one CalicoST row per
    sample. Checked against `SOLVERS`, `QA_SOLVER` and `samples`; fails where a baseline is missed, the figure is drawn
    for another solver, CalicoST is left out, or the outputs land elsewhere."""
    (study,) = [p for p in (cycle.root / "qa").iterdir()]
    baselines = ["calicost", *(s for s in SOLVERS if s != QA_SOLVER)]
    figures = {f"cnamaste_{QA_SOLVER}_baseline_{b}_one_shot_validation.pdf" for b in baselines}
    assert {p.name for p in study.iterdir()} == {
        "all_sample_metrics_one_shot.tsv",
        run_config_name(study.name),
    } | figures
    assert all((study / f).read_bytes()[:5] == get_pdf() for f in figures)
    table = pd.read_csv(study / "all_sample_metrics_one_shot.tsv", sep="\t")
    assert sorted(zip(table["sample_id"], table["solver"].fillna("calicost"), strict=True)) == sorted(
        [(s, v) for s, v in itertools.product(samples, SOLVERS)] + [(s, "calicost") for s in benchmarked(samples)]
    )


@pytest.mark.end2end
def test_run_sim_qa_scores_every_run_and_calicost_s_baseline(study_root: Path, samples: dict[str, str | None]) -> None:
    """`run_sim_qa --baseline_solver calicost`: one cnamaste row per sample and solver and one CalicoST row per
    sample, every panel in [0, 1]; CalicoST's `clone_ari` and `int_clone_ari` its archive's `run.json` `ari` to 4
    decimals (0.5303 easy, 0.069 hard), `clone_labels.tsv` scored against the same truth. Fails where a run is missed,
    read from another sample, or CalicoST's clones are scored otherwise than its own record."""
    assert scored(study_root, "--baseline_solver", "calicost") == 0
    table = table_of(study_root)
    rows = sorted(zip(table["sample_id"], table["method"], table["solver"].fillna(""), strict=True))
    assert rows == sorted(
        [(s, "cnamaste", v) for s, v in itertools.product(samples, SOLVERS)]
        + [(s, "calicost", "") for s in benchmarked(samples)]
    )
    panels = ["clone_ari", "int_clone_ari", "exact_altered", "sensitivity", "specificity"]
    assert ((table[panels] >= 0) & (table[panels] <= 1)).all().all(), table[panels]
    for sample, name in benchmarked(samples).items():
        with tarfile.open(BENCHMARKS / name / "calicost.tar.xz") as archive:
            recorded = json.load(archive.extractfile("run.json"))["ari"]
        row = table[(table["sample_id"] == sample) & (table["method"] == "calicost")].iloc[0]
        assert round(row["clone_ari"], 4) == round(recorded, 4) == round(row["int_clone_ari"], 4), (sample, row)


@pytest.mark.end2end
@pytest.mark.parametrize("one_shot", [False, True])
def test_run_sim_qa_writes_its_documented_outputs(study_root: Path, one_shot: bool) -> None:
    """Without and with `--baseline_solver calicost`, then `--one_shot`: `qa/` holds exactly
    `all_sample_metrics[_one_shot].tsv` and `cnamaste_<solver>[_baseline_calicost][_one_shot]_validation.pdf` per
    solver, each figure a PDF. Fails where a run's figure lands elsewhere or under another solver's name."""
    suffix = "_one_shot" if one_shot else ""
    flags = ["--one_shot"] if one_shot else []
    names = {f"all_sample_metrics{suffix}.tsv"}
    assert scored(study_root, "--overwrite", *flags) == 0
    assert scored(study_root, "--baseline_solver", "calicost", "--overwrite", *flags) == 0
    for solver in SOLVERS:
        names |= {
            f"cnamaste_{solver}{suffix}_validation.pdf",
            f"cnamaste_{solver}_baseline_calicost{suffix}_validation.pdf",
        }
    assert {p.name for p in (study_root / "qa").iterdir()} == names
    assert all((study_root / "qa" / n).read_bytes()[:5] == get_pdf() for n in names if n.endswith(".pdf"))


@pytest.mark.end2end
def test_a_second_run_sim_qa_is_refused_without_overwrite(study_root: Path) -> None:
    """A second `run_sim_qa` over the same `ROOT`: status 2, the table and figures byte for byte; `--overwrite`
    replaces them. Fails where a rerun silently replaces the last study's outputs."""
    assert scored(study_root) == 0
    before = {p: p.read_bytes() for p in (study_root / "qa").iterdir()}
    with pytest.raises(SystemExit) as stopped:
        scored(study_root)
    assert stopped.value.code == 2 and {p: p.read_bytes() for p in before} == before
    assert scored(study_root, "--overwrite") == 0


@pytest.mark.end2end
def test_a_solver_the_batch_did_not_run_is_refused(study_root: Path) -> None:
    """Status 2, nothing written: `--baseline_solver` naming a solver no batch job ran (`opengm_trws-lattice-lattice`,
    and the default solver where the batch does not run it), and the `ROOT` with its cnamaste runs removed. Fails where
    a solver's directory is read without a run in it, or a study with no run writes an empty table."""
    absent = ["opengm_trws-lattice-lattice", *([default_solver] if default_solver not in SOLVERS else [])]
    for solver in absent:
        with pytest.raises(SystemExit) as stopped:
            scored(study_root, "--baseline_solver", solver)
        assert stopped.value.code == 2
    shutil.rmtree(study_root / "nomixing_cnamaste_related")
    with pytest.raises(SystemExit) as stopped:
        scored(study_root)
    assert stopped.value.code == 2 and not (study_root / "qa").exists()


@pytest.mark.end2end
def test_run_sim_qa_config_scores_the_runs_it_names_beside_another(
    cycle: Study, study_root: Path, samples: dict[str, str | None]
) -> None:
    """Each solver's run copied beside itself as another configuration's (`run_00000000`): without `--config` every
    solver is refused, two runs in its directory, and only CalicoST's rows are scored; with it, the table equals the one
    scored before the copy, written to `qa/<id>/` beside `config_<id>.yaml`, `<id>` the configuration's own. Fails where
    a run of another configuration is scored as this one's, or the study's outputs land where another configuration's
    would."""
    config = cycle.work / "zenodo_sim_config.yaml"
    flags = [
        "--config",
        str(config),
        "-o",
        f"hmrf.random_seed={get_random_seeds()[0]}",
        "--baseline_solver",
        "calicost",
    ]
    assert scored(study_root, *flags) == 0
    (study,) = [p for p in (study_root / "qa").iterdir()]
    before = (study / "all_sample_metrics.tsv").read_bytes()
    assert re.fullmatch(r"[0-9a-f]{8}", study.name) and (study / run_config_name(study.name)).is_file()
    for sample, solver in itertools.product(samples, SOLVERS):
        own = run_dir_of(solver_dir(study_root, sample, solver))
        shutil.copytree(own, own.parent / "run_00000000", copy_function=os.link)
    assert scored(study_root) == 0
    assert set(table_of(study_root)["method"]) == {"calicost"}, "every solver refused: two runs, no configuration"
    assert scored(study_root, *flags, "--overwrite") == 0
    assert (study / "all_sample_metrics.tsv").read_bytes() == before
