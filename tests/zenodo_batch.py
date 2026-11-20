"""`batch_zenodo_run.sh` run by bash in a scratch study, external programs stubbed on `PATH`; shared by the
specification tests of the batch and of the cycle it feeds `run_sim_qa`.

A study is `<root>/simulated_data_related/<sample>/`, a working directory holding a template sample sheet and the
shipped `zenodo_sim_config.yaml`, and a `bin/` whose `uv` is the caller's stub and whose `gshuf` is GNU `shuf` where
the host has no `gshuf` (Linux). `ROOT`, `USE_EXISTING` and `QA` reach the script through the environment, as it reads
them, `QA` false unless a test asks for it, so the stub records the jobs alone;
`SOLVERS` and `RANDOM_SEEDS` are bash arrays, which no environment carries, so a test that asks for others runs a copy
of the script with those two declarations replaced and nothing else.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest

from cnamaste.qa.sims import get_project

BATCH = get_project() / "batch_zenodo_run.sh"
"""The batch that writes the runs this script reads: `run_sim_qa` scores every solver it runs."""


def batch_solvers() -> tuple[str, ...]:
    """`batch_zenodo_run.sh`'s `SOLVERS` default, in the order it declares them."""
    body = re.search(r"^SOLVERS=\((.*?)^\)", BATCH.read_text(), re.DOTALL | re.MULTILINE)
    assert body is not None, f"{BATCH.name} declares no SOLVERS array."
    return tuple(body.group(1).split())


def table_of(root: Path, one_shot: bool = False, all_shot: bool = False) -> pd.DataFrame:
    """The table `run_sim_qa` wrote under `root`."""
    suffix = "_one_shot" if one_shot else "_all_shot" if all_shot else ""
    return pd.read_csv(root / "qa" / f"all_sample_metrics{suffix}.tsv", sep="\t")


IDS = tuple(f"numcnas1.2_cnasize5e7_ploidy2_random{r}" for r in range(4))
"""Simulation-named samples, which the validation figure groups by and `run_sim_qa` therefore requires."""


def get_ids() -> tuple[str, ...]:
    """Return the ids, `IDS`."""
    return IDS


PDF = b"%PDF-"


def get_pdf() -> bytes:
    """Return the pdf, b"%PDF-"."""
    return PDF


TEXT = BATCH.read_text()
TEMPLATE_SAMPLE = re.search(r"^TEMPLATE_SAMPLE=(\S+)$", TEXT, re.MULTILINE).group(1)


def get_template_sample() -> str:
    """Return the template sample, `TEMPLATE_SAMPLE`."""
    return TEMPLATE_SAMPLE


TEMPLATE_ID = re.search(r"^TEMPLATE_ID=(\S+)$", TEXT, re.MULTILINE).group(1)


def get_template_id() -> str:
    """Return the template id, `TEMPLATE_ID`."""
    return TEMPLATE_ID


RANDOM_SEEDS = tuple(int(s) for s in re.search(r"^RANDOM_SEEDS=\(([^)]*)\)", TEXT, re.MULTILINE).group(1).split())
"""`batch_zenodo_run.sh`'s template tokens and seeds, as the script declares them."""


def get_random_seeds() -> tuple[int, ...]:
    """Return the random seeds, `RANDOM_SEEDS`."""
    return RANDOM_SEEDS


CONFIG = get_project() / "zenodo_sim_config.yaml"
DECODER_KEYS = {"lattice": "hmm_ab", "calicost": "calicost"}
"""A solver's decoder field -> the `ab_decode.decoder` it sets, stated here rather than read from the script."""
HEADER = "bam\tsample_id\tspaceranger_dir\tsnp_dir\n"


def get_header() -> str:
    """Return the header, `HEADER`."""
    return HEADER


UV_RECORDER = """#!/bin/bash
# NB each call's arguments, one per line, to $CALLS/<n>; status 3 where they name $STUB_FAIL.
n=$(ls "$CALLS" | wc -l | tr -d ' ')
printf '%s\\n' "$@" > "$CALLS/$(printf %04d "$n")"
if [[ -n "$STUB_FAIL" && "$*" == *"$STUB_FAIL"* ]]; then
    echo "stub uv: failing as asked" >&2
    exit 3
fi
"""
"""A `uv` that records its argv and computes nothing."""


@dataclass
class Study:
    """A scratch study: `root` the batch's `ROOT`, `work` its working directory, `calls` where the stub records."""

    root: Path
    work: Path
    bin: Path
    calls: Path

    def sheet_of(self, sample: str) -> Path:
        """The sheet the batch writes for `sample`, relative to `work` as the batch names it."""
        return Path("zenodo_sample_sheets") / f"zenodo_{sample}_sheet.tsv"

    def recorded(self) -> list[list[str]]:
        """Every argv the stub `uv` received, in call order."""
        return [p.read_text().splitlines() for p in sorted(self.calls.iterdir())]

    def jobs(self) -> list[list[str]]:
        """The `run run_cnamaste` argvs among `recorded`."""
        return [argv for argv in self.recorded() if argv[:2] == ["run", "run_cnamaste"]]

    def scorings(self) -> list[list[str]]:
        """The `run run_sim_qa` argvs among `recorded`."""
        return [argv for argv in self.recorded() if argv[:2] == ["run", "run_sim_qa"]]


def template(root: Path) -> str:
    """The shipped template's shape: one row naming `TEMPLATE_ID` and `TEMPLATE_SAMPLE`'s directory under `root`."""
    directory = f"{root}/simulated_data_related/{TEMPLATE_SAMPLE}/"
    return f"{HEADER}NULL\t{TEMPLATE_ID}\t{directory}\t{directory}\n"


def study_at(tmp_path: Path) -> Study:
    """The study `stage` lays out under `tmp_path`."""
    return Study(*(tmp_path / name for name in ("sim", "work", "bin", "calls")))


def stage(tmp_path: Path, samples: list[str], uv: str = UV_RECORDER, sheet: str | None = None) -> Study:
    """A study with `samples`' directories, the template `sheet` (default `template(root)`), and `uv` on `PATH`."""
    study = study_at(tmp_path)
    root, work, bin_dir, calls = study.root, study.work, study.bin, study.calls
    for path in (work, bin_dir, calls):
        path.mkdir(parents=True)
    for sample in samples:
        (root / "simulated_data_related" / sample).mkdir(parents=True, exist_ok=True)
    (work / "zenodo_sample_sheet.tsv").write_text(template(root) if sheet is None else sheet)
    shutil.copy(CONFIG, work / CONFIG.name)
    (bin_dir / "uv").write_text(uv)
    (bin_dir / "uv").chmod(0o755)
    if shutil.which("gshuf") is None:
        if shutil.which("shuf") is None:
            pytest.skip("neither gshuf nor shuf on PATH")
        (bin_dir / "gshuf").write_text('#!/bin/bash\nexec shuf "$@"\n')
        (bin_dir / "gshuf").chmod(0o755)
    return study


def script(study: Study, solvers: tuple[str, ...] | None = None, seeds: tuple[int, ...] | None = None) -> Path:
    """The shipped script, or where `solvers` or `seeds` are given a copy with `SOLVERS` / `RANDOM_SEEDS` replaced,
    each declaration found exactly once."""
    if solvers is None and seeds is None:
        return BATCH
    text = TEXT
    if solvers is not None:
        declared = "SOLVERS=(\n" + "\n".join(solvers) + "\n)"
        text, n = re.subn(r"^SOLVERS=\(.*?^\)", declared, text, flags=re.DOTALL | re.MULTILINE)
        assert n == 1, "SOLVERS declared once"
    if seeds is not None:
        declared = f"RANDOM_SEEDS=({' '.join(map(str, seeds))})"
        text, n = re.subn(r"^RANDOM_SEEDS=\([^)]*\)", declared, text, flags=re.MULTILINE)
        assert n == 1, "RANDOM_SEEDS declared once"
    copy = study.work.parent / "batch_zenodo_run.sh"
    copy.write_text(text)
    return copy


def run(
    study: Study, solvers: tuple[str, ...] | None = None, seeds: tuple[int, ...] | None = None, **env: str
) -> subprocess.CompletedProcess[str]:
    """The batch, by bash, in `study.work`, with `ROOT` the study's (trailing `/`, as the shipped default) and `env`."""
    inherited = {
        k: v
        for k, v in os.environ.items()
        if k not in ("QUIET", "MAX_SAMPLE", "USE_EXISTING", "STUB_FAIL", "QA", "CONFIG", "CALICOST_PLOIDY")
    }
    environment = (
        inherited
        | {
            "PATH": f"{study.bin}:{os.environ['PATH']}",
            "ROOT": f"{study.root}/",
            "CALLS": str(study.calls),
            "QA": "false",
        }
        | env
    )
    command = ["bash", str(script(study, solvers, seeds))]
    return subprocess.run(command, capture_output=True, text=True, env=environment, cwd=study.work, check=False)


def batch_command(study: Study, sample: str, seed: int, solver: str, use_existing: bool = False) -> list[str]:
    """`uv`'s argv for one job as `run_single_job` is documented to build it, stated here, decoder by `DECODER_KEYS`;
    `--use_existing` last where `USE_EXISTING` is true."""
    label, initializer, decoder = solver.split("-")
    out = f"{study.root}//nomixing_cnamaste_related/{sample}/{solver}"
    over_rides = [
        f"paths.sample_sheet={study.sheet_of(sample)}",
        f"paths.output_dir={out}",
        f"hmrf.random_seed={seed}",
        f"hmrf.label_solver={label}",
        f"hmm.initializer={initializer}",
        f"ab_decode.decoder={DECODER_KEYS[decoder]}",
    ]
    return [
        "run",
        "run_cnamaste",
        CONFIG.name,
        *(token for o in over_rides for token in ("-o", o)),
        *(["--use_existing"] if use_existing else []),
    ]


def qa_command(study: Study, seed: int, solver: str, baseline: str = "calicost") -> list[str]:
    """`uv`'s argv for the batch's scoring of one seed, `solver` against `baseline`, `QA` true, as the script is
    documented to build it."""
    return [
        "run",
        "run_sim_qa",
        "--root",
        f"{study.root}/",
        "--config",
        CONFIG.name,
        "-o",
        f"hmrf.random_seed={seed}",
        "--solver",
        solver,
        "--baseline_solver",
        baseline,
        "--calicost_ploidy",
        "diploid",
        "--one_shot",
        "--overwrite",
    ]
