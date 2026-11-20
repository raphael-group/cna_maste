"""`run_cnamaste`'s `-o key=value` overrides against the values the run reads, and the six `batch_zenodo_run.sh` passes.

`run_single_job` runs `run_cnamaste zenodo_sim_config.yaml` with `paths.sample_sheet`, `paths.output_dir`,
`hmrf.random_seed`, `hmrf.label_solver`, `hmm.initializer` and `ab_decode.decoder` set by `-o`, the last mapped
from the solver's decoder field by its own `case`. The referee is the command line itself: each value given there is
the value the run's readers (`label_solver_type`, `config.ab_decode.decoder`, `config.hmm.initializer`, the seed, `check_paths`)
resolve, read back from the shipped file with the override applied, and through `main` as the console script parses
it. The batch's `SOLVERS` and decoder mapping are read from the script, so a solver added there is checked here.

No stage runs: a run stops at `external.require` (an OpenGM solver, `$CNAMASTE_OPENGM_HOME` unset) or where it seeds
its generators, before any input is read.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numba import njit

from cnamaste import external
from cnamaste.config import YAMLConfig, label_solver_type
from cnamaste.qa.sims import get_project
from cnamaste.scripts import run_cnamaste as script

CONFIG = get_project() / "zenodo_sim_config.yaml"
BATCH = (get_project() / "batch_zenodo_run.sh").read_text()

SOLVERS = re.search(r"^SOLVERS=\(\n(.*?)\n\)", BATCH, re.DOTALL | re.MULTILINE).group(1).split()
"""`batch_zenodo_run.sh`'s solvers, `<label_solver>-<hmm.initializer>-<decoder>`, as the script lists them."""

DECODER_KEYS = dict(re.findall(r"^\s*(\w+)\) DECODER_KEY=(\w+) ;;", BATCH, re.MULTILINE))
"""`run_single_job`'s `case`: a solver's decoder field -> the `ab_decode.decoder` it passes."""

RESOLVED_DECODER = {"lattice": "hmm_ab", "calicost": "calicost"}
"""The decoder each field names, stated here rather than derived: `lattice` is the `(A, B)` HMM, `hmm_ab`."""


class Stop(Exception):
    """Raised where a run is stopped once it has its configuration."""


def batch_overrides(tmp_path: Path, seed: int, solver: str) -> list[str]:
    """The six `-o` `run_single_job` passes for `solver` and `seed`, with a sheet and an output directory that exist."""
    label, initializer, decoder = solver.split("-")
    sheet = tmp_path / "zenodo_sample_sheets" / "zenodo_Z001-U1_sheet.tsv"
    sheet.parent.mkdir(exist_ok=True)
    sheet.write_text("bam\tsample_id\tspaceranger_dir\tsnp_dir\n")
    out = tmp_path / "nomixing_cnamaste_related" / "Z001-U1" / solver
    out.mkdir(parents=True, exist_ok=True)
    return [
        f"paths.sample_sheet={sheet}",
        f"paths.output_dir={out}",
        f"hmrf.random_seed={seed}",
        f"hmrf.label_solver={label}",
        f"hmm.initializer={initializer}",
        f"ab_decode.decoder={DECODER_KEYS[decoder]}",
    ]


def argv(over_rides: list[str]) -> list[str]:
    """`run_cnamaste zenodo_sim_config.yaml -o ... -o ...`, as `run_single_job` spells it."""
    return ["run_cnamaste", str(CONFIG), *(token for o in over_rides for token in ("-o", o))]


def resolved(config: YAMLConfig) -> dict[str, Any]:
    """The six values as the run's readers resolve them."""
    return {
        "paths.sample_sheet": config.paths.sample_sheet,
        "paths.output_dir": config.paths.output_dir,
        "hmrf.random_seed": int(config.hmrf.random_seed),
        "hmrf.label_solver": label_solver_type(config),
        "hmm.initializer": config.hmm.initializer,
        "ab_decode.decoder": config.ab_decode.decoder,
    }


def expected(over_rides: list[str]) -> dict[str, Any]:
    """The six values as the command line states them: the seed a whole number."""
    stated: dict[str, Any] = dict(o.split("=", 1) for o in over_rides)
    stated["hmrf.random_seed"] = int(stated["hmrf.random_seed"])
    return stated


@pytest.fixture
def stop_logging() -> Any:
    """Close the log file a stopped run leaves on the logger, which only a finished run removes."""
    before = list(script.logger.handlers)
    yield
    for handler in [h for h in script.logger.handlers if h not in before]:
        script.logger.removeHandler(handler)
        handler.close()


def test_the_batch_s_solvers_and_decoder_map_are_what_these_tests_read() -> None:
    """The scan of `batch_zenodo_run.sh` finds its solvers and both decoder cases, each case naming a decoder the run
    takes. Checked against the script's text and `RESOLVED_DECODER`; fails where the scan reads nothing, so every
    parametrized row below would pass vacuously, or where a case passes a decoder the run would refuse."""
    assert SOLVERS and all(s.count("-") == 2 for s in SOLVERS), SOLVERS
    assert DECODER_KEYS == {"lattice": "hmm_ab", "calicost": "calicost"}, DECODER_KEYS
    for field, key in DECODER_KEYS.items():
        assert key == RESOLVED_DECODER[field], (field, key)


def test_the_console_script_hands_every_override_through_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """`main` parses `run_single_job`'s command line into the configuration path and its six `-o`, as given and in
    order, each a text `key=value` split at the first `=` only, and `--use_existing` off. Checked against the argv
    itself; fails where `-o` is not `append`, a value holding `=` is cut, or one is dropped."""
    over_rides = [
        "paths.sample_sheet=a=b.tsv",
        "paths.output_dir=out",
        "hmrf.random_seed=3",
        "hmrf.label_solver=icm",
        "hmm.initializer=gmm",
        "ab_decode.decoder=hmm_ab",
    ]
    handed: list[tuple[str, list[str], bool]] = []
    monkeypatch.setattr(
        script,
        "run_cnamaste",
        lambda path, over_rides=None, use_existing=False: handed.append((path, over_rides, use_existing)),
    )
    monkeypatch.setattr(sys, "argv", argv(over_rides))
    script.main()
    assert handed == [(str(CONFIG), over_rides, False)], "--use_existing off unless given"
    config = YAMLConfig.from_file(CONFIG)
    config.over_ride(handed[0][1])
    assert config.paths.sample_sheet == "a=b.tsv"


@pytest.mark.parametrize(
    ("key", "first", "text", "want"),
    [
        ("paths.sample_sheet", None, "elsewhere/sheet.tsv", "elsewhere/sheet.tsv"),
        ("paths.output_dir", None, "elsewhere/out", "elsewhere/out"),
        ("hmrf.random_seed", None, "7", 7),
        ("hmrf.label_solver", None, "icm", "icm"),
        ("hmm.initializer", None, "calicost", "calicost"),
        ("ab_decode.decoder", None, "calicost", "calicost"),
        ("ab_decode.decoder", "calicost", "hmm_ab", "hmm_ab"),
    ],
)
def test_an_override_is_the_value_the_run_reads(key: str, first: str | None, text: str, want: Any) -> None:
    """`-o key=text` on `zenodo_sim_config.yaml` moves the value the run's reader resolves from the one before it to
    `want`, the seed as a whole number, and leaves the other five as shipped. The value
    before is the shipped file's, or an earlier `-o key=first` where the file ships `want` (`hmm_ab`), so the later
    `-o` wins. Checked against that prior value, which each row differs from; fails where an override is skipped,
    lands on another key, or is read through a default."""
    before = [f"{key}={first}"] if first else []
    shipped = YAMLConfig.from_file(CONFIG)
    shipped.over_ride(before)
    shipped = resolved(shipped)
    assert shipped[key] != want, f"{key} is {want!r} before the override: the row cannot see it skipped"
    config = YAMLConfig.from_file(CONFIG)
    config.over_ride([*before, f"{key}={text}"])
    assert resolved(config) == shipped | {key: want}
    config.issue_warnings()


@pytest.mark.parametrize("solver", SOLVERS)
def test_each_batch_solver_resolves_and_validates(tmp_path: Path, solver: str) -> None:
    """Each `SOLVERS` entry, mapped as `run_single_job` maps it, gives a configuration `issue_warnings` and
    `check_paths` pass, whose six values are the command line's. Checked against `expected`, the argv's own text;
    fails where the batch names a solver, initializer or decoder the run refuses, or one the run reads otherwise."""
    over_rides = batch_overrides(tmp_path, seed=11, solver=solver)
    config = YAMLConfig.from_file(CONFIG)
    config.over_ride(over_rides)
    config.issue_warnings()
    config.check_paths()
    assert resolved(config) == expected(over_rides)


@pytest.mark.parametrize("solver", SOLVERS)
def test_the_batch_s_command_line_reaches_the_run_s_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, solver: str
) -> None:
    """`main` on `run_single_job`'s command line passes `issue_warnings` and `check_paths` with the six values it
    gave, then stops at `external.require` with `$CNAMASTE_OPENGM_HOME` unset, before its log. Checked against the
    argv's text and against the configuration `check_paths` was handed; fails where the console script validates a
    configuration other than the one the command line states, or a batch solver stops earlier."""
    monkeypatch.delenv(external.get_home_variable(), raising=False)
    assert solver.startswith("opengm_"), "every batch solver needs OpenGM, the stop this test reads"
    over_rides = batch_overrides(tmp_path, seed=5, solver=solver)
    checked: list[YAMLConfig] = []
    check_paths = YAMLConfig.check_paths
    monkeypatch.setattr(YAMLConfig, "check_paths", lambda self: (checked.append(self), check_paths(self))[1])
    monkeypatch.setattr(sys, "argv", argv(over_rides))
    with pytest.raises(RuntimeError, match=f"{external.get_home_variable()} is unset"):
        script.main()
    assert len(checked) == 1 and resolved(checked[0]) == expected(over_rides)
    assert not any(Path(expected(over_rides)["paths.output_dir"]).iterdir()), "nothing written before the stop"


def test_the_seed_override_seeds_the_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stop_logging: Any) -> None:
    """`-o hmrf.random_seed=7`, text in the configuration, seeds NumPy's and numba's generators with the whole number
    7. Checked against `np.random.seed(7)`'s first draw and the integer handed to `set_numba_seed`; fails where the
    seed is the shipped 0, or the text is not converted. ICM, so `external.require` passes; stopped at numba's seed."""
    over_rides = batch_overrides(tmp_path, seed=7, solver="icm-lattice-lattice")
    seeded: list[Any] = []

    def stop(value: Any) -> None:
        seeded.append(value)
        raise Stop

    monkeypatch.setattr(script, "set_numba_seed", stop)
    monkeypatch.setattr(sys, "argv", argv(over_rides))
    with pytest.raises(Stop):
        script.main()
    draw = np.random.random()
    np.random.seed(7)
    assert seeded == [7] and type(seeded[0]) is int and draw == np.random.random()


@njit
def draw_in_numba() -> tuple[float, float]:
    """Two draws from numba's generator, which `np.random.seed` called from Python does not seed."""
    return np.random.random(), np.random.random()


def test_set_numba_seed_seeds_numba_s_generator_as_numpy_s() -> None:
    """After `set_numba_seed(s)`, numba's two draws are NumPy's after `np.random.seed(s)`, bitwise (both the
    Mersenne Twister, seeded alike): the same seed repeats them, another differs. Fails where the seed does not reach
    the generator the kernels draw from."""
    script.set_numba_seed(11)
    first = draw_in_numba()
    script.set_numba_seed(11)
    assert draw_in_numba() == first
    script.set_numba_seed(12)
    assert draw_in_numba() != first
    np.random.seed(11)
    assert first == (np.random.random(), np.random.random())


@pytest.mark.parametrize(
    ("over_ride", "refused"),
    [
        ("hmrf.label_solver=simplex", "hmrf.label_solver 'simplex' is not one of"),
        ("hmm.initializer=nope", "hmm.initializer 'nope' is not one of"),
        ("ab_decode.decoder=nope", "ab_decode.decoder 'nope' is not one of"),
    ],
)
def test_an_invalid_value_is_refused_before_any_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, over_ride: str, refused: str
) -> None:
    """`main` on the batch's command line with one value replaced by one the run does not take raises `ValueError`
    naming it, in `issue_warnings`, before `check_paths`, `external.require` or the log. Checked against the valid
    values' tuples the message names; fails where an unknown solver, initializer or decoder starts a run."""
    monkeypatch.delenv(external.get_home_variable(), raising=False)
    over_rides = batch_overrides(tmp_path, seed=0, solver="opengm_expansion-lattice-lattice") + [over_ride]
    monkeypatch.setattr(YAMLConfig, "check_paths", lambda self: pytest.fail("check_paths reached"))
    monkeypatch.setattr(sys, "argv", argv(over_rides))
    with pytest.raises(ValueError, match=re.escape(refused)):
        script.main()
    assert not any(Path(expected(over_rides[:6])["paths.output_dir"]).iterdir())


@pytest.mark.parametrize(
    ("over_ride", "missing", "hint"),
    [
        ("hmrf.label_slover=icm", "hmrf.label_slover", "label_solver"),
        ("hmrff.label_solver=icm", "hmrff", "hmrf"),
        ("ab_decode.decodr=calicost", "ab_decode.decodr", "decoder"),
        ("hmrf.not_a_key_at_all=1", "hmrf.not_a_key_at_all", None),
    ],
)
def test_a_misspelled_override_is_refused(over_ride: str, missing: str, hint: str | None) -> None:
    """A key or section the configuration does not carry raises `ValueError` naming the override, what is missing
    and the closest names, and the configuration is left as it was. Fails where a misspelling is skipped and the run
    proceeds on the value it meant to replace."""
    config = YAMLConfig.from_file(CONFIG)
    before = (label_solver_type(config), config.ab_decode.decoder)
    with pytest.raises(
        ValueError, match=re.escape(f"-o {over_ride.split('=')[0]}=...: the configuration has no {missing}")
    ) as refused:
        config.over_ride([over_ride])
    assert (f"did you mean {hint}" in str(refused.value)) if hint else "did you mean" not in str(refused.value)
    assert (label_solver_type(config), config.ab_decode.decoder) == before


def test_an_override_without_a_value_is_refused() -> None:
    """`-o hmrf.label_solver`, no `=`: `ValueError` naming it and the form an override takes. Fails where it is
    dropped or raises with no message."""
    with pytest.raises(ValueError, match=re.escape("-o 'hmrf.label_solver': an override is <section>.<key>=<value>.")):
        YAMLConfig.from_file(CONFIG).over_ride(["hmrf.label_solver"])


def test_a_misspelled_override_stops_run_cnamaste_before_any_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The batch's six overrides with the solver key misspelled, through `main`: `ValueError` before `check_paths`,
    nothing written to the output directory. Fails where the run starts on the shipped solver."""
    over_rides = batch_overrides(tmp_path, seed=0, solver="opengm_icm-lattice-lattice")
    over_rides = [o.replace("hmrf.label_solver=", "hmrf.label_slover=") for o in over_rides]
    monkeypatch.setattr(YAMLConfig, "check_paths", lambda self: pytest.fail("check_paths reached"))
    monkeypatch.setattr(sys, "argv", argv(over_rides))
    with pytest.raises(ValueError, match="no hmrf.label_slover"):
        script.main()
    assert not any(Path(expected(over_rides[:6])["paths.output_dir"]).iterdir())
