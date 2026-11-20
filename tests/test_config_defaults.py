"""`config_defaults.yaml`, the one place a default is written, and the shipped configurations as overrides of it.

Referees: the YAML types each default as the code reads it (PyYAML reads `1e-4`, without a decimal point, as text);
each shipped file sets only keys the defaults carry, each to a value other than its default; and a section or key
the defaults do not carry is refused with the closest names, a retired one alike. It costs 1 s.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pytest
import yaml

from cnamaste.config import YAMLConfig, config_defaults, coordination_num, get_clone_init_types, typed
from cnamaste.initialize.spatial import get_slice_bisection_labelling
from cnamaste.qa.sims import get_project
from test_lattice_adjacency import two_slice_positions

SHIPPED = ("config.yaml", "zenodo_sim_config.yaml")


def test_every_default_is_a_scalar_of_its_own_type() -> None:
    """Each default, at any depth of nested sections, is None, a bool, an int, a float or text, and no text reads as a
    number: a float written without a decimal point is text to YAML 1.1, and would type its key as text. Fails where a
    default is written so."""
    stack = [(section, keys) for section, keys in config_defaults.items()]
    while stack:
        path, keys = stack.pop()
        for key, value in keys.items():
            if isinstance(value, dict):
                stack.append((f"{path}.{key}", value))
                continue
            assert value is None or isinstance(value, (bool, int, float, str)), f"{path}.{key}: {value!r}"
            if isinstance(value, str):
                assert not re.fullmatch(r"[-+]?[0-9._]+([eE][-+]?[0-9]+)?", value), f"{path}.{key}: {value!r}"


@pytest.mark.parametrize("name", SHIPPED)
def test_a_shipped_configuration_sets_only_what_differs_from_the_defaults(name: str) -> None:
    """Every key the file sets is one the defaults carry, at a value other than the default: an override, not a
    second copy that can drift. The resolved configuration passes `issue_warnings`. Fails where a file restates a
    default, or names a key nothing reads."""
    shipped = yaml.safe_load((get_project() / name).read_text())
    for section, keys in shipped.items():
        for key, value in keys.items():
            assert key in config_defaults[section], f"{name}: {section}.{key} is not a default's"
            default = config_defaults[section][key]
            assert typed(f"{section}.{key}", value, default) != default, f"{name}: {section}.{key} is its default"
    YAMLConfig.from_file(get_project() / name).issue_warnings()


@pytest.mark.parametrize(
    ("over", "refusal"),
    [
        ({"hmrf": {"label_slover": "icm"}}, r"hmrf carries no hmrf\.label_slover \(did you mean label_solver"),
        ({"hmrff": {"label_solver": "icm"}}, r"carries no section hmrff \(did you mean hmrf"),
        ({"nbinom": {"start_disp": 0.01}}, r"carries no section nbinom"),
        ({"hmrf": {"random_state": 0}}, r"hmrf carries no hmrf\.random_state \(did you mean random_seed\?\)"),
        *(
            ({section: {key: value}}, rf"{section} carries no {section}\.{key}: remove it; the section's keys are ")
            for section, key, value in (
                ("hmrf", "construct_adjacency_method", "lattice"),
                ("hmrf", "maxspots_pooling", 7),
                ("hmrf", "inertia", 0.0),
                ("hmrf", "np_merge", False),
                ("run", "cache", False),
                ("preprocessing", "tumorprop_file", "none.tsv"),
                ("hmrf", "tumorprop_threshold", 0.0),
                ("hmrf", "initialization_method", "rectangle"),
            )
        ),
    ],
    ids=[
        "key",
        "section",
        "retired section",
        "renamed random_state",
        *(
            "retired " + k
            for k in (
                "adjacency",
                "pooling",
                "inertia",
                "np_merge",
                "cache",
                "tumorprop_file",
                "tumorprop_threshold",
                "initialization_method",
            )
        ),
    ],
)
def test_a_key_the_defaults_do_not_carry_is_refused(over: dict[str, Any], refusal: str) -> None:
    """The shipped CalicoST configuration with one unknown key or section, a retired key at its neutral value among
    them: `ValueError` naming it, with the closest names and the section's keys. Fails where it is read as unset."""
    shipped = yaml.safe_load((get_project() / "zenodo_sim_config.yaml").read_text())
    merged = {**shipped, **{k: {**shipped.get(k, {}), **v} for k, v in over.items()}}
    with pytest.raises(ValueError, match=refusal):
        YAMLConfig(merged).issue_warnings()


@pytest.mark.parametrize("key", ["phasing.clone_init_type", "hmrf.rdr_clone_init_type"])
def test_each_clone_init_type_is_selectable_at_each_site(key: str) -> None:
    """Each site defaults to `rectangle`, takes each of `CLONE_INIT_TYPES` and refuses another, the sandboxed
    best_equal_partition among them; `hmrf.bisection_quantile_jitter` 0.5 is refused. Fails where a start nothing
    takes is accepted, or the default moves from cleanup's."""
    section, name = key.split(".")
    assert config_defaults[section][name] == "rectangle"
    shipped = yaml.safe_load((get_project() / "zenodo_sim_config.yaml").read_text())
    for init_type in get_clone_init_types():
        resolved = YAMLConfig({**shipped, section: {**shipped.get(section, {}), name: init_type}})
        resolved.issue_warnings()
        assert getattr(getattr(resolved, section), name) == init_type
    with pytest.raises(ValueError, match=f"{key} 'best_equal_partition' is not one of"):
        YAMLConfig({**shipped, section: {**shipped.get(section, {}), name: "best_equal_partition"}}).issue_warnings()
    with pytest.raises(ValueError, match="bisection_quantile_jitter is in"):
        YAMLConfig({**shipped, "hmrf": {**shipped["hmrf"], "bisection_quantile_jitter": 0.5}}).issue_warnings()


def test_square_takes_4_or_8_neighbours_and_triangular_6() -> None:
    """`hmrf.coordination_num`: unset is the lattice's first, 6 at `triangular` and 8 at `square`; `square` takes 4
    and 8; `square` 6 and `triangular` 4 are refused by `issue_warnings`. Fails where a k the lattice does not take is
    accepted, or the default moves from the 8 and 6 the run built before the key."""
    assert config_defaults["hmrf"]["coordination_num"] is None
    shipped = yaml.safe_load((get_project() / "zenodo_sim_config.yaml").read_text())
    for lattice, k, expected in (("triangular", None, 6), ("square", None, 8), ("square", 4, 4), ("square", 8, 8)):
        resolved = YAMLConfig({**shipped, "hmrf": {**shipped["hmrf"], "lattice_type": lattice, "coordination_num": k}})
        resolved.issue_warnings()
        assert coordination_num(resolved) == expected
    for lattice, k in (("square", 6), ("triangular", 4)):
        with pytest.raises(ValueError, match=f"hmrf.coordination_num {k} is not one of"):
            YAMLConfig(
                {**shipped, "hmrf": {**shipped["hmrf"], "lattice_type": lattice, "coordination_num": k}}
            ).issue_warnings()


def test_random_seed_changes_the_realization_only() -> None:
    """`hmrf.random_seed` 0 and 1: the resolved configurations differ at that key alone, and dev_tree easy's
    (`e4bbe03f`) phasing start keeps its clones per slice but moves its cut. Fails where the seed names another
    setting, or does not reach the bisection."""
    shipped = yaml.safe_load((get_project() / "zenodo_sim_config.yaml").read_text())
    resolved = [YAMLConfig({**shipped, "hmrf": {**shipped["hmrf"], "random_seed": seed}}).to_dict() for seed in (0, 1)]
    assert resolved[0]["hmrf"].pop("random_seed") == 0 and resolved[1]["hmrf"].pop("random_seed") == 1
    assert resolved[0] == resolved[1]
    _, physical, sample_ids = two_slice_positions("e4bbe03f")
    starts = [
        get_slice_bisection_labelling(
            physical, sample_ids, 8, np.random.default_rng(seed), config_defaults["hmrf"]["bisection_quantile_jitter"]
        )
        for seed in (0, 1)
    ]
    for s in (0, 1):
        assert set(starts[0][sample_ids == s]) == set(starts[1][sample_ids == s])
    assert not np.array_equal(starts[0], starts[1])
