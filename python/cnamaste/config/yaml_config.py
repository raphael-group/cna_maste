"""The run's configuration: a YAML file typed and defaulted by `config_defaults`, overridden by `-o`, and checked."""

from __future__ import annotations

import difflib
from pathlib import Path
from typing import Any, NamedTuple

import yaml

from cnamaste.config.clock import start_time
from cnamaste.config.defaults import (
    config_defaults,
    get_clone_init_types,
    get_contig_starts,
    get_expansion_label_orders,
    get_hmm_ab_solvers,
    get_hmm_initializers,
    get_integer_decoders,
    get_label_solvers,
    get_label_starts,
    get_lattice_types,
    get_logmu_shift_types,
    get_mstep_solvers,
    get_opengm_mincuts,
    get_ploidy_by_name,
)
from cnamaste.logger import get_logger

logger = get_logger(__name__, start_time=start_time)


def typed(key_path, value, default):
    """`value` as `default`'s type: a bool by `cast_text_to_bool`, an int where it is a whole number, a float; else as given.

    Args:
        key_path: names the key in a refusal.
        value: the file's value or `-o`'s text.
        default: the key's default.

    Returns:
        The typed value; None where `value` is None.

    Raises:
        ValueError: naming the key where the value does not read as the type.
    """
    if value is None or default is None:
        return value
    if isinstance(default, bool):
        return cast_text_to_bool(key_path, value)
    if isinstance(default, (int, float)):
        kind = type(default)
        try:
            number = float(value) if not isinstance(value, bool) else None
        except (TypeError, ValueError):
            number = None
        if number is None or (kind is int and not number.is_integer()):
            raise ValueError(f"{key_path} is {kind.__name__}, not {value!r}.")
        return kind(number) if kind is int else number
    return value


def cast_text_to_bool(key_path, value):
    """`-o`'s text for a boolean key as a boolean: true/1 or false/0, any case.

    Raises:
        ValueError: naming the key where the text is neither.
    """
    text = str(value).strip().lower()
    if text in ("true", "1"):
        return True
    if text in ("false", "0"):
        return False
    raise ValueError(f"{key_path} is a boolean: {value!r} is not true, false, 1 or 0.")


def get_section_defaults(path: str) -> dict[str, Any]:
    """The defaults of the section at dotted `path`, as `ab_decode.baf`; empty where `config_defaults` has none.

    Args:
        path: the section's dotted path.

    Returns:
        Its keys' defaults.
    """
    defaults: Any = config_defaults
    for key in path.split("."):
        defaults = defaults.get(key, {}) if isinstance(defaults, dict) else {}
    return defaults if isinstance(defaults, dict) else {}


def get_config_section_keys(obj: Any) -> list[str]:
    """The keys a configuration section carries: a `YAMLConfig`'s attributes or a dict's keys, else none."""
    if isinstance(obj, YAMLConfig):
        return sorted(vars(obj))
    return sorted(obj) if isinstance(obj, dict) else []


def describe_unknown_override(key_path: str, missing: str, names: list[str]) -> str:
    """The refusal of `-o key_path=...`, whose `missing` the configuration does not carry, with the closest `names`."""
    close = difflib.get_close_matches(missing.rsplit(".", 1)[-1], names, n=3)
    hint = f"; did you mean {', '.join(close)}?" if close else ""
    return f"-o {key_path}=...: the configuration has no {missing}, so the override would set nothing{hint}"


def describe_unknown_key(section: str, unknown: list[str], names: list[str]) -> str:
    """The refusal of keys a closed section does not carry, each with the closest of its `names`."""
    hints = []
    for key in unknown:
        close = difflib.get_close_matches(key, names, n=3)
        hints.append(f"{section}.{key}" + (f" (did you mean {', '.join(close)}?)" if close else ""))
    return f"{section} carries no {', '.join(hints)}: remove it; the section's keys are {', '.join(names)}."


class YAMLConfig:
    def __init__(self, config_dict: dict[str, Any], section: bool | str = False):
        # NB `section` the section's dotted path, as `ab_decode.baf`, whose `config_defaults` type its keys; False at
        #    the top.
        defaults = get_section_defaults(section) if isinstance(section, str) else {}
        for key, value in config_dict.items():
            if isinstance(value, dict):
                path = f"{section}.{key}" if isinstance(section, str) else key
                setattr(self, key, YAMLConfig(get_section_defaults(path) | value, section=path))
            elif isinstance(value, str) and value.lower() == "none":
                setattr(self, key, None)
            elif key in defaults:
                setattr(self, key, typed(f"{section}.{key}", value, defaults[key]))
            else:
                setattr(self, key, value)
        if not section:
            for key, values in config_defaults.items():
                if not hasattr(self, key):
                    setattr(self, key, YAMLConfig(dict(values), section=key))

    def __repr__(self) -> str:
        return f"YAMLConfig(\n{self.format_section(self.__dict__, indent=2)}\n)"

    def format_section(self, d: dict[str, Any], indent: int = 0) -> str:
        items = []
        space = " " * indent

        for key, value in d.items():
            if isinstance(value, YAMLConfig):
                formatted_value = f"\n{self.format_section(value.__dict__, indent + 2)}\n{space}"
            elif isinstance(value, dict):
                formatted_value = f"{{\n{self.format_section(value, indent + 2)}\n{space}}}"
            elif isinstance(value, str):
                formatted_value = f"'{value}'"
            else:
                formatted_value = repr(value)
            items.append(f"{space}{key}: {formatted_value}")

        return ",\n".join(items)

    def to_dict(self) -> dict[str, Any]:
        """The configuration as nested dicts, sections expanded: what `YAMLConfig(...)` reads back, None as None.

        Returns:
            The dict.
        """
        return {k: v.to_dict() if isinstance(v, YAMLConfig) else v for k, v in self.__dict__.items()}

    @classmethod
    def from_file(cls, config_path: str | Path) -> YAMLConfig:
        with Path.open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)

        config = cls(config_dict)

        return config

    def over_ride(self, over_rides):
        """Set each `-o <section>.<key>=<value>` on this configuration, in order, typed as the value it replaces.

        Args:
            over_rides: the `key.path=value` texts, or None.

        Raises:
            ValueError: where an override has no `=`, an unknown key path, or a value not of the key's type.
        """
        if over_rides:
            for over_ride in over_rides:
                if "=" not in over_ride:
                    raise ValueError(f"-o {over_ride!r}: an override is <section>.<key>=<value>.")

                key_path, value = over_ride.split("=", 1)
                keys = key_path.split(".")

                obj = self

                for depth, k in enumerate(keys[:-1]):
                    if isinstance(obj, YAMLConfig) and hasattr(obj, k):
                        obj = getattr(obj, k)
                    elif isinstance(obj, dict) and k in obj:
                        obj = obj[k]
                    else:
                        raise ValueError(
                            describe_unknown_override(
                                key_path, ".".join(keys[: depth + 1]), get_config_section_keys(obj)
                            )
                        )

                final_key = keys[-1]

                # NB untyped, "false" is a non-empty string and so true.
                current = (
                    getattr(obj, final_key, None)
                    if hasattr(obj, final_key)
                    else (obj.get(final_key) if isinstance(obj, dict) else None)
                )
                try:
                    value = typed(key_path, value, current)
                except ValueError as error:
                    raise ValueError(f"-o {over_ride!r}: {error}") from None

                if isinstance(obj, YAMLConfig) and hasattr(obj, final_key):
                    setattr(obj, final_key, value)
                    logger.info(f"Config override: {key_path} = {value}")
                elif isinstance(obj, dict) and final_key in obj:
                    obj[final_key] = value
                    logger.info(f"Config override: {key_path} = {value}")
                else:
                    raise ValueError(describe_unknown_override(key_path, key_path, get_config_section_keys(obj)))

    def check_paths(self):
        """Raise unless `paths.sample_sheet` is a file and `paths.output_dir` an existing directory."""
        paths = getattr(self, "paths", None)
        sheet, output_dir = getattr(paths, "sample_sheet", None), getattr(paths, "output_dir", None)
        if not isinstance(sheet, (str, Path)) or not str(sheet):
            raise ValueError(f"paths.sample_sheet must name a sample sheet file, not {sheet!r}.")
        if not Path(sheet).is_file():
            raise ValueError(f"paths.sample_sheet {sheet} does not exist.")
        if output_dir is None or not str(output_dir):
            raise ValueError("paths.output_dir is unset: set it in the configuration or by -o paths.output_dir=DIR.")
        if not Path(output_dir).is_dir():
            raise ValueError(f"paths.output_dir {output_dir} does not exist: create it first.")

    def issue_warnings(self):
        if self.annotation.clone_label is not None:
            logger.warning("Assuming known clone labels")
        if not self.phasing.run:
            logger.warning("Assuming no baf-based phasing")
        if int(self.hmrf.n_clones_rdr) == 1:
            logger.warning("Assuming no rdr-based clone identification")
        if self.hmrf.fixed_assignment:
            logger.warning("Assuming fixed assignment")
        # NB every section is closed: an unknown section or key is refused, not read as unset.
        for section in get_config_section_keys(self):
            if section not in config_defaults:
                close = difflib.get_close_matches(section, sorted(config_defaults), n=3)
                hint = f" (did you mean {', '.join(close)}?)" if close else ""
                raise ValueError(
                    f"the configuration carries no section {section}{hint}: its sections are "
                    f"{', '.join(config_defaults)}."
                )
            unknown = sorted(set(get_config_section_keys(getattr(self, section))) - set(config_defaults[section]))
            if unknown:
                raise ValueError(describe_unknown_key(section, unknown, sorted(config_defaults[section])))
        ab_decode = self.ab_decode
        if ab_decode.decoder not in get_integer_decoders():
            raise ValueError(f"ab_decode.decoder {ab_decode.decoder!r} is not one of {get_integer_decoders()}.")
        # NB a total of at least 2 so (1, 1) is a state.
        total, allele = ab_decode.max_total_copy, ab_decode.max_allele_copy
        if total < 2 or allele < 1 or allele > total:
            raise ValueError(
                f"ab_decode caps {{'max_total_copy': {total}, 'max_allele_copy': {allele}}}: max_total_copy >= 2, "
                "1 <= max_allele_copy <= max_total_copy."
            )
        names = assumed_ploidy(self)
        if not set(names) <= set(get_ploidy_by_name()):
            raise ValueError(
                f"ab_decode.assumed_ploidy {','.join(names)!r}: one or more of {sorted(get_ploidy_by_name())}."
            )
        # NB CalicoST's hill climb is diploid alone (`calicost.adapters.calicost_fixdiploid`).
        if ab_decode.decoder == "calicost" and names != ("diploid",):
            raise ValueError(f"ab_decode.decoder calicost is diploid alone, not assumed_ploidy {','.join(names)!r}.")
        if not 0.0 < ab_decode.int_clone_agreement <= 1.0:
            raise ValueError(f"ab_decode.int_clone_agreement is in (0, 1], not {ab_decode.int_clone_agreement}.")
        # NB expansion's cuts cannot hold the ploidy's A + B constraint.
        if ab_decode.solver not in get_hmm_ab_solvers():
            raise ValueError(f"ab_decode.solver {ab_decode.solver!r} is not one of {get_hmm_ab_solvers()}.")
        if ab_decode.solver == "expansion" and ab_decode.genome_wide_ploidy_constraint:
            raise ValueError(
                "ab_decode.solver expansion cannot hold genome_wide_ploidy_constraint: set it False, or solver icm."
            )
        if ab_decode.contig_start not in get_contig_starts():
            raise ValueError(f"ab_decode.contig_start {ab_decode.contig_start!r} is not one of {get_contig_starts()}.")
        # NB a recompute starts from the fit's Z_c, which apply_logmu_shift False does not take.
        if ab_decode.recompute_logmu_shift and not ab_decode.apply_logmu_shift:
            raise ValueError(
                "ab_decode.recompute_logmu_shift True recomputes Z_c from the fit's, which apply_logmu_shift False "
                "drops: set apply_logmu_shift True, or recompute_logmu_shift False."
            )
        if self.he.n_components is not None and self.he.n_components < 2:
            raise ValueError(f"he.n_components is at least 2, not {self.he.n_components}: one class cuts nothing.")
        # NB an OpenGM solver without the library stops the run at its start (`external.require`).
        if label_solver_type(self) not in get_label_solvers():
            raise ValueError(f"hmrf.label_solver {self.hmrf.label_solver!r} is not one of {get_label_solvers()}.")
        if label_start_type(self) not in get_label_starts():
            raise ValueError(f"hmrf.label_start {self.hmrf.label_start!r} is not one of {get_label_starts()}.")
        for key, init_type in (
            ("phasing.clone_init_type", self.phasing.clone_init_type),
            ("hmrf.rdr_clone_init_type", self.hmrf.rdr_clone_init_type),
        ):
            if init_type not in get_clone_init_types():
                raise ValueError(f"{key} {init_type!r} is not one of {get_clone_init_types()}.")
        phasing = self.phasing  # type: ignore[attr-defined]
        candidates = phasing.n_start_candidates
        if candidates < 1 or (candidates > 1 and phasing.clone_init_type != "median_bisection"):
            raise ValueError(
                f"phasing.n_start_candidates {candidates!r}: at least 1, and 1 unless phasing.clone_init_type is "
                "median_bisection."
            )
        if not 0.0 <= self.hmrf.bisection_quantile_jitter < 0.5:
            raise ValueError(
                f"hmrf.bisection_quantile_jitter is in [0, 0.5), not {self.hmrf.bisection_quantile_jitter!r}."
            )
        if lattice_type(self) not in get_lattice_types():
            raise ValueError(
                f"hmrf.lattice_type {lattice_type(self)!r} is not one of {tuple(get_lattice_types())}: set it, square for "
                "Visium HD's bins or triangular for Visium's array; unset is deprecated."
            )
        if get_labelling_options(self).expansion_label_order not in get_expansion_label_orders():
            raise ValueError(
                f"hmrf.expansion_label_order {self.hmrf.expansion_label_order!r} is not one of "
                f"{get_expansion_label_orders()}."
            )
        if coordination_num(self) not in get_lattice_types()[lattice_type(self)]:
            raise ValueError(
                f"hmrf.coordination_num {self.hmrf.coordination_num!r} is not one of "
                f"{get_lattice_types()[lattice_type(self)]} at hmrf.lattice_type {lattice_type(self)}."
            )
        if opengm_mincut_solver_type(self) not in get_opengm_mincuts():
            raise ValueError(
                f"hmrf.opengm_mincut_solver {self.hmrf.opengm_mincut_solver!r} is not one of {get_opengm_mincuts()}."
            )
        if self.ab_decode.opengm_mincut_solver not in get_opengm_mincuts():
            raise ValueError(
                f"ab_decode.opengm_mincut_solver {self.ab_decode.opengm_mincut_solver!r} is not one of {get_opengm_mincuts()}."
            )
        # NB each clone's read depth under its own library normalizer, absent or True; False is Z_c = 1.
        if hasattr(self.hmrf, "apply_logmu_shift"):
            if not isinstance(self.hmrf.apply_logmu_shift, bool):
                raise ValueError(f"hmrf.apply_logmu_shift is True or False, not {self.hmrf.apply_logmu_shift!r}.")
            if not self.hmrf.apply_logmu_shift:
                logger.warning("Assuming every clone's read depth at Z_c = 1 (hmrf.apply_logmu_shift is False)")
        if self.hmrf.logmu_shift_type not in get_logmu_shift_types():
            raise ValueError(
                f"hmrf.logmu_shift_type {self.hmrf.logmu_shift_type!r} is not one of {get_logmu_shift_types()}."
            )
        if self.hmm.initializer not in get_hmm_initializers():
            raise ValueError(f"hmm.initializer {self.hmm.initializer!r} is not one of {get_hmm_initializers()}.")
        if self.mstep.solver not in get_mstep_solvers():
            raise ValueError(f"mstep.solver {self.mstep.solver!r} is not one of {get_mstep_solvers()}.")


def he_components(config):
    """`he.n_components`: the classes the slide's grayscale is cut into (`he.get_he_image`).

    Args:
        config: a `YAMLConfig` or anything with the section.

    Returns:
        The count; `config_defaults`' where the configuration carries no `he` section, as a capture's does not.
    """
    value = getattr(getattr(config, "he", None), "n_components", None)
    return config_defaults["he"]["n_components"] if value is None else int(str(value))


def lattice_type(config):
    """`hmrf.lattice_type`, None where unset, which `issue_warnings` refuses.

    Args:
        config: a `YAMLConfig` or anything with the section.

    Returns:
        The lattice's name.
    """
    return getattr(getattr(config, "hmrf", None), "lattice_type", None)


def coordination_num(config):
    """`hmrf.coordination_num`, the k of each spot's k nearest neighbours; the lattice's default where unset.

    Args:
        config: a `YAMLConfig` or anything with the section, its `hmrf.lattice_type` one of `LATTICE_TYPES`.

    Returns:
        The k, `LATTICE_TYPES[lattice_type][0]` where unset.
    """
    value = getattr(getattr(config, "hmrf", None), "coordination_num", None)
    return get_lattice_types()[lattice_type(config)][0] if value is None else value


def opengm_mincut_solver_type(config):
    """`hmrf.opengm_mincut_solver`, `config_defaults`' where unset.

    Args:
        config: a `YAMLConfig` or anything with the section.

    Returns:
        The cut's name.
    """
    return getattr(
        getattr(config, "hmrf", None), "opengm_mincut_solver", config_defaults["hmrf"]["opengm_mincut_solver"]
    )


class LabellingOptions(NamedTuple):
    """`hmrf`'s reductions and expansion order for every label_solver but icm (`opengm.external_assignment`).

    Args:
        fix_dominant_spots: fix each spot whose field margin exceeds its summed coupling.
        prune_dominated_labels: drop each clone no spot can take.
        expansion_label_order: one of `get_expansion_label_orders()`.
        opengm_patch: `expansion`'s moves with T- #195's auxiliary node, else OpenGM's.
    """

    fix_dominant_spots: bool
    prune_dominated_labels: bool
    expansion_label_order: str
    opengm_patch: bool = False


def get_labelling_options(config):
    """`hmrf`'s `LabellingOptions`, `config_defaults`' where unset.

    Args:
        config: a `YAMLConfig` or anything with the section.

    Returns:
        The `LabellingOptions`.
    """
    hmrf = getattr(config, "hmrf", None)
    return LabellingOptions(*(getattr(hmrf, key, config_defaults["hmrf"][key]) for key in LabellingOptions._fields))


def get_penalize_phase_only_on_same_cnv(config):
    """`hmm.penalize_phase_only_on_same_cnv`, `config_defaults`' where unset.

    Args:
        config: a `YAMLConfig` or anything with the section.

    Returns:
        The flag.
    """
    default = config_defaults["hmm"]["penalize_phase_only_on_same_cnv"]
    return bool(getattr(getattr(config, "hmm", None), "penalize_phase_only_on_same_cnv", default))


def get_count_table_extent(config: Any) -> int:
    """`hmm.count_table_extent`, `config_defaults`' where unset: the largest count the emission's tables cover.

    Args:
        config: a `YAMLConfig` or anything with the section.

    Returns:
        The extent.
    """
    default = config_defaults["hmm"]["count_table_extent"]
    return int(getattr(getattr(config, "hmm", None), "count_table_extent", default))


def assumed_ploidy(config):
    """`ab_decode.assumed_ploidy`'s names, `config_defaults`' where unset.

    Args:
        config: a `YAMLConfig` or anything with the section.

    Returns:
        The names, in the order given, e.g. `("diploid", "tetraploid")`.
    """
    assumed = str(
        getattr(getattr(config, "ab_decode", None), "assumed_ploidy", config_defaults["ab_decode"]["assumed_ploidy"])
    )
    return tuple(name.strip() for name in assumed.split(","))


def normal_fraction_starts(config):
    """`ab_decode.normal_fraction_starts`, `config_defaults`' where unset.

    Args:
        config: a `YAMLConfig` or anything with the section.

    Returns:
        The tumour clones' first normal fractions, in the order given, e.g. `(0.05, 0.1, 0.15)`.
    """
    starts = str(
        getattr(
            getattr(config, "ab_decode", None),
            "normal_fraction_starts",
            config_defaults["ab_decode"]["normal_fraction_starts"],
        )
    )
    return tuple(float(start) for start in starts.split(","))


def label_solver_type(config):
    """`hmrf.label_solver`, `config_defaults`' where unset.

    Args:
        config: a `YAMLConfig` or anything with the section.

    Returns:
        The solver's name.
    """
    return getattr(getattr(config, "hmrf", None), "label_solver", config_defaults["hmrf"]["label_solver"])


def label_start_type(config):
    """`hmrf.label_start`, `config_defaults`' where unset.

    Args:
        config: a `YAMLConfig` or anything with the section.

    Returns:
        The start's name.
    """
    return getattr(getattr(config, "hmrf", None), "label_start", config_defaults["hmrf"]["label_start"])
