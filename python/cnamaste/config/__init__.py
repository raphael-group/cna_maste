"""The run's configuration: `YAMLConfig` (`cnamaste.config.yaml_config`), its defaults and choices
(`cnamaste.config.defaults`), and the one configuration a run holds."""

from cnamaste.config.clock import start_time
from cnamaste.config.defaults import (
    config_defaults,
    get_clone_init_types,
    get_config_defaults,
    get_contig_starts,
    get_expansion_label_orders,
    get_hmm_ab_solvers,
    get_hmm_initializers,
    get_integer_decoders,
    get_label_solvers,
    get_label_starts,
    get_lattice_types,
    get_logmu_shift_types,
    get_merge_stages,
    get_mstep_solvers,
    get_opengm_mincuts,
    get_ploidy_by_name,
)
from cnamaste.config.yaml_config import (
    YAMLConfig,
    assumed_ploidy,
    cast_text_to_bool,
    coordination_num,
    describe_unknown_key,
    describe_unknown_override,
    get_config_section_keys,
    get_count_table_extent,
    get_labelling_options,
    get_penalize_phase_only_on_same_cnv,
    he_components,
    label_solver_type,
    label_start_type,
    lattice_type,
    normal_fraction_starts,
    opengm_mincut_solver_type,
    typed,
)
from cnamaste.logger import get_logger

logger = get_logger(__name__, start_time=start_time)

__all__ = [
    "YAMLConfig",
    "assumed_ploidy",
    "cast_text_to_bool",
    "config_defaults",
    "coordination_num",
    "describe_unknown_key",
    "describe_unknown_override",
    "get_clone_init_types",
    "get_config_defaults",
    "get_config_section_keys",
    "get_contig_starts",
    "get_count_table_extent",
    "get_expansion_label_orders",
    "get_global_config",
    "get_hmm_ab_solvers",
    "get_hmm_initializers",
    "get_integer_decoders",
    "get_label_solvers",
    "get_label_starts",
    "get_labelling_options",
    "get_lattice_types",
    "get_logmu_shift_types",
    "get_merge_stages",
    "get_mstep_solvers",
    "get_opengm_mincuts",
    "get_penalize_phase_only_on_same_cnv",
    "get_ploidy_by_name",
    "he_components",
    "label_solver_type",
    "label_start_type",
    "lattice_type",
    "normal_fraction_starts",
    "opengm_mincut_solver_type",
    "set_global_config",
    "start_time",
    "typed",
]


_global_config = None


def set_global_config(config):
    assert config is None or isinstance(config, YAMLConfig), "Global config must be None or a YAMLConfig instance"

    global _global_config
    _global_config = config


def get_global_config():
    if _global_config is None:
        logger.warning("cnamaste config has not been defined.")

    else:
        assert isinstance(_global_config, YAMLConfig), "Global config must be None or a YAMLConfig instance"

    return _global_config
