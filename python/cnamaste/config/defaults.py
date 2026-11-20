"""The configuration's defaults and the values each choice takes."""

from __future__ import annotations

from importlib import resources
from typing import Any

import yaml

MERGE_STAGES = ("baf", "rdr")
"""The stages that fit clones and so can merge them: the BAF-only search, and the BAF/RDR one."""


def get_merge_stages() -> tuple[str, ...]:
    """Return the merge stages, ("baf", "rdr")."""
    return MERGE_STAGES


INTEGER_DECODERS = ("hmm_ab", "calicost")
"""`ab_decode.decoder`: the `(A, B)` HMM (the default) or CalicoST's diploid hill climb (copied in `ab_decode.decode`)."""


def get_integer_decoders() -> tuple[str, ...]:
    """Return the integer decoders, ("hmm_ab", "calicost")."""
    return INTEGER_DECODERS


HMM_AB_SOLVERS = ("icm", "expansion")
"""`ab_decode.solver`: how `ab_decode.decode.solve_clone_chains` decodes the coupled tumour clones."""


def get_hmm_ab_solvers() -> tuple[str, ...]:
    """Return the hmm ab solvers, ("icm", "expansion")."""
    return HMM_AB_SOLVERS


CONTIG_STARTS = ("anchored", "uniform")
"""`ab_decode.contig_start`: the hmm_ab decoder's prior on each contig's first bin."""


def get_contig_starts() -> tuple[str, ...]:
    """Return the contig starts, ("anchored", "uniform")."""
    return CONTIG_STARTS


PLOIDY_BY_NAME = {"diploid": 2, "triploid": 3, "tetraploid": 4}
"""`ab_decode.assumed_ploidy`'s names and the ploidy P each assumes, its balanced pair `(P / 2, P / 2)`."""


def get_ploidy_by_name() -> dict[str, int]:
    """Return the ploidy by name, {"diploid": 2, "triploid": 3, "tetraploid": 4}."""
    return PLOIDY_BY_NAME


LATTICE_TYPES = {"square": (8, 4), "triangular": (6,)}
"""`hmrf.lattice_type` -> the coordination numbers `hmrf.coordination_num` takes, the first its default."""


def get_lattice_types() -> dict[str, tuple[int, ...]]:
    """Return the lattice types, {"square": (8, 4), "triangular": (6,)}."""
    return LATTICE_TYPES


LABEL_SOLVERS = ("icm", "opengm_icm", "opengm_expansion", "opengm_trws", "expansion", "trws")
"""`hmrf.label_solver`: cnamaste's alpha expansion (`cnamaste.expansion`, the default), OpenGM's moves with a
Boykov-Kolmogorov cut in Rust; or ICM; or OpenGM's ICM, alpha expansion or TRW-S; or cnamaste's TRW-S
(`cnamaste.trws`, T- #198), OpenGM's `TRWSi` in Rust."""


def get_label_solvers() -> tuple[str, ...]:
    """Return the label solvers, ("icm", "opengm_icm", "opengm_expansion", "opengm_trws", "expansion", "trws")."""
    return LABEL_SOLVERS


EXPANSION_LABEL_ORDERS = ("index", "field_share")
"""`hmrf.expansion_label_order`: the order alpha expansion visits the clones, 0, 1, ... (OpenGM's), or by how many
spots' field each is best for, most first (`potts.get_field_share_order`)."""


def get_expansion_label_orders() -> tuple[str, ...]:
    """Return the expansion label orders, ("index", "field_share")."""
    return EXPANSION_LABEL_ORDERS


LABEL_STARTS = ("prev_assignment", "field_argmax")
"""`hmrf.label_start`: the labelling every label solver starts from, the iteration's `prev_assignment` (the default)
or `field_argmax` of the iteration's field."""


def get_label_starts() -> tuple[str, ...]:
    """Return the label starts, ("prev_assignment", "field_argmax")."""
    return LABEL_STARTS


OPENGM_MINCUTS = ("boykov_kolmogorov", "dinic")
"""`hmrf.opengm_mincut_solver`: the min s-t cut of OpenGM's alpha expansion and swap moves."""


def get_opengm_mincuts() -> tuple[str, ...]:
    """Return the opengm mincuts, ("boykov_kolmogorov", "dinic")."""
    return OPENGM_MINCUTS


CLONE_INIT_TYPES = ("rectangle", "median_bisection")
"""`phasing.clone_init_type` and `hmrf.rdr_clone_init_type`: CalicoST's grid, or a seeded rotated median bisection."""


def get_clone_init_types() -> tuple[str, ...]:
    """Return the clone init types, ("rectangle", "median_bisection")."""
    return CLONE_INIT_TYPES


LOGMU_SHIFT_TYPES = ("argmax", "posterior")
"""`hmrf.logmu_shift_type`: each bin's state weights in the RDR+BAF fit's `log Z_c`, the previous fit's argmax path
held through the fit; or each E step's posteriors, held through its M step (`hmm_nophasing.run_optimization_pipeline`)."""


def get_logmu_shift_types() -> tuple[str, ...]:
    """Return the logmu shift types, ("argmax", "posterior")."""
    return LOGMU_SHIFT_TYPES


HMM_INITIALIZERS = ("calicost", "lattice")
"""`hmm.initializer`: `lattice` (the default) or `calicost`, `cnamaste.initialize.genomic`'s starts."""


def get_hmm_initializers() -> tuple[str, ...]:
    """Return the hmm initializers, ("calicost", "lattice")."""
    return HMM_INITIALIZERS


MSTEP_SOLVERS = ("scipy_bfgs", "scipy_lbfgsb")
"""`mstep.solver`: scipy's BFGS (the default) or scipy's L-BFGS-B (`cnamaste.mstep`)."""


def get_mstep_solvers() -> tuple[str, ...]:
    """Return the M step's solvers, ("scipy_bfgs", "scipy_lbfgsb")."""
    return MSTEP_SOLVERS


def get_config_defaults() -> dict[str, dict[str, Any]]:
    """Return `config_defaults.yaml`, beside this module: section -> key -> default.

    Returns:
        The nested dict, as `yaml.safe_load` reads it.
    """
    return yaml.safe_load(resources.files("cnamaste.config").joinpath("config_defaults.yaml").read_text())


config_defaults = get_config_defaults()
"""Section -> key -> default (`get_config_defaults`); a default also types its key (`typed`)."""
