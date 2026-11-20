"""The HMM's initial parameters at each stage, by `hmm.initializer`: `lattice`, the `(A, B)` decoder's states, or
`calicost`, CalicoST's Gaussian mixture."""

import copy
from typing import Any

import numpy as np

from cnamaste.ab_decode.decode import decode_integer_copies
from cnamaste.ab_decode.lattice import get_ab_lattice, get_lattice_parameters
from cnamaste.calicost import gmm_start
from cnamaste.config import get_hmm_initializers, start_time
from cnamaste.emission import get_unconfounded
from cnamaste.hmm_params import CnaHmmParameters
from cnamaste.logger import get_logger
from cnamaste.spatio_genomic_counts import PseudobulkCounts

logger = get_logger(__name__, start_time=start_time)

LATTICE_T = 1.0 - 1.0e-6
"""The stay probability the lattice start decodes at: `ab_decode.t`'s default before it took the shipped 0.99, so the
start is unchanged. `sec:ab_decoder` names no stay probability for the start."""

LATTICE_DISPERSIONS = (0.5, 300.0)
"""The `(alpha, tau)` the lattice start decodes at, held rather than refitted."""

STAGE_CHANNELS = {"phasing": ("sp", True), "baf": ("sp", False), "rdr": ("smp", False)}
"""Each stage's fitted channels and whether its start is folded to the minor allele. NB phasing fits unphased counts,
so a state above 1/2 is its mirror and folding loses nothing; `baf` and `rdr` fit phased counts."""


def get_stage_channels(stage: str) -> tuple[str, bool]:
    """The channels `stage` fits and whether its start is folded to the minor allele.

    Args:
        stage: `phasing`, `baf` or `rdr`.

    Returns:
        The channels, e.g. `sp`, and the fold.

    Raises:
        ValueError: for an unknown stage.
    """
    if stage not in STAGE_CHANNELS:
        raise ValueError(f"{stage!r} is not one of {tuple(STAGE_CHANNELS)}.")
    return STAGE_CHANNELS[stage]


def get_default_parameters(n_states: int) -> CnaHmmParameters:
    """The start of every block no initializer sets: a uniform start, `log_mu` -0.1 to 0.1, `p` 0.05 to 0.45, alpha
    0.5 and tau 1,000.

    Args:
        n_states: the states.

    Returns:
        The `CnaHmmParameters`, `(n_states,)` each.
    """
    return CnaHmmParameters(
        np.log(np.ones(n_states) / n_states),
        np.linspace(-0.1, 0.1, n_states),
        np.linspace(0.05, 0.45, n_states),
        np.full(n_states, 0.5),
        # NB large dispersions tend to Binomial, a flat landscape: started just before.
        np.full(n_states, 1_000.0),
    )


def get_initial_parameters(counts: Any, labelling: np.ndarray, config: Any, *, stage: str) -> CnaHmmParameters:
    """The HMM's start at `stage`, on the pseudobulks of `labelling`'s clones stacked along bins.

    Args:
        counts: a `SpatioGenomicLabelledMultiCounts`.
        labelling: (n_spots,) each spot's clone.
        config: the run's.
        stage: `phasing`, `baf` or `rdr`.

    Returns:
        The `CnaHmmParameters`, `(n_states,)` each; `log_mu` and the dispersions the defaults where not started.

    Raises:
        ValueError: for an unknown stage or `hmm.initializer`.
    """
    params, only_minor = get_stage_channels(stage)
    pseudobulk = counts.with_labelling("clone", labelling).get_pseudobulk("clone").get_stacked()
    lengths = np.tile(counts.lengths, len(pseudobulk.clone_lengths))
    if config.hmm.initializer == "lattice":
        log_mu, p_binom = get_lattice_states(pseudobulk, lengths, config, params, only_minor)
    elif config.hmm.initializer == "calicost":
        log_mu, p_binom = get_calicost_states(pseudobulk, config, params, only_minor)
    else:
        raise ValueError(f"hmm.initializer {config.hmm.initializer!r} is not one of {get_hmm_initializers()}.")

    default = get_default_parameters(config.hmm.n_states)
    log_mu = np.asarray(log_mu).reshape(-1) if "m" in params else default.log_mu
    parameters = default._replace(log_mu=log_mu, p_binom=np.asarray(p_binom).reshape(-1))
    logger.info(f"Solved for hmm initialized parameters:\n{parameters.log_mu}\n{parameters.p_binom}")
    return parameters


def get_calicost_states(
    pseudobulk: PseudobulkCounts, config: Any, params: str, only_minor: bool
) -> tuple[np.ndarray | None, np.ndarray]:
    """CalicoST's `initialization_by_gmm` on the usable bins: those with a baseline where `m` is fit, and allele
    coverage where `p` is, in every clone; the BAF clip `hmm.gmm_min_binom_prob` to `hmm.gmm_max_binom_prob`.

    Args:
        pseudobulk: the stacked clones.
        config: the run's.
        params: the fitted channels.
        only_minor: fold to the minor allele.

    Returns:
        `(log_mu, p_binom)`, `log_mu` None where `m` is not fit.
    """
    low, high = float(config.hmm.gmm_min_binom_prob), float(config.hmm.gmm_max_binom_prob)
    counts = np.asarray(pseudobulk.X, dtype=np.float64)
    base = np.asarray(pseudobulk.normal_baseline_sized, dtype=np.float64).T
    total = np.asarray(pseudobulk.total_snp_umis, dtype=np.float64).T

    # NB CalicoST's has no inf guard: bins with no baseline and no coverage are dropped; the means are unchanged.
    usable = np.ones(counts.shape[0], dtype=bool)
    if "m" in params:
        usable &= np.all(base > 0.0, axis=1)
    if "p" in params:
        usable &= np.all(total > 0.0, axis=1)

    logger.info(
        f"Initializing HMM emission with CalicoST's GMM (only_minor={only_minor}, log_space=False, "
        f"clip=[{low}, {high}]); {int(usable.sum()):,} of {usable.size:,} bins usable."
    )
    log_mu, p_binom = gmm_start(
        config.hmm.n_states,
        counts[usable],
        base[usable],
        total[usable],
        params,
        random_state=config.hmm.random_state,
        in_log_space=False,
        only_minor=only_minor,
        min_binom_prob=low,
        max_binom_prob=high,
    )
    return log_mu, p_binom


def get_lattice_states(
    pseudobulk: PseudobulkCounts, lengths: np.ndarray, config: Any, params: str, only_minor: bool
) -> tuple[np.ndarray | None, np.ndarray]:
    """The `n_states` most used `(A, B)` states of `ab_decode.decode`'s decode, padded in the lattice's order.

    `only_minor` folds the lattice, keeping one state per minor fraction. `Z_c = 1`: the baseline is the normal
    reference; the dispersions are held at `LATTICE_DISPERSIONS`.

    Args:
        pseudobulk: the stacked clones.
        lengths: the stack's segment lengths.
        config: the run's.
        params: the fitted channels.
        only_minor: fold to the minor allele.

    Returns:
        `(log_mu, p_binom)`, `(n_states, 1)` each, `log_mu` None where `m` is not fit.
    """
    n_states = config.hmm.n_states
    X = np.asarray(pseudobulk.X, dtype=np.float64)
    counts = PseudobulkCounts(
        X[:, 0, :].T,
        np.asarray(pseudobulk.normal_baseline_sized, dtype=np.float64),
        X[:, 1, :].T,
        np.asarray(pseudobulk.total_snp_umis, dtype=np.float64),
        None,
        {},
    )
    alpha, tau = LATTICE_DISPERSIONS
    parameters = CnaHmmParameters(None, None, None, np.full(1, alpha), np.full(1, tau))
    decode_config = get_lattice_config(config, params, only_minor)
    decoded = decode_integer_copies(
        counts, parameters, get_unconfounded(X.shape[2]), np.asarray(lengths, dtype=np.int64), decode_config
    )

    pairs, counts_per = np.unique(decoded.copies.reshape(-1, 2), axis=0, return_counts=True)
    pairs = pairs[np.argsort(counts_per)[::-1][:n_states]]
    if pairs.shape[0] < n_states:
        # NB fewer states occupied than asked for: the rest in the lattice's order, the width the HMM expects.
        lattice = get_ab_lattice(
            params=params,
            fold=bool(only_minor),
            max_allele_copy=decode_config.ab_decode.max_allele_copy,
            max_total_copy=decode_config.ab_decode.max_total_copy,
        )
        spare = [pair for pair in lattice if not any(np.all(pair == row) for row in pairs)]
        pairs = np.vstack([pairs, np.array(spare[: n_states - pairs.shape[0]])])

    logger.info(
        f"Initializing HMM emission from the integer lattice (fold={bool(only_minor)}, "
        f"params={params!r}): {[tuple(int(v) for v in row) for row in pairs]}, "
        f"normal fractions {np.round(decoded.normal_fractions, 3).tolist()}, joint {decoded.score:,.1f}."
    )

    states = get_lattice_parameters(pairs, None, None)
    log_mu = states.log_mu.reshape(-1, 1) if "m" in params else None
    return log_mu, states.p_binom.reshape(-1, 1)


def get_lattice_config(config: Any, params: str, only_minor: bool) -> Any:
    """The configuration the lattice start decodes at: diploid, `fold` where `only_minor`, at `Z_m = 1`.

    Args:
        config: the run's.
        params: the fitted channels.
        only_minor: fold.

    Returns:
        A copy of `config`, its `ab_decode` overridden.
    """
    decode_config = copy.deepcopy(config)
    ab_decode = decode_config.ab_decode
    ab_decode.rdr, ab_decode.fold, ab_decode.fit_dispersions = "m" in params, bool(only_minor), False
    # NB at diploid with no A + B constraint, the start's decode needing no genome-wide hold.
    ab_decode.assumed_ploidy, ab_decode.genome_wide_ploidy_constraint = "diploid", False
    # NB the uniform contig start, not `ab_decode.contig_start`'s anchored one, which moved easy (2d4ce9a9)'s
    #    clones here, clone ARI 0.9884 to 0.9766, where it was measured for the final decode alone.
    ab_decode.specificity_prior, ab_decode.contig_start = 0.0, "uniform"
    # NB `sec:ab_decoder`'s start holds Z_m = 1, at its own stay probability.
    ab_decode.t, ab_decode.recompute_logmu_shift = LATTICE_T, False
    return decode_config
