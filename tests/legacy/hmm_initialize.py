"""Frozen copy of `hmm_initialize` before T- #102 PR 1, and of `hmm_nophasing.get_initial_params`; `lattice_init` takes
T- #100 PR 2's clone-major `PseudobulkCounts`, as the original did."""

from __future__ import annotations

import copy

import numpy as np

from cnamaste.ab_decode.decode import decode_integer_copies
from cnamaste.ab_decode.lattice import get_ab_lattice, get_lattice_parameters
from cnamaste.calicost import gmm_start as initialization_by_gmm
from cnamaste.config import YAMLConfig, get_global_config, get_hmm_initializers, start_time
from cnamaste.emission import get_unconfounded
from cnamaste.hmm_params import CnaHmmParameters
from cnamaste.hmm_utils import get_log_transmat
from cnamaste.logger import get_logger
from cnamaste.spatio_genomic_counts import PseudobulkCounts

logger = get_logger(__name__, start_time=start_time)

LATTICE_INIT_T = 1.0 - 1.0e-6
"""The stay probability the start decodes at: `ab_decode.t`'s default before it took the shipped 0.99, so the
start is unchanged. `sec:ab_decoder` names no stay probability for the start."""

__all__ = [
    "calicost_init",
    "configured_initializer",
    "lattice_init",
    "minor_allele_only_at_stage",
]


# NB equivalent to CalicoST utils_hmm.initialization_by_gmm, on the usable bins.
def calicost_init(
    n_states,
    X,
    base_nb_mean,
    total_bb_RD,
    params,
    lengths=None,
    log_transmat=None,
    log_sitewise_transmat=None,
    random_state=None,
    in_log_space=True,
    only_minor=True,
    mirrored_baf_augmentation=True,
):
    """`initialization_by_gmm`, CalicoST's start, on the module's shared signature and 4-tuple.

    `lengths`, `log_transmat` and `log_sitewise_transmat` are accepted and unused. The BAF clip is
    `hmm.gmm_min_binom_prob` / `hmm.gmm_max_binom_prob` where set; `mirrored_baf_augmentation` is accepted and ignored,
    as CalicoST does not augment.

    Args: `n_states`; `X` (n_obs, 2, n_spots) counts; `base_nb_mean`, `total_bb_RD` (n_obs, n_spots); `params` channels.
    Returns: `(log_mu, p_binom, alphas, taus)`, `None` where not fitted.
    """
    try:
        hmm = get_global_config().hmm
        low, high = float(hmm.gmm_min_binom_prob), float(hmm.gmm_max_binom_prob)
    except AttributeError:
        low, high = 0.1, 0.9

    # NB CalicoST's has no inf guard: bins with no baseline and no coverage are dropped; the means are unchanged.
    counts = np.asarray(X, dtype=np.float64)
    base = np.asarray(base_nb_mean, dtype=np.float64)
    total = np.asarray(total_bb_RD, dtype=np.float64)

    usable = np.ones(counts.shape[0], dtype=bool)

    if "m" in params:
        usable &= np.all(base > 0.0, axis=1)

    if "p" in params:
        usable &= np.all(total > 0.0, axis=1)

    logger.info(
        f"Initializing HMM emission with CalicoST's GMM (only_minor={only_minor}, "
        f"log_space={in_log_space}, clip=[{low}, {high}]); "
        f"{int(usable.sum()):,} of {usable.size:,} bins usable."
    )

    log_mu, p_binom = initialization_by_gmm(
        n_states,
        counts[usable],
        base[usable],
        total[usable],
        params,
        random_state=random_state,
        in_log_space=in_log_space,
        only_minor=only_minor,
        min_binom_prob=low,
        max_binom_prob=high,
    )

    return log_mu, p_binom, None, None


def lattice_init(
    n_states,
    X,
    base_nb_mean,
    total_bb_RD,
    params,
    lengths=None,
    log_transmat=None,
    log_sitewise_transmat=None,
    random_state=None,
    in_log_space=True,
    only_minor=True,
    mirrored_baf_augmentation=True,
):
    """A start decoded over `ab_decode.decode`'s own `(A, B)` lattice.

    Every center is a copy state the decoder can return. `only_minor` folds the lattice, keeping one state per minor
    fraction; otherwise it is reflected onto `(0, 1)`. `Z_c = 1`: the baseline is the normal reference. The dispersions
    are held at `hmm_emission`'s start rather than refitted.

    Args: `n_states`; `X` (n_obs, 2, n_spots) counts; `base_nb_mean`, `total_bb_RD` (n_obs, n_spots); `params` channels.
    Returns: `(log_mu, p_binom, alphas, taus)`, `None` where not fitted.
    """

    X = np.asarray(X, dtype=np.float64)
    counts = PseudobulkCounts(
        X[:, 0, :].T,
        np.asarray(base_nb_mean, dtype=np.float64).T,
        X[:, 1, :].T,
        np.asarray(total_bb_RD, dtype=np.float64).T,
        None,
        {},
    )

    if lengths is None:
        lengths = np.array([X.shape[0]], dtype=np.int64)

    alpha, tau = _starting_dispersions()
    parameters = CnaHmmParameters(None, None, None, np.full(1, alpha), np.full(1, tau))
    config = get_lattice_init_config(params, only_minor)
    decoded = decode_integer_copies(
        counts, parameters, get_unconfounded(X.shape[2]), np.asarray(lengths, dtype=np.int64), config
    )
    copies, fractions, score = decoded.copies, decoded.normal_fractions, decoded.score

    pairs, counts_per = np.unique(copies.reshape(-1, 2), axis=0, return_counts=True)
    order = np.argsort(counts_per)[::-1][:n_states]
    pairs = pairs[order]

    if pairs.shape[0] < n_states:
        # NB fewer states occupied than asked for: the rest are the lattice's
        #    own order, so the HMM is handed the width it expects.

        spare = [
            pair
            for pair in get_ab_lattice(
                params=params,
                fold=bool(only_minor),
                max_allele_copy=config.ab_decode.max_allele_copy,
                max_total_copy=config.ab_decode.max_total_copy,
            )
            if not any(np.all(pair == row) for row in pairs)
        ]
        pairs = np.vstack([pairs, np.array(spare[: n_states - pairs.shape[0]])])

    logger.info(
        f"Initializing HMM emission from the integer lattice (fold={bool(only_minor)}, "
        f"params={params!r}): {[tuple(int(v) for v in row) for row in pairs]}, "
        f"normal fractions {np.round(fractions, 3).tolist()}, joint {score:,.1f}."
    )

    states = get_lattice_parameters(pairs, None, None)
    log_mu = states.log_mu.reshape(-1, 1) if "m" in params else None
    p_binom = states.p_binom.reshape(-1, 1) if "p" in params else None

    return log_mu, p_binom, None, None


def get_lattice_init_config(params, only_minor):
    """The run's configuration as the start decodes: diploid, `fold` where `only_minor`, at `Z_m = 1`.

    Args: `params` the fit's channels; `only_minor` fold.
    Returns: a copy of the run's `YAMLConfig`, the defaults where none is set, its `ab_decode` overridden.
    """
    config = copy.deepcopy(get_global_config() or YAMLConfig({}))
    ab_decode = config.ab_decode
    ab_decode.rdr, ab_decode.fold, ab_decode.fit_dispersions = "m" in params, bool(only_minor), False
    # NB at diploid with no A + B constraint, the start's decode needing no genome-wide hold.
    ab_decode.assumed_ploidy, ab_decode.genome_wide_ploidy_constraint = "diploid", False
    # NB the uniform contig start, not `ab_decode.contig_start`'s anchored one, which moved easy (2d4ce9a9)'s
    #    clones here, clone ARI 0.9884 to 0.9766, where it was measured for the final decode alone.
    ab_decode.specificity_prior, ab_decode.contig_start = 0.0, "uniform"
    # NB `sec:ab_decoder`'s start holds Z_m = 1, at its own stay probability.
    ab_decode.t, ab_decode.recompute_logmu_shift = LATTICE_INIT_T, False
    return config


def _starting_dispersions():
    """`hmm_emission`'s own start, or the configuration's where it names one.

    Returns: the starting `(alpha, tau)`.
    """
    try:
        hmm = get_global_config().hmm
        return float(hmm.gmm_start_alpha), float(hmm.gmm_start_tau)
    except AttributeError:
        return 0.5, 300.0


def configured_initializer(requested=None):
    """The HMM's start, as `hmm.initializer` names it.

    A callable `requested` wins; `None` reads the configuration's `hmm.initializer`. Every start it can name is in
    this module: the run reaches nothing set aside.

    Args: `requested` an initializer callable, or None for the configuration's.
    Returns: the initializer function.
    """
    if callable(requested):
        return requested

    # NB an attribute read rather than `getattr`, because that is the form
    #    the configuration-key scanner recognizes as reading `hmm.initializer`.
    configuration = get_global_config()

    name = configuration.hmm.initializer

    if name == "calicost":
        return calicost_init
    elif name == "lattice":
        return lattice_init
    else:
        raise ValueError(f"hmm.initializer {name!r} is not one of {get_hmm_initializers()}.")


def minor_allele_only_at_stage(stage: str) -> bool:
    """Whether `stage`'s start is folded to the minor allele, refusing a stage it does not name.

    Args: `stage` `phasing` or `core_inference`.
    Raises: ValueError for an unknown stage.
    Returns: whether the stage fits the minor allele only.
    """
    # NB phasing fits unphased counts with `hmm_phased`, so a state above 1/2 is its mirror and folding loses
    #    nothing; core inference fits phased counts, where folding would merge a state with its mirror.
    ONLY_MINOR: dict[str, bool] = {"phasing": True, "core_inference": False}

    if stage not in ONLY_MINOR:
        raise ValueError(f"{stage!r} is not one of {tuple(ONLY_MINOR)}.")

    return ONLY_MINOR[stage]


def get_initial_params(
    t,
    n_states,
    n_spots,
    init_log_mu=None,  # DEPRECATE
    init_p_binom=None,  # DEPRECATE
    init_alphas=None,  # DEPRECATE
    init_taus=None,  # DEPRECATE
):
    """`hmm_nophasing.get_initial_params`, verbatim, `self.t` as `t`."""
    # TODO' use self.default_log_mu on class instance
    log_mu = (
        np.vstack([np.linspace(-0.1, 0.1, n_states) for _ in range(n_spots)]).T if init_log_mu is None else init_log_mu
    )

    # TODO' define self.default_p_binom on class instance
    p_binom = (
        np.vstack([np.linspace(0.05, 0.45, n_states) for _ in range(n_spots)]).T
        if init_p_binom is None
        else init_p_binom
    )

    # NB small alpha tend to Poisson. 0.1 ->
    alphas = 0.5 * np.ones((n_states, n_spots)) if init_alphas is None else init_alphas

    # TODO' define ...
    # NB large dispersions tend to Binomial, flat landscape, initialize just before.  30 -> 1_000
    taus = 1_000 * np.ones((n_states, n_spots)) if init_taus is None else init_taus

    # NB initialize start probability and emission probability
    log_startprob = np.log(np.ones(n_states) / n_states)

    log_transmat = get_log_transmat(n_states, t)

    return log_mu, p_binom, alphas, taus, log_startprob, log_transmat
