"""`baum_welch`: EM over clone columns with shared parameters, the E step `hmm.forward_backward`'s and the M step
`hmm_nophasing`'s, then the posteriors at the fit."""

from typing import Any, NamedTuple

import numpy as np

from cnamaste.config import start_time
from cnamaste.emission import CnaEmissionConfounders, get_encoded, get_log_emission
from cnamaste.fit_checks import record_fit_check
from cnamaste.hmm.forward_backward import get_contig_offsets, get_forward_backward, get_llf, get_log_gamma
from cnamaste.hmm_nophasing import hmm_nophasing
from cnamaste.hmm_params import CnaHmmParameters
from cnamaste.logger import get_logger
from cnamaste.spatio_genomic_counts import PseudobulkCounts

logger = get_logger(__name__, start_time=start_time)


class HmmTransitions(NamedTuple):
    """The chain's transitions: `T_state`, `(K, K)`; `T_phase` as each bin's log phase switch, `(n_bins,)`, None
    unphased; and the bins per contig, where each clone's chain restarts."""

    log_transmat: np.ndarray
    log_sitewise_transmat: np.ndarray | None
    lengths: np.ndarray


class HmmFit(NamedTuple):
    """`baum_welch`'s fit: the parameters, each clone's confounders, the log posteriors, `(n_clones, n_bins, K)` or
    `(n_clones, n_bins, 2, K)` phased, and the log-likelihood."""

    parameters: CnaHmmParameters
    confounders: CnaEmissionConfounders
    log_gamma: np.ndarray
    llf: float


def get_paths(log_gamma: np.ndarray) -> np.ndarray:
    """Each clone's most probable copy state per bin, the phase dropped.

    Args:
        log_gamma: `(n_clones, n_bins, K)`, or `(n_clones, n_bins, 2, K)` phased.

    Returns:
        The `(n_clones, n_bins)` states.
    """
    return np.argmax(log_gamma.reshape(*log_gamma.shape[:2], -1), axis=-1) % log_gamma.shape[-1]


def get_posterior_fit(
    counts: PseudobulkCounts,
    parameters: CnaHmmParameters,
    confounders: CnaEmissionConfounders,
    transitions: HmmTransitions,
    *,
    penalize: bool,
) -> tuple[np.ndarray, float]:
    """The log posteriors and log-likelihood at fixed parameters, on the encoded emission with every data term.

    Args:
        counts: clone-major `PseudobulkCounts`.
        parameters: the states'.
        confounders: each clone's.
        transitions: the `HmmTransitions`.
        penalize: `hmm.penalize_phase_only_on_same_cnv`.

    Returns:
        `(log_gamma, llf)`, `log_gamma` `(n_clones, n_bins, S)`.
    """
    phased = transitions.log_sitewise_transmat is not None
    encoded = get_log_emission(
        get_encoded(counts, common_zero_depth=False),
        parameters,
        confounders,
        layout="lattice",
        encoding="coded",
        phased=phased,
        data_terms=True,
    )
    log_alpha, log_beta = get_forward_backward(
        encoded,
        parameters.log_startprob,
        transitions.log_transmat,
        transitions.lengths,
        transitions.log_sitewise_transmat,
        penalize,
    )
    return get_log_gamma(log_alpha, log_beta), get_llf(log_alpha, get_contig_offsets(transitions.lengths))


def baum_welch(
    counts: PseudobulkCounts,
    parameters: CnaHmmParameters,
    confounders: CnaEmissionConfounders,
    transitions: HmmTransitions,
    *,
    params: str,
    tol: float,
    previous_log_gamma: np.ndarray | None,
    config: Any,
) -> HmmFit:
    """EM over clone columns, one chain per clone and contig with shared parameters, then the posteriors at the fit.

    Args:
        counts: clone-major `PseudobulkCounts`, `(n_clones, n_bins)`.
        parameters: the start (`initialize.genomic`); a channel `params` fits must be set.
        confounders: each clone's `theta_c`; `log Z_c` is the fit's where `previous_log_gamma` is given, else held.
        transitions: the `HmmTransitions`; phased where `log_sitewise_transmat` is set.
        params: the channels fitted, of `s`, `m`, `p`.
        tol: the EM tolerance on `mu` and `p`.
        previous_log_gamma: the previous fit's log posteriors, `(n_clones, n_bins, K)`, which weigh `log Z_c`, or
            None.
        config: the run's.

    Returns:
        The `HmmFit`.

    Raises:
        ValueError: where a channel `params` fits has no start.
    """
    missing = [
        name
        for name, channel, value in (("log_mu", "m", parameters.log_mu), ("p_binom", "p", parameters.p_binom))
        if channel in params and value is None
    ]
    if missing:
        raise ValueError(f"baum_welch fits params={params!r} without {', '.join(missing)}: pass the start.")

    n_states = transitions.log_transmat.shape[0]
    stay = np.exp(transitions.log_transmat[0, 0])
    logger.info(
        f"Solving HMM for {np.shape(counts.umis)} clone columns, {n_states} states, parameters={params}; t={stay:.10g}."
    )
    if "m" in params:
        logger.info(f"Assumed initial log_mu:\n{parameters.log_mu}")
    if "p" in params:
        logger.info(f"Assumed initial p_binom:\n{parameters.p_binom}")

    penalize = bool(config.hmm.penalize_phase_only_on_same_cnv)
    model = hmm_nophasing(params=params)
    fitted, fit_confounders = model.run_optimization_pipeline(
        counts,
        transitions,
        parameters,
        confounders,
        config=config,
        max_iter_em=int(config.hmm.max_iter_em),
        mstep_callback_cadence=int(config.hmm.mstep_callback_cadence),
        callback_warm_start=bool(config.hmm.callback_warm_start),
        tol=tol,
        log_gamma=previous_log_gamma,
        penalize=penalize,
    )

    to_log = ["Solved for best emission parameters:"]
    if "m" in params:
        to_log += [f"mu=\n{np.exp(fitted.log_mu)}", f"alphas=\n{fitted.alphas}"]
    if "p" in params:
        to_log += [f"p_binom=\n{fitted.p_binom}", f"taus=\n{fitted.taus}"]
    logger.info("\n".join(to_log))

    log_gamma, llf = get_posterior_fit(counts, fitted, fit_confounders, transitions, penalize=penalize)
    if transitions.log_sitewise_transmat is not None:
        log_gamma = log_gamma.reshape(*log_gamma.shape[:2], 2, n_states)
    logger.info(f"Solved HMM with LLF={llf:.6e} for log_mu.shape={np.shape(fitted.log_mu)}.")
    if not np.isfinite(llf):
        record_fit_check("non_finite_llf", f"the fit's llf is {llf} over {np.shape(counts.umis)} clone columns.")
    return HmmFit(fitted, fit_confounders, log_gamma, llf)
