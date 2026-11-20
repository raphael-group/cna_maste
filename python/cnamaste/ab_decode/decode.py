"""
(A,B) decoder for (allele-specific) integer copy numbers given a K-state (mu, p) hmm fit.

The reference is `cna-maste-paper` (81813c5), section `sec:ab_decoder`. Equations are cited by their label, not by
their number: `eqn:nb_emission` and `eqn:bb_emission` the emission, `eqn:ab_admixture` the normal admixture,
`eqn:ab_hmm` the scored copy path, `eqn:ab_coupling` the clone agreement prior.

    paper       here                              paper    here
    theta       normal_fraction                   eta_s    specificity_prior
    mu~, p~     `emission.get_admixed_moments`    eta_c    clone_coupling
    Z_m         exp(log_mu_shift)                 eta_t    the price `clone_viterbi_mean_ploidy_constraint` bisects
    G, M        n_obs, n_clones                   P        ploidy
    T_bar_m     the copy path's mean A + B        K        the ab_lattice's pairs

Known departures from the reference, each stated again where it is made: `clone_viterbi_mean_ploidy_constraint`
bisects the ploidy price where the paper steps it; `fold` scores the minor count, which the paper does not do, and is
reached only by `initialize.genomic`. The types and small helpers are `ab_decode.model`'.
"""

from __future__ import annotations

import functools
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from math import lgamma
from multiprocessing import get_context
from typing import Any

import numpy as np
import scipy.sparse as sp
from numba import njit
from scipy.optimize import minimize, minimize_scalar

from cnamaste.ab_decode.lattice import (
    get_ab_is_nonbalanced,
    get_ab_is_ploidy_neutral,
    get_ab_lattice,
    get_lattice_parameters,
)
from cnamaste.ab_decode.model import (
    AbDecodeResult,
    CloneChain,
    DecodeStatus,
    get_all_clone_agreement_prior,
    get_clone_agreement_prior,
    get_clone_chain,
    get_clone_rows,
    get_clone_status,
    get_coded_rows,
    get_copy_path_emission_loglike,
    get_copy_path_loglike,
    get_log_mu_shift_start,
    get_observed_baf_profiles,
    get_ploidy_band,
    in_ploidy_band,
)
from cnamaste.config import assumed_ploidy, get_hmm_ab_solvers, get_ploidy_by_name, normal_fraction_starts, start_time
from cnamaste.emission import (
    CnaEmissionConfounders,
    EncodedEmission,
    get_admixed_moments,
    get_decoded_emission,
    get_hard_state_posteriors,
    get_log_emission,
    get_log_mu_shift,
    get_minor_allele_count,
    get_nb_mean,
    get_state_major,
)
from cnamaste.external.digamma import digamma_element
from cnamaste.hmm.forward_backward import get_dense_encoded_emission, get_viterbi_paths
from cnamaste.hmm_emission_bb import bb_logpmf_element
from cnamaste.hmm_emission_nb import nb_logpmf_element
from cnamaste.hmm_nophasing import get_max_rdr_zeroed_baseline
from cnamaste.hmm_params import CnaHmmParameters
from cnamaste.hmm_utils import digamma_difference_element, get_normal_clone
from cnamaste.logger import get_logger
from cnamaste.opengm import opengm_session
from cnamaste.potts import PottsProblem, potts_problem
from cnamaste.spatio_genomic_counts import PseudobulkCounts

logger = get_logger(__name__, start_time=start_time)

ALPHA_FLOOR = 1e-6
"""The least NB dispersion the decoder's dispersion refit takes."""

PLOIDY_PRICE_MAX = 1.0e6
"""The largest price per bin `clone_viterbi_mean_ploidy_constraint` pays before it reports that no copy path holds the
ploidy."""

NORMAL_FRACTION_BOUNDS = (0.0, 1.0)
"""The interval a tumour clone's `theta_c` is fit on, `eqn:ab_admixture`; bounded Brent does not take 1."""

NORMAL_FRACTION_XATOL = 1.0e-4
"""Bounded Brent's absolute tolerance on `theta_c`, and the hard EM's on a round's change in it."""


ClonesDecode = tuple[dict[int, np.ndarray], dict[int, float], dict[int, DecodeStatus], float]
"""A clones decode's copy paths, own loglikes and `DecodeStatus`, by clone, and the joint loglike."""


def get_clone_viterbi(encoded: EncodedEmission, chain: CloneChain) -> tuple[np.ndarray, np.ndarray]:
    """Each encoded clone's best copy path on `chain`, `eqn:ab_hmm`: `forward_backward.get_viterbi_paths` on the chain's
    transitions and contig start prior.

    Args:
        encoded: the `(n_clones, n_classes, K)` `EncodedEmission`.
        chain: the `CloneChain`.

    Returns:
        `(n_clones, n_obs)` copy paths and `(n_clones,)` loglikes.
    """
    return get_viterbi_paths(encoded, chain.log_start, chain.transitions.log_transmat, chain.lengths)


def clone_viterbi_mean_ploidy_constraint(
    emission_loglike: np.ndarray,
    chain: CloneChain,
    totals: np.ndarray,
    ploidy: int,
    *,
    ploidy_price_steps: int,
    name: str,
) -> tuple[np.ndarray, float, float, str]:
    """`eqn:ab_hmm`'s `eta_t` term: hold the copy path's mean `A + B` within `PLOIDY_BAND` of the ploidy by pricing each
    state's total, every solve a plain Viterbi.

    The mean is linear in the copy path, so the price `price * (totals - ploidy)` moves it monotonically. The free path
    is kept where its mean already sits in the band; otherwise the price is bracketed by doubling and bisected to the
    smallest that brings the mean inside, a departure: the paper steps the price.

    Args:
        emission_loglike: `(K, n_obs)` log score.
        chain: the `CloneChain`.
        totals: `(K,)` each pair's `A + B`.
        ploidy: P.
        ploidy_price_steps: the price's bisections, once doubling has bracketed it.
        name: labels the log.

    Returns:
        The copy path, its loglike on the unpriced score, the price paid, and what held it; the free path at -inf where
        no price lands the mean in the band.
    """
    lo, hi = ploidy - get_ploidy_band(), ploidy + get_ploidy_band()
    encoded = get_dense_encoded_emission(emission_loglike.T[None])

    def decode(price: float) -> np.ndarray:
        # NB the price per pair is a term on the class tables: every bin of a pair pays it.
        priced = encoded._replace(rdr=encoded.rdr + price * (totals - ploidy))
        return get_clone_viterbi(priced, chain)[0][0]

    def mean(copy_path: np.ndarray) -> float:
        return float(totals[copy_path].mean())

    copy_path = free = decode(0.0)
    found = mean(copy_path)
    if in_ploidy_band(copy_path, totals, ploidy):
        return copy_path, get_copy_path_loglike(emission_loglike, chain, copy_path), 0.0, f"free, mean {found:.4f}"
    # NB the mean rises with the price: buy totals where the copy path is light, sell where it is heavy.
    want = (lambda m: m >= lo) if found < lo else (lambda m: m <= hi)
    step = 1.0 if found < lo else -1.0
    near, far = 0.0, step
    while abs(far) <= PLOIDY_PRICE_MAX and not want(mean(decode(far))):
        near, far = far, far * 2.0
    if abs(far) > PLOIDY_PRICE_MAX:
        logger.warning(
            f"{name}: no price under {PLOIDY_PRICE_MAX:.0e} brings the mean A + B to {ploidy} +/- {get_ploidy_band()}."
        )
        return free, -np.inf, float(far), "no price reaches the band"
    for _ in range(ploidy_price_steps):
        mid = 0.5 * (near + far)
        if want(mean(decode(mid))):
            far = mid
        else:
            near = mid
    copy_path = decode(far)
    found = mean(copy_path)
    if not in_ploidy_band(copy_path, totals, ploidy):
        logger.warning(f"{name}: the mean A + B steps over {ploidy} +/- {get_ploidy_band()}: {found:.4f} at {far:.4g}.")
        return free, -np.inf, float(far), f"stepped over the band, mean {found:.4f}"
    loglike = get_copy_path_loglike(emission_loglike, chain, copy_path)
    return copy_path, loglike, float(far), f"priced {far:.4g}, mean {found:.4f}"


def clone_viterbi(
    chain: CloneChain,
    clone: int,
    emission_loglike: np.ndarray,
    ploidy_band: tuple[np.ndarray, int] | None,
    *,
    ploidy_price_steps: int,
) -> tuple[np.ndarray, float]:
    """Clone `clone`'s best copy path on `chain`, `eqn:ab_hmm`, and its loglike.

    Args:
        chain: the `CloneChain`.
        clone: the clone, for the log.
        emission_loglike: `(K, n_obs)` log score.
        ploidy_band: `(totals, ploidy)` holds the mean `A + B` (`clone_viterbi_mean_ploidy_constraint`); None for none.
        ploidy_price_steps: the ploidy price's bisections.

    Returns:
        The copy path and its loglike, -inf where the band holds none.

    Raises:
        ValueError: where `emission_loglike` does not score the lattice's K pairs.
    """
    if emission_loglike.shape[0] != chain.K:
        raise ValueError(f"{emission_loglike.shape[0]} states scored against a {chain.K}-pair lattice")
    if ploidy_band is not None:
        copy_path, value, _, detail = clone_viterbi_mean_ploidy_constraint(
            emission_loglike, chain, *ploidy_band, ploidy_price_steps=ploidy_price_steps, name=f"hmm_ab: clone {clone}"
        )
        logger.info(f"hmm_ab: clone {clone} A + B {ploidy_band[1]} by mean, {detail}, loglike {value:.6e}.")
        return copy_path, float(value)
    paths, scores = get_clone_viterbi(get_dense_encoded_emission(emission_loglike.T[None]), chain)
    return paths[0], float(scores[0])


def get_clone_chains_potts_problem(
    chain: CloneChain, emission_loglikes: dict[int, np.ndarray], clones: Sequence[int]
) -> PottsProblem:
    """`eqn:ab_coupling` as one Potts field: `C x n_obs` nodes, `K` labels, every edge a reward for agreeing.

    A clone's step pays `log_switch` everywhere plus `log_stay - log_switch` where it stays; the agreement prior pays
    a constant plus `clone_coupling` per agreeing pair. Both rewards are non-negative, so the field is metric Potts and
    every expansion move is a minimum cut.

    Args:
        chain: the `CloneChain`.
        emission_loglikes: the `(K, n_obs)` log score, by clone.
        clones: the ids, in node order.

    Returns:
        The `PottsProblem`, `potts.potts_problem` as `hmrf` poses it, its nodes `c * n_obs + g`.
    """
    n_bins = emission_loglikes[clones[0]].shape[1]
    starts = np.concatenate([[0], np.cumsum(chain.lengths)[:-1]]).astype(np.int64)
    first_bin = np.zeros(n_bins, dtype=bool)
    first_bin[starts] = True
    # NB the unary is a cost, so the loglike is negated; a segment's first bin carries log_start.
    unary = np.concatenate(
        [-(emission_loglikes[c].T + np.where(first_bin[:, None], chain.log_start[None, :], 0.0)) for c in clones]
    )
    bins = np.arange(n_bins)
    inside = bins[:-1][~first_bin[1:]]
    chain_first = [i * n_bins + inside for i in range(len(clones))]
    chain_second = [i * n_bins + inside + 1 for i in range(len(clones))]
    pair_first, pair_second = [], []
    for i in range(len(clones)):
        for k in range(i + 1, len(clones)):
            pair_first.append(i * n_bins + bins)
            pair_second.append(k * n_bins + bins)
    coupling = np.concatenate(
        [
            np.full(sum(len(a) for a in chain_first), chain.log_stay - chain.log_switch),
            np.full(sum(len(a) for a in pair_first), chain.clone_coupling),
        ]
    )
    # NB potts_problem halves a stored edge's weight at spatial_weight 1: each edge stored once at twice its coupling.
    edges = (np.concatenate(chain_first + pair_first), np.concatenate(chain_second + pair_second))
    adjacency = sp.coo_matrix((2.0 * coupling, edges), shape=(len(unary), len(unary)))
    return potts_problem(-unary, adjacency, 1.0)


def get_own_loglikes(
    chain: CloneChain, emission_loglikes: dict[int, np.ndarray], copy_paths: dict[int, np.ndarray]
) -> dict[int, float]:
    """Each clone's own `eqn:ab_hmm` loglike, the agreement prior excluded.

    Args:
        chain: the `CloneChain`.
        emission_loglikes: the `(K, n_obs)` log score, by clone.
        copy_paths: the copy path, by clone.

    Returns:
        The loglike, by clone.
    """
    return {c: get_copy_path_loglike(emission_loglikes[c], chain, copy_paths[c]) for c in sorted(emission_loglikes)}


def solve_clone_chains(
    chain: CloneChain,
    emission_loglikes: dict[int, np.ndarray],
    ploidy_band: tuple[np.ndarray, int] | None,
    *,
    solver: str,
    mincut: str,
    icm_sweeps: int,
    ploidy_price_steps: int,
) -> ClonesDecode:
    """The tumour clones' joint `eqn:ab_coupling` decode, `get_clone_chains_potts_problem`, by `solver`.

    `expansion` is OpenGM's alpha expansion from each clone's own Viterbi, finished by `icm` under the band, since no cut holds a
    mean. `icm` decodes one clone at a time with the others held (`get_clone_agreement_prior`), swept to a fixed
    point, then under the band to a fixed point of its own.

    Args:
        chain: the `CloneChain`.
        emission_loglikes: the `(K, n_obs)` log score, by clone.
        ploidy_band: `(totals, ploidy)`, or None.
        solver: one of `icm`, `expansion`.
        mincut: OpenGM's min cut, for `expansion`.
        icm_sweeps: the most sweeps `icm` takes, free and then held.
        ploidy_price_steps: the ploidy price's bisections.

    Returns:
        The copy paths, each clone's own loglike and `DecodeStatus`, by clone, and the joint loglike, -inf where the
        ploidy holds no copy path.

    Raises:
        ValueError: on an unknown `solver`.
    """
    if solver not in get_hmm_ab_solvers():
        raise ValueError(f"solver is one of {get_hmm_ab_solvers()}, not {solver!r}.")
    clones = sorted(emission_loglikes)
    n_bins = emission_loglikes[clones[0]].shape[1]
    copy_paths = {
        c: clone_viterbi(chain, c, emission_loglikes[c], ploidy_band, ploidy_price_steps=ploidy_price_steps)[0]
        for c in clones
    }
    if solver == "expansion":
        start = np.concatenate([copy_paths[c] for c in clones])
        problem = get_clone_chains_potts_problem(chain, emission_loglikes, clones)
        run = opengm_session().solve(problem, "expansion", start=start, mincut=mincut)
        copy_paths = {c: run.labels[i * n_bins : (i + 1) * n_bins] for i, c in enumerate(clones)}
        logger.info(f"hmm_ab: OpenGM alpha expansion, {run.iterations} iterations over {len(clones)} clones.")
        if ploidy_band is None:
            own = get_own_loglikes(chain, emission_loglikes, copy_paths)
            joint = float(sum(own.values()) + get_all_clone_agreement_prior(copy_paths, chain.clone_coupling))
            return copy_paths, own, get_clone_status(own), joint

    def decode(c: int, held: bool) -> tuple[np.ndarray, float]:
        prior = get_clone_agreement_prior(copy_paths, c, chain.K, n_bins, chain.clone_coupling)
        return clone_viterbi(
            chain, c, emission_loglikes[c] + prior, ploidy_band if held else None, ploidy_price_steps=ploidy_price_steps
        )

    # NB the sweeps run on the free chain and the held ones last: a held decode is a bisection of Viterbis.
    for _ in range(icm_sweeps):
        moved = 0
        for c in clones:
            found = decode(c, held=False)[0]
            moved += int(np.count_nonzero(found != copy_paths[c]))
            copy_paths[c] = found
        if moved == 0:
            break
    else:
        logger.warning(f"hmm_ab: clones still moving after {icm_sweeps} ICM sweeps.")
    if ploidy_band is not None:
        # NB the first held pass moves clones onto the band; a later one that moves them is a sweep, to a fixed point.
        for sweep in range(icm_sweeps):
            held, failed = 0, set()
            for c in clones:
                found, value = decode(c, held=True)
                held += int(np.count_nonzero(found != copy_paths[c]))
                copy_paths[c] = found
                if not np.isfinite(value):
                    failed.add(c)
            if failed or (sweep and held == 0) or len(clones) == 1 or chain.clone_coupling == 0.0:
                break
        else:
            logger.warning(f"hmm_ab: the ploidy still moves clones after {icm_sweeps} held ICM sweeps.")
        if failed:
            logger.warning(f"hmm_ab: the ploidy holds no copy path for clones {sorted(failed)}.")
            own = get_own_loglikes(chain, emission_loglikes, copy_paths)
            for c in failed:
                own[c] = -np.inf
            return copy_paths, own, get_clone_status(own), -np.inf
    own = get_own_loglikes(chain, emission_loglikes, copy_paths)
    joint = float(sum(own.values()) + get_all_clone_agreement_prior(copy_paths, chain.clone_coupling))
    return copy_paths, own, get_clone_status(own), joint


def fit_normal_fraction(
    counts: PseudobulkCounts,
    clone: int,
    parameters: CnaHmmParameters,
    copy_path: np.ndarray,
    nonbalanced: np.ndarray,
    log_mu_shift: float,
    *,
    fold: bool,
    baf_only: bool,
) -> float:
    """A tumour clone's `theta_c` on its copy path's nonbalanced bins at its held `log Z_c`.

    Bounded Brent on `NORMAL_FRACTION_BOUNDS`, and the lower bound itself, which Brent does not evaluate, kept where it
    scores at least as high, so a pure clone and a flat score are 0.

    Args:
        counts: `(n_clones, n_obs)` the clones' `PseudobulkCounts`.
        clone: the clone fit.
        parameters: the lattice's, `ab_lattice.get_lattice_parameters`, with the dispersions.
        copy_path: the state per bin.
        nonbalanced: `(K,)` each lattice pair's `get_ab_is_nonbalanced`.
        log_mu_shift: the held `log Z_c`.
        fold: score the minor count.
        baf_only: score the B-allele fraction alone, the read depth dropped.

    Returns:
        `theta_c`.
    """
    bins = nonbalanced[copy_path]
    kept = get_clone_rows(counts, np.s_[clone : clone + 1, bins])
    if baf_only:
        kept = kept._replace(umis=None, normal_baseline_sized=None)

    def loglike(theta: float) -> float:
        confounders = CnaEmissionConfounders(np.full((1, 1), theta), np.full((1, 1), log_mu_shift))
        return get_copy_path_emission_loglike(
            kept, parameters, confounders, copy_path[bins], fold=fold, data_terms=False
        )

    lo, hi = NORMAL_FRACTION_BOUNDS
    sol = minimize_scalar(
        lambda theta: -loglike(theta), bounds=(lo, hi), method="bounded", options={"xatol": NORMAL_FRACTION_XATOL}
    )
    if not sol.success:
        logger.warning(f"hmm_ab: bounded Brent on theta_c stopped unconverged: {sol.message}")
    return lo if loglike(lo) >= -float(sol.fun) else float(sol.x)


@njit(cache=True, error_model="numpy")
def get_dispersion_loglike_gradient(
    y: np.ndarray, mean: np.ndarray, b_count: np.ndarray, total: np.ndarray, p: np.ndarray, alpha: float, tau: float
) -> tuple[float, float, float]:
    """The summed `eqn:nb_emission` plus `eqn:bb_emission` loglike over bins, less the terms the counts alone set, and
    its derivatives in `log alpha` and `log tau`, per bin.

    Args:
        y: `(n,)` read counts.
        mean: `(n,)` NB means, 0 where the bin carries no read depth.
        b_count: `(n,)` B-allele counts, the minor count where folded.
        total: `(n,)` allele totals.
        p: `(n,)` each bin's B fraction.
        alpha: NB dispersion.
        tau: BB concentration.

    Returns:
        The loglike, `d / d log alpha` and `d / d log tau`.
    """
    floored = max(alpha, 1.0e-10)
    r, size, log_alpha = 1.0 / floored, 1.0 / alpha, np.log(alpha)
    lgamma_r = lgamma(r)
    value, by_alpha, by_tau = 0.0, 0.0, 0.0
    for g in range(y.size):
        if mean[g] > 0.0:
            value += nb_logpmf_element(y[g], mean[g], floored, r, lgamma_r, 0.0)
            z = log_alpha + np.log(mean[g])
            w = 1.0 / (1.0 + np.exp(-z))
            by_alpha -= size * (
                digamma_element(y[g] + size)
                - digamma_element(size)
                - np.logaddexp(0.0, z)
                + w
                - alpha * y[g] * (1.0 - w)
            )
        a, b = max(p[g] * tau, 1e-10), max((1.0 - p[g]) * tau, 1e-10)
        value += bb_logpmf_element(b_count[g], total[g], a, b, False)
        by_a, by_b = digamma_difference_element(a, b_count[g]), digamma_difference_element(b, total[g] - b_count[g])
        by_tau += a * by_a + b * by_b - (a + b) * digamma_difference_element(a + b, total[g])
    return value, by_alpha, by_tau


def hmm_ab_decoder_dispersion_objective(
    log_dispersions: np.ndarray,
    counts: PseudobulkCounts,
    clones: Sequence[int],
    ab_lattice: np.ndarray,
    copy_paths: dict[int, np.ndarray],
    clone_confounders: CnaEmissionConfounders,
    *,
    fold: bool,
) -> tuple[float, np.ndarray]:
    """The negative `eqn:nb_emission` plus `eqn:bb_emission` loglike of the clones' copy paths at `(log alpha, log
    tau)`, less the terms the counts alone set, and its gradient: `get_dispersion_loglike_gradient` over every clone's
    bins.

    Args:
        log_dispersions: `(log alpha, log tau)`.
        counts: `(n_clones, n_obs)` the clones' `PseudobulkCounts`, both channels.
        clones: the clones scored.
        ab_lattice: `(K, 2)` the pairs.
        copy_paths: the state per bin, by clone.
        clone_confounders: each clone's `theta_c` and `log Z_c`, `(n_clones, 1)`.
        fold: score the minor count.

    Returns:
        The negative loglike and its `(2,)` gradient.
    """
    alpha, tau = (float(v) for v in np.exp(log_dispersions))
    parameters = get_lattice_parameters(ab_lattice, None, None)
    columns: list[list[np.ndarray]] = [[], [], [], [], []]
    for c in clones:
        mu, p = get_admixed_moments(parameters, clone_confounders.normal_fraction[c, 0])
        copy_path = copy_paths[c]
        base, total = counts.normal_baseline_sized[c], counts.total_snp_umis[c]
        mean = np.where(base > 0, get_nb_mean(base, mu[copy_path], clone_confounders.log_mu_shift[c, 0]), 0.0)
        b_count = get_minor_allele_count(counts.alt_snp_umis[c], total) if fold else counts.alt_snp_umis[c]
        for column, each in zip(columns, (counts.umis[c], mean, b_count, total, p[copy_path]), strict=True):
            column.append(np.asarray(each, dtype=np.float64))
    y, mean, b_count, total, p = (np.concatenate(column) for column in columns)
    value, by_alpha, by_tau = get_dispersion_loglike_gradient(y, mean, b_count, total, p, alpha, tau)
    return -value, -np.array([by_alpha, by_tau])


def get_clone_emission_loglikes(
    counts: PseudobulkCounts,
    parameters: CnaHmmParameters,
    confounders: CnaEmissionConfounders,
    prior: np.ndarray,
    *,
    params: str,
    clones: Sequence[int],
) -> dict[int, np.ndarray]:
    """Each tumour clone's `(K, n_obs)` encoded-lattice loglike, `eqn:nb_emission` plus `eqn:bb_emission`, less the
    `eta_s` prior and the terms the counts alone set, the same in every pair (`data_terms=False`).

    Args:
        counts: `(n_tumour, n_obs)` the tumour clones' `PseudobulkCounts`, `rdr` and `baf` encoded.
        parameters: the lattice's, with the dispersions.
        confounders: `(n_tumour, 1)` `theta_c` and `log Z_c`.
        prior: `(K,)` `eta_s` per pair off the ploidy-neutral pair.
        params: the channels scored, `m` read depth and `p` B-allele fraction.
        clones: the tumour clones' ids, in row order.

    Returns:
        The loglike, by clone.
    """
    encoded = get_log_emission(
        counts, parameters, confounders, layout="lattice", encoding="coded", phased=False, data_terms=False
    )
    rdr, baf = get_decoded_emission(encoded)
    # NB the encoded lattice scores both channels: the decode keeps its own.
    rdr, baf = (rdr if "m" in params else 0.0), (baf if "p" in params else 0.0)
    lattice = get_state_major(rdr + baf)
    return {c: lattice[:, :, i] - prior[:, None] for i, c in enumerate(clones)}


def hard_em_lattice(
    counts: PseudobulkCounts,
    parameters: CnaHmmParameters,
    ab_lattice: np.ndarray,
    clone_confounders: CnaEmissionConfounders,
    chain: CloneChain,
    config: Any,
    *,
    ploidy: int,
) -> AbDecodeResult:
    """`eqn:ab_hmm` and `eqn:ab_coupling` maximized by hard EM from each of `normal_fraction_starts`.

    Each round decodes every tumour clone's copy path (`solve_clone_chains`), recomputes its `log Z_c`
    on it under `recompute_logmu_shift`, fits each `theta_c` (`fit_normal_fraction`) and refits the dispersions; the
    best round over the starts is returned, under `recompute_logmu_shift` decoded again at its own `log Z_c` to a fixed
    point. A clone whose returned `theta_c` is its start is `DecodeStatus.NORMAL_FRACTION_UNMOVED`.

    Args:
        counts: `(n_clones, n_obs)` the clones' pseudobulks.
        parameters: the fit's `alphas` and `taus`.
        ab_lattice: `(K, 2)` the pairs.
        clone_confounders: each clone's first `theta_c`, 1 the normal clone held at (1, 1), 0 a tumour clone decoded
            from each of `normal_fraction_starts`, all 0 to take the normal clone by B fraction; and the fit's
            `log Z_c`, or None.
        chain: the `CloneChain`.
        config: the run's `YAMLConfig`, its `ab_decode`, `hmm` and `hmrf` sections read.
        ploidy: P.

    Returns:
        The `AbDecodeResult`, `ploidy` empty.

    Raises:
        ValueError: on `min_alpha` < 0 or `max_tau` < 1.
    """
    ab_decode = config.ab_decode
    params = "mp" if ab_decode.rdr else "p"
    if not ab_decode.min_alpha >= 0.0:
        raise ValueError(f"min_alpha is {ab_decode.min_alpha}: a floor on the NB dispersion is >= 0, 0 for none.")
    if not ab_decode.max_tau >= 1.0:
        raise ValueError(f"max_tau is {ab_decode.max_tau}: the BB concentration is bounded below by 1.")
    # NB a bin whose read depth exceeds max_rdr times its baseline carries no read-depth term, as the fit drops it.
    base = get_max_rdr_zeroed_baseline(counts.umis, counts.normal_baseline_sized, ab_decode.max_rdr)
    if "m" not in params:
        base = np.zeros_like(base)
    counts = PseudobulkCounts(
        np.asarray(counts.umis, dtype=np.float64),
        base,
        np.asarray(counts.alt_snp_umis, dtype=np.float64),
        np.asarray(counts.total_snp_umis, dtype=np.float64),
        None,
        {},
    )
    n_clones, n_obs = base.shape

    given = np.asarray(clone_confounders.normal_fraction, dtype=np.float64).reshape(-1)
    if not np.any(given):
        # NB one column is a start's clones stacked along the genome: no normal clone.
        normal_clone = -1 if n_clones == 1 else get_normal_clone(get_observed_baf_profiles(counts), ab_decode.baf.eps)
        given = np.zeros(n_clones)
        if normal_clone >= 0:
            given[normal_clone] = 1.0
            logger.info(f"hmm_ab: normal clone {normal_clone} by B fraction (theta = 1, decoded (1, 1)).")
    given = np.array(given, dtype=np.float64)
    tumour = [c for c in range(n_clones) if given[c] != 1.0]
    unset = [c for c in tumour if given[c] == 0.0]
    if not ab_decode.fit_normal:
        given[tumour] = 0.0
        start_runs = [given]
    elif unset:
        # NB a tumour clone given no theta_c is decoded from each of normal_fraction_starts, the best joint kept.
        start_runs = [
            np.where(np.isin(np.arange(n_clones), unset), start, given) for start in normal_fraction_starts(config)
        ]
    else:
        start_runs = [given]
    fit_log_z = np.asarray(clone_confounders.log_mu_shift, dtype=np.float64).reshape(-1)
    fit_log_z = fit_log_z if np.any(fit_log_z) else None

    def get_log_z_start(starts: np.ndarray) -> np.ndarray:
        log_z = np.zeros(n_clones) if fit_log_z is None else fit_log_z.copy()
        for c in tumour:
            log_z[c] = get_log_mu_shift_start(None if fit_log_z is None else fit_log_z[c], starts[c], ploidy)
        return log_z

    if not tumour:
        logger.warning(f"hmm_ab: no tumour clone of {n_clones}: (1, 1) at theta = 1.")
        return AbDecodeResult(
            np.ones((n_obs, n_clones, 2), dtype=np.int64), given, get_log_z_start(given), ab_decode.t, 0.0, {}, {}, {}
        )

    rows = np.asarray(tumour)
    coded_counts = get_coded_rows(counts, tumour, fold=ab_decode.fold)
    baf_only = ab_decode.normal_fraction_baf_only or "m" not in params
    tracks = {c: ab_decode.recompute_logmu_shift and base[c].sum() > 0 for c in tumour}
    prior = ab_decode.specificity_prior * ~get_ab_is_ploidy_neutral(ab_lattice, ploidy)
    totals = ab_lattice.sum(axis=1).astype(np.int64)
    ploidy_band = (totals, ploidy) if ab_decode.genome_wide_ploidy_constraint else None
    # NB the fit's alpha bounds the refit above; a floor above it is the decoder's alpha, held.
    alpha_max = max(float(np.asarray(parameters.alphas, dtype=np.float64).reshape(-1)[0]), ab_decode.min_alpha)
    tau_max = min(float(np.asarray(parameters.taus, dtype=np.float64).reshape(-1)[0]), ab_decode.max_tau)
    bounds = [(np.log(max(ab_decode.min_alpha, ALPHA_FLOOR)), np.log(alpha_max)), (0.0, np.log(ab_decode.max_tau))]

    def decode(f: np.ndarray, log_z: np.ndarray, alpha: float, tau: float) -> ClonesDecode:
        lattice = get_lattice_parameters(ab_lattice, alpha, tau)
        confounders = CnaEmissionConfounders(f[rows, None], log_z[rows, None])
        loglikes = get_clone_emission_loglikes(coded_counts, lattice, confounders, prior, params=params, clones=tumour)
        return solve_clone_chains(
            chain,
            loglikes,
            ploidy_band,
            solver=ab_decode.solver,
            mincut=ab_decode.opengm_mincut_solver,
            icm_sweeps=ab_decode.icm_sweeps,
            ploidy_price_steps=ab_decode.ploidy_price_steps,
        )

    def get_path_log_z(c: int, f: np.ndarray, copy_paths: dict[int, np.ndarray]) -> float:
        lattice = get_lattice_parameters(ab_lattice, None, None)
        weights = get_hard_state_posteriors(copy_paths[c], np.size(lattice.log_mu))
        admixed = lattice._replace(log_mu=np.log(get_admixed_moments(lattice, f[c])[0]))
        return float(get_log_mu_shift(admixed, weights, counts)[c, 0])

    nonbalanced = get_ab_is_nonbalanced(ab_lattice, ab_decode.nonbalance_min_baf_dist)
    best: tuple[Any, ...] | None = None
    for starts in start_runs:
        f, log_z, alpha, tau = starts.copy(), get_log_z_start(starts), alpha_max, tau_max
        copy_paths: dict[int, np.ndarray] = {}
        for it in range(ab_decode.max_iter):
            new, own, status, joint = decode(f, log_z, alpha, tau)
            changed = sum(int((new[c] != copy_paths[c]).sum()) for c in tumour) if copy_paths else -1
            copy_paths = new
            if best is None or joint > best[0]:
                best = (joint, dict(copy_paths), f.copy(), alpha, tau, log_z.copy(), dict(own), dict(status), starts)
            before = (f.copy(), alpha, tau, log_z.copy())
            # NB the round's copy path carries its own log Z_c, at the theta_c it was decoded at, before the refits.
            for c in tumour:
                if tracks[c]:
                    log_z[c] = get_path_log_z(c, f, copy_paths)
            lattice = get_lattice_parameters(ab_lattice, alpha, tau)
            for c in tumour:
                if ab_decode.fit_normal:
                    f[c] = fit_normal_fraction(
                        counts, c, lattice, copy_paths[c], nonbalanced, log_z[c], fold=ab_decode.fold, baf_only=baf_only
                    )
                # NB as on main, Z_c follows the fitted theta_c once the fit, at the held Z_c, is done.
                if tracks[c]:
                    log_z[c] = get_path_log_z(c, f, copy_paths)
            if ab_decode.fit_dispersions:
                objective = functools.partial(
                    hmm_ab_decoder_dispersion_objective,
                    counts=counts,
                    clones=tumour,
                    ab_lattice=ab_lattice,
                    copy_paths=copy_paths,
                    clone_confounders=CnaEmissionConfounders(f[:, None], log_z[:, None]),
                    fold=ab_decode.fold,
                )
                sol = minimize(objective, np.log([alpha, tau]), jac=True, method="L-BFGS-B", bounds=bounds)
                alpha, tau = (float(v) for v in np.exp(sol.x))
            if (
                it
                and changed == 0
                and np.allclose(before[0], f, rtol=0.0, atol=NORMAL_FRACTION_XATOL)
                and np.allclose(before[3], log_z, rtol=0.0, atol=1e-6)
                and np.isclose(before[1], alpha, rtol=1e-4)
                and np.isclose(before[2], tau, rtol=1e-3)
            ):
                break
        else:
            logger.warning(f"hmm_ab: no fixed point in {ab_decode.max_iter} rounds from theta = {starts.tolist()}.")
    assert best is not None
    value, copy_paths, f, alpha, tau, log_z, own, status, starts = best
    if ab_decode.recompute_logmu_shift:
        # NB a recomputed log Z_c is the previous round's copy path's: decode again at the path's own until they agree.
        for _ in range(ab_decode.max_iter):
            settled = log_z.copy()
            for c in tumour:
                if base[c].sum() > 0:
                    settled[c] = get_path_log_z(c, f, copy_paths)
            if np.allclose(settled, log_z, rtol=0.0, atol=1e-9):
                break
            log_z = settled
            copy_paths, own, status, value = decode(f, log_z, alpha, tau)
        else:
            logger.warning(f"hmm_ab: log Z_c and its copy path's log Z disagree after {ab_decode.max_iter} decodes.")
    if ab_decode.fit_normal:
        for c in tumour:
            if status[c] == DecodeStatus.OK and abs(f[c] - starts[c]) <= NORMAL_FRACTION_XATOL:
                status[c] = DecodeStatus.NORMAL_FRACTION_UNMOVED
                logger.warning(f"hmm_ab: clone {c}'s normal fraction {f[c]:.4f} is its start {starts[c]:.4f}.")
    copies = np.ones((n_obs, n_clones, 2), dtype=np.int64)
    for c in tumour:
        copies[:, c] = ab_lattice[copy_paths[c]]
    logger.info(
        f"hmm_ab: normal fractions {f.tolist()}, log Z_c {np.round(log_z, 4).tolist()}, alpha {alpha:.4e}, "
        f"tau {tau:.4e}, joint loglike {value:.6e}."
    )
    return AbDecodeResult(copies, f, log_z, ab_decode.t, float(value), own, {}, status)


def decode_at_ploidy(
    counts: PseudobulkCounts,
    parameters: CnaHmmParameters,
    clone_confounders: CnaEmissionConfounders,
    lengths: Any,
    config: Any,
    *,
    ploidy: int,
) -> AbDecodeResult:
    """Integer `(A, B)` per bin and clone at `ploidy`, `sec:ab_decoder`, every setting from `config`'s `ab_decode`.

    Args:
        counts: `(n_clones, n_obs)` the clones' pseudobulks.
        parameters: the fit's `alphas` and `taus`.
        clone_confounders: as `hard_em_lattice`.
        lengths: the segments' bin counts.
        config: the run's `YAMLConfig`.
        ploidy: P.

    Returns:
        The `AbDecodeResult`.

    Raises:
        NotImplementedError: on an odd ploidy.
        ValueError: on a ploidy that is not a whole number >= 2, an unknown solver or a negative `clone_coupling`.
    """
    if ploidy != int(ploidy) or ploidy < 1:
        raise ValueError(f"ploidy is {ploidy}: the assumed ploidy is a whole number of copies.")
    if ploidy % 2:
        raise NotImplementedError("odd ploidy, which has no neutral pair")
    ab_decode = config.ab_decode
    if ab_decode.solver not in get_hmm_ab_solvers():
        raise ValueError(f"solver is one of {get_hmm_ab_solvers()}, not {ab_decode.solver!r}.")
    if not ab_decode.clone_coupling >= 0.0:
        raise ValueError(f"clone_coupling is {ab_decode.clone_coupling}: nats per disagreeing pair, >= 0.")
    ab_lattice = get_ab_lattice(
        params="mp" if ab_decode.rdr else "p",
        fold=ab_decode.fold,
        max_allele_copy=ab_decode.max_allele_copy,
        max_total_copy=ab_decode.max_total_copy,
    )
    chain = get_clone_chain(ab_lattice, lengths, config, ploidy=int(ploidy))
    found = hard_em_lattice(counts, parameters, ab_lattice, clone_confounders, chain, config, ploidy=int(ploidy))
    named = {P: name for name, P in get_ploidy_by_name().items()}
    return found._replace(ploidy={c: named[int(ploidy)] for c in found.clone_scores})


def decode_integer_copies(
    counts: PseudobulkCounts,
    parameters: CnaHmmParameters,
    clone_confounders: CnaEmissionConfounders,
    lengths: Any,
    config: Any,
) -> AbDecodeResult:
    """The `(A, B)` decode at each of `ab_decode.assumed_ploidy`, each tumour clone's best, the lower on a tie.

    One ploidy decodes in process; several in worker processes, one per ploidy up to 4.

    Args:
        counts: `(n_clones, n_obs)` the clones' pseudobulks.
        parameters: the fit's `alphas` and `taus`.
        clone_confounders: as `hard_em_lattice`.
        lengths: the segments' bin counts.
        config: the run's `YAMLConfig`.

    Returns:
        The `AbDecodeResult`, `ploidy` each tumour clone's choice.

    Raises:
        NotImplementedError: on an odd ploidy.
        ValueError: on an empty or unknown `assumed_ploidy`.
    """
    names = assumed_ploidy(config)
    unknown = sorted(set(names) - set(get_ploidy_by_name()))
    if unknown or not names:
        raise ValueError(f"assumed_ploidy {list(names)}: one or more of {sorted(get_ploidy_by_name())}.")
    ploidies = sorted({get_ploidy_by_name()[name] for name in names})
    if any(P % 2 for P in ploidies):
        raise NotImplementedError("odd ploidy, which has no neutral pair")
    logger.info(
        f"Decoding integer (A, B) for {np.shape(counts.umis)[0]} clones over {np.shape(counts.umis)[1]} bins "
        f"at {list(names)}."
    )
    args = (counts, parameters, clone_confounders, lengths, config)
    if len(ploidies) == 1:
        return decode_at_ploidy(*args, ploidy=ploidies[0])
    with ProcessPoolExecutor(min(len(ploidies), 4), mp_context=get_context("fork")) as pool:
        futures = {P: pool.submit(decode_at_ploidy, *args, ploidy=P) for P in ploidies}
        runs = {P: future.result() for P, future in futures.items()}
    named = {P: name for name, P in get_ploidy_by_name().items()}
    first = runs[ploidies[0]]
    choice = {c: max(ploidies, key=lambda P: (runs[P].clone_scores[c], -P)) for c in sorted(first.clone_scores)}
    fields = ("copies", "normal_fractions", "log_mu_shift")
    chosen = {name: np.array(getattr(first, name), copy=True) for name in fields}
    for c, P in choice.items():
        for name in fields:
            if name == "copies":
                chosen[name][:, c] = runs[P].copies[:, c]
            else:
                chosen[name][c] = getattr(runs[P], name)[c]
        logger.info(
            f"hmm_ab: clone {c} loglike by ploidy "
            + ", ".join(f"{named[Q]}: {runs[Q].clone_scores[c]:.6e}" for Q in ploidies)
            + f"; {named[P]}."
        )
    clone_scores = {c: runs[P].clone_scores[c] for c, P in choice.items()}
    return AbDecodeResult(
        **chosen,
        stay_prob=first.stay_prob,
        score=float(sum(clone_scores.values())),
        clone_scores=clone_scores,
        ploidy={c: named[P] for c, P in choice.items()},
        status={c: runs[P].status[c] for c, P in choice.items()},
    )
