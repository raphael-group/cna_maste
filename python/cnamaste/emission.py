"""The emission of `eqn:nb_emission` and `eqn:bb_emission`, with the normal admixture of `eqn:ab_admixture`.

The reference is `cna-maste-paper`, section `sec:ab_decoder`. Paper and code, once:

    paper            here
    theta_m          normal_fraction
    mu, p            a state's depth exp(log_mu), or (A + B) / 2, and B fraction p_binom, or B / (A + B)
    mu~, p~          get_admixed_moments
    Z_m              exp(log_mu_shift), get_log_mu_shift
    lambda_g T_m     base_nb_mean
    phi, tau         alphas, taus

`get_log_emission` dispatches to the kernels in `hmm_emission_nb`, `hmm_emission_bb` and `hmm_nophasing`.
"""

from __future__ import annotations

import itertools
from math import exp
from typing import Any, Literal, NamedTuple, overload

import numpy as np
from numba import njit

from cnamaste import hmm_emission_bb, hmm_emission_nb
from cnamaste.hmm_emission_bb import bb_logpmf_vector, bb_switch_logpmf_vector
from cnamaste.hmm_emission_nb import nb_logpmf_vector
from cnamaste.hmm_emission_tabled import get_clone_decoded_fields, get_labelling_unary_encoding
from cnamaste.hmm_params import CnaHmmParameters
from cnamaste.hmm_utils import log_factorial
from cnamaste.spatio_genomic_counts import PseudobulkCounts, get_encoding_channels

# NB `get_log_emission`'s layouts:
# NB `lattice` every state at every bin, `(S, n_bins, n_cols)`.
# NB `path` the state `path` names per bin, the counts' shape.
# NB `field` each spot's sum over each clone's path, `(n_spots, n_clones)`.
LAYOUTS = ("lattice", "path", "field")

# NB `get_log_emission`'s encodings: None the counts themselves, `coded` their `rdr` and `baf` encodings.
ENCODINGS = (None, "coded")


class CnaEmissionConfounders(NamedTuple):
    """Each clone's normal fraction `theta_c` and library normalizer `log Z_c`, `(n_clones, 1)` each, in clone order; 0
    is none: unmixed, `Z_c = 1`."""

    normal_fraction: Any
    log_mu_shift: Any


def get_unconfounded(n_clones: int) -> CnaEmissionConfounders:
    """Confounders of 0 for `n_clones` clones: unmixed, `Z_c = 1`.

    Args:
        n_clones: the clones.

    Returns:
        The `CnaEmissionConfounders`.
    """
    return CnaEmissionConfounders(np.zeros((n_clones, 1)), np.zeros((n_clones, 1)))


def get_clone_confounders(confounders: CnaEmissionConfounders, n_clones: int) -> tuple[np.ndarray, np.ndarray]:
    """Each clone's `theta_c` and `log Z_c`, `(n_clones,)` each; one clone's for every clone.

    Args:
        confounders: `(n_clones, 1)` or `(1, 1)` each.
        n_clones: the clones.

    Returns:
        The normal fractions and the log normalizers.

    Raises:
        ValueError: on a confounder neither one per clone nor one.
    """
    values = [np.asarray(v, dtype=np.float64).reshape(-1) for v in confounders]
    for value in values:
        if value.size not in (1, n_clones):
            raise ValueError(f"a confounder is one per clone, {n_clones}, or one, not {value.size}.")
    return tuple(np.broadcast_to(value, n_clones).copy() for value in values)


class LogEmission(NamedTuple):
    """The read-depth and B-allele log emissions; 0.0 for a channel not scored."""

    rdr: Any
    baf: Any


def get_encoded(counts: PseudobulkCounts, *, common_zero_depth: bool) -> PseudobulkCounts:
    """The counts with their `rdr` and `baf` encodings (`ENCODING_CHANNELS`), per column.

    Args:
        counts: the `PseudobulkCounts`.
        common_zero_depth: as `spatio_genomic_counts.get_encoded_pairs`.

    Returns:
        The encoded counts.
    """
    for key in ("rdr", "baf"):
        counts = counts.update_encoding(key, get_encoding_channels()[key], common_zero_depth=common_zero_depth)
    return counts


def get_bin_major(channel: Any) -> Any:
    """A clone-major `(n_labels, n_bins)` channel as `(n_bins, n_labels)`, and back; None kept.

    Args:
        channel: the counts, or None.

    Returns:
        The transpose, or None.
    """
    return None if channel is None else np.transpose(channel)


def get_admixed_moments(parameters: CnaHmmParameters, normal_fraction: Any) -> tuple[np.ndarray, np.ndarray]:
    """`eqn:ab_admixture`: `mu~ = theta + (1 - theta) mu` and `p~ = (theta / 2 + (1 - theta) mu p) / mu~`.

    Args:
        parameters: the states' `log_mu` and `p_binom`.
        normal_fraction: `theta` in [0, 1], a scalar, or `(n_clones,)` for a trailing clone axis; None or a scalar 0
            returns `(exp(log_mu), p_binom)` unmixed.

    Returns:
        `mu~` and `p~`, each of the states' shape, plus the clone axis where `normal_fraction` has one.
    """
    mu = np.exp(np.asarray(parameters.log_mu, dtype=np.float64))
    p = np.asarray(parameters.p_binom, dtype=np.float64)
    if normal_fraction is None or (np.ndim(normal_fraction) == 0 and normal_fraction == 0.0):
        return mu, p
    theta = np.asarray(normal_fraction, dtype=np.float64)
    if theta.ndim:
        mu, p = mu[..., None], p[..., None]
    depth = theta + (1 - theta) * mu
    return depth, (theta / 2.0 + (1 - theta) * (mu * p)) / depth


def get_hard_state_posteriors(path: np.ndarray, n_states: int) -> np.ndarray:
    """A path as `get_log_mu_shift`'s state weights: one-hot, log 1 at the path's state and log 0 elsewhere.

    Args:
        path: `(n_obs,)` the state per bin.
        n_states: the states.

    Returns:
        The `(n_obs, n_states)` log weights, 0 or `-inf`.
    """
    path = np.asarray(path, dtype=np.int64).reshape(-1)
    log_weights = np.full((path.size, n_states), -np.inf)
    log_weights[np.arange(path.size), path] = 0.0
    return log_weights


@njit(nogil=True, cache=True, error_model="numpy")
def get_log_state_sums_by_clone(
    log_state_weights: np.ndarray, log_lambda: np.ndarray, clone_lengths: np.ndarray
) -> np.ndarray:
    """`log S_cs = logsumexp_g(log w_gs + log lambda_g)` over each clone's bins, guarded against `-inf`.

    `log Z_c = logsumexp_s(log mu_s + log S_cs)` (`get_log_normalizers`): `S_cs` holds what `log_mu` does not enter.

    Args:
        log_state_weights: `(n_obs, n_states)` `log w_gs`, each bin's weight per state, its rows summing to one.
        log_lambda: `(n_obs,)` `log lambda_g`, each clone's baseline over its sum.
        clone_lengths: `(n_clones,)` bins per clone, clone c's the c-th block.

    Returns:
        The `(n_clones, n_states)` sums.
    """
    n_states = log_state_weights.shape[1]
    log_state_sums = np.empty((len(clone_lengths), n_states), dtype=np.float64)
    start = 0
    for c in range(len(clone_lengths)):
        stop = start + clone_lengths[c]
        for s in range(n_states):
            top = -np.inf
            for g in range(start, stop):
                top = max(top, log_state_weights[g, s] + log_lambda[g])
            if np.isinf(top):
                log_state_sums[c, s] = top
            else:
                total = 0.0
                for g in range(start, stop):
                    total += np.exp(log_state_weights[g, s] + log_lambda[g] - top)
                log_state_sums[c, s] = top + np.log(total)
        start = stop
    return log_state_sums


def get_log_state_sums(log_state_weights: Any, counts: PseudobulkCounts) -> np.ndarray:
    """`get_log_state_sums_by_clone` of `counts`' clones: `(n_clones, n_states)` `log S_cs`.

    Args:
        log_state_weights: `(n_obs, n_states)` `log w_gs`; a path's by `get_hard_state_posteriors`. Clones stacked
            along the bins.
        counts: the `PseudobulkCounts`: `get_log_lambda` and `get_clone_lengths`; a clone per row where it has
            several, each at the same weights.

    Returns:
        The log state sums.
    """
    weights, log_lambda = np.asarray(log_state_weights, dtype=np.float64), counts.get_log_lambda()
    if log_lambda.ndim == 2 and log_lambda.shape[0] > 1:
        lengths = np.array([log_lambda.shape[1]])
        rows = [get_log_state_sums_by_clone(weights, np.ascontiguousarray(row), lengths) for row in log_lambda]
        return np.concatenate(rows)
    return get_log_state_sums_by_clone(weights, log_lambda.reshape(-1), counts.get_clone_lengths())


def get_log_normalizers(log_mu: np.ndarray, log_state_sums: np.ndarray) -> np.ndarray:
    """`log Z_c = logsumexp_s(log mu_s + log S_cs)`, `(n_clones,)`; `-inf` for a clone of no weight.

    Args:
        log_mu: `(n_states,)` each state's log depth.
        log_state_sums: `(n_clones, n_states)` `get_log_state_sums`.

    Returns:
        The log normalizers.
    """
    terms = log_mu[None, :] + log_state_sums
    top = np.max(terms, axis=1, keepdims=True)
    top = np.where(np.isinf(top), 0.0, top)
    with np.errstate(divide="ignore"):
        return np.asarray(top[:, 0] + np.log(np.sum(np.exp(terms - top), axis=1)))


def get_log_mu_shift(parameters: CnaHmmParameters, log_state_weights: Any, counts: PseudobulkCounts) -> np.ndarray:
    """Each clone's library normalizer `log Z_c = log sum_g lambda_g sum_s w_gs mu_s`, `eqn:nb_emission`, natural log.

    Args:
        parameters: the states' `log_mu`; `log mu~` of `eqn:ab_admixture` where admixed.
        log_state_weights: `(n_obs, n_states)` `log w_gs`, each bin's weight per state; a path's by
            `get_hard_state_posteriors`. Clones stacked along the bins.
        counts: the `PseudobulkCounts`: `get_log_lambda` and `get_clone_lengths`; a clone per row where it has
            several, each at the same weights.

    Returns:
        The `(n_clones, 1)` log normalizers.
    """
    log_mu = np.asarray(parameters.log_mu, dtype=np.float64).reshape(-1)
    return get_log_normalizers(log_mu, get_log_state_sums(log_state_weights, counts)).reshape(-1, 1)


def get_nb_mean(base_nb_mean: Any, mu: Any, log_mu_shift: Any) -> Any:
    """`eqn:nb_emission`'s mean, `base_nb_mean mu / Z_c`, as `base_nb_mean exp(-log Z_c) mu`; 0 is `Z_c = 1`.

    Args:
        base_nb_mean: the baseline per bin.
        mu: the depth, broadcast against it.
        log_mu_shift: a scalar `log Z_c`.

    Returns:
        The NB mean, the broadcast shape.
    """
    if log_mu_shift == 0.0:
        return base_nb_mean * mu
    return base_nb_mean * np.exp(-np.asarray(log_mu_shift)) * mu


def get_minor_allele_count(b: Any, n: Any) -> Any:
    """The minor allele's count, `min(b, n - b)`.

    Args:
        b: B-allele counts.
        n: allele totals.

    Returns:
        The minor counts.
    """
    return np.minimum(b, n - b)


def get_statewise_as_column(value: Any, n_states: int) -> np.ndarray:
    """A per-state parameter, or one shared, as the dense kernels read it, `(n_states, 1)`.

    Args:
        value: a scalar, `(n_states,)` or `(n_states, 1)`.
        n_states: the states.

    Returns:
        The `(n_states, 1)` float64 column.
    """
    return np.ascontiguousarray(np.broadcast_to(np.asarray(value, dtype=np.float64).reshape(-1, 1), (n_states, 1)))


def get_statewise(value: Any, n_states: int) -> np.ndarray:
    """A per-state parameter as the encoded kernel reads: `get_statewise_as_column`, or `(n_states, n_cols)` as given.

    Args:
        value: a scalar, `(n_states,)`, `(n_states, 1)` or `(n_states, n_cols)`.
        n_states: the states.

    Returns:
        The float64 `(n_states, 1)` or `(n_states, n_cols)`.
    """
    value = np.asarray(value, dtype=np.float64)
    return value if value.ndim == 2 and value.shape[1] > 1 else get_statewise_as_column(value, n_states)


def get_states_shared_value(value: Any) -> float:
    """The one dispersion every state shares, as the `path` layout scores it.

    Args:
        value: a scalar or one value per state.

    Returns:
        The value.

    Raises:
        ValueError: where the states' values differ.
    """
    values = np.asarray(value, dtype=np.float64).reshape(-1)
    if (values != values[0]).any():
        raise ValueError(f"the path layout scores one dispersion shared by every state, not {values.tolist()}.")
    return float(values[0])


def reshape_params_by_spot(param: Any, n_spots: int) -> np.ndarray:
    """A per-state parameter as `(n_states, n_spots)`: one column repeated for every spot, else kept.

    Args:
        param: `(n_states,)`, `(n_states, 1)` or `(n_states, n_spots)`.
        n_spots: the spots.

    Returns:
        The `(n_states, n_spots)` array.
    """
    param = np.asarray(param).reshape(len(param), -1)
    return np.repeat(param, n_spots, axis=1) if param.shape[1] == 1 else param


class EncodedEmission(NamedTuple):
    """Per-clone class tables, `(n_clones, n_classes, S)` or `(n_clones, n_classes, 2, K)` where phased, and each
    bin's class, `(n_clones, n_bins)`; a bin's log emission is `rdr[c, rdr_index[c, g]] + baf[c, baf_index[c, g]]`."""

    rdr: np.ndarray
    rdr_index: np.ndarray
    baf: np.ndarray
    baf_index: np.ndarray


def get_decoded_emission(encoded: EncodedEmission) -> LogEmission:
    """The encoded emission gathered to every bin, `(n_clones, n_bins, S)` or `(n_clones, n_bins, 2, K)` each.

    Args:
        encoded: the `EncodedEmission`.

    Returns:
        The `LogEmission`.
    """
    clones = np.arange(encoded.rdr_index.shape[0])[:, None]
    return LogEmission(encoded.rdr[clones, encoded.rdr_index], encoded.baf[clones, encoded.baf_index])


def get_class_index(counts: PseudobulkCounts, key: str) -> np.ndarray:
    """Each bin's class within its clone's unique pairs of the `key` encoding.

    Args:
        counts: the `PseudobulkCounts`, encoded.
        key: `rdr` or `baf`.

    Returns:
        The `(n_clones, n_bins)` classes.
    """
    offsets = np.cumsum([0] + [len(u) for u in counts.get_unique(key)[0][:-1]], dtype=np.int64)
    return counts.get_equivalence_class(key) - offsets[:, None]


def encode_emission(
    counts: PseudobulkCounts,
    parameters: CnaHmmParameters,
    thetas: np.ndarray,
    shifts: np.ndarray,
    *,
    phased: bool,
    data_terms: bool,
) -> EncodedEmission:
    """The `lattice` on the `rdr` and `baf` encodings, each clone's unique pairs scored once.

    A read-depth class is (clone, y, baseline), each block of bins at its own `log Z_c` its own classes; a B-allele
    class is (clone, b, n), its phase-1 row `b -> n - b`. `data_terms=False` leaves out `-lgamma(k + 1)` and
    `log C(n, k)`, which no parameter enters.

    Args:
        counts: the `PseudobulkCounts`, its `rdr` and `baf` encodings built.
        parameters: the states' `log_mu`, `p_binom`, `alphas` and `taus`.
        thetas: `(n_clones,)` each clone's `theta_c`, one for every clone stacked along bins.
        shifts: `(n_clones,)` each clone's `log Z_c`.
        phased: append the `(state, phase)` pairs.
        data_terms: keep the terms no parameter enters.

    Returns:
        The `EncodedEmission`, both channels scored.

    Raises:
        NotImplementedError: on a normal fraction per clone stacked along bins.
    """
    lengths, by_column = counts.get_clone_lengths(), counts.clone_lengths is None and thetas.size > 1
    if not by_column and len(set(thetas)) > 1:
        raise NotImplementedError("clones stacked along bins take one normal fraction for every clone.")
    n_states = np.shape(parameters.log_mu)[0]
    log_mu, p = (np.asarray(v, dtype=np.float64).reshape(n_states, -1) for v in (parameters.log_mu, parameters.p_binom))
    if by_column:
        # NB a clone per column: its own theta_c, and its log Z_c on log_mu; per-column parameters kept.
        if np.any(thetas):
            mu, admixed = get_admixed_moments(parameters, thetas)
            log_mu, p = np.where(thetas == 0.0, log_mu, np.log(mu)), np.where(thetas == 0.0, p, admixed)
        log_mu, shifts = log_mu - shifts, np.zeros(1)
    elif thetas[0] != 0.0:
        mu, p = get_admixed_moments(parameters, thetas[0])
        log_mu = np.log(mu)
    nb_endogs, nb_exposures = counts.get_unique("rdr")
    bb_endogs, bb_exposures = counts.get_unique("baf")
    n_clones = len(nb_endogs)
    log_mu, p = (
        reshape_params_by_spot(log_mu, n_clones),
        reshape_params_by_spot(np.reshape(p, (n_states, -1)), n_clones),
    )
    alphas = reshape_params_by_spot(get_statewise(parameters.alphas, n_states), n_clones)
    taus = reshape_params_by_spot(get_statewise(parameters.taus, n_states), n_clones)
    # NB x - 0.0 is x: a block at log Z_c = 0 scores exp(log_mu) exactly.
    block_shifts, block = np.unique(np.repeat(shifts, lengths), return_inverse=True)
    n_blocks, phases = len(block_shifts), ((2,) if phased else ())

    rdr = np.zeros((n_clones, n_blocks * max(map(len, nb_endogs)), *phases, n_states))
    baf = np.zeros((n_clones, max(map(len, bb_endogs)), *phases, n_states))
    for c in range(n_clones):
        n_unique = len(nb_endogs[c])
        log_k_factorial = log_factorial(nb_endogs[c]) if data_terms else np.zeros(n_unique)
        table = np.zeros((n_states, n_unique))
        for b, shift in enumerate(block_shifts):
            for i in range(n_states):
                rate = exp(log_mu[i, c] - shift)
                nb_logpmf_vector(nb_endogs[c], nb_exposures[c], rate, alphas[i, c], table[i], log_k_factorial)
            rdr[c, b * n_unique : (b + 1) * n_unique] = table.T[:, None] if phased else table.T
        table = np.zeros((n_states, len(bb_endogs[c])))
        for i in range(n_states):
            bb_logpmf_vector(bb_endogs[c], bb_exposures[c], p[i, c], taus[i, c], table[i], data_terms=data_terms)
        if phased:
            switched = bb_switch_logpmf_vector(table, bb_endogs[c], bb_exposures[c], p[:, c], taus[:, c])
            baf[c, : table.shape[1]] = np.stack((table.T, switched.T), axis=1)
        else:
            baf[c, : table.shape[1]] = table.T
    sizes = np.array([len(u) for u in nb_endogs], dtype=np.int64)[:, None]
    rdr_index = block.reshape(-1) * sizes + get_class_index(counts, "rdr")
    return EncodedEmission(rdr, rdr_index, baf, get_class_index(counts, "baf"))


def get_field_log_emission(
    counts: PseudobulkCounts,
    parameters: CnaHmmParameters,
    shifts: np.ndarray,
    *,
    path: np.ndarray,
    spot_weights: np.ndarray | None,
) -> LogEmission:
    """The `field`: `hmm_emission_tabled.get_clone_decoded_fields`, each spot's sum over each clone's decoded path.

    Args:
        counts: the `PseudobulkCounts` per spot, each spot its own label, `(n_spots, n_bins)`; its `labelling_unary`
            encoding where held, else built for this call.
        parameters: the states' `log_mu`, `p_binom`, `alphas` and `taus`.
        shifts: `(n_clones,)` each clone's `log Z_c`.
        path: `(n_clones, n_bins)` each clone's state, or `(state, phase)` pair, per bin.
        spot_weights: `(n_spots,)` each spot's read-depth weight.

    Returns:
        The `LogEmission`, `(n_spots, n_clones)` each, the read depth weighted by `spot_weights`.

    Raises:
        ValueError: on no `spot_weights`.
    """
    if spot_weights is None:
        raise ValueError("the field layout takes spot_weights.")
    scales = np.array([np.exp(-shift) for shift in shifts])
    log_mu, alphas, p_binom, taus = (
        np.asarray(v, dtype=np.float64).reshape(-1)
        for v in (parameters.log_mu, parameters.alphas, parameters.p_binom, parameters.taus)
    )
    held = counts.encodings.get("labelling_unary")
    encoding = get_labelling_unary_encoding(*counts[:4]) if held is None else held
    rdr, baf = get_clone_decoded_fields(*counts[:4], path, log_mu, alphas, p_binom, taus, scales, encoding=encoding)
    return LogEmission(spot_weights[:, None] * rdr, baf)


def get_dense_log_emission(
    counts: PseudobulkCounts,
    parameters: CnaHmmParameters,
    theta: float,
    shift: float,
    *,
    data_terms: bool,
    path: np.ndarray | None,
) -> LogEmission:
    """One clone's dense `lattice`, or its `path`, at its `theta_c` and `log Z_c`.

    Args:
        counts: the clone's `PseudobulkCounts`, a channel None not scored.
        parameters: the states' `log_mu`, `p_binom`, `alphas` and `taus`.
        theta: `theta_c`.
        shift: `log Z_c`.
        data_terms: keep the terms no parameter enters; the `path` layout only drops them.
        path: the state per bin, or None for the lattice.

    Returns:
        The `LogEmission`, `(S, n_bins, n_cols)` each, or the counts' shape on a path; 0.0 for a channel not scored.

    Raises:
        NotImplementedError: on `data_terms=False` on the lattice.
    """
    mu, p = get_admixed_moments(parameters, theta)
    rdr: Any = 0.0
    baf: Any = 0.0
    if path is not None:
        mu, p = mu[path], p[path]
        if counts.umis is not None:
            column = (slice(None),) + (None,) * (counts.umis.ndim - 1)
            mean = get_nb_mean(counts.normal_baseline_sized, mu[column], shift)
            alpha = get_states_shared_value(parameters.alphas)
            rdr = hmm_emission_nb.nb_logpmf_array(counts.umis, mean, alpha, data_terms=data_terms)
        if counts.alt_snp_umis is not None:
            column = (slice(None),) + (None,) * (counts.alt_snp_umis.ndim - 1)
            tau = get_states_shared_value(parameters.taus)
            baf = hmm_emission_bb.bb_logpmf_array(
                counts.alt_snp_umis, counts.total_snp_umis, p[column], tau, data_terms=data_terms
            )
        return LogEmission(rdr, baf)
    if not data_terms:
        raise NotImplementedError("data_terms=False on the dense lattice: score the encoded counts.")
    n_states = mu.shape[0]
    # NB unmixed, the states' own log_mu, not log(exp(log_mu)).
    log_mu = np.log(mu) if theta != 0.0 else np.asarray(parameters.log_mu, dtype=np.float64)
    if counts.umis is not None:
        base = np.asarray(counts.normal_baseline_sized, dtype=np.float64)
        exposure = base if shift == 0.0 else base * np.exp(-shift)
        rdr = hmm_emission_nb.nb_logpmf_dense(
            np.asarray(counts.umis, dtype=np.float64),
            exposure,
            log_mu.reshape(-1, 1),
            get_statewise_as_column(parameters.alphas, n_states),
        )
    if counts.alt_snp_umis is not None:
        baf = hmm_emission_bb.bb_logpmf_dense(
            np.asarray(counts.alt_snp_umis, dtype=np.float64),
            np.asarray(counts.total_snp_umis, dtype=np.float64),
            p.reshape(-1, 1),
            get_statewise_as_column(parameters.taus, n_states),
        )
    return LogEmission(rdr, baf)


@overload
def get_log_emission(
    counts: PseudobulkCounts,
    parameters: CnaHmmParameters,
    confounders: CnaEmissionConfounders,
    *,
    layout: str,
    encoding: None = None,
    phased: bool,
    data_terms: bool,
    path: np.ndarray | None = None,
    spot_weights: np.ndarray | None = None,
) -> LogEmission: ...


@overload
def get_log_emission(
    counts: PseudobulkCounts,
    parameters: CnaHmmParameters,
    confounders: CnaEmissionConfounders,
    *,
    layout: str,
    encoding: Literal["coded"],
    phased: bool,
    data_terms: bool,
    path: np.ndarray | None = None,
    spot_weights: np.ndarray | None = None,
) -> EncodedEmission: ...


def get_log_emission(
    counts: PseudobulkCounts,
    parameters: CnaHmmParameters,
    confounders: CnaEmissionConfounders,
    *,
    layout: str,
    encoding: str | None = None,
    phased: bool,
    data_terms: bool,
    path: np.ndarray | None = None,
    spot_weights: np.ndarray | None = None,
) -> LogEmission | EncodedEmission:
    """The read-depth and B-allele log emissions, `eqn:nb_emission` and `eqn:bb_emission`, at `get_admixed_moments`.

    A clone is a block of bins where `counts.clone_lengths` is set, else a column; a confounder of one clone holds
    for every clone. The B count is scored as `counts` holds it, the minor count where the caller folds it.

    Args:
        counts: the `PseudobulkCounts`.
        parameters: the states' `log_mu`, `p_binom`, `alphas` and `taus`.
        confounders: each clone's `theta_c` and `log Z_c`, `(n_clones, 1)` or `(1, 1)`.
        layout: one of `LAYOUTS`.
        encoding: None scores every bin, `coded` each class of the `rdr` and `baf` encodings (`ENCODING_CHANNELS`).
        phased: append the `(state, phase)` pairs, `p -> 1 - p`, `2 n_states` in all.
        data_terms: keep the terms no parameter enters; False is the M step's objective.
        path: the state per bin on the `path` layout, `(n_clones, n_bins)` on the `field` layout.
        spot_weights: each spot's weight, on the `field` layout.

    Returns:
        The `LogEmission`: clone-major on the lattice, `(n_clones, n_bins, S)` or `(n_clones, n_bins, 2, K)` where
        `phased`; `(n_bins, n_cols)` on a path; `(n_spots, n_clones)` on the field. The `EncodedEmission` where encoded.

    Raises:
        ValueError: on a layout not in `LAYOUTS`, an encoding not in `ENCODINGS`, or a `path` the layout does not
            take.
        NotImplementedError: on a combination no kernel scores: `phased` or an encoding off the lattice, or a normal
            fraction on the field.
    """
    if layout not in LAYOUTS:
        raise ValueError(f"layout is one of {LAYOUTS}, not {layout!r}.")
    if encoding not in ENCODINGS:
        raise ValueError(f"encoding is one of {ENCODINGS}, not {encoding!r}.")
    if (path is None) != (layout == "lattice"):
        raise ValueError(f"the {layout} layout takes a path: {path is not None}.")
    if layout == "field":
        thetas, shifts = get_clone_confounders(confounders, path.shape[0])
        if np.any(thetas):
            raise NotImplementedError(f"the field scores unmixed counts: normal_fraction is {thetas.tolist()}, not 0.")
        return get_field_log_emission(counts, parameters, shifts, path=path, spot_weights=spot_weights)
    if layout == "path":
        if encoding is not None or phased:
            raise NotImplementedError("the path layout scores dense, unphased states.")
        return get_state_major_log_emission(counts, parameters, confounders, path=path, data_terms=data_terms)
    if encoding is None and not phased:
        found = get_state_major_log_emission(counts, parameters, confounders, path=None, data_terms=data_terms)
        return LogEmission(*(get_clone_major(channel, phased=False) for channel in found))
    by_bins = counts.clone_lengths is not None
    n_labels = counts.get_equivalence_class("rdr").shape[0] if encoding else np.shape(counts.normal_baseline_sized)[0]
    n_clones = counts.get_clone_lengths().size if by_bins else n_labels
    thetas, shifts = get_clone_confounders(confounders, n_clones)
    if encoding is not None:
        return encode_emission(counts, parameters, thetas, shifts, phased=phased, data_terms=data_terms)
    if by_bins and np.any(shifts):
        # NB bitwise with T- #100 PR 1: the baseline over Z_c, not log_mu less log Z_c (6.9e-13 relative).
        exposure = counts.normal_baseline_sized * np.exp(-np.repeat(shifts, counts.get_clone_lengths()))
        counts, shifts = counts._replace(normal_baseline_sized=exposure), np.zeros_like(shifts)
    encoded = get_encoded(counts, common_zero_depth=False)
    return get_decoded_emission(
        encode_emission(encoded, parameters, thetas, shifts, phased=phased, data_terms=data_terms)
    )


def get_clone_major(log_emission: Any, *, phased: bool) -> Any:
    """`(S, n_bins, n_clones)` as the clone-major, C-order `(n_clones, n_bins, S)`, `(n_clones, n_bins, 2, K)` where
    `phased`; a channel not scored, 0.0, as is.

    Args:
        log_emission: `(S, n_bins, n_clones)`, `S = 2 K` where `phased`, the second `K` the switched phase.
        phased: split `S` into `(phase, state)`.

    Returns:
        The clone-major array.
    """
    if np.ndim(log_emission) == 0:
        return log_emission
    found = np.ascontiguousarray(np.transpose(log_emission, (2, 1, 0)))
    return found.reshape(*found.shape[:2], 2, -1) if phased else found


def get_state_major(log_emission: Any) -> Any:
    """`get_clone_major`'s inverse: C-order `(S, n_bins, n_clones)`, `(phase, state)` flattened, as the lattices read
    it.

    Args:
        log_emission: `(n_clones, n_bins, S)` or `(n_clones, n_bins, 2, K)`; 0.0 kept.

    Returns:
        The state-major array.
    """
    if np.ndim(log_emission) == 0:
        return log_emission
    found = np.asarray(log_emission)
    return np.ascontiguousarray(np.transpose(found.reshape(*found.shape[:2], -1), (2, 1, 0)))


def get_state_major_log_emission(
    counts: PseudobulkCounts,
    parameters: CnaHmmParameters,
    confounders: CnaEmissionConfounders,
    *,
    path: np.ndarray | None,
    data_terms: bool,
) -> LogEmission:
    """The dense `lattice`, `(S, n_bins, n_cols)`, or `path`, per clone: a block of bins where `counts.clone_lengths`
    is set, else a column.

    Args:
        counts: the `PseudobulkCounts`.
        parameters: the states' `log_mu`, `p_binom`, `alphas` and `taus`.
        confounders: each clone's `theta_c` and `log Z_c`.
        path: the state per bin, or None for the lattice.
        data_terms: keep the terms no parameter enters.

    Returns:
        The `LogEmission`.
    """
    layout = "lattice" if path is None else "path"
    lengths = counts.get_clone_lengths()
    # NB the dense kernels read bins along rows: the clone-major counts transposed.
    counts = counts._replace(**{name: get_bin_major(getattr(counts, name)) for name in counts._fields[:4]})
    n_cols = np.shape(counts.umis if counts.umis is not None else counts.alt_snp_umis)[1]
    by_column = counts.clone_lengths is None and n_cols > 1
    thetas, shifts = get_clone_confounders(confounders, n_cols if by_column else lengths.size)
    if by_column and (np.any(thetas != thetas[0]) or np.any(shifts != shifts[0])):
        clones = [
            get_dense_log_emission(
                PseudobulkCounts(*(None if v is None else v[:, [c]] for v in counts[:4]), None, {}),
                parameters,
                thetas[c],
                shifts[c],
                data_terms=data_terms,
                path=path,
            )
            for c in range(n_cols)
        ]
        return LogEmission(
            *(np.concatenate(channel, axis=2) if np.ndim(channel[0]) else 0.0 for channel in zip(*clones))
        )
    if by_column:
        return get_dense_log_emission(counts, parameters, thetas[0], shifts[0], data_terms=data_terms, path=path)
    bounds = np.concatenate([[0], np.cumsum(lengths)])
    blocks = [
        get_dense_log_emission(
            PseudobulkCounts(*(None if v is None else v[a:b] for v in counts[:4]), None, {}),
            parameters,
            thetas[c],
            shifts[c],
            data_terms=data_terms,
            path=None if path is None else path[a:b],
        )
        for c, (a, b) in enumerate(itertools.pairwise(bounds))
    ]
    axis = 0 if layout == "path" else 1
    return LogEmission(
        *(np.concatenate(channel, axis=axis) if np.ndim(channel[0]) else 0.0 for channel in zip(*blocks))
    )
