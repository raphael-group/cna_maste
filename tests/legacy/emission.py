"""Frozen copy of `emission.get_log_emission`'s lattice before T- #100 PR 2, at PR 1b (`2052aac`)."""

from math import exp

import numpy as np

from cnamaste import hmm_emission_bb, hmm_emission_nb
from cnamaste.emission import (
    CnaEmissionConfounders,
    LogEmission,
    get_admixed_moments,
    get_bin_major,
    get_clone_confounders,
    get_encoded,
    get_minor_allele_count,
    get_nb_mean,
    get_states_shared_value,
    get_statewise_as_column,
)
from cnamaste.hmm_emission_bb import bb_logpmf_vector, bb_switch_logpmf_vector
from cnamaste.hmm_emission_nb import nb_logpmf_vector
from cnamaste.hmm_params import CnaHmmParameters
from cnamaste.hmm_utils import log_factorial
from cnamaste.spatio_genomic_counts import PseudobulkCounts

LAYOUTS = ("lattice", "path", "field")


def reshape_params_by_spot(param, n_spots):
    """A per-state parameter as `(n_states, n_spots)`.

    The fit's `(n_states,)`, or `(n_states, 1)`, is repeated for every spot; one column per spot is kept as given.

    Args: `param` (n_states,), (n_states, 1) or (n_states, n_spots); `n_spots`.
    Returns: the `(n_states, n_spots)` array.
    """
    param = np.asarray(param).reshape(len(param), -1)
    return np.repeat(param, n_spots, axis=1) if param.shape[1] == 1 else param


def get_rdr_baf_emission_coded(
    counts,
    log_mu,
    alphas,
    p_binom,
    taus,
    log_mu_shift,
    clone_lengths,
    data_terms,
    phased,
):
    """The coded NB and beta-binomial emission, scored once per column's unique pairs and decoded to every observation.

    With `log_mu_shift`, each clone's read-depth mean is `base mu_s / Z_c`, `log_mu` less its `log Z_c` over the
    clone's `clone_lengths` block of observations; otherwise `base mu_s`.
    `data_terms=False` leaves out `-lgamma(k + 1)` and `log C(n, k)`, which no parameter enters. `phased` appends the
    (state, phase) pairs: the read depth repeated, the allele counts switched (`bb_switch_logpmf_vector`).

    Args:
        counts: the `PseudobulkCounts`, its `rdr` and `baf` encodings built.
        log_mu: (n_states,) or (n_states, n_spots).
        alphas: as `log_mu`.
        p_binom: as `log_mu`.
        taus: as `log_mu`.
        log_mu_shift: `(n_clones,)`, 0 for `Z_c = 1`.
        clone_lengths: bins per clone, or None.
        data_terms: keep the terms no parameter enters.
        phased: append the (state, phase) pairs.

    Returns:
        The read-depth and allele log emissions over `n_states`, or `2 n_states` where `phased`.
    """
    nb_endogs, nb_exposures = counts.get_unique("rdr")
    bb_endogs, bb_exposures = counts.get_unique("baf")
    n_spots = len(nb_endogs)
    assert len(bb_endogs) == n_spots, "Encodings must have identical spot counts"
    log_mu, alphas, p_binom, taus = (reshape_params_by_spot(a, n_spots) for a in (log_mu, alphas, p_binom, taus))
    n_states = log_mu.shape[0]

    log_emit_rdr_list, log_emit_baf_list, log_k_factorials = [], [], []

    # NB typically, n_spot is unity as clones are concatenataed along the genomic axis.
    for s in range(n_spots):
        nb_endog, nb_exposure = nb_endogs[s], nb_exposures[s]
        bb_endog, bb_exposure = bb_endogs[s], bb_exposures[s]

        log_emit_rdr_uniq = np.zeros((n_states, len(nb_endog)))
        log_emit_baf_uniq = np.zeros((n_states, len(bb_endog)))

        log_k_factorial = log_factorial(nb_endog) if data_terms else np.zeros(len(nb_endog))
        for i in range(n_states):
            nb_logpmf_vector(
                nb_endog, nb_exposure, exp(log_mu[i, s]), alphas[i, s], log_emit_rdr_uniq[i, :], log_k_factorial
            )
            bb_logpmf_vector(
                bb_endog, bb_exposure, p_binom[i, s], taus[i, s], log_emit_baf_uniq[i, :], data_terms=data_terms
            )

        if phased:
            log_emit_baf_uniq = np.vstack(
                (
                    log_emit_baf_uniq,
                    bb_switch_logpmf_vector(log_emit_baf_uniq, bb_endog, bb_exposure, p_binom[:, s], taus[:, s]),
                )
            )

        log_emit_rdr_list.append(log_emit_rdr_uniq)
        log_emit_baf_list.append(log_emit_baf_uniq)
        log_k_factorials.append(log_k_factorial)

    # NB the decode is clone-major, (S, n_labels, n_bins): the lattices read (S, n_bins, n_labels).
    log_emit_rdr = np.ascontiguousarray(
        counts.decode(np.concatenate(log_emit_rdr_list, axis=1), axis=1, encoding="rdr").transpose(0, 2, 1)
    )
    if np.any(log_mu_shift):
        # NB one shift per clone: the unique counts are evaluated once per shift, each clone's bins kept.
        shifts = np.repeat(log_mu_shift, clone_lengths)
        for shift in np.unique(shifts):
            for s in range(n_spots):
                for i in range(n_states):
                    nb_logpmf_vector(
                        nb_endogs[s],
                        nb_exposures[s],
                        exp(log_mu[i, s] - shift),
                        alphas[i, s],
                        log_emit_rdr_list[s][i, :],
                        log_k_factorials[s],
                    )
            block = shifts == shift
            log_emit_rdr[:, block] = counts.decode(
                np.concatenate(log_emit_rdr_list, axis=1), axis=1, encoding="rdr"
            ).transpose(0, 2, 1)[:, block]

    log_emit_baf = np.ascontiguousarray(
        counts.decode(np.concatenate(log_emit_baf_list, axis=1), axis=1, encoding="baf").transpose(0, 2, 1)
    )
    return (np.concatenate((log_emit_rdr, log_emit_rdr)) if phased else log_emit_rdr), log_emit_baf


def get_coded_log_emission(
    counts: PseudobulkCounts,
    parameters: CnaHmmParameters,
    confounders: CnaEmissionConfounders,
    *,
    phased: bool,
    data_terms: bool,
) -> LogEmission:
    """The `lattice` on the `rdr` and `baf` encodings: `hmm_nophasing.get_rdr_baf_emission_coded`, each unique pair
    scored once.

    Args:
        counts: the `PseudobulkCounts`, its `rdr` and `baf` encodings built.
        parameters: the states' `log_mu`, `p_binom`, `alphas` and `taus`.
        confounders: each clone's `log Z_c`, and one `theta` for every clone.
        phased: append the `(state, phase)` pairs.
        data_terms: keep the terms no parameter enters.

    Returns:
        The `LogEmission`, `(S, n_bins, n_cols)` each, every channel scored.

    Raises:
        NotImplementedError: on a normal fraction per clone.
    """
    lengths = counts.get_clone_lengths()
    thetas, shifts = get_clone_confounders(confounders, len(lengths))
    if len(set(thetas)) > 1:
        raise NotImplementedError("encoded counts take one normal fraction for every clone.")
    mu, p = get_admixed_moments(parameters, thetas[0])
    log_mu = np.asarray(parameters.log_mu, dtype=np.float64) if thetas[0] == 0.0 else np.log(mu)
    n_states = log_mu.shape[0]
    rdr, baf = get_rdr_baf_emission_coded(
        counts,
        log_mu.reshape(n_states, -1),
        get_statewise_as_column(parameters.alphas, n_states),
        p.reshape(n_states, -1),
        get_statewise_as_column(parameters.taus, n_states),
        shifts,
        clone_lengths=lengths,
        data_terms=data_terms,
        phased=phased,
    )
    return LogEmission(rdr, baf)


def get_log_emission(
    counts: PseudobulkCounts,
    parameters: CnaHmmParameters,
    confounders: CnaEmissionConfounders,
    *,
    layout: str,
    phased: bool,
    fold: bool,
    data_terms: bool,
    path: np.ndarray | None = None,
    spot_weights: np.ndarray | None = None,
) -> LogEmission:
    """The read-depth and B-allele log emissions, `eqn:nb_emission` and `eqn:bb_emission`, at `get_admixed_moments`.

    Args:
        counts: the `PseudobulkCounts`; the confounders expand along its `clone_lengths`.
        parameters: the states' `log_mu`, `p_binom`, `alphas` and `taus`.
        confounders: each clone's `theta_c` and `log Z_c`.
        layout: one of `LAYOUTS`.
        phased: append the `(state, phase)` pairs, `p -> 1 - p`.
        fold: score the minor count, `min(b, n - b)`.
        data_terms: keep the terms no parameter enters; False is the M step's objective.
        path: the state per bin, on the `path` layout.
        spot_weights: each spot's weight, on the `field` layout.

    Returns:
        The `LogEmission`.

    Raises:
        ValueError: on a layout not in `LAYOUTS`, or a `path` the layout does not take.
        NotImplementedError: on `data_terms=False` on the dense lattice, a normal fraction per clone on encoded or
            phased counts, `phased` on the `path` layout, or `spot_weights`.
    """
    if layout not in LAYOUTS:
        raise ValueError(f"layout is one of {LAYOUTS}, not {layout!r}.")
    if (path is None) != (layout == "lattice"):
        raise ValueError(f"the {layout} layout takes a path: {path is not None}.")
    if layout == "field":
        raise NotImplementedError("the field is unchanged: emission.get_log_emission.")
    if counts.encodings and layout == "lattice":
        return get_coded_log_emission(counts, parameters, confounders, phased=phased, data_terms=data_terms)
    if (not data_terms and path is None) or spot_weights is not None:
        raise NotImplementedError("data_terms=False on the dense lattice, and spot weights, follow.")
    lengths = counts.get_clone_lengths()
    # NB the dense kernels read bins along rows: the clone-major counts transposed.
    counts = counts._replace(**{name: get_bin_major(getattr(counts, name)) for name in counts._fields[:4]})
    thetas, shifts = get_clone_confounders(confounders, len(lengths))
    bounds = np.concatenate([[0], np.cumsum(lengths)])
    b_count = counts.alt_snp_umis
    if b_count is not None and fold:
        b_count = get_minor_allele_count(b_count, counts.total_snp_umis)
    if phased and layout == "lattice":
        # NB the (state, phase) pairs are the coded kernel's: encoded once, on the depth Z_c divides.
        exposure = np.asarray(counts.normal_baseline_sized, dtype=np.float64)
        if np.any(shifts):
            exposure = exposure * np.exp(-np.repeat(shifts, lengths))[:, None]
        coded = PseudobulkCounts(
            *(get_bin_major(v) for v in (counts.umis, exposure, b_count, counts.total_snp_umis)),
            counts.clone_lengths,
            {},
        )
        coded = get_encoded(coded, common_zero_depth=False)
        theta = confounders._replace(log_mu_shift=np.zeros_like(confounders.log_mu_shift))
        return get_coded_log_emission(coded, parameters, theta, phased=True, data_terms=data_terms)
    if phased:
        raise NotImplementedError("the path layout scores unphased states.")
    rdr, baf = [], []
    # NB one block per clone: its own theta_c and log Z_c.
    for c, theta in enumerate(thetas):
        bins = slice(bounds[c], bounds[c + 1])
        mu, p = get_admixed_moments(parameters, theta)
        shift = shifts[c]
        if path is not None:
            mu, p = mu[path[bins]], p[path[bins]]
            if counts.umis is not None:
                column = (slice(None),) + (None,) * (counts.umis.ndim - 1)
                mean = get_nb_mean(counts.normal_baseline_sized[bins], mu[column], shift)
                rdr.append(
                    hmm_emission_nb.nb_logpmf_array(
                        counts.umis[bins], mean, get_states_shared_value(parameters.alphas), data_terms=data_terms
                    )
                )
            if b_count is not None:
                column = (slice(None),) + (None,) * (b_count.ndim - 1)
                tau = get_states_shared_value(parameters.taus)
                baf.append(
                    hmm_emission_bb.bb_logpmf_array(
                        b_count[bins], counts.total_snp_umis[bins], p[column], tau, data_terms=data_terms
                    )
                )
            continue
        n_states = mu.shape[0]
        # NB unmixed, the states' own log_mu, not log(exp(log_mu)).
        log_mu = np.log(mu) if theta != 0.0 else np.asarray(parameters.log_mu, dtype=np.float64)
        if counts.umis is not None:
            base = np.asarray(counts.normal_baseline_sized[bins], dtype=np.float64)
            exposure = base if shift == 0.0 else base * np.exp(-shift)
            rdr.append(
                hmm_emission_nb.nb_logpmf_dense(
                    np.asarray(counts.umis[bins], dtype=np.float64),
                    exposure,
                    log_mu.reshape(-1, 1),
                    get_statewise_as_column(parameters.alphas, n_states),
                )
            )
        if b_count is not None:
            baf.append(
                hmm_emission_bb.bb_logpmf_dense(
                    np.asarray(b_count[bins], dtype=np.float64),
                    np.asarray(counts.total_snp_umis[bins], dtype=np.float64),
                    p.reshape(-1, 1),
                    get_statewise_as_column(parameters.taus, n_states),
                )
            )
    axis = 0 if layout == "path" else 1
    return LogEmission(np.concatenate(rdr, axis=axis) if rdr else 0.0, np.concatenate(baf, axis=axis) if baf else 0.0)
