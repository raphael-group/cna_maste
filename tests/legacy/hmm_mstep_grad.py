"""Frozen copy of `cnamaste.hmm_mstep_grad` before T- #100 PR 3, with `hmm_utils.digamma_difference`.

`optimize`'s analytic EM gradient, its `jac`, in `pack_params`' coordinates; the sampler's is `sandbox/hmm/mstep.py`."""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
import scipy.special
from scipy.special import digamma

from cnamaste import emission
from cnamaste.hmm_params import CnaHmmParameters, from_theta, get_theta_slices, get_unravel_theta
from cnamaste.spatio_genomic_counts import PseudobulkCounts, get_encoded_pairs


class HeldModel(NamedTuple):
    """What the EM cost holds: its encoded `counts`, `(n_obs, n_spots)`, the held parameters, the fit's own flags, each
    bin's `log_state_weights`, `(n_obs, n_states)`, None for `Z_c = 1`, and the `pseudobulk` whose views `log Z_c`
    reads."""

    counts: PseudobulkCounts
    held: CnaHmmParameters
    params: str
    optimize_nb: bool
    log_state_weights: np.ndarray | None
    pseudobulk: PseudobulkCounts


def get_normalizer_state_weights(log_mu: Any, log_state_weights: np.ndarray, counts: PseudobulkCounts) -> np.ndarray:
    """`d log Z_c / d log_mu_s`, `(n_clones, n_states)`: the softmax of `log w_gs + log_mu_s + log lambda_g` over a
    clone's bins and states, summed over its bins.

    Args:
        log_mu: `(n_states,)`.
        log_state_weights: `(n_obs, n_states)`.
        counts: the `PseudobulkCounts`: `get_log_lambda` and `get_clone_lengths`.

    Returns:
        The `(n_clones, n_states)` weights.
    """
    rates = np.asarray(log_mu, dtype=np.float64).reshape(-1)
    log_lambda, clone_lengths = counts.get_log_lambda().reshape(-1), counts.get_clone_lengths()
    bounds = np.concatenate(([0], np.cumsum(clone_lengths)))
    out = np.zeros((clone_lengths.size, rates.size))
    for clone in range(clone_lengths.size):
        first, last = int(bounds[clone]), int(bounds[clone + 1])
        terms = log_state_weights[first:last] + rates[None, :] + log_lambda[first:last, None]
        largest = terms.max(initial=-np.inf)
        if not np.isfinite(largest):
            continue
        weights = np.exp(terms - largest)
        # NB by bin, then over bins: a path's one-hot rows sum exactly, as the bins' pairwise sum did.
        weights /= weights.sum(axis=1).sum()
        out[clone] = weights.sum(axis=0)
    return out


class Coordinates:
    """`pack_params`' vector as the gradient reads it: `hmm_params`' `Theta`, the one layout both read.

    Args: `model` the `HeldModel`; `n_states`.
    """

    def __init__(self, model: HeldModel, n_states: int) -> None:
        self.model = model
        self.n_states = int(n_states)
        self.slices = get_theta_slices(self.n_states, model.params, model.optimize_nb)
        self.theta_blocks = [(name, place.stop - place.start) for name, place in self.slices.items()]
        self.width = sum(width for _, width in self.theta_blocks)
        self.unravel = get_unravel_theta(self.n_states, model.params, bool(model.optimize_nb))

    def slice(self, name: str) -> slice:
        """`name`'s place in `theta`, an empty slice where it carries no such block.

        Args: `name` a `Theta` field.
        Returns: the slice.
        """
        return self.slices.get(name, slice(0, 0))

    def unpack(self, theta: Any) -> CnaHmmParameters:
        """`theta` back into every block it carries; a block it does not carry reads as the model's held value.

        Args: `theta` the optimizer vector.
        Returns: the `CnaHmmParameters`, `(n_states, 1)` per fitted block.
        """
        parameters = from_theta(self.unravel(theta), self.model.held, self.n_states)
        # NB 1 / exp(-log tau), not exp(log tau): the last bit changes easy (2d4ce9a9) and hard (8797710b) labels.
        if "log_tau" in self.slices:
            log_tau = np.asarray(theta, dtype=np.float64)[self.slices["log_tau"]]
            parameters = parameters._replace(taus=np.ones(self.n_states) / np.exp(-log_tau))
        return parameters


def nb_logpmf_gradient(
    obs: np.ndarray, exposure: np.ndarray, rate: np.ndarray, dispersion: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """`d ell / d eta` and `d ell / d log alpha` of the NB log pmf; a bin with no exposure contributes zero to both.

    Args: `obs`, `exposure` (n_obs,); `rate` `eta` per state, (n_states,) or (n_states, n_obs); `dispersion` per state.
    Returns: the two terms, each `(n_states, n_obs)`.
    """
    eta = np.asarray(rate, dtype=np.float64)
    alpha = np.asarray(dispersion, dtype=np.float64).reshape(-1, 1)
    y = np.asarray(obs, dtype=np.float64).reshape(1, -1)
    exposure = np.asarray(exposure, dtype=np.float64).reshape(1, -1)

    # NB in `z = log(alpha m)` and `w = alpha m / (1 + alpha m)`, never `m`: where `exp` overflows the score is `-1 / alpha`.
    empty = exposure <= 0.0
    with np.errstate(divide="ignore"):
        z = np.log(alpha) + np.where(empty, 0.0, np.log(np.where(empty, 1.0, exposure))) + eta
    w = scipy.special.expit(z)
    size = 1.0 / alpha

    by_rate = np.where(empty, 0.0, y * (1.0 - w) - w * size)

    by_dispersion = np.where(
        empty,
        0.0,
        -size
        * (
            scipy.special.digamma(y + size)
            - scipy.special.digamma(size)
            - np.logaddexp(0.0, z)
            + w
            - alpha * y * (1.0 - w)
        ),
    )

    return by_rate, by_dispersion


def bb_logpmf_gradient(
    obs: np.ndarray, total: np.ndarray, p_binom: np.ndarray, taus: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """`d ell / d p` and `d ell / d log tau` of the beta-binomial log pmf.

    Args: `obs`, `total` (n_obs,) counts; `p_binom`, `taus` per state.
    Returns: the two terms, each `(n_states, n_obs)`.
    """
    p = np.asarray(p_binom, dtype=np.float64).reshape(-1, 1)
    tau = np.asarray(taus, dtype=np.float64).reshape(-1, 1)

    z = np.asarray(obs, dtype=np.float64).reshape(1, -1)
    n = np.asarray(total, dtype=np.float64).reshape(1, -1)

    a = np.maximum(p * tau, 1e-10)
    b = np.maximum((1.0 - p) * tau, 1e-10)

    # NB `hmm_utils.digamma_difference`, asymptotic above 100: precise at large tau, where the digammas cancel.
    by_a, by_b, by_ab = digamma_difference(a, z), digamma_difference(b, n - z), digamma_difference(a + b, n)

    return tau * (by_a - by_b), a * by_a + b * by_b - (a + b) * by_ab


def weighted_gradient(gamma: np.ndarray, theta: Any, coordinates: Coordinates) -> np.ndarray:
    """`-sum_ig gamma_ig d ell_ig / d theta`, the EM cost's gradient at fixed posteriors.

    Args: `gamma` (n_states, n_obs) posteriors; `theta`; `coordinates` the vector's.
    Returns: the gradient, shaped as `theta`.
    """
    model, n_states = coordinates.model, coordinates.n_states
    parameters = coordinates.unpack(theta)
    fitted = dict(coordinates.theta_blocks)
    theta_blocks: dict[str, np.ndarray] = {}

    if "log_mu" in fitted:
        obs = np.asarray(model.counts.umis, dtype=np.float64)[0]
        exposure = np.asarray(model.counts.normal_baseline_sized, dtype=np.float64)[0]
        rates = np.asarray(parameters.log_mu, dtype=np.float64).reshape(-1)
        shift = np.zeros(obs.size)
        weights_by_state = model.log_state_weights

        if weights_by_state is not None:
            rated = CnaHmmParameters(None, rates, None, None, None)
            log_mu_shift = emission.get_log_mu_shift(rated, weights_by_state, model.pseudobulk).reshape(-1)
            shift = np.repeat(log_mu_shift, model.pseudobulk.get_clone_lengths())

        eta = rates[:, None] - shift[None, :]
        by_eta, by_alpha = nb_logpmf_gradient(obs, exposure, eta, np.asarray(parameters.alphas, dtype=np.float64))
        weighted = gamma * by_eta
        by_rate = weighted.sum(axis=1)

        if weights_by_state is not None:
            # NB `d ell / d log_mu_j` less what the normalizer takes back, `delta_ij - w_cj`.
            weights = get_normalizer_state_weights(rates, weights_by_state, model.pseudobulk)
            bounds = np.concatenate(([0], np.cumsum(model.pseudobulk.get_clone_lengths())))

            for clone in range(bounds.size - 1):
                first, last = int(bounds[clone]), int(bounds[clone + 1])
                by_rate -= weights[clone] * weighted[:, first:last].sum()

        theta_blocks["log_mu"] = by_rate
        theta_blocks["log_alpha"] = np.array([(gamma * by_alpha).sum(axis=1).sum()])

    if "logit_p" in fitted:
        obs = np.asarray(model.counts.alt_snp_umis, dtype=np.float64)[0]
        total = np.asarray(model.counts.total_snp_umis, dtype=np.float64)[0]
        probabilities = np.asarray(parameters.p_binom, dtype=np.float64).reshape(-1)

        by_p, by_tau = bb_logpmf_gradient(obs, total, probabilities, np.asarray(parameters.taus, dtype=np.float64))

        # NB `theta` carries `logit p`, so the chain rule brings `p (1 - p)`.
        theta_blocks["logit_p"] = (gamma * by_p).sum(axis=1) * probabilities * (1.0 - probabilities)
        theta_blocks["log_tau"] = np.array([(gamma * by_tau).sum(axis=1).sum()])

    # NB the EM cost does not read `log_startprob`: its block is zero, kept for the vector's width.
    theta_blocks["startprob"] = np.zeros(n_states)

    out = [theta_blocks[name] for name, _ in coordinates.theta_blocks]

    # NB the cost is the NEGATIVE log-likelihood and every term above a score, so the sign flips once.
    return -np.concatenate(out) if out else np.zeros(0)


def as_encoded_counts(obs: np.ndarray, covariate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`(obs, covariate)` as an encoding holds them (`spatio_genomic_counts.get_encoded_pairs`): what `cost_fn` scored.

    Args: `obs` counts; `covariate` totals or baseline, as the encoder took them.
    Returns: the prepared `(obs, covariate)`.
    """
    counts, values = get_encoded_pairs(obs, covariate, common_zero_depth=True)
    return counts.astype(np.float64), values.astype(np.float64)


def em_gradient(
    model: Any,
    n_states: int,
    counts: PseudobulkCounts,
    base_nb_mean: np.ndarray,
    held: CnaHmmParameters,
    *,
    optimize_nb: bool,
    use_logit: bool,
    log_state_weights: np.ndarray | None,
) -> Any:
    """`jac` for `optimize`'s `em` cost: `weighted_gradient` at `model.state_posteriors`, read per call.

    Args:
        model: the fit.
        n_states: the states.
        counts: the fit's stacked `PseudobulkCounts`.
        base_nb_mean: its baseline, zeroed above `max_rdr`.
        held: the parameters a block the fit holds reads.
        optimize_nb: whether any bin has a read-depth baseline.
        use_logit: True: theta's BAF block is a logit.
        log_state_weights: `(n_obs, n_states)`, None for `Z_c = 1`.

    Returns:
        A function of `theta` returning its gradient.

    Raises:
        ValueError: where `use_logit` is False.
    """
    if not use_logit:
        msg = "em_gradient needs use_logit=True: theta's BAF block is a logit."
        raise ValueError(msg)

    coordinates = Coordinates(
        for_optimize(
            model.params, counts, base_nb_mean, held, optimize_nb=optimize_nb, log_state_weights=log_state_weights
        ),
        n_states,
    )
    baf = coordinates.slice("logit_p")

    def gradient(theta: np.ndarray) -> np.ndarray:
        gamma = model.state_posteriors

        if gamma is None:
            # NB `cost_fn` runs the callback on its first call to fill these.
            msg = "em_gradient was called before the E step set state_posteriors."
            raise RuntimeError(msg)

        gamma = np.asarray(gamma, dtype=np.float64)
        gamma = gamma[:, :, 0] if gamma.ndim == 3 else gamma
        theta = np.asarray(theta, dtype=np.float64)

        if gamma.shape[0] == n_states:
            return weighted_gradient(gamma, theta, coordinates)

        if gamma.shape[0] != 2 * n_states:
            msg = f"em_gradient: {gamma.shape[0]} posterior rows for {n_states} states."
            raise ValueError(msg)

        # NB phased: rows n..2n are the states at `p -> 1 - p`, `-logit p` in `theta`, a sign in and out.
        switched = theta.copy()
        switched[baf] *= -1.0

        lower = weighted_gradient(gamma[n_states:], switched, coordinates)
        lower[baf] *= -1.0

        return weighted_gradient(gamma[:n_states], theta, coordinates) + lower

    return gradient


def for_optimize(
    params: str,
    counts: PseudobulkCounts,
    base_nb_mean: np.ndarray,
    held: CnaHmmParameters,
    *,
    optimize_nb: bool,
    log_state_weights: np.ndarray | None,
) -> HeldModel:
    """The `HeldModel` whose coordinates are `optimize`'s own, counts scored through their encodings as `cost_fn` does.

    Args:
        params: the fit's channels.
        counts: the fit's stacked `PseudobulkCounts`.
        base_nb_mean: its baseline, zeroed above `max_rdr`.
        held: the held parameters.
        optimize_nb: whether any bin has a read-depth baseline.
        log_state_weights: `(n_obs, n_states)`, None for `Z_c = 1`.

    Returns:
        The `HeldModel`.
    """
    counts_nb, base = as_encoded_counts(counts.umis.T, base_nb_mean)
    counts_bb, total = as_encoded_counts(counts.alt_snp_umis.T, counts.total_snp_umis.T)
    held = CnaHmmParameters(*(None if v is None else np.asarray(v, dtype=np.float64).reshape(-1) for v in held))
    encoded = PseudobulkCounts(counts_nb.T, base.T, counts_bb.T, total.T, None, {})
    return HeldModel(encoded, held, params, optimize_nb, log_state_weights, counts)


def digamma_difference(x: np.ndarray | float, k: np.ndarray | float) -> np.ndarray:
    """`digamma(x + k) - digamma(x)`: for `x > 100` the asymptotic difference, written to cancel nothing, else directly.

    Args:
        x: positive reals.
        k: counts, broadcast against `x`.

    Returns:
        The elementwise difference, approximating `sum_{i < k} 1 / (x + i)`.
    """
    x, k = np.broadcast_arrays(np.asarray(x, dtype=np.float64), np.asarray(k, dtype=np.float64))
    big = x > 100.0
    xs, ks = np.where(big, x, 1.0), np.where(big, k, 0.0)
    asymptotic = np.log1p(ks / xs) + ks / (2 * xs * (xs + ks)) + ks * (2 * xs + ks) / (12 * xs**2 * (xs + ks) ** 2)
    xd, kd = np.where(big, 1.0, x), np.where(big, 0.0, k)
    return np.where(big, asymptotic, digamma(xd + kd) - digamma(xd))
