import time
from collections import Counter
from typing import Any

import jax
import jax.flatten_util
import jax.numpy as jnp
import numpy as np

from cnamaste import emission, hmm_mstep, mstep
from cnamaste.config import (
    config_defaults,
    get_global_config,
    get_logmu_shift_types,
    start_time,
)
from cnamaste.emission import EncodedEmission
from cnamaste.hmm.forward_backward import get_forward_backward, get_log_gamma
from cnamaste.hmm_params import (
    CnaHmmParameters,
    from_theta,
    get_theta_bounds,
    get_unravel_theta,
    to_theta,
)
from cnamaste.hmm_utils import get_normal_clone
from cnamaste.logger import get_logger
from cnamaste.spatio_genomic_counts import PseudobulkCounts

logger = get_logger(__name__, start_time=start_time)

# NB the finite logit a bounded fit starts a p=0,1 state at.  see `run_optimization_pipeline`."""
LOGIT_MAX = 50.0


def get_logit_max() -> float:
    """Return the finite logit a p=0,1 state starts at, `LOGIT_MAX` (50.0)."""
    return LOGIT_MAX


def get_fit_confounders(
    parameters: CnaHmmParameters,
    log_state_weights: np.ndarray | None,
    counts: PseudobulkCounts,
    held: "emission.CnaEmissionConfounders",
) -> "emission.CnaEmissionConfounders":
    """The fit's confounders: `held`'s `theta_c`, and each clone's `log Z_c` at `parameters` from the state weights,
    `held`'s without them.

    Args:
        parameters: the states' `log_mu`.
        log_state_weights: `(n_obs, n_states)`, or None to hold `log Z_c`.
        counts: the fit's stacked `PseudobulkCounts`, its views `log Z_c` reads.
        held: the confounders the fit starts from, one per clone.

    Returns:
        The `CnaEmissionConfounders`, one per clone of `counts`.
    """
    if log_state_weights is None:
        return held
    log_mu_shift = emission.get_log_mu_shift(parameters, log_state_weights, counts)
    return emission.CnaEmissionConfounders(np.asarray(held.normal_fraction).reshape(log_mu_shift.shape), log_mu_shift)


# NB used once, after the M step in `run_optimization_pipeline`: a state with no data has an unidentified rate and BAF.
UNOCCUPIED = 0.5
"""A state with less posterior mass than half a bin in the M step is unoccupied and assigned the normal state."""


def get_max_rdr_zeroed_baseline(X_nb, base_nb_mean, max_rdr=None):
    """`base_nb_mean` with the bins above `max_rdr` zeroed, as `optimize` zeroes them; a copy.

    Args:
        X_nb: (n_obs, n_spots) counts.
        base_nb_mean: (n_obs, n_spots).
        max_rdr: ceiling, or None for config `hmm.max_rdr`, one ceiling for the start, the fit and the decoder.

    Returns:
        The masked copy of `base_nb_mean`.
    """
    if max_rdr is None:
        try:
            max_rdr = float(get_global_config().hmm.max_rdr)
        except AttributeError:
            max_rdr = config_defaults["hmm"]["max_rdr"]
    baseline = np.asarray(base_nb_mean, dtype=np.float64).copy()

    with np.errstate(divide="ignore", invalid="ignore"):
        observed = np.asarray(X_nb, dtype=np.float64) / baseline
        observed[np.isnan(observed)] = 0.0
        baseline[observed > max_rdr] = 0.0

    return baseline


def get_hmm_baf_eps() -> float:
    """`hmm.baf.eps`, its default where no configuration is set.

    Returns:
        The BAF deadband about 0.5.
    """
    try:
        return float(get_global_config().hmm.baf.eps)
    except AttributeError:
        return float(config_defaults["hmm"]["baf"]["eps"])


def get_path_baf_profiles(paths: np.ndarray, p_binom: np.ndarray) -> np.ndarray:
    """Each clone's BAF per bin along its path, `get_normal_clone`'s profile.

    Args:
        paths: (n_obs, n_clones) states.
        p_binom: (n_states,).

    Returns:
        The `(n_obs, n_clones)` BAF.
    """
    return np.asarray(p_binom, dtype=np.float64).reshape(-1)[np.asarray(paths, dtype=np.int64)]


def get_neutral_state(paths, p_binom=None, eps_baf=None):
    """The state `labelling.get_neutral_pinned` pins to `mu = 1`.

    On the normal clone's bins, the most used state within `eps_baf` of balanced BAF, else the most used there; without
    `p_binom`, the most used state of every clone. Ties to the lower index. The pin supplies `mu`'s scale, which the
    shifted emission does not, so `mu` reads as depth against the normal clone's balanced state, as `ab_decode.decode` does.

    Args:
        paths: (n_obs, n_clones) states.
        p_binom: (n_states,) or None.
        eps_baf: deadband, or None for config `hmm.baf.eps`.

    Returns:
        The neutral state's index.
    """
    if eps_baf is None:
        eps_baf = get_hmm_baf_eps()
    paths = np.asarray(paths, dtype=np.int64)
    if p_binom is None:
        return int(np.argmax(np.bincount(paths.reshape(-1))))
    p_binom = np.asarray(p_binom, dtype=np.float64).reshape(-1)
    normal_clone = get_normal_clone(get_path_baf_profiles(paths, p_binom), eps_baf)
    usage = np.bincount(paths[:, normal_clone], minlength=p_binom.size)
    balanced = np.abs(p_binom - 0.5) <= eps_baf
    if np.any(balanced & (usage > 0)):
        usage = np.where(balanced, usage, -1)
    return int(np.argmax(usage))


def get_logmu_shift_type() -> str:
    """`hmrf.logmu_shift_type`, `argmax` where the configuration carries no `hmrf` section.

    Returns:
        One of `LOGMU_SHIFT_TYPES`.
    """
    # NB an attribute read, not `getattr`: the form the configuration-key scanner recognizes.
    try:
        return str(get_global_config().hmrf.logmu_shift_type)
    except AttributeError:
        return "argmax"


def get_previous_log_state_weights(log_gamma: np.ndarray, logmu_shift_type: str) -> np.ndarray:
    """The previous fit's posteriors as `emission.get_log_mu_shift`'s state weights, clones stacked along bins.

    Args:
        log_gamma: `(n_clones, n_bins, n_states)` the previous fit's log posteriors.
        logmu_shift_type: one of `LOGMU_SHIFT_TYPES`.

    Returns:
        The `(n_clones * n_bins, n_states)` log weights: one-hot on the argmax path, or `log_gamma`.

    Raises:
        ValueError: on a mode not in `LOGMU_SHIFT_TYPES`.
    """
    n_states = log_gamma.shape[-1]
    if logmu_shift_type == "argmax":
        return emission.get_hard_state_posteriors(np.argmax(log_gamma, axis=-1), n_states)
    if logmu_shift_type == "posterior":
        return np.ascontiguousarray(np.asarray(log_gamma, dtype=np.float64).reshape(-1, n_states))
    raise ValueError(f"hmrf.logmu_shift_type is one of {get_logmu_shift_types()}, not {logmu_shift_type!r}.")


def get_parameter_change(before: tuple[Any, ...], after: tuple[Any, ...]) -> float:
    """The EM step's largest change in a state's `mu` or `p`, `unpack_params`' tuples, against `tol`.

    Args:
        before: `(log_startprob, log_mu, p_binom, alphas, taus)` before the M step.
        after: the same after it.

    Returns:
        `max(|exp(log_mu') - exp(log_mu)|, |p' - p|)`, 0 for a channel not fitted.
    """
    mu = np.abs(np.exp(after[1]) - np.exp(before[1])) if before[1] is not None else np.zeros(1)
    p = np.abs(np.asarray(after[2]) - np.asarray(before[2])) if before[2] is not None else np.zeros(1)
    return float(max(np.max(mu), np.max(p)))


def get_clone_columns(encoded: EncodedEmission, n_clones: int) -> EncodedEmission:
    """Clones stacked along bins as encoded clone columns: each clone the stacked table, its bins' classes.

    Args:
        encoded: the stacked counts' `EncodedEmission`, one column.
        n_clones: the clones stacked.

    Returns:
        The clone-major `EncodedEmission`.
    """
    return EncodedEmission(
        np.repeat(encoded.rdr, n_clones, axis=0),
        encoded.rdr_index.reshape(n_clones, -1),
        np.repeat(encoded.baf, n_clones, axis=0),
        encoded.baf_index.reshape(n_clones, -1),
    )


class hmm_nophasing:
    """The M step: BFGS on the emission parameters at the E step's posteriors (`hmm.baum_welch`)."""

    def __init__(self, params="stmp"):
        self.params = params

    def get_bounds(
        self,
        n_states,
        optimize_nb=True,
        use_logit=True,
        max_alpha=1_000.0,
        min_alpha=1e-6,
        max_tau=5_000.0,
        min_tau=1e-4,
    ):
        """The `(min, max)` of each entry of the vector `pack_params` builds (`hmm_params.get_theta_bounds`).

        Args: `n_states`; `optimize_nb`, `use_logit` as `pack_params`; the dispersion bounds.

        Returns:
            The list of `(min, max)` tuples.
        """
        return get_theta_bounds(
            n_states, self.params, optimize_nb, use_logit, (min_alpha, max_alpha), (min_tau, max_tau)
        )

    def pack_params(
        self,
        log_startprob,
        log_mu,
        p_binom,
        alphas,
        taus,
        optimize_nb=True,
        use_logit=True,
    ):
        """The optimization vector: the fitted blocks of the canonical parameters, unconstrained (`hmm_params`).

        Args: `log_startprob`, `log_mu`, `p_binom`, `alphas`, `taus` canonical; `optimize_nb`, `use_logit`.

        Returns:
            The flat optimization vector.
        """
        parameters = CnaHmmParameters(log_startprob, log_mu, p_binom, alphas, taus)
        return np.asarray(jax.flatten_util.ravel_pytree(to_theta(parameters, self.params, optimize_nb, use_logit))[0])

    def unpack_params(
        self,
        x,
        n_states,
        log_startprob_init,
        log_mu_init,
        p_binom_init,
        alphas_init,
        taus_init,
        optimize_nb=True,
        use_logit=True,
    ):
        """The canonical parameters from the optimization vector; a block `x` omits is the `*_init` given.

        Args: `x` the vector; `n_states`; `*_init` for blocks `x` omits; `optimize_nb`, `use_logit` as `pack_params`.

        Returns:
            `(log_startprob, log_mu, p_binom, alphas, taus)`.
        """
        unravel = get_unravel_theta(n_states, self.params, bool(optimize_nb))
        held = CnaHmmParameters(log_startprob_init, log_mu_init, p_binom_init, alphas_init, taus_init)
        return tuple(None if v is None else np.asarray(v) for v in from_theta(unravel(x), held, n_states, use_logit))

    def run_optimization_pipeline(
        self,
        counts,
        transitions,
        init_parameters,
        confounders,
        *,
        config,
        max_iter_em,
        mstep_callback_cadence,
        callback_warm_start,
        tol,
        log_gamma,
        penalize,
        max_rdr=None,
        use_logit=True,
    ):
        """The EM fit: the emission parameters and each clone's confounders.

        Args:
            counts: clone-major `PseudobulkCounts`, `(n_clones, n_bins)`.
            transitions: `hmm.baum_welch.HmmTransitions`.
            init_parameters: the start, `CnaHmmParameters` (`initialize.genomic`); its `log_startprob` unread.
            confounders: the held `CnaEmissionConfounders`, one per clone.
            config: the run's; `mstep.solver` and `mstep.max_iter` read.
            max_iter_em: the E steps, each then an M step.
            mstep_callback_cadence: in a callback M step, an E step every this many solver iterations, 0 for none.
            callback_warm_start: the first M step is a callback M step; later ones hold their objective fixed.
            tol: the EM tolerance on `mu` and `p`.
            log_gamma: the previous fit's log posteriors, `(n_clones, n_bins, n_states)`, which weigh `log Z_c`, or
                None to hold `confounders`.
            penalize: `hmm.penalize_phase_only_on_same_cnv`.
            max_rdr: the read-depth ceiling, or None for `hmm.max_rdr`.
            use_logit: theta's BAF block is a logit.

        Returns:
            The `CnaHmmParameters`, `(n_states,)` each, and the `CnaEmissionConfounders`, one per clone.
        """
        # NB the objective and its gradient read clones stacked along bins, until T- #100 PR 3's class weights.
        n_clones = np.shape(counts.umis)[0]
        counts = counts.get_stacked()
        X, base_nb_mean, total_bb_RD, clone_lengths = (
            counts.X,
            np.transpose(counts.normal_baseline_sized),
            np.transpose(counts.total_snp_umis),
            counts.get_clone_lengths(),
        )
        log_transmat, log_sitewise_transmat, lengths = transitions
        n_states = log_transmat.shape[0]
        phased = log_sitewise_transmat is not None
        _, n_comp, n_spots = X.shape
        assert n_spots == 1, "Currently expects (a) clone(s) concatenated along the genomic axis."
        assert n_comp == 2

        # NB one ceiling, read from `hmm.max_rdr` where a run sets it, so the
        #    start, the sampler and the fit drop the same bins.
        base_nb_mean = get_max_rdr_zeroed_baseline(X[:, 0, :], base_nb_mean, max_rdr)

        optimize_nb = np.any(base_nb_mean > 0)

        if "m" in self.params:
            assert optimize_nb, "Cannot optimize negative binomial if normal baseline is not defined."

        # NB the previous fit's posteriors weigh each bin's states in log Z_c, which follows log_mu: their argmax
        #    path, or themselves (`hmrf.logmu_shift_type`); where that fit's clones are not this one's, unshifted.
        logmu_shift_type = get_logmu_shift_type()
        log_state_weights = None
        if log_gamma is not None and "m" in self.params:
            if log_gamma.shape[:2] == (n_clones, X.shape[0] // n_clones):
                log_state_weights = get_previous_log_state_weights(log_gamma, logmu_shift_type)
            else:
                logger.warning(
                    f"logmu_shift: the previous posteriors cover {log_gamma.shape[:2]} clone bins, this fit "
                    f"{(n_clones, X.shape[0] // n_clones)}: unshifted."
                )
        # NB posterior weights move with each E step: E steps outside the minimize, each M step at fixed posteriors.
        outer_e_step = log_state_weights is not None and logmu_shift_type == "posterior"
        if max_iter_em < 1:
            raise ValueError(f"hmm.max_iter_em={max_iter_em}: the fit takes at least one M step.")

        # NB `hmm_mstep`'s objective scores unmixed counts, `log Z_c` from the state weights or 0.
        held_shift = np.asarray(confounders.log_mu_shift) if log_state_weights is None else 0.0
        if np.any(np.asarray(confounders.normal_fraction) != 0.0) or np.any(held_shift != 0.0):
            raise NotImplementedError("the M step fits unconfounded clones: theta_c and a held log Z_c are 0.")

        encoded = emission.get_encoded(
            PseudobulkCounts(X[:, 0, :].T, base_nb_mean.T, X[:, 1, :].T, total_bb_RD.T, clone_lengths, {}),
            common_zero_depth=False,
        )

        logger.info(
            f"Encoders built. Medians: NB={np.median(base_nb_mean):.4f} ({encoded.compression_rate('rdr'):.2%} comp), "
            f"BB={np.median(total_bb_RD):.4f} ({encoded.compression_rate('baf'):.2%} comp)."
        )

        log_mu, p_binom, alphas, taus = (np.asarray(v).reshape(-1) for v in init_parameters[1:])
        log_startprob = np.log(np.ones(n_states) / n_states)

        logger.info(
            f"--- hmm initialized (EM) ---\n"
            f"log_mu:\n{np.array2string(log_mu, precision=4, suppress_small=True)}\n"
            f"p_binom:\n{np.array2string(p_binom, precision=4, suppress_small=True)}\n"
            f"alphas:\n{np.array2string(alphas, precision=4, suppress_small=True)}\n"
            f"taus:\n{np.array2string(taus, precision=2, suppress_small=True)}\n"
            "--------------------------------"
        )

        x0 = self.pack_params(
            log_startprob,
            log_mu,
            p_binom,
            alphas,
            taus,
            optimize_nb=optimize_nb,
            use_logit=use_logit,
        )

        self.encoded_emission, self.state_posteriors, self.class_weights, self.last_x = None, None, None, x0

        # NB the objective leaves out -lgamma(k + 1) and log C(n, k), the same in every state, unless hmm.fit_data_terms.
        try:
            fit_data_terms = bool(get_global_config().hmm.fit_data_terms)
        except AttributeError:
            fit_data_terms = False
        self.log_startprob = log_startprob
        self.iterations = 0
        self.e_steps = 0
        held = CnaHmmParameters(
            *(jnp.asarray(v, dtype=jnp.float64) for v in (log_startprob, log_mu, p_binom, alphas, taus))
        )
        structure = {
            "n_states": n_states,
            "params": self.params,
            "optimize_nb": bool(optimize_nb),
            "use_logit": use_logit,
        }

        # NB the E step at the cost's last evaluated vector, as the lattices read the emission at it.
        def e_step() -> None:
            _, this_log_mu, this_p_binom, this_alphas, this_taus = self.unpack_params(
                self.last_x, n_states, log_startprob, log_mu, p_binom, alphas, taus, optimize_nb, use_logit
            )
            parameters = CnaHmmParameters(None, this_log_mu, this_p_binom, this_alphas, this_taus)
            encoded_emission = emission.get_log_emission(
                encoded,
                parameters,
                get_fit_confounders(parameters, log_state_weights, counts, confounders),
                layout="lattice",
                encoding="coded",
                phased=phased,
                data_terms=fit_data_terms,
            )
            self.encoded_emission = get_clone_columns(encoded_emission, n_clones)
            log_alpha, log_beta = get_forward_backward(
                self.encoded_emission, self.log_startprob, log_transmat, lengths, log_sitewise_transmat, penalize
            )
            fit_log_gamma = get_log_gamma(log_alpha, log_beta)
            self.log_state_posteriors = np.ascontiguousarray(fit_log_gamma.reshape(-1, fit_log_gamma.shape[-1]).T)
            self.state_posteriors = np.exp(self.log_state_posteriors)
            self.class_weights = None
            self.e_steps += 1

        # NB an E step at the iterate every mstep_callback_cadence iterations.
        def callback(x: np.ndarray) -> None:
            self.iterations += 1
            if self.iterations % mstep_callback_cadence == 0:
                self.last_x = x
                e_step()

        def get_fit_data() -> tuple[CnaHmmParameters, hmm_mstep.ClassWeights]:
            if self.state_posteriors is None:
                e_step()
            if self.class_weights is None:
                # NB on the device once per E step, not once per cost.
                self.class_weights = jax.device_put(
                    hmm_mstep.get_class_weights(
                        encoded,
                        self.state_posteriors,
                        log_state_weights,
                        counts,
                        n_states=n_states,
                        data_terms=fit_data_terms,
                    )
                )
            return held, self.class_weights

        def cost_fn(params: np.ndarray) -> tuple[float, np.ndarray]:
            self.last_x = params
            cost, gradient = hmm_mstep.get_cost_and_gradient(params, *get_fit_data(), **structure)
            return float(cost), np.asarray(gradient)

        bounds = self.get_bounds(
            n_states,
            optimize_nb=optimize_nb,
            use_logit=use_logit,
        )

        self._said_bounded = getattr(self, "_said_bounded", False)
        solver = str(config.mstep.solver)

        start_time_opt = time.time()
        logger.info(f"Starting EM optimization with {solver}. Initial cost={cost_fn(x0)[0]:.6e}")

        # NB `hmm.bound_dispersions` applies `get_bounds`' range, `tau <= 5,000` and `alpha <= 1,000`, by L-BFGS-B, as
        #    BFGS takes no bounds. Off by default, as `config.yaml` sets it: on easy (2d4ce9a9) the bound binds in the
        #    phasing fit, 72 nats short of the unbounded optimum, and clone ARI falls 0.9836 -> 0.6977.
        try:
            bounded = bool(get_global_config().hmm.bound_dispersions)
        except AttributeError:
            bounded = False

        # NB BAF at 0 or 1 packs to an infinite logit, which L-BFGS-B returns as NaN; hence LOGIT_MAX.
        if bounded:
            x0 = np.where(np.isinf(x0), np.sign(x0) * get_logit_max(), x0)

        if bounded and not self._said_bounded:
            self._said_bounded = True

            logger.info("M step: applying get_bounds' own range, alpha in [1e-06, 1,000] and tau in [1e-04, 5,000].")

        # NB EM to `tol` on mu and p, or `max_iter_em` E steps, each then an M step; under `callback_warm_start` the
        #    first M step also E steps in its callback, later ones hold their objective fixed.
        nit, nfev, x, m_steps = 0, 0, x0, []
        for em_step in range(max_iter_em):
            # NB the first E step is the initial cost's, at x0.
            if em_step > 0:
                self.last_x = x
                e_step()
            if outer_e_step:
                log_state_weights = np.ascontiguousarray(self.log_state_posteriors.T)
                self.class_weights = None
            on_callback = em_step == 0 and callback_warm_start and mstep_callback_cadence > 0
            self.iterations = 0
            res = mstep.get_mstep_minimum(
                cost_fn,
                x,
                config,
                max_iter=int(config.mstep.max_iter),
                bounds=bounds if bounded else None,
                callback=callback if on_callback else None,
            )
            m_steps.append((on_callback, res))
            nit, nfev = nit + res.nit, nfev + res.nfev
            change = get_parameter_change(
                self.unpack_params(x, n_states, log_startprob, log_mu, p_binom, alphas, taus, optimize_nb, use_logit),
                self.unpack_params(
                    res.x, n_states, log_startprob, log_mu, p_binom, alphas, taus, optimize_nb, use_logit
                ),
            )
            x = res.x
            if change < tol:
                break

        logger.info(
            f"Optimization complete: {time.time() - start_time_opt:.2f}s | "
            f"{nit} iter | {nfev} fev | {self.e_steps} E steps | logmu_shift_type="
            f"{logmu_shift_type if log_state_weights is not None else None} | converged={res.success} | "
            + " | ".join(
                f"{name}: {sum(r.success for c, r in m_steps if c == on)}/{sum(c == on for c, _ in m_steps)} "
                f"converged, {dict(Counter(str(r.message) for c, r in m_steps if c == on))}"
                for name, on in (("callback M steps", True), ("fixed M steps", False))
            )
            + " | "
            f"negative ln. likelihood={res.fun:.6e}"
        )

        final_log_startprob, final_log_mu, final_p_binom, final_alphas, final_taus = self.unpack_params(
            res.x,
            n_states,
            log_startprob,
            log_mu,
            p_binom,
            alphas,
            taus,
            optimize_nb=optimize_nb,
            use_logit=use_logit,
        )

        # NB a state the M step's posteriors leave unoccupied (under half a bin) carries no data, so its rate and
        #    BAF are unidentified and BFGS can walk them anywhere: easy's RDR fit drove one to mu = 6.7e+09 under the
        #    analytic gradient. It is assigned the normal state, mu = 1 and p = 1/2.
        if self.state_posteriors is not None:
            occupancy = np.asarray(self.state_posteriors).sum(axis=1)
            if occupancy.size == 2 * n_states:
                occupancy = occupancy[:n_states] + occupancy[n_states:]
            unoccupied = np.flatnonzero(occupancy < UNOCCUPIED)
            if unoccupied.size:
                final_log_mu = np.array(final_log_mu, dtype=np.float64, copy=True)
                final_p_binom = np.array(final_p_binom, dtype=np.float64, copy=True)
                final_log_mu[unoccupied] = 0.0
                final_p_binom[unoccupied] = 0.5
                logger.info(
                    f"M step: states {unoccupied.tolist()} unoccupied (posterior mass < {UNOCCUPIED}); set to normal, mu = 1, p = 0.5."
                )

        final = CnaHmmParameters(
            *(
                np.asarray(v).reshape(-1)
                for v in (final_log_startprob, final_log_mu, final_p_binom, final_alphas, final_taus)
            )
        )
        fit_confounders = get_fit_confounders(final, log_state_weights, counts, confounders)
        encoded_emission = emission.get_log_emission(
            emission.get_encoded(counts._replace(normal_baseline_sized=base_nb_mean.T), common_zero_depth=False),
            final,
            fit_confounders,
            layout="lattice",
            encoding="coded",
            phased=phased,
            data_terms=True,
        )
        log_alpha, log_beta = get_forward_backward(
            get_clone_columns(encoded_emission, n_clones),
            final_log_startprob,
            log_transmat,
            lengths,
            log_sitewise_transmat,
            penalize,
        )
        log_gamma = get_log_gamma(log_alpha, log_beta).reshape(-1, log_alpha.shape[-1]).T
        state_prior = np.sum(np.exp(log_gamma), axis=1) / np.sum(np.exp(log_gamma))

        log_lines = [
            f"--- Final HMM State ({'phased' if phased else 'unphased'}) ---",
            f"p_binom:\n{np.array2string(final_p_binom, precision=3, suppress_small=True)}",
            f"taus:\n{np.array2string(final_taus, formatter={'float_kind': lambda x: f'{x:.3e}'})}",
        ]
        if optimize_nb:
            log_lines.extend(
                [
                    f"log_mu:\n{np.array2string(final_log_mu, precision=3, suppress_small=True)}",
                    f"alphas:\n{np.array2string(final_alphas, precision=3, suppress_small=True)}",
                ]
            )
        log_lines.extend(
            [
                f"State posteriors:\n{np.array2string(state_prior, formatter={'float_kind': lambda x: f'{x:.4e}'})}",
                f"Max updates (tol={tol:.6e}): mu={np.max(np.abs(np.exp(final_log_mu) - np.exp(log_mu))):.6e}",
            ]
        )
        logger.info("\n".join(log_lines))

        return final, fit_confounders
