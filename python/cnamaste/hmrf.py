"""
HMRF pipeline: the clone labelling, its field and the Potts coupling, and the outer loop that fits and relabels.

`run_cnamaste` enters here four times, in sequence:

1. `run_hmrf` on the BAF clones, then `get_merged_fit`;
2. `run_hmrf` again on the RDR refinement, then `get_merged_fit`;
3. `get_reindexed_fit` once the clones are final, which is what makes the normal clone 0;
4. `get_labelling_unary` and `field_argmax` for the written tables (1463, 1474): the labelling the field alone gives.
"""

import copy
import dataclasses
import time
from dataclasses import fields, is_dataclass
from typing import Any, NamedTuple

import numpy as np
from sklearn.metrics import adjusted_rand_score

from cnamaste.config import label_solver_type, label_start_type, opengm_mincut_solver_type, start_time
from cnamaste.emission import CnaEmissionConfounders, get_log_emission, get_unconfounded
from cnamaste.fit_checks import record_fit_check, record_fit_count
from cnamaste.hmm.baum_welch import HmmTransitions, baum_welch
from cnamaste.hmm_emission_tabled import get_labelling_unary_encoding
from cnamaste.hmm_params import CnaHmmParameters
from cnamaste.hmm_utils import get_log_transmat
from cnamaste.hmrf_fit import HmrfFit, get_hmrf_fit
from cnamaste.icm import icm_sweep_deque, merge_clones
from cnamaste.labelling import get_merged_labelling, get_neutral_pinned, get_reindexed_labelling
from cnamaste.logger import get_logger
from cnamaste.opengm import external_assignment
from cnamaste.potts import potts_energy
from cnamaste.spatio_genomic_counts import PseudobulkCounts, SpatioGenomicLabelledMultiCounts

logger = get_logger(__name__, start_time=start_time)


def field_argmax(single_llf):
    """Compute the labelling the field alone gives: each spot at its own best clone, ties to the lowest index.

    It is the energy's `spatial_weight = 0` minimum, so it bounds every labelling's field term from above and is what
    the Potts prior is asked to improve on. A solver's labelling and this one differ exactly where the coupling won.

    Args:
        single_llf: (n_spots, n_clones) the per-spot, per-clone log-likelihood, as `potts_energy` takes it.

    Returns:
        The labels, (n_spots,) int64.
    """
    field = np.asarray(single_llf, dtype=np.float64)
    if field.ndim != 2:
        raise ValueError(f"the field is {field.shape}, not (n_spots, n_clones)")
    return np.argmax(field, axis=1).astype(np.int64)


def get_labelling_unary_held(counts: SpatioGenomicLabelledMultiCounts) -> SpatioGenomicLabelledMultiCounts:
    """Return `counts` holding its `labelling_unary` encoding, built once: a relabelling keeps it, a new channel drops it
    (`SpatioGenomicLabelledMultiCounts._replace`).

    Args:
        counts: a `SpatioGenomicLabelledMultiCounts`.

    Returns:
        The counts, `encodings["labelling_unary"]` the field's count tables' classes and unique counts.
    """
    if "labelling_unary" in counts.encodings:
        return counts
    held = get_labelling_unary_encoding(
        counts.umis, counts.normal_baseline_sized, counts.alt_snp_umis, counts.total_snp_umis
    )
    return counts._replace(encodings=counts.encodings | {"labelling_unary": held})


def get_labelling_unary(counts, parameters, confounders, paths, *, phased):
    """Each spot's log-likelihood under each clone's decoded path, its read depth weighted by `get_spot_weights`: the
    field the labelling maximizes.

    Args:
        counts: a `SpatioGenomicLabelledMultiCounts`, its baseline `normal_baseline_sized`; its `labelling_unary`
            encoding read where held, else built (`get_labelling_unary_held`).
        parameters: the fit's `CnaHmmParameters`.
        confounders: the fit's `CnaEmissionConfounders`, clone c's depth mean at base mu / Z_c.
        paths: `(n_clones, n_bins)` each clone's decoded state, or `(state, phase)` pair where `phased`.
        phased: whether `paths` carries the phase.

    Returns:
        The `(n_spots, n_clones)` log-likelihood field.
    """
    counts = get_labelling_unary_held(counts)
    held = {"labelling_unary": counts.encodings["labelling_unary"]}
    spot_counts = PseudobulkCounts(
        counts.umis, counts.normal_baseline_sized, counts.alt_snp_umis, counts.total_snp_umis, None, held
    )
    rdr, baf = get_log_emission(
        spot_counts,
        parameters,
        confounders,
        layout="field",
        phased=phased,
        data_terms=True,
        path=np.ascontiguousarray(paths, dtype=np.int64),
        spot_weights=counts.get_spot_weights(),
    )
    return rdr + baf


def get_hmrf_labelling(
    counts: Any, res: HmrfFit, adjacency: Any, labelling: np.ndarray, config: Any, *, merge: bool, index: int
) -> tuple[np.ndarray, np.ndarray, float]:
    """Label spots by clone under the field and the Potts coupling, merging clones where that lowers the energy.

    The energy is the field and the Potts coupling alone: a stated difference from CalicoST, which adds
    `log_persample_weights` to its unary. Every solver starts from `hmrf.label_start`: `labelling`, or `field_argmax`
    of the field; OpenGM's TRW-S takes no start.

    Args:
        counts: a `SpatioGenomicLabelledMultiCounts`.
        res: the `HmrfFit`, its `pred_cnv` `(n_clones, n_bins)`.
        adjacency: sparse spot adjacency.
        labelling: `(n_spots,)` the start, not modified.
        config: the run's.
        merge: merge clones where that lowers the energy.
        index: the labelling's index within its stage, for the log.

    Returns:
        The labels, the `(n_spots, n_clones)` field and the log-likelihood.
    """
    n_bins, n_spots = counts.n_segments, counts.n_spots
    n_states = res.parameters.p_binom.shape[0]
    pred = res["pred_cnv"]
    spatial_weight = config.hmrf.spatial_weight
    started = time.time()
    new_assignment = copy.copy(labelling)

    logger.info(
        f"Solving (pooled) emission likelihood for X.shape={(n_bins, 2, n_spots)}, n_states={n_states} and "
        f"{pred.shape[0]} clones and merge={merge}."
    )
    field = get_labelling_unary(counts, res.parameters, res.confounders, pred, phased=False)

    if config.hmrf.fixed_assignment:
        logger.warning("Assuming a fixed clone assignment")
    else:
        solver = label_solver_type(config)
        if label_start_type(config) == "field_argmax":
            new_assignment[:] = field_argmax(field)
        energy_start = potts_energy(field, adjacency, new_assignment, spatial_weight)
        solve_start = time.time()

        # NB clones under `hmrf.min_spots_per_clone` empty into random clones above it.
        floor = int(config.hmrf.min_spots_per_clone)
        if solver == "icm":
            niter, new_cost, labels = icm_sweep_deque(
                single_llf=field,
                adj_indptr=adjacency.indptr,
                adj_indices=adjacency.indices,
                adj_weights=adjacency.data,
                new_assignment=new_assignment,
                spatial_weight=spatial_weight,
                posterior=None,
                min_clone_spots=floor,
            )
            new_assignment[:] = labels
        else:
            niter, new_cost = external_assignment(
                field, adjacency, new_assignment, spatial_weight, solver, min_clone_spots=floor
            )

        seconds = time.time() - solve_start
        energy_end = potts_energy(field, adjacency, new_assignment, spatial_weight)
        logger.info(
            f"Labelling {index} by {solver}:\n    spots: {n_spots:_}\n"
            f"    labels: {field.shape[1]}\n    stored edges: {adjacency.nnz:_}\n"
            f"    spatial_weight: {spatial_weight}\n    energy: {energy_start:.6e} -> {energy_end:.6e}\n"
            f"    seconds: {seconds:.2f}\n    iterations: {niter}\n"
            f"    clones: {len(np.unique(new_assignment))}"
        )
        logger.info(f"Ready for potential merging of {len(np.unique(new_assignment))} clones?  {merge}.")

        if merge:
            labels, new_cost, _ = merge_clones(
                field, adjacency.indptr, adjacency.indices, adjacency.data, new_assignment, spatial_weight
            )
            new_assignment[:] = labels

        _, n_clone_spots = np.unique(new_assignment, return_counts=True)
        fractions = [f"{x:.3f}" for x in n_clone_spots / n_clone_spots.sum()]
        logger.info(
            f"Found new clone assignment with new cost {new_cost:.6e} in {niter} iterations "
            f"({time.time() - started:.2f}s with clone breakdown=\n{fractions})."
        )

    logger.info("Computing ln likelihood for hmrf.")

    # NB the field at each spot's label, iid given the labelling, plus the coupling once per aligned edge.
    log_likelihood = np.sum(np.take_along_axis(field, new_assignment.astype(int)[:, None], axis=1))
    rows, cols = adjacency.nonzero()
    upper = rows < cols
    log_likelihood += spatial_weight * np.sum(new_assignment[rows[upper]] == new_assignment[cols[upper]])

    return new_assignment, field, log_likelihood


def get_hmm_fit(
    pseudobulk: PseudobulkCounts,
    parameters: CnaHmmParameters,
    transitions: HmmTransitions,
    config: Any,
    *,
    params: str,
    previous_log_gamma: np.ndarray | None,
) -> HmrfFit:
    """`baum_welch` on the clones' pseudobulks from `parameters`, unconfounded, as an `HmrfFit` with no labelling.

    Args:
        pseudobulk: the `PseudobulkCounts`, one column per clone.
        parameters: the start.
        transitions: the `HmmTransitions`.
        config: the run's.
        params: the fitted channels.
        previous_log_gamma: the last fit's posteriors where `log Z_c` reads them, else None.

    Returns:
        The `HmrfFit`.
    """
    fit = baum_welch(
        pseudobulk,
        parameters,
        get_unconfounded(len(pseudobulk.umis)),
        transitions,
        params=params,
        tol=config.hmm.tol,
        previous_log_gamma=previous_log_gamma,
        config=config,
    )
    return get_hmrf_fit(fit, transitions.log_transmat)


STABLE_ITERATIONS = 3
"""The span, in outer iterations, over which `LabellingTurnover.stable` counts a spot unchanged."""


class LabellingTurnover(NamedTuple):
    """One labelling against the stage's earlier ones (`get_labelling_turnover`)."""

    changed: int
    """Spots whose label differs from the last labelling."""
    returned: np.ndarray
    """`(n_spots,)` True where a spot changed back to a label it held at an earlier, not the last, labelling."""
    stable: int
    """Spots unchanged over the last `STABLE_ITERATIONS` iterations, or over every iteration where fewer."""


def get_labelling_turnover(history: np.ndarray, labelling: np.ndarray) -> LabellingTurnover:
    """Compare a labelling to the stage's earlier labellings, in one indexing.

    Args:
        history: `(n_labellings, n_spots)` the stage's start and each labelling since, oldest first.
        labelling: `(n_spots,)` the new labelling.

    Returns:
        The `LabellingTurnover`.
    """
    changed = labelling != history[-1]
    returned = changed & np.any(history[:-1] == labelling, axis=0)
    stable = np.all(history[-STABLE_ITERATIONS:] == labelling, axis=0)
    return LabellingTurnover(int(changed.sum()), returned, int(stable.sum()))


def get_reindexed_history(history: np.ndarray, remaining_clones: np.ndarray) -> np.ndarray:
    """Map each earlier labelling to the re-indexed clones, `remaining_clones[k] -> k`; a removed clone maps to -1.

    Args:
        history: `(n_labellings, n_spots)` labellings in the old indexing.
        remaining_clones: the sorted clones that hold spots.

    Returns:
        The labellings in the new indexing.
    """
    lookup = np.full(max(int(history.max()), int(remaining_clones.max())) + 1, -1)
    lookup[remaining_clones] = np.arange(len(remaining_clones))
    return lookup[history]


def run_hmrf(
    counts: Any, labelling: np.ndarray, adjacency: Any, config: Any, *, parameters: CnaHmmParameters, stage: str
) -> HmrfFit:
    """The HMM fit and the labelling, alternated until the labelling settles, then one merge and a last fit.

    The labelling settles at ARI >= `hmrf.assignment_ari_tolerance` to the last, at one clone, or at
    `hmrf.max_iter_outer - 2` iterations; the next iteration merges clones (`hmrf.merge_clones_<stage>`), and the
    one after fits and labels the merged clones. A final fit on that labelling gives the parameters.

    Args:
        counts: a `SpatioGenomicLabelledMultiCounts`, read depth zeroed at `baf`.
        labelling: `(n_spots,)` each spot's start clone, 0..N-1.
        adjacency: sparse spot adjacency.
        config: the run's.
        parameters: the start, `get_initial_parameters`'.
        stage: `baf` (fits `sp`) or `rdr` (fits `smp`).

    Returns:
        The `HmrfFit`, its labelling `new_assignment` and the one before it `prev_assignment`.
    """
    params = {"baf": "sp", "rdr": "smp"}[stage]
    stage_merges = bool(getattr(config.hmrf, f"merge_clones_{stage}"))
    max_iter_outer = config.hmrf.max_iter_outer
    n_states = config.hmm.n_states
    last_assignment = np.array(labelling, dtype=int)
    n_clones = int(np.max(last_assignment)) + 1

    if not stage_merges:
        logger.info(f"Fitting {n_clones} clones with no clone merge (hmrf.merge_clones_{stage} is False).")

    solver = label_solver_type(config)
    setup = f"\n    solver: {solver}\n    start: {label_start_type(config)}"
    if solver == "opengm_expansion":
        flows = {"boykov_kolmogorov": "Boykov-Kolmogorov (boost)", "dinic": "Dinic (infra/opengm/dinic.cxx)"}
        cut = opengm_mincut_solver_type(config)
        setup += f"\n    max flow: {flows.get(cut, cut)}"
    if solver == "expansion":
        setup += "\n    max flow: Boykov-Kolmogorov (infra/expansion)"
    if solver == "trws":
        setup += "\n    implementation: OpenGM's TRWSi in Rust (infra/expansion)"
    logger.info(f"Running hmrfmix_concatenate_pipeline for {n_clones} clones:{setup}")

    counts = get_labelling_unary_held(counts)

    # NB a normal baseline, zero for BAF only, is what log Z_c's lambda reads (`PseudobulkCounts.get_log_lambda`).
    has_normal_lambda = bool(counts.normal_baseline_sized.count_positive())
    if not has_normal_lambda:
        logger.warning("Found ill-defined normal baseline; corresponds to baf only run.")

    # NB each clone's read depth under its own library normalizer, from the second fit on.
    reads_posteriors = bool(config.hmrf.apply_logmu_shift) and has_normal_lambda
    pseudobulk = counts.with_labelling("clone", last_assignment).get_pseudobulk("clone")
    transitions = HmmTransitions(get_log_transmat(n_states, config.hmm.t), None, counts.lengths)
    _, log_mu, p_binom, alphas, taus = parameters

    merge = False
    hmm_fit = None
    r = 0
    history = last_assignment[None, :]
    oscillating = np.zeros(len(last_assignment), dtype=bool)
    returned_total, stable = 0, len(last_assignment)

    # NB r jumps to max_iter_outer - 1 once settled: that iteration merges, the last fits the merged clones.
    while r <= max_iter_outer:
        logger.info(
            f"----****  Solving iteration {r}/{max_iter_outer} of copy number state fitting & clone assignment "
            "(HMM + HMRF) ****----"
        )
        hmm_fit = get_hmm_fit(
            pseudobulk,
            CnaHmmParameters(None, log_mu, p_binom, alphas, taus),
            transitions,
            config,
            params=params,
            previous_log_gamma=hmm_fit["log_gamma"] if reads_posteriors and r > 0 else None,
        )
        pred = hmm_fit["pred_cnv"]

        new_assignment, _, total_llf = get_hmrf_labelling(
            counts, hmm_fit, adjacency, last_assignment, config, merge=merge, index=r
        )
        remaining_clones, new_assignment_reindexed = np.unique(new_assignment, return_inverse=True)
        n_spots = np.bincount(new_assignment)
        under = np.flatnonzero((n_spots > 0) & (n_spots < config.hmrf.min_spots_per_clone))
        if len(under):
            spots = n_spots[under].tolist()
            message = (
                f"iteration {r}: clones {under.tolist()} hold {spots} spots, under {config.hmrf.min_spots_per_clone}."
            )
            record_fit_check("labelling_below_min_spots", message)

        if len(remaining_clones) < len(pseudobulk.umis):
            message = (
                f"iteration {r}: {len(remaining_clones)} of {len(pseudobulk.umis)} clones hold spots; re-indexing."
            )
            record_fit_check("emptied_clone", message)
            new_assignment = new_assignment_reindexed
            history = get_reindexed_history(history, remaining_clones)
            hmm_fit["log_gamma"] = hmm_fit["log_gamma"][remaining_clones]
            hmm_fit["pred_cnv"] = hmm_fit["pred_cnv"][remaining_clones]

        hmm_fit["prev_assignment"] = last_assignment
        hmm_fit["new_assignment"] = new_assignment
        hmm_fit["total_llf"] = total_llf
        pseudobulk = counts.with_labelling("clone", new_assignment).get_updated_pseudobulk(
            "clone", pseudobulk, last_assignment
        )

        state_usage = np.bincount(pred.reshape(-1), minlength=n_states) / pred.size
        ari = adjusted_rand_score(last_assignment, new_assignment)
        logger.info(
            f"{np.count_nonzero(last_assignment != new_assignment)}/{len(last_assignment)} assignment changes with "
            f"ARI to last assignment: {ari:.4f}"
        )
        with np.printoptions(linewidth=np.inf):
            logger.info(f"Copy number state usage [%]:\n{100.0 * state_usage}")

        turnover = get_labelling_turnover(history, new_assignment)
        oscillating |= turnover.returned
        returned_total += int(turnover.returned.sum())
        stable = turnover.stable
        logger.info(
            f"Labelling {r} turnover: {turnover.changed} changed, {int(turnover.returned.sum())} back to an earlier "
            f"label ({int(oscillating.sum())} distinct spots in the stage); stable fraction "
            f"{1.0 - turnover.changed / len(new_assignment):.4f}, {turnover.stable / len(new_assignment):.4f} over "
            f"the last {min(STABLE_ITERATIONS, len(history))} iterations."
        )
        history = np.vstack([history, new_assignment])

        r += 1
        settled = ari >= config.hmrf.assignment_ari_tolerance or len(np.unique(new_assignment)) == 1
        if (settled or r == max_iter_outer - 2) and not merge:
            r = max_iter_outer - 1
            merge = stage_merges

        _, log_mu, p_binom, alphas, taus = hmm_fit.parameters
        last_assignment = new_assignment

    record_fit_count(f"flip_backs_{stage}", returned_total)
    record_fit_count(f"oscillating_spots_{stage}", int(oscillating.sum()))
    record_fit_count(f"stable_spots_{stage}", stable)

    final = get_hmm_fit(
        pseudobulk,
        CnaHmmParameters(None, log_mu, p_binom, alphas, taus),
        transitions,
        config,
        params=params,
        previous_log_gamma=hmm_fit["log_gamma"] if reads_posteriors else None,
    )
    hmm_fit.parameters, hmm_fit.confounders = final.parameters, final.confounders
    hmm_fit.log_transmat, hmm_fit.profile = final.log_transmat, final.profile

    # NB once, after the last fit and labelling, so no fit or start depends on it.
    paths = np.asarray(hmm_fit.profile.pred_cnv).T
    hmm_fit.parameters, hmm_fit.confounders = get_neutral_pinned(
        hmm_fit.parameters, hmm_fit.confounders, paths, "p" in params
    )
    return hmm_fit


def get_nested_copy(res):
    """A copy of the fit whose nested containers are copies too: HmrfFit writes a key into its profile or
    assignment, which a shallow copy shares with the result it was handed.

    Args:
        res: the `HmrfFit`.

    Returns:
        The copy.
    """
    parts = {f.name: copy.copy(getattr(res, f.name)) for f in fields(res) if is_dataclass(getattr(res, f.name))}
    return dataclasses.replace(res, **parts)


def get_clone_selected(res, clones):
    """A copy of the fit holding `clones`' columns, in that order: `pred_cnv`, `log_gamma` and the confounders.

    Args:
        res: the `HmrfFit`, clones concatenated along bins or one column each.
        clones: the clones kept.

    Returns:
        The copy, its nested containers copies.
    """
    selected = get_nested_copy(res)
    selected["pred_cnv"] = res["pred_cnv"][clones]
    selected["log_gamma"] = res["log_gamma"][clones]
    selected.confounders = CnaEmissionConfounders(*(np.asarray(v)[clones] for v in res.confounders))
    return selected


def get_merged_fit(res, total_snp_umis, min_spots, min_snp_umis):
    """The fit at `get_merged_labelling`: each group keeps its first clone's path and `log Z_c`; `total_llf` NaN.

    Args:
        res: the `HmrfFit`.
        total_snp_umis: the spots' SNP UMIs.
        min_spots: `hmrf.min_spots_per_clone`.
        min_snp_umis: the SNP UMI floor.

    Returns:
        The merged fit, a copy; `res` itself where it holds one clone.
    """
    labelling, groups = get_merged_labelling(res["new_assignment"], total_snp_umis, min_spots, min_snp_umis)
    if len(groups) == 1 and len(groups[0]) == 1:
        return res
    merged = get_clone_selected(res, [group[0] for group in groups])
    merged["new_assignment"] = labelling
    merged["total_llf"] = np.nan
    return merged


def get_reindexed_fit(res):
    """The fit at `get_reindexed_labelling`, the normal clone 0; each clone's path and `log Z_c` follow it.

    Args:
        res: the `HmrfFit`, one parameter per state.

    Returns:
        The renumbered fit, a copy.
    """
    paths = res["pred_cnv"].T
    labelling, order = get_reindexed_labelling(res["new_assignment"], paths, res.parameters.p_binom)
    reindexed = get_clone_selected(res, order)
    reindexed["new_assignment"] = labelling
    return reindexed
