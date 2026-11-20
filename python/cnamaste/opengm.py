"""The external labellings of `potts.potts_energy`, OpenGM's (the run's one worker, `external.session`) and cnamaste's
expansion (`cnamaste.expansion`), and `icm_sweep_deque`'s contract met by each."""

# NB atexit runs a registered callable when the interpreter exits: it closes the run's OpenGM worker, so no
#    subprocess outlives the run.
import atexit

import numpy as np

from cnamaste import expansion, external, trws
from cnamaste.config import get_global_config, get_labelling_options, opengm_mincut_solver_type, start_time
from cnamaste.icm import icm_sweep_deque
from cnamaste.logger import get_logger
from cnamaste.potts import (
    get_field_share_order,
    pairwise_potts_energy,
    potts_energy,
    potts_problem,
    reduce_potts_problem,
)

logger = get_logger(__name__, start_time=start_time)

_SESSION: list = []
"""The run's one OpenGM worker, started on the first labelling that asks for it and shut at exit."""

OPENGM_START_ALGORITHMS = ("icm", "expansion", "swap")
"""The OpenGM algorithms that start from the current labelling; the rest take none."""

EXPANSION_SOLVERS = ("opengm_expansion", "expansion")
"""The label solvers that are alpha expansion, which `hmrf.expansion_label_order` orders."""


def get_opengm_start_algorithms() -> tuple[str, ...]:
    """Return the opengm start algorithms, ("icm", "expansion", "swap")."""
    return OPENGM_START_ALGORITHMS


def opengm_session():
    """Return the run's OpenGM session, started anew if none or the last one closed.

    Returns:
        The session.
    """
    if not _SESSION or _SESSION[0].process is None:
        _SESSION[:] = [external.session()]
        atexit.register(_SESSION[0].close)
    return _SESSION[0]


def solve_labelling(problem, solver, start, max_iterations, tolerance, mincut, options):
    """Label `problem` by `solver`, after `options`' reductions and in its expansion order.

    Args:
        problem: a `PottsProblem`.
        solver: an `hmrf.label_solver` but icm.
        start: labels, used where the solver takes a start.
        max_iterations: the solver's iteration cap.
        tolerance: the solver's relative stop.
        mincut: OpenGM's cut.
        options: the `LabellingOptions`.

    Returns:
        The `OpenGMResult` on `problem`, its bound raised by the energy the reduction fixed.
    """
    reduced = reduce_potts_problem(problem, options.fix_dominant_spots, options.prune_dominated_labels)
    algorithm = solver.removeprefix("opengm_")
    if reduced.free.size == 0:
        labels = reduced.full_labels(np.zeros(0, dtype=np.int64))
        energy = pairwise_potts_energy(problem, labels)
        return external.OpenGMResult(labels, energy, energy, np.zeros(0), 0)
    reordered = solver in EXPANSION_SOLVERS and options.expansion_label_order == "field_share"
    order = get_field_share_order(reduced.problem.unary) if reordered else np.arange(reduced.clones.size)
    # NB the clones permuted so expansion's 0, 1, ... visits them in `order`; the labels come back through it.
    posed = reduced.problem._replace(unary=np.ascontiguousarray(reduced.problem.unary[:, order]))
    rank = np.argsort(order)
    begin = None
    if algorithm in get_opengm_start_algorithms():
        begin = rank[reduced.reduced_labels(start)]
    if solver == "trws":
        run = trws.solve(posed, max_iterations=max_iterations, tolerance=tolerance)
    elif solver == "expansion":
        run = expansion.solve(posed, start=begin, max_iterations=max_iterations, opengm_patch=options.opengm_patch)
    else:
        run = opengm_session().solve(
            posed, algorithm, max_iterations=max_iterations, tolerance=tolerance, start=begin, mincut=mincut
        )
    if not (options.fix_dominant_spots or options.prune_dominated_labels or reordered):
        return run
    labels = reduced.full_labels(order[run.labels])
    energy = pairwise_potts_energy(problem, labels)
    return run._replace(labels=labels, energy=energy, bound=run.bound + energy - run.energy)


def external_assignment(
    single_llf,
    adjacency_mat,
    new_assignment,
    spatial_weight,
    solver,
    min_clone_spots=200,
    max_iterations=1000,
    tolerance=1e-9,
    *,
    floor_by_solver=False,
):
    """Run `icm_sweep_deque`'s contract with `solver`, OpenGM's or cnamaste's expansion, updating `new_assignment` in
    place; OpenGM's expansion cuts by `hmrf.opengm_mincut_solver`, and every solver takes `hmrf`'s `LabellingOptions`.

    By default the solver's labelling passes through `icm_sweep_deque`, so the `min_clone_spots` floor and a fixed
    point of the ICM cost hold whatever the solver. `floor_by_solver=True` drops `icm_sweep_deque`: a clone under the
    floor is removed from the field, its spots start at their best remaining clone, and the solver runs again, until
    every occupied clone is at or above the floor. Opt-in and unmeasured against the default (TODO in 746945c).

    Args:
        single_llf: (n_spots, n_clones).
        adjacency_mat: sparse.
        new_assignment: labels.
        solver: an `hmrf.label_solver` but icm.
        floor_by_solver: the floor by re-solving rather than by `icm_sweep_deque`.

    Returns:
        The iterations, both counted, and the energy lowered.
    """
    n_clones = single_llf.shape[1]
    config = get_global_config()
    mincut, options = opengm_mincut_solver_type(config), get_labelling_options(config)
    problem = potts_problem(single_llf, adjacency_mat, spatial_weight)
    before = pairwise_potts_energy(problem, new_assignment)
    run = solve_labelling(problem, solver, new_assignment, max_iterations, tolerance, mincut, options)
    new_assignment[:] = run.labels
    niter = run.iterations

    logger.info(
        f"{solver}: energy {run.energy:.6e} from {before:.6e}, lower bound {run.bound:.6e}, {run.iterations} iterations."
    )

    if floor_by_solver:
        # NB a removed clone's field is the lowest finite value less the spread, so no spot prefers it and the
        #    problem stays finite for OpenGM.
        field = np.array(single_llf, dtype=np.float64, copy=True)
        removed = np.zeros(n_clones, dtype=bool)
        while min_clone_spots > 0:
            counts = np.bincount(new_assignment, minlength=n_clones)
            under = (counts > 0) & (counts < min_clone_spots)
            if not under.any() or not (counts >= min_clone_spots).any():
                break
            removed |= under
            kept = np.flatnonzero(~removed)
            field[:, removed] = field[:, kept].min() - (field[:, kept].max() - field[:, kept].min()) - 1.0
            moved = under[new_assignment]
            new_assignment[moved] = kept[np.argmax(single_llf[np.ix_(moved, kept)], axis=1)]
            logger.warning(
                f"{solver} left clones {np.flatnonzero(under).tolist()} under min_clone_spots={min_clone_spots}: removed and solved again."
            )
            run = solve_labelling(
                potts_problem(field, adjacency_mat, spatial_weight),
                solver,
                new_assignment,
                max_iterations,
                tolerance,
                mincut,
                options,
            )
            new_assignment[:] = run.labels
            niter += run.iterations
        after = potts_energy(single_llf, adjacency_mat, new_assignment, spatial_weight)
        return niter, before - after

    counts = np.bincount(new_assignment, minlength=n_clones)
    if min_clone_spots > 0 and 0 < counts[counts > 0].min() < min_clone_spots:
        logger.warning(
            f"{solver} left a clone under min_clone_spots={min_clone_spots} ({counts.tolist()}): icm_sweep_deque enforces it."
        )
    solved = potts_energy(single_llf, adjacency_mat, new_assignment, spatial_weight)

    more, _, labels = icm_sweep_deque(
        single_llf=single_llf,
        adj_indptr=adjacency_mat.indptr,
        adj_indices=adjacency_mat.indices,
        adj_weights=adjacency_mat.data,
        new_assignment=new_assignment,
        spatial_weight=spatial_weight,
        posterior=None,
        min_clone_spots=min_clone_spots,
    )
    new_assignment[:] = labels
    niter += more

    after = potts_energy(single_llf, adjacency_mat, new_assignment, spatial_weight)
    logger.info(f"{solver} then icm_sweep_deque: energy {after:.6e} from {solved:.6e}, {more} sweeps.")
    return niter, before - after
