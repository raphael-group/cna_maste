from collections import deque

import numpy as np
import scipy.sparse

# NB one logsumexp, in hmm_utils: -inf on an all -inf row, where icm's summed unguarded.
from cnamaste.config import start_time
from cnamaste.logger import get_logger

logger = get_logger(__name__, start_time=start_time)


def refuse_self_loops(rows, cols, weights):
    """Refuse an adjacency in which a stored edge of non-zero weight joins a spot to itself.

    Args:
        rows: the stored edges.
        cols: the stored edges.
        weights: the stored edges.

    Raises:
        ValueError: naming the count: a self-loop is aligned in every labelling, a constant in the energy.
    """
    loops = int(np.count_nonzero((np.asarray(rows) == np.asarray(cols)) & (np.asarray(weights) != 0)))
    if loops:
        raise ValueError(f"the adjacency has {loops} self-loops: a spot is not its own neighbour")


def calc_assignment_cost(
    single_llf,
    adj_spots,
    adj_neighbors,
    adj_weights,
    new_assignment,
    spatial_weight,
):
    """The labelling's cost, maximized: each spot's field plus `spatial_weight * w / 2` per agreeing stored edge.

    The one implementation of the Potts energy, `potts.potts_energy` its negation. Edges are read as stored,
    directed and not symmetrized, so `A` and `(A + A^T) / 2` score alike.

    Args:
        single_llf: (n_spots, n_clones).
        adj_*: the stored edges (COO rows, columns, weights), no self-loop.
        new_assignment: labels.
        spatial_weight: the coupling.

    Returns:
        The cost.

    Raises:
        ValueError: on a self-loop.
    """
    refuse_self_loops(adj_spots, adj_neighbors, adj_weights)
    labels = np.asarray(new_assignment)
    aligned = np.sum(np.asarray(adj_weights) * (labels[adj_spots] == labels[adj_neighbors]))
    field = np.asarray(single_llf, dtype=np.float64)
    return float(field[np.arange(labels.size), labels].sum() + spatial_weight * aligned / 2.0)


def merge_tallies(
    single_llf,
    adj_indptr,
    adj_indices,
    adj_weights,
    assignment,
    spatial_weight,
):
    """Tally what a merge decision needs, in one pass over the spots and their own edges.

    Args:
        single_llf: (n_spots, n_clones).
        adj_*: the adjacency as CSR (`scipy`'s `indptr`, `indices`, `data`).
        assignment: labels.
        spatial_weight: the coupling.

    Returns:
        `unary_sum[u, k]`, the field at label `k` summed over the spots now labelled `u`; `boundary_gain[u, v]`, the
            coupling across the `u`-`v` boundary, which merging would align; and the coupling already aligned.
    """
    n_spots, n_clones = single_llf.shape
    unary_sum = np.zeros((n_clones, n_clones), dtype=np.float64)
    boundary_gain = np.zeros((n_clones, n_clones), dtype=np.float64)
    aligned_cost = 0.0

    for i in range(n_spots):
        u = assignment[i]

        for k in range(n_clones):
            unary_sum[u, k] += single_llf[i, k]

        # NB the spot's own edges, which the CSR holds contiguously; every edge is seen from both ends, hence halved.
        for e in range(adj_indptr[i], adj_indptr[i + 1]):
            v = assignment[adj_indices[e]]
            half = spatial_weight * adj_weights[e] / 2.0

            if u == v:
                aligned_cost += half
            else:
                boundary_gain[u, v] += half

    return unary_sum, boundary_gain, aligned_cost


def tallied_cost(unary_sum, aligned_cost):
    """The assignment's cost from its tallies: each clone's own unary, plus the coupling already aligned.

    Args:
        unary_sum: from `merge_tallies`.
        aligned_cost: from `merge_tallies`.

    Returns:
        The cost, which `calc_assignment_cost` computes spot by spot.
    """
    unary_cost = 0.0

    for c in range(len(unary_sum)):
        unary_cost += unary_sum[c, c]

    return unary_cost + aligned_cost


def best_merge(unary_sum, boundary_gain, total_cost):
    """The clone pair whose merge gains most, and the cost it would leave.

    Only a pair sharing a boundary is considered: merging clones that touch nowhere gains no coupling, and the
    unary can only fall. Merging `u` into `v` trades `u`'s spots' unary for `v`'s and aligns the boundary both ways.

    Args:
        unary_sum: from `merge_tallies`.
        boundary_gain: from `merge_tallies`.
        total_cost: the assignment's cost.

    Returns:
        The best cost (`-inf` where no pair touches) and the pair `(u, v)`, `u` folding into `v`.
    """
    best_merge_cost = -np.inf
    best_merge_pair = (-1, -1)
    n_clones = len(unary_sum)

    for u in range(n_clones):
        for v in range(n_clones):
            # NB we cannot merge a clone with itself.
            if u == v:
                continue

            # NB only consider merging a clone pair if it reduces their boundary cost.
            if boundary_gain[u, v] > 0:
                delta_u_to_v = (unary_sum[u, v] - unary_sum[u, u]) + boundary_gain[u, v] + boundary_gain[v, u]

                if total_cost + delta_u_to_v > best_merge_cost:
                    best_merge_cost = total_cost + delta_u_to_v
                    best_merge_pair = (u, v)

    return best_merge_cost, best_merge_pair


def fold_clone(unary_sum, boundary_gain, aligned_cost, u, v):
    """Fold clone `u` into clone `v` in the tallies, in place, as a second pass over the merged labels would.

    The `u`-`v` boundary is aligned by the merge and joins the aligned coupling; `u`'s unary and its remaining
    boundaries become `v`'s, and `u` is left empty, which keeps it out of `best_merge`.

    Args: `unary_sum`, `boundary_gain` from `merge_tallies`, written in place; `aligned_cost`; the pair.

    Returns:
        The aligned coupling after the merge.
    """
    row_u, col_u = boundary_gain[u].copy(), boundary_gain[:, u].copy()
    aligned = aligned_cost + row_u[v] + col_u[v]

    unary_sum[v] += unary_sum[u]
    unary_sum[u] = 0.0

    boundary_gain[v] += row_u
    boundary_gain[:, v] += col_u
    boundary_gain[u] = 0.0
    boundary_gain[:, u] = 0.0
    boundary_gain[v, v] = 0.0

    return aligned


def merge_assignment(
    single_llf,
    adj_indptr,
    adj_indices,
    adj_weights,
    assignment,
    spatial_weight,
):
    """Compute the assignment's cost and the best pair of clones to merge.

    Args:
        single_llf: (n_spots, n_clones).
        adj_*: the adjacency as CSR.
        assignment: labels.
        spatial_weight: the coupling.

    Returns:
        The current cost, the best merge cost and the best merge pair.
    """
    unary_sum, boundary_gain, aligned_cost = merge_tallies(
        single_llf, adj_indptr, adj_indices, adj_weights, assignment, spatial_weight
    )
    total_cost = tallied_cost(unary_sum, aligned_cost)
    best_merge_cost, best_merge_pair = best_merge(unary_sum, boundary_gain, total_cost)
    _log_merge(best_merge_cost, best_merge_pair, total_cost, len(unary_sum))

    # NB the original cost, the new cost after merging this clone pair, and the pair.
    return total_cost, best_merge_cost, best_merge_pair


def _log_merge(best_merge_cost, best_merge_pair, total_cost, n_clones):
    if best_merge_cost > -np.inf:
        logger.info(
            f"Found best clone pair to merge={best_merge_pair} with improved cost={best_merge_cost:.3e}, given original cost={total_cost:.3e}."
        )
    else:
        logger.info(f"No beneficial clone merge available, for {n_clones} clones.")


def merge_clones(
    single_llf,
    adj_indptr,
    adj_indices,
    adj_weights,
    assignment,
    spatial_weight,
):
    """Merge the clone pair that gains most, while one gains, and return the labelling that leaves.

    The tallies are built once and folded at each merge (`fold_clone`), rather than recomputed: a merge changes
    only the merged clones' rows and columns, so every merge after the first costs the choice alone.

    Args: as `merge_tallies`; `assignment` is read, not written.

    Returns:
        The merged labels, their cost, and `(u, v, spots)` per merge, in the order taken.
    """
    labels = np.array(assignment, copy=True)
    unary_sum, boundary_gain, aligned_cost = merge_tallies(
        single_llf, adj_indptr, adj_indices, adj_weights, labels, spatial_weight
    )
    total_cost = tallied_cost(unary_sum, aligned_cost)
    merges = []

    while True:
        best_merge_cost, best_merge_pair = best_merge(unary_sum, boundary_gain, total_cost)
        _log_merge(best_merge_cost, best_merge_pair, total_cost, len(unary_sum))

        # NB only merge the clone pair if the cost is improved.
        if best_merge_cost <= total_cost:
            logger.info(
                f"No more beneficial merges available (best merge cost={best_merge_cost} given original cost={total_cost:.6e})."
            )
            return labels, total_cost, merges

        u, v = best_merge_pair
        num_merged_spots = int(np.count_nonzero(labels == u))
        labels[labels == u] = v
        aligned_cost = fold_clone(unary_sum, boundary_gain, aligned_cost, u, v)

        logger.info(
            f"Merged {num_merged_spots} spots from clone {u} into clone {v} with new cost={best_merge_cost} given original cost={total_cost:.6e}."
        )
        total_cost = best_merge_cost
        merges.append((u, v, num_merged_spots))


def icm_sweep_deque(
    single_llf,
    adj_indptr,
    adj_indices,
    adj_weights,
    new_assignment,
    spatial_weight,
    posterior,
    cost_zeropoint=0.0,
    min_clone_spots=200,
    random_state=0,
    energy_trace=None,
):
    """Run the ICM sweep of `potts.potts_energy`, `w / 2` per stored edge, to a fixed point: no spot moves.

    Half of each out- and in-edge per spot; an asymmetric adjacency runs on `(A + A^T) / 2`, a symmetric one
    on `A`. Labels are a copy; visiting order and floor draws from `random_state`'s generator. A tie goes to the
    lowest-numbered clone, not the spot's current one.
    Stated difference: a clone under `min_clone_spots` sends its spots to clones drawn at random among those at or
    above it, not to each spot's best, so the floor can raise the energy; the sweeps after it lower it again.

    Args:
        single_llf: (n_spots, n_clones).
        adj_*: CSR adjacency, no self-loop.
        new_assignment: labels.
        min_clone_spots: the floor.
        energy_trace: a list, opt-in, given `potts_energy` after each sweep's moves.

    Returns:
        The sweeps, the cost gained and the labels.

    Raises:
        ValueError: on a self-loop.
    """
    loop_rows = np.repeat(np.arange(len(adj_indptr) - 1), np.diff(adj_indptr))
    refuse_self_loops(loop_rows, adj_indices, adj_weights)
    stored = (loop_rows, np.asarray(adj_indices), np.asarray(adj_weights))

    new_assignment = np.array(new_assignment, dtype=np.int64, copy=True)
    rng = np.random.default_rng(random_state)
    n_spots, n_clones = single_llf.shape
    cost = cost_zeropoint

    adjacency = scipy.sparse.csr_matrix((adj_weights, adj_indices, adj_indptr), shape=(n_spots, n_spots))
    if (adjacency != adjacency.T).nnz > 0:
        half = ((adjacency + adjacency.T) / 2.0).tocsr()
        adj_indptr, adj_indices, adj_weights = half.indptr, half.indices, half.data

    w_edge = np.zeros(n_clones, dtype=np.float64)
    assignment_cost = np.zeros(n_clones, dtype=np.float64)
    niter = 0

    # 1. Initialize the Two Queues
    initial_nodes = np.arange(n_spots)
    rng.shuffle(initial_nodes)

    q_sweep = deque(initial_nodes)
    q_next = []

    in_queue = np.ones(n_spots, dtype=np.bool_)
    clone_counts = np.bincount(new_assignment, minlength=n_clones).astype(np.int32)

    logger.info(
        f"Starting (two-queue BFS) icm sweep with clone proportion:\n{clone_counts / clone_counts.sum()}, min_clone_spots={min_clone_spots}."
    )

    min_spot_guard = 0
    # NB a clone the floor dissolves leaves the label set, so a sweep cannot refill it (T- #22).
    dissolved = np.zeros(n_clones, dtype=np.bool_)

    while q_sweep or q_next:
        edits = 0

        while q_sweep:
            i = q_sweep.popleft()
            in_queue[i] = False

            # Fast CSR Neighbor Lookup
            w_edge[:] = 0.0
            start_idx = adj_indptr[i]
            end_idx = adj_indptr[i + 1]

            for k in range(start_idx, end_idx):
                neighbor = adj_indices[k]
                w_edge[new_assignment[neighbor]] += adj_weights[k]

            # Inlined Assignment Cost Calculation
            max_cost = -np.inf
            label = 0

            for c in range(n_clones):
                if dissolved[c]:
                    continue
                c_cost = single_llf[i, c] + w_edge[c] * spatial_weight

                assignment_cost[c] = c_cost
                if c_cost > max_cost:
                    max_cost = c_cost
                    label = c

            # Process Edits
            if label != new_assignment[i]:
                edits += 1
                cost += assignment_cost[label] - assignment_cost[new_assignment[i]]

                clone_counts[new_assignment[i]] -= 1
                clone_counts[label] += 1
                new_assignment[i] = label

                # 3. Add neighbors to the SECOND queue (q_next)
                for k in range(start_idx, end_idx):
                    neighbor = adj_indices[k]
                    if not in_queue[neighbor]:
                        q_next.append(neighbor)
                        in_queue[neighbor] = True

        sweep_edit_rate = edits / n_spots
        logger.info(f"Completed icm sweep epoch with a sweep edit rate={sweep_edit_rate:.6e}.")
        if energy_trace is not None:
            energy_trace.append(-calc_assignment_cost(single_llf, *stored, new_assignment, spatial_weight))

        # Minimum Spot Enforcement
        # NB the floor reads every occupied clone; an empty one does not switch it off (T- #22).
        if (min_clone_spots > 0) and np.any((clone_counts > 0) & (clone_counts < min_clone_spots)):
            eligible_clones_global = np.where(clone_counts >= min_clone_spots)[0]

            for c in range(n_clones):
                if clone_counts[c] > 0 and clone_counts[c] < min_clone_spots and len(eligible_clones_global) > 0:
                    dissolved[c] = True
                    spot_indices = np.where(new_assignment == c)[0]

                    for idx in spot_indices:
                        new_label = rng.choice(eligible_clones_global)

                        new_assignment[idx] = new_label
                        clone_counts[c] -= 1
                        clone_counts[new_label] += 1

                        # NB the forced spot and its neighbours return to the second queue (T- #22).
                        if not in_queue[idx]:
                            q_next.append(idx)
                            in_queue[idx] = True
                        start_idx = adj_indptr[idx]
                        end_idx = adj_indptr[idx + 1]
                        for k in range(start_idx, end_idx):
                            neighbor = adj_indices[k]
                            if not in_queue[neighbor]:
                                q_next.append(neighbor)
                                in_queue[neighbor] = True

            logger.warning(
                f"For enforcing min_clone_spot={min_clone_spots} with n_spots={n_spots}, found {len(eligible_clones_global)} valid clones. New clone proportion:\n{clone_counts / clone_counts.sum()}"
            )

            if len(eligible_clones_global) > 1:
                sweep_edit_rate = np.inf
                min_spot_guard += 1

        niter += 1

        if (sweep_edit_rate == 0.0) or (np.count_nonzero(clone_counts) <= 1) or (min_spot_guard > 10):
            break

        # 4. Randomize the second queue and promote it when the first is empty
        if q_next:
            rng.shuffle(q_next)
            q_sweep = deque(q_next)
            q_next = []

    return niter, cost, new_assignment
