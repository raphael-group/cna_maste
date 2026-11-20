import numpy as np

from cnamaste.config import start_time
from cnamaste.logger import get_logger

logger = get_logger(__name__, start_time=start_time)


def get_clone_indices(assignments, clone_ids):
    """Return the indices of the spots assigned to each clone id.

    Args:
        assignments: per-spot clone labels.
        clone_ids: the ids to extract.

    Returns:
        A list of index arrays, one per clone id.
    """
    return [np.where(assignments == cid)[0] for cid in clone_ids]


def get_clone_assignment(n_spots, clone_indices):
    """Each spot's clone from the spots of each clone, `get_clone_indices`' inverse.

    Args: `n_spots`; `clone_indices` the spots of each clone, every spot in one.

    Returns:
        The `(n_spots,)` labels.
    """
    assert sum(len(indices) for indices in clone_indices) == n_spots, "Total number of spots does not match n_spots."

    assignment = np.full(n_spots, -1, dtype=int)

    for idx, indices in enumerate(clone_indices):
        assignment[indices] = idx

    return assignment
