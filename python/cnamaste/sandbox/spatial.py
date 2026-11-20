"""`best_equal_partition`, the best of 10,000 random grids of `initialize.spatial.get_rectangle_labelling`, which no
`CLONE_INIT_TYPES` selects. Not shipped."""

# NB best_equal_partition, unread since its branch in run_cnamaste was dead, is to be restored if a run ever needs it.

import numpy as np

from cnamaste.config import start_time
from cnamaste.initialize.spatial import get_rectangle_labelling
from cnamaste.logger import get_logger

logger = get_logger(__name__, start_time=start_time)


def best_equal_partition(coords, x_part, y_part, n_trials=10_000):
    best_var = np.inf
    best_index = None
    best_assignment = None

    for trial in range(n_trials):
        clone_assignment = get_rectangle_labelling(coords, x_part, y_part, np.random.default_rng(trial))
        initial_clone_index = [np.flatnonzero(clone_assignment == cell) for cell in range(x_part * y_part)]
        sizes = [len(idx) for idx in initial_clone_index]
        var = np.var(sizes)
        if var < best_var:
            best_var = var
            best_index = initial_clone_index
            best_assignment = clone_assignment

            logger.info(f"New best partition at trial {trial}: variance={best_var:.2f}, sizes={sizes}")

    logger.info(f"Best partition variance after {n_trials} trials: {best_var:.2f}")
    return best_index, best_assignment
