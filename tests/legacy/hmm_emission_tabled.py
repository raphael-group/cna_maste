"""Frozen copies of `hmm_emission_tabled`'s realized-count marking and table layout before T- #100 PR 6."""

import numpy as np
from numba import njit


@njit(cache=True)
def mark_realized_counts(counts, realized):
    """Set `realized[j]` at every count j of a 1-D or 2-D `counts`, integers within `realized`.

    Args:
        counts: the counts.
        realized: (extent,) bool, updated.
    """
    for count in counts.ravel():
        realized[int(count)] = True


@njit(cache=True)
def mark_realized_differences(b_counts, totals, realized):
    """Set `realized[n - b]` at every (n, b) pair with b <= n of 2-D `b_counts` and `totals`.

    Args:
        b_counts: the B allele counts.
        totals: the allele depths, `b_counts`'s shape.
        realized: (extent,) bool, updated.
    """
    for row in range(totals.shape[0]):
        for column in range(totals.shape[1]):
            if b_counts[row, column] <= totals[row, column]:
                realized[int(totals[row, column] - b_counts[row, column])] = True


def is_realized_layout(n_distinct: int, extent: int) -> bool:
    """Return whether the tables cover the realized counts only: fewer than half of 0..extent - 1 are realized.

    Args:
        n_distinct: the number of distinct realized counts.
        extent: 1 + the largest count.

    Returns:
        True for the realized-only layout, False for every count.
    """
    return n_distinct < extent / 2


def get_count_slots(realized: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the slot map and the tabled counts: the realized counts where `is_realized_layout`, else 0..extent - 1.

    Args:
        realized: (extent,) bool, whether count j is read.

    Returns:
        `(slots, counts)`: (extent,) int32 column of count j, -1 where untabled; the counts per column.
    """
    distinct = np.flatnonzero(realized)
    if is_realized_layout(distinct.size, realized.size):
        slots = np.full(realized.size, -1, dtype=np.int32)
        slots[distinct] = np.arange(distinct.size, dtype=np.int32)
        return slots, distinct
    return np.arange(realized.size, dtype=np.int32), np.arange(realized.size)


def get_realized(count_values, b_counts, totals, extent):
    """`get_clone_decoded_fields`' marking of the realized counts, verbatim from its `extent > 0` branch.

    Args:
        count_values: the read counts' values.
        b_counts: the B allele counts read.
        totals: the allele depths read.
        extent: 1 + the largest count.

    Returns:
        The (extent,) bool realized counts.
    """
    realized = np.zeros(extent, dtype=np.bool_)
    # NB count 0 is read at the switched segments' absent depths.
    realized[0] = True
    mark_realized_counts(count_values, realized)
    mark_realized_counts(b_counts, realized)
    mark_realized_counts(totals, realized)
    mark_realized_differences(np.atleast_2d(b_counts), np.atleast_2d(totals), realized)
    return realized
