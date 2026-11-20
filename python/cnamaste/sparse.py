"""Fold a term over one row of a dense or CSR `(n, m)` array, and align a second array to a CSR's stored entries.

A caller writes one term and reaches every backing with it: `sum_over_row_dense` walks a dense row, `sum_over_row_csr`
the same row of a CSR with an absent entry its dtype's zero, and `sum_over_stored_entries` the stored entries alone,
which is exact where the term is +0.0 at an absent entry. A second array travels through `values_at_stored_entries`,
which reads it at the driver's stored entries.

`cnamaste.hmm_emission_tabled` folds the clone decoded unary this way and `cnamaste.normal_spot` its squared
deviations; `cnamaste.spatio_genomic_counts` holds the containers whose values they fold.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import scipy.sparse as sp
from numba import njit

from cnamaste.spatio_genomic_counts import canonical


@njit(inline="always")
def sum_over_stored_entries(
    term: Callable[..., float],
    indptr: np.ndarray,
    indices: np.ndarray,
    stored: np.ndarray,
    aligned: np.ndarray,
    row: int,
    clone: int,
    term_args: Any,
) -> float:
    """Sum `term(stored[entry], aligned[entry], row, column, clone, term_args)` over `row`'s stored entries, in order.

    Exact where `term` is +0.0 at every absent entry.

    Args:
        term: The summand.
        indptr: The CSR's row pointers.
        indices: The CSR's column indices.
        stored: The CSR's stored values.
        aligned: A second array's values at the same entries (`values_at_stored_entries`).
        row: The row summed.
        clone: Passed to `term`.
        term_args: Passed to `term`.

    Returns:
        The float sum.
    """
    total = 0.0
    for entry in range(indptr[row], indptr[row + 1]):
        total += term(stored[entry], aligned[entry], row, indices[entry], clone, term_args)
    return total


@njit(inline="always")
def sum_over_stored_entries_and_columns(
    term: Callable[..., float],
    indptr: np.ndarray,
    indices: np.ndarray,
    stored: np.ndarray,
    aligned: np.ndarray,
    columns: np.ndarray,
    row: int,
    clone: int,
    term_args: Any,
) -> float:
    """Compute `sum_over_stored_entries`, also at the sorted `columns` a row does not store, there at (0, 0), in order.

    Exact where `term` is +0.0 at every absent entry outside `columns`.

    Args:
        term: The summand.
        indptr: The CSR's row pointers.
        indices: The CSR's column indices, sorted within a row.
        stored: The CSR's stored values.
        aligned: A second array's values at the same entries (`values_at_stored_entries`).
        columns: The sorted columns also summed where absent.
        row: The row summed.
        clone: Passed to `term`.
        term_args: Passed to `term`.

    Returns:
        The float sum.
    """
    entry, row_end = indptr[row], indptr[row + 1]
    zero, aligned_zero = np.zeros(1, stored.dtype)[0], np.zeros(1, aligned.dtype)[0]
    position = 0
    total = 0.0
    while entry < row_end or position < len(columns):
        if entry < row_end and (position == len(columns) or indices[entry] <= columns[position]):
            if position < len(columns) and indices[entry] == columns[position]:
                position += 1
            total += term(stored[entry], aligned[entry], row, indices[entry], clone, term_args)
            entry += 1
        else:
            total += term(zero, aligned_zero, row, columns[position], clone, term_args)
            position += 1
    return total


@njit(inline="always")
def sum_over_row_dense(term: Callable[..., float], values: np.ndarray, row: int, clone: int, term_args: Any) -> float:
    """Sum `term(values[row, column], row, column, clone, term_args)` over every column of a dense `values`, in order.

    Args:
        term: The summand.
        values: The dense `(n, m)` array.
        row: The row summed.
        clone: Passed to `term`.
        term_args: Passed to `term`.

    Returns:
        The float sum.
    """
    total = 0.0
    for column in range(values.shape[1]):
        total += term(values[row, column], row, column, clone, term_args)
    return total


@njit(inline="always")
def sum_over_row_csr(
    term: Callable[..., float],
    n_columns: int,
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    row: int,
    clone: int,
    term_args: Any,
) -> float:
    """Compute `sum_over_row_dense` of a CSR: every column of `row` in order, an absent entry its dtype's zero.

    For a term not zero at zero; a dense operand travels in `term_args`.

    Args:
        term: The summand.
        n_columns: The row width.
        indptr: The CSR's row pointers.
        indices: The CSR's column indices, sorted within a row.
        data: The CSR's stored values.
        row: The row summed.
        clone: Passed to `term`.
        term_args: Passed to `term`.

    Returns:
        The float sum.
    """
    entry, row_end = indptr[row], indptr[row + 1]
    zero = np.zeros(1, data.dtype)[0]
    total = 0.0
    for column in range(n_columns):
        value = zero
        if entry < row_end and indices[entry] == column:
            value = data[entry]
            entry += 1
        total += term(value, row, column, clone, term_args)
    return total


@njit(cache=True)
def aligned_csr_values(
    driver_indptr: np.ndarray,
    driver_indices: np.ndarray,
    other_indptr: np.ndarray,
    other_indices: np.ndarray,
    other_data: np.ndarray,
) -> np.ndarray:
    """Read a second CSR's values at a driver CSR's stored entries, row by row, zero where it has none.

    Args:
        driver_indptr: The driver's row pointers.
        driver_indices: The driver's column indices, sorted within a row.
        other_indptr: The second CSR's row pointers.
        other_indices: The second CSR's column indices, sorted within a row.
        other_data: The second CSR's stored values.

    Returns:
        The second CSR's values aligned to the driver's entries.
    """
    aligned = np.zeros(len(driver_indices), other_data.dtype)
    for row in range(len(driver_indptr) - 1):
        other_entry, other_end = other_indptr[row], other_indptr[row + 1]
        for entry in range(driver_indptr[row], driver_indptr[row + 1]):
            while other_entry < other_end and other_indices[other_entry] < driver_indices[entry]:
                other_entry += 1
            if other_entry < other_end and other_indices[other_entry] == driver_indices[entry]:
                aligned[entry] = other_data[other_entry]
    return aligned


def values_at_stored_entries(driver: sp.csr_matrix, other: Any) -> np.ndarray:
    """Read `other`'s values at `driver`'s stored entries, for `sum_over_stored_entries`.

    Args:
        driver: A CSR `(n, m)`.
        other: A dense or sparse `(n, m)`.

    Returns:
        `other`'s values aligned to `driver`'s entries.
    """
    if sp.issparse(other):
        other_csr = canonical(other)
        return aligned_csr_values(driver.indptr, driver.indices, other_csr.indptr, other_csr.indices, other_csr.data)
    rows = np.repeat(np.arange(driver.shape[0]), np.diff(driver.indptr))
    return np.ascontiguousarray(np.asarray(other)[rows, driver.indices])
