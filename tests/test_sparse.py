"""`cnamaste.sparse` against plain loops in the same order, bitwise, and its two callers' dense and CSR arms."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import scipy.sparse as sp
from numba import njit

from cnamaste.hmm_emission_tabled import get_clone_decoded_fields, get_labelling_unary_encoding
from cnamaste.normal_spot import log1p_std_by_spot
from cnamaste.sparse import (
    aligned_csr_values,
    sum_over_row_csr,
    sum_over_row_dense,
    sum_over_stored_entries,
    sum_over_stored_entries_and_columns,
    values_at_stored_entries,
)
from cnamaste.spatio_genomic_counts import CountsEncoding

N_ROWS, N_COLUMNS = 6, 9


@njit()
def shifted_square(value: float, row: int, column: int, clone: int, term_args: tuple[float]) -> float:
    """A term not zero at zero: every absent entry contributes `term_args[0] ** 2`."""
    return (value - term_args[0]) ** 2 + 0.5 * row + 0.25 * column + clone


@njit()
def stored_product(stored: float, aligned: float, row: int, column: int, clone: int, term_args: tuple[float]) -> float:
    """A term +0.0 wherever `stored` is zero, as `sum_over_stored_entries` requires."""
    return stored * (aligned + term_args[0]) * (1.0 + row + column + clone)


@njit()
def stored_shifted_square(
    stored: float, aligned: float, row: int, column: int, clone: int, term_args: tuple[float]
) -> float:
    """`shifted_square` of `stored`, `aligned` ignored: equal to it wherever `stored` is the dense row's value."""
    return shifted_square(stored, row, column, clone, term_args)


@njit()
def value_of(value: float, row: int, column: int, clone: int, term_args: tuple[float]) -> float:
    """The entry itself: a sum of it shows the order of addition where floats cancel."""
    return value


@njit()
def stored_value_of(stored: float, aligned: float, row: int, column: int, clone: int, term_args: tuple[float]) -> float:
    return stored


@njit()
def row_sums_dense(values: np.ndarray, clone: int, term_args: tuple[float]) -> np.ndarray:
    return np.array(
        [sum_over_row_dense(shifted_square, values, row, clone, term_args) for row in range(values.shape[0])]
    )


@njit()
def row_sums_csr(
    n_columns: int, indptr: np.ndarray, indices: np.ndarray, data: np.ndarray, clone: int, term_args: tuple[float]
) -> np.ndarray:
    return np.array(
        [
            sum_over_row_csr(shifted_square, n_columns, indptr, indices, data, row, clone, term_args)
            for row in range(len(indptr) - 1)
        ]
    )


@njit()
def stored_sums(
    indptr: np.ndarray,
    indices: np.ndarray,
    stored: np.ndarray,
    aligned: np.ndarray,
    clone: int,
    term_args: tuple[float],
) -> np.ndarray:
    return np.array(
        [
            sum_over_stored_entries(stored_product, indptr, indices, stored, aligned, row, clone, term_args)
            for row in range(len(indptr) - 1)
        ]
    )


@njit()
def stored_square_sums(
    indptr: np.ndarray,
    indices: np.ndarray,
    stored: np.ndarray,
    aligned: np.ndarray,
    clone: int,
    term_args: tuple[float],
) -> np.ndarray:
    return np.array(
        [
            sum_over_stored_entries(stored_shifted_square, indptr, indices, stored, aligned, row, clone, term_args)
            for row in range(len(indptr) - 1)
        ]
    )


def get_counts(dtype: Any, seed: int = 7) -> np.ndarray:
    """Integer-valued `(N_ROWS, N_COLUMNS)` counts, ~60% zeros, with an empty row (1) and a full row (4)."""
    rng = np.random.default_rng(seed)
    counts = rng.integers(1, 9, size=(N_ROWS, N_COLUMNS)) * (rng.random((N_ROWS, N_COLUMNS)) < 0.4)
    counts[1], counts[4] = 0, rng.integers(1, 9, size=N_COLUMNS)
    return counts.astype(dtype)


def get_loop_sums(values: np.ndarray, clone: int, shift: float) -> np.ndarray:
    """Every row's `shifted_square` sum, in Python, position by position from column 0."""
    sums = np.zeros(values.shape[0])
    for row in range(values.shape[0]):
        total = 0.0
        for column in range(values.shape[1]):
            total += (float(values[row, column]) - shift) ** 2 + 0.5 * row + 0.25 * column + clone
        sums[row] = total
    return sums


@pytest.mark.parametrize("dtype", [np.int64, np.float32, np.float64])
def test_a_dense_row_sum_is_the_ordered_loop(dtype: Any) -> None:
    counts = get_counts(dtype)
    assert np.array_equal(row_sums_dense(counts, 2, (1.5,)), get_loop_sums(counts, 2, 1.5))


@pytest.mark.parametrize("dtype", [np.int64, np.float32, np.float64])
def test_a_csr_row_sum_is_the_dense_row_sum(dtype: Any) -> None:
    """Absent entries are the dtype's zero: the term is not zero there, so a skipped entry would show."""
    counts = get_counts(dtype)
    csr = sp.csr_matrix(counts)
    csr_sums = row_sums_csr(N_COLUMNS, csr.indptr, csr.indices, csr.data, 2, (1.5,))
    assert np.array_equal(csr_sums, row_sums_dense(counts, 2, (1.5,)))


def test_a_csr_row_sum_reads_an_explicitly_stored_zero_as_zero() -> None:
    counts = get_counts(np.float64)
    csr = sp.csr_matrix(counts)
    csr.data[0] = 0.0
    counts[0, csr.indices[0]] = 0.0
    csr_sums = row_sums_csr(N_COLUMNS, csr.indptr, csr.indices, csr.data, 0, (1.5,))
    assert np.array_equal(csr_sums, row_sums_dense(counts, 0, (1.5,)))


def test_every_sum_adds_in_column_order() -> None:
    """1e16 absorbs a following 1.0: left to right gives 1.0, right to left 0.0."""
    row = np.array([[1.0e16, 1.0, -1.0e16, 1.0]])
    csr = sp.csr_matrix(row)
    assert sum_over_row_dense_of_value(row) == 1.0
    assert sum_over_row_csr_of_value(4, csr.indptr, csr.indices, csr.data) == 1.0
    assert sum_over_stored_entries_of_value(csr.indptr, csr.indices, csr.data) == 1.0


@njit()
def sum_over_row_dense_of_value(values: np.ndarray) -> float:
    return sum_over_row_dense(value_of, values, 0, 0, (0.0,))


@njit()
def sum_over_row_csr_of_value(n_columns: int, indptr: np.ndarray, indices: np.ndarray, data: np.ndarray) -> float:
    return sum_over_row_csr(value_of, n_columns, indptr, indices, data, 0, 0, (0.0,))


@njit()
def sum_over_stored_entries_of_value(indptr: np.ndarray, indices: np.ndarray, data: np.ndarray) -> float:
    return sum_over_stored_entries(stored_value_of, indptr, indices, data, data, 0, 0, (0.0,))


def test_a_zero_width_row_sums_to_zero() -> None:
    empty = np.zeros((3, 0))
    csr = sp.csr_matrix(empty)
    assert np.array_equal(row_sums_dense(empty, 0, (1.5,)), np.zeros(3))
    assert np.array_equal(row_sums_csr(0, csr.indptr, csr.indices, csr.data, 0, (1.5,)), np.zeros(3))


def test_a_stored_entry_sum_is_the_full_row_sum_for_a_term_zero_at_absent_entries() -> None:
    """Rows of a term +0.0 wherever the driver is zero: the stored entries alone give the full row, bitwise."""
    counts, other = get_counts(np.float64), get_counts(np.float64, seed=11)
    csr = sp.csr_matrix(counts)
    sums = stored_sums(csr.indptr, csr.indices, csr.data, values_at_stored_entries(csr, other), 3, (0.5,))
    loop = np.zeros(N_ROWS)
    for row in range(N_ROWS):
        for column in range(N_COLUMNS):
            loop[row] += counts[row, column] * (other[row, column] + 0.5) * (1.0 + row + column + 3)
    assert np.array_equal(sums, loop)
    assert sums[1] == 0.0


def test_a_stored_entry_sum_skips_absent_entries() -> None:
    """For a term not zero at zero the stored entries alone differ from the full row: absent entries are skipped."""
    counts = get_counts(np.float64)
    csr = sp.csr_matrix(counts)
    sums = stored_square_sums(csr.indptr, csr.indices, csr.data, csr.data, 0, (1.5,))
    full = row_sums_csr(N_COLUMNS, csr.indptr, csr.indices, csr.data, 0, (1.5,))
    assert sums[4] == full[4]
    assert sums[1] == 0.0
    assert np.all(sums[np.diff(csr.indptr) < N_COLUMNS] != full[np.diff(csr.indptr) < N_COLUMNS])


@njit()
def stored_and_column_square_sums(
    indptr: np.ndarray, indices: np.ndarray, stored: np.ndarray, columns: np.ndarray, term_args: tuple[float]
) -> np.ndarray:
    return np.array(
        [
            sum_over_stored_entries_and_columns(
                stored_shifted_square, indptr, indices, stored, stored, columns, row, 0, term_args
            )
            for row in range(len(indptr) - 1)
        ]
    )


@pytest.mark.parametrize("columns", [np.arange(N_COLUMNS), np.array([0, 3, 4, 8]), np.array([], dtype=np.int64)])
def test_the_stored_entries_and_columns_sum_is_the_dense_row_sum(columns: np.ndarray) -> None:
    """Over each row's stored entries and `columns`, absent ones at 0, in column order: a Python loop over the dense
    row, bitwise; at every column, `sum_over_row_csr`'s full row. Fails where a stored column in `columns` counts
    twice, an absent one is skipped, or the merge adds out of column order."""
    counts = get_counts(np.float64)
    csr = sp.csr_matrix(counts)
    sums = stored_and_column_square_sums(csr.indptr, csr.indices, csr.data, columns, (1.5,))
    loop = np.zeros(N_ROWS)
    for row in range(N_ROWS):
        for column in range(N_COLUMNS):
            if counts[row, column] != 0 or column in columns:
                loop[row] += (counts[row, column] - 1.5) ** 2 + 0.5 * row + 0.25 * column
    assert np.array_equal(sums, loop)
    if len(columns) == N_COLUMNS:
        assert np.array_equal(sums, row_sums_csr(N_COLUMNS, csr.indptr, csr.indices, csr.data, 0, (1.5,)))


@pytest.mark.parametrize("backing", ["dense", "csr", "unsorted csr"])
def test_values_at_stored_entries_read_the_other_array_at_the_driver_entries(backing: str) -> None:
    """Dense and CSR (sorted or not) give the same values, 0 where the other array has no entry."""
    driver = sp.csr_matrix(get_counts(np.float64))
    other_dense = get_counts(np.float64, seed=11)
    other: Any = other_dense
    if backing != "dense":
        other = sp.csr_matrix(other_dense)
    if backing == "unsorted csr":
        for row in range(N_ROWS):
            entries = slice(other.indptr[row], other.indptr[row + 1])
            other.indices[entries], other.data[entries] = other.indices[entries][::-1], other.data[entries][::-1]
        other.has_sorted_indices = False
    rows = np.repeat(np.arange(N_ROWS), np.diff(driver.indptr))
    expected = other_dense[rows, driver.indices]
    aligned = values_at_stored_entries(driver, other)
    assert aligned.dtype == np.float64 and aligned.flags.c_contiguous
    assert np.array_equal(aligned, expected)
    assert np.any(expected == 0.0)


def test_aligned_csr_values_are_the_dense_values_at_the_driver_entries() -> None:
    driver, other = sp.csr_matrix(get_counts(np.float64)), sp.csr_matrix(get_counts(np.float64, seed=11))
    aligned = aligned_csr_values(driver.indptr, driver.indices, other.indptr, other.indices, other.data)
    rows = np.repeat(np.arange(N_ROWS), np.diff(driver.indptr))
    assert np.array_equal(aligned, other.toarray()[rows, driver.indices])


def get_field_inputs(seed: int = 3) -> dict[str, Any]:
    """A 5-spot, 12-segment, 3-state, 2-clone field: integer counts, so the `lgamma` tables are taken."""
    rng = np.random.default_rng(seed)
    n_spots, n_segments = 5, 12
    total = rng.integers(0, 6, size=(n_spots, n_segments)) * (rng.random((n_spots, n_segments)) < 0.6)
    b_counts = np.minimum(rng.integers(0, 4, size=total.shape), total)
    base = rng.random((n_spots, n_segments)) * (rng.random((n_spots, n_segments)) < 0.7)
    return {
        "X_nb": (rng.integers(0, 9, size=(n_spots, n_segments)) * (base > 0)).astype(np.float64),
        "base": base,
        "X_bb": b_counts.astype(np.float64),
        "total": total.astype(np.float64),
        "paths": rng.integers(0, 3, size=(2, n_segments)),
        "log_mu": np.log([0.5, 1.0, 1.5]),
        "alphas": np.array([0.05, 0.1, 0.2]),
        "p_binom": np.array([0.2, 0.5, 0.7]),
        "taus": np.array([30.0, 40.0, 50.0]),
        "scales": np.array([1.0, 1.2]),
    }


def get_encoding_of(arrays: dict[str, Any]) -> CountsEncoding:
    """The `labelling_unary` encoding of a field call's counts.

    Args:
        arrays: `get_field_inputs`' arrays, dense or CSR.

    Returns:
        The `CountsEncoding`.
    """
    return get_labelling_unary_encoding(arrays["X_nb"], arrays["base"], arrays["X_bb"], arrays["total"])


@pytest.mark.parametrize(
    "sparse_arrays", [("X_nb",), ("base", "X_nb"), ("total", "X_bb"), ("X_nb", "base", "X_bb", "total")]
)
def test_the_clone_decoded_fields_are_the_same_from_dense_or_csr(sparse_arrays: tuple[str, ...]) -> None:
    """`hmm_emission_tabled`'s dense, CSR and stored-entry arms, through this module, bitwise."""
    dense = get_field_inputs()
    mixed = {name: sp.csr_matrix(value) if name in sparse_arrays else value for name, value in dense.items()}
    expected = get_clone_decoded_fields(**dense, encoding=get_encoding_of(dense))
    assert all(np.all(np.isfinite(e)) and np.any(e != 0.0) for e in expected)
    found = get_clone_decoded_fields(**mixed, encoding=get_encoding_of(mixed))
    assert all(np.array_equal(f, e) for f, e in zip(found, expected, strict=True))


def test_log1p_std_by_spot_is_numpy_from_dense_or_csr() -> None:
    """`normal_spot`'s dense and CSR arms, through this module, bitwise, and numpy's `std`."""
    counts = get_counts(np.float64)
    expected = log1p_std_by_spot(counts)
    assert np.array_equal(log1p_std_by_spot(sp.csr_matrix(counts)), expected)
    assert np.allclose(expected, np.std(np.log1p(counts), axis=1), rtol=0, atol=1e-15)
