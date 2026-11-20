"""`hmm_utils`'s `njit` kernels against independent sums and scipy.

Referees: `math.fsum` of `log(x + i)` for `log_rising`; `scipy.special.gammaln` and `logsumexp`. Tolerances stated
per test.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
import scipy.special

from cnamaste.hmm_utils import log_factorial, log_rising, logsumexp


def test_log_rising_is_the_exact_sum_of_logs() -> None:
    """`log_rising(x, m)` == `fsum(log(x + i), i < m)`, x 1e-8 to 1e8, m 0 to 1e4, to 1e-13 relative (measured
    1.2e-14). Fails where a term is dropped, the sum is off by one, or `m log x` is not the leading term."""
    for x, m in itertools.product([1e-8, 1e-4, 0.3, 1.0, 7.7, 1e3, 1e8], [0, 1, 2, 10, 100, 1000, 10000]):
        expected = math.fsum(math.log(x + i) for i in range(m))
        np.testing.assert_allclose(log_rising(x, m), expected, rtol=1e-13, atol=0.0, err_msg=f"{x}, {m}")


def test_log_factorial_is_scipy_s_gammaln() -> None:
    """`log_factorial(k)` == `gammaln(k + 1)`, k 0 to 1e5, 1e6, 1e7 and 3.5, to 1e-15 relative (measured 2.9e-16
    compiled, 6.6e-16 in Python; not bitwise, numba's libm and scipy's lgamma differ by an ulp)."""
    k = np.concatenate([np.arange(0.0, 100_001.0), [1e6, 1e7, 3.5]])
    got = log_factorial(k)
    assert got[0] == 0.0 and got[1] == 0.0
    np.testing.assert_allclose(got, scipy.special.gammaln(k + 1.0), rtol=1e-15, atol=0.0)


@pytest.mark.parametrize(
    "a",
    [[-1.0, 0.5, 3.0], [1e3, -1e3, 999.0], [-np.inf, -np.inf], [-np.inf, -2.0], [4.2], [np.inf, 0.0], [-np.inf]],
    ids=["mixed", "wide", "all -inf", "one -inf", "single", "+inf", "single -inf"],
)
def test_logsumexp_is_scipy_s(a: list[float]) -> None:
    """scipy's `logsumexp`, to 1e-15 relative (measured bitwise): all -inf is -inf, not NaN; +inf is +inf; a single
    entry itself. Fails where the max is not guarded or not subtracted."""
    values = np.array(a)
    got, expected = logsumexp(values), scipy.special.logsumexp(values)
    if np.isinf(expected):
        assert got == expected
    else:
        np.testing.assert_allclose(got, expected, rtol=1e-15, atol=0.0)
