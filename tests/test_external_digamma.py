"""`external.digamma.digamma_element` against `scipy.special.digamma`, bit for bit."""

from __future__ import annotations

import numpy as np
import scipy.special
from numba import njit

from cnamaste.external.digamma import digamma_element


@njit
def get_digamma_elements(x: np.ndarray) -> np.ndarray:
    """`digamma_element` of each `x`, from `njit` code."""
    out = np.empty_like(x)
    for i in range(x.size):
        out[i] = digamma_element(x[i])
    return out


def test_digamma_element_is_scipy_s_digamma_bitwise() -> None:
    """`digamma_element`, SciPy's C `psi` bound by symbol, is `scipy.special.digamma` bit for bit, x 1e-6 to 1e6.
    Fails where a SciPy upgrade renames the Cython symbol or binds another routine."""
    x = np.geomspace(1e-6, 1e6, 10_001)
    assert np.array_equal(get_digamma_elements(x), scipy.special.digamma(x))
