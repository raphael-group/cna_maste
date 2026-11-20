"""SciPy's C digamma, bound by symbol so `njit` code can call it: `scipy.special.digamma` bit for bit, and cached."""

from typing import Any

from llvmlite import binding
from numba import types
from numba.extending import get_cython_function_address

# NB `__pyx_fuse_1psi` is Cython's float64 `psi` in `scipy.special.cython_special`; a named symbol, not an address,
#    so an `njit(cache=True)` caller relinks in a new process. `tests/test_external_digamma.py` fails on a rename.
binding.add_symbol("cnamaste_digamma", get_cython_function_address("scipy.special.cython_special", "__pyx_fuse_1psi"))
digamma_element: Any = types.ExternalFunction("cnamaste_digamma", types.float64(types.float64))
"""`scipy.special.digamma` of a float, bitwise, callable from `njit` code."""
