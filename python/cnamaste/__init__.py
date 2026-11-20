"""cnamaste: copy-number inference on spatial transcriptomics."""

import os
from pathlib import Path

# NB XLA on one thread: its pool otherwise contends with numba's prange kernels, 2.8x slower lattices on easy.
os.environ["XLA_FLAGS"] = " ".join(
    [os.environ.get("XLA_FLAGS", ""), "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"]
).strip()

import jax

# NB float64 throughout: the M step's objective and its gradient are held to the numpy kernels at 1e-12.
jax.config.update("jax_enable_x64", True)

# NB compiled objectives persist across runs, as numba's cache=True kernels do.
jax.config.update(
    "jax_compilation_cache_dir",
    str(Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "cnamaste" / "jax"),
)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
