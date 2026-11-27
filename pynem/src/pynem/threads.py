"""Thread limits for pynem's parallel kernels and the BLAS under NumPy."""

try:
    import numba
except ImportError:
    numba = None

from threadpoolctl import threadpool_info, threadpool_limits


def _forksafe_layer():
    """Take a fork-safe numba threading layer unless one was asked for."""
    if numba is not None and numba.config.THREADING_LAYER == "default":
        numba.config.THREADING_LAYER = "forksafe"


def set_threads(n):
    """Cap pynem's kernels at ``n`` threads, and the BLAS under NumPy at one.

    pynem's parallelism comes from its numba kernels, so the BLAS runs on one
    thread whatever ``n``. Its limit is only lowered, and only once: asking
    OpenBLAS for more threads than it started with can segfault, and setting
    its thread count again, e.g. in a forked worker, leaves idle threads spinning.

    :param n: Maximum number of threads, clamped to at least 1 and to numba's build maximum.
    :return: The numba thread count in force, or None without numba.
    """
    n = max(1, int(n))
    _forksafe_layer()
    if any(lib["num_threads"] > 1 for lib in threadpool_info()):
        threadpool_limits(1)
    if numba is None:
        return None
    n = min(n, numba.config.NUMBA_NUM_THREADS)
    numba.set_num_threads(n)
    return n
