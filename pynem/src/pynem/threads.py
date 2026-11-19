"""Thread limits for pynem's parallel kernels and the BLAS under NumPy."""

try:
    import numba
except ImportError:
    numba = None

from threadpoolctl import threadpool_limits


def _forksafe_layer():
    """Take a fork-safe numba threading layer unless one was asked for.

    GNU OpenMP aborts a child forked after the parent has run a parallel
    region, which is what a worker pool started after a fit does. forksafe
    takes TBB when installed and falls back to workqueue. NUMBA_THREADING_LAYER
    reaches us as config.THREADING_LAYER and still wins.
    """
    if numba is not None and numba.config.THREADING_LAYER == "default":
        numba.config.THREADING_LAYER = "forksafe"


def set_threads(n):
    """Cap pynem's kernels and the BLAS under NumPy at ``n`` threads.

    The cap is process-wide and permanent, which is what a caller about to fork
    wants. Returns the numba thread count in force, or None without numba.
    """
    n = max(1, int(n))
    _forksafe_layer()
    threadpool_limits(n)
    if numba is None:
        return None
    n = min(n, numba.config.NUMBA_NUM_THREADS)
    numba.set_num_threads(n)
    return n
