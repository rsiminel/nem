"""Threading layer and thread limits for pynem's parallel kernels."""

try:
    import numba
except ImportError:
    numba = None


def _forksafe_layer():
    """Take a fork-safe numba threading layer unless one was asked for.

    GNU OpenMP aborts a child forked after the parent has run a parallel
    region, which is what a worker pool started after a fit does. forksafe
    takes TBB when installed and falls back to workqueue. NUMBA_THREADING_LAYER
    reaches us as config.THREADING_LAYER and still wins.
    """
    if numba is not None and numba.config.THREADING_LAYER == "default":
        numba.config.THREADING_LAYER = "forksafe"
