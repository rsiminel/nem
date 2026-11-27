"""Thread use of pynem's parallel kernels, in threads and in forked workers.

Each scenario runs in a fresh interpreter: a numba threading-layer failure
aborts the whole process, and the layer is chosen once per process.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from pynem._fast import HAS_NUMBA

SRC = Path(__file__).parent.parent / "src"

DATA = """
import networkx as nx
import numpy as np
from pynem import NEM

def data(n, d):
    rng = np.random.default_rng(0)
    X = (rng.random((n, d)) < rng.random(d)).astype(float)
    return X, nx.random_regular_graph(6, n, seed=0)

def fit(X, G):
    return NEM(n_clusters=5, family="bernoulli", random_state=0).fit(X, graph=G)
"""

CONCURRENT_SCRIPT = DATA + """
from concurrent.futures import ThreadPoolExecutor

if __name__ == "__main__":
    X, G = data(3000, 100)
    with ThreadPoolExecutor(4) as ex:
        fits = list(ex.map(lambda _: fit(X, G), range(8)))
    print("fits", len(fits))
"""

FORKED_CPU_SCRIPT = DATA + """
import multiprocessing as mp
import time
import pynem

X, G = data(8000, 300)

def one_thread_fit(_):
    pynem.set_threads(1)
    wall, cpu = time.perf_counter(), time.process_time()
    fit(X, G)
    return (time.process_time() - cpu) / (time.perf_counter() - wall)

if __name__ == "__main__":
    pynem.set_threads(4)
    fit(X, G)  # the parent runs the parallel kernels before forking
    with mp.get_context("fork").Pool(1) as pool:
        print("ratio", pool.map_async(one_thread_fit, [0]).get(300)[0])
"""


def _run(tmp_path, script):
    """Run ``script`` in a fresh interpreter with numba's default thread settings."""
    env = {k: v for k, v in os.environ.items()
           if k not in ("NUMBA_THREADING_LAYER", "NUMBA_NUM_THREADS")}
    env["NUMBA_NUM_THREADS"] = "4"  # a thread pool to over-use, whatever the machine
    env["PYTHONPATH"] = str(SRC)
    path = tmp_path / "script.py"
    path.write_text(script)
    return subprocess.run([sys.executable, str(path)], env=env,
                          capture_output=True, text=True, timeout=600)


@pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
def test_concurrent_fits_in_threads(tmp_path):
    """Fits run from several Python threads at once must not abort the process."""
    done = _run(tmp_path, CONCURRENT_SCRIPT)
    assert done.returncode == 0 and "fits 8" in done.stdout, done.stderr[-2000:]


@pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="needs fork")
def test_set_threads_one_uses_one_core_in_forked_worker(tmp_path):
    """After set_threads(1), a forked worker's fit uses about one core.

    The parent fits with 4 threads first, as a caller scoring before forking
    does. CPU time over wall time counts every thread of the worker: well
    above 1 means more than one core was busy, e.g. BLAS threads started by
    the parent still spinning after the cap was lowered.
    """
    done = _run(tmp_path, FORKED_CPU_SCRIPT)
    assert done.returncode == 0, done.stderr[-2000:]
    ratio = float(done.stdout.split("ratio")[-1])
    assert ratio < 1.1, f"worker used {ratio:.2f} cores after set_threads(1)"


RAISE_SCRIPT = """
import numpy as np
import pynem

pynem.set_threads(4)
a = np.random.default_rng(0).random((1000, 1000))
print("product", (a @ a).shape)
"""


def test_set_threads_above_startup_blas_threads(tmp_path):
    """set_threads(n) above the BLAS thread count set at startup must not crash.

    OpenBLAS sizes its buffers for OPENBLAS_NUM_THREADS when it loads, so
    asking it for more threads later can segfault; clusters often set it to 1.
    """
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", PYTHONPATH=str(SRC))
    path = tmp_path / "script.py"
    path.write_text(RAISE_SCRIPT)
    done = subprocess.run([sys.executable, str(path)], env=env,
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 0 and "product" in done.stdout, (
        f"exit {done.returncode}: {done.stderr[-2000:]}")
