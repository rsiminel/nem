"""Optional Numba-accelerated kernels.

The sequential (Gauss-Seidel) E-step cannot be vectorised across nodes — each
node reads the already-updated memberships of its earlier neighbours. This loop
is JIT-compiled with Numba when available; otherwise the pure-Python loop in
``core.py`` is used. The two paths are numerically identical (float64, same
operations and visit order), so PPanGGOLiN reproduction is unaffected.
"""

import math

import numpy as np

try:
    from numba import njit, prange, get_num_threads
    HAS_NUMBA = True
except Exception:  # pragma: no cover - numba is an optional dependency
    HAS_NUMBA = False
    prange = range

    def get_num_threads():
        return 1


def num_threads():
    """Thread count for the chunked kernels, read on the Python side.

    Calling numba's get_num_threads() from inside an njit function makes it
    uncacheable ('uses dynamic globals'), which costs a recompile in every
    process. Callers pass the value in instead.
    """
    return get_num_threads()

    def njit(*args, **kwargs):
        """No-op fallback so the module imports without numba."""
        if args and callable(args[0]):
            return args[0]

        def wrap(func):
            return func

        return wrap


@njit(cache=True)
def seq_sweep(CM, log_pkfki, indptr, indices, data, beta, harden):
    """One in-place sequential (Gauss-Seidel) E-step sweep.

    Parameters
    ----------
    CM : (N, K) float64 — classification, updated in place and returned.
    log_pkfki : (N, K) float64 — precomputed log(p_k f_k(x_i)).
    indptr, indices, data : CSR arrays of the (directed) weighted adjacency.
    beta : float — spatial smoothing strength.
    harden : bool — if True apply the C-step (argmax -> one-hot), i.e. NCEM.

    Mirrors ``NEM._normalize_local`` / ``_harden_node`` for the mean-field and
    ICM (ncem) variants. Gibbs (gem) needs an RNG and stays on the Python path.
    """
    N = CM.shape[0]
    K = CM.shape[1]
    ctx = np.empty(K)
    lognum = np.empty(K)

    for i in range(N):
        # spatial context: sum_{j in N(i)} w_ij * CM[j]
        for k in range(K):
            ctx[k] = 0.0
        for p in range(indptr[i], indptr[i + 1]):
            j = indices[p]
            w = data[p]
            for k in range(K):
                ctx[k] += w * CM[j, k]

        # log numerator and its max
        maxlog = -np.inf
        for k in range(K):
            val = log_pkfki[i, k] + beta * ctx[k]
            lognum[k] = val
            if val > maxlog:
                maxlog = val

        if not np.isfinite(maxlog):
            for k in range(K):
                CM[i, k] = 1.0 / K
            continue

        total = 0.0
        for k in range(K):
            e = np.exp(lognum[k] - maxlog)
            lognum[k] = e
            total += e

        if total <= 0.0:
            for k in range(K):
                CM[i, k] = 1.0 / K
            continue

        if harden:
            best = 0
            bestval = lognum[0]
            for k in range(1, K):
                if lognum[k] > bestval:
                    bestval = lognum[k]
                    best = k
            for k in range(K):
                CM[i, k] = 1.0 if k == best else 0.0
        else:
            for k in range(K):
                CM[i, k] = lognum[k] / total

    return CM


_PROB_FLOOR = 1e-20
_ZERO_DISP_TOL = 1e-20

# The four density kernels share a shape: per-(class, variable) constants that
# do not depend on the observation, then one pass over the nodes. Both parts
# used to sit inside a `for k` loop, so the constants were rebuilt per class
# (harmless) and, for the one parallel kernel, the thread team was entered once
# per class instead of once per call. Hoisting the constants to (K, D) lets the
# node loop be a single prange, and lets the other three kernels have one too:
# node i only ever writes log_pkfki[i, :], so there is no reduction and the
# per-(i, k) accumulation order over d is exactly what it was.


@njit(cache=True)
def _bernoulli_consts(dispersions, proportions, K, D):
    """log(1-eps), log((1-eps)/eps) and the degenerate mask, per (class, var)."""
    log_pk = np.empty(K)
    a = np.zeros((K, D))
    b = np.zeros((K, D))
    degen = np.zeros((K, D), dtype=np.bool_)
    for k in range(K):
        log_pk[k] = math.log(max(proportions[k], _PROB_FLOOR))
        for d in range(D):
            v = dispersions[k, d]
            if v > _ZERO_DISP_TOL:
                a[k, d] = math.log(1.0 - v)
                b[k, d] = math.log((1.0 - v) / v)
            else:
                degen[k, d] = True
    return log_pk, a, b, degen


@njit(cache=True, parallel=True)
def density_bernoulli(Xf, observed, centers, dispersions, proportions,
                      weights, has_weights, all_observed, log_pkfki):
    N, D = Xf.shape
    K = centers.shape[0]
    log_pk, log_one_minus_v, log_ratio, degen = _bernoulli_consts(
        dispersions, proportions, K, D)

    if all_observed:
        # nothing missing: the per-element observed[] load and branch below is
        # pure overhead, and this is the path every PPanGGOLiN run takes
        for i in prange(N):
            for k in range(K):
                log_fki = 0.0
                invalid = False
                for d in range(D):
                    diff = Xf[i, d] - centers[k, d]
                    if degen[k, d]:
                        if abs(diff) > _ZERO_DISP_TOL:
                            if (not has_weights) or weights[d] > 0.0:
                                invalid = True
                                break
                        continue
                    w = weights[d] if has_weights else 1.0
                    log_fki += (log_one_minus_v[k, d]
                                - abs(diff) * log_ratio[k, d]) * w
                log_pkfki[i, k] = -np.inf if invalid else log_pk[k] + log_fki
    else:
        for i in prange(N):
            for k in range(K):
                log_fki = 0.0
                invalid = False
                for d in range(D):
                    if not observed[i, d]:
                        continue
                    diff = Xf[i, d] - centers[k, d]
                    if degen[k, d]:
                        if abs(diff) > _ZERO_DISP_TOL:
                            if (not has_weights) or weights[d] > 0.0:
                                invalid = True
                                break
                        continue
                    w = weights[d] if has_weights else 1.0
                    log_fki += (log_one_minus_v[k, d]
                                - abs(diff) * log_ratio[k, d]) * w
                log_pkfki[i, k] = -np.inf if invalid else log_pk[k] + log_fki


@njit(cache=True, parallel=True)
def density_normal(Xf, observed, centers, dispersions, proportions,
                   weights, has_weights, all_observed, log_pkfki):
    N, D = Xf.shape
    K = centers.shape[0]
    log_pk = np.empty(K)
    log_coef = np.zeros((K, D))
    degen = np.zeros((K, D), dtype=np.bool_)
    for k in range(K):
        log_pk[k] = math.log(max(proportions[k], _PROB_FLOOR))
        for d in range(D):
            v = dispersions[k, d]
            if v > _ZERO_DISP_TOL:
                log_coef[k, d] = -0.5 * math.log(2.0 * math.pi * v)
            else:
                degen[k, d] = True

    for i in prange(N):
        for k in range(K):
            log_fki = 0.0
            invalid = False
            for d in range(D):
                if not (all_observed or observed[i, d]):
                    continue
                diff = Xf[i, d] - centers[k, d]
                if degen[k, d]:
                    if abs(diff) > _ZERO_DISP_TOL:
                        if (not has_weights) or weights[d] > 0.0:
                            invalid = True
                            break
                    continue
                w = weights[d] if has_weights else 1.0
                log_fki += (log_coef[k, d]
                            - 0.5 * diff * diff / dispersions[k, d]) * w
            log_pkfki[i, k] = -np.inf if invalid else log_pk[k] + log_fki


@njit(cache=True, parallel=True)
def density_laplace(Xf, observed, centers, dispersions, proportions,
                    weights, has_weights, all_observed, log_pkfki):
    N, D = Xf.shape
    K = centers.shape[0]
    log_pk = np.empty(K)
    neg_log_2v = np.zeros((K, D))
    degen = np.zeros((K, D), dtype=np.bool_)
    for k in range(K):
        log_pk[k] = math.log(max(proportions[k], _PROB_FLOOR))
        for d in range(D):
            v = dispersions[k, d]
            if v > _ZERO_DISP_TOL:
                neg_log_2v[k, d] = -math.log(2.0 * v)
            else:
                degen[k, d] = True

    for i in prange(N):
        for k in range(K):
            log_fki = 0.0
            invalid = False
            for d in range(D):
                if not (all_observed or observed[i, d]):
                    continue
                diff = Xf[i, d] - centers[k, d]
                if degen[k, d]:
                    if abs(diff) > _ZERO_DISP_TOL:
                        if (not has_weights) or weights[d] > 0.0:
                            invalid = True
                            break
                    continue
                w = weights[d] if has_weights else 1.0
                log_fki += (neg_log_2v[k, d]
                            - abs(diff) / dispersions[k, d]) * w
            log_pkfki[i, k] = -np.inf if invalid else log_pk[k] + log_fki


@njit(cache=True, parallel=True)
def density_bernoulli_mag(Xf, observed, centers, dispersions, proportions,
                          completeness, weights, has_weights, all_observed,
                          log_pkfki):
    N, D = Xf.shape
    K = centers.shape[0]
    log_pk = np.empty(K)
    log_one_minus_mu_eff = np.zeros((K, D))
    log_diff = np.zeros((K, D))
    for k in range(K):
        log_pk[k] = math.log(max(proportions[k], _PROB_FLOOR))
        for d in range(D):
            v = dispersions[k, d]
            safe_v = 0.5 if v <= _ZERO_DISP_TOL else v
            mu = (1.0 - safe_v) if centers[k, d] == 1.0 else safe_v
            mu_eff = mu * completeness[d]
            if mu_eff < _PROB_FLOOR:
                mu_eff = _PROB_FLOOR
            elif mu_eff > 1.0 - _PROB_FLOOR:
                mu_eff = 1.0 - _PROB_FLOOR
            lome = math.log(1.0 - mu_eff)
            log_one_minus_mu_eff[k, d] = lome
            log_diff[k, d] = math.log(mu_eff) - lome

    for i in prange(N):
        for k in range(K):
            log_fki = 0.0
            for d in range(D):
                if not (all_observed or observed[i, d]):
                    continue
                w = weights[d] if has_weights else 1.0
                log_fki += (log_one_minus_mu_eff[k, d]
                            + Xf[i, d] * log_diff[k, d]) * w
            log_pkfki[i, k] = log_pk[k] + log_fki


@njit(cache=True, parallel=True)
def inertia_accum_byclass(Xf, observed, C, centers, use_abs, all_observed, Iner_KD):
    """Inertia parallelised over classes: one thread per k, no reduction buffer.

    Kept for the case where a per-chunk buffer would be too large to justify --
    which is exactly when K is big, and therefore also when this form has
    plenty of parallelism of its own.
    """
    N, D = Xf.shape
    K = centers.shape[0]
    for k in prange(K):
        for i in range(N):
            c = C[i, k]
            if c == 0.0:
                continue
            for d in range(D):
                if not (all_observed or observed[i, d]):
                    continue
                diff = Xf[i, d] - centers[k, d]
                if use_abs:
                    Iner_KD[k, d] += c * abs(diff)
                else:
                    Iner_KD[k, d] += c * diff * diff


@njit(cache=True, parallel=True)
def inertia_accum(Xf, observed, C, centers, use_abs, all_observed, nchunk, Iner_KD):
    """Inertia parallelised over node chunks, with the k loop innermost.

    The win over the by-class form is memory traffic, not thread count: that
    one sweeps the whole of Xf once per class, this one reads a row once and
    reuses it across all K. Measured faster at every K from 2 to 36 on a
    10000 x 2000 problem (0.3x to 0.7x), including at K well above the thread
    count where the by-class form is not thread-starved at all.

    nchunk is fixed by the problem shape by _inertia_chunks, never by the
    thread count, so the summation order and hence the result do not change
    with NUMBA_NUM_THREADS.

    The price is that each variable's total is a sum of per-chunk partial sums
    rather than one sweep over i, so it is not bit-identical to either the
    by-class form or the pure-numpy fallback.
    """
    N, D = Xf.shape
    K = centers.shape[0]
    buf = np.zeros((nchunk, K, D))

    for t in prange(nchunk):
        lo = N * t // nchunk
        hi = N * (t + 1) // nchunk
        for i in range(lo, hi):
            for k in range(K):
                c = C[i, k]
                if c == 0.0:
                    continue
                for d in range(D):
                    if not (all_observed or observed[i, d]):
                        continue
                    diff = Xf[i, d] - centers[k, d]
                    if use_abs:
                        buf[t, k, d] += c * abs(diff)
                    else:
                        buf[t, k, d] += c * diff * diff

    for t in range(nchunk):
        for k in range(K):
            for d in range(D):
                Iner_KD[k, d] += buf[t, k, d]


# The per-chunk buffer is nchunk * K * D float64. Past this it is not worth the
# memory: K is large in that regime, so the by-class kernel parallelises fine.
_INERTIA_BUF_CAP = 64 * 1024 * 1024


def _inertia_chunks(N, K, D):
    """How many node chunks to accumulate into, or 0 to use the by-class form.

    Deliberately a function of the problem shape alone. Deriving it from the
    thread count instead would make the summation order, and so the result,
    depend on NUMBA_NUM_THREADS.
    """
    nc = min(64, max(1, N // 256))
    while nc > 1 and nc * K * D * 8 > _INERTIA_BUF_CAP:
        nc //= 2
    return 0 if nc == 1 else nc
