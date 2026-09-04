"""Distribution families and parameter estimation for NEM."""

from enum import Enum

import numpy as np

from ._fast import (
    HAS_NUMBA,
    _inertia_chunks,
    inertia_accum_byclass,
    density_bernoulli,
    density_bernoulli_mag,
    density_laplace,
    density_normal,
    inertia_accum,
)

# Three semantically distinct floors that used to share a single ``EPSILON``.
# They are all kept at 1e-20 so the numerics are byte-for-byte unchanged (the
# PPanGGOLiN reproduction and the Gaussian examples are preserved exactly); the
# split only makes the *intent* explicit and gives a single place to retune one
# without disturbing the others.
PROB_FLOOR = 1e-20      # floor for probabilities/proportions before a log
VAR_FLOOR = 1e-20       # floor for dispersions/variances (kept tiny to match
                        # the reference C core, where a degenerate dimension is
                        # handled by ZERO_DISP_TOL below, not by the floor)
ZERO_DISP_TOL = 1e-20   # a dispersion <= this marks a degenerate (point-mass)
                        # dimension: it contributes no density term, and an
                        # observation off the mode there gets zero likelihood
DIV_GUARD = 1e-20       # generic guard against division by zero (counts, sums)

# Back-compat alias: external code (and older imports) may still reference
# EPSILON. It maps to the generic division guard.
EPSILON = DIV_GUARD

# A class whose total (soft) membership falls below this is considered empty
# and reinitialised (k-means++ style) by _reinit_empty_classes.
EMPTY_CLASS_WEIGHT = 1.0


class Family(Enum):
    NORMAL = "normal"
    LAPLACE = "laplace"
    BERNOULLI = "bernoulli"


class Dispersion(Enum):
    S__ = "s__"   # single dispersion for all
    SK_ = "sk_"   # per-class
    S_D = "s_d"   # per-variable
    SKD = "skd"   # full (per-class, per-variable)


class Proportion(Enum):
    EQUAL = "p_"  # equal proportions = 1/K
    FREE = "pk"   # free proportions


def compute_log_density(X, centers, dispersions, proportions, family,
                        weights=None, completeness=None, observed=None, Xf=None,
                        all_observed=None, binary=None):
    """Compute log(p_k * f_k(x_i)) for all i and k.

    Parameters
    ----------
    X : (N, D) array
    centers : (K, D) array
    dispersions : (K, D) array
    proportions : (K,) array
    family : Family enum
    weights : (D,) array or None
        Per-variable weights ``w_j > 0`` for the *weighted* NEM. Each variable's
        log-density contribution is multiplied by ``w_j`` before summing over D
        (a variable with ``w_j`` acts as if replicated ``w_j`` times). ``None``
        means all weights equal 1, i.e. the standard unweighted density —
        recovered byte-for-byte (the multiply is skipped entirely).
    completeness : (D,) array or None
        Per-variable (per-genome) completeness ``gamma_j in (0, 1]`` for the
        **MAG-aware Bernoulli** emission (mOTUpan-style). A present feature is
        only *detected* with probability ``gamma_j``, so the presence prob
        becomes ``mu_kj * gamma_j`` and an absence in an incomplete genome is
        forgiven. ``None`` (default) = full completeness, exact standard density.
        Only affects the Bernoulli family.
    observed, Xf : (N, D) arrays or None
        Precomputed ``~np.isnan(X)`` / ``np.where(observed, X, 0.0)``. X does
        not change across the iterations of a single fit, so callers that
        already have these (NEM._run_once) can pass them in to skip
        recomputing/reallocating a full (N, D) array on every call.
        ``None`` (default) recomputes them here

    Returns
    -------
    log_pkfki : (N, K) array
        log(p_k * f_k(x_i)) for each observation i and class k.
    """
    if observed is None:
        observed = ~np.isnan(X)            # (N, D)
    if Xf is None:
        Xf = np.where(observed, X, 0.0)    # NaN -> 0 (these terms are masked out)
    if all_observed is None:
        all_observed = bool(observed.all())
    if binary is None:
        binary = all_observed and bool(np.all((Xf == 0.0) | (Xf == 1.0)))

    # Bernoulli with binary centres collapses to one matmul; see
    # _bernoulli_closed_form. Everything it needs is settled for a whole fit
    # except the centres, which are binary by construction for this family.
    if (binary and family == Family.BERNOULLI and completeness is None
            and np.all((centers == 0.0) | (centers == 1.0))):
        return _bernoulli_closed_form(Xf, centers, dispersions, proportions,
                                      weights, dispersions <= ZERO_DISP_TOL)

    if HAS_NUMBA:
        return _compute_log_density_numba(Xf, observed, centers, dispersions,
                                          proportions, family, weights, completeness,
                                          all_observed)
    return _compute_log_density_numpy(Xf, observed, centers, dispersions,
                                      proportions, family, weights, completeness)


def _bernoulli_closed_form(Xf, centers, dispersions, proportions, weights,
                           degen):
    """log(p_k f_k(x_i)) for the Bernoulli family, as a single matmul.

    Valid only for binary data and binary centres, which is what the Bernoulli
    family means and what _estimate_bernoulli_centers always returns. Then

        |x_id - c_kd| = x_id + c_kd - 2 x_id c_kd

    so the per-class sum splits into a part that does not depend on i and a
    dot product against the observation:

        sum_d w_d lr_kd |x_id - c_kd| = A_k + sum_d x_id * g_kd
        A_k  = sum_d w_d lr_kd c_kd
        g_kd = w_d lr_kd (1 - 2 c_kd)

    which turns the triple-nested loop over (i, k, d) into (N, D) @ (D, K).
    On 15931 x 2002 that is 94 ms of hand-rolled kernel against 24 ms of BLAS,
    and it is what makes a sparse X exploitable at all -- the data is 8-16 %
    dense, so the loop was visiting an order of magnitude more cells than
    carry any information.

    Degenerate variables (dispersion at or below the floor) contribute no
    density term but still force a zero likelihood where the observation
    differs from the centre; they are handled separately below, and there were
    none at all in a converged staph or ecoli fit.
    """
    K, D = centers.shape
    a = np.zeros((K, D))
    lr = np.zeros((K, D))
    ok = ~degen
    a[ok] = np.log(1.0 - dispersions[ok])
    lr[ok] = np.log((1.0 - dispersions[ok]) / dispersions[ok])
    if weights is not None:
        a = a * weights[None, :]
        lr = lr * weights[None, :]

    base = (np.log(np.maximum(proportions, PROB_FLOOR))
            + a.sum(axis=1) - (lr * centers).sum(axis=1))
    g = np.ascontiguousarray((lr * (1.0 - 2.0 * centers)).T)   # (D, K)
    log_pkfki = base[None, :] - Xf @ g

    if degen.any():
        # a zero-dispersion variable admits only its own centre value
        for k in range(K):
            dd = np.flatnonzero(degen[k])
            if weights is not None:
                dd = dd[weights[dd] > 0.0]
            if dd.size:
                bad = (Xf[:, dd] != centers[k, dd]).any(axis=1)
                log_pkfki[bad, k] = -np.inf
    return log_pkfki


def _compute_log_density_numba(Xf, observed, centers, dispersions, proportions,
                               family, weights, completeness, all_observed):
    N, D = Xf.shape
    K = centers.shape[0]
    log_pkfki = np.empty((N, K))
    has_weights = weights is not None
    w = np.ascontiguousarray(weights, dtype=np.float64) if has_weights else np.empty(0)
    centers_c = np.ascontiguousarray(centers, dtype=np.float64)
    dispersions_c = np.ascontiguousarray(dispersions, dtype=np.float64)
    proportions_c = np.ascontiguousarray(proportions, dtype=np.float64)
    Xf_c = np.ascontiguousarray(Xf, dtype=np.float64)
    observed_c = np.ascontiguousarray(observed)

    if family == Family.NORMAL:
        density_normal(Xf_c, observed_c, centers_c, dispersions_c, proportions_c,
                       w, has_weights, all_observed, log_pkfki)
    elif family == Family.LAPLACE:
        density_laplace(Xf_c, observed_c, centers_c, dispersions_c, proportions_c,
                        w, has_weights, all_observed, log_pkfki)
    elif completeness is None:
        density_bernoulli(Xf_c, observed_c, centers_c, dispersions_c, proportions_c,
                          w, has_weights, all_observed, log_pkfki)
    else:
        gamma = np.ascontiguousarray(completeness, dtype=np.float64)
        density_bernoulli_mag(Xf_c, observed_c, centers_c, dispersions_c, proportions_c,
                              gamma, w, has_weights, all_observed, log_pkfki)
    return log_pkfki


def _compute_log_density_numpy(Xf, observed, centers, dispersions, proportions,
                               family, weights, completeness):
    N, D = Xf.shape
    K = centers.shape[0]
    log_pkfki = np.empty((N, K))

    for k in range(K):
        log_pk = np.log(max(proportions[k], PROB_FLOOR))
        v = dispersions[k]                       # (D,)
        zero_v = v <= ZERO_DISP_TOL              # (D,)
        diff = Xf - centers[k]                   # (N, D)
        absdiff = np.abs(diff)

        if family == Family.NORMAL:
            # -0.5 [ log(2π v) + (x - m)² / v ]
            safe_v = np.where(zero_v, 1.0, v)
            contrib = -0.5 * (np.log(2 * np.pi * safe_v) + diff * diff / safe_v)
        elif family == Family.LAPLACE:
            # -[ log(2 v) + |x - m| / v ]
            safe_v = np.where(zero_v, 1.0, v)
            contrib = -(np.log(2 * safe_v) + absdiff / safe_v)
        else:  # BERNOULLI: -[ -log(1 - v) + |x - m| log((1 - v) / v) ]
            safe_v = np.where(zero_v, 0.5, v)
            if completeness is None:
                contrib = np.log(1 - safe_v) - absdiff * np.log((1 - safe_v) / safe_v)
            else:
                # MAG-aware emission: presence prob mu_kj detected with
                # completeness gamma_j -> P(x=1)=mu*gamma, P(x=0)=1-mu*gamma.
                # mu = 1-eps if mode 1, eps if mode 0 (centers[k] is the 0/1 mode).
                mu = np.where(centers[k] == 1, 1 - safe_v, safe_v)          # (D,)
                mu_eff = np.clip(mu * completeness, PROB_FLOOR, 1 - PROB_FLOOR)
                contrib = (Xf * np.log(mu_eff)[None, :]
                           + (1 - Xf) * np.log(1 - mu_eff)[None, :])

        comp_bern = family == Family.BERNOULLI and completeness is not None
        if not comp_bern:
            contrib = np.where(zero_v[None, :], 0.0, contrib)  # zero-disp dims: no term
        contrib = np.where(observed, contrib, 0.0)          # unobserved dims: no term
        if weights is not None:
            contrib = contrib * weights[None, :]            # weighted NEM: w_j per variable
        log_fki = contrib.sum(axis=1)                       # (N,)

        # A zero-dispersion (point-mass) dimension gives -inf density to an
        # off-mode observation -- UNLESS that variable is weighted out (w_j = 0),
        # in which case it must be ignored entirely (the note's w_j=0 = variable
        # selection limit). weights=None -> all variables active -> exact old
        # behaviour, so the PPanGGOLiN reproduction is unaffected. With the
        # MAG-aware emission mu*gamma is always finite, so nothing is invalid.
        if comp_bern:
            invalid = np.zeros(N, dtype=bool)
        else:
            active = (np.ones(D, dtype=bool) if weights is None
                      else weights > 0)
            invalid = (observed & zero_v[None, :] & (absdiff > ZERO_DISP_TOL)
                       & active[None, :]).any(axis=1)
        log_pkfki[:, k] = np.where(invalid, -np.inf, log_pk + log_fki)

    return log_pkfki


def estimate_parameters(X, C, family, dispersion_model, proportion_model,
                        miss_mode="replace", old_centers=None,
                        old_dispersions=None, weights=None, completeness=None,
                        observed=None, Xf=None, all_observed=None):
    """M-step: estimate model parameters from soft classification.

    Parameters
    ----------
    X : (N, D) array
    C : (N, K) array — soft classification
    family : Family enum
    dispersion_model : Dispersion enum
    proportion_model : Proportion enum
    miss_mode : str — 'replace' or 'ignore'
    old_centers : (K, D) array or None
    old_dispersions : (K, D) array or None
    weights : (D,) array or None
        Per-variable weights for the weighted NEM. Centres and modes are
        **unchanged** by ``w_j`` (a per-column constant factors out of the
        weighted-likelihood normal equations, and out of the weighted median),
        so they are estimated as usual. The weights only matter for dispersion
        models that **pool across variables** (``s__`` and ``sk_``): there each
        column's inertia and count are weighted by ``w_j``. ``None`` = all ones.
    completeness : (D,) array or None
        Per-genome completeness ``gamma_j`` for the MAG-aware Bernoulli model.
        The presence rate is de-biased by ``gamma_j``: the mode is 1 iff the
        observed presence fraction exceeds ``0.5 * gamma_j`` (so a gene seen in
        only ``gamma_j`` of an incomplete genome's class still counts as
        present), and the dispersion follows from ``mu = frac / gamma``. Only
        used for the Bernoulli family. ``None`` = full completeness (standard).
    observed, Xf : (N, D) arrays or None
        Precomputed ``~np.isnan(X)`` / ``np.where(observed, X, 0.0)`` -- see
        ``compute_log_density``. ``None`` (default) recomputes them here.
    all_observed : bool or None
        Whether ``observed`` is entirely True. Like the two above it is fixed
        for a whole fit, so callers that know it should pass it; ``None``
        (default) derives it here with a full pass over ``observed``.

    Returns
    -------
    dict with keys 'centers', 'dispersions', 'proportions'.
    """
    N, D = X.shape
    K = C.shape[1]
    if observed is None:
        observed = ~np.isnan(X)  # (N, D)
    if Xf is None:
        Xf = np.where(observed, X, 0.0)
    if all_observed is None:
        all_observed = bool(observed.all())
    comp_bern = family == Family.BERNOULLI and completeness is not None

    # Class sizes
    raw_N_K = C.sum(axis=0)            # (K,) before clamping, for empty detection
    N_K = np.maximum(raw_N_K, DIV_GUARD)

    # Per-class, per-variable observed sizes. With nothing missing every column
    # sees the same weight, so N_KD is just raw_N_K broadcast over D -- and the
    # loop below is three full (N, D) float64 temporaries computing a value we
    # already have. That is the path every PPanGGOLiN run takes: presence is a
    # dense 0/1 matrix. On a 15931 x 2002 pangenome the loop costs 276 ms per
    # M-step against 0.1 ms for the broadcast, and it ran on every iteration.
    if all_observed:
        N_KD = np.repeat(raw_N_K[:, None], D, axis=1)
    else:
        N_KD = np.zeros((K, D))
        for k in range(K):
            N_KD[k] = (C[:, k:k+1] * observed).sum(axis=0)
    N_KD = np.maximum(N_KD, DIV_GUARD)

    # --- Centers ---
    # Bernoulli, like Laplace, uses the weighted MEDIAN as center estimator
    # (the reference NEM C code routes FAMILY_BERNOULLI to EstimParaLaplace).
    # For 0/1 data the weighted median is the binary MODE, which equals
    # thresholding the C-weighted fraction of 1s at 0.5 — computed here as two
    # matmuls, with no per-(k,d) sort (see _estimate_bernoulli_centers).
    if comp_bern:
        # MAG-aware: de-bias presence by completeness; sets mode AND dispersion.
        centers, dispersions = _estimate_bernoulli_completeness(
            C, observed, completeness, Xf, all_observed=all_observed)
    elif family == Family.BERNOULLI:
        centers = _estimate_bernoulli_centers(C, observed, Xf,
                                              all_observed=all_observed)
    elif family == Family.LAPLACE:
        centers = _estimate_laplace_centers(X, C, observed, K, D, N_K)
    else:
        centers = _estimate_mean_centers(C, observed, K, D, N, N_K, N_KD,
                                         miss_mode, old_centers, Xf)

    if not comp_bern:
        use_abs = family != Family.NORMAL
        Iner_KD = np.zeros((K, D))
        if HAS_NUMBA:
            _Xf = np.ascontiguousarray(Xf, dtype=np.float64)
            _ob = np.ascontiguousarray(observed)
            _C = np.ascontiguousarray(C, dtype=np.float64)
            _cen = np.ascontiguousarray(centers, dtype=np.float64)
            _nc = _inertia_chunks(N, K, D)
            if _nc:
                inertia_accum(_Xf, _ob, _C, _cen, use_abs, all_observed,
                              _nc, Iner_KD)
            else:
                inertia_accum_byclass(_Xf, _ob, _C, _cen, use_abs,
                                      all_observed, Iner_KD)
        else:
            unobserved = ~observed
            for k in range(K):
                diff = Xf - centers[k]
                diff[unobserved] = 0.0
                if use_abs:
                    Iner_KD[k] = (C[:, k:k+1] * np.abs(diff) * observed).sum(axis=0)
                else:
                    Iner_KD[k] = (C[:, k:k+1] * diff ** 2 * observed).sum(axis=0)

        if miss_mode == "replace" and family == Family.NORMAL and old_dispersions is not None:
            for k in range(K):
                n_miss_kd = N_K[k] - N_KD[k]
                Iner_KD[k] += np.maximum(n_miss_kd, 0) * old_dispersions[k]

        # --- Dispersions ---
        dispersions = _inertia_to_dispersions(
            Iner_KD, N_K, N_KD, N, D, K, dispersion_model, miss_mode, weights
        )

    # Clamp dispersions from below
    dispersions = np.maximum(dispersions, VAR_FLOOR)

    # --- Proportions ---
    if proportion_model == Proportion.EQUAL:
        proportions = np.full(K, 1.0 / K)
    else:
        proportions = N_K / N
        proportions = np.maximum(proportions, PROB_FLOOR)
        proportions /= proportions.sum()

    # --- Recover collapsed (empty) classes -------------------------------
    centers, dispersions, proportions = _reinit_empty_classes(
        X, centers, dispersions, proportions, raw_N_K, observed, Xf
    )

    return {
        "centers": centers,
        "dispersions": dispersions,
        "proportions": proportions,
    }


def _reinit_empty_classes(X, centers, dispersions, proportions, raw_N_K,
                          observed, Xf):
    """k-means++ style recovery for classes that have collapsed (emptied).

    A class whose total membership falls below ``EMPTY_CLASS_WEIGHT`` has its
    centre moved to the observation farthest from the dominant (largest) class
    centre — excluding points already chosen this round — its dispersion reset
    to the average dispersion of the populated classes, and its proportion
    floored so it can attract that point on the next E-step. The choice is
    deterministic (argmax distance, no RNG), so runs where no class is empty are
    left byte-for-byte unchanged (and PPanGGOLiN reproduction is preserved).
    """
    empty = np.where(raw_N_K < EMPTY_CLASS_WEIGHT)[0]
    if empty.size == 0:
        return centers, dispersions, proportions

    N = X.shape[0]
    populated = raw_N_K >= EMPTY_CLASS_WEIGHT
    if populated.any():
        default_disp = np.maximum(dispersions[populated].mean(axis=0), VAR_FLOOR)
    else:  # pathological: every class empty -> fall back to sample dispersion
        default_disp = np.maximum(np.nanvar(X, axis=0), VAR_FLOOR)
    dominant = int(np.argmax(raw_N_K))

    centers = centers.copy()
    dispersions = dispersions.copy()
    proportions = proportions.copy()

    # squared distance to the dominant centre, over observed variables
    d2 = ((Xf - centers[dominant]) ** 2 * observed).sum(axis=1)
    far_order = np.argsort(d2)[::-1]   # farthest observations first
    used = set()
    for k in empty:
        far = next((int(i) for i in far_order if i not in used),
                   int(far_order[0]))
        used.add(far)
        centers[k] = Xf[far]
        dispersions[k] = default_disp
        proportions[k] = max(proportions[k], 1.0 / N)

    proportions = proportions / proportions.sum()
    return centers, dispersions, proportions


def _estimate_mean_centers(C, observed, K, D, N, N_K, N_KD, miss_mode,
                           old_centers, Xf):
    """Weighted mean center estimation (Normal, Bernoulli)."""
    centers = np.zeros((K, D))

    for k in range(K):
        weighted_sum = (C[:, k:k+1] * Xf * observed).sum(axis=0)
        if miss_mode == "replace" and old_centers is not None:
            n_miss_kd = N_K[k] - N_KD[k]
            weighted_sum += np.maximum(n_miss_kd, 0) * old_centers[k]
            centers[k] = weighted_sum / N_K[k]
        else:
            centers[k] = weighted_sum / N_KD[k]

    return centers


def _weight_totals(C, observed, all_observed):
    """Per-class, per-variable total weight over observed entries.

    With nothing missing every column sums the same weights, so the (K, D)
    result is ``C.sum(axis=0)`` in every column -- returned as (K, 1) to
    broadcast, which skips both a full (N, D) float cast and the matmul that
    consumed it (49 ms per M-step on 15931 x 2002).

    Not bit-identical to the matmul it replaces: BLAS accumulates in blocks
    and numpy pairwise, so the totals differ by up to ~32 ULP. Callers use
    this only as the denominator of a fraction thresholded at 0.5, which is
    unaffected unless that fraction sits within ~1e-11 of the tie the code
    already breaks arbitrarily.
    """
    if all_observed:
        return C.sum(axis=0)[:, None]
    return C.T @ observed.astype(float)


def _estimate_bernoulli_centers(C, observed, Xf, all_observed=False):
    """Bernoulli center = binary mode (weighted median of {0,1}), vectorised.

    For 0/1 data the C-weighted median equals 1 iff the weighted fraction of 1s
    exceeds 0.5, which is exactly what the weighted median returns (ties at 0.5
    map to 0, as in ``_estimate_laplace_centers``). Computed with two matmuls
    instead of a per-(class, variable) sort.

    Returns
    -------
    centers : (K, D) array of {0.0, 1.0}
    """
    W_total = _weight_totals(C, observed, all_observed)   # (K, D) or (K, 1)
    W_ones = C.T @ Xf          # (K, D) weighted count of 1s
    with np.errstate(invalid="ignore", divide="ignore"):
        frac1 = np.where(W_total > 0, W_ones / np.maximum(W_total, DIV_GUARD), 0.0)
    return (frac1 > 0.5).astype(float)


def _estimate_bernoulli_completeness(C, observed, completeness, Xf,
                                     all_observed=False):
    """MAG-aware Bernoulli mode + dispersion, de-biased by genome completeness.

    The observed presence fraction ``frac`` of a class in genome ``j`` is
    deflated by incompleteness; dividing by ``gamma_j`` recovers the true
    presence rate ``mu = frac / gamma_j``. The mode is 1 iff ``mu > 0.5`` (i.e.
    ``frac > 0.5 * gamma_j``), and the dispersion is the distance of ``mu`` to
    its mode (``eps = 1-mu`` if mode 1, ``mu`` if mode 0). With ``gamma_j = 1``
    this reduces to the standard binary mode and the ``skd`` dispersion.

    Returns
    -------
    centers : (K, D) array of {0.0, 1.0}
    dispersions : (K, D) array — epsilon in [VAR_FLOOR, 0.5]
    """
    W_total = _weight_totals(C, observed, all_observed)
    W_ones = C.T @ Xf
    with np.errstate(invalid="ignore", divide="ignore"):
        frac1 = np.where(W_total > 0, W_ones / np.maximum(W_total, DIV_GUARD), 0.0)
    gamma = np.clip(np.asarray(completeness, dtype=float), VAR_FLOOR, 1.0)
    mu = np.clip(frac1 / gamma[None, :], 0.0, 1.0)        # de-biased presence prob
    centers = (mu > 0.5).astype(float)
    eps = np.where(centers == 1, 1.0 - mu, mu)
    dispersions = np.clip(eps, VAR_FLOOR, 0.5)
    return centers, dispersions


def _estimate_laplace_centers(X, C, observed, K, D, N_K):
    """Weighted median center estimation for Laplace family."""
    centers = np.zeros((K, D))
    for k in range(K):
        weights = C[:, k]
        for d in range(D):
            obs_mask = observed[:, d]
            if obs_mask.sum() == 0:
                centers[k, d] = 0.0
                continue
            vals = X[obs_mask, d]
            w = weights[obs_mask]
            # Weighted median
            idx = np.argsort(vals)
            vals_sorted = vals[idx]
            w_sorted = w[idx]
            cumw = np.cumsum(w_sorted)
            half = cumw[-1] / 2.0
            median_idx = np.searchsorted(cumw, half)
            median_idx = min(median_idx, len(vals_sorted) - 1)
            centers[k, d] = vals_sorted[median_idx]
    return centers


def _inertia_to_dispersions(Iner_KD, N_K, N_KD, N, D, K, model, miss_mode,
                            weights=None):
    """Convert inertia matrix to dispersions according to model.

    For the variable-pooling models (``s__``, ``sk_``) the sum over variables is
    weighted by ``weights`` (per-variable ``w_j``): the pooled variance maximises
    the weighted complete-data likelihood, giving ``sum_j w_j·Iner / sum_j w_j·N``
    (the column weight ``w_j`` cancels in the per-variable models ``skd``/``s_d``,
    which are therefore untouched). ``weights=None`` ⇒ all ones, recovering the
    original pooled estimators exactly (``sum_j w_j = D``).
    """
    dispersions = np.zeros((K, D))
    # Per-variable weights; w.sum() replaces the plain "D" column count in the
    # replace-mode denominators when pooling across variables.
    w = np.ones(D) if weights is None else np.asarray(weights, dtype=float)
    w_sum = w.sum()

    if model == Dispersion.S__:
        if miss_mode == "replace":
            v = (w * Iner_KD).sum() / (N * w_sum)
        else:
            v = (w * Iner_KD).sum() / (w * N_KD).sum()
        dispersions[:] = v

    elif model == Dispersion.SK_:
        for k in range(K):
            if miss_mode == "replace":
                vk = (w * Iner_KD[k]).sum() / (w_sum * N_K[k])
            else:
                vk = (w * Iner_KD[k]).sum() / (w * N_KD[k]).sum()
            dispersions[k, :] = vk

    elif model == Dispersion.S_D:
        for d in range(D):
            if miss_mode == "replace":
                vd = Iner_KD[:, d].sum() / N
            else:
                vd = Iner_KD[:, d].sum() / N_KD[:, d].sum()
            dispersions[:, d] = vd

    elif model == Dispersion.SKD:
        if miss_mode == "replace":
            for k in range(K):
                dispersions[k] = Iner_KD[k] / N_K[k]
        else:
            dispersions = Iner_KD / N_KD

    return dispersions
