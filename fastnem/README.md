# fastnem

A SIMD-accelerated and parallel C++ implementation of NEM (ppanggolin-related features only).

# `pynem` vs `fastnem`

## Phase: Fit loop
pynem: core.py: NEM.fit() / NEM._run_once()
fastnem: engine.hpp: fit() / detail::fit_from_classification()

## Phase: Initialization
pynem: core.py: NEM._initialize(), NEM._first_m_step()
fastnem: engine.hpp: init section of fit()

## Phase: M-step: class-density
pynem: models.py: compute_log_density(), numba kernels in _fast.py: density_bernoulli() /
density_bernoulli_mag()
fastnem: family.hpp: Bernoulli<T>::density() → density_standard() / density_mag_aware()

## Phase: M-step: estimate
pynem: models.py: estimate_parameters(), _estimate_bernoulli_centers(),
_estimate_bernoulli_completeness(), _inertia_to_dispersions(), numba kernel inertia_accum()
in _fast.py
fastnem: family.hpp: Bernoulli<T>::estimate() → estimate_standard() / estimate_mag_aware(),
using accumulate_observed_sums(), accumulate_abs_inertia(), inertia_to_dispersions()

## Phase: M-step: empty-class handling
pynem: models.py: _reinit_empty_classes()
fastnem: family.hpp: reinit_empty_classes()

## Phase: E-step: neighbor sum ("context")
pynem: spatial.py: NeighborhoodSystem.spatial_context() / compute_all_contexts()
fastnem: kernel.hpp: Kernel<T>::compute_all_contexts() / compute_context_row()

## Phase: E-step: Gauss-Seidel
pynem: core.py: NEM._e_step(), seq_sweep() in _fast.py
fastnem: kernel.hpp: Kernel<T>::e_step_seq()

## Phase: Criteria
pynem: core.py: NEM._compute_criteria()
fastnem: kernel.hpp: Kernel<T>::compute_criteria()

## Phase: Convergence
pynem: core.py: NEM._has_converged()
fastnem: convergence.hpp: has_converged()

# Benchmark and profiling

## Setup

```bash
pip install ./pynem[profile]
pip install ./fastnem/python/pyfastnem[dev]
```

## Test data

`nem/fastnem/benchs/data`

## `pynem` profiling

```bash
cd pynem/benchmarks
python ppanggolin_profile.py --data-dir ../../fastnem/benchs/data/ --k 3 --sm-degree 10
```

## `pynem` vs `pyfastnem`

```bash
cd fastnem/benchs
python run.py --data-dirs data/ --k 3 --sm-degree 10
```

### fastnem + pynem first optim pass

```
implementation    time (s)  speedup  mem (MiB)  mem ratio  agreement  n_iter
----------------------------------------------------------------------------
pynem (baseline)   328.707       --        993         --         --      80
pynem (opti)        83.633     3.9x        658       1.5x     100.0%      80
fastnem f64 @1t     39.117     8.4x        237       4.2x     100.0%      80
fastnem f64 @2t     26.039    12.6x        238       4.2x     100.0%      80
fastnem f64 @4t     18.566    17.7x        238       4.2x     100.0%      80
fastnem f64 @8t     14.605    22.5x        237       4.2x     100.0%      80
fastnem f64 @16t    14.507    22.7x        238       4.2x     100.0%      80
fastnem f32 @1t     50.802     6.5x        121       8.2x     100.0%     100
fastnem f32 @2t     30.137    10.9x        121       8.2x     100.0%     100
fastnem f32 @4t     20.506    16.0x        121       8.2x     100.0%     100
fastnem f32 @8t     20.065    16.4x        121       8.2x     100.0%     100
fastnem f32 @16t    22.147    14.8x        121       8.2x     100.0%     100
```

### fastnem + pynem second opti pass

- Improve fastnem parallel scaling
- fastnem is called through pyfastnem bindings (the memory overhead)
- add // to pynem numba kernels

```
implementation        time (s)  speedup  mem (MiB)  mem ratio  agreement  n_iter
--------------------------------------------------------------------------------
pynem_old (baseline)   313.419       --       1091         --         --      80
pynem                   61.234     5.1x        646       1.7x     100.0%      80
pyfastnem f64 @1t       44.117     7.1x        351       3.1x     100.0%      80
pyfastnem f64 @2t       25.550    12.3x        351       3.1x     100.0%      80
pyfastnem f64 @4t       15.593    20.1x        352       3.1x     100.0%      80
pyfastnem f64 @8t       11.618    27.0x        353       3.1x     100.0%      80
pyfastnem f64 @16t      10.602    29.6x        355       3.1x     100.0%      80
pyfastnem f32 @1t       54.189     5.8x        234       4.7x     100.0%     100
pyfastnem f32 @2t       30.305    10.3x        234       4.7x     100.0%     100
pyfastnem f32 @4t       18.322    17.1x        234       4.7x     100.0%     100
pyfastnem f32 @8t       17.996    17.4x        235       4.6x     100.0%     100
pyfastnem f32 @16t      20.317    15.4x        235       4.6x     100.0%     100
```
