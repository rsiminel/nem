# fastnem

A SIMD-accelerated and parallel C++ implementation of NEM (ppanggolin-related features only).

```
implementation    time (s)  speedup  mem (MiB)  mem ratio  agreement  n_iter
----------------------------------------------------------------------------
pynem (baseline)   328.707       --        993         --         --      80
fastpynem           83.633     3.9x        658       1.5x     100.0%      80
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
