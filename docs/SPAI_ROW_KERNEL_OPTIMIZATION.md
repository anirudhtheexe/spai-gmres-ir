# SPAI residual optimization and row-parallel experiment (24 October 2026)

## Baseline and intended change

Carson and Khan's [SPAI-GMRES-IR full text](https://arxiv.org/pdf/2202.10204)
uses QR least-squares fitting and a fitted residual norm to decide each adaptive
column's growth; the [BSPAI-GMRES-IR full text](https://arxiv.org/pdf/2307.03914)
retains that SPAI construction context while studying adaptive precision in
the preconditioner. The row scheduling and dense cache investigated below were
this project's proposed optimization, not a result or claimed novelty from
those papers. This project still uses uniform FP32/FP64 SPAI, not BSPAI.

The saved post-batching FP32 profile attributes 39.7% of summed GPU kernel time
to `fitted_residual` and 23.9% to `qr_guard`. These are historical kernel shares,
not predicted solver speedups. Both launch one thread. The residual kernel
recomputes every row product for its full and reduced diagnostics. The QR guard
and residual kernel repeatedly search sparse B after cuSOLVER overwrites the
dense reduced matrix.

The archived profile manifest used `stencil_8192.mtx` (SHA-256
`90d25d6332cc3729030c13e89deb74169b13aef8c1fe89b1ffa4af6709f09b1f`),
uniform FP32, `epsilon=0.3`, `alpha=1024`, `beta=8`, one warmup and one traced
trial, with 32,514 calls to each named kernel. Its own purpose field calls it
diagnostic constructor observation rather than timing evidence.

The comparison preserved scaled B, Householder QR, triangular solve, score
rule, precision, stopping criteria, column order within row dots, and the
ascending-row safe scaled norm recurrence. It compared a dense cache with
row-parallel checks against the original path and a one-pass serial residual.

## Validation plan

Build both FP32 and FP64 CUDA specializations. Run the CUDA SPAI suites,
independent Decimal reconstruction and solver integration checks. Compare
deterministic constructor output and per-fit diagnostics to a frozen baseline
where available; investigate any difference without changing tolerance or
failure policy. Then use a reproducible saved-profile matrix or equivalent
input to measure kernel time and end-to-end constructor time with repeated
runs, separating overhead, memory growth, and speedup claims. Record hardware,
toolchain, commands, inputs, parameters, and limitations here.

## Decision and production implementation, 24 October 2026

The row-parallel experiment cached a second dense `m*q` matrix and launched
separate row and finish kernels for both guards. It gave bitwise identical
public FP32 results on the archived stencil input, but its two paired median
constructor times were 2426–2432 ms against 2282–2307 ms for the isolated
same-flags pre-change baseline. It raised source-issued peak device bytes from
1,600,024 to 1,602,520 and allocation calls from 406 to 496. Parallel
scheduling and caching were therefore not retained. Extra launches and buffer
churn appear to outweigh saved searches for this workload, but kernel durations
were not collected to isolate that mechanism.

The production `src/cuda_spai.cu` retains the original gather, QR guard,
buffers, and launches. Its one-thread `fitted_residual` now computes each row
dot once, checks and stores that value, and feeds it immediately to the
unchanged safe norm recurrence. The reduced and full diagnostics use that same
norm because the complete shadow covers the fitted column's support. The
isolated fused variant returned identical public FP32 bits and measured
2146.1, 2153.3, and 2247.0 ms (median 2153.3 ms) on the same stencil case.
This suggests about 5–7% lower constructor time in these small samples; it is
exploratory evidence for one matrix, not a general or whole-solver speedup.
Full reconstruction, output hashes, and raw samples are recorded in
`SPAI_ROW_KERNEL_VALIDATION.md`.

Build environment: Windows x64, MSVC 19.44.35228, CUDA 13.4.59, RTX 4060
Laptop architecture 89, CMake 3.31.6/Ninja. `scripts/build_windows.cmd cuda 89`
and the Release variant both failed during CMake's compiler probe with MSVC
`C1902 Program database manager mismatch` before project compilation. A local
build-tree configuration overriding the probe's Debug PDB flags succeeded:

```
cmake -S cuda -B build/cuda-NoPdb -G Ninja -DCMAKE_CXX_COMPILER=cl \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 \
  "-DCMAKE_CXX_FLAGS_DEBUG=/Od /MDd /Z7" \
  "-DCMAKE_CUDA_FLAGS_DEBUG=-Xcompiler=/Z7"
cmake --build build/cuda-NoPdb --parallel 4
ctest --test-dir build/cuda-NoPdb --output-on-failure
```

After selecting the fused-only production source, both specializations
compiled. CUDA CTest passed 8/8. The independent FP64
SPAI Decimal80 reconstruction passed 436/436 checks over 60 fits; CUDA IR
Decimal80 passed 160/160 checks; four solver CLIs passed 32/32 independent
backward-error/error-path checks. The sparse allocation audit passed 3,363,954
checks, including actual CUDA allocation reconciliation. The FP32 focused
SPAI integration test passed and recorded a final original-system backward
error of `1.98682145e-07` after one GPU correction. The bitwise stencil
comparison covers public FP32 data but not every internal score history. No
FP64 performance result was collected. See `SPAI_ROW_KERNEL_VALIDATION.md`
for those limits.
