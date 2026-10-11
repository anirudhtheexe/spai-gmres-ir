# Phase 2A CPU milestone 2: SPAI construction

Date: 11 October 2026. Implementation completed; bounded milestone tests pass. Scope ends at SPAI construction and assembly; no GMRES, IR, BSPAI, CUDA, or performance measurements.

## Intended method and validation plan

The binding [algorithm specification](PHASE2A_ALGORITHM_SPEC.md) reconciles Carson--Khan Algorithm 2.1 and the pinned MATLAB `spai_dd.m`. Construct on `B=A^T D`, where `d[k]=1/max_j abs(A[k,j])`; form the left preconditioner `P=N^T D`. Use sorted identity patterns, reduced Householder QR least-squares fits, restricted-column candidate scoring, deterministic score/index ordering, at most beta acceptable new indices, and alpha growth/refit events after the initial fit. Test residual <= epsilon after each solved pattern and retain diagnostics for every fit.

All stored values, products, sums, QR operations, and scores use double. There is no normal-equations fallback, pivoting, altered pattern, or tolerance relaxation. Guard failed input/scaling, empty candidates, zero score denominators, nonfinite arithmetic, exact zero QR pivots, and iteration limits with explicit statuses. Roundoff-negative scores clamp only inside the prescribed `32*2^-53*(r_squared+abs(quotient))` bound, with counted clamp events. Rank uses exactly zero pivots; finite near-zero pivots are reported through `min(abs(Rjj))/norm_inf(reduced_matrix)` rather than truncated. Overflow/underflow that changes a nonzero scaled coefficient to zero fails explicitly. These exception/rank policies are project choices confirmed by the Research Agent, not claims of identical MATLAB library behavior.

Independent Validation Agent tests will compare scaling/orientation and complete residuals with wider precision and exact rational tiny-matrix references, growth/cap statuses, tie order, score guard boundaries, and malformed/range-limited input. Existing loader/CSR tests must continue to pass. Full solver convergence and published mixed-precision results remain outside scope.

## Components and explicit diagnostics

Added `include/bspai/spai.hpp` and `src/spai.cpp`; updated `CMakeLists.txt` to build SPAI and the independently authored `tests/test_spai.cpp`. Loader source/API and prior tests are unchanged.

`construct_spai` exposes scaling, scaled transpose B, right inverse N, left preconditioner P, global status, and per-column histories. Histories record solved patterns/shadows, coefficients, both residual norms, restricted candidates/scores/mean, selected indices, and QR pivot ratios. Completed fits/growths and attempted fits/growths are distinct; a failed attempted pattern is retained separately from the last solved pattern. Unavailable fit residuals are NaN, not zero. Failed results retain already fitted coefficients for diagnostics only, and must not be used as successful preconditioners. Assembly-range failure may leave assembled matrices unavailable. Allocation exhaustion propagates a standard exception rather than a numerical fallback.

Square dimensions, finite positive epsilon and nonzero diagonal are checked. Alpha zero means initial fit only; beta zero yields no-growth failure when another fit is needed. Nonsingularity remains an input assumption: local QR pivot checks do not certify nonsingularity of A and do not reject every singular matrix at loose epsilon. No arbitrary near-singular rank threshold is introduced.

## Execution evidence

Final independent SPAI suite: **190 SPAI checks and 37 existing loader/CSR checks, zero failures; CTest 2/2 passed**. The suite expanded from the implementation's earlier 183 checks with exact-zero-QR/failed-refit accounting and an independently recomputed near-dependent full residual; no test failed and no tolerance changed. Initial direct GNU warning-enabled build passed the earlier 174-check suite without diagnostics; the Validation Agent separately records its direct-build evidence and independent review in [the validation record](PHASE2A_VALIDATION_PLAN.md). No acceptance tolerance was changed to obtain a pass.

Numerical evidence from the independently authored test executable:

| Case | Recorded result |
|---|---|
| Tiny 3x3, epsilon=1e-12, alpha=2, beta=8 | Success; nnz(N)=nnz(P)=9; every column has 3 completed fits/2 growths, no score clamps |
| Wider recomputed full tiny residual | Maximum 7.0536646688527192e-16 |
| Tiny P versus exact rational A inverse | Maximum absolute entry error 1.9317772208260475e-16 |
| Tiny alpha=1 cap | `iteration_limit`; column 0 residual 0.23076923076923078 |
| Tight singular 2x2 fixture | Explicit `empty_candidates` failure; no claim of global rank detection |
| Near-dependent 2x2 | Maximum inverse relative error 2.3669808690840455e-7, pivot ratio for column 0 3.2927233162283187e-10, recorded column residual 2.384185791015625e-7 |

These tiny/stress results validate scoped construction behavior. The near-dependent error is materially larger than the tiny well-conditioned error, consistent with conditioning sensitivity; its predeclared epsilon=1e-5 and inverse allowance=2e-6 are distinct from the tiny test, not retroactive tolerance changes. Wider references use 64-bit-significand long double on this host, not IEEE quadruple.

Verified host/compiler: Windows, MinGW.org GCC 6.3.0, compiler target `mingw32` (32-bit x86), bundled CMake 3.31 and Ninja. CMake propagates `-fno-fast-math -ffp-contract=off`; on 32-bit GNU MinGW it additionally propagates `-msse2 -mfpmath=sse` to avoid default x87 excess precision. This is an approved numerical-semantics guard, not a performance optimization; SSE2 support is required for that configuration. Compile-time assertions require IEEE binary64 double. The code assumes the normal round-to-nearest environment; it does not change or detect a caller-modified floating-point rounding mode. MSVC uses `/fp:strict`, but MSVC and other platforms have not been tested.

Exact CMake commands, using the variables/paths recorded in [milestone 1](CPU_MILESTONE1.md):

```powershell
& $cpuCmake -S . -B build/ninja
& $cpuCmake --build build/ninja --verbose
& $cpuCtest --test-dir build/ninja --output-on-failure
& .\build\ninja\spai_tests.exe
& .\build\ninja\cpu_core_tests.exe
```

Direct equivalent correctness build:

```powershell
& 'C:\MinGW\bin\g++.exe' -std=c++11 -O0 -g -Wall -Wextra -Wpedantic -fno-fast-math -ffp-contract=off -msse2 -mfpmath=sse -Iinclude src/cpu.cpp src/spai.cpp tests/test_spai.cpp -o build/implementation_spai_tests.exe
& .\build\implementation_spai_tests.exe
```

The deterministic 3x3 test uses epsilon=1e-12, beta=8. Alpha=1 is an explicit accuracy-cap failure; alpha=2 completes construction. The mean-score filter chooses candidate 2 alone at column 0's first growth, showing that beta is a maximum, not a required number. The separate near-dependent 2x2 test has delta=2^-30, epsilon=1e-5, alpha=1, beta=1 and a predeclared analytic-inverse relative allowance of 2e-6. These are labelled project component/stress cases, not published datasets or convergence guarantees.

No GMRES, IR, BSPAI, CUDA, SuiteSparse acquisition, speedup, or end-to-end convergence result exists in this milestone. No MATLAB bitwise/source-equivalence claim is made: loop limits, equality acceptance and score clamps are the explicitly recorded project choices.
