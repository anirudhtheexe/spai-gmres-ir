# BSPAI–GMRES–IR Codebase Guide

This document describes the implementation that is actually present in the source tree. It complements the mathematical and validation reports; it does not replace [PHASE2A_ALGORITHM_SPEC.md](docs/PHASE2A_ALGORITHM_SPEC.md), [PHASE2A_VALIDATION_PLAN.md](docs/PHASE2A_VALIDATION_PLAN.md), or [PHASE2B_VALIDATION_RECORD.md](docs/PHASE2B_VALIDATION_RECORD.md). The current code is a uniform binary64 reference and CUDA implementation. Stage 4 sparse-storage GPU SPAI is implemented and independently validated on bounded sparse fixtures. Stage 5 BSPAI bucketing/mixed precision and performance optimization remain future work, not implemented or authorized.

## 1. Architecture at a glance

The project has two independent build graphs:

```text
root CMakeLists.txt
  bspai_cpu: cpu.cpp -> spai.cpp -> gmres.cpp -> ir.cpp
  CPU executables/tests

cuda/CMakeLists.txt (separate 64-bit MSVC/CUDA build)
  bspai_cuda_spmv: cpu.cpp + cuda_spmv.cu
  bspai_cuda_gmres: gmres.cpp + spai.cpp + cuda_gmres.cu
  bspai_cuda_spai: cuda_spai.cu + GPU GMRES/SpMV libraries
  bspai_cuda_ir: ir.cpp + cuda_ir.cpp + GPU libraries
  cuda_hybrid_ir / cuda_gpu_spai_ir and CUDA tests
```

`CsrMatrix` is the shared host representation. CPU SPAI constructs a sparse `P`; CPU GMRES and CPU IR consume it directly. CUDA SpMV, GPU GMRES, and GPU SPAI upload canonical CSR or dense construction workspaces to the selected device. The hybrid IR outer loop deliberately remains CPU staged: CPU computes the original residual, backward-error diagnostics, stopping, and `x += d`; GPU GMRES solves each correction equation. GPU SPAI downloads its final packed sparse P values, canonicalizes them through `CsrMatrix`, and then uploads `A/P` into the persistent GPU GMRES context.

The public API is in `include/bspai/`; implementation details are private to the corresponding `.cpp`/`.cu` file, often behind a PImpl (`std::unique_ptr<Impl>`). CUDA public operations are synchronous from the caller’s perspective and throw on runtime/allocation/library failures. Numerical solver failures are returned as named result statuses.

## 2. CPU data model and source map

### `include/bspai/cpu.hpp`, `src/cpu.cpp`

`Entry` is a zero-based `(row, col, value)` input tuple. `CsrMatrix` owns canonical CSR vectors: `offsets_`, `columns_`, and `values_`; rows are sorted, duplicate coordinates are summed in stable input order, and stored zeros are omitted. `CsrMatrix::from_entries()` validates dimensions, indices, finite values, duplicate-sum finiteness, and builds offsets. `rows()`, `cols()`, `row_offsets()`, `column_indices()`, and `values()` expose read-only storage. `multiply()` validates vector size/finiteness and performs row-major sparse matrix-vector multiplication, throwing on nonfinite output. `transpose()` emits entries and re-canonicalizes them.

`read_matrix_market()` accepts Matrix Market coordinate real/integer, general/symmetric input. It parses comments and dimensions, converts one-based indices to zero-based, expands symmetric off-diagonal entries, rejects malformed/trailing data and nonfinite values, and returns canonical CSR. It does not support complex, array, skew-symmetric, or hermitian formats.

### `include/bspai/spai.hpp`, `src/spai.cpp`

Public types:

- `SpaiStatus` names success, invalid input, QR/rank or numerical failure, candidate exhaustion, score failure, no-growth, and growth-budget outcomes.
- `SpaiOptions` contains `epsilon`, `alpha` (growth/refit budget), and `beta` (maximum selected candidates per growth).
- `ScoreResult` is the bounded candidate score plus a `clamped` flag.
- `SpaiFit` records one pattern/shadow fit, coefficients, candidates, scores, selected growth, residuals, mean score, and QR pivot ratio.
- `SpaiColumn` records a column’s status, attempted/current patterns, coefficients, counts, last diagnostics, and complete fit history. NaN diagnostics mean no completed fit exists.
- `SpaiResult` owns scaling, scaled transpose `B`, constructed inverse-like `N`, final preconditioner `P`, and per-column diagnostics.

Important internals in `src/spai.cpp`:

- `at()` performs binary search in a CSR row.
- `norm2()` uses `std::hypot` accumulation.
- `least_squares()` implements economy Householder QR and triangular back substitution. It rejects underdetermined systems, nonfinite reflectors, exact-zero pivots, and nonfinite coefficients; it does not use normal equations or an arbitrary rank threshold.
- `construct_column()` starts pattern `{k}`, forms the shadow rows touched by that pattern, builds the dense reduced problem, solves it with QR, computes reduced and full residuals, and records a `SpaiFit`. If the residual exceeds epsilon, it forms the restricted candidate union from shadow rows and row `k`, excludes the current pattern, evaluates `evaluate_spai_score()`, computes the mean, sorts by `(score,index)`, retains up to beta scores no greater than the mean, sorts the new pattern, and repeats until success or an explicit failure.
- `evaluate_spai_score()` computes `sqrt(residual_squared - dot^2/denominator_squared)`, allowing only the documented bounded negative-radicand clamp. Invalid/nonfinite/zero-denominator cases have explicit statuses.
- `construct_spai()` validates a square input and positive epsilon, requires nonzero diagonal, computes `d[k]=1/max(abs(A[k,*]))`, forms `B=A^T D`, constructs every column, retains diagnostics for failures, and assembles `P=N^T D` only as a successful result. A failed result must not be used as a successful preconditioner.

The exact scaling orientation is important: `D` scales rows of `A`, then the transpose is formed; final `D` multiplication is on the right of `N^T`.

### `include/bspai/gmres.hpp`, `src/gmres.cpp`

`GmresOptions` holds positive tolerance and an optional cap (`0` means `n`). `GmresIteration` stores estimate/recomputed residuals, original residual, Arnoldi subdiagonal, breakdown/verification/acceptance flags, and the candidate solution. `GmresResult` stores status, convergence, solution, initial/final diagnostics, history, basis, and unrotated Hessenberg data.

`apply_left_preconditioned(A,P,v)` computes exactly `P*(A*v)` with two CSR products and never forms `P*A`. `left_preconditioned_gmres()` validates square dimensions, vectors, tolerance, and cap; handles zero RHS and exact initial solutions; computes the left-preconditioned initial residual; and runs unrestarted Arnoldi. Each iteration:

1. Applies `P*(A*v_k)`.
2. Performs one-pass modified Gram–Schmidt against all previous basis vectors.
3. Stores the Hessenberg column and normalizes the next basis vector unless the subdiagonal is exactly zero.
4. Applies prior Givens rotations, creates the new Givens rotation, updates `g`, and estimates the residual.
5. Back-solves the rotated triangular system and forms a candidate from the initial guess plus the basis combination.
6. Recomputes the original and preconditioned residual for the candidate before accepting convergence.

Breakdown, zero pivots, iteration limits, nonconvergence, zero preconditioned RHS, invalid input, and arithmetic overflow have explicit statuses. A candidate is not accepted based only on the Hessenberg estimate.

### `include/bspai/ir.hpp`, `src/ir.cpp`

`IrOptions` combines SPAI and GMRES options with a maximum outer count capped at 20. `IrDiagnostics` stores `r=b-Ax`, infinity/2 norms, normwise and componentwise backward errors, and optional forward diagnostics against a supplied reference. `IrIteration` stores each iterate, diagnostics, and the inner GMRES result. `IrResult` stores construction, outer/inner histories, correction RHS vectors, update flags, terminal status, and solution.

`evaluate_ir_diagnostics()` validates dimensions/finiteness, computes the original residual and safe `hypot` norm, then computes the specified backward-error quantities. `ir_backward_tolerance()` is fixed at `10*2^-53` and is not configurable.

`refine_with_preconditioner()` initializes `x0=P*b`, checks the original backward error, then repeatedly solves `P*A*d=P*r` with zero initial correction, updates `x=x+d`, recomputes diagnostics, and stops on backward-error success, inner failure, unchanged iterate (stagnation), numerical failure, or the outer cap. Failed inner solves never update `x`. `spai_gmres_ir()` constructs CPU SPAI first and then delegates to this loop.

## 3. CUDA components

### `include/bspai/cuda_spmv.hpp`, `src/cuda_spmv.cu`

`CudaCsrSpmv` is a synchronous, non-thread-safe PImpl wrapper. Its constructor validates/copies canonical CSR arrays to device memory, creates a nonblocking stream and cuSPARSE generic CSR/dense-vector descriptors, and queries workspace for binary64 nontranspose `CUSPARSE_SPMV_CSR_ALG2`. `multiply()` uploads `x`, clears output, invokes cuSPARSE SpMV, synchronizes, downloads the result, and validates dimensions/finiteness. `DeviceGuard`, checked status helpers, `StreamDrain`, and RAII allocations restore device ownership and clean up on exceptions. The object owns its device CSR; it does not expose asynchronous calls.

### `include/bspai/cuda_gmres.hpp`, `src/cuda_gmres.cu`

`CudaGmresContext` owns immutable device CSR matrices `A/P`, one stream, cuSPARSE/cuBLAS handles/descriptors, and reusable vector/Krylov workspaces. `solve(rhs,initial)` resets per-solve basis/snapshot bookkeeping but retains matrices and workspace. `cuda_left_preconditioned_gmres()` is the one-shot wrapper; `cuda_safe_norm2()` exposes the safe-scaled device norm for tests.

The main device helpers are:

- `finite_flag`: scans device entries for nonfinite values.
- `divide_vector`: literal elementwise division used for normalization, avoiding reciprocal overflow.
- `safe_scaled_norm`: one-thread scale/sumsq recurrence for robust Euclidean norms, including the tested subnormal cases.
- `Engine` methods upload/download, synchronize, copy, axpy, dot, norm, normalize, finite validation, and CSR SpMV.
- `Matrix::apply()` invokes cuSPARSE for a device-resident CSR product.
- `Run::solve()` mirrors the CPU Arnoldi/Givens algorithm while keeping Krylov vectors, candidates, residuals, and history snapshots on device. Small Hessenberg/Givens/backsolve decisions remain host-controlled; scalar observations synchronize the stream.

GPU GMRES uses binary64 typed cuBLAS/cuSPARSE APIs, pedantic cuBLAS math, atomics disabled, host scalar mode, nonblocking streams, and custom kernels compiled without fused multiply-add contraction. It does not promise CPU bitwise equivalence. The safe norm is an explicit project correctness decision after the local cuBLAS subnormal discrepancy; it is not a performance design.

### `include/bspai/cuda_spai.hpp`, `src/cuda_spai.cu`

`cuda_construct_spai()` accepts canonical square CSR and ordinary `SpaiOptions`, and returns `CudaSpaiResult` containing the standard `SpaiResult` plus telemetry. `cuda_evaluate_spai_score()` exposes the device score kernel for direct validation.

Implemented workflow:

1. Validate canonical CSR, square dimensions, positive finite epsilon, and QR `INT_MAX` bounds.
2. Upload A CSR metadata/values; interpret its offsets/indices as B CSC and allocate only nnz(B) scaled values, scaling and status arrays.
3. `scale_matrix` computes row maxima/reciprocals and the scaled transpose `B=A^T D`; it records per-row failure flags.
4. Host sorted CSC-support unions form shadows; sparse B row adjacency forms candidates. Up to eight independent stream/handle slots allocate reduced buffers from actual m*q, metadata/candidate counts and queried vendor workspace; grow-only capacities are retained within the call, not n*n storage.
5. GPU `gather` builds each exact-size column-major reduced matrix; `Dgeqrf` computes QR; `Dormqr(LEFT,TRANSPOSE)` applies Qᵀ to the identity RHS; `qr_guard` checks info, finite values, exact-zero diagonals, pivot diagnostics, and solved RHS readiness.
6. `Dtrsv(UPPER,NONUNIT)` solves the triangular system. `fitted_residual` recomputes reduced/full residuals from immutable original `B`; `scores` evaluates all restricted candidates with the same bounded score formula as CPU.
7. Host downloads coefficients/diagnostics/status, records `SpaiFit`, applies the same epsilon/growth/mean/filter/tie rules, and schedules the next pattern. Every batch submits independent QR work before collecting results; telemetry records dispatch structure, not hardware overlap or speedup.
8. Host packs final solved column patterns/previously downloaded coefficients into compact N CSC/P CSR ordering. GPU `assemble_p` computes `P[k,j]=N[j,k]*d[j]` into packed values with range guards; downloaded values are canonicalized through `CsrMatrix::from_entries()`. No dense N/P buffer or CPU numerical fallback exists.

The current implementation is hybrid and sparse-staged. Sparse lookup zero-fills absent reduced entries and preserves ascending accumulation order. Source allocation telemetry includes actual call/current/peak/capacity/slot/workspace observations, excluding vendor-internal memory. Validation-only `tests/reference/cuda_spai_gate3.cu` preserves dense Gate3 numerics under renamed exports in a BUILD_TESTING-only target, never a production fallback. A singular input can produce a different explicit failure status from CPU when vendor QR returns a tiny finite pivot; partial P is never solver-usable.

### `include/bspai/cuda_ir.hpp`, `src/cuda_ir.cpp`

`cuda_refine_with_preconditioner()` creates one persistent `CudaGmresContext`, then performs the CPU outer loop from `ir.cpp` while dispatching each correction to GPU GMRES. `cuda_spai_gmres_ir()` retains CPU SPAI construction and calls that hybrid loop. `cuda_gpu_spai_gmres_ir()` calls `cuda_construct_spai()`, rejects any unsuccessful construction, then sends its canonical P to the same persistent GPU GMRES/CPU-outer path. `CudaIrResult` exposes construction telemetry, context/matrix upload counts, per-inner transfer counters, and whether GPU construction ran.

### CLI and environment probe

`src/cuda_hybrid_main.cpp` reads a square Matrix Market matrix, creates the documented equal-unit RHS, parses epsilon/outer/inner caps, selects CPU or GPU construction via `BSPAI_GPU_CONSTRUCTION`, prints status/diagnostics/telemetry, and returns 0 only on convergence, 1 on solver/construction failure, and 2 on usage/invalid input.

`src/cuda_environment.cpp` reports CUDA runtime/driver, device properties, cuSPARSE, cuBLAS, and cuSOLVER versions. `src/cuda_norm_probe.cpp` is a targeted diagnostic for norm behavior.

## 4. Memory, streams, and transfer rules

All CUDA code uses explicit device selection guards and RAII-like wrappers for allocations, handles, descriptors, streams, and cleanup. Public calls synchronize before returning, so host input lifetimes are safe. CUDA objects are not thread-safe unless a caller creates independent objects/contexts.

GPU SpMV owns one device CSR and transfers vectors per call. GPU GMRES uploads immutable A/P at context construction and retains them across correction solves. RHS and initial guesses are uploaded per solve; Arnoldi vectors and snapshots remain device resident until termination, after which solution/history are downloaded. Dot products, norms, finite flags, projected Hessenberg decisions, and Givens/backsolve scalar decisions synchronize with the host.

GPU SPAI retains sparse B CSC and separate compact slot buffers; scaled sparse values/metadata, fit coefficients, diagnostics/scores, packed output coefficients and final P values cross the host boundary. Host full histories remain enabled. Compact storage is pattern-dependent: reduced problems/P can still grow dense; unrestarted GMRES separately retains O(n*cap) Krylov/history storage. Final P is canonicalized on CPU and uploaded once into GPU GMRES. The telemetry counters count application-issued copies, not vendor-internal movement.

## 5. Precision policy and safety boundaries

The implemented algorithm is IEEE binary64. CPU builds disable fast contraction where supported; CUDA custom kernels use `--fmad=false`; vendor-library reduction/FMA order remains library-controlled. cuBLAS pedantic math, atomics-not-allowed, cuSOLVER deterministic results/default math, and typed `D*` APIs are explicitly configured/read back. No mixed precision or emulated publication configuration is implemented.

The CPU and GPU implementations may differ in coefficients, residual histories, and boundary pattern choices by normal floating-point effects. Validation uses independent high-precision Decimal oracles and predeclared bounds; it does not assert bitwise CPU/GPU equality. The known singular-status divergence and cuBLAS subnormal norm discrepancy are documented in the Gate reports and handoff.

## 6. Build and test infrastructure

The root CMake project builds `bspai_cpu`, `cpu_matrix_probe`, and CPU tests `cpu_core_tests`, `spai_tests`, `gmres_tests`, and `ir_tests`. The CUDA CMake project requires 64-bit MSVC/CUDA and links `CUDA::cudart`, `CUDA::cusparse`, `CUDA::cublas`, and `CUDA::cusolver`. It builds the four solver libraries, `cuda_hybrid_ir`, `cuda_gpu_spai_ir`, `cuda_environment_probe`, `cuda_norm_probe`, and CUDA tests `cuda_spmv_tests`, `cuda_gmres_tests`, `cuda_ir_tests`, `cuda_spai_tests` and `cuda_spai_sparse_tests`. The new target links the validation-only Gate3 reference and independently instrumented source allocation audit; production executables do not.

`scripts/build_cuda_msvc.cmd` configures `cuda/` with Ninja, MSVC, CUDA architecture 89, and Debug settings, then builds and runs CTest. Decimal and CLI scripts under `tests/check_*.py` are independent validation oracles and are run manually as documented in the validation record. `tests/fixtures/tiny_nonsymmetric.mtx` is the deterministic Matrix Market CLI fixture. Preserved suites are CPU946/SpMV323/GPU GMRES871/hybrid665/Gate3 SPAI1534 checks. Stage4 adds3,363,954 checks (mostly outside-shadow zeros),395 independent80-digit SPAI checks and208 IR checks, with exact allocation-ledger agreement and zero sanitizer errors/leaks; CPU CTest4/4 and CUDA CTest5/5. See [CUDA_STAGE4_SCALABLE_SPAI.md](docs/CUDA_STAGE4_SCALABLE_SPAI.md) for measured memory scaling and bounded claims.

## 7. Safe extension rules

- Preserve `CsrMatrix` canonical invariants and use `from_entries()` when producing host CSR. Add input-format behavior only with tests for malformed, symmetric, duplicate, empty-row, and nonfinite cases.
- Keep CPU SPAI as the reference oracle. Any GPU/backend change must preserve scaling orientation, QR (not normal equations), candidate ordering, exact computed-pivot policy, status handling, and P assembly. Add an independent test before changing a numerical decision.
- Keep GMRES left/unrestarted and verify recomputed residuals. Do not accept a Hessenberg estimate alone, add restart, alter MGS semantics, or materialize `P*A` without an approved specification change.
- Keep IR’s CPU diagnostics and fixed backward-error threshold as the baseline. A failed inner correction must not update the outer iterate.
- For CUDA changes, preserve device ownership guards, checked API calls, stream synchronization before host-lifetime release, and exception-safe cleanup. Extend transfer telemetry when adding a transfer; do not infer performance from dispatch counts.
- Stage4 storage work is complete. Do not begin Stage5 BSPAI bucketing/mixed precision or benchmarking without explicit authorization and a source-backed written validation/precision plan.
- Add tests to the relevant CMake target and independently compare CPU/GPU coefficients, patterns, residuals, statuses, solution, and backward errors. Update the appropriate report and `PROJECT_HANDOFF.md` after a milestone.

## 8. Actual versus future functionality

Implemented: Stage4 sparse B/compact QR/packed GPU P construction and allocation-audited validation; CPU uniform-double SPAI/GMRES/IR; CUDA binary64 CSR SpMV; GPU binary64 GMRES; hybrid CPU-outer/GPU-inner IR; GPU binary64 SPAI with host adaptive metadata, cuSOLVER QR, GPU scoring/residuals, and end-to-end CLI; deterministic regression/oracle infrastructure.

Not implemented: Stage5 BSPAI bucketing, adaptive/mixed precision, published mixed-precision reproduction, performance optimization/benchmarking, multi-GPU/MPI execution, a full SuiteSparse dataset, and the IEEE manuscript text/bibliography. These are plans or limitations, not hidden capabilities.

## Append-only Stage 4 implementation note — 11 October 2026

The current production `stage4-scalable-spai` path in `src/cuda_spai.cu` stores scaled `B` sparsely, derives shadows and candidate support from sparse topology, sizes reduced QR allocations from actual `|I|`, `|J|`, and candidate counts, and packs fitted columns before GPU `P` assembly. `CudaSpaiTelemetry` reports source-issued allocation calls/bytes, live and peak bytes, maximum allocation, sparse-B/output sizes, reduced dimensions, QR workspace sizes, and cleanup. The previous dense implementation is retained only in `tests/reference/cuda_spai_gate3.cu/.hpp` and is linked only by the validation target.

Stage 4 validation is recorded in [CUDA_STAGE4_SCALABLE_SPAI.md](docs/CUDA_STAGE4_SCALABLE_SPAI.md) and [PHASE2B_VALIDATION_RECORD.md](docs/PHASE2B_VALIDATION_RECORD.md). It passed 3,363,954 sparse C++ checks, 395 80-digit SPAI checks, 208 80-digit IR checks, CPU CTest 4/4, CUDA CTest 5/5, memcheck, initcheck, and an independent allocation pointer ledger. These results establish bounded sparse-family correctness, not universal asymptotic memory or performance behavior. Preserve the Gate 3 CPU/GPU numerical contract when extending this code.

## Append-only Stage 4.5A handoff note — 11 October 2026

Stage 4.5A is complete for untimed acquisition, numerical validation, and resource-feasibility screening. The consolidated report is [STAGE45A_NUMERICAL_VALIDATION.md](STAGE45A_NUMERICAL_VALIDATION.md); the validation-owned record is [STAGE4_5A_INDEPENDENT_VALIDATION.md](docs/STAGE4_5A_INDEPENDENT_VALIDATION.md); raw provenance is [MANIFEST.json](data/stage45a_provenance_20261011/MANIFEST.json); and lossless run records are under `results/stage45a_validation/`.

Research acquired ten unchanged CC-BY 4.0 SuiteSparse matrices. Six pilots, `sherman4` (n=1,104), and `orsreg_1` (n=2,205) were screened. `sherman5` and `add32` were acquired but deferred; no 10,000-row case was claimed. Validation retained 32 processes: 28 successes and four explicitly constrained nonconvergences. The independent Decimal80 collector performed 788,034 checks with zero discrepancies. CPU/GPU fixed-P comparisons and native-pipeline comparisons are kept separate.

The `gre_115` native CPU/GPU preconditioners differ at an exact-score tie near the beta-8 boundary; this is documented as a computed-ordering limitation, not corrected by a tolerance or rank-policy change. `steam3` demonstrates that a small backward error can coexist with materially larger forward error. Controlled 2-D/irregular synthetic additions remain deferred because no approved generator was supplied. `tests/untimed_screen.cpp` and `scripts/untimed_screen.py` are validation-only infrastructure; they contain no performance clocks or optimization.

Stage 4.5B benchmarking and profiling are not authorized by this checkpoint. Do not infer speedups, equal workloads, broad scalability, or published mixed-precision reproduction from these results.

## Append-only Stage 4.5A extension note — 11 October 2026

Validation-only extension infrastructure is now present at `scripts/generate_stage45a_synthetic.py`, `scripts/untimed_screen.py`, `tests/run_stage45a_assessment.py`, `tests/assess_stage45a.py`, `tests/collect_stage45a_assessment.py`, and `tests/compare_stage45a.py`. These are not linked into production solver targets. The generator/data are `data/stage45a_synthetic_v1/`; retained raw/assessment/comparison evidence is `results/stage45a_extension/`; the complete interpretation is [STAGE45A_EXTENSION.md](STAGE45A_EXTENSION.md).

The extension preserves the production CPU/CUDA numerical contract. It validates controlled CPU/GPU pairs plus native `sherman5` and `add32` pairs, but native construction may select different patterns; do not use those records as fixed-`P`, equal-work, timing, or GPU-equivalence evidence. Any future benchmark must retain the distinction between native and fixed-`P` paths and must not overwrite retained artifacts.
