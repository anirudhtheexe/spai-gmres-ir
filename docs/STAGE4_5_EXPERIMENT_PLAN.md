# Stage 4.5 Experiment Plan (Proposed — Approval Required)

## Scope

This is a pre-execution plan only. It does not authorize data acquisition, experiments, code changes, tolerance changes, benchmarking, profiling, BSPAI, mixed precision, reordering, regularization, restart, or performance optimization.

Execution, if approved, remains on `stage4-scalable-spai`. The preserved Gate 3 baseline is annotated tag `gate3-working-baseline` at `581e75e66e45d8d2e789d9898d3ef2987c66b509`. Stage 4's uniform-binary64 method remains unchanged: `B=A^T D`, `P=N^T D`, Householder QR, adaptive restricted candidates, unrestarted left-preconditioned GMRES, existing safe-scaled norms, and existing stopping/failure policy.

## A. Proposed matrix manifest

### Controlled synthetic families

| Family | Initial sizes / purpose | Classification |
|---|---|---|
| Existing nonsymmetric tridiagonal: lower/upper `2/1`, diagonal `8` and `5` | `n=16,64,256,1024`; continuity with Stage 4. Diagonal `8` is initial-fit control, `5` exercises growth. | Validation and baseline; not representative of general sparsity. |
| Strictly diagonally dominant nonsymmetric 2-D stencil | Modest grids first; grow only after resource screening. | Variable local patterns and QR shapes. |
| Strictly diagonally dominant uneven-block, irregular-degree, and arrowhead forms | Modest, then screened medium sizes. | Long rows and fill risk absent from tridiagonals. |
| Near-dependent, cancellation, extreme-range, singular, zero-diagonal cases | Existing/small fixtures. | Failure and decision-boundary diagnostics only, never successful-solve performance rows. |

Use deterministic formulae before any seeded randomness. Record generator version, all parameters and seed; use dyadic manufactured solutions where practical, but independently evaluate/solve the stored binary64 system because rounded `b=A*x*` does not make `x*` exact for stored `A,b`.

### Conditional SuiteSparse pilot

Candidate-page metadata was verified during plan preparation; archives were not downloaded. Admit a case only after its exact Matrix Market bytes/header, hash, diagonal suitability, license/citation and explicit-zero/duplicate handling have been checked.

| Candidate | Collection ID; dimensions / numerical nnz | Proposed role |
|---|---:|---|
| `vanHeukelum/cage5` | 906; `37 x 37` / 233 | Published-problem bridge; too small for scalability claims. |
| `HB/gre_115` | 161; `115 x 115` / 421 | Small nonsymmetric pilot. |
| `HB/pores_3` | 234; `532 x 532` / 3,474 | Larger application pilot, conditioning risk. |
| `HB/steam1` | 251; `240 x 240` / 2,248 | Paper-linked candidate; page reports 1,514 explicit zeros. |
| `HB/steam3` | 253; `80 x 80` / 314 | Severe-conditioning stress; page reports 614 explicit zeros. |
| `HB/saylr1` | 239; `238 x 238` / 1,128 | Ill-conditioned application stress. |

These pages identify real nonsymmetric, numerically full-rank matrix metadata, not a certificate for the acquired file or SPAI convergence. SuiteSparse matrices are CC-BY 4.0: retain matrix metadata/citations and disclose transformations. Preserve raw files. If canonical CSR removes explicit zeros or combines duplicates, record raw pattern entries, numerical nnz, canonical nnz, and hashes for raw and canonical data. Do not silently permute, regularize, or discard cases.

For each admitted case record collection/group/name/ID, source/citation/license URLs and dates, archive/extracted-file SHA-256 and bytes, Matrix Market banner/comments, declared entries, symmetry expansion and canonicalization, RHS/reference bytes and method.

## B. Frozen configuration and independent validation

For proposed new non-boundary cases use `epsilon=0.3`, `beta=8`, `alpha=min(n,ceil(n/8))`, inner `tau=1e-8`, inner cap `n`, outer cap `20`, and unchanged `eta <= 10*2^-53`. Historical Stage 4 continuity cases retain `alpha=5`. Declare any different setting before execution; no post-result tolerance adjustment. Deliberately capped negatives are separate rows.

1. **SPAI gate.** Independently reconstruct feasible controlled columns: scaling, shadows/candidates, selected patterns, QR fits, full residuals, and `P=N^T D`. Record per-column status, fits/growths, pivots, residuals, patterns, `nnz(P)`, residual distributions, and sparse `||I-PA||_F`. Successful construction requires every recorded residual to meet the unchanged epsilon rule; this does not establish nonsingular `P`.
2. **CPU/GPU construction comparison.** Require status/metadata agreement for controlled cases away from boundaries. On SuiteSparse cases retain the first candidate-score/mean/epsilon/pivot divergence and audit at higher precision; never automatically dismiss it as roundoff. Tiny-case coefficient bounds are not assumed universal.
3. **GMRES gate.** Use both fixed-`P` CPU/GPU comparisons (same canonical `A,P`, RHS, guess, tolerance, cap) and native-pipeline comparisons (each builds P; report work/pattern differences). Record every Hessenberg estimate, recomputed preconditioned residual, original residual, Arnoldi/breakdown diagnostic, final status and solution. Estimate-only convergence is rejected.
4. **IR gate.** Independently check `x0=P*b`, correction RHS, zero correction initial guess, accepted updates, original residuals and backward errors. Failed corrections may not update `x`. Success uses the frozen binary64 eta criterion; forward error is diagnostic only.
5. **Oracle and failures.** Decimal80 checks cover all feasible controlled fits. Preselect SuiteSparse samples: fixed indices plus largest pattern, worst residual and smallest pivot; recheck every terminal solution/divergence. Preserve rows for malformed/rectangular/zero-diagonal input, range failure, singular/near-dependent columns, alpha/beta zero, ties/boundaries, zero RHS, zero normalization, breakdown, caps, stagnation and overflow. Existing CPU/GPU singular-status differences are findings, not exclusions.

Gate evidence must be reviewed before presenting performance results. Unsupported, resource-aborted, nonconvergent and failed cases remain manifest rows.

## C. Proposed baseline methodology

Primary time is synchronized monotonic host elapsed time around the full public call. Parsing/CSR preparation is excluded from primary solver time but may be a distinct application-total measurement.

| Measurement | API / required included work |
|---|---|
| CPU SPAI | `construct_spai`: scaling, adaptive fits/growth, assembly, diagnostics. |
| GPU SPAI | `cuda_construct_spai`: handles/allocations, transfers, GPU QR/residual work, host metadata, downloads, packaging, cleanup. |
| CPU GMRES | `left_preconditioned_gmres` with supplied canonical P. |
| GPU one-shot GMRES | `cuda_left_preconditioned_gmres`: setup/uploads, solve, diagnostic downloads, teardown. |
| GPU reusable GMRES | Separately report `CudaGmresContext` construction, synchronous `solve`, destruction. |
| CPU full pipeline | `spai_gmres_ir`. |
| Hybrid pipeline | `cuda_spai_gmres_ir`, including CPU SPAI/outer arithmetic and GPU correction solves. |
| GPU full pipeline | `cuda_gpu_spai_gmres_ir`, including P staging/reupload, context setup, corrections, diagnostics, cleanup. |

CUDA events may supplement, never replace, primary wall time: private/multiple streams and host work mean event durations are not additive end-to-end time. Start without outstanding work and stop only after public API completion/synchronization.

Proposed protocol: three untimed warm-ups then twenty sequential measured trials per matrix/config/API. Rebuild SPAI every construction/native-pipeline trial. A reusable GMRES context may be reused only in its explicitly labelled setup/solve/teardown series and resets its guess. Alternate CPU/GPU order deterministically; no concurrent benchmark load; keep every raw sample; no outlier removal. Report count, minimum, median, mean, standard deviation and IQR plus every failure/interruption.

Record source commit/dirty hash, build/compiler/flags, CPU/GPU, driver/runtime/library versions, precision/modes, parameters, timing boundary, warm-up/order, RHS/matrix hash, status, fits/growths/`nnz(P)`, iterations, residuals/backward errors, transfers and available allocation telemetry. Store JSON Lines: one manifest plus one record per sample. A Release comparison requires independent numerical revalidation; historical cross-toolchain Debug checks are correctness evidence, not fair performance comparisons.

Current telemetry supports GPU-SPAI transfer bytes/source-owned allocation peaks. It does not provide CPU peak memory, GPU-GMRES allocation/byte totals, vendor-internal GPU memory, utilization, power, or temperature; mark those unavailable rather than infer them. For bottleneck attribution only, one profiler trace per validated tier may be collected separately and never combined with timing statistics; profiling does not authorize optimization.

## D. Feasibility limits and approval gate

Sparse Stage 4 staging removes production dense `n^2` SPAI buffers but does not guarantee linear memory: adaptive fill, histories, vendor QR workspace and unrestarted GMRES storage can dominate. Before increasing size, screen source allocation telemetry, available GPU/host memory and a declared wall-time budget. Start at documented `n<=1024`; resource-aborted rows are unvalidated, never passes. The small/moderate SuiteSparse pilot cannot establish broad scalability.

Approval is required before acquiring matrices, creating a harness, executing validation/timing/profiling, or writing result records. The first execution report must distinguish validated outcomes from limits, failures and unresolved CPU/GPU differences; no speedup, general scalability, numerical equivalence, or published-result-reproduction claim follows from this plan.

## Append-only execution amendment — Stage 4.5A complete

The user approved execution as two checkpoints. Stage 4.5A has now completed
untimed provenance acquisition, numerical validation, and resource screening.
Its authoritative result is [STAGE45A_NUMERICAL_VALIDATION.md](../STAGE45A_NUMERICAL_VALIDATION.md), with raw provenance and retained records linked there. Stage 4.5B baseline benchmarking and bottleneck profiling remain a separate checkpoint and require explicit approval.

The executed screen retained 32 processes (28 solver successes and four
intentional solver nonconvergences) and 788,034 independent Decimal80 checks
with zero discrepancies. It screened six pilots, `sherman4`, and `orsreg_1`
at n=2,205. `sherman5` and `add32` remain acquired but unexecuted. No timing,
profiling, optimization, BSPAI, mixed precision, or tolerance change occurred.
