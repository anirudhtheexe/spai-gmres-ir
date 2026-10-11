# Stage 4.5A — Numerical Validation and Resource-Feasibility Report

## Scope and status

Stage 4.5A is complete for the acquired SuiteSparse pilot set, one intermediate matrix, and the first conditional larger-tier matrix. This was an untimed correctness/resource screen only. No performance timing, warm-up/repetition benchmark, profiler trace, optimization, BSPAI, mixed precision, tolerance change, restart, regularization, or production numerical-code change was made.

The frozen baseline remains uniform binary64 with `B=A^T D`, `P=N^T D`, QR SPAI, `epsilon=.3`, `beta=8`, `alpha=ceil(n/8)`, unrestarted left-preconditioned GMRES with `tau=1e-8`, maximum inner iterations `n`, outer cap 20, and `eta<=10*2^-53`. Gate 3 remains preserved at `gate3-working-baseline` (`581e75e66e45d8d2e789d9898d3ef2987c66b509`); work remains on `stage4-scalable-spai`.

## Data provenance and screening

Research acquired unchanged SuiteSparse archives and Matrix Market files under `data/stage45a_provenance_20261011/`. Exact source URLs, headers, authors, collection identifiers, CC-BY-4.0 basis, archive/member SHA-256 values, byte counts, static checks, and exclusions are in [MANIFEST.json](data/stage45a_provenance_20261011/MANIFEST.json).

Independent validation rechecked all ten archive/file hashes, coordinate counts, zeros, duplicates, finite values, diagonal availability, and scaling preflight: 111,091 static checks passed. All acquired A files are coordinate-real-general; raw explicit zeros were preserved. This is a format/range screen, not a rank or convergence certificate.

| Tier | Matrix | n | canonical nnz | Stage 4.5A outcome |
|---|---|---:|---:|---|
| Pilot | cage5 | 37 | 233 | CPU/GPU native and fixed-P solver success. |
| Pilot | gre_115 | 115 | 421 | CPU/GPU success; native P pattern discrepancy recorded below. |
| Pilot | pores_3 | 532 | 3,474 | CPU/GPU native and fixed-P solver success. |
| Pilot | steam1 | 240 | 2,248 | CPU/GPU native and fixed-P solver success. |
| Pilot | steam3 | 80 | 314 | CPU/GPU native and fixed-P solver success; forward/backward distinction recorded below. |
| Pilot | saylr1 | 238 | 1,128 | CPU/GPU native and fixed-P solver success. |
| Intermediate | sherman4 | 1,104 | 3,786 | CPU/GPU native solver success. |
| Conditional larger | orsreg_1 | 2,205 | 14,133 | CPU/GPU native solver success. |
| Deferred larger | sherman5 | 3,312 | 20,793 | Acquired/preflighted; not executed in this bounded screen. |
| Deferred larger | add32 | 4,960 | 19,848 | Acquired/preflighted; not executed in this bounded screen. |

No 10,000-row case was acquired or claimed. `HB/sherman3` (5,005 rows) was excluded because its collection metadata reports numerical rank 2,898; it was not downloaded.

## Independent numerical evidence

Validation used sequential processes with a 300-second external watchdog per case/configuration, no artificial memory cap, and no altered solver setting. `tests/untimed_screen.cpp` and `scripts/untimed_screen.py` preserve complete CSR, construction, solver, transfer, and memory diagnostics; they contain no performance clocks. Existing CUDA CTest passed 5/5 after adding the validation-only target.

`tests/assess_stage45a.py` independently recomputed original residuals and backward errors in Decimal80 from stored A, b, and x. It also audited feasible SPAI fits, sparse P quality, and CPU/GPU results. The consolidated artifact [ASSESSMENT_FINAL.json](results/stage45a_validation/ASSESSMENT_FINAL.json) records **788,034 checks, zero discrepancies** across all stored cases. Every reported solver success had independently checked residual and normwise backward error, including the periodic manufactured-solution rows; no manufactured x was presumed exact merely because it generated b.

Native and fixed-P comparisons are separate:

- Fixed-P comparisons used the same canonical host A/P/RHS/options and passed for all six pilots.
- Native comparisons constructed backend-specific P and must not be interpreted as equal work. All screened native cases reached solver success, but P or selection order can differ.

The largest completed case, `orsreg_1` (n=2,205), passed CPU/GPU native screening. Independent terminal Decimal80 normwise backward errors were `2.327362097105388e-16` (CPU) and `2.0402971761e-16` (GPU), both below the unchanged `10*2^-53` stopping threshold. Its P nnz was 11,025 on both paths; two outer steps used inner iteration sequences 58/76 on both paths. This is feasibility evidence for one structure, not a broad scalability claim.

## Recorded discrepancies and limitations

1. **gre_115 native P differs.** CPU P had 1,731 nnz and GPU P 1,739. The first selected-set difference is column 5, first fit: CPU selected `[20,13,23,24,46,59,90,103]`; GPU selected `[13,20,24,23,46,59,80,90]`. The exact-fraction singleton least-squares/restricted-score witness in `tests/compare_stage45a.py` identifies exact score ties split differently by computed rounding at the beta-8 boundary. Both paths follow their computed-score order, reach success in two outer steps with 17+17 inner iterations, and have solution infinity-difference `1.67e-16`. This is a transparent native-pattern-equivalence limitation, not a tolerance/rank fix target.
2. **Backward success does not guarantee small forward error.** `steam3` has Decimal normwise backward error about `1.04e-16`, yet relative forward error about `8.86e-11`; `pores_3` is about `7e-13`. No forward-error stopping condition was added.
3. **No resource abort occurred** among executed configurations. This does not establish feasibility for `sherman5`, `add32`, arbitrary sparse patterns, dense adaptive growth, host histories, vendor-internal memory, or unrestarted-GMRES worst cases.
4. **Controlled synthetic rows are deferred.** The approved new deterministic 2-D/irregular/arrowhead generator was not supplied in the initial harness; historical Stage 4 synthetic evidence was not substituted. This is an incomplete coverage limitation.

## Required classifications

Executed rows are preserved as `solver_success` or the deliberately constrained negative rows as `solver_nonconvergence`. The negative fixed-P `cage5` cases demonstrate explicit classification without reinterpretation: outer cap zero returned `iteration_limit`, and inner cap one returned `inner_failure`; CPU/GPU retained the same last-state Decimal normwise backward error `0.07406510201651704`. No executed row was discarded.

`construction_failure`, `resource_abort`, and `unsupported_input` remain supported record classifications but were not observed in the admitted positive suite. Raw JSONL records preserve source status/message and classification. Deferred matrices are **not run**, not successes or failures.

## Resource evidence

For `sherman4`, GPU SPAI recorded peak source-owned allocation 1,839,688 bytes, largest reduced shape 61x23, and released all source allocations; process peak working set was 461,459,456 bytes. Before the run, available host memory was about 2.65 GB and GPU free memory about 7.45 GB. These source-owned values exclude vendor-internal allocations and are not total-GPU-memory or performance measurements.

## Recommendation

The numerical prerequisites for a carefully bounded Stage 4.5B benchmark are met for the successfully screened matrix subset, subject to explicit approval. Before any timing, Stage 4.5B must retain native-versus-fixed-P separation, include all classifications, document CPU/thread/BLAS/compiler/GPU/power/clock/temperature environment fields, and use predeclared trial counts. It must not claim equal workload for native pipelines, numerical equivalence of `gre_115` P, broad scalability, or published mixed-precision reproduction.

Stage 4.5B is **not authorized by this report**. No timed performance experiment or profiling was executed.
