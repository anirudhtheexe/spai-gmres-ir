# Stage 4.5A extension: controlled and conditional-large validation

**Date:** 11 October 2026
**Status:** complete, untimed numerical validation only. Stage 4.5B has not begun.

## Scope

The extension retains the frozen uniform-binary64 SPAI–GMRES–IR contract: `B=A^T D`, `P=N^T D`, QR fits, `epsilon=0.3`, `beta=8`, `alpha=ceil(n/8)`, unrestarted GMRES (`tau=1e-8`), 20 outer steps, and normwise backward-error acceptance at `10u`. It made no algorithm, precision, tolerance, ordering, fallback, or production-source change. Every execution is untimed, with a 300-second external watchdog and no artificial memory cap; neither is a performance measurement.

## New inputs

`data/stage45a_synthetic_v1/MANIFEST.json` records deterministic generator `stage45a-dyadic-v1`, matrix/generator hashes, frozen options, and periodic-7 manufactured diagnostics. These project-designed cases are strictly row diagonally dominant because each diagonal is `sum(abs(offdiagonal))+1/2`:

| Family | n / canonical nnz | Structure |
|---|---|---|
| Nonsymmetric 2-D stencil | 64/288, 256/1216, 1024/4992 | Directional dyadic couplings. |
| Uneven-block irregular | 64/304, 256/1236, 1024/4951 | Repeating 3/5/8/13 blocks and directed links. |
| Arrowhead-plus-chain | 64/220, 256/828, 1024/3196 | Sparse hubs of degree 16/32/64 and an asymmetric chain. |

The retained SuiteSparse provenance manifest is `data/stage45a_provenance_20261011/MANIFEST.json`:

| Matrix | Source / ID | n | stored / canonical nnz |
|---|---:|---:|---:|
| `sherman5` | `HB/sherman5`, 246 | 3312 | 20793 / 20793 |
| `add32` | `Hamm/add32`, 540 | 4960 | 23884 / 19848 |

`add32` has 4036 retained explicit zeros removed by canonical CSR; `sherman5` has none. Static input/scaling preflight is not a rank, nonsingularity, convergence, or general resource-feasibility certificate. Application records use `equal_unit2`, so they make no manufactured-forward-error claim.

## Verified outcomes

`results/stage45a_extension/ASSESSMENT_EXTENSION_WITH_APPLICATION.json` contains exactly 22 assessments corresponding to 22 raw JSONL records. Reconciliation verified their raw SHA-256 values, `solver_success` classifications, `pass` outcomes, empty failure lists, **1,068,784 independent Decimal80 checks**, and zero discrepancies.

All 18 synthetic CPU/GPU records succeeded. They include independent construction/residual/backward-error checks and manufactured diagnostics; Decimal normwise backward errors range from `4.7566e-17` to `2.2849e-16`, while manufactured forward diagnostics range from `1.48e-16` to `7.40e-16`.

| Application matrix | CPU Decimal nBE | GPU Decimal nBE | CPU/GPU outer | CPU/GPU inner |
|---|---:|---:|---|---|
| `sherman5` | `2.3179e-17` | `1.0064e-17` | 2 / 2 | 76+77 / 76+77 |
| `add32` | `8.3589e-17` | `7.3445e-17` | 2 / 2 | 10+11 / 10+11 |

All reported nBEs are below unchanged `10u`. Application construction checks are sampled rather than exhaustive: `sherman5` audits 16 fits per backend (5939 unchecked); `add32` audits 17 CPU fits (10515 unchecked) and 17 GPU fits (10523 unchecked).

## Preserved cross-backend differences

Native CPU/GPU runs are not fixed-`P` or equal-work comparisons. Stencil and arrowhead final patterns agree; irregular cases have selected-order differences but equal final patterns/nnz. `sherman5` has equal `P` nnz (23414) but differs at final-pattern column 2495 (`||x_cpu-x_gpu||_inf=3.8858e-15`). `add32` differs in 92 final pattern/status columns, has `P` nnz 24554/24595, and `||x_cpu-x_gpu||_inf=2.7001e-13`. These are retained discrepancies; no tolerance, forced pattern, rank-policy, or fallback change conceals them. No fixed-`P` application comparison exists.

## Resources, regressions, and limits

No record is a timeout, allocation failure, unsupported input, construction failure, or solver nonconvergence. This is bounded evidence only; source allocation/free-memory snapshots do not prove memory scaling, absence of vendor allocation, capacity, leak freedom, or performance. `sherman5_screen_stderr.log` is a later artifact-preservation replay rejection (`FileExistsError`), not a solver/resource result.

On 11 October 2026, explicit Visual Studio CTest reruns passed CPU **4/4** (`spai`, `core`, `gmres`, `ir`) and CUDA **5/5** (`spai`, sparse-SPAI, IR, GMRES, SpMV). The first CPU sandbox invocation reported `BAD_COMMAND`; the unchanged suite passed under host execution. CTest durations are not benchmark results.

Current production hashes match the retained extension records; no Stage 4.5A numerical source, tolerance, precision, rank-guard, or stopping-policy change is evidenced. Stage 4 sparse-storage changes and the validation/provenance artifacts remain uncommitted.

## Gate conclusion

The bounded Stage 4.5A extension is complete. It does not establish performance, speedup, broad scalability, exhaustive application-SPAI correctness, CPU/GPU equivalence, BSPAI/mixed precision behavior, or published-result reproduction. Stage 4.5B remains separate and requires explicit approval.
