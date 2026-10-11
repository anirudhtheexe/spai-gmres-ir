# BSPAI–GMRES–IR Project Handoff

**Handoff date:** 11 October 2026  
**Authoritative status:** Stage 4 sparse-storage GPU SPAI is complete and independently validated. Stop here; do not begin Stage 5 BSPAI/mixed precision without explicit user authorization. Current Git branch: `stage4-scalable-spai`; preserved baseline tag: `gate3-working-baseline`.

## Start here

1. Read [AGENTS.md](AGENTS.md), this handoff, [PAPER_INDEX.md](PAPER_INDEX.md), and the specifications in `docs/` before changing numerical code.
2. Read [CODEBASE_GUIDE.md](CODEBASE_GUIDE.md), then inspect the current source and validation records, especially [CUDA_STAGE3_GPU_SPAI.md](docs/CUDA_STAGE3_GPU_SPAI.md) and [PHASE2B_VALIDATION_RECORD.md](docs/PHASE2B_VALIDATION_RECORD.md).
3. Verify the four project-local specialist definitions in `.codex/agents/`. Their exact model identifiers must be supported by the new account before use; do not substitute silently.
4. Rebuild and rerun the documented tests on the target machine before extending the code. Preserve any discrepancy rather than changing tolerances or stopping criteria to make it disappear.

## Project purpose

This research project establishes a reproducible numerical baseline for SPAI–GMRES–IR, develops a CUDA implementation, and is intended to lead to an IEEE conference paper. The central research baseline is Carson–Khan’s SPAI/BSPAI–GMRES–IR work. Work completed here is uniform binary64 correctness work, **not** a reproduction of the published mixed-precision BSPAI results and not a performance study.

`PAPER_INDEX.md` is the literature entry point. Its priority items are the original SPAI–GMRES–IR paper and the BSPAI–GMRES–IR paper, followed by GPU/MPI SPAI and GPU GMRES/SpMV literature. It also lists the original SPAI–GMRES–IR and ADAP-SPAI–GMRES–IR GitHub repositories. Its access/review labels must be respected: do not treat a listed link as proof that a full text or repository has been independently verified.

## Mathematical contract

The binding executable contract is [PHASE2A_ALGORITHM_SPEC.md](docs/PHASE2A_ALGORITHM_SPEC.md), with independent requirements in [PHASE2A_VALIDATION_PLAN.md](docs/PHASE2A_VALIDATION_PLAN.md).

- Fixed SPAI scaling/orientation: `B = A^T D`, where `d[k] = 1 / max_j |A[k,j]|`; construct `N` and assemble `P = N^T D`.
- SPAI starts from identity patterns; uses adaptive sparse-pattern growth, restricted candidate scoring, documented ascending tie/order rules, and the stated epsilon/growth limits.
- Least-squares fits use QR, never normal equations. The frozen rank guard is an exactly-zero **computed** QR diagonal, not an arbitrary near-rank threshold.
- GMRES is double-precision, unrestarted, left preconditioned, one-pass modified Gram–Schmidt with Givens rotations. Apply `P*(A*v)` without constructing `P*A`; recompute the true preconditioned residual before accepting convergence.
- Iterative refinement starts `x0=P*b`, uses a zero initial correction, evaluates `r=b-A*x`, and uses the specified normwise backward-error criterion (`10*2^-53`) with the documented outer limit.

Do not alter this policy, precision semantics, tolerance, scaling orientation, rank handling, or benchmark methodology without a written plan and independent validation.

## Development history and current architecture

### Phase 1 — investigation

[PHASE1_REPORT.md](docs/PHASE1_REPORT.md) records the literature/repository investigation, verified findings, gaps, and recommendations. The project deliberately distinguishes verified source-backed claims from proposals.

### Phase 2A — CPU uniform-double baseline

- Milestone 1: CSR/reference core — [CPU_MILESTONE1.md](docs/CPU_MILESTONE1.md)
- Milestone 2: CPU SPAI construction — [CPU_MILESTONE2.md](docs/CPU_MILESTONE2.md)
- Milestone 3: CPU GMRES — [CPU_MILESTONE3.md](docs/CPU_MILESTONE3.md)
- Milestone 4: end-to-end CPU SPAI–GMRES–IR — [CPU_MILESTONE4.md](docs/CPU_MILESTONE4.md)

CPU sources are `src/cpu.cpp`, `src/spai.cpp`, `src/gmres.cpp`, and `src/ir.cpp`, with public interfaces in `include/bspai/`. The CPU regression baseline is 946 checks: 37 CSR/core, 190 SPAI, 280 GMRES, and 439 IR; CPU CTest is 4/4.

### Phase 2B — CUDA uniform-double baseline

- CUDA Milestone 5A: cuSPARSE binary64 CSR SpMV — [CUDA_MILESTONE5A.md](docs/CUDA_MILESTONE5A.md)
- Gate 1: GPU-resident left-preconditioned GMRES — [CUDA_STAGE1_GMRES.md](docs/CUDA_STAGE1_GMRES.md)
- Gate 2: CPU SPAI / GPU GMRES hybrid IR — [CUDA_STAGE2_HYBRID_IR.md](docs/CUDA_STAGE2_HYBRID_IR.md)
- Gate 3: GPU SPAI construction and end-to-end solver — [CUDA_STAGE3_GPU_SPAI.md](docs/CUDA_STAGE3_GPU_SPAI.md)
- Stage 4: sparse B/compact GPU SPAI workspace and output — [CUDA_STAGE4_SCALABLE_SPAI.md](docs/CUDA_STAGE4_SCALABLE_SPAI.md)

CUDA source components are:

- `src/cuda_spmv.cu`, `include/bspai/cuda_spmv.hpp`: cuSPARSE binary64 CSR SpMV.
- `src/cuda_gmres.cu`, `include/bspai/cuda_gmres.hpp`: GPU-resident vector GMRES. It uses a project safe-scaled device norm because the tested cuBLAS 13.7.0 norm path mishandled local subnormal fixtures.
- `src/cuda_ir.cpp`, `include/bspai/cuda_ir.hpp`: CPU-staged outer IR with persistent GPU GMRES context, plus GPU-SPAI integration.
- `src/cuda_spai.cu`, `include/bspai/cuda_spai.hpp`: GPU numerical SPAI construction using independent bounded stream slots, cuSOLVER QR, GPU residual/scoring and GPU `P` assembly.
- `src/cuda_hybrid_main.cpp`: shared Matrix Market CLI. `cuda_hybrid_ir` uses CPU SPAI; `cuda_gpu_spai_ir` enables GPU SPAI.

`cuda/CMakeLists.txt` defines the separate x64 CUDA build. The CPU root `CMakeLists.txt` remains the independent reference build.

### Historical Gate 3 result

The final independent Gate 3 evidence is in [PHASE2B_VALIDATION_RECORD.md](docs/PHASE2B_VALIDATION_RECORD.md): 1,534 GPU-SPAI C++ checks, 436 independent 80-digit SPAI checks over 60 fits, 43 independent 80-digit IR checks, and 15 CLI checks passed. CUDA memcheck reported zero errors/leaks and initcheck zero errors. Previously validated suites also passed: CPU 946 checks, SpMV 323, GPU GMRES 871, and hybrid IR 665; CUDA CTest 4/4 and CPU CTest 4/4.

These are deterministic, small-matrix, uniform-double correctness results. They are not performance numbers, scalable-GPU claims, general high-dimensional robustness proof, or a published mixed-precision reproduction.

### Current Stage 4 result

The authoritative independent record passes 3,363,954 new sparse C++ checks (mostly exact outside-shadow zero assertions), 395 new80-digit SPAI checks over48fits and208 new80-digit IR checks over3states. Full new-suite memcheck0errors/0leaks and initcheck0errors. All prior suites pass; CPU CTest4/4, CUDA CTest5/5. Independent cudaMalloc/cudaFree pointer-ledger measurements exactly match source allocation telemetry, including zero outstanding allocations on tested normal/failure returns; vendor internal allocations are excluded.

For the adaptive deterministic tridiagonal family n16/64/256/1024, source-owned peak bytes are173808/181680/213168/339120, max reduced shape5x3 and queried QRworkspace20992bytes. This establishes sparse storage on tested local-pattern families, not timing, general asymptotic performance or arbitrary-fill bounds. See the Stage4 report for commands/hashes, validation-only frozen Gate3 library isolation, host staging and all limitations.

## Known limitations and unresolved discrepancies

1. **Stage 4 is implemented and validated on bounded sparse families through n1024.** B uses sparse CSC; QR buffers use actual reduced shapes; N/P are compactly packed. Actual patterns can still grow dense. Retained host histories and unrestarted GMRES Krylov/history buffers can dominate memory; this is not a universal linear-memory solver claim.
2. GPU SPAI remains hybrid host controlled: sorted patterns, candidate metadata, mean/filter/tie decisions, and adaptive scheduling remain on CPU. Stream telemetry proves independent work was submitted before collection, not actual hardware overlap or speedup.
3. The outer iterative-refinement loop is CPU staged. The final GPU P is downloaded/canonicalized and uploaded into the persistent GMRES context once; no direct device-to-device ownership transfer is claimed.
4. On literal singular `[[1,2],[1,2]]`, CPU reports `rank_deficient` while GPU reports `empty_candidates` because GPU Householder QR produces a tiny finite pivot. Both fail; no partial P is supplied to GMRES. Do not add a threshold, structural fallback, or CPU fallback to conceal this documented difference.
5. The tested cuBLAS 13.7.0 `Dnrm2` path gave incorrect local subnormal-norm results. The intentionally serial safe-scaled device norm is a correctness baseline, not a performance optimization.
6. BSPAI bucketing, mixed precision, benchmark methodology, speedups, and publication-quality experiments remain unimplemented.
7. `PAPER_INDEX.md` has links and citations but the project `papers/` directory has no locally available PDFs at this handoff. Full texts must be lawfully reacquired and review status updated before relying on new literature claims.
8. Git was subsequently initialized with authorization: current branch `stage4-scalable-spai`, baseline commit `581e75e`, tag `gate3-working-baseline`; no remote is configured. Stage4 source/report changes are currently uncommitted at this handoff update. Preserve the tag; do not reset or overwrite user changes.

## Environment and dependencies

Latest validated CUDA environment (machine specific):

- NVIDIA RTX 4060 Laptop GPU, compute capability 8.9, 8,188 MiB; driver 616.92.
- CUDA Toolkit/nvcc 13.4.59; runtime/driver API 13.4.
- cuSPARSE 12.8.6, cuBLAS 13.7.0, cuSOLVER 12.3.2.
- 64-bit MSVC 19.44.35228, CMake 3.31.6-msvc6, Ninja 1.12.1, Python 3.12.
- CUDA architecture `89`; Windows x64. The CUDA build requires CMake, Ninja, Visual Studio Build Tools, NVIDIA driver/runtime, cuBLAS, cuSPARSE, and cuSOLVER.

The supplied script contains absolute paths for this particular Windows installation. On a new machine, edit/copy it only after verifying installed tool locations, CUDA architecture, and compiler compatibility. Set `CUBLAS_EMULATE_DOUBLE_PRECISION=0`; the implementation also sets/queries documented cuBLAS and cuSOLVER modes. The old environment encountered sandbox DLL-relocation failures; verified CUDA execution occurred outside that sandbox. A new account/session must independently establish its own execution permissions.

## Build, execution, and test commands

From the project root, the existing CUDA script configures `cuda/` into `build/cuda-ninja`, builds, and runs CUDA CTest:

```powershell
cmd /c scripts\build_cuda_msvc.cmd
$env:PATH='C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin\x64;'+$env:PATH
$env:CUBLAS_EMULATE_DOUBLE_PRECISION='0'
& 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\ctest.exe' --test-dir build\cuda-ninja --output-on-failure
```

GPU SPAI CLI smoke test (explicitly uses a small validation fixture and is not a benchmark):

```powershell
build\cuda-ninja\cuda_gpu_spai_ir.exe tests\fixtures\tiny_nonsymmetric.mtx 0.5
```

The default epsilon `0.3` is expected to transparently fail construction for that tiny fixture because `alpha=1`; use the explicit `0.5` smoke configuration only as documented.

For independent CUDA numerical checks, use the exact commands in the Stage4 report and updated validation record, including the new `cuda_spai_sparse_tests` allocator audit plus `tests/check_cuda_spai_decimal.py`, `tests/check_cuda_spai_cli.py`, Compute Sanitizer memcheck/initcheck, and `build/validation-cuda`. These records include pinned artifact hashes and full command lines.

For the CPU baseline, configure root `CMakeLists.txt` with a compiler supporting C++11 and no fast-math, then run CTest and the four executables: `cpu_core_tests`, `spai_tests`, `gmres_tests`, `ir_tests`. Existing historical CPU validation used `build/ninja`; build directories are machine-generated and should be regenerated on the target system.

## Project-local specialist agents

The following supported Codex definitions are present under `.codex/agents/`:

| Role | File | Exact configured model | Reasoning |
|---|---|---|---|
| Research | `bspai_gmres_ir_researcher.toml` | `gpt-6.1-sol` | medium |
| Implementation | `bspai_gmres_ir_implementer.toml` | `gpt-5.6-terra` | medium |
| Validation | `bspai_gmres_ir_validator.toml` | `gpt-5.6-terra` | medium |
| Manuscript | `bspai_gmres_ir_manuscript.toml` | `gpt-6-luna` | medium |

These definitions contain the full responsibilities and should be invoked by a new main session as appropriate. Model availability and any account-level model entitlement are account specific; if a target account cannot select one of these identifiers, report that fact and ask the user how to proceed rather than silently substituting another model.

## Manuscript and research materials

The unmodified IEEE template is in `manuscript/IEEE_Conference_Template/` and includes `conference_101719.tex`, `conference_101719.pdf`, `IEEEtran.cls`, and `IEEEtran_HOWTO.pdf`. There is no `references.bib` and no project-authored manuscript draft at handoff. The manuscript agent must preserve the template, create citations only from verified publications, and never invent results.

Research index and reports are local. Paper full texts are externally referenced in `PAPER_INDEX.md`; none are present under `papers/` at handoff. Repository URLs are external and need their own network access/authorization.

## Immediate next milestone — do not start without authorization

**Stage 5: BSPAI bucketing and mixed precision is future work, not authorized or implemented.** First obtain explicit approval and a source-backed precision/bucketing/validation specification; do not infer that Stage4 storage validation authorizes changed precision, preconditioner application, convergence rules or benchmarking.

Stage4 is complete: sparse B CSC, compact actual reduced QR buffers, packed GPU P assembly, unchanged frozen SPAI numerics and independent allocation/numerical/sanitizer evidence. Preserve the CPU reference and the validation-only Gate3 dense reference. Review [CUDA_STAGE4_SCALABLE_SPAI.md](docs/CUDA_STAGE4_SCALABLE_SPAI.md) and the authoritative validation record before any extension.

After explicit future approval: Stage5 BSPAI bucketing/mixed precision; Stage6 performance optimization and benchmarking; Stage7 IEEE manuscript finalization. None has started.

## Transfer audit

Transfer the entire project directory, including hidden `.codex/agents/`, source, headers, `cuda/`, `scripts/`, tests, fixtures, docs, `manuscript/`, `PAPER_INDEX.md`, and this handoff. Build outputs in `build/` are machine-specific and nonessential; regenerate them. Do not transfer credentials, tokens, user profile data, or the prior conversation.

Git is now initialized on `stage4-scalable-spai` with the preserved Gate3 tag and no configured remote; include .git when transferring recoverable history, and preserve current uncommitted Stage4 files. Verify repository/worktree state after copying. Check the new account can access the configured models, a suitable CUDA-capable host, toolkit/compiler paths, official source links, and any Codex-specific agent support. Those account/machine services do not transfer with files.

### Final handoff audit — 11 October 2026

- The project-local `.codex/config.toml` preserves the coordinator/default setting `gpt-5.6-terra`, medium reasoning; the four full role definitions are present in `.codex/agents/` with the exact requested model identifiers and responsibilities.
- All source/header files, both CMake configurations, CUDA script, CPU/CUDA test suites, decimal/CLI validation scripts, tiny Matrix Market fixture, reports/specifications, Gate 3 validation evidence, and the IEEE template assets are present in the project folder.
- The current machine reran `cmd /d /c scripts\build_cuda_msvc.cmd`: configure/build succeeded and CUDA CTest passed 4/4. CPU CTest on `build/ninja` also passed 4/4. These commands refresh generated test logs only; no numerical source or tolerance was changed during handoff.
- `papers/` is empty. No locally cached reference repositories, MATLAB reference implementation, source archive, dependency lockfile, README/LICENSE, unified all-check runner, or project manuscript/bibliography is available. The IEEE directory is the generic template only, with its placeholder/example bibliography.
- Build directories, binaries, CMake caches, Debug runtimes, and saved test logs are machine-generated. A stale historical `LastTestsFailed.log` may exist under `build/`; the fresh CTest runs above are the current result. Regenerate build artifacts after transfer rather than using them as portable evidence.
- No active build/test processes were found during the audit. Live agent conversations, Codex account entitlements, custom-agent loading behavior, access to external papers/repositories, MATLAB/Advanpix licenses, CUDA driver/toolkit/MSVC/Python installations, and the prior conversation do not transfer with this folder.

## Append-only Stage 4 completion update — 11 October 2026

Stage 4 is implemented and independently validated on `stage4-scalable-spai`. Production GPU SPAI now uses sparse CSC storage for `B=A^T D`, compact reduced-problem QR workspaces, compact fitted-column packing, and GPU assembly of `P=N^T D`. The prior dense Gate 3 constructor remains under `tests/reference/` as a validation-only comparison target and is never a production fallback.

Final evidence includes 3,363,954 new sparse C++ checks, 395 independent 80-digit SPAI checks over 48 fits, 208 independent 80-digit end-to-end IR checks, CPU CTest 4/4, CUDA CTest 5/5, zero memcheck errors/leaks, zero initcheck errors, and exact agreement between allocator instrumentation and reported telemetry. Deterministic sparse families were tested through `n=1024`; measured source-owned adaptive peak allocation ranged from 173,808 bytes at `n=16` to 339,120 bytes at `n=1024`.

These are bounded sparse-family correctness results. Vendor-internal allocations, dense pattern growth, host histories, compact QR worst cases, unrestarted GMRES histories, and CPU-staged packaging remain possible memory costs. No universal linear-memory, high-dimensional robustness, overlap, speedup, or performance claim is made. Stage 5 BSPAI bucketing/mixed precision remains unauthorized and unimplemented.

## Append-only Stage 4.5A validation update — 11 October 2026

Stage 4.5A completed untimed acquisition, independent numerical validation and resource-feasibility screening. See `STAGE45A_NUMERICAL_VALIDATION.md`, `docs/STAGE4_5A_INDEPENDENT_VALIDATION.md`, the provenance manifest under `data/stage45a_provenance_20261011/`, and raw retained results under `results/stage45a_validation/`. Ten CC-BY 4.0 SuiteSparse matrices were acquired unchanged with raw/archive hashes; six pilots, `sherman4` (n=1,104), and the first conditional larger matrix `orsreg_1` (n=2,205) were screened. The conditional n=3,312/n=4,960 matrices were acquired but not run.

There were 32 retained processes: 28 solver successes and four intentionally constrained solver nonconvergences. The final Decimal80 assessment performed 788,034 checks with zero discrepancies. CPU CTest 4/4 and CUDA CTest 5/5 passed. `gre_115` has a documented CPU/GPU native-P pattern discrepancy induced by computed-score ordering near exact ties; it was not hidden by a tolerance or rank-policy change. `steam3` demonstrates that a successful backward error need not imply small forward error. No timing, warm-up, repeated trials, profiling, speedup claim, optimization, BSPAI, or mixed precision occurred. Stage 4.5B baseline benchmarking requires separate explicit approval.

## Append-only Stage 4.5A extension completion — 11 October 2026

The approved extension is complete; see [STAGE45A_EXTENSION.md](STAGE45A_EXTENSION.md). It adds nine deterministic controlled matrices at `n=64,256,1024` and the retained SuiteSparse `sherman5` (`n=3312`) and `add32` (`n=4960`) runs. `results/stage45a_extension/ASSESSMENT_EXTENSION_WITH_APPLICATION.json` reconciles 22 raw CPU/GPU records: all are `solver_success`, all independently pass, with 1,068,784 Decimal80 checks and zero discrepancies. The 18 controlled records carry manufactured diagnostics; application records use `equal_unit2` and do not claim forward-truth validation.

Native construction differs across some CPU/GPU runs: `sherman5` differs at one final pattern column with equal `P` nnz; `add32` differs in 92 columns with `P` nnz 24554/24595. Application QR checks are sampled, not exhaustive. The original `gre_115` and `steam3` qualifications remain unchanged. Explicit CTest reruns now pass CPU 4/4 and CUDA 5/5; CPU required host execution after a sandbox `BAD_COMMAND`. No new algorithm or benchmark work occurred. Stage 4.5B remains unstarted and requires approval.
