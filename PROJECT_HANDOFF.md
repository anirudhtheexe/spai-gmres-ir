# BSPAI–GMRES–IR Project Handoff

**Handoff date:** 11 October 2026  
**Authoritative status:** Phase 2B Gate 3 is complete and independently validated. Do not begin Stage 4 without explicit user authorization.

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

CUDA source components are:

- `src/cuda_spmv.cu`, `include/bspai/cuda_spmv.hpp`: cuSPARSE binary64 CSR SpMV.
- `src/cuda_gmres.cu`, `include/bspai/cuda_gmres.hpp`: GPU-resident vector GMRES. It uses a project safe-scaled device norm because the tested cuBLAS 13.7.0 norm path mishandled local subnormal fixtures.
- `src/cuda_ir.cpp`, `include/bspai/cuda_ir.hpp`: CPU-staged outer IR with persistent GPU GMRES context, plus GPU-SPAI integration.
- `src/cuda_spai.cu`, `include/bspai/cuda_spai.hpp`: GPU numerical SPAI construction using independent bounded stream slots, cuSOLVER QR, GPU residual/scoring and GPU `P` assembly.
- `src/cuda_hybrid_main.cpp`: shared Matrix Market CLI. `cuda_hybrid_ir` uses CPU SPAI; `cuda_gpu_spai_ir` enables GPU SPAI.

`cuda/CMakeLists.txt` defines the separate x64 CUDA build. The CPU root `CMakeLists.txt` remains the independent reference build.

### Gate 3 result

The final independent Gate 3 evidence is in [PHASE2B_VALIDATION_RECORD.md](docs/PHASE2B_VALIDATION_RECORD.md): 1,534 GPU-SPAI C++ checks, 436 independent 80-digit SPAI checks over 60 fits, 43 independent 80-digit IR checks, and 15 CLI checks passed. CUDA memcheck reported zero errors/leaks and initcheck zero errors. Previously validated suites also passed: CPU 946 checks, SpMV 323, GPU GMRES 871, and hybrid IR 665; CUDA CTest 4/4 and CPU CTest 4/4.

These are deterministic, small-matrix, uniform-double correctness results. They are not performance numbers, scalable-GPU claims, general high-dimensional robustness proof, or a published mixed-precision reproduction.

## Known limitations and unresolved discrepancies

1. **Stage 4 is not implemented.** Current GPU SPAI uses dense `O(n^2)` B/N/P staging plus as many as eight dense QR workspaces. It is not a scalable sparse-memory implementation.
2. GPU SPAI remains hybrid host controlled: sorted patterns, candidate metadata, mean/filter/tie decisions, and adaptive scheduling remain on CPU. Stream telemetry proves independent work was submitted before collection, not actual hardware overlap or speedup.
3. The outer iterative-refinement loop is CPU staged. The final GPU P is downloaded/canonicalized and uploaded into the persistent GMRES context once; no direct device-to-device ownership transfer is claimed.
4. On literal singular `[[1,2],[1,2]]`, CPU reports `rank_deficient` while GPU reports `empty_candidates` because GPU Householder QR produces a tiny finite pivot. Both fail; no partial P is supplied to GMRES. Do not add a threshold, structural fallback, or CPU fallback to conceal this documented difference.
5. The tested cuBLAS 13.7.0 `Dnrm2` path gave incorrect local subnormal-norm results. The intentionally serial safe-scaled device norm is a correctness baseline, not a performance optimization.
6. BSPAI bucketing, mixed precision, benchmark methodology, speedups, and publication-quality experiments remain unimplemented.
7. `PAPER_INDEX.md` has links and citations but the project `papers/` directory has no locally available PDFs at this handoff. Full texts must be lawfully reacquired and review status updated before relying on new literature claims.
8. The project folder is not a Git repository. There is no transferable commit history, remote, branch state, or worktree metadata. Create/version-control a copy only under explicit authorization.

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

For independent CUDA numerical checks, use the exact commands in the Gate 3 validation record, including `tests/check_cuda_spai_decimal.py`, `tests/check_cuda_spai_cli.py`, Compute Sanitizer memcheck/initcheck, and `build/validation-cuda`. These records include pinned artifact hashes and full command lines.

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

**Stage 4: Scalable Sparse GPU SPAI Construction.** Replace the current dense B/N/P staging and dense per-slot QR workspaces with sparse representations and exact reduced-problem-sized QR workspaces, while preserving the frozen adaptive SPAI algorithm: scaling/orientation, pattern growth, candidate scoring, QR semantics, exact computed-pivot guard, status behavior, and `P=N^T D` assembly.

Recommended preparation after explicit authorization:

1. Ask Research to revisit prior GPU/MPI SPAI work and identify source-backed sparse storage/workspace approaches relevant to the frozen contract.
2. Ask Implementation to write a design/validation plan before changing code. Represent B and per-fit restricted submatrices sparsely; allocate QR storage from actual `|I| x |J|` dimensions rather than `n^2`; preserve independently issued column fits without treating scheduling as performance evidence.
3. Ask Validation to freeze current Gate 3 comparisons and add memory-footprint/structural tests before accepting a sparse replacement. Re-run all CPU/GPU numerical regressions and explicitly compare patterns, residuals, coefficients, P, status, and end-to-end IR.

After Stage 4: Stage 5 BSPAI bucketing and mixed precision; Stage 6 performance optimization and benchmarking; Stage 7 IEEE conference paper finalization.

## Transfer audit

Transfer the entire project directory, including hidden `.codex/agents/`, source, headers, `cuda/`, `scripts/`, tests, fixtures, docs, `manuscript/`, `PAPER_INDEX.md`, and this handoff. Build outputs in `build/` are machine-specific and nonessential; regenerate them. Do not transfer credentials, tokens, user profile data, or the prior conversation.

At handoff creation, no known worktree exists because this folder is not a Git repository. Verify this again after copying. Check the new account can access the configured models, a suitable CUDA-capable host, toolkit/compiler paths, official source links, and any Codex-specific agent support. Those account/machine services do not transfer with files.

### Final handoff audit — 11 October 2026

- The project-local `.codex/config.toml` preserves the coordinator/default setting `gpt-5.6-terra`, medium reasoning; the four full role definitions are present in `.codex/agents/` with the exact requested model identifiers and responsibilities.
- All source/header files, both CMake configurations, CUDA script, CPU/CUDA test suites, decimal/CLI validation scripts, tiny Matrix Market fixture, reports/specifications, Gate 3 validation evidence, and the IEEE template assets are present in the project folder.
- The current machine reran `cmd /d /c scripts\build_cuda_msvc.cmd`: configure/build succeeded and CUDA CTest passed 4/4. CPU CTest on `build/ninja` also passed 4/4. These commands refresh generated test logs only; no numerical source or tolerance was changed during handoff.
- `papers/` is empty. No locally cached reference repositories, MATLAB reference implementation, source archive, dependency lockfile, README/LICENSE, unified all-check runner, or project manuscript/bibliography is available. The IEEE directory is the generic template only, with its placeholder/example bibliography.
- Build directories, binaries, CMake caches, Debug runtimes, and saved test logs are machine-generated. A stale historical `LastTestsFailed.log` may exist under `build/`; the fresh CTest runs above are the current result. Regenerate build artifacts after transfer rather than using them as portable evidence.
- No active build/test processes were found during the audit. Live agent conversations, Codex account entitlements, custom-agent loading behavior, access to external papers/repositories, MATLAB/Advanpix licenses, CUDA driver/toolkit/MSVC/Python installations, and the prior conversation do not transfer with this folder.
