# Project handoff - source review, 24 October 2026

## FP32 CUDA GMRES persistent context, 25 October 2026

The latest requested optimization was to reuse unchanged FP32 GMRES CUDA
state across outer refinement corrections. Read
`docs/FP32_GMRES_CONTEXT_PLAN.md` and the independent focused gate plus final
results in `docs/FP32_GMRES_CONTEXT_VALIDATION.md`. Production now exposes
`Fp32CudaGmresContext` in `include/bspai/fp32_cuda_gmres.hpp`. It retains one
device copy of A and P, a stream, cuBLAS/cuSPARSE handles and descriptors,
eight device work vectors, and cuSPARSE workspaces. The FP32 IR loop constructs
it lazily for the first required correction and reuses it thereafter. The
public one-shot API remains available. Krylov basis/Hessenberg storage stays
per solve; arithmetic and stopping rules were not changed.

The saved pre-context slab source and reproducible benchmark inputs/scripts
are in `results/fp32_gmres_context_20261025/`. Independent focused tests found
bitwise equality for repeated context and one-shot solves, including invalid
inputs and cap 16/18 transitions; source-issued CSR upload count stayed at two
per context. Full CUDA CTest passed 8/8 after rebuild, and all four solver
CLIs passed 32 Decimal80 checks. On archived `stencil_8192`, two corrections
and 15 total inner iterations took 10.50–10.85 ms median per baseline run and
9.71–9.80 ms per context run with fixed P, about 7–10% less time for this
isolated refinement loop. Complete GPU SPAI–GMRES–IR run medians stayed around
2.15–2.16 s with no clear overall change. Prechange and context results had
identical final FP32 solution and diagnostics within each benchmark scope.
These timings cover one matrix and GPU; no general speedup is established.
Invalid context construction still lacks a dedicated test, and a broader
matrix campaign would be needed before any general performance claim.

## FP32 CUDA GMRES storage growth, 25 October 2026

The user subsequently requested an investigation of eager FP32 GPU Krylov
basis and host Hessenberg allocation at maximum iteration cap. Read
`docs/FP32_GMRES_STORAGE_PLAN.md` and the independent
`docs/FP32_GMRES_STORAGE_VALIDATION.md` before further changes. Production
`src/fp32_cuda_gmres.cu` now allocates no basis/Hessenberg before its
zero-iteration checks. It adds stable 16-vector device slabs and grows the
host projected system in 16-column chunks as Arnoldi reaches them. The
unrestarted algorithm, cap, arithmetic, convergence tests, and statuses are
unchanged. A contiguous-growth draft was discarded because reallocation
could nearly double transient basis memory near the cap.

Independent validation passed the FP32 focused test with a cap-16/cap-18
boundary, exact bitwise history prefixes, CPU and independently recomputed
residual comparisons, and `n=12000` exact-initial/one-step cases. The complete
CUDA CTest suite passed 8/8 and all four solver CLIs passed 32 Decimal80 and
error-path checks after the final source rebuild. At `n=12000`, the old eager
basis request was 576,048,000 bytes; the one-step case now requests one
768,000-byte basis slab and 1,088 bytes of host H float data. Exact-initial
return requests neither. These are source-owned allocation counts, not live
whole-device/host peak measurements; no runtime speedup or high-iteration
resource result was measured at that validation milestone.

The 25 October follow-up measured runtime with an isolated, same-flags
pre-change FP32 GMRES body and the production slab body. Raw trials, exact
input generator, source reconstruction and build recipe are retained in
`results/fp32_gmres_storage_20261025/`; interpretation is in
`docs/FP32_GMRES_STORAGE_VALIDATION.md`. On a deterministic `n=12000` identity
problem, median public-call wall time fell from 220.103 ms to 1.331 ms for a
one-iteration solve and from 16.634 ms to 0.907 ms for exact-initial return.
On a 40x40 cap-18 bidiagonal case crossing a slab boundary, medians were
11.564 ms before and 11.598 ms after, indistinguishable at measured spread.
All exposed numerical result signatures matched bitwise in every trial.
This confirms a large early-convergence call-time benefit on the named
inputs, not an end-to-end SPAI-GMRES-IR or general-matrix speedup. The next
milestone, if requested, is a representative matrix campaign or live total
device/host peak measurement; do not infer either from these three cases.

The subsequent whole-solver timing follow-up used the archived `stencil_8192`
matrix and Experiment 2A `gpu_pipeline` harness with the same final GPU SPAI
and SpMV libraries in both executables. It isolated the old eager versus new
slab GMRES storage body, ran serial baseline/current/current/baseline with
one warmup and three measured trials per process, and retained raw files in
`results/fp32_gmres_storage_20261025/`. Six-run pooled complete-pipeline
medians were 2639.473 ms eager and 2290.805 ms slab: 13.21% less time
(`1.15x`) on this one FP32 workload. Both orderings favored the slab version;
run-to-run medians varied. All 16 outputs matched in status, construction
counts, two outer steps, 15 inner iterations, backward error and serialized
FP32 solution. Details and caveats are in
`docs/FP32_GMRES_STORAGE_VALIDATION.md`. Do not generalize this percentage to
other matrices, hardware, or a multi-run uncertainty interval without a
representative campaign.

## SPAI FP32 residual optimization decision, 24 October 2026

The user subsequently authorized investigation and improvement of the saved
post-batching FP32 SPAI `fitted_residual` and `qr_guard` hotspots. This narrow
request supersedes the earlier no-new-benchmark/no-numerical-work restriction
for these kernels only. Read `docs/SPAI_ROW_KERNEL_OPTIMIZATION.md` and the
independent `docs/SPAI_ROW_KERNEL_VALIDATION.md` before continuing. The shared
`src/cuda_spai.cu` body now computes each fitted row residual once and reuses
it for scoring and both diagnostics while retaining the original one-thread
launch, sparse lookup, ascending-row safe norm, and QR guard. FP32 and FP64
compile from the same body. An isolated row-parallel/dense-cache candidate
returned identical public FP32 bits but was 5–7% slower on the archived
stencil_8192 input, so it was not retained. The immutable research archive and
previous dirty work remain untouched.

The final fused-only source built with MSVC 19.44/CUDA 13.4 for sm89 in
`build/cuda-NoPdb`. The normal Debug and Release helper configurations failed
before project compilation due MSVC C1902 PDB manager mismatch. Local CMake
Debug-flag overrides `/Z7` for C++ and CUDA allowed a Release build. All eight
CUDA CTest suites passed; independent Decimal80 SPAI 436 checks, IR 160 checks,
four CLI 32 checks, and the sparse allocation audit passed after final rebuild.
The FP32 focused test accepted one outer correction with final backward error
1.98682145e-07. In independent same-flags isolated builds, a public FP32
constructor dump was bitwise identical across pre-change, row-parallel, and
fused-only variants. Three fused-only stencil constructor trials had median
2153 ms against 2282–2307 ms baseline medians (exploratory 5–7% lower), with
unchanged peak source-issued memory and allocation count. No kernel-time,
FP64 performance, other-matrix, or end-to-end speedup is established. The
immediate next milestone is any broader, predeclared FP32 matrix campaign if
one is requested; keep the smaller fused-only change as the measured outcome.

## Read-only project review, 24 October 2026

The current project was mapped without rebuilding, rerunning solvers, changing
numerical source, or extracting the research archive. The active implementation
is four uniform-precision SPAI-GMRES-IR paths (CPU/GPU, FP32/FP64); BSPAI and
mixed precision remain future work. The presentation source package also exists
under `presentation/`; see its PACKAGE_RECORD.md for scope and caveats.

Delivery discrepancy discovered during this review: the current
`submission/SPAI_GMRES_IR_Submission.zip` is 147,400 bytes with 64 source
members plus its manifest. The recorded final package in
`safeguards/cleanup_20261019/final_submission_integrity.json` is 313,131 bytes
with 99 source files. The current ZIP's manifest paths all still exist in the
working tree; its `.gitignore` and `README.md` hashes differ from live files.
The other 62 manifest-listed live files match. This checkout's submission ZIP
must therefore not be represented as the final validated package or uploaded
without regeneration and an independent package check. No ZIP was rewritten
during this review. The original dirty working tree was left intact.

## Active authorization and status

The user authorized a complete cleanup, readable code, testing, reduced disk
usage and a Teams source submission, retaining CPU/GPU FP32/FP64. They permit
subagents only with gpt-6-sol and asked for unattended decisions and a resumable
record. Earlier audit-only/stop restrictions are superseded within this scope.
Do not start new algorithms, benchmarks, BSPAI, mixed precision or manuscript
work. Never reset the pre-existing dirty/untracked work.

Read AGENTS.md, CODEBASE_GUIDE.md, PAPER_INDEX.md, docs/CLEANUP_PLAN.md and
CLEANUP_RECORD.md. Consult the numerical specs and validation record before any
numerical action. Local specialist responsibilities remain in .codex/agents;
this cleanup used the user-authorized gpt-6-sol model override.

## Completed so far

- Initial inventory, Git status and source checkpoint:
  `safeguards/cleanup_20261019/` (source_before.zip is the pre-edit recovery copy).
- Independent source/build/archive reviews in docs/CLEANUP_*REVIEW.md.
- All 28 original production files formatted; 47,661 original tokens matched.
- Consolidated CMake, optional CUDA, explicit required tests, portable Windows
  tool discovery, and four thin solver CLIs.
- Reproduced and fixed FP32 unused-column NaN acceptance; added finite-input/
  output and allocation size guards. Accepted finite arithmetic is unchanged.
- Corrected the validation snapshot serializer's old MinGW hexfloat problem
  using exact IEEE-bit hexadecimal output, independently round-trip checked.
- Lossless research ZIP: 1,426 files / 1,846,708,788 original bytes, compressed
  to 632,840,001 bytes. Every file SHA-256 verified before loose originals were
  removed. The archive preserves original relative paths and a full manifest.
- Final core regressions pass CPU 5/5 and CUDA 8/8. Decimal80 gates and all four
  CLI backward-error/error-path checks pass. An independent fresh extraction
  passes both CPU and CUDA builds/tests, all 32 CLI checks, 500-check snapshot
  oracles on GCC/MSVC, and production builds with tests disabled.

## Delivery state

Cleanup work and its recorded validation are complete, but the current
`submission/SPAI_GMRES_IR_Submission.zip` differs from the recorded final
package as described above. Regenerate and independently verify it before any
Teams upload.
README.md describes build/run commands and package scope. Final submission
integrity, source comparison, directory inventory and Git status are saved under
`safeguards/cleanup_20261019/`. The large evidence archive and Git history remain
in the working checkout; all generated build trees and caches are removed.
No solver, build, profiler, benchmark or background maintenance remains active.
Future work now begins with a step-by-step presentation-preparation walkthrough.
The first teaching milestone is a source-map and executable call-chain explanation;
then cover SPAI construction, GMRES, iterative refinement, CUDA differences, and
the distinction between active regression evidence and archived experimental
results. No numerical extension or benchmark rerun is authorized.

Four role-aligned beginner briefing sources for a four-person presentation team
now live in `presentation/team_guides/`. They are intended for PDF conversion:
algorithm/research, CPU FP64 implementation, FP32/CUDA variants, and
validation/results/delivery. They state the active implementation boundary
explicitly: uniform-precision SPAI-GMRES-IR is implemented; BSPAI and mixed
precision are not active implementation claims.

`presentation/team_guides/00_overall_project_and_code_flow.md` is the shared
master briefing for all four members. It documents the repository map, CPU
FP64 execution graph, algorithm stages, active variant scope, recorded
validation evidence, archive boundary, and presentation-safe wording.

On 25 October 2026, beginner foundations were added at the start of all five
team guides. They define source code, headers, CMake/build/test, CPU/GPU,
FP32/FP64, matrix/vector/sparse data, and the high-level algorithm before the
more detailed role-specific sections.

Git packing completed without pruning objects or reflogs. All 1,258 objects
remain; full fsck passes. Removed only 19 Git-reported interrupted temporary
files after recording their paths/sizes/hashes. No commits, branches or tags
were rewritten. See docs/CLEANUP_RECORD.md and CLEANUP_PACKAGE_VALIDATION.md.

## Recovery and retained evidence

`archives/research_20261019.zip` contains historical results/data/provenance,
profiles, benchmark and validation campaign tools, root planning/research
reports, manuscript template and report packages. Use
`python scripts/research_archive.py verify` or `restore`; restore refuses to
replace a changed file. Do not add the archive to the Teams source ZIP.

The old append-only handoff/guide and all source/build files are in
`safeguards/cleanup_20261019/source_before.zip`. Earlier independent sibling
backups recorded by the old handoff are outside this task's writable workspace
and were not modified. All old raw results retain their original claims and
limitations. Current numerical contracts are still the FP64 Phase 2A and separate
FP32 specification, not bucketed/mixed-precision BSPAI.

Known qualifications: CPU/GPU native patterns may differ at ties; singular QR
statuses differ; FP32 GPU score history/status mapping is limited; FP32 duplicate
summation order remains original; original MSVC FP64 CTest wide-oracle suites
require GCC/Clang instead. Unrestarted GMRES can use substantial memory.

## Tested environment

Windows x64; GCC 6.3 CPU (32-bit, SSE2 binary64 flags), MSVC 19.44.35228;
CUDA Toolkit 13.4.59; RTX 4060 Laptop architecture 89; driver 616.92; CMake
3.31.6, Ninja 1.12.1, Python 3.12. Host execution was required after sandbox
DLL relocation failures. GPU tests run serially; no benchmark was launched.

## Presentation teaching package, 25 October 2026

The four role-aligned beginner guides and shared overall code-flow guide under
`presentation/team_guides/` were reviewed for unexplained jargon and improved
with reading routes plus role-specific glossaries. They explain source code,
headers, build/CMake/test, sparse matrices, CPU/GPU, FP32/FP64, numerical
terms, active implementation scope, and evidence boundaries before detailed
content. PDFs were generated in `presentation/team_guides/pdf/` with the local
`presentation/render_team_guides.py` renderer. Five outputs have valid PDF
signatures and nonzero page counts; no solver code or numerical result was
changed by this documentation work.

## Implementation-detailed teaching package revision, 25 October 2026

All five Markdown teaching guides were expanded after comparison with the
current source, algorithm specifications, validation records, and current
handoff. Each guide remains beginner-first but now includes exact active call
routes, CPU/GPU ownership, data movement, stopping and verification equations,
precision policy, status/failure interpretation, and scope boundaries. The
validation/delivery guide now discusses the recorded CPU 5/5 and CUDA 8/8
CTest results, Decimal80 evidence, deterministic-fixture errors, persistent
FP32 GMRES context measurements, lazy 16-vector slab measurements, fused SPAI
measurements, reproducibility requirements, and the current submission ZIP
discrepancy. Claims are explicitly limited to the recorded matrices and GPU.

`presentation/render_team_guides.py` now renders display mathematics through
Matplotlib mathtext images instead of stripping LaTeX notation. The five PDFs
were regenerated. Structural checks found valid `%PDF`/`%%EOF` boundaries,
12/10/7/6/8 pages respectively, and embedded equation images in every guide.
This was documentation-only work; no solver source or numerical behavior was
changed and no new solver benchmark was run.

## Parallel implementation teaching expansion, 25 October 2026

The five teaching guides were expanded again at the user's request with
top-to-bottom file/function/API flows, input parameters, returned result
structures, and function-internal explanations. The CUDA guide now explicitly
labels host-serial control, custom GPU-kernel parallelism, vendor-library GPU
parallelism, cross-stream column concurrency, synchronization points, and
host/device transfers. It documents Runtime, cuSPARSE, cuBLAS, and cuSOLVER
calls; persistent and per-solve buffer ownership; SPAI/GMRES telemetry; and the
recorded sparse-storage, stream-slot, fused-residual, rejected row-parallel,
persistent-context, and lazy-basis-slab optimization history. The validation
guide connects those changes to recorded timing and numerical-equivalence
evidence and states missing broader parallel-performance studies.

The PDFs were regenerated without changing the renderer or the established
math presentation. Structural checks found valid PDF boundaries and page
counts of 13/11/9/10/10 for overall/algorithm/CPU/CUDA/validation. No solver
source, numerical policy, tests, or benchmark results were changed.
