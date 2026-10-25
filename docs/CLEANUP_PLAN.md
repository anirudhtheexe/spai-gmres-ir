# Submission cleanup plan — 19 October 2026

All five stages completed. Execution and independent delivery validation are
recorded in CLEANUP_RECORD.md and CLEANUP_PACKAGE_VALIDATION.md. The final
source ZIP is the Teams deliverable; the working checkout retains recovery data.

## Authorization and invariants

The user authorized a full cleanup, readable code, testing, smaller disk use,
and a Teams submission package. Retain CPU FP32, CPU FP64, CUDA FP32 and CUDA
FP64. This supersedes earlier audit-only restrictions. No new numerical method,
precision policy, tolerance, benchmark campaign, or manuscript work is planned.
All pre-existing dirty and untracked work belongs to the project.

## Stages and acceptance

1. **Inventory/checkpoint:** capture initial Git status, complete file/size
   inventory, and a recovery ZIP of source/configuration/documentation before
   editing. Read bootstrap, contracts, current source and validation records.
2. **Baseline:** fresh CPU and CUDA builds/tests; record environment and actual
   failures, including the documented MSVC long-double oracle limitation.
3. **Source/build cleanup:** independently review source; consistently format
   compressed code without changing tokens or arithmetic; simplify duplicated
   CMake definitions and isolate historical experiment tools. Provide portable
   build instructions and a usable solver entry point for each backend/precision.
4. **Evidence/storage:** archive historical research files losslessly at their
   original relative paths. Verify every archive member with SHA-256 before
   removing its loose original. Keep Git history. Remove only generated build
   trees, root object and caches after verification/testing.
5. **Validation/package:** rebuild and run meaningful unit, independent Decimal,
   and CLI checks. Create an explicit-source-allowlist submission ZIP, extract it
   to a clean directory and verify it builds and runs. Record limitations and
   final sizes; update the handoff after each material milestone.

## Resume protocol

Read PROJECT_HANDOFF.md first, then this plan and CLEANUP_RECORD.md. Inspect
current Git status; never reset existing edits. Archive manifests and checkpoint
files live in safeguards/cleanup_20261019/. Do not delete evidence unless the
verified archive already contains matching bytes. Numerical tests run serially
on GPU; no benchmark processes are authorized by this cleanup.

## Focused guard correction plan

The source review identified missing finite-vector checks in FP32 CPU/GPU SpMV,
missing GPU output checks, and unchecked size products. Correct only these
invalid-input/allocation boundaries. Keep finite arithmetic, duplicate summation
order, numerical tolerances and diagnostic mappings unchanged. Add unused-column
NaN/Inf, output-overflow, and maximum-row-count fixtures; run CPU/GPU FP32
regressions plus all original suites and independent residual checks. Avoid
enormous allocations in boundary tests. Deferred items are unstable duplicate
ordering and the documented limited GPU SPAI score/status history.
