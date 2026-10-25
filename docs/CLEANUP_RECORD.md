# Cleanup execution record — 19 October 2026

Status: complete. The final source ZIP is ready for Teams upload.

Fresh baseline: CPU GCC 6.3.0 Debug CTest 5/5 and MSVC/CUDA Debug CTest 8/8
pass. MinGW required host execution after sandbox DLL relocation failure;
the first failed configure/no-tests invocation is not a passing baseline.
Logs are in safeguards/cleanup_20261019/. Consolidated CPU build passes 5/5.

Build changes: common strict-arithmetic interface and library/test helpers;
CPU source compiled once per build; CUDA is opt-in from root, with the old
cuda/ entry point retained. Required tests are no longer silently omitted by
EXISTS checks. Historical campaign targets are removed from the active build;
their original CMake and sources remain recoverable from checkpoints/archive.
The standalone CUDA entry excludes the three known MSVC-incompatible CPU
wide-long-double suites; all five CPU suites are tested separately with GCC.

Four thin solver CLIs delegate to existing precision-specific public APIs.
Matrix Market input is parsed/canonicalized using the shared FP64 reader and
explicitly rounded for FP32; nonzero input underflow and overflow are rejected.
Solver arithmetic, epsilon policy, stopping and correction behavior are intact.

Initial checkout: branch stage4-scalable-spai, existing dirty tracked and
untracked work. Initial largest categories: results 1,826,043,479 bytes; .git
711,311,362 bytes; build 419,322,795 bytes; data 9,780,022 bytes. Source is
165,793 bytes. Size is dominated by retained experiments and generated output.

Bootstrap, paper index, CPU algorithm and FP32 contracts, earlier cleanup audit,
and specialist role responsibilities read. Historical handoff entries contradict
later milestones; the new handoff will identify current state explicitly.

No active compiler, solver, benchmark or profiler was observed at inventory.
clang-format 19.1.7 was installed in build/cleanup-tools as a temporary cleanup
tool; it is not a project dependency. Initial sandbox network lookup failed;
the approved download succeeded.

## Source corrections after formatting

The formatting stage preserved 47,661 tokens across all 28 existing production
files. Subsequent boundary corrections are separate changes:

- FP32 CPU SpMV now rejects NaN/Inf anywhere in the input vector, including
  unused sparse columns. The added fixture failed against the original baseline
  library, then passed after correction. Row-offset count SIZE_MAX is rejected
  before allocation. Original duplicate order and finite arithmetic are intact.
- FP32 CUDA SpMV rejects nonfinite inputs and outputs; byte counts and signed
  CUDA index conversions are range checked. Tests include unused-column NaN/Inf
  and finite-input output overflow. GPU GMRES checks the Krylov element-count
  product before allocation. The enormous-dimension branch is source-reviewed;
  it was not exercised using impractically large allocations.
- The historical validation-only snapshot serializer crashed under MinGW 6.3
  before emitting records. A small std::hexfloat reproduction printed incorrect
  text for 1.25 (`2e-081`). Serialization now derives exact hex literals from
  IEEE bits using integer formatting. MinGW Decimal80 now passes all 16
  snapshots/500 checks. The prior failing invocation is retained in the logs.
- CUDA Python checkers discover runtime paths using CUDA_PATH/nvcc rather than
  the original user's hardcoded CUDA 13.4 installation directory.

No precision, tolerance, arithmetic order, QR rank decision, stopping rule,
score heuristic, fallback, pattern order or benchmark methodology was changed.
The FP32 duplicate-order and GPU diagnostic limitations remain documented.
Final flag review caught the first consolidated build applying the old FP64-only
MinGW 32-bit SSE2 guard to the FP32 library as well. It was moved back to
`bspai_cpu` before final admission, preserving original per-precision flags;
the amended extracted CPU configuration was rebuilt and passed all checks.

## Current numerical evidence

Final GCC CPU CTest: 5/5. Final MSVC/CUDA CTest: 8/8. Original baseline was
also 5/5 and 8/8. All five/eight suite names are retained; no test was removed
to achieve these results. MSVC Decimal80 snapshot: 16 records/500 checks pass
after the final serializer rebuild, independently confirmed in the extraction.

Fresh independent GPU Decimal80: SPAI 60 fits/436 checks, GMRES 48 snapshots/
383 checks plus 125 norm snapshots, IR 21 snapshots/160 checks; zero failures.
The historical CLI probes pass. New CLI checks: 32/32 across all four programs,
including explicit usage/input/cap/solver-failure exits and Decimal80 backward
error reconstruction. Errors CPU FP32/FP64: 2.53746e-8 / 5.40158e-17; GPU
FP32/FP64: 1.44998e-8 / 2.02559e-17, below their unchanged precision targets.
These are correctness checks on small deterministic fixtures, not benchmarks.
After restoring the original MinGW FP32 compiler flags, final CPU FP32 backward
error is 4.34993e-8; the other three errors remain as above. The final extracted
package passes all 32 CLI checks under that corrected configuration.

Logs/commands and failed reproductions: safeguards/cleanup_20261019/.

## Archive transaction

The explicit 61-root selection and initial inventory are in safeguards. Archive
creation, full per-member SHA-256 verification, source rechecks, and a second
full verification before removal succeeded: 1,426 files / 1,846,708,788 original
bytes -> 632,840,001 compressed bytes. The independent reviewer confirmed all
61 selections are represented and all ZIP member names are unique. The archive
tool was hardened to reject absolute/traversal/symlink paths, duplicate members,
and a source that changes immediately before deletion. All historical research
bytes remain in archives/research_20261019.zip at their original relative paths.
Packaging writes/verifies a temporary ZIP before replacing an existing output.

Archive SHA-256:
`4fb497b86b65fb4123ad9c80d8652ec17b8c0bf28d820795baed78427f329bbf`.
Manifest and integrity metadata are in safeguards. A six-category sample
restored through the actual restore function with matching hashes. Five safety
tests pass: changed-source prune refusal, lossless restore/overwrite refusal,
duplicate member rejection, protected/traversal path rejection, and preservation
of a completed submission when a mandatory source file is missing. The first
safety-test attempt hit sandbox default-temp permissions; tests now use a
verified project-local generated temporary directory.

## Git storage

Ran `git -c gc.reflogExpire=never -c gc.reflogExpireUnreachable=never gc --no-prune`.
All 1,258 real objects remain, including unreachable objects in a cruft pack;
no commits, tags, branches, reflogs, or objects were pruned. HEAD remains
`8eac10eec719bf24d55e219884d7a35bcd25aaaa`. Before/after full fsck reports only
the same historical dangling blobs, with no corrupt/missing object errors.
After packing, Git identified 19 interrupted `tmp_obj_*` files as garbage.
Their exact paths/sizes/SHA-256 were recorded, containment and no active Git
writer checked, then 216,758,492 generated bytes were removed. Final Git object
count: 1,258 packed, zero loose, zero garbage; pack size 431.05 MiB. Git history
and all dirty source work remain recoverable.

## Generated output and documentation

Removed 17 stale/generated build subdirectories after saving their path/count/
byte inventory, plus the generated root .obj. Active extracted-package builds
and the temporary formatter are retained until their checks finish, then removed.
Current-document local Markdown links and source formatting checks pass. The
append-only handoff/guide were replaced with concise current entry points;
their full originals remain in source_before.zip. Historical reports keep their
original claims/limitations, and docs/README.md explains archived paths.

The submission ZIP is deliberately source-only and builds independently of the
large evidence archive. It includes all four backends, the deterministic fixture,
all five CPU/eight GPU suites, exact-bit/Decimal80 checkers and project role
instructions. The ZIP manifest and clean extraction checks are the delivery
gate; no generated executable, cache, Git object or profile belongs in it.

## Final independent delivery gate

See CLEANUP_PACKAGE_VALIDATION.md for the independent fresh-extraction record.
CPU 5/5, CUDA 8/8, all four CLI checks 32/32, GCC/MSVC Decimal80 500/500 each,
the documented Windows CPU helper (2/2 plus 500 Decimal checks), and both CPU
and CUDA production-only builds pass. Production-only CTest inventories are
zero and contain no frozen reference or test targets. The final CMake amendment
was staged into that extraction and retested. The final ZIP's code/build/test
bytes are compared to the staged tested extraction; the newly added safety
suite is separately tested 5/5. Final documentation is included when regenerated.

Approximate disk change: 2.98 GB originally to 1.09 GB now (decimal units,
about 63% reduction), including all retained evidence and Git objects. Of the
remaining checkout, the research archive is 632,840,001 bytes and .git is
452,061,550 bytes. Active code/tests/docs are under 1 MB; the Teams ZIP is
about 0.31 MB. Exact final sizes, manifests and final ZIP SHA-256 are in
safeguards/cleanup_20261019/final_*.json/csv. The older sibling backups outside
this workspace were not modified or included in these size figures.

No open cleanup issue remains. Known numerical limitations are preserved in
CODEBASE_GUIDE.md, the contracts, independent review and historical validation
record. No benchmark/sanitizer/performance campaign was rerun for this cleanup.
