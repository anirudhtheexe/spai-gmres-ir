# Documentation index

Current entry points:

- [Cleanup record](CLEANUP_RECORD.md): changes, validation, size and recovery.
- [Cleanup plan](CLEANUP_PLAN.md): scope, invariants and staged execution.
- [Source review](CLEANUP_SOURCE_REVIEW.md): independently identified issues.
- [Build/archive review](CLEANUP_INDEPENDENT_REVIEW.md): independent safety review.
- [Formatting verification](CLEANUP_FORMATTING.md): original token comparison.
- [FP64 algorithm contract](PHASE2A_ALGORITHM_SPEC.md).
- [FP32 algorithm contract](EXPERIMENT2A_FP32_SPEC.md).
- [CPU validation plan](PHASE2A_VALIDATION_PLAN.md).
- [Historical GPU validation](PHASE2B_VALIDATION_RECORD.md).

CPU/CUDA milestone reports in this directory remain as historical descriptions
of the numerical implementation. Their original build/data/result paths describe
the research checkout before cleanup. Historical raw files and root reports are
inside `archives/research_20261019.zip`, at their original relative paths. Use
the root README and codebase guide for current build commands and file layout.

The compact submission contains contracts and tests. Large raw research evidence
and machine-specific checkpoint logs stay in the working checkout.
