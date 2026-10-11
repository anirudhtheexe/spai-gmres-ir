# BSPAI–GMRES–IR Research Project Guidance

## Fresh-session bootstrap

Before taking any implementation, numerical, research, validation, or manuscript action, read [PROJECT_HANDOFF.md](PROJECT_HANDOFF.md) and [CODEBASE_GUIDE.md](CODEBASE_GUIDE.md). Then consult [PAPER_INDEX.md](PAPER_INDEX.md), the relevant algorithm specification under `docs/`, the current source, and the applicable validation record. Do not rely on prior conversation history as project state.

Use the project-local specialist definitions under `.codex/agents/` when independent research, implementation, validation, or manuscript work is needed. Keep their responsibilities separate: research establishes source-backed facts; implementation owns requested production changes; validation independently tests without changing the implementation under test; manuscript owns only the IEEE paper and bibliography.

Record every future milestone, material numerical decision, validation result, environment assumption, discrepancy, and unresolved issue in project files. Update `PROJECT_HANDOFF.md` when the project state or immediate next milestone changes, so a later session can resume without access to this conversation.

This project reproduces and optimizes BSPAI–GMRES–IR with parallel computing and CUDA. Its primary literature entry point is [PAPER_INDEX.md](PAPER_INDEX.md). Begin with priority papers 1 and 2, then use the GPU, MPI/CUDA, GMRES, SpMV, and foundational references relevant to the task.

## Research-first workflow

- Consult the relevant full-text papers before making or recommending algorithmic, numerical, precision, convergence, preconditioning, parallelization, or CUDA design decisions. If full text is unavailable, state that limitation and do not present abstract-level information as verified implementation detail.
- Clearly separate established research results from this project's proposed contributions. Cite the source paper(s) for established methods, assumptions, and reported results; label hypotheses, extensions, and optimizations as proposed until validated.
- Prioritize numerical correctness, stability, and reproducibility before performance optimization. Do not treat a speedup as success if it changes the mathematical method, precision semantics, convergence behavior, or solution quality without explicit investigation.
- Validate implementations against trusted numerical references, such as a high-precision or well-established CPU implementation, published test cases, and independently checked residual/backward-error calculations. Compare both convergence behavior and final numerical accuracy.
- Make experiments reproducible: record matrix/source data, hardware and software environment, precision configuration, solver and preconditioner parameters, stopping criteria, random seeds, commands, and measurement methodology.
- Document every material change, its rationale, test results, numerical comparisons, performance results when applicable, and unresolved problems or limitations. Keep claims proportional to the evidence collected.
- Use subagents for independent literature review, baseline validation, CUDA/parallel design investigation, or experiment review when that provides a meaningful independent check. Reconcile their findings against the papers and recorded evidence.

## Scope discipline

- Treat the Carson–Khan BSPAI–GMRES–IR work as the central baseline. Consult prior GPU/MPI SPAI work before claiming novelty for parallel construction or CUDA implementation choices.
- Preserve a clear baseline-versus-proposal comparison in code, documentation, and reports.
- Before changing an algorithm, precision policy, stopping rule, or benchmark methodology, document the intended change and validation plan.
- Report unresolved numerical discrepancies promptly; do not hide them with altered tolerances, silent fallbacks, or selective benchmarks.
