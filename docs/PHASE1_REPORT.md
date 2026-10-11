# Phase 1 — Existing Work Investigation

Date: 11 October 2026.  Scope: literature and read-only source inspection only. No algorithms were implemented, changed, built, or benchmarked.

## Evidence reviewed

The priority full texts were read: Carson and Khan's [SPAI–GMRES–IR preprint](https://arxiv.org/pdf/2202.10204) (arXiv v2, 31 August 2022), and Khan and Carson's [published BSPAI–GMRES–IR article](https://link.springer.com/article/10.1007/s00366-025-02187-z). The latter is the authoritative Phase 1 source where it differs from its earlier arXiv v1. Prior GPU/MPI SPAI was checked through [HybridSPAI](https://www.tbench.org/single/1-9.pdf). Source inspection used the fixed-SPAI repository at commit `b25e6f7c1cf1c3035686e9adb5ef9f35e2256ae1` and the CUDA SPAI candidate at `350c67622288c3168b4d3b8332c3ee9d4dfa8f29`.

## Verified research baseline

### SPAI–GMRES–IR

The original method constructs a sparse approximate inverse through independent column least-squares problems (QR), adaptively expanding sparsity using construction tolerance `epsilon`, maximum pattern-growth steps `alpha`, and new indices `beta`. It constructs on the scaled transpose and transposes the result to form a left preconditioner `P`; iterative refinement uses `x0 = P b` and solves `P A d_i = P r_i`.

The paper distinguishes construction/initial-solve (`u_f`), outer working (`u`), residual (`u_r`), GMRES (`u_g`), and matrix/preconditioner product (`u_p`) precisions. Construction success depends on the construction precision, tolerance, and conditioning; the analysis does not promise successful construction or a GMRES convergence-rate bound for every sparse matrix.

### BSPAI–GMRES–IR

**Verified:** “BSPAI” here means *bucketed* SPAI, not block SPAI. After SPAI construction, preconditioner entries are partitioned by magnitude using bucket tolerance `epsilon_b`. Each bucket's dot products use its assigned precision and partial results accumulate in the highest bucket precision. `epsilon` (construction) and `epsilon_b` (bucketing) are separate parameters. The paper applies `A` uniformly and the preconditioner adaptively.

The published experiments use **unrestarted GMRES** to match the analysis; the paper states fixed-restart GMRES is not generally backward stable. Any restarted variant is therefore a future, separately validated proposal rather than a faithful baseline.

## Existing implementations and reuse boundary

| Item | Verified status | Reuse / limitation |
|---|---|---|
| [Noaman67khan/SPAI-GMRES-IR](https://github.com/Noaman67khan/SPAI-GMRES-IR/tree/b25e6f7c1cf1c3035686e9adb5ef9f35e2256ae1) | BSD-2-Clause MATLAB baseline. Fixed constructors: `spai_dd.m`, `spai_ss.m`, `spai_hh.m`; IR/GMRES and experiment utilities are present. MATLAB, Advanpix, Higham's `chop`, and Multi_precision_NLA_kernels are required. | Useful specification/comparison oracle after ambiguity resolution. It is serial, has dense allocations in paths, and contains no C++, CUDA, MPI, OpenMP, or BSPAI bucketing implementation. |
| [Noaman67khan/ADAP-SPAI-GMRES-IR](https://github.com/Noaman67khan/ADAP-SPAI-GMRES-IR) | Returned 404 through public GitHub page and API on 11 October 2026. | Adaptive source, license, and exact drivers were **not inspected**. This does not establish whether it was renamed, private, or removed. |
| [HybridSPAI](https://www.tbench.org/single/1-9.pdf) | Published MPI/CUDA multi-GPU static-pattern SPAI: index discovery, submatrix extraction, QR, triangular solve, assembly, and MPI gathering; evaluated with GPU BiCGSTAB. | Establishes that parallel/GPU SPAI construction is prior art. It is not BSPAI–GMRES–IR and cannot be taken as a numerical or precision-semantic baseline. |
| [nbz406/SPAI](https://github.com/nbz406/SPAI/tree/350c67622288c3168b4d3b8332c3ee9d4dfa8f29) | C++/CUDA Visual Studio project using CUDA runtime, cuBLAS, CUB, batched double QR and triangular solves, plus Python experiments. | Candidate design reference only: no license file/API license was found; no GMRES–IR or adaptive precision behavior was verified. Do not copy code without license clarification. |
| Other established software | Grote SPAI 3.2 was announced as C/MPI with MATLAB interfaces; PETSc exposes `PCSPAI`; ViennaCL has OpenCL SPAI; Loe et al. implement mixed-precision GPU GMRES–IR with Trilinos/Kokkos. | These differ in construction, preconditioner, precision semantics, or solver. Their existence is not proof of equivalence to this baseline. |

The local workspace initially contains only project guidance and the paper index; it has no repository, build system, test suite, implementation, or CUDA code.

## Verified gaps and cautions

The authors explicitly identify: small numerical examples rather than intensive high-performance large-scale study; a need to determine which matrices and parameters benefit; decoupling storage from computation precision; and extension to other preconditioners/adaptive operations (for example triangular solves). BSPAI can reduce storage while increasing GMRES iterations or give little benefit; the paper expects clearest benefit when `u_f = u`.

No inspected source was a complete C++/CUDA BSPAI–GMRES–IR implementation. This is a scoped-search finding, **not proof that none exists**. It is also not a novelty claim: GPU/MPI SPAI construction is documented prior work.

Before any native baseline, resolve these evidence-backed discrepancies: the MATLAB construction loop/tolerance differs from paper pseudocode (`1:alpha`, `< epsilon` versus `0:alpha`, `<= epsilon`); the MATLAB routine calls absent `findscaling`; paper unit roundoff differs from MATLAB `eps`; and its diagnostic reference/residual paths downcast or compute parts in double. The BSPAI preprint/table also has matrix-name and precision-setting inconsistencies; use the published article and pinned reproduction drivers/data.

## Recommended implementation strategy — proposal, not begun

1. Freeze a reproducible CPU reference first: Matrix Market/SuiteSparse loader, exact scaling/transposition convention, deterministic SPAI pattern growth and QR least-squares construction, uniform-precision SPAI application, unrestarted GMRES, IR, and independent diagnostics.
2. Match a selected published configuration before adding CUDA: separate construction, storage, application, accumulation, GMRES, outer, and residual precision controls. Retain a uniform-SPAI baseline beside BSPAI.
3. Add BSPAI bucketing from the published specification and compare bucket boundaries, counts, dropped entries, nnz, outer steps, and inner iterations against the reference.
4. Only after numerical agreement, introduce CUDA components incrementally (SpMV/application first, then construction). Treat reductions, FMA, subnormals, FP16 behavior, transfer costs, and ordering as validation variables. Do not assume MATLAB `chop` matches hardware arithmetic.
5. Evaluate parallel construction designs against prior GPU/MPI SPAI literature; any claim of improvement needs an apples-to-apples baseline and end-to-end measurements.

## Proposed validation approach — pending approval

### Reproduction set and published settings

Use the reported SuiteSparse matrices: `pores_3`, `steam1`, `steam3`, `saylr1`, `bfwa782`, `cage5`, `gre_115`, `orsreg_1`, `sherman4`, and BSPAI's `hor_131`. The RHS is all ones normalized to unit 2-norm. Use identity SPAI pattern, natural ordering, `beta=8`, `alpha=ceil(n/beta)`, scale columns of `A^T` to maximum magnitude one, construct on `A^T D`, and form `P=M^T D`.

For the original baseline, published configurations include `(u_f,u,u_r)` of `(single,double,quad)`, `(half,single,double)`, and `(single,single,double)`, with `u_g=u_p=u`; GMRES tolerances are `1e-4` for single and `1e-8` for double. For published BSPAI tables: double working uses `epsilon_b=2^-53, 2^-37` with double/single/half/drop buckets; single working uses `epsilon_b=2^-24, 2^-18` with single/half/drop buckets. Pin all exact matrix hashes, drivers, paper revision, dependency versions, and construction tolerances before execution.

### Independent checks and acceptance evidence

Use a genuinely high-precision CPU reference without downcasting for truth comparisons. Independently recompute at every outer step: true residual, normwise and componentwise backward error, forward error, and preconditioned/inner residual histories. Component tests should cover SPAI column residuals, QR solutions, candidate selection, scaling/transposition, bucket membership, dropping, and mixed-precision SpMV.

Compare numerical trajectories and structural quantities (preconditioner nnz, bucket counts, outer steps, GMRES iterations), not only final answers or bitwise values. Add separately labelled stress cases for zero diagonal, ill-conditioning, dynamic range, cancellation, empty buckets, exact residual, and nonconvergence. Define acceptance thresholds before measurements; report all nonconvergence and regressions.

After correctness, time CPU/uniform-SPAI/BSPAI variants separately for construction, bucketing, SpMV/application, GMRES, transfers, and total solution. Measure indices/metadata/workspace as well as values. Record hardware, compiler/CUDA/library versions, flags, seeds, warmups, synchronization/timing scope, repetitions, median, and spread. A proposed initial policy is three warmups and at least ten timed repetitions.

## Phase gate

Phase 2 is **not started**. Approval is required before any algorithm, implementation, benchmark, or source acquisition/modification work begins.
