# BSPAI Stage 0 feasibility audit

**Scope:** read-only architecture and literature audit, 13 October 2026. No BSPAI implementation, numerical-policy change, test, benchmark, or profiler run occurred.

## Target method

The target is Khan and Carson, *Mixed Precision Iterative Refinement with Adaptive Precision Sparse Approximate Inverse Preconditioning*, *Engineering with Computers* 41 (2025), 3833-3846, DOI `10.1007/s00366-025-02187-z` (published PDF and arXiv `2307.03914` accessible). BSPAI means **bucketed** sparse approximate inverse, not block SPAI.

The paper constructs SPAI first and buckets final P using separate `epsilon_b`. With unit roundoffs `u_1 < ... < u_q`, Eq. (2.4) defines `T_k=epsilon_b ||P||/u_k`; `B_1=(T_2,infinity)`, intermediate `B_k=(T_(k+1),T_k]`, and `B_q=[0,T_q]`. Equal thresholds enter the lower-precision bucket. Bucket products use their bucket precision and partial sums accumulate in `u_1`; `u_q=1` represents drops. These are paper facts, not implementation choices for this repository.

Before a numerical specification is approved, establish the threshold norm, conversion/FMA/subnormal rules, post-conversion membership semantics, dropped-value representation, and initialization application policy from full text or official source. The adaptive MATLAB repository remains unverified.

## Architecture and integration seam

| Component | Existing implementation | BSPAI requirement | Disposition |
|---|---|---|---|
| CPU SPAI | `src/spai.cpp:253`; `SpaiResult::preconditioner` in `include/bspai/spai.hpp` | Bucket completed P | Reuse unchanged |
| CUDA SPAI | `src/cuda_spai.cu:179`; `cuda_construct_spai` returns canonical host P | Bucket same final P | Reuse unchanged; no fallback |
| FP32 | Separate `src/fp32_cpu.cpp:21`, `src/fp32_cuda_spai.cu:101` | Explicit future precision-role policy | Do not merge into frozen FP64 API |
| Sparse storage | Sorted CSR `CsrMatrix` / `Fp32CsrMatrix` | Bucket indices, values, metadata, source-P hash | New abstraction |
| Application | CPU `P*(A*v)` at `src/gmres.cpp:90`; GPU immutable CSR P in `CudaGmresContext` | Same operator contract with bucket arithmetic | New adapter interface |
| GMRES / IR | `left_preconditioned_gmres`, `refine_with_preconditioner`; CUDA analogues | Consume adapter, retain Krylov/IR contracts | Integrate only after standalone validation |
| Serialization / checks | Matrix Market P and SHA-256 in Stage 4.5/Experiment 2A runners; Decimal80 checkers | Versioned bucket format and membership/application checkers | New BSPAI-only artifacts |

The lowest-risk future seam is an immutable preconditioner-application interface: a uniform-CSR adapter preserves current behavior, while a bucketed adapter implements the paper’s published arithmetic. This avoids duplicating GMRES. The first bounded milestone is CPU-only: consume an existing canonical P, construct the published bucket representation, and verify application to vectors—no GMRES or IR.

## CUDA assessment

Recorded target: RTX 4060 Laptop (CC 8.9), CUDA 13.4.59, cuSPARSE 12.8.6, cuBLAS 13.7.0, cuSOLVER 12.3.2. Existing CSR SpMV supports uniform `CUDA_R_64F` / `CUDA_R_32F`, not a mixed bucket operator. Options—one CSR per bucket, tagged packed arrays, or custom kernels—remain open. Separate CSR is easiest to validate but adds launches; packed/custom forms require defined conversion and accumulation semantics.

The primary numerical risk is unverified threshold/norm/rounding behavior at precision boundaries, especially FP16 subnormal/overflow cases. The primary performance risk is multiplying tiny kernels and host/device coordination; existing evidence already identifies launch/transfer coordination as material. Storage savings alone do not imply solver speedup.

## Validation, benchmarking, and isolation

First validate exact membership (including open/closed endpoints), zeros/drops, sign, extrema, subnormals, overflow/underflow, sparse structure, serialization, and CPU/GPU application bounds. Then assess fixed-P GMRES convergence and IR backward error; do not require bitwise equality to uniform P. Use a fresh BSPAI-only results root and distinguish application timing from complete solver timing.

Do not create a worktree yet: HEAD omits material uncommitted/untracked baseline content. After a verified snapshot and plan review, use a separately copied development directory with isolated build, results, locks, docs, and checkpoints.
