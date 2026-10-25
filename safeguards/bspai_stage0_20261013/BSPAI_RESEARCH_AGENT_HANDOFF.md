# BSPAI literature-research handoff

Research only. Do not edit production code, tolerances, benchmark runners, or evidence.

## Starting sources

1. Khan and Carson, *Mixed Precision Iterative Refinement with Adaptive Precision Sparse Approximate Inverse Preconditioning*, Engineering with Computers 41 (2025), DOI `10.1007/s00366-025-02187-z`; published PDF and arXiv `2307.03914`.
2. Carson and Khan, *Mixed Precision Iterative Refinement with Sparse Approximate Inverse Preconditioning*, SIAM J. Sci. Comput. (2023), DOI `10.1137/22M1487709`; arXiv `2202.10204`.
3. Local `PAPER_INDEX.md`, `docs/PHASE1_REPORT.md`, `docs/PHASE2A_ALGORITHM_SPEC.md`.

## Required answers

1. Identify the exact norm in `T_k=epsilon_b ||P||/u_k` and cite page/equation/algorithm.
2. Confirm final-scaled-P versus pre-conversion bucket membership.
3. Capture endpoints, dropped-value convention, partial-product/accumulator precision, and `P*b` initialization policy.
4. Establish roles `u_f,u,u_r,u_g,u_p` and all precision combinations actually tested.
5. Locate, license-check, and commit-pin official adaptive source if available.
6. Separate MATLAB experimental facts from native CUDA proposals.
7. List explicit mismatches requiring a future decision against this project’s growth, rank, score-clamp, stopping, and FP16 policies.

## Integration facts

Current final P is canonical CSR `SpaiResult::preconditioner`; CPU GMRES uses `P*(A*v)` at `src/gmres.cpp:90`, and GPU GMRES uploads immutable CSR P through `CudaGmresContext`. Any future proposal must preserve a uniform adapter and may not invent threshold norms or mixed-format SpMV behavior.
