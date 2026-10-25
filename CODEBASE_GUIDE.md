# Codebase guide

The project contains separate uniform FP32 and FP64 SPAI?GMRES?IR solvers for
CPU and CUDA. See README.md for current build/run commands. Contracts remain in
`docs/PHASE2A_ALGORITHM_SPEC.md` and `docs/EXPERIMENT2A_FP32_SPEC.md`; the old
append-only guide is preserved in `safeguards/cleanup_20261019/source_before.zip`.

## Source map

| Component | FP64 | FP32 |
| --- | --- | --- |
| CSR/input | `src/cpu.cpp`, `include/bspai/cpu.hpp` | CSR in `src/fp32_cpu.cpp`, `include/bspai/fp32.hpp` |
| CPU SPAI/GMRES/IR | `src/spai.cpp`, `src/gmres.cpp`, `src/ir.cpp` | `src/fp32_cpu.cpp` |
| CUDA SpMV | `src/cuda_spmv.cu` | `src/fp32_cuda_spmv.cu` |
| CUDA GMRES | `src/cuda_gmres.cu` | `src/fp32_cuda_gmres.cu` |
| CUDA SPAI | `src/cuda_spai.cu` | `src/fp32_cuda_spai.cu` |
| CUDA outer integration | `src/cuda_ir.cpp` | FP32 CUDA GMRES/SPAI translation units |
| Submission CLI | `src/solver_main.cpp`, compiled for each backend/precision | same adapter |

The FP32 GPU SPAI source specializes the shared CUDA constructor by compile-time
API/type mappings and inclusion. Keep both `.cu` files together and test both
precisions when changing that body. A large templated numerical rewrite was
avoided during submission cleanup.

## Mathematical and numerical boundaries

SPAI constructs `B=A^T D` using row-max scaling, solves adaptive reduced column
fits by Householder QR, and assembles `P=N^T D`. Sorted candidates and documented
score guards, growth budgets and exact-computed-zero rank handling are fixed.
GMRES applies `P*(A*v)` with one MGS pass and Givens rotations; convergence is
checked against a recomputed preconditioned residual. IR starts at `P*b` and
accepts only the original-system backward-error target. FP64 uses `10*2^-53`;
FP32 uses `10*2^-24`. The GPU outer refinement loop is CPU staged.

Input parsing for the submission CLIs uses the existing Matrix Market real/
integer coordinate reader. FP32 rounds canonical stored entries explicitly to
float before entering its solver, rejecting range loss to zero or infinity.
No mixed precision or BSPAI bucketing is implemented.

## Build layout

Root CMake owns the CPU libraries and optional CUDA graph. Strict arithmetic
flags and common target helpers are defined once. `cmake/CudaTargets.cmake`
adds GPU libraries/tests when `BSPAI_ENABLE_CUDA=ON`. The `cuda/` CMake entry
point adds the root with CUDA enabled and original CPU wide-oracle tests off.
There are five CPU and eight CUDA regression suites. `BUILD_TESTING=OFF`
excludes all regression/reference-only targets.

Required test files are explicit dependencies: missing tests now fail configure
rather than silently disappearing. The frozen dense Gate 3 GPU constructor in
`tests/reference/` and allocation instrumentation are validation-only.

Production programs: `cpu_fp32_ir`, `cpu_fp64_ir`, `gpu_fp32_ir`, `gpu_fp64_ir`.
The old `cuda_hybrid_ir`, `cuda_gpu_spai_ir`, `cpu_matrix_probe`, environment
probe and norm probe remain for diagnostics and existing independent checkers.

## Validation

Run CTest as in README. Python checkers use only the standard library:

- `check_release_cpu_decimal.py`: 16 exact-bit FP64 snapshots, Decimal80 checks.
- `check_cuda_{spai,gmres,ir}_decimal.py`: independent numerical reconstruction.
- `check_cuda_{spai,ir}_cli.py`: historical CLI status/failure paths.
- `check_solver_cli.py`: new CLI contracts and Decimal80 original-system errors.

`cuda_test_env.py` discovers runtime paths from CUDA_PATH/nvcc. The snapshot
serializer constructs hexadecimal literals from IEEE bits to work around the
old MinGW runtime's broken hexfloat formatting. Independent oracles never feed
values back into a solver. See docs/CLEANUP_RECORD.md for exact outcomes.

## Retained research and limitations

Historical campaigns, input provenance, profiles, raw snapshots, report packages,
and future BSPAI plans live in `archives/research_20261019.zip`; original paths
are retained inside it. Restore with `scripts/research_archive.py restore`.
The current build intentionally excludes campaign harnesses. The pre-cleanup
source/build configurations are in the source checkpoint ZIP.

Known numerical qualifications remain: exact-score ties may select different
native CPU/GPU patterns; singular QR failure statuses can differ; unrestarted
GMRES histories and dense pattern growth can require large memory; GPU FP32
SPAI exposes only the last fit's score records and coarser failure statuses.
FP32 duplicate summation still uses its original sort order. No new performance,
robustness, published mixed-precision, or full BSPAI reproduction claim is made.
