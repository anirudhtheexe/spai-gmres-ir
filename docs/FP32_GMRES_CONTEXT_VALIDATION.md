# FP32 CUDA GMRES persistent-context validation

Date: 25 October 2026. This validation record covers the proposed reuse of
one FP32 CUDA GMRES context for multiple corrections with fixed canonical
`A` and `P`. The binding arithmetic and stopping rules are in
`EXPERIMENT2A_FP32_SPEC.md` and `PHASE2A_ALGORITHM_SPEC.md`. The comparison
baseline is the existing public one-shot
`fp32_cuda_left_preconditioned_gmres` API, with identical matrices, right-hand
sides, options, and initial guesses. This is a resource-lifetime change
proposed by the project, not a result claimed from Carson and Khan.
The full text of [Carson and Khan's SPAI-GMRES-IR paper](https://arxiv.org/html/2202.10204)
and the [published Khan and Carson BSPAI-GMRES-IR article](https://link.springer.com/article/10.1007/s00366-025-02187-z)
were consulted for the residual/correction/update and preconditioner-reuse
context. Neither source specifies this CUDA object lifetime or bitwise API
regression criterion.

## Predeclared validation plan

1. On a fixed nonsymmetric 3x3 `A` and identity `P`, solve multiple distinct
   right-hand sides through one context. Include a nonzero initial guess,
   exact initial solution, and repeated first right-hand side after intervening
   calls. Compare status, convergence flag, message, iteration count, final
   original and left-preconditioned residuals, solution, and all three exposed
   histories with one-shot calls. Compare float payloads bitwise, including
   NaN diagnostic defaults on error paths.
2. On deterministic 40x40 upper bidiagonal `A`, identity `P`, and `b=e_last`,
   solve with caps 16 and 18 through the same context, then repeat cap 16.
   Require each complete result to equal its one-shot counterpart bitwise.
   This exercises reuse across the existing basis-storage growth boundary and
   nonconverged `iteration_limit` paths.
3. Interleave invalid RHS length, nonfinite RHS, invalid initial-guess length,
   nonfinite initial guess, and invalid tolerance/cap calls with valid calls.
   Compare exact status and message with one-shot and require the later valid
   call to remain bitwise identical. Check invalid construction separately if
   the context constructor exposes exceptions rather than result status.
4. Require `matrix_uploads()==2` after construction and after every solve.
   Require `matrix_upload_bytes()` to equal the CSR offset, column and value
   payloads for both matrices: `8*(n+1)+8*nnz+4*nnz` per nonempty matrix.
   Inspect the source to ensure only context construction performs those CSR
   host-to-device copies. These counters and source audit show source-issued
   matrix transfers, not a CUDA API trace or vendor-internal transfers.
5. Run the focused CUDA GMRES test and relevant complete CUDA CTest suite
   serially after the production interface is ready. Compare the current
   frozen Decimal80 GMRES and IR checks where applicable. Record exact build
   and run commands, source/test hashes, environment and results below.

All fixtures are deterministic; no random seed or timing statistics apply.
No performance speedup is claimed by this correctness gate.

## Results

The focused test was rebuilt after the final context-destructor edit, using
MSVC 19.44, CUDA Toolkit 13.4, Release configuration, and sm89 in the existing
`build/cuda-NoPdb` tree. The host had the RTX 4060 Laptop GPU and driver
616.92 recorded in the immediately preceding storage validation. The build
ran `vcvars64.bat`, then the Visual Studio bundled CMake:

```text
cmake --build build/cuda-NoPdb --target fp32_cuda_gmres_tests -j 2
ctest --test-dir build/cuda-NoPdb --output-on-failure -R ^fp32_cuda_gmres_tests$
```

The CUDA runtime `bin/x64` directory was prepended to `PATH` for CTest. Build
and link succeeded; CTest passed 1/1. The test executable printed the existing
nonzero-initial estimate `3.16252368e-07`, recomputed relative residual
`6.9005452e-08`, and cap-18 independent relative residual `0.22941573`.
These are pre-existing numerical checks retained alongside the new context
checks.

The fixed 3x3 context passed ten calls: two distinct right-hand sides, an
exact-initial solve, six invalid input/options calls, and a repeat of the
first nonzero-initial solve. Every exposed result field matched a separately
created one-shot call bitwise, including NaN diagnostic payloads on invalid
paths. The 40x40 context passed cap-16, cap-18, and repeated cap-16 solves;
each result matched its one-shot counterpart bitwise. The existing CPU and
independently recomputed residual checks also passed. Test fixtures have no
randomness; no timing was measured.

The 3x3 context reported exactly two matrix uploads and 184 CSR payload bytes
after construction and after every call. The 40x40 context reported two
matrix uploads and 2,084 CSR payload bytes after construction and after all
three calls. A source audit found the row-offset, column and value H2D calls
only in `Matrix` construction, reached from the context's `Run` constructor;
the solve body contains RHS and initial-vector H2D calls but no matrix H2D.
The byte counters are calculated by the implementation and were not checked
against a CUDA API trace; vendor transfers and allocator activity remain
outside this claim.

Source SHA-256: `src/fp32_cuda_gmres.cu`
`B01643621AB0044D148732D4682809983936A339DB316EAE283991DC1521C79B`;
`include/bspai/fp32_cuda_gmres.hpp`
`E3FCBD0892E75F90019E9C2F7209F5BDC8BB59650439BEB566853AA2FBA57360`;
`tests/test_fp32_cuda_gmres.cpp`
`2BD33ED2C38717B51F281BAC2BCE866B8F771F9BEB346E479A313A3B516D635D`.

The one-shot API now constructs a fresh instance of the same context body.
Therefore its bitwise comparison independently tests reuse/reset behavior,
while the existing CPU and residual checks constrain numerical behavior; it
is not a second CUDA arithmetic implementation or a saved prechange binary.
The focused test did not directly prove that the outer IR loop takes multiple
corrections on these inputs or exercise invalid context construction. A later
archived multi-correction timing run covers the former, below. Invalid context
construction remains untested.

## Complete regression and archived multi-correction timing

The parent rebuilt the full `build/cuda-NoPdb` tree with MSVC 19.44, CUDA
13.4, sm89 and the local `/Z7` configuration, then ran `ctest --test-dir
build/cuda-NoPdb --output-on-failure` with CUDA runtime DLLs on `PATH` and
`CUBLAS_EMULATE_DOUBLE_PRECISION=0`. All eight CUDA suites passed. The four
submission CLIs passed `tests/check_solver_cli.py`: 32 Decimal80 backward-error
and error-path checks, zero failures. The FP32 GPU CLI's independently checked
backward error was `1.4499757216065174e-8`. Build/run command scripts are in
`results/fp32_gmres_context_20261025/`.

The immediate pre-context slab GMRES source was saved before editing as
`results/fp32_gmres_context_20261025/fp32_cuda_gmres_before_context.cu`
(SHA-256 `19020F72E354237B201382E20C12D366B57ACC23D3EE79069FE1207297BDB2D9`).
The same Experiment 2A harness was linked once with that source and once with
the current context source. Both used NVCC `-DNDEBUG -arch=sm_89 --fmad=false`
and host `/O2 /MD /EHsc /fp:strict /Z7`, plus the same FP32 CPU, GPU SPAI and
SpMV libraries. Hardware was the RTX 4060 Laptop GPU, driver 616.92. The
archived `stencil_8192` matrix had SHA-256
`90d25d6332cc3729030c13e89deb74169b13aef8c1fe89b1ffa4af6709f09b1f`.
The right hand side was `A*x`, with deterministic periodic-seven `x`; SPAI
epsilon 0.3, alpha 1024, beta 8, GMRES tolerance `1e-4`, cap 8192, outer cap
20. No random seed applied. Runs were serial in baseline/context/context/
baseline order; each fixed-P process had two warmups and twelve measured
samples, and each complete pipeline process had one warmup and three measured
samples. Synchronization bracketed every measured public API call. The fixed
P was constructed once by CPU SPAI outside timing and serialized to
`build/fp32-context-bench/stencil_8192_P.mtx`; complete-pipeline runs built P
with GPU SPAI inside timing. See the archived scripts and raw JSONL files for
commands and all samples.

The fixed-P A had 40,576 nonzeros and P had 40,197. Their two CSR payloads
totaled 1,100,364 source-issued H2D bytes per context construction (64-bit
offsets/columns and FP32 values). With two corrections, the old one-shot
path issued that matrix payload twice; the persistent path issued it once.

| Timed scope | Baseline run medians (ms) | Context run medians (ms) | Interpretation |
| --- | ---: | ---: | --- |
| Fixed-P refinement, two corrections, 15 total GMRES iterations | 10.853, 10.504 | 9.798, 9.710 | Roughly 7–10% lower medians for this isolated correction loop |
| Complete GPU SPAI–GMRES–IR pipeline | 2164.834, 2160.272 | 2152.202, 2163.933 | No clear whole-pipeline time change at this spread |

All samples in both scopes had `success`, two outer steps, two inner attempts,
and 15 inner iterations. Within each scope, all serialized FP32 solutions had
one SHA-256 value and all final independently recomputed residual/backward-
error diagnostics were identical. The fixed-P result had backward error
`7.40717851e-8`, while the complete pipeline using GPU SPAI had the same
backward error but a different P and solution. The local fixed-P result thus
checks numerical equivalence against the saved prechange CUDA body as well
as the new one-shot API. `summarize.py` prints per-run medians, hashes, and
diagnostics from all raw samples.

The performance evidence is one matrix on one GPU, with small millisecond
differences and no confidence interval. The context removes source-issued
matrix uploads and setup on later corrections, but SPAI construction dominates
the complete measured pipeline. No general whole-solver speedup is claimed.
