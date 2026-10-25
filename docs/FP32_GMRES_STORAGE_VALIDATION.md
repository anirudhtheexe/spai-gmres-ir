# FP32 CUDA GMRES storage-growth validation

Date: 25 October 2026. Validation owns this record and
`tests/test_fp32_cuda_gmres.cpp`; production changes are independently owned.
This assesses the storage-only proposal in `FP32_GMRES_STORAGE_PLAN.md`.
The binding numerical contract remains `EXPERIMENT2A_FP32_SPEC.md` and
`PHASE2A_ALGORITHM_SPEC.md`.

## Sources and boundary

Full text consulted: Carson and Khan, *Mixed Precision Iterative Refinement
with Sparse Approximate Inverse Preconditioning* (2023), Algorithm 3.1,
https://arxiv.org/pdf/2202.10204; Khan and Carson, *Mixed Precision Iterative
Refinement with Adaptive Precision Sparse Approximate Inverse Preconditioning*
(2025), https://arxiv.org/pdf/2307.03914; and Loe et al., *A Study of Mixed
Precision Strategies for GMRES on GPUs* (2021),
https://arxiv.org/pdf/2109.01232. The papers support the GMRES-IR context;
16-column demand-grown storage is this project's proposed resource policy.
No published result is attributed to the storage policy.

## Predeclared tests

The existing fixed 3x3 checks retain strict and contract-tolerance convergence,
nonzero initial guess, diagnostic histories, and an outer IR correction. New
checks use deterministic matrices and no random seed:

1. A 40x40 upper bidiagonal matrix with unit diagonal and superdiagonal,
   identity P, and right-hand side e_last. The cap-16 solve stops at the first
   storage boundary; the cap-18 solve crosses the planned boundary at k=16.
   Both must report `iteration_limit`, retain the last candidate, and have
   history lengths equal to the cap. Their first 16 history entries must have
   identical FP32 bits. The cap-18 result is compared with the existing CPU
   FP32 implementation and an independent double-accumulated residual of
   the returned FP32 solution. The double calculation is validation-only.
2. A 12000x12000 identity A=P with one nonzero right-hand-side entry checks
   exact-initial return before Arnoldi and one-step success at default cap=n.
   Returned solutions are compared exactly and residuals independently checked.

For memory evidence, the prechange eager basis has `12000*(12000+1)*4 =
576,048,000` bytes. The final proposal uses 16-vector fixed device slabs.
Its first slab is `12000*16*4 = 768,000` bytes. These are source-owned
allocation sizes, not measured whole-device peaks. A pre/post
`cudaMemGetInfo` delta would only assess release/leak behavior and cannot
establish peak use.

## Results

The focused target rebuilt with MSVC 19.44, CUDA 13.4, CMake Release,
`CMAKE_CUDA_ARCHITECTURES=89`, strict host FP32 flags, and the existing
`build/cuda-NoPdb` `/Z7` configuration. The device was an NVIDIA GeForce RTX
4060 Laptop GPU, 8188 MiB reported VRAM, driver 616.92. CUDA runtime DLLs
were taken from `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin\x64`.
The source SHA-256 was
`19020F72E354237B201382E20C12D366B57ACC23D3EE79069FE1207297BDB2D9`;
the test SHA-256 was
`2219B0A6587B5559D9936C9FEC18B49D8D681E731B61DB59C9DF67BCD89E736F`.

Commands: activate BuildTools `VC/Auxiliary/Build/vcvars64.bat`; then run
`cmake --build build/cuda-NoPdb --target fp32_cuda_gmres_tests -j 2` with the
Visual Studio bundled CMake; prepend the CUDA DLL directory to `PATH` and run
`build/cuda-NoPdb/solver/fp32_cuda_gmres_tests.exe`. The target compiled and
linked; the executable exited 0 and printed `FP32 CUDA GMRES tests passed`.

The 40x40 cap-16 and cap-18 tests both returned `iteration_limit` at their
configured caps. The first 16 entries of each Hessenberg-estimate, original-
residual, and preconditioned-relative history were exactly bitwise equal.
The cap-18 independent double-accumulated relative residual was `0.22941573`,
far above the unchanged `1e-4` threshold. The GPU reported residual and CPU
FP32 oracle each agreed with that independent residual within `5e-5`; GPU
and CPU solution entries differed by at most the test's `5e-4` bound.
The 12000x12000 exact-initial case returned at iteration 0, and the zero-
initial case succeeded at iteration 1 with an exactly matching sparse unit
solution and independently zero residual. The previously existing 3x3
strict-tolerance, nonzero-initial, and IR checks passed in the same run.

Source audit of `src/fp32_cuda_gmres.cu` found eight length-n FP32 work vectors
(`8*12000*4 = 384,000` bytes) and two identity CSR uploads, each with
`(12001*8)+(12000*8)+(12000*4) = 240,008` bytes. On the exact-initial path,
no basis slab or Hessenberg storage is allocated. On the one-step path, one
16-vector slab adds 768,000 bytes, for 1,632,016 bytes of these enumerated
source-owned device allocations, excluding cuSPARSE workspace, allocator
overhead, handles, and vendor-internal allocations. The projected host H has
`17*16*4 = 1,088` float-data bytes, plus `cs`, `sn`, and `g` float data of
196 bytes and container overhead. Before the change, the eager basis alone
was 576,048,000 bytes on both paths; for the one-step path, the fully sized
host H float data was another 576,048,000 bytes. The final slab allocation
never relocates earlier basis vectors during growth.

These figures are statically verified source-issued byte counts, not a live
CUDA allocation trace or total process/device peak. The validation-only SPAI
allocator audit includes a `.cu` source clone and separate CMake target;
porting it for GMRES would require additional build plumbing beyond the
owned test file, so no allocator interception was claimed. At this focused
test stage, no runtime comparison or whole-device peak was measured. The
subsequent full regression and runtime comparison are recorded below.

## Full regression after the final slab change

The parent rebuilt the complete `build/cuda-NoPdb` tree and ran
`ctest --test-dir build/cuda-NoPdb --output-on-failure` with
`CUBLAS_EMULATE_DOUBLE_PRECISION=0`: all eight CUDA suites passed. The four
submission solver CLIs passed 32 independent Decimal80 backward-error and
error-path checks through `tests/check_solver_cli.py`; the FP32 GPU CLI's
independent backward error on the small fixture was approximately
`1.4499757216065174e-8`. This establishes bounded numerical regression
coverage for the production slab implementation. No total GPU peak, host RSS,
or high-iteration memory experiment was measured. Runtime timing is recorded
separately below.

## Same-build runtime comparison, 25 October 2026

The subsequent user question authorized a focused runtime comparison. A
validation-only script reconstructed the immediate pre-storage-change eager
allocation body from the current FP32 CUDA source. The two translation units
differ only in basis/Hessenberg storage management and its pointer access
(10 inserted and 35 removed lines in the reconstructed baseline diff); the
GMRES arithmetic and stop logic are shared. Current-source SHA-256 is
`19020F72E354237B201382E20C12D366B57ACC23D3EE79069FE1207297BDB2D9`;
reconstructed baseline SHA-256 is
`EAA64B7E696283BB31BC9F899E09027897857AC4EF317FA961C3EB6154239401`.
The reconstruction script, exact baseline source, timing harness, and 180 raw
JSONL samples are saved under `results/fp32_gmres_storage_20261025/`; raw
sample SHA-256 is
`416E868A411C4CA4CCCE8E4A74CE3C32C1C0F27A59EB6F570E9C07CAC653A299`.

Environment: Windows x64, MSVC 19.44.35228, CUDA 13.4.59, RTX 4060 Laptop
sm89, driver 616.92. Both FP32 CUDA bodies used NVCC `-DNDEBUG`, sm89,
`--fmad=false`, and host `/O2 /MD /EHsc /fp:strict /Z7`; the common harness
used MSVC `/O2 /MD /Z7 /EHsc /std:c++17`. Both linked the same FP32 CPU and
CUDA SpMV libraries from `build/cuda-NoPdb` and the same CUDA runtime,
cuBLAS, cuSPARSE and cuSOLVER libraries. No random inputs or seeds were used.
The harness constructs matrices outside timing, synchronizes before and after
each public GMRES call, performs three warmups followed by twelve measured
calls per process, and ran four serial processes in baseline/current/current/
baseline order for each mode. It hashes status, iteration count, full solution,
all three exposed histories, and final residual diagnostics as stored FP32
bits. Each mode had exactly one signature across both variants and every run.

| Input and cap | Baseline median | Slab median | Observation |
| --- | ---: | ---: | --- |
| `n=12000` identity, exact initial solution, cap=`n` | 16.634 ms | 0.907 ms | ~18.3x lower call time; zero iterations |
| `n=12000` identity, zero initial solution, cap=`n` | 220.103 ms | 1.331 ms | ~165x lower call time; one iteration |
| `n=40` upper bidiagonal, `b=e_last`, cap=18 | 11.564 ms | 11.598 ms | indistinguishable at observed spread; crosses first slab boundary |

The exact-initial case returned status code 2 and zero iterations; the one-step
case returned code 0 and one iteration; the cap-18 case returned code 6 and
18 iterations. Per-process medians and every individual sample are in the raw
file. The large early-convergence difference includes avoiding a 576 MB device
basis allocation and, for the one-step case, 576 MB of host Hessenberg zero
initialization. The current slab path has no observed runtime penalty at 18
iterations on this input, but the result does not prove equal performance for
longer or other solves. This is public-call wall timing, not isolated GPU
kernel time or end-to-end SPAI-GMRES-IR timing. GPU clocks/power were not pinned;
the paired order and warmups reduce, but do not remove, run-to-run effects.

## Predeclared full-pipeline timing follow-up

To answer the whole-solver question, compare the same reconstructed eager
GMRES object and production slab GMRES with the **same final FP32 GPU SPAI,
SpMV, and CPU libraries**. Use the archived deterministic
`stencil_8192.mtx` (SHA-256
`90d25d6332cc3729030c13e89deb74169b13aef8c1fe89b1ffa4af6709f09b1f`)
and archived Experiment 2A harness in `gpu_pipeline` mode with periodic7 RHS,
`epsilon=.3`, `alpha=1024`, `beta=8`, `innercap=8192`, `outercap=20`.
Measure the complete public pipeline call with GPU synchronization, one warmup
and three retained trials per process, running baseline/current/current/
baseline serially. Before comparing medians, require matching construction
status, preconditioner nnz, fit/growth counts, IR status, outer and inner step
counts, backward error, and binary32-roundtripped solution. Retain raw output,
build commands, hashes, and environment. Treat this as one representative
matrix, not a prediction for all matrices.

### Full-pipeline result

The archived harness source (`pipeline_benchmark.cpp`, SHA-256
`002A799F12EEE26A16BAEB0800B1FFFB2F3E48FF043C43778D6BBDC07BA327C5`)
was linked twice with identical final FP32 CUDA SPAI, SpMV and CPU libraries;
one executable used the reconstructed eager GMRES object and one used the
production slab GMRES library. Both used MSVC `/O2 /MD /Z7 /EHsc /std:c++17`;
the GMRES objects used the flags recorded above. The executable SHA-256 values
were `AAFFF177D2763C108B389016BEE6D09955519C0C4665CCABD80AB462C5D87327`
(eager) and `649A4549CCC87C4A0720724DC6662BF94A51EB7C24CB165757B4A46DB5FE2BAE`
(slab). The four raw JSONL files are retained under
`results/fp32_gmres_storage_20261025/`, along with the harness source.

| Serial process | Measured complete-pipeline times (ms) | Median (ms) |
| --- | --- | ---: |
| Eager first | 2574.141, 2526.665, 2602.676 | 2574.141 |
| Slab second | 2378.246, 2352.679, 2392.172 | 2378.246 |
| Slab reverse first | 2201.410, 2228.931, 2203.246 | 2203.246 |
| Eager reverse second | 2797.402, 2676.270, 2727.955 | 2727.955 |

Across the six measured calls per variant, the pooled medians were
`2639.473 ms` eager and `2290.805 ms` slab: **13.21% lower elapsed time**
(`1.15x` speedup) on this complete FP32 GPU SPAI-GMRES-IR workload. Both
orderings favor the slab version, though the per-process medians shifted
materially, so this small experiment does not support a universal percentage.
All 16 samples (warmups included) returned construction success and IR success,
40,197 P nonzeros, 16,257 fits, 8,065 growths, two outer steps and 15 total
inner iterations. After timing fields were removed, every JSON result had the
same SHA-256
`33c21e9f0f358670c9e35748e041c94f1b42b63c7f8825d3d3af0e69a331f9a9`,
including backward error and the binary32-roundtripped solution. No solver
precision, cap, preconditioner or stopping criterion was changed.

This is one archived matrix/RHS and the tested Windows RTX 4060 Laptop stack.
It includes SPAI construction and CPU-staged outer refinement, but excludes
matrix-file parsing and process startup. GPU clocks/power were not pinned;
no kernel-level attribution, uncertainty interval, other matrix-family result,
or general end-to-end speedup was established.
