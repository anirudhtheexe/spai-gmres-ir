# Phase 2B Milestone 5A: CUDA SpMV validation

Scope: CPU-to-device canonical CSR transfer and binary64 cuSPARSE nontranspose SpMV only. No GPU SPAI/GMRES/IR, mixed precision, BSPAI, performance, or published-table reproduction is in scope. Validation owns `tests/test_cuda_spmv.cpp` and this record; production code/manuscript are not modified.

## Pre-execution checks and proposed acceptance policy

Deterministic fixtures: nonsymmetric3x3, rectangular5x4 with leading/internal/trailing empty rows, Matrix Market duplicate-canonicalized4x3, allzero3x5,1x1, tiny normal/subnormal dyadic2x2,257-entry long row, and decimal/cancellation inputs. Exact representable integer/dyadic cases require exact CPU/GPU/reference values; tiny values receive no absolute-one tolerance floor. Decimal cases compare independently trusted CPU CSR output with GPU output using per-row differential bound

`4*gamma(2*m+1)*sum_j |a_ij*x_j| + m*denorm_min`, `gamma(k)=k*u/(1-k*u)`, `u=2^-53`, where m is row nonzeros.

This is a documented project acceptance threshold, not a published or vendor accuracy guarantee. Maximum tested row length257. It allows reduction/FMA differences and does not establish high relative accuracy in cancellation-dominated rows. CPU/GPU bitwise identity is not required except exact dyadic cases. Repeated executions of identical input must be bitwise identical on the pinned GPU/library/algo; that is a separate local determinism check. No random seed, timing, warm-up, or performance methodology applies.

Each device object is exercised repeatedly, with changing RHS/zero input, rejected wrong-size/NaN/infinite inputs, and reuse after errors. Independent checks cover shape, finite output, final CUDA synchronization, invalid device, overlapping object lifetimes, repeated construction/destruction, and overflowing computed output. Lifetime exercises are not proof of leak freedom. Source inspection and compute-sanitizer, if executable, supplement runtime results; allocation-failure injection and huge index-width cases may remain untested.

Research checked official CUDA13.4 [cuSPARSE documentation section6.6.7](https://docs.nvidia.com/cuda/cusparse/index.html#cusparsespmv): generic CSR supports64-bit indices and uniformCUDA_R_64F; CSR_ALG2+NON_TRANSPOSE is documented deterministic; asynchronous calls require completion before host checks. Validation independently checked its algorithm table and execution properties. FMA/reduction differences from CPU are expected and caller `--fmad=false` does not control library internals. The implementation must pin the algorithm and completion/error contracts. These API facts do not claim novelty or solver equivalence.

## Initial environment and launch investigation

Read-only queries found NVIDIA GeForce RTX4060 Laptop GPU, compute capability8.9,8188MiB, driver616.92, WDDM; installed nvcc CUDA13.4.59. The driver display version is not the toolkit/library version. MSVC14.44.35207 is installed. Runtime/cuSPARSE versions will be queried by the test executable.

An initial attempt to rerun the existing four CPU CTest executables returned BAD_COMMAND for all four, before test execution. Direct invocation of `build/ninja/cpu_core_tests.exe` reported Windows `Illegal System DLL Relocation`; even `C:/MinGW/bin/g++.exe` subsequently failed to launch with that error, preventing a fresh static-runtime GCC build. This is an environment launch failure, not numerical assertion evidence. The existing946-check result remains historically verified but is not yet a Phase2B rerun. MSVC has53-bit long double and cannot silently replace the64-bit wider-oracle GCC configuration used by earlier tests.

## Authoritative final execution result

On2026-10-11, Validation independently executed the unchanged CPU binaries outside the sandbox: **37 CSR+190 SPAI+280 GMRES+439 IR=946 checks, zero failures**; CPU CTest **4/4 passed**. This isolates the earlier sandbox launch failure. The prior test hashes below remain unchanged. No host/toolchain setting or original numerical acceptance policy was changed to obtain these results.

The independent CUDA build in `build/validation-cuda` passed **323 checks, zero failures**, CUDA CTest **1/1**. Maximum CPU/GPU absolute difference was **1.1102230246251565e-16**, maximum ratio to the prescribed row-differential bound **0.012077294685990324**. Exact dyadic, tiny normal and minimum-subnormal outputs passed exact comparisons. Repeated results were bitwise identical on this pinned configuration; this does not imply cross-device/version reproducibility.

Additional final cases cover negative/out-of-range device, zero dimensions,64-bit index and byte-count overflow before allocation, moved-from malformed CPU CSR, and ownership of CSR uploaded from a temporary host matrix. Rejected inputs and computed overflow leave the object reusable in the tested cases. Zero-nnz input deliberately uses the implementation's GPU-output-clear shortcut, not a cuSPARSE zero-nnz descriptor. No transpose operation is implemented or tested.

| Environment item | Independently verified value |
|---|---|
| GPU | NVIDIA GeForce RTX4060 Laptop GPU, CC8.9,8585216000 bytes, WDDM |
| Driver |616.92; runtime driver API query13040 |
| Toolkit/nvcc |13.4.59 |
| Runtime API query |13040 |
| cuSPARSE query/header |12806 /12.8.6.49 |
| Host compiler |MSVC19.44.35228.0 x64; tool directory14.44.35207 |
| Build |Debug Ninja,CUDA architecture89, host /W4 /fp:strict on CPU library and test; nvcc host /W4,/fp:strict;CUDA_STANDARD11 |
| CUDA-test reference precision |double53 bits; long double53 bits, **not** a wider oracle; exact dyadic references plus trusted independent CPU CSR differential comparison |
| Original CPU-test oracle |Unchanged MinGW6.3.0 binary64/SSE2/no-contraction, long double64 bits |
| Compute Sanitizer |2026.3.0.0, build38637409 |

The CUDA build emitted vendor `cusparse.h` C4996 warnings for deprecated declarations; inspected project code calls generic nondeprecated APIs. These warnings were not suppressed. No project-source warning was observed in the final test compilation. Final emitted test rules explicitly include /fp:strict; the initial310-check build's test-target flag inheritance was incomplete and Implementation added that flag before final acceptance.

### Memory/error-path evidence and limits

Compute Sanitizer memcheck with full CUDA leak checks passed **321 checks, zero test failures**, **0 bytes leaked in0 allocations**, **ERROR SUMMARY:0 errors**. Its explicit `--memory-safety-only` mode skips exactly the two deliberate invalid-device CUDA API calls, leaving all numerical, dimension/index/byte overflow, finite-input/output, object-lifetime and reuse checks active. The full323-check default suite separately verifies those two invalid-device errors. No global sanitizer API-error suppression was used.

An initial raw sanitizer run of the older310-check default suite reported **3 API errors** around its deliberately invalid device and subsequent last-error observation, with zero leaked bytes. This is retained rather than described as a clean full-suite sanitizer run. The explicit sanitizer mode separates expected API-failure tests from memory checks; it does not remove a numerical discrepancy or loosen tolerances.

Independent source audit found owned nonblocking stream/handle/descriptors/buffers; checked canonical CSR/index/byte ranges; per-H2D synchronization preserving host lifetimes; checked NON_TRANSPOSE/CSR_ALG2/CUDA_R_64F calls; cleared output; checked D2H completion; and a scoped stream drain before result destruction on errors. Partial constructor resources are owned by RAII cleanup; cleanup failures are reported rather than thrown. Multi-device switching is implemented with a guard but not dynamically exercised on this single-GPU machine. Allocation-failure injection, arbitrary asynchronous hardware faults and cleanup-error injection remain untested. Passing memcheck on these paths is not a proof of all possible memory/error-path safety.

### Retained build/probe failures

The first CUDA test compilation failed MSVC C2312 because the validation helper instantiated duplicate `std::exception` catch clauses. Validation changed that test-only invalid-device expectation to the API's runtime_error type; no production change was needed. A later asynchronous build was overlapped with execution of its previous binary, causing LNK1168 (test executable locked for writing); the old310-check run is not counted as final-source validation. Validation then waited for the build and reran it successfully before final323-check execution. The sandbox configure also stalled at compiler ABI detection and was interrupted; the separate approved host-side configure succeeded. These are launch/build/test-harness problems, not hidden numerical assertion failures.

## Commands

All executable/build steps below were run outside the sandbox after its launch restriction was isolated. Configuration and build used these exact commands from PowerShell:

```powershell
cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" && "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" -S cuda -B build/validation-cuda -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_COMPILER=cl "-DCMAKE_CUDA_COMPILER=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.4/bin/nvcc.exe" -DCMAKE_CUDA_ARCHITECTURES=89 "-DCMAKE_MAKE_PROGRAM=C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/Common7/IDE/CommonExtensions/Microsoft/CMake/Ninja/ninja.exe"'
cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" && "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" --build build/validation-cuda --verbose'
cmd /c 'set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin\x64;%PATH%" && build\validation-cuda\cuda_spmv_tests.exe && "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\compute-sanitizer\compute-sanitizer.exe" --tool memcheck --leak-check full --error-exitcode 99 build\validation-cuda\cuda_spmv_tests.exe --memory-safety-only && "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\ctest.exe" --test-dir build/validation-cuda --output-on-failure'
.\build\ninja\cpu_core_tests.exe
.\build\ninja\spai_tests.exe
.\build\ninja\gmres_tests.exe
.\build\ninja\ir_tests.exe
& 'C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/ctest.exe' --test-dir build/ninja --output-on-failure
```

Preliminary raw sanitizer command was the same memcheck/leak/error-exitcode command **without** `--memory-safety-only`; its3 expected API-error reports are described above. Reproducibility inputs reside entirely in `tests/test_cuda_spmv.cpp`; no external matrix data or random generation was used. No timing or speedup result follows from CTest runtimes.

## Final SHA-256 evidence

| Artifact | SHA-256 |
|---|---|
| `tests/test_cuda_spmv.cpp` | `D659C7AA4C26E5B9F3BB3B46B74A5BF6A05D3B29B44427B5B72566661CF029EB` |
| `src/cuda_spmv.cu` | `EFD9CCC5056F4F9C15ABCA4AFEA10C4E0CB01443CDAA347AA02E7040F18EF901` |
| `include/bspai/cuda_spmv.hpp` | `5EEB4870CC4F3DC5DC703C56E1758E75618017C330279317E15F04CE12CCDD6B` |
| `cuda/CMakeLists.txt` | `F95FBE433617A1B3B40A0C8A57386AF4EFDA62674111AE18D19514EAC7E0CF17` |
| `src/cpu.cpp` | `071D6614723DFF930BBAD66F7E7C0CA9A0265B3620EE38FD607367FBF47EF37C` |
| `tests/test_cpu.cpp` | `1469F4A3BEB8E511BB26D178B5B13FB366B9CD15D0DF8F23B40488FB6BA59A86` |
| `tests/test_spai.cpp` | `AC7A22B1ED2EE86ED544E08A04CC56A7E6C7F57A4E2A1191D0F8F3FCFADADCFF` |
| `tests/test_gmres.cpp` | `137BA13A946E63ED756E3CF4B6BBF3C8B767A51D835595A8C5C627699465FD7A` |
| `tests/test_ir.cpp` | `6FC4A6341F42C7627CEB1F2B1EDD3F73DBA21291902D0EA3553532361A733EC4` |

No unresolved numerical discrepancy was found within this bounded5A contract. The one-ULP-scale CPU/GPU difference is expected and below the predeclared bound. No solver convergence, wider/high-precision general GPU accuracy, source-paper reproduction, GPU performance, or optimization success is established. Stop at5A pending further approval.

## Gate 1: GPU GMRES pre-execution validation contract

Gate1 authorizes only unrestarted left-preconditioned GPU GMRES, retaining946 CPU and323 GPU-SpMV checks. CPU-constructed SPAI may supplyP uploaded to the GPU; neither GPU construction nor outer IR is in scope. Validation owns `tests/test_cuda_gmres.cpp` and this record; no production/manuscript edits.

Proposed independent checks: compare CPU and GPU convergence/status/iterations and final solutions for identity, signed diagonal, nonsymmetric3x3, integer/noninteger/general RHS, zero/exact/nonzero initial guesses, cap failures, full-cap nonconvergence, exact Arnoldi breakdown, normalization/input/range failures, and approximate/full CPU-SPAI preconditioners. Main tau1e-8 remains unchanged. Exact/rational well-conditioned solution references require infinity-norm forward agreement within1e-12*max(1,reference norm); CPU/GPU solution differences use the same documented bound. Independent original and preconditioned residual diagnostics use fixed-order dense binary64 loops and stable hypot, with agreement allowance768*machine_epsilon times relevant data/operator scales. This allowance accounts for GPU FMA/reduction order; it does not relax the implementation's recorded recomputed acceptance threshold. Recorded accepted candidates must satisfy their actual GPU recomputed relative residual<=tau and have a verification trigger.

Separate strict-tolerance/estimate-gap fixtures target explicit nonconvergence and rejection when estimate<=tau but recomputed residual>tau. They are branch probes, not changes to the main1e-8 policy. CPU/GPU strict statuses and iteration counts can differ if finite-precision dot/norm/products differ; any such observation must be investigated and recorded rather than forced to match. MSVC long double remains53bits, so this GPU suite does not claim a general wider oracle; exact/rational references and trusted CPU comparisons are bounded evidence.

Residency audit must establish uploadedA/P, GPU vector/Krylov basis ownership, two-productP*(A*v), GPU vector updates/dots/norms, transfers and synchronization. Scalar host reads and small projected-system host work must be explicitly documented. Full periteration vector snapshots, if exposed only for validation, must be distinguished from normal resident execution; lack of snapshots limits independent periteration residual reconstruction. No performance inference follows from residency.

### Gate 1 blocker: independently reproduced cuBLAS subnormal norm discrepancy

**Gate1 is not approved.** The first independent GPU solver build passed ordinary cases but the unchanged identity RHS[2^-1030,0] failed:364 checks,6 failures (status/convergence/count,basis normalization/orthogonality,and exact tiny solution). The independent80-digit Decimal snapshot audit returned a failure because the underlying GPU test failed; its first99-check draft's original-residual absolute-one scale floor was then removed to prevent tiny norm under-reporting from passing a broad absolute allowance. Main1e-8 and strict1e-30 tolerances and the tiny fixture remain unchanged.

Validation-only `tests/probe_cuda_norm.cu` directly calls legacy `cublasDnrm2` and `cublasDnrm2_64` without any solver. It uploads and bitwise reads back[2^-k,0,...] for everyk0..1074, lengths1,2,3, DEFAULT_MATH(0) and PEDANTIC_MATH(2), HOSTpointer mode, atomics-not-allowed, CUBLAS_EMULATE_DOUBLE_PRECISION=0. Loaded library path was `C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.4/bin/x64/cublas64_13.dll`, runtime cuBLAS130700. All powersk<=1022 were exact. Every one of52 subnormal powersk1023..1074 was incorrect in all6 mode/length combinations:12900 classic/64-bit norm comparisons,312 failing mode/length/exponent cases (each involves both calls). Atk1030, exact8.6916947597937554e-311 was reported as5.1806537865363094e-318 (ratio2^-24); minimum-subnormal norm was0. Both APIs agreed with each other, not with the exact norm. This establishes a discrepancy at isolated library calls on this pinned stack, not a claim about cuBLAS internals or FTZ root cause. Other hardware/library versions are untested.

Standalone commands (approved host execution):

```powershell
cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" && "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin\nvcc.exe" -arch=sm_89 --fmad=false -Xcompiler=/W4,/fp:strict tests/probe_cuda_norm.cu -lcublas -o build/probe_cuda_norm.exe'
cmd /c 'set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin\x64;%PATH%" && set "CUBLAS_EMULATE_DOUBLE_PRECISION=0" && build\probe_cuda_norm.exe'
```

The initial probe compile lacked its own<string> include and was corrected in the validation-only source; no production change followed from that compile error.

Parent authorizes a documented deterministic safe-scaled device norm proposal for every GPU GMRES Euclidean norm, with scalar-only host observation. Predeclared primitive acceptance: exact singleton powers from minimum-subnormal through largest normalpowers; signedzero; dyadic3-4-5 vectors at exponents-1074,-1030,-1022,-512,0,512,1021; mixed minimum/maximum finite entries; naive-square overflow/underflow cases; long vectors; stable CPU hypot plus80-digit Decimal references. Singleton/dyadic expected-representable norms require exact values; general cases use rowlength-aware rounding bound `8*gamma(3*n+1)*norm+8*denorm_min`, without an absolute-one floor, and explicit ULP/absolute subnormal checks. Nonfinite entries and genuinely unrepresentable norms must fail explicitly; empty-vector behavior must be pinned. Preserve original/residual zero semantics, fixed tolerance, literal device division, all-vector residency and whole-GMRES/regression suites. This proposal is not yet validated and is not performance optimization.

### Authoritative Gate 1 corrected execution evidence

The approved single-thread deterministic safe-scaled GPU norm replaced cuBLAS norms **for every GPU GMRES norm**, without changing stopping/zero/breakdown policies. Validation independently tested the corrected implementation: **871 C++ checks, zero failures**; **383 independent80-digit Decimal checks, zero failures**, covering48 GMRES final/history snapshots and125 norm fixtures. The initial failed vendor-norm solver/probe evidence above remains valid and retained; this correction does not establish that the vendor primitive itself is fixed. Gate1's scoped correctness evidence is now satisfied; further integration requires separate authorization.

The unchanged identity RHS[2^-1030,0] now succeeds in one iteration with exact stored solution and valid basis normalization. All singleton subnormal powers2^-1023..2^-1074, smallest normal2^-1022, signed singleton/zero/empty cases and selected normal/large powers pass exact norm checks. Dyadic3-4-5 norms, mixed max/min finite values,257-entry vector, naive-square underflow/overflow cases pass the predeclared stable-hypot/80-digit bounds. Repeated norms agree exactly. Truly overflowing norms and nonfinite input fail explicitly. This serial safe-scaled recurrence is a correctness/debugging project choice, not a claim of efficient parallel norm reduction or a bitwise port of modern Netlib's multiphase norm.

Main cases agree with the strict-host CPU GMRES status/convergence/iteration count and exact/rational references. Maximum CPU/GPU solution infinity difference **4.4408920985006262e-16**; maximum independently recomputed relative-residual difference **1.1102230246251565e-16**. Approximate CPU-SPAI P converges in3 GPU iterations, full CPU-SPAI P in1; uploaded preconditioners are not GPU-constructed. Strict tau1e-30 returns explicitnonconvergence with final relative3.0067321592990152e-16. A separately declared finite grid{1e-17,1e-16,2e-16,5e-16} exercises actual estimate-gap rejection at tau2e-16: estimate1.085730288427049e-16 <=tau, recomputed3.0067321592990152e-16 >tau; verification occurs and candidate is not accepted. The main1e-8 and strict1e-30 configurations were never relaxed.

Decimal conversion uses exact stored binary64 inputs via `Decimal.from_float`,80 significant decimal digits, independent dense products/norms and no production solver calls to derive reference residuals. Maximum80-digit original-norm difference2.660747733157166089e-16; maximum80-digit relative-preconditioned-residual difference1.121829756956074676e-16. C++ initial/percandidate/final original-norm comparisons and Decimal original-norm comparisons have no absolute-one floor; tiny absolute scales are preserved. Higher-precision acceptance comparisons retain the declared rounding allowance rather than claiming GPU and exact-real residuals coincide.

Complete Gate1 memcheck executed all871 checks with **0 test failures, ERROR SUMMARY:0 errors,0 bytes leaked in0 allocations**. No invalid-device exclusion mode or sanitizer suppression was used for this suite. Independently rerun regressions passed **946 CPU checks**, CPU CTest4/4; **323 SpMV checks**, combined CUDA CTest2/2. Previous CPU/SpMV validation sources/hashes remain unchanged. No timing/performance interpretation follows from CTest durations.

Residency verified by source inspection and counters: A/P each upload their immutable CSR arrays before Arnoldi; rhs/initial vectors upload before Arnoldi; all Krylov/work/candidate vectors and history snapshots are GPU allocations. Operator application is two NON_TRANSPOSE/CSR_ALG2/FP64 products P*(A*v), never materializedP*A. MGS dots and vector copies/updates use cuBLASFP64; safe norms/literal vector division/finite scans use device kernels. Givens, smallH/triangular solves and scalar decisions are host-controlled. Full histories/basis are copiedD2D during iteration and downloaded **only after termination**, so their independent audit does not insert periteration vectorD2H.

Typical3-iteration case:8 initial uploads (six CSRarrays+rhs+initial),8 posttermination vector downloads (finalsolution+4basis+3candidates), **0 Arnoldi vectorH2D/D2H**,6 dot,12 norm,39 finite-flag host observations. Scalars are synchronizing observations: cuBLASdot returns a double to host; safe norm explicitly reads one double; finite scan reads one int. Host alpha/coefficient pointers passed to cuBLAS are scalar controls, not full vector transfers, and internal library scalar movement is not independently traced by the counters. This is **vector-resident, host-controlled** GMRES, not fully asynchronous/device-only control and not performance evidence.

Native configuration: cuBLAS PEDANTIC_MATH set and queried back exactly, atomicsNOT_ALLOWED, HOSTpointer mode, dedicatedstream; CUBLAS_EMULATE_DOUBLE_PRECISION=0 explicitly set in final runs. Custom kernels compile with --fmad=false; this does not control cuBLAS/cuSPARSE internal FMA/reduction order. Environment remains the5A stack plus queried cuBLAS130700 and Python3.12. CUDA C++ comparisons use53-bit MSVClongdouble; independent80-digit Python audits supply the additional precision. Final test compile removed an MSVCgetenv warning using its documented environment-query function; numerical assertions/count remained unchanged.

Additional commands, same approved host build/environment as5A:

```powershell
cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" && "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" --build build/validation-cuda --verbose'
& C:/Users/Anirudh/AppData/Local/Programs/Python/Python312/python.exe tests/check_cuda_gmres_decimal.py build/validation-cuda/cuda_gmres_tests.exe
cmd /c 'set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin\x64;%PATH%" && set "CUBLAS_EMULATE_DOUBLE_PRECISION=0" && "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\compute-sanitizer\compute-sanitizer.exe" --tool memcheck --leak-check full --error-exitcode 99 build\validation-cuda\cuda_gmres_tests.exe'
cmd /c 'set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin\x64;%PATH%" && set "CUBLAS_EMULATE_DOUBLE_PRECISION=0" && "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\ctest.exe" --test-dir build/validation-cuda --output-on-failure && build\validation-cuda\cuda_spmv_tests.exe'
```

The Python harness sets installed runtimeDLLpath and emulation0 for its child, reads stdout snapshotJSON without creating result files, and fails if the test executable fails or snapshots are missing. CPU rerun commands are the unchanged four executables/CTest listed under5A.

| Gate1 artifact at tested revision | SHA-256 |
|---|---|
| `src/cuda_gmres.cu` | `063883BB946A8FC71121F388387C3D3CB01E4785A5DFF9B0FCDC9B061C213C8D` |
| `include/bspai/cuda_gmres.hpp` | `9F2BF92F3C2A231C522A53A5012AA2BB71DF7B38A5B3AE015F458C36045AAB2D` |
| `tests/test_cuda_gmres.cpp` | `B2658B9952BF17C4D534571EC5B2145A37D32846A63DEF44BA9255B0CB5BFE3B` |
| `tests/check_cuda_gmres_decimal.py` | `7087E72D34FE118EC70FF25F77FD7E04DBB700B6B266A86E0A3583700A035BAC` |
| `tests/probe_cuda_norm.cu` | `8BCD9D30A380E76ADA366C988B81D2F1FCA2F89BA920005434120520F416B142` |
| `cuda/CMakeLists.txt` | `87DEEB75DABA28F83E9BC00237CD8C624C1C87727647660C24D7C6CD096E4066` |

No unresolved corrected-solver discrepancy was found within this boundedGate1 scope. Limitations: deterministic tiny cases rather than SuiteSparse/published table reproduction; no general high-dimension robustness guarantee, cross-toolkit bitwise equivalence, GPU construction, IR or performance result. Validation changed only its tests/oracle/record, never production or manuscript.

## Gate 2: hybrid CPU-SPAI/GPU-GMRES IR pre-execution contract

Separately authorized after Gate1, this gate covers CPU fixed-SPAI construction, GPU GMRES corrections and the hybrid original-system IR loop; no GPU construction/mixedprecision/BSPAI/performance. Preserve946 CPU,323 SpMV and871 GPU GMRES assertions. Tests/records only are owned by Validation.

Main configuration stays uniformbinary64, SPAIepsilon0.3/alpha2/beta8, inner tau1e-8, unrestartedcapn, original normwise outerstop10*2^-53/defaultcap20. CPU outer diagnostics must match independent fixed-order binary64 residual/normwise/componentwise calculations (16epsrelative for nonzero diagnostics, exactzero convention); every snapshot also goes to an independent80-digit Decimal residual/BE oracle. Exact/rational well-conditioned forward and CPU/hybrid solution agreement remains1e-12*max(1,truth norm); the nearlydependent2x2delta2^-30 case records backward-versus-forward limitation, not a forward-accuracy guarantee.

Cases include nonsymmetric/exact/rational/general RHS, exactPb/zeroRHS, cap0/1, coarse-inner0.9cap and1stagnation probes, strict1e-30inner failure, failed construction, original/diagnostic overflow and invalid inputs. Coarse/strict fixtures remain separated from the primary inner1e-8. Check x0=P*b, everyouteroriginal residual andBE, zero initial correction withunscaled originalrhs, accepted xnext=x+d, failure retention/updates/status, success-beforecap/stagnation andoptional-reference diagnostic-only behavior. No failed inner candidate may update the outer iterate.

Residency/reuse evidence must show immutableA/P uploaded onlyonce for an actual multi-correction IR run (or explicitly deferred until correction is needed), perinner0ArnoldivecH2D/D2H, and documented outerCPU/GPU boundaries. A scalar-only host-controlled inner loop remains allowed. The runnable Matrix Market executable must be tested against its documented RHS/configuration and failure exitstatus using validation-owned or existing deterministicfixtures, not a published experiment substitute. CLI/build commands, outputs, telemetry, hashes and limitations will be recorded before declaringGate2passed.

## Gate 1: authoritative GPU GMRES execution result

The all-norm deterministic safe-scaled device norm was implemented after the documented cuBLAS subnormal-norm discrepancy. It preserves the Euclidean metric, literal device division, exact-zero semantics, and the existing GMRES tolerance; it is a serial correctness baseline rather than an optimization or a subnormal-only fallback.

Independent validation passed **871 CUDA-GMRES checks with zero failures** and an **80-digit Decimal audit with 383 checks**, including 125 norm fixtures and 48 GMRES snapshots. Every binary64 subnormal singleton and the smallest-normal boundary passed exact checks. The preserved identity RHS `[2^-1030, 0]` succeeds exactly in one GMRES iteration. The main-case maximum CPU/GPU solution infinity-norm difference was `4.4408920985006262e-16`; independently recomputed relative-residual differences were at most `1.1102230246251565e-16`.

The required estimate-gap rejection was observed at `tau=2e-16`: estimate `1.085730288427049e-16` was below the tolerance while the independently recomputed preconditioned relative residual `3.0067321592990152e-16` was above it, so no success was accepted. Main `tau=1e-8`, unrestarted left preconditioning, one-pass MGS, and named failure statuses are unchanged.

Residency telemetry on a three-iteration solve recorded eight initialization uploads, eight termination downloads, zero Arnoldi vector H2D/D2H transfers, six dot scalar observations, twelve norm scalar observations, and 39 finite-flag scalar observations. CUDA CTest passed **2/2** (`cuda_gmres_tests`, `cuda_spmv_tests`). The existing CPU regression suite remains **946 checks with zero failures** and CTest **4/4**.

This closes GPU-GMRES Gate 1 only. It does not validate GPU iterative refinement, a persistent device-resident outer loop, GPU SPAI construction, mixed precision, performance, or a complete end-to-end GPU solver.

## Gate 2: authoritative independent hybrid execution evidence

Validation independently built the frozen implementation in `build/validation-cuda` and passed **665 C++ hybrid checks, 160 independent 80-digit Decimal checks over 21 outer snapshots, and 23 runnable CLI checks**, all with zero failures. Full hybrid Compute Sanitizer memcheck executed all 665 assertions with **0 errors and 0 bytes leaked in 0 allocations**, without exclusions/suppression. The scoped Gate 2 correctness evidence is satisfied; no later implementation phase is authorized by this record.

Preserved regressions were independently rerun against the new persistent-context revision: **37 CSR +190 SPAI +280 GMRES +439 IR =946 CPU checks**, CPU CTest **4/4**; **323 CUDA SpMV +871 CUDA GMRES** checks, combined CUDA CTest **3/3** including hybrid IR. The unchanged GPU-GMRES Decimal oracle again passed **383 checks**, 48 solver snapshots and125 norm fixtures. CTest durations are not benchmark evidence. Environment is the previously recorded Windows/RTX4060Laptop/CUDA13.4.59/cuBLAS130700/cuSPARSE12806/MSVC19.44 x64 stack, with explicit `CUBLAS_EMULATE_DOUBLE_PRECISION=0`, PEDANTIC_MATH and native binary64. Debug host `/W4 /fp:strict`, custom CUDA `--fmad=false`, sm89 flags were observed in the verbose independent build. No random cases, warm-ups, timed repetitions, speedups, or performance claims are involved.

Validation-only artifacts are `tests/test_cuda_ir.cpp`, `tests/check_cuda_ir_decimal.py`, and `tests/check_cuda_ir_cli.py`. The C++ harness reuses Validation's independent fixed-order CPU residual/update audit helpers from `test_ir.cpp` by renaming its main; it does not run/alter the older439 checks or use production diagnostics to derive expected results. Main3x3 nonsymmetric matrix, its rational inverse, general RHS and2x2delta2^-30 ill-conditioned fixture remain as specified in the pre-execution contract. Tested policies include x0=P*b, every recorded original residual/2-norm/infinity norm/normwise and componentwise BE, unscaled original correction RHS, zero correction initial guess, accepted xnext=x+d, exact/zero initial success, cap0/1/3, stagnation, inner cap/nonconvergence failure retention, construction/invalid-input failure, and successful correction followed by failed outer diagnostics. Optional reference overflow remains flagged diagnostic-only; mandatory residual-norm/backward-denominator overflow fails. Success is checked before cap/stagnation; unsuccessful inner candidates never update x. No tolerance or precision policy changed.

For the exact3x3 RHS[2,-5,7], the hybrid makes one accepted correction and returns[0.99999999999999989,-2,3], recorded original normwise BE1.4325458382260084e-17, componentwise BE5.5511151231257827e-17. Noninteger RHS[.1,-.7,1.3] agrees with the exact rational-inverse reference within the declared1e-12 forward bound. CPU/hybrid terminal status/count agreement and solutions within1e-12*max(1,reference norm) pass on the comparison fixtures; bitwise CPU/GPU equivalence is not claimed. The ill-conditioned fixture succeeds at initialization with recorded binary64 BE1.1102232853449324e-16 while forward relative error2.3669808668796226e-7: this explicitly demonstrates that small backward error does not guarantee small forward error.

The80-digit oracle converts each stored double using `Decimal.from_float`, independently reconstructs original r and both BEs, and uses perrow rounding allowance `64*u*(abs(b_i)+sum_j abs(A_ij*x_j))+denorm_min` (no absolute-one floor); norm/BE allowances propagate this bound. Recorded acceptance still requires the unchanged eta<=10u. Exact-real and working-binary64 diagnostics are not asserted identical. Maximum observed80-digit normwise-BE discrepancy was4.2976375146780241762395561730779896e-17. The oracle is validation-only and never supplies a production fallback or modifies stopping.

### Residency and persistent workspace audit

An actual three-correction coarse-inner fixture (tau.9,outercap3, explicitly separate from main1e-8) initializes one persistent context and **two immutable matrix objects**. Their six CSRarray uploads occur only at construction; each correction adds exactly two vector uploads (original residual RHS and zero initial guess). Total is **12 uploads,12 termination downloads,0 Arnoldi vector H2D/D2H** over three corrections. Full candidate/basis diagnostic snapshots are accumulated D2D then downloaded only after each inner termination. Standalone context tests change RHS across general/integer/zero/general solves, reject invalid RHS without transfers, recover afterward, and agree with CPU results, checking workspace reset/reuse. Aggregate scalar-observation/input counters and their categories are reconciled for every inner attempt.

Independent source review found the context owns `Run` and uploadedA/P/workspaces, resets only basis/snapshot/loop bookkeeping at each solve, and does not upload matrix arrays in `.solve`. `cuda_ir.cpp` uses the existing CPU routines for x0, original residual/BE and outer update. Thus the **outer loop is CPU-staged**, with RHS/zero-guess H2D and correction/diagnostic D2H at every inner boundary. Inner vectors remain GPU-resident, while projectedH/Givens/backsolve and scalar reductions/finite flags synchronize with host. This is not a fully device-resident/asynchronous outer solver. Counter evidence is cross-checked with source, not an independent CUDA API profiler trace; hidden library-internal scalar movement is unmeasured. Serial safe-scaled GPU norms remain the Gate1 correctness baseline, not a parallel optimization.

### Runnable Matrix Market CLI and retained investigation failures

`tests/fixtures/tiny_nonsymmetric.mtx` is the deterministic3x3/7nnz validation fixture. The CLI uses equal-unit2 RHS1/sqrt(n), beta8, alpha=ceil(n/8), main inner tau1e-8 and outer10u. With explicit epsilon.5 it exits0, converges after one outer/three inner iterations, returns the rational reference(12,2,16)/(50sqrt3), original normwise BE5.4015805797291102e-17 and final inner true-relative1.1719012182897898e-16. **Default epsilon.3 fails construction at alpha1 and exits1 on this fixture**; this is retained as expected parameter-dependent failure, not silently changed defaults. Tight epsilon1e-12 also fails transparently. Cap0 at epsilon.5 exits1 with iteration_limit/no correction; malformed/nonfinite/nonpositive epsilon, missing file/arguments and out-of-range caps exit2. The first validation build had C3538 from combining distinct CPU/GPU result types in one auto declaration; only the harness was fixed. The initial CLI assertion expected `outer_iteration_limit`, whereas actual documented status name is `iteration_limit`; that test-label error was corrected after direct output inspection. No production change was made to obtain pass.

### Exact independent commands

All compiled executions used approved host execution because the previously documented sandbox DLL relocation issue persists; no OS/runtime configuration was changed. Same configure command as5A, followed by:

```powershell
cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" && "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" --build build/validation-cuda --verbose'
& C:/Users/Anirudh/AppData/Local/Programs/Python/Python312/python.exe tests/check_cuda_ir_decimal.py build/validation-cuda/cuda_ir_tests.exe
& C:/Users/Anirudh/AppData/Local/Programs/Python/Python312/python.exe tests/check_cuda_ir_cli.py build/validation-cuda/cuda_hybrid_ir.exe tests/fixtures/tiny_nonsymmetric.mtx
& C:/Users/Anirudh/AppData/Local/Programs/Python/Python312/python.exe tests/check_cuda_gmres_decimal.py build/validation-cuda/cuda_gmres_tests.exe
$env:PATH='C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin\x64;'+$env:PATH
$env:CUBLAS_EMULATE_DOUBLE_PRECISION='0'
& 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.4/bin/compute-sanitizer.bat' --tool memcheck --leak-check full --error-exitcode 9 build/validation-cuda/cuda_ir_tests.exe
& 'C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/ctest.exe' --test-dir build/validation-cuda --output-on-failure
foreach($test in @('cpu_core_tests','spai_tests','gmres_tests','ir_tests')) { & "build/ninja/$test.exe"; if($LASTEXITCODE -ne 0){exit $LASTEXITCODE} }
& 'C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/ctest.exe' --test-dir build/ninja --output-on-failure
foreach($test in @('cuda_spmv_tests','cuda_gmres_tests')) { & "build/validation-cuda/$test.exe"; if($LASTEXITCODE -ne 0){exit $LASTEXITCODE} }
```

| Gate2 artifact at independently tested revision | SHA-256 |
|---|---|
| `src/cuda_gmres.cu` | `A208711A73C13A58244B3A3D79F88446E000D9566C2AA47E186B3F541F22C0C8` |
| `include/bspai/cuda_gmres.hpp` | `8327475F724CA0D1A18B6432C40D2B6B9B86709D115053FD94117D0F157E786C` |
| `src/cuda_ir.cpp` | `E927C97E4DA5EDB43A3477E07B8F0F67DDD57E94333CF3FA17FB0361363829F7` |
| `include/bspai/cuda_ir.hpp` | `A40776A7D74A543B5D4A1A23A42C0858B1E1A75978CA0A56F18C7B1ECF00D5F7` |
| `src/cuda_hybrid_main.cpp` | `85DBA2747C3DC00B8D3F82DAEFB1EFA252E754FE5BEF74E39CD8F2AD03D31F87` |
| `cuda/CMakeLists.txt` | `91B66B647FD8EE947C9341D102A46C0522C972ABCB516B746D92AB7FB53BDF49` |
| `tests/test_cuda_ir.cpp` | `A5D852B5544215CD7BEFB1FA00E9B3EA26432509D138A65204D1C5436C5EA41B` |
| `tests/check_cuda_ir_decimal.py` | `AB5FC9D774AC7F696EE8331856D5C90C59764A4AD245690DE7F27DAA0F78DE1D` |
| `tests/check_cuda_ir_cli.py` | `C1D8365113B7C30522D3CD8896A01D112DF5560F7B2C9C8D43643E8233B4EA66` |
| `tests/fixtures/tiny_nonsymmetric.mtx` | `1E33F403765FE840682964D706917878DCB968B3F73E1CFB71529E82A87F691D` |

Prior GPU-GMRES/SpMV test/oracle hashes match Gate1/5A. No unresolved discrepancy was found in the bounded corrected hybrid scope. The isolated cuBLAS subnormal-norm discrepancy above is still unresolved in the vendor primitive and avoided by the explicitly authorized all-norm safe-scaled implementation. This is deterministic tiny-case uniform-double correctness, **not published mixed-precision experiment reproduction**, general high-dimensional robustness, GPU SPAI/BSPAI, cross-stack equivalence or performance validation. Validation edited only validation-owned artifacts; production/manuscript were not modified.

## Gate 3: GPU SPAI pre-execution validation contract

Separately authorized GPU fixed-SPAI construction is tested against the frozen CPU contract, not BSPAI. Reconsulted P1 full-text section2.2 (QR least-squares, independent columns and adaptive pattern growth) and Wang/Li/Gao full-text sections2/3 (prior static GPU SPAI): GPU construction is established prior work, and their static scheduling/QR choices are not a numerically equivalent reference for this adaptive baseline. Project policies remain those in PHASE2A_ALGORITHM_SPEC.md, including right orientation, explicit growth budget, restricted score and range/failure guards.

Validation requires independent scaling/orientation, complete-shadow reduced/full residuals, least-squares coefficients, candidate sets/scores/mean/order/tie filtering, solved-pattern/failure-state alignment, and P=N^T D. Exact integer/dyadic structure, sorted sets/counts and genuinely equal-score index ties are exact expectations; general floating-point coefficients/residuals use the previous well-conditioned768eps*max(1,reference) comparison bound, not bitwise CPU/GPU equality. Exact repeated fixed-stack device results will be checked separately from CPU agreement. An independent80-digit Decimal reduced QR/residual oracle must provide precision beyond MSVC's53-bit long double. Stress near-dependent fits keep their existing2e-6 relative coefficient/1e-5 residual diagnostic thresholds, not a claim of binary64 forward accuracy. Pattern/status comparisons near floating-point decision boundaries must be recorded if they differ, never forced by altered epsilon/rank rules.

Fixtures include identity/signed diagonal, full/approximate3x3, ties/beta1, beta0/alpha0/caps, epsilon equality/oneULP boundary from the GPU's own recorded residual, uneven independent column histories, singular/degenerate and input/range failures. CPU source inspection confirms construction continues through all columns after an unsuccessful fit, retaining the smallest failing index's global status/message; assembly range failures can return early. Device completion/failure semantics will be compared to this actual behavior, not an assumed CPU early exit. Test GPU-produced P with existing GPU GMRES and CPU-staged IR, exact/rational original-system diagnostics, construction-failure propagation and regression suites946CPU/323SpMV/871GMRES/665hybrid. No performance measurement or new precision/stopping rule is authorized by this contract.

Gate3's independent QR oracle uses80-digit MGS on exact stored reduced entries, not normal equations. Candidate score comparisons near a zero radicand use squared-score bounds768eps*max(1,residual_squared), because square root magnifies cancellation; this is an oracle comparison bound, not changed production clamp policy or tolerance. A singular reduced problem can produce exactly zero CPU pivot versus tiny finite vendor pivot; exact-zero project rank guards remain unchanged, and each resulting failure/status is observed rather than a global singularity certificate. The planned implementation uses native FP64 cuSOLVERDgeqrf/Dormqr and cuBLASDtrsv; their info statuses alone do not diagnose rank.

## Gate 3: authoritative independent GPU SPAI evidence

The frozen GPU SPAI revision independently passes **1,534 C++ checks**, **436 independent80-digit QR/residual/score checks over60 fits**, **43 independent80-digit original-system IR checks over5 GPU-P outer snapshots**, and **15 GPU-construction CLI checks**, with zero failures. Full1,534-check Compute Sanitizer **memcheck:0 errors/0 bytes leaked**, and **initcheck:0 errors**; no exclusions or suppressions. The scoped nonsingular-baseline correctness evidence is satisfied, with the explicit singular-status equivalence limitation below. Implementation description is [CUDA_STAGE3_GPU_SPAI.md](CUDA_STAGE3_GPU_SPAI.md); this independent record supplies evidence rather than relying on that report's test claims.

Preserved suites were independently executed again: **946 CPU checks**, CPU CTest4/4; **323 SpMV,871 GPU GMRES,665 CPU-SPAI/GPU-GMRES IR** checks, combined CUDA CTest4/4 including the new SPAI suite. Prior383 GMRES Decimal/160 hybrid Decimal/23 original CLI checks also pass. Prior CPU/GPU tests were not modified. Counts1,512 C++/434 Decimal were preliminary checkpoints, superseded by the final1,534/436 above, not added cumulatively.

### Numerical, structural and failure evidence

Signed diagonal/identity-pattern, full and approximate3x3, capalpha0/1, beta0/1, exact device tie, own-residual epsilon equality/oneULP-lower, canonical scaling/assembly, and8x8 uneven-block histories pass. Scaling is independently checked against original row maxima, B=A^T D, P=N^T D with right-side unequal scaling. Full3x3 P agrees with the exact rational inverse. Each successful reduced fit is checked against independent MGS QR (not normal equations), complete-shadow full/reduced residuals, candidate union/exclusion/restricted scoring, mean over all scores, `(score,index)` order, beta filter, growth union and fitted-pattern publication. Ordinary CPU/GPU statuses/patterns/counts agree on the comparison fixtures; coefficients/residuals satisfy predeclared floating-point bounds, not asserted bitwise equal. Repeated full3x3 GPU N/P values are bitwise identical on this pinned stack. Epsilon-boundary tests use the GPU's own residual rather than silently substituting CPU rounding.

Ordinary80-digit coefficient max absolute discrepancy **3.3306690738754696e-16**, reduced-residual max discrepancy **1.1430106177670572e-16**. Near-dependent2x2delta2^-30 stress: positive pivot ratio3.2927228541949734e-10; maximum80-digit relative coefficient discrepancy **9.5447261316827792e-8** under the previously declared2e-6 diagnostic bound. Absolute discrepancy102.4857164621 is on roughly1e9 coefficients, and must not be presented as ordinary accuracy. Maximum accepted exact-real full residual **5.7721292910986588e-7**, independently below unchanged stress epsilon1e-5; max stress working/exact-real residual difference1.9559410307718066e-8. The stress tolerance is separately labelled, not a relaxation of the main1e-12 construction fixture or1e-8 GMRES rule.

The triangular input[[1,denorm_min],[0,1]] retains the minimum-subnormal offdiagonal in B/shadow; the recorded original-B fit residual equalsdenorm_min exactly. This is a bounded vendor-QR/subnormal case, not proof that all internal cuSOLVER norm paths support the entire input range. QR/norm policies do not use the previously failing cuBLAS norm primitive. Shared actual device score-formula tests cover analytic score, oneULP-negative clamp, rounded32u clamp boundary/oneULP-past rejection, grossnegative, zero/negative denominator, negative residual squared and nonfinite input. Guard status/clamp flags and successful scores match CPU binary64 results exactly in these direct probes. Invalid epsilon/dimensions/zero diagonal/scaling reciprocal overflow/structure underflow fail; an unrepresentable inverse/pivot-range probe fails rank_deficient rather than returning success. No numerical-rank threshold/regularization/CPU fallback is added.

**Preserved status-equivalence limitation:** for singular[[1,2],[1,2]], CPU reports rank_deficient while GPU reports empty_candidates. GPU columns complete2 fits with positive QRpivotratio **2.9086035161589745e-17**, full residuals **2** and **0.5**, zero remaining candidates. Thus its computed pivot is finite nonzero under the frozen rule, and later failure is explicit. For singular all-ones2x2, both report empty_candidates; GPU pivotratio2.3721342664653315e-17, full residuals5 and3.6055512754639896. These inputs violate the mathematical nonsingular baseline assumption; computed-pivot checks are not global rank certificates. Parent/Research explicitly accepted transparent status divergence and rejected structural/near-zero/fallback changes. The integrated singular fixture reports construction_failure with **zero GMRES contexts**, proving partialP is not used. These differences remain limitations, not hidden successful comparisons.

GPU-produced fullP passes direct GPU/CPU GMRES and exact solution checks. GPU SPAI+GPU GMRES+CPU-staged IR succeeds on exact/general/zero RHS, matches CPU IR status/count/solution in the exact fixture, checks x0=P*b/original residual andboth BEs/accepted updates and inner residency. Exact RHS[2,-5,7] yields x=[0.99999999999999967,-1.9999999999999996,3], eta_n2.8650916764520168e-17 after1 correction. Independent80-digit outer-BE max discrepancy2.1488187573390127e-17. The new CLI at explicit epsilon.5,alpha1,equal-unit2 RHS succeeds in1outer/3inner, eta_n5.4015805797291114e-17, inner true-relative1.6573185966669716e-16. Default epsilon.3/alpha1 fails transparently with no inner context; outercap0 and innercap1 propagate failures, malformedcaps/nonfiniteepsilon exit2. Old CPU-construction CLI remains unchanged in policy and passes all23 priorchecks.

### Device work, ownership and scope

Independent source review confirms device scaling/gather, cuSOLVERDgeqrf/Dormqr, cuBLASDtrsv, original-B residual recomputation, restricted score kernel, Ncommit and Pnumeric assembly; no CPU numerical fallback or normal equations. R diagonal exact-zero/nonfinite guards precede triangular solve, and cuSOLVER devInfo is treated as API error status, not singularity detection. Sorted J/I/candidate metadata/mean/filter/round scheduling remain host-controlled. Up to8 distinct stream/handle slots submit exact-size, unpadded reduced QR/Q problems before collecting any result. The8x8 uneven fixture reports8 QR submissions beforecollection and distinct completed-fit lengths. This establishes dispatch structure, **not measured concurrent execution, utilization or speedup**.

DenseB/N/P staging and up to8 dense n*n reduced workspaces entail **O(n²) storage**, bounded to tiny tests here; this is not a validated scalable sparse-construction implementation. B, perfit coefficients/diagnostics/scores and finalN/P are downloaded for host metadata/records. Canonical finalP is staged through hostCSR then uploadedonce to the existing persistent GPU-GMRES context. The outer IR remainsCPU, with the previously recorded boundary transfers. This is GPU numerical SPAI with host adaptive control, not entirely device-resident construction/IR.

Two source-audit exceptional-path findings were reported to Implementation before finalization: raw solver/BLAS handles could leak during partially failed Slot construction, and score dispatch could read stale/unwritten residual diagnostics following a failed fit. The owner introduced handleRAII and device fit-status guarding; independent final source review and memcheck/initcheck found no issue on tested paths. No allocation/API fault injection was performed, so exceptional cleanup is source-audited beyond tested numerical failures. A transient duplicated score-helper declaration was caught in review and removed by its owner before final build. Validation's early compile warning from implicit longdouble-to-double JSON conversion was fixed with explicit conversion; production warnings from generated CUDAstubC4211/vendor headers remain unsuppressed.

### Environment, commands and tested hashes

Existing RTX4060Laptop/CC8.9/driver616.92/CUDA13.4.59/MSVC19.44 x64 stack is unchanged. Independent environment probe now reports **cuSOLVER12.3.2**, cuBLAS130700, cuSPARSE12806, runtime/driverAPI13040. CUBLAS_EMULATE_DOUBLE_PRECISION=0 is explicit; cuSOLVER deterministic-results andDEFAULT_MATH and cuBLASPEDANTIC modes are set and queried in source. `/W4 /fp:strict`, sm89, `--fmad=false` observed in independent verbose build; MSVClongdouble53bits, with80-digit Python oracle supplying wider precision. Deterministic fixtures/no random seed; no benchmark warm-up/timing/repetition methodology is claimed. All compiled commands require approved host execution because the prior sandbox DLL issue remains; no OS/toolchain installation was changed.

Same independent configure/build/CPU regression commands recorded underGate2, plus:

```powershell
& C:/Users/Anirudh/AppData/Local/Programs/Python/Python312/python.exe tests/check_cuda_spai_decimal.py build/validation-cuda/cuda_spai_tests.exe
& C:/Users/Anirudh/AppData/Local/Programs/Python/Python312/python.exe tests/check_cuda_ir_decimal.py build/validation-cuda/cuda_spai_tests.exe
& C:/Users/Anirudh/AppData/Local/Programs/Python/Python312/python.exe tests/check_cuda_spai_cli.py build/validation-cuda/cuda_gpu_spai_ir.exe tests/fixtures/tiny_nonsymmetric.mtx
$env:PATH='C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin\x64;'+$env:PATH
$env:CUBLAS_EMULATE_DOUBLE_PRECISION='0'
& build/validation-cuda/cuda_environment_probe.exe
& 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.4/bin/compute-sanitizer.bat' --tool memcheck --leak-check full --error-exitcode 9 build/validation-cuda/cuda_spai_tests.exe
& 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.4/bin/compute-sanitizer.bat' --tool initcheck --error-exitcode 9 build/validation-cuda/cuda_spai_tests.exe
foreach($test in @('cuda_spmv_tests','cuda_gmres_tests','cuda_ir_tests','cuda_spai_tests')) { & "build/validation-cuda/$test.exe"; if($LASTEXITCODE -ne 0){exit $LASTEXITCODE} }
& 'C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/ctest.exe' --test-dir build/validation-cuda --output-on-failure
```

| Gate3 artifact at independently tested revision | SHA-256 |
|---|---|
| `src/cuda_spai.cu` | `32F19AD4AC35F016224C8FF218B3444B25F9F3DA1BDBBC232A7BBF1DA4FBC106` |
| `include/bspai/cuda_spai.hpp` | `2656C36B2B914FE3D5B0EF867F29F8AD0C43C101DCE5681253F7E67F17047A83` |
| `src/cuda_ir.cpp` | `8DC6B5492370C21D9995AE32FA43FDA585BB138A5C891EF664EDD5D4C01B8077` |
| `include/bspai/cuda_ir.hpp` | `DDE7B6BFB299EBF5B6CF853ECAFEF82EFADA82EE30DF911D250A8261949BBBB0` |
| `src/cuda_hybrid_main.cpp` | `6E2C2E5F6D4BCA6ED8D3DBFEC95C23BA43471A316522A776A1644736FAFEDA94` |
| `src/cuda_environment.cpp` | `E53CF79D26EDEFD072B927FC202CAA2314D6E2AE10FF6AC324D1D1A69EA769BA` |
| `cuda/CMakeLists.txt` | `06F9FDD7285D5CCA48D4A97E32BF44DFA0F4DAF150F2534C111F5B2446EBD033` |
| `tests/test_cuda_spai.cpp` | `3D1DD5E9DD5E0D1BCA4D214F778EC6597F34BF7243827A8F61F47EF96168E19D` |
| `tests/check_cuda_spai_decimal.py` | `E44C64F4FBB3D5341B06FFA1FE30D2F01436A83EF3853A98948B734B1D390F48` |
| `tests/check_cuda_spai_cli.py` | `BEC8EAD23EF5A2E9F2B5566B7606ABE27F9684F6A1FA656D0ADB2A0898AAAF55` |

No unreported baseline discrepancy remains in the tested scope; singular status divergence and vendor-norm behavior remain explicitly recorded limitations. Evidence is tiny deterministic **uniform-double correctness, not published mixed-precision reproduction**, BSPAI, GPU optimization, scalable performance, general high-dimensional stability or cross-stack equivalence. Validation changed only validation tests/oracles/records and did not edit production or manuscript. Stop atGate3.
