# CUDA milestone 5A: binary64 CSR SpMV

Bounded component implemented and tested, 11 October 2026. This is a separate component and build; CPU SPAI/GMRES/IR and the root CMake project are unchanged.

Source basis: [cuSPARSE generic SpMV](https://docs.nvidia.com/cuda/cusparse/index.html#cusparsespmv) supports uniform CUDA_R_64F, int64 indices and nontranspose CSR_ALG2 deterministic execution. CPU/GPU bitwise equality is not promised: prebuilt library reduction/FMA behavior is not controlled by application compiler flags. This is an established vendor primitive, not a novel GPU SPAI contribution.

Project policies: checked int64/byte conversions and canonical CSR structure (offsets, unique sorted in-range columns, finite values); owned device allocations/descriptors/handle; one dedicated nonblocking stream; blocking public completion via stream synchronization; finite inputs/results; alpha1/beta0; no preprocessing or optimizations. Each H2D upload synchronizes before its host storage can disappear; scoped stream draining also protects D2H result lifetime during error unwinding. Positive rectangular/nonsymmetric matrices and empty rows are supported. Zero-nnz output is device-cleared explicitly instead of creating an empty sparse descriptor. Same object is not thread-safe. Destructors synchronize/free without throwing and log cleanup failures to stderr; operation errors throw with CUDA/cuSPARSE status. Device selection is restored after operations. No CPU fallback occurs on GPU failures.

Observed GPU: RTX 4060 Laptop GPU, compute capability8.9, 8188MiB (8585216000 bytes), driver616.92. nvcc13.4.59, 64-bit MSVC19.44.35228, CMake3.31.6-msvc6 and Ninja1.12.1. Header versions cuSPARSE12.8.6.49/cuBLAS13.7.0.27. The loaded-library inventory executable reports runtime/driver API13040 (13.4), cuSPARSE12806 (12.8.6), cuBLAS130700 (13.7.0), one device. cuBLAS is inventory only, not required by the SpMV library. Matching toolkit import libraries and runtime DLLs were resolved from CUDA v13.4; CMake also links its nvJitLink dependency.

## Build and execution record

New implementation files: `include/bspai/cuda_spmv.hpp`, `src/cuda_spmv.cu`, `src/cuda_environment.cpp` (inventory only), `cuda/CMakeLists.txt`, `scripts/build_cuda_msvc.cmd`, and this report. Validation owns `tests/test_cuda_spmv.cpp` and [its independent record](PHASE2B_VALIDATION_RECORD.md). Existing algorithm sources, prior tests and manuscript were not edited by Implementation.

The separate `build/cuda-ninja` configuration is x64 MSVC with CUDA architecture89, Debug, C++11/CUDA11. CPU reference and test host arithmetic use `/fp:strict`; CUDA compilation also forwards strict host flags. These flags do not control arithmetic inside prebuilt cuSPARSE. No fast-math option or custom GPU kernel was introduced. Root `build/ninja` remains the original 32-bit GNU baseline with its wider oracle.

Run from the workspace root:

```powershell
cmd /c scripts\build_cuda_msvc.cmd
$env:PATH='C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.4/bin/x64;'+$env:PATH
.\build\cuda-ninja\cuda_environment_probe.exe
.\build\cuda-ninja\cuda_spmv_tests.exe
& 'C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/ctest.exe' --test-dir build/ninja --output-on-failure
.\build\ninja\cpu_core_tests.exe
.\build\ninja\spai_tests.exe
.\build\ninja\gmres_tests.exe
.\build\ninja\ir_tests.exe
```

The script records exact vcvars64, CUDA DLL path, CMake/nvcc/Ninja paths, architecture and configure/build/test commands. It must run outside the current sandbox: sandbox CPU launches failed with Illegal System DLL Relocation, while the MSVC ABI probe failed C1902 (PDB manager mismatch). Host execution outside the sandbox succeeds without changing toolchains, system installations or numerical policies. A first shell quoting attempt failed before configuration; the explicit script corrected it. CIM hardware query was denied; nvidia-smi and runtime queries supplied GPU evidence. `cl /Bv` printed the compiler version then its expected missing-source error.

The first CUDA test compile exposed a Validation-owned duplicate exception-handler template, rejected by MSVC C2312. Validation corrected that harness; no production numerical change or tolerance relaxation followed. Builds succeed with vendor `cusparse.h` C4996 deprecated-declaration warnings. The implementation calls only current generic APIs; no project-source warning was observed. Vendor warnings remain unsuppressed and are not a warning-free build claim.

## Validation results and boundaries

Latest GPU suite: **323 checks, zero failures**, CUDA CTest **1/1**. Independent exact dyadic cases (including subnormals) agree exactly. Decimal/cancellation fixtures use the predeclared row-length-aware bound in the validation record, with no absolute-one tolerance floor. Maximum observed CPU/GPU absolute difference `1.1102230246251565e-16`; maximum differential-bound fraction `0.012077294685990324`. These are local component observations, not universal accuracy or high-relative-accuracy claims. Repeated execution satisfies local bitwise determinism under pinned CSR_ALG2/nontranspose hardware/library conditions.

Implementation, Parent and Validation each reran the unchanged CPU tests outside the sandbox: **37 core +190 SPAI +280 GMRES +439 IR =946 checks**, zero failures, CTest **4/4**. Wider precision remains64 significand bits in the original GNU run, whereas MSVC CUDA tests have53-bit long double and do not claim a wider independent oracle.

Independent `compute-sanitizer --tool memcheck --leak-check full --error-exitcode 99` passed the explicit test `--memory-safety-only` mode:321 checks, zero failures, `ERROR SUMMARY: 0`, zero bytes leaked in zero allocations. That mode omits exactly two deliberate invalid-device calls, which remain covered by the full323-check ordinary run. Initial raw310-check sanitizer execution reported three expected invalid-device API errors with zero leaks; that investigation is retained, not erased or globally suppressed. Exact sanitizer commands/hashes are recorded by Validation separately.

Huge index-width/byte-count and injected allocation/driver-failure paths are guarded but not exhaustively runtime-tested; ordinary invalid vectors/devices, computed overflow/recovery, allzero matrices, empty rows, rectangular dimensions and repeated object lifetimes were tested. The largest tested row has257 entries. Cleanup logging is not proof that all possible teardown errors are recoverable. No asynchronous caller API or multi-thread/multi-GPU execution is supported.

No solver port, mixed precision, bucketing, performance measurement, novel parallel SPAI claim or solver convergence claim. Stop at Milestone5A.
