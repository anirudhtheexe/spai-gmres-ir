# Phase 2A CPU milestone 1

Date: 11 October 2026. Status: completed and tested; stopped at the requested first milestone.

This is portable C++11 infrastructure for debugging with uniform `double` values. It contains no SPAI construction, GMRES, iterative refinement, BSPAI bucketing, CUDA, parallel optimization, or published-result reproduction. The Phase 1 report is historical and was not edited.

## Components and choices

- `include/bspai/cpu.hpp` / `src/cpu.cpp`: canonical CSR creation, transpose, matrix-vector multiplication, and Matrix Market loading.
- `src/main.cpp`: diagnostic probe printing dimensions, zero-based entries, and `A * ones`.
- `CMakeLists.txt`: library/probe and independent Validation Agent test target registered with CTest.
- `tests/test_cpu.cpp`, `tests/fixtures/tiny_nonsymmetric.mtx`: independently authored by the Validation Agent, not copied from implementation calculations.

The loader accepts coordinate `real` or `integer`, `general` or `symmetric`, with exact standard-case header tokens. It rejects zero dimensions, unsupported formats, nonfinite values, invalid one-based coordinates, extra fields, truncated data, and trailing coordinates. Symmetric off-diagonal entries are mirrored and diagonal entries are stored once. No triangular storage enforcement is implemented; callers must use a valid one-triangle symmetric input.

CSR uses zero-based `size_t` indices and sorted columns. Duplicate coordinates accumulate in stable input order using double precision; exact resulting zeros are removed. Duplicates that overflow are errors. Multiplication proceeds in ascending column order with double values/products/accumulation and rejects nonfinite inputs/results. There is no normalization or scaling. Transpose constructs canonical CSR of the transposed matrix. These are explicit infrastructure choices, not research claims or precision-equivalence guarantees.

## Reproducible build and checks

Working directory: `C:\Users\Anirudh\Desktop\pcapproj`. GNU MinGW GCC 6.3.0, bundled CMake 3.31, bundled Ninja, Debug build. No downloaded dependencies or random data. On this host use these PowerShell commands:

```powershell
$cpuCmake = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe'
$cpuCtest = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\ctest.exe'
& $cpuCmake -S . -B build/ninja -G Ninja -DCMAKE_CXX_COMPILER=C:/MinGW/bin/g++.exe '-DCMAKE_MAKE_PROGRAM=C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/Common7/IDE/CommonExtensions/Microsoft/CMake/Ninja/ninja.exe' -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTING=ON
& $cpuCmake --build build/ninja
& $cpuCtest --test-dir build/ninja --output-on-failure
& .\build\ninja\cpu_core_tests.exe
& .\build\ninja\cpu_matrix_probe.exe tests/fixtures/tiny_nonsymmetric.mtx
```

Normal portable commands when CMake and a compiler are on PATH:

```text
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTING=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

Direct compiler check also succeeded without warnings:

```powershell
& 'C:\MinGW\bin\g++.exe' -std=c++11 -O0 -g -Wall -Wextra -Wpedantic -fno-fast-math -Iinclude src/cpu.cpp src/main.cpp -o build/cpu_matrix_probe.exe
```

CTest: 1/1 test passed. Independent executable: **37 checks, 0 failures**; `double` significand 53 bits, `long double` 64 bits. Tests include exact matrix/CSR/manufactured-product comparisons, rectangular/symmetric input, transpose orientation, duplicate cancellation, empty rows, malformed inputs, overflow, and one wider-precision floating-product bound. Probe on the 3x3 fixture prints `nnz=7` and `A * ones = [5,8,4]`. No solver convergence, backward-error, or performance measurement was made.

## Limitations and pending work

The MinGW Makefiles generator failed in compiler detection with `mingw32-make: INTERNAL: readdir: Invalid argument`. The Visual Studio 17 2022 generator found no C++ compiler. The bundled CMake/Ninja/GNU combination succeeds; only that CMake combination and the direct GNU build are verified.

The implementation is not tested on Linux or MSVC. Integer values are stored as double, so exact representation above 2^53 is not promised. The strict parser does not implement array/complex/pattern/Hermitian/skew-symmetric input or arbitrary header capitalization. Allocation exhaustion is reported through standard exceptions; there is no out-of-core loading.

Research Agent findings on scaling, construction loop conventions, restricted candidate scoring, and numerical guards remain specification inputs for a later milestone. No implementation choice for those algorithms has been made in this milestone. Proceed only after the parent reviews this result and reconciles the forthcoming specification.
