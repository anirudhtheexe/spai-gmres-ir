# SPAI–GMRES–IR: CPU and CUDA, FP32 and FP64

This project solves sparse real square linear systems using an adaptive sparse
approximate inverse (SPAI), left-preconditioned unrestarted GMRES, and iterative
refinement. It retains four implementations: CPU FP32, CPU FP64, GPU FP32 and
GPU FP64. GPU refinement uses a CPU outer loop. Bucketed SPAI (BSPAI) and mixed
precision are future research; these implementations use one precision each.

## Build and test

Requirements: CMake 3.18+, C++11 compiler; Python 3.10+ for independent checks
and packaging. CUDA builds additionally need a compatible 64-bit host compiler,
NVIDIA GPU/driver and CUDA Toolkit with cuBLAS, cuSPARSE and cuSOLVER. Tested
here: GCC 6.3 CPU; MSVC 19.44/CUDA 13.4; RTX 4060 Laptop, architecture 89.

CPU with GCC/Clang (run from the project root):

```sh
cmake -S . -B build/cpu -DCMAKE_BUILD_TYPE=Debug
cmake --build build/cpu
ctest --test-dir build/cpu --output-on-failure
```

This runs all five CPU suites. The original FP64 SPAI/GMRES/IR tests require
`long double` wider than `double`. MSVC does not provide that type. On Windows,
the helper discovers Visual Studio C++ Build Tools and its CMake/Ninja tools:

```bat
scripts\build_windows.cmd cpu
scripts\build_windows.cmd cuda 89
```

The CPU helper runs `cpu_core_tests`, `fp32_cpu_tests`, and the independent
Decimal80 snapshot gate; it does not claim all five original CPU tests pass
under MSVC. The CUDA helper runs all eight GPU suites. Replace `89` with your
GPU architecture. The helper's default `native` requires CMake 3.24+. Set
`CUDA_PATH` to your toolkit; no user-specific installation path is embedded.
An optional third argument selects `Release` (default `Debug`).

Portable CUDA configuration from a supported compiler environment:

```sh
cmake -S cuda -B build/cuda -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_BUILD_TYPE=Debug
cmake --build build/cuda
ctest --test-dir build/cuda --output-on-failure
```

Alternatively, configure the root with `-DBSPAI_ENABLE_CUDA=ON` to build both
backends in one tree. `-DBUILD_TESTING=OFF` builds only libraries and programs.
Set `CUBLAS_EMULATE_DOUBLE_PRECISION=0` when running CUDA FP64.

## Run

Each solver accepts `MATRIX.mtx [epsilon=0.3] [outercap=20] [innercap=0]`.
The RHS has equal components and unit 2-norm; `innercap=0` selects matrix size.
Use the supplied deterministic small fixture with explicit epsilon `0.5`:

```sh
build/cpu/cpu_fp32_ir tests/fixtures/tiny_nonsymmetric.mtx 0.5
build/cpu/cpu_fp64_ir tests/fixtures/tiny_nonsymmetric.mtx 0.5
build/cuda/solver/gpu_fp32_ir tests/fixtures/tiny_nonsymmetric.mtx 0.5
build/cuda/solver/gpu_fp64_ir tests/fixtures/tiny_nonsymmetric.mtx 0.5
```

On Windows append `.exe`. The Windows helper's output trees are
`build/cpu-Debug` and `build/cuda-Debug/solver`. Exit codes: 0 converged,
1 construction/solver failure, 2 invalid input. FP32 CLI input is parsed and
canonicalized by the shared binary64 Matrix Market reader, then explicitly
rounded to binary32; entries that overflow or become zero are rejected.

## What to submit

Upload **`submission/SPAI_GMRES_IR_Submission.zip`** to Teams. It contains the
source, headers, build files, tests, tiny input fixture, numerical specifications,
and cleanup/validation documentation. It includes a per-file SHA-256 manifest.
It builds without the research archive, Git database or generated build files.

Recreate it with `python scripts/package_submission.py`. This is a source
submission; it does not include the historical benchmark datasets or raw traces.
If a rubric also asks for the report/evidence, the original report packages are
preserved inside the research archive described below.

## Research evidence and recovery

The working checkout retains `archives/research_20261019.zip`. It contains all
historical results, profiles, large inputs/provenance, campaign scripts, report
packages, and planning reports at their original relative paths. Every file was
SHA-256 verified before removing loose originals. Explore the ZIP directly or:

```sh
python scripts/research_archive.py verify
python scripts/research_archive.py restore
```

Restoration refuses to overwrite a changed file. It expands about 1.85 GB;
reproduction of old experiments also requires the archived original build
configuration in `safeguards/cleanup_20261019/source_before.zip`. The compact
submission omits archives, safeguards, `.git`, local Codex settings and all
generated files. The small project specialist instruction files are included.

See [CODEBASE_GUIDE.md](CODEBASE_GUIDE.md) for source structure,
[docs/CLEANUP_RECORD.md](docs/CLEANUP_RECORD.md) for changes and testing, and
[PAPER_INDEX.md](PAPER_INDEX.md) for references. Known numerical limitations
remain recorded in the specifications and validation record.
