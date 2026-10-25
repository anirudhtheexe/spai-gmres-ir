# FP32 GMRES storage timing artifacts

Date: 25 October 2026. See `docs/FP32_GMRES_STORAGE_VALIDATION.md` for the
interpretation, environment, numerical checks, and limitations.

`benchmark.cpp` constructs three deterministic matrices outside the measured
interval and times the public FP32 CUDA GMRES call with synchronization before
and after. Each process has three warmups and twelve retained trials. The
serial process order per mode was baseline, current, current, baseline.
`benchmark_samples.jsonl` contains all 180 runs; `run_file` identifies the
process. Status, iterations, solution, histories, and final residuals were
hashed using their stored FP32 bits. The hashes agree across implementations.

`reconstruct_baseline.py` starts from the production source and restores only
the eager basis/Hessenberg storage code. Its generated
`baseline_fp32_cuda_gmres.cu` is retained here. The baseline is the immediate
pre-storage-change algorithm, not repository HEAD. It and the production body
were compiled with CUDA 13.4 for sm89, `--fmad=false`, `-DNDEBUG`, and MSVC
`/O2 /MD /EHsc /fp:strict /Z7`. The common benchmark was compiled with MSVC
`/O2 /MD /Z7 /EHsc /std:c++17`. Both versions linked the same libraries in
`build/cuda-NoPdb/solver`.

After activating the Visual Studio 2022 BuildTools x64 environment, the
essential build commands from the project root are:

```text
python results/fp32_gmres_storage_20261025/reconstruct_baseline.py
nvcc -Iinclude -DNDEBUG -arch=sm_89 --fmad=false -Xcompiler=/O2,/MD,/EHsc,/fp:strict,/Z7 -c results/fp32_gmres_storage_20261025/baseline_fp32_cuda_gmres.cu -o build/fp32-gmres-bench/baseline.obj
cl /O2 /MD /Z7 /EHsc /std:c++17 /Iinclude /I"%CUDA_PATH%\include" results/fp32_gmres_storage_20261025/benchmark.cpp build/fp32-gmres-bench/baseline.obj build/cuda-NoPdb/solver/bspai_fp32_cuda_spmv.lib build/cuda-NoPdb/solver/bspai_fp32_cpu.lib build/cuda-NoPdb/solver/bspai_cpu.lib /Fe:build/fp32-gmres-bench/baseline.exe /link /LIBPATH:"%CUDA_PATH%\lib\x64" cudart.lib cublas.lib cusparse.lib cusolver.lib
cl /O2 /MD /Z7 /EHsc /std:c++17 /Iinclude /I"%CUDA_PATH%\include" results/fp32_gmres_storage_20261025/benchmark.cpp build/cuda-NoPdb/solver/bspai_fp32_cuda_gmres.lib build/cuda-NoPdb/solver/bspai_fp32_cuda_spmv.lib build/cuda-NoPdb/solver/bspai_fp32_cpu.lib build/cuda-NoPdb/solver/bspai_cpu.lib /Fe:build/fp32-gmres-bench/current.exe /link /LIBPATH:"%CUDA_PATH%\lib\x64" cudart.lib cublas.lib cusparse.lib cusolver.lib
```

Invoke either executable as `EXE baseline|current exact|one|chunk`. The build
tree uses local CMake `/Z7` overrides because the standard MSVC compiler probe
hit `C1902 Program database manager mismatch` on this host. The executable
hashes in this run were `CA2FD234CEC7900DBAC95B2644A39ABB1A11EF3958CFC89CF55EB50B35EDAB7E`
for baseline and `9BCDFA04315242E4FA8007FCAD0F87F040BF555F5F2972C4F481976FA91C044C`
for current.

## Complete FP32 GPU pipeline follow-up

`pipeline_benchmark.cpp` is the archived Experiment 2A harness. It was linked
with the same final `bspai_fp32_cuda_spai`, `bspai_fp32_cuda_spmv`, FP32 CPU,
and CUDA libraries for both variants. The eager executable used the baseline
GMRES object above; the slab executable used the production GMRES library.
Run each executable with:

```text
--mode gpu_pipeline --matrix build/spai-row-validation/stencil_8192.mtx --rhs periodic7 --epsilon 0.3 --alpha 1024 --beta 8 --innercap 8192 --outercap 20 --warmups 1 --trials 3
```

The input matrix is archived as
`data/stage45b_fp64_scalability_v1/stencil_8192.mtx` in
`archives/research_20261019.zip`; its SHA-256 is
`90d25d6332cc3729030c13e89deb74169b13aef8c1fe89b1ffa4af6709f09b1f`.
The files `pipeline_baseline_1.jsonl`, `pipeline_current_1.jsonl`,
`pipeline_current_2.jsonl`, and `pipeline_baseline_2.jsonl` preserve the
baseline/current/current/baseline serial order. Each contains one warmup and
three measured samples. Their combined interpretation and numerical-output
comparison are in `docs/FP32_GMRES_STORAGE_VALIDATION.md`.
