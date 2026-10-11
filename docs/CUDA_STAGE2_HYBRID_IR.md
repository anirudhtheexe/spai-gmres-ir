# Phase 2B Gate 2: hybrid binary64 SPAI–GMRES–IR

11 October 2026. Implementation and bounded independent validation complete; stage handoff awaits parent approval. This stage implements only CPU SPAI construction and CPU outer refinement with persistent GPU GMRES corrections. No GPU SPAI, mixed precision, BSPAI bucketing, benchmark or performance optimization is included.

## Authority and intended numerical behavior

Carson–Khan [P1 Algorithm 3.1 and section 4](https://arxiv.org/pdf/2202.10204v2) and the [published BSPAI article sections 3–4](https://link.springer.com/article/10.1007/s00366-025-02187-z) establish the SPAI-preconditioned refinement sequence. The exact executable choices, source reconciliation and exceptional-case rules remain in [the binding CPU specification](PHASE2A_ALGORITHM_SPEC.md); this all-binary64 project baseline does not reproduce published mixed-precision/quadruple-residual experiments.

Parent explicitly authorized **CPU outer staging** for Gate 2. CPU `construct_spai` builds P unchanged, including B=A^T D and P=N^T D. The unchanged CPU products compute x0=P*b, original r=b-A*x, and the unchanged `evaluate_ir_diagnostics` computes original residual, normwise/componentwise backward errors and optional forward diagnostics. Each correction invokes the existing GPU one-pass MGS, unrestarted left-preconditioned GMRES with zero initial correction. The CPU updates x+=d in ascending vector order. The inner algorithm, all-norm safe-scaled device primitive, tau and true-residual acceptance remain unchanged from [Gate 1](CUDA_STAGE1_GMRES.md).

The fixed outer stop remains eta_n <= 10*2^-53, checked initially and after each fully diagnosed update. Caps 0..20 are accepted; cap zero evaluates initialization only. Failed inner solves are retained/reported and their candidates are never applied. Success precedes exact unchanged-iterate stagnation and cap checks. Mandatory arithmetic overflow is explicit numerical_failure; optional forward-diagnostic overflow alone is exempt exactly as in CPU IR. Neither P nonsingularity nor all-input convergence is certified. CPU/GPU rounding may change inner histories and outer results; no bitwise equivalence or inherited theoretical mixed-precision guarantee is asserted.

## Files and persistent ownership

- `include/bspai/cuda_gmres.hpp`, `src/cuda_gmres.cu`: new synchronous noncopyable `CudaGmresContext`, owning immutable GPU A/P and one reusable Krylov/candidate workspace. Existing GMRES solve body and safe norm are reused, not replaced. Each call resets only logical basis/snapshot counters, allocates fresh host scalar/history state and uploads its own RHS/initial guess. Matrix buffers/descriptors/workspaces survive across corrections. Device selection is restored after each synchronous call and teardown; cleanup errors are reported without throwing.
- `include/bspai/cuda_ir.hpp`, `src/cuda_ir.cpp`: `cuda_refine_with_preconditioner` for component validation and `cuda_spai_gmres_ir` for CPU construction plus hybrid refinement. Results retain the existing `IrResult` diagnostics/status/history plus context and transfer telemetry.
- `src/cuda_hybrid_main.cpp`: Matrix Market runnable executable with parameters, construction/outer/inner diagnostics, transfer counts and final solution.
- `cuda/CMakeLists.txt`: separate `bspai_cuda_ir`, executable and validation-owned test registration. Root CPU build/source/tests remain unchanged; no manuscript files are modified.

The public context owns uploaded matrices rather than retaining caller references; it is not thread-safe. Runtime/CUDA/library/allocation errors remain checked exceptions, not CPU fallbacks. Invalid vector input returns existing GMRES invalid_input. Finite-precision numerical failures retain existing status semantics. A hybrid initial success still initializes one GPU context; construction failure does not create one. This is a correctness staging choice, not optimized initialization.

## Residency and transfer accounting

This stage is **hybrid CPU outer / GPU vector-resident inner**, not an entirely GPU-resident refinement loop.

- `context_constructions=1`, `matrix_uploads=2` denote one context and two immutable **matrix objects initialized**, not two CSR arrays, bytes or two upload API calls. For nonempty A and P, `construction_transfers.vector_uploads=6` records offset/index/value arrays for each. An empty matrix uses the existing zero-matrix path and uploads no CSR arrays.
- Every inner call uploads exactly two n-entry binary64 vectors: the current CPU original residual and an explicitly zero correction initial guess. A/P are never re-uploaded by `solve`.
- No full-vector transfers occur within Arnoldi. Safe norms, dots and finite-status flags are scalar/status observations, as in Gate 1. All category counts and host scalar inputs are reported in the existing `CudaGmresTransfers` fields.
- Only after each inner termination does the context download its final correction plus completed basis and candidate snapshots. Downloads equal `1 + inner.basis.size() + inner.history.size()`. These debug histories support independent validation, are not required algorithmically, and add O(n*cap) device memory/storage and post-inner transfers.
- CPU original residual, outer iterates, backward/forward diagnostics and updates are already host data and introduce no additional device copies. The device buffers remain reusable even though the outer vectors are staged on CPU.
- `CudaIrResult.construction_transfers`, `.inner_transfers` and `.total_transfers` separate one-time construction, per-inner deltas and cumulative application-issued transfers. These counters are source instrumentation, not a CUPTI claim about vendor-internal traffic. Kernel dimensions/control metadata are not vector transfers.

## Reproducible build and CLI

The existing 64-bit Debug CUDA/MSVC configuration is retained. From the repository root:

```powershell
cmd /c scripts\build_cuda_msvc.cmd
$env:PATH='C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin\x64;'+$env:PATH
$env:CUBLAS_EMULATE_DOUBLE_PRECISION='0'
build\cuda-ninja\cuda_hybrid_ir.exe tests\fixtures\tiny_nonsymmetric.mtx 0.5
```

The existing script records/configures MSVC x64, CUDA 13.4, architecture 89, native pedantic FP64 cuBLAS and emulation0, `/fp:strict` host and `--fmad=false` custom kernels; it builds and runs CUDA CTest. Environment and vendor-norm limitation are recorded in Gate 1/5A and [the independent validation record](PHASE2B_VALIDATION_RECORD.md). Installed hardware is RTX4060 Laptop GPU, driver616.92; runtime13040, loaded cuSPARSE12806 and cuBLAS130700. Prebuilt library arithmetic is not controlled by custom-kernel compiler flags. Host execution outside the sandbox is required because of the previously isolated sandbox tool-launch issue; no system/compiler changes are made.

CLI contract:

```text
cuda_hybrid_ir matrix.mtx [epsilon=0.3] [outercap=20] [innercap=0(default n)]
```

Input uses the unchanged canonical Matrix Market real/integer coordinate general/symmetric loader; a nonempty square matrix is required. RHS is equal-component `1/sqrt(n)` in binary64. Parameters are natural ordering, identity SPAI pattern, beta8, alpha=ceil(n/8), tau1e-8 and fixed outer tolerance; optional arguments alter only construction epsilon/caps. Output includes exact parameters, construction status/nnz(P), all outer BE/residual values, inner statuses/iterations/estimates/recomputed values, applied-update flags, aggregate transfers and final x. Exit0 means verified outer convergence, exit1 solver/construction/runtime failure and exit2 input/usage error. A failed solve is not relabelled successful. The tiny fixture is a hand-check validation matrix, not published result reproduction.

## Evidence and remaining gate

Final frozen independent hybrid validation passes **665 C++ checks/0 failures**, CUDA CTest **3/3**. The independent 80-digit audit passes **160 checks across21 outer snapshots** (maximum nBE discrepancy `4.2976375146780241762395561730779896e-17`), and the CLI audit passes **23 checks**. Validation's full665-check compute-sanitizer memcheck reports0 errors/0 leaks, with no exclusions. The preserved GPU SpMV323 and GPU GMRES871 checks pass with zero failures. Original CPU946 checks (37/190/280/439) rerun outside the sandbox with CTest4/4; original CPU production files, root CMake and 5A SpMV source hashes are unchanged. Tests and independent record are owned by Validation; implementation has not changed those files.

The independent three-correction reuse fixture records one context, two immutable matrix initializations,12 uploads (six one-time CSR arrays plus two vectors per correction) and12 post-inner diagnostic downloads. All Arnoldi vector transfer counters remainzero. The context also survives changing RHS, zero RHS, repeated RHS and an invalid-dimension call without matrix re-upload or stale workspace history. This verifies scoped lifetime/reuse, not efficient asynchronous performance.

The primary tiny exact/rational cases agree with trusted CPU IR status/count/accuracy under the declared validation bounds. The nearly dependent2x2 case has recorded original nBE `1.1102232853449324e-16` but relative forward error `2.3669808668796226e-7`, retained as a backward-versus-forward limitation rather than a forward-accuracy success claim. Failed inner solves, cap/stagnation, optional-reference overflow, mandatory norm/BE overflow, failed construction and failure-state retention are independently exercised.

The tiny nonsymmetric Matrix Market fixture with explicit `epsilon=0.5`, beta8, alpha1, tau1e-8, default n-inner/20-outer caps converges in one outer correction and three inner iterations. Recorded final original residual_inf is `1.1102230246251565e-16`, nBE `5.4015805797291102e-17`, cBE `9.6148134319178178e-17`; verified inner relative residual is `1.1719012182897898e-16`. Solution is `[0.13856406460551024,0.023094010767584994,0.18475208614068031]`, to be independently checked against `[12,2,16]/(50*sqrt(3))`. Telemetry reports one context, two immutable matrix initializations, eight uploads (six CSR arrays plus residual/zero guess), eight post-inner downloads and zero Arnoldi vector transfers. This explicit CLI demonstration is not the main epsilon0.3 reference configuration or a published reproduction.

**Preserved CLI failure:** the same tiny fixture at default epsilon0.3 and mandated alpha=ceil(3/8)=1 reports construction_failure: column2 exceeds its growth/refit budget, with no GPU context/correction and exit1. No alpha increase, convergence-tolerance change or silent fallback hides that failure. The explicit epsilon0.5 demonstration is separately labelled; it does not alter the default contract.

No timing or speedup measurements are performed. Gate 2 work stops at this implementation and its bounded validation; no subsequent stage is authorized here. CPU outer staging adds two vector uploads and diagnostic downloads per correction; debug snapshot memory/traffic, serial safe norm and scalar synchronization are known unoptimized limitations. A supported GPU is required even when initial CPU diagnostics would already converge because the context is initialized before that check.
