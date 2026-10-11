# Stage 4: sparse-storage GPU SPAI construction

Implementation and scoped independent validation complete on `stage4-scalable-spai`; preserved tag `gate3-working-baseline`. Stop here for parent acceptance; no next stage is started.

## Scope and source basis

This is a storage/workspace replacement, not a new SPAI algorithm, precision policy, convergence criterion or performance study. The binding reference is [PHASE2A_ALGORITHM_SPEC.md](PHASE2A_ALGORITHM_SPEC.md), Carson–Khan SPAI §2.2/Algorithm 2.1 and the published BSPAI paper §4 orientation. Research independently confirmed original A CSR is exactly B CSC for B=A^T D. Sparse CSC/reduced extraction is established prior art (HybridSPAI §3.2); no novelty claim is made.

## Intended change and numerical invariants

- Store B in CSC using original A CSR offsets/indices, GPU-scaled values. Zero-fill absent entries via sparse lookup; do not drop small values or change structure.
- Allocate each QR slot from actual m=|I|, q=|J|, candidate count and queried cuSOLVER workspace. Retain grow-only capacity within a call, not n*n capacity.
- Preserve exact sorted metadata/order, native FP64 cuSOLVER QR/Dormqr/Dtrsv, exact-zero computed pivot guard, restricted scores, bounded clamp, epsilon/alpha/beta and complete-column failure semantics.
- Final fitted patterns/coefficients are compactly packed by host. Packed N column k becomes P row k; GPU computes coefficient*d[j]. No dense N/P and no CPU numerical fallback.
- Keep existing GPU GMRES/hybrid IR integration and its CPU outer staging. Final sparse P still passes through host canonical CSR then uploads into the persistent context.

Removing zero-only full-residual rows preserves the norm's mathematical value: complete shadow includes every nonzero of B(:,J), and nonzero diagonal ensures k is in the shadow. Within each included row, coefficient products remain in ascending J order. General cross-backend equality remains bounded, not globally bitwise promised.

## Memory and transfer contract

`CudaSpaiTelemetry` counts source-issued successful cudaMalloc calls, cumulative bytes, current/outstanding allocations, peak live bytes and maximum allocation. Vendor-library handle/internal allocations are excluded. B bytes include CSC offsets/indices/scaled values; output bytes cover packed row-index/N-value/P-value buffers (host offsets not uploaded). Reduced dimensions/elements and queried workspace/slot maxima are separately recorded. Current/outstanding counts are zero after normal public return; exceptions remain RAII cleaned.

Explicit host staging: O(nnz(B)) scaled values/support CSR, sorted adaptive sets, per-fit numerical diagnostics/coefficient/score downloads, final fitted coefficient repacking/upload, P value download and canonical CSR packaging. Full histories remain enabled and can dominate host memory; no bounded-history redesign is authorized. Device memory is O(n+nnz(B)+packed output+sum of retained compact slot capacities/workspaces), not an unconditional linear bound: difficult patterns can grow to dense reduced systems or dense P.

Gate3 dense numerical source is preserved at `tests/reference/cuda_spai_gate3.cu` with exported symbols renamed. Its separate BUILD_TESTING-only library is never linked as a production fallback.

## Validation plan

Rebuild/run all frozen CPU/GPU suites. Independently compare sparse vs CPU and frozen Gate3 on patterns, fitted coefficients, residuals, scores, statuses, P and end-to-end IR, including guard/stress/epsilon boundaries. Add deterministic progressively larger sparse fixtures with bounded local patterns and record actual allocator peaks, maximum reduced shapes and vendor workspace sizes; run memcheck/initcheck. Preserve numerical discrepancies without changed thresholds. No timing/overlap/speedup claim is authorized.

The plan was executed below; the independent authoritative record is [PHASE2B_VALIDATION_RECORD.md](PHASE2B_VALIDATION_RECORD.md).

## Initial owner-side evidence (not final independent Stage4 acceptance)

`cmd /d /c scripts\build_cuda_msvc.cmd` configured/rebuilt successfully, CUDA CTest4/4. Existing CPU executables in build/ninja reran 37+190+280+439=946 checks, zero failures. Existing CUDA suites reran 323SpMV/871GMRES/665hybrid/1534SPAI checks, zero failures. Existing Python scripts check_cuda_spai_decimal.py/check_cuda_ir_decimal.py/check_cuda_gmres_decimal.py/check_cuda_spai_cli.py/check_cuda_ir_cli.py reran 436SPAI60fits,43GPU-P IR5snapshots,383GMRES,160hybrid21snapshots,15GPUCLI and23originalCLI checks, zero failures.

Ordinary/stress SPAI metrics exactly match Gate3's recorded aggregates: ordinary coefficient difference3.3306690738754696e-16 and residual difference1.1430106177670572e-16; stress relative coefficient difference9.5447261316827792e-8, accepted true full residual5.7721292910986588e-7. The unsupported-singular CPU rank_deficient/GPU empty_candidates discrepancy remains unchanged. This does not prove equivalence on all matrices.

Tiny CLI epsilon.5 has source device peak35136bytes, sparse B144bytes, packed output72bytes, max reduced elements3, queried QR workspace12800bytes and outstanding allocations0. This is one tiny fixture, not a scaling result. A source-audit allocator telemetry bug (resize did not decrement outstanding count) was corrected before freezing/testing new Stage4 allocation assertions; numerical code/criteria were unaffected. Removed obsolete unused host lookup helper; generated CUDAstub C4211 warnings remain unsuppressed.

All compiled execution uses approved host execution on the existing RTX4060Laptop/CUDA13.4.59/MSVC19.44 x64 stack due the recorded sandbox DLL-relocation issue. No installations/system changes or timing experiments were performed. Final independent evidence is recorded below.

## Progressive sparse tests: owner-side execution of independent-authored suite

New `cuda_spai_sparse_tests` target links the validation-only Gate3 reference and a validation-owned source-instrumentation TU, not a production fallback. Normal production executables' link graphs exclude that reference. The allocator audit wraps actual source cudaMalloc/cudaFree calls after vendor headers and verifies call counts/cumulative bytes/live peak/largest allocation/normal-return cleanup exactly against telemetry. BUILD_TESTING=OFF omits both reference and audit target.

Owner-side rebuild via the same script passes CUDA CTest5/5 and the new suite3,363,954checks0failures. Inputs are deterministic nonsymmetric tridiagonal families: row scaling2^(i%5-2), lower2,upper1,diagonal8 or5 before scaling, epsilon.3,alpha5,beta8. Diagonal8 completes initial fits; diagonal5 exercises adaptive growth. CPU comparisons cover n<=256, dense Gate3 comparisons n<=64; n1024 has independent full sparse residual/support checks. A manufactured-solution n64 end-to-end comparison also passes. No random seeds or timing methodology apply.

| n | nnz(B) | peak bytes, diagonal8 | peak bytes, diagonal5 |
|---|---|---|---|
|16|46|106168|173808|
|64|190|111736|181680|
|256|766|134008|213168|
|1024|3070|223096|339120|

Diagonal8 max reduced shape3x1, queried workspace12800bytes; diagonal5 max5x3, workspace20992bytes. At n1024 diagonal5 packed output73656bytes (solved patterns, including any zero coefficients), largest individual source allocation24560bytes. Counts/capacities are allocator observations, not virtual-memory estimates. Vendor internal memory remains excluded. These bounded local-pattern families establish absence of unconditional n² construction buffers on tested paths, not an arbitrary-sparsity guarantee, asymptotic runtime result, physical stream overlap or speedup.

These owner-side results are corroborated by the final independent results below.

## Final independent evidence and reproducible commands

Validation independently rebuilt and passed **3,363,954 new sparse C++ checks**, **395 new80-digit SPAI checks over48fits**, and **208 new80-digit original-system IR checks over3states**, zero failures. The large C++ count is mostly exact outside-shadow zero assertions; it does not represent millions of distinct problems. Complete new-suite Compute Sanitizer memcheck reports0errors/0leaks, initcheck0errors, without suppressions/exclusions. Its independent allocation pointer ledger exactly matches all reported source allocation counts/cumulative bytes/current/peak/largest bytes and normal-return outstanding counts. No allocation/API fault injection was performed; exceptional paths remain additionally source-audited.

Preserved suites independently pass946CPU/323SpMV/871GPU GMRES/665hybrid IR/1534GPU SPAI checks; CPU CTest4/4, CUDA CTest5/5. Existing436SPAI/43GPU-P IR/383GMRES/160hybrid Decimal and15GPUCLI/23originalCLI checks also pass. Independent source provenance reconstructed the exact Gate3 source hash from the preserved reference after undoing only exported-symbol renames and newline normalization. It is a validation-only library; default construction never calls it.

From project root (same installed tools/flags as Gate3):

```powershell
cmd /d /c scripts\build_cuda_msvc.cmd
$env:PATH='C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin\x64;'+$env:PATH
$env:CUBLAS_EMULATE_DOUBLE_PRECISION='0'
& build/cuda-ninja/cuda_spai_sparse_tests.exe
& 'C:/Users/Anirudh/AppData/Local/Programs/Python/Python312/python.exe' tests/check_cuda_spai_decimal.py build/cuda-ninja/cuda_spai_sparse_tests.exe
& 'C:/Users/Anirudh/AppData/Local/Programs/Python/Python312/python.exe' tests/check_cuda_ir_decimal.py build/cuda-ninja/cuda_spai_sparse_tests.exe
& 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.4/bin/compute-sanitizer.bat' --tool memcheck --leak-check full --error-exitcode 9 build/cuda-ninja/cuda_spai_sparse_tests.exe
& 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.4/bin/compute-sanitizer.bat' --tool initcheck --error-exitcode 9 build/cuda-ninja/cuda_spai_sparse_tests.exe
```

Independent execution uses build/validation-cuda with separately configured/rebuilt strict flags. Existing CPU/frozen oracle/CLI rerun commands are in the authoritative validation record; use its exact final commands/hashes for independent reproducibility.

| Production/artifact at validated freeze | SHA-256 |
|---|---|
|src/cuda_spai.cu|0F258C94616E636DC2EE3802D5E9534E851B193F5A756B2C2A0ADEDA89B8964B|
|include/bspai/cuda_spai.hpp|0866BD05990E2E707F60E4900A65101511FE562A726A3B0090A39BAD02BAA191|
|src/cuda_hybrid_main.cpp|AFECCF7AA0EF66AA5995AAC28928419EB24B482640DFD4A5DE426A3048D0DB4D|
|cuda/CMakeLists.txt|CDA781D5A3F23C25B5AF24A539694E7147D95678C7AF8AC7B437582095B85D52|
|tests/reference/cuda_spai_gate3.cu|4BE8173114E422D89700BEDF0680037FC96C04FF137B6750B499355DDCA578ED|

## Remaining limitations and stop boundary

Storage validation is bounded to these deterministic sparse families through n1024 and the earlier tiny/stress cases. Full histories/host adaptive scheduling, coefficient repacking, CPU sparse packaging, serial safe-norm work, vendor allocation exclusion, sparse P download/reupload and CPU outer IR remain. Dense fill in actual J/I/P is not forbidden and can still consume substantial memory; unchanged unrestarted GMRES separately retains O(n*cap) Krylov/history buffers. This milestone does not make the entire solver universally linear-memory or optimized.

The singular computed-pivot status discrepancy and cuBLAS subnormal norm issue remain as documented in prior gates; neither is concealed or fixed by new tolerances/fallbacks. No arbitrary rank threshold, reduced-column padding, QR update, reorthogonalization, restart, precision change, bucketed SpMV, benchmark, speedup or published mixed-precision reproduction is claimed. Full-text P1/P2 methods were reconsulted; a fresh pinned MATLAB URL fetch was cache-missed this turn, so source equivalence relies on the preserved local CPU/Gate3 references and prior pinned audit, not a new upstream fetch claim.

Stop after Stage4 implementation/validation. Further BSPAI/mixed-precision/performance stages require fresh explicit authorization.
