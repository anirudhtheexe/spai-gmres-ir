# SPAI row-kernel independent validation record

Date: 24 October 2026. Status: source review and validation plan only; no
independent build, solver execution, or timing result has been obtained for this
change. This record concerns the proposed row-parallel checks in
`src/cuda_spai.cu`, compiled for FP64 and through `src/fp32_cuda_spai.cu` for
FP32. The numerical contracts are `PHASE2A_ALGORITHM_SPEC.md` and
`EXPERIMENT2A_FP32_SPEC.md`. Existing CUDA validation evidence is in
`PHASE2B_VALIDATION_RECORD.md`; it predates this change.

## Source review

The gather kernel now writes each `B(I,J)` entry once to the QR work matrix and
an immutable original matrix. Row kernels use the latter after cuSOLVER changes
the QR work matrix. Each fitted residual still sums columns in ascending `J`
order. The finish kernel checks coefficients and residuals, and applies the
existing scaled norm recurrence in ascending shadow-row order. The prior
kernel repeated the same row dot products separately for reduced and full
diagnostics; the new code stores one residual vector and reports its norm in
both fields. Since the shadow contains the complete support of each fitted
column and the target row, those two norms should agree, subject to numerical
verification. Candidate scores still consume this residual vector.

The QR guard still checks transformed RHS, factor entries, tau, and exact-zero
pivots, and computes the ratio from original reduced-row absolute sums. The
split changes scheduling and adds an `m*q` device buffer plus `m` row flags;
it does not visibly change pivot thresholds or score policy. Stream ordering
connects gather, QR, row checks, triangular solve, residual, and score kernels
within each slot. This is a source-level observation, not proof of identical
finite-precision results. In particular, FP32 compiles the shared body by
mapping `double` to `float` and D CUDA APIs to S APIs, so both specializations
require execution evidence.

Potential discrepancies to investigate include changed fit/P bits due to
compiler contraction or storage, the first failed status when several QR
conditions coexist, exact epsilon-boundary acceptance, and any changed
coefficient or score history. A nonfinite row absolute sum can make the pivot
ratio nonfinite without setting `row_bad`; this behavior already existed in
the original guard and is outside the proposed optimization's claim. Do not
silently alter its failure policy during this validation.

## Numerical admission plan

1. Freeze the pre-optimization and candidate source hashes, compile each with
   the same compiler, flags, CUDA libraries, device, and build type, and run
   them serially. The pre-optimization baseline must be the state immediately
   before this row change; the repository HEAD is older and includes other
   differences. Capture all build commands, logs, and executable hashes.
2. For both precisions, compare constructor and per-column status, fit/growth
   counts, attempted and solved patterns, shadow/candidate/selected lists,
   coefficient bits, reduced/full residual bits, QR pivot-ratio bits, score
   and clamp histories, scaling bits, and serialized `N` and `P` patterns and
   values. An expected bitwise mismatch still needs an independent numerical
   explanation; do not raise tolerance to pass it. Include an epsilon-equality
   boundary, subnormal row, uneven column histories, exact score tie, near
   dependent fit, rank/invalid input, and empty-candidate or capped cases.
3. Run both CUDA SPAI suites, `check_cuda_spai_decimal.py`, the applicable
   CUDA IR/CLI suites, and the FP32 CUDA GMRES/IR integration checks. Independently
   reconstruct fit residuals and restricted scores from serialized binary
   values, plus original-system backward error of terminal IR solutions. The
   existing FP32 focused test is a small successful construction and does not
   by itself cover all FP64 edge fixtures or all FP32 score histories.
4. Check source-issued peak bytes and allocation counts. The duplicate dense
   reduced matrix can be material for large patterns; vendor memory remains
   outside this ledger. Report any failure or numerical divergence before
   collecting performance results.

## Performance method, after numerical admission

Use an immutable input with recorded SHA-256, precision, options, and known
successful baseline/candidate outputs. Record GPU model, clocks/power policy,
driver/runtime/cuSOLVER/cuBLAS, compiler/flags, CPU, and free device memory.
Run one process at a time, alternate baseline and candidate order, warm both,
then retain repeated measured trials with all statuses and numerical checks.
Use CUDA event or profiler GPU durations for `qr_guard`, `fitted_residual`,
gather, QR, and total kernels; use synchronized host wall time for the full
constructor and separately report setup, adaptive fitting, and packing.
Report medians and spread with raw samples, along with peak source-issued
device bytes and allocation counts. The historical 39.7% and 23.9% kernel
shares in `SPAI_ROW_KERNEL_OPTIMIZATION.md` are motivation only, not a speedup
prediction or current measurement. No performance result is asserted here.

## Candidate FP32 stencil measurement, 24 October 2026

The parent agent reported that the candidate `build/cuda-NoPdb` CUDA build
passed all eight CTest suites, with 436 SPAI Decimal80 checks, 160 IR Decimal80
checks, and 32 CLI checks. These are parent-reported results; this independent
run measured only the FP32 GPU SPAI constructor. The experiment is exploratory
because no same-flags immediate pre-optimization executable was available.

The exact archived `tests/experiment2a_fp32_benchmark.cpp` was extracted to
`build/spai-row-validation/benchmark.cpp` (SHA-256
`002a799f12eee26a16baeb0800b1fffb2f3e48ff043c43778d6bbdc07ba327c5`).
The archived `data/stage45b_fp64_scalability_v1/stencil_8192.mtx` was extracted
beside it; its SHA-256 is
`90d25d6332cc3729030c13e89deb74169b13aef8c1fe89b1ffa4af6709f09b1f`,
matching its provenance file. The 64-by-128 grid matrix has `n=8192`, uses
fixed generator weights and no random seed. The benchmark's matrix reader
rounds stored entries to FP32 before constructing the input CSR.

The validated libraries are from the existing MSVC 19.44.35228 / CUDA 13.4
Release build in `build/cuda-NoPdb`. A separate harness was compiled with
MSVC `/O2 /MD /Z7 /EHsc /std:c++17`, linked against those libraries and CUDA
runtime/cuBLAS/cuSPARSE/cuSOLVER. Executable SHA-256:
`0D7F0892C35567380D3F68BF1A2220C1053FD7FF6FA37EB071A8FF471C6D7970`.
CUDA source hashes at measurement: `src/cuda_spai.cu`
`A35A66A63694F3C252ACDEBB3C4E2728BBCDEE9FEC5924F27D563FEBB5DEC1E9`,
`src/fp32_cuda_spai.cu`
`11D7451DF966FFAA3F18649AF1DB93D7FB73F84696F13A238EE9DE36BD75E604`.
GPU: RTX 4060 Laptop, driver 616.92, reported memory 8188 MiB. The sampled
idle SM clock was 225 MHz; running clock and power policy were not recorded.

The standalone CMake attempt failed at its compiler probe with this host's
known MSVC `C1902` PDB manager mismatch, including a retry that appended
`/Zi` after `/Z7`. Direct `cl` compilation with `/Z7` succeeded. The serial
measurement command, after prepending the CUDA 13.4 `bin` directory to PATH,
was:

```powershell
& build/spai-row-validation/spai_row_benchmark.exe `
  --matrix build/spai-row-validation/stencil_8192.mtx `
  --mode gpu_spai --epsilon 0.3 --alpha 1024 --beta 8 `
  --innercap 8192 --warmups 1 --trials 3
```

The harness synchronizes the GPU before and after each timed public API call.
It produced one warmup at 2381.420 ms and measured wall times of 2334.858,
2352.447, and 2393.941 ms; median 2352.447 ms, range 59.083 ms. Every run
reported success, 40,197 P nonzeros, 16,257 fits, 8,065 growths, and 253,747
score evaluations. The measured telemetry adaptive-fit wall intervals were
2315.617, 2332.755, and 2374.342 ms. Source-issued peak live device bytes
were 1,602,520, with 496 allocation calls in each run; this excludes vendor
internal allocations and CUDA context. Raw samples are saved in
`build/spai-row-validation/candidate_samples.jsonl` as PowerShell UTF-16 text,
SHA-256 `c9a248ea034aa7cd0f3a4bcbc338acdf0b6771951b8aa1d9558bb495831fa200`.

The archived prior experiment reports the same construction counts for this
matrix, but it was built and timed on an earlier source/toolchain state.
Repository HEAD also predates unrelated cleanup and packing changes. Neither
is an immediate, same-flags baseline for this row-kernel change. No speedup,
regression ratio, or kernel-time reduction is inferred from these timings;
the archived kernel percentages remain motivation only. A defensible paired
comparison requires reconstructing the exact pre-change body in an isolated
copy, checking its numerical outputs, and then alternating serial timed runs
under identical build flags and environment.

## Same-flags three-variant FP32 comparison, 24 October 2026

This follow-up supersedes the candidate-only speedup limitation above for the
specific FP32 stencil input. It remains exploratory rather than a general GPU
performance result. All sources and executables below are validation-only
under `build/spai-row-validation`; production source and `build/cuda-NoPdb`
were not edited.

`reconstruct_baseline.py` (SHA-256
`1BA47C3674A5573C28240BF9A4D3640AA8FF9DEC9E4047A5A7C02E5CEEE7F173`)
copied the current shared CUDA body, replaced its gather/QR guard/fitted
residual kernel section with the exact old section from Git HEAD, removed only
the two new slot buffers and their allocation/telemetry sums, and restored
the old kernel launch arguments. This retains all current unrelated packing,
scoring, batching, and formatted control flow. Its output
`baseline/cuda_spai.cu` has SHA-256
`57BA4F7F092CEFE8E0353C324EF80C623EC75B5D98D1743A70C937AF850925FB`.
The `baseline/fp32_cuda_spai.cu` specialization is a byte-for-byte copy of
current `src/fp32_cuda_spai.cu`, SHA-256
`11D7451DF966FFAA3F18649AF1DB93D7FB73F84696F13A238EE9DE36BD75E604`.
The legacy kernel bodies agree with the pre-change style preserved in
`tests/reference/cuda_spai_gate3.cu`, accounting for its different dense-B
signature. The reconstructed source is a faithful *row-change baseline*;
it is not asserted to be an archival byte-for-byte file from 12 October.

`reconstruct_fused.py` (SHA-256
`9EE99E9D1D09AEA0637E297E2A9F3D9C15CC42D682DBE274A3E533837768B493`)
started from that baseline and changed only `fitted_residual`: after checking
the coefficients, it computes each row dot once, checks finiteness, stores
the residual, and applies the old scaled `accumulate` recurrence in ascending
row order. Both diagnostics receive the resulting norm. Its isolated shared
source SHA-256 is
`F3DC49CBAF6092E19C746E21E2AE650FFF29ABB751AA8C7F00F776D3A584F927`.
The gather, QR guard, slot storage, and launches remain the baseline ones.

The candidate, baseline, and fused FP32 CUDA translation units were each
compiled separately with CUDA 13.4 NVCC, MSVC 19.44, `-DNDEBUG`,
`--generate-code=arch=compute_89,code=[compute_89,sm_89]`, `--fmad=false`,
and host `/GR /EHsc /O2 /Ob2 /MD /W4 /fp:strict /Z7`. The same archived
benchmark source was compiled for each executable with `/O2 /MD /Z7` and
linked to the corresponding CUDA object plus the same validated
`build/cuda-NoPdb` FP32 GMRES, SpMV, and CPU libraries. The baseline,
candidate, fused executable SHA-256 values respectively are:

- `5C32CBB0E86F028543501EA9375A8E798EB4CAAE87CD5B7ADBDFB024903E9F76`
- `B3AD0A4AA62E1C2A3431788B1EB730F76084A0CCD36FB10744120553792771F2`
- `F87C8B01FB5D62ED9BE29E75C12D547D6AF22E315F618A10B13D0449731AA3AB`

The validation-only `probe.cpp` serializes public FP32 result data as integer
float bit patterns: scaling, B/N/P CSR, each column's status, fit/growth
counts, pattern, coefficients, residual and pivot diagnostics, shadow, and
exposed last-fit score history. Baseline, row-parallel candidate, and fused
probe outputs are all exactly 3,744,283 bytes with the same SHA-256
`0fcf0dfbe79ca9a5816e855ac6f612c286ad370ac2b79d740439e05448d4b79b`.
This establishes bitwise public output equality for this input. The FP32
wrapper does not expose every native fit's score history, so this dump is not
an exhaustive internal-history comparison. Each probe returned zero.

All timing runs used the same `stencil_8192.mtx` and frozen options above,
were serial, and each used one warmup plus three measured samples. The
benchmark synchronizes before and after each public API call. Raw PowerShell
UTF-16 JSONL files are retained under `build/spai-row-validation/`.

| Variant/order | Measured wall time (ms) | Median (ms) | Raw SHA-256 |
| --- | --- | ---: | --- |
| Baseline, first | 2281.071, 2282.430, 2291.831 | 2282.430 | `41e92027b4e6e839d745edaccc6fa5a5ee6ac5659de30f2d35ba3441ce5c6221` |
| Row-parallel, second | 2360.043, 2461.634, 2431.504 | 2431.504 | `f065f18eaf9d49db0d46d6d25f2c8d21a9f870029da2b3cdd92d076747e9ab0c` |
| Row-parallel, reverse first | 2426.400, 2530.083, 2425.623 | 2426.400 | `9c26001f6d723a3d0b7358ebab023e4b80b0509838d144d4fcb70a831cd4c7f1` |
| Baseline, reverse second | 2389.662, 2307.161, 2254.752 | 2307.161 | `a15f6606c8e37461710ac2f56367e3fd6638521dc70d133d707aa4d111ab729b` |
| Fused one-pass | 2146.068, 2153.269, 2247.020 | 2153.269 | `424ea2ae343540396a3ef8362ff32d1221bbc91aace3591f2fcb79d239b7b99f` |

All samples reported success, 40,197 P nonzeros, 16,257 fits, 8,065
growths, and 253,747 scores. The row-parallel candidate's two paired medians
were about 6.5% and 5.2% slower than their nearby baseline medians. The
fused median was about 5.7% below the first baseline median and 6.7% below
the reverse baseline median; this is a small exploratory sample taken after
those pairs, without a contemporaneous reversed fused/baseline pair.
Baseline and fused reported 1,600,024 peak source-issued device bytes and
406 allocation calls; row-parallel reported 1,602,520 bytes and 496 calls.
These are source telemetry values, not total GPU footprint. The mechanism
suggested by the data is that extra row kernels/launches and buffer churn
outweighed parallel row work for this sparse fit distribution, while removing
one serial residual pass helped. GPU kernel durations were not collected, so
the mechanism remains an inference. No conclusion is made for other matrix
families, larger reduced fits, FP64, or end-to-end IR performance.

## Final production admission, 24 October 2026

The isolated fused-only body was copied to production `src/cuda_spai.cu` and
formatted with the project clang-format configuration. A normalized diff
against the reconstructed pre-change body contains only the one-pass
`fitted_residual` change. Final production source SHA-256 is
`A45D86A148559A46F77D4E5355E401193A92A3403D63D5C5F34A0A176885765B`.
It was rebuilt in `build/cuda-NoPdb` with the flags above. CUDA CTest passed
8/8, FP64 SPAI Decimal80 passed 436 checks over 60 fits, CUDA IR Decimal80
passed 160 checks, and the four submission CLIs passed 32 Decimal80/error-path
checks. The sparse CUDA SPAI suite includes 3,363,954 allocation and behavior
checks. These final runs occurred after the production source was reduced to
the fused-only variant. The earlier row-parallel candidate test results are
not being used as admission evidence for the final source.
