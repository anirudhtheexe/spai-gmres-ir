# Stage4.5A independent numerical assessment

This validation-owned record is distinct from the screening harness and Research data-provenance record. Stage4.5A is authorized for **untimed correctness/resource screening only**. No performance timing, profiling, optimization, BSPAI, mixed precision, production edit, reordering, restart, regularization, fallback or stopping-policy change is authorized here.

## Pre-execution policy

Preserve the approved `STAGE4_5_EXPERIMENT_PLAN.md` and `PHASE2A_ALGORITHM_SPEC.md`. Frozen ordinary epsilon=.3,beta8,alpha=ceil(n/8),tau=1e-8,innercap=n,outercap20,eta<=10*2^-53. Historical tridiagonal continuity rows retain alpha5. Each case/config runs in one untimed process with a300-second safety watchdog; timeout/allocation failure/OOM is `resource_abort`, not a pass or solver nonconvergence. No artificial memory cap is imposed. CPU/GPU free-memory snapshots are recorded where supported. Resource budgets may not trigger changed mathematical parameters. Every manifest entry is retained.

`tests/assess_stage45a.py` is an independent Decimal80 read-only numerical assessor of recorded binary64 inputs/results. It does not call the solver or manufacture a correction. The sparse original-system residual is evaluated from the stored A,b,x; manufactured x is independently evaluated against stored b rather than assumed exact. All available outer states and all terminal successful solves are checked, including unsuccessful cases' retained last states where finite.

The predeclared rowwise diagnostic residual rounding envelope for p stored row entries is

`max(64*u,gamma_(2p+3))*(|b_i|+sum_j |a_ij|*|x_j|)+(2p+3)*denorm_min`,

where u=2^-53 and gamma_k=ku/(1-ku). Norms and backward-error diagnostics propagate this component envelope with operation-length terms. This extends the historical64u tiny-case diagnostic envelope for long rows; **it does not change production epsilon, tau or eta**. Every reported successful IR terminal must retain recorded eta<=10u. Decimal eta>10u plus the propagated diagnostic envelope is a discrepancy; Decimal eta near that boundary is explicitly recorded, not silently rounded into success.

SPAI support/orientation/P assembly and each available fit residual are independently checked against rounded stored scaled B values; Decimal80 reduced QR is used on feasible fits. QR coefficients on arbitrary ill-conditioned systems do not have a universal tiny-case absolute bound. All feasible controlled fits are audited; for application matrices deterministic samples include fixed indices, maximum final pattern, largest final residual and minimum recorded pivot, plus every detected divergence. Samples/cost limits must be disclosed before claiming coverage. Whole-column full residual and P quality use sparse arithmetic where feasible, not dense PA storage.

GMRES acceptance uses independently reconstructed two-stage P*(A*x) and P*b, propagating A and P product rounding envelopes separately. Recorded accepted residual must obey tau; estimate-only acceptance is never enough. CPU/GPU native pipeline comparisons are distinct from same-P correction comparisons. Any differing status/pattern/selection/iteration history is retained and investigated rather than automatically dismissed as roundoff. Unsupported singular A/P cases cannot certify original-system solvability.

## Executed evidence (2026-10-11)

Independent execution completed eight SuiteSparse native CPU/GPU pairs (six pilots plus sherman4 and orsreg_1), six same-CPU-P CPU/GPU pairs, and two labelled negative pairs: 32 processes total, 28 solver successes and four deliberately retained solver nonconvergences. No watchdog timeout/OOM/resource abort occurred. No case beyond n=2205 was launched. Every process used the approved 300-second safety watchdog, without reporting elapsed duration as performance evidence. Raw records and initial assessments are retained under `results/stage45a_validation/`; final reassessment of the same records is `ASSESSMENT_FINAL.json`: **788034 independent checks, zero discrepancies, all 32 records assessed pass**. The negative cases pass validation of their failure semantics; they remain solver nonconvergences, not solver successes. No solver was rerun by that collector. Final aggregate SHA256: `FAF54A5E349C148EF3B74CF50801D20267394E249CE9F4C923CB0CEBF3E3C8F2`.

All successful terminal solutions passed independent Decimal80 original-system residual, normwise/componentwise backward-error and recorded stopping-policy checks. All available outer states, correction RHS, applied updates, inner candidates and recomputed preconditioned residuals were assessed. The stored terminal eta and independent Decimal eta are below the unchanged 10u criterion in these runs; no terminal boundary-uncertain success was observed. No performance, published mixed-precision reproduction or global nonsingularity certificate is claimed.

| Native matrix | n | P nnz CPU/GPU | Outer steps | Inner iterations CPU/GPU | Decimal nBE CPU/GPU |
|---|---:|---:|---:|---|---|
| cage5 | 37 | 213/213 | 2 | 11+12 / 11+12 | 1.994e-17 / 3.062e-17 |
| gre_115 | 115 | 1731/1739 | 2 | 17+17 / 17+17 | 3.599e-17 / 3.726e-17 |
| pores_3 | 532 | 9265/9265 | 2 | 55+56 / 55+56 | 9.583e-17 / 6.477e-17 |
| steam1 | 240 | 961/961 | 2 | 7+7 / 7+7 | 3.698e-21 / 3.621e-21 |
| steam3 | 80 | 248/248 | 1 | 8 / 8 | 1.041e-16 / 1.041e-16 |
| saylr1 | 238 | 3396/3396 | 2 | 44+45 / 44+45 | 1.081e-19 / 9.700e-20 |
| sherman4 | 1104 | 8499/8499 | 2 | 49+50 / 49+50 | 5.363e-17 / 6.016e-17 |
| orsreg_1 | 2205 | 11025/11025 | 2 | 58+76 / 58+76 | 2.327e-16 / 2.040e-16 |

All six same-P pairs use a retained successful CPU-native canonical P, exported with 17 significant digits; they preserve original A/b and distinguish correction-backend behavior from constructing a different P. Status/outer/inner counts agree. Their maximum solution infinity differences are cage5 1.110e-16, gre_115 1.804e-16, pores_3 3.134e-13, steam1 1.388e-17, steam3 1.388e-16, saylr1 2.418e-15. They are not bitwise-equivalence claims.

The labelled cage5 fixed-P `negative_outer0` pair returns `solver_nonconvergence/iteration_limit` with zero inner attempts. The labelled `negative_inner1` pair returns `solver_nonconvergence/inner_failure` and an inner iteration-limit failure; the failed candidate is not applied. All four retain x0 and Decimal nBE 0.0740651020165. These are intentional lower-cap tests, not ordinary-cap failures or tolerance changes. Independent assessments pass 44/45 and 46/48 checks respectively.

## Provenance and oracle coverage

Independent raw parser/hash/byte/count/diagonal/scaling checks corroborated all ten entries in `data/stage45a_provenance_20261011/MANIFEST.json`: 111091 checks, zero failures. Original explicit zeros remain in the acquired files and are omitted by canonical CSR as documented. These static checks are not rank or convergence certificates. Auxiliary published array-format RHS files were preserved but not used; screening uses the harness's stored binary64 equal-unit-2 RHS. No random seeds or manufactured-solution truth are involved.

All recorded SPAI fits have independently recomputed reduced/full residuals, support, scaling, scoring, selection, fitted-pattern association and P assembly checks. Independent Decimal80 QR checks are sampled, not exhaustive: CPU/GPU checked/remaining fits are cage5 11/54 each; gre_115 24/358 and 24/359; pores_3 65/2261 each; steam1 19/477 each; steam3 19/145 each; saylr1 33/1022 each; sherman4 17/2153 each; orsreg_1 13/6161 each. The predetermined sample indices and m*q^2<=100000 feasibility limit are retained in each assessment. Application coefficient differences are recorded rather than accepted by a universal condition-independent coefficient bound. orsreg_1 maximum scaled sampled coefficient differences are CPU 1.156e-13/GPU 1.700e-14.

P quality is recomputed sparsely in Decimal80, with `frobenius_BN_minus_I` and actual assembled `frobenius_I_minus_PA` reported separately. Initial draft assessments had labelled the BN quantity as PA; that validation-only metric-label error was corrected before final reassessment. Initial artifacts are preserved, not overwritten. No production result changed. For orsreg_1 actual ||I-PA||F is about 11.4477, while every successful individual column residual is <=epsilon=.3; neither is a forward-accuracy certificate.

Independent sparse Gaussian elimination with partial row pivoting at 80 and 120 decimal digits was performed for all six pilots. Predeclared relative precision-consistency threshold 1e-50 passes (observed about 1e-76--1e-79); reference residual/backward errors are about 1e-120 or smaller. Files `{matrix}_reference80_120.json` retain methods and results. Relative forward errors CPU/GPU: cage5 1.058e-16/1.022e-16; gre_115 3.223e-16/2.774e-16; pores_3 7.413e-13/6.567e-13; steam1 1.732e-16/1.732e-16; steam3 8.857e-11/8.857e-11; saylr1 5.783e-15/2.002e-14. In particular steam3's small backward error does not imply near-unit-roundoff forward accuracy. No direct high-precision solve was performed for n1104/n2205; those have Decimal residual/BE and sampled QR checks only.

## Retained CPU/GPU decision discrepancy

gre_115 final patterns differ in columns 5,66,75; P nnz differs 1731 versus 1739. The first differing selected set is column5 initial fit. Exact Fraction arithmetic on the stored rounded B proves equal squared scores for the beta-boundary group [46,59,80,90,103,105,109,111], approximately .5765704159646164402. Binary64 CPU scores split that group between .5765704159646166 and .5765704159646167; GPU scores are all .5765704159646166. The fitted singleton coefficient differs by two ulps. CPU selects [20,13,23,24,46,59,90,103]; GPU selects [13,20,24,23,46,59,80,90]. Both obey their computed-score ordering/exact-computed-tie rules; this explains, but does not remove, the pattern-equivalence limitation. Raw fits and exact rational witness are retained in `gre_115_native_comparison.json`. No near-tie tolerance, forced pattern, fallback or precision change was introduced. Both finish with two outer and 34 inner iterations; solution infinity difference is 1.665e-16. Other native pairs have equal final patterns/statuses, but some selected order differs; exact metadata equivalence is not claimed.

## Resource feasibility and scope limits

GPU source-owned SPAI peak bytes for the eight native cases, in table order: 832844,4043964,3446864,1368384,669456,5275736,1839688,1209724. All returned current/outstanding source allocations are zero. Vendor-library internal allocations and GMRES/context/history allocations are not included in this SPAI ledger. orsreg_1 source allocation calls=849, cumulative issued bytes=3339108, largest allocation=113064, sparse B bytes=243776, packed output bytes=264600, max reduced shape25x5/workspace67072, eight streams. These corroborate feasibility for this case, not universal linear memory.

orsreg_1 GPU-process snapshots: host free before/after 1984843776/1788932096 bytes (host total16308531200), GPU free before/after7451181056/7449083904 (total8585216000), process peak working set566611968. They include context/library/history effects and are not equivalent to source allocation counters. The after-GPU difference alone is not a demonstrated leak. No artificial memory cap was imposed. Unrestarted GMRES and retained full candidate/basis/outer histories can still require O(n*cap) or worse aggregate storage; QR patterns can become dense. n3312/n4960 are **not run**, not passes, failures or resource aborts.

Controlled tridiagonal continuity, newly proposed 2-D/irregular/arrowhead matrices, manufactured RHS and new synthetic larger tiers are deferred/unvalidated in Stage4.5A: their approved generator/infrastructure was unavailable. Historical Stage4 controlled results are not substituted for these missing runs. No suite entry is silently excluded. No timed performance experiment, profiler, sanitizer rerun or Release revalidation was performed here.

## Reproduction commands and environment

All commands run from the repository root. Windows sandbox DLL relocation required approved host execution for compiled binaries. Python3.12 path `C:/Users/Anirudh/AppData/Local/Programs/Python/Python312/python.exe`; below `python` means that executable. Build the independent screening target after vcvars64 using bundled CMake: `cmake --build build/validation-cuda --target untimed_screen`. The existing independent build is Debug/MSVC19.44.35228 x64, host /W4 /fp:strict, architecture89, custom CUDA --fmad=false. Native FP64/PEDANTIC and deterministic policies remain frozen; CUBLAS_EMULATE_DOUBLE_PRECISION=0. Runtime environment: RTX4060 Laptop CC8.9, driver616.92, CUDA13.4.59 runtime/driverAPI13040, cuSPARSE12806, cuBLAS130700, cuSOLVER12.3.2, binary64 digits53. No warm-up/repetition/timing policy applies to this untimed one-process-per-case screen.

```powershell
python tests/assess_stage45a.py data/stage45a_provenance_20261011/MANIFEST.json --provenance
python tests/run_stage45a_assessment.py --cases cage5 gre_115 pores_3 steam1 steam3 saylr1
python tests/run_stage45a_assessment.py --cases cage5 gre_115 pores_3 steam1 steam3 saylr1 --backends cpu_fixed_p gpu_fixed_p --fixed-p-from cpu_native
python tests/run_stage45a_assessment.py --cases sherman4
python tests/run_stage45a_assessment.py --cases orsreg_1
python tests/run_stage45a_assessment.py --cases cage5 --backends cpu_fixed_p gpu_fixed_p --fixed-p-from cpu_native --label negative_outer0 --outercap 0
python tests/run_stage45a_assessment.py --cases cage5 --backends cpu_fixed_p gpu_fixed_p --fixed-p-from cpu_native --label negative_inner1 --innercap 1
python tests/collect_stage45a_assessment.py results/stage45a_validation --output results/stage45a_validation/ASSESSMENT_FINAL.json
python tests/compare_stage45a.py results/stage45a_validation/gre_115_cpu_native.jsonl results/stage45a_validation/gre_115_gpu_native.jsonl --output NEW_COMPARISON.json
python tests/reference_stage45a.py results/stage45a_validation/steam3_cpu_native.jsonl results/stage45a_validation/steam3_gpu_native.jsonl --output NEW_REFERENCE.json
ctest --test-dir build/ninja --output-on-failure
ctest --test-dir build/validation-cuda --output-on-failure
```

Artifacts use exclusive creation. For a rerun, select a new `--output-dir`/output filename; do not overwrite retained evidence. Comparison/reference commands were run for each relevant pair; existing named artifacts remain the evidence. Raw JSONL retains exact source command, input/runner/source hashes, provenance bytes/base64, derived canonical/RHS hashes, compiler and resource metadata.

Fresh independent regressions pass CPU CTest4/4 (37+190+280+439=946 checks) and CUDA CTest5/5 (1534 SPAI+3363954 sparse+665 IR+871 GMRES+323 SpMV=3367347 checks), zero failures. CTest's incidental durations are not performance results. Logs are retained under each build's `Testing/Temporary/LastTest.log`.

SHA256 screening runner: `651B750D8B761F3B7036C7F6E8E24FC79EAF2838495E08DD144F967FA6966A97`. Validation assessor `2E481CE8F9F8ACCC802465DE825E364B9F4FC6043DA59C35EC40E0675510F4EF`; coordinator `D21F9E963AE81262178AB2EFE1873C5397C91F1E30C0C3E61F0348A257144374`; collector `CF8553387994FDC3A7EB5778884D63E5D5B3A71B4C2D85A20CB1BE8EF773315D`; direct reference `FBF5F2455C5F8678935A687FCD3C86141B28370736ADF9A8A6756303306CA4C3`; comparison `40D74815DAB4405326D75067267BB56E95720E59B2A1E7FC326CEAB1006DC0F0`.

Production cuda_spai.cu remains `0F258C94616E636DC2EE3802D5E9534E851B193F5A756B2C2A0ADEDA89B8964B`, cuda_gmres.cu `A208711A73C13A58244B3A3D79F88446E000D9566C2AA47E186B3F541F22C0C8`, cuda_ir.cpp `8DC6B5492370C21D9995AE32FA43FDA585BB138A5C891EF664EDD5D4C01B8077`; all source/header hashes are pinned in raw records. No production or existing regression test was edited by validation.

## Recommendation before Stage4.5B

Do not interpret this screen as benchmark authorization. First complete the missing controlled/manufactured tier and explicitly decide whether to extend application tiers given available memory/history costs. Preserve gre_115's decision divergence and steam3's forward-error limitation as benchmark qualifications, not exclusions. Freshly rebuild/revalidate any Release configuration before timing; require a named baseline, unchanged A/b/P/options, separated construction/solve/transfer phases, warm-up and repetitions/statistics and an approved timing plan. The watchdog and free-memory snapshots cannot support speedup claims.

## Append-only independent extension assessment — 11 October 2026

This supersedes only the earlier statements that controlled/manufactured rows and `sherman5`/`add32` were deferred. The retained extension aggregate is `results/stage45a_extension/ASSESSMENT_EXTENSION_WITH_APPLICATION.json`: exactly 22 raw records, 1,068,784 Decimal80 checks, zero discrepancies, and all 22 `solver_success`/`pass`. The raw records, raw-hash reconciliation, and input manifests are retained rather than regenerated.

The 18 controlled CPU/GPU records carry periodic-7 manufactured diagnostics. The four application records correctly use `equal_unit2`; they have independent residual/backward-error checks but no manufactured forward reference. `sherman5` and `add32` application construction assessments sample feasible QR fits rather than exhaustively audit them. Native CPU/GPU pattern differences are retained: one `sherman5` column and 92 `add32` columns, respectively. No fixed-`P` application comparison exists. These results support bounded solver-success classifications, not CPU/GPU construction equivalence, exhaustive SPAI correctness, performance, scalability, or benchmark readiness by themselves.

Fresh regression verification used the explicit Visual Studio CTest executable: CPU 4/4 and CUDA 5/5 passed. The CPU sandbox launcher could not start the already-built executables (`BAD_COMMAND`), then the same untouched CPU tests passed with host execution. No test tolerance, solver code, or validation criterion changed.
