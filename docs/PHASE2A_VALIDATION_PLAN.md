# Phase 2A independent validation plan

Date: 11 October 2026. Scope: uniform double CPU baseline. This plan proposes acceptance thresholds; it does not claim published mixed-precision reproduction. Validation owns `tests/` and this document, and does not modify implementation or manuscript files.

## Evidence and baseline boundaries

The [original SPAI paper](https://arxiv.org/pdf/2202.10204), Algorithm 3.1 and section 4, constructs a left preconditioner by solving on the scaled transpose, uses an initial application to b, and solves correction equations with left-preconditioned GMRES. The [published BSPAI paper](https://link.springer.com/article/10.1007/s00366-025-02187-z), section 4, explicitly uses unrestarted GMRES. No bucketing or CUDA belongs to this milestone.

The pinned [upstream double-working solver](https://github.com/Noaman67khan/SPAI-GMRES-IR/blob/b25e6f7c1cf1c3035686e9adb5ef9f35e2256ae1/sgmresir3_spai.m) monitors forward/normwise/componentwise errors against MATLAB eps, but downcasts its reference solution and evaluates monitored residual in double. Future independent diagnostics must not rely solely on these monitored values. Paper unit roundoff is half C++ `numeric_limits<double>::epsilon()`; their distinction must remain explicit.

The subsequent [Research specification](PHASE2A_ALGORITHM_SPEC.md) resolves these as explicit project choices: alpha counts growth/refit events, construction uses <=, restricted-column candidate scoring with bounded negative-score handling, and P=N^T D. Inner stopping is recomputed relative preconditioned 2-norm residual <=1e-8; outer stopping is double normwise backward error <=10*2^-53 with cap20. These policies differ from the upstream monitor and are not a published reproduction claim. The current implementation milestone contains only Matrix Market loading and CSR operations. Numerical checks below remain pending an implementation/API.

## Deterministic independent component tests

`tests/test_cpu.cpp` is a standalone C++11 validation executable wired into the implementation build. Proposed loader contract: coordinate real/integer, general/symmetric; canonical sorted CSR; duplicate sums; exact zero removal; finite values; invalid/unsupported inputs rejected. Exact integer fixtures require exact agreement, without floating tolerances.

Checks cover unsorted entries, rectangular matrices, empty rows and zero-nnz inputs, duplicate cancellation/summation, symmetric expansion without diagonal duplication, one-based conversion, transpose orientation and transpose twice, vector length mismatch, invalid indices/banner/storage/field/symmetry, truncated/extra coordinates, nonfinite input and duplicate overflow. Floating SpMV is checked against independently accumulated long double using a fixed `8*machine_epsilon*sum_abs_products` bound for its three-term positive example. Wider precision is reported, not assumed.

The tiny nonsymmetric fixture is A = [[4,1,0],[2,5,1],[0,1,3]]. Its determinant is exactly 50 and its inverse is [[14,-3,1],[-6,12,-4],[2,-4,18]]/50. Manufactured x = [1,-2,3], b = [2,-5,7] are exact. Paper-style b = ones/sqrt(3) gives x = [6,1,8]/(25*sqrt(3)). These are project fixtures, not published SuiteSparse problems.

## Pending numerical acceptance checks

Proposed thresholds are fixed before execution. Changing them requires a recorded rationale and investigation of the previous failure. Let e = C++ double machine epsilon. Initial well-conditioned tiny-fixture checks use a scaled allowance `256*n*e` for algebraic relative errors; this is a proposed conservative CPU diagnostic threshold, not a theorem about arbitrary matrices.

| Component | Independent check | Proposed acceptance / limitation |
|---|---|---|
| Scaling/orientation | Form B=A^T D independently, D_jj=1/max_i abs(A^T_ij); compare produced P with N^T D. | Algebraic scaled tolerance above; check P*A, not A*P alone, and test nonsymmetric A. |
| SPAI construction | Evaluate each complete residual e_k-B*n_k in wider precision; record growth sets, residual norms, cap status. | With epsilon=1e-12, beta=8, alpha=2 growth/refits on tiny fixture, require actual residual <= epsilon plus scaled rounding allowance; alpha=1 is an explicit cap case. Hitting cap is a reported failure of requested accuracy, never implicit success. Test the explicit 32u score-clamp boundary separately. |
| Preconditioner application | Compare P*v with independently computed dense rational inverse on full-pattern tiny case. | Scaled allowance above; no normal-equations oracle. |
| Residual diagnostics | Recompute b-A*x independently, plus normwise eta=norm_inf(r)/(norm_inf(A)*norm_inf(x)+norm_inf(b)), componentwise max_i abs(r_i)/(abs(A)*abs(x)+abs(b))_i. | Agreement within propagated rounding allowance; handle zero denominator: 0/0=0, nonzero/0=infinity. Exact-solution residual test must terminate cleanly. |
| GMRES | Identity/diagonal/nonsymmetric examples, zero RHS, exact initial guess, Arnoldi breakdown, and intentionally unattainable tolerance. Recompute true and preconditioned residual separately. | Unrestarted, one-pass MGS; inner tolerance 1e-8 is recomputed preconditioned relative 2-norm residual, with n-iteration cap. No convergence claim from recursive residual alone. |
| Outer IR | Record x at every step, inner iterations/status, actual residual and both backward errors. Known-solution and cap/nonconvergence examples. | Validate specified double eta<=10*2^-53 and cap20 separately from independent wider-precision diagnostics; bound evaluation-rounding discrepancies explicitly. Proposed tiny well-conditioned forward/backward acceptance <=1e-12. Single uniform precision is not the double/quad published experiment. |
| Reference | Exact rational inverse/solution for integer tiny fixture; long double independent residual and dense pivoted solve after verifying mantissa width. | Wider precision is a secondary check, not IEEE quadruple. Retain reference at wider precision. If long double is not wider or ill-conditioning exceeds its useful accuracy, mark inconclusive and obtain an approved multiprecision oracle. |

Future SuiteSparse validation will use Research-pinned matrix identity, provenance/hash, dimensions, and configuration. Published iteration counts are comparison targets only when the input, precision policy, algorithm, and stopping instrumentation match; this uniform-double milestone cannot claim mixed-precision reproduction.

## Required evidence

Record matrix/file provenance, exact expected/reference values, hardware/OS/compiler/build flags, source revision, precisions, epsilon/alpha/beta, GMRES tolerance/max iterations/restart policy, outer criteria/cap, seeds (none for the fixed fixtures), exact commands, and all statuses/discrepancies. No timing or speedup claims are authorized. Loader tests do not establish SPAI, GMRES, IR, or mixed-precision correctness.

## Executed loader/CSR milestone

Independent direct build and test on 11 October 2026 passed **37 checks, 0 failures**, process exit code 0. Compile emitted no diagnostics. This result covers only loading, canonical CSR, multiplication and transpose; no solver exists in the tested API. No numerical solver, CUDA, bucketing, performance, or published experiment has been validated.

Environment: AMD Ryzen 7 7745HX; Windows 11 Home Single Language 64-bit, version 10.0.26200; MinGW.org GCC-6.3.0-1, g++ 6.3.0. Executable reports double mantissa 53 bits and long double mantissa 64 bits. Compiler flags use C++11, O0/debug, warnings and no fast math. Inputs are fixed, no random seed/warmups/repetitions/timing methodology applies because this is correctness-only.

Commands, from workspace root:

```powershell
C:/MinGW/bin/g++.exe -std=c++11 -Wall -Wextra -Wpedantic -fno-fast-math -O0 -g -Iinclude src/cpu.cpp tests/test_cpu.cpp -o tests/validation_cpu_tests.exe
.\tests\validation_cpu_tests.exe
```

The generated executable was subsequently moved unchanged to the ignored `build/validation_cpu_tests.exe` directory and rerun successfully. Future equivalent builds should set `-o build/validation_cpu_tests.exe` directly. No source file was moved.

An additional independent execution of the implementation's Ninja build test registration passed **CTest 1/1**:

```powershell
& 'C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/ctest.exe' --test-dir build/ninja --output-on-failure
```

SHA-256 at validation time (no Git revision was available):

| Artifact | SHA-256 |
|---|---|
| `src/cpu.cpp` | `071D6614723DFF930BBAD66F7E7C0CA9A0265B3620EE38FD607367FBF47EF37C` |
| `include/bspai/cpu.hpp` | `C498DFFB61D9807674CE2D88B1D0FB62DADA35ED9EB0FCCC8A60C30EDED5A8E1` |
| `tests/test_cpu.cpp` | `1469F4A3BEB8E511BB26D178B5B13FB366B9CD15D0DF8F23B40488FB6BA59A86` |
| `tests/fixtures/tiny_nonsymmetric.mtx` | `1E33F403765FE840682964D706917878DCB968B3F73E1CFB71529E82A87F691D` |

Known limits: no external dataset was acquired/tested here; the SuiteSparse cage5 input remains pending provenance/hash validation. Long double is wider than double here but is not IEEE quadruple and supplies no mixed-precision reproduction evidence. Large matrix/resource exhaustion, all valid Matrix Market encodings, and numerical solvers are beyond these 37 checks. Exact lowercase Matrix Market banner field strings are the supported implementation contract. No failures were observed in the executed suite; this is scoped evidence, not proof for every input.

## Milestone 2 SPAI validation

Scope is construction only; no GMRES, IR, BSPAI, CUDA or performance tests. New independent suite `tests/test_spai.cpp` preserves the previous 37 loader/core checks in their own file. API results and histories are checked against independently formed dense matrices, wider-precision MGS QR least-squares fits, and the tiny fixture's exact rational inverse. The implementation chooses Householder QR; the oracle is independently structured and never forms normal equations. Code inspection must additionally confirm the QR implementation, because agreement of final results alone cannot prove which algorithm was used.

Before first execution, the tiny well-conditioned algebraic allowance remains `768*machine_epsilon` relative to max(1,abs(reference)); independent full column residual allowance is epsilon plus `768*machine_epsilon`. Restricted scores, complete-candidate arithmetic mean, and deterministic tied candidate selection are checked separately. The epsilon equality probe uses a reproducibly recorded binary64 initial residual as epsilon, then one ULP below it; this is a branch-boundary check, separate from independent residual accuracy.

A pre-execution hand calculation corrects the earlier alpha=1 full-inverse proposal: for column0, B has column0 [1,1/4,0], initial coefficient16/17, residual [-1/17,4/17] on shadow{0,1}. Candidate2's restricted score is1/17, smaller than candidate1's approximately0.142; the mean filter permits only candidate2. Thus alpha=1 returns the solved pattern{0,2} with unmet1e-12 tolerance; alpha=2 permits the full inverse. Beta=8 is a maximum, not a requirement to add all candidates. No tolerance was changed to hide this budget distinction.

Additional prepared checks cover diagonal/no-growth success, alpha0 and alpha1 budgets, beta0 no-growth, singular tight-fit failure, zero diagonal/nonsquare/invalid epsilon, reciprocal scaling overflow, underflow of a scaled nonzero, and score helper cases (positive, bounded clamp, excessive negative, zero denominator, invalid/nonfinite input). Exact-zero QR pivots are local failure detection only, not a global singularity certificate. A loose construction tolerance may pass on a singular matrix; no universal singularity-detection claim is made.

The near-dependent nonsingular stress fixture A=[[1,1],[1,1+2^-30]] uses the exact analytic inverse. Its separate predeclared construction epsilon is1e-5 and inverse relative allowance2e-6; this tests retaining a small nonzero QR pivot without an arbitrary rank cutoff. It does not share the well-conditioned acceptance threshold. The pivot ratio must be positive and below1e-8. Score-clamp/budget behavior and all failures will be reported without altering production code.

MinGW32 double arithmetic uses the implementation's explicit `-msse2 -mfpmath=sse -ffp-contract=off` guard in addition to no fast math. Independent inspection of `build/ninja/build.ninja` confirms these flags on production `src/spai.cpp` and both test translation units; verbose Ninja rebuild also emitted them. Wider-reference calculations use the separately reported long double type. Source inspection verified implicit Householder reflectors applied to the RHS and triangular back substitution, with no normal-equations fallback.

### Authoritative independent execution result

Final suite: **190 SPAI checks, 0 failures; 37 prior loader/core checks, 0 failures; CTest 2/2 passed**, all process exit codes0. Direct compilation emitted no diagnostics. Same Windows/CPU/GCC environment as Milestone1, now with explicit binary64 intermediate and no-contraction guards. No random inputs, performance timing, warmups or benchmark repetitions were used.

The initial174 SPAI checks passed; the suite was expanded to183 with diagnostic accounting, numeric summaries and clamp-boundary brackets, then189 with the exact-zero QR fixture, and190 with an independently recomputed stress-case residual. Every executed version passed; tolerances were unchanged. These earlier counts are superseded by190.

| Observed quantity / case | Result |
|---|---|
| Tiny alpha2, epsilon1e-12, beta8 | success; nnz(N)=9, nnz(P)=9; each column3 completed fits and2 completed growths;0 score clamps |
| Tiny maximum wider full column residual | `7.0536646688527192e-16` |
| Tiny maximum absolute P-entry error versus rational inverse | `1.9317772208260475e-16` |
| Tiny alpha1 | `iteration_limit`; column0 fitted pattern{0,2}, full residual`0.23076923076923078`; no unsolved pattern was published |
| Tight singular [[1,1],[1,1]] | `empty_candidates`, explicitly failed requested accuracy |
| Exact-zero QR [[1,2],[1,2]] | `rank_deficient`; failed refit attempt retained prior solved coefficients/pattern and recorded attempted pattern |
| Near-dependent2x2, delta2^-30 | success; max inverse relative error`2.3669808690840455e-7` versus predeclared2e-6 allowance |
| Near-dependent residual | recorded column0`2.384185791015625e-7`; independent maximum wider full residual`2.3843361371181198e-7`, both below predeclared epsilon1e-5 |
| Near-dependent column0 pivot ratio | `3.2927233162283187e-10`, retained as nonzero |
| Score-boundary bracket | r2=denominator=1; dot=1+16*machine_epsilon permits clamp; next binary64 dot toward2 returns `invalid_score` |

Exact commands from workspace root:

```powershell
C:/MinGW/bin/g++.exe -std=c++11 -Wall -Wextra -Wpedantic -fno-fast-math -ffp-contract=off -msse2 -mfpmath=sse -O0 -g -Iinclude src/cpu.cpp src/spai.cpp tests/test_spai.cpp -o build/validation_spai_tests.exe
.\build\validation_spai_tests.exe
C:/MinGW/bin/g++.exe -std=c++11 -Wall -Wextra -Wpedantic -fno-fast-math -ffp-contract=off -msse2 -mfpmath=sse -O0 -g -Iinclude src/cpu.cpp tests/test_cpu.cpp -o build/validation_cpu_tests.exe
.\build\validation_cpu_tests.exe
& 'C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe' --build build/ninja --verbose
& 'C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/ctest.exe' --test-dir build/ninja --output-on-failure
```

Validation-only branch investigation `tests/probe_rank_cases.cpp` enumerated all25 rank-one matrices [[p,q],[p,q]] for integers p,q in1..5, with epsilon1e-12/alpha2/beta1 under the same compiler/precision flags. All returned failure:8 `rank_deficient`,17 `empty_candidates`. The first exact-zero branch case [[1,2],[1,2]] was retained in the main regression suite. This is documented branch coverage, not a global singularity classifier or a selected success benchmark.

```powershell
C:/MinGW/bin/g++.exe -std=c++11 -Wall -Wextra -Wpedantic -fno-fast-math -ffp-contract=off -msse2 -mfpmath=sse -O0 -g -Iinclude src/cpu.cpp src/spai.cpp tests/probe_rank_cases.cpp -o build/validation_rank_probe.exe
.\build\validation_rank_probe.exe
```

Final SHA-256 evidence for new/changed artifacts; unchanged CPU/header/test hashes are recorded above:

| Artifact | SHA-256 |
|---|---|
| `src/spai.cpp` | `FBC42843A3C52529217D5E2870D94E74D385AFA2C37FAE90B2A0517A424BF8EB` |
| `include/bspai/spai.hpp` | `085EF3D365BC2662DBDA6F4D039D2749138C9375396242C8FE7EBB9E3825D6B1` |
| `tests/test_spai.cpp` | `AC7A22B1ED2EE86ED544E08A04CC56A7E6C7F57A4E2A1191D0F8F3FCFADADCFF` |
| `tests/probe_rank_cases.cpp` | `8F2582189A290A1F345F0516888F2904E2AE67BC87AA5232F31311DCC0770203` |
| `CMakeLists.txt` | `E0F764FAAF07A4627F328B0668862A411AF789C7EDA7CA70F66F8EFD08B4C1B5` |

No discrepancies were observed against the scoped acceptance checks. Limitations: exact-zero rank guards do not certify nonsingularity; the loose-epsilon singular case is outside that claim. The normal-equations sensitivity comment describes the purpose of the near-dependent fixture; no alternative normal-equations implementation was benchmarked or compared. No SuiteSparse data was acquired/tested in Milestone2; no published counts are reproduction targets here. Wider long double is not quadruple. No full GMRES/IR solver, mixed-precision, BSPAI, CUDA, parallel correctness, or performance has been validated. Production code and manuscript were not modified by Validation.

## Milestone 3 GMRES validation

Scope: unrestarted uniform-double left-preconditioned GMRES only. Prior190 SPAI and37 loader checks remain unchanged. No IR, BSPAI, CUDA or performance work is included.

Before execution, the well-conditioned fixtures use inner tau1e-8 and independently checked forward error <=1e-12, with diagnostic agreement allowance768*machine_epsilon times the relevant norm scale. Cases include identity, negative/positive diagonal, the nonsymmetric3x3, noninteger RHS, nonzero/exact initial guess, and both exact and approximate SPAI preconditioners. Exact tiny inverse/rational solutions and wider long double residuals provide independent references. No GMRES oracle uses the implementation's Hessenberg data to derive the reference solution.

Research choices: only exact h_next==0 is breakdown; small finite Arnoldi subdiagonals remain. Zero Givens/backsolve pivot is explicit failure unless an existing well-defined candidate passes the recomputed criterion. The limit is1..n (defaultn), cap>n rejected, lower cap failure is iteration_limit and full-n failure nonconvergence. No tolerance relaxation, reorthogonalization or restart is introduced. The original residual is checked first: exact initial solution succeeds even when P*b=0. Otherwise ||P*b||2==0 is a named failure; zero RHS with nonzero initial residual does not silently substitute a denominator.

Independent checks will recompute each original residual and the specified preconditioned residual P*b-P*(A*x), distinguish its binary64 operational evaluation from the wider residual, check histories/iteration counts/verification triggers, and verify no success from an estimated residual alone. Explicit tau1e-30 stress cases investigate estimates below tolerance while the recomputed residual remains above; these are documented nonconvergence tests, not revised main stopping tolerances. Invalid dimensions/nonfinite input, zero preconditioner, singular/inconsistent equations, and arithmetic overflow are failure cases. Numerical instability investigations report forward error and basis orthogonality rather than assuming small residual implies an accurate solution.

Operator order fixture (no solve): A=[[1e16,1e16],[1,1+machine_epsilon]], P=[[1,-1e16],[0,1]], v=[1,1]. Under specified binary64/no-FMA arithmetic, two sequential products give[0,2], while explicitly rounded materialization of P*A followed by a product gives[-2,2]. An independent fixed-order dense loop tests the public two-product helper; the materialized alternative is only a validation contrast. Source inspection additionally checks the GMRES operator calls this helper, one MGS pass, and Givens/backsolve.

### Authoritative Milestone 3 execution result

Final independent execution passed **280 GMRES checks with zero failures**. The retained suites passed **190 SPAI checks** and **37 loader/CSR checks**, all with zero failures; CTest passed **3/3**. The first `tau=1e-16` gap probe is retained as an informative coverage miss: it was below both the estimate `1.2831951913535754e-16` and recomputed residual `5.1192497462734993e-16`, so the solver correctly returned `nonconvergence` but did not enter the intended branch. A separate `tau=2e-16` fixture then established the required rejection path: estimate <= tau, recomputed residual > tau, and no accepted solution. No operational tolerance, implementation rule, or acceptance threshold changed.

The identity, diagonal, nonsymmetric, zero-RHS/exact-initial, cap, breakdown, general-initial-guess, Hessenberg/MGS, and two-product operator checks passed. The SPAI-preconditioned 3x3 solve completed in three iterations with wider original residual `6.6901877333569169e-16` and operational relative preconditioned residual `4.0270357740038596e-16`. The strict `1e-30` full-cap case correctly returned `nonconvergence`.

An unsupported-input diagnostic found that singular `P=diag(1,0)` can make the preconditioned residual vanish while the original residual is nonzero. Per Research's documented ruling, this remains an explicit nonsingular-preconditioner input assumption; no extra original-residual convergence rule was silently added. The near-dependent stress case returned a failure status (`breakdown`) with wider original residual `2.935454526555012e-7` and relative forward error `6.7179421740666064e-8`, not a success claim.

The final rerun used the previously recorded CMake/CTest commands and the three saved executables under the same binary64/no-contraction/SSE2 guard configuration. No performance measurements, SuiteSparse reproduction, iterative refinement, BSPAI, CUDA, or mixed-precision validation occurred.

Detailed audit trail: the189-check initial suite passed but gap coverage was false. The expanded275-check draft with branch-probe tau1e-16 **failed one required coverage assertion**, exit1; there were no numerical/Arnoldi assertion failures. Choosing the separately documented2e-16 branch probe between the measured estimate and recomputed residual produced276 passing checks. Four final assumption/breakdown assertions produced280. The main1e-8 and strict1e-30 configurations stayed unchanged. No implementation or primary acceptance threshold was relaxed. Normalization/raw Arnoldi relations and used-basis orthogonality were independently checked; an unused final near-breakdown vector is excluded from orthogonality because n+1 vectors cannot be orthogonal in dimensionn.

Exact independent direct-build command, followed by the recorded verbose CMake/CTest checks:

```powershell
C:/MinGW/bin/g++.exe -std=c++11 -Wall -Wextra -Wpedantic -fno-fast-math -ffp-contract=off -msse2 -mfpmath=sse -O0 -g -Iinclude src/cpu.cpp src/spai.cpp src/gmres.cpp tests/test_gmres.cpp -o build/validation_gmres_tests.exe
.\build\validation_gmres_tests.exe
& 'C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe' --build build/ninja --verbose
& 'C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/ctest.exe' --test-dir build/ninja --output-on-failure
.\build\ninja\spai_tests.exe
.\build\ninja\cpu_core_tests.exe
```

Source inspection independently verified the GMRES production implementation's two products, one-pass ascending MGS, Givens/triangular solve, exactzero guards, and recomputed-residual acceptance. Guard flags are present in the emitted Ninja production/test rules. Direct compilation emitted no diagnostics; the environment/reference precision are unchanged from earlier milestones.

SHA-256 at Milestone3 testing (prior SPAI/CSR tests retain their previous hashes):

| Artifact | SHA-256 |
|---|---|
| `src/gmres.cpp` | `78AE262B6E90A6A901686C66D7ECECF6F2FA68DD1E186E759D913CF4F7B7BA5F` |
| `include/bspai/gmres.hpp` | `4ED7598923AC3DDC5A5E15CD5BEA2FD64B31EA6EB029C09A7711891C70919E0D` |
| `tests/test_gmres.cpp` | `137BA13A946E63ED756E3CF4B6BBF3C8B767A51D835595A8C5C627699465FD7A` |
| `CMakeLists.txt` | `448480ADAF088B190D21AA7A5D9489E89CE24351E9B5C819AB69CC7596A3049C` |

## Milestone 4 uniform-double iterative-refinement validation

Scope: fixed-SPAI construction followed by uniform-binary64 GMRES correction/outer IR only; preserve 37 CSR, 190 SPAI, and 280 GMRES checks. This does not reproduce the published mixed-precision or quadruple-residual experiments. No timing, CUDA, BSPAI, or parallel claims are tested.

Pre-execution acceptance policy: original normwise backward error <=10*2^-53 is the unchanged solver stopping rule; componentwise backward error and forward error are diagnostics. Main fixtures use SPAI epsilon0.3/alpha2/beta8, inner tau1e-8, outer cap20. Well-conditioned exact/rational 3x3 references require relative forward error <=1e-12. Independent operational residual components must agree exactly; norms/backward errors agree within16*machine_epsilon relative (exact zero required for zero reference). Wider long-double residual comparisons allow768*machine_epsilon times the data/iterate norm scale. Wider precision has64 mantissa bits here, not quadruple.

Every recorded state is independently checked for original b-A*x, both residual norms, normwise/componentwise backward errors, x0=P*b, zero initial inner correction, unscaled correction RHS, xnext=x+d, attempt/update counts and terminal state. Failures must retain the previous outer iterate and the failed inner result without applying its candidate. Exact/zero-RHS success precedes cap checks; unchanged unconverged iterates stagnate; outer caps0..20 are permitted and21 rejected. Separate coarse-inner tau0.9/cap1 and tau1/stagnation fixtures exercise those statuses without changing the primary tau1e-8. A separate strict tau1e-30 probes full inner nonconvergence. The nearly dependent2x2 delta2^-30 is an explicitly ill-conditioned backward-success/forward-error limitation fixture, not a working-precision forward-accuracy requirement.

Range checks distinguish nonfinite original residual/BE denominators (explicit numerical failure) from optional-reference forward-diagnostic overflow (infinity plus flag; must not change solver success). Optional deliberately wrong reference is used only to establish diagnostic-only behavior, never as a solution oracle.

Initial build investigation: Implementation's first execution of the validation draft stopped at35 checks with `std::bad_alloc`. Validation independently identified its own erroneous iterator range from two separate temporary `product` vectors in an unused local diagnostic. That validation-only range was removed before independent execution. This was a test-harness defect, not evidence of a production solver failure; the failed draft is retained in this audit trail.

### Authoritative Milestone 4 execution result

Independent warning-enabled direct compilation passed **439 IR checks, zero failures**, followed by the CMake/Ninja suite passing. All unchanged regressions passed: **37 CSR,190 SPAI,280 GMRES checks**, zero failures; CTest **4/4 passed**. Execution date2026-10-11; hardware/software/wider-reference precision are unchanged from the environment recorded above. Deterministic synthetic inputs have no random seed, warm-up, repetitions, or performance measurements; CTest wall time is not a benchmark.

| Fixture / diagnostic | Independently observed result |
|---|---|
| Nonsymmetric3x3, exact truth[1,-2,3] | One outer update/one inner attempt; wider normwise BE1.4325458382260084e-17; wider componentwise BE5.551115123125783e-17; relative forward error3.7007434154171885e-17 |
| Same3x3, general binary64 RHS[0.1,-0.7,1.3] | Success and agreement with wider rational inverse within proposed1e-12 forward bound |
| Near-dependent2x2, delta2^-30, RHS[1,0] | Success at initialization; wider original infinity residual2.3667234927415848e-7; normwise BE1.1020917671417224e-16; relative forward error2.3669808668796228e-7 |
| A=diag(1e20,1),P=diag(1e-20,0.5),b=[1,1] | Initial normwise success despite componentwise BE approximately1/3: confirms componentwise error is diagnostic, not a silently added stop |
| One-ULP original-stop bracket, A=[1],b=[1] | P=[1+10*machine_epsilon] succeeds at cap0; its next larger binary64 value fails with iteration_limit |
| Outer range probe A=diag(0.5,1e308),P=diag(1e-308,1e-308),b=[1,0] | Inner converges; outer normwise denominator overflows; numerical_failure after one attempt, zero committed updates, previous fully diagnosed iterate retained |
| Optional finite reference[denorm_min,0], identity system,b=[2,0] | Exact initial success unchanged; optional forward ratio infinity with explicit overflow flag/message |

Exact initialization, zero RHS/zero-denominator convention, independent per-state residual/normwise/componentwise error, cap0, cap1, success-before-cap, stagnation, lower-cap/full-cap inner failures, construction failure, invalid inputs, original-residual overflow, and backward-denominator overflow all passed. The final additional guard test A=I2,P=0.01*I2,b=[1.3e308,1.3e308] has finite residual components but overflowing true2-norm: stable hypot accumulation explicitly returns numerical_failure per the conservative diagnostic contract. No squared-norm artifact or optional-reference exemption is applied to that mandatory diagnostic. Inner failure never applies its candidate. A converged inner attempt followed by failed outer diagnostics is separately recorded with update_applied=false; this does not mean the inner solve failed.

Additional branch investigation is retained: the first outer-range fixture used identityP. Its initial normwise BE was approximately5e-309, so production correctly accepted before attempting a correction. The initial expanded test expected an inner attempt and failed its coverage assertion; an unchecked empty-history access prevented a complete summary. After guarding the test access, the376-check draft reported two failed fixture expectations. No production failure was identified. The separately documented scaled-P fixture above makes initial BE unmet and reaches the intended range failure without changing any stopping policy or main threshold;378 checks passed. Final success-before-cap and componentwise-diagnostic assertions produced438 passing checks. The previously passing321-check core was retained.

Exact final commands:

```powershell
C:/MinGW/bin/g++.exe -std=c++11 -Wall -Wextra -Wpedantic -fno-fast-math -ffp-contract=off -msse2 -mfpmath=sse -O0 -g -Iinclude src/cpu.cpp src/spai.cpp src/gmres.cpp src/ir.cpp tests/test_ir.cpp -o build/validation_ir_tests.exe
.\build\validation_ir_tests.exe
& 'C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe' --build build/ninja --verbose
& 'C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/ctest.exe' --test-dir build/ninja --output-on-failure
.\build\ninja\cpu_core_tests.exe
.\build\ninja\spai_tests.exe
.\build\ninja\gmres_tests.exe
.\build\ninja\ir_tests.exe
```

Direct compilation emitted no warnings. An initial independent verbose CMake rerun reported no work to do; the final439-check rerun visibly rebuilt the IR test with -fno-fast-math,-ffp-contract=off,-msse2,-mfpmath=sse. Validation also inspected emitted `build/ninja/build.ninja` rules to verify these guards on production and test compilations. Source inspection independently verified x0=P*b, original-residual diagnostics, zero initial correction/unscaled RHS, reuse ofP, separate failed-attempt records, acceptance-before-stagnation/cap, fixed10u stopping, and range checks. Validation did not change any production source or manuscript.

SHA-256 at final Milestone4 execution (previous three test hashes independently rechecked unchanged):

| Artifact | SHA-256 |
|---|---|
| `src/ir.cpp` | `823309F4329FC3A394605B8896B046966B27DF804A50D6F3AE616006280F65FF` |
| `include/bspai/ir.hpp` | `F5AC2CAED34B9839DF984DF90F5D0BAC4FB8E69C4AA85BF4DFA45A0EE811EB5B` |
| `tests/test_ir.cpp` | `6FC4A6341F42C7627CEB1F2B1EDD3F73DBA21291902D0EA3553532361A733EC4` |
| `CMakeLists.txt` | `03F46A78A49C9F22F1BEE89F357FA03C0B9D426C1116E7981C25A9EFE317AFE9` |

No unresolved discrepancy against this scoped uniform-double contract was observed. The forward-accuracy/componentwise limitations above are prominent and intentional consequences of normwise stopping, not hidden failures. NonsingularA/P remain mathematical assumptions rather than globally certified properties. No SuiteSparse/published table reproduction, MATLAB-source equivalence, quadruple/mixed precision, BSPAI, CUDA, parallel validation, or performance claim follows from these deterministic small CPU tests. Wider long double plus exact/rational references are bounded independent checks, not a general high-precision validation library.
