# Phase 2A CPU milestone 4: uniform-double SPAI--GMRES--IR

Date: 11 October 2026. Completed bounded milestone. Scope is the all-double outer refinement integration only.

## Intended method and validation plan

Use [the binding specification](PHASE2A_ALGORITHM_SPEC.md) and existing validated components without modifying them. Construct P once through SPAI, initialize x0=P*b, evaluate original residual r=b-A*x, and pass that residual directly to existing GMRES with a zero correction initial guess. Apply a successful correction x+=d, then recompute original diagnostics. A failed construction or inner solve is a terminal failure; do not apply its partial correction.

Stop on original-system normwise backward error <=10*2^-53, checked at initialization and after each applied update. Check convergence before stagnation/cap. An exactly unchanged iterate above tolerance reports stagnation. Default outer cap is 20; configured caps 0..20 are permitted, with cap zero evaluating only the initial state. No forward-error stopping, tolerance loosening, residual normalization, alternative solver, automatic preconditioner replacement, or suppressed inner failure is permitted.

Histories retain all committed outer x/residual/backward-error diagnostics and all attempted inner results, their correction RHS and whether an update was applied. Forward error is optional and diagnostic only, measured against a caller-provided binary64 vector; independent higher-precision truth remains the Validation Agent's responsibility. Nonsingularity of A/P remains an assumption, not globally certified.

The residual/correction/update structure, left preconditioner and initial x0=P*b are source-supported by Carson--Khan [P1 Algorithm 3.1](https://arxiv.org/pdf/2202.10204v2) and Khan--Carson [P2 sections 3--4](https://link.springer.com/article/10.1007/s00366-025-02187-z). Entirely double arithmetic, the exact backward-error threshold/caps, range/failure policies, and optional reference API are explicitly specified project choices. Expected numerical effect: successful correction solves reduce the original error, but no guarantee of convergence or published mixed-precision accuracy is asserted.

## API and range policies

`include/bspai/ir.hpp` and `src/ir.cpp` add `spai_gmres_ir` (construction plus refinement), `refine_with_preconditioner` (component interface with supplied P), `evaluate_ir_diagnostics`, and named IR statuses. The supplied-P interface does not use construction options. No existing production component was modified.

`history` contains initialization and each fully computed/committed update; `outer_steps` counts those updates. Parallel `inner_history`, `correction_rhs`, and `update_applied` arrays record every inner attempt, including failures; `inner_attempts` counts attempts. A failed inner result remains unchanged and its candidate is never adopted. A successful inner correction followed by failed outer update/diagnostics is also not committed. Allocation failures remain ordinary exceptions, not numerical statuses.

Every diagnostic uses original A and b, deterministic CSR-order products, binary64 subtraction, and explicit backward-error denominators. A zero denominator gives zero only for a zero numerator, otherwise infinity. Nonfinite original residual, residual norm, matrix norm, or backward-error denominator reports `numerical_failure`; no overflowed denominator can produce false convergence. This is a conservative supported-range guard, not scaled backward-error evaluation. If initialization diagnostics fail, no valid outer state exists and solution/history remain empty.

Research explicitly confirmed that nonfinite mandatory original residual 2-norm also fails, even if individual components or backward-error denominators are finite. Its accumulation uses `hypot`, avoiding naive squaring overflow when the norm is representable. This range policy is a project choice, not a paper convergence rule; optional forward error is the sole overflow exemption.

Optional finite binary64 reference subtraction/ratio overflow records positive infinity plus `forward_diagnostic_overflow` and a message; it does not change convergence. Zero reference truth uses absolute infinity-norm error. No reference vector means forward error remains unavailable (NaN).

Validation owns tests/test_ir.cpp. Planned checks include x0=Pb, correction equations/zero initialization, each applied update, original backward-error acceptance, independent exact/wider final accuracy, zero RHS, construction/inner/cap/stagnation/range failures, and all previous regressions. No SuiteSparse acquisition, published mixed-precision reproduction, BSPAI bucketing, CUDA, or performance measurement is included.

## Execution evidence

The implementation and CMake registration compiled without warnings under the existing GNU MinGW 6.3.0/CMake/Ninja Debug environment. Emitted flags retain `-fno-fast-math -ffp-contract=off -msse2 -mfpmath=sse`; direct C++11 warning-enabled compilation of `src/ir.cpp` also passed. Existing suites passed unchanged: 37 core, 190 SPAI and 280 GMRES checks.

Final independent validation passed **439 IR checks**, together with unchanged **37 core, 190 SPAI and 280 GMRES checks** (946 total, zero failures). CTest passed **4/4**. The independent source audit and artifact hashes are recorded in [the validation record](PHASE2A_VALIDATION_PLAN.md).

The tiny nonsymmetric manufactured solution required one outer correction/inner attempt: wider normwise backward error `1.4325458382260084e-17`, componentwise backward error `5.551115123125783e-17`, and relative infinity-norm forward error `3.7007434154171885e-17`. The noninteger-RHS rational-inverse comparison and every recorded original residual/update also passed. Tests include construction and inner failures without adopting candidates, exact/zero-RHS starts, cap0/cap1, stagnation, one-ULP backward-threshold brackets, original residual/denominator overflow and optional reference overflow.

The nearly dependent 2x2 case with delta=`2^-30` succeeded at initialization under the specified backward criterion. Independent wider diagnostics were residual infinity norm `2.3667234927415848e-7`, normwise backward error `1.1020917671417224e-16`, and relative forward error `2.3669808668796228e-7`. This demonstrates the distinction between backward success and forward accuracy; it is not a working-precision forward-accuracy claim.

Initial IR test execution stopped after 35 checks with `std::bad_alloc`. Inspection found a validation-test iterator range formed from two distinct temporary `product()` vectors. Validation removed that test-only defect. A later range-coverage draft expected an inner correction with `A=diag(.5,1e308), P=I, b=[1,0]`; it failed that expectation because the initial normwise backward error is about `5e-309` and correctly succeeds before any inner solve. The unchecked draft then accessed absent inner history. The final distinct range probe uses nonsingular `P=diag(1e-308,1e-308)`: initialization is above tolerance, the inner correction converges, then original backward-denominator overflow safely fails without committing the update. These draft-test failures are retained in the validation audit; no production algorithm or numerical tolerance changed.

Reproduction commands (from the workspace root; CMake/Ninja paths and initial configure are recorded in prior milestone documents):

```powershell
& 'C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe' -S . -B build/ninja
& 'C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe' --build build/ninja --verbose
& 'C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/ctest.exe' --test-dir build/ninja --output-on-failure
C:/MinGW/bin/g++.exe -std=c++11 -O0 -g -Wall -Wextra -Wpedantic -fno-fast-math -ffp-contract=off -msse2 -mfpmath=sse -Iinclude -c src/ir.cpp -o build/ir_warning_check.o
C:/MinGW/bin/g++.exe -std=c++11 -O0 -g -Wall -Wextra -Wpedantic -fno-fast-math -ffp-contract=off -msse2 -mfpmath=sse -Iinclude src/cpu.cpp src/spai.cpp src/gmres.cpp src/ir.cpp tests/test_ir.cpp -o build/implementation_ir_tests.exe
.\build\ninja\ir_tests.exe
```

All arithmetic remains uniform binary64. Wider validation references use the independently recorded 64-significand-bit `long double`, not arbitrary precision. Published cage5 data/source equivalence, mixed precision, bucketing and GPU behavior remain unverified. No performance measurements or claims were made.

Changed production artifacts: `include/bspai/ir.hpp`, `src/ir.cpp`, `CMakeLists.txt` (library source and test registration), and this report. Independent Validation owns `tests/test_ir.cpp` and its validation record. No manuscript, previous test or previous algorithm code was edited by Implementation. Work stops at this milestone.
