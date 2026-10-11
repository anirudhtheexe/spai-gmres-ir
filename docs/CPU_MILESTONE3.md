# Phase 2A CPU milestone 3: unrestarted GMRES

Date: 11 October 2026. Implementation and independent validation complete. Scope is binary64 GMRES only; no outer iterative refinement, BSPAI, CUDA or performance work.

## Intended method and validation plan

Follow [the binding specification](PHASE2A_ALGORITHM_SPEC.md): solve `P*A*x=P*rhs`, apply `P*(A*v)` as two products, use one ascending-order modified Gram--Schmidt pass and Givens rotations, and never restart or reorthogonalize. Default initial guess is zero. The cap defaults to n; explicit caps 1..n are allowed and larger caps are rejected. Requested tolerances must be positive and finite.

Only exactly computed zero Arnoldi subdiagonal is breakdown; small finite values are retained. Zero Givens/backsolve pivots fail unless an already well-defined candidate passes verification. No invented coefficients, alternate least-squares solver, or tolerance changes are allowed. These are project exceptional-case choices confirmed by Research, consistent with the paper-supported method; they are not a claim of MATLAB/library equivalence.

Compute original `rhs-A*x0` first. An exactly zero residual returns with zero iterations. Otherwise `||P*rhs||2=0` is an explicit normalization failure, including zero RHS with a non-solution initial guess. There is no substituted denominator. General initial guesses start from `P*rhs-P*(A*x0)` and normalization remains `||P*rhs||2`.

Every complete Arnoldi iteration forms a candidate and recomputes both original and preconditioned residuals for diagnostics. Only an estimated-convergence, exact-breakdown, or iteration-cap trigger can accept that candidate, and only after the recomputed relative preconditioned residual meets tolerance. Extra diagnostic products do not change the basis or least-squares process. Lower-cap failure is `iteration_limit`; n-cap failure is `nonconvergence`; a failed breakdown is separate. Returned raw Hessenberg entries and basis vectors support independent checks.

Validation owns `tests/test_gmres.cpp`. Planned checks include exact identity/diagonal/tiny nonsymmetric solutions, zero RHS and exact initial guesses, nonzero general initial guesses, explicit breakdown and cap failures, recursive-estimate rejection, independent wider residuals, orthogonality/Hessenberg evidence, and a cancellation fixture distinguishing two-product application from a materialized rounded P*A. Existing 37 CSR and 190 SPAI checks must continue to pass.

## Execution evidence

Added `include/bspai/gmres.hpp`, `src/gmres.cpp`, and this record. Updated CMake to build the GMRES source and independently owned `tests/test_gmres.cpp` target. CSR/SPAI source and tests remain unchanged. No public data was downloaded.

The API accepts original A, P, RHS, options, and an optional initial guess. It exposes solution/status, initial/final original 2-norm residuals and relative preconditioned residuals, iteration history, basis, and raw unrotated Hessenberg matrix. Iteration history includes candidate solution, estimate, verified residuals, exact subdiagonal, and verification/breakdown/acceptance flags. Allocation failures propagate standard exceptions; finite/range arithmetic failures return `numerical_failure`. Iterations count attempted Arnoldi steps; incomplete numerical-error steps may have no completed history entry. Exact original initial residuals report zero relative residual by an explicit exact-residual convention, including a zero preconditioned RHS, without evaluating 0/0.

An initial source build emitted no warnings and passed **189 GMRES checks**, with previous **190 SPAI and 37 CSR checks** retained; CTest **3/3** passed. The first expanded direct-compiler coverage probe had **275 checks, one failed assertion**: the proposed separate gap-test tolerance 1e-16 was below both the estimate (1.2831951913535754e-16) and verified residual (5.1192497462734993e-16), so it did not exercise the asserted estimate-below/true-above branch. The solver correctly reported nonconvergence. The test fixture alone was corrected to a separate 2e-16 bracket; the principal tolerance 1e-8, strict 1e-30 test, algorithm, and acceptance rule are unchanged.

Final independent validation passed **280 GMRES checks with zero failures**. The retained suites passed **190 SPAI checks** and **37 CSR checks**, also with zero failures; CTest passed **3/3**. A clean no-work rebuild and all three saved test executables were rerun from this workspace after documentation reconciliation.

Observed initial numerical evidence:

| Project fixture | Result |
|---|---|
| Tiny nonsymmetric A, identity P, noninteger RHS | 3 iterations; wider original residual 9.0748423910279679e-16; wider preconditioned relative residual 6.1322102067017616e-16 |
| Tiny A, epsilon=0.3 SPAI P, same RHS | 3 iterations; wider original residual 6.6901877333569169e-16; operational preconditioned relative residual 4.0270357740038596e-16 |
| Requested tolerance 1e-30, n=3 cap | `nonconvergence`; final operational relative residual 5.1192497462734993e-16 |
| Cancellation operator fixture | `P*(A*[1,1])=[0,2]`, versus separately rounded materialized `(P*A)*[1,1]=[-2,2]` |

No thresholds were relaxed. Exact rational solutions and independently computed wider residuals are the tiny-case numerical references. Long double has a 64-bit significand here and is not IEEE quadruple. Successful left-preconditioned residual convergence is not a global certificate for original-system accuracy with arbitrary singular/poorly conditioned P; original residuals remain separate diagnostics.

Build uses the Milestone 2 numerical guards: binary64 double, no fast math/contraction, and SSE2 intermediates for the 32-bit GNU MinGW target. Compiler GCC 6.3.0, bundled CMake 3.31/Ninja, Debug build on the same Windows host. Standard round-to-nearest environment is assumed, not changed. Commands use the paths/variables recorded in [Milestone 1](CPU_MILESTONE1.md):

```powershell
& $cpuCmake -S . -B build/ninja
& $cpuCmake --build build/ninja --verbose
& $cpuCtest --test-dir build/ninja --output-on-failure
& .\build\ninja\gmres_tests.exe
```

Independent direct-compiler command:

```powershell
& 'C:\MinGW\bin\g++.exe' -std=c++11 -O0 -g -Wall -Wextra -Wpedantic -fno-fast-math -ffp-contract=off -msse2 -mfpmath=sse -Iinclude src/cpu.cpp src/spai.cpp src/gmres.cpp tests/test_gmres.cpp -o build/implementation_gmres_tests.exe
& .\build\implementation_gmres_tests.exe
```

The direct command was executed on the expanded coverage probe described above. The separate `2e-16` coverage fixture exercised the required rejection path: estimate `1.2831951913535754e-16 <= tau`, recomputed residual `5.1192497462734993e-16 > tau`, and status `nonconvergence`. The near-dependent stress case returned `breakdown`, with wider original residual `2.935454526555012e-7` and relative forward error `6.7179421740666064e-8`; it is a documented failure to meet the requested tolerance, not success.

With singular `P=diag(1,0)`, `A=I`, `b=[1,1]`, and `x0=[1,0]`, the current left-preconditioned criterion accepts at zero iterations because its preconditioned relative residual is zero although the original residual is one. This is an unsupported-input limitation, not a solver defect: the method requires a nonsingular preconditioner and does not certify that property. The original paper's analysis uses `P^{-1}` and conditioning involving `P`, so no silent original-residual stopping condition was added. Original residuals remain mandatory diagnostics.

Other compilers/platforms, SuiteSparse inputs, published iteration-count reproduction, MATLAB stopping equivalence, reorthogonalized/restarted methods, and mixed-precision convergence guarantees remain unverified. The source and report stop at inner GMRES; outer refinement is not implemented.
