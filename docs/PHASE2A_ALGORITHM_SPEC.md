# Phase 2A: uniform-double CPU reference specification

Date: 11 October 2026. Research specification only; no BSPAI, mixed-precision, parallel, or CUDA implementation is specified here. Phase 1 remains unchanged.

## Authority and evidence

- **P1:** Carson–Khan, [SPAI–GMRES–IR, arXiv v2](https://arxiv.org/pdf/2202.10204v2): Algorithm 2.1, printed p. 6; Algorithm 3.1, p. 13; experiments, pp. 16–17. PDF page numbers are one greater than the zero-based browser page indices. The [HTML full text](https://arxiv.org/html/2202.10204v2) renumbers algorithms; use the PDF numbering.
- **P2:** Khan–Carson, [published BSPAI article](https://link.springer.com/article/10.1007/s00366-025-02187-z), §§2.2–3 and §4. Use the published article rather than its 2023 preprint.
- **M:** [fixed MATLAB repository](https://github.com/Noaman67khan/SPAI-GMRES-IR/tree/b25e6f7c1cf1c3035686e9adb5ef9f35e2256ae1), pinned during Phase 1. The [rendered `spai_dd.m`](https://github.com/Noaman67khan/SPAI-GMRES-IR/blob/main/spai_dd.m) was additionally read here. Its moving-branch rendering does not independently establish equality with the pinned commit; retained pinned-source inspection must supply that check.

**Paper-verified:** independent QR column fits, identity initial pattern, adaptive candidate growth, left SPAI preconditioning, `x0=P*b`, residual/correction/update refinement, natural ordering, scaling below, `beta=8`, `alpha=ceil(n/beta)`, and double-working GMRES tolerance `1e-8`. P1 distinguishes five precision roles; P2 uses unrestarted GMRES to match its analysis. P1's double-working experiments use a quadruple residual, so an entirely double solver is a **project reference configuration**, not a reproduction of those tables. No inherited mixed-precision convergence guarantee is asserted.

## Executable contract and explicitly chosen policies

The following is the project's disambiguated contract. Where a finite-precision or exceptional-case rule is not fixed by the sources, it is labelled **choice**.

### Input, scaling, and orientation

Use a real, square, nonsingular matrix `A`, natural ordering, and a nonzero diagonal for the identity-pattern baseline. Reject nonfinite entries and invalid dimensions. A zero diagonal is outside this baseline's verified problem selection; report it rather than changing ordering or the initial pattern automatically.

Define `d[k]=1/max_j(abs(A[k,j]))`, `D=diag(d)`, and `B=transpose(A)*D`. Fit a right approximate inverse `N` of `B`. Then **`P=transpose(N)*D`**, with `D` on the right. Keep original `A` and `b` for all solves and diagnostics. Indeed, `(B*N)^T=N^T*D*A=P*A`; moving `D` to the left changes the method. Column scaling of `A^T` is row-max scaling of the original `A`. Cite P1 §4 and P2 §4; do not depend on the missing MATLAB `findscaling` helper.

**Milestone 2 range guard choice:** reject a zero/nonfinite row maximum or reciprocal, nonfinite scaled value, and any original nonzero that becomes zero during scaling. The last check prevents silent changes to the shadow/candidate pattern. These guards delimit the supported binary64 input range; they are not additional rules claimed from the papers. Also reject nonfinite output when forming `P`.

### SPAI construction

For column `k`, start with sorted `J={k}`. Its shadow is `I={i : some B[i,j]!=0 for j in J}`. With the allowed nonzero diagonal, `k` belongs to `I`. Solve the reduced least-squares problem

`min_m ||B(I,J)*m-e_k(I)||_2`

using an economy QR and a triangular solve; never form normal equations or an explicit inverse. Recompute the QR after each growth (**choice**); QR updates are an established optional implementation, not required for this reference. A double Householder QR is the selected CPU realization (**choice**, consistent with the paper's QR requirement); rank-deficient/failed fits return explicit failure rather than a fallback to normal equations, SVD, or a changed pattern.

**Milestone 2 QR guard choice:** exactly zero `R[j,j]` gives `rank_deficient`; nonfinite pivots/intermediates give a separate nonfinite failure. No relative rank threshold, pivoted QR, or regularization is introduced. Record `min_j(abs(R[j,j]))/||B(I,J)||_infinity` as a conditioning indicator; a small finite ratio is not itself a rank decision. For nonsingular `B`, every subset of its columns is exactly independent, and deleting rows outside the shadow deletes only all-zero rows of that subset. Thus each reduced matrix is full column rank in exact arithmetic. This reasoning does not guarantee stable numerical fits for ill-conditioned subsets. Nor is checking reduced QR pivots a global nonsingularity test for arbitrary input `A`.

Compute `s=B(I,J)*m-e_k(I)` and publish coefficients only for the pattern just solved. Construction success means `||s||_2 <= epsilon`. Also record a separately recomputed full residual `||B*N(:,k)-e_k||_2`.

For an unconverged column, form `L=I union {k}` and

`C=(union over l in L of {j:B[l,j]!=0}) minus J`.

For each candidate use the **restricted-column** score in P1 Algorithm 2.1 and M:

`delta_j=||s||_2^2-(s^T*B(I,j))^2/||B(I,j)||_2^2`, `rho_j=sqrt(delta_j)`.

Do not substitute the full-column norm. The score is the published heuristic; it need not equal a new full least-squares residual. Compute the mean over all candidate scores once; select at most `beta` candidates with `rho_j <= mean`, ordered by `(rho_j,j)` (**choice:** smallest index breaks ties), union them into `J`, and refit. An empty set, zero denominator, nonfinite score, or no growth before convergence returns a named construction failure with column/residual details. Do not divide by an empty candidate count or loop beyond available candidates.

**Loop resolution:** define `alpha` as the maximum number of completed growth-and-refit events. Perform an initial fit, then at most `alpha` such events: at most `alpha+1` fits. On the last fit, test the residual and return either success or iteration-limit failure; never enlarge an unsolved pattern. This **choice** interprets the paper's `0:alpha` fit loop coherently and differs from MATLAB's `1:alpha` fit limit. Record fit and growth counts separately. A capped result is not construction success.

**Score-rounding choice:** permit a negative `delta_j` only within `32*u*(||s||_2^2+abs((s^T*B(I,j))^2/||B(I,j)||_2^2))`; clamp that value to zero and count/report the event. A larger negative value fails. Here `u=2^-53`. This guard is explicit and differs from MATLAB's unconditional `sqrt(abs(delta_j))`; it must be covered by validation.

### GMRES and iterative refinement

Set `x0=P*b`. At outer step `i`, compute `r=b-A*x_i` in double, solve `P*A*d=P*r` by unrestarted modified Gram–Schmidt GMRES, and update `x_{i+1}=x_i+d` in double. Apply the operator as two products `P*(A*v)`; do not materialize a rounded `P*A` operator. Reuse `P` across outer steps. The correction's initial guess is zero (**choice**).

**Preconditioner assumption and certification limit:** equivalence of the left-preconditioned and original equations requires nonsingular `P`. P1 §3.1 uses inverses/condition numbers of the preconditioner and preconditioned system, and P2 §3 bounds errors involving `kappa_infinity(M)`; finite bounds presume invertibility. Nonsingular `P` is therefore a mathematical input assumption for this solver, not something the SPAI constructor or GMRES API globally certifies. A positive construction status alone is not that certificate. Retain the prescribed preconditioned stopping criterion rather than silently adding an original-residual stopping condition.

For example, `A=I_2`, `P=diag(1,0)`, `b=[1,1]`, `x0=[1,0]` has zero preconditioned residual but original residual `[0,1]`. The GMRES API can accept its preconditioned criterion at zero iterations; this unsupported singular-preconditioner input does not certify a solution of `A*x=b`. Record this as a diagnostic limitation test. Even an ill-conditioned nonsingular `P` can amplify differences between original and preconditioned residuals, so independent original-system diagnostics remain required. The outer IR stopping condition below uses original-system backward error.

Use one MGS pass in deterministic ascending basis order and Givens rotations for the Hessenberg least-squares update (**choices**). Do not silently introduce reorthogonalization, restarting, or flexible GMRES. Limit the Arnoldi process to `n` iterations (**choice**); failure to satisfy the criterion reports inner nonconvergence. On breakdown, compute the available correction and verify its residual; accept only if the stopping criterion holds. A zero preconditioned RHS with a nonzero original residual is a failure, not convergence.

**Milestone 3 guard choices:** Arnoldi breakdown means the computed `h_next` is exactly zero; retain small finite values rather than applying an unspecified near-zero threshold. Nonfinite arithmetic fails. If both new Givens inputs are zero, or a required triangular pivot is zero, return an explicit breakdown/rank failure unless an already well-defined candidate satisfies the recomputed criterion; no division by zero or alternative least-squares fallback is permitted. A configured inner cap may lie in `1..n`, defaulting to `n`: reject a cap above `n`; an exhausted smaller cap is `iteration_limit`, while an exhausted `n`-iteration cap is `nonconvergence`. These status distinctions are project choices, not statements from the papers.

For a general GMRES initial guess, compute `r_original=b-A*x0` first. An exactly zero original residual succeeds in zero iterations. Otherwise, if `||P*b||_2=0`, return a named normalization/preconditioner-RHS failure, including the case `b=0` with a non-solution initial guess. Do not substitute a denominator of one or change the prescribed normalization. Thus an exact initial solution is accepted even if `P*b=0`; a non-solution is not. The refinement correction calls still use zero initial guesses.

**Inner stopping choice:** define `tau` as the relative 2-norm of the left-preconditioned correction residual,

`||P*r-P*(A*d)||_2 / ||P*r||_2 <= tau`.

Use the Hessenberg residual estimate to trigger a check, then recompute the displayed numerator before accepting. Record estimate and recomputed residual. This pins what `tau=1e-8` means; P1 §3.1 discusses backward-error-based stopping as well, so that discussion alone does not establish equivalence to this relative-residual rule. Exact MATLAB normalization remains a source-reconciliation item below.

**Outer stopping choice:** accept when double-computed normwise backward error `eta_n=||b-A*x||_infinity/(||A||_infinity*||x||_infinity+||b||_infinity)` is at most `10*u`. Check at initialization and after each update. A cap of 20 outer correction steps is a failure limit, not convergence. Report unchanged iterates/stagnation without loosening the tolerance. Do not stop on a known reference solution's forward error: it is a diagnostic only. Uniform-double residuals do not establish quadruple-residual attainable-accuracy guarantees.

**Milestone 4 cap/failure choices:** permit configured outer caps in `0..20`, default `20`; reject larger caps. Cap zero performs initialization and its diagnostics only: success if the initial criterion holds, otherwise `iteration_limit`. Any failed inner solve propagates without adopting its correction or updating the current outer iterate. After a valid update, check original-system backward-error success before checking exact unchanged-iterate stagnation or the cap. An unchanged iterate with backward error above the threshold is `stagnation`, not success. Retain the last accepted outer iterate and explicit failure reason/history.

### Diagnostics and independent acceptance

Record initial/final and per-step original residual, `eta_n`, componentwise backward error `max_i |r_i|/(|A|*|x|+|b|)_i`, and relative infinity-norm forward error when independently computed reference truth is available. A zero denominator contributes zero only when its numerator is zero; otherwise it contributes infinity. Forward error for zero reference truth is reported as absolute error.

**Milestone 4 diagnostic range choices:** a nonfinite original residual or normwise/componentwise backward-error denominator is `numerical_failure`; never interpret division by an overflowed infinity as zero error. This deliberately bounds supported input range; stable scaled-ratio evaluation would need a separately documented change. An optional finite binary64 reference is diagnostic only and does not constitute an independent high-precision truth certificate. Overflow in its forward-error subtraction or ratio is recorded as positive infinity with an explicit diagnostic-overflow flag/message; that optional-diagnostic overflow alone must not fail the solver or change its stopping decision. A component API taking an externally supplied `P` is permissible for validation, with the same unverified nonsingular-preconditioner assumption.

The mandatory original-residual 2-norm also must be finite: its final overflow is `numerical_failure`, even when the residual components and normwise backward error remain finite. Compute this norm with `hypot`/scaled accumulation, so avoidable intermediate squaring overflow is not mistaken for an unrepresentable norm. This is an explicitly conservative project range guard, not a paper-prescribed condition or an additional accuracy criterion. The optional-forward-diagnostic exemption above does not extend to mandatory residual norms.

Construction records include per-column fits/growths, residuals, failure statuses, and `nnz(N),nnz(P)`. Record inner iterations per outer step and both original and preconditioned residuals; one cannot stand in for the other. Recompute validation diagnostics separately using a documented higher-precision or otherwise trusted oracle. Binary64 diagnostics alone cannot certify binary64 accuracy independently. Hash matrix files and record compiler/library/version, flags, QR realization, parameters, ordering, and stopping policies.

## Initial matrix and exact configuration

Use [SuiteSparse `vanHeukelum/cage5`](https://sparse.tamu.edu/vanHeukelum/cage5), ID **906**, real nonsymmetric **37 x 37**, **233 nonzeros**, full numerical rank according to the collection. This agrees with P1 Table 4.1. Its original Table 4.4 includes construction `epsilon=0.3` and reports 213 preconditioner nonzeros for the single-construction/single-working/double-residual experiment; **213 and its iteration counts are contextual published results, not acceptance targets for all-double arithmetic**.

| Setting | Phase 2A contract | Evidence/status |
|---|---|---|
| RHS | `b_i=1/sqrt(37)` | Published equal-component unit-2-norm convention |
| Pattern/order | identity; natural | Published |
| Scaling/orientation | `B=A^T D`, `P=N^T D` | Published |
| Construction | `epsilon=0.3`, `beta=8`, `alpha=ceil(37/8)=5` | Published parameters; loop meaning disambiguated above |
| Arithmetic | IEEE binary64 throughout; `u=2^-53` | Project choice |
| GMRES | no restart; one-pass MGS; `tau=1e-8`; cap 37 | No restart/tolerance paper-supported; exact implementation/stopping policies above are choices |
| IR | `eta_n <= 10*2^-53`; cap 20 | Project choice |

Data bytes and SHA-256 remain to be verified after lawful acquisition; metadata verification is not data verification. Until then, use the Validation Agent's separately labelled 3 x 3 hand-check case for component tests, not as a replacement for the named published problem.

## Conflict ledger and unresolved limits

**M source behavior verified from the rendered constructor:** `spai_dd.m` uses economy `qr`, stores a fitted column before growth, `1:alpha`, strict `<epsilon`, sorted union/set difference, restricted scoring, and unconditional absolute value around the score radicand. The CPU contract instead uses `<=`, an explicit fit/growth budget, bounded negative-score handling, and exhaustion checks. These are recorded reconciliation choices; they preclude claiming a bitwise MATLAB port.

**Resolved:** missing candidate exclusion in printed pseudocode is supplied by its prose and M; `r`/`s` naming inconsistencies do not define two residuals; final enlarged-but-unsolved pattern is never stored; scaling uses the explicit paper formula rather than an unavailable helper; IEEE unit roundoff is half machine epsilon (`2^-53`, not `2^-52`).

**Unresolved / limits:** exact pinned GMRES/IR source stopping and normalization still require retained source inspection; rendered-main equality to the pinned constructor needs checking. No claim of matching MATLAB histories or reproducing published tables is allowed until these are checked. These limits do not prevent the explicitly specified all-double reference, but they block a source-equivalence claim. No hardware low-precision, BSPAI, CUDA, or theoretical convergence claim is made by this document.
