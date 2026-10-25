# Experiment 2A: uniform-binary32 extension contract

Date: 11 October 2026. This is an experiment-specific extension authorized by
the Experiment 2A request. It does not amend the binary64 Phase 2A reference
contract or claim a reproduction of the published mixed-precision methods.

## Frozen method

The mathematical workflow remains the Phase 2A workflow: form `B=A^T D` with
`d[i]=1/max_j |A[i,j]|`, construct `N` by independent adaptive QR least-squares
fits, assemble `P=N^T D`, apply left-preconditioned unrestarted GMRES as
`P*(A*v)`, and perform iterative refinement from `x0=P*b`.

The production FP32 path uses IEEE binary32 for all solver matrix entries,
vectors, QR work, candidate scores, Hessenberg/Givens work, residuals,
backward errors, and updates. Host timers, JSON serialization, hashes, and an
external Decimal80 oracle are outside the solver and must be labelled as such.
No binary64 residual, accumulation, fallback solve, restart, regularization,
or alternate preconditioner is permitted inside this path.

## Parameters and exceptional cases

- `epsilon=0.3`, `beta=8`, `alpha=ceil(n/8)`;
- GMRES relative left-preconditioned residual tolerance `1e-4`, iteration cap
  `n`, one ascending-order MGS pass and Givens rotations;
- outer correction cap `20`; normwise backward-error target `10*2^-24`;
- Householder QR is required for each reduced SPAI fit; normal equations,
  pivoting, SVD and rank thresholds are prohibited;
- an exactly-zero computed QR or Arnoldi pivot retains the Phase 2A named
  failure/breakdown semantics. A breakdown candidate may succeed only after a
  recomputed residual satisfies the frozen FP32 tolerance;
- score rounding uses `u=2^-24` and permits a negative score radicand only
  through `32*u*(||s||_2^2+abs(q))`, where `q=(s^T B(I,j))^2/||B(I,j)||_2^2`.
  It is then clamped to zero and recorded.

The candidate denominator is restricted to the current shadow rows `I`.
Candidate ordering is `(rho,index)` and only the at-most-`beta` scores no
larger than the one-shot candidate mean are added. All input, numerical,
construction, cap, nonconvergence and stagnation failures remain explicit;
no failed correction is applied.

## Evidence and claims

FP32 input provenance records both original stored binary64 inputs and their
binary32-rounded counterparts. Independent Decimal80 checks must verify
binary32 bit round trips, scaling, reconstructed fits/`P`, GMRES residual
acceptance, and every accepted iterative-refinement update. CPU/GPU native
constructions may have different valid patterns; fixed-P comparisons require
the identical serialized FP32 `P` bit hash.

This contract is an implementation/measurement extension. The papers support
the requested single-working GMRES tolerance but do not establish a guarantee
for fully uniform-FP32 residual computation. Any resulting timings are
exploratory until the documented admission and validation work completes.
