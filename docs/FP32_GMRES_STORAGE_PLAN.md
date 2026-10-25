# FP32 CUDA GMRES demand-grown storage

Date: 25 October 2026. This is a storage optimization of the existing
uniform-FP32, unrestarted left-preconditioned GMRES. It does not change the
maximum iteration count, arithmetic precision, MGS/Givens order, convergence
test, or failure statuses. The project contracts are
`PHASE2A_ALGORITHM_SPEC.md` and `EXPERIMENT2A_FP32_SPEC.md`; the former records
the unrestarted algorithm and project-specific cap/status rules. Carson and
Khan's full-text SPAI-GMRES-IR and Khan and Carson's BSPAI-GMRES-IR were
consulted for the method context, but chunked storage is a project proposal,
not a result reported by those papers.

## Baseline and proposed change

`src/fp32_cuda_gmres.cu` currently allocates `n*(cap+1)` FP32 device elements
for the Krylov basis before the zero-iteration residual checks, then creates
`(cap+1)*cap` zeroed FP32 host Hessenberg elements before Arnoldi. At
`n=cap=12000`, each array is 576,048,000 bytes (549.36 MiB), excluding
allocator overhead. Archived FP32 examples generally terminate in far fewer
iterations, but that observation is workload-specific and is not used as a
new iteration limit.

Start with no basis or Hessenberg allocation. After the initial residual
checks, allocate a bounded first slab of basis vectors and projected-system
rows/columns. Add fixed 16-vector device slabs on demand; each vector keeps
its original device address, so growth does not require a device copy or a
simultaneous old/new basis allocation. Grow `h`, `cs`, `sn`, and `g` in 16-column
host chunks, zero-initializing only newly exposed entries. Keep all numerical
operations and index order unchanged. The chunk size is an implementation
resource policy, not a stopping rule. A preliminary contiguous-growth design
was discarded before testing because its transient peak could nearly double
the full-cap device-basis allocation.

## Validation plan

Compare deterministic FP32 baseline/candidate outputs and independent
original/preconditioned residuals for zero-iteration success, a one-step
case, a case crossing at least one chunk boundary, configured cap exhaustion,
breakdown, and normal IR integration. Include `n=12000` with early convergence
and record actual source-issued CUDA allocation size/peak plus host projected
capacity. Run all CUDA suites and applicable independent Decimal80/CLI checks.
Record build flags, hardware, input provenance, numerical differences, and any
allocation or runtime regressions before claiming improvement. No performance
claim follows from nominal byte counts alone.

## Outcome

The slab implementation is in `src/fp32_cuda_gmres.cu`. The validation-only
record `FP32_GMRES_STORAGE_VALIDATION.md` documents the chunk-boundary and
`n=12000` numerical checks, source-owned memory arithmetic, environment, and
full CUDA/CLI regression results. This is an observed ordinary-case resource
reduction in source-requested storage. Near the full cap, basis and
Hessenberg capacity still grows to the original `O(n*cap)` size by design;
the method remains unrestarted and retains the original maximum iteration
count.

A subsequent paired same-build benchmark answered the runtime question on
three deterministic cases; see the timing section of
`FP32_GMRES_STORAGE_VALIDATION.md` and retained raw data in
`results/fp32_gmres_storage_20261025/`. At `n=12000`, median public GMRES call
time fell from 220.103 ms to 1.331 ms for a one-iteration solve and from
16.634 ms to 0.907 ms for an exact-initial return. A cap-18 solve crossing a
slab boundary measured 11.564 ms versus 11.598 ms, within the observed
spread. These are input-specific call timings, not whole-solver or general
speedup claims. No whole-device peak was measured.

The next complete-pipeline follow-up on archived `stencil_8192` held the final
GPU SPAI/SpMV code fixed and varied only eager versus slab GMRES storage.
Six-run pooled medians were 2639.473 ms and 2290.805 ms, respectively,
13.21% less complete-pipeline call time for the slab implementation on that
matrix. See the validation record and raw files for the paired-order spread
and exact matching outputs. This single result does not establish an expected
percentage across other matrices.
