# FP32 GMRES persistent CUDA context plan (2026-10-25)

The uniform FP32 inner GMRES is currently called once per refinement correction. Each call creates a stream, cuBLAS and cuSPARSE handles, dense and sparse descriptors, eight device work vectors, and CSR copies of unchanged A and P. The unrestarted GMRES arithmetic, stopping test, and FP32 precision policy are specified in `EXPERIMENT2A_FP32_SPEC.md`; this change affects resource lifetime only.

Introduce a device-bound context that owns those resources and one uploaded copy of each matrix. Construct it lazily at the first correction and reuse it for later corrections. Retain the one-shot entry point for existing callers. Keep the growing Krylov basis and projected host data per solve, and preserve their existing order of operations. Count the two matrix uploads and their CSR payload bytes for validation.

Validation: independently compare every exposed solver result field with the existing one-shot path across distinct and repeated right-hand sides, exact initial solutions, iteration limits, and invalid inputs. Run the FP32 CUDA unit tests and CLI checks. Measure repeated solves and an archived multi-correction FP32 pipeline with identical source, flags, inputs and hardware; report setup savings separately from complete pipeline time. Save raw results and environment details. Treat any numerical discrepancy as a failure, and do not infer a general speedup from one workload.

Baseline source snapshot: `results/fp32_gmres_context_20261025/fp32_cuda_gmres_before_context.cu`, SHA-256 `19020F72E354237B201382E20C12D366B57ACC23D3EE79069FE1207297BDB2D9`.
