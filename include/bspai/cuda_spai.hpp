#ifndef BSPAI_CUDA_SPAI_HPP
#define BSPAI_CUDA_SPAI_HPP
#include "bspai/spai.hpp"
namespace bspai {
struct CudaSpaiSlotMemory {
    std::size_t max_rows = 0, max_columns = 0, max_candidates = 0;
    std::size_t reduced_capacity_bytes = 0, workspace_capacity_bytes = 0, total_capacity_bytes = 0;
};
struct CudaSpaiTelemetry {
    // Source-issued device allocations only; vendor handle/internal memory excluded.
    std::size_t device_allocation_calls = 0, device_free_calls = 0, device_reallocation_calls = 0;
    std::size_t device_allocated_bytes = 0, device_freed_bytes = 0, current_device_bytes = 0;
    std::size_t peak_device_bytes = 0, max_single_allocation_bytes = 0;
    std::size_t outstanding_device_allocations = 0;
    std::size_t sparse_b_bytes = 0, packed_output_bytes = 0;
    std::size_t max_reduced_rows = 0, max_pattern_columns = 0, max_reduced_elements = 0;
    std::size_t max_qr_workspace_bytes = 0, max_slot_bytes = 0;
    std::size_t nnz_b = 0, nnz_n = 0, nnz_p = 0;
    std::vector<CudaSpaiSlotMemory> slot_memory;
    std::size_t uploads = 0, downloads = 0, upload_bytes = 0, download_bytes = 0;
    std::size_t upload_min_bytes = 0, upload_max_bytes = 0, download_min_bytes = 0,
                download_max_bytes = 0;
    std::size_t upload_le_256_bytes = 0, upload_le_4096_bytes = 0, upload_gt_4096_bytes = 0;
    std::size_t download_le_256_bytes = 0, download_le_4096_bytes = 0, download_gt_4096_bytes = 0;
    std::size_t upload_synchronizations = 0, download_synchronizations = 0,
                explicit_synchronizations = 0;
    double allocation_wall_ms = 0, free_wall_ms = 0;
    std::size_t streams_created = 0, max_fit_batch = 0, parallel_fit_batches = 0;
    std::size_t qr_submissions = 0, max_qr_submissions_before_collection = 0;
    std::size_t gpu_score_evaluations = 0;
    // Diagnostic-only host wall-clock intervals.  These are not CUDA event
    // timings and do not participate in any numerical decision.
    double construction_wall_total_ms = 0;
    double construction_wall_matrix_setup_ms = 0;
    double construction_wall_adaptive_fits_ms = 0;
    double construction_wall_packing_assembly_ms = 0;
    bool gpu_scaling = false, gpu_preconditioner_assembly = false;
};
struct CudaSpaiResult {
    explicit CudaSpaiResult(std::size_t n) : construction(n) {
    }
    SpaiResult construction;
    CudaSpaiTelemetry telemetry;
};
// Native binary64 GPU QR/residual/scoring with host adaptive metadata.
// Canonical square CSR, unchanged SpaiOptions. No CPU numerical fallback.
CudaSpaiResult cuda_construct_spai(const CsrMatrix &a, const SpaiOptions &options, int device = 0);
ScoreResult cuda_evaluate_spai_score(double residual_squared, double dot,
                                     double denominator_squared, int device = 0);
}
#endif
