#ifndef BSPAI_CUDA_SPAI_HPP
#define BSPAI_CUDA_SPAI_HPP
#include "bspai/spai.hpp"
namespace bspai {
struct CudaSpaiSlotMemory {
    std::size_t max_rows=0,max_columns=0,max_candidates=0;
    std::size_t reduced_capacity_bytes=0,workspace_capacity_bytes=0,total_capacity_bytes=0;
};
struct CudaSpaiTelemetry {
    // Source-issued device allocations only; vendor handle/internal memory excluded.
    std::size_t device_allocation_calls=0,device_allocated_bytes=0,current_device_bytes=0;
    std::size_t peak_device_bytes=0,max_single_allocation_bytes=0;
    std::size_t outstanding_device_allocations=0;
    std::size_t sparse_b_bytes=0,packed_output_bytes=0;
    std::size_t max_reduced_rows=0,max_pattern_columns=0,max_reduced_elements=0;
    std::size_t max_qr_workspace_bytes=0,max_slot_bytes=0;
    std::size_t nnz_b=0,nnz_n=0,nnz_p=0;
    std::vector<CudaSpaiSlotMemory> slot_memory;
    std::size_t uploads=0,downloads=0,upload_bytes=0,download_bytes=0;
    std::size_t streams_created=0,max_fit_batch=0,parallel_fit_batches=0;
    std::size_t qr_submissions=0,max_qr_submissions_before_collection=0;
    std::size_t gpu_score_evaluations=0;
    bool gpu_scaling=false,gpu_preconditioner_assembly=false;
};
struct CudaSpaiResult {
    explicit CudaSpaiResult(std::size_t n):construction(n){}
    SpaiResult construction;
    CudaSpaiTelemetry telemetry;
};
// Native binary64 GPU QR/residual/scoring with host adaptive metadata.
// Canonical square CSR, unchanged SpaiOptions. No CPU numerical fallback.
CudaSpaiResult cuda_construct_spai(const CsrMatrix& a,const SpaiOptions& options,int device=0);
ScoreResult cuda_evaluate_spai_score(double residual_squared,double dot,
    double denominator_squared,int device=0);
}
#endif
