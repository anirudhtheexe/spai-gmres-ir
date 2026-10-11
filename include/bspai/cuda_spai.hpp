#ifndef BSPAI_CUDA_SPAI_HPP
#define BSPAI_CUDA_SPAI_HPP
#include "bspai/spai.hpp"
namespace bspai {
struct CudaSpaiTelemetry {
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
