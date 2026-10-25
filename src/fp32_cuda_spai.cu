// The numerical body below is deliberately the compact GPU-SPAI constructor
// specialization.  Keeping the source algorithm in one audited body prevents
// an FP32-only divergence in sparse storage, stream batching, or P=N^T D.
#include "bspai/fp32_cuda_spai.hpp"
#include "bspai/fp32_cuda_gmres.hpp"
#include <limits>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <climits>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <memory>
#include <numeric>
#include <stdexcept>

namespace bspai {
namespace fp32_cuda_spai_detail {
enum class Status {
    success,
    not_attempted,
    invalid_input,
    rank_deficient,
    numerical_failure,
    empty_candidates,
    zero_denominator,
    invalid_score,
    no_growth,
    iteration_limit
};
inline const char *status_name(Status) {
    return "FP32 CUDA SPAI status";
}
struct ScoreResult {
    float score = 0;
    Status status = Status::invalid_input;
    bool clamped = false;
};
struct Fit {
    std::vector<std::size_t> pattern, shadow, candidates, selected;
    std::vector<float> coefficients, scores;
    float residual = 0, full_residual = 0, mean_score = 0, qr_pivot_ratio = 0;
};
struct Column {
    Status status = Status::invalid_input;
    std::string message;
    std::vector<std::size_t> attempted_pattern, pattern;
    std::vector<float> coefficients;
    std::size_t fit_attempts = 0, growth_attempts = 0, fits = 0, growths = 0, score_clamps = 0;
    float reduced_residual = 0, full_residual = 0, qr_pivot_ratio = 0;
    std::vector<Fit> history;
};
struct Result {
    explicit Result(std::size_t n)
        : scaled_transpose(n, n), inverse(n, n), preconditioner(n, n), columns(n) {
    }
    Status status = Status::invalid_input;
    std::string message;
    std::vector<float> scaling;
    Fp32CsrMatrix scaled_transpose, inverse, preconditioner;
    std::vector<Column> columns;
};
struct CudaResult {
    explicit CudaResult(std::size_t n) : construction(n) {
    }
    Result construction;
    Fp32CudaSpaiTelemetry telemetry;
};
}
using Fp32CudaSpaiStatusCompat = fp32_cuda_spai_detail::Status;
using Fp32CudaSpaiScoreCompat = fp32_cuda_spai_detail::ScoreResult;
using Fp32CudaSpaiFitCompat = fp32_cuda_spai_detail::Fit;
using Fp32CudaSpaiColumnCompat = fp32_cuda_spai_detail::Column;
using Fp32CudaSpaiResultCompatInner = fp32_cuda_spai_detail::Result;
using Fp32CudaSpaiResultCompat = fp32_cuda_spai_detail::CudaResult;
inline const char *fp32_cuda_spai_status_name_compat(Fp32CudaSpaiStatusCompat s) {
    return fp32_cuda_spai_detail::status_name(s);
}
}

// This inclusion is a type-only specialization of the frozen, compact FP64
// implementation.  The mappings below select the S/float CUDA APIs while
// retaining the same QR, score, sparse-storage, host scheduling and assembly
// control flow.  The FP32 score guard is selected explicitly below.
#define BSPAI_CUDA_SPAI_HPP
#define CudaSpaiSlotMemory Fp32CudaSpaiSlotMemory
#define CudaSpaiTelemetry Fp32CudaSpaiTelemetry
#define CudaSpaiResult Fp32CudaSpaiResultCompat
#define SpaiResult Fp32CudaSpaiResultCompatInner
#define SpaiStatus Fp32CudaSpaiStatusCompat
#define ScoreResult Fp32CudaSpaiScoreCompat
#define SpaiFit Fp32CudaSpaiFitCompat
#define SpaiColumn Fp32CudaSpaiColumnCompat
#define SpaiOptions Fp32SpaiOptions
#define CsrMatrix Fp32CsrMatrix
#define Entry Fp32Entry
#define cuda_construct_spai fp32_cuda_construct_spai_compat
#define cuda_evaluate_spai_score fp32_cuda_evaluate_spai_score_compat
#define spai_status_name fp32_cuda_spai_status_name_compat
#define double float
#define BSPAI_SPAI_SCORE_UNIT_ROUNDOFF 5.9604644775390625e-8f
#define BSPAI_SPAI_SCORE_ZERO 0.0f
#define cusolverDnDgeqrf_bufferSize cusolverDnSgeqrf_bufferSize
#define cusolverDnDormqr_bufferSize cusolverDnSormqr_bufferSize
#define cusolverDnDgeqrf cusolverDnSgeqrf
#define cusolverDnDormqr cusolverDnSormqr
#undef cublasDtrsv
#define cublasDtrsv cublasStrsv
#include "cuda_spai.cu"
#undef cublasDtrsv
#undef cusolverDnDormqr
#undef cusolverDnDgeqrf
#undef cusolverDnDormqr_bufferSize
#undef cusolverDnDgeqrf_bufferSize
#undef double
#undef BSPAI_SPAI_SCORE_ZERO
#undef BSPAI_SPAI_SCORE_UNIT_ROUNDOFF
#undef spai_status_name
#undef cuda_evaluate_spai_score
#undef cuda_construct_spai
#undef Entry
#undef CsrMatrix
#undef SpaiOptions
#undef SpaiColumn
#undef SpaiFit
#undef ScoreResult
#undef SpaiStatus
#undef SpaiResult
#undef CudaSpaiResult
#undef CudaSpaiTelemetry
#undef CudaSpaiSlotMemory

namespace bspai {
namespace {
Fp32SpaiStatus fp32_status(fp32_cuda_spai_detail::Status s) {
    using S = fp32_cuda_spai_detail::Status;
    switch (s) {
    case S::success:
        return Fp32SpaiStatus::success;
    case S::invalid_input:
        return Fp32SpaiStatus::invalid_input;
    case S::rank_deficient:
        return Fp32SpaiStatus::rank_deficient;
    case S::empty_candidates:
        return Fp32SpaiStatus::empty_candidates;
    case S::no_growth:
        return Fp32SpaiStatus::no_growth;
    case S::iteration_limit:
        return Fp32SpaiStatus::iteration_limit;
    default:
        return Fp32SpaiStatus::numerical_failure;
    }
}
}
Fp32CudaSpaiResult fp32_cuda_construct_spai(const Fp32CsrMatrix &a, const Fp32SpaiOptions &o,
                                            int device) {
    Fp32CudaSpaiResult out(a.rows());
    const auto native = fp32_cuda_construct_spai_compat(a, o, device);
    out.telemetry = native.telemetry;
    out.construction.status = fp32_status(native.construction.status);
    out.construction.message = native.construction.message;
    out.construction.scaling = native.construction.scaling;
    out.construction.scaled_transpose = native.construction.scaled_transpose;
    out.construction.inverse = native.construction.inverse;
    out.construction.preconditioner = native.construction.preconditioner;
    out.construction.columns.resize(native.construction.columns.size());
    for (std::size_t k = 0; k < native.construction.columns.size(); ++k) {
        const auto &in = native.construction.columns[k];
        auto &dst = out.construction.columns[k];
        dst.status = fp32_status(in.status);
        dst.pattern = in.pattern;
        dst.coefficients = in.coefficients;
        dst.fits = in.fits;
        dst.growths = in.growths;
        dst.reduced_residual = in.reduced_residual;
        dst.full_residual = in.full_residual;
        dst.qr_pivot_ratio = in.qr_pivot_ratio;
        dst.negative_score_clamps = in.score_clamps;
        if (!in.history.empty()) {
            const auto &fit = in.history.back();
            dst.shadow = fit.shadow;
            for (std::size_t j = 0; j < fit.candidates.size() && j < fit.scores.size(); ++j) {
                Fp32SpaiScore score;
                score.candidate = fit.candidates[j];
                score.rho = fit.scores[j];
                score.accepted = std::find(fit.selected.begin(), fit.selected.end(),
                                           score.candidate) != fit.selected.end();
                dst.score_history.push_back(score);
            }
        }
    }
    out.construction.inverse_nnz = out.construction.inverse.values().size();
    out.construction.preconditioner_nnz = out.construction.preconditioner.values().size();
    return out;
}

Fp32IrResult fp32_cuda_spai_gmres_ir(const Fp32CsrMatrix &a, const std::vector<float> &rhs,
                                     const Fp32IrOptions &options, int device) {
    Fp32IrResult result(a.rows());
    Fp32CudaSpaiResult cuda_construction = fp32_cuda_construct_spai(a, options.spai, device);
    if (cuda_construction.construction.status != Fp32SpaiStatus::success) {
        result.construction = std::move(cuda_construction.construction);
        result.status = Fp32IrStatus::construction_failure;
        result.message = result.construction.message;
        return result;
    }
    Fp32IrResult refined = fp32_cuda_refine_with_preconditioner(
        a, cuda_construction.construction.preconditioner, rhs, options, device);
    refined.construction = std::move(cuda_construction.construction);
    return refined;
}
}
