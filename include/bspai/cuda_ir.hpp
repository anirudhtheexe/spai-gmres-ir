#ifndef BSPAI_CUDA_IR_HPP
#define BSPAI_CUDA_IR_HPP
#include "bspai/ir.hpp"
#include "bspai/cuda_gmres.hpp"
#include "bspai/cuda_spai.hpp"
namespace bspai {
struct CudaIrResult {
    explicit CudaIrResult(std::size_t n) : solver(n) {
    }
    IrResult solver;
    std::size_t context_constructions = 0, matrix_uploads = 0;
    CudaGmresTransfers construction_transfers, total_transfers;
    std::vector<CudaGmresTransfers> inner_transfers;
    bool gpu_construction_performed = false;
    CudaSpaiTelemetry gpu_spai_telemetry;
};
// Hybrid: unchanged CPU SPAI/outer arithmetic, persistent GPU inner GMRES.
CudaIrResult cuda_refine_with_preconditioner(const CsrMatrix &a, const CsrMatrix &p,
                                             const std::vector<double> &rhs,
                                             const IrOptions &options,
                                             const std::vector<double> &reference = {},
                                             int device = 0);
CudaIrResult cuda_spai_gmres_ir(const CsrMatrix &a, const std::vector<double> &rhs,
                                const IrOptions &options, const std::vector<double> &reference = {},
                                int device = 0);
CudaIrResult cuda_gpu_spai_gmres_ir(const CsrMatrix &a, const std::vector<double> &rhs,
                                    const IrOptions &options,
                                    const std::vector<double> &reference = {}, int device = 0);
}
#endif
