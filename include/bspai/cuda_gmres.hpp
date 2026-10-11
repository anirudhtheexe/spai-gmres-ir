#ifndef BSPAI_CUDA_GMRES_HPP
#define BSPAI_CUDA_GMRES_HPP
#include "bspai/gmres.hpp"
#include <memory>
namespace bspai {
struct CudaGmresTransfers {
    std::size_t vector_uploads=0,vector_downloads=0;
    std::size_t arnoldi_vector_uploads=0,arnoldi_vector_downloads=0;
    std::size_t scalar_observations=0;
    std::size_t dot_observations=0,norm_observations=0,finite_flag_observations=0;
    std::size_t host_scalar_inputs=0,axpy_scalar_inputs=0,spmv_scalar_inputs=0,division_scalar_inputs=0;
};
struct CudaGmresResult {
    GmresResult solver;
    CudaGmresTransfers transfers;
};
// Persistent immutable A/P and Krylov workspace; solve calls are synchronous.
// Not thread-safe. Construction/runtime failures throw; numerical failures
// use the same GmresResult statuses as the one-shot API.
class CudaGmresContext {
public:
    CudaGmresContext(const CsrMatrix& a,const CsrMatrix& p,
        const GmresOptions& options,int device=0);
    ~CudaGmresContext();
    CudaGmresContext(const CudaGmresContext&)=delete;
    CudaGmresContext& operator=(const CudaGmresContext&)=delete;
    CudaGmresResult solve(const std::vector<double>& rhs,
        const std::vector<double>& initial_guess={});
    CudaGmresTransfers construction_transfers() const;
    CudaGmresTransfers total_transfers() const;
    std::size_t matrix_uploads() const; // Two immutable matrix initializations.
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
// Standalone safe-scaled device norm primitive for independent validation.
double cuda_safe_norm2(const std::vector<double>& x,int device=0);
// All vector histories are downloaded only after termination. No CPU fallback.
CudaGmresResult cuda_left_preconditioned_gmres(const CsrMatrix& a,const CsrMatrix& p,
    const std::vector<double>& rhs,const GmresOptions& options,
    const std::vector<double>& initial_guess={},int device=0);
}
#endif
