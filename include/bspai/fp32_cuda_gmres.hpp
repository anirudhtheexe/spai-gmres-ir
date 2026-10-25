#ifndef BSPAI_FP32_CUDA_GMRES_HPP
#define BSPAI_FP32_CUDA_GMRES_HPP
#include "bspai/fp32.hpp"
#include <cstddef>
#include <memory>
namespace bspai {
// Device-bound reusable inner solver. A and P must remain unchanged for the
// lifetime of this context. Sequential solves may use different right sides.
class Fp32CudaGmresContext {
  public:
    Fp32CudaGmresContext(const Fp32CsrMatrix &, const Fp32CsrMatrix &, int device = 0);
    ~Fp32CudaGmresContext();
    Fp32CudaGmresContext(const Fp32CudaGmresContext &) = delete;
    Fp32CudaGmresContext &operator=(const Fp32CudaGmresContext &) = delete;
    Fp32GmresResult solve(const std::vector<float> &, const Fp32GmresOptions &,
                            const std::vector<float> &initial_guess = {});
    std::size_t matrix_uploads() const noexcept;
    std::size_t matrix_upload_bytes() const noexcept;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
// Uniform-FP32 GPU-resident Arnoldi/vector path. Matrices and Krylov vectors
// stay on device for one solve; only scalar Hessenberg data is observed on host.
Fp32GmresResult fp32_cuda_left_preconditioned_gmres(const Fp32CsrMatrix &, const Fp32CsrMatrix &,
                                                    const std::vector<float> &,
                                                    const Fp32GmresOptions &,
                                                    const std::vector<float> &initial_guess = {},
                                                    int device = 0);
// CPU-staged uniform-FP32 outer refinement reusing the GPU-resident inner
// GMRES above.  P is supplied explicitly; no GPU SPAI construction is implied.
Fp32IrResult fp32_cuda_refine_with_preconditioner(const Fp32CsrMatrix &, const Fp32CsrMatrix &,
                                                  const std::vector<float> &, const Fp32IrOptions &,
                                                  int device = 0);
}
#endif
