#ifndef BSPAI_FP32_CUDA_SPMV_HPP
#define BSPAI_FP32_CUDA_SPMV_HPP
#include "bspai/fp32.hpp"
#include <memory>
namespace bspai {
// Experiment 2A uniform-binary32 CSR SpMV. Synchronous and not thread-safe.
class Fp32CudaCsrSpmv {
  public:
    explicit Fp32CudaCsrSpmv(const Fp32CsrMatrix &, int device = 0);
    ~Fp32CudaCsrSpmv();
    Fp32CudaCsrSpmv(const Fp32CudaCsrSpmv &) = delete;
    Fp32CudaCsrSpmv &operator=(const Fp32CudaCsrSpmv &) = delete;
    std::vector<float> multiply(const std::vector<float> &);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
#endif
