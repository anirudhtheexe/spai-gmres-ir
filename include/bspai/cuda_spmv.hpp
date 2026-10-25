#ifndef BSPAI_CUDA_SPMV_HPP
#define BSPAI_CUDA_SPMV_HPP
#include "bspai/cpu.hpp"
#include <memory>
namespace bspai {
// Synchronous binary64 non-transpose product; same object is not thread-safe.
class CudaCsrSpmv {
  public:
    explicit CudaCsrSpmv(const CsrMatrix &matrix, int device = 0);
    ~CudaCsrSpmv();
    CudaCsrSpmv(const CudaCsrSpmv &) = delete;
    CudaCsrSpmv &operator=(const CudaCsrSpmv &) = delete;
    std::size_t rows() const;
    std::size_t cols() const;
    std::vector<double> multiply(const std::vector<double> &x);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
#endif
