#include "bspai/fp32_cuda_spmv.hpp"
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cstdint>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
namespace bspai {
namespace {
void ck(cudaError_t s, const char *what) {
    if (s != cudaSuccess)
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(s));
}
void ck(cusparseStatus_t s, const char *what) {
    if (s != CUSPARSE_STATUS_SUCCESS)
        throw std::runtime_error(std::string(what) + ": cuSPARSE status " + std::to_string((int)s));
}
std::int64_t i64(std::size_t n) {
    if (n > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()))
        throw std::overflow_error("FP32 CUDA index overflow");
    return static_cast<std::int64_t>(n);
}
std::size_t bytes(std::size_t count, std::size_t width) {
    if (count > std::numeric_limits<std::size_t>::max() / width)
        throw std::overflow_error("FP32 CUDA allocation size overflow");
    return count * width;
}
}
struct Fp32CudaCsrSpmv::Impl {
    std::size_t m, n, nnz;
    int device;
    cudaStream_t stream = nullptr;
    cusparseHandle_t h = nullptr;
    cusparseSpMatDescr_t A = nullptr;
    cusparseDnVecDescr_t X = nullptr, Y = nullptr;
    void *ro = nullptr, *ci = nullptr, *av = nullptr, *x = nullptr, *y = nullptr, *buf = nullptr;
    Impl(std::size_t r, std::size_t c, std::size_t z, int d) : m(r), n(c), nnz(z), device(d) {
    }
    ~Impl() {
        cudaSetDevice(device);
        if (stream)
            cudaStreamSynchronize(stream);
        if (X)
            cusparseDestroyDnVec(X);
        if (Y)
            cusparseDestroyDnVec(Y);
        if (A)
            cusparseDestroySpMat(A);
        if (h)
            cusparseDestroy(h);
        for (void *p : {buf, y, x, av, ci, ro})
            if (p)
                cudaFree(p);
        if (stream)
            cudaStreamDestroy(stream);
    }
};
Fp32CudaCsrSpmv::Fp32CudaCsrSpmv(const Fp32CsrMatrix &a, int device) {
    if (!a.rows() || !a.cols() || a.row_offsets().size() != a.rows() + 1 ||
        a.values().size() != a.column_indices().size())
        throw std::invalid_argument("invalid FP32 CUDA CSR");
    ck(cudaSetDevice(device), "cudaSetDevice");
    impl_.reset(new Impl(a.rows(), a.cols(), a.values().size(), device));
    Impl &p = *impl_;
    ck(cudaStreamCreateWithFlags(&p.stream, cudaStreamNonBlocking), "cudaStreamCreate");
    ck(cusparseCreate(&p.h), "cusparseCreate");
    ck(cusparseSetStream(p.h, p.stream), "cusparseSetStream");
    std::vector<std::int64_t> ro(a.row_offsets().begin(), a.row_offsets().end()),
        ci(a.column_indices().begin(), a.column_indices().end());
    auto alloc = [](void **q, std::size_t b) {
        if (b)
            ck(cudaMalloc(q, b), "cudaMalloc");
    };
    alloc(&p.ro, bytes(ro.size(), sizeof(std::int64_t)));
    alloc(&p.ci, bytes(ci.size(), sizeof(std::int64_t)));
    alloc(&p.av, bytes(a.values().size(), sizeof(float)));
    alloc(&p.x, bytes(a.cols(), sizeof(float)));
    alloc(&p.y, bytes(a.rows(), sizeof(float)));
    ck(cudaMemcpyAsync(p.ro, ro.data(), ro.size() * sizeof(std::int64_t), cudaMemcpyHostToDevice,
                       p.stream),
       "H2D ro");
    ck(cudaMemcpyAsync(p.ci, ci.data(), ci.size() * sizeof(std::int64_t), cudaMemcpyHostToDevice,
                       p.stream),
       "H2D ci");
    ck(cudaMemcpyAsync(p.av, a.values().data(), a.values().size() * sizeof(float),
                       cudaMemcpyHostToDevice, p.stream),
       "H2D values");
    ck(cusparseCreateCsr(&p.A, i64(p.m), i64(p.n), i64(p.nnz), p.ro, p.ci, p.av, CUSPARSE_INDEX_64I,
                         CUSPARSE_INDEX_64I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F),
       "CreateCsr");
    ck(cusparseCreateDnVec(&p.X, i64(p.n), p.x, CUDA_R_32F), "CreateDnVec x");
    ck(cusparseCreateDnVec(&p.Y, i64(p.m), p.y, CUDA_R_32F), "CreateDnVec y");
    float alpha = 1.0f, beta = 0.0f;
    std::size_t bytes = 0;
    ck(cusparseSpMV_bufferSize(p.h, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, p.A, p.X, &beta, p.Y,
                               CUDA_R_32F, CUSPARSE_SPMV_CSR_ALG2, &bytes),
       "SpMV buffer");
    alloc(&p.buf, bytes);
    ck(cudaStreamSynchronize(p.stream), "sync");
}
Fp32CudaCsrSpmv::~Fp32CudaCsrSpmv() = default;
std::vector<float> Fp32CudaCsrSpmv::multiply(const std::vector<float> &v) {
    Impl &p = *impl_;
    if (v.size() != p.n)
        throw std::invalid_argument("FP32 CUDA SpMV vector size");
    for (float value : v)
        if (!std::isfinite(value))
            throw std::invalid_argument("nonfinite FP32 CUDA SpMV input");
    ck(cudaSetDevice(p.device), "cudaSetDevice");
    ck(cudaMemcpyAsync(p.x, v.data(), p.n * sizeof(float), cudaMemcpyHostToDevice, p.stream),
       "H2D x");
    float alpha = 1.0f, beta = 0.0f;
    ck(cusparseSpMV(p.h, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, p.A, p.X, &beta, p.Y, CUDA_R_32F,
                    CUSPARSE_SPMV_CSR_ALG2, p.buf),
       "SpMV");
    std::vector<float> r(p.m);
    ck(cudaMemcpyAsync(r.data(), p.y, p.m * sizeof(float), cudaMemcpyDeviceToHost, p.stream),
       "D2H y");
    ck(cudaStreamSynchronize(p.stream), "sync");
    for (float value : r)
        if (!std::isfinite(value))
            throw std::overflow_error("nonfinite FP32 CUDA SpMV output");
    return r;
}
}
