#include "bspai/fp32_cuda_gmres.hpp"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace bspai {
namespace {
void ck(cudaError_t s, const char *w) {
    if (s != cudaSuccess)
        throw std::runtime_error(std::string(w) + ": " + cudaGetErrorString(s));
}
void ck(cublasStatus_t s, const char *w) {
    if (s != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string(w) + ": cuBLAS " + std::to_string((int)s));
}
void ck(cusparseStatus_t s, const char *w) {
    if (s != CUSPARSE_STATUS_SUCCESS)
        throw std::runtime_error(std::string(w) + ": cuSPARSE " + std::to_string((int)s));
}
std::int64_t i64(std::size_t n) {
    if (n > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()))
        throw std::overflow_error("CUDA index overflow");
    return static_cast<std::int64_t>(n);
}
std::size_t fb(std::size_t n) {
    if (n > std::numeric_limits<std::size_t>::max() / sizeof(float))
        throw std::overflow_error("CUDA allocation overflow");
    return n * sizeof(float);
}
float finite(float x, const char *w) {
    if (!std::isfinite(x))
        throw std::overflow_error(w);
    return x;
}
bool canonical(const Fp32CsrMatrix &a) {
    if (!a.rows() || a.rows() != a.cols() || a.row_offsets().size() != a.rows() + 1 ||
        a.values().size() != a.column_indices().size() || a.row_offsets().front() != 0 ||
        a.row_offsets().back() != a.values().size())
        return false;
    for (std::size_t r = 0; r < a.rows(); ++r) {
        if (a.row_offsets()[r] > a.row_offsets()[r + 1])
            return false;
        bool first = true;
        std::size_t prev = 0;
        for (std::size_t q = a.row_offsets()[r]; q < a.row_offsets()[r + 1]; ++q) {
            if (a.column_indices()[q] >= a.cols() || !std::isfinite(a.values()[q]) ||
                (!first && a.column_indices()[q] <= prev))
                return false;
            prev = a.column_indices()[q];
            first = false;
        }
    }
    return true;
}
struct Device {
    int old = 0;
    bool active = false;
    explicit Device(int d) {
        ck(cudaGetDevice(&old), "cudaGetDevice");
        ck(cudaSetDevice(d), "cudaSetDevice");
        active = true;
    }
    ~Device() {
        if (active)
            cudaSetDevice(old);
    }
};
struct Mem {
    void *p = nullptr;
    explicit Mem(std::size_t bytes = 0) {
        if (bytes)
            ck(cudaMalloc(&p, bytes), "cudaMalloc");
    }
    ~Mem() {
        if (p)
            cudaFree(p);
    }
    Mem(const Mem &) = delete;
    float *f() const {
        return static_cast<float *>(p);
    }
};
struct Matrix {
    cusparseSpMatDescr_t d = nullptr;
    Mem ro, ci, v, work;
    std::size_t m = 0, n = 0, nnz = 0;
    Matrix(const Fp32CsrMatrix &a, cudaStream_t stream)
        : ro(a.row_offsets().size() * sizeof(std::int64_t)),
          ci(a.values().size() * sizeof(std::int64_t)), v(fb(a.values().size())), m(a.rows()),
          n(a.cols()), nnz(a.values().size()) {
        std::vector<std::int64_t> r(a.row_offsets().begin(), a.row_offsets().end()),
            c(a.column_indices().begin(), a.column_indices().end());
        ck(cudaMemcpyAsync(ro.p, r.data(), r.size() * sizeof(std::int64_t), cudaMemcpyHostToDevice,
                           stream),
           "CSR row offsets H2D");
        if (nnz) {
            ck(cudaMemcpyAsync(ci.p, c.data(), c.size() * sizeof(std::int64_t),
                               cudaMemcpyHostToDevice, stream),
               "CSR columns H2D");
            ck(cudaMemcpyAsync(v.p, a.values().data(), fb(nnz), cudaMemcpyHostToDevice, stream),
               "CSR values H2D");
        }
        ck(cusparseCreateCsr(&d, i64(m), i64(n), i64(nnz), ro.p, ci.p, v.p, CUSPARSE_INDEX_64I,
                             CUSPARSE_INDEX_64I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F),
           "cusparseCreateCsr FP32");
    }
    ~Matrix() {
        if (d)
            cusparseDestroySpMat(d);
    }
};
struct Engine {
    cudaStream_t stream = nullptr;
    cublasHandle_t blas = nullptr;
    cusparseHandle_t sparse = nullptr;
    cusparseDnVecDescr_t x = nullptr, y = nullptr;
    std::size_t n;
    Engine(std::size_t count, float *xv, float *yv) : n(count) {
        ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate");
        ck(cublasCreate(&blas), "cublasCreate");
        ck(cusparseCreate(&sparse), "cusparseCreate");
        ck(cublasSetStream(blas, stream), "cublasSetStream");
        ck(cusparseSetStream(sparse, stream), "cusparseSetStream");
        ck(cusparseCreateDnVec(&x, i64(n), xv, CUDA_R_32F), "cusparseCreateDnVec X");
        ck(cusparseCreateDnVec(&y, i64(n), yv, CUDA_R_32F), "cusparseCreateDnVec Y");
    }
    ~Engine() {
        if (stream)
            cudaStreamSynchronize(stream);
        if (x)
            cusparseDestroyDnVec(x);
        if (y)
            cusparseDestroyDnVec(y);
        if (blas)
            cublasDestroy(blas);
        if (sparse)
            cusparseDestroy(sparse);
        if (stream)
            cudaStreamDestroy(stream);
    }
    void copy(const float *a, float *b) {
        ck(cublasScopy_64(blas, i64(n), a, 1, b, 1), "cublasScopy_64");
    }
    void axpy(float alpha, const float *a, float *b) {
        ck(cublasSaxpy_64(blas, i64(n), &alpha, a, 1, b, 1), "cublasSaxpy_64");
    }
    void scale(float alpha, float *a) {
        ck(cublasSscal_64(blas, i64(n), &alpha, a, 1), "cublasSscal_64");
    }
    float dot(const float *a, const float *b) {
        float value = 0;
        ck(cublasSdot_64(blas, i64(n), a, 1, b, 1, &value), "cublasSdot_64");
        return finite(value, "nonfinite FP32 CUDA dot");
    }
    float norm(const float *a) {
        float value = 0;
        ck(cublasSnrm2_64(blas, i64(n), a, 1, &value), "cublasSnrm2_64");
        return finite(value, "nonfinite FP32 CUDA norm");
    }
    void apply(Matrix &a, float *in, float *out) {
        float one = 1, zero = 0;
        std::size_t need = 0;
        ck(cusparseDnVecSetValues(x, in), "cusparse set X");
        ck(cusparseDnVecSetValues(y, out), "cusparse set Y");
        ck(cusparseSpMV_bufferSize(sparse, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, a.d, x, &zero, y,
                                   CUDA_R_32F, CUSPARSE_SPMV_CSR_ALG2, &need),
           "cusparse SpMV bufferSize");
        if (!a.work.p && need)
            ck(cudaMalloc(&a.work.p, need), "cudaMalloc SpMV workspace");
        ck(cusparseSpMV(sparse, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, a.d, x, &zero, y,
                        CUDA_R_32F, CUSPARSE_SPMV_CSR_ALG2, a.work.p),
           "cusparse SpMV");
    }
};
} // namespace

Fp32GmresResult fp32_cuda_left_preconditioned_gmres(const Fp32CsrMatrix &a, const Fp32CsrMatrix &p,
                                                    const std::vector<float> &rhs,
                                                    const Fp32GmresOptions &o,
                                                    const std::vector<float> &initial_guess,
                                                    int device) {
    Fp32GmresResult result;
    const std::size_t n = a.rows(), cap = o.max_iterations ? o.max_iterations : n;
    if (!canonical(a) || !canonical(p) || p.rows() != n || rhs.size() != n ||
        (!initial_guess.empty() && initial_guess.size() != n) || !std::isfinite(o.tolerance) ||
        o.tolerance <= 0 || !cap || cap > n) {
        result.message = "invalid FP32 CUDA GMRES input";
        return result;
    }
    for (float x : rhs)
        if (!std::isfinite(x)) {
            result.message = "nonfinite FP32 CUDA RHS";
            return result;
        }
    for (float x : initial_guess)
        if (!std::isfinite(x)) {
            result.message = "nonfinite FP32 CUDA initial guess";
            return result;
        }
    const std::vector<float> initial =
        initial_guess.empty() ? std::vector<float>(n, 0) : initial_guess;
    result.solution = initial;
    try {
        if (cap == std::numeric_limits<std::size_t>::max() ||
            n > std::numeric_limits<std::size_t>::max() / (cap + 1))
            throw std::overflow_error("FP32 CUDA Krylov element count overflow");
        Device device_guard(device);
        Mem dr(fb(n)), dx(fb(n)), dinitial(fb(n)), dcandidate(fb(n)), dax(fb(n)), dpb(fb(n)),
            dpa(fb(n)), dw(fb(n));
        Engine e(n, dx.f(), dax.f());
        Matrix A(a, e.stream), P(p, e.stream);
        ck(cudaMemcpyAsync(dr.p, rhs.data(), fb(n), cudaMemcpyHostToDevice, e.stream),
           "FP32 RHS H2D");
        ck(cudaMemcpyAsync(dx.p, initial.data(), fb(n), cudaMemcpyHostToDevice, e.stream),
           "FP32 initial H2D");
        ck(cudaMemcpyAsync(dinitial.p, initial.data(), fb(n), cudaMemcpyHostToDevice, e.stream),
           "FP32 immutable initial H2D");
        // Exact initial solution is accepted before forming P*b, matching the FP32 contract.
        e.apply(A, dx.f(), dax.f());
        e.copy(dr.f(), dw.f());
        e.axpy(-1, dax.f(), dw.f());
        result.final_original_residual = e.norm(dw.f());
        if (result.final_original_residual == 0) {
            result.status = Fp32GmresStatus::exact_initial_solution;
            result.converged = true;
            result.final_preconditioned_relative = 0;
            return result;
        }
        e.apply(P, dr.f(), dpb.f());
        const float denominator = e.norm(dpb.f());
        if (denominator == 0) {
            result.status = Fp32GmresStatus::zero_preconditioned_rhs;
            return result;
        }
        e.apply(P, dax.f(), dpa.f());
        e.copy(dpb.f(), dw.f());
        e.axpy(-1, dpa.f(), dw.f());
        const float beta = e.norm(dw.f());
        result.final_preconditioned_relative =
            finite(beta / denominator, "nonfinite FP32 CUDA relative residual");
        if (result.final_preconditioned_relative <= o.tolerance) {
            result.status = Fp32GmresStatus::success;
            result.converged = true;
            return result;
        }
        constexpr std::size_t storage_chunk = 16;
        std::vector<std::unique_ptr<Mem>> basis_chunks;
        auto basis_vector = [&](std::size_t j) -> float * {
            const std::size_t chunk = j / storage_chunk;
            while (basis_chunks.size() <= chunk) {
                const std::size_t first = basis_chunks.size() * storage_chunk;
                const std::size_t count = std::min(storage_chunk, cap + 1 - first);
                basis_chunks.emplace_back(new Mem(fb(n * count)));
            }
            return basis_chunks[chunk]->f() + n * (j % storage_chunk);
        };
        std::size_t projected_capacity = std::min(cap, storage_chunk);
        e.copy(dw.f(), basis_vector(0));
        e.scale(finite(1.0f / beta, "invalid FP32 CUDA normalization"), basis_vector(0));
        std::vector<std::vector<float>> h(projected_capacity + 1,
                                          std::vector<float>(projected_capacity, 0));
        std::vector<float> cs(projected_capacity, 0), sn(projected_capacity, 0),
            g(projected_capacity + 1, 0);
        g[0] = beta;
        for (std::size_t k = 0; k < cap; ++k) {
            if (k == projected_capacity) {
                const std::size_t next_capacity =
                    projected_capacity + std::min(storage_chunk, cap - projected_capacity);
                for (auto &row : h)
                    row.resize(next_capacity, 0);
                h.resize(next_capacity + 1, std::vector<float>(next_capacity, 0));
                cs.resize(next_capacity, 0);
                sn.resize(next_capacity, 0);
                g.resize(next_capacity + 1, 0);
                projected_capacity = next_capacity;
            }
            float *vk = basis_vector(k);
            e.apply(A, vk, dax.f());
            e.apply(P, dax.f(), dw.f());
            for (std::size_t j = 0; j <= k; ++j) {
                float *vj = basis_vector(j);
                h[j][k] = e.dot(vj, dw.f());
                e.axpy(-h[j][k], vj, dw.f());
            }
            h[k + 1][k] = e.norm(dw.f());
            const bool arnoldi_breakdown = h[k + 1][k] == 0;
            if (!arnoldi_breakdown) {
                e.copy(dw.f(), basis_vector(k + 1));
                e.scale(finite(1.0f / h[k + 1][k], "invalid FP32 CUDA Arnoldi normalization"),
                        basis_vector(k + 1));
            }
            for (std::size_t j = 0; j < k; ++j) {
                const float u = h[j][k], v = h[j + 1][k];
                h[j][k] = finite(cs[j] * u + sn[j] * v, "nonfinite FP32 CUDA Givens rotation");
                h[j + 1][k] = finite(-sn[j] * u + cs[j] * v, "nonfinite FP32 CUDA Givens rotation");
            }
            const float radius =
                finite(std::hypot(h[k][k], h[k + 1][k]), "nonfinite FP32 CUDA Givens radius");
            if (radius == 0) {
                result.status = Fp32GmresStatus::breakdown;
                result.message = "zero FP32 CUDA Givens pivot";
                break;
            }
            cs[k] = finite(h[k][k] / radius, "nonfinite FP32 CUDA cosine");
            sn[k] = finite(h[k + 1][k] / radius, "nonfinite FP32 CUDA sine");
            h[k][k] = radius;
            const float old = g[k];
            g[k] = finite(cs[k] * old, "nonfinite FP32 CUDA Givens RHS");
            g[k + 1] = finite(-sn[k] * old, "nonfinite FP32 CUDA Givens RHS");
            result.hessenberg_residual_history.push_back(std::fabs(g[k + 1]));
            std::vector<float> y(k + 1, 0);
            bool triangular_breakdown = false;
            for (std::size_t row = k + 1; row-- > 0;) {
                if (h[row][row] == 0) {
                    triangular_breakdown = true;
                    break;
                }
                float value = g[row];
                for (std::size_t col = row + 1; col <= k; ++col)
                    value = finite(value - h[row][col] * y[col],
                                   "nonfinite FP32 CUDA triangular solve");
                y[row] = finite(value / h[row][row], "nonfinite FP32 CUDA triangular solve");
            }
            if (triangular_breakdown) {
                result.status = Fp32GmresStatus::breakdown;
                result.message = "zero FP32 CUDA triangular pivot";
                break;
            }
            // Each projected GMRES candidate is x0 + V_k*y_k.  dx holds only the
            // most recently verified candidate for the returned solution and must not
            // become the base of the next projected candidate.
            e.copy(dinitial.f(), dcandidate.f());
            for (std::size_t j = 0; j <= k; ++j)
                e.axpy(y[j], basis_vector(j), dcandidate.f());
            e.apply(A, dcandidate.f(), dax.f());
            e.copy(dr.f(), dw.f());
            e.axpy(-1, dax.f(), dw.f());
            result.final_original_residual = e.norm(dw.f());
            e.apply(P, dax.f(), dpa.f());
            e.copy(dpb.f(), dw.f());
            e.axpy(-1, dpa.f(), dw.f());
            result.final_preconditioned_relative =
                finite(e.norm(dw.f()) / denominator, "nonfinite FP32 CUDA relative residual");
            result.original_residual_history.push_back(result.final_original_residual);
            result.preconditioned_relative_history.push_back(result.final_preconditioned_relative);
            result.iterations = k + 1;
            // Retain the last formed candidate for every ordinary termination path.
            e.copy(dcandidate.f(), dx.f());
            if (result.final_preconditioned_relative <= o.tolerance) {
                result.status = Fp32GmresStatus::success;
                result.converged = true;
                break;
            }
            if (arnoldi_breakdown) {
                result.status = Fp32GmresStatus::breakdown;
                result.message = "FP32 CUDA Arnoldi breakdown candidate failed residual check";
                break;
            }
            if (k + 1 == cap)
                result.status =
                    cap < n ? Fp32GmresStatus::iteration_limit : Fp32GmresStatus::nonconvergence;
        }
        ck(cudaMemcpyAsync(result.solution.data(), dx.p, fb(n), cudaMemcpyDeviceToHost, e.stream),
           "FP32 solution D2H");
        ck(cudaStreamSynchronize(e.stream), "FP32 CUDA GMRES synchronize");
        return result;
    } catch (const std::exception &error) {
        result.status = Fp32GmresStatus::numerical_failure;
        result.message = error.what();
        return result;
    }
}

Fp32IrResult fp32_cuda_refine_with_preconditioner(const Fp32CsrMatrix &a, const Fp32CsrMatrix &p,
                                                  const std::vector<float> &rhs,
                                                  const Fp32IrOptions &o, int device) {
    Fp32IrResult result(a.rows());
    if (!canonical(a) || !canonical(p) || p.rows() != a.rows() || rhs.size() != a.rows() ||
        o.max_outer_steps > 20) {
        result.message = "invalid FP32 CUDA iterative-refinement input";
        return result;
    }
    try {
        result.solution = p.multiply(rhs);
        for (std::size_t step = 0; step <= o.max_outer_steps; ++step) {
            const Fp32IrDiagnostics diagnostics =
                fp32_evaluate_ir_diagnostics(a, rhs, result.solution);
            result.outer_history.push_back(diagnostics);
            if (diagnostics.normwise_backward_error <= fp32_ir_backward_tolerance()) {
                result.status = Fp32IrStatus::success;
                result.converged = true;
                return result;
            }
            if (step == o.max_outer_steps)
                break;
            Fp32GmresResult inner = fp32_cuda_left_preconditioned_gmres(a, p, diagnostics.residual,
                                                                        o.gmres, {}, device);
            result.inner_history.push_back(inner);
            ++result.inner_attempts;
            if (!inner.converged) {
                result.status = Fp32IrStatus::inner_failure;
                result.message = fp32_gmres_status_name(inner.status);
                return result;
            }
            const std::vector<float> previous = result.solution;
            for (std::size_t i = 0; i < result.solution.size(); ++i)
                result.solution[i] = finite(result.solution[i] + inner.solution[i],
                                            "nonfinite FP32 CUDA outer update");
            ++result.outer_steps;
            const Fp32IrDiagnostics after = fp32_evaluate_ir_diagnostics(a, rhs, result.solution);
            if (after.normwise_backward_error <= fp32_ir_backward_tolerance()) {
                result.outer_history.push_back(after);
                result.status = Fp32IrStatus::success;
                result.converged = true;
                return result;
            }
            if (result.solution == previous) {
                result.outer_history.push_back(after);
                result.status = Fp32IrStatus::stagnation;
                return result;
            }
        }
        result.status = Fp32IrStatus::iteration_limit;
        return result;
    } catch (const std::exception &error) {
        result.status = Fp32IrStatus::numerical_failure;
        result.message = error.what();
        return result;
    }
}
} // namespace bspai
