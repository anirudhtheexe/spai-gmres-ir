#include "bspai/cuda_gmres.hpp"
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

namespace bspai {
static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559 &&
                  std::numeric_limits<double>::digits == 53,
              "GPU GMRES requires IEEE binary64");
static_assert(sizeof(std::size_t) == 8, "GPU GMRES requires a 64-bit host");
namespace {
void checked(cudaError_t s, const char *op) {
    if (s != cudaSuccess)
        throw std::runtime_error(std::string(op) + ": " + cudaGetErrorString(s));
}
void checked(cusparseStatus_t s, const char *op) {
    if (s != CUSPARSE_STATUS_SUCCESS)
        throw std::runtime_error(std::string(op) + ": " + cusparseGetErrorString(s));
}
void checked(cublasStatus_t s, const char *op) {
    if (s != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string(op) + ": cuBLAS status " +
                                 std::to_string(static_cast<int>(s)));
}
void release(cudaError_t s) {
    if (s != cudaSuccess)
        std::cerr << "GPU GMRES cleanup: " << cudaGetErrorString(s) << '\n';
}
void release(cusparseStatus_t s) {
    if (s != CUSPARSE_STATUS_SUCCESS)
        std::cerr << "GPU GMRES cuSPARSE cleanup status " << s << '\n';
}
void release(cublasStatus_t s) {
    if (s != CUBLAS_STATUS_SUCCESS)
        std::cerr << "GPU GMRES cuBLAS cleanup status " << s << '\n';
}
double finite(double v) {
    if (!std::isfinite(v))
        throw std::overflow_error("Nonfinite GPU GMRES arithmetic");
    return v;
}
std::size_t product(std::size_t a, std::size_t b) {
    if (a && b > std::numeric_limits<std::size_t>::max() / a)
        throw std::overflow_error("GPU GMRES allocation size overflow");
    return a * b;
}
std::int64_t index64(std::size_t n) {
    if (n > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()))
        throw std::overflow_error("GPU GMRES index exceeds int64");
    return static_cast<std::int64_t>(n);
}
bool canonical(const CsrMatrix &a) {
    if (a.rows() == std::numeric_limits<std::size_t>::max() ||
        a.row_offsets().size() != a.rows() + 1 || a.row_offsets().front() != 0 ||
        a.row_offsets().back() != a.values().size() ||
        a.column_indices().size() != a.values().size())
        return false;
    for (std::size_t row = 0; row < a.rows(); ++row) {
        const auto first = a.row_offsets()[row], last = a.row_offsets()[row + 1];
        if (first > last || last > a.values().size())
            return false;
        for (std::size_t k = first; k < last; ++k)
            if (a.column_indices()[k] >= a.cols() ||
                (k > first && a.column_indices()[k - 1] >= a.column_indices()[k]) ||
                !std::isfinite(a.values()[k]))
                return false;
    }
    return true;
}
struct DeviceGuard {
    int old;
    explicit DeviceGuard(int d) {
        checked(cudaGetDevice(&old), "cudaGetDevice");
        checked(cudaSetDevice(d), "cudaSetDevice");
    }
    ~DeviceGuard() {
        release(cudaSetDevice(old));
    }
};
struct Memory {
    void *p = nullptr;
    explicit Memory(std::size_t bytes = 0) {
        if (bytes)
            checked(cudaMalloc(&p, bytes), "cudaMalloc");
    }
    ~Memory() {
        if (p)
            release(cudaFree(p));
    }
    Memory(const Memory &) = delete;
    Memory &operator=(const Memory &) = delete;
    double *doubles() {
        return static_cast<double *>(p);
    }
};
struct Stream {
    cudaStream_t p = nullptr;
    Stream() {
        checked(cudaStreamCreateWithFlags(&p, cudaStreamNonBlocking), "cudaStreamCreate");
    }
    ~Stream() {
        release(cudaStreamSynchronize(p));
        release(cudaStreamDestroy(p));
    }
};
struct Blas {
    cublasHandle_t p = nullptr;
    Blas() {
        checked(cublasCreate(&p), "cublasCreate");
    }
    ~Blas() {
        release(cublasDestroy(p));
    }
};
struct Sparse {
    cusparseHandle_t p = nullptr;
    Sparse() {
        checked(cusparseCreate(&p), "cusparseCreate");
    }
    ~Sparse() {
        release(cusparseDestroy(p));
    }
};
struct SparseDescriptor {
    cusparseSpMatDescr_t p = nullptr;
    ~SparseDescriptor() {
        if (p)
            release(cusparseDestroySpMat(p));
    }
};
struct DenseDescriptor {
    cusparseDnVecDescr_t p = nullptr;
    ~DenseDescriptor() {
        if (p)
            release(cusparseDestroyDnVec(p));
    }
};

__global__ void finite_flag(const double *x, std::size_t n, int *flag) {
    for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; i < n;
         i += static_cast<std::size_t>(blockDim.x) * gridDim.x)
        if (!isfinite(x[i]))
            atomicExch(flag, 1);
}
__global__ void divide_vector(const double *x, double *y, std::size_t n, double divisor) {
    for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; i < n;
         i += static_cast<std::size_t>(blockDim.x) * gridDim.x)
        y[i] = x[i] / divisor;
}
// Deterministic GPU LASSQ recurrence. Serial debugging baseline, not optimized.
__global__ void safe_scaled_norm(const double *x, std::size_t n, double *result) {
    double scale = 0, ssq = 1;
    for (std::size_t i = 0; i < n; ++i) {
        const double a = fabs(x[i]);
        if (!isfinite(a)) {
            *result = a;
            return;
        }
        if (a != 0) {
            if (scale < a) {
                const double ratio = scale / a;
                ssq = 1 + ssq * ratio * ratio;
                scale = a;
            } else {
                const double ratio = a / scale;
                ssq = ssq + ratio * ratio;
            }
        }
    }
    *result = scale == 0 ? 0 : scale * sqrt(ssq);
}
struct Engine {
    Stream stream;
    Blas blas;
    Sparse sparse;
    Memory flag{sizeof(int)}, norm_result{sizeof(double)};
    std::size_t n;
    CudaGmresTransfers &transfers;
    bool in_arnoldi = false;
    Engine(std::size_t size, CudaGmresTransfers &t) : n(size), transfers(t) {
        checked(cublasSetStream(blas.p, stream.p), "cublasSetStream");
        checked(cublasSetPointerMode(blas.p, CUBLAS_POINTER_MODE_HOST), "cublasSetPointerMode");
        checked(cublasSetMathMode(blas.p, CUBLAS_PEDANTIC_MATH), "cublasSetMathMode");
        cublasMath_t actual = CUBLAS_DEFAULT_MATH;
        checked(cublasGetMathMode(blas.p, &actual), "cublasGetMathMode");
        if (actual != CUBLAS_PEDANTIC_MATH)
            throw std::runtime_error("Native pedantic FP64 math mode not retained");
        checked(cublasSetAtomicsMode(blas.p, CUBLAS_ATOMICS_NOT_ALLOWED), "cublasSetAtomicsMode");
        checked(cusparseSetStream(sparse.p, stream.p), "cusparseSetStream");
        checked(cusparseSetPointerMode(sparse.p, CUSPARSE_POINTER_MODE_HOST),
                "cusparseSetPointerMode");
    }
    void sync() {
        checked(cudaStreamSynchronize(stream.p), "GMRES stream synchronization");
    }
    void upload(void *d, const void *h, std::size_t bytes) {
        if (!bytes)
            return;
        checked(cudaMemcpyAsync(d, h, bytes, cudaMemcpyHostToDevice, stream.p), "GMRES H2D");
        sync();
        ++transfers.vector_uploads;
        if (in_arnoldi)
            ++transfers.arnoldi_vector_uploads;
    }
    std::vector<double> download(const double *d) {
        std::vector<double> h(n);
        checked(cudaMemcpyAsync(h.data(), d, product(n, sizeof(double)), cudaMemcpyDeviceToHost,
                                stream.p),
                "GMRES D2H");
        sync();
        ++transfers.vector_downloads;
        if (in_arnoldi)
            ++transfers.arnoldi_vector_downloads;
        return h;
    }
    void validate(const double *x) {
        checked(cudaMemsetAsync(flag.p, 0, sizeof(int), stream.p), "GMRES finite flag clear");
        finite_flag<<<static_cast<unsigned>(std::min<std::size_t>((n + 255) / 256, 65535)), 256, 0,
                      stream.p>>>(x, n, static_cast<int *>(flag.p));
        checked(cudaGetLastError(), "GMRES finite flag kernel");
        int host = 0;
        checked(cudaMemcpyAsync(&host, flag.p, sizeof(int), cudaMemcpyDeviceToHost, stream.p),
                "GMRES finite status D2H");
        sync();
        ++transfers.scalar_observations;
        ++transfers.finite_flag_observations;
        if (host)
            throw std::overflow_error("Nonfinite GPU GMRES vector");
    }
    void copy(const double *x, double *y) {
        checked(cublasDcopy_64(blas.p, index64(n), x, 1, y, 1), "cublasDcopy_64");
    }
    void axpy(double a, const double *x, double *y) {
        ++transfers.host_scalar_inputs;
        ++transfers.axpy_scalar_inputs;
        checked(cublasDaxpy_64(blas.p, index64(n), &a, x, 1, y, 1), "cublasDaxpy_64");
        validate(y);
    }
    void subtract(const double *x, const double *y, double *out) {
        copy(x, out);
        axpy(-1, y, out);
    }
    double norm(const double *x) {
        safe_scaled_norm<<<1, 1, 0, stream.p>>>(x, n, norm_result.doubles());
        checked(cudaGetLastError(), "GMRES safe scaled norm kernel");
        double v = 0;
        checked(
            cudaMemcpyAsync(&v, norm_result.p, sizeof(double), cudaMemcpyDeviceToHost, stream.p),
            "GMRES norm scalar D2H");
        sync();
        ++transfers.scalar_observations;
        ++transfers.norm_observations;
        return finite(v);
    }
    double dot(const double *x, const double *y) {
        double v = 0;
        checked(cublasDdot_64(blas.p, index64(n), x, 1, y, 1, &v), "cublasDdot_64");
        sync();
        ++transfers.scalar_observations;
        ++transfers.dot_observations;
        return finite(v);
    }
    void normalize(const double *x, double *y, double divisor) {
        if (!std::isfinite(divisor) || divisor <= 0)
            throw std::overflow_error("Invalid GPU GMRES normalization divisor");
        ++transfers.host_scalar_inputs;
        ++transfers.division_scalar_inputs;
        divide_vector<<<static_cast<unsigned>(std::min<std::size_t>((n + 255) / 256, 65535)), 256,
                        0, stream.p>>>(x, y, n, divisor);
        checked(cudaGetLastError(), "GMRES division kernel");
        validate(y);
    }
};
struct Matrix {
    Engine &e;
    std::size_t nnz;
    Memory offsets, columns, values, workspace;
    SparseDescriptor descriptor;
    DenseDescriptor vx, vy;
    Matrix(Engine &engine, const CsrMatrix &a, double *temporary_x, double *temporary_y)
        : e(engine), nnz(a.values().size()),
          offsets(product(a.row_offsets().size(), sizeof(std::int64_t))),
          columns(product(nnz, sizeof(std::int64_t))), values(product(nnz, sizeof(double))) {
        if (!nnz)
            return;
        std::vector<std::int64_t> ro, ci;
        ro.reserve(a.row_offsets().size());
        ci.reserve(nnz);
        for (auto k : a.row_offsets())
            ro.push_back(index64(k));
        for (auto k : a.column_indices())
            ci.push_back(index64(k));
        e.upload(offsets.p, ro.data(), product(ro.size(), sizeof(std::int64_t)));
        e.upload(columns.p, ci.data(), product(nnz, sizeof(std::int64_t)));
        e.upload(values.p, a.values().data(), product(nnz, sizeof(double)));
        checked(cusparseCreateCsr(&descriptor.p, index64(a.rows()), index64(a.cols()), index64(nnz),
                                  offsets.p, columns.p, values.p, CUSPARSE_INDEX_64I,
                                  CUSPARSE_INDEX_64I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F),
                "GMRES cusparseCreateCsr");
        checked(cusparseCreateDnVec(&vx.p, index64(a.cols()), temporary_x, CUDA_R_64F),
                "GMRES vector descriptor X");
        checked(cusparseCreateDnVec(&vy.p, index64(a.rows()), temporary_y, CUDA_R_64F),
                "GMRES vector descriptor Y");
        const double alpha = 1, beta = 0;
        std::size_t size = 0;
        checked(cusparseSpMV_bufferSize(e.sparse.p, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                                        descriptor.p, vx.p, &beta, vy.p, CUDA_R_64F,
                                        CUSPARSE_SPMV_CSR_ALG2, &size),
                "GMRES SpMV bufferSize");
        if (size)
            checked(cudaMalloc(&workspace.p, size), "GMRES SpMV workspace");
    }
    void apply(const double *x, double *y) {
        checked(cudaMemsetAsync(y, 0, product(e.n, sizeof(double)), e.stream.p),
                "GMRES SpMV clear Y");
        if (nnz) {
            checked(cusparseDnVecSetValues(vx.p, const_cast<double *>(x)), "GMRES vector X values");
            checked(cusparseDnVecSetValues(vy.p, y), "GMRES vector Y values");
            const double alpha = 1, beta = 0;
            e.transfers.host_scalar_inputs += 2;
            e.transfers.spmv_scalar_inputs += 2;
            checked(cusparseSpMV(e.sparse.p, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, descriptor.p,
                                 vx.p, &beta, vy.p, CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2,
                                 workspace.p),
                    "GMRES cusparseSpMV");
        }
        e.validate(y);
    }
};
struct Run {
    Engine e;
    std::size_t n, cap;
    Memory rhs, initial, accepted, candidate, ax, pa, work, residual, pb, basis, snapshots;
    Matrix a, p;
    std::size_t basis_count = 0, snapshot_count = 0;
    Run(const CsrMatrix &ah, const CsrMatrix &ph, std::size_t budget, CudaGmresTransfers &t)
        : e(ah.rows(), t), n(ah.rows()), cap(budget), rhs(product(n, sizeof(double))),
          initial(product(n, sizeof(double))), accepted(product(n, sizeof(double))),
          candidate(product(n, sizeof(double))), ax(product(n, sizeof(double))),
          pa(product(n, sizeof(double))), work(product(n, sizeof(double))),
          residual(product(n, sizeof(double))), pb(product(n, sizeof(double))),
          basis(product(product(n, cap + 1), sizeof(double))),
          snapshots(product(product(n, cap), sizeof(double))),
          a(e, ah, initial.doubles(), ax.doubles()), p(e, ph, initial.doubles(), pa.doubles()) {
    }
    ~Run() {
        release(cudaStreamSynchronize(e.stream.p));
    }
    double *v(std::size_t j) {
        return basis.doubles() + product(n, j);
    }
    void snapshot() {
        e.copy(accepted.doubles(), snapshots.doubles() + product(n, snapshot_count));
        ++snapshot_count;
    }
    void check_candidate(GmresResult &result, GmresIteration &record, const double *x,
                         double denominator) {
        a.apply(x, ax.doubles());
        e.subtract(rhs.doubles(), ax.doubles(), residual.doubles());
        record.original_residual = e.norm(residual.doubles());
        p.apply(ax.doubles(), pa.doubles());
        e.subtract(pb.doubles(), pa.doubles(), residual.doubles());
        record.recomputed_relative = finite(e.norm(residual.doubles()) / denominator);
        result.final_original_residual = record.original_residual;
        result.final_preconditioned_relative = record.recomputed_relative;
    }
    void finish(GmresResult &result) {
        e.in_arnoldi = false;
        e.sync();
        result.solution = e.download(accepted.doubles());
        for (std::size_t j = 0; j < basis_count; ++j)
            result.basis.push_back(e.download(v(j)));
        for (std::size_t j = 0; j < snapshot_count; ++j)
            result.history[j].solution = e.download(snapshots.doubles() + product(n, j));
    }
};
bool candidate_coefficients(const std::vector<std::vector<double>> &h, const std::vector<double> &g,
                            std::size_t count, std::vector<double> &y) {
    y.assign(count, 0);
    for (std::size_t row = count; row-- > 0;) {
        if (h[row][row] == 0)
            return false;
        double value = g[row];
        for (std::size_t col = row + 1; col < count; ++col)
            value = finite(value - finite(h[row][col] * y[col]));
        y[row] = finite(value / h[row][row]);
    }
    return true;
}
void solve(Run &run, GmresResult &result, const GmresOptions &options,
           const std::vector<double> &rhs, const std::vector<double> &initial) {
    Engine &e = run.e;
    const std::size_t cap = run.cap, n = run.n;
    e.upload(run.rhs.p, rhs.data(), product(n, sizeof(double)));
    e.upload(run.initial.p, initial.data(), product(n, sizeof(double)));
    e.copy(run.initial.doubles(), run.accepted.doubles());
    run.a.apply(run.initial.doubles(), run.ax.doubles());
    e.subtract(run.rhs.doubles(), run.ax.doubles(), run.residual.doubles());
    result.initial_original_residual = e.norm(run.residual.doubles());
    result.final_original_residual = result.initial_original_residual;
    if (result.initial_original_residual == 0) {
        result.converged = true;
        result.status = std::all_of(rhs.begin(), rhs.end(),
                                    [](double v) {
                                        return v == 0;
                                    })
                            ? GmresStatus::zero_rhs
                            : GmresStatus::exact_initial_solution;
        result.initial_preconditioned_relative = result.final_preconditioned_relative = 0;
        return;
    }
    run.p.apply(run.rhs.doubles(), run.pb.doubles());
    const double denominator = e.norm(run.pb.doubles());
    if (denominator == 0) {
        result.status = GmresStatus::zero_preconditioned_rhs;
        result.message = "Zero preconditioned RHS with nonzero original initial residual";
        return;
    }
    run.p.apply(run.ax.doubles(), run.pa.doubles());
    e.subtract(run.pb.doubles(), run.pa.doubles(), run.residual.doubles());
    const double beta = e.norm(run.residual.doubles());
    result.initial_preconditioned_relative = result.final_preconditioned_relative =
        finite(beta / denominator);
    if (result.initial_preconditioned_relative <= options.tolerance) {
        result.status = GmresStatus::success;
        result.converged = true;
        return;
    }
    e.normalize(run.residual.doubles(), run.v(0), beta);
    run.basis_count = 1;
    result.hessenberg.assign(cap + 1, std::vector<double>(cap, 0));
    std::vector<std::vector<double>> rotated(cap + 1, std::vector<double>(cap, 0));
    std::vector<double> cosine(cap, 0), sine(cap, 0), g(cap + 1, 0);
    g[0] = beta;
    e.in_arnoldi = true;
    for (std::size_t k = 0; k < cap; ++k) {
        result.iterations = k + 1;
        run.a.apply(run.v(k), run.ax.doubles());
        run.p.apply(run.ax.doubles(), run.work.doubles());
        for (std::size_t j = 0; j <= k; ++j) {
            const double h = e.dot(run.v(j), run.work.doubles());
            result.hessenberg[j][k] = h;
            e.axpy(-h, run.v(j), run.work.doubles());
        }
        const double hnext = e.norm(run.work.doubles());
        result.hessenberg[k + 1][k] = hnext;
        const bool breakdown = hnext == 0;
        if (!breakdown) {
            e.normalize(run.work.doubles(), run.v(k + 1), hnext);
            ++run.basis_count;
        }
        for (std::size_t row = 0; row <= k + 1; ++row)
            rotated[row][k] = result.hessenberg[row][k];
        for (std::size_t row = 0; row < k; ++row) {
            const double upper = rotated[row][k], lower = rotated[row + 1][k];
            rotated[row][k] = finite(cosine[row] * upper + sine[row] * lower);
            rotated[row + 1][k] = finite(-sine[row] * upper + cosine[row] * lower);
        }
        GmresIteration record;
        record.iteration = k + 1;
        record.breakdown = breakdown;
        record.arnoldi_subdiagonal = hnext;
        const double upper = rotated[k][k], radius = finite(std::hypot(upper, rotated[k + 1][k]));
        bool defined = radius != 0;
        std::vector<double> y;
        if (defined) {
            cosine[k] = finite(upper / radius);
            sine[k] = finite(rotated[k + 1][k] / radius);
            rotated[k][k] = radius;
            rotated[k + 1][k] = 0;
            const double old = g[k];
            g[k] = finite(cosine[k] * old);
            g[k + 1] = finite(-sine[k] * old);
            record.estimated_relative = finite(std::abs(g[k + 1]) / denominator);
            defined = candidate_coefficients(rotated, g, k + 1, y);
        }
        if (!defined) {
            record.breakdown = true;
            record.verification_triggered = true;
            run.check_candidate(result, record, run.accepted.doubles(), denominator);
            record.accepted = record.recomputed_relative <= options.tolerance;
            result.history.push_back(record);
            run.snapshot();
            result.converged = record.accepted;
            result.status = record.accepted ? GmresStatus::success : GmresStatus::breakdown;
            result.message = record.accepted ? "Previously defined candidate verified at zero pivot"
                                             : "Zero Givens/triangular pivot";
            return;
        }
        e.copy(run.initial.doubles(), run.candidate.doubles());
        for (std::size_t j = 0; j <= k; ++j)
            e.axpy(y[j], run.v(j), run.candidate.doubles());
        run.check_candidate(result, record, run.candidate.doubles(), denominator);
        record.verification_triggered =
            breakdown || record.estimated_relative <= options.tolerance || k + 1 == cap;
        record.accepted =
            record.verification_triggered && record.recomputed_relative <= options.tolerance;
        e.copy(run.candidate.doubles(), run.accepted.doubles());
        result.history.push_back(record);
        run.snapshot();
        if (record.accepted) {
            result.status = GmresStatus::success;
            result.converged = true;
            return;
        }
        if (breakdown) {
            result.status = GmresStatus::breakdown;
            result.message = "Arnoldi breakdown candidate failed recomputed residual criterion";
            return;
        }
    }
    result.status = cap < n ? GmresStatus::iteration_limit : GmresStatus::nonconvergence;
    result.message = "Iteration cap reached without verified convergence";
}
}
struct CudaGmresContext::Impl {
    std::size_t n;
    int device;
    GmresOptions options;
    CudaGmresTransfers transfers, construction;
    std::unique_ptr<Run> run;
    Impl(const CsrMatrix &a, const CsrMatrix &p, const GmresOptions &o, int d)
        : n(a.rows()), device(d), options(o) {
        if (n == 0 || a.cols() != n || p.rows() != n || p.cols() != n || !canonical(a) ||
            !canonical(p) || !std::isfinite(o.tolerance) || o.tolerance <= 0 ||
            o.max_iterations > n)
            throw std::invalid_argument("Invalid persistent GPU GMRES matrix/options");
        index64(n);
        DeviceGuard guard(device);
        run.reset(new Run(a, p, o.max_iterations ? o.max_iterations : n, transfers));
        construction = transfers;
    }
    ~Impl() {
        int old = 0;
        const auto status = cudaGetDevice(&old);
        release(status);
        if (status == cudaSuccess) {
            release(cudaSetDevice(device));
            run.reset();
            release(cudaSetDevice(old));
        } else
            run.reset();
    }
};
namespace {
CudaGmresTransfers difference(const CudaGmresTransfers &after, const CudaGmresTransfers &before) {
    CudaGmresTransfers d;
    d.vector_uploads = after.vector_uploads - before.vector_uploads;
    d.vector_downloads = after.vector_downloads - before.vector_downloads;
    d.arnoldi_vector_uploads = after.arnoldi_vector_uploads - before.arnoldi_vector_uploads;
    d.arnoldi_vector_downloads = after.arnoldi_vector_downloads - before.arnoldi_vector_downloads;
    d.scalar_observations = after.scalar_observations - before.scalar_observations;
    d.dot_observations = after.dot_observations - before.dot_observations;
    d.norm_observations = after.norm_observations - before.norm_observations;
    d.finite_flag_observations = after.finite_flag_observations - before.finite_flag_observations;
    d.host_scalar_inputs = after.host_scalar_inputs - before.host_scalar_inputs;
    d.axpy_scalar_inputs = after.axpy_scalar_inputs - before.axpy_scalar_inputs;
    d.spmv_scalar_inputs = after.spmv_scalar_inputs - before.spmv_scalar_inputs;
    d.division_scalar_inputs = after.division_scalar_inputs - before.division_scalar_inputs;
    return d;
}
}
CudaGmresContext::CudaGmresContext(const CsrMatrix &a, const CsrMatrix &p,
                                   const GmresOptions &options, int device)
    : impl_(new Impl(a, p, options, device)) {
}
CudaGmresContext::~CudaGmresContext() = default;
CudaGmresTransfers CudaGmresContext::construction_transfers() const {
    return impl_->construction;
}
CudaGmresTransfers CudaGmresContext::total_transfers() const {
    return impl_->transfers;
}
std::size_t CudaGmresContext::matrix_uploads() const {
    return 2;
}
CudaGmresResult CudaGmresContext::solve(const std::vector<double> &rhs,
                                        const std::vector<double> &initial_guess) {
    CudaGmresResult out;
    auto &result = out.solver;
    auto &state = *impl_;
    if (rhs.size() != state.n || (!initial_guess.empty() && initial_guess.size() != state.n)) {
        result.message = "Invalid persistent GPU GMRES vector dimensions";
        return out;
    }
    for (double v : rhs)
        if (!std::isfinite(v)) {
            result.message = "RHS must be finite";
            return out;
        }
    for (double v : initial_guess)
        if (!std::isfinite(v)) {
            result.message = "Initial guess must be finite";
            return out;
        }
    const auto initial = initial_guess.empty() ? std::vector<double>(state.n, 0) : initial_guess;
    result.solution = initial;
    const auto before = state.transfers;
    DeviceGuard guard(state.device);
    Run &run = *state.run;
    run.basis_count = 0;
    run.snapshot_count = 0;
    run.e.in_arnoldi = false;
    try {
        bspai::solve(run, result, state.options, rhs, initial);
    } catch (const std::overflow_error &error) {
        result.status = GmresStatus::numerical_failure;
        result.message = error.what();
    }
    run.finish(result);
    out.transfers = difference(state.transfers, before);
    return out;
}

double cuda_safe_norm2(const std::vector<double> &x, int device) {
    for (double v : x)
        if (!std::isfinite(v))
            throw std::invalid_argument("Device norm input must be finite");
    index64(x.size());
    DeviceGuard guard(device);
    CudaGmresTransfers transfers;
    Engine e(x.size(), transfers);
    Memory input(product(x.size(), sizeof(double)));
    e.upload(input.p, x.data(), product(x.size(), sizeof(double)));
    return e.norm(input.doubles());
}
CudaGmresResult cuda_left_preconditioned_gmres(const CsrMatrix &a, const CsrMatrix &p,
                                               const std::vector<double> &rhs,
                                               const GmresOptions &options,
                                               const std::vector<double> &initial_guess,
                                               int device) {
    CudaGmresResult out;
    GmresResult &result = out.solver;
    const std::size_t n = a.rows();
    if (n == 0 || a.cols() != n || p.rows() != n || p.cols() != n || rhs.size() != n ||
        (!initial_guess.empty() && initial_guess.size() != n) ||
        !std::isfinite(options.tolerance) || options.tolerance <= 0 || options.max_iterations > n ||
        !canonical(a) || !canonical(p)) {
        result.message =
            "Invalid dimensions, canonical CSR, initial guess, tolerance, or iteration cap";
        return out;
    }
    for (double v : rhs)
        if (!std::isfinite(v)) {
            result.message = "RHS must be finite";
            return out;
        }
    for (double v : initial_guess)
        if (!std::isfinite(v)) {
            result.message = "Initial guess must be finite";
            return out;
        }
    index64(n);
    const auto initial = initial_guess.empty() ? std::vector<double>(n, 0) : initial_guess;
    result.solution = initial;
    DeviceGuard guard(device);
    Run run(a, p, options.max_iterations ? options.max_iterations : n, out.transfers);
    try {
        solve(run, result, options, rhs, initial);
    } catch (const std::overflow_error &error) {
        result.status = GmresStatus::numerical_failure;
        result.message = error.what();
    }
    run.finish(result);
    return out;
}
}
