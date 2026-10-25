#include "bspai/cuda_spai.hpp"
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
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>

namespace bspai {
namespace {
using WallClock = std::chrono::steady_clock;
double elapsed_ms(WallClock::time_point begin) {
    return std::chrono::duration<double, std::milli>(WallClock::now() - begin).count();
}
struct WallClockTotal {
    CudaSpaiTelemetry &telemetry;
    WallClock::time_point begin;
    bool finished = false;
    explicit WallClockTotal(CudaSpaiTelemetry &value) : telemetry(value), begin(WallClock::now()) {
    }
    void finish() {
        if (!finished) {
            telemetry.construction_wall_total_ms = elapsed_ms(begin);
            finished = true;
        }
    }
    ~WallClockTotal() {
        finish();
    }
};
void check(cudaError_t s, const char *op) {
    if (s != cudaSuccess)
        throw std::runtime_error(std::string(op) + ": " + cudaGetErrorString(s));
}
void check(cusolverStatus_t s, const char *op) {
    if (s != CUSOLVER_STATUS_SUCCESS)
        throw std::runtime_error(std::string(op) + ": cuSOLVER status " +
                                 std::to_string(static_cast<int>(s)));
}
void check(cublasStatus_t s, const char *op) {
    if (s != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string(op) + ": cuBLAS status " +
                                 std::to_string(static_cast<int>(s)));
}
void cleanup(cudaError_t s) {
    if (s != cudaSuccess)
        std::cerr << "GPU SPAI cleanup: " << cudaGetErrorString(s) << '\n';
}
std::size_t bytes(std::size_t n, std::size_t width) {
    if (n && width > std::numeric_limits<std::size_t>::max() / n)
        throw std::overflow_error("GPU SPAI size overflow");
    return n * width;
}
int dim(std::size_t n) {
    if (n > INT_MAX)
        throw std::invalid_argument("GPU QR requires dimensions <= INT_MAX");
    return static_cast<int>(n);
}
struct Guard {
    int old;
    explicit Guard(int device) {
        check(cudaGetDevice(&old), "SPAI get device");
        check(cudaSetDevice(device), "SPAI set device");
    }
    ~Guard() {
        cleanup(cudaSetDevice(old));
    }
};
struct Memory {
    void *p = nullptr;
    std::size_t capacity = 0;
    CudaSpaiTelemetry *telemetry = nullptr;
    explicit Memory(std::size_t size = 0, CudaSpaiTelemetry *t = nullptr) : telemetry(t) {
        allocate(size);
    }
    ~Memory() {
        if (p) {
            const auto begin = WallClock::now();
            cleanup(cudaFree(p));
            if (telemetry) {
                telemetry->free_wall_ms += elapsed_ms(begin);
                ++telemetry->device_free_calls;
                telemetry->device_freed_bytes += capacity;
                telemetry->current_device_bytes -= capacity;
                --telemetry->outstanding_device_allocations;
            }
        }
    }
    void allocate(std::size_t size) {
        if (size <= capacity)
            return;
        if (p) {
            const auto begin = WallClock::now();
            check(cudaFree(p), "SPAI resize free");
            if (telemetry) {
                telemetry->free_wall_ms += elapsed_ms(begin);
                ++telemetry->device_free_calls;
                ++telemetry->device_reallocation_calls;
                telemetry->device_freed_bytes += capacity;
                telemetry->current_device_bytes -= capacity;
                --telemetry->outstanding_device_allocations;
            }
            p = nullptr;
            capacity = 0;
        }
        if (!size)
            return;
        const auto begin = WallClock::now();
        check(cudaMalloc(&p, size), "SPAI allocation");
        capacity = size;
        if (telemetry) {
            ++telemetry->device_allocation_calls;
            ++telemetry->outstanding_device_allocations;
            telemetry->device_allocated_bytes += size;
            telemetry->current_device_bytes += size;
            telemetry->peak_device_bytes =
                std::max(telemetry->peak_device_bytes, telemetry->current_device_bytes);
            telemetry->max_single_allocation_bytes =
                std::max(telemetry->max_single_allocation_bytes, size);
            telemetry->allocation_wall_ms += elapsed_ms(begin);
        }
    }
    double *d() {
        return static_cast<double *>(p);
    }
    int *i() {
        return static_cast<int *>(p);
    }
    std::uint64_t *u() {
        return static_cast<std::uint64_t *>(p);
    }
    Memory(const Memory &) = delete;
    Memory &operator=(const Memory &) = delete;
};
struct SparseB {
    const std::uint64_t *offsets;
    const std::uint64_t *rows;
    const double *values;
    __device__ double at(std::size_t row, std::size_t col) const {
        auto first = offsets[col], last = offsets[col + 1];
        while (first < last) {
            const auto mid = first + (last - first) / 2;
            if (rows[mid] < row)
                first = mid + 1;
            else
                last = mid;
        }
        return first < offsets[col + 1] && rows[first] == row ? values[first] : 0;
    }
};
struct Stream {
    cudaStream_t p = nullptr;
    Stream() {
        check(cudaStreamCreateWithFlags(&p, cudaStreamNonBlocking), "SPAI create stream");
    }
    ~Stream() {
        cleanup(cudaStreamSynchronize(p));
        cleanup(cudaStreamDestroy(p));
    }
};
void record_upload(CudaSpaiTelemetry &t, std::size_t size) {
    ++t.uploads;
    t.upload_bytes += size;
    t.upload_min_bytes = t.upload_min_bytes ? std::min(t.upload_min_bytes, size) : size;
    t.upload_max_bytes = std::max(t.upload_max_bytes, size);
    if (size <= 256)
        ++t.upload_le_256_bytes;
    else if (size <= 4096)
        ++t.upload_le_4096_bytes;
    else
        ++t.upload_gt_4096_bytes;
}
void record_download(CudaSpaiTelemetry &t, std::size_t size) {
    ++t.downloads;
    t.download_bytes += size;
    t.download_min_bytes = t.download_min_bytes ? std::min(t.download_min_bytes, size) : size;
    t.download_max_bytes = std::max(t.download_max_bytes, size);
    if (size <= 256)
        ++t.download_le_256_bytes;
    else if (size <= 4096)
        ++t.download_le_4096_bytes;
    else
        ++t.download_gt_4096_bytes;
}
// Callers that keep their host buffers alive may enqueue several ordered copies and
// complete them with one stream wait.  This changes only host-side wait placement.
void enqueue_upload(Memory &target, const void *host, std::size_t size, Stream &s,
                    CudaSpaiTelemetry &t) {
    if (!size)
        return;
    check(cudaMemcpyAsync(target.p, host, size, cudaMemcpyHostToDevice, s.p), "SPAI H2D");
    record_upload(t, size);
}
void complete_uploads(Stream &s, CudaSpaiTelemetry &t) {
    check(cudaStreamSynchronize(s.p), "SPAI upload lifetime");
    ++t.upload_synchronizations;
}
void enqueue_download(void *host, const void *device, std::size_t size, Stream &s,
                      CudaSpaiTelemetry &t) {
    if (!size)
        return;
    check(cudaMemcpyAsync(host, device, size, cudaMemcpyDeviceToHost, s.p), "SPAI D2H");
    record_download(t, size);
}
void complete_downloads(Stream &s, CudaSpaiTelemetry &t) {
    check(cudaStreamSynchronize(s.p), "SPAI download lifetime");
    ++t.download_synchronizations;
}
void upload(Memory &target, const void *host, std::size_t size, Stream &s, CudaSpaiTelemetry &t) {
    if (!size)
        return;
    enqueue_upload(target, host, size, s, t);
    complete_uploads(s, t);
}
void download(void *host, const void *device, std::size_t size, Stream &s, CudaSpaiTelemetry &t) {
    if (!size)
        return;
    enqueue_download(host, device, size, s, t);
    complete_downloads(s, t);
}
void sync(Stream &s, CudaSpaiTelemetry *t = nullptr) {
    check(cudaStreamSynchronize(s.p), "SPAI stream completion");
    if (t)
        ++t->explicit_synchronizations;
}
__device__ void accumulate(double x, double &scale, double &sum) {
    const double a = fabs(x);
    if (a != 0) {
        if (scale < a) {
            const double r = scale / a;
            sum = 1 + sum * r * r;
            scale = a;
        } else {
            const double r = a / scale;
            sum += r * r;
        }
    }
}
__global__ void scale_matrix(const std::uint64_t *offsets, const std::uint64_t *columns,
                             const double *values, std::size_t n, double *d, double *b,
                             int *status) {
    const std::size_t row = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (row >= n)
        return;
    double maximum = 0, diagonal = 0;
    for (auto j = offsets[row]; j < offsets[row + 1]; ++j) {
        maximum = fmax(maximum, fabs(values[j]));
        if (columns[j] == row)
            diagonal = values[j];
    }
    status[row] = 0;
    if (diagonal == 0) {
        status[row] = 1;
        return;
    }
    const double scale = 1 / maximum;
    if (!isfinite(scale) || scale == 0) {
        status[row] = 2;
        return;
    }
    d[row] = scale;
    for (auto j = offsets[row]; j < offsets[row + 1]; ++j) {
        const double v = values[j] * scale;
        if (!isfinite(v) || v == 0) {
            status[row] = 3;
            return;
        }
        b[j] = v;
    }
}
__global__ void gather(SparseB b, std::size_t n, const std::uint64_t *shadow,
                       const std::uint64_t *pattern, int m, int q, std::size_t k, double *reduced,
                       double *rhs) {
    for (std::size_t at = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
         at < static_cast<std::size_t>(m) * q;
         at += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
        const std::size_t row = at % m, col = at / m;
        reduced[at] = b.at(shadow[row], pattern[col]);
    }
    for (std::size_t row = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
         row < static_cast<std::size_t>(m); row += static_cast<std::size_t>(blockDim.x) * gridDim.x)
        rhs[row] = shadow[row] == k ? 1 : 0;
}
__global__ void qr_guard(SparseB b, std::size_t n, const std::uint64_t *shadow,
                         const std::uint64_t *pattern, int m, int q, const double *r,
                         const double *tau, const double *rhs, int *status, double *diagnostics) {
    int code = static_cast<int>(SpaiStatus::success);
    double minimum = INFINITY, inf = 0;
    for (int row = 0; row < m; ++row) {
        double sum = 0;
        for (int col = 0; col < q; ++col) {
            sum += fabs(b.at(shadow[row], pattern[col]));
            if (!isfinite(r[row + static_cast<std::size_t>(col) * m]))
                code = static_cast<int>(SpaiStatus::numerical_failure);
        }
        inf = fmax(inf, sum);
        if (!isfinite(rhs[row]))
            code = static_cast<int>(SpaiStatus::numerical_failure);
    }
    for (int j = 0; j < q; ++j) {
        const double pivot = r[j + static_cast<std::size_t>(j) * m];
        if (!isfinite(pivot) || !isfinite(tau[j]))
            code = static_cast<int>(SpaiStatus::numerical_failure);
        else if (pivot == 0 && code == static_cast<int>(SpaiStatus::success))
            code = static_cast<int>(SpaiStatus::rank_deficient);
        minimum = fmin(minimum, fabs(pivot));
    }
    diagnostics[0] = minimum / inf;
    *status = code;
}
__global__ void fitted_residual(SparseB b, std::size_t n, const std::uint64_t *shadow,
                                const std::uint64_t *pattern, int m, int q, std::size_t k,
                                const double *coefficients, double *residual, double *diagnostics,
                                int *status) {
    double scale = 0, sum = 1;
    for (int j = 0; j < q; ++j)
        if (!isfinite(coefficients[j])) {
            *status = static_cast<int>(SpaiStatus::numerical_failure);
            return;
        }
    for (int i = 0; i < m; ++i) {
        const auto row = shadow[i];
        double value = 0;
        for (int j = 0; j < q; ++j)
            value += b.at(row, pattern[j]) * coefficients[j];
        value -= row == k ? 1 : 0;
        if (!isfinite(value)) {
            *status = static_cast<int>(SpaiStatus::numerical_failure);
            return;
        }
        residual[i] = value;
        accumulate(value, scale, sum);
    }
    diagnostics[1] = scale == 0 ? 0 : scale * sqrt(sum);
    diagnostics[2] = diagnostics[1];
    *status = isfinite(diagnostics[1]) && isfinite(diagnostics[2])
                  ? static_cast<int>(SpaiStatus::success)
                  : static_cast<int>(SpaiStatus::numerical_failure);
}
__device__ int score_value(double rsq, double dot, double denominator, double &score,
                           int &clamped) {
    int code = static_cast<int>(SpaiStatus::success);
    clamped = 0;
    score = 0;
    if (!isfinite(rsq) || !isfinite(dot) || !isfinite(denominator))
        code = static_cast<int>(SpaiStatus::numerical_failure);
    else if (denominator == 0)
        code = static_cast<int>(SpaiStatus::zero_denominator);
    else if (rsq < 0 || denominator < 0)
        code = static_cast<int>(SpaiStatus::invalid_score);
    if (code == static_cast<int>(SpaiStatus::success)) {
        const double quotient = dot * dot / denominator;
        double delta = rsq - quotient;
#ifndef BSPAI_SPAI_SCORE_UNIT_ROUNDOFF
#define BSPAI_SPAI_SCORE_UNIT_ROUNDOFF 1.1102230246251565e-16
#endif
        const double bound = 32 * BSPAI_SPAI_SCORE_UNIT_ROUNDOFF * (rsq + fabs(quotient));
        if (!isfinite(quotient) || !isfinite(delta) || !isfinite(bound))
            code = static_cast<int>(SpaiStatus::numerical_failure);
        else if (delta < 0) {
            if (-delta > bound)
                code = static_cast<int>(SpaiStatus::invalid_score);
            else {
                delta = 0;
                clamped = 1;
            }
        }
        if (code == static_cast<int>(SpaiStatus::success))
            score = sqrt(delta);
    }
    return code;
}
__global__ void single_score(double rsq, double dot, double denominator, double *score, int *status,
                             int *clamped) {
    *status = score_value(rsq, dot, denominator, *score, *clamped);
}
// One record is the exact contiguous representation of the three prior score
// outputs.  It removes two D2H API calls per growth decision without changing
// score bits, status ordering, clamp semantics, or transferred payload bytes.
struct ScoreWire {
    double score;
    int status;
    int clamped;
};
static_assert(sizeof(ScoreWire) == sizeof(double) + 2 * sizeof(int),
              "ScoreWire must not add padding");
__global__ void scores(SparseB b, std::size_t n, const std::uint64_t *shadow, int m,
                       const std::uint64_t *candidates, std::size_t count, const double *residual,
                       const double *diagnostics, const int *fit_status, ScoreWire *output) {
    const std::size_t index = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (index >= count)
        return;
    if (*fit_status != static_cast<int>(SpaiStatus::success)) {
        output[index].status = *fit_status;
        output[index].clamped = 0;
        output[index].score = 0;
        return;
    }
    const auto j = candidates[index];
    double dot = 0, denominator = 0;
    for (int i = 0; i < m; ++i) {
        const double value = b.at(shadow[i], j);
        dot += residual[i] * value;
        denominator += value * value;
    }
    output[index].status = score_value(diagnostics[1] * diagnostics[1], dot, denominator,
                                       output[index].score, output[index].clamped);
}
// Packed N CSC column k becomes P CSR row k; multiply by d[j], not d[k].
__global__ void assemble_p(const double *coefficients, const std::uint64_t *rows, const double *d,
                           std::size_t count, double *p, int *flag) {
    const std::size_t at = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (at >= count)
        return;
    const double coefficient = coefficients[at], value = coefficient * d[rows[at]];
    p[at] = value;
    if (!isfinite(value) || (coefficient != 0 && value == 0))
        atomicExch(flag, 1);
}
struct Handles {
    cusolverDnHandle_t solver = nullptr;
    cublasHandle_t blas = nullptr;
    ~Handles() {
        if (blas && cublasDestroy(blas) != CUBLAS_STATUS_SUCCESS)
            std::cerr << "SPAI BLAS cleanup failed\n";
        if (solver && cusolverDnDestroy(solver) != CUSOLVER_STATUS_SUCCESS)
            std::cerr << "SPAI solver cleanup failed\n";
    }
};
struct Slot {
    Stream stream;
    Handles handles;
    cusolverDnHandle_t &solver;
    cublasHandle_t &blas;
    Memory reduced, rhs, tau, work, pattern, shadow, candidates, residual, diagnostics, infos,
        score_results;
    std::size_t column = 0;
    SpaiFit fit;
    int m = 0, q = 0, lwork = 0;
    bool qr_ok = false;
    Slot(CudaSpaiTelemetry &t)
        : solver(handles.solver), blas(handles.blas), reduced(0, &t), rhs(0, &t), tau(0, &t),
          work(0, &t), pattern(0, &t), shadow(0, &t), candidates(0, &t), residual(0, &t),
          diagnostics(3 * sizeof(double), &t), infos(3 * sizeof(int), &t), score_results(0, &t) {
        check(cusolverDnCreate(&solver), "SPAI solver create");
        check(cublasCreate(&blas), "SPAI BLAS create");
        check(cusolverDnSetStream(solver, stream.p), "SPAI solver stream");
        check(cublasSetStream(blas, stream.p), "SPAI BLAS stream");
        check(cusolverDnSetDeterministicMode(solver, CUSOLVER_DETERMINISTIC_RESULTS),
              "SPAI deterministic QR");
        cusolverDeterministicMode_t deterministic;
        check(cusolverDnGetDeterministicMode(solver, &deterministic),
              "SPAI query deterministic QR");
        if (deterministic != CUSOLVER_DETERMINISTIC_RESULTS)
            throw std::runtime_error("Deterministic QR mode not retained");
        check(cusolverDnSetMathMode(solver, CUSOLVER_DEFAULT_MATH), "SPAI solver native math");
        cusolverMathMode_t math;
        check(cusolverDnGetMathMode(solver, &math), "SPAI query solver math");
        if (math != CUSOLVER_DEFAULT_MATH)
            throw std::runtime_error("Native solver math not retained");
        check(cublasSetPointerMode(blas, CUBLAS_POINTER_MODE_HOST), "SPAI scalar mode");
        check(cublasSetMathMode(blas, CUBLAS_PEDANTIC_MATH), "SPAI native BLAS math");
        cublasMath_t actual;
        check(cublasGetMathMode(blas, &actual), "SPAI query BLAS math");
        if (actual != CUBLAS_PEDANTIC_MATH)
            throw std::runtime_error("Native BLAS math not retained");
        check(cublasSetAtomicsMode(blas, CUBLAS_ATOMICS_NOT_ALLOWED), "SPAI atomics mode");
    }
    ~Slot() {
        cleanup(cudaStreamSynchronize(stream.p));
    }
};
bool canonical(const CsrMatrix &a) {
    if (a.rows() == std::numeric_limits<std::size_t>::max() ||
        a.row_offsets().size() != a.rows() + 1 || a.row_offsets().front() != 0 ||
        a.row_offsets().back() != a.values().size() ||
        a.column_indices().size() != a.values().size())
        return false;
    for (std::size_t row = 0; row < a.rows(); ++row) {
        auto first = a.row_offsets()[row], last = a.row_offsets()[row + 1];
        if (first > last || last > a.values().size())
            return false;
        for (auto j = first; j < last; ++j)
            if (a.column_indices()[j] >= a.cols() || !std::isfinite(a.values()[j]) ||
                a.values()[j] == 0 ||
                (j > first && a.column_indices()[j - 1] >= a.column_indices()[j]))
                return false;
    }
    return true;
}
} // namespace

ScoreResult cuda_evaluate_spai_score(double residual_squared, double dot,
                                     double denominator_squared, int device) {
    Guard guard(device);
    Stream stream;
    Memory score(sizeof(double)), codes(2 * sizeof(int));
    CudaSpaiTelemetry telemetry;
    single_score<<<1, 1, 0, stream.p>>>(residual_squared, dot, denominator_squared, score.d(),
                                        codes.i(), codes.i() + 1);
    check(cudaGetLastError(), "SPAI score helper kernel");
    ScoreResult result;
    int status[2];
    download(&result.score, score.p, sizeof(double), stream, telemetry);
    download(status, codes.p, sizeof(status), stream, telemetry);
    result.status = static_cast<SpaiStatus>(status[0]);
    result.clamped = status[1] != 0;
    return result;
}

CudaSpaiResult cuda_construct_spai(const CsrMatrix &a, const SpaiOptions &options, int device) {
    const auto n = a.rows();
    CudaSpaiResult out(n);
    auto &result = out.construction;
    auto &telemetry = out.telemetry;
    result.status = SpaiStatus::invalid_input;
    WallClockTotal total_wall_clock(telemetry);
    if (!n || a.cols() != n || !canonical(a) || !std::isfinite(options.epsilon) ||
        options.epsilon <= 0 || n > INT_MAX) {
        result.message =
            "Require canonical square matrix, positive finite epsilon and int QR dimensions";
        total_wall_clock.finish();
        return out;
    }
    auto build = [&]() {
        Guard guard(device);
        Stream main;
        const auto matrix_setup_begin = WallClock::now();
        const auto nnz = a.values().size();
        Memory offsets(bytes(n + 1, sizeof(std::uint64_t)), &telemetry),
            indices(bytes(nnz, sizeof(std::uint64_t)), &telemetry),
            values(bytes(nnz, sizeof(double)), &telemetry), d(bytes(n, sizeof(double)), &telemetry),
            b(bytes(nnz, sizeof(double)), &telemetry), status(bytes(n, sizeof(int)), &telemetry);
        telemetry.nnz_b = nnz;
        telemetry.sparse_b_bytes = offsets.capacity + indices.capacity + b.capacity;
        std::vector<std::uint64_t> ro(a.row_offsets().begin(), a.row_offsets().end()),
            ci(a.column_indices().begin(), a.column_indices().end());
        upload(offsets, ro.data(), offsets.capacity, main, telemetry);
        upload(indices, ci.data(), indices.capacity, main, telemetry);
        upload(values, a.values().data(), values.capacity, main, telemetry);
        scale_matrix<<<static_cast<unsigned>((n + 255) / 256), 256, 0, main.p>>>(
            offsets.u(), indices.u(), values.d(), n, d.d(), b.d(), status.i());
        check(cudaGetLastError(), "SPAI scaling kernel");
        std::vector<int> scale_status(n);
        download(scale_status.data(), status.p, bytes(n, sizeof(int)), main, telemetry);
        telemetry.gpu_scaling = true;
        for (std::size_t k = 0; k < n; ++k)
            if (scale_status[k]) {
                result.message = "GPU scaling/diagonal range failure at row " + std::to_string(k);
                return;
            }
        result.scaling.resize(n);
        std::vector<double> host_b(nnz);
        download(result.scaling.data(), d.p, bytes(n, sizeof(double)), main, telemetry);
        download(host_b.data(), b.p, bytes(nnz, sizeof(double)), main, telemetry);
        std::vector<Entry> entries;
        entries.reserve(nnz);
        for (std::size_t col = 0; col < n; ++col)
            for (auto at = ro[col]; at < ro[col + 1]; ++at)
                entries.push_back(Entry{static_cast<std::size_t>(ci[at]), col, host_b[at]});
        result.scaled_transpose = CsrMatrix::from_entries(n, n, std::move(entries));
        result.columns.resize(n);
        telemetry.construction_wall_matrix_setup_ms = elapsed_ms(matrix_setup_begin);
        const auto adaptive_fits_begin = WallClock::now();
        SparseB device_b{offsets.u(), indices.u(), b.d()};
        std::vector<std::unique_ptr<Slot>> slots;
        for (std::size_t j = 0; j < std::min<std::size_t>(n, 8); ++j)
            slots.emplace_back(new Slot(telemetry));
        telemetry.streams_created = slots.size();
        telemetry.slot_memory.resize(slots.size());
        std::vector<std::vector<std::size_t>> patterns(n);
        for (std::size_t k = 0; k < n; ++k)
            patterns[k] = {k};
        std::vector<bool> done(n, false);
        std::size_t remaining = n;
        while (remaining) {
            std::vector<std::size_t> active;
            for (std::size_t k = 0; k < n; ++k)
                if (!done[k])
                    active.push_back(k);
            for (std::size_t begin = 0; begin < active.size(); begin += slots.size()) {
                const auto count = std::min(slots.size(), active.size() - begin);
                telemetry.max_fit_batch = std::max(telemetry.max_fit_batch, count);
                if (count > 1)
                    ++telemetry.parallel_fit_batches;
                std::size_t submitted = 0;
                // Allocate/query every slot before launching any QR in this batch.
                for (std::size_t s = 0; s < count; ++s) {
                    auto &slot = *slots[s];
                    slot.column = active[begin + s];
                    auto &column = result.columns[slot.column];
                    slot.fit = SpaiFit{};
                    slot.fit.pattern = patterns[slot.column];
                    slot.qr_ok = false;
                    column.attempted_pattern = slot.fit.pattern;
                    ++column.fit_attempts;
                    if (column.fits)
                        ++column.growth_attempts;
                    for (auto col : slot.fit.pattern)
                        for (auto at = ro[col]; at < ro[col + 1]; ++at)
                            slot.fit.shadow.push_back(static_cast<std::size_t>(ci[at]));
                    std::sort(slot.fit.shadow.begin(), slot.fit.shadow.end());
                    slot.fit.shadow.erase(
                        std::unique(slot.fit.shadow.begin(), slot.fit.shadow.end()),
                        slot.fit.shadow.end());
                    slot.m = dim(slot.fit.shadow.size());
                    slot.q = dim(slot.fit.pattern.size());
                    if (slot.m < slot.q) {
                        column.status = SpaiStatus::rank_deficient;
                        column.message = "Reduced shadow has fewer rows than columns";
                        done[slot.column] = true;
                        --remaining;
                        continue;
                    }
                    for (auto row : slot.fit.shadow)
                        for (auto at = result.scaled_transpose.row_offsets()[row];
                             at < result.scaled_transpose.row_offsets()[row + 1]; ++at)
                            slot.fit.candidates.push_back(
                                result.scaled_transpose.column_indices()[at]);
                    for (auto at = result.scaled_transpose.row_offsets()[slot.column];
                         at < result.scaled_transpose.row_offsets()[slot.column + 1]; ++at)
                        slot.fit.candidates.push_back(result.scaled_transpose.column_indices()[at]);
                    std::sort(slot.fit.candidates.begin(), slot.fit.candidates.end());
                    slot.fit.candidates.erase(
                        std::unique(slot.fit.candidates.begin(), slot.fit.candidates.end()),
                        slot.fit.candidates.end());
                    std::vector<std::size_t> filtered;
                    std::set_difference(slot.fit.candidates.begin(), slot.fit.candidates.end(),
                                        slot.fit.pattern.begin(), slot.fit.pattern.end(),
                                        std::back_inserter(filtered));
                    slot.fit.candidates.swap(filtered);
                    const auto m = slot.fit.shadow.size(), q = slot.fit.pattern.size(),
                               c = slot.fit.candidates.size();
                    slot.reduced.allocate(bytes(bytes(m, q), sizeof(double)));
                    slot.rhs.allocate(bytes(m, sizeof(double)));
                    slot.tau.allocate(bytes(q, sizeof(double)));
                    slot.pattern.allocate(bytes(q, sizeof(std::uint64_t)));
                    slot.shadow.allocate(bytes(m, sizeof(std::uint64_t)));
                    slot.candidates.allocate(bytes(c, sizeof(std::uint64_t)));
                    slot.residual.allocate(bytes(m, sizeof(double)));
                    slot.score_results.allocate(bytes(c, sizeof(ScoreWire)));
                    telemetry.max_reduced_rows = std::max(telemetry.max_reduced_rows, m);
                    telemetry.max_pattern_columns = std::max(telemetry.max_pattern_columns, q);
                    telemetry.max_reduced_elements =
                        std::max(telemetry.max_reduced_elements, bytes(m, q));
                    int qr_work = 0, q_work = 0;
                    check(cusolverDnDgeqrf_bufferSize(slot.solver, slot.m, slot.q, slot.reduced.d(),
                                                      slot.m, &qr_work),
                          "SPAI QR workspace");
                    check(cusolverDnDormqr_bufferSize(slot.solver, CUBLAS_SIDE_LEFT, CUBLAS_OP_T,
                                                      slot.m, 1, slot.q, slot.reduced.d(), slot.m,
                                                      slot.tau.d(), slot.rhs.d(), slot.m, &q_work),
                          "SPAI Q workspace");
                    if (qr_work < 0 || q_work < 0)
                        throw std::runtime_error("Negative solver workspace");
                    slot.lwork = std::max(qr_work, q_work);
                    slot.work.allocate(bytes(static_cast<std::size_t>(slot.lwork), sizeof(double)));
                    slot.qr_ok = true;
                    telemetry.max_qr_workspace_bytes =
                        std::max(telemetry.max_qr_workspace_bytes, slot.work.capacity);
                    telemetry.max_slot_bytes =
                        std::max(telemetry.max_slot_bytes,
                                 slot.reduced.capacity + slot.rhs.capacity + slot.tau.capacity +
                                     slot.work.capacity + slot.pattern.capacity +
                                     slot.shadow.capacity + slot.candidates.capacity +
                                     slot.residual.capacity + slot.diagnostics.capacity +
                                     slot.infos.capacity + slot.score_results.capacity);
                    auto &memory = telemetry.slot_memory[s];
                    memory.max_rows = std::max(memory.max_rows, m);
                    memory.max_columns = std::max(memory.max_columns, q);
                    memory.max_candidates = std::max(memory.max_candidates, c);
                    memory.reduced_capacity_bytes = slot.reduced.capacity;
                    memory.workspace_capacity_bytes = slot.work.capacity;
                    memory.total_capacity_bytes =
                        slot.reduced.capacity + slot.rhs.capacity + slot.tau.capacity +
                        slot.work.capacity + slot.pattern.capacity + slot.shadow.capacity +
                        slot.candidates.capacity + slot.residual.capacity +
                        slot.diagnostics.capacity + slot.infos.capacity +
                        slot.score_results.capacity;
                    // size_t is statically checked below to match uint64 metadata.
                    // The metadata vectors remain immutable until this single wait completes.
                    enqueue_upload(slot.pattern, slot.fit.pattern.data(),
                                   bytes(slot.fit.pattern.size(), sizeof(std::uint64_t)),
                                   slot.stream, telemetry);
                    enqueue_upload(slot.shadow, slot.fit.shadow.data(),
                                   bytes(slot.fit.shadow.size(), sizeof(std::uint64_t)),
                                   slot.stream, telemetry);
                    enqueue_upload(slot.candidates, slot.fit.candidates.data(),
                                   bytes(slot.fit.candidates.size(), sizeof(std::uint64_t)),
                                   slot.stream, telemetry);
                    complete_uploads(slot.stream, telemetry);
                }
                for (std::size_t s = 0; s < count; ++s) {
                    auto &slot = *slots[s];
                    if (!slot.qr_ok)
                        continue;
                    gather<<<32, 256, 0, slot.stream.p>>>(
                        device_b, n, slot.shadow.u(), slot.pattern.u(), slot.m, slot.q, slot.column,
                        slot.reduced.d(), slot.rhs.d());
                    check(cudaGetLastError(), "SPAI reduced gather");
                    check(cusolverDnDgeqrf(slot.solver, slot.m, slot.q, slot.reduced.d(), slot.m,
                                           slot.tau.d(), slot.work.d(), slot.lwork, slot.infos.i()),
                          "SPAI GPU QR");
                    check(cusolverDnDormqr(slot.solver, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, slot.m, 1,
                                           slot.q, slot.reduced.d(), slot.m, slot.tau.d(),
                                           slot.rhs.d(), slot.m, slot.work.d(), slot.lwork,
                                           slot.infos.i() + 1),
                          "SPAI GPU Q transpose");
                    qr_guard<<<1, 1, 0, slot.stream.p>>>(
                        device_b, n, slot.shadow.u(), slot.pattern.u(), slot.m, slot.q,
                        slot.reduced.d(), slot.tau.d(), slot.rhs.d(), slot.infos.i() + 2,
                        slot.diagnostics.d());
                    check(cudaGetLastError(), "SPAI QR guards");
                    ++submitted;
                    ++telemetry.qr_submissions;
                }
                telemetry.max_qr_submissions_before_collection =
                    std::max(telemetry.max_qr_submissions_before_collection, submitted);
                for (std::size_t s = 0; s < count; ++s) {
                    auto &slot = *slots[s];
                    if (!slot.qr_ok)
                        continue;
                    int info[3];
                    download(info, slot.infos.p, sizeof(info), slot.stream, telemetry);
                    if (info[0] || info[1])
                        throw std::runtime_error("cuSOLVER devInfo rejected QR/Q application");
                    if (info[2] != static_cast<int>(SpaiStatus::success)) {
                        auto &column = result.columns[slot.column];
                        column.status = static_cast<SpaiStatus>(info[2]);
                        column.message = "GPU QR exact-pivot/nonfinite guard failed";
                        slot.qr_ok = false;
                        done[slot.column] = true;
                        --remaining;
                    }
                }
                for (std::size_t s = 0; s < count; ++s) {
                    auto &slot = *slots[s];
                    if (!slot.qr_ok)
                        continue;
                    check(cublasDtrsv(slot.blas, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
                                      CUBLAS_DIAG_NON_UNIT, slot.q, slot.reduced.d(), slot.m,
                                      slot.rhs.d(), 1),
                          "SPAI GPU triangular solve");
                    fitted_residual<<<1, 1, 0, slot.stream.p>>>(
                        device_b, n, slot.shadow.u(), slot.pattern.u(), slot.m, slot.q, slot.column,
                        slot.rhs.d(), slot.residual.d(), slot.diagnostics.d(), slot.infos.i() + 2);
                    check(cudaGetLastError(), "SPAI fitted residual");
                    if (!slot.fit.candidates.empty()) {
                        scores<<<static_cast<unsigned>((slot.fit.candidates.size() + 255) / 256),
                                 256, 0, slot.stream.p>>>(
                            device_b, n, slot.shadow.u(), slot.m, slot.candidates.u(),
                            slot.fit.candidates.size(), slot.residual.d(), slot.diagnostics.d(),
                            slot.infos.i() + 2, static_cast<ScoreWire *>(slot.score_results.p));
                        check(cudaGetLastError(), "SPAI restricted scores");
                        telemetry.gpu_score_evaluations += slot.fit.candidates.size();
                    }
                }
                for (std::size_t s = 0; s < count; ++s) {
                    auto &slot = *slots[s];
                    if (!slot.qr_ok)
                        continue;
                    auto &column = result.columns[slot.column];
                    auto &fit = slot.fit;
                    int fit_status = 0;
                    double diagnostics[3];
                    fit.coefficients.resize(slot.q);
                    enqueue_download(fit.coefficients.data(), slot.rhs.p,
                                     bytes(slot.q, sizeof(double)), slot.stream, telemetry);
                    enqueue_download(diagnostics, slot.diagnostics.p, sizeof(diagnostics),
                                     slot.stream, telemetry);
                    enqueue_download(&fit_status, slot.infos.i() + 2, sizeof(int), slot.stream,
                                     telemetry);
                    complete_downloads(slot.stream, telemetry);
                    if (fit_status != static_cast<int>(SpaiStatus::success)) {
                        column.status = static_cast<SpaiStatus>(fit_status);
                        column.message = "GPU coefficients/residual nonfinite";
                        done[slot.column] = true;
                        --remaining;
                        continue;
                    }
                    fit.qr_pivot_ratio = diagnostics[0];
                    fit.residual = diagnostics[1];
                    fit.full_residual = diagnostics[2];
                    if (column.fits)
                        ++column.growths;
                    ++column.fits;
                    column.pattern = fit.pattern;
                    column.coefficients = fit.coefficients;
                    column.reduced_residual = fit.residual;
                    column.full_residual = fit.full_residual;
                    column.qr_pivot_ratio = fit.qr_pivot_ratio;
                    const auto candidates = fit.candidates;
                    fit.candidates.clear();
                    column.history.push_back(fit);
                    auto &record = column.history.back();
                    if (fit.residual <= options.epsilon) {
                        column.status = SpaiStatus::success;
                        done[slot.column] = true;
                        --remaining;
                        sync(slot.stream, &telemetry);
                        continue;
                    }
                    if (column.growths == options.alpha) {
                        column.status = SpaiStatus::iteration_limit;
                        column.message =
                            "Requested accuracy not reached within growth/refit budget";
                        done[slot.column] = true;
                        --remaining;
                        sync(slot.stream, &telemetry);
                        continue;
                    }
                    record.candidates = candidates;
                    if (candidates.empty()) {
                        column.status = SpaiStatus::empty_candidates;
                        column.message = "No candidate remains before convergence";
                        done[slot.column] = true;
                        --remaining;
                        sync(slot.stream, &telemetry);
                        continue;
                    }
                    std::vector<ScoreWire> score_results(candidates.size());
                    enqueue_download(score_results.data(), slot.score_results.p,
                                     bytes(candidates.size(), sizeof(ScoreWire)), slot.stream,
                                     telemetry);
                    complete_downloads(slot.stream, telemetry);
                    bool failed = false;
                    for (std::size_t j = 0; j < candidates.size(); ++j) {
                        const auto &score = score_results[j];
                        if (score.status != static_cast<int>(SpaiStatus::success)) {
                            column.status = static_cast<SpaiStatus>(score.status);
                            column.message = "GPU candidate scoring failed at index " +
                                             std::to_string(candidates[j]);
                            failed = true;
                            break;
                        }
                        if (score.clamped)
                            ++column.score_clamps;
                        record.scores.push_back(score.score);
                    }
                    if (failed) {
                        done[slot.column] = true;
                        --remaining;
                        continue;
                    }
#ifndef BSPAI_SPAI_SCORE_ZERO
#define BSPAI_SPAI_SCORE_ZERO 0.0
#endif
                    record.mean_score = std::accumulate(record.scores.begin(), record.scores.end(),
                                                        BSPAI_SPAI_SCORE_ZERO) /
                                        static_cast<double>(record.scores.size());
                    if (!std::isfinite(record.mean_score)) {
                        column.status = SpaiStatus::numerical_failure;
                        column.message = "Candidate mean score is nonfinite";
                        done[slot.column] = true;
                        --remaining;
                        continue;
                    }
                    std::vector<std::size_t> order(candidates.size());
                    std::iota(order.begin(), order.end(), 0);
                    std::sort(order.begin(), order.end(), [&](std::size_t x, std::size_t y) {
                        return record.scores[x] < record.scores[y] ||
                               (record.scores[x] == record.scores[y] &&
                                candidates[x] < candidates[y]);
                    });
                    for (auto j : order) {
                        if (record.selected.size() == options.beta)
                            break;
                        if (record.scores[j] <= record.mean_score)
                            record.selected.push_back(candidates[j]);
                    }
                    if (record.selected.empty()) {
                        column.status = SpaiStatus::no_growth;
                        column.message = "No acceptable growth permitted before convergence";
                        done[slot.column] = true;
                        --remaining;
                        continue;
                    }
                    auto &pattern = patterns[slot.column];
                    pattern.insert(pattern.end(), record.selected.begin(), record.selected.end());
                    std::sort(pattern.begin(), pattern.end());
                }
            }
        }
        for (auto &slot : slots)
            sync(slot->stream, &telemetry);
        result.status = SpaiStatus::success;
        for (std::size_t k = 0; k < n; ++k)
            if (result.columns[k].status != SpaiStatus::success) {
                result.status = result.columns[k].status;
                result.message = "Column " + std::to_string(k) + ": " + result.columns[k].message;
                break;
            }
        telemetry.construction_wall_adaptive_fits_ms = elapsed_ms(adaptive_fits_begin);
        const auto packing_assembly_begin = WallClock::now();
        // Host packs metadata and previously downloaded fitted coefficients; scaling remains GPU numerical work.
        std::vector<std::uint64_t> packed_rows;
        std::vector<double> coefficients;
        std::vector<std::size_t> packed_offsets(n + 1, 0);
        for (std::size_t k = 0; k < n; ++k) {
            const auto &column = result.columns[k];
            for (std::size_t j = 0; j < column.pattern.size(); ++j) {
                packed_rows.push_back(column.pattern[j]);
                coefficients.push_back(column.coefficients[j]);
            }
            packed_offsets[k + 1] = coefficients.size();
        }
        const auto count = coefficients.size();
        Memory packed_indices(bytes(count, sizeof(std::uint64_t)), &telemetry),
            packed_n(bytes(count, sizeof(double)), &telemetry),
            packed_p(bytes(count, sizeof(double)), &telemetry);
        telemetry.packed_output_bytes =
            packed_indices.capacity + packed_n.capacity + packed_p.capacity;
        upload(packed_indices, packed_rows.data(), packed_indices.capacity, main, telemetry);
        upload(packed_n, coefficients.data(), packed_n.capacity, main, telemetry);
        check(cudaMemsetAsync(status.p, 0, sizeof(int), main.p), "SPAI assembly guard clear");
        if (count)
            assemble_p<<<static_cast<unsigned>((count + 255) / 256), 256, 0, main.p>>>(
                packed_n.d(), packed_indices.u(), d.d(), count, packed_p.d(), status.i());
        check(cudaGetLastError(), "SPAI GPU packed P assembly");
        telemetry.gpu_preconditioner_assembly = true;
        int assembly_status = 0;
        std::vector<double> host_p(count);
        download(&assembly_status, status.p, sizeof(int), main, telemetry);
        download(host_p.data(), packed_p.p, bytes(count, sizeof(double)), main, telemetry);
        if (assembly_status) {
            result.status = SpaiStatus::numerical_failure;
            result.message = "GPU preconditioner assembly outside binary64 range";
            return;
        }
        std::vector<Entry> n_entries, p_entries;
        n_entries.reserve(count);
        p_entries.reserve(count);
        for (std::size_t k = 0; k < n; ++k)
            for (auto at = packed_offsets[k]; at < packed_offsets[k + 1]; ++at) {
                const auto j = static_cast<std::size_t>(packed_rows[at]);
                n_entries.push_back(Entry{j, k, coefficients[at]});
                p_entries.push_back(Entry{k, j, host_p[at]});
            }
        result.inverse = CsrMatrix::from_entries(n, n, std::move(n_entries));
        result.preconditioner = CsrMatrix::from_entries(n, n, std::move(p_entries));
        telemetry.nnz_n = result.inverse.values().size();
        telemetry.nnz_p = result.preconditioner.values().size();
        telemetry.construction_wall_packing_assembly_ms = elapsed_ms(packing_assembly_begin);
    };
    build();
    total_wall_clock.finish();
    return out;
}
static_assert(sizeof(std::size_t) == sizeof(std::uint64_t),
              "GPU SPAI metadata requires 64-bit size_t");
}
