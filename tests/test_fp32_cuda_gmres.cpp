#include "bspai/fp32_cuda_gmres.hpp"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>
namespace {
void check(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
void close(const std::vector<float>& got, const std::vector<float>& expected, float tolerance) {
    check(got.size() == expected.size(), "FP32 CUDA vector size");
    for (std::size_t i = 0; i < got.size(); ++i) check(std::fabs(got[i] - expected[i]) <= tolerance, "FP32 CUDA solution mismatch");
}
void finite_history(const std::vector<float>& history, std::size_t iterations, const char* message) {
    check(history.size() == iterations, message);
    for (float value : history) check(std::isfinite(value), message);
}
std::uint32_t bits(float value) {
    std::uint32_t result = 0;
    std::memcpy(&result, &value, sizeof result);
    return result;
}
void identical_bits(const std::vector<float>& got, const std::vector<float>& expected,
                    const char* message) {
    check(got.size() == expected.size(), message);
    for (std::size_t i = 0; i < got.size(); ++i) check(bits(got[i]) == bits(expected[i]), message);
}
void identical_result(const bspai::Fp32GmresResult& got,
                      const bspai::Fp32GmresResult& expected, const char* message) {
    check(got.status == expected.status && got.converged == expected.converged &&
              got.message == expected.message && got.iterations == expected.iterations,
          message);
    identical_bits(got.solution, expected.solution, message);
    check(bits(got.final_original_residual) == bits(expected.final_original_residual) &&
              bits(got.final_preconditioned_relative) ==
                  bits(expected.final_preconditioned_relative),
          message);
    identical_bits(got.hessenberg_residual_history, expected.hessenberg_residual_history,
                   message);
    identical_bits(got.original_residual_history, expected.original_residual_history, message);
    identical_bits(got.preconditioned_relative_history,
                   expected.preconditioned_relative_history, message);
}
float independent_relative_residual(const bspai::Fp32CsrMatrix& a,
                                    const std::vector<float>& rhs,
                                    const std::vector<float>& solution) {
    double residual2 = 0, rhs2 = 0;
    for (std::size_t row = 0; row < a.rows(); ++row) {
        double ax = 0;
        for (std::size_t q = a.row_offsets()[row]; q < a.row_offsets()[row + 1]; ++q)
            ax += double(a.values()[q]) * double(solution[a.column_indices()[q]]);
        const double residual = double(rhs[row]) - ax;
        residual2 += residual * residual;
        rhs2 += double(rhs[row]) * double(rhs[row]);
    }
    return float(std::sqrt(residual2 / rhs2));
}
bspai::Fp32CsrMatrix identity(std::size_t n) {
    std::vector<bspai::Fp32Entry> entries;
    entries.reserve(n);
    for (std::size_t i = 0; i < n; ++i) entries.push_back({i, i, 1.0f});
    return bspai::Fp32CsrMatrix::from_entries(n, n, std::move(entries));
}
}
int main() {
    try {
        const bspai::Fp32CsrMatrix a = bspai::Fp32CsrMatrix::from_entries(3, 3, {
            {0,0,4.0f}, {0,1,-1.0f}, {1,0,-2.0f}, {1,1,5.0f}, {1,2,-1.0f}, {2,1,-1.0f}, {2,2,3.0f}});
        const bspai::Fp32CsrMatrix p = bspai::Fp32CsrMatrix::from_entries(3, 3, {{0,0,1.0f}, {1,1,1.0f}, {2,2,1.0f}});
        const std::vector<float> truth{0.25f, -0.5f, 0.75f};
        const std::vector<float> rhs = a.multiply(truth);
        // Retain the formerly failing, stricter smoke case as regression
        // evidence; the Experiment 2A contract tolerance remains 1e-4 below.
        bspai::Fp32GmresOptions strict_gmres; strict_gmres.tolerance = 2.0e-5f; strict_gmres.max_iterations = 3;
        const bspai::Fp32GmresResult gpu = bspai::fp32_cuda_left_preconditioned_gmres(a, p, rhs, strict_gmres);
        check(gpu.converged, "FP32 CUDA GMRES convergence");
        check(gpu.iterations > 0 && gpu.iterations <= 3, "FP32 CUDA GMRES iteration bound");
        check(std::isfinite(gpu.final_original_residual) && std::isfinite(gpu.final_preconditioned_relative), "FP32 CUDA GMRES diagnostics");
        close(gpu.solution, truth, 3.0e-4f);

        // A nonzero x0 forces a multi-step projected solve.  Every candidate
        // must be formed as x0 + V_k*y_k; starting a later candidate from the
        // prior candidate instead double-counts earlier Krylov corrections.
        bspai::Fp32GmresOptions gmres; gmres.tolerance = 1.0e-4f; gmres.max_iterations = 3;
        const std::vector<float> initial_guess{0.10f, -0.20f, 0.30f};
        const bspai::Fp32GmresResult cpu_from_x0 = bspai::fp32_left_preconditioned_gmres(
            a, p, rhs, gmres, initial_guess);
        const bspai::Fp32GmresResult gpu_from_x0 = bspai::fp32_cuda_left_preconditioned_gmres(
            a, p, rhs, gmres, initial_guess);
        check(cpu_from_x0.converged && gpu_from_x0.converged, "FP32 CUDA GMRES immutable-x0 regression convergence");
        check(cpu_from_x0.iterations > 1 && gpu_from_x0.iterations > 1, "FP32 CUDA GMRES immutable-x0 regression multi-iteration");
        check(cpu_from_x0.iterations == gpu_from_x0.iterations, "FP32 CUDA GMRES immutable-x0 regression iteration count");
        check(gpu_from_x0.final_preconditioned_relative <= gmres.tolerance, "FP32 CUDA GMRES immutable-x0 regression residual acceptance");
        finite_history(gpu_from_x0.hessenberg_residual_history, gpu_from_x0.iterations, "FP32 CUDA GMRES Hessenberg-history diagnostics");
        finite_history(gpu_from_x0.original_residual_history, gpu_from_x0.iterations, "FP32 CUDA GMRES original-residual-history diagnostics");
        finite_history(gpu_from_x0.preconditioned_relative_history, gpu_from_x0.iterations, "FP32 CUDA GMRES preconditioned-residual-history diagnostics");
        check(gpu_from_x0.preconditioned_relative_history.back() == gpu_from_x0.final_preconditioned_relative,
              "FP32 CUDA GMRES final recomputed residual history");
        close(gpu_from_x0.solution, cpu_from_x0.solution, 5.0e-4f);
        // A persistent context must reset all solve-local state while keeping
        // the same uploaded A/P. Compare every exposed result bit, including
        // the diagnostic defaults on invalid calls.
        bspai::Fp32CudaGmresContext context(a, p);
        const std::size_t matrix_bytes =
            2 * (a.rows() + 1) * sizeof(std::int64_t) +
            (a.values().size() + p.values().size()) *
                (sizeof(std::int64_t) + sizeof(float));
        check(context.matrix_uploads() == 2 && context.matrix_upload_bytes() == matrix_bytes,
              "FP32 CUDA context construction matrix uploads");
        const std::vector<float> rhs_second = a.multiply({-0.75f, 0.25f, 0.5f});
        const std::vector<float> invalid_short{1.0f};
        const std::vector<float> invalid_nonfinite{
            1.0f, std::numeric_limits<float>::quiet_NaN(), 0.0f};
        bspai::Fp32GmresOptions invalid_cap = gmres;
        invalid_cap.max_iterations = 4;
        bspai::Fp32GmresOptions invalid_tolerance = gmres;
        invalid_tolerance.tolerance = 0.0f;
        auto compare_context = [&](const std::vector<float>& b,
                                   const bspai::Fp32GmresOptions& options,
                                   const std::vector<float>& guess) {
            const auto expected = bspai::fp32_cuda_left_preconditioned_gmres(
                a, p, b, options, guess);
            const auto actual = context.solve(b, options, guess);
            identical_result(actual, expected, "FP32 CUDA persistent/one-shot result bits");
            check(context.matrix_uploads() == 2 &&
                      context.matrix_upload_bytes() == matrix_bytes,
                  "FP32 CUDA context repeated matrix upload");
        };
        compare_context(rhs, gmres, initial_guess);
        compare_context(rhs_second, gmres, {});
        compare_context(rhs, gmres, truth); // Exact-initial path between corrections.
        compare_context(invalid_short, gmres, {});
        compare_context(invalid_nonfinite, gmres, {});
        compare_context(rhs, gmres, invalid_short);
        compare_context(rhs, gmres, invalid_nonfinite);
        compare_context(rhs, invalid_cap, {});
        compare_context(rhs, invalid_tolerance, {});
        compare_context(rhs, gmres, initial_guess); // Repeat after all other paths.
        bspai::Fp32IrOptions ir; ir.gmres = strict_gmres; ir.max_outer_steps = 3;
        const bspai::Fp32IrResult refined = bspai::fp32_cuda_refine_with_preconditioner(a, p, rhs, ir);
        check(refined.converged, "FP32 CUDA iterative refinement convergence");
        const bspai::Fp32IrDiagnostics diagnostics = bspai::fp32_evaluate_ir_diagnostics(a, rhs, refined.solution);
        check(std::isfinite(diagnostics.normwise_backward_error) && diagnostics.normwise_backward_error <= bspai::fp32_ir_backward_tolerance(), "FP32 CUDA iterative refinement backward error");

        // A shift Jordan block with b=e_last needs more than 16 steps at the
        // frozen tolerance. Compare a cap at the first chunk boundary with a
        // cap two steps beyond it. The common prefix must be bitwise equal.
        constexpr std::size_t chain_n = 40;
        std::vector<bspai::Fp32Entry> chain_entries;
        for (std::size_t i = 0; i < chain_n; ++i) {
            chain_entries.push_back({i, i, 1.0f});
            if (i + 1 < chain_n) chain_entries.push_back({i, i + 1, 1.0f});
        }
        const auto chain = bspai::Fp32CsrMatrix::from_entries(chain_n, chain_n, std::move(chain_entries));
        const auto chain_p = identity(chain_n);
        std::vector<float> chain_rhs(chain_n, 0.0f);
        chain_rhs.back() = 1.0f;
        bspai::Fp32GmresOptions at_boundary; at_boundary.max_iterations = 16;
        bspai::Fp32GmresOptions past_boundary; past_boundary.max_iterations = 18;
        const auto before_growth = bspai::fp32_cuda_left_preconditioned_gmres(
            chain, chain_p, chain_rhs, at_boundary);
        const auto after_growth = bspai::fp32_cuda_left_preconditioned_gmres(
            chain, chain_p, chain_rhs, past_boundary);
        const auto cpu_after_growth = bspai::fp32_left_preconditioned_gmres(
            chain, chain_p, chain_rhs, past_boundary);
        check(before_growth.status == bspai::Fp32GmresStatus::iteration_limit &&
              before_growth.iterations == 16, "FP32 CUDA GMRES boundary cap");
        check(after_growth.status == bspai::Fp32GmresStatus::iteration_limit &&
              after_growth.iterations == 18, "FP32 CUDA GMRES chunk growth cap");
        check(cpu_after_growth.status == bspai::Fp32GmresStatus::iteration_limit &&
              cpu_after_growth.iterations == 18, "FP32 CPU GMRES chunk oracle cap");
        finite_history(after_growth.hessenberg_residual_history, 18, "FP32 CUDA growth Hessenberg history");
        finite_history(after_growth.original_residual_history, 18, "FP32 CUDA growth original history");
        finite_history(after_growth.preconditioned_relative_history, 18, "FP32 CUDA growth relative history");
        for (std::size_t i = 0; i < 16; ++i) {
            check(bits(before_growth.hessenberg_residual_history[i]) ==
                  bits(after_growth.hessenberg_residual_history[i]), "FP32 CUDA growth Hessenberg prefix bits");
            check(bits(before_growth.original_residual_history[i]) ==
                  bits(after_growth.original_residual_history[i]), "FP32 CUDA growth original prefix bits");
            check(bits(before_growth.preconditioned_relative_history[i]) ==
                  bits(after_growth.preconditioned_relative_history[i]), "FP32 CUDA growth relative prefix bits");
        }
        const float independent = independent_relative_residual(chain, chain_rhs, after_growth.solution);
        check(std::isfinite(independent) && independent > past_boundary.tolerance,
              "FP32 CUDA growth independently nonconverged");
        check(std::fabs(independent - after_growth.final_preconditioned_relative) < 5.0e-5f,
              "FP32 CUDA growth independent residual agreement");
        check(std::fabs(independent - cpu_after_growth.final_preconditioned_relative) < 5.0e-5f,
              "FP32 CUDA growth CPU residual agreement");
        close(after_growth.solution, cpu_after_growth.solution, 5.0e-4f);
        bspai::Fp32CudaGmresContext chain_context(chain, chain_p);
        const std::size_t chain_matrix_bytes =
            2 * (chain_n + 1) * sizeof(std::int64_t) +
            (chain.values().size() + chain_p.values().size()) *
                (sizeof(std::int64_t) + sizeof(float));
        check(chain_context.matrix_uploads() == 2 &&
                  chain_context.matrix_upload_bytes() == chain_matrix_bytes,
              "FP32 CUDA chain context construction matrix uploads");
        identical_result(chain_context.solve(chain_rhs, at_boundary), before_growth,
                         "FP32 CUDA context cap-16 result bits");
        identical_result(chain_context.solve(chain_rhs, past_boundary), after_growth,
                         "FP32 CUDA context cap-18 result bits");
        identical_result(chain_context.solve(chain_rhs, at_boundary), before_growth,
                         "FP32 CUDA context repeated cap-16 result bits");
        check(chain_context.matrix_uploads() == 2 &&
                  chain_context.matrix_upload_bytes() == chain_matrix_bytes,
              "FP32 CUDA chain context repeated matrix upload");

        // The former eager allocation reserved n*(cap+1) floats even for an
        // exact initial solution. This full-cap case also checks one-step use.
        constexpr std::size_t large_n = 12000;
        const auto large_identity = identity(large_n);
        std::vector<float> large_rhs(large_n, 0.0f);
        large_rhs.back() = 1.0f;
        bspai::Fp32GmresOptions full_cap; // zero means cap=n
        const auto exact = bspai::fp32_cuda_left_preconditioned_gmres(
            large_identity, large_identity, large_rhs, full_cap, large_rhs);
        check(exact.status == bspai::Fp32GmresStatus::exact_initial_solution &&
              exact.converged && exact.iterations == 0 && exact.solution == large_rhs,
              "FP32 CUDA large exact initial solution");
        const auto one_step = bspai::fp32_cuda_left_preconditioned_gmres(
            large_identity, large_identity, large_rhs, full_cap);
        check(one_step.status == bspai::Fp32GmresStatus::success && one_step.converged &&
              one_step.iterations == 1, "FP32 CUDA large one-step solve");
        check(one_step.solution == large_rhs &&
              independent_relative_residual(large_identity, large_rhs, one_step.solution) == 0,
              "FP32 CUDA large one-step exact oracle");
        std::cout << std::setprecision(9)
                  << "FP32 CUDA nonzero-x0 estimate=" << gpu_from_x0.hessenberg_residual_history.back()
                  << " recomputed_relative=" << gpu_from_x0.final_preconditioned_relative << '\n'
                  << "FP32 CUDA growth cap=18 independent_relative=" << independent << '\n'
                  << "FP32 CUDA GMRES tests passed\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
