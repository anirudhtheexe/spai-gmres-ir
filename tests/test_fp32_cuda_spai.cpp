// Experiment 2A validation-only GPU-SPAI constructor and fixed-P gate.
// It never substitutes a CPU constructor for the CUDA result under test.
#include "bspai/fp32_cuda_spai.hpp"
#include "bspai/fp32_cuda_gmres.hpp"
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace {
void check(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

long double norm2(const std::vector<float>& x) {
    long double sum = 0;
    for (float value : x) sum += static_cast<long double>(value) * value;
    return std::sqrt(sum);
}

long double relative_error(const std::vector<float>& got, const std::vector<float>& reference) {
    std::vector<float> difference(got.size());
    for (std::size_t i = 0; i < got.size(); ++i) difference[i] = got[i] - reference[i];
    return norm2(difference) / norm2(reference);
}

long double preconditioned_relative_residual(const bspai::Fp32CsrMatrix& a,
                                              const bspai::Fp32CsrMatrix& p,
                                              const std::vector<float>& rhs,
                                              const std::vector<float>& solution) {
    const std::vector<float> ax = a.multiply(solution);
    std::vector<float> residual(rhs.size());
    for (std::size_t i = 0; i < rhs.size(); ++i) residual[i] = rhs[i] - ax[i];
    return norm2(p.multiply(residual)) / norm2(p.multiply(rhs));
}
}

int main() {
    try {
        const bspai::Fp32CsrMatrix a = bspai::Fp32CsrMatrix::from_entries(3, 3, {
            {0, 0, 4.0f}, {0, 1, -1.0f},
            {1, 0, -2.0f}, {1, 1, 5.0f}, {1, 2, -1.0f},
            {2, 1, -1.0f}, {2, 2, 3.0f}});
        bspai::Fp32SpaiOptions spai;
        spai.epsilon = .3f;
        spai.alpha = 5;
        spai.beta = 8;

        const bspai::Fp32SpaiResult cpu = bspai::fp32_construct_spai(a, spai);
        const bspai::Fp32CudaSpaiResult gpu = bspai::fp32_cuda_construct_spai(a, spai);
        check(cpu.status == bspai::Fp32SpaiStatus::success, "FP32 CPU SPAI fixture construction");
        check(gpu.construction.status == bspai::Fp32SpaiStatus::success, "FP32 GPU SPAI fixture construction");
        check(gpu.telemetry.gpu_scaling && gpu.telemetry.gpu_preconditioner_assembly,
              "FP32 GPU SPAI scaling and P assembly telemetry");
        check(gpu.telemetry.qr_submissions > 0 && gpu.telemetry.streams_created > 0,
              "FP32 GPU SPAI QR/stream telemetry");
        std::size_t fp32_fits = 0, fp32_growths = 0;
        for (const bspai::Fp32SpaiColumn& column : gpu.construction.columns) {
            fp32_fits += column.fits;
            fp32_growths += column.growths;
        }
        // Setup/assembly make five D2H calls; each completed fit contributes QR
        // info plus three fit results, and each growth now has one packed score
        // record transfer rather than score/status/clamp as three transfers.
        check(gpu.telemetry.downloads == 5 + 4 * fp32_fits + fp32_growths,
              "FP32 packed score-result D2H transfer accounting");
        check(gpu.construction.inverse_nnz != 0 && gpu.construction.preconditioner_nnz != 0,
              "FP32 GPU SPAI nonempty compact outputs");
        check(gpu.construction.columns.size() == a.rows(), "FP32 GPU SPAI column count");
        for (const bspai::Fp32SpaiColumn& column : gpu.construction.columns) {
            check(column.status == bspai::Fp32SpaiStatus::success, "FP32 GPU SPAI column success");
            check(column.pattern.size() == column.coefficients.size() && !column.pattern.empty(),
                  "FP32 GPU SPAI compact coefficient alignment");
            check(std::isfinite(column.reduced_residual) && column.reduced_residual <= spai.epsilon,
                  "FP32 GPU SPAI frozen epsilon acceptance");
        }

        const std::vector<float> truth{.25f, -.5f, .75f};
        const std::vector<float> rhs = a.multiply(truth);
        const std::vector<float> x0{.10f, -.20f, .30f};
        bspai::Fp32GmresOptions gmres;
        gmres.tolerance = 1.0e-4f;
        gmres.max_iterations = 3;
        const bspai::Fp32GmresResult cpu_fixed = bspai::fp32_left_preconditioned_gmres(
            a, gpu.construction.preconditioner, rhs, gmres, x0);
        const bspai::Fp32GmresResult gpu_fixed = bspai::fp32_cuda_left_preconditioned_gmres(
            a, gpu.construction.preconditioner, rhs, gmres, x0);
        check(cpu_fixed.converged && gpu_fixed.converged, "FP32 GPU-built fixed-P CPU/GPU GMRES convergence");
        check(cpu_fixed.status == gpu_fixed.status, "FP32 GPU-built fixed-P CPU/GPU GMRES status");
        check(cpu_fixed.iterations == gpu_fixed.iterations, "FP32 GPU-built fixed-P CPU/GPU GMRES iteration count");
        check(gpu_fixed.final_preconditioned_relative <= gmres.tolerance,
              "FP32 GPU-built fixed-P CUDA residual acceptance");
        const long double cpu_residual = preconditioned_relative_residual(a, gpu.construction.preconditioner, rhs, cpu_fixed.solution);
        const long double gpu_residual = preconditioned_relative_residual(a, gpu.construction.preconditioner, rhs, gpu_fixed.solution);
        check(cpu_residual <= gmres.tolerance && gpu_residual <= gmres.tolerance,
              "independent FP32 fixed-P preconditioned residual reconstruction");
        const long double cpu_error = relative_error(cpu_fixed.solution, truth);
        const long double gpu_error = relative_error(gpu_fixed.solution, truth);
        check(cpu_error <= 2.0e-3L && gpu_error <= 2.0e-3L,
              "higher-precision FP32 fixed-P manufactured-solution accuracy");

        // End-to-end integration uses the frozen alpha=ceil(n/8) setting,
        // constructs P on the requested backend, and requires an actual
        // correction rather than accepting an already-exact starting point.
        const bspai::Fp32CsrMatrix integration_a = bspai::Fp32CsrMatrix::from_entries(3, 3, {
            {0, 0, 4.0f}, {0, 1, -.10f},
            {1, 0, -.20f}, {1, 1, 5.0f}, {1, 2, -.10f},
            {2, 1, -.10f}, {2, 2, 3.0f}});
        const std::vector<float> integration_rhs = integration_a.multiply(truth);
        bspai::Fp32IrOptions ir;
        ir.spai.epsilon = .3f;
        ir.spai.beta = 8;
        ir.spai.alpha = (a.rows() + 7) / 8;
        ir.gmres.tolerance = 1.0e-4f;
        ir.gmres.max_iterations = a.rows();
        ir.max_outer_steps = 20;
        const bspai::Fp32IrResult cpu_pipeline = bspai::fp32_spai_gmres_ir(integration_a, integration_rhs, ir);
        const bspai::Fp32IrResult gpu_pipeline = bspai::fp32_cuda_spai_gmres_ir(integration_a, integration_rhs, ir);
        check(cpu_pipeline.construction.status == bspai::Fp32SpaiStatus::success &&
                  gpu_pipeline.construction.status == bspai::Fp32SpaiStatus::success,
              "FP32 CPU/GPU pipeline SPAI construction");
        check(cpu_pipeline.converged && gpu_pipeline.converged,
              "FP32 CPU/GPU pipeline iterative-refinement convergence");
        check(cpu_pipeline.outer_steps >= 1 && gpu_pipeline.outer_steps >= 1 &&
                  cpu_pipeline.inner_attempts >= 1 && gpu_pipeline.inner_attempts >= 1,
              "FP32 CPU/GPU pipeline accepted iterative-refinement correction");
        check(gpu_pipeline.outer_history.size() >= 2,
              "FP32 GPU pipeline initial and post-correction outer history");
        check(gpu_pipeline.outer_history.front().normwise_backward_error > bspai::fp32_ir_backward_tolerance() &&
                  gpu_pipeline.outer_history.back().normwise_backward_error <= bspai::fp32_ir_backward_tolerance(),
              "FP32 GPU pipeline outer-history backward-error progression");
        const long double cpu_pipeline_error = relative_error(cpu_pipeline.solution, truth);
        const long double gpu_pipeline_error = relative_error(gpu_pipeline.solution, truth);
        check(cpu_pipeline_error <= 2.0e-3L && gpu_pipeline_error <= 2.0e-3L,
              "FP32 CPU/GPU pipeline manufactured-solution accuracy");
        std::cout << std::setprecision(9)
                  << "FP32 CUDA SPAI fixed-P iterations CPU=" << cpu_fixed.iterations
                  << " GPU=" << gpu_fixed.iterations
                  << " reported_GPU=" << gpu_fixed.final_preconditioned_relative
                  << " recomputed_GPU=" << static_cast<double>(gpu_residual)
                  << " forward_GPU=" << static_cast<double>(gpu_error) << '\n'
                  << "FP32 CUDA SPAI pipeline outer_steps CPU=" << cpu_pipeline.outer_steps
                  << " GPU=" << gpu_pipeline.outer_steps
                  << " initial_BE=" << gpu_pipeline.outer_history.front().normwise_backward_error
                  << " final_BE=" << gpu_pipeline.outer_history.back().normwise_backward_error
                  << " forward_GPU=" << static_cast<double>(gpu_pipeline_error) << '\n'
                  << "FP32 CUDA SPAI tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FP32 CUDA SPAI test failure: " << error.what() << '\n';
        return 1;
    }
}
