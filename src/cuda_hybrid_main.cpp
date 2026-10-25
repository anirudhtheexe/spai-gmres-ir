#include "bspai/cuda_ir.hpp"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
std::size_t count(const char *text) {
    std::string s(text);
    if (s.empty() || s.find_first_not_of("0123456789") != std::string::npos)
        throw std::invalid_argument("Expected unsigned integer cap");
    std::size_t used = 0;
    const auto value = std::stoull(s, &used);
    if (used != s.size())
        throw std::invalid_argument("Invalid cap");
    return static_cast<std::size_t>(value);
}
}
int main(int argc, char **argv) {
    if (argc < 2 || argc > 5) {
        std::cerr
            << "Usage: cuda_hybrid_ir matrix.mtx [epsilon=0.3] [outercap=20] [innercap=0(default n)]\n";
        return 2;
    }
    try {
        std::ifstream input(argv[1]);
        if (!input)
            throw std::invalid_argument("Cannot open Matrix Market input");
        const auto a = bspai::read_matrix_market(input);
        if (!a.rows() || a.rows() != a.cols())
            throw std::invalid_argument("Hybrid IR requires a nonempty square matrix");
        bspai::IrOptions options;
        options.spai.epsilon = 0.3;
        options.spai.beta = 8;
        options.spai.alpha = a.rows() / 8 + (a.rows() % 8 != 0);
        if (argc >= 3) {
            std::size_t used = 0;
            std::string s(argv[2]);
            options.spai.epsilon = std::stod(s, &used);
            if (used != s.size())
                throw std::invalid_argument("Invalid epsilon");
        }
        if (argc >= 4)
            options.max_outer_steps = count(argv[3]);
        if (argc >= 5)
            options.gmres.max_iterations = count(argv[4]);
        if (!std::isfinite(options.spai.epsilon) || options.spai.epsilon <= 0 ||
            options.max_outer_steps > 20 || options.gmres.max_iterations > a.rows())
            throw std::invalid_argument(
                "Require finite positive epsilon, outercap<=20, innercap<=n");
        const std::vector<double> rhs(a.rows(), 1 / std::sqrt(static_cast<double>(a.rows())));
#ifdef BSPAI_GPU_CONSTRUCTION
        const auto out = bspai::cuda_gpu_spai_gmres_ir(a, rhs, options);
#else
        const auto out = bspai::cuda_spai_gmres_ir(a, rhs, options);
#endif
        const auto &result = out.solver;
        std::cout << std::setprecision(17) << "matrix=" << argv[1] << " n=" << a.rows()
                  << " nnz=" << a.values().size() << "\n"
                  << "precision=binary64 cpu_spai="
                  << (!out.gpu_construction_performed ? "true" : "false")
                  << " cpu_outer=true gpu_gmres=true rhs=equal_unit2 gpu_spai="
                  << (out.gpu_construction_performed ? "true" : "false") << "\n"
                  << "epsilon=" << options.spai.epsilon << " beta=" << options.spai.beta
                  << " alpha=" << options.spai.alpha << " tau=" << options.gmres.tolerance
                  << " innercap=" << options.gmres.max_iterations
                  << " outercap=" << options.max_outer_steps
                  << " backward_tolerance=" << bspai::ir_backward_tolerance() << "\n"
                  << "status=" << bspai::ir_status_name(result.status)
                  << " converged=" << result.converged << " outer_steps=" << result.outer_steps
                  << " inner_attempts=" << result.inner_attempts << "\n"
                  << "message=" << result.message << "\n"
                  << "contexts=" << out.context_constructions
                  << " matrix_uploads=" << out.matrix_uploads
                  << " vector_uploads=" << out.total_transfers.vector_uploads
                  << " vector_downloads=" << out.total_transfers.vector_downloads
                  << " arnoldi_uploads=" << out.total_transfers.arnoldi_vector_uploads
                  << " arnoldi_downloads=" << out.total_transfers.arnoldi_vector_downloads
                  << " scalar_observations=" << out.total_transfers.scalar_observations << "\n";
        if (result.construction_performed)
            std::cout << "spai_status=" << bspai::spai_status_name(result.construction.status)
                      << " nnz_P=" << result.construction.preconditioner.values().size() << "\n";
        if (out.gpu_construction_performed) {
            const auto &t = out.gpu_spai_telemetry;
            std::cout << "construction_streams=" << t.streams_created
                      << " max_fit_batch=" << t.max_fit_batch
                      << " parallel_fit_batches=" << t.parallel_fit_batches
                      << " qr_submissions=" << t.qr_submissions
                      << " max_qr_before_collection=" << t.max_qr_submissions_before_collection
                      << " construction_uploads=" << t.uploads
                      << " construction_downloads=" << t.downloads << "\n";
            std::cout << "construction_peak_device_bytes=" << t.peak_device_bytes
                      << " max_allocation_bytes=" << t.max_single_allocation_bytes
                      << " sparse_B_bytes=" << t.sparse_b_bytes
                      << " packed_output_bytes=" << t.packed_output_bytes
                      << " max_reduced_elements=" << t.max_reduced_elements
                      << " max_QR_workspace_bytes=" << t.max_qr_workspace_bytes
                      << " outstanding_device_allocations=" << t.outstanding_device_allocations
                      << "\n";
        }
        for (const auto &h : result.history) {
            std::cout << "outer=" << h.step << " residual_inf=" << h.diagnostics.residual_inf
                      << " residual_2=" << h.diagnostics.residual_2
                      << " eta_n=" << h.diagnostics.normwise_backward_error
                      << " eta_c=" << h.diagnostics.componentwise_backward_error << "\n";
        }
        for (std::size_t j = 0; j < result.inner_history.size(); ++j) {
            const auto &h = result.inner_history[j];
            std::cout << "inner=" << j + 1 << " status=" << bspai::gmres_status_name(h.status)
                      << " iterations=" << h.iterations
                      << " true_relative=" << h.final_preconditioned_relative
                      << " update_applied=" << result.update_applied[j] << "\n";
            for (const auto &k : h.history)
                std::cout << "inner_iteration=" << k.iteration
                          << " estimate=" << k.estimated_relative
                          << " true_relative=" << k.recomputed_relative
                          << " accepted=" << k.accepted << "\n";
        }
        for (std::size_t j = 0; j < result.solution.size(); ++j)
            std::cout << "x[" << j << "]=" << result.solution[j] << "\n";
        return result.converged ? 0 : (result.status == bspai::IrStatus::invalid_input ? 2 : 1);
    } catch (const std::invalid_argument &e) {
        std::cerr << "Input error: " << e.what() << "\n";
        return 2;
    } catch (const std::out_of_range &e) {
        std::cerr << "Input range error: " << e.what() << "\n";
        return 2;
    } catch (const std::exception &e) {
        std::cerr << "Checked runtime error: " << e.what() << "\n";
        return 1;
    }
}
