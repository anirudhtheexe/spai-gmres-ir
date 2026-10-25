#include "bspai/cuda_ir.hpp"
#include <cmath>
#include <memory>
#include <stdexcept>
#include <utility>

namespace bspai {
namespace {
bool valid(const CsrMatrix &a, const std::vector<double> &rhs, const IrOptions &o,
           const std::vector<double> &reference) {
    if (!a.rows() || a.rows() != a.cols() || rhs.size() != a.rows() ||
        (!reference.empty() && reference.size() != a.rows()) || o.max_outer_steps > 20 ||
        !std::isfinite(o.gmres.tolerance) || o.gmres.tolerance <= 0 ||
        o.gmres.max_iterations > a.rows())
        return false;
    for (double v : rhs)
        if (!std::isfinite(v))
            return false;
    for (double v : reference)
        if (!std::isfinite(v))
            return false;
    return true;
}
double finite_update(double x) {
    if (!std::isfinite(x))
        throw std::overflow_error("Nonfinite hybrid outer update");
    return x;
}
}

CudaIrResult cuda_refine_with_preconditioner(const CsrMatrix &a, const CsrMatrix &p,
                                             const std::vector<double> &rhs,
                                             const IrOptions &options,
                                             const std::vector<double> &reference, int device) {
    CudaIrResult out(a.rows());
    IrResult &result = out.solver;
    if (!valid(a, rhs, options, reference) || p.rows() != a.rows() || p.cols() != a.cols()) {
        result.message = "Invalid hybrid dimensions, finite inputs, inner options, or outer cap";
        return out;
    }
    std::unique_ptr<CudaGmresContext> context;
    try {
        context.reset(new CudaGmresContext(a, p, options.gmres, device));
        out.context_constructions = 1;
        out.matrix_uploads = context->matrix_uploads();
        out.construction_transfers = context->construction_transfers();
        // CPU outer staging deliberately reuses the frozen diagnostic routine.
        // Only the inner-solver call differs from the CPU IR contract.
        auto refine = [&]() {
            IrIteration initial;
            initial.x = p.multiply(rhs);
            initial.diagnostics = evaluate_ir_diagnostics(a, rhs, initial.x, reference);
            result.solution = initial.x;
            result.history.push_back(initial);
            if (initial.diagnostics.normwise_backward_error <= ir_backward_tolerance()) {
                result.status = IrStatus::success;
                result.converged = true;
                return;
            }
            for (std::size_t step = 0; step < options.max_outer_steps; ++step) {
                const auto residual = result.history.back().diagnostics.residual;
                result.correction_rhs.push_back(residual);
                result.update_applied.push_back(false);
                CudaGmresResult correction = context->solve(residual);
                out.inner_transfers.push_back(correction.transfers);
                result.inner_history.push_back(std::move(correction.solver));
                ++result.inner_attempts;
                const auto &inner = result.inner_history.back();
                if (!inner.converged) {
                    result.status = IrStatus::inner_failure;
                    result.message = "Correction GPU GMRES failed: " +
                                     std::string(gmres_status_name(inner.status)) + ": " +
                                     inner.message;
                    return;
                }
                IrIteration next;
                next.step = step + 1;
                next.x = result.solution;
                for (std::size_t i = 0; i < next.x.size(); ++i)
                    next.x[i] = finite_update(next.x[i] + inner.solution[i]);
                next.diagnostics = evaluate_ir_diagnostics(a, rhs, next.x, reference);
                next.has_inner = true;
                next.inner = inner;
                const bool unchanged = next.x == result.solution;
                result.solution = next.x;
                result.history.push_back(next);
                result.update_applied.back() = true;
                ++result.outer_steps;
                if (next.diagnostics.normwise_backward_error <= ir_backward_tolerance()) {
                    result.status = IrStatus::success;
                    result.converged = true;
                    return;
                }
                if (unchanged) {
                    result.status = IrStatus::stagnation;
                    result.message =
                        "Unchanged hybrid outer iterate above original backward-error tolerance";
                    return;
                }
            }
            result.status = IrStatus::iteration_limit;
            result.message =
                "Hybrid outer correction cap reached above original backward-error tolerance";
        };
        refine();
    } catch (const std::overflow_error &error) {
        result.status = IrStatus::numerical_failure;
        result.message = error.what();
    } catch (const std::invalid_argument &error) {
        result.status = IrStatus::invalid_input;
        result.message = error.what();
    }
    if (context)
        out.total_transfers = context->total_transfers();
    return out;
}

CudaIrResult cuda_spai_gmres_ir(const CsrMatrix &a, const std::vector<double> &rhs,
                                const IrOptions &options, const std::vector<double> &reference,
                                int device) {
    CudaIrResult out(a.rows());
    IrResult &result = out.solver;
    if (!valid(a, rhs, options, reference)) {
        result.message = "Invalid hybrid dimensions, finite inputs, inner options, or outer cap";
        return out;
    }
    result.construction_performed = true;
    result.construction = construct_spai(a, options.spai);
    if (result.construction.status != SpaiStatus::success) {
        result.status = IrStatus::construction_failure;
        result.message =
            "CPU SPAI failed: " + std::string(spai_status_name(result.construction.status)) + ": " +
            result.construction.message;
        return out;
    }
    CudaIrResult refined = cuda_refine_with_preconditioner(a, result.construction.preconditioner,
                                                           rhs, options, reference, device);
    refined.solver.construction = std::move(result.construction);
    refined.solver.construction_performed = true;
    return refined;
}
CudaIrResult cuda_gpu_spai_gmres_ir(const CsrMatrix &a, const std::vector<double> &rhs,
                                    const IrOptions &options, const std::vector<double> &reference,
                                    int device) {
    CudaIrResult out(a.rows());
    if (!valid(a, rhs, options, reference)) {
        out.solver.message = "Invalid GPU-construction hybrid dimensions/options";
        return out;
    }
    auto construction = cuda_construct_spai(a, options.spai, device);
    out.solver.construction_performed = true;
    out.gpu_construction_performed = true;
    out.gpu_spai_telemetry = construction.telemetry;
    out.solver.construction = std::move(construction.construction);
    if (out.solver.construction.status != SpaiStatus::success) {
        out.solver.status = IrStatus::construction_failure;
        out.solver.message =
            "GPU SPAI failed: " + std::string(spai_status_name(out.solver.construction.status)) +
            ": " + out.solver.construction.message;
        return out;
    }
    auto refined = cuda_refine_with_preconditioner(a, out.solver.construction.preconditioner, rhs,
                                                   options, reference, device);
    refined.solver.construction = std::move(out.solver.construction);
    refined.solver.construction_performed = true;
    refined.gpu_construction_performed = true;
    refined.gpu_spai_telemetry = out.gpu_spai_telemetry;
    return refined;
}
}
