#include "bspai/ir.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace bspai {
static_assert(std::numeric_limits<double>::is_iec559 &&
              std::numeric_limits<double>::digits == 53,
              "Iterative refinement requires IEEE binary64 double");
namespace {

double finite(double value) {
    if (!std::isfinite(value)) throw std::overflow_error("Nonfinite outer residual/backward-error arithmetic");
    return value;
}

void validate_vectors(const CsrMatrix& a, const std::vector<double>& rhs,
                      const std::vector<double>& x, const std::vector<double>& reference) {
    if (a.rows() == 0 || a.rows() != a.cols() || rhs.size() != a.rows() ||
        x.size() != a.cols() || (!reference.empty() && reference.size() != a.cols()))
        throw std::invalid_argument("Invalid diagnostic matrix/vector dimensions");
    for (double value : rhs) if (!std::isfinite(value)) throw std::invalid_argument("RHS must be finite");
    for (double value : x) if (!std::isfinite(value)) throw std::invalid_argument("Iterate must be finite");
    for (double value : reference) if (!std::isfinite(value)) throw std::invalid_argument("Reference must be finite");
}

double ratio(double numerator, double denominator) {
    if (denominator == 0) return numerator == 0 ? 0 : std::numeric_limits<double>::infinity();
    return numerator / denominator;
}

bool valid_options(const CsrMatrix& a, const std::vector<double>& rhs,
                    const IrOptions& options, const std::vector<double>& reference) {
    if (a.rows() == 0 || a.rows() != a.cols() || rhs.size() != a.rows() ||
        (!reference.empty() && reference.size() != a.rows()) || options.max_outer_steps > 20 ||
        !std::isfinite(options.gmres.tolerance) || options.gmres.tolerance <= 0 ||
        options.gmres.max_iterations > a.rows()) return false;
    for (double value : rhs) if (!std::isfinite(value)) return false;
    for (double value : reference) if (!std::isfinite(value)) return false;
    return true;
}

} // namespace

const char* ir_status_name(IrStatus status) {
    switch (status) {
    case IrStatus::success: return "success";
    case IrStatus::invalid_input: return "invalid_input";
    case IrStatus::construction_failure: return "construction_failure";
    case IrStatus::inner_failure: return "inner_failure";
    case IrStatus::stagnation: return "stagnation";
    case IrStatus::iteration_limit: return "iteration_limit";
    case IrStatus::numerical_failure: return "numerical_failure";
    }
    return "unknown";
}

double ir_backward_tolerance() { return 10 * std::ldexp(1.0, -53); }

IrDiagnostics evaluate_ir_diagnostics(const CsrMatrix& a, const std::vector<double>& rhs,
                                      const std::vector<double>& x,
                                      const std::vector<double>& reference) {
    validate_vectors(a, rhs, x, reference);
    IrDiagnostics diagnostics;
    const std::vector<double> ax = a.multiply(x);
    diagnostics.residual.resize(a.rows());
    double matrix_inf = 0, x_inf = 0, rhs_inf = 0;
    for (double value : x) x_inf = std::max(x_inf, std::abs(value));
    for (std::size_t row = 0; row < a.rows(); ++row) {
        const double residual = finite(rhs[row] - ax[row]);
        diagnostics.residual[row] = residual;
        diagnostics.residual_inf = std::max(diagnostics.residual_inf, std::abs(residual));
        diagnostics.residual_2 = finite(std::hypot(diagnostics.residual_2, residual));
        rhs_inf = std::max(rhs_inf, std::abs(rhs[row]));
        double row_sum = 0, component_denominator = std::abs(rhs[row]);
        for (std::size_t index = a.row_offsets()[row]; index < a.row_offsets()[row + 1]; ++index) {
            const double value = std::abs(a.values()[index]);
            row_sum = finite(row_sum + value);
            component_denominator = finite(component_denominator + finite(value * std::abs(x[a.column_indices()[index]])));
        }
        matrix_inf = std::max(matrix_inf, row_sum);
        diagnostics.componentwise_backward_error = std::max(diagnostics.componentwise_backward_error,
                                                            ratio(std::abs(residual), component_denominator));
    }
    const double denominator = finite(finite(matrix_inf * x_inf) + rhs_inf);
    diagnostics.normwise_backward_error = ratio(diagnostics.residual_inf, denominator);
    if (!reference.empty()) {
        diagnostics.has_reference = true;
        double reference_inf = 0, error_inf = 0;
        for (std::size_t i = 0; i < x.size(); ++i) {
            reference_inf = std::max(reference_inf, std::abs(reference[i]));
            const double error = std::abs(x[i] - reference[i]);
            if (!std::isfinite(error)) diagnostics.forward_diagnostic_overflow = true;
            error_inf = std::max(error_inf, error);
        }
        diagnostics.forward_error = reference_inf == 0 ? error_inf : error_inf / reference_inf;
        if (!std::isfinite(diagnostics.forward_error)) diagnostics.forward_diagnostic_overflow = true;
        if (diagnostics.forward_diagnostic_overflow) {
            diagnostics.forward_error = std::numeric_limits<double>::infinity();
            diagnostics.forward_diagnostic_message = "Optional binary64 forward-error diagnostic overflow";
        }
    }
    return diagnostics;
}

IrResult::IrResult(std::size_t n) : construction(n) {}

IrResult refine_with_preconditioner(const CsrMatrix& a, const CsrMatrix& p,
                                    const std::vector<double>& rhs, const IrOptions& options,
                                    const std::vector<double>& reference) {
    IrResult result(a.rows());
    if (!valid_options(a, rhs, options, reference) || p.rows() != a.rows() || p.cols() != a.cols()) {
        result.message = "Invalid dimensions, finite inputs, inner options, or outer cap";
        return result;
    }
    try {
        IrIteration initial;
        initial.x = p.multiply(rhs);
        initial.diagnostics = evaluate_ir_diagnostics(a, rhs, initial.x, reference);
        result.solution = initial.x;
        result.history.push_back(initial);
        if (initial.diagnostics.normwise_backward_error <= ir_backward_tolerance()) {
            result.status = IrStatus::success; result.converged = true; return result;
        }
        for (std::size_t step = 0; step < options.max_outer_steps; ++step) {
            const std::vector<double> residual = result.history.back().diagnostics.residual;
            result.correction_rhs.push_back(residual);
            result.update_applied.push_back(false);
            result.inner_history.push_back(left_preconditioned_gmres(a, p, residual, options.gmres));
            ++result.inner_attempts;
            const GmresResult& inner = result.inner_history.back();
            if (!inner.converged) {
                result.status = IrStatus::inner_failure;
                result.message = "Correction GMRES failed: " + std::string(gmres_status_name(inner.status)) + ": " + inner.message;
                return result;
            }
            IrIteration next;
            next.step = step + 1;
            next.x = result.solution;
            for (std::size_t i = 0; i < next.x.size(); ++i) next.x[i] = finite(next.x[i] + inner.solution[i]);
            next.diagnostics = evaluate_ir_diagnostics(a, rhs, next.x, reference);
            next.has_inner = true;
            next.inner = inner;
            const bool unchanged = next.x == result.solution;
            result.solution = next.x;
            result.history.push_back(next);
            result.update_applied.back() = true;
            ++result.outer_steps;
            if (next.diagnostics.normwise_backward_error <= ir_backward_tolerance()) {
                result.status = IrStatus::success; result.converged = true; return result;
            }
            if (unchanged) {
                result.status = IrStatus::stagnation;
                result.message = "Unchanged outer iterate above original backward-error tolerance";
                return result;
            }
        }
        result.status = IrStatus::iteration_limit;
        result.message = "Outer correction cap reached above original backward-error tolerance";
    } catch (const std::overflow_error& error) {
        result.status = IrStatus::numerical_failure;
        result.message = error.what();
    }
    return result;
}

IrResult spai_gmres_ir(const CsrMatrix& a, const std::vector<double>& rhs,
                       const IrOptions& options, const std::vector<double>& reference) {
    IrResult result(a.rows());
    if (!valid_options(a, rhs, options, reference)) {
        result.message = "Invalid dimensions, finite inputs, inner options, or outer cap";
        return result;
    }
    result.construction_performed = true;
    result.construction = construct_spai(a, options.spai);
    if (result.construction.status != SpaiStatus::success) {
        result.status = IrStatus::construction_failure;
        result.message = "SPAI failed: " + std::string(spai_status_name(result.construction.status)) + ": " + result.construction.message;
        return result;
    }
    IrResult refined = refine_with_preconditioner(a, result.construction.preconditioner, rhs, options, reference);
    refined.construction = std::move(result.construction);
    refined.construction_performed = true;
    return refined;
}

} // namespace bspai
