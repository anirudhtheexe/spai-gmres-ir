#ifndef BSPAI_IR_HPP
#define BSPAI_IR_HPP

#include "bspai/gmres.hpp"
#include "bspai/spai.hpp"

namespace bspai {

enum class IrStatus {
    success, invalid_input, construction_failure, inner_failure,
    stagnation, iteration_limit, numerical_failure
};
const char* ir_status_name(IrStatus status);

struct IrOptions {
    SpaiOptions spai;
    GmresOptions gmres;
    std::size_t max_outer_steps = 20;
};

struct IrDiagnostics {
    std::vector<double> residual;
    double residual_inf = 0, residual_2 = 0;
    double normwise_backward_error = 0, componentwise_backward_error = 0;
    double forward_error = std::numeric_limits<double>::quiet_NaN();
    bool has_reference = false;
    bool forward_diagnostic_overflow = false;
    std::string forward_diagnostic_message;
};

struct IrIteration {
    std::size_t step = 0;
    std::vector<double> x;
    IrDiagnostics diagnostics;
    bool has_inner = false;
    GmresResult inner;
};

struct IrResult {
    explicit IrResult(std::size_t n);
    IrStatus status = IrStatus::invalid_input;
    std::string message;
    bool converged = false, construction_performed = false;
    std::size_t outer_steps = 0, inner_attempts = 0;
    std::vector<double> solution;
    std::vector<IrIteration> history;
    SpaiResult construction;
    std::vector<GmresResult> inner_history;
    std::vector<std::vector<double>> correction_rhs;
    std::vector<bool> update_applied;
};

double ir_backward_tolerance(); // Fixed 10*2^-53, not configurable.
IrDiagnostics evaluate_ir_diagnostics(const CsrMatrix& a,
                                      const std::vector<double>& rhs,
                                      const std::vector<double>& x,
                                      const std::vector<double>& reference = {});
IrResult refine_with_preconditioner(const CsrMatrix& a, const CsrMatrix& p,
                                    const std::vector<double>& rhs,
                                    const IrOptions& options,
                                    const std::vector<double>& reference = {});
IrResult spai_gmres_ir(const CsrMatrix& a, const std::vector<double>& rhs,
                       const IrOptions& options,
                       const std::vector<double>& reference = {});

} // namespace bspai
#endif
