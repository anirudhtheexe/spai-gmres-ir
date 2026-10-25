// Experiment 2A unit checks.  These test only the isolated uniform-FP32 API.
#include "bspai/fp32.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {
void check(bool condition, const char *message) {
    if (!condition)
        throw std::runtime_error(message);
}
}

int main() {
    try {
        const bspai::Fp32CsrMatrix a = bspai::Fp32CsrMatrix::from_entries(3, 3,
                                                                          {{0, 0, 4.0f},
                                                                           {0, 1, -1.0f},
                                                                           {1, 0, -2.0f},
                                                                           {1, 1, 5.0f},
                                                                           {1, 2, -1.0f},
                                                                           {2, 1, -1.0f},
                                                                           {2, 2, 3.0f}});
        const std::vector<float> x{0.25f, -0.5f, 0.75f};
        const std::vector<float> b = a.multiply(x);
        // Even an unused sparse column must not conceal a nonfinite input.
        const auto sparse = bspai::Fp32CsrMatrix::from_entries(2, 2, {{0, 0, 1.0f}});
        for (float bad :
             {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity()}) {
            bool rejected = false;
            try {
                sparse.multiply({1.0f, bad});
            } catch (const std::invalid_argument &) {
                rejected = true;
            }
            check(rejected, "FP32 SpMV rejects nonfinite unused-column input");
        }
        bool rejected_rows = false;
        try {
            bspai::Fp32CsrMatrix impossible(std::numeric_limits<std::size_t>::max(), 1);
        } catch (const std::overflow_error &) {
            rejected_rows = true;
        }
        check(rejected_rows, "FP32 CSR rejects row-offset count overflow");
        bspai::Fp32IrOptions options;
        options.spai.alpha = 3;
        options.spai.beta = 2;
        options.gmres.max_iterations = 3;
        const bspai::Fp32SpaiResult p = bspai::fp32_construct_spai(a, options.spai);
        check(p.status == bspai::Fp32SpaiStatus::success, "FP32 SPAI construction");
        check(p.inverse_nnz != 0 && p.preconditioner_nnz != 0, "FP32 SPAI nnz diagnostics");
        check(!p.columns[0].shadow.empty(), "FP32 SPAI shadow diagnostic");
        const bspai::Fp32GmresResult g =
            bspai::fp32_left_preconditioned_gmres(a, p.preconditioner, b, options.gmres);
        check(g.converged, "FP32 GMRES convergence");
        check(g.hessenberg_residual_history.size() == g.iterations &&
                  g.preconditioned_relative_history.size() == g.iterations,
              "FP32 GMRES residual histories");
        const bspai::Fp32GmresResult zero = bspai::fp32_left_preconditioned_gmres(
            a, p.preconditioner, std::vector<float>(3, 0.0f), options.gmres);
        check(zero.converged && zero.status == bspai::Fp32GmresStatus::zero_rhs,
              "FP32 GMRES zero RHS");
        const bspai::Fp32IrResult r = bspai::fp32_spai_gmres_ir(a, b, options);
        check(r.converged, "FP32 IR convergence");
        check(!r.outer_history.empty(), "FP32 IR outer history");
        const bspai::Fp32IrDiagnostics d = bspai::fp32_evaluate_ir_diagnostics(a, b, r.solution);
        check(std::isfinite(d.normwise_backward_error) &&
                  d.normwise_backward_error <= bspai::fp32_ir_backward_tolerance(),
              "FP32 backward error");
        std::cout << "FP32 CPU tests passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "FP32 CPU test failure: " << error.what() << '\n';
        return 1;
    }
}
