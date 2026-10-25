#ifndef BSPAI_GMRES_HPP
#define BSPAI_GMRES_HPP

#include "bspai/cpu.hpp"
#include <limits>
#include <string>

namespace bspai {

enum class GmresStatus {
    success,
    zero_rhs,
    exact_initial_solution,
    invalid_input,
    zero_preconditioned_rhs,
    breakdown,
    iteration_limit,
    nonconvergence,
    numerical_failure
};
const char *gmres_status_name(GmresStatus status);

struct GmresOptions {
    double tolerance = 1e-8;
    std::size_t max_iterations = 0; // Default n; explicit values must be 1..n.
};

struct GmresIteration {
    std::size_t iteration = 0;
    double estimated_relative = std::numeric_limits<double>::quiet_NaN();
    double recomputed_relative = std::numeric_limits<double>::quiet_NaN();
    double original_residual = std::numeric_limits<double>::quiet_NaN();
    double arnoldi_subdiagonal = std::numeric_limits<double>::quiet_NaN();
    bool breakdown = false, verification_triggered = false, accepted = false;
    std::vector<double> solution;
};

struct GmresResult {
    GmresStatus status = GmresStatus::invalid_input;
    std::string message;
    bool converged = false;
    std::size_t iterations = 0;
    std::vector<double> solution;
    double initial_original_residual = std::numeric_limits<double>::quiet_NaN();
    double initial_preconditioned_relative = std::numeric_limits<double>::quiet_NaN();
    double final_original_residual = std::numeric_limits<double>::quiet_NaN();
    double final_preconditioned_relative = std::numeric_limits<double>::quiet_NaN();
    std::vector<GmresIteration> history;
    std::vector<std::vector<double>> basis, hessenberg; // Unrotated Arnoldi H.
};

// Exactly two sparse products, never materialize the rounded matrix P*A.
std::vector<double> apply_left_preconditioned(const CsrMatrix &a, const CsrMatrix &p,
                                              const std::vector<double> &v);
GmresResult left_preconditioned_gmres(const CsrMatrix &a, const CsrMatrix &p,
                                      const std::vector<double> &rhs, const GmresOptions &options,
                                      const std::vector<double> &initial_guess = {});

} // namespace bspai
#endif
