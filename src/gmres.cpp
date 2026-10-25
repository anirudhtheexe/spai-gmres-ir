#include "bspai/gmres.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace bspai {
static_assert(std::numeric_limits<double>::is_iec559 && std::numeric_limits<double>::digits == 53,
              "GMRES requires IEEE binary64 double");
namespace {

double finite(double value) {
    if (!std::isfinite(value))
        throw std::overflow_error("Nonfinite GMRES arithmetic");
    return value;
}

double norm2(const std::vector<double> &values) {
    double result = 0;
    for (double value : values)
        result = finite(std::hypot(result, value));
    return result;
}

std::vector<double> subtract(const std::vector<double> &a, const std::vector<double> &b) {
    std::vector<double> result(a.size());
    for (std::size_t i = 0; i < a.size(); ++i)
        result[i] = finite(a[i] - b[i]);
    return result;
}

struct ResidualCheck {
    double original;
    double relative;
};

ResidualCheck check_residual(const CsrMatrix &a, const CsrMatrix &p, const std::vector<double> &rhs,
                             const std::vector<double> &preconditioned_rhs, double denominator,
                             const std::vector<double> &x) {
    const std::vector<double> ax = a.multiply(x);
    ResidualCheck result;
    result.original = norm2(subtract(rhs, ax));
    result.relative = finite(norm2(subtract(preconditioned_rhs, p.multiply(ax))) / denominator);
    return result;
}

bool form_candidate(const std::vector<std::vector<double>> &rotated, const std::vector<double> &g,
                    const std::vector<std::vector<double>> &basis,
                    const std::vector<double> &initial, std::size_t count,
                    std::vector<double> &candidate) {
    std::vector<double> y(count, 0);
    for (std::size_t row = count; row-- > 0;) {
        if (rotated[row][row] == 0)
            return false;
        double value = g[row];
        for (std::size_t col = row + 1; col < count; ++col)
            value = finite(value - finite(rotated[row][col] * y[col]));
        y[row] = finite(value / rotated[row][row]);
    }
    candidate = initial;
    for (std::size_t col = 0; col < count; ++col)
        for (std::size_t row = 0; row < candidate.size(); ++row)
            candidate[row] = finite(candidate[row] + finite(basis[col][row] * y[col]));
    return true;
}

void store_check(GmresResult &result, GmresIteration &record, const ResidualCheck &check) {
    record.recomputed_relative = check.relative;
    record.original_residual = check.original;
    result.final_original_residual = check.original;
    result.final_preconditioned_relative = check.relative;
}

} // namespace

const char *gmres_status_name(GmresStatus status) {
    switch (status) {
    case GmresStatus::success:
        return "success";
    case GmresStatus::zero_rhs:
        return "zero_rhs";
    case GmresStatus::exact_initial_solution:
        return "exact_initial_solution";
    case GmresStatus::invalid_input:
        return "invalid_input";
    case GmresStatus::zero_preconditioned_rhs:
        return "zero_preconditioned_rhs";
    case GmresStatus::breakdown:
        return "breakdown";
    case GmresStatus::iteration_limit:
        return "iteration_limit";
    case GmresStatus::nonconvergence:
        return "nonconvergence";
    case GmresStatus::numerical_failure:
        return "numerical_failure";
    }
    return "unknown";
}

std::vector<double> apply_left_preconditioned(const CsrMatrix &a, const CsrMatrix &p,
                                              const std::vector<double> &v) {
    if (a.rows() == 0 || a.rows() != a.cols() || p.rows() != a.rows() || p.cols() != a.cols())
        throw std::invalid_argument(
            "Left-preconditioned operator requires equal square dimensions");
    return p.multiply(a.multiply(v));
}

GmresResult left_preconditioned_gmres(const CsrMatrix &a, const CsrMatrix &p,
                                      const std::vector<double> &rhs, const GmresOptions &options,
                                      const std::vector<double> &initial_guess) {
    GmresResult result;
    const std::size_t n = a.rows();
    if (n == 0 || a.cols() != n || p.rows() != n || p.cols() != n || rhs.size() != n ||
        (!initial_guess.empty() && initial_guess.size() != n) ||
        !std::isfinite(options.tolerance) || options.tolerance <= 0 || options.max_iterations > n) {
        result.message = "Invalid dimensions, initial guess, tolerance, or iteration cap";
        return result;
    }
    for (double value : rhs)
        if (!std::isfinite(value)) {
            result.message = "RHS must be finite";
            return result;
        }
    for (double value : initial_guess)
        if (!std::isfinite(value)) {
            result.message = "Initial guess must be finite";
            return result;
        }
    const std::vector<double> initial =
        initial_guess.empty() ? std::vector<double>(n, 0) : initial_guess;
    result.solution = initial;
    const std::size_t cap = options.max_iterations == 0 ? n : options.max_iterations;
    try {
        const std::vector<double> ax0 = a.multiply(initial);
        const std::vector<double> original_initial = subtract(rhs, ax0);
        result.initial_original_residual = norm2(original_initial);
        result.final_original_residual = result.initial_original_residual;
        if (result.initial_original_residual == 0) {
            const bool rhs_zero = std::all_of(rhs.begin(), rhs.end(), [](double value) {
                return value == 0;
            });
            result.status = rhs_zero ? GmresStatus::zero_rhs : GmresStatus::exact_initial_solution;
            result.converged = true;
            // Exact-original-residual convention, without evaluating 0/0.
            result.initial_preconditioned_relative = result.final_preconditioned_relative = 0;
            return result;
        }
        const std::vector<double> preconditioned_rhs = p.multiply(rhs);
        const double denominator = norm2(preconditioned_rhs);
        if (denominator == 0) {
            result.status = GmresStatus::zero_preconditioned_rhs;
            result.message = "Zero preconditioned RHS with nonzero original initial residual";
            return result;
        }
        const std::vector<double> residual = subtract(preconditioned_rhs, p.multiply(ax0));
        const double beta = norm2(residual);
        result.initial_preconditioned_relative = finite(beta / denominator);
        result.final_preconditioned_relative = result.initial_preconditioned_relative;
        if (result.initial_preconditioned_relative <= options.tolerance) {
            result.status = GmresStatus::success;
            result.converged = true;
            return result;
        }
        std::vector<double> v(n);
        for (std::size_t row = 0; row < n; ++row)
            v[row] = finite(residual[row] / beta);
        result.basis.push_back(v);
        result.hessenberg.assign(cap + 1, std::vector<double>(cap, 0));
        std::vector<std::vector<double>> rotated(cap + 1, std::vector<double>(cap, 0));
        std::vector<double> cosine(cap, 0), sine(cap, 0), g(cap + 1, 0);
        g[0] = beta;
        for (std::size_t iteration = 0; iteration < cap; ++iteration) {
            result.iterations = iteration + 1;
            std::vector<double> w = apply_left_preconditioned(a, p, result.basis[iteration]);
            for (std::size_t col = 0; col <= iteration; ++col) {
                double dot = 0;
                for (std::size_t row = 0; row < n; ++row)
                    dot = finite(dot + finite(result.basis[col][row] * w[row]));
                result.hessenberg[col][iteration] = dot;
                for (std::size_t row = 0; row < n; ++row)
                    w[row] = finite(w[row] - finite(dot * result.basis[col][row]));
            }
            const double next_norm = norm2(w);
            result.hessenberg[iteration + 1][iteration] = next_norm;
            const bool breakdown = next_norm == 0;
            if (!breakdown) {
                for (double &value : w)
                    value = finite(value / next_norm);
                result.basis.push_back(w);
            }
            for (std::size_t row = 0; row <= iteration + 1; ++row)
                rotated[row][iteration] = result.hessenberg[row][iteration];
            for (std::size_t row = 0; row < iteration; ++row) {
                const double upper = rotated[row][iteration], lower = rotated[row + 1][iteration];
                rotated[row][iteration] = finite(cosine[row] * upper + sine[row] * lower);
                rotated[row + 1][iteration] = finite(-sine[row] * upper + cosine[row] * lower);
            }
            GmresIteration record;
            record.iteration = iteration + 1;
            record.breakdown = breakdown;
            record.arnoldi_subdiagonal = next_norm;
            const double upper = rotated[iteration][iteration];
            const double radius = finite(std::hypot(upper, rotated[iteration + 1][iteration]));
            bool candidate_defined = radius != 0;
            std::vector<double> candidate;
            if (candidate_defined) {
                cosine[iteration] = finite(upper / radius);
                sine[iteration] = finite(rotated[iteration + 1][iteration] / radius);
                rotated[iteration][iteration] = radius;
                rotated[iteration + 1][iteration] = 0;
                const double old_g = g[iteration];
                g[iteration] = finite(cosine[iteration] * old_g);
                g[iteration + 1] = finite(-sine[iteration] * old_g);
                record.estimated_relative = finite(std::abs(g[iteration + 1]) / denominator);
                candidate_defined =
                    form_candidate(rotated, g, result.basis, initial, iteration + 1, candidate);
            }
            if (!candidate_defined) {
                // Recheck only a previously defined candidate; never invent y.
                record.breakdown = true;
                record.verification_triggered = true;
                record.solution = result.solution;
                store_check(
                    result, record,
                    check_residual(a, p, rhs, preconditioned_rhs, denominator, result.solution));
                record.accepted = record.recomputed_relative <= options.tolerance;
                result.history.push_back(record);
                result.status = record.accepted ? GmresStatus::success : GmresStatus::breakdown;
                result.converged = record.accepted;
                result.message = record.accepted
                                     ? "Previously defined candidate verified at zero pivot"
                                     : "Zero Givens/triangular pivot";
                return result;
            }
            record.solution = candidate;
            store_check(result, record,
                        check_residual(a, p, rhs, preconditioned_rhs, denominator, candidate));
            record.verification_triggered =
                breakdown || record.estimated_relative <= options.tolerance || iteration + 1 == cap;
            record.accepted =
                record.verification_triggered && record.recomputed_relative <= options.tolerance;
            result.solution = candidate;
            result.history.push_back(record);
            if (record.accepted) {
                result.status = GmresStatus::success;
                result.converged = true;
                return result;
            }
            if (breakdown) {
                result.status = GmresStatus::breakdown;
                result.message = "Arnoldi breakdown candidate failed recomputed residual criterion";
                return result;
            }
        }
        result.status = cap < n ? GmresStatus::iteration_limit : GmresStatus::nonconvergence;
        result.message = "Iteration cap reached without verified convergence";
    } catch (const std::overflow_error &error) {
        result.status = GmresStatus::numerical_failure;
        result.message = error.what();
    }
    return result;
}

} // namespace bspai
