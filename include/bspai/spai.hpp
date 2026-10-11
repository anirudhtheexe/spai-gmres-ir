#ifndef BSPAI_SPAI_HPP
#define BSPAI_SPAI_HPP

#include "bspai/cpu.hpp"
#include <limits>
#include <string>

namespace bspai {

enum class SpaiStatus {
    success, not_attempted, invalid_input, rank_deficient, numerical_failure,
    empty_candidates, zero_denominator, invalid_score, no_growth, iteration_limit
};
const char* spai_status_name(SpaiStatus status);

struct SpaiOptions {
    double epsilon = 0.3;
    std::size_t alpha = 5;
    std::size_t beta = 8;
};

struct ScoreResult {
    SpaiStatus status = SpaiStatus::not_attempted;
    double score = 0;
    bool clamped = false;
};
ScoreResult evaluate_spai_score(double residual_squared, double dot,
                                double denominator_squared);

struct SpaiFit {
    std::vector<std::size_t> pattern, shadow, candidates, selected;
    std::vector<double> coefficients, scores;
    double residual = 0, full_residual = 0, mean_score = 0;
    double qr_pivot_ratio = 0;
};

struct SpaiColumn {
    SpaiStatus status = SpaiStatus::not_attempted;
    std::string message;
    std::vector<std::size_t> pattern, attempted_pattern;
    std::vector<double> coefficients;
    std::size_t fits = 0, growths = 0, fit_attempts = 0, growth_attempts = 0, score_clamps = 0;
    // NaN means no completed fit is available; failure is never zero residual.
    double reduced_residual = std::numeric_limits<double>::quiet_NaN();
    double full_residual = std::numeric_limits<double>::quiet_NaN();
    double qr_pivot_ratio = std::numeric_limits<double>::quiet_NaN();
    std::vector<SpaiFit> history;
};

struct SpaiResult {
    explicit SpaiResult(std::size_t n);
    SpaiStatus status = SpaiStatus::not_attempted;
    std::string message;
    std::vector<double> scaling;
    CsrMatrix scaled_transpose, inverse, preconditioner;
    std::vector<SpaiColumn> columns;
};

// Failed results retain only already fitted coefficients for diagnostics.
// inverse/preconditioner must not be used as a successful SPAI on failure.
// Nonsingularity is an input assumption, not globally certified here.
SpaiResult construct_spai(const CsrMatrix& a, const SpaiOptions& options);

} // namespace bspai
#endif
