#include "bspai/spai.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace bspai {
static_assert(std::numeric_limits<double>::is_iec559 && std::numeric_limits<double>::digits == 53 &&
                  std::numeric_limits<double>::max_exponent == 1024,
              "SPAI requires IEEE binary64 double");
namespace {
double at(const CsrMatrix &a, std::size_t row, std::size_t col) {
    const auto first = a.column_indices().begin() + a.row_offsets()[row];
    const auto last = a.column_indices().begin() + a.row_offsets()[row + 1];
    const auto found = std::lower_bound(first, last, col);
    return found == last || *found != col ? 0.0 : a.values()[found - a.column_indices().begin()];
}

double norm2(const std::vector<double> &v) {
    double result = 0;
    for (double value : v)
        result = std::hypot(result, value);
    return result;
}

// Implicit economy Householder QR: apply the same reflectors to the RHS.
SpaiStatus least_squares(std::vector<double> r, std::vector<double> rhs, std::size_t rows,
                         std::size_t cols, std::vector<double> &solution, double &pivot_ratio) {
    if (rows < cols)
        return SpaiStatus::rank_deficient;
    double inf_norm = 0;
    for (std::size_t i = 0; i < rows; ++i) {
        double sum = 0;
        for (std::size_t j = 0; j < cols; ++j)
            sum += std::abs(r[i * cols + j]);
        inf_norm = std::max(inf_norm, sum);
    }
    double min_pivot = std::numeric_limits<double>::infinity();
    for (std::size_t j = 0; j < cols; ++j) {
        std::vector<double> v(rows - j);
        for (std::size_t i = j; i < rows; ++i)
            v[i - j] = r[i * cols + j];
        const double length = norm2(v);
        if (!std::isfinite(length))
            return SpaiStatus::numerical_failure;
        if (length == 0)
            return SpaiStatus::rank_deficient;
        const double diagonal = -std::copysign(length, v[0]);
        // Normalize first, avoiding x0-alpha overflow for large inputs.
        for (double &value : v)
            value /= length;
        v[0] += std::copysign(1.0, v[0]);
        const double vnorm = norm2(v);
        for (double &value : v)
            value /= vnorm;
        for (std::size_t col = j; col < cols; ++col) {
            double dot = 0;
            for (std::size_t i = j; i < rows; ++i)
                dot += v[i - j] * r[i * cols + col];
            for (std::size_t i = j; i < rows; ++i) {
                r[i * cols + col] -= 2 * v[i - j] * dot;
                if (!std::isfinite(r[i * cols + col]))
                    return SpaiStatus::numerical_failure;
            }
        }
        double dot = 0;
        for (std::size_t i = j; i < rows; ++i)
            dot += v[i - j] * rhs[i];
        for (std::size_t i = j; i < rows; ++i) {
            rhs[i] -= 2 * v[i - j] * dot;
            if (!std::isfinite(rhs[i]))
                return SpaiStatus::numerical_failure;
        }
        r[j * cols + j] = diagonal;
        for (std::size_t i = j + 1; i < rows; ++i)
            r[i * cols + j] = 0;
        min_pivot = std::min(min_pivot, std::abs(diagonal));
    }
    pivot_ratio = inf_norm == 0 ? 0 : min_pivot / inf_norm;
    solution.assign(cols, 0);
    for (std::size_t j = cols; j-- > 0;) {
        double value = rhs[j];
        for (std::size_t k = j + 1; k < cols; ++k)
            value -= r[j * cols + k] * solution[k];
        const double diagonal = r[j * cols + j];
        if (!std::isfinite(diagonal))
            return SpaiStatus::numerical_failure;
        if (diagonal == 0)
            return SpaiStatus::rank_deficient;
        solution[j] = value / diagonal;
        if (!std::isfinite(solution[j]))
            return SpaiStatus::numerical_failure;
    }
    return SpaiStatus::success;
}

SpaiColumn construct_column(const CsrMatrix &b, std::size_t k, const SpaiOptions &options) {
    SpaiColumn column;
    std::vector<std::size_t> pattern(1, k);
    for (;;) {
        column.attempted_pattern = pattern;
        ++column.fit_attempts;
        if (column.fits != 0)
            ++column.growth_attempts;
        SpaiFit fit;
        fit.pattern = pattern;
        for (std::size_t row = 0; row < b.rows(); ++row) {
            for (std::size_t j : pattern) {
                if (at(b, row, j) != 0) {
                    fit.shadow.push_back(row);
                    break;
                }
            }
        }
        const std::size_t rows = fit.shadow.size(), cols = pattern.size();
        if (cols != 0 && rows > std::numeric_limits<std::size_t>::max() / cols) {
            column.status = SpaiStatus::numerical_failure;
            column.message = "Reduced matrix size overflows";
            return column;
        }
        std::vector<double> reduced(rows * cols), rhs(rows, 0);
        for (std::size_t i = 0; i < rows; ++i) {
            rhs[i] = fit.shadow[i] == k ? 1.0 : 0.0;
            for (std::size_t j = 0; j < cols; ++j)
                reduced[i * cols + j] = at(b, fit.shadow[i], pattern[j]);
        }
        const SpaiStatus fit_status =
            least_squares(reduced, rhs, rows, cols, fit.coefficients, fit.qr_pivot_ratio);
        if (fit_status != SpaiStatus::success) {
            column.status = fit_status;
            column.message = "Householder QR/triangular fit failed";
            return column;
        }
        std::vector<double> residual(rows, 0), full_coefficients(b.cols(), 0);
        for (std::size_t i = 0; i < rows; ++i) {
            for (std::size_t j = 0; j < cols; ++j)
                residual[i] += reduced[i * cols + j] * fit.coefficients[j];
            residual[i] -= rhs[i];
        }
        for (std::size_t j = 0; j < cols; ++j)
            full_coefficients[pattern[j]] = fit.coefficients[j];
        std::vector<double> full_residual;
        try {
            full_residual = b.multiply(full_coefficients);
        } catch (const std::overflow_error &) {
            column.status = SpaiStatus::numerical_failure;
            column.message = "Full fitted residual is nonfinite";
            return column;
        }
        full_residual[k] -= 1;
        fit.residual = norm2(residual);
        fit.full_residual = norm2(full_residual);
        if (!std::isfinite(fit.residual) || !std::isfinite(fit.full_residual)) {
            column.status = SpaiStatus::numerical_failure;
            column.message = "Residual norm is nonfinite";
            return column;
        }
        if (column.fits != 0)
            ++column.growths;
        ++column.fits;
        column.pattern = pattern;
        column.coefficients = fit.coefficients;
        column.reduced_residual = fit.residual;
        column.full_residual = fit.full_residual;
        column.qr_pivot_ratio = fit.qr_pivot_ratio;
        column.history.push_back(fit);
        if (fit.residual <= options.epsilon) {
            column.status = SpaiStatus::success;
            return column;
        }
        if (column.growths == options.alpha) {
            column.status = SpaiStatus::iteration_limit;
            column.message = "Requested accuracy not reached within growth/refit budget";
            return column;
        }
        SpaiFit &recorded = column.history.back();
        std::vector<bool> candidate(b.cols(), false);
        for (std::size_t row : fit.shadow)
            for (std::size_t index = b.row_offsets()[row]; index < b.row_offsets()[row + 1];
                 ++index)
                candidate[b.column_indices()[index]] = true;
        // k belongs to the shadow for the verified nonzero-diagonal baseline.
        for (std::size_t index = b.row_offsets()[k]; index < b.row_offsets()[k + 1]; ++index)
            candidate[b.column_indices()[index]] = true;
        for (std::size_t j : pattern)
            candidate[j] = false;
        for (std::size_t j = 0; j < b.cols(); ++j)
            if (candidate[j])
                recorded.candidates.push_back(j);
        if (recorded.candidates.empty()) {
            column.status = SpaiStatus::empty_candidates;
            column.message = "No candidate remains before convergence";
            return column;
        }
        for (std::size_t j : recorded.candidates) {
            double dot = 0, denominator = 0;
            for (std::size_t i = 0; i < rows; ++i) {
                const double value = at(b, fit.shadow[i], j);
                dot += residual[i] * value;
                denominator += value * value;
            }
            const ScoreResult score =
                evaluate_spai_score(fit.residual * fit.residual, dot, denominator);
            if (score.status != SpaiStatus::success) {
                column.status = score.status;
                column.message = "Candidate scoring failed for index " + std::to_string(j);
                return column;
            }
            if (score.clamped)
                ++column.score_clamps;
            recorded.scores.push_back(score.score);
        }
        recorded.mean_score = std::accumulate(recorded.scores.begin(), recorded.scores.end(), 0.0) /
                              static_cast<double>(recorded.scores.size());
        if (!std::isfinite(recorded.mean_score)) {
            column.status = SpaiStatus::numerical_failure;
            column.message = "Candidate mean score is nonfinite";
            return column;
        }
        std::vector<std::size_t> order(recorded.candidates.size());
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](std::size_t i, std::size_t j) {
            return recorded.scores[i] < recorded.scores[j] ||
                   (recorded.scores[i] == recorded.scores[j] &&
                    recorded.candidates[i] < recorded.candidates[j]);
        });
        for (std::size_t i : order) {
            if (recorded.selected.size() == options.beta)
                break;
            if (recorded.scores[i] <= recorded.mean_score)
                recorded.selected.push_back(recorded.candidates[i]);
        }
        if (recorded.selected.empty()) {
            column.status = SpaiStatus::no_growth;
            column.message = "No acceptable growth permitted before convergence";
            return column;
        }
        pattern.insert(pattern.end(), recorded.selected.begin(), recorded.selected.end());
        std::sort(pattern.begin(), pattern.end());
    }
}
} // namespace

const char *spai_status_name(SpaiStatus status) {
    switch (status) {
    case SpaiStatus::success:
        return "success";
    case SpaiStatus::not_attempted:
        return "not_attempted";
    case SpaiStatus::invalid_input:
        return "invalid_input";
    case SpaiStatus::rank_deficient:
        return "rank_deficient";
    case SpaiStatus::numerical_failure:
        return "numerical_failure";
    case SpaiStatus::empty_candidates:
        return "empty_candidates";
    case SpaiStatus::zero_denominator:
        return "zero_denominator";
    case SpaiStatus::invalid_score:
        return "invalid_score";
    case SpaiStatus::no_growth:
        return "no_growth";
    case SpaiStatus::iteration_limit:
        return "iteration_limit";
    }
    return "unknown";
}

ScoreResult evaluate_spai_score(double residual_squared, double dot, double denominator_squared) {
    ScoreResult result;
    if (!std::isfinite(residual_squared) || !std::isfinite(dot) ||
        !std::isfinite(denominator_squared)) {
        result.status = SpaiStatus::numerical_failure;
        return result;
    }
    if (denominator_squared == 0) {
        result.status = SpaiStatus::zero_denominator;
        return result;
    }
    if (residual_squared < 0 || denominator_squared < 0) {
        result.status = SpaiStatus::invalid_score;
        return result;
    }
    const double quotient = dot * dot / denominator_squared;
    double delta = residual_squared - quotient;
    const double bound = 32 * std::ldexp(1.0, -53) * (residual_squared + std::abs(quotient));
    if (!std::isfinite(quotient) || !std::isfinite(delta) || !std::isfinite(bound)) {
        result.status = SpaiStatus::numerical_failure;
        return result;
    }
    if (delta < 0) {
        if (-delta > bound) {
            result.status = SpaiStatus::invalid_score;
            return result;
        }
        delta = 0;
        result.clamped = true;
    }
    result.score = std::sqrt(delta);
    result.status = SpaiStatus::success;
    return result;
}

SpaiResult::SpaiResult(std::size_t n)
    : scaled_transpose(CsrMatrix::from_entries(n, n, {})),
      inverse(CsrMatrix::from_entries(n, n, {})),
      preconditioner(CsrMatrix::from_entries(n, n, {})) {
}

SpaiResult construct_spai(const CsrMatrix &a, const SpaiOptions &options) {
    SpaiResult result(a.rows());
    result.status = SpaiStatus::invalid_input;
    if (a.rows() == 0 || a.rows() != a.cols() || !std::isfinite(options.epsilon) ||
        options.epsilon <= 0) {
        result.message = "Require positive square dimensions and positive finite epsilon";
        return result;
    }
    result.scaling.resize(a.rows());
    for (std::size_t row = 0; row < a.rows(); ++row) {
        if (at(a, row, row) == 0) {
            result.message = "Zero diagonal at row " + std::to_string(row);
            return result;
        }
        double maximum = 0;
        for (std::size_t index = a.row_offsets()[row]; index < a.row_offsets()[row + 1]; ++index)
            maximum = std::max(maximum, std::abs(a.values()[index]));
        const double scale = 1.0 / maximum;
        if (!std::isfinite(scale) || scale == 0) {
            result.message = "Reciprocal row scaling outside binary64 range";
            return result;
        }
        result.scaling[row] = scale;
    }
    std::vector<Entry> entries;
    for (std::size_t row = 0; row < a.rows(); ++row)
        for (std::size_t index = a.row_offsets()[row]; index < a.row_offsets()[row + 1]; ++index) {
            const double value = a.values()[index] * result.scaling[row];
            if (!std::isfinite(value) || value == 0) {
                result.message = "Scaling changes nonzero structure/range";
                return result;
            }
            entries.push_back(Entry{a.column_indices()[index], row, value});
        }
    result.scaled_transpose = CsrMatrix::from_entries(a.rows(), a.rows(), entries);
    result.status = SpaiStatus::success;
    result.columns.resize(a.rows());
    std::vector<Entry> inverse_entries, preconditioner_entries;
    for (std::size_t k = 0; k < a.rows(); ++k) {
        result.columns[k] = construct_column(result.scaled_transpose, k, options);
        const SpaiColumn &column = result.columns[k];
        if (column.status != SpaiStatus::success && result.status == SpaiStatus::success) {
            result.status = column.status;
            result.message = "Column " + std::to_string(k) + ": " + column.message;
        }
        for (std::size_t i = 0; i < column.pattern.size(); ++i) {
            const std::size_t j = column.pattern[i];
            inverse_entries.push_back(Entry{j, k, column.coefficients[i]});
            const double value = column.coefficients[i] * result.scaling[j];
            if (!std::isfinite(value) || (column.coefficients[i] != 0 && value == 0)) {
                result.status = SpaiStatus::numerical_failure;
                result.message = "Preconditioner assembly outside binary64 range";
                return result;
            }
            preconditioner_entries.push_back(Entry{k, j, value});
        }
    }
    result.inverse = CsrMatrix::from_entries(a.rows(), a.rows(), inverse_entries);
    result.preconditioner = CsrMatrix::from_entries(a.rows(), a.rows(), preconditioner_entries);
    return result;
}

} // namespace bspai
