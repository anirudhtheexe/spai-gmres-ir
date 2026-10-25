#ifndef BSPAI_FP32_HPP
#define BSPAI_FP32_HPP

// Experiment 2A's intentionally separate uniform-binary32 solver API.
// No type in this header aliases the frozen binary64 production API.
#include <cstddef>
#include <limits>
#include <string>
#include <vector>

namespace bspai {

struct Fp32Entry {
    std::size_t row, col;
    float value;
};

class Fp32CsrMatrix {
  public:
    // Empty canonical matrices are required in explicit failure results.
    Fp32CsrMatrix(std::size_t rows, std::size_t cols);
    static Fp32CsrMatrix from_entries(std::size_t rows, std::size_t cols,
                                      std::vector<Fp32Entry> entries);
    std::size_t rows() const {
        return rows_;
    }
    std::size_t cols() const {
        return cols_;
    }
    const std::vector<std::size_t> &row_offsets() const {
        return offsets_;
    }
    const std::vector<std::size_t> &column_indices() const {
        return columns_;
    }
    const std::vector<float> &values() const {
        return values_;
    }
    std::vector<float> multiply(const std::vector<float> &x) const;
    Fp32CsrMatrix transpose() const;

  private:
    std::size_t rows_, cols_;
    std::vector<std::size_t> offsets_, columns_;
    std::vector<float> values_;
};

enum class Fp32SpaiStatus {
    success,
    invalid_input,
    rank_deficient,
    numerical_failure,
    empty_candidates,
    no_growth,
    iteration_limit
};
const char *fp32_spai_status_name(Fp32SpaiStatus);
struct Fp32SpaiOptions {
    float epsilon = .3f;
    std::size_t alpha = 5, beta = 8;
};
struct Fp32SpaiScore {
    std::size_t candidate = 0;
    float rho = 0;
    bool accepted = false;
};
struct Fp32SpaiColumn {
    Fp32SpaiStatus status = Fp32SpaiStatus::invalid_input;
    std::vector<std::size_t> pattern;
    std::vector<float> coefficients;
    std::size_t fits = 0, growths = 0;
    float reduced_residual = std::numeric_limits<float>::quiet_NaN();
    float full_residual = std::numeric_limits<float>::quiet_NaN();
    std::vector<std::size_t> shadow;
    std::vector<Fp32SpaiScore> score_history;
    std::size_t negative_score_clamps = 0;
    float qr_pivot_ratio = std::numeric_limits<float>::quiet_NaN();
};
struct Fp32SpaiResult {
    explicit Fp32SpaiResult(std::size_t n);
    Fp32SpaiStatus status = Fp32SpaiStatus::invalid_input;
    std::string message;
    std::vector<float> scaling;
    Fp32CsrMatrix scaled_transpose, inverse, preconditioner;
    std::vector<Fp32SpaiColumn> columns;
    std::size_t inverse_nnz = 0, preconditioner_nnz = 0;
};
Fp32SpaiResult fp32_construct_spai(const Fp32CsrMatrix &, const Fp32SpaiOptions &);

enum class Fp32GmresStatus {
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
const char *fp32_gmres_status_name(Fp32GmresStatus);
struct Fp32GmresOptions {
    float tolerance = 1e-4f;
    std::size_t max_iterations = 0;
};
struct Fp32GmresResult {
    Fp32GmresStatus status = Fp32GmresStatus::invalid_input;
    bool converged = false;
    std::string message;
    std::size_t iterations = 0;
    std::vector<float> solution;
    float final_original_residual = std::numeric_limits<float>::quiet_NaN();
    float final_preconditioned_relative = std::numeric_limits<float>::quiet_NaN();
    std::vector<float> hessenberg_residual_history;
    std::vector<float> preconditioned_relative_history;
    std::vector<float> original_residual_history;
};
Fp32GmresResult fp32_left_preconditioned_gmres(const Fp32CsrMatrix &, const Fp32CsrMatrix &,
                                               const std::vector<float> &, const Fp32GmresOptions &,
                                               const std::vector<float> &initial_guess = {});

enum class Fp32IrStatus {
    success,
    invalid_input,
    construction_failure,
    inner_failure,
    stagnation,
    iteration_limit,
    numerical_failure
};
const char *fp32_ir_status_name(Fp32IrStatus);
struct Fp32IrOptions {
    Fp32SpaiOptions spai;
    Fp32GmresOptions gmres;
    std::size_t max_outer_steps = 20;
};
struct Fp32IrDiagnostics {
    float residual_inf = 0, residual_2 = 0, normwise_backward_error = 0;
    std::vector<float> residual;
};
struct Fp32IrResult {
    explicit Fp32IrResult(std::size_t n);
    Fp32IrStatus status = Fp32IrStatus::invalid_input;
    bool converged = false;
    std::string message;
    std::size_t outer_steps = 0, inner_attempts = 0;
    std::vector<float> solution;
    Fp32SpaiResult construction;
    std::vector<Fp32GmresResult> inner_history;
    std::vector<Fp32IrDiagnostics> outer_history;
};
float fp32_ir_backward_tolerance();
Fp32IrDiagnostics fp32_evaluate_ir_diagnostics(const Fp32CsrMatrix &, const std::vector<float> &,
                                               const std::vector<float> &);
Fp32IrResult fp32_refine_with_preconditioner(const Fp32CsrMatrix &, const Fp32CsrMatrix &,
                                             const std::vector<float> &, const Fp32IrOptions &);
Fp32IrResult fp32_spai_gmres_ir(const Fp32CsrMatrix &, const std::vector<float> &,
                                const Fp32IrOptions &);

} // namespace bspai
#endif
