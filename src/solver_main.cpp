// Shared command-line adapter. All arithmetic after input conversion is delegated
// to the existing CPU/CUDA precision-specific solver APIs.
#include "bspai/cpu.hpp"
#include "bspai/ir.hpp"
#ifdef BSPAI_SOLVER_FP32
#include "bspai/fp32.hpp"
#ifdef BSPAI_SOLVER_CUDA
#include "bspai/fp32_cuda_spai.hpp"
#endif
#elif defined(BSPAI_SOLVER_CUDA)
#include "bspai/cuda_ir.hpp"
#endif

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {
std::size_t parse_count(const char *text) {
    const std::string value(text);
    if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
        throw std::invalid_argument("Expected an unsigned integer cap");
    const unsigned long long count = std::stoull(value);
    if (count > std::numeric_limits<std::size_t>::max())
        throw std::out_of_range("Cap exceeds size_t range");
    return static_cast<std::size_t>(count);
}

#ifdef BSPAI_SOLVER_FP32
using Scalar = float;
using Options = bspai::Fp32IrOptions;
bspai::Fp32CsrMatrix solver_matrix(const bspai::CsrMatrix &input) {
    std::vector<bspai::Fp32Entry> entries;
    for (std::size_t row = 0; row < input.rows(); ++row) {
        for (std::size_t q = input.row_offsets()[row]; q < input.row_offsets()[row + 1]; ++q) {
            const double stored = input.values()[q];
            if (std::fabs(stored) > std::numeric_limits<float>::max())
                throw std::invalid_argument("Matrix entry exceeds FP32 range");
            const float rounded = static_cast<float>(stored);
            if (stored != 0 && rounded == 0)
                throw std::invalid_argument("Matrix entry underflows FP32 range");
            entries.push_back({row, input.column_indices()[q], rounded});
        }
    }
    return bspai::Fp32CsrMatrix::from_entries(input.rows(), input.cols(), entries);
}
#else
using Scalar = double;
using Options = bspai::IrOptions;
const bspai::CsrMatrix &solver_matrix(const bspai::CsrMatrix &input) {
    return input;
}
#endif
} // namespace

int main(int argc, char **argv) {
    if (argc < 2 || argc > 5) {
        std::cerr << "Usage: " << argv[0]
                  << " MATRIX.mtx [epsilon=0.3] [outercap=20] [innercap=0(default n)]\n";
        return 2;
    }
    try {
        std::ifstream stream(argv[1]);
        if (!stream)
            throw std::invalid_argument("Cannot open Matrix Market input");
        const auto input = bspai::read_matrix_market(stream);
        if (!input.rows() || input.rows() != input.cols())
            throw std::invalid_argument("Solver requires a nonempty square matrix");
        const auto matrix = solver_matrix(input);
        Options options;
        options.spai.alpha = matrix.rows() / 8 + (matrix.rows() % 8 != 0);
        options.spai.beta = 8;
        if (argc >= 3) {
            const std::string text(argv[2]);
            std::size_t used = 0;
            const double epsilon = std::stod(text, &used);
            if (used != text.size() || !std::isfinite(epsilon) || epsilon <= 0 ||
                epsilon > std::numeric_limits<Scalar>::max())
                throw std::invalid_argument("Require a finite positive epsilon in precision range");
            options.spai.epsilon = static_cast<Scalar>(epsilon);
            if (options.spai.epsilon == 0)
                throw std::invalid_argument("Epsilon underflows precision");
        }
        if (argc >= 4)
            options.max_outer_steps = parse_count(argv[3]);
        if (argc >= 5)
            options.gmres.max_iterations = parse_count(argv[4]);
        if (options.max_outer_steps > 20 || options.gmres.max_iterations > matrix.rows())
            throw std::invalid_argument("Require outercap<=20 and innercap<=n");
        const std::vector<Scalar> rhs(matrix.rows(), Scalar(1) / std::sqrt(Scalar(matrix.rows())));
#ifdef BSPAI_SOLVER_FP32
#ifdef BSPAI_SOLVER_CUDA
        const auto result = bspai::fp32_cuda_spai_gmres_ir(matrix, rhs, options);
#else
        const auto result = bspai::fp32_spai_gmres_ir(matrix, rhs, options);
#endif
        const char *status = bspai::fp32_ir_status_name(result.status);
        const auto diagnostics =
            result.outer_history.empty() ? bspai::Fp32IrDiagnostics() : result.outer_history.back();
#else
#ifdef BSPAI_SOLVER_CUDA
        const auto native = bspai::cuda_gpu_spai_gmres_ir(matrix, rhs, options);
        const auto &result = native.solver;
#else
        const auto result = bspai::spai_gmres_ir(matrix, rhs, options);
#endif
        const char *status = bspai::ir_status_name(result.status);
        const auto diagnostics =
            result.history.empty() ? bspai::IrDiagnostics() : result.history.back().diagnostics;
#endif
        std::cout << std::setprecision(std::numeric_limits<Scalar>::max_digits10) << "precision=FP"
                  << sizeof(Scalar) * 8
#ifdef BSPAI_SOLVER_CUDA
                  << " backend=GPU cpu_outer=true\n"
#else
                  << " backend=CPU\n"
#endif
                  << "n=" << matrix.rows() << " nnz=" << matrix.values().size()
                  << " epsilon=" << options.spai.epsilon << " tau=" << options.gmres.tolerance
                  << '\n'
                  << "status=" << status << " converged=" << result.converged
                  << " outer_steps=" << result.outer_steps << '\n'
                  << "message=" << result.message << '\n';
        if (result.converged)
            std::cout << "residual_inf=" << diagnostics.residual_inf
                      << " backward_error=" << diagnostics.normwise_backward_error << '\n';
        for (std::size_t i = 0; i < result.solution.size(); ++i)
            std::cout << "x[" << i << "]=" << result.solution[i] << '\n';
        return result.converged ? 0 : 1;
    } catch (const std::invalid_argument &error) {
        std::cerr << "Input error: " << error.what() << '\n';
        return 2;
    } catch (const std::out_of_range &error) {
        std::cerr << "Input error: " << error.what() << '\n';
        return 2;
    } catch (const std::exception &error) {
        std::cerr << "Solver error: " << error.what() << '\n';
        return 1;
    }
}
