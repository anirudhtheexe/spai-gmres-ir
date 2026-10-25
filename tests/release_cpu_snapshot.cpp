// Release-admission snapshot only.  It does not contain assertions or change
// solver policy; Decimal80 validation is performed by check_release_cpu_decimal.py.
#include "bspai/ir.hpp"
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace {
typedef std::vector<std::vector<double> > Dense;

bspai::CsrMatrix matrix(const Dense &a) {
    std::vector<bspai::Entry> entries;
    for (std::size_t i = 0; i < a.size(); ++i)
        for (std::size_t j = 0; j < a[i].size(); ++j)
            if (a[i][j] != 0)
                entries.push_back(bspai::Entry{i, j, a[i][j]});
    return bspai::CsrMatrix::from_entries(a.size(), a[0].size(), entries);
}
void text(const std::string &value) {
    std::cout << '"';
    for (std::size_t i = 0; i < value.size(); ++i) {
        const char c = value[i];
        if (c == '"' || c == '\\')
            std::cout << '\\';
        std::cout << c;
    }
    std::cout << '"';
}
void number(double value) {
    if (!std::isfinite(value))
        throw std::runtime_error("Snapshot requires finite binary64 values");
    // The hexadecimal value round-trips through C/Python.  The appended raw
    // IEEE-754 payload makes that claim independently checkable, including -0.
    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof bits);
    // Old MinGW/MSVCRT hexfloat formatting is broken (and can crash). Build the
    // exact hexadecimal literal from IEEE bits using only integer formatting.
    const unsigned exponent = static_cast<unsigned>((bits >> 52) & 0x7ff);
    const std::uint64_t fraction = bits & UINT64_C(0x000fffffffffffff);
    std::ostringstream out;
    if (bits >> 63)
        out << '-';
    out << "0x" << (exponent ? '1' : '0') << '.' << std::hex << std::setw(13) << std::setfill('0')
        << fraction << 'p' << std::dec << (exponent ? static_cast<int>(exponent) - 1023 : -1022)
        << '@' << std::hex << bits;
    text(out.str());
}
void indices(const std::vector<std::size_t> &values) {
    std::cout << '[';
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i)
            std::cout << ',';
        std::cout << values[i];
    }
    std::cout << ']';
}
void values(const std::vector<double> &values) {
    std::cout << '[';
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i)
            std::cout << ',';
        number(values[i]);
    }
    std::cout << ']';
}
void vectors(const std::vector<std::vector<double> > &rows) {
    std::cout << '[';
    for (std::size_t i = 0; i < rows.size(); ++i) {
        if (i)
            std::cout << ',';
        values(rows[i]);
    }
    std::cout << ']';
}
void csr(const bspai::CsrMatrix &a) {
    std::cout << "{\"rows\":" << a.rows() << ",\"cols\":" << a.cols() << ",\"offsets\":";
    indices(a.row_offsets());
    std::cout << ",\"indices\":";
    indices(a.column_indices());
    std::cout << ",\"values\":";
    values(a.values());
    std::cout << '}';
}
void fit(const bspai::SpaiFit &f) {
    std::cout << "{\"J\":";
    indices(f.pattern);
    std::cout << ",\"I\":";
    indices(f.shadow);
    std::cout << ",\"coefficients\":";
    values(f.coefficients);
    std::cout << ",\"candidates\":";
    indices(f.candidates);
    std::cout << ",\"scores\":";
    values(f.scores);
    std::cout << ",\"selected\":";
    indices(f.selected);
    std::cout << ",\"residual\":";
    number(f.residual);
    std::cout << ",\"full_residual\":";
    number(f.full_residual);
    std::cout << ",\"mean_score\":";
    number(f.mean_score);
    std::cout << ",\"pivot_ratio\":";
    number(f.qr_pivot_ratio);
    std::cout << '}';
}
void construction(const bspai::SpaiResult &r) {
    std::cout << "{\"status\":";
    text(bspai::spai_status_name(r.status));
    std::cout << ",\"scaling\":";
    values(r.scaling);
    std::cout << ",\"B\":";
    csr(r.scaled_transpose);
    std::cout << ",\"N\":";
    csr(r.inverse);
    std::cout << ",\"P\":";
    csr(r.preconditioner);
    std::cout << ",\"columns\":[";
    for (std::size_t i = 0; i < r.columns.size(); ++i) {
        if (i)
            std::cout << ',';
        const bspai::SpaiColumn &c = r.columns[i];
        std::cout << "{\"status\":";
        text(bspai::spai_status_name(c.status));
        std::cout << ",\"pattern\":";
        indices(c.pattern);
        std::cout << ",\"coefficients\":";
        values(c.coefficients);
        std::cout << ",\"history\":[";
        for (std::size_t j = 0; j < c.history.size(); ++j) {
            if (j)
                std::cout << ',';
            fit(c.history[j]);
        }
        std::cout << "]}";
    }
    std::cout << "]}";
}
void gmres_iteration(const bspai::GmresIteration &h) {
    std::cout << "{\"iteration\":" << h.iteration << ",\"solution\":";
    values(h.solution);
    std::cout << ",\"estimate\":";
    number(h.estimated_relative);
    std::cout << ",\"recomputed\":";
    number(h.recomputed_relative);
    std::cout << ",\"original\":";
    number(h.original_residual);
    std::cout << ",\"subdiagonal\":";
    number(h.arnoldi_subdiagonal);
    std::cout << ",\"breakdown\":" << (h.breakdown ? "true" : "false")
              << ",\"verified\":" << (h.verification_triggered ? "true" : "false")
              << ",\"accepted\":" << (h.accepted ? "true" : "false") << '}';
}
void gmres(const bspai::GmresResult &r) {
    std::cout << "{\"status\":";
    text(bspai::gmres_status_name(r.status));
    std::cout << ",\"converged\":" << (r.converged ? "true" : "false")
              << ",\"iterations\":" << r.iterations;
    std::cout << ",\"solution\":";
    values(r.solution);
    std::cout << ",\"initial_original\":";
    number(r.initial_original_residual);
    std::cout << ",\"initial_preconditioned\":";
    number(r.initial_preconditioned_relative);
    std::cout << ",\"final_original\":";
    number(r.final_original_residual);
    std::cout << ",\"final_preconditioned\":";
    number(r.final_preconditioned_relative);
    std::cout << ",\"basis\":";
    vectors(r.basis);
    std::cout << ",\"H\":";
    vectors(r.hessenberg);
    std::cout << ",\"history\":[";
    for (std::size_t i = 0; i < r.history.size(); ++i) {
        if (i)
            std::cout << ',';
        gmres_iteration(r.history[i]);
    }
    std::cout << "]}";
}
void diagnostics(const bspai::IrDiagnostics &d) {
    std::cout << "{\"residual\":";
    values(d.residual);
    std::cout << ",\"inf\":";
    number(d.residual_inf);
    std::cout << ",\"norm2\":";
    number(d.residual_2);
    std::cout << ",\"nbe\":";
    number(d.normwise_backward_error);
    std::cout << ",\"cbe\":";
    number(d.componentwise_backward_error);
    std::cout << '}';
}
void ir(const bspai::IrResult &r) {
    std::cout << "{\"status\":";
    text(bspai::ir_status_name(r.status));
    std::cout << ",\"converged\":" << (r.converged ? "true" : "false")
              << ",\"outer_steps\":" << r.outer_steps << ",\"inner_attempts\":" << r.inner_attempts;
    std::cout << ",\"solution\":";
    values(r.solution);
    std::cout << ",\"history\":[";
    for (std::size_t i = 0; i < r.history.size(); ++i) {
        if (i)
            std::cout << ',';
        std::cout << "{\"step\":" << r.history[i].step << ",\"x\":";
        values(r.history[i].x);
        std::cout << ",\"diagnostics\":";
        diagnostics(r.history[i].diagnostics);
        std::cout << ",\"has_inner\":" << (r.history[i].has_inner ? "true" : "false");
        if (r.history[i].has_inner) {
            std::cout << ",\"inner\":";
            gmres(r.history[i].inner);
        }
        std::cout << '}';
    }
    std::cout << "],\"rhs\":";
    vectors(r.correction_rhs);
    std::cout << ",\"updates\":[";
    for (std::size_t i = 0; i < r.update_applied.size(); ++i) {
        if (i)
            std::cout << ',';
        std::cout << (r.update_applied[i] ? "true" : "false");
    }
    std::cout << "],\"inner_history\":[";
    for (std::size_t i = 0; i < r.inner_history.size(); ++i) {
        if (i)
            std::cout << ',';
        gmres(r.inner_history[i]);
    }
    std::cout << "]}";
}
void emit(const char *name, const bspai::CsrMatrix &a, const std::vector<double> &b,
          const bspai::SpaiOptions &sp, const bspai::CsrMatrix *fixed_p,
          const bspai::GmresResult *gm, const bspai::IrResult *refinement) {
    std::cout << "RELEASE_CPU_SNAPSHOT {\"schema\":2,\"case\":";
    text(name);
    std::cout << ",\"A\":";
    csr(a);
    std::cout << ",\"b\":";
    values(b);
    std::cout << ",\"epsilon\":";
    number(sp.epsilon);
    std::cout << ",\"alpha\":" << sp.alpha << ",\"beta\":" << sp.beta;
    if (fixed_p) {
        std::cout << ",\"fixed_P\":";
        csr(*fixed_p);
    }
    if (gm) {
        std::cout << ",\"gmres\":";
        gmres(*gm);
    }
    if (refinement) {
        if (refinement->construction_performed) {
            std::cout << ",\"construction\":";
            construction(refinement->construction);
        }
        std::cout << ",\"ir\":";
        ir(*refinement);
    }
    std::cout << "}\n";
}
void status_emit(const char *name, const char *family, const char *expected, const char *actual) {
    std::cout << "RELEASE_CPU_SNAPSHOT {\"schema\":2,\"case\":";
    text(name);
    std::cout << ",\"family\":";
    text(family);
    std::cout << ",\"expected_status\":";
    text(expected);
    std::cout << ",\"actual_status\":";
    text(actual);
    std::cout << "}\n";
}
}

int main() {
    try {
        if (!std::numeric_limits<double>::is_iec559 || std::numeric_limits<double>::digits != 53)
            throw std::runtime_error("Release snapshot requires IEEE binary64 double");
        const bspai::CsrMatrix a = matrix(Dense{{4, 1, 0}, {2, 5, 1}, {0, 1, 3}}),
                               identity = matrix(Dense{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}});
        const std::vector<double> b = {2, -5, 7}, general = {.1, -.7, 1.3};
        bspai::SpaiOptions construction_options;
        construction_options.epsilon = 1e-12;
        construction_options.alpha = 2;
        construction_options.beta = 8;
        const bspai::SpaiResult direct = bspai::construct_spai(a, construction_options);
        if (direct.status != bspai::SpaiStatus::success)
            throw std::runtime_error("Snapshot construction fixture unexpectedly failed");
        bspai::GmresOptions gm_options;
        gm_options.tolerance = 1e-8;
        const bspai::GmresResult direct_gmres =
            bspai::left_preconditioned_gmres(a, identity, general, gm_options);
        emit("gmres_identity_nonsymmetric", a, general, construction_options, &identity,
             &direct_gmres, 0);
        // Direct construction is retained rather than discarded by the IR fixture.
        bspai::IrResult direct_holder(3);
        direct_holder.construction = direct;
        direct_holder.construction_performed = true;
        emit("spai_direct_full", a, std::vector<double>{2, -5, 7}, construction_options, 0, 0,
             &direct_holder);
        bspai::IrOptions ir_options;
        ir_options.spai.epsilon = .3;
        ir_options.spai.alpha = 2;
        ir_options.spai.beta = 8;
        ir_options.gmres.tolerance = 1e-8;
        const bspai::IrResult refined = bspai::spai_gmres_ir(a, b, ir_options, {1, -2, 3});
        if (!refined.converged || refined.construction.status != bspai::SpaiStatus::success)
            throw std::runtime_error("Snapshot IR fixture unexpectedly failed");
        emit("spai_gmres_ir", a, b, ir_options.spai, 0, 0, &refined);
        // Force an actual correction, preserving the zero initial correction used by IR.
        bspai::IrOptions forced = ir_options;
        forced.max_outer_steps = 3;
        const bspai::IrResult forced_ir =
            bspai::refine_with_preconditioner(a, identity, b, forced, {1, -2, 3});
        emit("ir_identity_preconditioner_correction", a, b, forced.spai, &identity, 0, &forced_ir);
        // A bounded inner solve must be retained as an explicit no-update failure.
        bspai::IrOptions inner_fail = forced;
        inner_fail.gmres.max_iterations = 1;
        const bspai::IrResult failed_ir =
            bspai::refine_with_preconditioner(a, identity, b, inner_fail, {1, -2, 3});
        emit("ir_inner_iteration_limit", a, b, inner_fail.spai, &identity, 0, &failed_ir);
        bspai::SpaiOptions growth;
        growth.epsilon = 1e-12;
        growth.alpha = 1;
        growth.beta = 1;
        const bspai::CsrMatrix tie = matrix(Dense{{1, 1, 1}, {1, 1, 0}, {1, 0, 1}});
        const bspai::IrResult growth_holder = [&]() {
            bspai::IrResult r(3);
            r.construction = bspai::construct_spai(tie, growth);
            r.construction_performed = true;
            return r;
        }();
        emit("spai_tie_growth", tie, std::vector<double>{1, 0, 0}, growth, 0, 0, &growth_holder);
        const double delta = std::ldexp(1.0, -30);
        const bspai::CsrMatrix near = matrix(Dense{{1, 1}, {1, 1 + delta}});
        bspai::SpaiOptions nearopt;
        nearopt.epsilon = 1e-5;
        nearopt.alpha = 1;
        nearopt.beta = 1;
        bspai::IrResult nearholder(2);
        nearholder.construction = bspai::construct_spai(near, nearopt);
        nearholder.construction_performed = true;
        emit("spai_near_dependent", near, std::vector<double>{1, 1 + delta}, nearopt, 0, 0,
             &nearholder);
        // Status-only cases have no finite diagnostic payload; they preserve the
        // original failure/edge semantics without serializing NaNs as data.
        bspai::SpaiOptions boundary = construction_options;
        boundary.alpha = 0;
        const bspai::SpaiResult initial = bspai::construct_spai(a, boundary);
        boundary.epsilon = initial.columns[0].reduced_residual;
        const bspai::SpaiResult equal = bspai::construct_spai(a, boundary);
        status_emit("spai_epsilon_equal_column0", "spai", "success",
                    bspai::spai_status_name(equal.columns[0].status));
        boundary.epsilon = std::nextafter(boundary.epsilon, 0.0);
        status_emit("spai_epsilon_one_ulp_below", "spai", "iteration_limit",
                    bspai::spai_status_name(bspai::construct_spai(a, boundary).status));
        boundary = construction_options;
        boundary.beta = 0;
        status_emit("spai_beta_zero", "spai", "no_growth",
                    bspai::spai_status_name(bspai::construct_spai(a, boundary).status));
        status_emit(
            "spai_rank_zero_pivot", "spai", "rank_deficient",
            bspai::spai_status_name(
                bspai::construct_spai(matrix(Dense{{1, 2}, {1, 2}}), construction_options).status));
        status_emit(
            "spai_scaling_overflow", "spai", "invalid_input",
            bspai::spai_status_name(
                bspai::construct_spai(matrix(Dense{{1e-320, 0}, {0, 1}}), construction_options)
                    .status));
        const bspai::GmresResult zero =
            bspai::left_preconditioned_gmres(a, identity, std::vector<double>(3, 0), gm_options);
        status_emit("gmres_zero_rhs", "gmres", "zero_rhs", bspai::gmres_status_name(zero.status));
        const bspai::GmresResult exact = bspai::left_preconditioned_gmres(
            a, identity, b, gm_options, std::vector<double>{1, -2, 3});
        status_emit("gmres_exact_initial", "gmres", "exact_initial_solution",
                    bspai::gmres_status_name(exact.status));
        bspai::GmresOptions cap_options;
        cap_options.tolerance = 1e-30;
        cap_options.max_iterations = 1;
        const bspai::GmresResult cap =
            bspai::left_preconditioned_gmres(a, identity, b, cap_options);
        status_emit("gmres_iteration_cap", "gmres", "iteration_limit",
                    bspai::gmres_status_name(cap.status));
        const bspai::IrResult zeroir =
            bspai::refine_with_preconditioner(a, identity, std::vector<double>(3, 0), forced);
        status_emit("ir_zero_rhs", "ir", "success", bspai::ir_status_name(zeroir.status));
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "RELEASE_CPU_SNAPSHOT_ERROR " << e.what() << '\n';
        return 1;
    }
}
