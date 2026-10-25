#include "bspai/fp32.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
namespace bspai {
namespace {
float ck(float x) {
    if (!std::isfinite(x))
        throw std::overflow_error("nonfinite uniform-FP32 arithmetic");
    return x;
}
float n2(const std::vector<float> &x) {
    float s = 0;
    for (float v : x)
        s = ck(std::hypot(s, v));
    return s;
}
float dp(const std::vector<float> &a, const std::vector<float> &b) {
    float s = 0;
    for (std::size_t i = 0; i < a.size(); ++i)
        s = ck(s + a[i] * b[i]);
    return s;
}
std::vector<float> minus(const std::vector<float> &a, const std::vector<float> &b) {
    std::vector<float> r(a.size());
    for (std::size_t i = 0; i < a.size(); ++i)
        r[i] = ck(a[i] - b[i]);
    return r;
}
bool valid(const Fp32CsrMatrix &a) {
    if (!a.rows() || a.rows() != a.cols())
        return false;
    for (float v : a.values())
        if (!std::isfinite(v))
            return false;
    return true;
}
float u() {
    return std::ldexp(1.f, -24);
}
std::size_t offset_count(std::size_t rows) {
    if (rows == std::numeric_limits<std::size_t>::max())
        throw std::overflow_error("FP32 CSR row-offset count overflow");
    return rows + 1;
}
// Packed unpivoted Householder QR; diagonal is the exact computed R pivot.
struct QR {
    std::vector<std::vector<float> > a;
    std::vector<float> tau, diag;
    QR(std::vector<std::vector<float> > x)
        : a(std::move(x)), tau(a.empty() ? 0 : a[0].size()), diag(tau.size()) {
    }
    void factor() {
        for (std::size_t j = 0; j < tau.size(); ++j) {
            float norm = 0;
            for (std::size_t i = j; i < a.size(); ++i)
                norm = ck(std::hypot(norm, a[i][j]));
            if (norm == 0) {
                diag[j] = 0;
                continue;
            }
            float beta = a[j][j] >= 0 ? -norm : norm, v0 = ck(a[j][j] - beta), vv = 1;
            a[j][j] = beta;
            for (std::size_t i = j + 1; i < a.size(); ++i) {
                a[i][j] = ck(a[i][j] / v0);
                vv = ck(vv + a[i][j] * a[i][j]);
            }
            tau[j] = ck(2 / vv);
            diag[j] = beta;
            for (std::size_t k = j + 1; k < tau.size(); ++k) {
                float w = a[j][k];
                for (std::size_t i = j + 1; i < a.size(); ++i)
                    w = ck(w + a[i][j] * a[i][k]);
                w = ck(tau[j] * w);
                a[j][k] = ck(a[j][k] - w);
                for (std::size_t i = j + 1; i < a.size(); ++i)
                    a[i][k] = ck(a[i][k] - a[i][j] * w);
            }
        }
    }
    std::vector<float> solve(std::vector<float> y) const {
        for (std::size_t j = 0; j < tau.size(); ++j) {
            float w = y[j];
            for (std::size_t i = j + 1; i < a.size(); ++i)
                w = ck(w + a[i][j] * y[i]);
            w = ck(tau[j] * w);
            y[j] = ck(y[j] - w);
            for (std::size_t i = j + 1; i < a.size(); ++i)
                y[i] = ck(y[i] - a[i][j] * w);
        }
        std::vector<float> x(tau.size());
        for (std::size_t i = tau.size(); i-- > 0;) {
            if (diag[i] == 0)
                throw std::domain_error("exact-zero computed FP32 QR pivot");
            float s = y[i];
            for (std::size_t j = i + 1; j < tau.size(); ++j)
                s = ck(s - a[i][j] * x[j]);
            x[i] = ck(s / diag[i]);
        }
        return x;
    }
};
}
Fp32CsrMatrix::Fp32CsrMatrix(std::size_t r, std::size_t c)
    : rows_(r), cols_(c), offsets_(offset_count(r), 0) {
}
Fp32CsrMatrix Fp32CsrMatrix::from_entries(std::size_t r, std::size_t c, std::vector<Fp32Entry> e) {
    for (const auto &x : e)
        if (x.row >= r || x.col >= c || !std::isfinite(x.value))
            throw std::invalid_argument("invalid FP32 CSR entry");
    std::sort(e.begin(), e.end(), [](const Fp32Entry &a, const Fp32Entry &b) {
        return a.row != b.row ? a.row < b.row : a.col < b.col;
    });
    std::vector<Fp32Entry> z;
    for (std::size_t i = 0; i < e.size();) {
        std::size_t j = i + 1;
        float s = e[i].value;
        while (j < e.size() && e[j].row == e[i].row && e[j].col == e[i].col) {
            s = ck(s + e[j].value);
            ++j;
        }
        if (s != 0)
            z.push_back({e[i].row, e[i].col, s});
        i = j;
    }
    Fp32CsrMatrix out(r, c);
    for (const auto &x : z)
        ++out.offsets_[x.row + 1];
    for (std::size_t i = 0; i < r; ++i)
        out.offsets_[i + 1] += out.offsets_[i];
    for (const auto &x : z) {
        out.columns_.push_back(x.col);
        out.values_.push_back(x.value);
    }
    return out;
}
std::vector<float> Fp32CsrMatrix::multiply(const std::vector<float> &x) const {
    if (x.size() != cols_)
        throw std::invalid_argument("FP32 SpMV size");
    for (float value : x)
        if (!std::isfinite(value))
            throw std::invalid_argument("nonfinite FP32 SpMV input");
    std::vector<float> y(rows_);
    for (std::size_t i = 0; i < rows_; ++i)
        for (std::size_t q = offsets_[i]; q < offsets_[i + 1]; ++q)
            y[i] = ck(y[i] + values_[q] * x[columns_[q]]);
    return y;
}
Fp32CsrMatrix Fp32CsrMatrix::transpose() const {
    std::vector<Fp32Entry> e;
    for (std::size_t i = 0; i < rows_; ++i)
        for (std::size_t q = offsets_[i]; q < offsets_[i + 1]; ++q)
            e.push_back({columns_[q], i, values_[q]});
    return from_entries(cols_, rows_, std::move(e));
}
Fp32SpaiResult::Fp32SpaiResult(std::size_t n)
    : scaled_transpose(n, n), inverse(n, n), preconditioner(n, n), columns(n) {
}
const char *fp32_spai_status_name(Fp32SpaiStatus s) {
    static const char *n[] = {"success",           "invalid_input",    "rank_deficient",
                              "numerical_failure", "empty_candidates", "no_growth",
                              "iteration_limit"};
    return n[(int)s];
}
Fp32SpaiResult fp32_construct_spai(const Fp32CsrMatrix &a, const Fp32SpaiOptions &o) {
    Fp32SpaiResult out(a.rows());
    const std::size_t n = a.rows();
    if (!valid(a) || !std::isfinite(o.epsilon) || o.epsilon <= 0 || !o.beta) {
        out.message = "invalid FP32 SPAI input";
        return out;
    }
    try {
        out.scaling.resize(n);
        std::vector<Fp32Entry> be;
        for (std::size_t i = 0; i < n; ++i) {
            float mx = 0;
            bool d = false;
            for (std::size_t q = a.row_offsets()[i]; q < a.row_offsets()[i + 1]; ++q) {
                mx = std::max(mx, std::fabs(a.values()[q]));
                d = d || (a.column_indices()[q] == i && a.values()[q] != 0);
            }
            if (!d || mx == 0) {
                out.message = "zero diagonal or row";
                return out;
            }
            out.scaling[i] = ck(1 / mx);
            for (std::size_t q = a.row_offsets()[i]; q < a.row_offsets()[i + 1]; ++q) {
                float v = ck(a.values()[q] * out.scaling[i]);
                if (a.values()[q] != 0 && v == 0) {
                    out.message = "FP32 scaling underflow changes pattern";
                    return out;
                }
                be.push_back({a.column_indices()[q], i, v});
            }
        }
        out.scaled_transpose = Fp32CsrMatrix::from_entries(n, n, std::move(be));
        const Fp32CsrMatrix &b = out.scaled_transpose;
        const Fp32CsrMatrix bt = b.transpose();
        std::vector<Fp32Entry> ie;
        for (std::size_t k = 0; k < n; ++k) {
            Fp32SpaiColumn &c = out.columns[k];
            std::vector<std::size_t> J(1, k);
            for (;;) {
                std::vector<bool> mark(n);
                for (std::size_t j : J)
                    for (std::size_t q = bt.row_offsets()[j]; q < bt.row_offsets()[j + 1]; ++q)
                        mark[bt.column_indices()[q]] = true;
                c.shadow.clear();
                for (std::size_t i = 0; i < n; ++i)
                    if (mark[i])
                        c.shadow.push_back(i);
                std::vector<std::vector<float> > M(c.shadow.size(), std::vector<float>(J.size()));
                for (std::size_t jj = 0; jj < J.size(); ++jj)
                    for (std::size_t q = bt.row_offsets()[J[jj]]; q < bt.row_offsets()[J[jj] + 1];
                         ++q) {
                        std::size_t i = bt.column_indices()[q],
                                    p = std::lower_bound(c.shadow.begin(), c.shadow.end(), i) -
                                        c.shadow.begin();
                        M[p][jj] = bt.values()[q];
                    }
                QR qr(M);
                qr.factor();
                float mn = std::numeric_limits<float>::infinity(), ni = 0;
                for (const auto &row : M) {
                    float s = 0;
                    for (float v : row)
                        s = ck(s + std::fabs(v));
                    ni = std::max(ni, s);
                }
                for (float v : qr.diag) {
                    if (v == 0) {
                        c.status = Fp32SpaiStatus::rank_deficient;
                        out.status = c.status;
                        out.message = "exact-zero computed FP32 QR pivot";
                        return out;
                    }
                    mn = std::min(mn, std::fabs(v));
                }
                c.qr_pivot_ratio = ck(mn / ni);
                std::vector<float> e(c.shadow.size());
                e[std::lower_bound(c.shadow.begin(), c.shadow.end(), k) - c.shadow.begin()] = 1;
                std::vector<float> x = qr.solve(e), s(c.shadow.size());
                for (std::size_t i = 0; i < s.size(); ++i) {
                    float t = 0;
                    for (std::size_t j = 0; j < J.size(); ++j)
                        t = ck(t + M[i][j] * x[j]);
                    s[i] = ck(t - e[i]);
                }
                std::vector<float> full(n);
                for (std::size_t j = 0; j < J.size(); ++j)
                    for (std::size_t q = bt.row_offsets()[J[j]]; q < bt.row_offsets()[J[j] + 1];
                         ++q)
                        full[bt.column_indices()[q]] =
                            ck(full[bt.column_indices()[q]] + bt.values()[q] * x[j]);
                full[k] = ck(full[k] - 1);
                c.pattern = J;
                c.coefficients = x;
                c.reduced_residual = n2(s);
                c.full_residual = n2(full);
                ++c.fits;
                if (c.reduced_residual <= o.epsilon) {
                    c.status = Fp32SpaiStatus::success;
                    break;
                }
                if (c.growths == o.alpha) {
                    c.status = Fp32SpaiStatus::iteration_limit;
                    out.status = c.status;
                    out.message = "FP32 SPAI growth limit";
                    return out;
                }
                std::vector<bool> C(n);
                for (std::size_t i : c.shadow)
                    for (std::size_t q = b.row_offsets()[i]; q < b.row_offsets()[i + 1]; ++q)
                        C[b.column_indices()[q]] = true;
                for (std::size_t j : J)
                    C[j] = false;
                std::vector<Fp32SpaiScore> score;
                float ss = ck(c.reduced_residual * c.reduced_residual);
                for (std::size_t j = 0; j < n; ++j)
                    if (C[j]) {
                        float num = 0, den = 0;
                        for (std::size_t ii = 0; ii < c.shadow.size(); ++ii) {
                            float v = 0;
                            for (std::size_t q = b.row_offsets()[c.shadow[ii]];
                                 q < b.row_offsets()[c.shadow[ii] + 1]; ++q)
                                if (b.column_indices()[q] == j) {
                                    v = b.values()[q];
                                    break;
                                }
                            num = ck(num + s[ii] * v);
                            den = ck(den + v * v);
                        }
                        if (den == 0) {
                            c.status = Fp32SpaiStatus::numerical_failure;
                            out.status = c.status;
                            out.message = "zero restricted candidate denominator";
                            return out;
                        }
                        float quot = ck(num * num / den), delta = ck(ss - quot),
                              bound = ck(32 * u() * (ss + std::fabs(quot)));
                        if (delta < 0) {
                            if (-delta > bound) {
                                c.status = Fp32SpaiStatus::numerical_failure;
                                out.status = c.status;
                                out.message = "invalid FP32 candidate score";
                                return out;
                            }
                            delta = 0;
                            ++c.negative_score_clamps;
                        }
                        score.push_back({j, std::sqrt(delta), false});
                    }
                if (score.empty()) {
                    c.status = Fp32SpaiStatus::empty_candidates;
                    out.status = c.status;
                    out.message = "no candidate remains";
                    return out;
                }
                float mean = 0;
                for (const auto &v : score)
                    mean = ck(mean + v.rho);
                mean = ck(mean / score.size());
                std::sort(score.begin(), score.end(),
                          [](const Fp32SpaiScore &a, const Fp32SpaiScore &b) {
                              return a.rho != b.rho ? a.rho < b.rho : a.candidate < b.candidate;
                          });
                std::size_t add = 0;
                for (auto &v : score)
                    if (add < o.beta && v.rho <= mean) {
                        v.accepted = true;
                        J.push_back(v.candidate);
                        ++add;
                    }
                c.score_history.insert(c.score_history.end(), score.begin(), score.end());
                if (!add) {
                    c.status = Fp32SpaiStatus::no_growth;
                    out.status = c.status;
                    out.message = "no acceptable candidate growth";
                    return out;
                }
                std::sort(J.begin(), J.end());
                ++c.growths;
            }
            for (std::size_t j = 0; j < c.pattern.size(); ++j)
                ie.push_back({c.pattern[j], k, c.coefficients[j]});
        }
        out.inverse = Fp32CsrMatrix::from_entries(n, n, std::move(ie));
        out.inverse_nnz = out.inverse.values().size();
        std::vector<Fp32Entry> pe;
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t q = out.inverse.row_offsets()[i]; q < out.inverse.row_offsets()[i + 1];
                 ++q)
                pe.push_back({out.inverse.column_indices()[q], i,
                              ck(out.inverse.values()[q] * out.scaling[i])});
        out.preconditioner = Fp32CsrMatrix::from_entries(n, n, std::move(pe));
        out.preconditioner_nnz = out.preconditioner.values().size();
        out.status = Fp32SpaiStatus::success;
        return out;
    } catch (const std::exception &e) {
        out.status = Fp32SpaiStatus::numerical_failure;
        out.message = e.what();
        return out;
    }
}
const char *fp32_gmres_status_name(Fp32GmresStatus s) {
    static const char *n[] = {"success",
                              "zero_rhs",
                              "exact_initial_solution",
                              "invalid_input",
                              "zero_preconditioned_rhs",
                              "breakdown",
                              "iteration_limit",
                              "nonconvergence",
                              "numerical_failure"};
    return n[(int)s];
}
Fp32GmresResult fp32_left_preconditioned_gmres(const Fp32CsrMatrix &a, const Fp32CsrMatrix &p,
                                               const std::vector<float> &b,
                                               const Fp32GmresOptions &o,
                                               const std::vector<float> &ig) {
    Fp32GmresResult r;
    const std::size_t n = a.rows();
    if (!valid(a) || p.rows() != n || p.cols() != n || b.size() != n ||
        (!ig.empty() && ig.size() != n) || !std::isfinite(o.tolerance) || o.tolerance <= 0) {
        r.message = "invalid FP32 GMRES input";
        return r;
    }
    try {
        std::size_t cap = o.max_iterations ? o.max_iterations : n;
        if (!cap || cap > n) {
            r.message = "invalid FP32 GMRES cap";
            return r;
        }
        std::vector<float> x = ig.empty() ? std::vector<float>(n) : ig,
                           orig = minus(b, a.multiply(x));
        r.final_original_residual = n2(orig);
        if (r.final_original_residual == 0) {
            r.status =
                n2(b) == 0 ? Fp32GmresStatus::zero_rhs : Fp32GmresStatus::exact_initial_solution;
            r.converged = true;
            r.solution = x;
            r.final_preconditioned_relative = 0;
            return r;
        }
        std::vector<float> pb = p.multiply(b);
        float den = n2(pb);
        if (den == 0) {
            r.status = Fp32GmresStatus::zero_preconditioned_rhs;
            r.solution = x;
            return r;
        }
        std::vector<float> res = minus(pb, p.multiply(a.multiply(x)));
        float beta = n2(res);
        if (ck(beta / den) <= o.tolerance) {
            r.status = Fp32GmresStatus::success;
            r.converged = true;
            r.solution = x;
            r.final_preconditioned_relative = ck(beta / den);
            return r;
        }
        std::vector<std::vector<float> > v(1, res), h(cap + 1, std::vector<float>(cap));
        for (float &z : v[0])
            z = ck(z / beta);
        std::vector<float> cs(cap), sn(cap), g(cap + 1);
        g[0] = beta;
        for (std::size_t k = 0; k < cap; ++k) {
            std::vector<float> w = p.multiply(a.multiply(v[k]));
            for (std::size_t j = 0; j <= k; ++j) {
                h[j][k] = dp(v[j], w);
                for (std::size_t i = 0; i < n; ++i)
                    w[i] = ck(w[i] - h[j][k] * v[j][i]);
            }
            h[k + 1][k] = n2(w);
            bool breakdown = h[k + 1][k] == 0;
            if (!breakdown) {
                for (float &z : w)
                    z = ck(z / h[k + 1][k]);
                v.push_back(w);
            }
            for (std::size_t j = 0; j < k; ++j) {
                float q = h[j][k], z = h[j + 1][k];
                h[j][k] = ck(cs[j] * q + sn[j] * z);
                h[j + 1][k] = ck(-sn[j] * q + cs[j] * z);
            }
            float rad = std::hypot(h[k][k], h[k + 1][k]);
            if (rad == 0) {
                r.status = Fp32GmresStatus::breakdown;
                r.solution = x;
                return r;
            }
            cs[k] = ck(h[k][k] / rad);
            sn[k] = ck(h[k + 1][k] / rad);
            h[k][k] = rad;
            g[k + 1] = ck(-sn[k] * g[k]);
            g[k] = ck(cs[k] * g[k]);
            r.hessenberg_residual_history.push_back(std::fabs(g[k + 1]));
            std::vector<float> y(k + 1);
            for (std::size_t i = k + 1; i-- > 0;) {
                if (h[i][i] == 0) {
                    r.status = Fp32GmresStatus::breakdown;
                    r.solution = x;
                    return r;
                }
                float s = g[i];
                for (std::size_t j = i + 1; j <= k; ++j)
                    s = ck(s - h[i][j] * y[j]);
                y[i] = ck(s / h[i][i]);
            }
            std::vector<float> z = x;
            for (std::size_t j = 0; j <= k; ++j)
                for (std::size_t i = 0; i < n; ++i)
                    z[i] = ck(z[i] + v[j][i] * y[j]);
            orig = minus(b, a.multiply(z));
            r.final_original_residual = n2(orig);
            r.final_preconditioned_relative = ck(n2(minus(pb, p.multiply(a.multiply(z)))) / den);
            r.original_residual_history.push_back(r.final_original_residual);
            r.preconditioned_relative_history.push_back(r.final_preconditioned_relative);
            r.iterations = k + 1;
            r.solution = z;
            if (r.final_preconditioned_relative <= o.tolerance) {
                r.status = Fp32GmresStatus::success;
                r.converged = true;
                return r;
            }
            if (breakdown) {
                r.status = Fp32GmresStatus::breakdown;
                return r;
            }
        }
        r.status = cap < n ? Fp32GmresStatus::iteration_limit : Fp32GmresStatus::nonconvergence;
        return r;
    } catch (const std::exception &e) {
        r.status = Fp32GmresStatus::numerical_failure;
        r.message = e.what();
        return r;
    }
}
const char *fp32_ir_status_name(Fp32IrStatus s) {
    static const char *n[] = {"success",          "invalid_input", "construction_failure",
                              "inner_failure",    "stagnation",    "iteration_limit",
                              "numerical_failure"};
    return n[(int)s];
}
Fp32IrResult::Fp32IrResult(std::size_t n) : construction(n) {
}
float fp32_ir_backward_tolerance() {
    return 10 * u();
}
Fp32IrDiagnostics fp32_evaluate_ir_diagnostics(const Fp32CsrMatrix &a, const std::vector<float> &b,
                                               const std::vector<float> &x) {
    if (!valid(a) || b.size() != a.rows() || x.size() != a.cols())
        throw std::invalid_argument("invalid FP32 IR diagnostics input");
    Fp32IrDiagnostics d;
    d.residual = minus(b, a.multiply(x));
    d.residual_2 = n2(d.residual);
    float an = 0, xn = 0, bn = 0;
    for (std::size_t i = 0; i < a.rows(); ++i) {
        float rs = 0;
        for (std::size_t q = a.row_offsets()[i]; q < a.row_offsets()[i + 1]; ++q)
            rs = ck(rs + std::fabs(a.values()[q]));
        an = std::max(an, rs);
        d.residual_inf = std::max(d.residual_inf, std::fabs(d.residual[i]));
        xn = std::max(xn, std::fabs(x[i]));
        bn = std::max(bn, std::fabs(b[i]));
    }
    float den = ck(an * xn + bn);
    d.normwise_backward_error =
        den == 0 ? (d.residual_inf == 0 ? 0 : std::numeric_limits<float>::infinity())
                 : ck(d.residual_inf / den);
    return d;
}
Fp32IrResult fp32_refine_with_preconditioner(const Fp32CsrMatrix &a, const Fp32CsrMatrix &p,
                                             const std::vector<float> &b, const Fp32IrOptions &o) {
    Fp32IrResult r(a.rows());
    if (!valid(a) || p.rows() != a.rows() || p.cols() != a.cols() || b.size() != a.rows() ||
        o.max_outer_steps > 20) {
        r.message = "invalid FP32 IR input";
        return r;
    }
    try {
        r.solution = p.multiply(b);
        for (std::size_t s = 0; s <= o.max_outer_steps; ++s) {
            Fp32IrDiagnostics d = fp32_evaluate_ir_diagnostics(a, b, r.solution);
            r.outer_history.push_back(d);
            if (d.normwise_backward_error <= fp32_ir_backward_tolerance()) {
                r.status = Fp32IrStatus::success;
                r.converged = true;
                return r;
            }
            if (s == o.max_outer_steps)
                break;
            Fp32GmresResult g = fp32_left_preconditioned_gmres(a, p, d.residual, o.gmres);
            r.inner_history.push_back(g);
            ++r.inner_attempts;
            if (!g.converged) {
                r.status = Fp32IrStatus::inner_failure;
                r.message = fp32_gmres_status_name(g.status);
                return r;
            }
            std::vector<float> old = r.solution;
            for (std::size_t i = 0; i < a.rows(); ++i)
                r.solution[i] = ck(r.solution[i] + g.solution[i]);
            ++r.outer_steps;
            Fp32IrDiagnostics after = fp32_evaluate_ir_diagnostics(a, b, r.solution);
            if (after.normwise_backward_error <= fp32_ir_backward_tolerance()) {
                r.outer_history.push_back(after);
                r.status = Fp32IrStatus::success;
                r.converged = true;
                return r;
            }
            if (r.solution == old) {
                r.outer_history.push_back(after);
                r.status = Fp32IrStatus::stagnation;
                return r;
            }
        }
        r.status = Fp32IrStatus::iteration_limit;
        return r;
    } catch (const std::exception &e) {
        r.status = Fp32IrStatus::numerical_failure;
        r.message = e.what();
        return r;
    }
}
Fp32IrResult fp32_spai_gmres_ir(const Fp32CsrMatrix &a, const std::vector<float> &b,
                                const Fp32IrOptions &o) {
    Fp32IrResult r(a.rows());
    r.construction = fp32_construct_spai(a, o.spai);
    if (r.construction.status != Fp32SpaiStatus::success) {
        r.status = Fp32IrStatus::construction_failure;
        r.message = r.construction.message;
        return r;
    }
    Fp32IrResult q = fp32_refine_with_preconditioner(a, r.construction.preconditioner, b, o);
    q.construction = r.construction;
    return q;
}
} // namespace bspai
