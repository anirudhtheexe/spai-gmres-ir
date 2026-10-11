#include "bspai/spai.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
int checks = 0, failures = 0;
typedef std::vector<std::vector<long double> > Dense;
const long double eps = std::numeric_limits<double>::epsilon();
void check(bool ok, const std::string& label) {
    ++checks;
    if (!ok) { ++failures; std::cerr << "FAIL: " << label << '\n'; }
}
bool near(long double x, long double y, long double allowance = 768 * eps) {
    return std::isfinite(x) && std::isfinite(y) && std::fabs(x-y) <= allowance * std::max(1.0L, std::fabs(y));
}
Dense dense(const bspai::CsrMatrix& a) {
    Dense out(a.rows(), std::vector<long double>(a.cols(),0));
    for (std::size_t i=0; i<a.rows(); ++i)
        for (std::size_t k=a.row_offsets()[i]; k<a.row_offsets()[i+1]; ++k)
            out[i][a.column_indices()[k]] = a.values()[k];
    return out;
}
bspai::CsrMatrix matrix(const Dense& a) {
    std::vector<bspai::Entry> entries;
    for (std::size_t i=0;i<a.size();++i)
        for (std::size_t j=0;j<a[i].size();++j)
            if(a[i][j]!=0) entries.push_back(bspai::Entry{i,j,static_cast<double>(a[i][j])});
    return bspai::CsrMatrix::from_entries(a.size(),a[0].size(),entries);
}
long double norm(const std::vector<long double>& v) {
    long double result=0;
    for (long double value:v) result=std::hypot(result,value);
    return result;
}
// Wider independent MGS QR oracle for small well-conditioned reduced systems.
// No normal equations: each column is orthogonalized and the triangular system solved.
std::vector<long double> oracle_fit(const Dense& b, const std::vector<std::size_t>& rows,
                                  const std::vector<std::size_t>& cols,std::size_t target) {
    const std::size_t m=rows.size(), n=cols.size();
    Dense q(m,std::vector<long double>(n)),r(n,std::vector<long double>(n));
    for (std::size_t j=0;j<n;++j) {
        std::vector<long double> v(m);
        for(std::size_t i=0;i<m;++i) v[i]=b[rows[i]][cols[j]];
        for(std::size_t k=0;k<j;++k) {
            for(std::size_t i=0;i<m;++i) r[k][j]+=q[i][k]*v[i];
            for(std::size_t i=0;i<m;++i) v[i]-=r[k][j]*q[i][k];
        }
        r[j][j]=norm(v);
        if(r[j][j]==0) throw std::runtime_error("Independent QR oracle encountered zero pivot");
        for(std::size_t i=0;i<m;++i) q[i][j]=v[i]/r[j][j];
    }
    std::vector<long double> x(n,0);
    for(std::size_t j=0;j<n;++j)
        for(std::size_t i=0;i<m;++i) if(rows[i]==target) x[j]+=q[i][j];
    for(std::size_t j=n;j-->0;) {
        for(std::size_t k=j+1;k<n;++k) x[j]-=r[j][k]*x[k];
        x[j]/=r[j][j];
    }
    return x;
}
template<class Result> void audit_result(const Dense& original,const Result& result,double tolerance) {
    const std::size_t n=original.size();
    const Dense b=dense(result.scaled_transpose), inv=dense(result.inverse), p=dense(result.preconditioner);
    for(std::size_t j=0;j<n;++j) {
        long double rowmax=0;
        for(std::size_t i=0;i<n;++i) rowmax=std::max(rowmax,std::fabs(original[j][i]));
        check(near(result.scaling[j],1/rowmax),"rowmax scaling");
        for(std::size_t i=0;i<n;++i) {
            check(near(b[i][j],original[j][i]/rowmax),"scaled-transpose orientation");
            check(near(p[i][j],inv[j][i]*result.scaling[j]),"P=N^T D right scaling");
        }
        const auto& column=result.columns[j];
        check(column.status==bspai::SpaiStatus::success,"successful column status");
        check(column.fits==column.history.size() && column.growths+1==column.fits,"fit/growth counts");
        check(column.fit_attempts==column.fits && column.growth_attempts==column.growths,"completed successful attempt counts");
        check(column.history.front().pattern==std::vector<std::size_t>({j}),"identity initial pattern");
        check(column.pattern==column.history.back().pattern,"assembled solved pattern");
        std::vector<long double> residual(n,0);
        for(std::size_t i=0;i<n;++i) {
            for(std::size_t k=0;k<n;++k) residual[i]+=b[i][k]*inv[k][j];
            if(i==j) residual[i]-=1;
        }
        check(norm(residual)<=tolerance+768*eps,"independent full column residual meets epsilon");
        check(near(column.full_residual,norm(residual)),"recorded full column residual");
        for(const auto& fit:column.history) {
            std::vector<std::size_t> expected_shadow;
            for(std::size_t i=0;i<n;++i) {
                bool occupied=false;
                for(std::size_t col:fit.pattern) occupied=occupied||b[i][col]!=0;
                if(occupied) expected_shadow.push_back(i);
            }
            check(fit.shadow==expected_shadow,"fit shadow");
            const auto reference=oracle_fit(b,fit.shadow,fit.pattern,j);
            check(reference.size()==fit.coefficients.size(),"fit coefficient count");
            for(std::size_t k=0;k<reference.size();++k)
                check(near(fit.coefficients[k],reference[k]),"fit coefficients independent QR oracle");
            std::vector<long double> s(fit.shadow.size(),0);
            for(std::size_t i=0;i<fit.shadow.size();++i) {
                for(std::size_t k=0;k<fit.pattern.size();++k) s[i]+=b[fit.shadow[i]][fit.pattern[k]]*fit.coefficients[k];
                if(fit.shadow[i]==j)s[i]-=1;
            }
            check(near(fit.residual,norm(s)),"recorded reduced residual");
            check(near(fit.full_residual,norm(s)),"fit full residual equals reduced with complete shadow");
            if(!fit.candidates.empty()) {
                check(fit.candidates.size()==fit.scores.size(),"candidate score count");
                long double mean=0;
                for(std::size_t c=0;c<fit.candidates.size();++c) {
                    const std::size_t candidate=fit.candidates[c];
                    check(!std::binary_search(fit.pattern.begin(),fit.pattern.end(),candidate),"existing pattern excluded from candidates");
                    long double dot=0,den=0;
                    for(std::size_t i=0;i<fit.shadow.size();++i) {
                        const long double v=b[fit.shadow[i]][candidate];
                        dot+=s[i]*v;den+=v*v;
                    }
                    const long double residualnorm=norm(s);
                    const long double radicand=std::max(0.0L,residualnorm*residualnorm-dot*dot/den);
                    const long double expected_score=std::sqrt(radicand);
                    check(near(fit.scores[c],expected_score),"restricted candidate score");
                    mean+=fit.scores[c];
                }
                check(near(fit.mean_score,mean/fit.candidates.size()),"mean over complete candidate list");
            }
        }
    }
}
}

int main() {
    using bspai::SpaiStatus;
    try {
        std::cout << std::setprecision(17);
        check(std::numeric_limits<long double>::digits>std::numeric_limits<double>::digits,"wider reference mantissa available");
        const Dense a={{4,1,0},{2,5,1},{0,1,3}};
        bspai::SpaiOptions options;options.epsilon=1e-12;options.alpha=2;options.beta=8;
        const auto result=bspai::construct_spai(matrix(a),options);
        check(result.status==SpaiStatus::success,"tiny full inverse construction");
        if(result.status==SpaiStatus::success) {
            audit_result(a,result,options.epsilon);
            const Dense expected={{14.0L/50,-3.0L/50,1.0L/50},{-6.0L/50,12.0L/50,-4.0L/50},{2.0L/50,-4.0L/50,18.0L/50}};
            const Dense p=dense(result.preconditioner);
            long double max_inverse_error=0, max_full_residual=0;
            for(std::size_t i=0;i<3;++i) for(std::size_t j=0;j<3;++j)
                { check(near(p[i][j],expected[i][j]),"exact rational A inverse");
                  max_inverse_error=std::max(max_inverse_error,std::fabs(p[i][j]-expected[i][j])); }
            const Dense b=dense(result.scaled_transpose),inverse=dense(result.inverse);
            for(std::size_t j=0;j<3;++j) {
                std::vector<long double> r(3,0);
                for(std::size_t i=0;i<3;++i) {
                    for(std::size_t k=0;k<3;++k) r[i]+=b[i][k]*inverse[k][j];
                    if(i==j)r[i]-=1;
                }
                max_full_residual=std::max(max_full_residual,norm(r));
            }
            std::cout << "tiny: nnzN=" << result.inverse.values().size() << ", nnzP=" << result.preconditioner.values().size()
                      << ", max_wide_full_residual=" << max_full_residual << ", max_rational_inverse_abs_error=" << max_inverse_error << '\n';
            for(std::size_t j=0;j<3;++j)
                std::cout << "tiny column" << j << ": fits=" << result.columns[j].fits << ", growths=" << result.columns[j].growths
                          << ", score_clamps=" << result.columns[j].score_clamps << '\n';
            const auto x=result.preconditioner.multiply({2,-5,7});
            check(near(x[0],1)&&near(x[1],-2)&&near(x[2],3),"preconditioner manufactured solution");
            check(result.columns[0].history[0].selected==std::vector<std::size_t>({2}),"restricted first growth chooses candidate2");
        }
        options.alpha=1;
        const auto cap=bspai::construct_spai(matrix(a),options);
        check(cap.status==SpaiStatus::iteration_limit,"alpha1 reports required accuracy cap");
        check(cap.columns[0].fits==2 && cap.columns[0].growths==1,"alpha1 allows initial plus one refit");
        check(cap.columns[0].pattern==std::vector<std::size_t>({0,2}),"cap publishes solved pattern only");
        check(cap.columns[0].coefficients.size()==cap.columns[0].pattern.size(),"cap coefficient/pattern alignment");
        check(cap.columns[0].full_residual>options.epsilon,"cap residual remains unconverged");
        std::cout << "alpha1: status=" << bspai::spai_status_name(cap.status)
                  << ", column0_full_residual=" << cap.columns[0].full_residual << '\n';
        options.alpha=0;
        const auto initial=bspai::construct_spai(matrix(a),options);
        check(initial.status==SpaiStatus::iteration_limit,"alpha0 initial fit failure status");
        check(initial.columns[0].fits==1 && initial.columns[0].growths==0,"alpha0 budget");
        check(initial.columns[0].pattern==std::vector<std::size_t>({0}),"alpha0 identity pattern retained");
        // Equality uses the recorded deterministic residual as a binary64 boundary probe.
        options.epsilon=initial.columns[0].reduced_residual;
        const auto equality=bspai::construct_spai(matrix(a),options);
        check(equality.columns[0].status==SpaiStatus::success,"residual exactly epsilon uses <=");
        options.epsilon=std::nextafter(options.epsilon,0.0);
        const auto below=bspai::construct_spai(matrix(a),options);
        check(below.columns[0].status==SpaiStatus::iteration_limit,"epsilon one ULP below residual does not accept");
        options.epsilon=1e-12;options.alpha=2;options.beta=1;
        const Dense tie={{1,1,1},{1,1,0},{1,0,1}};
        const auto tied=bspai::construct_spai(matrix(tie),options);
        check(!tied.columns[0].history.empty(),"tie history available");
        if(!tied.columns[0].history.empty()) {
            const auto& fit=tied.columns[0].history[0];
            check(fit.candidates==std::vector<std::size_t>({1,2}),"tie candidates sorted");
            check(fit.scores.size()==2 && fit.scores[0]==fit.scores[1],"equal scores actually exercised");
            check(fit.selected==std::vector<std::size_t>({1}),"score tie selects smallest index with beta1");
        }
        options.beta=0;
        check(bspai::construct_spai(matrix(a),options).status==SpaiStatus::no_growth,"beta0 explicit no-growth failure");
        options.beta=8;options.alpha=0;
        const auto diagonal=bspai::construct_spai(matrix({{2,0},{0,-4}}),options);
        check(diagonal.status==SpaiStatus::success,"diagonal succeeds without growth");
        check(diagonal.columns[0].fits==1 && diagonal.columns[1].fits==1,"diagonal identity fits only");
        const Dense dp=dense(diagonal.preconditioner);
        check(near(dp[0][0],.5L)&&near(dp[1][1],-.25L),"diagonal inverse including negative entry");
        options.alpha=3;
        const auto singular=bspai::construct_spai(matrix({{1,1},{1,1}}),options);
        check(singular.status!=SpaiStatus::success && !singular.message.empty(),"tight singular fixture explicit failure (no global rank assertion)");
        std::cout << "singular tight fit: status=" << bspai::spai_status_name(singular.status) << '\n';
        for(const auto& column:singular.columns) {
            check(column.fit_attempts>=column.fits && column.growth_attempts>=column.growths,"failed fit attempted/completed diagnostics");
            check(column.pattern.size()==column.coefficients.size(),"failed fit retains solved coefficient alignment");
            if(column.history.empty()) check(std::isnan(column.full_residual),"unavailable residual marked NaN rather than zero");
        }
        const auto zero_pivot=bspai::construct_spai(matrix({{1,2},{1,2}}),options);
        check(zero_pivot.status==SpaiStatus::rank_deficient,"exact-zero QR pivot fixture failure");
        bool exercised_zero_pivot=false;
        for(const auto& column:zero_pivot.columns) if(column.status==SpaiStatus::rank_deficient) {
            exercised_zero_pivot=true;
            check(column.fit_attempts==2 && column.fits==1 && column.growth_attempts==1 && column.growths==0,
                  "failed QR refit attempted but not completed");
            check(column.pattern.size()==1 && column.coefficients.size()==1 && column.attempted_pattern.size()==2,
                  "failed QR preserves earlier solved pattern and records attempted pattern");
        }
        check(exercised_zero_pivot,"exact-zero QR branch actually exercised");
        std::cout << "exact-zero QR fixture: status=" << bspai::spai_status_name(zero_pivot.status) << '\n';
        check(bspai::construct_spai(matrix({{0,1},{1,1}}),options).status==SpaiStatus::invalid_input,"zero diagonal rejected");
        check(bspai::construct_spai(matrix({{1,2,0},{0,1,2}}),options).status==SpaiStatus::invalid_input,"nonsquare input rejected");
        options.epsilon=0;
        check(bspai::construct_spai(matrix(a),options).status==SpaiStatus::invalid_input,"zero epsilon rejected");
        options.epsilon=-1;
        check(bspai::construct_spai(matrix(a),options).status==SpaiStatus::invalid_input,"negative epsilon rejected");
        options.epsilon=std::numeric_limits<double>::quiet_NaN();
        check(bspai::construct_spai(matrix(a),options).status==SpaiStatus::invalid_input,"NaN epsilon rejected");
        options.epsilon=1e-12;
        check(bspai::construct_spai(matrix({{1e-320L,0},{0,1}}),options).status!=SpaiStatus::success,"reciprocal scaling overflow failure");
        check(bspai::construct_spai(matrix({{1e308L,1e-300L},{0,1}}),options).status!=SpaiStatus::success,"scaled nonzero underflow failure");
        const auto score=bspai::evaluate_spai_score(4,1,2);
        check(score.status==SpaiStatus::success && near(score.score,std::sqrt(3.5L)),"score analytic positive case");
        const auto clamp=bspai::evaluate_spai_score(1,std::nextafter(1.0,2.0),1);
        check(clamp.status==SpaiStatus::success && clamp.clamped && clamp.score==0,"bounded negative radicand clamp");
        const double boundary_dot=1+16*std::numeric_limits<double>::epsilon();
        check(bspai::evaluate_spai_score(1,boundary_dot,1).status==SpaiStatus::success,"score clamp allowed at rounded 32u boundary");
        check(bspai::evaluate_spai_score(1,std::nextafter(boundary_dot,2.0),1).status==SpaiStatus::invalid_score,"score clamp forbidden one ULP past boundary");
        check(bspai::evaluate_spai_score(1,1.1,1).status==SpaiStatus::invalid_score,"large negative radicand failure");
        check(bspai::evaluate_spai_score(1,1,0).status!=SpaiStatus::success,"zero score denominator failure");
        check(bspai::evaluate_spai_score(-1,1,1).status!=SpaiStatus::success,"negative residual squared failure");
        check(bspai::evaluate_spai_score(1,std::numeric_limits<double>::infinity(),1).status!=SpaiStatus::success,"nonfinite score input failure");
        // Normal equations lose the tiny determinant; analytic inverse supplies the oracle.
        const double delta=std::ldexp(1.0,-30);
        options.epsilon=1e-5;options.alpha=1;options.beta=1;
        const auto close=bspai::construct_spai(matrix({{1,1},{1,1+static_cast<long double>(delta)}}),options);
        check(close.status==SpaiStatus::success,"nonsingular near-dependent columns retained without arbitrary rank cutoff");
        if(close.status==SpaiStatus::success) {
            const Dense cp=dense(close.preconditioner);
            const Dense exact={{(1+delta)/delta,-1.0L/delta},{-1.0L/delta,1.0L/delta}};
            long double max_relative_error=0;
            for(std::size_t i=0;i<2;++i)for(std::size_t j=0;j<2;++j)
                { check(near(cp[i][j],exact[i][j],2e-6L),"near-dependent analytic inverse relative accuracy");
                  max_relative_error=std::max(max_relative_error,std::fabs((cp[i][j]-exact[i][j])/exact[i][j])); }
            check(close.columns[0].qr_pivot_ratio>0 && close.columns[0].qr_pivot_ratio<1e-8,"small nonzero QR pivot ratio reported");
            const Dense cb=dense(close.scaled_transpose),cn=dense(close.inverse);
            long double max_wide_residual=0;
            for(std::size_t j=0;j<2;++j) {
                std::vector<long double> r(2,0);
                for(std::size_t i=0;i<2;++i) {
                    for(std::size_t k=0;k<2;++k)r[i]+=cb[i][k]*cn[k][j];
                    if(i==j)r[i]-=1;
                }
                max_wide_residual=std::max(max_wide_residual,norm(r));
            }
            check(max_wide_residual<=options.epsilon,"near-dependent independent full residual meets stress tolerance");
            std::cout << "near-dependent: max_inverse_relative_error=" << max_relative_error
                      << ", pivot_ratio0=" << close.columns[0].qr_pivot_ratio << ", recorded_residual0=" << close.columns[0].full_residual
                      << ", max_wide_full_residual=" << max_wide_residual << '\n';
        }
    } catch(const std::exception& e) { ++failures; std::cerr<<"UNEXPECTED EXCEPTION: "<<e.what()<<'\n'; }
    std::cout<<"Independent SPAI validation: "<<checks<<" checks, "<<failures<<" failures\n";
    return failures?1:0;
}
