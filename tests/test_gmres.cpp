#include "bspai/gmres.hpp"
#include "bspai/spai.hpp"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
typedef std::vector<std::vector<long double> > Dense;
int checks=0, failures=0;
const long double eps=std::numeric_limits<double>::epsilon();
void check(bool ok,const std::string& name) {
    ++checks;if(!ok){++failures;std::cerr<<"FAIL: "<<name<<'\n';}
}
bool near(long double actual,long double expected,long double scale=1) {
    return std::isfinite(actual)&&std::isfinite(expected)&&
           std::fabs(actual-expected)<=768*eps*std::max(scale,std::fabs(expected));
}
bspai::CsrMatrix matrix(const Dense& a) {
    std::vector<bspai::Entry> entries;
    for(std::size_t i=0;i<a.size();++i)for(std::size_t j=0;j<a[i].size();++j)
        if(a[i][j]!=0)entries.push_back({i,j,static_cast<double>(a[i][j])});
    return bspai::CsrMatrix::from_entries(a.size(),a[0].size(),entries);
}
Dense dense(const bspai::CsrMatrix& a) {
    Dense d(a.rows(),std::vector<long double>(a.cols(),0));
    for(std::size_t i=0;i<a.rows();++i)for(std::size_t k=a.row_offsets()[i];k<a.row_offsets()[i+1];++k)
        d[i][a.column_indices()[k]]=a.values()[k];
    return d;
}
Dense identity(std::size_t n) {
    Dense a(n,std::vector<long double>(n,0));for(std::size_t i=0;i<n;++i)a[i][i]=1;return a;
}
std::vector<double> operational_product(const Dense& a,const std::vector<double>& x) {
    std::vector<double> y(a.size(),0);
    for(std::size_t i=0;i<a.size();++i)for(std::size_t j=0;j<x.size();++j)
        if(a[i][j]!=0)y[i]+=static_cast<double>(a[i][j])*x[j];
    return y;
}
std::vector<long double> wider_product(const Dense& a,const std::vector<long double>& x) {
    std::vector<long double> y(a.size(),0);
    for(std::size_t i=0;i<a.size();++i)for(std::size_t j=0;j<x.size();++j)y[i]+=a[i][j]*x[j];
    return y;
}
std::vector<long double> widen(const std::vector<double>& x) { return std::vector<long double>(x.begin(),x.end()); }
long double norm(const std::vector<long double>& x) {
    long double n=0;for(long double value:x)n=std::hypot(n,value);return n;
}
double operational_norm(const std::vector<double>& x) {
    double n=0;for(double value:x)n=std::hypot(n,value);return n;
}
double operational_relative(const Dense& a,const Dense& p,const std::vector<double>& b,const std::vector<double>& x) {
    const auto pb=operational_product(p,b), pax=operational_product(p,operational_product(a,x));
    std::vector<double> residual(pb.size());for(std::size_t i=0;i<pb.size();++i)residual[i]=pb[i]-pax[i];
    const double denominator=operational_norm(pb);
    return denominator==0?(operational_norm(residual)==0?0:std::numeric_limits<double>::infinity()):operational_norm(residual)/denominator;
}
long double original_residual(const Dense& a,const std::vector<double>& b,const std::vector<double>& x) {
    auto r=wider_product(a,widen(x));for(std::size_t i=0;i<r.size();++i)r[i]=b[i]-r[i];return norm(r);
}
long double wider_relative(const Dense& a,const Dense& p,const std::vector<double>& b,const std::vector<double>& x) {
    const auto pb=wider_product(p,widen(b)),pax=wider_product(p,wider_product(a,widen(x)));
    std::vector<long double> r(pb.size());for(std::size_t i=0;i<r.size();++i)r[i]=pb[i]-pax[i];
    return norm(pb)==0?(norm(r)==0?0:std::numeric_limits<long double>::infinity()):norm(r)/norm(pb);
}
std::vector<long double> rational_tiny_solution(const std::vector<double>& b) {
    return wider_product({{14.0L/50,-3.0L/50,1.0L/50},{-6.0L/50,12.0L/50,-4.0L/50},{2.0L/50,-4.0L/50,18.0L/50}},widen(b));
}
template<class Result> void audit(const Dense& a,const Dense& p,const std::vector<double>& b,
                                 const Result& result,double tolerance,std::size_t cap,
                                 const std::vector<long double>* reference=nullptr) {
    check(result.iterations<=cap,"iteration count obeys cap");
    check(result.history.size()==result.iterations,"history records every iteration");
    check(result.solution.size()==b.size(),"solution dimensions");
    if(result.solution.size()!=b.size())return;
    const double relative=operational_relative(a,p,b,result.solution);
    check(near(result.final_preconditioned_relative,relative),"final residual independently operationally recomputed");
    check(near(result.final_original_residual,original_residual(a,b,result.solution),norm(widen(b))+norm(wider_product(a,widen(result.solution)))),
          "final original residual wider reference");
    if(result.converged) {
        check(relative<=tolerance,"accepted final solution satisfies recomputed criterion");
        check(wider_relative(a,p,b,result.solution)<=tolerance+768*eps,"accepted well-conditioned wider residual");
        if(reference) {
            long double error=0,size=0;
            for(std::size_t i=0;i<b.size();++i){error=std::max(error,std::fabs(result.solution[i]-(*reference)[i]));size=std::max(size,std::fabs((*reference)[i]));}
            check(error<=1e-12L*std::max(1.0L,size),"independent reference solution accuracy");
        }
    }
    for(std::size_t j=0;j<result.history.size();++j) {
        const auto& h=result.history[j];
        check(h.iteration==j+1,"history iteration ordinal");
        check(h.solution.size()==b.size(),"history candidate dimensions");
        if(h.solution.size()!=b.size())continue;
        const double own=operational_relative(a,p,b,h.solution);
        check(near(h.recomputed_relative,own),"candidate residual independently recomputed");
        check(near(h.original_residual,original_residual(a,b,h.solution),norm(widen(b))+norm(wider_product(a,widen(h.solution)))),"candidate original wider residual");
        if(h.accepted) {
            check(h.verification_triggered,"acceptance has verification trigger");
            check(own<=tolerance,"acceptance never uses estimate alone");
        }
        check(!h.breakdown || h.arnoldi_subdiagonal==0,"only exact zero Arnoldi subdiagonal breaks down");
    }
    if(!result.history.empty()) {
        check(result.basis.size()>=result.iterations,"Arnoldi basis contains every used vector");
        check(result.hessenberg.size()==cap+1,"raw Hessenberg row dimensions");
        for(const auto& q:result.basis) {
            check(q.size()==b.size(),"basis vector dimensions");
            check(near(norm(widen(q)),1),"basis vector normalization");
        }
        long double max_orthogonality=0,max_relation_defect=0;
        for(std::size_t j=0;j<result.iterations;++j) {
            auto relation=wider_product(p,wider_product(a,widen(result.basis[j])));
            const long double relation_scale=std::max(1.0L,norm(relation));
            for(std::size_t k=0;k<=j+1 && k<result.basis.size();++k)
                for(std::size_t row=0;row<b.size();++row)relation[row]-=result.hessenberg[k][j]*result.basis[k][row];
            max_relation_defect=std::max(max_relation_defect,norm(relation)/relation_scale);
            for(std::size_t k=0;k<j;++k) {
                long double dot=0;for(std::size_t row=0;row<b.size();++row)dot+=static_cast<long double>(result.basis[j][row])*result.basis[k][row];
                max_orthogonality=std::max(max_orthogonality,std::fabs(dot));
            }
        }
        check(max_relation_defect<=768*eps,"independent wider Arnoldi relation");
        // Excludes unused final near-breakdown vector: n+1 vectors cannot be orthogonal in R^n.
        check(max_orthogonality<=1e-12L,"well-conditioned used basis orthogonality");
    }
}
}

int main() {
    using bspai::GmresStatus;
    try {
        std::cout<<std::setprecision(17);
        check(std::numeric_limits<long double>::digits>std::numeric_limits<double>::digits,"wider oracle available");
        bspai::GmresOptions options;options.tolerance=1e-8;
        const Dense i3=identity(3);
        const std::vector<double> general={.1,-.7,1.3};
        const std::vector<long double> identity_reference=widen(general);
        const auto ident=bspai::left_preconditioned_gmres(matrix(i3),matrix(i3),general,options);
        check(ident.converged,"identity general RHS converges");audit(i3,i3,general,ident,options.tolerance,3,&identity_reference);
        const Dense diagonal={{2,0,0},{0,-3,0},{0,0,5}};
        const std::vector<double> diagonal_b={2,6,15};const std::vector<long double> diagonal_ref={1,-2,3};
        const auto diag=bspai::left_preconditioned_gmres(matrix(diagonal),matrix(i3),diagonal_b,options);
        check(diag.converged,"mixed-sign diagonal converges");audit(diagonal,i3,diagonal_b,diag,options.tolerance,3,&diagonal_ref);
        const Dense tiny={{4,1,0},{2,5,1},{0,1,3}};
        const auto truth=rational_tiny_solution(general);
        const auto non=bspai::left_preconditioned_gmres(matrix(tiny),matrix(i3),general,options);
        check(non.converged,"nonsymmetric noninteger RHS converges");audit(tiny,i3,general,non,options.tolerance,3,&truth);
        const auto guessed=bspai::left_preconditioned_gmres(matrix(tiny),matrix(i3),general,options,{.5,-.3,.2});
        check(guessed.converged,"nonzero initial guess converges");audit(tiny,i3,general,guessed,options.tolerance,3,&truth);
        const auto exact=bspai::left_preconditioned_gmres(matrix(tiny),matrix(i3),{2,-5,7},options,{1,-2,3});
        check(exact.converged&&exact.iterations==0&&exact.status==GmresStatus::exact_initial_solution,"exact initial solution before Arnoldi");
        const auto zero=bspai::left_preconditioned_gmres(matrix(tiny),matrix(i3),{0,0,0},options);
        check(zero.converged&&zero.iterations==0&&zero.status==GmresStatus::zero_rhs,"zero RHS zero guess clean acceptance");
        const auto zero_bad=bspai::left_preconditioned_gmres(matrix(tiny),matrix(i3),{0,0,0},options,{1,0,0});
        check(!zero_bad.converged&&zero_bad.status==GmresStatus::zero_preconditioned_rhs,"zero normalization with nonzero original residual failure");
        const Dense zp={{0,0,0},{0,0,0},{0,0,0}};
        const auto annihilated=bspai::left_preconditioned_gmres(matrix(i3),matrix(zp),{1,0,0},options);
        check(!annihilated.converged&&annihilated.status==GmresStatus::zero_preconditioned_rhs,"nonzero RHS annihilated by P fails");
        const auto annihilated_exact=bspai::left_preconditioned_gmres(matrix(i3),matrix(zp),{1,0,0},options,{1,0,0});
        check(annihilated_exact.converged&&annihilated_exact.iterations==0,"original exact solution accepted even P RHS zero");
        // Observe the explicit criterion's limitation outside a nonsingular-P assumption.
        const Dense projection={{1,0},{0,0}};
        const auto projected=bspai::left_preconditioned_gmres(matrix(identity(2)),matrix(projection),{1,1},options,{1,0});
        check(projected.converged&&projected.iterations==0,"singular P nullspace satisfies stated preconditioned criterion");
        check(projected.final_preconditioned_relative==0&&projected.final_original_residual==1,
              "singular P preserves nonzero original residual diagnostic (not an original solve certificate)");
        std::cout<<"singular P limitation: accepted="<<projected.converged<<", original_residual="<<projected.final_original_residual
                 <<", preconditioned_relative="<<projected.final_preconditioned_relative<<'\n';
        const auto happy=bspai::left_preconditioned_gmres(matrix(i3),matrix(i3),{1,0,0},options);
        check(happy.converged&&happy.iterations==1,"exact Arnoldi happy breakdown solves");
        check(happy.history.size()==1&&happy.history[0].breakdown&&happy.history[0].arnoldi_subdiagonal==0,"exact happy breakdown branch exercised");
        const auto unhappy=bspai::left_preconditioned_gmres(matrix(zp),matrix(i3),{1,0,0},options);
        check(!unhappy.converged&&unhappy.status==GmresStatus::breakdown,"zero Hessenberg/Givens singular breakdown failure");
        options.max_iterations=1;
        const auto capped=bspai::left_preconditioned_gmres(matrix(diagonal),matrix(i3),{1,1,1},options);
        check(!capped.converged&&capped.status==GmresStatus::iteration_limit&&capped.iterations==1,"lower cap named nonconvergence");
        audit(diagonal,i3,{1,1,1},capped,options.tolerance,1);
        options.max_iterations=4;
        check(bspai::left_preconditioned_gmres(matrix(tiny),matrix(i3),general,options).status==GmresStatus::invalid_input,"cap greater than dimension rejected");
        options.max_iterations=0;
        check(bspai::left_preconditioned_gmres(matrix({{1,2},{3,4}}),matrix(i3),general,options).status==GmresStatus::invalid_input,"preconditioner dimension mismatch rejected");
        check(bspai::left_preconditioned_gmres(matrix(tiny),matrix(i3),{1,2},options).status==GmresStatus::invalid_input,"RHS mismatch rejected");
        check(bspai::left_preconditioned_gmres(matrix(tiny),matrix(i3),general,options,{1,2}).status==GmresStatus::invalid_input,"initial guess mismatch rejected");
        check(bspai::left_preconditioned_gmres(matrix(tiny),matrix(i3),{1,std::numeric_limits<double>::quiet_NaN(),2},options).status==GmresStatus::invalid_input,"nonfinite RHS rejected");
        options.tolerance=0;
        check(bspai::left_preconditioned_gmres(matrix(tiny),matrix(i3),general,options).status==GmresStatus::invalid_input,"zero tolerance rejected");
        options.tolerance=std::numeric_limits<double>::infinity();
        check(bspai::left_preconditioned_gmres(matrix(tiny),matrix(i3),general,options).status==GmresStatus::invalid_input,"nonfinite tolerance rejected");
        options.tolerance=1e-8;
        const auto overflow=bspai::left_preconditioned_gmres(matrix({{1e308L,0},{0,1}}),matrix({{2,0},{0,1}}),{1,0},options);
        check(!overflow.converged&&overflow.status==GmresStatus::numerical_failure,"operator overflow explicit failure");
        const Dense round_a={{1e16L,1e16L},{1,1+eps}},round_p={{1,-1e16L},{0,1}};
        const auto two=bspai::apply_left_preconditioned(matrix(round_a),matrix(round_p),{1,1});
        const auto own_two=operational_product(round_p,operational_product(round_a,{1,1}));
        check(two==own_two&&two==std::vector<double>({0,2}),"two sequential binary64 products semantics");
        Dense pa(2,std::vector<long double>(2));
        for(std::size_t i=0;i<2;++i)for(std::size_t j=0;j<2;++j) {
            double value=0;for(std::size_t k=0;k<2;++k)value+=static_cast<double>(round_p[i][k])*static_cast<double>(round_a[k][j]);pa[i][j]=value;
        }
        const auto materialized=operational_product(pa,{1,1});
        check(materialized==std::vector<double>({-2,2})&&materialized!=two,"rounded materialization contrast actually differs");
        bspai::SpaiOptions sp;sp.epsilon=.3;sp.alpha=2;sp.beta=8;
        const auto approximate=bspai::construct_spai(matrix(tiny),sp);
        check(approximate.status==bspai::SpaiStatus::success,"prior milestone approximate SPAI setup");
        if(approximate.status==bspai::SpaiStatus::success) {
            const auto solved=bspai::left_preconditioned_gmres(matrix(tiny),approximate.preconditioner,general,options);
            check(solved.converged,"approximate SPAI GMRES convergence");audit(tiny,dense(approximate.preconditioner),general,solved,options.tolerance,3,&truth);
            std::cout<<"SPAI GMRES: iterations="<<solved.iterations<<", wider_original_residual="<<original_residual(tiny,general,solved.solution)
                     <<", operational_preconditioned_relative="<<operational_relative(tiny,dense(approximate.preconditioner),general,solved.solution)<<'\n';
        }
        sp.epsilon=1e-12;
        const auto full=bspai::construct_spai(matrix(tiny),sp);
        check(full.status==bspai::SpaiStatus::success,"prior milestone full SPAI setup");
        if(full.status==bspai::SpaiStatus::success) {
            const auto solved=bspai::left_preconditioned_gmres(matrix(tiny),full.preconditioner,general,options);
            check(solved.converged,"full SPAI GMRES convergence");audit(tiny,dense(full.preconditioner),general,solved,options.tolerance,3,&truth);
        }
        options.tolerance=1e-30;
        const auto strict=bspai::left_preconditioned_gmres(matrix(tiny),matrix(i3),general,options);
        audit(tiny,i3,general,strict,options.tolerance,3);
        check(!strict.converged&&strict.status==GmresStatus::nonconvergence,"full dimension unattainable tolerance explicit nonconvergence");
        bool rejected_gap=false;
        for(const auto& step:strict.history)
            if(step.estimated_relative<=options.tolerance&&step.recomputed_relative>options.tolerance) {
                rejected_gap=true;check(!step.accepted,"estimated-zero residual gap rejected");
            }
        std::cout<<"strict tolerance: status="<<bspai::gmres_status_name(strict.status)<<", final_recomputed="<<strict.final_preconditioned_relative
                 <<", final_estimate="<<strict.history.back().estimated_relative<<", estimate-below-tolerance gap exercised="<<rejected_gap<<'\n';
        // Branch-coverage probe: 1e-16 failed to bracket the measured estimate
        // 1.283e-16 and recomputed residual 5.119e-16. This explicit 2e-16
        // probe changes neither the primary 1e-8 nor strict 1e-30 configuration.
        options.tolerance=2e-16;
        const auto gap=bspai::left_preconditioned_gmres(matrix(tiny),matrix(i3),general,options);
        bool required_gap=false;
        for(const auto& step:gap.history)if(step.estimated_relative<=options.tolerance&&step.recomputed_relative>options.tolerance) {
            required_gap=true;
            check(step.verification_triggered&&!step.accepted,"required estimate gap triggers verification and rejection");
        }
        check(required_gap,"estimate<=tau while recomputed>tau branch actually exercised");
        check(!gap.converged&&gap.status==GmresStatus::nonconvergence,"estimate gap cannot report convergence");
        std::cout<<"required residual gap: tau="<<options.tolerance<<", estimate="<<gap.history.back().estimated_relative
                 <<", recomputed="<<gap.final_preconditioned_relative<<", status="<<bspai::gmres_status_name(gap.status)<<'\n';
        options.tolerance=1e-8;
        const double delta=std::ldexp(1.0,-30);
        const Dense unstable_a={{1,1},{1,1+static_cast<long double>(delta)}},i2=identity(2);
        const auto unstable=bspai::left_preconditioned_gmres(matrix(unstable_a),matrix(i2),{1,0},options);
        const std::vector<long double> unstable_truth={(1+delta)/delta,-1.0L/delta};
        long double unstable_forward=0;
        for(std::size_t k=0;k<2;++k)unstable_forward=std::max(unstable_forward,std::fabs((unstable.solution[k]-unstable_truth[k])/unstable_truth[k]));
        check(std::isfinite(unstable_forward),"near-dependent forward accuracy diagnostic finite");
        if(unstable.converged)check(operational_relative(unstable_a,i2,{1,0},unstable.solution)<=options.tolerance,"near-dependent success obeys operational criterion");
        else check(unstable.status==GmresStatus::nonconvergence||unstable.status==GmresStatus::breakdown,"near-dependent failure explicitly classified");
        check(unstable.status==GmresStatus::breakdown&&!unstable.history.empty()&&unstable.history.back().breakdown,
              "near-dependent Arnoldi breakdown branch exercised");
        check(unstable.final_preconditioned_relative>options.tolerance&&!unstable.history.back().accepted,
              "near-dependent breakdown candidate rejected by recomputed criterion");
        std::cout<<"near-dependent GMRES: status="<<bspai::gmres_status_name(unstable.status)<<", forward_relative="<<unstable_forward
                 <<", wider_original_residual="<<original_residual(unstable_a,{1,0},unstable.solution)<<'\n';
        std::cout<<"nonsymmetric GMRES: iterations="<<non.iterations<<", wider_original_residual="<<original_residual(tiny,general,non.solution)
                 <<", wider_preconditioned_relative="<<wider_relative(tiny,i3,general,non.solution)<<'\n';
    } catch(const std::exception& e){++failures;std::cerr<<"UNEXPECTED EXCEPTION: "<<e.what()<<'\n';}
    std::cout<<"Independent GMRES validation: "<<checks<<" checks, "<<failures<<" failures\n";
    return failures?1:0;
}
