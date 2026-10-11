#include "bspai/ir.hpp"
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
int checks=0,failures=0;
const long double eps=std::numeric_limits<double>::epsilon();
void check(bool ok,const std::string& name){++checks;if(!ok){++failures;std::cerr<<"FAIL: "<<name<<'\n';}}
bspai::CsrMatrix matrix(const Dense& a){
    std::vector<bspai::Entry> entries;
    for(std::size_t i=0;i<a.size();++i)for(std::size_t j=0;j<a[i].size();++j)
        if(a[i][j]!=0)entries.push_back({i,j,static_cast<double>(a[i][j])});
    return bspai::CsrMatrix::from_entries(a.size(),a[0].size(),entries);
}
Dense dense(const bspai::CsrMatrix& a){
    Dense d(a.rows(),std::vector<long double>(a.cols(),0));
    for(std::size_t i=0;i<a.rows();++i)for(std::size_t k=a.row_offsets()[i];k<a.row_offsets()[i+1];++k)d[i][a.column_indices()[k]]=a.values()[k];
    return d;
}
Dense identity(std::size_t n){Dense a(n,std::vector<long double>(n,0));for(std::size_t i=0;i<n;++i)a[i][i]=1;return a;}
std::vector<double> product(const Dense& a,const std::vector<double>& x){
    std::vector<double> y(a.size(),0);
    for(std::size_t i=0;i<a.size();++i)for(std::size_t j=0;j<x.size();++j)if(a[i][j]!=0)y[i]+=static_cast<double>(a[i][j])*x[j];
    return y;
}
long double norm2(const std::vector<long double>& x){long double v=0;for(long double a:x)v=std::hypot(v,a);return v;}
struct TruthDiagnostics {std::vector<long double> residual;long double inf,norm2,nbe,cbe;};
TruthDiagnostics wide_diagnostics(const Dense& a,const std::vector<double>& b,const std::vector<double>& x){
    TruthDiagnostics d;d.residual.resize(b.size());d.inf=0;d.cbe=0;
    long double anorm=0,xnorm=0,bnorm=0;
    for(std::size_t i=0;i<b.size();++i){
        long double ax=0,rowsum=0,den=std::fabs(static_cast<long double>(b[i]));
        for(std::size_t j=0;j<x.size();++j){ax+=a[i][j]*x[j];rowsum+=std::fabs(a[i][j]);den+=std::fabs(a[i][j])*std::fabs(static_cast<long double>(x[j]));}
        d.residual[i]=b[i]-ax;d.inf=std::max(d.inf,std::fabs(d.residual[i]));
        const long double component=den==0?(d.residual[i]==0?0:std::numeric_limits<long double>::infinity()):std::fabs(d.residual[i])/den;
        d.cbe=std::max(d.cbe,component);anorm=std::max(anorm,rowsum);xnorm=std::max(xnorm,std::fabs(static_cast<long double>(x[i])));bnorm=std::max(bnorm,std::fabs(static_cast<long double>(b[i])));
    }
    const long double den=anorm*xnorm+bnorm;d.nbe=den==0?(d.inf==0?0:std::numeric_limits<long double>::infinity()):d.inf/den;d.norm2=norm2(d.residual);return d;
}
TruthDiagnostics operational_diagnostics(const Dense& a,const std::vector<double>& b,const std::vector<double>& x){
    TruthDiagnostics d;const auto ax=product(a,x);d.residual.resize(b.size());double anorm=0,xnorm=0,bnorm=0,inf=0,cbe=0,r2=0;
    for(std::size_t i=0;i<b.size();++i){
        const double r=b[i]-ax[i];d.residual[i]=r;inf=std::max(inf,std::abs(r));r2=std::hypot(r2,r);
        double row=0,den=0;for(std::size_t j=0;j<x.size();++j){row+=std::abs(static_cast<double>(a[i][j]));den+=std::abs(static_cast<double>(a[i][j]))*std::abs(x[j]);}
        den+=std::abs(b[i]);cbe=std::max(cbe,den==0?(r==0?0:std::numeric_limits<double>::infinity()):std::abs(r)/den);
        anorm=std::max(anorm,row);xnorm=std::max(xnorm,std::abs(x[i]));bnorm=std::max(bnorm,std::abs(b[i]));
    }
    const double den=anorm*xnorm+bnorm;d.inf=inf;d.norm2=r2;d.cbe=cbe;d.nbe=den==0?(inf==0?0:std::numeric_limits<double>::infinity()):inf/den;return d;
}
bool relative_near(long double x,long double y){return std::isfinite(x)&&std::isfinite(y)&&(y==0?x==0:std::fabs(x-y)<=16*eps*std::fabs(y));}
bool scaled_near(long double x,long double y,long double scale=1){return std::isfinite(x)&&std::isfinite(y)&&std::fabs(x-y)<=768*eps*std::max(scale,std::fabs(y));}
void audit_diagnostics(const Dense& a,const std::vector<double>& b,const std::vector<double>& x,const bspai::IrDiagnostics& d){
    const auto own=operational_diagnostics(a,b,x),wide=wide_diagnostics(a,b,x);
    check(d.residual.size()==b.size(),"diagnostic residual dimensions");
    for(std::size_t i=0;i<b.size()&&i<d.residual.size();++i){
        check(d.residual[i]==static_cast<double>(own.residual[i]),"operational original residual components exact");
        long double scale=std::fabs(static_cast<long double>(b[i]));for(std::size_t j=0;j<x.size();++j)scale+=std::fabs(a[i][j])*std::fabs(static_cast<long double>(x[j]));
        check(scaled_near(d.residual[i],wide.residual[i],std::max(1.0L,scale)),"wider original residual rounding bound");
    }
    check(relative_near(d.residual_inf,own.inf),"independent residual infinity norm");
    check(relative_near(d.residual_2,own.norm2),"independent residual 2norm");
    check(relative_near(d.normwise_backward_error,own.nbe),"independent operational normwise backward error");
    check(relative_near(d.componentwise_backward_error,own.cbe),"independent operational componentwise backward error");
}
template<class Result>void audit(const Dense& a,const Dense& p,const std::vector<double>& b,const Result& r,std::size_t cap){
    check(r.outer_steps<=cap,"outer update count respects cap");
    check(r.inner_attempts==r.inner_history.size()&&r.inner_attempts==r.correction_rhs.size()&&r.inner_attempts==r.update_applied.size(),"parallel inner attempt records");
    check(r.history.size()==r.outer_steps+1,"outer records only completed updates plus initial state");
    if(r.history.empty())return;
    check(r.history[0].x==product(p,b),"x0 equals independent P*b");
    check(r.history[0].step==0&&!r.history[0].has_inner,"initial state has no correction");
    for(std::size_t i=0;i<r.history.size();++i){
        const auto& h=r.history[i];check(h.step==i,"outer history step ordinal");audit_diagnostics(a,b,h.x,h.diagnostics);
        if(i>0){
            check(h.has_inner&&h.inner.converged,"updated state has accepted inner correction");
            std::vector<double> expected=h.inner.solution;for(std::size_t j=0;j<b.size();++j)expected[j]+=r.history[i-1].x[j];
            check(h.x==expected,"outer update xnext=x+d binary64");
        }
    }
    std::size_t updates=0;
    for(std::size_t attempt=0;attempt<r.inner_attempts;++attempt){
        const auto& state=r.history[updates];const auto diag=operational_diagnostics(a,b,state.x);
        std::vector<double> rhs(diag.residual.begin(),diag.residual.end());
        check(r.correction_rhs[attempt]==rhs,"unscaled original residual is correction RHS");
        const auto& inner=r.inner_history[attempt];
        check(relative_near(inner.initial_original_residual,diag.norm2),"inner solve initial zero correction guess");
        if(inner.initial_original_residual>0)check(inner.initial_preconditioned_relative==1,"inner zero initial correction preconditioned relative residual1");
        if(r.update_applied[attempt]){check(inner.converged,"update only accepted inner correction");++updates;}
        else check(!inner.converged||r.status==bspai::IrStatus::numerical_failure,"unapplied attempt is failed inner or failed outer diagnostics");
    }
    check(updates==r.outer_steps,"completed update count matches applied corrections");
    check(r.solution==r.history.back().x,"terminal solution retains last outer iterate");
    if(r.converged){check(r.status==bspai::IrStatus::success,"converged terminal success status");check(r.history.back().diagnostics.normwise_backward_error<=bspai::ir_backward_tolerance(),"success on original normwise backward error");}
}
long double forward_error(const std::vector<double>& x,const std::vector<long double>& truth){long double err=0,size=0;for(std::size_t i=0;i<x.size();++i){err=std::max(err,std::fabs(x[i]-truth[i]));size=std::max(size,std::fabs(truth[i]));}return size==0?err:err/size;}
}

int main(){
    using bspai::IrStatus;
    try{
        std::cout<<std::setprecision(17);check(std::numeric_limits<long double>::digits>53,"wider oracle available");
        check(bspai::ir_backward_tolerance()==10*std::ldexp(1.0,-53),"fixed original backward tolerance10u");
        const Dense a={{4,1,0},{2,5,1},{0,1,3}},i3=identity(3);const std::vector<double>b={2,-5,7};
        bspai::IrOptions options;options.spai.epsilon=.3;options.spai.alpha=2;options.spai.beta=8;options.gmres.tolerance=1e-8;
        const auto solve=bspai::spai_gmres_ir(matrix(a),b,options,{1,-2,3});
        check(solve.converged&&solve.construction_performed&&solve.construction.status==bspai::SpaiStatus::success,"end-to-end nonsymmetric uniformdouble solve");
        if(solve.construction.status==bspai::SpaiStatus::success)audit(a,dense(solve.construction.preconditioner),b,solve,20);
        check(forward_error(solve.solution,{1,-2,3})<=1e-12,"well-conditioned exact reference solution accuracy");
        if(!solve.history.empty()){
            const auto wide=wide_diagnostics(a,b,solve.solution);
            check(wide.nbe<=bspai::ir_backward_tolerance()+16*eps,"well-conditioned independent wider backward acceptance");
            std::cout<<"IR tiny: steps="<<solve.outer_steps<<", inner_attempts="<<solve.inner_attempts<<", wide_nbe="<<wide.nbe<<", wide_cbe="<<wide.cbe<<", forward="<<forward_error(solve.solution,{1,-2,3})<<'\n';
        }
        const std::vector<double> general={.1,-.7,1.3};
        const Dense inv={{14.0L/50,-3.0L/50,1.0L/50},{-6.0L/50,12.0L/50,-4.0L/50},{2.0L/50,-4.0L/50,18.0L/50}};
        std::vector<long double> truth(3,0);for(std::size_t i=0;i<3;++i)for(std::size_t j=0;j<3;++j)truth[i]+=inv[i][j]*general[j];
        const auto nonspecific=bspai::spai_gmres_ir(matrix(a),general,options);
        check(nonspecific.converged,"end-to-end noninteger RHS");audit(a,dense(nonspecific.construction.preconditioner),general,nonspecific,20);
        check(forward_error(nonspecific.solution,truth)<=1e-12,"noninteger RHS wider rational inverse accuracy");
        const auto wrong_truth=bspai::spai_gmres_ir(matrix(a),b,options,{1e9,1e9,1e9});
        check(wrong_truth.converged&&wrong_truth.solution==solve.solution,"optional diagnostic truth does not change stopping");
        check(wrong_truth.history.back().diagnostics.forward_error>.9,"deliberately incorrect reference remains diagnostic only");
        const auto explicit_d=bspai::evaluate_ir_diagnostics(matrix(a),b,{.9,-2,3.1},{1,-2,3});
        audit_diagnostics(a,b,{.9,-2,3.1},explicit_d);
        check(explicit_d.has_reference&&scaled_near(explicit_d.forward_error,1.0L/30),"forward error diagnostic against exact truth");
        const Dense diagonal={{2,0},{0,-4}};const auto exact=bspai::spai_gmres_ir(matrix(diagonal),{2,8},options,{1,-2});
        check(exact.converged&&exact.outer_steps==0&&exact.inner_attempts==0,"exact initial Pb terminates without correction");
        audit(diagonal,dense(exact.construction.preconditioner),{2,8},exact,20);
        const auto zero=bspai::spai_gmres_ir(matrix(a),{0,0,0},options,{0,0,0});
        check(zero.converged&&zero.outer_steps==0&&zero.history[0].diagnostics.normwise_backward_error==0&&zero.history[0].diagnostics.componentwise_backward_error==0,"zero RHS and denominator convention");
        audit(a,dense(zero.construction.preconditioner),{0,0,0},zero,20);
        const auto zero_truth=bspai::evaluate_ir_diagnostics(matrix(identity(2)),{0,0},{1,-2},{0,0});
        check(zero_truth.has_reference&&zero_truth.forward_error==2,"zero truth reports absolute forward error");
        options.max_outer_steps=0;
        const auto cap0=bspai::refine_with_preconditioner(matrix(a),matrix(i3),b,options);
        check(!cap0.converged&&cap0.status==IrStatus::iteration_limit&&cap0.outer_steps==0&&cap0.inner_attempts==0,"outer cap0 checks only x0 then fails");
        audit(a,i3,b,cap0,0);
        const auto cap0exact=bspai::refine_with_preconditioner(matrix(identity(2)),matrix(identity(2)),{1,2},options);
        check(cap0exact.converged&&cap0exact.outer_steps==0,"initial success precedes outer cap0");
        options.max_outer_steps=1;options.gmres.tolerance=.9;
        const auto cap1=bspai::refine_with_preconditioner(matrix(a),matrix(i3),b,options);
        check(!cap1.converged&&cap1.status==IrStatus::iteration_limit&&cap1.outer_steps==1,"coarse-inner outer cap1 nonconvergence");audit(a,i3,b,cap1,1);
        options.max_outer_steps=20;options.gmres.tolerance=1;
        const auto stagnation=bspai::refine_with_preconditioner(matrix(a),matrix(i3),b,options);
        check(!stagnation.converged&&stagnation.status==IrStatus::stagnation&&stagnation.outer_steps==1,"unchanged unconverged iterate named stagnation");
        check(stagnation.solution==product(i3,b)&&stagnation.inner_history[0].converged&&stagnation.inner_history[0].iterations==0,"zero correction accepted only by coarse inner criterion");audit(a,i3,b,stagnation,20);
        options.gmres.tolerance=1e-8;options.gmres.max_iterations=1;
        const auto inner_cap=bspai::refine_with_preconditioner(matrix(a),matrix(i3),general,options);
        check(!inner_cap.converged&&inner_cap.status==IrStatus::inner_failure&&inner_cap.outer_steps==0&&inner_cap.inner_attempts==1,"failed inner cap propagates without outer update");
        check(inner_cap.inner_history[0].status==bspai::GmresStatus::iteration_limit,"inner iteration limit retained");audit(a,i3,general,inner_cap,20);
        options.gmres.max_iterations=0;options.gmres.tolerance=1e-30;
        const auto inner_full=bspai::refine_with_preconditioner(matrix(a),matrix(i3),general,options);
        check(!inner_full.converged&&inner_full.status==IrStatus::inner_failure&&inner_full.outer_steps==0,"full inner nonconvergence propagates without applying candidate");
        check(inner_full.inner_history[0].status==bspai::GmresStatus::nonconvergence,"full inner nonconvergence status retained");audit(a,i3,general,inner_full,20);
        options.gmres.tolerance=1e-8;options.spai.epsilon=1e-12;
        const auto construction=bspai::spai_gmres_ir(matrix({{1,2},{1,2}}),{1,0},options);
        check(!construction.converged&&construction.status==IrStatus::construction_failure&&construction.inner_attempts==0,"construction failure prevents inner solves");
        options.spai.epsilon=.3;
        const double delta=std::ldexp(1.0,-30);const Dense ill={{1,1},{1,1+static_cast<long double>(delta)}};
        const auto ill_solve=bspai::spai_gmres_ir(matrix(ill),{1,0},options,{(1+delta)/delta,-1/delta});
        check(ill_solve.converged,"ill-conditioned original backward criterion result");audit(ill,dense(ill_solve.construction.preconditioner),{1,0},ill_solve,20);
        const auto ill_wide=wide_diagnostics(ill,{1,0},ill_solve.solution);const long double ill_forward=forward_error(ill_solve.solution,{(1+delta)/delta,-1.0L/delta});
        check(std::isfinite(ill_forward)&&ill_wide.nbe<=bspai::ir_backward_tolerance()+16*eps,"ill-conditioned independent backward/forward diagnostics");
        std::cout<<"IR ill-conditioned: status="<<bspai::ir_status_name(ill_solve.status)<<", steps="<<ill_solve.outer_steps<<", wide_residual="<<ill_wide.inf<<", wide_nbe="<<ill_wide.nbe<<", forward_relative="<<ill_forward<<'\n';
        options.max_outer_steps=21;
        check(bspai::refine_with_preconditioner(matrix(a),matrix(i3),b,options).status==IrStatus::invalid_input,"outer cap greater20 rejected");
        options.max_outer_steps=20;
        check(bspai::spai_gmres_ir(matrix(a),{1,2},options).status==IrStatus::invalid_input,"invalid RHS dimensions rejected");
        check(bspai::spai_gmres_ir(matrix(a),{1,std::numeric_limits<double>::infinity(),3},options).status==IrStatus::invalid_input,"nonfinite RHS rejected");
        const auto denominator=bspai::refine_with_preconditioner(matrix({{1e308L,0},{0,1}}),matrix(identity(2)),{2,0},options);
        check(!denominator.converged&&denominator.status==IrStatus::numerical_failure,"original residual overflow fails explicitly");
        const auto residual_norm_overflow=bspai::refine_with_preconditioner(matrix(identity(2)),matrix({{.01L,0},{0,.01L}}),{1.3e308,1.3e308},options);
        check(!residual_norm_overflow.converged&&residual_norm_overflow.status==IrStatus::numerical_failure,"finite residual components but true hypot norm overflow fails explicitly");
        const auto denominator_only=bspai::refine_with_preconditioner(matrix({{1e308L,0},{0,1}}),matrix(identity(2)),{0,2},options);
        check(!denominator_only.converged&&denominator_only.status==IrStatus::numerical_failure,"backward denominator overflow despite exact original residual fails");
        const Dense outer_range={{.5L,0},{0,1e308L}};
        const Dense outer_range_p={{1e-308L,0},{0,1e-308L}};
        const auto outer_overflow=bspai::refine_with_preconditioner(matrix(outer_range),matrix(outer_range_p),{1,0},options);
        check(!outer_overflow.converged&&outer_overflow.status==IrStatus::numerical_failure&&outer_overflow.outer_steps==0&&outer_overflow.inner_attempts==1,"outer diagnostic overflow after converged correction retains prior state");
        std::cout<<"IR outer range probe: status="<<bspai::ir_status_name(outer_overflow.status)<<", attempts="<<outer_overflow.inner_attempts<<", message="<<outer_overflow.message<<'\n';
        check(!outer_overflow.inner_history.empty()&&outer_overflow.inner_history[0].converged&&!outer_overflow.update_applied[0]&&outer_overflow.solution==product(outer_range_p,{1,0}),"converged inner correction is not committed after failed outer diagnostics");
        if(!outer_overflow.history.empty())audit(outer_range,outer_range_p,{1,0},outer_overflow,20);
        options.max_outer_steps=0;
        const double below=1+10*std::numeric_limits<double>::epsilon();
        const double above=std::nextafter(below,2.0);
        const auto stop_below=bspai::refine_with_preconditioner(matrix({{1}}),matrix({{below}}),{1},options);
        const auto stop_above=bspai::refine_with_preconditioner(matrix({{1}}),matrix({{above}}),{1},options);
        check(stop_below.converged&&stop_below.history[0].diagnostics.normwise_backward_error<=bspai::ir_backward_tolerance(),"one-ULP lower outer threshold bracket success");
        check(!stop_above.converged&&stop_above.status==IrStatus::iteration_limit&&stop_above.history[0].diagnostics.normwise_backward_error>bspai::ir_backward_tolerance(),"one-ULP upper outer threshold bracket failure");
        audit({{1}},{{below}},{1},stop_below,0);audit({{1}},{{above}},{1},stop_above,0);
        options.max_outer_steps=20;
        options.max_outer_steps=1;options.gmres.tolerance=1e-8;
        const auto cap_success=bspai::spai_gmres_ir(matrix(a),b,options,{1,-2,3});
        check(cap_success.converged&&cap_success.outer_steps==1,"successful update is accepted before exhausted outer cap");
        audit(a,dense(cap_success.construction.preconditioner),b,cap_success,1);
        options.max_outer_steps=20;
        const Dense disparate={{1e20L,0},{0,1}},disparate_p={{1e-20L,0},{0,.5L}};
        const auto norm_only=bspai::refine_with_preconditioner(matrix(disparate),matrix(disparate_p),{1,1},options,{1e-20,1});
        check(norm_only.converged&&norm_only.outer_steps==0&&norm_only.history[0].diagnostics.componentwise_backward_error>bspai::ir_backward_tolerance(),"componentwise error is diagnostic not an added outer stop");
        audit(disparate,disparate_p,{1,1},norm_only,20);
        const auto reference_overflow=bspai::refine_with_preconditioner(matrix(identity(2)),matrix(identity(2)),{2,0},options,{std::numeric_limits<double>::denorm_min(),0});
        check(reference_overflow.converged&&reference_overflow.outer_steps==0,"optional forward overflow does not fail exact solver");
        check(reference_overflow.history[0].diagnostics.has_reference&&reference_overflow.history[0].diagnostics.forward_diagnostic_overflow&&std::isinf(reference_overflow.history[0].diagnostics.forward_error)&&!reference_overflow.history[0].diagnostics.forward_diagnostic_message.empty(),"optional forward diagnostic overflow explicitly flagged");
        check(bspai::spai_gmres_ir(matrix(a),b,options,{1,2}).status==IrStatus::invalid_input,"optional reference dimensions rejected");
        check(bspai::spai_gmres_ir(matrix(a),b,options,{1,2,std::numeric_limits<double>::infinity()}).status==IrStatus::invalid_input,"nonfinite optional reference rejected");
    }catch(const std::exception& e){++failures;std::cerr<<"UNEXPECTED EXCEPTION: "<<e.what()<<'\n';}
    std::cout<<"Independent IR validation: "<<checks<<" checks, "<<failures<<" failures\n";return failures?1:0;
}
