#include "bspai/cuda_gmres.hpp"
#include "bspai/spai.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
typedef std::vector<std::vector<double> > Dense;
int checks=0,failures=0;
const double eps=std::numeric_limits<double>::epsilon();
double max_solution_difference=0,max_residual_difference=0;
void check(bool ok,const std::string& label){++checks;if(!ok){++failures;std::cerr<<"FAIL: "<<label<<'\n';}}
bspai::CsrMatrix matrix(const Dense& a){std::vector<bspai::Entry> e;for(std::size_t i=0;i<a.size();++i)for(std::size_t j=0;j<a[i].size();++j)if(a[i][j]!=0)e.push_back({i,j,a[i][j]});return bspai::CsrMatrix::from_entries(a.size(),a[0].size(),e);}
Dense dense(const bspai::CsrMatrix& a){Dense d(a.rows(),std::vector<double>(a.cols(),0));for(std::size_t i=0;i<a.rows();++i)for(std::size_t k=a.row_offsets()[i];k<a.row_offsets()[i+1];++k)d[i][a.column_indices()[k]]=a.values()[k];return d;}
Dense identity(std::size_t n){Dense a(n,std::vector<double>(n,0));for(std::size_t i=0;i<n;++i)a[i][i]=1;return a;}
std::vector<double> product(const Dense& a,const std::vector<double>& x){std::vector<double> y(a.size(),0);for(std::size_t i=0;i<a.size();++i)for(std::size_t j=0;j<x.size();++j)if(a[i][j]!=0)y[i]+=a[i][j]*x[j];return y;}
double norm(const std::vector<double>& x){double v=0;for(double a:x)v=std::hypot(v,a);return v;}
double maxabs(const std::vector<double>& x){double v=0;for(double a:x)v=std::max(v,std::abs(a));return v;}
bool near(double x,double y,double scale=1){return std::isfinite(x)&&std::isfinite(y)&&std::abs(x-y)<=768*eps*std::max(scale,std::abs(y));}
struct Residuals{double original,relative,scale;};
Residuals residuals(const Dense& a,const Dense& p,const std::vector<double>& b,const std::vector<double>& x){
    const auto ax=product(a,x),pb=product(p,b),pax=product(p,ax);std::vector<double> r(b.size()),pr(b.size());
    for(std::size_t i=0;i<b.size();++i){r[i]=b[i]-ax[i];pr[i]=pb[i]-pax[i];}
    const double denominator=norm(pb);const double relative=denominator==0?(norm(pr)==0?0:std::numeric_limits<double>::infinity()):norm(pr)/denominator;
    return {norm(r),relative,norm(b)+norm(ax)+norm(pb)+norm(pax)};
}
std::vector<double> tiny_truth(const std::vector<double>& b){const Dense inv={{14.0/50,-3.0/50,1.0/50},{-6.0/50,12.0/50,-4.0/50},{2.0/50,-4.0/50,18.0/50}};return product(inv,b);}
void json_vector(const std::vector<double>& x){std::cout<<'[';for(std::size_t i=0;i<x.size();++i){if(i)std::cout<<',';std::cout<<x[i];}std::cout<<']';}
void json_matrix(const Dense& a){std::cout<<'[';for(std::size_t i=0;i<a.size();++i){if(i)std::cout<<',';json_vector(a[i]);}std::cout<<']';}
void oracle_record(const Dense& a,const Dense& p,const std::vector<double>& b,const std::vector<double>& x,double original,double relative,bool accepted,double tau){
    std::cout<<"ORACLE {\"A\":";json_matrix(a);std::cout<<",\"P\":";json_matrix(p);std::cout<<",\"b\":";json_vector(b);std::cout<<",\"x\":";json_vector(x);std::cout<<",\"original\":"<<original<<",\"relative\":"<<relative<<",\"accepted\":"<<(accepted?"true":"false")<<",\"tau\":"<<tau<<"}\n";
}
void norm_case(const std::vector<double>& x,bool exact=false){
    const double expected=norm(x),gpu=bspai::cuda_safe_norm2(x),n=static_cast<double>(3*x.size()+1),u=std::ldexp(1.0,-53),gamma=n*u/(1-n*u);
    const double allowance=8*gamma*expected+8*std::numeric_limits<double>::denorm_min();
    check(std::isfinite(gpu)&&std::abs(gpu-expected)<=allowance,"safe GPU norm versus stable CPU hypot scaled bound");
    if(exact)check(gpu==expected,"safe GPU norm exact singleton/dyadic reference");
    check(gpu==bspai::cuda_safe_norm2(x),"safe GPU norm deterministic repeat");
    std::cout<<"NORM_ORACLE {\"values\":";json_vector(x);std::cout<<",\"norm\":"<<gpu<<",\"exact\":"<<(exact?"true":"false")<<"}\n";
}
void validate_norm(){
    norm_case({},true);norm_case({0,-0.0,0},true);
    for(int exponent=1022;exponent<=1074;++exponent){const double x=std::ldexp(1.0,-exponent);norm_case({x},true);norm_case({-x,0},true);}
    for(int exponent:std::vector<int>({-900,-512,0,512,1023})){norm_case({std::ldexp(1.0,exponent),0},true);}
    for(int exponent:std::vector<int>({-1074,-1030,-1022,-512,0,512,1021})){const double unit=std::ldexp(1.0,exponent);norm_case({3*unit,-4*unit,0},true);}
    norm_case({std::numeric_limits<double>::max(),std::numeric_limits<double>::denorm_min()},true);
    norm_case({1e200,-1e200});norm_case({1e-200,-1e-200});norm_case({.1,-.7,1.3});
    std::vector<double> long_vector(257);for(std::size_t i=0;i<long_vector.size();++i)long_vector[i]=std::ldexp((i%2?-1.0:1.0),static_cast<int>(i%20)-10);norm_case(long_vector);
    bool overflow=false;try{bspai::cuda_safe_norm2({std::numeric_limits<double>::max(),std::numeric_limits<double>::max()});}catch(const std::overflow_error&){overflow=true;}check(overflow,"genuinely unrepresentable norm explicit overflow");
    bool nonfinite=false;try{bspai::cuda_safe_norm2({1,std::numeric_limits<double>::infinity()});}catch(const std::invalid_argument&){nonfinite=true;}check(nonfinite,"nonfinite norm input explicit rejection");
}
void audit(const Dense& a,const Dense& p,const std::vector<double>& b,const bspai::GmresResult& r,const bspai::GmresOptions& o,bool well_conditioned=true){
    const std::size_t cap=o.max_iterations?o.max_iterations:b.size();check(r.iterations<=cap,"GPU iteration cap");check(r.history.size()==r.iterations,"GPU history length");check(r.solution.size()==b.size(),"GPU final solution dimensions");if(r.solution.size()!=b.size())return;
    const auto own=residuals(a,p,b,r.solution);max_residual_difference=std::max(max_residual_difference,std::abs(own.relative-r.final_preconditioned_relative));
    check(near(r.final_original_residual,own.original,std::max(std::numeric_limits<double>::denorm_min(),own.scale)),"GPU final original residual independently computed without absolute-one floor");
    check(near(r.final_preconditioned_relative,own.relative),"GPU final true preconditioned residual independently computed");
    oracle_record(a,p,b,r.solution,r.final_original_residual,r.final_preconditioned_relative,r.converged,o.tolerance);
    if(r.converged){check(r.final_preconditioned_relative<=o.tolerance,"GPU accepted final recomputed criterion");check(own.relative<=o.tolerance+768*eps,"independent accepted residual with declared rounding allowance");}
    for(std::size_t i=0;i<r.history.size();++i){const auto& h=r.history[i];check(h.iteration==i+1,"GPU history ordinal");check(h.solution.size()==b.size(),"posttermination history candidate snapshot");if(h.solution.size()!=b.size())continue;
        const auto q=residuals(a,p,b,h.solution);check(near(h.original_residual,q.original,std::max(std::numeric_limits<double>::denorm_min(),q.scale)),"each GPU candidate original residual independent without absolute-one floor");check(near(h.recomputed_relative,q.relative),"each GPU candidate preconditioned residual independent");
        oracle_record(a,p,b,h.solution,h.original_residual,h.recomputed_relative,h.accepted,o.tolerance);
        if(h.accepted){check(h.verification_triggered,"GPU acceptance verification triggered");check(h.recomputed_relative<=o.tolerance,"GPU acceptance not estimate-only");}
        if(h.arnoldi_subdiagonal==0)check(h.breakdown,"GPU exact zero Arnoldi subdiagonal flagged");
    }
    if(well_conditioned&&!r.history.empty()){
        check(r.basis.size()>=r.iterations,"GPU posttermination basis snapshot exists");check(r.hessenberg.size()==cap+1,"GPU raw Hessenberg shape");
        double defect=0,orthogonality=0;
        for(std::size_t j=0;j<r.iterations&&j<r.basis.size();++j){check(r.basis[j].size()==b.size(),"GPU used basis dimensions");check(near(norm(r.basis[j]),1),"GPU used basis normalization");auto relation=product(p,product(a,r.basis[j]));const double scale=std::max(1.0,norm(relation));
            for(std::size_t k=0;k<=j+1&&k<r.basis.size();++k)for(std::size_t row=0;row<b.size();++row)relation[row]-=r.hessenberg[k][j]*r.basis[k][row];defect=std::max(defect,norm(relation)/scale);
            for(std::size_t k=0;k<j;++k){double dot=0;for(std::size_t row=0;row<b.size();++row)dot+=r.basis[j][row]*r.basis[k][row];orthogonality=std::max(orthogonality,std::abs(dot));}
        }
        check(defect<=768*eps,"independent GPU Arnoldi relation");check(orthogonality<=1e-12,"GPU used basis orthogonality well-conditioned");
    }
}
void compare(const Dense& a,const Dense& p,const std::vector<double>& b,const bspai::GmresOptions& o,const std::vector<double>& initial={},const std::vector<double>& truth={}){
    const auto cpu=bspai::left_preconditioned_gmres(matrix(a),matrix(p),b,o,initial);const auto gpu=bspai::cuda_left_preconditioned_gmres(matrix(a),matrix(p),b,o,initial);
    check(gpu.transfers.arnoldi_vector_uploads==0&&gpu.transfers.arnoldi_vector_downloads==0,"no GPU Arnoldi loop vector host transfers");
    check(gpu.transfers.vector_uploads>0&&gpu.transfers.vector_downloads>0,"GPU inputs uploaded and snapshots downloaded");
    check(gpu.transfers.scalar_observations==gpu.transfers.dot_observations+gpu.transfers.norm_observations+gpu.transfers.finite_flag_observations,"GPU scalar observation accounting");
    check(gpu.solver.status==cpu.status,"main CPU/GPU terminal status agreement");check(gpu.solver.converged==cpu.converged,"main CPU/GPU convergence agreement");check(gpu.solver.iterations==cpu.iterations,"main CPU/GPU iteration count agreement");audit(a,p,b,gpu.solver,o);
    const auto initial_residual=residuals(a,p,b,initial.empty()?std::vector<double>(b.size(),0):initial);
    check(near(gpu.solver.initial_original_residual,initial_residual.original,std::max(std::numeric_limits<double>::denorm_min(),initial_residual.scale)),"GPU initial original residual independent no floor");
    check(initial_residual.original==0?gpu.solver.initial_preconditioned_relative==0:near(gpu.solver.initial_preconditioned_relative,initial_residual.relative),"GPU initial preconditioned residual and exact convention");
    if(gpu.solver.solution.size()==cpu.solution.size()){double diff=0;for(std::size_t i=0;i<cpu.solution.size();++i)diff=std::max(diff,std::abs(cpu.solution[i]-gpu.solver.solution[i]));max_solution_difference=std::max(max_solution_difference,diff);check(diff<=1e-12*std::max(1.0,maxabs(cpu.solution)),"main CPU/GPU solution agreement");}
    if(!truth.empty()&&gpu.solver.solution.size()==truth.size()){double error=0;for(std::size_t i=0;i<truth.size();++i)error=std::max(error,std::abs(gpu.solver.solution[i]-truth[i]));check(error<=1e-12*std::max(1.0,maxabs(truth)),"GPU independent exact/rational reference solution");}
    std::cout<<"GPU main case: status="<<bspai::gmres_status_name(gpu.solver.status)<<", iterations="<<gpu.solver.iterations<<", true_relative="<<gpu.solver.final_preconditioned_relative<<'\n';
    std::cout<<"GPU residency: uploads="<<gpu.transfers.vector_uploads<<", downloads_aftertermination="<<gpu.transfers.vector_downloads<<", Arnoldi_H2D="<<gpu.transfers.arnoldi_vector_uploads<<", Arnoldi_D2H="<<gpu.transfers.arnoldi_vector_downloads<<", dot="<<gpu.transfers.dot_observations<<", norm="<<gpu.transfers.norm_observations<<", finite_flag="<<gpu.transfers.finite_flag_observations<<'\n';
}
}

int main(){
    try{
        std::cout<<std::setprecision(17);const Dense i3=identity(3),tiny={{4,1,0},{2,5,1},{0,1,3}},diagonal={{2,0,0},{0,-3,0},{0,0,5}};const std::vector<double> general={.1,-.7,1.3};bspai::GmresOptions o;o.tolerance=1e-8;
        char* emulation=nullptr;std::size_t emulation_size=0;
        const int environment_status=_dupenv_s(&emulation,&emulation_size,"CUBLAS_EMULATE_DOUBLE_PRECISION");
        std::cout<<"CUBLAS_EMULATE_DOUBLE_PRECISION="<<(environment_status?"(unavailable)":(emulation?emulation:"(unset)"))<<'\n';std::free(emulation);
        validate_norm();
        compare(i3,i3,general,o,{},general);compare(diagonal,i3,{2,6,15},o,{}, {1,-2,3});compare(tiny,i3,general,o,{},tiny_truth(general));compare(tiny,i3,general,o,{.5,-.3,.2},tiny_truth(general));compare(tiny,i3,{2,-5,7},o,{1,-2,3},{1,-2,3});compare(tiny,i3,{0,0,0},o,{}, {0,0,0});compare(i3,i3,{1,0,0},o,{}, {1,0,0});
        const double tiny_rhs=std::ldexp(1.0,-1030);
        compare(identity(2),identity(2),{tiny_rhs,0},o,{}, {tiny_rhs,0});
        const auto tiny_beta=bspai::cuda_left_preconditioned_gmres(matrix(identity(2)),matrix(identity(2)),{tiny_rhs,0},o);
        check(tiny_beta.solver.converged&&tiny_beta.solver.solution==std::vector<double>({tiny_rhs,0}),"literal device division at subnormal beta preserves exact solution");
        const auto happy=bspai::cuda_left_preconditioned_gmres(matrix(i3),matrix(i3),{1,0,0},o);check(happy.solver.converged&&happy.solver.iterations==1&&happy.solver.history[0].breakdown&&happy.solver.history[0].arnoldi_subdiagonal==0,"GPU exact happy Arnoldi breakdown");
        const Dense zero(3,std::vector<double>(3,0));const auto unhappy=bspai::cuda_left_preconditioned_gmres(matrix(zero),matrix(i3),{1,0,0},o);check(!unhappy.solver.converged&&unhappy.solver.status==bspai::GmresStatus::breakdown,"GPU zero Givens/triangular breakdown failure");
        const auto zero_bad=bspai::cuda_left_preconditioned_gmres(matrix(tiny),matrix(i3),{0,0,0},o,{1,0,0});check(!zero_bad.solver.converged&&zero_bad.solver.status==bspai::GmresStatus::zero_preconditioned_rhs,"GPU zero RHS nonsolution normalization failure");
        const auto annihilated=bspai::cuda_left_preconditioned_gmres(matrix(i3),matrix(zero),{1,0,0},o);check(!annihilated.solver.converged&&annihilated.solver.status==bspai::GmresStatus::zero_preconditioned_rhs,"GPU annihilated preconditioned RHS failure");
        const auto annihilated_exact=bspai::cuda_left_preconditioned_gmres(matrix(i3),matrix(zero),{1,0,0},o,{1,0,0});check(annihilated_exact.solver.converged&&annihilated_exact.solver.iterations==0,"GPU exact original guess precedes zero normalization");
        o.max_iterations=1;compare(diagonal,i3,{1,1,1},o);const auto cap=bspai::cuda_left_preconditioned_gmres(matrix(diagonal),matrix(i3),{1,1,1},o);check(cap.solver.status==bspai::GmresStatus::iteration_limit&&!cap.solver.converged,"GPU smaller cap nonconvergence status");
        o.max_iterations=4;check(bspai::cuda_left_preconditioned_gmres(matrix(tiny),matrix(i3),general,o).solver.status==bspai::GmresStatus::invalid_input,"GPU cap>n invalid");o.max_iterations=0;
        check(bspai::cuda_left_preconditioned_gmres(matrix(tiny),matrix(i3),{1,2},o).solver.status==bspai::GmresStatus::invalid_input,"GPU RHS dimensions invalid");check(bspai::cuda_left_preconditioned_gmres(matrix(tiny),matrix(i3),general,o,{1,2}).solver.status==bspai::GmresStatus::invalid_input,"GPU initial dimensions invalid");
        check(bspai::cuda_left_preconditioned_gmres(matrix(tiny),matrix(i3),{1,std::numeric_limits<double>::quiet_NaN(),2},o).solver.status==bspai::GmresStatus::invalid_input,"GPU nonfinite RHS invalid");o.tolerance=0;check(bspai::cuda_left_preconditioned_gmres(matrix(tiny),matrix(i3),general,o).solver.status==bspai::GmresStatus::invalid_input,"GPU zero tolerance invalid");o.tolerance=1e-8;
        bspai::SpaiOptions sp;sp.epsilon=.3;sp.alpha=2;sp.beta=8;const auto approximate=bspai::construct_spai(matrix(tiny),sp);check(approximate.status==bspai::SpaiStatus::success,"CPU approximate SPAI preconditioner fixture");if(approximate.status==bspai::SpaiStatus::success)compare(tiny,dense(approximate.preconditioner),general,o,{},tiny_truth(general));
        sp.epsilon=1e-12;const auto full=bspai::construct_spai(matrix(tiny),sp);check(full.status==bspai::SpaiStatus::success,"CPU full SPAI preconditioner fixture");if(full.status==bspai::SpaiStatus::success)compare(tiny,dense(full.preconditioner),general,o,{},tiny_truth(general));
        o.tolerance=1e-30;const auto strict=bspai::cuda_left_preconditioned_gmres(matrix(tiny),matrix(i3),general,o);audit(tiny,i3,general,strict.solver,o);check(!strict.solver.converged,"GPU strict full cap cannot claim estimate-only success");check(strict.solver.status==bspai::GmresStatus::nonconvergence||strict.solver.status==bspai::GmresStatus::breakdown,"GPU strict nonconvergence explicitly classified");
        std::cout<<"GPU strict: status="<<bspai::gmres_status_name(strict.solver.status)<<", relative="<<strict.solver.final_preconditioned_relative<<'\n';
        bool gap_seen=false;
        for(double tolerance:std::vector<double>({1e-17,1e-16,2e-16,5e-16})){
            o.tolerance=tolerance;const auto gap=bspai::cuda_left_preconditioned_gmres(matrix(tiny),matrix(i3),general,o);audit(tiny,i3,general,gap.solver,o);
            for(const auto& h:gap.solver.history)if(h.estimated_relative<=tolerance&&h.recomputed_relative>tolerance){gap_seen=true;check(h.verification_triggered&&!h.accepted,"GPU estimate gap is verified and rejected");std::cout<<"GPU estimate gap: tau="<<tolerance<<", estimate="<<h.estimated_relative<<", true="<<h.recomputed_relative<<'\n';}
        }
        check(gap_seen,"actual GPU estimate<=tau/true>tau branch coverage");
    }catch(const std::exception& e){++failures;std::cerr<<"UNEXPECTED EXCEPTION: "<<e.what()<<'\n';}
    std::cout<<"Maximum CPU/GPU solution infinity difference="<<max_solution_difference<<", maximum independent relative-residual difference="<<max_residual_difference<<'\n';std::cout<<"Independent CUDA GMRES validation: "<<checks<<" checks, "<<failures<<" failures\n";return failures?1:0;
}
