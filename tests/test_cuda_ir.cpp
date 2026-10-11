#include "bspai/cuda_ir.hpp"
// Reuse Validation's independent fixed-order residual/update audit helpers.
// The saved CPU executable's main is not called; its 439 checks remain untouched.
#define main saved_cpu_ir_validation_main
#include "test_ir.cpp"
#undef main

namespace {
void json_vector_ir(const std::vector<double>& v){std::cout<<'[';for(std::size_t i=0;i<v.size();++i){if(i)std::cout<<',';std::cout<<v[i];}std::cout<<']';}
void json_matrix_ir(const Dense& a){std::cout<<'[';for(std::size_t i=0;i<a.size();++i){if(i)std::cout<<',';std::vector<double> row(a[i].begin(),a[i].end());json_vector_ir(row);}std::cout<<']';}
void emit_outer(const Dense& a,const std::vector<double>& b,const bspai::IrResult& result){
    for(const auto& state:result.history){const auto& d=state.diagnostics;
        std::cout<<"IR_ORACLE {\"A\":";json_matrix_ir(a);std::cout<<",\"b\":";json_vector_ir(b);std::cout<<",\"x\":";json_vector_ir(state.x);std::cout<<",\"residual\":";json_vector_ir(d.residual);
        std::cout<<",\"inf\":"<<d.residual_inf<<",\"norm2\":"<<d.residual_2<<",\"nbe\":"<<d.normwise_backward_error<<",\"cbe\":"<<d.componentwise_backward_error<<",\"accepted\":"<<(d.normwise_backward_error<=bspai::ir_backward_tolerance()?"true":"false")<<"}\n";
    }
}
void transfers(const bspai::CudaIrResult& gpu){
    const auto& r=gpu.solver;check(gpu.inner_transfers.size()==r.inner_attempts,"hybrid per-inner transfer records");
    check(gpu.context_constructions<=1,"hybrid context constructed at most once");
    if(r.inner_attempts){check(gpu.context_constructions==1&&gpu.matrix_uploads==2,"one persistent context and two immutable matrices");check(gpu.construction_transfers.vector_uploads==6,"canonical nonempty A/P uploaded six CSRarrays once");}
    std::size_t uploads=gpu.construction_transfers.vector_uploads,downloads=gpu.construction_transfers.vector_downloads,scalars=gpu.construction_transfers.scalar_observations,inputs=gpu.construction_transfers.host_scalar_inputs;
    for(std::size_t i=0;i<gpu.inner_transfers.size();++i){const auto& t=gpu.inner_transfers[i];check(t.vector_uploads==2,"per-inner only residual and zero initial guess uploaded");check(t.arnoldi_vector_uploads==0&&t.arnoldi_vector_downloads==0,"hybrid no Arnoldi-loop vector host transfers");check(t.vector_downloads==1+r.inner_history[i].basis.size()+r.inner_history[i].history.size(),"terminated correction and debug snapshots downloads only");check(t.scalar_observations==t.dot_observations+t.norm_observations+t.finite_flag_observations,"hybrid observed scalar accounting");uploads+=t.vector_uploads;downloads+=t.vector_downloads;scalars+=t.scalar_observations;}
    check(gpu.total_transfers.vector_uploads==uploads&&gpu.total_transfers.vector_downloads==downloads&&gpu.total_transfers.scalar_observations==scalars,"hybrid aggregate transfer accounting");
    for(const auto& t:gpu.inner_transfers){check(t.host_scalar_inputs==t.axpy_scalar_inputs+t.spmv_scalar_inputs+t.division_scalar_inputs,"hybrid host scalar input accounting");inputs+=t.host_scalar_inputs;}
    check(gpu.total_transfers.host_scalar_inputs==inputs,"hybrid aggregate host scalar input accounting");
    std::cout<<"Hybrid reuse: context="<<gpu.context_constructions<<", matrices="<<gpu.matrix_uploads<<", attempts="<<r.inner_attempts<<", uploads="<<gpu.total_transfers.vector_uploads<<", downloads="<<gpu.total_transfers.vector_downloads<<'\n';
}
void compare_ir(const Dense& a,const Dense& p,const std::vector<double>& b,const bspai::IrOptions& options,const std::vector<double>& truth={}){
    const auto cpu=bspai::refine_with_preconditioner(matrix(a),matrix(p),b,options,truth);
    const auto gpu=bspai::cuda_refine_with_preconditioner(matrix(a),matrix(p),b,options,truth);
    check(cpu.status==gpu.solver.status&&cpu.converged==gpu.solver.converged,"CPU/hybrid IR terminal status agreement");check(cpu.outer_steps==gpu.solver.outer_steps&&cpu.inner_attempts==gpu.solver.inner_attempts,"CPU/hybrid outer counts agreement");
    if(cpu.solution.size()==gpu.solver.solution.size()){long double error=0;for(std::size_t i=0;i<cpu.solution.size();++i)error=std::max(error,std::fabs(static_cast<long double>(cpu.solution[i])-gpu.solver.solution[i]));check(error<=1e-12L*std::max(1.0L,norm2(std::vector<long double>(cpu.solution.begin(),cpu.solution.end()))),"CPU/hybrid well-conditioned solution agreement");}
    audit(a,p,b,gpu.solver,options.max_outer_steps);transfers(gpu);emit_outer(a,b,gpu.solver);
}
}

int main(){
    using bspai::IrStatus;
    try{
        std::cout<<std::setprecision(17);const Dense a={{4,1,0},{2,5,1},{0,1,3}},i3=identity(3);const std::vector<double>b={2,-5,7},general={.1,-.7,1.3};
        bspai::IrOptions o;o.spai.epsilon=.3;o.spai.alpha=2;o.spai.beta=8;o.gmres.tolerance=1e-8;
        const auto gpu=bspai::cuda_spai_gmres_ir(matrix(a),b,o,{1,-2,3});
        const auto cpu=bspai::spai_gmres_ir(matrix(a),b,o,{1,-2,3});
        check(gpu.solver.converged&&gpu.solver.construction_performed&&gpu.solver.construction.status==bspai::SpaiStatus::success,"hybrid end-to-end CPU SPAI/GPU corrections");
        check(gpu.solver.status==cpu.status&&gpu.solver.outer_steps==cpu.outer_steps,"hybrid end-to-end CPU IR agreement");
        audit(a,dense(gpu.solver.construction.preconditioner),b,gpu.solver,20);transfers(gpu);emit_outer(a,b,gpu.solver);
        check(forward_error(gpu.solver.solution,{1,-2,3})<=1e-12,"hybrid exact nonsymmetric reference accuracy");
        const auto variable=bspai::cuda_spai_gmres_ir(matrix(a),general,o);check(variable.solver.converged,"hybrid noninteger RHS success");audit(a,dense(variable.solver.construction.preconditioner),general,variable.solver,20);transfers(variable);emit_outer(a,general,variable.solver);
        const Dense inv={{14.0L/50,-3.0L/50,1.0L/50},{-6.0L/50,12.0L/50,-4.0L/50},{2.0L/50,-4.0L/50,18.0L/50}};std::vector<long double> truth(3,0);for(std::size_t i=0;i<3;++i)for(std::size_t j=0;j<3;++j)truth[i]+=inv[i][j]*general[j];check(forward_error(variable.solver.solution,truth)<=1e-12,"hybrid rational inverse reference accuracy");
        compare_ir(a,i3,b,o,{1,-2,3});compare_ir(identity(2),identity(2),{1,2},o,{1,2});compare_ir(a,i3,{0,0,0},o,{0,0,0});
        const auto zero=bspai::cuda_spai_gmres_ir(matrix(a),{0,0,0},o);check(zero.solver.converged&&zero.solver.inner_attempts==0,"hybrid full solver zero RHS no correction");
        o.max_outer_steps=0;compare_ir(a,i3,b,o);compare_ir(identity(2),identity(2),{1,2},o);
        o.max_outer_steps=1;o.gmres.tolerance=.9;compare_ir(a,i3,b,o);
        o.max_outer_steps=3;const auto multiple=bspai::cuda_refine_with_preconditioner(matrix(a),matrix(i3),b,o);check(multiple.solver.outer_steps==3&&multiple.solver.inner_attempts==3&&multiple.solver.status==IrStatus::iteration_limit,"explicit three-correction GPU reuse fixture");audit(a,i3,b,multiple.solver,3);transfers(multiple);emit_outer(a,b,multiple.solver);
        o.max_outer_steps=20;o.gmres.tolerance=1;compare_ir(a,i3,b,o);
        o.gmres.tolerance=1e-8;o.gmres.max_iterations=1;compare_ir(a,i3,general,o);
        o.gmres.max_iterations=0;o.gmres.tolerance=1e-30;compare_ir(a,i3,general,o);
        o.gmres.tolerance=1e-8;o.spai.epsilon=1e-12;const auto failed=bspai::cuda_spai_gmres_ir(matrix({{1,2},{1,2}}),{1,0},o);check(!failed.solver.converged&&failed.solver.status==IrStatus::construction_failure&&failed.solver.inner_attempts==0&&failed.context_constructions==0,"hybrid construction failure blocks GPU context");
        o.spai.epsilon=.3;const double delta=std::ldexp(1.0,-30);const Dense ill={{1,1},{1,1+static_cast<long double>(delta)}};const auto ill_gpu=bspai::cuda_spai_gmres_ir(matrix(ill),{1,0},o,{(1+delta)/delta,-1/delta});
        check(ill_gpu.solver.converged,"hybrid ill-conditioned original backward criterion success");audit(ill,dense(ill_gpu.solver.construction.preconditioner),{1,0},ill_gpu.solver,20);emit_outer(ill,{1,0},ill_gpu.solver);
        std::cout<<"Hybrid ill-conditioned forward="<<forward_error(ill_gpu.solver.solution,{(1+delta)/delta,-1.0L/delta})<<", original_nbe="<<ill_gpu.solver.history.back().diagnostics.normwise_backward_error<<'\n';
        const auto wrong=bspai::cuda_spai_gmres_ir(matrix(a),b,o,{1e9,1e9,1e9});check(wrong.solver.converged&&wrong.solver.solution==gpu.solver.solution&&wrong.solver.history.back().diagnostics.forward_error>.9,"hybrid optional forward truth never changes stopping");
        o.max_outer_steps=21;check(bspai::cuda_spai_gmres_ir(matrix(a),b,o).solver.status==IrStatus::invalid_input,"hybrid outer cap>20 invalid");o.max_outer_steps=20;
        check(bspai::cuda_spai_gmres_ir(matrix(a),{1,2},o).solver.status==IrStatus::invalid_input,"hybrid invalid RHS dimensions");
        const auto range=bspai::cuda_refine_with_preconditioner(matrix({{1e308L,0},{0,1}}),matrix(identity(2)),{0,2},o);check(range.solver.status==IrStatus::numerical_failure&&!range.solver.converged,"hybrid original denominator overflow fails");
        const Dense range_a={{.5L,0},{0,1e308L}},range_p={{1e-308L,0},{0,1e-308L}};
        const auto outer_range=bspai::cuda_refine_with_preconditioner(matrix(range_a),matrix(range_p),{1,0},o);
        check(outer_range.solver.status==IrStatus::numerical_failure&&outer_range.solver.outer_steps==0&&outer_range.solver.inner_attempts==1,"hybrid post-correction diagnostic overflow retains prior state");
        check(!outer_range.solver.inner_history.empty()&&outer_range.solver.inner_history[0].converged&&!outer_range.solver.update_applied[0]&&outer_range.solver.solution==product(range_p,{1,0}),"hybrid converged correction not committed after failed diagnostics");
        audit(range_a,range_p,{1,0},outer_range.solver,20);transfers(outer_range);
        const auto norm_range=bspai::cuda_refine_with_preconditioner(matrix(identity(2)),matrix({{.01L,0},{0,.01L}}),{1.3e308,1.3e308},o);
        check(norm_range.solver.status==IrStatus::numerical_failure&&!norm_range.solver.converged,"hybrid finite residual components but unrepresentable original norm failure");
        const auto optional=bspai::cuda_refine_with_preconditioner(matrix(identity(2)),matrix(identity(2)),{2,0},o,{std::numeric_limits<double>::denorm_min(),0});
        check(optional.solver.converged&&optional.solver.history[0].diagnostics.forward_diagnostic_overflow&&std::isinf(optional.solver.history[0].diagnostics.forward_error),"hybrid optional forward overflow flagged without failing solve");
        check(bspai::cuda_spai_gmres_ir(matrix(a),{1,std::numeric_limits<double>::infinity(),3},o).solver.status==IrStatus::invalid_input,"hybrid nonfinite RHS rejected");
        check(bspai::cuda_spai_gmres_ir(matrix(a),b,o,{1,2}).solver.status==IrStatus::invalid_input,"hybrid invalid diagnostic truth dimensions");
        o.max_outer_steps=1;const auto final_cap=bspai::cuda_spai_gmres_ir(matrix(a),b,o);check(final_cap.solver.converged&&final_cap.solver.outer_steps==1,"hybrid success precedes exhausted outer cap");audit(a,dense(final_cap.solver.construction.preconditioner),b,final_cap.solver,1);transfers(final_cap);
        bspai::CudaGmresContext context(matrix(a),matrix(i3),o.gmres);
        check(context.matrix_uploads()==2&&context.construction_transfers().vector_uploads==6,"standalone persistent context immutable upload count");
        for(const auto& rhs:std::vector<std::vector<double>>{general,b,{0,0,0},general}){
            const auto again=context.solve(rhs);const auto reference=bspai::left_preconditioned_gmres(matrix(a),matrix(i3),rhs,o.gmres);
            check(again.solver.status==reference.status&&again.solver.converged==reference.converged,"persistent context changed RHS status agreement");
            check(again.transfers.vector_uploads==2&&again.transfers.arnoldi_vector_uploads==0&&again.transfers.arnoldi_vector_downloads==0,"persistent repeated solve does not reupload matrices");
            long double discrepancy=0;for(std::size_t j=0;j<rhs.size();++j)discrepancy=std::max(discrepancy,std::fabs(static_cast<long double>(again.solver.solution[j])-reference.solution[j]));
            check(discrepancy<=1e-12,"persistent workspace reset and CPU solution agreement");
        }
        const auto before=context.total_transfers();const auto invalid=context.solve({1,2});check(invalid.solver.status==bspai::GmresStatus::invalid_input&&context.total_transfers().vector_uploads==before.vector_uploads,"persistent invalid call has no transfers");
        check(context.solve(b).solver.converged,"persistent context remains usable after invalid RHS");
    }catch(const std::exception& e){++failures;std::cerr<<"UNEXPECTED EXCEPTION: "<<e.what()<<'\n';}
    std::cout<<"Independent hybrid CUDA IR validation: "<<checks<<" checks, "<<failures<<" failures\n";return failures?1:0;
}
