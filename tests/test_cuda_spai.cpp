#include "bspai/cuda_spai.hpp"
#include "bspai/cuda_ir.hpp"
#define main saved_cpu_spai_validation_main
#include "test_spai.cpp"
#undef main

namespace {
void json_values(const std::vector<double>& values){std::cout<<'[';for(std::size_t i=0;i<values.size();++i){if(i)std::cout<<',';std::cout<<values[i];}std::cout<<']';}
void json_indices(const std::vector<std::size_t>& values){std::cout<<'[';for(std::size_t i=0;i<values.size();++i){if(i)std::cout<<',';std::cout<<values[i];}std::cout<<']';}
void json_dense(const Dense& a){std::cout<<'[';for(std::size_t i=0;i<a.size();++i){if(i)std::cout<<',';std::vector<double> row;for(auto v:a[i])row.push_back(static_cast<double>(v));json_values(row);}std::cout<<']';}
void emit_fits(const bspai::SpaiResult& r,double epsilon,bool stress=false){
    for(std::size_t k=0;k<r.columns.size();++k)for(const auto& fit:r.columns[k].history){
        std::cout<<"SPAI_ORACLE {\"B\":";json_dense(dense(r.scaled_transpose));std::cout<<",\"I\":";json_indices(fit.shadow);std::cout<<",\"J\":";json_indices(fit.pattern);
        std::cout<<",\"coefficients\":";json_values(fit.coefficients);std::cout<<",\"candidates\":";json_indices(fit.candidates);std::cout<<",\"scores\":";json_values(fit.scores);
        std::cout<<",\"target\":"<<k<<",\"residual\":"<<fit.residual<<",\"full\":"<<fit.full_residual<<",\"epsilon\":"<<epsilon<<",\"stress\":"<<(stress?"true":"false")<<",\"accepted\":"<<(fit.residual<=epsilon?"true":"false")<<"}\n";
    }
}
void structure(const bspai::SpaiResult& r,const bspai::SpaiOptions& o){
    for(std::size_t k=0;k<r.columns.size();++k){const auto& c=r.columns[k];
        check(c.fits==c.history.size()&&c.fits<=o.alpha+1,"device completed-fit history/budget");
        check(c.fit_attempts>=c.fits&&c.growth_attempts>=c.growths,"attempted vs completed work");
        check(c.pattern.size()==c.coefficients.size(),"failed/successful solved pattern coefficient alignment");
        if(c.history.empty()){check(std::isnan(c.full_residual),"unavailable device residual explicitly NaN");continue;}
        check(c.history.front().pattern==std::vector<std::size_t>{k},"device identity initial pattern");check(c.pattern==c.history.back().pattern,"device publishes only last solved pattern");
        for(std::size_t h=0;h<c.history.size();++h){const auto& f=c.history[h];
            check(std::is_sorted(f.pattern.begin(),f.pattern.end())&&std::adjacent_find(f.pattern.begin(),f.pattern.end())==f.pattern.end(),"sorted unique extraction set");
            check(f.shadow.size()>=f.pattern.size(),"unpadded tall reduced problem dimensions");
            check(f.selected.size()<=o.beta,"candidate selection beta bound");
            if(!f.candidates.empty()){const auto b=dense(r.scaled_transpose);auto rows=f.shadow;rows.push_back(k);std::vector<std::size_t> expected_candidates;
                for(std::size_t j=0;j<b.size();++j){bool occupied=false;for(auto row:rows)occupied=occupied||b[row][j]!=0;if(occupied&&!std::binary_search(f.pattern.begin(),f.pattern.end(),j))expected_candidates.push_back(j);}
                check(expected_candidates==f.candidates,"complete restricted candidate union excludes J");
            }
            check(f.scores.size()<=f.candidates.size(),"candidate score diagnostics bounded by candidate count");
            if(f.scores.size()!=f.candidates.size())continue;
            std::vector<std::pair<double,std::size_t>> acceptable;
            for(std::size_t j=0;j<f.candidates.size();++j)if(f.scores[j]<=f.mean_score)acceptable.push_back({f.scores[j],f.candidates[j]});
            std::sort(acceptable.begin(),acceptable.end());std::vector<std::size_t> selected;for(std::size_t j=0;j<std::min(o.beta,acceptable.size());++j)selected.push_back(acceptable[j].second);
            check(f.selected==selected,"independent score/mean/tie order selection");
            if(h+1<c.history.size()){auto next=f.pattern;next.insert(next.end(),selected.begin(),selected.end());std::sort(next.begin(),next.end());next.erase(std::unique(next.begin(),next.end()),next.end());check(next==c.history[h+1].pattern,"growth union equals next solved pattern");}
        }
        if(c.status==bspai::SpaiStatus::success)check(c.reduced_residual<=o.epsilon,"GPU column accepts unchanged epsilon");
    }
}
bspai::CudaSpaiResult compare(const Dense& a,const bspai::SpaiOptions& o,bool successful=true){
    const auto gpu=bspai::cuda_construct_spai(matrix(a),o);const auto cpu=bspai::construct_spai(matrix(a),o);const auto& r=gpu.construction;
    check(r.status==cpu.status,"ordinary CPU/GPU constructor status agreement");check((r.status==bspai::SpaiStatus::success)==successful,"expected constructor success/failure");structure(r,o);
    check(r.columns.size()==cpu.columns.size(),"CPU/GPU all-column diagnostic size");
    for(std::size_t k=0;k<r.columns.size()&&k<cpu.columns.size();++k){const auto& c=r.columns[k];const auto& ref=cpu.columns[k];check(c.status==ref.status,"ordinary per-column CPU/GPU status");check(c.pattern==ref.pattern&&c.fits==ref.fits&&c.growths==ref.growths,"ordinary CPU/GPU pattern/work count");
        for(std::size_t j=0;j<c.coefficients.size()&&j<ref.coefficients.size();++j)check(near(c.coefficients[j],ref.coefficients[j]),"ordinary CPU/GPU coefficients within declared allowance");
    }
    if(successful){audit_result(a,r,o.epsilon);check(gpu.telemetry.gpu_scaling&&gpu.telemetry.gpu_preconditioner_assembly,"GPU numeric scaling/P assembly flags");check(gpu.telemetry.qr_submissions>0,"device QR work submitted");}
    emit_fits(r,o.epsilon);std::cout<<"GPU SPAI status="<<bspai::spai_status_name(r.status)<<", streams="<<gpu.telemetry.streams_created<<", fit_batch="<<gpu.telemetry.max_fit_batch<<", qr_before_collect="<<gpu.telemetry.max_qr_submissions_before_collection<<'\n';return gpu;
}
void solve_audit(const Dense& a,const std::vector<double>& b,const bspai::CudaIrResult& r){
    check(r.gpu_construction_performed&&r.solver.construction_performed,"IR explicitly marks GPU SPAI construction");check(r.solver.converged,"GPU P/GPU GMRES/CPU outer converges");
    const auto p=dense(r.solver.construction.preconditioner);std::vector<double> initial(b.size(),0);for(std::size_t i=0;i<b.size();++i)for(std::size_t j=0;j<b.size();++j)initial[i]+=static_cast<double>(p[i][j])*b[j];check(r.solver.history[0].x==initial,"GPU-produced P initializes x0=P*b");
    for(std::size_t h=0;h<r.solver.history.size();++h){const auto& state=r.solver.history[h];std::vector<long double> residual(b.size());long double ainf=0,xinf=0,binf=0,cbe=0;
        for(std::size_t i=0;i<b.size();++i){long double ax=0,row=0,den=std::fabs(b[i]);for(std::size_t j=0;j<b.size();++j){ax+=a[i][j]*state.x[j];row+=std::fabs(a[i][j]);den+=std::fabs(a[i][j])*std::fabs(state.x[j]);}residual[i]=b[i]-ax;ainf=std::max(ainf,row);xinf=std::max(xinf,std::fabs(static_cast<long double>(state.x[i])));binf=std::max(binf,std::fabs(static_cast<long double>(b[i])));cbe=std::max(cbe,den?std::fabs(residual[i])/den:0);check(near(state.diagnostics.residual[i],residual[i]),"independent original outer residual with GPU P");}
        long double rinf=0;for(auto v:residual)rinf=std::max(rinf,std::fabs(v));const long double den=ainf*xinf+binf;check(near(state.diagnostics.normwise_backward_error,den?rinf/den:0)&&near(state.diagnostics.componentwise_backward_error,cbe),"independent original BE with GPU P");
        if(h){for(std::size_t i=0;i<b.size();++i)check(state.x[i]==r.solver.history[h-1].x[i]+state.inner.solution[i],"GPU P IR accepted correction update");}
        const auto& d=state.diagnostics;std::cout<<"IR_ORACLE {\"A\":";json_dense(a);std::cout<<",\"b\":";json_values(b);std::cout<<",\"x\":";json_values(state.x);std::cout<<",\"residual\":";json_values(d.residual);std::cout<<",\"inf\":"<<d.residual_inf<<",\"norm2\":"<<d.residual_2<<",\"nbe\":"<<d.normwise_backward_error<<",\"cbe\":"<<d.componentwise_backward_error<<",\"accepted\":"<<(d.normwise_backward_error<=bspai::ir_backward_tolerance()?"true":"false")<<"}\n";
    }
    check(r.solver.history.back().diagnostics.normwise_backward_error<=bspai::ir_backward_tolerance(),"GPU P IR fixed outer acceptance");for(const auto& t:r.inner_transfers)check(t.arnoldi_vector_uploads==0&&t.arnoldi_vector_downloads==0,"GPU P IR inner vector residency");
}
}
int main(){using bspai::SpaiStatus;try{
    std::cout<<std::setprecision(17);const Dense a={{4,1,0},{2,5,1},{0,1,3}};bspai::SpaiOptions o;o.epsilon=1e-12;o.alpha=2;o.beta=8;
    const auto full=compare(a,o);const auto repeat=bspai::cuda_construct_spai(matrix(a),o);check(full.construction.inverse.values()==repeat.construction.inverse.values()&&full.construction.preconditioner.values()==repeat.construction.preconditioner.values(),"fixed-stack repeated GPU construction bitwise values");
    const Dense inverse={{14.0L/50,-3.0L/50,1.0L/50},{-6.0L/50,12.0L/50,-4.0L/50},{2.0L/50,-4.0L/50,18.0L/50}};const auto p=dense(full.construction.preconditioner);for(std::size_t i=0;i<3;++i)for(std::size_t j=0;j<3;++j)check(near(p[i][j],inverse[i][j]),"GPU full P exact rational inverse");
    o.epsilon=.3;compare(a,o);o.epsilon=1e-12;o.alpha=1;compare(a,o,false);o.alpha=0;const auto first=compare(a,o,false);
    o.epsilon=first.construction.columns[0].reduced_residual;const auto equality=bspai::cuda_construct_spai(matrix(a),o);check(equality.construction.columns[0].status==SpaiStatus::success,"GPU <=epsilon equality boundary");o.epsilon=std::nextafter(o.epsilon,0.0);check(bspai::cuda_construct_spai(matrix(a),o).construction.columns[0].status==SpaiStatus::iteration_limit,"GPU oneULP below own residual rejects");
    o.epsilon=1e-12;o.alpha=2;o.beta=1;const auto tied=compare({{1,1,1},{1,1,0},{1,0,1}},o);const auto& tf=tied.construction.columns[0].history[0];check(tf.scores.size()==2&&tf.scores[0]==tf.scores[1]&&tf.selected==std::vector<std::size_t>{1},"actual exact device score tie chooses lowest index");
    o.beta=0;compare(a,o,false);o.beta=8;o.alpha=0;compare({{2,0},{0,-4}},o);
    const double tiny=std::numeric_limits<double>::denorm_min();o.epsilon=tiny;const auto subnormal=bspai::cuda_construct_spai(matrix({{1,static_cast<long double>(tiny)},{0,1}}),o);check(subnormal.construction.status==SpaiStatus::success,"GPU QR preserves subnormal offdiagonal singleton fit");check(subnormal.construction.columns[0].full_residual==tiny&&subnormal.construction.columns[0].history[0].shadow==std::vector<std::size_t>{0,1},"GPU tiny original-B residual retained exactly after vendorQR");emit_fits(subnormal.construction,o.epsilon);o.epsilon=1e-12;
    Dense uneven(8,std::vector<long double>(8,0));for(std::size_t i=0;i<3;++i)for(std::size_t j=0;j<3;++j)uneven[i][j]=a[i][j];uneven[3][3]=2;uneven[4][4]=-4;uneven[5][5]=8;uneven[6][6]=3;uneven[6][7]=1;uneven[7][6]=2;uneven[7][7]=5;o.alpha=2;
    const auto batch=compare(uneven,o);std::size_t minfits=99,maxfits=0;for(const auto& c:batch.construction.columns){minfits=std::min(minfits,c.fits);maxfits=std::max(maxfits,c.fits);}check(minfits<maxfits,"uneven actual pattern growth histories exercised");check(batch.telemetry.max_qr_submissions_before_collection>1&&batch.telemetry.parallel_fit_batches>0,"multiple QR submissions before collection (not timing/overlap proof)");
    o.alpha=3;for(const auto& singular:std::vector<Dense>{{{1,1},{1,1}},{{1,2},{1,2}}}){const auto gpu=bspai::cuda_construct_spai(matrix(singular),o);const auto cpu=bspai::construct_spai(matrix(singular),o);check(gpu.construction.status!=SpaiStatus::success&&!gpu.construction.message.empty(),"degenerate fixture reports failure without rank certification");structure(gpu.construction,o);std::cout<<"Degenerate CPU="<<bspai::spai_status_name(cpu.status)<<", GPU="<<bspai::spai_status_name(gpu.construction.status)<<'\n';
        for(std::size_t k=0;k<gpu.construction.columns.size();++k){const auto& c=gpu.construction.columns[k];std::cout<<"Degenerate GPU column="<<k<<", status="<<bspai::spai_status_name(c.status)<<", fits="<<c.fits<<", pivot_ratio="<<c.qr_pivot_ratio<<", full_residual="<<c.full_residual<<", candidates="<<(c.history.empty()?0:c.history.back().candidates.size())<<'\n';}
    }
    for(const auto& bad:std::vector<Dense>{{{0,1},{1,1}},{{1,2,0},{0,1,2}},{{1e-320L,0},{0,1}},{{1e308L,1e-300L},{0,1}}})check(bspai::cuda_construct_spai(matrix(bad),o).construction.status!=SpaiStatus::success,"GPU input/scaling range failure");
    for(double epsilon:std::vector<double>{0,-1,std::numeric_limits<double>::quiet_NaN(),std::numeric_limits<double>::infinity()}){o.epsilon=epsilon;check(bspai::cuda_construct_spai(matrix(a),o).construction.status==SpaiStatus::invalid_input,"GPU invalid epsilon rejected");}
    const double boundary=1+16*std::numeric_limits<double>::epsilon();
    struct ScoreCase{double r,dot,den;SpaiStatus status;bool clamp;};
    for(const auto& c:std::vector<ScoreCase>{{4,1,2,SpaiStatus::success,false},{1,std::nextafter(1.0,2.0),1,SpaiStatus::success,true},{1,boundary,1,SpaiStatus::success,true},{1,std::nextafter(boundary,2.0),1,SpaiStatus::invalid_score,false},{1,1.1,1,SpaiStatus::invalid_score,false},{1,1,0,SpaiStatus::zero_denominator,false},{-1,1,1,SpaiStatus::invalid_score,false},{1,1,-1,SpaiStatus::invalid_score,false},{1,std::numeric_limits<double>::infinity(),1,SpaiStatus::numerical_failure,false}}){
        const auto got=bspai::cuda_evaluate_spai_score(c.r,c.dot,c.den);const auto ref=bspai::evaluate_spai_score(c.r,c.dot,c.den);check(got.status==c.status&&got.status==ref.status,"actual shared device score guard status");check(got.clamped==c.clamp,"actual device score clamp boundary flag");if(got.status==SpaiStatus::success)check(got.score==ref.score,"actual device score result exact binary64 comparison");
    }
    o.epsilon=1e-12;o.alpha=2;const auto coefficient_range=bspai::cuda_construct_spai(matrix({{static_cast<long double>(tiny),1},{0,1}}),o);check(coefficient_range.construction.status!=SpaiStatus::success,"GPU unrepresentable inverse/pivot range fails explicitly");std::cout<<"GPU coefficient range status="<<bspai::spai_status_name(coefficient_range.construction.status)<<'\n';
    o.epsilon=1e-5;o.alpha=1;o.beta=1;const double delta=std::ldexp(1.0,-30);const auto stress=bspai::cuda_construct_spai(matrix({{1,1},{1,1+static_cast<long double>(delta)}}),o);check(stress.construction.status==SpaiStatus::success,"GPU near-dependent finite pivot stress accepted without cutoff");structure(stress.construction,o);emit_fits(stress.construction,o.epsilon,true);std::cout<<"GPU near-dependent status="<<bspai::spai_status_name(stress.construction.status)<<", pivot="<<stress.construction.columns[0].qr_pivot_ratio<<'\n';
    bspai::IrOptions ir;ir.spai.epsilon=.3;ir.spai.alpha=2;ir.spai.beta=8;const auto integrated=bspai::cuda_gpu_spai_gmres_ir(matrix(a),{2,-5,7},ir,{1,-2,3});solve_audit(a,{2,-5,7},integrated);for(std::size_t i=0;i<3;++i)check(near(integrated.solver.solution[i],std::vector<double>{1,-2,3}[i],1e-12L),"GPU construction/GMRES/IR exact solution reference");
    const auto cpu_integrated=bspai::spai_gmres_ir(matrix(a),{2,-5,7},ir);check(integrated.solver.status==cpu_integrated.status&&integrated.solver.outer_steps==cpu_integrated.outer_steps,"GPU SPAI end-to-end CPU IR status/count agreement");for(std::size_t j=0;j<3;++j)check(near(integrated.solver.solution[j],cpu_integrated.solution[j],1e-12L),"GPU SPAI end-to-end CPU IR solution agreement");
    const auto general=bspai::cuda_gpu_spai_gmres_ir(matrix(a),{.1,-.7,1.3},ir);solve_audit(a,{.1,-.7,1.3},general);const auto zero=bspai::cuda_gpu_spai_gmres_ir(matrix(a),{0,0,0},ir);solve_audit(a,{0,0,0},zero);check(zero.solver.inner_attempts==0,"GPU construction with zeroRHS accepts original initialization");
    const auto direct=bspai::cuda_left_preconditioned_gmres(matrix(a),full.construction.preconditioner,{2,-5,7},ir.gmres);const auto cpu_direct=bspai::left_preconditioned_gmres(matrix(a),full.construction.preconditioner,{2,-5,7},ir.gmres);check(direct.solver.converged&&direct.solver.status==cpu_direct.status,"GPU-produced full P direct GPU/CPU GMRES agreement");for(std::size_t j=0;j<3;++j)check(near(direct.solver.solution[j],std::vector<double>{1,-2,3}[j],1e-12L),"GPU-produced P direct GMRES exactsolution");
    ir.spai.epsilon=1e-12;const auto failure=bspai::cuda_gpu_spai_gmres_ir(matrix({{1,2},{1,2}}),{1,0},ir);check(failure.solver.status==bspai::IrStatus::construction_failure&&!failure.solver.converged&&failure.context_constructions==0,"failed GPU construction prevents inner context/partial P use");
}catch(const std::exception& e){++failures;std::cerr<<"UNEXPECTED EXCEPTION: "<<e.what()<<'\n';}
std::cout<<"Independent CUDA SPAI validation: "<<checks<<" checks, "<<failures<<" failures\n";return failures?1:0;}
