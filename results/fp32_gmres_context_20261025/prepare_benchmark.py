from pathlib import Path

path = Path(__file__).with_name("context_benchmark.cpp")
source = path.read_text()
needle = 'if(a.mode=="prepare_fixed_p")'
replacement = '''if(a.mode=="gpu_ir_fixed") {
    if(a.p_path.empty()) throw std::invalid_argument("missing --p");
    const auto P=load(a.p_path);
    samples(a,true,[&](){return bspai::fp32_cuda_refine_with_preconditioner(A,P,b,o);},
      [&](std::ostream&z,const bspai::Fp32IrResult&r){
        z<<"\\"status\\":"<<quote(bspai::fp32_ir_status_name(r.status))
         <<",\\"converged\\":"<<(r.converged?"true":"false")
         <<",\\"outer_steps\\":"<<r.outer_steps
         <<",\\"inner_attempts\\":"<<r.inner_attempts<<',';
        work(z,r);z<<',';diag(z,A,b,r.solution);
        z<<",\\"solution\\":";values(z,r.solution);
      });
    return 0;
} if(a.mode=="prepare_fixed_p")'''
assert source.count(needle) == 1
path.write_text(source.replace(needle, replacement))
