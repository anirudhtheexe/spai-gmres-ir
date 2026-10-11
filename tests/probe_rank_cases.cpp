// Validation-only branch-coverage investigation; not a benchmark or input selector.
#include "bspai/spai.hpp"
#include <iostream>
#include <vector>
int main() {
    bspai::SpaiOptions o; o.epsilon=1e-12;o.alpha=2;o.beta=1;
    for(int p=1;p<=5;++p)for(int q=1;q<=5;++q) {
        const auto a=bspai::CsrMatrix::from_entries(2,2,{{0,0,double(p)},{0,1,double(q)},{1,0,double(p)},{1,1,double(q)}});
        const auto r=bspai::construct_spai(a,o);
        std::cout<<p<<' '<<q<<' '<<bspai::spai_status_name(r.status)<<'\n';
    }
}
