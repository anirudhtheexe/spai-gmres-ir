#include "bspai/cuda_spmv.hpp"
#include <cuda_runtime_api.h>
#include <cusparse.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
int checks=0, failures=0;
long double maximum_cpu_difference=0,maximum_bound_fraction=0;
void check(bool ok,const std::string& name){++checks;if(!ok){++failures;std::cerr<<"FAIL: "<<name<<'\n';}}
template<class Error,class Function>void rejects(Function f,const std::string& name){bool expected=false;try{f();}catch(const Error&){expected=true;}catch(const std::exception& e){std::cerr<<"Unexpected exception for "<<name<<": "<<e.what()<<'\n';}check(expected,name);}
bspai::CsrMatrix make(std::size_t m,std::size_t n,std::vector<bspai::Entry> entries){return bspai::CsrMatrix::from_entries(m,n,entries);}
bool same_bits(const std::vector<double>& a,const std::vector<double>& b){return a.size()==b.size()&&(a.empty()||std::memcmp(a.data(),b.data(),a.size()*sizeof(double))==0);}
void compare(const bspai::CsrMatrix& a,const std::vector<double>& x,const std::vector<double>& y,const std::vector<double>& exact={}){
    const auto cpu=a.multiply(x);check(y.size()==a.rows(),"GPU result dimension");
    if(y.size()!=a.rows())return;
    for(std::size_t i=0;i<a.rows();++i){
        const std::size_t count=a.row_offsets()[i+1]-a.row_offsets()[i];
        long double sumabs=0;
        for(std::size_t k=a.row_offsets()[i];k<a.row_offsets()[i+1];++k)sumabs+=std::fabs(static_cast<long double>(a.values()[k])*x[a.column_indices()[k]]);
        const long double u=std::ldexp(1.0L,-53),n=static_cast<long double>(2*count+1);
        const long double gamma=n*u/(1-n*u);
        // Project differential threshold, not a cuSPARSE or paper guarantee.
        // No absolute-one floor: tiny-scale erroneous zeros must not pass.
        const long double allowance=4*gamma*sumabs+static_cast<long double>(count)*std::numeric_limits<double>::denorm_min();
        check(std::isfinite(y[i]),"GPU output finite");
        const long double difference=std::fabs(static_cast<long double>(y[i])-cpu[i]);
        maximum_cpu_difference=std::max(maximum_cpu_difference,difference);
        if(allowance>0)maximum_bound_fraction=std::max(maximum_bound_fraction,difference/allowance);
        check(difference<=allowance,"rowlength/scale-aware CPU versus GPU bound");
        if(!exact.empty())check(y[i]==exact[i]&&cpu[i]==exact[i],"independent exact dyadic reference");
    }
}
void exercise(const bspai::CsrMatrix& a,const std::vector<double>& x,const std::vector<double>& exact={}){
    bspai::CudaCsrSpmv gpu(a);check(gpu.rows()==a.rows()&&gpu.cols()==a.cols(),"device object retains shape");
    const auto first=gpu.multiply(x);compare(a,x,first,exact);
    for(int repeat=0;repeat<4;++repeat){const auto again=gpu.multiply(x);check(same_bits(first,again),"same input repeated deterministic bits");}
    const std::vector<double> zero(a.cols(),0);const auto zero_product=gpu.multiply(zero);
    compare(a,zero,zero_product,std::vector<double>(a.rows(),0));
    check(same_bits(first,gpu.multiply(x)),"changed input does not leave stale vector/output");
    rejects<std::invalid_argument>([&](){gpu.multiply(std::vector<double>(a.cols()+1,1));},"wrong input length rejected");
    std::vector<double> bad=x;bad[0]=std::numeric_limits<double>::quiet_NaN();
    rejects<std::invalid_argument>([&](){gpu.multiply(bad);},"NaN input rejected");
    bad[0]=std::numeric_limits<double>::infinity();
    rejects<std::invalid_argument>([&](){gpu.multiply(bad);},"infinite input rejected");
    check(same_bits(first,gpu.multiply(x)),"object usable after rejected input");
}
}

int main(int argc,char** argv){
    try{
        const bool memory_safety_only=argc==2&&std::string(argv[1])=="--memory-safety-only";
        if(argc>1&&!memory_safety_only)throw std::invalid_argument("Only --memory-safety-only is supported");
        if(memory_safety_only)std::cout<<"Sanitizer mode: skipping only the two deliberate invalid-device CUDA API calls\n";
        std::cout<<std::setprecision(17);int runtime=0,driver=0,devices=0;
        check(cudaRuntimeGetVersion(&runtime)==cudaSuccess,"query runtime version");
        check(cudaDriverGetVersion(&driver)==cudaSuccess,"query driver API version");
        check(cudaGetDeviceCount(&devices)==cudaSuccess&&devices>0,"CUDA device available");
        if(devices<=0)throw std::runtime_error("No CUDA device; validation cannot execute");
        cudaDeviceProp prop;check(cudaGetDeviceProperties(&prop,0)==cudaSuccess,"query device properties");
        cusparseHandle_t handle=nullptr;check(cusparseCreate(&handle)==CUSPARSE_STATUS_SUCCESS,"query handle creation");int library=0;
        if(handle){check(cusparseGetVersion(handle,&library)==CUSPARSE_STATUS_SUCCESS,"query cuSPARSE version");check(cusparseDestroy(handle)==CUSPARSE_STATUS_SUCCESS,"query handle cleanup");}
        std::cout<<"GPU="<<prop.name<<", CC="<<prop.major<<'.'<<prop.minor<<", bytes="<<prop.totalGlobalMem<<", runtime="<<runtime<<", driver_api="<<driver<<", cuSPARSE="<<library<<", double_digits="<<std::numeric_limits<double>::digits<<", long_double_digits="<<std::numeric_limits<long double>::digits<<'\n';
        const auto tiny=make(3,3,{{0,0,4},{0,1,1},{1,0,2},{1,1,5},{1,2,1},{2,1,1},{2,2,3}});
        exercise(tiny,{1,-2,3},{2,-5,7});
        // Rectangular, leading/internal/trailing empty rows; sorted transfer.
        exercise(make(5,4,{{3,3,8},{1,2,-2},{1,0,4}}),{.5,2,3,-.25},{0,-4,0,-2,0});
        // Duplicate cancellation/canonicalization occurs in the trusted CPU loader.
        std::istringstream mm("%%MatrixMarket matrix coordinate real general\n4 3 6\n3 2 8\n1 1 3\n3 2 -2\n1 1 -3\n4 3 0\n2 1 2\n");
        const auto canonical=bspai::read_matrix_market(mm);check(canonical.values()==std::vector<double>({2,6}),"duplicate-canonicalized transfer fixture");
        exercise(canonical,{2,-.5,1},{0,4,-3,0});
        exercise(make(3,5,{}),{1,2,3,4,5},{0,0,0});
        exercise(make(1,1,{{0,0,-4}}),{.25},{-1});
        // Tiny normal and subnormal FP64 values are not discarded under a broad absolute tolerance.
        exercise(make(2,2,{{0,0,std::ldexp(1.0,-900)},{1,1,std::numeric_limits<double>::denorm_min()}}),{2,1},{std::ldexp(1.0,-899),std::numeric_limits<double>::denorm_min()});
        std::vector<bspai::Entry> row;std::vector<double> x(257,0);double exact=0;
        for(std::size_t j=0;j<257;++j){const double value=(j%2?-1:1)*std::ldexp(1.0,static_cast<int>(j%7)-3);x[j]=static_cast<double>(static_cast<int>(j%5)-2);row.push_back({0,j,value});exact+=value*x[j];}
        exercise(make(2,257,row),x,{exact,0});
        // Decimal inputs allow legitimate FMA/reduction differences, without claiming bitwise CPU equality.
        exercise(make(2,4,{{0,0,.1},{0,1,-.7},{0,2,1.3},{0,3,.2},{1,0,1e16},{1,1,1},{1,2,-1e16}}),{1,1,1,1});
        if(!memory_safety_only){
            rejects<std::runtime_error>([&](){bspai::CudaCsrSpmv invalid(tiny,devices);},"out-of-range device rejected");
            rejects<std::runtime_error>([&](){bspai::CudaCsrSpmv invalid(tiny,-1);},"negative device rejected");
        }
        rejects<std::invalid_argument>([&](){bspai::CudaCsrSpmv invalid(make(0,2,{}));},"zero row dimension rejected");
        rejects<std::invalid_argument>([&](){bspai::CudaCsrSpmv invalid(make(2,0,{}));},"zero column dimension rejected");
        rejects<std::overflow_error>([&](){bspai::CudaCsrSpmv invalid(make(1,std::numeric_limits<std::size_t>::max()/2+1,{}));},"64-bit index overflow rejected before allocation");
        rejects<std::overflow_error>([&](){bspai::CudaCsrSpmv invalid(make(1,std::numeric_limits<std::size_t>::max()/sizeof(double)+1,{}));},"byte count overflow rejected before allocation");
        auto moved=tiny;const auto retained=std::move(moved);(void)retained;
        rejects<std::invalid_argument>([&](){bspai::CudaCsrSpmv invalid(moved);},"moved-from noncanonical host CSR rejected");
        bspai::CudaCsrSpmv copied_from_temporary(make(2,2,{{0,1,2},{1,0,-4}}));
        compare(make(2,2,{{0,1,2},{1,0,-4}}),{1,3},copied_from_temporary.multiply({1,3}),{6,-4});
        // Lifetime isolation and destruction/reconstruction, not an allocation leak measurement.
        for(int i=0;i<3;++i){bspai::CudaCsrSpmv a(tiny),b(canonical);compare(tiny,{1,-2,3},a.multiply({1,-2,3}),{2,-5,7});compare(canonical,{2,-.5,1},b.multiply({2,-.5,1}),{0,4,-3,0});}
        const auto large=make(1,1,{{0,0,std::numeric_limits<double>::max()}});
        bspai::CudaCsrSpmv large_gpu(large);
        rejects<std::runtime_error>([&](){large_gpu.multiply({2});},"nonfinite computed output rejected");
        compare(large,{.5},large_gpu.multiply({.5}),{std::numeric_limits<double>::max()*.5});
        check(cudaDeviceSynchronize()==cudaSuccess,"final CUDA synchronization status");
    }catch(const std::exception& e){++failures;std::cerr<<"UNEXPECTED EXCEPTION: "<<e.what()<<'\n';}
    std::cout<<"Maximum CPU/GPU absolute difference="<<maximum_cpu_difference<<", maximum differential-bound fraction="<<maximum_bound_fraction<<'\n';
    std::cout<<"Independent CUDA SpMV validation: "<<checks<<" checks, "<<failures<<" failures\n";return failures?1:0;
}
