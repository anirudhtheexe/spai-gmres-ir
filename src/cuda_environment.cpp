#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <iostream>
int main() {
    int runtime=0,driver=0,count=0,sparse=0,blas=0,solver_major=0,solver_minor=0,solver_patch=0;
    if(cudaRuntimeGetVersion(&runtime)!=cudaSuccess || cudaDriverGetVersion(&driver)!=cudaSuccess ||
       cudaGetDeviceCount(&count)!=cudaSuccess) return 1;
    cusparseHandle_t sh=nullptr;cublasHandle_t bh=nullptr;
    if(cusparseCreate(&sh)!=CUSPARSE_STATUS_SUCCESS) return 1;
    const auto s=cusparseGetVersion(sh,&sparse);
    const auto sd=cusparseDestroy(sh);
    if(s!=CUSPARSE_STATUS_SUCCESS || sd!=CUSPARSE_STATUS_SUCCESS) return 1;
    if(cublasCreate(&bh)!=CUBLAS_STATUS_SUCCESS) return 1;
    const auto b=cublasGetVersion(bh,&blas);
    const auto bd=cublasDestroy(bh);
    if(b!=CUBLAS_STATUS_SUCCESS || bd!=CUBLAS_STATUS_SUCCESS) return 1;
    if(cusolverGetProperty(MAJOR_VERSION,&solver_major)!=CUSOLVER_STATUS_SUCCESS ||
       cusolverGetProperty(MINOR_VERSION,&solver_minor)!=CUSOLVER_STATUS_SUCCESS ||
       cusolverGetProperty(PATCH_LEVEL,&solver_patch)!=CUSOLVER_STATUS_SUCCESS)return 1;
    std::cout << "runtime="<<runtime<<", driver_api="<<driver<<", cuSPARSE="<<sparse
              <<", cuBLAS="<<blas<<", cuSOLVER="<<solver_major<<'.'<<solver_minor<<'.'<<solver_patch<<", device_count="<<count<<'\n';
    for(int i=0;i<count;++i){cudaDeviceProp p{};if(cudaGetDeviceProperties(&p,i)!=cudaSuccess)return 1;
        std::cout<<"device="<<i<<", name="<<p.name<<", CC="<<p.major<<'.'<<p.minor
                 <<", memory_bytes="<<p.totalGlobalMem<<'\n';}
    return 0;
}
