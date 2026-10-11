// Validation-only cuBLAS norm probe: no solver, no production changes.
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <windows.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
void cuda_check(cudaError_t s){if(s!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(s));}
void blas_check(cublasStatus_t s){if(s!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("cuBLAS status "+std::to_string(s));}
int main(){
    double* device=nullptr;cublasHandle_t handle=nullptr;
    try{
        std::cout<<std::setprecision(17);cuda_check(cudaMalloc(reinterpret_cast<void**>(&device),3*sizeof(double)));blas_check(cublasCreate(&handle));blas_check(cublasSetPointerMode(handle,CUBLAS_POINTER_MODE_HOST));blas_check(cublasSetAtomicsMode(handle,CUBLAS_ATOMICS_NOT_ALLOWED));int version=0;blas_check(cublasGetVersion(handle,&version));
        const char* env=std::getenv("CUBLAS_EMULATE_DOUBLE_PRECISION");std::cout<<"cuBLAS="<<version<<", CUBLAS_EMULATE_DOUBLE_PRECISION="<<(env?env:"(unset)")<<'\n';
        char loaded[MAX_PATH]={0};HMODULE module=GetModuleHandleA("cublas64_13.dll");if(module&&GetModuleFileNameA(module,loaded,MAX_PATH))std::cout<<"Loaded cuBLAS="<<loaded<<'\n';else std::cout<<"Loaded cuBLAS path unavailable\n";
        int failures=0,total=0;const cublasMath_t modes[]={CUBLAS_DEFAULT_MATH,CUBLAS_PEDANTIC_MATH};
        for(auto mode:modes){blas_check(cublasSetMathMode(handle,mode));cublasMath_t readback;blas_check(cublasGetMathMode(handle,&readback));
            for(int count=1;count<=3;++count){int first=-1,last=-1,bad=0;
                for(int exponent=0;exponent<=1074;++exponent){const double x=std::ldexp(1.0,-exponent);std::vector<double> host(count,0);host[0]=x;
                    cuda_check(cudaMemcpy(device,host.data(),count*sizeof(double),cudaMemcpyHostToDevice));std::vector<double> readback_input(count);cuda_check(cudaMemcpy(readback_input.data(),device,count*sizeof(double),cudaMemcpyDeviceToHost));if(std::memcmp(host.data(),readback_input.data(),count*sizeof(double)))throw std::runtime_error("Uploaded device input differs bitwise");
                    double legacy=0,wide=0;blas_check(cublasDnrm2(handle,count,device,1,&legacy));blas_check(cublasDnrm2_64(handle,count,device,1,&wide));cuda_check(cudaDeviceSynchronize());total+=2;
                    if(legacy!=x||wide!=x){if(first<0)first=exponent;last=exponent;++bad;++failures;
                        if(exponent<1022||exponent<=1032||exponent>=1072)std::cout<<"NORM mode="<<readback<<", n="<<count<<", exponent="<<exponent<<", exact="<<x<<", legacy="<<legacy<<", wide="<<wide<<", ratio="<<(x?wide/x:0)<<'\n';
                    }
                }
                std::cout<<"SUMMARY mode="<<readback<<", n="<<count<<", failed_exponents="<<bad<<", first="<<first<<", last="<<last<<'\n';
            }
        }
        blas_check(cublasDestroy(handle));handle=nullptr;cuda_check(cudaFree(device));device=nullptr;std::cout<<"Norm comparisons="<<total<<", failing_cases="<<failures<<" (observational probe; nonzero means discrepancy detected)\n";return failures?2:0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';if(handle)cublasDestroy(handle);if(device)cudaFree(device);return 1;}
}
