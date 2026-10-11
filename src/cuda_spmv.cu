#include "bspai/cuda_spmv.hpp"
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace bspai {
static_assert(sizeof(double)==8 && std::numeric_limits<double>::is_iec559 &&
              std::numeric_limits<double>::digits==53,"CUDA SpMV requires IEEE binary64");
static_assert(sizeof(std::size_t)==8,"CUDA SpMV requires a 64-bit host build");
namespace {
void check(cudaError_t s, const char* op) {
    if (s != cudaSuccess) throw std::runtime_error(std::string(op)+": "+cudaGetErrorString(s));
}
void check(cusparseStatus_t s, const char* op) {
    if (s != CUSPARSE_STATUS_SUCCESS) throw std::runtime_error(std::string(op)+": "+cusparseGetErrorString(s));
}
void cleanup(cudaError_t s) noexcept {
    if(s != cudaSuccess) std::cerr << "CUDA cleanup: " << cudaGetErrorString(s) << '\n';
}
void cleanup(cusparseStatus_t s) noexcept {
    if(s != CUSPARSE_STATUS_SUCCESS) std::cerr << "cuSPARSE cleanup: " << cusparseGetErrorString(s) << '\n';
}
struct DeviceGuard {
    int previous;
    explicit DeviceGuard(int device) { check(cudaGetDevice(&previous),"cudaGetDevice"); check(cudaSetDevice(device),"cudaSetDevice"); }
    ~DeviceGuard(){ cleanup(cudaSetDevice(previous)); }
};
struct StreamDrain {
    cudaStream_t stream;
    ~StreamDrain(){ cleanup(cudaStreamSynchronize(stream)); }
};
std::int64_t index64(std::size_t v) {
    if(v > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) throw std::overflow_error("CSR index exceeds int64");
    return static_cast<std::int64_t>(v);
}
std::size_t bytes(std::size_t count,std::size_t width) {
    if(count > std::numeric_limits<std::size_t>::max()/width) throw std::overflow_error("Device allocation byte count overflow");
    return count*width;
}
}
struct CudaCsrSpmv::Impl {
    std::size_t m,n,nnz;
    int device;
    cudaStream_t stream=nullptr;
    cusparseHandle_t handle=nullptr;
    cusparseSpMatDescr_t matrix=nullptr;
    cusparseDnVecDescr_t vx=nullptr,vy=nullptr;
    void *offsets=nullptr,*columns=nullptr,*values=nullptr,*x=nullptr,*y=nullptr,*workspace=nullptr;
    Impl(std::size_t rows,std::size_t cols,std::size_t count,int dev):m(rows),n(cols),nnz(count),device(dev){}
    ~Impl() {
        int previous=0;
        const cudaError_t got=cudaGetDevice(&previous);
        cleanup(got); cleanup(cudaSetDevice(device));
        if(stream) cleanup(cudaStreamSynchronize(stream));
        if(vx) cleanup(cusparseDestroyDnVec(vx));
        if(vy) cleanup(cusparseDestroyDnVec(vy));
        if(matrix) cleanup(cusparseDestroySpMat(matrix));
        if(handle) cleanup(cusparseDestroy(handle));
        for(void* p:{workspace,y,x,values,columns,offsets}) if(p) cleanup(cudaFree(p));
        if(stream) cleanup(cudaStreamDestroy(stream));
        if(got==cudaSuccess) cleanup(cudaSetDevice(previous));
    }
    void allocate(void** p,std::size_t size) { if(size) check(cudaMalloc(p,size),"cudaMalloc"); }
    void upload(void* dest,const void* source,std::size_t size) {
        if(size) {
            check(cudaMemcpyAsync(dest,source,size,cudaMemcpyHostToDevice,stream),"cudaMemcpyAsync H2D");
            check(cudaStreamSynchronize(stream),"H2D synchronization");
        }
    }
};
CudaCsrSpmv::CudaCsrSpmv(const CsrMatrix& a,int device) {
    if(a.rows()==0 || a.cols()==0) throw std::invalid_argument("CUDA SpMV requires positive dimensions");
    index64(a.rows()); index64(a.cols()); index64(a.values().size());
    if(a.row_offsets().size()!=a.rows()+1 || a.row_offsets().front()!=0 ||
       a.row_offsets().back()!=a.values().size() || a.column_indices().size()!=a.values().size())
        throw std::invalid_argument("Invalid canonical CSR structure");
    for(std::size_t row=0;row<a.rows();++row) {
        const auto first=a.row_offsets()[row],last=a.row_offsets()[row+1];
        if(first>last || last>a.values().size()) throw std::invalid_argument("Invalid CSR offsets");
        for(std::size_t k=first;k<last;++k)
            if(a.column_indices()[k]>=a.cols() || (k>first && a.column_indices()[k-1]>=a.column_indices()[k]) ||
               !std::isfinite(a.values()[k])) throw std::invalid_argument("Invalid canonical CSR indices/values");
    }
    std::vector<std::int64_t> offsets,columns;
    offsets.reserve(a.row_offsets().size()); columns.reserve(a.column_indices().size());
    for(auto v:a.row_offsets()) offsets.push_back(index64(v));
    for(auto v:a.column_indices()) columns.push_back(index64(v));
    const auto ob=bytes(offsets.size(),sizeof(std::int64_t)),cb=bytes(columns.size(),sizeof(std::int64_t));
    const auto vb=bytes(a.values().size(),sizeof(double)),xb=bytes(a.cols(),sizeof(double)),yb=bytes(a.rows(),sizeof(double));
    DeviceGuard guard(device);
    impl_.reset(new Impl(a.rows(),a.cols(),a.values().size(),device));
    Impl& p=*impl_;
    check(cudaStreamCreateWithFlags(&p.stream,cudaStreamNonBlocking),"cudaStreamCreateWithFlags");
    p.allocate(&p.x,xb); p.allocate(&p.y,yb);
    if(p.nnz) {
        p.allocate(&p.offsets,ob); p.allocate(&p.columns,cb); p.allocate(&p.values,vb);
        p.upload(p.offsets,offsets.data(),ob); p.upload(p.columns,columns.data(),cb); p.upload(p.values,a.values().data(),vb);
        check(cusparseCreate(&p.handle),"cusparseCreate");
        check(cusparseSetStream(p.handle,p.stream),"cusparseSetStream");
        check(cusparseSetPointerMode(p.handle,CUSPARSE_POINTER_MODE_HOST),"cusparseSetPointerMode");
        check(cusparseCreateCsr(&p.matrix,index64(p.m),index64(p.n),index64(p.nnz),p.offsets,p.columns,p.values,
              CUSPARSE_INDEX_64I,CUSPARSE_INDEX_64I,CUSPARSE_INDEX_BASE_ZERO,CUDA_R_64F),"cusparseCreateCsr");
        check(cusparseCreateDnVec(&p.vx,index64(p.n),p.x,CUDA_R_64F),"cusparseCreateDnVec x");
        check(cusparseCreateDnVec(&p.vy,index64(p.m),p.y,CUDA_R_64F),"cusparseCreateDnVec y");
        double alpha=1,beta=0;std::size_t size=0;
        check(cusparseSpMV_bufferSize(p.handle,CUSPARSE_OPERATION_NON_TRANSPOSE,&alpha,p.matrix,p.vx,&beta,p.vy,
              CUDA_R_64F,CUSPARSE_SPMV_CSR_ALG2,&size),"cusparseSpMV_bufferSize");
        p.allocate(&p.workspace,size);
    }
    check(cudaStreamSynchronize(p.stream),"CSR upload synchronization");
}
CudaCsrSpmv::~CudaCsrSpmv()=default;
std::size_t CudaCsrSpmv::rows()const{return impl_->m;}
std::size_t CudaCsrSpmv::cols()const{return impl_->n;}
std::vector<double> CudaCsrSpmv::multiply(const std::vector<double>& x) {
    Impl& p=*impl_;
    if(x.size()!=p.n) throw std::invalid_argument("CUDA SpMV vector length mismatch");
    for(double v:x) if(!std::isfinite(v)) throw std::invalid_argument("CUDA SpMV vector must be finite");
    DeviceGuard guard(p.device);
    std::vector<double> result(p.m);
    StreamDrain drain{p.stream}; // Host result remains alive through any error cleanup.
    p.upload(p.x,x.data(),bytes(p.n,sizeof(double)));
    check(cudaMemsetAsync(p.y,0,bytes(p.m,sizeof(double)),p.stream),"cudaMemsetAsync y");
    if(p.nnz) {
        const double alpha=1,beta=0;
        check(cusparseSpMV(p.handle,CUSPARSE_OPERATION_NON_TRANSPOSE,&alpha,p.matrix,p.vx,&beta,p.vy,
              CUDA_R_64F,CUSPARSE_SPMV_CSR_ALG2,p.workspace),"cusparseSpMV");
    }
    check(cudaMemcpyAsync(result.data(),p.y,bytes(p.m,sizeof(double)),cudaMemcpyDeviceToHost,p.stream),"cudaMemcpyAsync D2H");
    check(cudaStreamSynchronize(p.stream),"SpMV synchronization");
    for(double v:result) if(!std::isfinite(v)) throw std::overflow_error("Nonfinite GPU SpMV result");
    return result;
}
}
