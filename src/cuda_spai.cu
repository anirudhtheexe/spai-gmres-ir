#include "bspai/cuda_spai.hpp"
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <algorithm>
#include <cmath>
#include <climits>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>

namespace bspai { namespace {
void check(cudaError_t s,const char* op){if(s!=cudaSuccess)throw std::runtime_error(std::string(op)+": "+cudaGetErrorString(s));}
void check(cusolverStatus_t s,const char* op){if(s!=CUSOLVER_STATUS_SUCCESS)throw std::runtime_error(std::string(op)+": cuSOLVER status "+std::to_string(static_cast<int>(s)));}
void check(cublasStatus_t s,const char* op){if(s!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error(std::string(op)+": cuBLAS status "+std::to_string(static_cast<int>(s)));}
void cleanup(cudaError_t s){if(s!=cudaSuccess)std::cerr<<"GPU SPAI cleanup: "<<cudaGetErrorString(s)<<'\n';}
std::size_t bytes(std::size_t n,std::size_t width){if(n && width>std::numeric_limits<std::size_t>::max()/n)throw std::overflow_error("GPU SPAI size overflow");return n*width;}
int dim(std::size_t n){if(n>INT_MAX)throw std::invalid_argument("GPU QR requires dimensions <= INT_MAX");return static_cast<int>(n);}
struct Guard{int old;explicit Guard(int device){check(cudaGetDevice(&old),"SPAI get device");check(cudaSetDevice(device),"SPAI set device");}~Guard(){cleanup(cudaSetDevice(old));}};
struct Memory{
    void* p=nullptr;std::size_t capacity=0;
    explicit Memory(std::size_t size=0){allocate(size);}~Memory(){if(p)cleanup(cudaFree(p));}
    void allocate(std::size_t size){if(size<=capacity)return;if(p){check(cudaFree(p),"SPAI resize free");p=nullptr;capacity=0;}check(cudaMalloc(&p,size),"SPAI allocation");capacity=size;}
    double* d(){return static_cast<double*>(p);}int* i(){return static_cast<int*>(p);}std::uint64_t* u(){return static_cast<std::uint64_t*>(p);}
    Memory(const Memory&)=delete;Memory&operator=(const Memory&)=delete;
};
struct Stream{cudaStream_t p=nullptr;Stream(){check(cudaStreamCreateWithFlags(&p,cudaStreamNonBlocking),"SPAI create stream");}~Stream(){cleanup(cudaStreamSynchronize(p));cleanup(cudaStreamDestroy(p));}};
void upload(Memory& target,const void* host,std::size_t size,Stream& s,CudaSpaiTelemetry& t){if(!size)return;check(cudaMemcpyAsync(target.p,host,size,cudaMemcpyHostToDevice,s.p),"SPAI H2D");check(cudaStreamSynchronize(s.p),"SPAI upload lifetime");++t.uploads;t.upload_bytes+=size;}
void download(void* host,const void* device,std::size_t size,Stream& s,CudaSpaiTelemetry& t){if(!size)return;check(cudaMemcpyAsync(host,device,size,cudaMemcpyDeviceToHost,s.p),"SPAI D2H");check(cudaStreamSynchronize(s.p),"SPAI download lifetime");++t.downloads;t.download_bytes+=size;}
void sync(Stream& s){check(cudaStreamSynchronize(s.p),"SPAI stream completion");}
__device__ void accumulate(double x,double& scale,double& sum){
    const double a=fabs(x);if(a!=0){if(scale<a){const double r=scale/a;sum=1+sum*r*r;scale=a;}else{const double r=a/scale;sum+=r*r;}}
}
__global__ void scale_matrix(const std::uint64_t* offsets,const std::uint64_t* columns,const double* values,
    std::size_t n,double* d,double* b,int* status){
    const std::size_t row=blockIdx.x*static_cast<std::size_t>(blockDim.x)+threadIdx.x;if(row>=n)return;
    double maximum=0,diagonal=0;for(auto j=offsets[row];j<offsets[row+1];++j){maximum=fmax(maximum,fabs(values[j]));if(columns[j]==row)diagonal=values[j];}
    status[row]=0;if(diagonal==0){status[row]=1;return;}const double scale=1/maximum;
    if(!isfinite(scale)||scale==0){status[row]=2;return;}d[row]=scale;
    for(auto j=offsets[row];j<offsets[row+1];++j){const double v=values[j]*scale;if(!isfinite(v)||v==0){status[row]=3;return;}b[columns[j]+row*n]=v;}
}
__global__ void gather(const double* b,std::size_t n,const std::uint64_t* shadow,const std::uint64_t* pattern,
    int m,int q,std::size_t k,double* reduced,double* rhs){
    for(std::size_t at=blockIdx.x*static_cast<std::size_t>(blockDim.x)+threadIdx.x;at<static_cast<std::size_t>(m)*q;at+=static_cast<std::size_t>(blockDim.x)*gridDim.x){
        const std::size_t row=at%m,col=at/m;reduced[at]=b[shadow[row]+pattern[col]*n];
    }
    for(std::size_t row=blockIdx.x*static_cast<std::size_t>(blockDim.x)+threadIdx.x;row<static_cast<std::size_t>(m);row+=static_cast<std::size_t>(blockDim.x)*gridDim.x)rhs[row]=shadow[row]==k?1:0;
}
__global__ void qr_guard(const double* b,std::size_t n,const std::uint64_t* shadow,const std::uint64_t* pattern,
    int m,int q,const double* r,const double* tau,const double* rhs,int* status,double* diagnostics){
    int code=static_cast<int>(SpaiStatus::success);double minimum=INFINITY,inf=0;
    for(int row=0;row<m;++row){double sum=0;for(int col=0;col<q;++col){sum+=fabs(b[shadow[row]+pattern[col]*n]);if(!isfinite(r[row+static_cast<std::size_t>(col)*m]))code=static_cast<int>(SpaiStatus::numerical_failure);}inf=fmax(inf,sum);if(!isfinite(rhs[row]))code=static_cast<int>(SpaiStatus::numerical_failure);}
    for(int j=0;j<q;++j){const double pivot=r[j+static_cast<std::size_t>(j)*m];if(!isfinite(pivot)||!isfinite(tau[j]))code=static_cast<int>(SpaiStatus::numerical_failure);else if(pivot==0&&code==static_cast<int>(SpaiStatus::success))code=static_cast<int>(SpaiStatus::rank_deficient);minimum=fmin(minimum,fabs(pivot));}
    diagnostics[0]=minimum/inf;*status=code;
}
__global__ void fitted_residual(const double* b,std::size_t n,const std::uint64_t* shadow,const std::uint64_t* pattern,
    int m,int q,std::size_t k,const double* coefficients,double* residual,double* diagnostics,int* status){
    double reduced_scale=0,reduced_sum=1,full_scale=0,full_sum=1;
    for(int j=0;j<q;++j)if(!isfinite(coefficients[j])){*status=static_cast<int>(SpaiStatus::numerical_failure);return;}
    for(std::size_t row=0;row<n;++row){double value=0;for(int j=0;j<q;++j)value+=b[row+pattern[j]*n]*coefficients[j];value-=row==k?1:0;
        if(!isfinite(value)){*status=static_cast<int>(SpaiStatus::numerical_failure);return;}accumulate(value,full_scale,full_sum);
    }
    for(int i=0;i<m;++i){double value=0;for(int j=0;j<q;++j)value+=b[shadow[i]+pattern[j]*n]*coefficients[j];value-=shadow[i]==k?1:0;
        residual[i]=value;accumulate(value,reduced_scale,reduced_sum);
    }
    diagnostics[1]=reduced_scale==0?0:reduced_scale*sqrt(reduced_sum);diagnostics[2]=full_scale==0?0:full_scale*sqrt(full_sum);
    *status=isfinite(diagnostics[1])&&isfinite(diagnostics[2])?static_cast<int>(SpaiStatus::success):static_cast<int>(SpaiStatus::numerical_failure);
}
__device__ int score_value(double rsq,double dot,double denominator,double& score,int& clamped){
    int code=static_cast<int>(SpaiStatus::success);clamped=0;score=0;
    if(!isfinite(rsq)||!isfinite(dot)||!isfinite(denominator))code=static_cast<int>(SpaiStatus::numerical_failure);
    else if(denominator==0)code=static_cast<int>(SpaiStatus::zero_denominator);
    else if(rsq<0||denominator<0)code=static_cast<int>(SpaiStatus::invalid_score);
    if(code==static_cast<int>(SpaiStatus::success)){
        const double quotient=dot*dot/denominator;double delta=rsq-quotient;const double bound=32*1.1102230246251565e-16*(rsq+fabs(quotient));
        if(!isfinite(quotient)||!isfinite(delta)||!isfinite(bound))code=static_cast<int>(SpaiStatus::numerical_failure);
        else if(delta<0){if(-delta>bound)code=static_cast<int>(SpaiStatus::invalid_score);else{delta=0;clamped=1;}}
        if(code==static_cast<int>(SpaiStatus::success))score=sqrt(delta);
    }return code;
}
__global__ void single_score(double rsq,double dot,double denominator,double* score,int* status,int* clamped){
    *status=score_value(rsq,dot,denominator,*score,*clamped);
}
__global__ void scores(const double* b,std::size_t n,const std::uint64_t* shadow,int m,const std::uint64_t* candidates,
    std::size_t count,const double* residual,const double* diagnostics,const int* fit_status,double* output,int* statuses,int* clamps){
    const std::size_t index=blockIdx.x*static_cast<std::size_t>(blockDim.x)+threadIdx.x;if(index>=count)return;
    if(*fit_status!=static_cast<int>(SpaiStatus::success)){statuses[index]=*fit_status;clamps[index]=0;output[index]=0;return;}
    const auto j=candidates[index];double dot=0,denominator=0;for(int i=0;i<m;++i){const double value=b[shadow[i]+j*n];dot+=residual[i]*value;denominator+=value*value;}
    statuses[index]=score_value(diagnostics[1]*diagnostics[1],dot,denominator,output[index],clamps[index]);
}
__global__ void commit_column(double* inverse,std::size_t n,std::size_t k,const std::uint64_t* pattern,int q,const double* coefficients){
    for(std::size_t row=0;row<n;++row)inverse[row+k*n]=0;for(int j=0;j<q;++j)inverse[pattern[j]+k*n]=coefficients[j];
}
__global__ void assemble_p(const double* inverse,const double* d,std::size_t n,double* p,int* flag){
    const std::size_t at=blockIdx.x*static_cast<std::size_t>(blockDim.x)+threadIdx.x;if(at>=n*n)return;const auto j=at%n,k=at/n;
    const double coefficient=inverse[at],value=coefficient*d[j];p[k+j*n]=value;if(!isfinite(value)||(coefficient!=0&&value==0))atomicExch(flag,1);
}
struct Handles{
    cusolverDnHandle_t solver=nullptr;cublasHandle_t blas=nullptr;
    ~Handles(){if(blas&&cublasDestroy(blas)!=CUBLAS_STATUS_SUCCESS)std::cerr<<"SPAI BLAS cleanup failed\n";if(solver&&cusolverDnDestroy(solver)!=CUSOLVER_STATUS_SUCCESS)std::cerr<<"SPAI solver cleanup failed\n";}
};
struct Slot{
    Stream stream;Handles handles;cusolverDnHandle_t& solver;cublasHandle_t& blas;
    Memory reduced,rhs,tau,work,pattern,shadow,candidates,residual,diagnostics,infos,score_values,score_status,clamps;
    std::size_t column=0;SpaiFit fit;int m=0,q=0,lwork=0;bool qr_ok=false;
    Slot(std::size_t n):solver(handles.solver),blas(handles.blas),reduced(bytes(bytes(n,n),sizeof(double))),rhs(bytes(n,sizeof(double))),tau(bytes(n,sizeof(double))),
        pattern(bytes(n,sizeof(std::uint64_t))),shadow(bytes(n,sizeof(std::uint64_t))),candidates(bytes(n,sizeof(std::uint64_t))),
        residual(bytes(n,sizeof(double))),diagnostics(3*sizeof(double)),infos(3*sizeof(int)),score_values(bytes(n,sizeof(double))),score_status(bytes(n,sizeof(int))),clamps(bytes(n,sizeof(int))){
        check(cusolverDnCreate(&solver),"SPAI solver create");check(cublasCreate(&blas),"SPAI BLAS create");
        check(cusolverDnSetStream(solver,stream.p),"SPAI solver stream");check(cublasSetStream(blas,stream.p),"SPAI BLAS stream");
        check(cusolverDnSetDeterministicMode(solver,CUSOLVER_DETERMINISTIC_RESULTS),"SPAI deterministic QR");
        cusolverDeterministicMode_t deterministic;check(cusolverDnGetDeterministicMode(solver,&deterministic),"SPAI query deterministic QR");
        if(deterministic!=CUSOLVER_DETERMINISTIC_RESULTS)throw std::runtime_error("Deterministic QR mode not retained");
        check(cusolverDnSetMathMode(solver,CUSOLVER_DEFAULT_MATH),"SPAI solver native math");
        cusolverMathMode_t math;check(cusolverDnGetMathMode(solver,&math),"SPAI query solver math");if(math!=CUSOLVER_DEFAULT_MATH)throw std::runtime_error("Native solver math not retained");
        check(cublasSetPointerMode(blas,CUBLAS_POINTER_MODE_HOST),"SPAI scalar mode");check(cublasSetMathMode(blas,CUBLAS_PEDANTIC_MATH),"SPAI native BLAS math");
        cublasMath_t actual;check(cublasGetMathMode(blas,&actual),"SPAI query BLAS math");if(actual!=CUBLAS_PEDANTIC_MATH)throw std::runtime_error("Native BLAS math not retained");
        check(cublasSetAtomicsMode(blas,CUBLAS_ATOMICS_NOT_ALLOWED),"SPAI atomics mode");
    }
    ~Slot(){cleanup(cudaStreamSynchronize(stream.p));}
};
double at(const CsrMatrix& b,std::size_t row,std::size_t col){for(auto j=b.row_offsets()[row];j<b.row_offsets()[row+1];++j){if(b.column_indices()[j]==col)return b.values()[j];if(b.column_indices()[j]>col)break;}return 0;}
std::vector<Entry> dense_entries(const std::vector<double>& values,std::size_t n){std::vector<Entry> result;for(std::size_t col=0;col<n;++col)for(std::size_t row=0;row<n;++row)if(values[row+col*n]!=0)result.push_back(Entry{row,col,values[row+col*n]});return result;}
bool canonical(const CsrMatrix& a){if(a.rows()==std::numeric_limits<std::size_t>::max()||a.row_offsets().size()!=a.rows()+1||a.row_offsets().front()!=0||a.row_offsets().back()!=a.values().size()||a.column_indices().size()!=a.values().size())return false;
    for(std::size_t row=0;row<a.rows();++row){auto first=a.row_offsets()[row],last=a.row_offsets()[row+1];if(first>last||last>a.values().size())return false;for(auto j=first;j<last;++j)if(a.column_indices()[j]>=a.cols()||!std::isfinite(a.values()[j])||a.values()[j]==0||(j>first&&a.column_indices()[j-1]>=a.column_indices()[j]))return false;}return true;}
} // namespace

ScoreResult cuda_evaluate_spai_score(double residual_squared,double dot,double denominator_squared,int device){
    Guard guard(device);Stream stream;Memory score(sizeof(double)),codes(2*sizeof(int));CudaSpaiTelemetry telemetry;
    single_score<<<1,1,0,stream.p>>>(residual_squared,dot,denominator_squared,score.d(),codes.i(),codes.i()+1);check(cudaGetLastError(),"SPAI score helper kernel");
    ScoreResult result;int status[2];download(&result.score,score.p,sizeof(double),stream,telemetry);download(status,codes.p,sizeof(status),stream,telemetry);
    result.status=static_cast<SpaiStatus>(status[0]);result.clamped=status[1]!=0;return result;
}

CudaSpaiResult cuda_construct_spai(const CsrMatrix& a,const SpaiOptions& options,int device){
    const auto n=a.rows();CudaSpaiResult out(n);auto& result=out.construction;auto& telemetry=out.telemetry;result.status=SpaiStatus::invalid_input;
    if(!n||a.cols()!=n||!canonical(a)||!std::isfinite(options.epsilon)||options.epsilon<=0||n>INT_MAX){result.message="Require canonical square matrix, positive finite epsilon and int QR dimensions";return out;}
    Guard guard(device);Stream main;const auto matrix_bytes=bytes(bytes(n,n),sizeof(double));
    Memory offsets(bytes(n+1,sizeof(std::uint64_t))),indices(bytes(a.values().size(),sizeof(std::uint64_t))),values(bytes(a.values().size(),sizeof(double))),
        d(bytes(n,sizeof(double))),b(matrix_bytes),inverse(matrix_bytes),p(matrix_bytes),status(bytes(n,sizeof(int)));
    std::vector<std::uint64_t> ro(a.row_offsets().begin(),a.row_offsets().end()),ci(a.column_indices().begin(),a.column_indices().end());
    upload(offsets,ro.data(),offsets.capacity,main,telemetry);upload(indices,ci.data(),indices.capacity,main,telemetry);upload(values,a.values().data(),values.capacity,main,telemetry);
    check(cudaMemsetAsync(b.p,0,matrix_bytes,main.p),"SPAI clear B");check(cudaMemsetAsync(inverse.p,0,matrix_bytes,main.p),"SPAI clear N");
    scale_matrix<<<static_cast<unsigned>((n+255)/256),256,0,main.p>>>(offsets.u(),indices.u(),values.d(),n,d.d(),b.d(),status.i());check(cudaGetLastError(),"SPAI scaling kernel");
    std::vector<int> scale_status(n);download(scale_status.data(),status.p,bytes(n,sizeof(int)),main,telemetry);sync(main);telemetry.gpu_scaling=true;
    for(std::size_t k=0;k<n;++k)if(scale_status[k]){result.message="GPU scaling/diagonal range failure at row "+std::to_string(k);return out;}
    result.scaling.resize(n);std::vector<double> host_b(bytes(n,n));download(result.scaling.data(),d.p,bytes(n,sizeof(double)),main,telemetry);download(host_b.data(),b.p,matrix_bytes,main,telemetry);sync(main);
    result.scaled_transpose=CsrMatrix::from_entries(n,n,dense_entries(host_b,n));result.columns.resize(n);
    std::vector<std::unique_ptr<Slot>> slots;for(std::size_t j=0;j<std::min<std::size_t>(n,8);++j)slots.emplace_back(new Slot(n));telemetry.streams_created=slots.size();
    std::vector<std::vector<std::size_t>> patterns(n);for(std::size_t k=0;k<n;++k)patterns[k]={k};
    std::vector<bool> done(n,false);std::size_t remaining=n;
    while(remaining){
        std::vector<std::size_t> active;for(std::size_t k=0;k<n;++k)if(!done[k])active.push_back(k);
        for(std::size_t begin=0;begin<active.size();begin+=slots.size()){
            const auto count=std::min(slots.size(),active.size()-begin);telemetry.max_fit_batch=std::max(telemetry.max_fit_batch,count);if(count>1)++telemetry.parallel_fit_batches;
            std::size_t submitted=0;
            // Allocate/query every slot before launching any QR in this batch.
            for(std::size_t s=0;s<count;++s){auto& slot=*slots[s];slot.column=active[begin+s];auto& column=result.columns[slot.column];slot.fit=SpaiFit{};slot.fit.pattern=patterns[slot.column];slot.qr_ok=false;
                column.attempted_pattern=slot.fit.pattern;++column.fit_attempts;if(column.fits)++column.growth_attempts;
                for(std::size_t row=0;row<n;++row)for(auto j:slot.fit.pattern)if(at(result.scaled_transpose,row,j)!=0){slot.fit.shadow.push_back(row);break;}
                slot.m=dim(slot.fit.shadow.size());slot.q=dim(slot.fit.pattern.size());
                if(slot.m<slot.q){column.status=SpaiStatus::rank_deficient;column.message="Reduced shadow has fewer rows than columns";done[slot.column]=true;--remaining;continue;}
                std::vector<bool> candidates(n,false);for(auto row:slot.fit.shadow)for(auto j=result.scaled_transpose.row_offsets()[row];j<result.scaled_transpose.row_offsets()[row+1];++j)candidates[result.scaled_transpose.column_indices()[j]]=true;
                for(auto j=result.scaled_transpose.row_offsets()[slot.column];j<result.scaled_transpose.row_offsets()[slot.column+1];++j)candidates[result.scaled_transpose.column_indices()[j]]=true;
                for(auto j:slot.fit.pattern)candidates[j]=false;for(std::size_t j=0;j<n;++j)if(candidates[j])slot.fit.candidates.push_back(j);
                int qr_work=0,q_work=0;check(cusolverDnDgeqrf_bufferSize(slot.solver,slot.m,slot.q,slot.reduced.d(),slot.m,&qr_work),"SPAI QR workspace");
                check(cusolverDnDormqr_bufferSize(slot.solver,CUBLAS_SIDE_LEFT,CUBLAS_OP_T,slot.m,1,slot.q,slot.reduced.d(),slot.m,slot.tau.d(),slot.rhs.d(),slot.m,&q_work),"SPAI Q workspace");
                if(qr_work<0||q_work<0)throw std::runtime_error("Negative solver workspace");slot.lwork=std::max(qr_work,q_work);slot.work.allocate(bytes(static_cast<std::size_t>(slot.lwork),sizeof(double)));slot.qr_ok=true;
                // size_t is statically checked below to match uint64 metadata.
                upload(slot.pattern,slot.fit.pattern.data(),bytes(slot.fit.pattern.size(),sizeof(std::uint64_t)),slot.stream,telemetry);
                upload(slot.shadow,slot.fit.shadow.data(),bytes(slot.fit.shadow.size(),sizeof(std::uint64_t)),slot.stream,telemetry);
                upload(slot.candidates,slot.fit.candidates.data(),bytes(slot.fit.candidates.size(),sizeof(std::uint64_t)),slot.stream,telemetry);
            }
            for(std::size_t s=0;s<count;++s){auto& slot=*slots[s];if(!slot.qr_ok)continue;
                gather<<<32,256,0,slot.stream.p>>>(b.d(),n,slot.shadow.u(),slot.pattern.u(),slot.m,slot.q,slot.column,slot.reduced.d(),slot.rhs.d());check(cudaGetLastError(),"SPAI reduced gather");
                check(cusolverDnDgeqrf(slot.solver,slot.m,slot.q,slot.reduced.d(),slot.m,slot.tau.d(),slot.work.d(),slot.lwork,slot.infos.i()),"SPAI GPU QR");
                check(cusolverDnDormqr(slot.solver,CUBLAS_SIDE_LEFT,CUBLAS_OP_T,slot.m,1,slot.q,slot.reduced.d(),slot.m,slot.tau.d(),slot.rhs.d(),slot.m,slot.work.d(),slot.lwork,slot.infos.i()+1),"SPAI GPU Q transpose");
                qr_guard<<<1,1,0,slot.stream.p>>>(b.d(),n,slot.shadow.u(),slot.pattern.u(),slot.m,slot.q,slot.reduced.d(),slot.tau.d(),slot.rhs.d(),slot.infos.i()+2,slot.diagnostics.d());check(cudaGetLastError(),"SPAI QR guards");++submitted;++telemetry.qr_submissions;
            }
            telemetry.max_qr_submissions_before_collection=std::max(telemetry.max_qr_submissions_before_collection,submitted);
            for(std::size_t s=0;s<count;++s){auto& slot=*slots[s];if(!slot.qr_ok)continue;int info[3];download(info,slot.infos.p,sizeof(info),slot.stream,telemetry);sync(slot.stream);
                if(info[0]||info[1])throw std::runtime_error("cuSOLVER devInfo rejected QR/Q application");
                if(info[2]!=static_cast<int>(SpaiStatus::success)){auto& column=result.columns[slot.column];column.status=static_cast<SpaiStatus>(info[2]);column.message="GPU QR exact-pivot/nonfinite guard failed";slot.qr_ok=false;done[slot.column]=true;--remaining;}
            }
            for(std::size_t s=0;s<count;++s){auto& slot=*slots[s];if(!slot.qr_ok)continue;
                check(cublasDtrsv(slot.blas,CUBLAS_FILL_MODE_UPPER,CUBLAS_OP_N,CUBLAS_DIAG_NON_UNIT,slot.q,slot.reduced.d(),slot.m,slot.rhs.d(),1),"SPAI GPU triangular solve");
                fitted_residual<<<1,1,0,slot.stream.p>>>(b.d(),n,slot.shadow.u(),slot.pattern.u(),slot.m,slot.q,slot.column,slot.rhs.d(),slot.residual.d(),slot.diagnostics.d(),slot.infos.i()+2);check(cudaGetLastError(),"SPAI fitted residual");
                if(!slot.fit.candidates.empty()){scores<<<static_cast<unsigned>((slot.fit.candidates.size()+255)/256),256,0,slot.stream.p>>>(b.d(),n,slot.shadow.u(),slot.m,slot.candidates.u(),slot.fit.candidates.size(),slot.residual.d(),slot.diagnostics.d(),slot.infos.i()+2,slot.score_values.d(),slot.score_status.i(),slot.clamps.i());check(cudaGetLastError(),"SPAI restricted scores");telemetry.gpu_score_evaluations+=slot.fit.candidates.size();}
            }
            for(std::size_t s=0;s<count;++s){auto& slot=*slots[s];if(!slot.qr_ok)continue;auto& column=result.columns[slot.column];auto& fit=slot.fit;int fit_status=0;double diagnostics[3];
                fit.coefficients.resize(slot.q);download(fit.coefficients.data(),slot.rhs.p,bytes(slot.q,sizeof(double)),slot.stream,telemetry);download(diagnostics,slot.diagnostics.p,sizeof(diagnostics),slot.stream,telemetry);download(&fit_status,slot.infos.i()+2,sizeof(int),slot.stream,telemetry);sync(slot.stream);
                if(fit_status!=static_cast<int>(SpaiStatus::success)){column.status=static_cast<SpaiStatus>(fit_status);column.message="GPU coefficients/residual nonfinite";done[slot.column]=true;--remaining;continue;}
                fit.qr_pivot_ratio=diagnostics[0];fit.residual=diagnostics[1];fit.full_residual=diagnostics[2];
                if(column.fits)++column.growths;++column.fits;column.pattern=fit.pattern;column.coefficients=fit.coefficients;column.reduced_residual=fit.residual;column.full_residual=fit.full_residual;column.qr_pivot_ratio=fit.qr_pivot_ratio;
                const auto candidates=fit.candidates;fit.candidates.clear();column.history.push_back(fit);auto& record=column.history.back();
                commit_column<<<1,1,0,slot.stream.p>>>(inverse.d(),n,slot.column,slot.pattern.u(),slot.q,slot.rhs.d());check(cudaGetLastError(),"SPAI commit fitted N column");
                if(fit.residual<=options.epsilon){column.status=SpaiStatus::success;done[slot.column]=true;--remaining;sync(slot.stream);continue;}
                if(column.growths==options.alpha){column.status=SpaiStatus::iteration_limit;column.message="Requested accuracy not reached within growth/refit budget";done[slot.column]=true;--remaining;sync(slot.stream);continue;}
                record.candidates=candidates;if(candidates.empty()){column.status=SpaiStatus::empty_candidates;column.message="No candidate remains before convergence";done[slot.column]=true;--remaining;sync(slot.stream);continue;}
                std::vector<double> candidate_scores(candidates.size());std::vector<int> statuses(candidates.size()),clamped(candidates.size());
                download(candidate_scores.data(),slot.score_values.p,bytes(candidates.size(),sizeof(double)),slot.stream,telemetry);download(statuses.data(),slot.score_status.p,bytes(candidates.size(),sizeof(int)),slot.stream,telemetry);download(clamped.data(),slot.clamps.p,bytes(candidates.size(),sizeof(int)),slot.stream,telemetry);sync(slot.stream);
                bool failed=false;for(std::size_t j=0;j<candidates.size();++j){if(statuses[j]!=static_cast<int>(SpaiStatus::success)){column.status=static_cast<SpaiStatus>(statuses[j]);column.message="GPU candidate scoring failed at index "+std::to_string(candidates[j]);failed=true;break;}if(clamped[j])++column.score_clamps;record.scores.push_back(candidate_scores[j]);}
                if(failed){done[slot.column]=true;--remaining;continue;}
                record.mean_score=std::accumulate(record.scores.begin(),record.scores.end(),0.0)/static_cast<double>(record.scores.size());
                if(!std::isfinite(record.mean_score)){column.status=SpaiStatus::numerical_failure;column.message="Candidate mean score is nonfinite";done[slot.column]=true;--remaining;continue;}
                std::vector<std::size_t> order(candidates.size());std::iota(order.begin(),order.end(),0);std::sort(order.begin(),order.end(),[&](std::size_t x,std::size_t y){return record.scores[x]<record.scores[y]||(record.scores[x]==record.scores[y]&&candidates[x]<candidates[y]);});
                for(auto j:order){if(record.selected.size()==options.beta)break;if(record.scores[j]<=record.mean_score)record.selected.push_back(candidates[j]);}
                if(record.selected.empty()){column.status=SpaiStatus::no_growth;column.message="No acceptable growth permitted before convergence";done[slot.column]=true;--remaining;continue;}
                auto& pattern=patterns[slot.column];pattern.insert(pattern.end(),record.selected.begin(),record.selected.end());std::sort(pattern.begin(),pattern.end());
            }
        }
    }
    for(auto& slot:slots)sync(slot->stream);
    result.status=SpaiStatus::success;for(std::size_t k=0;k<n;++k)if(result.columns[k].status!=SpaiStatus::success){result.status=result.columns[k].status;result.message="Column "+std::to_string(k)+": "+result.columns[k].message;break;}
    check(cudaMemsetAsync(status.p,0,sizeof(int),main.p),"SPAI assembly guard clear");
    assemble_p<<<static_cast<unsigned>((bytes(n,n)+255)/256),256,0,main.p>>>(inverse.d(),d.d(),n,p.d(),status.i());check(cudaGetLastError(),"SPAI GPU P assembly");telemetry.gpu_preconditioner_assembly=true;
    int assembly_status=0;std::vector<double> host_n(bytes(n,n)),host_p(bytes(n,n));download(&assembly_status,status.p,sizeof(int),main,telemetry);download(host_n.data(),inverse.p,matrix_bytes,main,telemetry);download(host_p.data(),p.p,matrix_bytes,main,telemetry);sync(main);
    if(assembly_status){result.status=SpaiStatus::numerical_failure;result.message="GPU preconditioner assembly outside binary64 range";return out;}
    result.inverse=CsrMatrix::from_entries(n,n,dense_entries(host_n,n));result.preconditioner=CsrMatrix::from_entries(n,n,dense_entries(host_p,n));return out;
}
static_assert(sizeof(std::size_t)==sizeof(std::uint64_t),"GPU SPAI metadata requires 64-bit size_t");
}
