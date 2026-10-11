// Validation-only source instrumentation: vendor internal allocations excluded.
#include "cuda_spai_allocation_audit.hpp"
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <map>
#include <algorithm>
namespace validation {
AllocationAudit state;
std::map<void*,std::size_t> live;
cudaError_t audit_malloc(void** p,std::size_t n){const auto s=cudaMalloc(p,n);if(s==cudaSuccess){live[*p]=n;++state.calls;state.total+=n;state.current+=n;state.peak=std::max(state.peak,state.current);state.maximum=std::max(state.maximum,n);state.outstanding=live.size();}return s;}
cudaError_t audit_free(void* p){const auto s=cudaFree(p);if(s==cudaSuccess){const auto it=live.find(p);if(it!=live.end()){state.current-=it->second;live.erase(it);}state.outstanding=live.size();}return s;}
AllocationAudit allocation_audit(){return state;}
}
namespace bspai {
CudaSpaiResult cuda_construct_spai_allocation_reference(const CsrMatrix&,const SpaiOptions&,int);
ScoreResult cuda_evaluate_spai_score_allocation_reference(double,double,double,int);
}
#define cudaMalloc validation::audit_malloc
#define cudaFree validation::audit_free
#define cuda_construct_spai cuda_construct_spai_allocation_reference
#define cuda_evaluate_spai_score cuda_evaluate_spai_score_allocation_reference
#include "../src/cuda_spai.cu"
#undef cudaMalloc
#undef cudaFree
#undef cuda_construct_spai
#undef cuda_evaluate_spai_score
namespace validation {
bspai::CudaSpaiResult audited_construct(const bspai::CsrMatrix& a,const bspai::SpaiOptions& o){if(!live.empty())throw std::runtime_error("Previous source allocations still live");state=AllocationAudit{};return bspai::cuda_construct_spai_allocation_reference(a,o,0);}
}
