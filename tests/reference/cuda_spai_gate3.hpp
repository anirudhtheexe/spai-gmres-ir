#ifndef BSPAI_CUDA_SPAI_GATE3_REFERENCE_HPP
#define BSPAI_CUDA_SPAI_GATE3_REFERENCE_HPP
#include "bspai/cuda_spai.hpp"
namespace bspai {
// Validation-only frozen Gate3 numerical source; never a production fallback.
CudaSpaiResult cuda_construct_spai_dense_reference(const CsrMatrix&,const SpaiOptions&,int device=0);
}
#endif
