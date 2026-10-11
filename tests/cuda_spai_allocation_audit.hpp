#pragma once
#include "bspai/cuda_spai.hpp"
namespace validation {
struct AllocationAudit {std::size_t calls=0,total=0,current=0,peak=0,maximum=0,outstanding=0;};
bspai::CudaSpaiResult audited_construct(const bspai::CsrMatrix&,const bspai::SpaiOptions&);
AllocationAudit allocation_audit();
}
