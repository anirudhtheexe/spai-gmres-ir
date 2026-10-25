#include "bspai/fp32_cuda_spmv.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
int main() {
    try {
        const bspai::Fp32CsrMatrix a = bspai::Fp32CsrMatrix::from_entries(
            3, 3, {{0, 0, 2.0f}, {0, 2, -1.0f}, {1, 1, 3.0f}, {2, 0, 4.0f}});
        const std::vector<float> x{.25f, -.5f, .75f};
        const std::vector<float> expected = a.multiply(x);
        bspai::Fp32CudaCsrSpmv gpu(a);
        const std::vector<float> got = gpu.multiply(x);
        for (std::size_t i = 0; i < got.size(); ++i)
            if (std::fabs(got[i] - expected[i]) > 1e-6f)
                throw std::runtime_error("FP32 CUDA SpMV mismatch");
        const auto sparse = bspai::Fp32CsrMatrix::from_entries(2, 2, {{0, 0, 1.0f}});
        bspai::Fp32CudaCsrSpmv sparse_gpu(sparse);
        for (float bad :
             {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity()}) {
            bool rejected = false;
            try {
                sparse_gpu.multiply({1.0f, bad});
            } catch (const std::invalid_argument &) {
                rejected = true;
            }
            if (!rejected)
                throw std::runtime_error("FP32 CUDA SpMV accepted nonfinite input");
        }
        const auto huge =
            bspai::Fp32CsrMatrix::from_entries(1, 1, {{0, 0, std::numeric_limits<float>::max()}});
        bspai::Fp32CudaCsrSpmv huge_gpu(huge);
        bool rejected_output = false;
        try {
            huge_gpu.multiply({2.0f});
        } catch (const std::overflow_error &) {
            rejected_output = true;
        }
        if (!rejected_output)
            throw std::runtime_error("FP32 CUDA SpMV accepted infinite output");
        std::cout << "FP32 CUDA SpMV passed\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
