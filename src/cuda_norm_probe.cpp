// Primitive investigation only; not a solver fallback or a performance test.
#define NOMINMAX
#include <windows.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
void check(cudaError_t s) {
    if (s != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(s));
}
void check(cublasStatus_t s) {
    if (s != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("cuBLAS status " + std::to_string(s));
}
int main() {
    try {
        std::cout << std::setprecision(17);
        cudaStream_t stream;
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        cublasHandle_t handle;
        check(cublasCreate(&handle));
        check(cublasSetStream(handle, stream));
        check(cublasSetAtomicsMode(handle, CUBLAS_ATOMICS_NOT_ALLOWED));
        double *device = nullptr, *scalar = nullptr;
        check(cudaMalloc(reinterpret_cast<void **>(&device), 2 * sizeof(double)));
        check(cudaMalloc(reinterpret_cast<void **>(&scalar), sizeof(double)));
        int version = 0;
        check(cublasGetVersion(handle, &version));
        char path[MAX_PATH] = {};
        GetModuleFileNameA(GetModuleHandleA("cublas64_13.dll"), path, MAX_PATH);
        char *emulation = nullptr;
        std::size_t environment_size = 0;
        if (_dupenv_s(&emulation, &environment_size, "CUBLAS_EMULATE_DOUBLE_PRECISION") != 0)
            throw std::runtime_error("Environment query failed");
        std::cout << "cuBLAS=" << version << ", loaded=" << path
                  << ", CUBLAS_EMULATE_DOUBLE_PRECISION=" << (emulation ? emulation : "unset")
                  << '\n';
        std::free(emulation);
        for (auto math : {CUBLAS_DEFAULT_MATH, CUBLAS_PEDANTIC_MATH}) {
            check(cublasSetMathMode(handle, math));
            cublasMath_t actual;
            check(cublasGetMathMode(handle, &actual));
            for (int exponent : {0, -900, -1022, -1030, -1074}) {
                double input[2] = {std::ldexp(1.0, exponent), 0}, back[2] = {};
                check(
                    cudaMemcpyAsync(device, input, sizeof(input), cudaMemcpyHostToDevice, stream));
                check(cudaStreamSynchronize(stream));
                check(cudaMemcpyAsync(back, device, sizeof(back), cudaMemcpyDeviceToHost, stream));
                check(cudaStreamSynchronize(stream));
                for (int n : {1, 2})
                    for (auto mode : {CUBLAS_POINTER_MODE_HOST, CUBLAS_POINTER_MODE_DEVICE}) {
                        check(cublasSetPointerMode(handle, mode));
                        double classic = 0, wide = 0;
                        check(cublasDnrm2(handle, n, device, 1,
                                          mode == CUBLAS_POINTER_MODE_HOST ? &classic : scalar));
                        check(cudaStreamSynchronize(stream));
                        if (mode == CUBLAS_POINTER_MODE_DEVICE)
                            check(cudaMemcpy(&classic, scalar, sizeof(double),
                                             cudaMemcpyDeviceToHost));
                        check(cublasDnrm2_64(handle, n, device, 1,
                                             mode == CUBLAS_POINTER_MODE_HOST ? &wide : scalar));
                        check(cudaStreamSynchronize(stream));
                        if (mode == CUBLAS_POINTER_MODE_DEVICE)
                            check(
                                cudaMemcpy(&wide, scalar, sizeof(double), cudaMemcpyDeviceToHost));
                        std::cout << "math=" << actual << ", exponent=" << exponent << ", n=" << n
                                  << ", pointermode=" << mode << ", uploaded=" << input[0]
                                  << ", readback=" << back[0] << ", classic=" << classic
                                  << ", int64=" << wide << '\n';
                    }
            }
        }
        check(cudaFree(scalar));
        check(cudaFree(device));
        check(cublasDestroy(handle));
        check(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
