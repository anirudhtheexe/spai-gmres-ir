#include "bspai/fp32_cuda_gmres.hpp"
#include <cuda_runtime_api.h>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static std::uint64_t signature(const bspai::Fp32GmresResult &r) {
    std::uint64_t hash = 1469598103934665603ULL;
    auto mix = [&](std::uint32_t bits) { hash = (hash ^ bits) * 1099511628211ULL; };
    auto number = [&](float value) {
        std::uint32_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        mix(bits);
    };
    mix(static_cast<std::uint32_t>(r.status));
    mix(static_cast<std::uint32_t>(r.iterations));
    for (float value : r.solution) number(value);
    for (float value : r.hessenberg_residual_history) number(value);
    for (float value : r.original_residual_history) number(value);
    for (float value : r.preconditioned_relative_history) number(value);
    number(r.final_original_residual);
    number(r.final_preconditioned_relative);
    return hash;
}

int main(int argc, char **argv) {
    try {
        if (argc != 3) throw std::invalid_argument("usage: benchmark VARIANT exact|one|chunk");
        const std::string variant = argv[1], mode = argv[2];
        const std::size_t n = mode == "chunk" ? 40 : 12000;
        if (mode != "exact" && mode != "one" && mode != "chunk")
            throw std::invalid_argument("invalid mode");
        std::vector<bspai::Fp32Entry> entries, identity;
        entries.reserve(mode == "chunk" ? 2*n-1 : n);
        identity.reserve(n);
        for (std::size_t i=0; i<n; ++i) {
            entries.push_back({i,i,1.0f});
            if (mode == "chunk" && i+1<n) entries.push_back({i,i+1,1.0f});
            identity.push_back({i,i,1.0f});
        }
        const auto a = bspai::Fp32CsrMatrix::from_entries(n,n,entries);
        const auto p = bspai::Fp32CsrMatrix::from_entries(n,n,identity);
        std::vector<float> rhs(n,0), initial;
        rhs[mode == "chunk" ? n-1 : 0] = 1.0f;
        if (mode == "exact") initial = rhs;
        bspai::Fp32GmresOptions options;
        options.tolerance = 1.0e-4f;
        options.max_iterations = mode == "chunk" ? 18 : 0;
        std::uint64_t expected_hash = 0;
        for (int t=0; t<15; ++t) {
            cudaDeviceSynchronize();
            const auto start = std::chrono::steady_clock::now();
            const auto result = bspai::fp32_cuda_left_preconditioned_gmres(
                a,p,rhs,options,initial);
            if (cudaDeviceSynchronize() != cudaSuccess)
                throw std::runtime_error("CUDA synchronization failed");
            const auto end = std::chrono::steady_clock::now();
            const std::uint64_t hash = signature(result);
            if (t == 0) expected_hash = hash;
            if (hash != expected_hash) throw std::runtime_error("nondeterministic result");
            const double ms = std::chrono::duration<double,std::milli>(end-start).count();
            std::cout << std::fixed << std::setprecision(6)
                      << "{\"variant\":\"" << variant << "\",\"mode\":\"" << mode
                      << "\",\"phase\":\"" << (t<3?"warmup":"measure")
                      << "\",\"trial\":" << t << ",\"ms\":" << ms
                      << ",\"status\":" << static_cast<int>(result.status)
                      << ",\"iterations\":" << result.iterations
                      << ",\"hash\":" << hash << "}\n";
        }
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
