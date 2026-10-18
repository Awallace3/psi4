// Host-only CUDA runtime identity. Separate process, no kernels/Psi4/cuEST calls.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>

#if CUDART_VERSION != 13030
#error "Identity probe requires the campaign's pinned CUDA 13.3 headers"
#endif

static void check(cudaError_t rc) {
    if (rc != cudaSuccess) {
        std::cerr << cudaGetErrorString(rc) << '\n';
        std::exit(1);
    }
}

int main() {
    int count = 0, runtime = 0;
    check(cudaGetDeviceCount(&count));
    if (count != 1) {
        std::cerr << "Expected exactly one CUDA-visible device, got " << count << '\n';
        return 1;
    }
    cudaDeviceProp prop{};
    check(cudaGetDeviceProperties(&prop, 0));
    check(cudaRuntimeGetVersion(&runtime));
    const auto* b = reinterpret_cast<const unsigned char*>(prop.uuid.bytes);
    char uuid[41];
    std::snprintf(uuid, sizeof(uuid),
        "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
        b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15]);
    std::cout << "{\"count\":1,\"logical_device\":0,\"name\":" << std::quoted(prop.name)
              << ",\"uuid\":" << std::quoted(uuid)
              << ",\"memory_bytes\":" << prop.totalGlobalMem
              << ",\"header_version\":" << CUDART_VERSION
              << ",\"runtime_version\":" << runtime << "}\n";
}
