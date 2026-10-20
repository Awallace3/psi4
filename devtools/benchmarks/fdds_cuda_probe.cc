// Job-local ABI probe: compile against the same pinned toolkit as Psi4.
#include <cuda_runtime_api.h>
#include <cstdio>

int main() {
    int runtime = 0;
    auto status = cudaRuntimeGetVersion(&runtime);
    if (status != cudaSuccess) {
        std::fprintf(stderr, "cudaRuntimeGetVersion: %s\n", cudaGetErrorString(status));
        return 1;
    }
    std::printf("CUDART_VERSION=%d runtime=%d\n", CUDART_VERSION, runtime);
    if (runtime != CUDART_VERSION) {
        std::fprintf(stderr, "Pinned CUDA headers/runtime do not match\n");
        return 2;
    }
    cudaDeviceProp properties{};
    status = cudaGetDeviceProperties(&properties, 0);
    if (status != cudaSuccess) {
        std::fprintf(stderr, "cudaGetDeviceProperties: %s\n", cudaGetErrorString(status));
        return 3;
    }
    std::printf("device=%s memory=%zu compute=%d.%d\n", properties.name,
                properties.totalGlobalMem, properties.major, properties.minor);
    return 0;
}
