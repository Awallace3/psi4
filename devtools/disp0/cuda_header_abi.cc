// ABI-only build probe. No GPU operations, dispersion or elementwise algorithm.
#include <cuda_runtime.h>
#include <iostream>

int main() {
    int runtime = 0;
    const auto status = cudaRuntimeGetVersion(&runtime);
    // Keep an actual reference to the header-selected symbol, without calling
    // a device API on this CPU-only preparation allocation.
    auto volatile properties_entry = &cudaGetDeviceProperties;
    (void)properties_entry;
    std::cout << "CUDART_VERSION=" << CUDART_VERSION
              << "\nruntime_version=" << runtime << "\nstatus=" << status << "\n";
    return status != cudaSuccess || CUDART_VERSION != 13030 || runtime != 13030;
}
