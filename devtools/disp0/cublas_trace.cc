// Diagnostic interposer only. All matrix mathematics is forwarded unchanged
// to NVIDIA cuBLAS; this file implements no dispersion/elementwise algorithm.
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <string>

namespace {
using Dgemm = decltype(&cublasDgemm_v2);
Dgemm actual_dgemm() {
    static const auto actual = reinterpret_cast<Dgemm>(dlsym(RTLD_NEXT, "cublasDgemm_v2"));
    return actual;
}

std::string escaped(const char* value) {
    std::string out;
    for (const unsigned char c : std::string(value ? value : "")) {
        if (c == '\\' || c == '"') out += '\\';
        if (c >= 32) out += static_cast<char>(c);
    }
    return out;
}

void record(int m, int n, int k, int status) {
    const char* path = std::getenv("DISP0_CUBLAS_TRACE");
    if (!path || !*path) return;
    const std::string phase = escaped(std::getenv("DISP0_TRACE_PHASE"));
    const std::string line = "{\"api\":\"cublasDgemm_v2\",\"pid\":" + std::to_string(getpid()) +
                             ",\"m\":" + std::to_string(m) + ",\"n\":" + std::to_string(n) +
                             ",\"k\":" + std::to_string(k) + ",\"status\":" + std::to_string(status) +
                             ",\"phase\":\"" + phase + "\"}\n";
    const int fd = open(path, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0600);
    if (fd >= 0) {
        // One append per record; a missing/truncated record causes the external
        // evidence check to fail, never changes cuBLAS numerical execution.
        (void)write(fd, line.data(), line.size());
        close(fd);
    }
}
}  // namespace

extern "C" cublasStatus_t cublasDgemm_v2(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const double* alpha, const double* a, int lda,
    const double* b, int ldb, const double* beta, double* c, int ldc) {
    const auto actual = actual_dgemm();
    if (!actual) {
        record(m, n, k, CUBLAS_STATUS_NOT_INITIALIZED);
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    const auto status = actual(handle, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    record(m, n, k, static_cast<int>(status));
    return status;
}

extern "C" int disp0_cublas_forwarding_identity(char* buffer, size_t size) {
    const auto actual = actual_dgemm();
    Dl_info info{};
    if (!actual || actual == &cublasDgemm_v2 ||
        !dladdr(reinterpret_cast<void*>(actual), &info) || !info.dli_fname) {
        std::snprintf(buffer, size, "{\"status\":1}");
        return 1;
    }
    std::snprintf(buffer, size, "{\"status\":0,\"library\":\"%s\"}",
                  escaped(info.dli_fname).c_str());
    return 0;
}

// ABI-safe identity probe compiled against the same explicit CUDA13 headers.
// It never changes the selected device and reports errors to the caller.
extern "C" int disp0_cuda_identity(char* buffer, size_t size) {
    int count = 0, device = -1, runtime = 0, driver = 0;
    cudaDeviceProp properties{};
    cudaUUID_t uuid{};
    cudaError_t status = cudaGetDeviceCount(&count);
    if (status == cudaSuccess) status = cudaGetDevice(&device);
    if (status == cudaSuccess) status = cudaGetDeviceProperties(&properties, device);
    if (status == cudaSuccess) status = cudaDeviceGetUuid(&uuid, device);
    if (status == cudaSuccess) status = cudaRuntimeGetVersion(&runtime);
    if (status == cudaSuccess) status = cudaDriverGetVersion(&driver);
    if (status != cudaSuccess) {
        std::snprintf(buffer, size, "{\"status\":%d,\"count\":%d}", static_cast<int>(status), count);
        return static_cast<int>(status);
    }
    char text[80];
    const auto* u = reinterpret_cast<const unsigned char*>(uuid.bytes);
    std::snprintf(text, sizeof(text),
                  "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
                  u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7],
                  u[8], u[9], u[10], u[11], u[12], u[13], u[14], u[15]);
    std::snprintf(buffer, size,
                  "{\"status\":0,\"count\":%d,\"logical_device\":%d,\"name\":\"%s\","
                  "\"uuid\":\"%s\",\"header_version\":%d,\"runtime_version\":%d,\"driver_version\":%d}",
                  count, device, escaped(properties.name).c_str(), text, CUDART_VERSION, runtime, driver);
    return 0;
}
