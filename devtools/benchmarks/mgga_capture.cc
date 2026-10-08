// Passive cuEST C-API capture for a single fixed-density XC plan.
// No Psi4 ABI/private-layout access; never change inputs or API outputs.
#include <cuest.h>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <vector>

namespace {
std::unordered_map<cuestXCIntPlan_t, uint64_t> points;

void fail(const char* message) {
    std::fprintf(stderr, "[MGGA-CAPTURE] %s\n", message);
    std::abort();
}
void check(cuestStatus_t status) {
    if (status != CUEST_STATUS_SUCCESS) fail("cuEST capture API failed");
}
void cuda_check(cudaError_t status) {
    if (status != cudaSuccess) fail(cudaGetErrorString(status));
}
template <class T> T real_api(const char* name) {
    auto pointer = dlsym(RTLD_NEXT, name);
    if (!pointer) fail(name);
    return reinterpret_cast<T>(pointer);
}
const char* directory() { return std::getenv("MGGA_CAPTURE_DIR"); }

void save(const char* name, const double* device, size_t count) {
    std::vector<double> host(count);
    cuda_check(cudaMemcpy(host.data(), device, count * sizeof(double), cudaMemcpyDeviceToHost));
    auto path = std::string(directory()) + "/" + name;
    auto file = std::fopen(path.c_str(), "wbx"); // Reject repeated/overwritten captures.
    if (!file) fail("Cannot create exclusive capture file");
    if (std::fwrite(host.data(), sizeof(double), count, file) != count)
        fail("Short capture write");
    if (std::fclose(file)) fail("Capture close failed");
}

void save_grid(cuestHandle_t handle, cuestXCIntPlan_t plan, uint64_t count) {
    cuestXCIntegrationGridComputeParameters_t parameters{};
    check(cuestParametersCreate(CUEST_XCINTEGRATIONGRIDCOMPUTE_PARAMETERS,
                                reinterpret_cast<void**>(&parameters)));
    double* coordinates{};
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&coordinates), count * 3 * sizeof(double)));
    cuestWorkspaceDescriptor_t descriptor{};
    check(cuestXCIntegrationGridComputeWorkspaceQuery(
        handle, plan, parameters, &descriptor, coordinates));
    cuestWorkspace_t workspace{};
    workspace.hostBufferSizeInBytes = descriptor.hostBufferSizeInBytes;
    workspace.deviceBufferSizeInBytes = descriptor.deviceBufferSizeInBytes;
    void* host{};
    void* device{};
    if (descriptor.hostBufferSizeInBytes) {
        host = std::malloc(descriptor.hostBufferSizeInBytes);
        if (!host) fail("Host allocation failed");
    }
    if (descriptor.deviceBufferSizeInBytes)
        cuda_check(cudaMalloc(&device, descriptor.deviceBufferSizeInBytes));
    workspace.hostBuffer = reinterpret_cast<uintptr_t>(host);
    workspace.deviceBuffer = reinterpret_cast<uintptr_t>(device);
    check(cuestXCIntegrationGridCompute(handle, plan, parameters, &workspace, coordinates));
    save("coordinates.f64", coordinates, count * 3);
    cuda_check(cudaFree(coordinates));
    if (device) cuda_check(cudaFree(device));
    std::free(host);
    check(cuestParametersDestroy(CUEST_XCINTEGRATIONGRIDCOMPUTE_PARAMETERS, parameters));
    auto path = std::string(directory()) + "/grid.json";
    auto file = std::fopen(path.c_str(), "wx");
    if (!file) fail("Cannot create grid metadata");
    std::fprintf(file, "{\"npoints\":%llu,\"coordinates_shape\":[%llu,3],"
                       "\"density_shape\":[%llu,5],\"dtype\":\"native-float64\"}\n",
                 static_cast<unsigned long long>(count),
                 static_cast<unsigned long long>(count),
                 static_cast<unsigned long long>(count));
    if (std::fclose(file)) fail("Grid metadata close failed");
}
} // namespace

extern "C" cuestStatus_t cuestXCIntPlanCreate(
    cuestHandle_t handle, const cuestAOBasis_t basis, const cuestMolecularGrid_t grid,
    cuestXCIntPlanParametersFunctional_t functional, const cuestXCIntPlanParameters_t parameters,
    cuestWorkspace_t* persistent, cuestWorkspace_t* temporary, cuestXCIntPlan_t* plan) {
    static auto real = real_api<decltype(&cuestXCIntPlanCreate)>("cuestXCIntPlanCreate");
    auto status = real(handle, basis, grid, functional, parameters, persistent, temporary, plan);
    if (directory() && status == CUEST_STATUS_SUCCESS) {
        uint64_t count{};
        check(cuestQuery(handle, CUEST_MOLECULARGRID, grid,
                         CUEST_MOLECULARGRID_NUM_POINT, &count, sizeof(count)));
        points[*plan] = count;
        save_grid(handle, *plan, count);
    }
    return status;
}

extern "C" cuestStatus_t cuestXCIntegrationWeightCompute(
    cuestHandle_t handle, const cuestXCIntPlan_t plan,
    cuestXCIntegrationWeightComputeParametersWeightType_t type,
    const cuestXCIntegrationWeightComputeParameters_t parameters,
    cuestWorkspace_t* workspace, double* weights) {
    static auto real = real_api<decltype(&cuestXCIntegrationWeightCompute)>(
        "cuestXCIntegrationWeightCompute");
    auto status = real(handle, plan, type, parameters, workspace, weights);
    if (directory() && status == CUEST_STATUS_SUCCESS) {
        if (type != CUEST_XCINTEGRATIONWEIGHT_PARAMETERS_WEIGHTTYPE_TOTAL || !points.count(plan))
            fail("Unexpected weight capture");
        save("weights.f64", weights, points.at(plan));
    }
    return status;
}

extern "C" cuestStatus_t cuestXCDensityCompute(
    cuestHandle_t handle, const cuestXCIntPlan_t plan,
    cuestXCAdvancedComputeParametersApproximation_t approximation,
    const cuestXCDensityComputeParameters_t parameters,
    const cuestWorkspaceDescriptor_t* variable, cuestWorkspace_t* workspace,
    uint64_t occupied, const double* coefficients, double* density) {
    static auto real = real_api<decltype(&cuestXCDensityCompute)>("cuestXCDensityCompute");
    auto status = real(handle, plan, approximation, parameters, variable, workspace,
                       occupied, coefficients, density);
    if (directory() && status == CUEST_STATUS_SUCCESS) {
        if (approximation != CUEST_XCADVANCED_PARAMETERS_APPROXIMATION_METAGGA || !points.count(plan))
            fail("Unexpected density capture");
        save("density.f64", density, points.at(plan) * 5);
    }
    return status;
}
