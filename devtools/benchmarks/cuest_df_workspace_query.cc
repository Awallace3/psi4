// Diagnostic only: reproduce Psi4's basis/pair/DF query without allocating a DF plan.
// Input is produced from Psi4 GaussianShell::exp/coef (normalized, Bohr geometry).
#include <cuest.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

static void check(cuestStatus_t s) {
    if (s != CUEST_STATUS_SUCCESS)
        throw std::runtime_error("cuEST status " + std::to_string(static_cast<int>(s)));
}
static void cuda_check(cudaError_t s) {
    if (s != cudaSuccess) throw std::runtime_error(cudaGetErrorString(s));
}
struct Workspace {
    cuestWorkspace_t w{};
    explicit Workspace(const cuestWorkspaceDescriptor_t& d) {
        w.hostBufferSizeInBytes = d.hostBufferSizeInBytes;
        w.deviceBufferSizeInBytes = d.deviceBufferSizeInBytes;
        if (w.hostBufferSizeInBytes) {
            w.hostBuffer = reinterpret_cast<uintptr_t>(std::malloc(w.hostBufferSizeInBytes));
            if (!w.hostBuffer) throw std::bad_alloc();
        }
        if (w.deviceBufferSizeInBytes) {
            void* p = nullptr;
            auto status = cudaMalloc(&p, w.deviceBufferSizeInBytes);
            if (status != cudaSuccess) {
                std::free(reinterpret_cast<void*>(w.hostBuffer));
                w.hostBuffer = 0;
                cuda_check(status);
            }
            w.deviceBuffer = reinterpret_cast<uintptr_t>(p);
        }
    }
    ~Workspace() {
        if (w.deviceBuffer) cudaFree(reinterpret_cast<void*>(w.deviceBuffer));
        std::free(reinterpret_cast<void*>(w.hostBuffer));
    }
    Workspace(const Workspace&) = delete;
    Workspace& operator=(const Workspace&) = delete;
};
struct Basis {
    cuestAOBasis_t basis = nullptr;
    std::unique_ptr<Workspace> persistent;
    Basis(std::istream& input, uint64_t natom, cuestHandle_t handle) {
        std::vector<uint64_t> counts(natom);
        std::vector<cuestAOShell_t> shells;
        cuestAOShellParameters_t shell_params;
        check(cuestParametersCreate(CUEST_AOSHELL_PARAMETERS,
                                   reinterpret_cast<void**>(&shell_params)));
        for (uint64_t atom = 0; atom < natom; ++atom) {
            input >> counts[atom];
            if (!input || counts[atom] > 1000) throw std::runtime_error("Invalid shell count");
            for (uint64_t shell = 0; shell < counts[atom]; ++shell) {
                uint64_t pure, am, nprim;
                input >> pure >> am >> nprim;
                if (!input || pure > 1 || am > 10 || !nprim || nprim > 1000)
                    throw std::runtime_error("Invalid shell");
                std::vector<double> exps(nprim), coefs(nprim);
                for (auto& x : exps) input >> x;
                for (auto& x : coefs) input >> x;
                if (!input) throw std::runtime_error("Truncated primitive data");
                cuestAOShell_t s;
                check(cuestAOShellCreate(handle, pure, am, nprim, exps.data(),
                                        coefs.data(), shell_params, &s));
                shells.push_back(s);
            }
        }
        check(cuestParametersDestroy(CUEST_AOSHELL_PARAMETERS, shell_params));
        cuestAOBasisParameters_t params;
        check(cuestParametersCreate(CUEST_AOBASIS_PARAMETERS, reinterpret_cast<void**>(&params)));
        cuestWorkspaceDescriptor_t p{}, t{};
        check(cuestAOBasisCreateWorkspaceQuery(handle, natom, counts.data(), shells.data(),
                                             params, &p, &t, nullptr));
        persistent = std::make_unique<Workspace>(p);
        Workspace temporary(t);
        check(cuestAOBasisCreate(handle, natom, counts.data(), shells.data(), params,
                                &persistent->w, &temporary.w, &basis));
        check(cuestParametersDestroy(CUEST_AOBASIS_PARAMETERS, params));
        for (auto s : shells) check(cuestAOShellDestroy(s));
    }
    ~Basis() { if (basis) cuestAOBasisDestroy(basis); }
};

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("Usage: cuest_df_workspace_query basis.txt");
        std::ifstream input(argv[1]);
        uint64_t natom;
        input >> natom;
        if (!input || !natom || natom > 100000) throw std::runtime_error("Invalid atom count");
        std::vector<double> xyz(3 * natom);
        for (auto& x : xyz) input >> x;
        if (!input) throw std::runtime_error("Truncated coordinates");
        cudaStream_t stream{};
        cublasHandle_t blas{};
        cusolverDnHandle_t solver{};
        cuda_check(cudaStreamCreate(&stream));
        if (cublasCreate(&blas) != CUBLAS_STATUS_SUCCESS ||
            cusolverDnCreate(&solver) != CUSOLVER_STATUS_SUCCESS)
            throw std::runtime_error("Cannot initialize BLAS/solver");
        cublasSetStream(blas, stream);
        cusolverDnSetStream(solver, stream);
        cuestHandleParameters_t params;
        check(cuestParametersCreate(CUEST_HANDLE_PARAMETERS, &params));
        check(cuestParametersConfigure(CUEST_HANDLE_PARAMETERS, params,
              CUEST_HANDLE_PARAMETERS_CUDASTREAM, &stream, sizeof(stream)));
        check(cuestParametersConfigure(CUEST_HANDLE_PARAMETERS, params,
              CUEST_HANDLE_PARAMETERS_CUBLAS, &blas, sizeof(blas)));
        check(cuestParametersConfigure(CUEST_HANDLE_PARAMETERS, params,
              CUEST_HANDLE_PARAMETERS_CUSOLVER, &solver, sizeof(solver)));
        cuestHandle_t handle{};
        check(cuestCreate(params, &handle));
        check(cuestParametersDestroy(CUEST_HANDLE_PARAMETERS, params));
        check(cuestSetMathMode(handle, CUEST_NATIVE_FP64_MATH_MODE));
        {
            Basis primary(input, natom, handle), auxiliary(input, natom, handle);
            std::string extra;
            if (input >> extra) throw std::runtime_error("Unexpected trailing data");
            cuestAOPairListParameters_t pair_params;
            check(cuestParametersCreate(CUEST_AOPAIRLIST_PARAMETERS,
                                       reinterpret_cast<void**>(&pair_params)));
            cuestWorkspaceDescriptor_t p{}, t{};
            check(cuestAOPairListCreateWorkspaceQuery(handle, primary.basis, natom, xyz.data(),
                                                    1.e-12, pair_params, &p, &t, nullptr));
            Workspace pair_persistent(p);
            cuestAOPairList_t pairs{};
            {
                Workspace temporary(t);
                check(cuestAOPairListCreate(handle, primary.basis, natom, xyz.data(), 1.e-12,
                                           pair_params, &pair_persistent.w, &temporary.w, &pairs));
            }
            check(cuestParametersDestroy(CUEST_AOPAIRLIST_PARAMETERS, pair_params));
            // Match the benchmark's explicit INTS_TOLERANCE=1e-12.
            cuestDFIntPlanParameters_t df_params;
            check(cuestParametersCreate(CUEST_DFINTPLAN_PARAMETERS,
                                       reinterpret_cast<void**>(&df_params)));
            const double alpha = 1., beta = 0., omega = 0.;
            check(cuestParametersConfigure(CUEST_DFINTPLAN_PARAMETERS, df_params,
                  CUEST_DFINTPLAN_PARAMETERS_EXCHANGE_FRACTION, &alpha, sizeof(alpha)));
            check(cuestParametersConfigure(CUEST_DFINTPLAN_PARAMETERS, df_params,
                  CUEST_DFINTPLAN_PARAMETERS_LRC_EXCHANGE_FRACTION, &beta, sizeof(beta)));
            check(cuestParametersConfigure(CUEST_DFINTPLAN_PARAMETERS, df_params,
                  CUEST_DFINTPLAN_PARAMETERS_LRC_EXCHANGE_OMEGA, &omega, sizeof(omega)));
            check(cuestDFIntPlanCreateWorkspaceQuery(handle, primary.basis, auxiliary.basis,
                                                   pairs, df_params, &p, &t, nullptr));
            size_t free_bytes{}, total_bytes{};
            cuda_check(cudaMemGetInfo(&free_bytes, &total_bytes));
            std::cout << "{\"df_persistent_bytes\":" << p.deviceBufferSizeInBytes
                      << ",\"df_construction_temporary_bytes\":" << t.deviceBufferSizeInBytes
                      << ",\"df_host_persistent_bytes\":" << p.hostBufferSizeInBytes
                      << ",\"df_host_temporary_bytes\":" << t.hostBufferSizeInBytes
                      << ",\"device_free_bytes\":" << free_bytes
                      << ",\"device_total_bytes\":" << total_bytes
                      << ",\"ints_tolerance\":1e-12,\"exchange_fraction\":1.0"
                      << ",\"df_plan_allocated\":false,\"diagnostic_only\":true}\n";
            check(cuestParametersDestroy(CUEST_DFINTPLAN_PARAMETERS, df_params));
            check(cuestAOPairListDestroy(pairs));
        }
        check(cuestDestroy(handle));
        cusolverDnDestroy(solver);
        cublasDestroy(blas);
        cudaStreamDestroy(stream);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
