"""Compile actual runtime control flow against CPU-only failure-injection stubs.

This verifies cleanup/exception/retry semantics, NOT real CUDA/vendor execution.
"""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class Lifecycle(unittest.TestCase):
    def test_getter_and_parameter_failures_unwind_and_allow_retry(self):
        source = (ROOT / "psi4/src/cuest_runtime.cc").read_text()
        owner = source.split("struct CuestHandleParametersOwner {", 1)[1].split("\n};", 1)[0]
        cleanup = source.split("void cuest_cleanup_noexcept() {", 1)[1].split("// Some CUDA status", 1)[0]
        init = source.split("void cuest_init() {", 1)[1].split("}  // namespace", 1)[0]
        native = source.split("std::array<std::uint32_t, 3> native_version() {", 1)[1].split("\nvoid shutdown()", 1)[0]
        common = (ROOT / "psi4/src/psi4/libfock/cuESTCommon.h").read_text()
        check = common.split("inline void check_cuest(", 1)[1].split("constexpr double to_gib", 1)[0]
        harness = r"""
#include <array>
#include <cstdint>
#include <string>
#include <sstream>
#include <vector>
#include <stdexcept>
#include <cassert>
using cudaError_t=int;
using cublasStatus_t=int;
using cusolverStatus_t=int;
using cuestStatus_t=int;
using cuestHandle_t=void*;
using cuestHandleParameters_t=void*;
constexpr int cudaSuccess=0, CUBLAS_STATUS_SUCCESS=0, CUSOLVER_STATUS_SUCCESS=0, CUEST_STATUS_SUCCESS=0;
constexpr int CUEST_HANDLE_PARAMETERS=1, CUEST_HANDLE_PARAMETERS_CUDASTREAM=2,
              CUEST_HANDLE_PARAMETERS_CUBLAS=3, CUEST_HANDLE_PARAMETERS_CUSOLVER=4;
void *cuest_handle=nullptr, *cusolver_handle=nullptr, *cublas_handle=nullptr, *stream_handle=nullptr;
struct cudaDeviceProp {int major=9, minor=0; const char* name="CPU mock";};
struct Out {template<class... T> void Printf(const char*,T...){}};
Out out;
Out* outfile=&out;
struct PsiException:std::runtime_error {
    PsiException(const std::string& m,const char*,int):std::runtime_error(m){}
};
#define PSIEXCEPTION(message) PsiException(message,__FILE__,__LINE__)
std::vector<std::string> events;
int fail_getter=-1, getter_calls=0;
bool fail_configure=false, fail_create=false, fail_blas_stream=false, fail_solver_stream=false;
std::array<std::uint32_t,3> reported{0,2,50};
cudaError_t cudaGetDeviceCount(int* n){*n=1;return 0;}
cudaError_t cudaGetDevice(int* n){*n=0;return 0;}
cudaError_t cudaGetDeviceProperties(cudaDeviceProp*,int){return 0;}
cudaError_t cudaStreamCreate(void** h){*h=reinterpret_cast<void*>(1);events.push_back("create_stream");return 0;}
cudaError_t cudaStreamDestroy(void*){events.push_back("destroy_stream");return 0;}
cublasStatus_t cublasCreate(void** h){*h=reinterpret_cast<void*>(2);events.push_back("create_blas");return 0;}
cublasStatus_t cublasDestroy(void*){events.push_back("destroy_blas");return 0;}
cusolverStatus_t cusolverDnCreate(void** h){*h=reinterpret_cast<void*>(3);events.push_back("create_solver");return 0;}
cusolverStatus_t cusolverDnDestroy(void*){events.push_back("destroy_solver");return 0;}
int cublasSetStream(void*,void*){return fail_blas_stream?20:0;}
int cusolverDnSetStream(void*,void*){return fail_solver_stream?21:0;}
cuestStatus_t cuestParametersCreate(int,void** h){*h=reinterpret_cast<void*>(4);events.push_back("create_parameters");return 0;}
cuestStatus_t cuestParametersDestroy(int,void*){events.push_back("destroy_parameters");return 0;}
cuestStatus_t cuestParametersConfigure(int,void*,int,void*,std::size_t){return fail_configure?17:0;}
cuestStatus_t cuestCreate(void*,void** h){
    if(fail_create)return 18;
    *h=reinterpret_cast<void*>(5);events.push_back("create_cuest");return 0;
}
cuestStatus_t cuestDestroy(void*){events.push_back("destroy_cuest");return 0;}
cuestStatus_t get(void* h,std::uint32_t* v,int i) {
    assert(h);++getter_calls;if(i==fail_getter)return 19;
    *v=reported[i];return 0;
}
cuestStatus_t cuestGetMajorVersion(void* h,std::uint32_t* v){return get(h,v,0);}
cuestStatus_t cuestGetMinorVersion(void* h,std::uint32_t* v){return get(h,v,1);}
cuestStatus_t cuestGetPatchVersion(void* h,std::uint32_t* v){return get(h,v,2);}
std::string cuda_failure_message(const std::string& m,int,const std::string& = ""){return m;}
std::string describe_cuda_device(int,const cudaDeviceProp&){return "CPU mock";}
namespace psi {namespace cuest_runtime {std::array<std::uint32_t,3> native_version();}}
namespace psi {namespace cuest_common {
inline void check_cuest(
"""
        harness += check + "\n}}\nstruct CuestHandleParametersOwner {" + owner + "\n};\n"
        harness += "void cuest_cleanup_noexcept() {" + cleanup
        harness += "void cuest_init() {" + init
        harness += "\nnamespace psi {namespace cuest_runtime {\nstd::array<std::uint32_t,3> native_version() {" + native + "\n}}\n"
        harness += r"""
bool all_null(){return !cuest_handle&&!cusolver_handle&&!cublas_handle&&!stream_handle;}
void check_tail(const std::vector<std::string>& tail) {
    assert(events.size()>=tail.size());
    assert(std::vector<std::string>(events.end()-tail.size(),events.end())==tail);
}
int main() {
    for(int mode=0;mode<8;++mode) {
        events.clear();getter_calls=0;fail_getter=-1;
        reported={0,2,50};fail_configure=false;fail_create=false;
        fail_blas_stream=false;fail_solver_stream=false;
        if(mode<3)fail_getter=mode;
        if(mode==3)reported={0,2,2};
        if(mode==4)fail_configure=true;
        if(mode==5)fail_create=true;
        if(mode==6)fail_blas_stream=true;
        if(mode==7)fail_solver_stream=true;
        bool threw=false;
        try {cuest_init();} catch(const PsiException& e) {
            threw=true;
            if(mode<3)assert(std::string(e.what()).find("status code 19")!=std::string::npos);
            if(mode==3)assert(std::string(e.what()).find("required 0.2.50")!=std::string::npos);
            if(mode==4)assert(std::string(e.what()).find("status code 17")!=std::string::npos);
            if(mode==5)assert(std::string(e.what()).find("status code 18")!=std::string::npos);
            if(mode==6)assert(std::string(e.what()).find("cublasSetStream")!=std::string::npos);
            if(mode==7)assert(std::string(e.what()).find("cusolverDnSetStream")!=std::string::npos);
        }
        assert(threw&&all_null());
        if(mode<3)assert(getter_calls==mode+1);
        if(mode<4)check_tail({"destroy_cuest","destroy_solver","destroy_blas","destroy_stream"});
        else if(mode<6)check_tail({"destroy_parameters","destroy_solver","destroy_blas","destroy_stream"});
        else {
            assert(getter_calls==0);
            check_tail({"destroy_solver","destroy_blas","destroy_stream"});
        }
        int destructions=0;
        for(auto& e:events)if(e=="destroy_parameters")++destructions;
        assert(destructions==(mode<6?1:0));
        // A failed initialization must not poison the next independent request.
        fail_getter=-1;reported={0,2,50};fail_configure=false;fail_create=false;
        fail_blas_stream=false;fail_solver_stream=false;
        cuest_init();assert(!all_null());
        cuest_cleanup_noexcept();assert(all_null());
    }
}
"""
        with tempfile.TemporaryDirectory() as tmp:
            cpp, executable = Path(tmp) / "lifecycle.cc", Path(tmp) / "lifecycle"
            cpp.write_text(harness)
            result = subprocess.run(["c++", "-std=c++17", "-Wall", "-Wextra", "-Werror",
                                     str(cpp), "-o", str(executable)],
                                    text=True, capture_output=True, timeout=45)
            self.assertEqual(result.returncode, 0, result.stderr)
            result = subprocess.run([str(executable)], text=True, capture_output=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
