"""Read-only policy getter control flow against host stubs, not native CUDA."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class RuntimePolicy(unittest.TestCase):
    def test_live_policy_statuses_domains_and_streams(self):
        source = (ROOT / "psi4/src/cuest_runtime.cc").read_text()
        body = source.split("RuntimePolicy current_policy() {", 1)[1].split(
            "std::array<std::uint32_t, 3> native_version()", 1)[0]
        harness = r"""
#include <cassert>
#include <stdexcept>
#include "cuest_runtime.h"
#define PSIEXCEPTION(message) std::runtime_error(message)
using cudaStream_t=void*;
using cuestMathMode_t=int;
using cublasPointerMode_t=int;
using cublasMath_t=int;
constexpr int CUBLAS_STATUS_SUCCESS=0, CUSOLVER_STATUS_SUCCESS=0;
void *cuest_handle=nullptr, *cublas_handle=nullptr, *cusolver_handle=nullptr, *stream_handle=nullptr;
int fail_at=-1, calls=0, native_mode=1, pointer_mode=0, blas_mode=0;
bool wrong_blas_stream=false, wrong_solver_stream=false;
int status(){return calls++ == fail_at ? 17 : 0;}
void check_status(int value){if(value)throw std::runtime_error("vendor status");}
#define CHECK_CUEST(expression) check_status(expression)
int cuestGetMathMode(void*,int* mode){*mode=native_mode;return status();}
int cublasGetPointerMode(void*,int* mode){*mode=pointer_mode;return status();}
int cublasGetMathMode(void*,int* mode){*mode=blas_mode;return status();}
int cublasGetStream(void*,void** stream){
    *stream=wrong_blas_stream?nullptr:stream_handle;return status();
}
int cusolverDnGetStream(void*,void** stream){
    *stream=wrong_solver_stream?nullptr:stream_handle;return status();
}
namespace psi {namespace cuest_runtime {
RuntimePolicy current_policy() {
"""
        harness += body + "\n}}\n"
        harness += r"""
void live() {
    cuest_handle=reinterpret_cast<void*>(1);cublas_handle=reinterpret_cast<void*>(2);
    cusolver_handle=reinterpret_cast<void*>(3);stream_handle=reinterpret_cast<void*>(4);
}
template<class Call> void throws(Call call){
    bool raised=false;try{call();}catch(const std::runtime_error&){raised=true;}
    assert(raised);
}
int main(){
    using psi::cuest_runtime::current_policy;
    for(int missing=0;missing<4;++missing){
        live();calls=0;
        if(missing==0)cuest_handle=nullptr;
        if(missing==1)cublas_handle=nullptr;
        if(missing==2)cusolver_handle=nullptr;
        if(missing==3)stream_handle=nullptr;
        throws([]{current_policy();});assert(calls==0);
    }
    live();
    for(int mode : {0,1}){
        native_mode=mode;calls=0;
        auto policy=current_policy();
        assert(calls==5 && policy.native_math_mode==mode);
        assert(policy.cublas_pointer_mode==0 && policy.cublas_math_mode==0);
        assert(policy.shared_stream && native_mode==mode);
    }
    for(int failing=0;failing<5;++failing){
        calls=0;fail_at=failing;
        throws([]{current_policy();});assert(calls==failing+1);
        assert(cuest_handle && cublas_handle && cusolver_handle && stream_handle);
    }
    fail_at=-1;
    pointer_mode=1;blas_mode=2;
    auto policy=current_policy();
    assert(policy.cublas_pointer_mode==1 && policy.cublas_math_mode==2);
    assert(pointer_mode==1 && blas_mode==2);
    wrong_blas_stream=true;assert(!current_policy().shared_stream);
    wrong_blas_stream=false;wrong_solver_stream=true;
    assert(!current_policy().shared_stream);
}
"""
        with tempfile.TemporaryDirectory() as tmp:
            cpp, exe = Path(tmp) / "policy.cc", Path(tmp) / "policy"
            cpp.write_text(harness)
            result = subprocess.run(
                ["c++", "-std=c++17", "-DUSING_cuEST", "-Wall", "-Wextra", "-Werror",
                 "-I" + str(ROOT / "psi4/src"), str(cpp), "-o", str(exe)],
                capture_output=True, text=True, timeout=45)
            self.assertEqual(result.returncode, 0, result.stderr)
            result = subprocess.run([str(exe)], capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_private_binding_has_only_value_fields(self):
        core = (ROOT / "psi4/src/core.cc").read_text()
        binding = core.split('core.def("_cuest_runtime_policy"', 1)[1].split(
            "#endif", 1)[0]
        for field in ("native_math_mode", "cublas_pointer_mode",
                      "cublas_math_mode", "shared_stream"):
            self.assertIn(f'receipt["{field}"] = policy.{field};', binding)
        self.assertNotIn("ensure_cuest_initialized", binding)
        self.assertNotIn("reinterpret_cast", binding)
