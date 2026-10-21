"""CPU-only compiled ABI/semantic regression for the actual native-version body.

Vendor calls are stubs with the verified status_t(handle, uint32_t*) signatures.
This does not compile/load the full CUDA backend or exercise a real SDK handle.
"""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class NativeVersion(unittest.TestCase):
    def test_actual_body_against_verified_abi_stubs(self):
        source = (ROOT / "psi4/src/cuest_runtime.cc").read_text()
        body = source.split("std::array<std::uint32_t, 3> native_version() {", 1)[1]
        body = body.split("\nvoid shutdown()", 1)[0]
        program = r"""
#include <array>
#include <cstdint>
#include <stdexcept>
#include <cassert>
using cuestHandle_t = void*;
using cuestStatus_t = int;
constexpr int CUEST_STATUS_SUCCESS = 0;
cuestHandle_t cuest_handle = nullptr;
int status = 0, calls = 0, fail_index = -1;
std::array<std::uint32_t, 3> reported{0, 2, 50};
cuestStatus_t getter(cuestHandle_t handle, std::uint32_t* value, int index) {
    assert(handle && value);
    ++calls;
    if (status && index == fail_index) return status;
    *value = reported[index];
    return 0;
}
cuestStatus_t cuestGetMajorVersion(cuestHandle_t h, std::uint32_t* v) {return getter(h, v, 0);}
cuestStatus_t cuestGetMinorVersion(cuestHandle_t h, std::uint32_t* v) {return getter(h, v, 1);}
cuestStatus_t cuestGetPatchVersion(cuestHandle_t h, std::uint32_t* v) {return getter(h, v, 2);}
#define PSIEXCEPTION(message) std::runtime_error(message)
#define CHECK_CUEST(call) do {if ((call) != CUEST_STATUS_SUCCESS) throw std::runtime_error("status");} while(0)
std::array<std::uint32_t, 3> native_version() {
"""
        program += body + r"""
int main() {
    bool threw = false;
    try {native_version();} catch (const std::runtime_error&) {threw = true;}
    assert(threw && calls == 0); // no hidden initialization or getters on null
    cuest_handle = reinterpret_cast<void*>(1);
    assert(native_version() == reported && calls == 3);
    for (auto wrong : {std::array<std::uint32_t,3>{1,2,50}, {0,3,50}, {0,2,2}}) {
        reported = wrong; calls = 0; threw = false;
        try {native_version();} catch (const std::runtime_error&) {threw = true;}
        assert(threw && calls == 3);
    }
    reported = {0, 2, 50}; status = 1;
    for (fail_index = 0; fail_index < 3; ++fail_index) {
        calls = 0; threw = false;
        try {native_version();} catch (const std::runtime_error&) {threw = true;}
        assert(threw && calls == fail_index+1); // no later getter after failure
    }
}
"""
        compiler = shutil.which("c++")
        self.assertIsNotNone(compiler, "CPU compiler required for this offline gate")
        with tempfile.TemporaryDirectory() as tmp:
            cpp, binary = Path(tmp) / "test.cc", Path(tmp) / "test"
            cpp.write_text(program)
            result = subprocess.run([compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror",
                                     str(cpp), "-o", str(binary)],
                                    text=True, capture_output=True, timeout=45)
            self.assertEqual(result.returncode, 0, result.stderr)
            result = subprocess.run([str(binary)], text=True, capture_output=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_cpu_header_compiles_without_cuda(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "cpu.cc"
            source.write_text('#include "cuest_runtime.h"\n'
                              'int main() {psi::cuest_runtime::shutdown(); '
                              'static_assert(!psi::cuest_runtime::built_with_cuest());}\n')
            result = subprocess.run(["c++", "-std=c++17", "-fsyntax-only",
                                     "-I", str(ROOT / "psi4/src"), str(source)],
                                    text=True, capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_init_checks_after_handle_created_and_parameters_destroyed(self):
        source = (ROOT / "psi4/src/cuest_runtime.cc").read_text()
        init = source.split("void cuest_init()", 1)[1].split("}  // namespace", 1)[0]
        self.assertLess(init.index("cuestCreate(parameters.value, &cuest_handle)"),
                        init.index("psi::cuest_runtime::native_version();"))
        self.assertLess(init.index("cuestParametersDestroy(CUEST_HANDLE_PARAMETERS, parameters.value)"),
                        init.index("psi::cuest_runtime::native_version();"))
        self.assertIn("cuest_cleanup_noexcept();", init)
        self.assertIn("CUEST_VER_PATCH == 50", source)

    def test_binding_explicit_tuple(self):
        source = (ROOT / "psi4/src/core.cc").read_text()
        block = source.split('core.def("_cuest_runtime_version"', 1)[1].split("#endif", 1)[0]
        self.assertIn("py::make_tuple(version[0], version[1], version[2])", block)


if __name__ == "__main__":
    unittest.main()
