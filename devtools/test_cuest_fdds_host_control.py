"""Host-only control/layout tests of the actual FDDS CUDA adapter bodies.

Like Disp0's serial kernel shim, this is preparation for a device gate, not a
CUDA/cuBLAS numerical pass. A queued CPU model checks the BLAS argument mapping,
capacity decisions, borrowed-buffer drains and propagation of injected failures.
No vendor library is loaded; production algorithms are extracted, not copied.
"""
from pathlib import Path
import ast
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]


class WrapperControl(unittest.TestCase):
    def wrapper(self, core):
        source = (ROOT / "psi4/driver/procrouting/sapt/sapt_mp2_terms.py").read_text()
        function = next(n for n in ast.parse(source).body
                        if isinstance(n, ast.FunctionDef) and n.name == "_fdds_dot3")
        namespace = {"core": core}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "<actual _fdds_dot3>", "exec"),
             namespace)
        return namespace["_fdds_dot3"]

    def test_response_product_preflight_and_left_association(self):
        events = []

        class Operand:
            def __init__(self, name):
                self.name = name

            def dot(self, other):
                events.append((self.name, other.name))
                return Operand(f"({self.name}{other.name})")

        original = [Operand(name) for name in ("a", "b", "c")]
        core = SimpleNamespace(get_global_option=lambda key: True if key == "USE_CUEST" else 0)

        def preflight(*args):
            self.assertEqual(args[:3], tuple(original))
            return None

        core._cuest_fdds_dot3 = preflight
        result = self.wrapper(core)(*original)
        self.assertEqual(events, [("a", "b"), ("(ab)", "c")])
        self.assertEqual(result.name, "((ab)c)")
        events.clear()
        accepted = object()
        core._cuest_fdds_dot3 = lambda *args: accepted
        self.assertIs(self.wrapper(core)(*original), accepted)
        self.assertEqual(events, [])

    def test_response_product_runtime_failure_never_retries(self):
        class Poison:
            def dot(self, other):
                raise AssertionError("runtime error must not reach native retry")

        def failed(*args):
            raise RuntimeError("injected queued endpoint failure")

        core = SimpleNamespace(get_global_option=lambda key: True if key == "USE_CUEST" else 0,
                               _cuest_fdds_dot3=failed)
        with self.assertRaisesRegex(RuntimeError, "injected queued"):
            self.wrapper(core)(Poison(), Poison(), Poison())


class HostControl(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        source = (ROOT / "psi4/src/cuest_gemm.cc").read_text()
        header = (ROOT / "psi4/src/cuest_gemm.h").read_text().replace(
            '#include "psi4/libmints/typedefs.h"', "")
        buffers = source.split("class DeviceBuffer {", 1)[1].split(
            "cublasOperation_t to_cublas_op", 1)[0]
        helpers = source.split("size_t fdds_product(", 1)[1].split(
            "void check_cusolver(", 1)[0]
        adapters = source.split("size_t CuestDFMetric::capacity(", 1)[1].split(
            "/*! Multiply a list", 1)[0]
        harness = MODEL + header
        harness += "\nnamespace psi {\nclass DeviceBuffer {" + buffers
        harness += "\nsize_t fdds_product(" + helpers
        harness += "\nsize_t CuestDFMetric::capacity(" + adapters + "\n}\n"
        harness += CASES
        cpp = Path(cls.tmp.name) / "host_control.cc"
        cls.executable = Path(cls.tmp.name) / "host_control"
        cpp.write_text(harness)
        result = subprocess.run(
            ["c++", "-std=c++17", "-O0", "-Wall", "-Wextra", "-Werror",
             str(cpp), "-o", str(cls.executable)],
            text=True, capture_output=True, timeout=45)
        if result.returncode:
            raise AssertionError(result.stderr)

    def run_case(self, name):
        result = subprocess.run([str(self.executable), name],
                                text=True, capture_output=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_capacity_reduction_and_runtime_query_failure(self):
        self.run_case("capacity")

    def test_layouts_unequal_tails_and_fresh_inputs(self):
        self.run_case("layouts")

    def test_enqueue_errors_drain_and_propagate_without_retry(self):
        self.run_case("faults")


MODEL = r"""
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#define USING_cuEST
#define PSIEXCEPTION(message) std::runtime_error(message)
using cudaError_t = int;
using cudaStream_t = void*;
using cublasStatus_t = int;
using cublasPointerMode_t = int;
constexpr int cudaSuccess = 0, CUBLAS_STATUS_SUCCESS = 0;
constexpr int CUBLAS_POINTER_MODE_HOST = 0, CUBLAS_POINTER_MODE_DEVICE = 1;
constexpr int cudaMemcpyHostToDevice = 1, cudaMemcpyDeviceToHost = 2;
enum cublasOperation_t {CUBLAS_OP_N, CUBLAS_OP_T};
void* cublas_handle = reinterpret_cast<void*>(1);
std::vector<std::function<void()>> pending;
std::unordered_map<void*, size_t> allocations;
std::string fail_kind;
int fail_ordinal = 0, seen = 0, gemms = 0, syncs = 0, initializations = 0;
int pointer_mode = CUBLAS_POINTER_MODE_HOST;
size_t free_bytes = 1 << 20;
bool fail(const std::string& kind) {
    return kind == fail_kind && ++seen == fail_ordinal;
}
void inject(const std::string& kind, int ordinal = 1) {
    fail_kind = kind; fail_ordinal = ordinal; seen = 0;
}
void drain() {
    auto work = std::move(pending); pending.clear();
    for (auto& call : work) call();
}
cudaError_t cudaStreamSynchronize(cudaStream_t) {
    ++syncs; drain(); return fail("sync") ? 7 : 0;
}
cudaError_t cudaMalloc(void** ptr, size_t bytes) {
    if (fail("alloc")) return 7;
    *ptr = new unsigned char[bytes]; allocations[*ptr] = bytes; return 0;
}
cudaError_t cudaFree(void* ptr) {
    // Legacy cudaMalloc/cudaFree may synchronize. Method-level assertions below
    // still require explicit drains before returning borrowed host storage.
    drain(); assert(allocations.erase(ptr) == 1);
    delete[] static_cast<unsigned char*>(ptr); return 0;
}
const char* cudaGetErrorString(cudaError_t) {return "injected CPU-model failure";}
cudaError_t cudaMemGetInfo(size_t* free, size_t* total) {
    if (fail("memory")) return 7;
    *free = free_bytes; *total = free_bytes; return 0;
}
cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t bytes, int kind, cudaStream_t) {
    if (fail(kind == cudaMemcpyHostToDevice ? "upload" : "download")) return 7;
    pending.push_back([=] {std::memcpy(dst, src, bytes);}); return 0;
}
cudaError_t cudaMemsetAsync(void* dst, int value, size_t bytes, cudaStream_t) {
    if (fail("zero")) return 7;
    pending.push_back([=] {std::memset(dst, value, bytes);}); return 0;
}
cublasStatus_t cublasGetPointerMode(void*, cublasPointerMode_t* mode) {
    *mode = pointer_mode; return fail("pointer") ? 7 : 0;
}
cublasStatus_t cublasGetStream(void*, cudaStream_t* stream) {
    *stream = reinterpret_cast<void*>(2); return fail("stream") ? 7 : 0;
}
cublasStatus_t cublasDgemm(void*, cublasOperation_t ta, cublasOperation_t tb,
                         int m, int n, int k, const double* alpha,
                         const double* a, int lda, const double* b, int ldb,
                         const double* beta, double* c, int ldc) {
    ++gemms;
    if (fail("gemm")) return 7;
    const double av = *alpha, bv = *beta;
    pending.push_back([=] {
        for (int j = 0; j < n; ++j) for (int i = 0; i < m; ++i) {
            double sum = 0;
            for (int l = 0; l < k; ++l) {
                double x = ta == CUBLAS_OP_N ? a[i + l * lda] : a[l + i * lda];
                double y = tb == CUBLAS_OP_N ? b[l + j * ldb] : b[j + l * ldb];
                sum += x * y;
            }
            c[i + j * ldc] = av * sum + (bv == 0 ? 0 : bv * c[i + j * ldc]);
        }
    });
    return 0;
}
void timer_on(const char*) {}
void timer_off(const char*) {}
namespace psi {
namespace cuest_common {void ensure_cuest_initialized() {++initializations;}}
class Matrix {
    size_t q_;
    std::vector<double> data_;
public:
    explicit Matrix(size_t q) : q_(q), data_(q * q) {}
    int nirrep() const {return 1;}
    int symmetry() const {return 0;}
    size_t rowdim() const {return q_;}
    size_t coldim() const {return q_;}
    double* get_pointer(int) {return data_.data();}
};
using SharedMatrix = std::shared_ptr<Matrix>;
}
"""

CASES = r"""
using namespace psi;
void close(double a, double b) {assert(std::abs(a-b) < 1.e-11);}
std::vector<double> values(size_t n, double shift = 0) {
    std::vector<double> x(n);
    for (size_t i = 0; i < n; ++i) x[i] = shift + (int(i % 11) - 5) * 0.125;
    return x;
}
template<class Call> void throws(Call call) {
    bool raised = false;
    try {call();} catch (const std::runtime_error&) {raised = true;}
    assert(raised);
}
void capacity() {
    for (size_t available : {size_t(24), size_t(25), size_t(30)}) {
        free_bytes = available * sizeof(double);
        const size_t rows = available == 24 ? 0 : available == 25 ? 1 : 2;
        assert(CuestFDDSContractions::capacity(4, 100, false) == rows);
    }
    for (size_t available : {size_t(124), size_t(125), size_t(150)}) {
        free_bytes = available * sizeof(double);
        const size_t rows = available == 124 ? 0 : available == 125 ? 1 : 2;
        assert(CuestFDDSContractions::capacity(4, 100, true) == rows);
    }
    free_bytes = 100 * sizeof(double);  // 80 model doubles available.
    assert(CuestDFMetric::capacity(4, 100) == 32); // 16 + 2*32
    assert(CuestFDDSContractions::capacity(4, 100, false) == 16); // 16 + 4*16
    assert(CuestFDDSContractions::capacity(4, 100, true) == 0); // five fixed outputs
    free_bytes = 200 * sizeof(double);
    assert(CuestFDDSContractions::capacity(4, 100, true) == 4); // 80 + 20*4
    auto blocks = CuestFDDSExchange::capacity(3, 2, 4, 9, 7);
    assert(blocks.first && blocks.second);
    assert(blocks.first <= 9 && blocks.second <= 7);
    assert(exchange_storage(3, 2, 4, blocks.first, blocks.second) <= 160);
    free_bytes = 0;
    assert(CuestDFMetric::capacity(4, 100) == 0);
    assert(CuestFDDSExchange::capacity(3, 2, 4, 1, 1) == std::make_pair(size_t(0),size_t(0)));
    initializations = 0;
    assert(CuestDFMetric::capacity(0, 100) == 0);
    assert(CuestFDDSContractions::capacity(0, 100, false) == 0);
    assert(initializations == 0 && allocations.empty() && gemms == 0);
    free_bytes = 1000;
    inject("memory");
    throws([] {CuestFDDSContractions::capacity(3, 4, false);});
    assert(allocations.empty() && gemms == 0);
    inject("");
    auto metric = values(9);
    pointer_mode = CUBLAS_POINTER_MODE_HOST;
    initializations = 0;
    const size_t too_large = std::numeric_limits<int>::max();
    throws([&] {CuestFDDSContractions x(too_large, too_large, true);});
    assert(initializations == 0 && allocations.empty());
    throws([&] {CuestDFMetric x(too_large, too_large, metric.data(), 0);});
    assert(initializations == 0 && allocations.empty());
    throws([&] {CuestFDDSExchange x(too_large, too_large, too_large, 2, 3);});
    assert(initializations == 0 && allocations.empty());
    inject("");
    pointer_mode = CUBLAS_POINTER_MODE_DEVICE;
    throws([&] {CuestDFMetric x(3, 18, metric.data(), 0);});
    throws([] {CuestFDDSContractions x(3, 2, false);});
    throws([] {CuestFDDSExchange x(3, 2, 4, 2, 3);});
    assert(allocations.empty() && gemms == 0);
}
void layouts() {
    const size_t q = 4, p = 2;
    auto metric = values(q*q, .25);
    {
        CuestDFMetric fold(q, p*3*q, metric.data(), 0);
        for (auto layout : {CuestDFMetric::Layout::pqQ, CuestDFMetric::Layout::Qpq,
                            CuestDFMetric::Layout::pQq, CuestDFMetric::Layout::Qpq_to_pqQ}) {
            for (size_t r : {size_t(3), size_t(1), size_t(3)}) {
                auto input = values(p*r*q, double(r));
                std::vector<double> output(input.size()), expected(input.size(), 0);
                for (size_t i = 0; i < p; ++i) for (size_t j = 0; j < r; ++j)
                    for (size_t a = 0; a < q; ++a) for (size_t b = 0; b < q; ++b) {
                        if (layout == CuestDFMetric::Layout::pqQ)
                            expected[(i*r+j)*q+a] += input[(i*r+j)*q+b]*metric[b*q+a];
                        else if (layout == CuestDFMetric::Layout::Qpq)
                            expected[(a*p+i)*r+j] += metric[a*q+b]*input[(b*p+i)*r+j];
                        else if (layout == CuestDFMetric::Layout::pQq)
                            expected[(i*q+a)*r+j] += metric[a*q+b]*input[(i*q+b)*r+j];
                        else
                            expected[(i*r+j)*q+a] += input[(b*p+i)*r+j]*metric[b*q+a];
                    }
                fold.fold(layout, input.data(), output.data(), p, r);
                assert(pending.empty());
                for (size_t i = 0; i < output.size(); ++i) close(output[i], expected[i]);
            }
        }
    }
    for (bool hybrid : {false, true}) {
        CuestFDDSContractions contraction(q, 3, hybrid);
        const size_t ni = hybrid ? 5 : 1;
        std::vector<std::vector<double>> sums(ni, std::vector<double>(q*q));
        for (size_t rows : {size_t(3), size_t(1), size_t(2)}) {
            std::vector<std::vector<double>> input;
            for (size_t i = 0; i < ni; ++i) input.push_back(values(rows*q, i+.1*rows));
            if (hybrid) contraction.hybrid(input[0].data(), input[1].data(), input[2].data(),
                                           input[3].data(), input[4].data(), rows);
            else contraction.uncoupled(input[0].data(), rows);
            assert(pending.empty());
            size_t left[] = {0,0,0,1,4}, right[] = {1,2,3,4,2};
            for (size_t out = 0; out < ni; ++out)
                for (size_t a = 0; a < q; ++a) for (size_t b = 0; b < q; ++b)
                    for (size_t row = 0; row < rows; ++row)
                        sums[out][a*q+b] += input[hybrid ? left[out] : 0][row*q+a] *
                                           input[hybrid ? right[out] : 0][row*q+b];
            // Reuse caller scratch immediately after a returned upload.
            for (auto& x : input) std::fill(x.begin(), x.end(), -999.);
        }
        std::vector<SharedMatrix> output;
        for (size_t i = 0; i < ni; ++i) output.push_back(std::make_shared<Matrix>(q));
        contraction.finish(output);
        assert(pending.empty());
        for (size_t i = 0; i < ni; ++i) for (size_t j = 0; j < q*q; ++j)
            close(output[i]->get_pointer(0)[j], sums[i][j]);
    }
    {
        const size_t v = 2, k = 3;
        CuestFDDSExchange exchange(q, v, k, 2, 3);
        for (size_t outer_rows : {size_t(2), size_t(1), size_t(2)}) {
            auto outer = values(outer_rows*k*q, .1*outer_rows);
            std::vector<double> expected(outer_rows*v*q), qexpected(expected.size());
            exchange.begin(outer.data(), outer_rows);
            assert(pending.empty());
            for (size_t inner_rows : {size_t(3), size_t(1)}) {
                auto integrals = values(inner_rows*v*q, .2*inner_rows);
                auto raw = values(inner_rows*k*q, .3);
                auto orthogonal = values(inner_rows*k*q, -.2);
                for (size_t a = 0; a < outer_rows; ++a)
                    for (size_t b = 0; b < inner_rows; ++b)
                        for (size_t i = 0; i < v; ++i) for (size_t j = 0; j < k; ++j) {
                            double pair = 0;
                            for (size_t s = 0; s < q; ++s)
                                pair += integrals[(b*v+i)*q+s] * outer[(a*k+j)*q+s];
                            for (size_t s = 0; s < q; ++s) {
                                expected[(a*v+i)*q+s] += pair * raw[(b*k+j)*q+s];
                                qexpected[(a*v+i)*q+s] += pair * orthogonal[(b*k+j)*q+s];
                            }
                        }
                exchange.accumulate(integrals.data(), raw.data(), orthogonal.data(), inner_rows);
                assert(pending.empty());
                for (auto* field : {&integrals, &raw, &orthogonal})
                    std::fill(field->begin(), field->end(), -999.);
            }
            std::vector<double> output(expected.size()), qoutput(expected.size());
            exchange.finish(output.data(), qoutput.data());
            assert(pending.empty());
            for (size_t i = 0; i < output.size(); ++i) {
                close(output[i], expected[i]); close(qoutput[i], qexpected[i]);
            }
        }
    }
    assert(allocations.empty() && pending.empty());
}
void faults() {
    auto metric = values(9), input = values(18), output = values(18);
    for (auto stage : {"upload", "gemm", "download", "sync"}) {
        {
            CuestDFMetric fold(3, 18, metric.data(), 0);
            inject(stage);
            const int calls_before = gemms;
            throws([&] {fold.fold(CuestDFMetric::Layout::pqQ, input.data(), output.data(), 2, 3);});
            assert(pending.empty());
            assert(gemms - calls_before <= 1); // no hidden GEMM/retry
            inject("");
        }
        assert(allocations.empty());
        {
            CuestFDDSContractions contraction(3, 2, true);
            drain();
            inject(stage);
            if (std::string(stage) == "download") {
                contraction.hybrid(input.data(), input.data(), input.data(), input.data(), input.data(), 2);
                std::vector<SharedMatrix> out(5);
                for (auto& x : out) x = std::make_shared<Matrix>(3);
                inject("download", 2); // first destination may already be queued
                throws([&] {contraction.finish(out);});
            } else {
                throws([&] {contraction.hybrid(input.data(), input.data(), input.data(),
                                              input.data(), input.data(), 2);});
            }
            assert(pending.empty());
            inject("");
        }
        assert(allocations.empty());
    }
    for (int ordinal = 1; ordinal <= 3; ++ordinal) {
        inject("alloc", ordinal);
        throws([&] {CuestDFMetric fold(3, 18, metric.data(), 0);});
        assert(allocations.empty() && pending.empty());
    }
    inject("upload");
    throws([&] {CuestDFMetric fold(3, 18, metric.data(), 0);});
    assert(allocations.empty() && pending.empty());
    inject("");
    for (auto stage : {"upload", "gemm", "download", "sync"}) {
        {
            CuestFDDSExchange exchange(3, 2, 3, 1, 2);
            exchange.begin(input.data(), 1);
            inject(stage, std::string(stage) == "upload" ? 2 : 1);
            if (std::string(stage) == "download") {
                exchange.accumulate(input.data(), input.data(), input.data(), 2);
                inject("download", 2);
                throws([&] {exchange.finish(output.data(), input.data());});
            } else {
                throws([&] {exchange.accumulate(input.data(), input.data(), input.data(), 2);});
            }
            assert(pending.empty());
            inject("");
        }
        assert(allocations.empty());
    }
}
int main(int argc, char** argv) {
    assert(argc == 2);
    if (std::string(argv[1]) == "capacity") capacity();
    else if (std::string(argv[1]) == "layouts") layouts();
    else if (std::string(argv[1]) == "faults") faults();
    else assert(false);
    assert(allocations.empty() && pending.empty());
}
"""
