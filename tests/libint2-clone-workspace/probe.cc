#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "psi4/libmints/integral.h"
#include "psi4/libmints/twobody.h"
#include "psi4/libmints/basisset.h"
#include <algorithm>
#include <memory>
#include <string>
#include <vector>

namespace py = pybind11;
using namespace psi;

static std::vector<double> normal_integrals(TwoBodyAOInt& engine, int deriv) {
    // A multicenter quartet avoids identically zero same-center derivatives.
    int shells[] = {0, engine.basis2()->nshell() - 1,
                    std::max(0, engine.basis3()->nshell() - 2), engine.basis4()->nshell() - 1};
    size_t count = engine.basis1()->shell(shells[0]).nfunction() *
                   engine.basis2()->shell(shells[1]).nfunction() *
                   engine.basis3()->shell(shells[2]).nfunction() *
                   engine.basis4()->shell(shells[3]).nfunction();
    size_t computed = 0;
    if (deriv == 0) computed = engine.compute_shell(shells[0], shells[1], shells[2], shells[3]);
    if (deriv == 1) computed = engine.compute_shell_deriv1(shells[0], shells[1], shells[2], shells[3]);
    if (deriv == 2) computed = engine.compute_shell_deriv2(shells[0], shells[1], shells[2], shells[3]);
    // Only the requested derivative's chunks are populated by compute_shell.
    const size_t chunks = deriv == 0 ? 1 : deriv == 1 ? 12 : 78;
    if (computed == 0) return std::vector<double>(count * chunks, 0.0);
    std::vector<double> result;
    for (size_t chunk = 0; chunk < chunks; ++chunk) {
        const auto* buffer = engine.buffers().at(chunk);
        if (!buffer) throw std::runtime_error("Missing computed integral chunk");
        result.insert(result.end(), buffer, buffer + count);
    }
    return result;
}

static py::dict probe(std::shared_ptr<IntegralFactory> factory, const std::string& op, int deriv) {
    std::unique_ptr<TwoBodyAOInt> original;
    const std::vector<std::pair<double, double>> coefficients{{0.7, 0.3}, {1.4, -0.2}};
    if (op == "coulomb") original = factory->eri(deriv);
    else if (op == "erf") original = factory->erf_eri(0.37, deriv);
    else if (op == "erfc") original = factory->erf_complement_eri(0.37, deriv);
    else if (op == "yukawa") original = factory->yukawa_eri(0.63, deriv);
    else if (op == "f12") original = factory->f12(coefficients, deriv);
    else if (op == "f12_squared") original = factory->f12_squared(coefficients, deriv);
    else if (op == "f12g12") original = factory->f12g12(coefficients, deriv);
    else if (op == "f12dc") original = factory->f12_double_commutator(coefficients, deriv);
    else throw std::runtime_error("Unknown operator");

    std::unique_ptr<TwoBodyAOInt> copy(original->clone());
    std::unique_ptr<TwoBodyAOInt> copy2(copy->clone());
    py::dict result;
    std::vector<TwoBodyAOInt*> engines{original.get(), copy.get(), copy2.get()};
    std::vector<std::vector<double>> normal, rebuilt, bounds;
    for (auto* engine : engines) normal.push_back(normal_integrals(*engine, deriv));
    // Keep sources alive: pre-existing sieve lambdas capture their source object.
    for (auto* engine : engines) {
        if (!engine->sieve_initialized()) engine->initialize_sieve();
        if (engine->function_pairs().empty()) {
            bounds.push_back({0.0, 0.0, 0.0});  // Two-center integrals have no sieve.
        } else {
            bounds.push_back({engine->function_ceiling2(0, 0, 0, 0),
                              engine->shell_ceiling2(0, 0, 0, 0),
                              static_cast<double>(engine->function_pairs().size())});
        }
        rebuilt.push_back(normal_integrals(*engine, deriv));
    }
    result["normal"] = normal;
    result["rebuilt"] = rebuilt;
    result["bounds"] = bounds;
    return result;
}

PYBIND11_MODULE(_libint2_clone_test, module) {
    module.def("probe", &probe);
}
