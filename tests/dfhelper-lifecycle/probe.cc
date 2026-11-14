#include <pybind11/pybind11.h>
#include "psi4/lib3index/dfhelper.h"

// Exercise the C++-only release API without adding a production Python binding.
PYBIND11_MODULE(_dfhelper_lifecycle_test, module) {
    module.def("release_tensor", &psi::DFHelper::release_tensor);
}
