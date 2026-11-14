cmake_minimum_required(VERSION 3.20)
project(libint2_clone_workspace LANGUAGES CXX)
find_package(psi4 CONFIG REQUIRED)
pybind11_add_module(_libint2_clone_test probe.cc)
target_link_libraries(_libint2_clone_test PRIVATE psi4::core)
enable_testing()
add_test(NAME libint2-clone-workspace
         COMMAND "${Python_EXECUTABLE}" -m pytest "${CMAKE_CURRENT_SOURCE_DIR}/test_clone.py" -q -x)
set_tests_properties(libint2-clone-workspace PROPERTIES
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}:${psi4_SYS_PATH};OMP_NUM_THREADS=2;MKL_NUM_THREADS=2;PSI4_REQUIRE_CLONE_TEST=1"
    TIMEOUT 600)
