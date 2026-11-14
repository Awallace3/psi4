cmake_minimum_required(VERSION 3.20)
project(dfhelper_lifecycle LANGUAGES CXX)
find_package(psi4 CONFIG REQUIRED)
pybind11_add_module(_dfhelper_lifecycle_test probe.cc)
target_link_libraries(_dfhelper_lifecycle_test PRIVATE psi4::core)
enable_testing()
add_test(NAME dfhelper-lifecycle
         COMMAND "${Python_EXECUTABLE}" -m pytest "${CMAKE_CURRENT_SOURCE_DIR}/test_release.py"
                 "--confcutdir=${CMAKE_CURRENT_SOURCE_DIR}" "--rootdir=${CMAKE_CURRENT_SOURCE_DIR}"
                 -o addopts= -q)
set_tests_properties(dfhelper-lifecycle PROPERTIES
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}:${psi4_SYS_PATH};OMP_NUM_THREADS=2;MKL_NUM_THREADS=2;PSI4_REQUIRE_DFHELPER_TEST=1"
    TIMEOUT 120)
