# Can also run without building Psi4: cmake -S this_directory -B build.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 3.16)
    project(dfhelper_jk_blocking_test LANGUAGES CXX)
    enable_testing()
endif()

add_executable(dfhelper_jk_blocking_test.x dfhelper_jk_blocking_test.cc)
target_compile_features(dfhelper_jk_blocking_test.x PRIVATE cxx_std_17)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    add_test(NAME dfhelper-jk-blocking COMMAND dfhelper_jk_blocking_test.x)
    set_tests_properties(dfhelper-jk-blocking PROPERTIES TIMEOUT 10)
endif()
