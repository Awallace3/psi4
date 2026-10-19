cmake_minimum_required(VERSION 3.25.2)
project(disp0_catch_config_probe LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
include(CheckCXXSourceCompiles)
option(DISP0_REPRODUCE_SOURCE_ALIAS "Reproduce the failed non-imported target probe" OFF)

if(DISP0_REPRODUCE_SOURCE_ALIAS)
    # A check_* try_compile can import an installed IMPORTED target, but not
    # this source-backed alias from its parent project. This diagnostic models
    # the failed FetchContent path; it is not a Catch2 or dispersion algorithm.
    add_library(catch_source STATIC fallback.cc)
    add_library(Catch2::Catch2 ALIAS catch_source)
    set(CMAKE_REQUIRED_LIBRARIES Catch2::Catch2)
    check_cxx_source_compiles("int main() { return 0; }" DISP0_CATCH2_LINKS)
else()
    find_package(Catch2 3 CONFIG REQUIRED)
    get_target_property(is_imported Catch2::Catch2 IMPORTED)
    if(NOT is_imported)
        message(FATAL_ERROR "The Catch2 probe requires an installed IMPORTED target")
    endif()
    set(CMAKE_REQUIRED_LIBRARIES Catch2::Catch2)
    # Same symbol tested unconditionally by Einsums d17bb76e SetupCatch.
    check_cxx_source_compiles(
        "#include <catch2/catch_tostring.hpp>
         #include <string_view>
         int main() {
             return static_cast<int>(Catch::Detail::stringify(std::string_view{\"x\"}).size());
         }"
        DISP0_CATCH2_LINKS)
endif()
if(NOT DISP0_CATCH2_LINKS)
    message(FATAL_ERROR "Catch2 link probe failed")
endif()
message(STATUS "DISP0 imported Catch2 link probe PASS")
