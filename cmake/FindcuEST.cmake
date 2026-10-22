# Exact, content-addressed vendor SDK admission. No CONFIG/system/static fallback.
# Set cuEST_ROOT to an extracted cuest_PID prefix and cuEST_CUDA_RUNTIME_ROOT to
# the read-only CUDA runtime prefix. CPU-only builds never include this module.
include(FindPackageHandleStandardArgs)
find_package(Python 3.10 COMPONENTS Interpreter REQUIRED)
find_package(CUDAToolkit REQUIRED)
set(cuEST_ROOT "${cuEST_ROOT}" CACHE PATH "Canonical cuEST 0.2.50 SDK prefix")
if(NOT cuEST_ROOT AND DEFINED ENV{cuEST_ROOT})
    set(cuEST_ROOT "$ENV{cuEST_ROOT}" CACHE PATH "Canonical cuEST 0.2.50 SDK prefix" FORCE)
endif()
if(NOT IS_DIRECTORY "${cuEST_ROOT}")
    message(FATAL_ERROR "cuEST 0.2.50 requires explicit cuEST_ROOT (no system/conda fallback)")
endif()
file(REAL_PATH "${cuEST_ROOT}" _cuEST_root)
set(cuEST_CUDA_RUNTIME_ROOT "${cuEST_CUDA_RUNTIME_ROOT}" CACHE PATH "CUDA runtime closure prefix")
if(NOT IS_DIRECTORY "${cuEST_CUDA_RUNTIME_ROOT}")
    message(FATAL_ERROR "Set cuEST_CUDA_RUNTIME_ROOT to the selected CUDA runtime prefix")
endif()
if(NOT CUDAToolkit_VERSION_MAJOR MATCHES "^(12|13)$")
    message(FATAL_ERROR "Canonical cuEST 0.2.50 supports CUDA toolkit 12 or 13 only")
endif()
set(_cuEST_include "${_cuEST_root}/include")
set(_cuEST_library "${_cuEST_root}/lib/${CUDAToolkit_VERSION_MAJOR}/libcuest.so")
# Do not silently repair a stale cache (including old conda library overrides).
foreach(_item INCLUDE_DIR LIBRARY)
    if(_item STREQUAL "INCLUDE_DIR")
        set(_wanted "${_cuEST_include}")
    else()
        set(_wanted "${_cuEST_library}")
    endif()
    if(DEFINED cuEST_${_item} AND NOT "${cuEST_${_item}}" STREQUAL "")
        file(REAL_PATH "${cuEST_${_item}}" _given)
        file(REAL_PATH "${_wanted}" _canonical)
        if(NOT _given STREQUAL _canonical)
            message(FATAL_ERROR "Stale/cross-CUDA cuEST_${_item}; use a fresh build")
        endif()
    endif()
endforeach()
set(cuEST_INCLUDE_DIR "${_cuEST_include}" CACHE PATH "Verified cuEST headers")
set(cuEST_LIBRARY "${_cuEST_library}" CACHE FILEPATH "Verified cuEST shared library")
if(cuEST_STATIC_LIBRARY)
    message(FATAL_ERROR "Unverified static cuEST libraries are not supported")
endif()
set(cuEST_IDENTITY_FILE "${CMAKE_BINARY_DIR}/cuest-sdk-identity.json")
if(cuEST_SUPERBUILD_INNER AND NOT EXISTS "${cuEST_PARENT_IDENTITY}")
    message(FATAL_ERROR "Superbuild inner configure requires the outer cuEST identity")
endif()
get_target_property(_cuEST_cudart CUDA::cudart IMPORTED_LOCATION)
if(NOT _cuEST_cudart)
    message(FATAL_ERROR "CUDA::cudart has no pinned imported location")
endif()
set(_cuEST_expected "")
if(cuEST_PARENT_IDENTITY)
    list(APPEND _cuEST_expected --expected "${cuEST_PARENT_IDENTITY}")
endif()
execute_process(
    COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_LIST_DIR}/cuest_sdk_contract.py"
        --sdk-root "${_cuEST_root}" --cuda-major "${CUDAToolkit_VERSION_MAJOR}"
        --cuda-root "${cuEST_CUDA_RUNTIME_ROOT}"
        --cudart "${_cuEST_cudart}"
        ${_cuEST_expected} --output "${cuEST_IDENTITY_FILE}"
    RESULT_VARIABLE _cuEST_result OUTPUT_VARIABLE _cuEST_out ERROR_VARIABLE _cuEST_error)
if(NOT _cuEST_result EQUAL 0)
    message(FATAL_ERROR "cuEST 0.2.50 identity/closure admission failed:\n${_cuEST_out}${_cuEST_error}")
endif()
set(cuEST_VERSION "0.2.50")
set_property(GLOBAL PROPERTY PSI4_CUEST_IDENTITY "${cuEST_IDENTITY_FILE}")
find_package_handle_standard_args(cuEST
    REQUIRED_VARS cuEST_INCLUDE_DIR cuEST_LIBRARY cuEST_IDENTITY_FILE
    VERSION_VAR cuEST_VERSION)
if(NOT TARGET cuEST::cuEST)
    add_library(cuEST::cuEST SHARED IMPORTED GLOBAL)
    set_target_properties(cuEST::cuEST PROPERTIES
        IMPORTED_LOCATION "${cuEST_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${cuEST_INCLUDE_DIR}"
        INTERFACE_LINK_LIBRARIES "CUDA::cudart"
        SYSTEM FALSE
        PSI4_CUEST_CONTRACT_IDENTITY "${cuEST_IDENTITY_FILE}")
else()
    get_target_property(_cuEST_owned cuEST::cuEST PSI4_CUEST_CONTRACT_IDENTITY)
    if(NOT _cuEST_owned STREQUAL cuEST_IDENTITY_FILE)
        message(FATAL_ERROR "Foreign pre-existing cuEST target cannot bypass admission")
    endif()
    foreach(_prop IN ITEMS IMPORTED_LOCATION INTERFACE_INCLUDE_DIRECTORIES INTERFACE_LINK_LIBRARIES SYSTEM)
        get_target_property(_value cuEST::cuEST "${_prop}")
        if(_prop STREQUAL "IMPORTED_LOCATION")
            set(_expected "${cuEST_LIBRARY}")
        elseif(_prop STREQUAL "INTERFACE_INCLUDE_DIRECTORIES")
            set(_expected "${cuEST_INCLUDE_DIR}")
        elseif(_prop STREQUAL "INTERFACE_LINK_LIBRARIES")
            set(_expected "CUDA::cudart")
        else()
            set(_expected FALSE)
        endif()
        if(NOT "${_value}" STREQUAL "${_expected}")
            message(FATAL_ERROR "cuEST target property changed: ${_prop}")
        endif()
    endforeach()
    get_target_property(_configs cuEST::cuEST IMPORTED_CONFIGURATIONS)
    foreach(_config IN ITEMS NOCONFIG RELEASE DEBUG RELWITHDEBINFO MINSIZEREL ${_configs})
        string(TOUPPER "${_config}" _config)
        get_target_property(_value cuEST::cuEST "IMPORTED_LOCATION_${_config}")
        if(_value)
            message(FATAL_ERROR "Configuration-specific cuEST target binding is not admitted")
        endif()
    endforeach()
    if(_configs)
        message(FATAL_ERROR "Foreign cuEST imported configurations are not admitted")
    endif()
endif()

# Exact SDK admission is not a substitute for compile/link feature detection.
# Language-free admission fixtures do not compile; production configures use CXX.
if(CMAKE_CXX_COMPILER_LOADED)
  include(CheckCXXSourceCompiles)
  include(CMakePushCheckState)
  cmake_push_check_state(RESET)
  set(CMAKE_REQUIRED_LIBRARIES cuEST::cuEST)
  unset(cuEST_HAS_DIRECT_JK CACHE)
  check_cxx_source_compiles([=[
    #include <cuest.h>
    int main() {
      auto volatile a = &cuestDirectIntPlanCreate;
      auto volatile b = &cuestDirectIntPlanCreateWorkspaceQuery;
      auto volatile c = &cuestDirectIntPlanDestroy;
      auto volatile d = &cuestDirectCoulombCompute;
      auto volatile e = &cuestDirectCoulombComputeWorkspaceQuery;
      auto volatile f = &cuestDirectSymmetricExchangeCompute;
      auto volatile g = &cuestDirectSymmetricExchangeComputeWorkspaceQuery;
      return !(a && b && c && d && e && f && g);
    }
  ]=] cuEST_HAS_DIRECT_JK)
  cmake_pop_check_state()
endif()
