# FindcuEST.cmake
# ---------------
# Finds cuEST headers and libraries.
#
# Result variables:
#   cuEST_FOUND
#   cuEST_INCLUDE_DIR
#   cuEST_LIBRARY            (shared, if found)
#   cuEST_STATIC_LIBRARY     (static, if found)
#
# Imported targets:
#   cuEST::cuEST         - shared library target if available, else static
#
# Hints:
#   -DcuEST_ROOT=/path/to/prefix
#   or set environment variable cuEST_ROOT

cmake_minimum_required(VERSION 3.15)

include(FindPackageHandleStandardArgs)

# ---- user hints ----
set(_cuEST_ROOT_HINTS "")
if(DEFINED cuEST_ROOT)
  list(APPEND _cuEST_ROOT_HINTS "${cuEST_ROOT}")
endif()
if(DEFINED ENV{cuEST_ROOT})
  list(APPEND _cuEST_ROOT_HINTS "$ENV{cuEST_ROOT}")
endif()

# ---- include dir ----
find_path(cuEST_INCLUDE_DIR
  NAMES cuest.h
  HINTS ${_cuEST_ROOT_HINTS}
  PATH_SUFFIXES include
)

# ---- detect CUDA major version for matching library ----
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
  string(REGEX MATCH "^([0-9]+)" _cuEST_CUDA_MAJOR "${CUDAToolkit_VERSION}")
  message(STATUS "cuEST: detected CUDA toolkit major version ${_cuEST_CUDA_MAJOR}")
else()
  set(_cuEST_CUDA_MAJOR "")
  message(STATUS "cuEST: CUDA toolkit not detected, will search all library subdirectories")
endif()

# ---- library dir suffixes (prefer matching CUDA version) ----
set(_cuEST_LIB_SUFFIXES "")
if(_cuEST_CUDA_MAJOR)
  list(APPEND _cuEST_LIB_SUFFIXES "lib/${_cuEST_CUDA_MAJOR}")
endif()
list(APPEND _cuEST_LIB_SUFFIXES lib lib/13 lib/12)

# Shared
find_library(cuEST_LIBRARY
  # The canonical 0.2.50 archive can be staged without an unversioned symlink.
  NAMES cuest libcuest.so.0.2.50
  HINTS ${_cuEST_ROOT_HINTS}
  PATH_SUFFIXES ${_cuEST_LIB_SUFFIXES}
)

# Static
find_library(cuEST_STATIC_LIBRARY
  NAMES cuest_static
  HINTS ${_cuEST_ROOT_HINTS}
  PATH_SUFFIXES ${_cuEST_LIB_SUFFIXES}
)

# ---- found? ----
# Headers alone must not report success without creating an imported target.
if(cuEST_LIBRARY)
  set(_cuEST_LINK_LIBRARY "${cuEST_LIBRARY}")
else()
  set(_cuEST_LINK_LIBRARY "${cuEST_STATIC_LIBRARY}")
endif()
find_package_handle_standard_args(cuEST
  REQUIRED_VARS cuEST_INCLUDE_DIR _cuEST_LINK_LIBRARY
  HANDLE_COMPONENTS
)

mark_as_advanced(cuEST_INCLUDE_DIR cuEST_LIBRARY cuEST_STATIC_LIBRARY)

# ---- imported targets ----
if(cuEST_FOUND)
  # Convenience target that picks shared if available.
  find_package(CUDAToolkit REQUIRED)
  if(NOT TARGET cuEST::cuEST)
    if(cuEST_LIBRARY)
    add_library(cuEST::cuEST SHARED IMPORTED GLOBAL)
      set_target_properties(cuEST::cuEST PROPERTIES
        IMPORTED_LOCATION "${cuEST_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${cuEST_INCLUDE_DIR}"
	INTERFACE_LINK_LIBRARIES "CUDA::cudart"
      )
    elseif(cuEST_STATIC_LIBRARY)
      add_library(cuEST::cuEST STATIC IMPORTED GLOBAL)
      set_target_properties(cuEST::cuEST PROPERTIES
        IMPORTED_LOCATION "${cuEST_STATIC_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${cuEST_INCLUDE_DIR}"
	INTERFACE_LINK_LIBRARIES "CUDA::cudart"
      )
    endif()
  endif()
endif()
