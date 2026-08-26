# - Find GTFock, the MPI-distributed Fock-build engine
#
# GTFock does not export a CMake package config file, so this module locates the
# headers and libraries that `gtfock_psi4`'s `build_deps.sh` installs into its
# `_install/` prefix (see doc/sphinxman/source/gtfock.rst). Point CMake at that
# prefix with either -DGTFock_ROOT=/path/to/gtfock_psi4/_install or by adding it
# to CMAKE_PREFIX_PATH.
#
# GTFock's Fock build is MPI-parallel and evaluates its two-electron integrals
# with Simint, so both are requirements of the imported target rather than
# separate opt-in choices.
#
# This module sets:
#
#  GTFock_FOUND         - True when every GTFock piece was located
#  GTFock_INCLUDE_DIRS  - Include directories for pfock.h, CInt.h, GTMatrix.h
#  GTFock_LIBRARIES     - The GTFock/CInt/GTMatrix/Simint libraries
#
# and defines the imported target:
#
#  GTFock::gtfock       - GTFock plus CInt, GTMatrix, Simint, and MPI

include(FindPackageHandleStandardArgs)

find_path(GTFock_PFOCK_INCLUDE_DIR
  NAMES pfock.h
  PATH_SUFFIXES include
  DOC "Directory holding GTFock's pfock.h")

find_path(GTFock_CINT_INCLUDE_DIR
  NAMES CInt.h
  PATH_SUFFIXES include
  DOC "Directory holding GTFock's CInt.h")

find_path(GTFock_GTMATRIX_INCLUDE_DIR
  NAMES GTMatrix.h
  PATH_SUFFIXES include
  DOC "Directory holding GTFock's GTMatrix.h")

find_library(GTFock_PFOCK_LIBRARY
  NAMES gtfock
  PATH_SUFFIXES lib lib64
  DOC "The GTFock PFock library")

find_library(GTFock_CINT_LIBRARY
  NAMES cint
  PATH_SUFFIXES lib lib64
  DOC "The GTFock CInt integral-driver library")

find_library(GTFock_GTMATRIX_LIBRARY
  NAMES GTMatrix
  PATH_SUFFIXES lib lib64
  DOC "The GTMatrix distributed-matrix library")

# Simint is resolved as a plain library path rather than through
# find_package(simint CONFIG), on purpose. That find would define the
# simint::simint imported target in the calling directory scope, and Psi4 gates
# its own Simint ERI engine on `if(TARGET simint::simint)` rather than on
# ENABLE_simint -- so enabling GTFock would silently switch on an unrelated
# integral engine that the build still reports as disabled, and without the
# am${MAX_AM_ERI} component check Psi4's own Simint block requires. GTFock's
# public headers never include <simint/simint.h> (libcint's CInt.h says so
# outright), so the library alone is all this target needs.
#
# The search is confined to the prefix GTFock itself was found in, because the
# only correct Simint here is the one GTFock was linked against: Psi4's shim
# reads _SIMINT_OSTEI_MAXAM out of GTFock's CInt.h to bound shell angular
# momentum, and a different Simint on the system would make that guard describe
# a library the build does not use.
if(GTFock_PFOCK_LIBRARY)
    get_filename_component(_gtfock_libdir "${GTFock_PFOCK_LIBRARY}" DIRECTORY)
    get_filename_component(_gtfock_prefix "${_gtfock_libdir}" DIRECTORY)
    find_library(GTFock_SIMINT_LIBRARY
      NAMES simint
      HINTS "${_gtfock_prefix}" "${GTFock_ROOT}" ENV GTFock_ROOT
      PATH_SUFFIXES lib lib64
      NO_DEFAULT_PATH
      DOC "The Simint ERI library GTFock was built against")
    unset(_gtfock_libdir)
    unset(_gtfock_prefix)
endif()

set(GTFock_INCLUDE_DIRS
  ${GTFock_PFOCK_INCLUDE_DIR}
  ${GTFock_CINT_INCLUDE_DIR}
  ${GTFock_GTMATRIX_INCLUDE_DIR})
if(GTFock_INCLUDE_DIRS)
    list(REMOVE_DUPLICATES GTFock_INCLUDE_DIRS)
endif()
set(GTFock_LIBRARIES
  ${GTFock_PFOCK_LIBRARY}
  ${GTFock_CINT_LIBRARY}
  ${GTFock_GTMATRIX_LIBRARY}
  ${GTFock_SIMINT_LIBRARY})

find_package_handle_standard_args(GTFock
  REQUIRED_VARS
    GTFock_PFOCK_LIBRARY
    GTFock_CINT_LIBRARY
    GTFock_GTMATRIX_LIBRARY
    GTFock_SIMINT_LIBRARY
    GTFock_PFOCK_INCLUDE_DIR
    GTFock_CINT_INCLUDE_DIR
    GTFock_GTMATRIX_INCLUDE_DIR
  REASON_FAILURE_MESSAGE
    "Build GTFock first with gtfock_psi4's build_deps.sh (GTF_COMBINED_JK=OFF) and pass -DGTFock_ROOT=<gtfock_psi4>/_install. See doc/sphinxman/source/gtfock.rst.")

if(GTFock_FOUND AND NOT TARGET GTFock::gtfock)
    # GTFock's Fock build is MPI-parallel and its ERIs come from Simint; a
    # GTFock target without both is not usable. GTFock is C, so ask only for the
    # C components and let Psi4's C++ shim sit on top of the C API.
    find_package(MPI REQUIRED COMPONENTS C)
    find_package(OpenMP REQUIRED COMPONENTS C)

    add_library(GTFock::gtfock UNKNOWN IMPORTED)
    set_target_properties(GTFock::gtfock PROPERTIES
      IMPORTED_LOCATION "${GTFock_PFOCK_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${GTFock_INCLUDE_DIRS}")
    target_link_libraries(GTFock::gtfock
      INTERFACE
        "${GTFock_CINT_LIBRARY}"
        "${GTFock_GTMATRIX_LIBRARY}"
        "${GTFock_SIMINT_LIBRARY}"
        MPI::MPI_C
        OpenMP::OpenMP_C)
endif()

mark_as_advanced(
  GTFock_PFOCK_INCLUDE_DIR
  GTFock_CINT_INCLUDE_DIR
  GTFock_GTMATRIX_INCLUDE_DIR
  GTFock_PFOCK_LIBRARY
  GTFock_CINT_LIBRARY
  GTFock_GTMATRIX_LIBRARY
  GTFock_SIMINT_LIBRARY)
