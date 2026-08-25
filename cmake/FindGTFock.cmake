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
#  GTFock_LIBRARIES     - The GTFock/CInt/GTMatrix libraries
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
  ${GTFock_GTMATRIX_LIBRARY})

find_package_handle_standard_args(GTFock
  REQUIRED_VARS
    GTFock_PFOCK_LIBRARY
    GTFock_CINT_LIBRARY
    GTFock_GTMATRIX_LIBRARY
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
    find_package(simint 0.8 CONFIG REQUIRED)
    find_package(OpenMP REQUIRED COMPONENTS C)

    add_library(GTFock::gtfock UNKNOWN IMPORTED)
    set_target_properties(GTFock::gtfock PROPERTIES
      IMPORTED_LOCATION "${GTFock_PFOCK_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${GTFock_INCLUDE_DIRS}")
    target_link_libraries(GTFock::gtfock
      INTERFACE
        "${GTFock_CINT_LIBRARY}"
        "${GTFock_GTMATRIX_LIBRARY}"
        simint::simint
        MPI::MPI_C
        OpenMP::OpenMP_C)
endif()

mark_as_advanced(
  GTFock_PFOCK_INCLUDE_DIR
  GTFock_CINT_INCLUDE_DIR
  GTFock_GTMATRIX_INCLUDE_DIR
  GTFock_PFOCK_LIBRARY
  GTFock_CINT_LIBRARY
  GTFock_GTMATRIX_LIBRARY)
