list(APPEND sources
  native_response.cc
  casimir_grid.cc
  explicit_basis.cc
  partitioned_response.cc
  multipole_transform.cc
  lw_localization.cc
  harmonic_transform.cc
  aux_coulomb.cc
  orbital_coulomb.cc
  isa_grid.cc
  tables.cc
  )

# Keep the heavy Libint2 engine implementation out of the Gaussian-kernel unity TU.
set_source_files_properties(aux_coulomb.cc orbital_coulomb.cc harmonic_transform.cc PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
set_source_files_properties(native_response.cc PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
psi4_add_module(lib isapol sources)

target_link_libraries(isapol
  PRIVATE
    fock
    Libint2::cxx
  )

# aux_coulomb.cc and orbital_coulomb.cc include the Libint2 engine and each take ~4.3 GiB
# to compile, so throttle them like l2intf (libmints) to avoid OOM with Ninja parallelism.
cmake_host_system_information(RESULT _numcore QUERY NUMBER_OF_PHYSICAL_CORES)
if(_numcore LESS 5 AND UNIX AND NOT APPLE)
  set_property(TARGET isapol PROPERTY JOB_POOL_COMPILE one_job)
endif()

# This module exists to reproduce another code's arithmetic bit for bit, so the
# compiler must not be free to re-associate it.  Without this, GCC contracts
# `r*x + c` (isa_grid.cc) into an FMA on any host that has one, which shifts
# grid coordinates by ~1 ulp relative to the CamCASP reference and makes our own
# results depend on the -march the user happened to build with.
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-ffp-contract=off" HAVE_FFP_CONTRACT_OFF)
if(HAVE_FFP_CONTRACT_OFF)
  target_compile_options(isapol PRIVATE -ffp-contract=off)
endif()
