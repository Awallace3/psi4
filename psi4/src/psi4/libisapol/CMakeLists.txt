list(APPEND sources
  explicit_basis.cc
  partitioned_response.cc
  isa_grid.cc
  tables.cc
  )

psi4_add_module(lib isapol sources)

target_link_libraries(isapol
  PRIVATE
    fock
  )

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
