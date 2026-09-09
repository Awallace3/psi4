list(APPEND sources
  native_response.cc
  point_response.cc
  casimir_grid.cc
  pfit.cc
  fit_points.cc
  explicit_basis.cc
  partitioned_response.cc
  isotropic_dispersion.cc
  anisotropic_dispersion.cc
  multipole_transform.cc
  t_functions.cc
  lw_localization.cc
  aux_coulomb.cc
  orbital_coulomb.cc
  ov_fit.cc
  isa_grid.cc
  isa_fit.cc
  isa_sweep.cc
  isa_shape.cc
  isa_controller.cc
  recoupling_tables.cc
  realcg_tables.cc
  recoupled_dispersion.cc
  tables.cc
  )

# Keep the heavy Libint2 engine implementation out of the Gaussian-kernel unity TU.
set_source_files_properties(aux_coulomb.cc orbital_coulomb.cc PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
set_source_files_properties(native_response.cc point_response.cc PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
psi4_add_module(lib isapol sources)

target_link_libraries(isapol
  PRIVATE
    fock
    Libint2::cxx
  )

# This module exists to reproduce another code's arithmetic bit for bit, so the
# compiler must not be free to re-associate it.  Without this, GCC contracts
# `r*x + c` (isa_grid.cc) into an FMA on any host that has one, which shifts
# grid coordinates by ~1 ulp relative to the CamCASP reference and makes our own
# results depend on the -march the user happened to build with.  See SPEC.md 3.5.5.
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-ffp-contract=off" HAVE_FFP_CONTRACT_OFF)
if(HAVE_FFP_CONTRACT_OFF)
  target_compile_options(isapol PRIVATE -ffp-contract=off)
endif()
