# Create interface library for required dependencies
add_library(psi4_required_deps INTERFACE)

target_link_libraries(psi4_required_deps
  INTERFACE
    tgt::lapack     # Provides OpenMP flags and BLAS/LAPACK headers
    Libint2::cxx    # Required for integral engine
    Libxc::xc       # Required for DFT functionals; target provides USING_Libxc
    gau2grid::gg    # Required for DFT grid generation
)

set (CMAKE_OPTIMIZE_DEPENDENCIES ON)

# Create interface library for common optional dependencies to avoid repeating if(TARGET...) blocks
add_library(psi4_optional_deps INTERFACE)

if(TARGET gauxc::gauxc)
  target_link_libraries(psi4_optional_deps INTERFACE gauxc::gauxc)
  target_compile_definitions(psi4_optional_deps INTERFACE USING_gauxc)
endif()

if(TARGET ECPINT::ecpint)
  target_link_libraries(psi4_optional_deps INTERFACE ECPINT::ecpint)
  target_compile_definitions(psi4_optional_deps INTERFACE USING_ecpint)
endif()

if(TARGET OpenOrbitalOptimizer::OpenOrbitalOptimizer)
  # Note: USING_OpenOrbitalOptimizer is NOT added to psi4_optional_deps because:
  # 1. USING_OpenOrbitalOptimizer causes armadillo headers to be included in matrix.h
  # 2. Armadillo declares BLAS functions which conflict with libqt's BLAS declarations
  # 3. Only libmints, libscf_solver, scfgrad actually need OpenOrbitalOptimizer
  # Solution: libmints adds USING_ as PUBLIC, which keeps it away from upstream modules like libqt
  target_link_libraries(psi4_optional_deps INTERFACE OpenOrbitalOptimizer::OpenOrbitalOptimizer)
endif()

if(TARGET BrianQC::static_wrapper)
  target_link_libraries(psi4_optional_deps INTERFACE BrianQC::static_wrapper)
  target_compile_definitions(psi4_optional_deps INTERFACE USING_BrianQC)
endif()

if(TARGET IntegratorXX::IntegratorXX)
  target_link_libraries(psi4_optional_deps INTERFACE IntegratorXX::IntegratorXX)
  target_compile_definitions(psi4_optional_deps INTERFACE USING_IntegratorXX)
endif()

if(TARGET simint::simint)
  target_link_libraries(psi4_optional_deps INTERFACE simint::simint)
  target_compile_definitions(psi4_optional_deps INTERFACE USING_simint)
endif()

if(TARGET dkh::dkh)
  target_link_libraries(psi4_optional_deps INTERFACE dkh::dkh)
  # USING_dkh provided by dkh::dkh target itself
endif()

if(TARGET PCMSolver::pcm)
  target_link_libraries(psi4_optional_deps INTERFACE PCMSolver::pcm)
  # USING_PCMSolver provided by PCMSolver::pcm target itself
endif()

if(TARGET CheMPS2::chemps2)
  target_link_libraries(psi4_optional_deps INTERFACE CheMPS2::chemps2)
  # USING_CheMPS2 provided by CheMPS2::chemps2 target itself
endif()

if(TARGET ambit::ambit)
  target_link_libraries(psi4_optional_deps INTERFACE ambit::ambit)
  # USING_ambit provided by ambit::ambit target itself
endif()

if(TARGET Einsums::Einsums)
  target_link_libraries(psi4_optional_deps INTERFACE Einsums::Einsums)
  target_compile_definitions(psi4_optional_deps INTERFACE USING_Einsums)
endif()

if(TARGET cuEST::cuEST)
  target_link_libraries(psi4_optional_deps INTERFACE cuEST::cuEST)
  target_compile_definitions(psi4_optional_deps INTERFACE USING_cuEST)

  # cuEST adds members to public Psi4 classes, so plugins must use the same ABI.
  target_compile_definitions(core INTERFACE USING_cuEST)

  # Grid-K exchange and Direct-J Coulomb use the cuEST 0.3 entry points;
  # probe their exact signatures rather than header presence. An SDK that
  # fails a probe still provides DF-J/DF-K, and requesting the missing mode
  # raises at run time.
  include(CheckCXXSourceCompiles)
  set(CMAKE_REQUIRED_LIBRARIES cuEST::cuEST)
  check_cxx_source_compiles("
#include <cuest.h>
#include <type_traits>
using Create = cuestStatus_t (*)(cuestHandle_t, cuestMolecularGrid_t, cuestOEIntPlan_t,
    cuestGridKIntPlanParameters_t, const cuestWorkspaceDescriptor_t*,
    cuestWorkspace_t*, cuestWorkspace_t*, cuestGridKIntPlan_t*);
using Query = cuestStatus_t (*)(cuestHandle_t, cuestMolecularGrid_t, cuestOEIntPlan_t,
    cuestGridKIntPlanParameters_t, const cuestWorkspaceDescriptor_t*,
    cuestWorkspaceDescriptor_t*, cuestWorkspaceDescriptor_t*, cuestGridKIntPlan_t*);
static_assert(std::is_same<decltype(&cuestGridKIntPlanCreate), Create>::value, \"Grid-K create API\");
static_assert(std::is_same<decltype(&cuestGridKIntPlanCreateWorkspaceQuery), Query>::value, \"Grid-K query API\");
int main() {
    auto sym = &cuestGridKSymmetricExchangeCompute;
    auto nonsym = &cuestGridKNonsymmetricExchangeCompute;
    const auto fit = CUEST_GRIDKINTPLAN_PARAMETERS_FITTING_ALGORITHM_OVERLAP_SYMMETRIC;
    const auto nmo = CUEST_GRIDKINTPLAN_FITTING_NUM_MO;
    return (sym && nonsym && fit >= 0 && nmo >= 0) ? 0 : 1;
}
" PSI4_CUEST_HAS_GRIDK_API)
  check_cxx_source_compiles("
#include <cuest.h>
int main() {
    auto create = &cuestDirectIntPlanCreate;
    auto compute = &cuestDirectCoulombCompute;
    auto query = &cuestDirectCoulombComputeWorkspaceQuery;
    const auto threshold = CUEST_DIRECTCOULOMBCOMPUTE_PARAMETERS_DENSITY_SCHWARZ_THRESHOLD;
    const auto jit = CUEST_DIRECTCOULOMBCOMPUTE_PARAMETERS_JIT_USAGE_MODE;
    return (create && compute && query && threshold >= 0 && jit >= 0) ? 0 : 1;
}
" PSI4_CUEST_HAS_DIRECTJ_API)
  unset(CMAKE_REQUIRED_LIBRARIES)
  foreach(_mode GRIDK DIRECTJ)
    if(PSI4_CUEST_HAS_${_mode}_API)
      target_compile_definitions(psi4_optional_deps INTERFACE PSI_CUEST_HAS_${_mode})
      # These switch cuESTJK class members, so plugins must see them too.
      target_compile_definitions(core INTERFACE PSI_CUEST_HAS_${_mode})
      message(STATUS "cuEST ${_mode} API found: CUEST_${_mode} enabled")
    else()
      message(STATUS "cuEST ${_mode} API not found: CUEST_${_mode} unavailable; DF-JK remains available")
    endif()
  endforeach()
endif()

foreach(dir_name
  lib3index
  libciomr
  libcubeprop
  libdiis
  libdisp
  libdpd
  libfock
  libfunctional
  libiwl
  libmints
  libmoinfo
  liboptions
  libplugin
  libpsi4util
  libpsio
  libpsipcm
  libqt
  libsapt_solver
  libscf_solver
  libtrans
  )
  add_subdirectory(${dir_name})
endforeach ()

foreach(dir_name
  ambit_interface
  cc
  cctransort
  dct
  detci
  dfep2
  dfmp2
  dfocc
  dlpno
  dmrg
  fisapt
  fnocc
  mcscf
  mrcc
  occ
  psimrcc
  sapt
  scfgrad
  f12
  dummy_einsums
  dummy_integratorxx
  )
  add_subdirectory(${dir_name})
endforeach ()
