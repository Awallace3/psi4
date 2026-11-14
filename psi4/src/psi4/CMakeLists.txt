# Create interface library for required dependencies
add_library(psi4_required_deps INTERFACE)

target_link_libraries(psi4_required_deps
  INTERFACE
    tgt::lapack     # Provides OpenMP flags and BLAS/LAPACK headers
    Libint2::cxx    # Required for integral engine
    Libxc::xc       # Required for DFT functionals; target provides USING_Libxc
    gau2grid::gg    # Required for DFT grid generation
)

if(Libxc_ENABLE_CUDA)
  if(Libxc_VERSION VERSION_LESS 7.1.0)
    message(FATAL_ERROR "CUDA LibXC integration requires LibXC 7.1.0 or newer for explicit host/device dispatch")
  endif()
  enable_language(CUDA)
  target_compile_definitions(psi4_required_deps INTERFACE USING_Libxc_CUDA)
endif()

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

  # The XC response kernel needs the nonsymmetric XC density (cuEST 0.2.2). Require the
  # declaration, the parameter type and a resolvable symbol, so an older SDK keeps the
  # explicit unsupported diagnostic instead of failing to link or silently changing grids.
  # The define changes no class layout.
  include(CheckCXXSourceCompiles)
  set(CMAKE_REQUIRED_LIBRARIES cuEST::cuEST)
  check_cxx_source_compiles("
#include <cuest.h>
int main() {
    cuestXCNonsymmetricDensityComputeParameters_t parameters{};
    (void)parameters;
    const auto type = CUEST_XCNONSYMMETRICDENSITYCOMPUTE_PARAMETERS;
    auto compute = &cuestXCNonsymmetricDensityCompute;
    auto query = &cuestXCNonsymmetricDensityComputeWorkspaceQuery;
    return (compute && query && type) ? 0 : 1;
}" PSI4_CUEST_HAS_NONSYMMETRIC_XC)
  # Header guards alone cannot identify these optional preview APIs: cuEST
  # 0.3 retains the headers but changes Grid-K construction and Direct-J
  # screening. Keep DF-JK available; reject optional modes until their actual
  # API is supported, rather than guessing argument or threshold semantics.
  check_cxx_source_compiles("
#include <cuest.h>
#include <type_traits>
using Create = cuestStatus_t (*)(cuestHandle_t, cuestAOBasis_t,
    cuestAOPairList_t, cuestMolecularGrid_t, cuestGridKIntPlanParameters_t,
    cuestWorkspace_t*, cuestWorkspace_t*, cuestGridKIntPlan_t*);
using Query = cuestStatus_t (*)(cuestHandle_t, cuestAOBasis_t,
    cuestAOPairList_t, cuestMolecularGrid_t, cuestGridKIntPlanParameters_t,
    cuestWorkspaceDescriptor_t*, cuestWorkspaceDescriptor_t*, cuestGridKIntPlan_t*);
static_assert(std::is_same<decltype(&cuestGridKIntPlanCreate), Create>::value, \"Grid-K create API\");
static_assert(std::is_same<decltype(&cuestGridKIntPlanCreateWorkspaceQuery), Query>::value, \"Grid-K query API\");
int main() { return 0; }
" PSI4_CUEST_HAS_PREVIEW_GRIDK)
  check_cxx_source_compiles("
#include <cuest.h>
int main() {
    auto create = &cuestDirectIntPlanCreate;
    auto compute = &cuestDirectCoulombCompute;
    auto query = &cuestDirectCoulombComputeWorkspaceQuery;
    const auto threshold = CUEST_DIRECTCOULOMBCOMPUTE_PARAMETERS_SCHWARZ_THRESHOLD;
    const auto jit = CUEST_DIRECTCOULOMBCOMPUTE_PARAMETERS_JIT_USAGE_MODE;
    return (create && compute && query && threshold >= 0 && jit >= 0) ? 0 : 1;
}
" PSI4_CUEST_HAS_PREVIEW_DIRECTJ)
  unset(CMAKE_REQUIRED_LIBRARIES)
  foreach(_mode GRIDK DIRECTJ)
    if(PSI4_CUEST_HAS_PREVIEW_${_mode})
      target_compile_definitions(psi4_optional_deps INTERFACE PSI_CUEST_HAS_${_mode})
      # These switches control cuESTJK class members, including for plugins.
      target_compile_definitions(core INTERFACE PSI_CUEST_HAS_${_mode})
      message(STATUS "cuEST ${_mode} preview API supported")
    else()
      message(STATUS "cuEST ${_mode} API unsupported: optional mode disabled; DF-JK remains available")
    endif()
  endforeach()
  if(PSI4_CUEST_HAS_NONSYMMETRIC_XC)
    target_compile_definitions(psi4_optional_deps INTERFACE PSI4_CUEST_NONSYMMETRIC_XC)
    message(STATUS "cuEST nonsymmetric XC density found: GPU XC response kernel enabled")
  else()
    message(STATUS "cuEST nonsymmetric XC density not found: GPU XC response kernel disabled")
  endif()
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
