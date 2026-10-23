list(APPEND sources
  CDJK.cc
  CompositeJK.cc
  cuESTJK.cc
  cuest_device_response.cc
  cuest_xc_response.cc
  COSK.cc
  DirectDFJ.cc
  DirectJK.cc
  DiskDFJK.cc
  DiskJK.cc
  GTFockJK.cc
  LinK.cc
  MemDFJK.cc
  PKJK.cc
  PK_workers.cc
  PKmanagers.cc
  SplitJK.cc
  apps.cc
  cubature.cc
  hamiltonian.cc
  jk.cc
  points.cc
  sap.cc
  snLinK.cc
  solver.cc
  soscf.cc
  v.cc
  wrapper.cc
  )

if(ENABLE_cuEST AND TARGET CUDA::nvrtc AND TARGET CUDA::cuda_driver)
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/cuest_xc_response_math.h")
  file(READ "${CMAKE_CURRENT_SOURCE_DIR}/cuest_xc_response_math.h" CUEST_RESPONSE_MATH_SOURCE)
  configure_file(cuest_device_math_source.h.in cuest_device_math_source.h @ONLY)
  list(APPEND sources cuest_device_kernels.cc)
endif()

if(Libxc_ENABLE_CUDA AND TARGET cuEST::cuEST)
  list(APPEND sources cuest_xc_gpu.cu)
endif()

psi4_add_module(lib fock sources)
if (psi4_ENABLE_PRECOMPILE_HEADERS)
    target_precompile_headers(fock REUSE_FROM psio)
endif()

# example for turning off unity build for a directory while developint
# set_property(TARGET fock PROPERTY UNITY_BUILD OFF)

target_link_libraries(fock
  PRIVATE
    pybind11::headers
  )
if(ENABLE_cuEST AND TARGET CUDA::nvrtc AND TARGET CUDA::cuda_driver)
  target_compile_definitions(fock PRIVATE PSI4_CUEST_DEVICE_CG)
  target_include_directories(fock PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
  target_link_libraries(fock PRIVATE CUDA::nvrtc CUDA::cuda_driver)
endif()
