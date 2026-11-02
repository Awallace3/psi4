list(APPEND sources
  becke_roussel.cc
  hirshfeld.cc
  xdm.cc
  xdm_data.cc
  xdm_data_ftot.cc
  )

# XDM_GPU: Becke-Roussel, Hirshfeld and moment kernels. Rides on the cuEST switch,
# which owns the CUDA runtime and device initialization.
if(TARGET cuEST::cuEST)
  enable_language(CUDA)
  list(APPEND sources xdm_gpu.cu)
endif()

psi4_add_module(lib xdm sources)
