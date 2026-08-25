if(${ENABLE_GTFock})
    # GTFock has no CMake package config and no conda-forge package yet, so it
    # can only be detected externally: it is built out-of-tree by gtfock_psi4's
    # `build_deps.sh` (IntelLLVM icx/icpx + OpenMPI + Simint). Detect it here in
    # the superbuild so a missing or misplaced install fails at configure time
    # with an actionable message instead of at psi4-core link time.
    find_package(GTFock MODULE REQUIRED)

    message(STATUS "${Cyan}Found GTFock${ColourReset}: ${GTFock_PFOCK_LIBRARY}")
    add_library(gtfock_external INTERFACE)  # dummy

    # Hand psi4-core the resolved location so it repeats this detection without
    # re-searching CMAKE_PREFIX_PATH.
    get_filename_component(_gtfock_libdir "${GTFock_PFOCK_LIBRARY}" DIRECTORY)
    get_filename_component(_gtfock_prefix "${_gtfock_libdir}" DIRECTORY)
    set(GTFock_ROOT ${_gtfock_prefix} CACHE PATH "path to the externally built GTFock install prefix" FORCE)
else()
    add_library(gtfock_external INTERFACE)  # dummy
endif()
