# Register in the superbuild; psi4-core is a separate ExternalProject.
ExternalProject_Get_Property(psi4-core BINARY_DIR)
add_test(NAME dfhelper-jk-blocking
         COMMAND ${BINARY_DIR}/src/psi4/lib3index/test/dfhelper_jk_blocking_test.x${CMAKE_EXECUTABLE_SUFFIX})
set_tests_properties(dfhelper-jk-blocking PROPERTIES
                     LABELS "psi;quicktests;smoketests;df;scf"
                     TIMEOUT 10)
