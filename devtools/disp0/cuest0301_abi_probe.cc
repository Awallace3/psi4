// Compile-only ABI checks for the SAPT factory/DFMO ctypes boundary.
// No library loading, GPU execution, or numerical qualification occurs here.
// Pointer constness/pointee types are erased to match ctypes' void-pointer
// representation; enum width, scalar widths, counts and workspace offsets
// remain checked. Pointer ownership and array layouts require runtime tests.
#include <cuest.h>
#include <cstddef>
#include <cstdint>
#include <type_traits>

template<class T, bool E = std::is_enum_v<T>, bool P = std::is_pointer_v<T>>
struct abi_type { using type = T; };
template<class T> struct abi_type<T, true, false> {
    static_assert(sizeof(T) == sizeof(int));
    using type = int;
};
template<class T> struct abi_type<T, false, true> { using type = void*; };
template<class T> using abi_t = typename abi_type<T>::type;
template<class T> struct abi_signature;
template<class R, class... A> struct abi_signature<R (*)(A...)> {
    using type = abi_t<R> (*)(abi_t<A>...);
};
using H = void*;
using U = uint64_t;
#define ABI(name, ...) \
    static_assert(std::is_same_v<typename abi_signature<decltype(&name)>::type, \
                                 int (*)(__VA_ARGS__)>, #name)
ABI(cuestGetMajorVersion, H,H);
ABI(cuestGetMinorVersion, H,H);
ABI(cuestGetPatchVersion, H,H);
ABI(cuestGetMathMode, H,H);
ABI(cuestParametersCreate, int,H);
ABI(cuestParametersDestroy, int,H);
ABI(cuestDFMOIntegralsComputeWorkspaceQuery, H,H,H,H,H,U,H,H,H,H,H);
ABI(cuestDFMOIntegralsCompute, H,H,H,H,H,U,H,H,H,H,H);
ABI(cuestParametersConfigure, int,H,int,H,size_t);
ABI(cuestParametersQuery, int,H,int,H,size_t);
ABI(cuestCreate, H,H);
ABI(cuestDestroy, H);
ABI(cuestSetMathMode, H,int);
ABI(cuestQuery, H,int,H,int,H,size_t);
ABI(cuestAOShellCreate, H,int32_t,U,U,H,H,H,H);
ABI(cuestAOShellDestroy, H);
ABI(cuestAOBasisCreateWorkspaceQuery, H,U,H,H,H,H,H,H);
ABI(cuestAOBasisCreate, H,U,H,H,H,H,H,H);
ABI(cuestAOBasisDestroy, H);
ABI(cuestAOPairListCreateWorkspaceQuery, H,H,U,H,double,H,H,H,H);
ABI(cuestAOPairListCreate, H,H,U,H,double,H,H,H,H);
ABI(cuestAOPairListDestroy, H);
ABI(cuestDFIntPlanCreateWorkspaceQuery, H,H,H,H,H,H,H,H);
ABI(cuestDFIntPlanCreate, H,H,H,H,H,H,H,H);
ABI(cuestDFIntPlanDestroy, H);
#undef ABI

static_assert(sizeof(void*) == 8 && sizeof(size_t) == 8 && sizeof(uintptr_t) == 8);
static_assert(sizeof(int) == 4 && sizeof(uint32_t) == 4 && sizeof(uint64_t) == 8);
static_assert(sizeof(cuestWorkspaceDescriptor_t) == 16);
static_assert(offsetof(cuestWorkspaceDescriptor_t, hostBufferSizeInBytes) == 0);
static_assert(offsetof(cuestWorkspaceDescriptor_t, deviceBufferSizeInBytes) == 8);
static_assert(sizeof(cuestWorkspace_t) == 32);
static_assert(offsetof(cuestWorkspace_t, hostBuffer) == 0);
static_assert(offsetof(cuestWorkspace_t, hostBufferSizeInBytes) == 8);
static_assert(offsetof(cuestWorkspace_t, deviceBuffer) == 16);
static_assert(offsetof(cuestWorkspace_t, deviceBufferSizeInBytes) == 24);

#define VALUE(name, number) static_assert(name == number, #name)
VALUE(CUEST_NATIVE_FP64_MATH_MODE, 1);
VALUE(CUEST_HANDLE_PARAMETERS, 0);
VALUE(CUEST_AOSHELL_PARAMETERS, 1);
VALUE(CUEST_AOBASIS_PARAMETERS, 2);
VALUE(CUEST_AOPAIRLIST_PARAMETERS, 3);
VALUE(CUEST_DFINTPLAN_PARAMETERS, 5);
VALUE(CUEST_DFMOINTEGRALSCOMPUTE_PARAMETERS, 49);
VALUE(CUEST_AOBASIS, 2);
VALUE(CUEST_AOBASIS_IS_PURE, 0);
VALUE(CUEST_AOBASIS_IS_CART, 1);
VALUE(CUEST_AOBASIS_IS_MIXED, 2);
VALUE(CUEST_AOBASIS_NUM_ATOM, 3);
VALUE(CUEST_AOBASIS_NUM_SHELL, 4);
VALUE(CUEST_AOBASIS_NUM_AO, 5);
VALUE(CUEST_AOBASIS_NUM_CART, 6);
VALUE(CUEST_AOBASIS_NUM_PRIMITIVE, 7);
VALUE(CUEST_AOBASIS_MAX_L, 8);
VALUE(CUEST_DFINTPLAN_PARAMETERS_FITTING_CUTOFF, 0);
VALUE(CUEST_DFINTPLAN_PARAMETERS_FITTING_RELATIVE_CONDITIONING, 1);
VALUE(CUEST_DFINTPLAN_PARAMETERS_EXCHANGE_FRACTION, 2);
VALUE(CUEST_DFINTPLAN_PARAMETERS_FITTING_ALGORITHM, 3);
VALUE(CUEST_DFINTPLAN_PARAMETERS_LRC_EXCHANGE_FRACTION, 4);
VALUE(CUEST_DFINTPLAN_PARAMETERS_LRC_EXCHANGE_OMEGA, 5);
VALUE(CUEST_DFINTPLAN_PARAMETERS_THREE_INDEX_INTEGRAL_DIRECT, 6);
VALUE(CUEST_DFINTPLAN_PARAMETERS_FITTING_ALGORITHM_MATRIXPOWER, 1);
VALUE(CUEST_DFINTPLAN_PARAMETERS_CACHE_OPERATOR_WEIGHTING, 7);
VALUE(CUEST_DFINTPLAN_PARAMETERS_CACHE_OPERATOR_WEIGHTING_FULL, 0);
VALUE(CUEST_DFMOINTEGRALSCOMPUTE_PARAMETERS_OPERATOR_WEIGHTING, 0);
VALUE(CUEST_DFMOINTEGRALSCOMPUTE_PARAMETERS_OPERATOR_WEIGHTING_FULL, 0);
VALUE(CUEST_DFINTPLAN_PARAMETERS_DETAIL_THREE_INDEX_FITTING_FIXEDPOINT_EMULATION_USAGE_MODE, 8);
VALUE(CUEST_EMULATION_USAGE_MODE_OFF, 1);
#undef VALUE
