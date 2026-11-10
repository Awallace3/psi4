/* SPDX-License-Identifier: LGPL-3.0-only */
#ifndef PSI4_CUEST_WORKSPACE_POLICY_H
#define PSI4_CUEST_WORKSPACE_POLICY_H
#include <cstddef>
#include <limits>
#include <stdexcept>
namespace psi {
namespace cuest_workspace {
/// cuEST uses decimal bytes, as does CUEST_XC_RESPONSE_BUDGET_MB.
/// This bounds only the variable Grid-K scratch, not fixed SDK overhead,
/// Direct-J scratch, plans, AO buffers, or the complete device footprint.
inline std::size_t gridk_variable_bytes(int megabytes) {
    constexpr std::size_t mb = 1000000;
    if (megabytes <= 0)
        throw std::invalid_argument("CUEST_GRIDK_WORKSPACE_MB must be positive.");
    if (static_cast<std::size_t>(megabytes) > std::numeric_limits<std::size_t>::max() / mb)
        throw std::overflow_error("CUEST_GRIDK_WORKSPACE_MB exceeds the platform byte-size limit.");
    return static_cast<std::size_t>(megabytes) * mb;
}
}  // namespace cuest_workspace
}  // namespace psi
#endif
