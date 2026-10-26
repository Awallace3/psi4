/* SPDX-License-Identifier: LGPL-3.0-only */
#ifndef PSI4_CUEST_RESPONSE_BATCH_H
#define PSI4_CUEST_RESPONSE_BATCH_H
#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace psi {
namespace cuest_response {
struct DensityBatch {
    std::size_t first;
    std::size_t count;
};

/// Layout is [spin][storage_rhs][AO][AO]. A short final tile retains that
/// pitch, so its two spin panels must not be mistaken for one dense batch.
/// Build these schedules once; the response iteration does not allocate them.
inline std::vector<DensityBatch> density_batches(std::size_t storage_rhs, std::size_t actual_rhs,
                                                std::size_t nspin, std::size_t batch_size) {
    if (!storage_rhs || !actual_rhs || actual_rhs > storage_rhs ||
        !nspin || nspin > 2 || !batch_size)
        throw std::invalid_argument("Invalid cuEST response density batch shape.");
    if (storage_rhs > std::numeric_limits<std::size_t>::max() / nspin)
        throw std::overflow_error("cuEST response density batch shape overflow.");
    std::vector<DensityBatch> batches;
    const auto append = [&](std::size_t first, std::size_t count) {
        for (std::size_t offset = 0; offset < count;) {
            const auto size = std::min(batch_size, count - offset);
            batches.push_back({first + offset, size});
            offset += size;
        }
    };
    if (actual_rhs == storage_rhs) {
        append(0, storage_rhs * nspin);
    } else {
        for (std::size_t spin = 0; spin < nspin; ++spin)
            append(spin * storage_rhs, actual_rhs);
    }
    return batches;
}
}  // namespace cuest_response
}  // namespace psi
#endif
