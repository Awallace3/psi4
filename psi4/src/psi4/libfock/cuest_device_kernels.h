/* Device-only algebra shared by the resident CPHF/CPKS implementation.
 * SPDX-License-Identifier: LGPL-3.0-only
 */
#ifndef PSI4_CUEST_DEVICE_KERNELS_H
#define PSI4_CUEST_DEVICE_KERNELS_H

#ifdef PSI4_CUEST_DEVICE_CG
#include <cstddef>
#include <memory>

namespace psi {
class CuestDeviceKernels {
   public:
    static CuestDeviceKernels& instance();
    template <typename... Args>
    void launch(const char* name, std::size_t count, Args... args) {
        if (!count) return;
        void* parameters[] = {static_cast<void*>(&args)...};
        launch_impl(name, count, parameters);
    }
    ~CuestDeviceKernels();
   private:
    CuestDeviceKernels();
    struct Impl;
    std::unique_ptr<Impl> impl_;
    void launch_impl(const char*, std::size_t, void**);
};
}  // namespace psi
#endif
#endif
