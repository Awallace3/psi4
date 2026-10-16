/*
 * @BEGIN LICENSE
 *
 * Psi4: an open-source quantum chemistry software package
 *
 * Copyright (c) 2007-2026 The Psi4 Developers.
 *
 * The copyrights for code used from other parties are included in
 * the corresponding files.
 *
 * This file is part of Psi4.
 *
 * Psi4 is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * Psi4 is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License along
 * with Psi4; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * @END LICENSE
 */

#ifndef PSI4_LIBFOCK_CUEST_XC_RESPONSE_H
#define PSI4_LIBFOCK_CUEST_XC_RESPONSE_H

#include <cstddef>
#include <cstdint>
#include <string>

#ifdef USING_cuEST
#include <cuest.h>
#endif

namespace psi {

#ifdef USING_cuEST
/// Grow-only device allocation. Never shrinks within a V object's lifetime, and frees
/// without profiling or throwing, so it is safe to release during stack unwinding.
class CuestDeviceArray {
   public:
    CuestDeviceArray() = default;
    CuestDeviceArray(const CuestDeviceArray&) = delete;
    CuestDeviceArray& operator=(const CuestDeviceArray&) = delete;
    ~CuestDeviceArray() { release(); }

    /// A device pointer with room for at least `bytes`, or null for zero bytes.
    void* ensure_bytes(std::size_t bytes, const char* what);
    double* ensure(std::size_t count, const char* what) {
        return static_cast<double*>(ensure_bytes(count * sizeof(double), what));
    }
    std::size_t capacity_bytes() const { return capacity_; }
    void release();

   private:
    void* pointer_ = nullptr;
    std::size_t capacity_ = 0;
};

/// Device and SDK scratch of the cuEST-grid XC response kernel, owned by one VBase.
///
/// Allocation capacity survives between calls. On the CUDA LibXC route the weights,
/// reference density and derivative fields also survive, and are reused while the key
/// below matches: the V's reference epoch (bumped by set_D, set_Cocc and initialize),
/// the spin integration, the point count and the functional's components. The host
/// LibXC route keeps its fields in host memory only for one call, so it always
/// recomputes. VBase::finalize destroys this before the XC integral plan and grid it
/// was used with, so a grid change can never reuse it.
struct CuestXCResponseWorkspace {
    CuestDeviceArray weights;
    CuestDeviceArray reference_a, reference_b;
    // Derivative fields: RKS (A, B, C, S) or polarized LibXC blocks.
    CuestDeviceArray kernel_A, kernel_B, kernel_C, kernel_S;
    CuestDeviceArray v2rho2, v2rhosigma, v2sigma2, vsigma;
    // Per-component LibXC scratch and inputs.
    CuestDeviceArray xc_rho, xc_sigma, t_vrho, t_vsigma, t_v2rho2, t_v2rhosigma, t_v2sigma2;
    CuestDeviceArray t_A, t_B, t_C, t_S;
    // Per-right-hand-side fields.
    CuestDeviceArray response_a, response_b, tile;
    CuestDeviceArray potential_a, potential_b;
    CuestDeviceArray left, right, ao;
    CuestDeviceArray counter;
    // SDK temporary workspace, reused while large enough.
    CuestDeviceArray sdk_device;
    void* sdk_host = nullptr;
    std::size_t sdk_host_capacity = 0;
    cuestWorkspace_t sdk_view{};
    // Key of the reference state held in weights, reference_a/b and the derivative fields.
    bool reference_valid = false;
    std::uint64_t reference_epoch = 0;
    int reference_spin = -1;
    std::size_t reference_npoints = 0;
    std::string reference_functional;

    CuestXCResponseWorkspace() = default;
    CuestXCResponseWorkspace(const CuestXCResponseWorkspace&) = delete;
    CuestXCResponseWorkspace& operator=(const CuestXCResponseWorkspace&) = delete;
    ~CuestXCResponseWorkspace();

    /// A temporary workspace view at least as large as the descriptor requests.
    cuestWorkspace_t* sdk_workspace(const cuestWorkspaceDescriptor_t& descriptor);
};
#endif

}  // namespace psi

#endif
