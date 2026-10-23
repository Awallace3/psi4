/* Resident response operators. SPDX-License-Identifier: LGPL-3.0-only */
#ifndef PSI4_CUEST_DEVICE_RESPONSE_H
#define PSI4_CUEST_DEVICE_RESPONSE_H
#ifdef USING_cuEST
#include "cuESTJK.h"
#include "cuest_xc_response.h"
#include <array>
#include <memory>
#include <vector>
namespace psi {
class VBase;

/// All AO arrays are spin-major, then RHS-major, row-major nbf x nbf.
/// Right factors are [rhs, occupied, nbf]; left factors are [occupied, nbf].
/// Borrowed buffers must remain valid throughout the session.
struct CuestDeviceJKView {
    std::size_t nrhs = 0;
    std::vector<std::size_t> nocc;
    std::array<const double*, 2> left{}, right{};
    const double* density = nullptr;
    double* coulomb = nullptr;
    double* exchange = nullptr;
    bool compute_exchange = false;
    /// Consumer requires only K+K^T, not general K. Allows a representative
    /// K((D+D^T)/2) with the same Hermitian sum for symmetric Grid-K.
    bool exchange_sum_only = false;
};

/// Query/allocate once, then compute entirely between device buffers.
class CuestJKDeviceResponse {
   public:
    CuestJKDeviceResponse(cuESTJK& jk, CuestDeviceJKView view);
    void compute();
    std::size_t device_workspace_bytes() const { return scratch_.deviceBufferSizeInBytes; }
    std::size_t density_batch_size() const { return batch_size_; }
    bool symmetric_gridk() const { return symmetric_gridk_; }
   private:
    cuESTJK& jk_;
    CuestDeviceJKView view_;
    std::size_t batch_size_ = 0;
    bool symmetric_gridk_ = false;
    std::string plan_key_;
    CuestDeviceArray symmetric_density_;
    CuestXCResponseWorkspace workspace_;
    cuestWorkspace_t* work_ = nullptr;
    cuestWorkspaceDescriptor_t scratch_{}, variable_{};
};

/// One fixed reference XC kernel; physical reference preparation happens once.
/// apply() only reads/writes device buffers, never host matrices/grid fields.
class CuestXCDeviceResponse {
   public:
    ~CuestXCDeviceResponse();
    void prepare(const std::array<const double*, 2>& left,
                 const std::array<const double*, 2>& right,
                 const std::array<double*, 2>& potential);
    void apply(const std::array<const double*, 2>& left,
               const std::array<const double*, 2>& right,
               const std::array<double*, 2>& potential);
   private:
    friend class VBase;
    struct Impl;
    explicit CuestXCDeviceResponse(std::unique_ptr<Impl>);
    std::unique_ptr<Impl> impl_;
};
}  // namespace psi
#endif
#endif
