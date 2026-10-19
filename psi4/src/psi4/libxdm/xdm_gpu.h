/*
 * @BEGIN LICENSE
 *
 * Psi4: an open-source quantum chemistry software package
 *
 * Copyright (c) 2007-2026 The Psi4 Developers.
 *
 * This file is part of Psi4.
 *
 * Psi4 is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * @END LICENSE
 */

// Device side of the XDM moment integration (option XDM_GPU).
//
// The host builds the DFT grid (the same grid, blocks and basis-function
// screening as the host path) and hands the blocks over. Everything per point
// runs on the device: basis-function values with first and second
// derivatives, the spin densities, gradients, kinetic-energy densities and
// Laplacians, the exchange-hole curvature, the Becke-Roussel b parameter, the
// Hirshfeld promolecule and the five Hirshfeld-weighted moment integrals per
// atom. Consecutive blocks are batched into chunks whose collocation and
// density-GEMM buffers fit a fixed device workspace, so device memory does not
// grow with the grid.

#ifndef PSI4_SRC_PSI4_LIBXDM_XDM_GPU_H
#define PSI4_SRC_PSI4_LIBXDM_XDM_GPU_H

#ifdef USING_cuEST

#include <array>
#include <cstddef>
#include <memory>
#include <vector>

namespace psi {
namespace xdm {

class ProatomDensity;

/// Host description of the primary basis, as gau2grid sees it in BasisFunctions.
struct XDMDeviceBasis {
    int nbf = 0;
    bool puream = false;
    std::vector<double> center;  ///< [nshell][3], bohr
    std::vector<int> am;
    std::vector<int> nprim;
    std::vector<int> prim_offset;  ///< into exps/coefs
    std::vector<int> func_offset;  ///< first AO index of the shell
    std::vector<int> nfunc;
    std::vector<double> exps;
    std::vector<double> coefs;  ///< GaussianShell::coefs(): normalized contraction coefficients
    /// gau2grid pure-function order of the host collocation (GG_SPHERICAL_*), when puream.
    int pure_order = 0;
    int nshell() const { return static_cast<int>(am.size()); }
};

/// Row-major (2L+1) x (L+1)(L+2)/2 transform from CCA-ordered Cartesian components to
/// pure functions in gau2grid order ``order``, read off gau2grid itself.
std::vector<double> xdm_pure_transform(int L, int order);

class DeviceXDMIntegrator {
   public:
    /// @param atomic_nums      true atomic numbers [natom]; ghosts included, Z < 1 rows get no moments
    /// @param atom_coords      [natom] coordinates in bohr
    /// @param Da, Db           row-major nbf x nbf AO spin density matrices, read until finish();
    ///                         Db == nullptr means restricted (beta = alpha)
    /// @param workspace_bytes  device budget for the per-chunk collocation, GEMM and density-matrix
    ///                         buffers; raised to the minimum that holds one grid block if smaller
    DeviceXDMIntegrator(int natom, const int* atomic_nums, const std::array<double, 3>* atom_coords,
                        const ProatomDensity& proatom, const XDMDeviceBasis& basis, const double* Da, const double* Db,
                        std::size_t workspace_bytes);
    ~DeviceXDMIntegrator();
    DeviceXDMIntegrator(const DeviceXDMIntegrator&) = delete;
    DeviceXDMIntegrator& operator=(const DeviceXDMIntegrator&) = delete;

    /// Queue one grid block: its points and weights, and the global indices of the shells
    /// the host screening keeps for it (BlockOPoints::shells_local_to_global).
    void add_block(std::size_t npoints, const double* x, const double* y, const double* z, const double* w,
                   const std::vector<int>& shells);

    /// Per-atom integrals, in this order.
    enum Moment { MM1, MM2, MM3, VOLUME, CHARGE, NMOMENT };

    /// Flush, wait for the device and return the integrated moments, [natom][NMOMENT].
    /// nbhole_failed/total count Becke-Roussel solves exactly as the host loop does.
    std::vector<std::array<double, NMOMENT>> finish(std::size_t& nbhole_failed, std::size_t& nbhole_total);

    /// Peak device bytes this integrator allocated.
    std::size_t device_bytes() const;
    /// Growth of the process's device memory (cudaMemGetInfo) from construction to its
    /// peak, sampled once per chunk: this integrator's buffers plus anything cuBLAS adds.
    std::size_t device_growth() const;
    /// Device memory already in use when the integrator was built (the SCF's, the context).
    std::size_t device_in_use_at_start() const;
    /// Points processed so far.
    std::size_t npoints() const;
    /// Chunks sent to the device so far.
    std::size_t nchunks() const;

   private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace xdm
}  // namespace psi

#endif  // USING_cuEST
#endif
