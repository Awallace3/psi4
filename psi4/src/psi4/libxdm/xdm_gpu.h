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
#include <string>
#include <utility>
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

/// cuEST's points blocked on the device (XDM_GRID_BLOCKING DEVICE). The host build's two
/// sieves are applied, the kept points are sorted along a Morton curve and cut at octant
/// boundaries into blocks of at most max_points, and each block's shells are screened with
/// BlockOPoints' bound()/populate() test. The points stay on the device, block after block.
class DeviceGridBlocks {
   public:
    /// @param d_xyz, d_w       cuEST's interleaved points and total weights (cudaMalloc'd);
    ///                         taken over and freed as soon as the sorted copy exists
    /// @param weights_cutoff   DFT_WEIGHTS_TOLERANCE: points with |w| below it are dropped
    /// @param max_distance     with DFT_REMOVE_DISTANT_POINTS, BasisExtents::maxR(): points
    ///                         farther than this from every atom are dropped; < 0 keeps them
    /// @param shell_extents    BasisExtents::shell_extents(), one radius per shell
    DeviceGridBlocks(double* d_xyz, double* d_w, std::size_t npoints, double weights_cutoff, double max_distance,
                     int natom, const std::array<double, 3>* atom_coords, int max_points,
                     const XDMDeviceBasis& basis, const std::vector<double>& shell_extents);
    ~DeviceGridBlocks();
    DeviceGridBlocks(const DeviceGridBlocks&) = delete;
    DeviceGridBlocks& operator=(const DeviceGridBlocks&) = delete;

    /// Points kept by the sieves (all of them sit in some block).
    std::size_t npoints() const;
    /// Blocks, including those no shell reaches (their shell list is empty).
    std::size_t nblocks() const;
    /// Blocks with at least one shell: the ones that are integrated.
    std::size_t nblocks_with_shells() const;
    const std::vector<std::size_t>& block_start() const;
    const std::vector<std::size_t>& block_npoints() const;
    /// Per-block significant shells, CSR: shells()[shell_offset()[b] .. shell_offset()[b + 1]).
    const std::vector<int>& shell_offset() const;
    const std::vector<int>& shells() const;
    /// Device arrays of the sorted kept points, [npoints()] each.
    const double* x() const;
    const double* y() const;
    const double* z() const;
    const double* w() const;
    /// The sorted points on the host, for DFTGrid::adopt_host_blocks().
    void copy_to_host(std::vector<double>& x, std::vector<double>& y, std::vector<double>& z,
                      std::vector<double>& w) const;
    /// Peak device bytes held while blocking (cuEST's arrays included) and bytes held afterwards.
    std::size_t peak_device_bytes() const;
    std::size_t resident_device_bytes() const;

   private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class DeviceXDMIntegrator {
   public:
    /// @param atomic_nums      true atomic numbers [natom]; ghosts included, Z < 1 rows get no moments
    /// @param atom_coords      [natom] coordinates in bohr
    /// @param Da, Db           row-major nbf x nbf AO spin density matrices, read until finish();
    ///                         Db == nullptr means restricted (beta = alpha)
    /// @param workspace_bytes  device budget for the per-chunk collocation, GEMM and density-matrix
    ///                         buffers; raised to the minimum that holds one grid block if smaller
    /// @param resident_density allow the full density matrices to live on the device (taken out of
    ///                         the budget when they fit in half of it); false gathers them on the host
    /// @param gemm_ratio       close a chunk before a block that would make its GEMM (over the union of
    ///                         its blocks' columns) exceed this multiple of the blocks' own GEMMs; 0 = off
    DeviceXDMIntegrator(int natom, const int* atomic_nums, const std::array<double, 3>* atom_coords,
                        const ProatomDensity& proatom, const XDMDeviceBasis& basis, const double* Da, const double* Db,
                        std::size_t workspace_bytes, bool resident_density = true, double gemm_ratio = 0.0);
    ~DeviceXDMIntegrator();
    DeviceXDMIntegrator(const DeviceXDMIntegrator&) = delete;
    DeviceXDMIntegrator& operator=(const DeviceXDMIntegrator&) = delete;

    /// Queue one grid block: its points and weights, and the global indices of the shells
    /// the host screening keeps for it (BlockOPoints::shells_local_to_global).
    void add_block(std::size_t npoints, const double* x, const double* y, const double* z, const double* w,
                   const std::vector<int>& shells);

    /// Queue every block of a device-blocked grid. Its points are read in place on the device,
    /// so nothing is staged through the host; blocks without shells are skipped, as the host
    /// blockers drop them. Do not mix with add_block() in one integration.
    void add_device_blocks(const DeviceGridBlocks& grid);

    /// Whether the full density matrices live on the device (gathered per chunk there) rather
    /// than being gathered on the host for every chunk. Decided at construction: when allowed,
    /// they are resident if they fit in half of the workspace budget, which they are taken out of.
    bool density_resident() const;

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
    /// Device time per stage of the chunk pipeline, in seconds, from events on the stream
    /// (valid after finish()): copies, density gather, collocation, GEMM, density,
    /// Becke-Roussel + promolecule, moments.
    std::vector<std::pair<std::string, double>> stage_seconds() const;
    /// GEMM multiply-adds {done, needed}: done over each chunk's union columns, needed over
    /// each grid block's own columns.
    std::array<double, 2> gemm_work() const;

   private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace xdm
}  // namespace psi

#endif  // USING_cuEST
#endif
