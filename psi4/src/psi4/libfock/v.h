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
// The interface to cuEST was contributed by NVIDIA under the following terms:
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LGPL-3.0-only

#ifndef LIBFOCK_DFT_H
#define LIBFOCK_DFT_H
#include "psi4/libmints/typedefs.h"
#include "psi4/pragma.h"
#include "psi4/libpsi4util/memory_ledger.h"
#include <vector>
#include <map>
#include <memory>
#include <unordered_map>
#include <string>
#include <vector>

#ifdef USING_cuEST
#include <cuest.h>
#endif

namespace psi {
class BasisSet;
class Options;
class DFTGrid;
class PointFunctions;
class SuperFunctional;
class BlockOPoints;
struct CuestXCResponseWorkspace;
class CuestXCDeviceResponse;

/// One XC response density: Dx = left * right^T when both factors are set, else dense.
struct VxDensity {
    SharedMatrix left;
    SharedMatrix right;
    SharedMatrix dense;
};

/// Spin integration of an XC response build, in the conventions of RV::compute_Vx_full
/// (half-alpha output; triplet means beta = -alpha) and UV::compute_Vx (alpha/beta pairs).
enum class VxSpin { RKSSinglet, RKSTriplet, UKS };

// => BASE CLASS <= //

/**
 * Class VBase
 *
 * Class to compute KS-V matrices and
 * K-matrix-vector products
 **/

class PSI_API VBase {
   protected:
    /// Debug flag
    int debug_;
    /// Print flag
    int print_;
    /// Number of threads
    int num_threads_;
    /// Number of basis functions;
    int nbf_;
    /// Rho threshold for the second derivative;
    double v2_rho_cutoff_;
    /// VV10 interior kernel threshold
    double vv10_rho_cutoff_;
    /// Options object, used to build grid
    Options& options_;
    /// Basis set used in the integration
    std::shared_ptr<BasisSet> primary_;
    /// Desired superfunctional kernel
    std::shared_ptr<SuperFunctional> functional_;
    /// Desired superfunctional kernel
    std::vector<std::shared_ptr<SuperFunctional>> functional_workers_;
    /// Point function computer (densities, gammas, basis values)
    std::vector<std::shared_ptr<PointFunctions>> point_workers_;
    /// Integration grid, built by KSPotential
    std::shared_ptr<DFTGrid> grid_;
    /// Only SCF potentials opt into the initial grid; response grids stay final.
    bool initial_grid_ = false;
    /// VV10 Grid
    std::shared_ptr<DFTGrid> vv10_grid_;
#ifdef USING_cuEST
    /// XC Integral Plan
    cuestXCIntPlan_t cuest_xcint_plan_ = nullptr;
    /// XC Integral Plan Persistent Workspace
    cuestWorkspace_t *cuest_xcint_ws_ptr_ = nullptr;
    /// VV10 Integral Plan
    cuestXCIntPlan_t cuest_vv10_xcint_plan_ = nullptr;
    /// VV10 Integral Plan Persistent Workspace
    cuestWorkspace_t *cuest_vv10_xcint_ws_ptr_ = nullptr;
    /// Vector of GPU matrices of occupied orbitals
    double* d_Coccs_AO_ = nullptr;
    std::vector<uint64_t> d_Cocc_noccs_;
    /// Host copies of the factors last given to set_Cocc. The XC response uses the
    /// device copies for its reference density only once these reproduce D_AO_.
    std::vector<SharedMatrix> cuest_Cocc_;
    /// -1 unknown, 0 no, 1 yes. Reset by every set_D and set_Cocc.
    int cuest_Cocc_matches_D_ = -1;
    /// Scratch of the cuEST-grid XC response kernel; released before the plan/grid.
    std::unique_ptr<CuestXCResponseWorkspace> cuest_response_;
    /// Bumped by set_D, set_Cocc and initialize. The XC response reuses its reference density and
    /// derivative kernel only while this is unchanged (grid changes also pass through finalize,
    /// which destroys cuest_response_).
    uint64_t cuest_reference_epoch_ = 0;
#endif
    /// Quadrature values obtained during integration
    std::map<std::string, double> quad_values_;
    // Caches collocation grids
    std::unordered_map<size_t, std::map<std::string, SharedMatrix>> cache_map_;
    int cache_map_deriv_;
    /// What the cache above is costing the process right now, reported so that an SCF
    /// started while this V is alive is not handed the same memory a second time.
    MemoryClaim cache_claim_;

    /// AO2USO matrix (if not C1)
    SharedMatrix AO2USO_;
    SharedMatrix USO2AO_;

    /// Vector of C1 D matrices (built by USO2AO)
    std::vector<SharedMatrix> D_AO_;

    // GRAC data
    bool grac_initialized_;

    // VV10 dispersion, return vv10_nlc energy
    void prepare_vv10_cache(DFTGrid& nlgrid, SharedMatrix D,
                            std::vector<std::map<std::string, SharedVector>>& vv10_cache,
                            std::vector<std::shared_ptr<PointFunctions>>& nl_point_workers, int ansatz = 1);
    double vv10_nlc(SharedMatrix D, SharedMatrix ret);
    SharedMatrix vv10_nlc_gradient(SharedMatrix D);

    /// Set things up
    void common_init();
    /// Destructor cleanup is unprofiled and best-effort; explicit cleanup is checked.
    void finalize_impl(bool checked);

    /// Whether the exchange-correlation quadrature runs on the GPU.
    ///
    /// True only when USE_CUEST and CUEST_XC are both set.  USE_CUEST alone
    /// selects the cuEST density-fitted J/K builder; CUEST_XC additionally
    /// hands the XC grid to cuEST.  Splitting the two lets a calculation keep
    /// GPU J/K -- the dominant cost -- while running XC on the CPU, which is
    /// the only way to reach XC features cuEST has no branch for (SAP,
    /// meta-GGA/VV10 response kernels, analytic Hessians). Restricted GRAC potentials
    /// are evaluated on the host grid arrays and integrated by cuEST.
    ///
    /// Every site that reads this must agree with DFTGrid::buildGridFromOptions
    /// in cubature.cc, which uses the same conjunction to decide whether to
    /// build the GPU grid or a CPU-blocked one.  A disagreement means either an
    /// empty grid_->blocks() on the CPU path or a missing grid_->cuest_grid()
    /// on the GPU path.
    bool use_cuest_xc() const;

    /// Guard for CPU-only XC routines that cuEST has no branch for.
    ///
    /// When the cuEST XC path is active, the DFT quadrature grid is built on
    /// the GPU and MolecularGrid's CPU-side post-processing is skipped entirely
    /// (cubature.cc: "if (is_cuest) return;" before postProcess()), so
    /// grid_->blocks() is empty and functional_workers_ is never populated.
    /// Any CPU XC routine reached under cuEST would therefore either index an
    /// empty vector (segfault) or loop over zero blocks and silently return a
    /// zero XC contribution.  Routines without a cuEST implementation must call
    /// this to fail loudly instead.  \p what names the operation for the user.
    /// Setting CUEST_XC false restores the CPU XC path and clears the guard
    /// without giving up cuEST J/K.
    void throw_if_cuest_unsupported(const std::string& what) const;

    /// LDA/GGA XC response kernel on the cuEST grid (cuest_xc_response.cc). ret[i] receives
    /// the response to inputs[i] in the representation of ret[i]; UKS inputs are alpha/beta
    /// pairs [a0, b0, a1, b1, ...]. Throws the unsupported diagnostic when the linked cuEST
    /// SDK has no nonsymmetric XC density.
    void compute_Vx_cuest(const std::vector<VxDensity>& inputs, std::vector<SharedMatrix>& ret, VxSpin spin);
    /// Whether the last set_Cocc factors reproduce the current D_AO_ (cached until either changes).
    bool cuest_Cocc_reproduces_D();

   public:
    VBase(std::shared_ptr<SuperFunctional> functional, std::shared_ptr<BasisSet> primary, Options& options);
    virtual ~VBase();

    static std::shared_ptr<VBase> build_V(std::shared_ptr<BasisSet> primary,
                                          std::shared_ptr<SuperFunctional> functional, Options& options,
                                          const std::string& type = "RV");

    std::shared_ptr<BasisSet> basis() const { return primary_; }
    std::shared_ptr<SuperFunctional> functional() const { return functional_; }
    std::vector<std::shared_ptr<PointFunctions>> properties() const { return point_workers_; }
    std::shared_ptr<DFTGrid> grid() const { return grid_; }
    std::shared_ptr<BlockOPoints> get_block(int block);
    size_t nblocks();
    std::map<std::string, double>& quadrature_values() { return quad_values_; }

    // Creates a collocation cache map based on stride
    void build_collocation_cache(size_t memory);
    void clear_collocation_cache();

    // Set the D matrix, get it back if needed
    void set_D(std::vector<SharedMatrix> Dvec);
    void set_Cocc(std::vector<SharedMatrix> Cocc);
    const std::vector<SharedMatrix>& Dao() const { return D_AO_; }

    // Set the site of the grac shift
    void set_grac_shift(double value);

    /// Throws by default
    virtual void compute_V(std::vector<SharedMatrix> ret);
    /// Throws by default. Compute the orbital derivative of the KS potential for each spin,
    /// contract against Dx, and putting the result in ret.
    virtual void compute_Vx(const std::vector<SharedMatrix> Dx, std::vector<SharedMatrix> ret);
    /// compute_Vx for response densities given as factors, Dx[i] = left[i] * right[i]^T, in the
    /// representation of Matrix::doublet(left[i], right[i], false, true). ret[i] must have that
    /// product's dimensions and symmetry. The factors are only borrowed for the call.
    /// The base implementation forms each product and calls compute_Vx.
    virtual void compute_Vx_factored(const std::vector<SharedMatrix>& left, const std::vector<SharedMatrix>& right,
                                     std::vector<SharedMatrix> ret);
    virtual std::vector<SharedMatrix> compute_fock_derivatives();
    virtual SharedMatrix compute_gradient();
    virtual SharedMatrix compute_hessian();

    void set_print(int print) { print_ = print; }
    void set_debug(int debug) { debug_ = debug; }

    /// Whether this potential's XC response kernel runs on the cuEST grid. True when it was
    /// initialized on a cuEST grid, whatever USE_CUEST/CUEST_XC say now (another potential
    /// may have changed them since), and also when they now request cuEST for a potential
    /// initialized on the CPU grid, so that compute_Vx raises instead of mixing grids.
    bool uses_cuest_xc() const {
#ifdef USING_cuEST
        if (cuest_xcint_plan_) return true;
#endif
        return use_cuest_xc();
    }
#ifdef USING_cuEST
    std::unique_ptr<CuestXCDeviceResponse> prepare_device_response(const std::vector<std::size_t>& nocc);
#endif

    virtual void initialize();
    virtual void finalize();
    /// Rebuild the grid, workers and GPU integral plans without changing the functional.
    void set_grid(const std::string& name);
    bool initial_grid() const { return initial_grid_; }

    virtual void print_header() const;
};

// => Derived Classes <= //
class SAP : public VBase {
   protected:
   public:
    SAP(std::shared_ptr<SuperFunctional> functional, std::shared_ptr<BasisSet> primary, Options& options);
    ~SAP() override;

    void initialize() override;
    void finalize() override;

    void compute_V(std::vector<SharedMatrix> ret) override;
    void print_header() const override;
};

class RV : public VBase {
   protected:
   public:
    RV(std::shared_ptr<SuperFunctional> functional, std::shared_ptr<BasisSet> primary, Options& options);
    ~RV() override;

    void initialize() override;
    void finalize() override;

    // compute_V assuming same orbitals for different spin. Computes V_alpha, not spin-summed V.
    void compute_V(std::vector<SharedMatrix> ret) override;
    /// Compute the orbital derivative of the KS potential, contract against Dx, and
    /// putting the result in ret. ret[i] is Vx where x = Dx[i]. The "true" vector has
    /// 2^-0.5 Dx[i] for each input spin case and returns **half** the α component of the output.
    /// The singlet flag controls whether to assume singlet spin-integration (β components
    /// are the α components) or triplet (β components are -α components)
    void compute_Vx_full(const std::vector<SharedMatrix> Dx, std::vector<SharedMatrix> ret, bool singlet);
    /// A convenience function to call compute_Vx_full for singlets.
    /// And no, we can't just make singlet a default argument. Then compute_Vx has different signatures for
    /// different VBase subclasses, so we can't call compute_Vx from VBase, which breaks the hessian code.
    void compute_Vx(const std::vector<SharedMatrix> Dx, std::vector<SharedMatrix> ret) override { compute_Vx_full(Dx, ret, true); };
    /// compute_Vx_full for Dx[i] = left[i] * right[i]^T. On the cuEST grid, C1 factors are
    /// contracted directly; elsewhere the product is formed and compute_Vx_full is called.
    void compute_Vx_full_factored(const std::vector<SharedMatrix>& left, const std::vector<SharedMatrix>& right,
                                  std::vector<SharedMatrix> ret, bool singlet);
    void compute_Vx_factored(const std::vector<SharedMatrix>& left, const std::vector<SharedMatrix>& right,
                             std::vector<SharedMatrix> ret) override {
        compute_Vx_full_factored(left, right, ret, true);
    }
    std::vector<SharedMatrix> compute_fock_derivatives() override;
    SharedMatrix compute_gradient() override;
    SharedMatrix compute_hessian() override;

    void print_header() const override;
};

class UV : public VBase {
   protected:
   public:
    UV(std::shared_ptr<SuperFunctional> functional, std::shared_ptr<BasisSet> primary, Options& options);
    ~UV() override;

    void initialize() override;
    void finalize() override;

    void compute_V(std::vector<SharedMatrix> ret) override;
    /// Compute the orbital derivative of the KS potential, contract against Dx, and
    /// putting the result in ret. ret[i] is Vx where x = Dx[i].
    /// ret[2n], ret[2n+1] are alpha and beta Vx where x concatenates Dx[2n] (α) and Dx[2n+1] (β).
    void compute_Vx(const std::vector<SharedMatrix> Dx, std::vector<SharedMatrix> ret) override;
    /// Factored compute_Vx; left/right follow the same [a0, b0, a1, b1, ...] pair order.
    void compute_Vx_factored(const std::vector<SharedMatrix>& left, const std::vector<SharedMatrix>& right,
                             std::vector<SharedMatrix> ret) override;
    std::vector<SharedMatrix> compute_fock_derivatives() override;
    SharedMatrix compute_gradient() override;
    SharedMatrix compute_hessian() override;

    void print_header() const override;
};
}
#endif
