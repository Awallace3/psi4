/*
 * @BEGIN LICENSE
 *
 * Psi4: an open-source quantum chemistry software package
 *
 * Copyright (c) 2007-2025 The Psi4 Developers.
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

#include "cuESTJK.h"

#ifdef USING_cuEST

#include "cuESTCommon.h"

#include "psi4/libfock/cubature.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/integral.h"
#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libpsi4util/process.h"
#include "psi4/liboptions/liboptions.h"
#include "psi4/psi4-dec.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cublas_v2.h>
#include <cusolverDn.h>

extern cublasHandle_t cublas_handle;

namespace psi {

static bool uses_overlap_fit(const std::string& k_build) {
    return k_build == "GRID" || k_build == "GRID_FIT";
}

cuESTJK::cuESTJK(std::shared_ptr<BasisSet> primary, std::shared_ptr<BasisSet> auxiliary, Options& options)
    : JK(primary),
      options_(options),
      auxiliary_(auxiliary),
      condition_(1.0E-12),
      pq_threshold_(1.0E-14),
      cuest_primary_basis_(nullptr),
      cuest_auxiliary_basis_(nullptr),
      cuest_pair_list_(nullptr),
      cuest_df_plan_(nullptr),
      cuest_coulomb_compute_params_(nullptr),
      cuest_exchange_compute_params_(nullptr),
      initialized_(false) 
{
    cuest_common::ensure_cuest_initialized();

    cuest_primary_basis_ = primary->cuest_basis();

    condition_ = options.get_double("DF_FITTING_CONDITION");
    pq_threshold_ = options.get_double("INTS_TOLERANCE");
    dfk_slices_ = options.get_int("CUEST_DFK_SLICES");
    dfk_moduli_ = options.get_int("CUEST_DFK_MODULI");
}

void cuESTJK::prepare_auxiliary_basis() {
    // Only a tasked DF builder needs a spherical-harmonic auxiliary basis.
    if (!auxiliary_->has_puream()) {
        throw PSIEXCEPTION("cuESTJK requires a spherical-harmonic auxiliary (DF) basis set; "
                           "the auxiliary basis set '" + auxiliary_->name() +
                           "' is Cartesian. Set DF_BASIS_SCF/DF_BASIS_MP2 to a basis with spherical "
                           "harmonics, or add 'puream true' scoped to the auxiliary basis if supported; "
                           "the orbital (primary) basis may remain Cartesian.");
    }

    cuest_auxiliary_basis_ = auxiliary_->cuest_basis();
}

cuESTJK::~cuESTJK() {
    destroy_cuest_objects();
}

void cuESTJK::preiterations()
{
    gridk_ = options_.get_str("CUEST_K_BUILD") != "DF";
    directj_ = options_.get_str("CUEST_J_BUILD") == "DIRECT";
#ifndef PSI_CUEST_HAS_GRIDK
    if (gridk_)
        throw PSIEXCEPTION("CUEST_K_BUILD GRID requires cuEST's Grid-K API (cuEST >= 0.3), which the cuEST this "
                           "Psi4 was built against does not provide. DF-K remains available.");
#endif
#ifndef PSI_CUEST_HAS_DIRECTJ
    if (directj_)
        throw PSIEXCEPTION("CUEST_J_BUILD DIRECT requires cuEST's Direct-J API (cuEST >= 0.3), which the cuEST this "
                           "Psi4 was built against does not provide. DF-J remains available.");
#endif
    if (needs_df_plan()) prepare_auxiliary_basis();
    if (directj_ && do_J_ && !(directj_density_threshold() >= 0.0))
        throw PSIEXCEPTION("CUEST_DIRECTJ_DENSITY_THRESHOLD must be nonnegative.");
    if (gridk_ && do_K_) {
        (void)gridk_workspace_bytes();  // Validate before allocating any plan.
        if (!(options_.get_double("CUEST_GRIDK_COLLOCATION_THRESHOLD") >= 0.0))
            throw PSIEXCEPTION("CUEST_GRIDK_COLLOCATION_THRESHOLD must be nonnegative.");
        if (uses_overlap_fit(options_.get_str("CUEST_K_BUILD"))) {
            const auto cutoff = options_.get_double("CUEST_GRIDK_FIT_CUTOFF");
            if (!(cutoff > 0.0 && cutoff < 1.0)) throw PSIEXCEPTION("CUEST_GRIDK_FIT_CUTOFF must be in (0, 1).");
            if (!(options_.get_double("CUEST_GRIDK_FIT_ORTHO_CUTOFF") > 0.0))
                throw PSIEXCEPTION("CUEST_GRIDK_FIT_ORTHO_CUTOFF must be positive.");
        }
    }

    auto mol = primary_->molecule();
    size_t natom = mol->natom();

    cuestWorkspaceDescriptor_t* persistentWorkspaceDescriptor = (cuestWorkspaceDescriptor_t*) malloc(sizeof(cuestWorkspaceDescriptor_t));
    cuestWorkspaceDescriptor_t* temporaryWorkspaceDescriptor = (cuestWorkspaceDescriptor_t*) malloc(sizeof(cuestWorkspaceDescriptor_t));

    // AO Pair List
    cuestAOPairListParameters_t pair_params;
    CHECK_CUEST(cuestParametersCreate(CUEST_AOPAIRLIST_PARAMETERS, reinterpret_cast<void**>(&pair_params)));

    CHECK_CUEST(cuestAOPairListCreateWorkspaceQuery(cuest_handle, cuest_primary_basis_,
        static_cast<uint64_t>(natom), mol->geometry().pointer()[0], pq_threshold_, pair_params, persistentWorkspaceDescriptor, temporaryWorkspaceDescriptor, nullptr));

    cuest_pair_list_ws_ptr_ = cuest_common::allocateWorkspace(persistentWorkspaceDescriptor);
    cuestWorkspace_t* temporaryPairListWorkspace = cuest_common::allocateWorkspace(temporaryWorkspaceDescriptor);

    CHECK_CUEST(cuestAOPairListCreate(cuest_handle, cuest_primary_basis_,
        static_cast<uint64_t>(natom), mol->geometry().pointer()[0], pq_threshold_, pair_params, cuest_pair_list_ws_ptr_, temporaryPairListWorkspace, &cuest_pair_list_));

    cuest_common::freeWorkspace(temporaryPairListWorkspace);
    CHECK_CUEST(cuestParametersDestroy(CUEST_AOPAIRLIST_PARAMETERS, pair_params));

    // DF Int Plan List. Direct-J + Grid-K needs no DF plan at all; the auxiliary
    // basis is then unused.
    if (needs_df_plan()) {
    cuestDFIntPlanParameters_t dfint_params;
    CHECK_CUEST(cuestParametersCreate(CUEST_DFINTPLAN_PARAMETERS, reinterpret_cast<void**>(&dfint_params)));
    
    // Psi4 and cuEST both represent exact exchange as alpha / r + beta erf(omega r) / r.
    double exchange_fraction = omega_alpha_;
    double lrc_exchange_fraction = omega_beta_;
    double lr_exchange_omega = omega_;

    //Set omega params if applicable 
    CHECK_CUEST(cuestParametersConfigure(
        CUEST_DFINTPLAN_PARAMETERS, 
        dfint_params,
        CUEST_DFINTPLAN_PARAMETERS_EXCHANGE_FRACTION,
        &exchange_fraction,
        sizeof(double)));

    CHECK_CUEST(cuestParametersConfigure(
        CUEST_DFINTPLAN_PARAMETERS, 
        dfint_params,
        CUEST_DFINTPLAN_PARAMETERS_LRC_EXCHANGE_FRACTION,
        &lrc_exchange_fraction,
        sizeof(double)));

    CHECK_CUEST(cuestParametersConfigure(
        CUEST_DFINTPLAN_PARAMETERS, 
        dfint_params,
        CUEST_DFINTPLAN_PARAMETERS_LRC_EXCHANGE_OMEGA,
        &lr_exchange_omega,
        sizeof(double)));

    CHECK_CUEST(cuestDFIntPlanCreateWorkspaceQuery(cuest_handle,
        cuest_primary_basis_, cuest_auxiliary_basis_, cuest_pair_list_,
        dfint_params, persistentWorkspaceDescriptor, temporaryWorkspaceDescriptor, nullptr));

    cuest_dfint_plan_ws_ptr_ = cuest_common::allocateWorkspace(persistentWorkspaceDescriptor);;
    cuestWorkspace_t* temporaryDFIntPlanWorkspace = cuest_common::allocateWorkspace(temporaryWorkspaceDescriptor);;

    CHECK_CUEST(cuestDFIntPlanCreate(cuest_handle,
        cuest_primary_basis_, cuest_auxiliary_basis_, cuest_pair_list_,
        dfint_params, cuest_dfint_plan_ws_ptr_, temporaryDFIntPlanWorkspace, &cuest_df_plan_));

    cuest_common::freeWorkspace(temporaryDFIntPlanWorkspace);
    CHECK_CUEST(cuestParametersDestroy(CUEST_DFINTPLAN_PARAMETERS, dfint_params));

    // Create J/K parameters
    CHECK_CUEST(cuestParametersCreate(CUEST_DFCOULOMBCOMPUTE_PARAMETERS, reinterpret_cast<void**>(&cuest_coulomb_compute_params_)));
    CHECK_CUEST(cuestParametersCreate(CUEST_DFSYMMETRICEXCHANGECOMPUTE_PARAMETERS, reinterpret_cast<void**>(&cuest_exchange_compute_params_)));

    // Set J & K compute parameters
    CHECK_CUEST(cuestParametersConfigure(
        CUEST_DFSYMMETRICEXCHANGECOMPUTE_PARAMETERS,
        cuest_exchange_compute_params_,
        CUEST_DFSYMMETRICEXCHANGECOMPUTE_PARAMETERS_INT8_SLICE_COUNT,
        &dfk_slices_,
        sizeof(uint64_t)));

    CHECK_CUEST(cuestParametersConfigure(
        CUEST_DFSYMMETRICEXCHANGECOMPUTE_PARAMETERS, 
        cuest_exchange_compute_params_,
        CUEST_DFSYMMETRICEXCHANGECOMPUTE_PARAMETERS_INT8_MODULUS_COUNT,
        &dfk_moduli_,
        sizeof(uint64_t)));
    }  // needs_df_plan()

    free(persistentWorkspaceDescriptor);
    free(temporaryWorkspaceDescriptor);

#ifdef PSI_CUEST_HAS_DIRECTJ
    if (directj_ && do_J_) build_directj_plan();
#endif
#ifdef PSI_CUEST_HAS_GRIDK
    if (gridk_ && do_K_) build_gridk_plan();
#endif

    // Set global math mode, if CUEST_NATIVE_FP64_MATH_MODE, all forms of mixed precision emulation will be turned off
    if (!options_.get_bool("CUEST_MIXED_PRECISION")) {
        CHECK_CUEST(cuestSetMathMode(
            cuest_handle,
            CUEST_NATIVE_FP64_MATH_MODE
        ));
    }

    initialized_ = true;
}

void cuESTJK::destroy_cuest_objects() {
#ifdef PSI_CUEST_HAS_GRIDK
    if (cuest_gridk_exchange_params_) {
        CHECK_CUEST(cuestParametersDestroy(CUEST_GRIDKSYMMETRICEXCHANGECOMPUTE_PARAMETERS,
                                           cuest_gridk_exchange_params_));
        cuest_gridk_exchange_params_ = nullptr;
    }
    if (cuest_gridk_nonsym_exchange_params_) {
        CHECK_CUEST(cuestParametersDestroy(CUEST_GRIDKNONSYMMETRICEXCHANGECOMPUTE_PARAMETERS,
                                           cuest_gridk_nonsym_exchange_params_));
        cuest_gridk_nonsym_exchange_params_ = nullptr;
    }
    if (cuest_gridk_plan_) {
        CHECK_CUEST(cuestGridKIntPlanDestroy(cuest_gridk_plan_));
        cuest_gridk_plan_ = nullptr;
    }
    if (cuest_gridk_plan_ws_) {
        cuest_common::freeWorkspace(cuest_gridk_plan_ws_);
        cuest_gridk_plan_ws_ = nullptr;
    }
    if (cuest_gridk_oeint_plan_) {
        CHECK_CUEST(cuestOEIntPlanDestroy(cuest_gridk_oeint_plan_));
        cuest_gridk_oeint_plan_ = nullptr;
    }
    if (cuest_gridk_oeint_plan_ws_) {
        cuest_common::freeWorkspace(cuest_gridk_oeint_plan_ws_);
        cuest_gridk_oeint_plan_ws_ = nullptr;
    }
    gridk_grid_.reset();  // only after the plan that references it
#endif
#ifdef PSI_CUEST_HAS_DIRECTJ
    if (cuest_directj_params_) {
        CHECK_CUEST(cuestParametersDestroy(CUEST_DIRECTCOULOMBCOMPUTE_PARAMETERS, cuest_directj_params_));
        cuest_directj_params_ = nullptr;
    }
    if (cuest_directj_plan_) {
        CHECK_CUEST(cuestDirectIntPlanDestroy(cuest_directj_plan_));
        cuest_directj_plan_ = nullptr;
    }
    if (cuest_directj_plan_ws_) {
        cuest_common::freeWorkspace(cuest_directj_plan_ws_);
        cuest_directj_plan_ws_ = nullptr;
    }
#endif
    if (cuest_df_plan_) {
        CHECK_CUEST(cuestDFIntPlanDestroy(cuest_df_plan_));
        cuest_df_plan_ = nullptr;
    }
    if (cuest_pair_list_) {
        CHECK_CUEST(cuestAOPairListDestroy(cuest_pair_list_));
        cuest_pair_list_ = nullptr;
    }
    if (cuest_pair_list_ws_ptr_) {
        cuest_common::freeWorkspace(cuest_pair_list_ws_ptr_);
        cuest_pair_list_ws_ptr_ = nullptr;
    }
    if (cuest_dfint_plan_ws_ptr_) {
        cuest_common::freeWorkspace(cuest_dfint_plan_ws_ptr_);
        cuest_dfint_plan_ws_ptr_ = nullptr;
    }
    if (cuest_coulomb_compute_params_) {
        CHECK_CUEST(cuestParametersDestroy(CUEST_DFCOULOMBCOMPUTE_PARAMETERS, cuest_coulomb_compute_params_));
        cuest_coulomb_compute_params_ = nullptr;
    }
    if (cuest_exchange_compute_params_) {
        CHECK_CUEST(cuestParametersDestroy(CUEST_DFSYMMETRICEXCHANGECOMPUTE_PARAMETERS, cuest_exchange_compute_params_));
        cuest_exchange_compute_params_ = nullptr;
    }
}

double cuESTJK::directj_density_threshold() const {
    return options_["CUEST_DIRECTJ_DENSITY_THRESHOLD"].has_changed() ?
        options_.get_double("CUEST_DIRECTJ_DENSITY_THRESHOLD") : pq_threshold_;
}

size_t cuESTJK::gridk_workspace_bytes() const {
    const auto mb = options_.get_int("CUEST_GRIDK_WORKSPACE_MB");
    if (mb <= 0) throw PSIEXCEPTION("CUEST_GRIDK_WORKSPACE_MB must be positive.");
    return static_cast<size_t>(mb) * size_t(1000000);
}

void cuESTJK::query_coulomb_workspace(cuestWorkspaceDescriptor_t* desc, const double* density, double* coulomb) {
#ifdef PSI_CUEST_HAS_DIRECTJ
    if (directj_) {
        CHECK_CUEST(cuestDirectCoulombComputeWorkspaceQuery(cuest_handle, cuest_directj_plan_, cuest_directj_params_,
                                                            desc, 1, density, coulomb));
        return;
    }
#endif
    CHECK_CUEST(cuestDFCoulombComputeWorkspaceQuery(cuest_handle, cuest_df_plan_, cuest_coulomb_compute_params_, desc,
                                                    density, coulomb));
}

void cuESTJK::compute_coulomb(cuestWorkspace_t* scratch, const double* density, double* coulomb) {
#ifdef PSI_CUEST_HAS_DIRECTJ
    if (directj_) {
        // Four-center J sees only the symmetric part of D, so signed and
        // nonsymmetric response densities are passed through unchanged.
        CHECK_CUEST(cuestDirectCoulombCompute(cuest_handle, cuest_directj_plan_, cuest_directj_params_, scratch, 1,
                                              density, coulomb));
        return;
    }
#endif
    CHECK_CUEST(cuestDFCoulombCompute(cuest_handle, cuest_df_plan_, cuest_coulomb_compute_params_, scratch, density,
                                      coulomb));
}

namespace {
struct WorkspaceDeleter {
    void operator()(cuestWorkspace_t* ws) const { cuest_common::freeWorkspace(ws); }
};
using Workspace = std::unique_ptr<cuestWorkspace_t, WorkspaceDeleter>;
Workspace make_workspace(const cuestWorkspaceDescriptor_t& desc) {
    return Workspace(cuest_common::allocateWorkspace(&desc));
}
}  // namespace

#ifdef PSI_CUEST_HAS_DIRECTJ
void cuESTJK::build_directj_plan() {
    cuestDirectIntPlanParameters_t params = nullptr;
    CHECK_CUEST(cuestParametersCreate(CUEST_DIRECTINTPLAN_PARAMETERS, reinterpret_cast<void**>(&params)));
    cuestWorkspaceDescriptor_t persistent{}, temporary{};
    CHECK_CUEST(cuestDirectIntPlanCreateWorkspaceQuery(cuest_handle, cuest_primary_basis_, cuest_pair_list_, params,
                                                       &persistent, &temporary, nullptr));
    cuest_directj_plan_ws_ = cuest_common::allocateWorkspace(&persistent);
    {
        auto scratch = make_workspace(temporary);
        CHECK_CUEST(cuestDirectIntPlanCreate(cuest_handle, cuest_primary_basis_, cuest_pair_list_, params,
                                             cuest_directj_plan_ws_, scratch.get(), &cuest_directj_plan_));
    }
    CHECK_CUEST(cuestParametersDestroy(CUEST_DIRECTINTPLAN_PARAMETERS, params));

    CHECK_CUEST(cuestParametersCreate(CUEST_DIRECTCOULOMBCOMPUTE_PARAMETERS,
                                      reinterpret_cast<void**>(&cuest_directj_params_)));
    // DEFAULT keeps the SDK's own choice (JIT on CUDA 13+, ahead-of-time otherwise).
    const auto jit_policy = options_.get_str("CUEST_DIRECTJ_JIT");
    if (jit_policy != "DEFAULT") {
        const cuestJITUsageMode_t jit = jit_policy == "ON" ? CUEST_JIT_USAGE_MODE_ON : CUEST_JIT_USAGE_MODE_OFF;
        CHECK_CUEST(cuestParametersConfigure(CUEST_DIRECTCOULOMBCOMPUTE_PARAMETERS, cuest_directj_params_,
                                             CUEST_DIRECTCOULOMBCOMPUTE_PARAMETERS_JIT_USAGE_MODE, &jit,
                                             sizeof(jit)));
    }
    const double threshold = directj_density_threshold();
    CHECK_CUEST(cuestParametersConfigure(CUEST_DIRECTCOULOMBCOMPUTE_PARAMETERS, cuest_directj_params_,
                                         CUEST_DIRECTCOULOMBCOMPUTE_PARAMETERS_DENSITY_SCHWARZ_THRESHOLD, &threshold,
                                         sizeof(double)));
}
#endif

#ifdef PSI_CUEST_HAS_GRIDK
void cuESTJK::build_gridk_plan() {
    // The exchange quadrature follows the DFT_* grid options; it is a separate
    // GPU-resident grid from any XC grid.
    gridk_grid_ = std::make_unique<DFTGrid>(primary_->molecule(), primary_, options_, true);

    // Same operator convention as the DF plan: alpha / r + beta erf(omega r) / r.
    // A non-LRC functional still carries x_beta = 1 with omega = 0; do not hand
    // cuEST a long-range fraction without a range-separation parameter.
    double alpha = omega_alpha_;
    double beta = (omega_ == 0.0) ? 0.0 : omega_beta_;
    double omega = (beta == 0.0 && std::isfinite(omega_)) ? 0.0 : omega_;

    cuestGridKIntPlanParameters_t params = nullptr;
    CHECK_CUEST(cuestParametersCreate(CUEST_GRIDKINTPLAN_PARAMETERS, reinterpret_cast<void**>(&params)));
    const double collocation = options_.get_double("CUEST_GRIDK_COLLOCATION_THRESHOLD");
    CHECK_CUEST(cuestParametersConfigure(CUEST_GRIDKINTPLAN_PARAMETERS, params,
                                         CUEST_GRIDKINTPLAN_PARAMETERS_THRESHOLD_COLLOCATION, &collocation, sizeof(double)));
    CHECK_CUEST(cuestParametersConfigure(CUEST_GRIDKINTPLAN_PARAMETERS, params,
                                         CUEST_GRIDKINTPLAN_PARAMETERS_EXCHANGE_FRACTION, &alpha, sizeof(double)));
    CHECK_CUEST(cuestParametersConfigure(CUEST_GRIDKINTPLAN_PARAMETERS, params,
                                         CUEST_GRIDKINTPLAN_PARAMETERS_LRC_EXCHANGE_FRACTION, &beta, sizeof(double)));
    CHECK_CUEST(cuestParametersConfigure(CUEST_GRIDKINTPLAN_PARAMETERS, params,
                                         CUEST_GRIDKINTPLAN_PARAMETERS_LRC_EXCHANGE_OMEGA, &omega, sizeof(double)));

    // Overlap fit: dress the collocation so the quadrature reproduces the
    // analytic overlap on the retained subspace, removing most of the grid error.
    const bool fit = uses_overlap_fit(options_.get_str("CUEST_K_BUILD"));
    if (fit) {
        const cuestGridKIntPlanParametersFittingAlgorithm_t algorithm =
            CUEST_GRIDKINTPLAN_PARAMETERS_FITTING_ALGORITHM_OVERLAP_SYMMETRIC;
        const double ortho_cutoff = options_.get_double("CUEST_GRIDK_FIT_ORTHO_CUTOFF");
        const double fit_cutoff = options_.get_double("CUEST_GRIDK_FIT_CUTOFF");
        CHECK_CUEST(cuestParametersConfigure(CUEST_GRIDKINTPLAN_PARAMETERS, params,
                                             CUEST_GRIDKINTPLAN_PARAMETERS_FITTING_ALGORITHM, &algorithm,
                                             sizeof(algorithm)));
        CHECK_CUEST(cuestParametersConfigure(CUEST_GRIDKINTPLAN_PARAMETERS, params,
                                             CUEST_GRIDKINTPLAN_PARAMETERS_FITTING_ORTHOGONALIZATION_CUTOFF,
                                             &ortho_cutoff, sizeof(double)));
        CHECK_CUEST(cuestParametersConfigure(CUEST_GRIDKINTPLAN_PARAMETERS, params,
                                             CUEST_GRIDKINTPLAN_PARAMETERS_FITTING_CUTOFF, &fit_cutoff,
                                             sizeof(double)));
    }

    cuestWorkspaceDescriptor_t persistent{}, temporary{};
    {
        cuestOEIntPlanParameters_t oe_params = nullptr;
        CHECK_CUEST(cuestParametersCreate(CUEST_OEINTPLAN_PARAMETERS, reinterpret_cast<void**>(&oe_params)));
        CHECK_CUEST(cuestOEIntPlanCreateWorkspaceQuery(cuest_handle, cuest_primary_basis_, cuest_pair_list_,
                                                       oe_params, &persistent, &temporary, nullptr));
        cuest_gridk_oeint_plan_ws_ = cuest_common::allocateWorkspace(&persistent);
        auto scratch = make_workspace(temporary);
        CHECK_CUEST(cuestOEIntPlanCreate(cuest_handle, cuest_primary_basis_, cuest_pair_list_, oe_params,
                                         cuest_gridk_oeint_plan_ws_, scratch.get(), &cuest_gridk_oeint_plan_));
        CHECK_CUEST(cuestParametersDestroy(CUEST_OEINTPLAN_PARAMETERS, oe_params));
    }

    cuestWorkspaceDescriptor_t variable{};
    variable.deviceBufferSizeInBytes = gridk_workspace_bytes();
    persistent = {};
    temporary = {};
    CHECK_CUEST(cuestGridKIntPlanCreateWorkspaceQuery(cuest_handle, gridk_grid_->cuest_grid(),
                                                      cuest_gridk_oeint_plan_, params, &variable, &persistent,
                                                      &temporary, nullptr));
    cuest_gridk_plan_ws_ = cuest_common::allocateWorkspace(&persistent);
    {
        auto scratch = make_workspace(temporary);
        CHECK_CUEST(cuestGridKIntPlanCreate(cuest_handle, gridk_grid_->cuest_grid(), cuest_gridk_oeint_plan_, params,
                                            &variable, cuest_gridk_plan_ws_, scratch.get(), &cuest_gridk_plan_));
    }
    if (fit && print_) {
        uint64_t nmo = 0;
        double lo = 0.0, hi = 0.0;
        CHECK_CUEST(cuestQuery(cuest_handle, CUEST_GRIDKINTPLAN, cuest_gridk_plan_, CUEST_GRIDKINTPLAN_FITTING_NUM_MO,
                               &nmo, sizeof(nmo)));
        CHECK_CUEST(cuestQuery(cuest_handle, CUEST_GRIDKINTPLAN, cuest_gridk_plan_,
                               CUEST_GRIDKINTPLAN_FITTING_MIN_EIGENVALUE, &lo, sizeof(lo)));
        CHECK_CUEST(cuestQuery(cuest_handle, CUEST_GRIDKINTPLAN, cuest_gridk_plan_,
                               CUEST_GRIDKINTPLAN_FITTING_MAX_EIGENVALUE, &hi, sizeof(hi)));
        outfile->Printf("    Grid-K overlap fit: %llu of %d functions retained, quadrature overlap eigenvalues [%.3e, %.3e]\n\n",
                        static_cast<unsigned long long>(nmo), primary_->nbf(), lo, hi);
    }
    CHECK_CUEST(cuestParametersDestroy(CUEST_GRIDKINTPLAN_PARAMETERS, params));
    CHECK_CUEST(cuestParametersCreate(CUEST_GRIDKSYMMETRICEXCHANGECOMPUTE_PARAMETERS,
                                      reinterpret_cast<void**>(&cuest_gridk_exchange_params_)));
    CHECK_CUEST(cuestParametersCreate(CUEST_GRIDKNONSYMMETRICEXCHANGECOMPUTE_PARAMETERS,
                                      reinterpret_cast<void**>(&cuest_gridk_nonsym_exchange_params_)));
}

void cuESTJK::compute_gridk_JK() {
    if (D_ao_.empty() || (!do_J_ && !do_K_)) return;
    using clock = std::chrono::high_resolution_clock;
    const auto start = clock::now();
    const size_t nbf = primary_->nbf();
    const size_t bytes = nbf * nbf * sizeof(double);
    const auto check_cuda = [](cudaError_t status) {
        if (status != cudaSuccess)
            throw PSIEXCEPTION(std::string("cuESTJK Grid-K CUDA error: ") + cudaGetErrorString(status));
    };

    // Grid-K consumes the AO density, not orbital factors, so one staging
    // buffer per matrix serves any number of densities, symmetric or not.
    cuestWorkspaceDescriptor_t matrix{};
    matrix.deviceBufferSizeInBytes = bytes;
    auto density = make_workspace(matrix);
    auto coulomb = do_J_ ? make_workspace(matrix) : Workspace{};
    auto exchange = do_K_ ? make_workspace(matrix) : Workspace{};
    auto d_D = reinterpret_cast<double*>(density->deviceBuffer);
    auto d_J = do_J_ ? reinterpret_cast<double*>(coulomb->deviceBuffer) : nullptr;
    auto d_K = do_K_ ? reinterpret_cast<double*>(exchange->deviceBuffer) : nullptr;

    cuestWorkspaceDescriptor_t j_desc{}, k_desc{}, variable{};
    if (do_K_) variable.deviceBufferSizeInBytes = gridk_workspace_bytes();
    if (do_J_) query_coulomb_workspace(&j_desc, d_D, d_J);
    if (do_K_) {
        if (lr_symmetric_) {
            CHECK_CUEST(cuestGridKSymmetricExchangeComputeWorkspaceQuery(
                cuest_handle, cuest_gridk_plan_, cuest_gridk_exchange_params_, &variable, &k_desc, 1, d_D, d_K));
        } else {
            CHECK_CUEST(cuestGridKNonsymmetricExchangeComputeWorkspaceQuery(
                cuest_handle, cuest_gridk_plan_, cuest_gridk_nonsym_exchange_params_, &variable, &k_desc, 1, d_D,
                d_K));
        }
    }
    cuestWorkspaceDescriptor_t total{};
    total.hostBufferSizeInBytes = std::max(j_desc.hostBufferSizeInBytes, k_desc.hostBufferSizeInBytes);
    total.deviceBufferSizeInBytes = std::max(j_desc.deviceBufferSizeInBytes, k_desc.deviceBufferSizeInBytes);
    auto scratch = make_workspace(total);

    for (size_t n = 0; n < D_ao_.size(); ++n) {
        check_cuda(cudaMemcpy(d_D, D_ao_[n]->get_pointer(), bytes, cudaMemcpyHostToDevice));
        if (do_J_) {
            compute_coulomb(scratch.get(), d_D, d_J);
            check_cuda(cudaDeviceSynchronize());
            check_cuda(cudaMemcpy(J_ao_[n]->get_pointer(), d_J, bytes, cudaMemcpyDeviceToHost));
        }
        if (do_K_) {
            // Response and SAPT densities are signed and nonsymmetric. Never
            // symmetrize D to reach the cheaper SCF kernel.
            if (lr_symmetric_) {
                CHECK_CUEST(cuestGridKSymmetricExchangeCompute(cuest_handle, cuest_gridk_plan_,
                                                               cuest_gridk_exchange_params_, &variable,
                                                               scratch.get(), 1, d_D, d_K));
            } else {
                CHECK_CUEST(cuestGridKNonsymmetricExchangeCompute(cuest_handle, cuest_gridk_plan_,
                                                                  cuest_gridk_nonsym_exchange_params_, &variable,
                                                                  scratch.get(), 1, d_D, d_K));
            }
            check_cuda(cudaDeviceSynchronize());
            check_cuda(cudaMemcpy(K_ao_[n]->get_pointer(), d_K, bytes, cudaMemcpyDeviceToHost));
        }
    }
    // Surface asynchronous kernel errors before the buffers are released.
    check_cuda(cudaDeviceSynchronize());
    if (bench_)
        outfile->Printf("    cuESTJK compute_JK (%s + Grid-K): total=%7.2fms\n", directj_ ? "Direct-J" : "DF-J",
                        std::chrono::duration<double, std::milli>(clock::now() - start).count());
}
#endif

size_t cuESTJK::memory_estimate() {
    return 0;
}

void cuESTJK::print_header() const {
    if (print_) {
        const bool directj = options_.get_str("CUEST_J_BUILD") == "DIRECT";
        const auto k_build = options_.get_str("CUEST_K_BUILD");
        const bool gridk = k_build != "DF";
        outfile->Printf("  ==> cuESTJK: GPU-Accelerated J/K Matrices <==\n\n");
        outfile->Printf("    J algorithm:           %11s\n", (directj ? "Direct" : "DF"));
        outfile->Printf("    K algorithm:           %11s\n", (uses_overlap_fit(k_build) ? "Grid (fit)" : gridk ? "Grid (raw)" : "DF"));
        outfile->Printf("    J tasked:              %11s\n", (do_J_ ? "Yes" : "No"));
        outfile->Printf("    K tasked:              %11s\n", (do_K_ ? "Yes" : "No"));
        outfile->Printf("    wK tasked:             %11s\n", (do_wK_ ? "Yes" : "No"));
        if (do_wK_) 
        outfile->Printf("    Omega:                 %11.3E\n", omega_);
        outfile->Printf("    Pseudoinverse cutoff:  %11.1E\n", condition_);
        outfile->Printf("    Threshold PQ:          %11.1E\n", pq_threshold_);
        if (gridk) {
            outfile->Printf("    Grid-K scratch budget: %8lld MB\n",
                            static_cast<long long>(options_.get_int("CUEST_GRIDK_WORKSPACE_MB")));
        }
        outfile->Printf("\n");
    }
}

void cuESTJK::compute_JK() {
#ifdef PSI_CUEST_HAS_GRIDK
    if (gridk_) {
        compute_gridk_JK();
        return;
    }
#endif
    using clock = std::chrono::high_resolution_clock;
    auto t_total_start = clock::now();
 
    int nbf = primary_->nbf();
    size_t nbf2_bytes = static_cast<size_t>(nbf) * nbf * sizeof(double);
 
    auto t0 = clock::now();
    double* d_D = nullptr;
    double* d_J = nullptr;
    double* d_K = nullptr;
    double* d_C = nullptr;
 
    cudaMalloc(reinterpret_cast<void**>(&d_D), nbf2_bytes);
    cudaMalloc(reinterpret_cast<void**>(&d_J), nbf2_bytes);
    cudaMalloc(reinterpret_cast<void**>(&d_K), nbf2_bytes);
    auto t_alloc = clock::now();
 
    cuestWorkspaceDescriptor_t* j_temp_desc = (cuestWorkspaceDescriptor_t*) malloc(sizeof(cuestWorkspaceDescriptor_t));
    cuestWorkspaceDescriptor_t* k_temp_desc = (cuestWorkspaceDescriptor_t*) malloc(sizeof(cuestWorkspaceDescriptor_t));

    cuestWorkspaceDescriptor_t* variableBufferSize = (cuestWorkspaceDescriptor_t*) malloc(sizeof(cuestWorkspaceDescriptor_t));
    variableBufferSize->hostBufferSizeInBytes = 0;
    variableBufferSize->deviceBufferSizeInBytes = 8000000000;
    
    size_t max_host = 0, max_device = 0;
 
    if (do_J_) {
        query_coulomb_workspace(j_temp_desc, d_D, d_J);
        max_host = std::max(max_host, j_temp_desc->hostBufferSizeInBytes);
        max_device = std::max(max_device, j_temp_desc->deviceBufferSizeInBytes);
    }
 
    if (do_K_) {
        for (size_t N = 0; N < D_ao_.size(); N++) {
            int nocc = C_left_ao_[N]->ncol();
            if (nocc == 0) continue;
 
            CHECK_CUEST(cuestDFSymmetricExchangeComputeWorkspaceQuery(
                cuest_handle,
                cuest_df_plan_,
                cuest_exchange_compute_params_,
                variableBufferSize,
                k_temp_desc,
                static_cast<uint64_t>(nocc),
                nullptr,
                d_K));
            max_host = std::max(max_host, k_temp_desc->hostBufferSizeInBytes);
            max_device = std::max(max_device, k_temp_desc->deviceBufferSizeInBytes);
        }
    }
 
    cuestWorkspaceDescriptor_t* total_desc = (cuestWorkspaceDescriptor_t*) malloc(sizeof(cuestWorkspaceDescriptor_t));
    total_desc->hostBufferSizeInBytes = max_host;
    total_desc->deviceBufferSizeInBytes = max_device;
 
    cuestWorkspace_t* temporaryJKWorkspace = cuest_common::allocateWorkspace(total_desc);

    auto t_ws = clock::now();
    double ms_J_compute = 0.0, ms_K_compute = 0.0;
    double ms_memcpy_h2d = 0.0, ms_memcpy_d2h = 0.0;
    double ms_transpose = 0.0;
 
    for (size_t N = 0; N < D_ao_.size(); N++) {
        if (do_J_) {
            auto tc0 = clock::now();
            cudaMemcpy(d_D, D_ao_[N]->get_pointer(), nbf2_bytes, cudaMemcpyHostToDevice);
            auto tc1 = clock::now();
            compute_coulomb(temporaryJKWorkspace, d_D, d_J);
            cudaDeviceSynchronize();
            auto tc2 = clock::now();
            cudaMemcpy(J_ao_[N]->get_pointer(), d_J, nbf2_bytes, cudaMemcpyDeviceToHost);
            auto tc3 = clock::now();
            ms_memcpy_h2d += std::chrono::duration<double, std::milli>(tc1 - tc0).count();
            ms_J_compute += std::chrono::duration<double, std::milli>(tc2 - tc1).count();
            ms_memcpy_d2h += std::chrono::duration<double, std::milli>(tc3 - tc2).count();
        }
 
        if (do_K_) {
            int nocc = C_left_ao_[N]->ncol();
            if (nocc > 0) {
                size_t c_bytes = static_cast<size_t>(nocc) * nbf * sizeof(double);
 
                auto tt0 = clock::now();
                std::vector<double> C_row_major(nocc * nbf);
                double** Cp = C_left_ao_[N]->pointer();
                for (int i = 0; i < nocc; i++) {
                    for (int mu = 0; mu < nbf; mu++) {
                        C_row_major[i * nbf + mu] = Cp[mu][i];
                    }
                }
                auto tt1 = clock::now();
                ms_transpose += std::chrono::duration<double, std::milli>(tt1 - tt0).count();
 
                cudaFree(d_C);
                cudaMalloc(reinterpret_cast<void**>(&d_C), c_bytes);
 
                auto tk0 = clock::now();
                cudaMemcpy(d_C, C_row_major.data(), c_bytes, cudaMemcpyHostToDevice);
                auto tk1 = clock::now();
 
                CHECK_CUEST(cuestDFSymmetricExchangeCompute(
                    cuest_handle,
                    cuest_df_plan_,
                    cuest_exchange_compute_params_,
                    variableBufferSize,
                    temporaryJKWorkspace,
                    static_cast<uint64_t>(nocc),
                    d_C,
                    d_K));
                cudaDeviceSynchronize();
                auto tk2 = clock::now();
 
                cudaMemcpy(K_ao_[N]->get_pointer(), d_K, nbf2_bytes, cudaMemcpyDeviceToHost);
                auto tk3 = clock::now();
 
                ms_memcpy_h2d += std::chrono::duration<double, std::milli>(tk1 - tk0).count();
                ms_K_compute += std::chrono::duration<double, std::milli>(tk2 - tk1).count();
                ms_memcpy_d2h += std::chrono::duration<double, std::milli>(tk3 - tk2).count();
            }
        }
    }
 
    free(j_temp_desc);
    free(k_temp_desc);
    free(total_desc);
    free(variableBufferSize);

    cuest_common::freeWorkspace(temporaryJKWorkspace);

    auto t_free_start = clock::now();
    cudaFree(d_D);
    cudaFree(d_J);
    cudaFree(d_K);
    if (d_C) cudaFree(d_C);
    auto t_free_end = clock::now();
 
    double ms_alloc = std::chrono::duration<double, std::milli>(t_alloc - t0).count();
    double ms_wsquery = std::chrono::duration<double, std::milli>(t_ws - t_alloc).count();
    double ms_free = std::chrono::duration<double, std::milli>(t_free_end - t_free_start).count();
    double ms_total = std::chrono::duration<double, std::milli>(t_free_end - t_total_start).count();
 
    outfile->Printf("    cuESTJK compute_JK: total=%7.2fms | alloc=%5.2fms ws=%5.2fms "
                     "J=%6.2fms K=%6.2fms memcpy(H2D)=%5.2fms memcpy(D2H)=%5.2fms "
                     "transpose=%5.2fms free=%5.2fms\n",
                     ms_total, ms_alloc, ms_wsquery, ms_J_compute, ms_K_compute,
                     ms_memcpy_h2d, ms_memcpy_d2h, ms_transpose, ms_free);
}

void cuESTJK::postiterations() 
{
    destroy_cuest_objects();
}

} // namespace psi

#endif // USING_cuEST
