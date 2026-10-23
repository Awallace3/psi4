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
#include "cubature.h"

#include "psi4/libmints/molecule.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/integral.h"
#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libpsi4util/process.h"
#include "psi4/liboptions/liboptions.h"
#include "psi4/libpsi4util/exception.h"
#include "psi4/psi4-dec.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

#include <cublas_v2.h>
#include <cusolverDn.h>

extern cublasHandle_t cublas_handle;

namespace psi {

namespace {

// Initial construction only: building a second basis can symmetrize the shared
// molecule again, moving a coordinate by one representable double while the
// first basis owns its original snapshot. This also admits an indistinguishable
// one-ULP preconstruction edit, not a general geometry tolerance. Later shell
// and non-C1 frame dependency keys remain exact.
bool initial_shell_coordinate_matches(double actual, double expected) {
    return std::isfinite(actual) && std::isfinite(expected) &&
           (actual == expected || std::nextafter(actual, expected) == expected);
}

// Plan construction must unwind on allocation/SDK failure. The older shared
// workspace helper exits on OOM and cannot provide that recovery contract.
bool release_plan_workspace(cuestWorkspace_t* ws) noexcept {
    if (!ws) return true;
    bool ok = true;
    if (ws->deviceBuffer) ok = cudaFree(reinterpret_cast<void*>(ws->deviceBuffer)) == cudaSuccess;
    if (ws->hostBuffer) std::free(reinterpret_cast<void*>(ws->hostBuffer));
    std::free(ws);
    return ok;
}

struct PlanWorkspaceDeleter {
    void operator()(cuestWorkspace_t* ws) const noexcept { release_plan_workspace(ws); }
};
using PlanWorkspace = std::unique_ptr<cuestWorkspace_t, PlanWorkspaceDeleter>;

void release_plan_scratch(PlanWorkspace& ws) {
    if (!release_plan_workspace(ws.release()))
        throw PSIEXCEPTION("cuESTJK temporary workspace cleanup failed.");
}

PlanWorkspace plan_workspace(const cuestWorkspaceDescriptor_t& desc) {
    PlanWorkspace ws(static_cast<cuestWorkspace_t*>(std::calloc(1, sizeof(cuestWorkspace_t))));
    if (!ws) throw std::bad_alloc();
    ws->hostBufferSizeInBytes = desc.hostBufferSizeInBytes;
    ws->deviceBufferSizeInBytes = desc.deviceBufferSizeInBytes;
    if (desc.hostBufferSizeInBytes) {
        ws->hostBuffer = reinterpret_cast<uintptr_t>(std::malloc(desc.hostBufferSizeInBytes));
        if (!ws->hostBuffer) throw std::bad_alloc();
    }
    if (desc.deviceBufferSizeInBytes) {
        void* device = nullptr;
        const auto status = cudaMalloc(&device, desc.deviceBufferSizeInBytes);
        if (status != cudaSuccess) {
            throw PSIEXCEPTION(std::string("cuESTJK plan workspace allocation failed: ") + cudaGetErrorString(status));
        }
        ws->deviceBuffer = reinterpret_cast<uintptr_t>(device);
    }
    return ws;
}

template <typename T>
class PlanParameters {
    decltype(CUEST_AOPAIRLIST_PARAMETERS) kind_;
    T value_ = nullptr;
   public:
    explicit PlanParameters(decltype(CUEST_AOPAIRLIST_PARAMETERS) kind) : kind_(kind) {
        const auto status = cuestParametersCreate(kind_, reinterpret_cast<void**>(&value_));
        if (status != CUEST_STATUS_SUCCESS) {
            if (value_) cuestParametersDestroy(kind_, value_);
            CHECK_CUEST(status);
        }
    }
    ~PlanParameters() noexcept { if (value_) cuestParametersDestroy(kind_, value_); }
    PlanParameters(const PlanParameters&) = delete;
    PlanParameters& operator=(const PlanParameters&) = delete;
    T get() const { return value_; }
    void close() {
        if (value_) CHECK_CUEST(cuestParametersDestroy(kind_, value_));
        value_ = nullptr;
    }
};

struct PlanShells {
    std::vector<cuestAOShell_t> values;
    ~PlanShells() noexcept { for (auto shell : values) if (shell) cuestAOShellDestroy(shell); }
    void close() {
        for (auto& shell : values) {
            if (shell) CHECK_CUEST(cuestAOShellDestroy(shell));
            shell = nullptr;
        }
    }
};

template <typename T>
void key_value(std::string& key, const T& value) {
    key.append(reinterpret_cast<const char*>(&value), sizeof(value));
}

void key_double(std::string& key, double value) {
    if (!std::isfinite(value)) throw PSIEXCEPTION("cuESTJK plan dependencies must be finite.");
    key_value(key, value);
}

std::string shell_centers_key(const std::shared_ptr<BasisSet>& primary,
                              const std::shared_ptr<BasisSet>& auxiliary) {
    std::string key;
    for (const auto& basis : {primary, auxiliary}) {
        key_value(key, basis->nshell());
        for (int s = 0; s < basis->nshell(); ++s)
            for (int k = 0; k < 3; ++k) key_double(key, basis->shell(s).coord(k));
    }
    return key;
}

std::string molecular_frame_key(const std::shared_ptr<BasisSet>& primary,
                                const std::shared_ptr<BasisSet>& auxiliary) {
    std::string key;
    for (const auto& basis : {primary, auxiliary}) {
        const auto mol = basis->molecule();
        const auto xyz = mol->geometry();
        key_value(key, mol->natom());
        for (int a = 0; a < mol->natom(); ++a)
            for (int k = 0; k < 3; ++k) key_double(key, xyz.get(a, k));
    }
    return key;
}

void basis_key(std::string& key, const std::shared_ptr<BasisSet>& basis) {
    const auto mol = basis->molecule();
    key_value(key, mol->natom());
    key_value(key, basis->nbf());
    key_value(key, basis->nshell());
    key_value(key, basis->has_puream());
    const auto xyz = mol->geometry();
    for (int a = 0; a < mol->natom(); ++a) {
        key_double(key, mol->Z(a));  // includes the current ghost mask
        key_value(key, mol->true_atomic_number(a));
        for (int k = 0; k < 3; ++k) key_double(key, xyz.get(a, k));
        key_value(key, basis->nshell_on_center(a));
        for (int q = 0; q < basis->nshell_on_center(a); ++q) {
            const int s = basis->shell_on_center(a, q);
            const auto& shell = basis->shell(s);
            key_value(key, s);
            key_value(key, shell.am());
            key_value(key, shell.is_pure());
            key_value(key, shell.nprimitive());
            for (int p = 0; p < shell.nprimitive(); ++p) {
                key_double(key, shell.exps()[p]);
                key_double(key, shell.coefs()[p]);
            }
        }
    }
}
}  // namespace

cuESTJK::cuESTJK(std::shared_ptr<BasisSet> primary, std::shared_ptr<BasisSet> auxiliary, Options& options)
    : JK(primary),
      options_(options),
      auxiliary_(auxiliary),
      condition_(1.0E-12),
      pq_threshold_(1.0E-14),
      initialized_(false),
      cuest_primary_basis_(nullptr),
      cuest_auxiliary_basis_(nullptr),
      cuest_pair_list_(nullptr),
      cuest_df_plan_(nullptr),
      cuest_coulomb_compute_params_(nullptr),
      cuest_exchange_compute_params_(nullptr),
      cuest_nonsym_exchange_compute_params_(nullptr),
      // Set in preiterations(); the destructor and the plan rebuild both free
      // whatever is here, so they must not start out as garbage.
      cuest_pair_list_ws_ptr_(nullptr),
      cuest_dfint_plan_ws_ptr_(nullptr),
      pair_list_device_bytes_(0),
      dfint_plan_device_bytes_(0),
      jk_temp_device_bytes_(0)
{
    cuest_common::ensure_cuest_initialized();

    // cuEST's density-fitting integral plan requires a spherical-harmonic
    // auxiliary basis; the orbital (primary) basis may be Cartesian. Without
    // this check, a Cartesian aux basis fails deep inside cuEST with an
    // opaque "status code" error out of preiterations()/cuestDFIntPlan*.
    if (!auxiliary->has_puream()) {
        throw PSIEXCEPTION("cuESTJK requires a spherical-harmonic auxiliary (DF) basis set; "
                           "the auxiliary basis set '" + auxiliary->name() +
                           "' is Cartesian. Set DF_BASIS_SCF/DF_BASIS_MP2 to a basis with spherical "
                           "harmonics, or add 'puream true' scoped to the auxiliary basis if supported; "
                           "the orbital (primary) basis may remain Cartesian.");
    }

    condition_ = options.get_double("DF_FITTING_CONDITION");
    pq_threshold_ = options.get_double("INTS_TOLERANCE");
    dfk_slices_ = options.get_int("CUEST_DFK_SLICES");
    dfk_moduli_ = options.get_int("CUEST_DFK_MODULI");
    // A JK is tied to these BasisSet lifetimes. Geometry may subsequently
    // change on their molecules, but independent shell displacements may not.
    for (const auto& basis : {primary_, auxiliary_}) {
        const auto xyz = basis->molecule()->geometry();
        for (int s = 0; s < basis->nshell(); ++s)
            for (int k = 0; k < 3; ++k)
                if (!initial_shell_coordinate_matches(basis->shell(s).coord(k),
                                                      xyz.get(basis->shell_to_center(s), k)))
                    throw PSIEXCEPTION("cuESTJK requires basis shell centers consistent with molecule geometry.");
    }
    shell_centers_ = shell_centers_key(primary_, auxiliary_);
    primary_natom_ = primary_->molecule()->natom();
    auxiliary_natom_ = auxiliary_->molecule()->natom();
    primary_symmetry_ = primary_->molecule()->schoenflies_symbol();
    auxiliary_symmetry_ = auxiliary_->molecule()->schoenflies_symbol();
    molecular_frame_ = molecular_frame_key(primary_, auxiliary_);
}

cuESTJK::~cuESTJK() noexcept {
    if (!destroy_cuest_objects())
        std::fprintf(stderr, "cuESTJK cleanup failed; GPU resources may remain until context teardown.\n");
}

std::string cuESTJK::current_plan_key() const {
    if (primary_->molecule()->natom() != primary_natom_ ||
        auxiliary_->molecule()->natom() != auxiliary_natom_ ||
        primary_->molecule()->schoenflies_symbol() != primary_symmetry_ ||
        auxiliary_->molecule()->schoenflies_symbol() != auxiliary_symmetry_)
        throw PSIEXCEPTION("cuESTJK molecular topology/symmetry changed; construct new basis sets and JK.");
    // JK's base class constructs AO2USO once. Rebuilding only the backend plan
    // cannot refresh that symmetry transform for a displaced non-C1 frame.
    if (AO2USO_->nirrep() != 1 && molecular_frame_key(primary_, auxiliary_) != molecular_frame_)
        throw PSIEXCEPTION("cuESTJK non-C1 geometry changed; construct new basis sets and JK.");
    if (shell_centers_key(primary_, auxiliary_) != shell_centers_)
        throw PSIEXCEPTION("cuESTJK does not support independent BasisSet::move_atom displacements; "
                           "construct new basis sets and JK.");
    std::string key;
    basis_key(key, primary_);
    basis_key(key, auxiliary_);
    key_double(key, omega_alpha_);
    key_double(key, omega_beta_);
    key_double(key, omega_);
    key_double(key, condition_);
    key_double(key, options_.get_double("DF_FITTING_CONDITION"));
    key_double(key, cutoff_overridden_ ? pq_threshold_ :
               (options_.get_str("SCREENING") == "NONE" ? 0.0 : options_.get_double("INTS_TOLERANCE")));
    key_value(key, do_J_);
    key_value(key, do_K_);
    key_value(key, do_wK_);
    key_value(key, options_.get_bool("CUEST_MIXED_PRECISION"));
    key_value(key, options_.get_int("CUEST_DFK_SLICES"));
    key_value(key, options_.get_int("CUEST_DFK_MODULI"));
    key_value(key, options_.get_bool("CUEST_GRIDK"));
    if (options_.get_bool("CUEST_GRIDK") && (do_K_ || do_wK_)) {
        // Keep the exchange quadrature tied to the live DFT grid options, not
        // to CUEST_XC or the lifetime of a separate VBase/XC grid.
        for (const auto* name : {"DFT_RADIAL_SCHEME", "DFT_PRUNING_SCHEME", "DFT_GRID_NAME",
                                 "DFT_NUCLEAR_SCHEME", "DFT_BLOCK_SCHEME"}) {
            const auto value = options_.get_str(name);
            key_value(key, value.size());
            key.append(value);
        }
        for (const auto* name : {"DFT_RADIAL_POINTS", "DFT_SPHERICAL_POINTS",
                                 "DFT_BLOCK_MAX_POINTS", "DFT_BLOCK_MIN_POINTS"})
            key_value(key, options_.get_int(name));
        for (const auto* name : {"DFT_BS_RADIUS_ALPHA", "DFT_PRUNING_ALPHA", "DFT_WEIGHTS_TOLERANCE",
                                 "DFT_BASIS_TOLERANCE", "DFT_BLOCK_MAX_RADIUS"})
            key_double(key, options_.get_double(name));
        key_value(key, options_.get_bool("DFT_REMOVE_DISTANT_POINTS"));
    }
    return key;
}

void cuESTJK::build_cuest_basis(const std::shared_ptr<BasisSet>& basis, cuestAOBasis_t& result,
                               cuestWorkspace_t*& workspace) {
    PlanParameters<cuestAOShellParameters_t> shell_params(CUEST_AOSHELL_PARAMETERS);
    PlanShells shells;
    // Reserve before creating SDK objects so vector growth cannot leak a shell.
    shells.values.reserve(basis->nshell());
    const int natom = basis->molecule()->natom();
    std::vector<uint64_t> shells_per_atom(natom);
    for (int a = 0; a < natom; ++a) {
        shells_per_atom[a] = basis->nshell_on_center(a);
        for (int q = 0; q < basis->nshell_on_center(a); ++q) {
            const auto& shell = basis->shell(basis->shell_on_center(a, q));
            shells.values.push_back(nullptr);
            CHECK_CUEST(cuestAOShellCreate(cuest_handle, shell.is_pure() ? 1 : 0, shell.am(),
                                          shell.nprimitive(), shell.exps(), shell.coefs(), shell_params.get(),
                                          &shells.values.back()));
        }
    }
    PlanParameters<cuestAOBasisParameters_t> params(CUEST_AOBASIS_PARAMETERS);
    cuestWorkspaceDescriptor_t persistent{}, temporary{};
    CHECK_CUEST(cuestAOBasisCreateWorkspaceQuery(cuest_handle, natom, shells_per_atom.data(),
                shells.values.data(), params.get(), &persistent, &temporary, nullptr));
    workspace = plan_workspace(persistent).release();
    auto scratch = plan_workspace(temporary);
    CHECK_CUEST(cuestAOBasisCreate(cuest_handle, natom, shells_per_atom.data(), shells.values.data(),
                                 params.get(), workspace, scratch.get(), &result));
    release_plan_scratch(scratch);
    params.close();
    shells.close();
    shell_params.close();
}

void cuESTJK::preiterations()
{
    try {
        if (cleanup_failed_) throw PSIEXCEPTION("cuESTJK cleanup failed; construct a new JK object.");
        const auto key = current_plan_key();
        // The handle is shared by interleaved objects: restore its mode even on hits.
        configure_math_mode();
        if (initialized_ && key == plan_key_) {
            if (print_ >= 3) outfile->Printf("    cuESTJK plan reused #%llu\n",
                                           static_cast<unsigned long long>(plan_creations_));
            return;
        }
        // Fail closed; never overlap two potentially huge plans during a rebuild.
        if (!destroy_cuest_objects()) throw PSIEXCEPTION("cuESTJK cleanup failed; construct a new JK object.");
        build_cuest_plan(key);
    } catch (...) {
        destroy_cuest_objects();
        throw;
    }
}

void cuESTJK::build_cuest_plan(const std::string& key) {
    gridk_ = options_.get_bool("CUEST_GRIDK");
#ifndef PSI_CUEST_HAS_GRIDK
    if (gridk_)
        throw PSIEXCEPTION("CUEST_GRIDK requires cuEST symmetric and nonsymmetric Grid-K APIs; "
                           "rebuild Psi4 with a supporting cuEST SDK (e.g. 0.2.50).");
#endif
    if (!auxiliary_->has_puream()) throw PSIEXCEPTION("cuESTJK requires a spherical-harmonic auxiliary basis.");
    if (options_.get_int("CUEST_DFK_SLICES") <= 0 || options_.get_int("CUEST_DFK_MODULI") <= 0)
        throw PSIEXCEPTION("cuESTJK precision slice/modulus counts must be positive.");
    dfk_slices_ = options_.get_int("CUEST_DFK_SLICES");
    dfk_moduli_ = options_.get_int("CUEST_DFK_MODULI");
    if (!cutoff_overridden_)
        pq_threshold_ = options_.get_str("SCREENING") == "NONE" ? 0.0 : options_.get_double("INTS_TOLERANCE");
    if (pq_threshold_ < 0.0) throw PSIEXCEPTION("cuESTJK pair threshold must be nonnegative.");

    auto mol = primary_->molecule();
    size_t natom = mol->natom();
    const auto xyz = mol->geometry();
    const auto aux_xyz = auxiliary_->molecule()->geometry();
    if (auxiliary_->molecule()->natom() != natom)
        throw PSIEXCEPTION("cuESTJK primary and auxiliary must have the same molecular centers.");
    for (size_t a = 0; a < natom; ++a)
        for (int k = 0; k < 3; ++k)
            if (xyz.get(a, k) != aux_xyz.get(a, k))
                throw PSIEXCEPTION("cuESTJK primary and auxiliary must have the same molecular centers.");
    build_cuest_basis(primary_, cuest_primary_basis_, cuest_primary_basis_ws_);
    build_cuest_basis(auxiliary_, cuest_auxiliary_basis_, cuest_auxiliary_basis_ws_);

    cuestWorkspaceDescriptor_t persistentWorkspaceDescriptor{}, temporaryWorkspaceDescriptor{};

    // AO Pair List
    PlanParameters<cuestAOPairListParameters_t> pair_params(CUEST_AOPAIRLIST_PARAMETERS);

    CHECK_CUEST(cuestAOPairListCreateWorkspaceQuery(cuest_handle, cuest_primary_basis_,
        static_cast<uint64_t>(natom), xyz.pointer()[0], pq_threshold_, pair_params.get(),
        &persistentWorkspaceDescriptor, &temporaryWorkspaceDescriptor, nullptr));

    pair_list_device_bytes_ = persistentWorkspaceDescriptor.deviceBufferSizeInBytes;

    cuest_pair_list_ws_ptr_ = plan_workspace(persistentWorkspaceDescriptor).release();
    auto temporaryPairListWorkspace = plan_workspace(temporaryWorkspaceDescriptor);

    CHECK_CUEST(cuestAOPairListCreate(cuest_handle, cuest_primary_basis_,
        static_cast<uint64_t>(natom), xyz.pointer()[0], pq_threshold_, pair_params.get(),
        cuest_pair_list_ws_ptr_, temporaryPairListWorkspace.get(), &cuest_pair_list_));
    release_plan_scratch(temporaryPairListWorkspace);
    pair_params.close();

    // DF Int Plan List
    PlanParameters<cuestDFIntPlanParameters_t> dfint_params(CUEST_DFINTPLAN_PARAMETERS);
    
    // Psi4 and cuEST both represent exact exchange as alpha / r + beta erf(omega r) / r.
    double exchange_fraction = omega_alpha_;
    double lrc_exchange_fraction = omega_beta_;
    double lr_exchange_omega = omega_;

    //Set omega params if applicable 
    CHECK_CUEST(cuestParametersConfigure(
        CUEST_DFINTPLAN_PARAMETERS, 
        dfint_params.get(),
        CUEST_DFINTPLAN_PARAMETERS_EXCHANGE_FRACTION,
        &exchange_fraction,
        sizeof(double)));

    CHECK_CUEST(cuestParametersConfigure(
        CUEST_DFINTPLAN_PARAMETERS, 
        dfint_params.get(),
        CUEST_DFINTPLAN_PARAMETERS_LRC_EXCHANGE_FRACTION,
        &lrc_exchange_fraction,
        sizeof(double)));

    CHECK_CUEST(cuestParametersConfigure(
        CUEST_DFINTPLAN_PARAMETERS, 
        dfint_params.get(),
        CUEST_DFINTPLAN_PARAMETERS_LRC_EXCHANGE_OMEGA,
        &lr_exchange_omega,
        sizeof(double)));

    CHECK_CUEST(cuestDFIntPlanCreateWorkspaceQuery(cuest_handle,
        cuest_primary_basis_, cuest_auxiliary_basis_, cuest_pair_list_,
        dfint_params.get(), &persistentWorkspaceDescriptor, &temporaryWorkspaceDescriptor, nullptr));

    dfint_plan_device_bytes_ = persistentWorkspaceDescriptor.deviceBufferSizeInBytes;

    cuest_dfint_plan_ws_ptr_ = plan_workspace(persistentWorkspaceDescriptor).release();
    auto temporaryDFIntPlanWorkspace = plan_workspace(temporaryWorkspaceDescriptor);

    CHECK_CUEST(cuestDFIntPlanCreate(cuest_handle,
        cuest_primary_basis_, cuest_auxiliary_basis_, cuest_pair_list_,
        dfint_params.get(), cuest_dfint_plan_ws_ptr_, temporaryDFIntPlanWorkspace.get(), &cuest_df_plan_));
    // Count SDK creations, including a creation followed by a later configure
    // failure, rather than counting initialize() calls or comparing matrices.
    if (plan_creations_ != std::numeric_limits<uint64_t>::max()) ++plan_creations_;
    if (print_ >= 3) outfile->Printf("    cuESTJK plan created #%llu\n",
                                   static_cast<unsigned long long>(plan_creations_));
    release_plan_scratch(temporaryDFIntPlanWorkspace);
    dfint_params.close();

    // Create J/K parameters
    CHECK_CUEST(cuestParametersCreate(CUEST_DFCOULOMBCOMPUTE_PARAMETERS, reinterpret_cast<void**>(&cuest_coulomb_compute_params_)));
    CHECK_CUEST(cuestParametersCreate(CUEST_DFSYMMETRICEXCHANGECOMPUTE_PARAMETERS, reinterpret_cast<void**>(&cuest_exchange_compute_params_)));
    CHECK_CUEST(cuestParametersCreate(CUEST_DFNONSYMMETRICEXCHANGECOMPUTE_PARAMETERS, reinterpret_cast<void**>(&cuest_nonsym_exchange_compute_params_)));

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

    // The nonsymmetric exchange kernel has the same two emulated-GEMM knobs and
    // must be given the same values: a K built with C_left != C_right would
    // otherwise carry a different precision from the K built beside it in the
    // same SAPT term.
    CHECK_CUEST(cuestParametersConfigure(
        CUEST_DFNONSYMMETRICEXCHANGECOMPUTE_PARAMETERS,
        cuest_nonsym_exchange_compute_params_,
        CUEST_DFNONSYMMETRICEXCHANGECOMPUTE_PARAMETERS_INT8_SLICE_COUNT,
        &dfk_slices_,
        sizeof(uint64_t)));

    CHECK_CUEST(cuestParametersConfigure(
        CUEST_DFNONSYMMETRICEXCHANGECOMPUTE_PARAMETERS,
        cuest_nonsym_exchange_compute_params_,
        CUEST_DFNONSYMMETRICEXCHANGECOMPUTE_PARAMETERS_INT8_MODULUS_COUNT,
        &dfk_moduli_,
        sizeof(uint64_t)));

#ifdef PSI_CUEST_HAS_GRIDK
    if (gridk_ && (do_K_ || do_wK_)) build_gridk_plan();
#endif

    // Publish validity only after every create/configure succeeded.
    plan_key_ = key;
    initialized_ = true;

    if (print_) {
        // Report what the card is actually being asked for.  Every other JK
        // builder prints its memory (MemDFJK prints "AOs need X GiB; user
        // supplied Y GiB") and falls back to disk when it does not fit; cuEST
        // has neither a budget nor a fallback, so an over-large system simply
        // dies in cudaMalloc.  Printing the queried sizes next to the free
        // device memory turns that into a diagnosis rather than a guess.
        size_t dev_free = 0, dev_total = 0;
        const bool have_dev = cuest_common::device_memory(&dev_free, &dev_total);
        outfile->Printf("    Device memory, AO pair list:  %8.3f [GiB]\n",
                        pair_list_device_bytes_ * cuest_common::to_gib);
        outfile->Printf("    Device memory, DF int plan:   %8.3f [GiB]\n",
                        dfint_plan_device_bytes_ * cuest_common::to_gib);
#ifdef PSI_CUEST_HAS_GRIDK
        if (cuest_gridk_plan_)
            outfile->Printf("    Device memory, Grid-K plan:   %8.3f [GiB]\n",
                            gridk_plan_device_bytes_ * cuest_common::to_gib);
#endif
        if (have_dev) {
            outfile->Printf("    Device memory, free/total:    %8.3f / %.3f [GiB]\n",
                            dev_free * cuest_common::to_gib, dev_total * cuest_common::to_gib);
        }
        outfile->Printf("\n");
    }
}

#ifdef PSI_CUEST_HAS_GRIDK
void cuESTJK::build_gridk_plan() {
    gridk_grid_ = std::make_unique<DFTGrid>(primary_->molecule(), primary_, options_, true);
    PlanParameters<cuestGridKIntPlanParameters_t> params(CUEST_GRIDKINTPLAN_PARAMETERS);
    CHECK_CUEST(cuestParametersConfigure(CUEST_GRIDKINTPLAN_PARAMETERS, params.get(),
                CUEST_GRIDKINTPLAN_PARAMETERS_EXCHANGE_FRACTION, &omega_alpha_, sizeof(double)));
    CHECK_CUEST(cuestParametersConfigure(CUEST_GRIDKINTPLAN_PARAMETERS, params.get(),
                CUEST_GRIDKINTPLAN_PARAMETERS_LRC_EXCHANGE_FRACTION, &omega_beta_, sizeof(double)));
    CHECK_CUEST(cuestParametersConfigure(CUEST_GRIDKINTPLAN_PARAMETERS, params.get(),
                CUEST_GRIDKINTPLAN_PARAMETERS_LRC_EXCHANGE_OMEGA, &omega_, sizeof(double)));

    cuestWorkspaceDescriptor_t persistent{}, temporary{};
    CHECK_CUEST(cuestGridKIntPlanCreateWorkspaceQuery(
        cuest_handle, cuest_primary_basis_, cuest_pair_list_, gridk_grid_->cuest_grid(),
        params.get(), &persistent, &temporary, nullptr));
    gridk_plan_device_bytes_ = persistent.deviceBufferSizeInBytes;
    cuest_gridk_plan_ws_ = plan_workspace(persistent).release();
    auto scratch = plan_workspace(temporary);
    CHECK_CUEST(cuestGridKIntPlanCreate(
        cuest_handle, cuest_primary_basis_, cuest_pair_list_, gridk_grid_->cuest_grid(),
        params.get(), cuest_gridk_plan_ws_, scratch.get(), &cuest_gridk_plan_));
    release_plan_scratch(scratch);
    params.close();
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
    // Bound staging to one density regardless of the number of CPKS right-hand
    // sides. The queried scratch is shared by J, K, and all densities.
    cuestWorkspaceDescriptor_t matrix_desc{};
    matrix_desc.deviceBufferSizeInBytes = bytes;
    auto density = plan_workspace(matrix_desc);
    auto coulomb = do_J_ ? plan_workspace(matrix_desc) : PlanWorkspace{};
    auto exchange = do_K_ ? plan_workspace(matrix_desc) : PlanWorkspace{};
    auto d_D = reinterpret_cast<double*>(density->deviceBuffer);
    auto d_J = do_J_ ? reinterpret_cast<double*>(coulomb->deviceBuffer) : nullptr;
    auto d_K = do_K_ ? reinterpret_cast<double*>(exchange->deviceBuffer) : nullptr;

    cuestWorkspaceDescriptor_t j_desc{}, k_desc{}, variable{};
    variable.deviceBufferSizeInBytes = 8000000000;
    if (do_J_)
        CHECK_CUEST(cuestDFCoulombComputeWorkspaceQuery(
            cuest_handle, cuest_df_plan_, cuest_coulomb_compute_params_, &j_desc, d_D, d_J));
    if (do_K_) {
        if (lr_symmetric_) {
            CHECK_CUEST(cuestGridKSymmetricExchangeComputeWorkspaceQuery(
                cuest_handle, cuest_gridk_plan_, cuest_gridk_exchange_params_, &variable, &k_desc,
                1, d_D, d_K));
        } else {
            CHECK_CUEST(cuestGridKNonsymmetricExchangeComputeWorkspaceQuery(
                cuest_handle, cuest_gridk_plan_, cuest_gridk_nonsym_exchange_params_, &variable, &k_desc,
                1, d_D, d_K));
        }
    }
    cuestWorkspaceDescriptor_t total{};
    total.hostBufferSizeInBytes = std::max(j_desc.hostBufferSizeInBytes, k_desc.hostBufferSizeInBytes);
    total.deviceBufferSizeInBytes = std::max(j_desc.deviceBufferSizeInBytes, k_desc.deviceBufferSizeInBytes);
    const bool first_report = jk_temp_device_bytes_ == 0;
    jk_temp_device_bytes_ = total.deviceBufferSizeInBytes;
    if (print_ && first_report)
        outfile->Printf("    Device memory, J/Grid-K scratch: %8.3f [GiB]\n\n",
                        jk_temp_device_bytes_ * cuest_common::to_gib);
    auto scratch = plan_workspace(total);
    for (size_t n = 0; n < D_ao_.size(); ++n) {
        check_cuda(cudaMemcpy(d_D, D_ao_[n]->get_pointer(), bytes, cudaMemcpyHostToDevice));
        if (do_J_) {
            CHECK_CUEST(cuestDFCoulombCompute(cuest_handle, cuest_df_plan_, cuest_coulomb_compute_params_,
                                             scratch.get(), d_D, d_J));
            check_cuda(cudaDeviceSynchronize());
            check_cuda(cudaMemcpy(J_ao_[n]->get_pointer(), d_J, bytes, cudaMemcpyDeviceToHost));
        }
        if (do_K_) {
            // A response density is signed and generally nonsymmetric. Never
            // symmetrize D (or the returned K) to use the cheaper SCF kernel.
            if (lr_symmetric_) {
                CHECK_CUEST(cuestGridKSymmetricExchangeCompute(
                    cuest_handle, cuest_gridk_plan_, cuest_gridk_exchange_params_, &variable,
                    scratch.get(), 1, d_D, d_K));
            } else {
                CHECK_CUEST(cuestGridKNonsymmetricExchangeCompute(
                    cuest_handle, cuest_gridk_plan_, cuest_gridk_nonsym_exchange_params_, &variable,
                    scratch.get(), 1, d_D, d_K));
            }
            check_cuda(cudaDeviceSynchronize());
            check_cuda(cudaMemcpy(K_ao_[n]->get_pointer(), d_K, bytes, cudaMemcpyDeviceToHost));
        }
    }
    // Check asynchronous kernel errors before releasing any of their buffers.
    check_cuda(cudaDeviceSynchronize());
    if (bench_)
        outfile->Printf("    cuESTJK compute_JK (DF-J + Grid-K): total=%7.2fms\n",
                        std::chrono::duration<double, std::milli>(clock::now() - start).count());
}
#endif

void cuESTJK::configure_math_mode() {
    // The handle outlives individual JK objects. Always set both directions:
    // allowing emulation must undo native FP64 selected by an earlier build.
    // Reapply before compute as well, since SAPT can reuse an initialized JK.
    CHECK_CUEST(cuestSetMathMode(
        cuest_handle,
        options_.get_bool("CUEST_MIXED_PRECISION") ? CUEST_DEFAULT_MATH_MODE : CUEST_NATIVE_FP64_MATH_MODE));
}

bool cuESTJK::destroy_cuest_objects() noexcept {
    // Invalidate first. Destruction must not throw (including during unwind).
    // On an SDK destruction failure retain dependent objects/storage instead
    // of freeing memory still potentially referenced by the failed object.
    initialized_ = false;
    plan_key_.clear();
    jk_temp_device_bytes_ = 0;
#ifdef PSI_CUEST_HAS_GRIDK
    if (cuest_gridk_plan_) {
        if (cuestGridKIntPlanDestroy(cuest_gridk_plan_) != CUEST_STATUS_SUCCESS) {
            // A failed SDK destruction may leave a live plan referencing its
            // grid. Preserve that grid even when this JK is destructed.
            gridk_grid_.release();
            return !(cleanup_failed_ = true);
        }
        cuest_gridk_plan_ = nullptr;
    }
    gridk_grid_.reset();
    if (cuest_gridk_exchange_params_) {
        if (cuestParametersDestroy(CUEST_GRIDKSYMMETRICEXCHANGECOMPUTE_PARAMETERS,
                                  cuest_gridk_exchange_params_) != CUEST_STATUS_SUCCESS)
            return !(cleanup_failed_ = true);
        cuest_gridk_exchange_params_ = nullptr;
    }
    if (cuest_gridk_nonsym_exchange_params_) {
        if (cuestParametersDestroy(CUEST_GRIDKNONSYMMETRICEXCHANGECOMPUTE_PARAMETERS,
                                  cuest_gridk_nonsym_exchange_params_) != CUEST_STATUS_SUCCESS)
            return !(cleanup_failed_ = true);
        cuest_gridk_nonsym_exchange_params_ = nullptr;
    }
    if (!release_plan_workspace(cuest_gridk_plan_ws_)) cleanup_failed_ = true;
    cuest_gridk_plan_ws_ = nullptr;
    gridk_plan_device_bytes_ = 0;
#endif
    if (cuest_df_plan_) {
        if (cuestDFIntPlanDestroy(cuest_df_plan_) != CUEST_STATUS_SUCCESS)
            return !(cleanup_failed_ = true);
        cuest_df_plan_ = nullptr;
    }
    if (cuest_pair_list_) {
        if (cuestAOPairListDestroy(cuest_pair_list_) != CUEST_STATUS_SUCCESS)
            return !(cleanup_failed_ = true);
        cuest_pair_list_ = nullptr;
    }
    if (cuest_coulomb_compute_params_) {
        if (cuestParametersDestroy(CUEST_DFCOULOMBCOMPUTE_PARAMETERS, cuest_coulomb_compute_params_) != CUEST_STATUS_SUCCESS)
            return !(cleanup_failed_ = true);
        cuest_coulomb_compute_params_ = nullptr;
    }
    if (cuest_exchange_compute_params_) {
        if (cuestParametersDestroy(CUEST_DFSYMMETRICEXCHANGECOMPUTE_PARAMETERS, cuest_exchange_compute_params_) != CUEST_STATUS_SUCCESS)
            return !(cleanup_failed_ = true);
        cuest_exchange_compute_params_ = nullptr;
    }
    if (cuest_nonsym_exchange_compute_params_) {
        if (cuestParametersDestroy(CUEST_DFNONSYMMETRICEXCHANGECOMPUTE_PARAMETERS, cuest_nonsym_exchange_compute_params_) != CUEST_STATUS_SUCCESS)
            return !(cleanup_failed_ = true);
        cuest_nonsym_exchange_compute_params_ = nullptr;
    }
    if (cuest_primary_basis_) {
        if (cuestAOBasisDestroy(cuest_primary_basis_) != CUEST_STATUS_SUCCESS)
            return !(cleanup_failed_ = true);
        cuest_primary_basis_ = nullptr;
    }
    if (cuest_auxiliary_basis_) {
        if (cuestAOBasisDestroy(cuest_auxiliary_basis_) != CUEST_STATUS_SUCCESS)
            return !(cleanup_failed_ = true);
        cuest_auxiliary_basis_ = nullptr;
    }
    for (auto ws : {&cuest_pair_list_ws_ptr_, &cuest_dfint_plan_ws_ptr_,
                    &cuest_primary_basis_ws_, &cuest_auxiliary_basis_ws_}) {
        if (!release_plan_workspace(*ws)) cleanup_failed_ = true;
        *ws = nullptr;
    }
    return !cleanup_failed_;
}

void cuESTJK::rebuild_cuest_plan() {
    // Before initialize() the plan does not exist yet and preiterations() will
    // pick the new fractions up on its own.
    if (!initialized_) return;
    preiterations();
}

void cuESTJK::set_cutoff(double cutoff) {
    cutoff_overridden_ = true;
    cutoff_ = cutoff;
    pq_threshold_ = cutoff;
    rebuild_cuest_plan();
}

void cuESTJK::set_omega_alpha(double alpha) {
    if (alpha == omega_alpha_) return;
    omega_alpha_ = alpha;
    rebuild_cuest_plan();
}

void cuESTJK::set_omega(double omega) {
    if (omega == omega_) return;
    omega_ = omega;
    rebuild_cuest_plan();
}

void cuESTJK::set_omega_beta(double beta) {
    if (beta == omega_beta_) return;
    omega_beta_ = beta;
    rebuild_cuest_plan();
}

size_t cuESTJK::memory_estimate() {
    return 0;
}

void cuESTJK::print_header() const {
    if (print_) {
        cuestMathMode_t math_mode;
        CHECK_CUEST(cuestGetMathMode(cuest_handle, &math_mode));
        outfile->Printf("  ==> cuESTJK: GPU-Accelerated %s Matrices <==\n\n",
                        options_.get_bool("CUEST_GRIDK") ? "DF-J + Grid-K" : "Density-Fitted J/K");
        outfile->Printf("    J tasked:              %11s\n", (do_J_ ? "Yes" : "No"));
        outfile->Printf("    K tasked:              %11s\n", (do_K_ ? "Yes" : "No"));
        outfile->Printf("    wK tasked:             %11s\n", (do_wK_ ? "Yes" : "No"));
        if (do_wK_) 
        outfile->Printf("    Omega:                 %11.3E\n", omega_);
        outfile->Printf("    Pseudoinverse cutoff:  %11.1E\n", condition_);
        outfile->Printf("    Threshold PQ:          %11.1E\n", pq_threshold_);
        outfile->Printf("    Math mode:             %s\n",
                        math_mode == CUEST_NATIVE_FP64_MATH_MODE ? "Native FP64"
                                                               : "Default (mixed precision permitted)");
        outfile->Printf("\n");
    }
}

void cuESTJK::compute_JK() {
    if (!initialized_) {
        throw PSIEXCEPTION("cuESTJK plan is not initialized. Call initialize() after "
                           "finalize() or a failed plan rebuild.");
    }
    // Detect live geometry/shell/options mutations even if a consumer forgot a
    // fresh initialize(). Never resurrect an explicitly finalized/failed plan.
    try {
        if (current_plan_key() != plan_key_) preiterations();
        configure_math_mode();
    } catch (...) {
        destroy_cuest_objects();
        throw;
    }

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

    // One pair of coefficient buffers per build, not per density. SAPT cache
    // builds carry several densities and SAD carries alpha/beta together.
    // Keep the lifetime local so retained scratch cannot crowd out later SCFs.
    size_t max_nocc = 0;
    if (do_K_) {
        for (size_t N = 0; N < D_ao_.size(); ++N) {
            const int nocc = C_left_ao_[N]->ncol();
            if (!lr_symmetric_ && C_right_ao_[N]->ncol() != nocc) {
                throw PSIEXCEPTION("cuESTJK: C_left and C_right must have the same number of columns; "
                                   "cuEST's nonsymmetric exchange kernel contracts them over a shared "
                                   "occupied index.");
            }
            max_nocc = std::max(max_nocc, static_cast<size_t>(nocc));
        }
    }
    const size_t max_coeffs = max_nocc * static_cast<size_t>(nbf);
    std::vector<double> C_row_major(max_coeffs);
    std::vector<double> C_right_row_major(lr_symmetric_ ? 0 : max_coeffs);
 
    auto t0 = clock::now();
    double* d_D = nullptr;
    double* d_J = nullptr;
    double* d_K = nullptr;
    double* d_C = nullptr;
    double* d_C_right = nullptr;
 
    if (do_J_) {
        cudaMalloc(reinterpret_cast<void**>(&d_D), nbf2_bytes);
        cudaMalloc(reinterpret_cast<void**>(&d_J), nbf2_bytes);
    }
    if (do_K_) cudaMalloc(reinterpret_cast<void**>(&d_K), nbf2_bytes);
    if (max_coeffs) {
        cudaMalloc(reinterpret_cast<void**>(&d_C), max_coeffs * sizeof(double));
        if (!lr_symmetric_)
            cudaMalloc(reinterpret_cast<void**>(&d_C_right), max_coeffs * sizeof(double));
    }
    auto t_alloc = clock::now();
 
    cuestWorkspaceDescriptor_t* j_temp_desc = (cuestWorkspaceDescriptor_t*) malloc(sizeof(cuestWorkspaceDescriptor_t));
    cuestWorkspaceDescriptor_t* k_temp_desc = (cuestWorkspaceDescriptor_t*) malloc(sizeof(cuestWorkspaceDescriptor_t));

    cuestWorkspaceDescriptor_t* variableBufferSize = (cuestWorkspaceDescriptor_t*) malloc(sizeof(cuestWorkspaceDescriptor_t));
    variableBufferSize->hostBufferSizeInBytes = 0;
    variableBufferSize->deviceBufferSizeInBytes = 8000000000;
    
    size_t max_host = 0, max_device = 0;
 
    if (do_J_) {
        CHECK_CUEST(cuestDFCoulombComputeWorkspaceQuery(
            cuest_handle,
            cuest_df_plan_,
            cuest_coulomb_compute_params_,
            j_temp_desc,
            d_D,
            d_J));
        max_host = std::max(max_host, j_temp_desc->hostBufferSizeInBytes);
        max_device = std::max(max_device, j_temp_desc->deviceBufferSizeInBytes);
    }
 
    if (do_K_) {
        for (size_t N = 0; N < D_ao_.size(); N++) {
            int nocc = C_left_ao_[N]->ncol();
            if (nocc == 0) continue;
 
            if (lr_symmetric_) {
                CHECK_CUEST(cuestDFSymmetricExchangeComputeWorkspaceQuery(
                    cuest_handle,
                    cuest_df_plan_,
                    cuest_exchange_compute_params_,
                    variableBufferSize,
                    k_temp_desc,
                    static_cast<uint64_t>(nocc),
                    nullptr,
                    d_K));
            } else {
                CHECK_CUEST(cuestDFNonsymmetricExchangeComputeWorkspaceQuery(
                    cuest_handle,
                    cuest_df_plan_,
                    cuest_nonsym_exchange_compute_params_,
                    variableBufferSize,
                    k_temp_desc,
                    1,
                    static_cast<uint64_t>(nocc),
                    nullptr,
                    nullptr,
                    d_K));
            }
            max_host = std::max(max_host, k_temp_desc->hostBufferSizeInBytes);
            max_device = std::max(max_device, k_temp_desc->deviceBufferSizeInBytes);
        }
    }
 
    cuestWorkspaceDescriptor_t* total_desc = (cuestWorkspaceDescriptor_t*) malloc(sizeof(cuestWorkspaceDescriptor_t));
    total_desc->hostBufferSizeInBytes = max_host;
    total_desc->deviceBufferSizeInBytes = max_device;
 
    // The temporary workspace is re-queried on every build because it depends on
    // how many densities and occupied columns this build carries.  Report it the
    // first time only: it is the peak the card has to absorb on top of the two
    // persistent workspaces, and repeating it every SCF iteration would bury
    // the output.
    const bool first_jk_report = (jk_temp_device_bytes_ == 0);
    jk_temp_device_bytes_ = max_device;
    if (print_ && first_jk_report) {
        outfile->Printf("    Device memory, J/K scratch:   %8.3f [GiB]\n\n",
                        jk_temp_device_bytes_ * cuest_common::to_gib);
    }

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
            CHECK_CUEST(cuestDFCoulombCompute(
                cuest_handle,
                cuest_df_plan_,
                cuest_coulomb_compute_params_,
                temporaryJKWorkspace,
                d_D,
                d_J));
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
 
                // cuEST wants the occupied index first (nocc x nbf, row-major);
                // a Psi4 coefficient matrix is nbf x nocc.
                auto pack_transposed = [nocc, nbf](const SharedMatrix& C, double* out) {
                    double** Cp = C->pointer();
                    for (int i = 0; i < nocc; i++) {
                        for (int mu = 0; mu < nbf; mu++) {
                            out[i * nbf + mu] = Cp[mu][i];
                        }
                    }
                };
 
                auto tt0 = clock::now();
                pack_transposed(C_left_ao_[N], C_row_major.data());
 
                // C_left != C_right happens all over SAPT's exchange terms.  The
                // symmetric kernel would quietly return K built from C_left
                // twice, which is a wrong answer that still converges and still
                // looks plausible -- so the two paths are selected here, never
                // approximated by one another.
                if (!lr_symmetric_) {
                    pack_transposed(C_right_ao_[N], C_right_row_major.data());
                }
                auto tt1 = clock::now();
                ms_transpose += std::chrono::duration<double, std::milli>(tt1 - tt0).count();
 
                auto tk0 = clock::now();
                cudaMemcpy(d_C, C_row_major.data(), c_bytes, cudaMemcpyHostToDevice);
                if (!lr_symmetric_) {
                    cudaMemcpy(d_C_right, C_right_row_major.data(), c_bytes, cudaMemcpyHostToDevice);
                }
                auto tk1 = clock::now();
 
                if (lr_symmetric_) {
                    CHECK_CUEST(cuestDFSymmetricExchangeCompute(
                        cuest_handle,
                        cuest_df_plan_,
                        cuest_exchange_compute_params_,
                        variableBufferSize,
                        temporaryJKWorkspace,
                        static_cast<uint64_t>(nocc),
                        d_C,
                        d_K));
                } else {
                    CHECK_CUEST(cuestDFNonsymmetricExchangeCompute(
                        cuest_handle,
                        cuest_df_plan_,
                        cuest_nonsym_exchange_compute_params_,
                        variableBufferSize,
                        temporaryJKWorkspace,
                        1,
                        static_cast<uint64_t>(nocc),
                        d_C,
                        d_C_right,
                        d_K));
                }
                cudaDeviceSynchronize();
                auto tk2 = clock::now();
 
                cudaMemcpy(K_ao_[N]->get_pointer(), d_K, nbf2_bytes, cudaMemcpyDeviceToHost);
                auto tk3 = clock::now();
 
                ms_memcpy_h2d += std::chrono::duration<double, std::milli>(tk1 - tk0).count();
                ms_K_compute += std::chrono::duration<double, std::milli>(tk2 - tk1).count();
                ms_memcpy_d2h += std::chrono::duration<double, std::milli>(tk3 - tk2).count();
            } else {
                // In particular, integer-occupation SAD on hydrogen has no
                // beta occupied orbitals. Never leave a previous build's K.
                K_ao_[N]->zero();
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
    if (d_C_right) cudaFree(d_C_right);
    auto t_free_end = clock::now();
 
    double ms_alloc = std::chrono::duration<double, std::milli>(t_alloc - t0).count();
    double ms_wsquery = std::chrono::duration<double, std::milli>(t_ws - t_alloc).count();
    double ms_free = std::chrono::duration<double, std::milli>(t_free_end - t_free_start).count();
    double ms_total = std::chrono::duration<double, std::milli>(t_free_end - t_total_start).count();
 
    if (bench_) outfile->Printf("    cuESTJK compute_JK: total=%7.2fms | alloc=%5.2fms ws=%5.2fms "
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
