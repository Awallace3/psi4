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
#include "psi4/libfock/jk.h"

#include "psi4/libfock/gtfock_interface.h"
#include "psi4/libmints/basisset.h"
#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libpsi4util/exception.h"
#include "psi4/psi4-dec.h"

namespace psi {

namespace {
/// The GTFock engine is only created on the first compute_JK(), so report a
/// GTFock-less build here rather than several driver layers later.
void require_gtfock() {
    if (!MinimalInterface::enabled()) {
        throw PSIEXCEPTION("Psi4 was not compiled with GTFock support. Reconfigure with -DENABLE_GTFock=ON.");
    }
}
}  // namespace

GTFockJK::GTFockJK(std::shared_ptr<psi::BasisSet> Primary) : JK(Primary) { require_gtfock(); }

GTFockJK::GTFockJK(std::shared_ptr<psi::BasisSet> Primary, size_t NMats, bool AreSymm)
    : JK(Primary), NMats_(static_cast<int>(NMats)), are_symm_(AreSymm), fixed_shape_(true) {
    require_gtfock();
}

size_t GTFockJK::memory_estimate() {
    // GTFock allocates its own distributed buffers outside Psi4's accounting,
    // so there is nothing here for Psi4's memory planner to reserve.
    return 0;
}

void GTFockJK::compute_JK() {
    if (do_wK_) {
        throw PSIEXCEPTION("GTFockJK: range-separated (wK) integrals are not available from GTFock.");
    }

    // zero out J, K, and wK matrices
    zero();

    // GTFock fixes the density count when its engine is created. The
    // three-argument constructor pins it up front; otherwise adopt whatever
    // libfock is asking for on the first build and reuse the engine after that,
    // so an SCF pays engine setup once rather than once per iteration.
    const int requested = static_cast<int>(C_left_.size());
    if (!Impl_) {
        if (!fixed_shape_) {
            NMats_ = requested;
            are_symm_ = lr_symmetric_;
        }
        Impl_ = std::make_shared<MinimalInterface>(primary_, static_cast<size_t>(NMats_), are_symm_);
    }
    if (requested != NMats_) {
        throw PSIEXCEPTION("GTFockJK: this engine was built for " + std::to_string(NMats_) +
                           " density matrices but " + std::to_string(requested) + " were supplied.");
    }

    Impl_->SetP(D_ao_);
    if (do_J_) Impl_->GetJ(J_ao_);
    if (do_K_) Impl_->GetK(K_ao_);
}

void GTFockJK::print_header() const {
    if (print_) {
        outfile->Printf("  ==> GTFockJK: MPI-Distributed J/K <==\n\n");
        if (Impl_) {
            outfile->Printf("    MPI ranks:          %d\n", Impl_->mpi_size());
            outfile->Printf("    Process grid:       %d x %d\n", Impl_->nprow(), Impl_->npcol());
        } else {
            outfile->Printf("    MPI ranks:          %d\n", MinimalInterface::world_size());
        }
        outfile->Printf("    Densities per build: %d\n\n", NMats_);
    }
}

}  // namespace psi
