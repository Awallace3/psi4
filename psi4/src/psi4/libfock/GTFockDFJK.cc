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

#include "psi4/libfock/gtfock_df_interface.h"
#include "psi4/libfock/gtfock_interface.h"
#include "psi4/libmints/basisset.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libpsi4util/exception.h"
#include "psi4/psi4-dec.h"

namespace psi {

GTFockDFJK::GTFockDFJK(std::shared_ptr<BasisSet> Primary, std::shared_ptr<BasisSet> Auxiliary)
    : JK(Primary), auxiliary_(Auxiliary) {
    // The engine is not built until preiterations(), so report a GTFock-less
    // build here rather than several driver layers later.
    if (!MinimalDFInterface::enabled()) {
        throw PSIEXCEPTION(
            "Psi4 was not compiled with GTFock density-fitting support. SCF_TYPE GTFOCK_DF needs a GTFock "
            "installation that ships libgtfockdf and gtfock_pdf.h; reconfigure with -DENABLE_GTFock=ON against "
            "one that does.");
    }
}

size_t GTFockDFJK::memory_estimate() {
    // What this rank holds in steady state is its slice of the fitted tensor.
    // Once the engine exists GTFock can say exactly how large that slice is;
    // before then, estimate it as an even split of the auxiliary index over the
    // ranks. Either way this is a per-rank figure, which is the one Psi4's
    // planner is comparing against this process's memory.
    //
    // The setup peak is roughly twice this, because the three-center integrals
    // are formed in an AO-pair distribution and then redistributed by auxiliary
    // index; that transient is GTFock's own and is not reported here.
    if (Impl_) return Impl_->local_tensor_doubles();
    if (!auxiliary_ || !primary_) return 0;
    const size_t nbf = primary_->nbf();
    const size_t naux = auxiliary_->nbf();
    const size_t npair = nbf * (nbf + 1) / 2;
    const int world = MinimalInterface::world_size();
    const size_t nranks = world > 0 ? static_cast<size_t>(world) : 1;
    return ((naux + nranks - 1) / nranks) * npair;
}

void GTFockDFJK::preiterations() {
    // Building the engine here rather than on the first Fock build is the whole
    // point of this class: GTFock's DF engine keeps no process-global state, so
    // it can be created before the iterations start, which puts the fitting
    // setup inside JK::initialize() exactly where MemDFJK's is.
    Impl_ = std::make_shared<MinimalDFInterface>(primary_, auxiliary_, fitting_condition_,
                                                 nthreads_ > 0 ? nthreads_ : omp_nthread_);
}

void GTFockDFJK::compute_JK() {
    if (do_wK_) {
        throw PSIEXCEPTION("GTFockDFJK: range-separated (wK) integrals are not available from GTFock.");
    }
    if (!Impl_) {
        throw PSIEXCEPTION("GTFockDFJK: compute_JK() was called before initialize(); the DF engine does not exist.");
    }
    // GTFock's exchange contraction consumes occupied orbitals rather than a
    // density, so it can only build K for a density that factors as
    // D = Cocc Cocc^T. A non-symmetric build (C_left != C_right) does not.
    if (do_K_ && !lr_symmetric_) {
        throw PSIEXCEPTION(
            "GTFockDFJK: non-symmetric densities (C_left != C_right) are not supported. GTFock's fitted "
            "exchange contracts occupied orbitals directly, which assumes D = C C^T. SOSCF, stability "
            "analysis, and response-type J/K builds need another SCF_TYPE.");
    }

    zero();

    // Unlike the exact GTFock path, which has one global density matrix baked
    // into its engine, PDF_computeJK takes one density at a time, so any number
    // of them can be driven through the same engine -- open shell included. All
    // ranks run the same replicated Psi4 driver, so they agree on the count and
    // stay in step through the collective inside each call.
    const size_t nmats = D_ao_.size();
    for (size_t i = 0; i < nmats; ++i) {
        const std::shared_ptr<Matrix> none;
        Impl_->compute_JK(D_ao_[i], do_K_ ? C_left_ao_[i] : none, do_J_ ? J_ao_[i] : none,
                          do_K_ ? K_ao_[i] : none);
    }
}

void GTFockDFJK::postiterations() {
    // Dropping the engine frees this rank's slice of the fitted tensor, which is
    // the largest thing this JK owns. See ~MinimalDFInterface for why it is safe
    // to do here and why it is skipped once MPI has been finalized.
    Impl_.reset();
}

void GTFockDFJK::print_header() const {
    if (print_) {
        outfile->Printf("  ==> GTFockDFJK: MPI-Distributed Density-Fitted J/K <==\n\n");

        outfile->Printf("    J tasked:           %11s\n", (do_J_ ? "Yes" : "No"));
        outfile->Printf("    K tasked:           %11s\n", (do_K_ ? "Yes" : "No"));
        outfile->Printf("    wK tasked:          %11s\n", (do_wK_ ? "Yes" : "No"));
        outfile->Printf("    OpenMP threads:     %11d\n", nthreads_ > 0 ? nthreads_ : omp_nthread_);
        outfile->Printf("    Fitting Condition:  %11.0E\n", fitting_condition_);
        if (Impl_) {
            outfile->Printf("    MPI ranks:          %11d\n", Impl_->mpi_size());
            outfile->Printf("    Aux functions:      %11d\n", Impl_->naux());
            // The interesting number on more than one rank: naux split this way
            // across ranks is what makes this different from MemDFJK.
            outfile->Printf("    Aux on this rank:   %11d\n", Impl_->nlocal_aux());
            if (Impl_->nmetric_null() > 0) {
                outfile->Printf("    Metric null space:  %11d\n", Impl_->nmetric_null());
            }
            outfile->Printf("    Tensor [MiB]:       %11.1f\n\n",
                            static_cast<double>(Impl_->local_tensor_doubles()) * 8.0 / 1048576.0);
        } else {
            // preiterations() has not run, so there is no partition to report.
            outfile->Printf("    Engine:             %11s\n\n", "deferred");
        }

        outfile->Printf("   => Auxiliary Basis Set <=\n\n");
        if (auxiliary_) auxiliary_->print_by_level("outfile", print_);
    }
}

}  // namespace psi
