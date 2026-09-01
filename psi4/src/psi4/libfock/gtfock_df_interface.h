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

#ifndef PSI4_LIBFOCK_GTFOCK_DF_INTERFACE_H
#define PSI4_LIBFOCK_GTFOCK_DF_INTERFACE_H

#include <cstddef>
#include <memory>
#include <vector>

#include "psi4/pragma.h"

namespace psi {

class BasisSet;
class Matrix;

/*! \brief Psi4's side of the Psi4 <-> GTFock density-fitting bridge.
 *
 *  This is the DF counterpart of MinimalInterface. Where that class drives
 *  GTFock's exact four-center engine (`pfock.h`), this one drives the
 *  distributed density-fitting engine `gtfock_psi4` grew for it
 *  (`gtfock_pdf.h`, library `gtfockdf`). The engine builds the fitted tensor
 *  \f$B_Q^{mn}\f$ once, distributes it over ranks by auxiliary index Q, and
 *  contracts it against a density each iteration; a build costs one
 *  MPI_Allreduce of \f$2 n_{bf}^2\f$ doubles.
 *
 *  Two differences from the exact path drive the whole design here:
 *
 *  - `PDF_t` keeps no file-scope state, so a process may own several engines
 *    and may destroy them. `MinimalInterface` cannot: `fock_task.c` caches the
 *    basis and every screening buffer in globals filled exactly once. That is
 *    why this class owns its engine outright and creates it from
 *    GTFockDFJK::preiterations() rather than from the first Fock build.
 *  - The setup cost therefore lands inside `JK::initialize()`, exactly where
 *    MemDFJK's does, so the two are finally comparable on a timer.
 *
 *  Every entry point is collective over MPI_COMM_WORLD and must be reached by
 *  all ranks in the same order. MPI must already be running; that is the Python
 *  layer's job (`psi4.driver.gtfock`, which imports mpi4py).
 */
class MinimalDFInterface {
   public:
    /*! \param primary   orbital basis; must be Cartesian
     *  \param auxiliary fitting basis; must be Cartesian
     *  \param fitting_condition relative eigenvalue cutoff for inverting the
     *         Coulomb metric, i.e. Psi4's DF_FITTING_CONDITION
     *  \param nthreads  OpenMP threads for the three-center integrals */
    MinimalDFInterface(std::shared_ptr<BasisSet> primary, std::shared_ptr<BasisSet> auxiliary,
                       double fitting_condition, int nthreads);
    ~MinimalDFInterface();

    MinimalDFInterface(const MinimalDFInterface&) = delete;
    MinimalDFInterface& operator=(const MinimalDFInterface&) = delete;

    /*! One density's J and K, both replicated on every rank on return.
     *
     *  \param D    nbf x nbf density; required whenever J is wanted
     *  \param Cocc nbf x nocc occupied coefficients with any occupation factor
     *              already folded in, so that D = Cocc Cocc^T; required
     *              whenever K is wanted
     *  \param J,K  destinations, overwritten, either may be null to skip
     *
     *  Every matrix must be a single dense C1 block of the stated shape: the
     *  transfer hands GTFock `Matrix::pointer(0)[0]` and a length, so a
     *  symmetry-blocked or mis-sized matrix would be read or written past its
     *  end rather than refused. */
    void compute_JK(const std::shared_ptr<Matrix>& D, const std::shared_ptr<Matrix>& Cocc,
                    const std::shared_ptr<Matrix>& J, const std::shared_ptr<Matrix>& K);

    int nbf() const { return nbf_; }
    int naux() const { return naux_; }
    /// Auxiliary functions this rank owns after redistribution.
    int nlocal_aux() const { return nlocal_aux_; }
    /// Auxiliary functions the fitting-condition cutoff dropped, on every rank.
    int nmetric_null() const { return nmetric_null_; }
    /*! AO-pair elements this rank computed three-center integrals for, before
     *  redistribution. Zero is legal: with few shells and many ranks the
     *  AO-element partition can leave a rank empty. */
    int nlocal_pairs() const { return nlocal_pairs_; }
    /// Doubles in this rank's slice of the fitted tensor.
    size_t local_tensor_doubles() const { return local_tensor_doubles_; }

    int mpi_rank() const { return mpi_rank_; }
    int mpi_size() const { return mpi_size_; }
    /// J/K builds this engine has run.
    size_t jk_builds() const { return jk_builds_; }

    /// True when Psi4 was compiled against a GTFock that ships libgtfockdf.
    static bool enabled();
    /// Total distributed-DF J/K builds this process has run, over all engines.
    static size_t total_jk_builds();
    /*! {nbf, naux, nlocal_aux, nmetric_null, nlocal_pairs} of the most recent
     *  engine, or five -1s if none. nlocal_aux differing across ranks and
     *  summing to naux is what shows the tensor was really distributed. */
    static std::vector<int> last_partition();
    /// Doubles in this rank's slice of the most recent engine's tensor, or 0.
    static size_t last_local_tensor_doubles();

   private:
    /// Refuse basis sets whose answer would be wrong, not merely slow.
    void check_supported(const std::shared_ptr<BasisSet>& primary, const std::shared_ptr<BasisSet>& auxiliary) const;

    struct Impl;
    std::unique_ptr<Impl> impl_;

    double fitting_condition_ = 1.0e-12;
    int nthreads_ = 1;
    size_t jk_builds_ = 0;
    int nbf_ = 0;
    int naux_ = 0;
    int nlocal_aux_ = 0;
    int nmetric_null_ = 0;
    int nlocal_pairs_ = 0;
    size_t local_tensor_doubles_ = 0;
    int mpi_rank_ = 0;
    int mpi_size_ = 1;
};

}  // namespace psi

#endif  // PSI4_LIBFOCK_GTFOCK_DF_INTERFACE_H
