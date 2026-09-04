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

#ifndef PSI4_LIBFOCK_GTFOCK_INTERFACE_H
#define PSI4_LIBFOCK_GTFOCK_INTERFACE_H

#include <memory>
#include <string>
#include <vector>

#include "psi4/pragma.h"

namespace psi {

class BasisSet;
class Matrix;

/*! \brief Psi4's side of the Psi4 <-> GTFock bridge.
 *
 *  GTFock is an MPI-distributed Fock-build engine with a C API (`pfock.h`,
 *  `CInt.h`, `GTMatrix.h`). This class owns one GTFock `PFock_t` engine plus the
 *  `BasisSet_t` imported from Psi4, and translates between Psi4 `Matrix`
 *  objects and GTFock's distributed `GTMatrix` handles. Every entry point is
 *  collective over `MPI_COMM_WORLD`, which is the only communicator GTFock
 *  knows about.
 *
 *  MPI must already be initialized when the engine is created. In Psi4 that is
 *  the job of the Python layer (`psi4.driver.gtfock`, which imports mpi4py), so
 *  the C++ side only checks and reports.
 *
 *  Only the pieces the milestone-2 prototype needs are implemented; see
 *  `check_supported()` for the cases that are refused outright rather than
 *  answered with a wrong number.
 */
class MinimalInterface {
   public:
    /*! \param nmats   number of density matrices per Fock build (must be 1)
     *  \param are_symm whether those densities are symmetric
     *  \param cutoff   Schwarz screening tolerance handed to GTFock */
    MinimalInterface(std::shared_ptr<BasisSet> primary, size_t nmats, bool are_symm, double cutoff);
    ~MinimalInterface();

    MinimalInterface(const MinimalInterface&) = delete;
    MinimalInterface& operator=(const MinimalInterface&) = delete;

    /*! Every matrix passed to or filled by these three must be a single dense
     *  `nbf x nbf` C1 block, because each transfer moves `nbf*nbf` contiguous
     *  doubles through `Matrix::pointer(0)`; a null, symmetry-blocked, or
     *  wrongly sized matrix is refused rather than read or written past its
     *  end. `GetJ`/`GetK` do not resize their destinations; callers must hand
     *  them matrices that are already that shape. */
    /// Push densities into GTFock and run the distributed Fock build.
    void SetP(const std::vector<std::shared_ptr<Matrix>>& Ps);
    /// Pull the Coulomb matrices of the last build, in Psi4's convention.
    void GetJ(std::vector<std::shared_ptr<Matrix>>& Js);
    /// Pull the exchange matrices of the last build, in Psi4's convention.
    void GetK(std::vector<std::shared_ptr<Matrix>>& Ks);

    /// Rank of this process in MPI_COMM_WORLD, as GTFock sees it.
    int mpi_rank() const { return mpi_rank_; }
    /// Size of MPI_COMM_WORLD, as GTFock sees it.
    int mpi_size() const { return mpi_size_; }
    /// GTFock's process grid, nprow x npcol.
    int nprow() const { return nprow_; }
    int npcol() const { return npcol_; }
    /// Inclusive AO row/column range of the block this rank owns.
    int start_row() const { return start_row_; }
    int end_row() const { return end_row_; }
    int start_col() const { return start_col_; }
    int end_col() const { return end_col_; }
    /*! GTFock's task decomposition of the panel this rank owns: how many task
     *  blocks it holds along each dimension, and how many tasks that makes. */
    int nblks_row() const { return nblks_row_; }
    int nblks_col() const { return nblks_col_; }
    int ntasks() const { return ntasks_; }
    /// Number of GTFock Fock builds this engine has run.
    size_t fock_builds() const { return fock_builds_; }

    /// True when Psi4 was compiled with GTFock support.
    static bool enabled();
    /// {nprow, npcol} of the most recent GTFock engine, or {-1, -1} if none.
    static std::vector<int> process_grid();
    /// {start_row, end_row, start_col, end_col} of the AO block this rank owned
    /// in the most recent GTFock engine, or four -1s if none. Distinct values
    /// across ranks are evidence GTFock really distributed the build.
    static std::vector<int> local_block();
    /// {nblks_row, nblks_col, ntasks} of the task decomposition this rank owned
    /// in the most recent GTFock engine, or three -1s if none. Counts above one
    /// show the rank's AO panel was itself split into blocks, which is what
    /// separates a real distributed build from a system too small to decompose.
    static std::vector<int> local_task_shape();
    /// Total GTFock Fock builds run by this process since it started.
    static size_t total_fock_builds();
    /// True when MPI has been initialized (by mpi4py, ordinarily).
    static bool mpi_initialized();
    /// Rank/size of MPI_COMM_WORLD without needing an engine; -1 if no MPI.
    static int world_rank();
    static int world_size();

   private:
    /// Refuse basis sets and task shapes whose answer would be wrong, not slow.
    void check_supported(std::shared_ptr<BasisSet> primary, size_t nmats, bool are_symm) const;

    struct Impl;
    std::unique_ptr<Impl> impl_;

    size_t nmats_ = 1;
    bool are_symm_ = true;
    double cutoff_ = 0.0;
    size_t fock_builds_ = 0;
    bool density_was_nonzero_ = false;
    int nbf_ = 0;
    int mpi_rank_ = 0;
    int mpi_size_ = 1;
    int nprow_ = 1;
    int npcol_ = 1;
    int start_row_ = 0;
    int end_row_ = 0;
    int start_col_ = 0;
    int end_col_ = 0;
    int nblks_row_ = 0;
    int nblks_col_ = 0;
    int ntasks_ = 0;
};

}  // namespace psi

#endif  // PSI4_LIBFOCK_GTFOCK_INTERFACE_H
