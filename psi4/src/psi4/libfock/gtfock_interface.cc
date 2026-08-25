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

#include "psi4/libfock/gtfock_interface.h"

#include "psi4/libpsi4util/exception.h"

#ifdef USING_GTFock

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include <mpi.h>

#include "psi4/libmints/basisset.h"
#include "psi4/libmints/gshell.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/psi4-dec.h"

extern "C" {
#include "CInt.h"
#include "GTMatrix.h"
#include "pfock.h"
}

namespace psi {

namespace {

/// Process-wide tally, so a test can prove GTFock actually ran rather than
/// Psi4 quietly falling back to its own integrals.
size_t gtfock_total_fock_builds = 0;

/// Snapshot of the most recent engine's decomposition. GTFock hands each rank a
/// different AO block, so exposing these lets a multi-rank test show the work
/// really was split rather than replicated.
std::vector<int> gtfock_last_grid{-1, -1};
std::vector<int> gtfock_last_block{-1, -1, -1, -1};

/*! Factor GTFock's process count into the most square nprow x npcol grid, the
 *  same way GTFock's own SCF driver does. GTFock requires nprow*npcol to equal
 *  the communicator size exactly, so a prime rank count degenerates to 1 x N. */
void split_procs(int nprocs, int& nprow, int& npcol) {
    nprow = static_cast<int>(std::floor(std::sqrt(static_cast<double>(nprocs))));
    if (nprow < 1) nprow = 1;
    while (nprocs % nprow != 0) nprow--;
    npcol = nprocs / nprow;
}

/*! Everything that identifies one GTFock engine: the imported basis payload
 *  plus the engine shape. Two requests with equal keys can share an engine. */
struct GTFockEngineKey {
    int natom = 0;
    int nshell = 0;
    int nprim = 0;
    int pure = 0;
    int nbf = 0;
    size_t nmats = 1;
    bool symm = true;
    double cutoff = 0.0;
    std::vector<int> Zs, shells_per_atom, prims_per_shell, am;
    std::vector<double> x, y, z, cc, alpha;

    bool operator==(const GTFockEngineKey&) const = default;
};

/*! The one GTFock engine this process may own.
 *
 *  fock_task.c keeps the basis, the Simint handle, and every screening and
 *  blocking buffer in file-scope globals that `init_block_buf()` fills exactly
 *  once -- "Using global variables is a bad habit, but it is convenient",
 *  says GTFock's own comment -- and never refreshes. A second PFock engine therefore
 *  runs against the first engine's pointers, and destroying the first turns
 *  those into dangling reads. So: build at most one engine, keep it for the
 *  life of the process, and reuse it whenever the request matches.
 */
struct GTFockEngine {
    GTFockEngineKey key;
    BasisSet_t basis = nullptr;
    PFock_t pfock = nullptr;
    int nprow = 1;
    int npcol = 1;
    /// Whether this engine has already been checked for GTF_COMBINED_JK=ON.
    bool combined_jk_checked = false;
};

/// Deliberately never deleted. Releasing GTMatrix windows at static-destruction
/// time would run after mpi4py's atexit MPI_Finalize; MPI_Finalize reclaims
/// them anyway.
GTFockEngine* gtfock_engine = nullptr;

/*! True when a distributed GTMatrix is identically zero everywhere.
 *
 *  GTFock's one-sided get is not collective, so one rank reads the whole global
 *  matrix and broadcasts the verdict; every rank must agree, because the callers
 *  turn it into a collective throw. */
bool gtmatrix_is_zero(GTMatrix_t matrix, int nbf, int rank) {
    int is_zero = 1;
    if (rank == 0) {
        std::vector<double> buffer(static_cast<size_t>(nbf) * nbf, 0.0);
        GTM_getBlock(matrix, 0, nbf, 0, nbf, buffer.data(), nbf);
        for (double value : buffer) {
            if (value != 0.0) {
                is_zero = 0;
                break;
            }
        }
    }
    MPI_Bcast(&is_zero, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return is_zero != 0;
}

}  // namespace

struct MinimalInterface::Impl {
    GTFockEngine* engine = nullptr;  // non-owning: the process owns the engine
};

bool MinimalInterface::enabled() { return true; }

size_t MinimalInterface::total_fock_builds() { return gtfock_total_fock_builds; }

std::vector<int> MinimalInterface::process_grid() { return gtfock_last_grid; }

std::vector<int> MinimalInterface::local_block() { return gtfock_last_block; }

bool MinimalInterface::mpi_initialized() {
    int flag = 0;
    MPI_Initialized(&flag);
    return flag != 0;
}

int MinimalInterface::world_rank() {
    if (!mpi_initialized()) return -1;
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
}

int MinimalInterface::world_size() {
    if (!mpi_initialized()) return -1;
    int size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    return size;
}

void MinimalInterface::check_supported(std::shared_ptr<BasisSet> primary, size_t nmats, bool are_symm) const {
    // GTFock's GTMatrix-backed engine keeps exactly one global density matrix
    // and asserts num_dmat == 1 deep inside fock_task.c, so refuse here where
    // the message can still be useful.
    if (nmats != 1) {
        throw PSIEXCEPTION(
            "GTFock: this build drives one density matrix per Fock build, but " + std::to_string(nmats) +
            " were requested. The prototype covers closed-shell RHF (SCF_TYPE GTFOCK); use another SCF_TYPE "
            "for open-shell or multi-density work.");
    }
    // A basis imported as spherical makes libcint size shells as 2l+1 while the
    // Simint driver underneath fills Cartesian shell blocks. The counts diverge
    // above l=1, and even at l=1 the ordering does not match: Simint lays a p
    // shell out as px, py, pz while Psi4 orders pure shells by m. So GTFock
    // would read the density under permuted AO labels and return a wrong J/K,
    // silently, for a plain spherical s/p basis. Cartesian only.
    if (primary->has_puream()) {
        throw PSIEXCEPTION(
            "GTFock: spherical-harmonic basis sets are not supported; GTFock requires a Cartesian basis. "
            "GTFock's Simint path fills Cartesian shell blocks (px, py, pz) while Psi4 orders pure shells "
            "by m, and above l = 1 the two do not even agree on how many functions a shell has. Set "
            "PUREAM false, or choose a Cartesian basis set, to use SCF_TYPE GTFOCK.");
    }
    // libcint hard-codes _SIMINT_OSTEI_MAXAM to match the max angular momentum
    // the linked Simint was generated for, and derives _SIMINT_AM_PAIRS from it.
    // CInt_SIMINT_getShellpairAMIndex forms am_P * (MAXAM + 1) + am_Q with no
    // bound check, and fock_task.c subscripts an array of exactly _SIMINT_AM_PAIRS
    // ket shell-pair lists with it, so a shell above the ceiling writes past the
    // end of that array instead of failing. Refuse here.
    for (int s = 0; s < primary->nshell(); ++s) {
        const int am = primary->shell(s).am();
        if (am > _SIMINT_OSTEI_MAXAM) {
            throw PSIEXCEPTION(
                "GTFock: shell " + std::to_string(s) + " has angular momentum l = " + std::to_string(am) +
                ", above the maximum of " + std::to_string(_SIMINT_OSTEI_MAXAM) +
                " this GTFock/Simint build supports. GTFock indexes its shell-pair work lists by "
                "angular momentum against a table sized for l <= " +
                std::to_string(_SIMINT_OSTEI_MAXAM) +
                ", so a higher shell would corrupt memory rather than fail. Choose a basis set whose "
                "maximum angular momentum is at most " +
                std::to_string(_SIMINT_OSTEI_MAXAM) + " (through g functions), or use another SCF_TYPE.");
        }
    }
    // PFock_create(..., symm=0) puts GTFock in its nosymm mode, where the whole
    // post-build correction branch in PFock_computeFock is commented out
    // ("GTMatrix cannot handle this yet...") so gtm_Fmat/gtm_Kmat never get the
    // symmetrization the symm branch applies, and num_dmat2 becomes 2 while
    // fock_buf.c requires 1. That returns a wrong J/K rather than failing, so
    // refuse non-symmetric densities at the one place both GTFockJK
    // constructors and the adopt-on-first-build path funnel through.
    if (!are_symm) {
        throw PSIEXCEPTION(
            "GTFock: non-symmetric densities (C_left != C_right) are not supported. GTFock's nosymm mode "
            "skips the symmetrization its Fock build depends on, so it would return a wrong J/K. SOSCF, "
            "stability analysis, and response-type builds need another SCF_TYPE.");
    }
}

MinimalInterface::MinimalInterface(std::shared_ptr<BasisSet> primary, size_t nmats, bool are_symm, double cutoff)
    : impl_(new Impl), nmats_(nmats), are_symm_(are_symm), cutoff_(cutoff) {
    check_supported(primary, nmats, are_symm);

    // GTFock refuses to build without MPI, and it is the Python layer's job to
    // have started MPI through mpi4py. Say so plainly instead of letting
    // PFock_create fail with its own terser message.
    if (!mpi_initialized()) {
        throw PSIEXCEPTION(
            "GTFock: MPI is not initialized. Drive this path from Python with `import psi4.driver.gtfock` "
            "(which imports mpi4py) under `mpirun`; see doc/sphinxman/source/gtfock.rst.");
    }
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size_);
    split_procs(mpi_size_, nprow_, npcol_);

    nbf_ = primary->nbf();
    // J and K are broadcast as one nbf*nbf block, and MPI counts are int.
    if (static_cast<long long>(nbf_) * nbf_ > 2147483647LL) {
        throw PSIEXCEPTION("GTFock: nbf is too large for this prototype's single-block J/K broadcast.");
    }

    // <<< Marshal Psi4's basis and molecule into GTFock's CInt layout >>>
    // CInt_importBasisSet expects raw, .gbs-style contraction coefficients:
    // GTFock (for spherical) and Simint (always) apply their own normalization
    // on top. Psi4 keeps those in original_coef(); coef() is already normalized.
    GTFockEngineKey key;
    key.nmats = nmats;
    key.symm = are_symm;
    key.cutoff = cutoff_;
    key.nbf = nbf_;
    key.pure = primary->has_puream() ? 1 : 0;

    std::shared_ptr<Molecule> molecule = primary->molecule();
    key.natom = molecule->natom();
    key.nshell = primary->nshell();
    key.nprim = primary->nprimitive();

    key.Zs.resize(key.natom);
    key.shells_per_atom.resize(key.natom);
    key.x.resize(key.natom);
    key.y.resize(key.natom);
    key.z.resize(key.natom);
    for (int a = 0; a < key.natom; ++a) {
        key.Zs[a] = static_cast<int>(molecule->Z(a));
        key.x[a] = molecule->x(a);
        key.y[a] = molecule->y(a);
        key.z[a] = molecule->z(a);
        key.shells_per_atom[a] = primary->nshell_on_center(a);
    }

    key.prims_per_shell.resize(key.nshell);
    key.am.resize(key.nshell);
    key.cc.reserve(key.nprim);
    key.alpha.reserve(key.nprim);
    for (int s = 0; s < key.nshell; ++s) {
        const GaussianShell& shell = primary->shell(s);
        key.prims_per_shell[s] = shell.nprimitive();
        key.am[s] = shell.am();
        for (int p = 0; p < shell.nprimitive(); ++p) {
            key.alpha.push_back(shell.exp(p));
            key.cc.push_back(shell.original_coef(p));
        }
    }

    // <<< Reuse this process's engine, or build the one it gets >>>
    if (gtfock_engine != nullptr) {
        if (!(gtfock_engine->key == key)) {
            throw PSIEXCEPTION(
                "GTFock: this process already built a GTFock engine for a different basis, screening "
                "tolerance, or task shape. "
                "GTFock caches its basis and integral buffers in global state that it fills once per "
                "process, so only one engine can exist. Run the second system in a fresh process.");
        }
    } else {
        auto* engine = new GTFockEngine();
        engine->key = key;
        engine->nprow = nprow_;
        engine->npcol = npcol_;

        if (CInt_createBasisSet(&engine->basis) != CINT_STATUS_SUCCESS) {
            delete engine;
            throw PSIEXCEPTION("GTFock: CInt_createBasisSet failed.");
        }
        if (CInt_importBasisSet(engine->basis, key.natom, key.Zs.data(), key.x.data(), key.y.data(), key.z.data(),
                                key.nprim, key.nshell, key.pure, key.shells_per_atom.data(), key.prims_per_shell.data(),
                                key.am.data(), key.cc.data(), key.alpha.data()) != CINT_STATUS_SUCCESS) {
            CInt_destroyBasisSet(engine->basis);
            delete engine;
            throw PSIEXCEPTION("GTFock: CInt_importBasisSet failed.");
        }
        if (CInt_getNumFuncs(engine->basis) != nbf_) {
            const int got = CInt_getNumFuncs(engine->basis);
            CInt_destroyBasisSet(engine->basis);
            delete engine;
            throw PSIEXCEPTION("GTFock: imported basis has " + std::to_string(got) + " functions but Psi4's has " +
                               std::to_string(nbf_) + ".");
        }

        // A negative task count lets GTFock pick its own task blocking. The
        // screening tolerance is the JK object's own cutoff, handed over
        // verbatim: GTFock and Psi4 share one screening convention. Both store
        // the unsquared shell-pair maximum max|(MN|MN)| and test the product of
        // two of them against the squared threshold, so INTS_TOLERANCE maps
        // straight onto GTFock's tolscr with no conversion. GTFock additionally
        // weights that product by the largest relevant density element, which
        // Psi4's default Schwarz/CSAM screening does not, so at the same
        // tolerance GTFock screens somewhat more aggressively.
        if (PFock_create(engine->basis, nprow_, npcol_, -1, cutoff_, static_cast<int>(nmats_), are_symm_ ? 1 : 0,
                         &engine->pfock) != PFOCK_STATUS_SUCCESS) {
            CInt_destroyBasisSet(engine->basis);
            delete engine;
            throw PSIEXCEPTION(
                "GTFock: PFock_create failed. Check that the process count factors into a grid no "
                "larger than the shell count.");
        }
        // GTFock's own SCF driver sets these two on the public struct rather
        // than through a setter; fock_task.c asserts on num_dmat.
        engine->pfock->num_dmat = static_cast<int>(nmats_);
        engine->pfock->num_dmat2 = static_cast<int>(nmats_) * (engine->pfock->nosymm + 1);

        gtfock_engine = engine;
    }

    impl_->engine = gtfock_engine;
    nprow_ = gtfock_engine->nprow;
    npcol_ = gtfock_engine->npcol;
    start_row_ = gtfock_engine->pfock->sfunc_row;
    end_row_ = gtfock_engine->pfock->efunc_row;
    start_col_ = gtfock_engine->pfock->sfunc_col;
    end_col_ = gtfock_engine->pfock->efunc_col;
    gtfock_last_grid = {nprow_, npcol_};
    gtfock_last_block = {start_row_, end_row_, start_col_, end_col_};
}

// The engine outlives every MinimalInterface: see GTFockEngine for why it is
// neither rebuilt nor released.
MinimalInterface::~MinimalInterface() = default;

void MinimalInterface::SetP(const std::vector<std::shared_ptr<Matrix>>& Ps) {
    if (Ps.size() != nmats_) {
        throw PSIEXCEPTION("GTFock: expected " + std::to_string(nmats_) + " density matrices, got " +
                           std::to_string(Ps.size()) + ".");
    }
    const std::shared_ptr<Matrix>& P = Ps[0];
    if (P->rowspi()[0] != nbf_ || P->colspi()[0] != nbf_) {
        throw PSIEXCEPTION("GTFock: density matrix is not nbf x nbf in C1.");
    }

    // Psi4 replicates the density on every rank, so a single writer avoids
    // redundant one-sided traffic; the sync makes it visible to all ranks. The
    // put covers the whole global matrix, so pre-zeroing would be both dead work
    // and a race against rank 0's one-sided writes into the other ranks' blocks.
    if (mpi_rank_ == 0) {
        GTM_putBlock(impl_->engine->pfock->gtm_Dmat, 0, nbf_, 0, nbf_, P->pointer(0)[0], nbf_);
    }
    GTM_sync(impl_->engine->pfock->gtm_Dmat);

    // Remembered so the combined-JK probe below can tell a genuinely zero
    // exchange (zero density) from a GTFock built without a separate K matrix.
    density_was_nonzero_ = P->absmax() != 0.0;

    if (PFock_computeFock(impl_->engine->basis, impl_->engine->pfock) != PFOCK_STATUS_SUCCESS) {
        throw PSIEXCEPTION("GTFock: PFock_computeFock failed.");
    }
    // Everything below reads the result matrices one-sidedly out of other ranks'
    // blocks, so no rank may run ahead of another's last local write to them.
    MPI_Barrier(MPI_COMM_WORLD);
    ++fock_builds_;
    ++gtfock_total_fock_builds;

    // A GTFock built with GTF_COMBINED_JK=ON folds exchange into gtm_Fmat and
    // leaves gtm_Kmat at zero, so GetJ would hand back J - K/2 as the Coulomb
    // matrix. Detect it here rather than in GetK: do_K_ is false for any pure
    // functional, and that path never calls GetK at all. A nonzero density
    // always gives a nonzero K, so one probe per engine settles it.
    if (density_was_nonzero_ && !impl_->engine->combined_jk_checked) {
        if (gtmatrix_is_zero(impl_->engine->pfock->gtm_Kmat, nbf_, mpi_rank_)) {
            throw PSIEXCEPTION(
                "GTFock returned an identically zero exchange matrix, which means it was built with "
                "GTF_COMBINED_JK=ON. Rebuild GTFock with -DGTF_COMBINED_JK=OFF so J and K come back "
                "separately.");
        }
        impl_->engine->combined_jk_checked = true;
    }
}

namespace {

/*! Fetch the whole global matrix from a GTMatrix on rank 0, broadcast it, and
 *  write it into `result` scaled into Psi4's convention.
 *
 *  GTFock's one-sided get is not collective, so having every rank pull the same
 *  global block would serialize on the owners; one reader plus a broadcast is
 *  both cheaper and what GTFock's own tests do. */
void gather_scaled(std::shared_ptr<Matrix>& result, GTMatrix_t source, int nbf, int rank, double scale) {
    std::vector<double> buffer(static_cast<size_t>(nbf) * nbf, 0.0);
    if (rank == 0) {
        GTM_getBlock(source, 0, nbf, 0, nbf, buffer.data(), nbf);
    }
    MPI_Bcast(buffer.data(), nbf * nbf, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double** rp = result->pointer(0);
    for (int i = 0; i < nbf; ++i) {
        for (int j = 0; j < nbf; ++j) {
            rp[i][j] = scale * buffer[static_cast<size_t>(i) * nbf + j];
        }
    }
}

}  // namespace

void MinimalInterface::GetJ(std::vector<std::shared_ptr<Matrix>>& Js) {
    if (Js.size() != nmats_) {
        throw PSIEXCEPTION("GTFock: expected " + std::to_string(nmats_) + " J matrices, got " +
                           std::to_string(Js.size()) + ".");
    }
    // GTFock accumulates 2J into gtm_Fmat (its "Fock" matrix in the separate-K
    // build), so halve it to reach Psi4's J.
    gather_scaled(Js[0], impl_->engine->pfock->gtm_Fmat, nbf_, mpi_rank_, 0.5);
}

void MinimalInterface::GetK(std::vector<std::shared_ptr<Matrix>>& Ks) {
    if (Ks.size() != nmats_) {
        throw PSIEXCEPTION("GTFock: expected " + std::to_string(nmats_) + " K matrices, got " +
                           std::to_string(Ks.size()) + ".");
    }
    // gtm_Kmat holds -K, ready to be added to 2J; flip the sign for Psi4's K.
    gather_scaled(Ks[0], impl_->engine->pfock->gtm_Kmat, nbf_, mpi_rank_, -1.0);
}

}  // namespace psi

#else  // USING_GTFock

namespace psi {

struct MinimalInterface::Impl {};

bool MinimalInterface::enabled() { return false; }
size_t MinimalInterface::total_fock_builds() { return 0; }
std::vector<int> MinimalInterface::process_grid() { return {-1, -1}; }
std::vector<int> MinimalInterface::local_block() { return {-1, -1, -1, -1}; }
bool MinimalInterface::mpi_initialized() { return false; }
int MinimalInterface::world_rank() { return -1; }
int MinimalInterface::world_size() { return -1; }

void MinimalInterface::check_supported(std::shared_ptr<BasisSet>, size_t, bool) const {}

MinimalInterface::MinimalInterface(std::shared_ptr<BasisSet>, size_t, bool, double) {
    throw PSIEXCEPTION("Psi4 was not compiled with GTFock support. Reconfigure with -DENABLE_GTFock=ON.");
}
MinimalInterface::~MinimalInterface() = default;
void MinimalInterface::SetP(const std::vector<std::shared_ptr<Matrix>>&) {}
void MinimalInterface::GetJ(std::vector<std::shared_ptr<Matrix>>&) {}
void MinimalInterface::GetK(std::vector<std::shared_ptr<Matrix>>&) {}

}  // namespace psi

#endif  // USING_GTFock
