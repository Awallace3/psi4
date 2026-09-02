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

#include "psi4/libfock/gtfock_df_interface.h"

#include "psi4/libpsi4util/exception.h"

#ifdef USING_GTFock_DF

#include <string>
#include <utility>
#include <vector>

#include <mpi.h>

#include "psi4/libmints/basisset.h"
#include "psi4/libmints/gshell.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/molecule.h"

extern "C" {
#include "CInt.h"
#include "gtfock_df.h"
#include "gtfock_pdf.h"
}

namespace psi {

namespace {

/// Process-wide tally, so a test can prove the distributed DF engine actually
/// ran rather than Psi4 quietly falling back to MemDFJK.
size_t gtfockdf_total_jk_builds = 0;

/// Snapshot of the most recent engine's partition, for the same reason the
/// exact path snapshots its process grid: on more than one rank, nlocal_aux
/// differing between ranks and summing to naux is the evidence the fitted
/// tensor was distributed rather than replicated.
std::vector<int> gtfockdf_last_partition{-1, -1, -1, -1, -1};
size_t gtfockdf_last_local_tensor_doubles = 0;

/// Where the most recent engine build spent its wall time, by phase. Setup is a
/// third of SCF here and does not scale like the integrals do, so which phase
/// it is matters; see doc/sphinxman/source/gtfock.rst.
std::vector<std::pair<std::string, double>> gtfockdf_last_setup_phases;

/*! Marshal one Psi4 BasisSet into GTFock's CInt layout.
 *
 *  CInt_importBasisSet wants raw, .gbs-style contraction coefficients, because
 *  Simint applies its own normalization on top; Psi4 keeps those in
 *  original_coef(), while coef() is already normalized. The DF engine imports
 *  two basis sets this way, the orbital basis and the fitting basis, and both
 *  go through here.
 *
 *  Returns a basis the caller owns and must CInt_destroyBasisSet. */
BasisSet_t import_basis(const std::shared_ptr<BasisSet>& basis, const char* what) {
    std::shared_ptr<Molecule> molecule = basis->molecule();
    const int natom = molecule->natom();
    const int nshell = basis->nshell();
    const int nprim = basis->nprimitive();

    std::vector<int> Zs(natom), shells_per_atom(natom);
    std::vector<double> x(natom), y(natom), z(natom);
    for (int a = 0; a < natom; ++a) {
        Zs[a] = static_cast<int>(molecule->Z(a));
        x[a] = molecule->x(a);
        y[a] = molecule->y(a);
        z[a] = molecule->z(a);
        shells_per_atom[a] = basis->nshell_on_center(a);
    }

    std::vector<int> prims_per_shell(nshell), am(nshell);
    std::vector<double> cc, alpha;
    cc.reserve(nprim);
    alpha.reserve(nprim);
    for (int s = 0; s < nshell; ++s) {
        const GaussianShell& shell = basis->shell(s);
        prims_per_shell[s] = shell.nprimitive();
        am[s] = shell.am();
        for (int p = 0; p < shell.nprimitive(); ++p) {
            alpha.push_back(shell.exp(p));
            cc.push_back(shell.original_coef(p));
        }
    }

    BasisSet_t imported = nullptr;
    if (CInt_createBasisSet(&imported) != CINT_STATUS_SUCCESS) {
        throw PSIEXCEPTION(std::string("GTFock DF: CInt_createBasisSet failed for the ") + what + " basis.");
    }
    if (CInt_importBasisSet(imported, natom, Zs.data(), x.data(), y.data(), z.data(), nprim, nshell, 0,
                            shells_per_atom.data(), prims_per_shell.data(), am.data(), cc.data(),
                            alpha.data()) != CINT_STATUS_SUCCESS) {
        CInt_destroyBasisSet(imported);
        throw PSIEXCEPTION(std::string("GTFock DF: CInt_importBasisSet failed for the ") + what + " basis.");
    }
    if (CInt_getNumFuncs(imported) != basis->nbf()) {
        const int got = CInt_getNumFuncs(imported);
        CInt_destroyBasisSet(imported);
        throw PSIEXCEPTION(std::string("GTFock DF: the imported ") + what + " basis has " + std::to_string(got) +
                           " functions but Psi4's has " + std::to_string(basis->nbf()) + ".");
    }
    return imported;
}

/*! Refuse any matrix that is not one dense C1 block of the expected shape.
 *
 *  Every transfer here hands PDF_computeJK `Matrix::pointer(0)[0]` and an
 *  implied length. pointer(0) returns the first irrep block, so a
 *  symmetry-blocked or mis-sized matrix would be read or written past its end
 *  rather than refused. GTFockDFJK::C1() is true, so libfock hands it AO
 *  matrices and any other shape is a caller error worth naming. */
void check_c1_shape(const std::shared_ptr<Matrix>& mat, int rows, int cols, const char* what) {
    if (!mat) {
        throw PSIEXCEPTION(std::string("GTFock DF: ") + what + " is null.");
    }
    if (mat->nirrep() != 1 || mat->rowspi()[0] != rows || mat->colspi()[0] != cols) {
        throw PSIEXCEPTION(std::string("GTFock DF: ") + what + " must be a single " + std::to_string(rows) + " x " +
                           std::to_string(cols) + " C1 block.");
    }
}

}  // namespace

struct MinimalDFInterface::Impl {
    BasisSet_t primary = nullptr;
    BasisSet_t auxiliary = nullptr;
    PDF_t pdf = nullptr;
};

bool MinimalDFInterface::enabled() { return true; }

size_t MinimalDFInterface::total_jk_builds() { return gtfockdf_total_jk_builds; }

std::vector<int> MinimalDFInterface::last_partition() { return gtfockdf_last_partition; }

size_t MinimalDFInterface::last_local_tensor_doubles() { return gtfockdf_last_local_tensor_doubles; }

std::vector<std::pair<std::string, double>> MinimalDFInterface::last_setup_phases() {
    return gtfockdf_last_setup_phases;
}

void MinimalDFInterface::check_supported(const std::shared_ptr<BasisSet>& primary,
                                         const std::shared_ptr<BasisSet>& auxiliary) const {
    // The Simint driver under GTFDF fills Cartesian shell blocks while Psi4
    // orders pure shells by m; above l = 1 the two do not even agree on how many
    // functions a shell has, and at l = 1 the ordering already differs (Simint
    // lays out px, py, pz). A spherical basis would therefore give a wrong J/K
    // silently, so refuse both basis sets unless they are Cartesian.
    const BasisSet* bases[2] = {primary.get(), auxiliary.get()};
    const char* labels[2] = {"orbital", "fitting"};
    for (int b = 0; b < 2; ++b) {
        if (bases[b]->has_puream()) {
            throw PSIEXCEPTION(
                std::string("GTFock DF: the ") + labels[b] +
                " basis is spherical-harmonic; GTFock's distributed DF engine requires Cartesian basis sets. "
                "Its Simint path fills Cartesian shell blocks (px, py, pz) while Psi4 orders pure shells by m. "
                "Set PUREAM false, or choose Cartesian orbital and fitting basis sets, to use SCF_TYPE "
                "GTFOCK_DF.");
        }
    }
    // Simint dispatches three- and two-center integrals through generated
    // tables indexed by angular momentum, with no bound check, so a shell above
    // the ceiling the linked Simint was generated for reads past them rather
    // than failing. GTFDF_create refuses such a basis too, but only after the
    // import; screen here where the message can name the shell. Note this asks
    // Simint itself through GTFDF_maxSupportedAM(), not libcint's separately
    // hardcoded _SIMINT_OSTEI_MAXAM, which can be lower and does not bound this
    // engine.
    const int max_am = GTFDF_maxSupportedAM();
    for (int b = 0; b < 2; ++b) {
        for (int s = 0; s < bases[b]->nshell(); ++s) {
            const int am = bases[b]->shell(s).am();
            if (am > max_am) {
                throw PSIEXCEPTION(std::string("GTFock DF: shell ") + std::to_string(s) + " of the " + labels[b] +
                                   " basis has angular momentum l = " + std::to_string(am) +
                                   ", above the maximum of " + std::to_string(max_am) +
                                   " the Simint linked into this GTFock was generated for. Simint indexes its "
                                   "integral tables by angular momentum with no bound check, so a higher shell "
                                   "would read past them rather than fail. Choose basis sets with l <= " +
                                   std::to_string(max_am) + ", or use another SCF_TYPE.");
            }
        }
    }
    // Fitting bases are commonly larger than the orbital basis, and J and K come
    // back as replicated nbf x nbf blocks through an MPI_Allreduce whose count
    // is an int.
    if (static_cast<long long>(primary->nbf()) * primary->nbf() > 2147483647LL) {
        throw PSIEXCEPTION("GTFock DF: nbf is too large for this prototype's replicated nbf x nbf J/K.");
    }
}

MinimalDFInterface::MinimalDFInterface(std::shared_ptr<BasisSet> primary, std::shared_ptr<BasisSet> auxiliary,
                                       double fitting_condition, int nthreads)
    : impl_(new Impl), fitting_condition_(fitting_condition), nthreads_(nthreads) {
    if (!primary || !auxiliary) {
        throw PSIEXCEPTION("GTFock DF: both an orbital and a fitting basis are required.");
    }
    // An empty fitting basis is what Psi4 hands a JK object that never asked for
    // one: proc.py sets DF_BASIS_SCF to the zero basis unless SCF_TYPE is on its
    // list of fitted methods. Name that rather than letting the metric inversion
    // fail on a rank-zero matrix.
    if (auxiliary->nbf() == 0) {
        throw PSIEXCEPTION(
            "GTFock DF: the fitting basis is empty. SCF_TYPE GTFOCK_DF needs DF_BASIS_SCF; this usually means "
            "the JK object was built with BasisSet::zero_ao_basis_set() by a code path that has not been "
            "taught about GTFOCK_DF.");
    }
    check_supported(primary, auxiliary);

    // The engine is collective and it is the Python layer's job to have started
    // MPI through mpi4py. Say so plainly rather than letting PDF_create fail
    // inside MPI_Comm_dup.
    int mpi_started = 0;
    MPI_Initialized(&mpi_started);
    if (mpi_started == 0) {
        throw PSIEXCEPTION(
            "GTFock DF: MPI is not initialized. Drive this path from Python with `import psi4.driver.gtfock` "
            "(which imports mpi4py) under `mpirun`; see doc/sphinxman/source/gtfock.rst.");
    }
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size_);

    nbf_ = primary->nbf();
    naux_ = auxiliary->nbf();

    impl_->primary = import_basis(primary, "orbital");
    try {
        impl_->auxiliary = import_basis(auxiliary, "fitting");
    } catch (...) {
        CInt_destroyBasisSet(impl_->primary);
        impl_->primary = nullptr;
        throw;
    }

    // This is the expensive call: all three-center integrals, the Coulomb
    // metric and its factorization, the fit, and the redistribution that gives
    // each rank its own slice of Q. Unlike the exact path's PFock engine there is no global state
    // behind it, so it may be built here, destroyed, and built again.
    if (PDF_create(MPI_COMM_WORLD, impl_->primary, impl_->auxiliary, nthreads_, fitting_condition_, &impl_->pdf) !=
        CINT_STATUS_SUCCESS) {
        CInt_destroyBasisSet(impl_->auxiliary);
        CInt_destroyBasisSet(impl_->primary);
        impl_->auxiliary = nullptr;
        impl_->primary = nullptr;
        throw PSIEXCEPTION(
            "GTFock DF: PDF_create failed. Check that the orbital and fitting basis sets are Cartesian and "
            "that enough memory is available for this rank's slice of the fitted tensor.");
    }

    nlocal_aux_ = PDF_nLocalAuxFuncs(impl_->pdf);
    nmetric_null_ = PDF_nMetricNullVectors(impl_->pdf);
    nlocal_pairs_ = PDF_nLocalPairElements(impl_->pdf);
    local_tensor_doubles_ = PDF_localTensorSize(impl_->pdf);
    // Read the phase clocks now rather than on demand: the engine may be
    // destroyed long before a benchmark asks, and PDF_phaseName is the only
    // place these names live.
    setup_phases_.reserve(PDF_NPHASES);
    for (int phase = 0; phase < PDF_NPHASES; ++phase) {
        const PDF_Phase p = static_cast<PDF_Phase>(phase);
        const char* name = PDF_phaseName(p);
        setup_phases_.emplace_back(name != nullptr ? name : "?", PDF_phaseSeconds(impl_->pdf, p));
    }
    gtfockdf_last_partition = {nbf_, naux_, nlocal_aux_, nmetric_null_, nlocal_pairs_};
    gtfockdf_last_local_tensor_doubles = local_tensor_doubles_;
    gtfockdf_last_setup_phases = setup_phases_;
}

MinimalDFInterface::~MinimalDFInterface() {
    if (impl_->pdf != nullptr) {
        // PDF_destroy frees the communicator PDF_create duplicated, and
        // MPI_Comm_free is collective and illegal after MPI_Finalize. Psi4's JK
        // objects are held by Python, whose garbage collector may well run after
        // mpi4py's atexit handler has finalized MPI, so leak the engine in that
        // case: the process is on its way out and MPI_Finalize has already
        // reclaimed the communicator.
        int finalized = 0;
        MPI_Finalized(&finalized);
        if (!finalized) {
            PDF_destroy(impl_->pdf);
        }
        impl_->pdf = nullptr;
    }
    if (impl_->auxiliary != nullptr) CInt_destroyBasisSet(impl_->auxiliary);
    if (impl_->primary != nullptr) CInt_destroyBasisSet(impl_->primary);
}

void MinimalDFInterface::compute_JK(const std::shared_ptr<Matrix>& D, const std::shared_ptr<Matrix>& Cocc,
                                    const std::shared_ptr<Matrix>& J, const std::shared_ptr<Matrix>& K) {
    const double* Dp = nullptr;
    const double* Coccp = nullptr;
    double* Jp = nullptr;
    double* Kp = nullptr;
    int nocc = 0;

    if (J) {
        check_c1_shape(J, nbf_, nbf_, "J matrix");
        Jp = J->pointer(0)[0];
    }
    if (K) {
        check_c1_shape(K, nbf_, nbf_, "K matrix");
        Kp = K->pointer(0)[0];
    }
    // D is what the Coulomb contraction needs; Cocc is what the exchange
    // contraction needs. Ask for each only when the corresponding output was
    // requested, so a pure functional never has to supply orbitals.
    if (Jp != nullptr) {
        check_c1_shape(D, nbf_, nbf_, "density matrix");
        Dp = D->pointer(0)[0];
    }
    if (Kp != nullptr) {
        if (!Cocc) {
            throw PSIEXCEPTION("GTFock DF: exchange was requested without occupied orbital coefficients.");
        }
        nocc = Cocc->colspi()[0];
        check_c1_shape(Cocc, nbf_, nocc, "occupied coefficient matrix");
        // A zero-column Cocc means a zero density and hence a zero K. GTFock
        // would handle it, but the caller has already zeroed K, so skip the
        // collective rather than reduce nbf^2 zeros.
        if (nocc == 0) {
            Kp = nullptr;
        } else {
            Coccp = Cocc->pointer(0)[0];
        }
    }
    if (Jp == nullptr && Kp == nullptr) return;

    if (PDF_computeJK(impl_->pdf, Dp, Coccp, nocc, Jp, Kp) != CINT_STATUS_SUCCESS) {
        throw PSIEXCEPTION("GTFock DF: PDF_computeJK failed.");
    }
    ++jk_builds_;
    ++gtfockdf_total_jk_builds;
}

}  // namespace psi

#else  // USING_GTFock_DF

namespace psi {

struct MinimalDFInterface::Impl {};

bool MinimalDFInterface::enabled() { return false; }
size_t MinimalDFInterface::total_jk_builds() { return 0; }
std::vector<int> MinimalDFInterface::last_partition() { return {-1, -1, -1, -1, -1}; }
size_t MinimalDFInterface::last_local_tensor_doubles() { return 0; }
std::vector<std::pair<std::string, double>> MinimalDFInterface::last_setup_phases() { return {}; }

void MinimalDFInterface::check_supported(const std::shared_ptr<BasisSet>&, const std::shared_ptr<BasisSet>&) const {}

MinimalDFInterface::MinimalDFInterface(std::shared_ptr<BasisSet>, std::shared_ptr<BasisSet>, double, int) {
    throw PSIEXCEPTION(
        "Psi4 was not compiled with GTFock density-fitting support. SCF_TYPE GTFOCK_DF needs a GTFock "
        "installation that ships libgtfockdf and gtfock_pdf.h; reconfigure with -DENABLE_GTFock=ON against "
        "one that does.");
}
MinimalDFInterface::~MinimalDFInterface() = default;
void MinimalDFInterface::compute_JK(const std::shared_ptr<Matrix>&, const std::shared_ptr<Matrix>&,
                                    const std::shared_ptr<Matrix>&, const std::shared_ptr<Matrix>&) {}

}  // namespace psi

#endif  // USING_GTFock_DF
