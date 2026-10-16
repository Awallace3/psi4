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

#include "psi4/libqt/qt.h"
#include "psi4/psi4-dec.h"
#include "psi4/libmints/vector.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/basisset.h"
#include "psi4/libmints/integral.h"
#include "psi4/libmints/eri.h"
#include "psi4/libmints/coordentry.h"
#include "psi4/libpsi4util/exception.h"
#include "psi4/libsapt_solver/fdds_disp.h"
#include "psi4/libmints/3coverlap.h"
#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/liboptions/liboptions.h"
#include "psi4/libpsi4util/process.h"
#include "psi4/lib3index/dfhelper.h"

#include <filesystem>
#include <iomanip>
#include <limits>
#include <system_error>

#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

// OMP
#ifdef _OPENMP
#include <omp.h>
#endif

namespace psi {

namespace sapt {

namespace {

size_t cmul(size_t a, size_t b) {
    size_t r;
    if (__builtin_mul_overflow(a, b, &r)) throw PSIEXCEPTION("FDDS: resource size overflows size_t.");
    return r;
}
size_t cadd(std::initializer_list<size_t> terms) {
    size_t r = 0;
    for (size_t t : terms)
        if (__builtin_add_overflow(r, t, &r)) throw PSIEXCEPTION("FDDS: resource size overflows size_t.");
    return r;
}
size_t cmax(std::initializer_list<size_t> terms) { return std::max(terms); }
size_t doubles_for(size_t bytes) { return bytes / sizeof(double) + (bytes % sizeof(double) != 0); }

// Declared-path storage beside the numerical payload, in doubles. A C1 Matrix also allocates one
// row pointer per row (block_matrix); each is rounded up to whole doubles, so the charge is an
// upper bound whatever the pointer width. Every DFHelper holds its Qshell_aggs_/pshell_aggs_ shell
// offsets (nshell + 1 size_t each) from construction.
constexpr size_t kRowPointer = (sizeof(double*) + sizeof(double) - 1) / sizeof(double);
size_t mat(size_t rows, size_t cols) { return cmul(rows, cadd({cols, kRowPointer})); }
size_t dfh_offsets(const BasisSet& primary, const BasisSet& auxiliary) {
    return doubles_for(cmul(cadd({(size_t)primary.nshell(), (size_t)auxiliary.nshell(), 2}), sizeof(size_t)));
}

// Existing directory the process may create files in, checked without creating anything.
// POSIX asks access(W_OK | X_OK); Windows _access(.., 2) on a directory reports existence only,
// so the read-only attribute is checked there as well.
bool writable_directory(const std::string& path) {
    if (path.empty()) return false;
    std::error_code ec;
    if (!std::filesystem::is_directory(std::filesystem::path(path), ec) || ec) return false;
#ifdef _WIN32
    const auto perms = std::filesystem::status(std::filesystem::path(path), ec).permissions();
    if (ec || (perms & std::filesystem::perms::owner_write) == std::filesystem::perms::none) return false;
    return _access(path.c_str(), 2) == 0;
#else
    return access(path.c_str(), W_OK | X_OK) == 0;
#endif
}

// (P|Q) over one auxiliary basis, filled symmetrically; the legacy constructor loop. The declared
// path asks for Libint2ERI directly (IntegralFactory::eri()'s arguments), whatever INTEGRAL_PACKAGE says.
SharedMatrix form_coulomb_metric(std::shared_ptr<BasisSet> auxiliary_, size_t nthread, bool libint2 = false) {
    size_t naux = auxiliary_->nbf();
    auto metric = std::make_shared<Matrix>("Inv Coulomb Metric", naux, naux);

    std::shared_ptr<BasisSet> zero = BasisSet::zero_ao_basis_set();

    // ==> (P|Q) Metric <==
    IntegralFactory metric_factory(auxiliary_, zero, auxiliary_, zero);

    std::vector<std::shared_ptr<TwoBodyAOInt>> metric_ints(nthread);
    std::vector<const double*> metric_buff(nthread);
    for (size_t thread = 0; thread < nthread; thread++) {
        metric_ints[thread] = libint2 ? std::make_shared<Libint2ERI>(
                                            &metric_factory,
                                            Process::environment.options.get_double("INTS_TOLERANCE"), 0, true, false)
                                      : std::shared_ptr<TwoBodyAOInt>(metric_factory.eri());
        metric_buff[thread] = metric_ints[thread]->buffer();
    }

    double** metricp = metric->pointer();

#pragma omp parallel for schedule(dynamic) num_threads(nthread)
    for (size_t MU = 0; MU < auxiliary_->nshell(); ++MU) {
        size_t nummu = auxiliary_->shell(MU).nfunction();

        size_t thread = 0;
#ifdef _OPENMP
        thread = omp_get_thread_num();
#endif

        // Triangular
        for (size_t NU = 0; NU <= MU; ++NU) {
            size_t numnu = auxiliary_->shell(NU).nfunction();

            metric_ints[thread]->compute_shell(MU, 0, NU, 0);
            metric_buff[thread] = metric_ints[thread]->buffer();

            size_t index = 0;
            // #pragma simd collapse(2)
            for (size_t mu = 0; mu < nummu; ++mu) {
                size_t omu = auxiliary_->shell(MU).function_index() + mu;

                for (size_t nu = 0; nu < numnu; ++nu, ++index) {
                    size_t onu = auxiliary_->shell(NU).function_index() + nu;

                    metricp[omu][onu] = metricp[onu][omu] = metric_buff[thread][index];
                }
            }
        }
    }
    return metric;
}

SharedMatrix form_aux_overlap(std::shared_ptr<BasisSet> auxiliary) {
    IntegralFactory factory(auxiliary);
    std::shared_ptr<OneBodyAOInt> overlap(factory.ao_overlap());
    auto S = std::make_shared<Matrix>("Auxiliary Overlap", auxiliary->nbf(), auxiliary->nbf());
    overlap->compute(S);
    return S;
}

}  // namespace

FDDS_Dispersion::FDDS_Dispersion(std::shared_ptr<BasisSet> primary, std::shared_ptr<BasisSet> auxiliary,
                                 std::map<std::string, SharedMatrix> matrix_cache,
                                 std::map<std::string, SharedVector> vector_cache,
                                 bool is_hybrid)
    : FDDS_Dispersion(primary, auxiliary, matrix_cache, vector_cache, is_hybrid, false) {}

FDDS_Dispersion::FDDS_Dispersion(std::shared_ptr<BasisSet> primary, std::shared_ptr<BasisSet> auxiliary,
                               std::map<std::string, SharedMatrix> matrix_cache,
                               std::map<std::string, SharedVector> vector_cache, bool is_hybrid, bool single_monomer)
    : primary_(primary), auxiliary_(auxiliary), matrix_cache_(matrix_cache), vector_cache_(vector_cache),
      is_hybrid_(is_hybrid), single_monomer_(single_monomer), naux_(auxiliary ? auxiliary->nbf() : 0) {
    validate_orbitals();

    // ==> Form Metric <==

    timer_on("Form JS");
    size_t metric_threads = 1;
#ifdef _OPENMP
    metric_threads = omp_get_max_threads();
#endif
    metric_ = form_coulomb_metric(auxiliary_, metric_threads);

    metric_inv_ = metric_->clone();
    metric_inv_->power(-1.0, 1.e-12);

    // ==> Form Aux overlap <==

    aux_overlap_ = form_aux_overlap(auxiliary_);
    timer_off("Form JS");

    // ==> Form 3-index object <==

    // Build C Stack
    std::vector<SharedMatrix> Cstack_vec;
    Cstack_vec.push_back(matrix_cache_["Cocc_A"]);
    Cstack_vec.push_back(matrix_cache_["Cvir_A"]);
    if (!single_monomer_) {
        Cstack_vec.push_back(matrix_cache_["Cocc_B"]);
        Cstack_vec.push_back(matrix_cache_["Cvir_B"]);
    }

    size_t nthread = 1;
#ifdef _OPENMP
    nthread = omp_get_max_threads();
#endif
    size_t doubles = budget_doubles();
    size_t max_MO = 0;
    for (auto& mat : Cstack_vec) max_MO = std::max(max_MO, (size_t)mat->ncol());

    // Build DFHelper
    dfh_ = std::make_shared<DFHelper>(primary_, auxiliary_);
    dfh_->set_memory(doubles);
    if (is_hybrid_) {
        dfh_->set_method("DIRECT");
    } else {
        dfh_->set_method("DIRECT_iaQ");
    }
    dfh_->set_nthreads(nthread);
    dfh_->set_metric_pow(0.0);
    dfh_->initialize();
    dfh_->print_header();

    // Define spaces
    dfh_->add_space("a", Cstack_vec[0]);
    dfh_->add_space("r", Cstack_vec[1]);
    if (!single_monomer_) {
        dfh_->add_space("b", Cstack_vec[2]);
        dfh_->add_space("s", Cstack_vec[3]);
    }

    // add transformations
    dfh_->add_transformation("arQ", "a", "r", "pqQ");
    if (!single_monomer_) dfh_->add_transformation("bsQ", "b", "s", "pqQ");
    if (is_hybrid_) {
        dfh_->add_transformation("raQ", "r", "a", "pqQ");
        dfh_->add_transformation("Qar", "a", "r", "Qpq");
        if (!single_monomer_) {
            dfh_->add_transformation("sbQ", "s", "b", "pqQ");
            dfh_->add_transformation("Qbs", "b", "s", "Qpq");
        }
    }

    // transform
    dfh_->set_release_core_AO_before_metric(true);
    dfh_->transform();

    // transformations specific for hybrid functional

    if (is_hybrid_) {
        // Contracted 3-index integrals to reproduce 4-index ERI
        // Clear spaces to re-order spaces and transformations in DFHelper
        // Clear transformations to avoid overwriting pqQ tensors 
        dfh_->clear_spaces();
        dfh_->clear_transformations();
        dfh_->set_method("DIRECT_iaQ");
        dfh_->set_metric_pow(-0.5);
        dfh_->initialize();

        dfh_->add_space("a", Cstack_vec[0]);
        dfh_->add_space("r", Cstack_vec[1]);
        if (!single_monomer_) {
            dfh_->add_space("b", Cstack_vec[2]);
            dfh_->add_space("s", Cstack_vec[3]);
        }

        dfh_->add_transformation("aaR", "a", "a", "pqQ");
        dfh_->add_transformation("arR", "a", "r", "pqQ");
        dfh_->add_transformation("rrR", "r", "r", "pqQ");
        if (!single_monomer_) {
            dfh_->add_transformation("bbR", "b", "b", "pqQ");
            dfh_->add_transformation("bsR", "b", "s", "pqQ");
            dfh_->add_transformation("ssR", "s", "s", "pqQ");
        }
        dfh_->set_release_core_AO_before_metric(true);
        dfh_->transform();
    }

    dfh_->clear_spaces();

    if (is_hybrid_) {
        // QR Factorization of (ar|Q)
        timer_on("FDDS: QR");
        R_A_ = QR("A");
        if (!single_monomer_) R_B_ = QR("B");
        timer_off("FDDS: QR");

        // form (ar|(Q)X|Q) = (ar'|a'r) (a'r'|(Q)|Q)
        timer_on("FDDS: Form X");
        form_X("A");
        if (!single_monomer_) form_X("B");
        timer_off("FDDS: Form X");

        // form (ar|(Q)Y|Q) = (aa'|rr') (a'r'|(Q)|Q)
        timer_on("FDDS: Form Y");
        form_Y("A");
        if (!single_monomer_) form_Y("B");
        timer_off("FDDS: Form Y");
    }

}

FDDS_Dispersion::FDDS_Dispersion(std::shared_ptr<BasisSet> primary, std::shared_ptr<BasisSet> auxiliary,
                                 std::map<std::string, SharedMatrix> matrix_cache,
                                 std::map<std::string, SharedVector> vector_cache, bool is_hybrid, Deferred)
    : primary_(primary), auxiliary_(auxiliary), matrix_cache_(matrix_cache), vector_cache_(vector_cache),
      is_hybrid_(is_hybrid), single_monomer_(true), naux_(auxiliary ? auxiliary->nbf() : 0) {
    validate_orbitals();
}

void FDDS_Dispersion::validate_orbitals() const {
    if (!primary_ || !auxiliary_ || primary_->nbf() == 0 || auxiliary_->nbf() == 0) {
        throw PSIEXCEPTION("FDDS: nonempty primary and auxiliary bases are required.");
    }
    const std::vector<std::string> monomers = single_monomer_ ? std::vector<std::string>{"A"}
                                                           : std::vector<std::string>{"A", "B"};
    for (const auto& monomer : monomers) {
        for (const std::string space : {"occ", "vir"}) {
            const auto ck = "C" + space + "_" + monomer;
            const auto ek = "eps_" + space + "_" + monomer;
            if (!matrix_cache_.count(ck) || !matrix_cache_.at(ck) ||
                !vector_cache_.count(ek) || !vector_cache_.at(ek)) {
                throw PSIEXCEPTION("FDDS: missing orbital data: " + ck + " / " + ek);
            }
            const auto& C = matrix_cache_.at(ck);
            const auto& eps = vector_cache_.at(ek);
            if (C->nirrep() != 1 || eps->nirrep() != 1 || C->nrow() != primary_->nbf() ||
                C->ncol() == 0 || C->ncol() != eps->dim(0)) {
                throw PSIEXCEPTION("FDDS: orbital dimensions must be C1, nbf by nmo, with matching nonempty energies.");
            }
            for (int p = 0; p < C->nrow(); ++p)
                for (int i = 0; i < C->ncol(); ++i)
                    if (!std::isfinite(C->get(p, i)))
                        throw PSIEXCEPTION("FDDS: orbital coefficients must be finite.");
            for (int i = 0; i < eps->dim(0); ++i)
                if (!std::isfinite(eps->get(i)))
                    throw PSIEXCEPTION("FDDS: orbital energies must be finite.");
        }
    }

}

FDDS_Dispersion::~FDDS_Dispersion() {}

size_t FDDS_Dispersion::budget_doubles() const {
    if (work_doubles_) return work_doubles_;
    return Process::environment.get_memory() * 0.8 / sizeof(double);
}

int FDDS_Dispersion::blocking_threads() const {
    return nthread_ ? nthread_ : Process::environment.get_n_threads();
}

void FDDS_Dispersion::check_monomer(const std::string& monomer, bool needs_hybrid) const {
    if (monomer != "A" && (monomer != "B" || single_monomer_)) {
        throw PSIEXCEPTION("FDDS: requested monomer is not prepared.");
    }
    if (needs_hybrid && !is_hybrid_) {
        throw PSIEXCEPTION("FDDS: exchange intermediates require hybrid preparation.");
    }
}

std::vector<SharedMatrix> FDDS_Dispersion::project_densities(std::vector<SharedMatrix> densities) {
    for (const auto& density : densities) {
        if (!density || density->nirrep() != 1 || density->nrow() != primary_->nbf() ||
            density->ncol() != primary_->nbf()) {
            throw PSIEXCEPTION("FDDS: densities must be C1 nbf by nbf matrices.");
        }
    }
    // Perform the contraction
    // (PQS) (S|R)^-1 (R|pq) Dpq -> PQ

    // ==> Contract (R|pq) Dpq -> R <== //

    std::shared_ptr<BasisSet> zero = BasisSet::zero_ao_basis_set();
    size_t nthread = 1;
#ifdef _OPENMP
    nthread = omp_get_max_threads();
#endif

    // Build integral threads
    IntegralFactory df_factory(auxiliary_, zero, primary_, primary_);

    std::vector<std::shared_ptr<TwoBodyAOInt>> df_ints(nthread);
    std::vector<const double*> df_buff(nthread);
    for (size_t thread = 0; thread < nthread; thread++) {
        df_ints[thread] = std::shared_ptr<TwoBodyAOInt>(df_factory.eri());
        df_buff[thread] = df_ints[thread]->buffer();
    }

    size_t naux = auxiliary_->nbf();
    size_t nbf = primary_->nbf();
    size_t nbf2 = nbf * nbf;

    // Pack the DF pairs
    std::vector<std::pair<size_t, size_t>> df_pairs;
    for (size_t M = 0; M < primary_->nshell(); M++) {
        for (size_t N = 0; N <= M; N++) {
            df_pairs.push_back(std::pair<size_t, size_t>(M, N));
        }
    }

    // Check on memory real quick
    size_t doubles = budget_doubles();
    size_t mem_size = nbf2 * auxiliary_->max_nprimitive() * nthread;
    if (mem_size > doubles) {
        std::stringstream message;
        double mem_gb = ((double)(mem_size) / 0.8 * sizeof(double));
        message << "FDDS Dispersion requires at least nbf^2 * max_ang * nthread of memory." << std::endl;
        message << "       After taxes this is " << std::setprecision(2) << mem_gb << " GB of memory.";
        throw PSIEXCEPTION(message.str());
    }

    // Build result and temp vectors
    std::vector<SharedVector> aux_dens;
    for (size_t i = 0; i < densities.size(); i++) {
        aux_dens.push_back(std::make_shared<Vector>(naux));
    }

    std::vector<SharedMatrix> collapse_temp;
    for (size_t i = 0; i < nthread; i++) {
        collapse_temp.push_back(std::make_shared<Matrix>(auxiliary_->max_function_per_shell(), nbf2));
    }

// Do the contraction
#pragma omp parallel for schedule(dynamic) num_threads(nthread)
    for (size_t Rshell = 0; Rshell < auxiliary_->nshell(); Rshell++) {
        size_t thread = 0;
#ifdef _OPENMP
        thread = omp_get_thread_num();
#endif

        collapse_temp[thread]->zero();
        double** tempp = collapse_temp[thread]->pointer();

        size_t num_r = auxiliary_->shell(Rshell).nfunction();
        size_t index_r = auxiliary_->shell(Rshell).function_index();

        // Loop over our PQ shells
        for (auto PQshell : df_pairs) {
            size_t Pshell = PQshell.first;
            size_t Qshell = PQshell.second;

            df_ints[thread]->compute_shell(Rshell, 0, Pshell, Qshell);
            df_buff[thread] = df_ints[thread]->buffer();

            size_t num_p = primary_->shell(Pshell).nfunction();
            size_t index_p = primary_->shell(Pshell).function_index();

            size_t num_q = primary_->shell(Qshell).nfunction();
            size_t index_q = primary_->shell(Qshell).function_index();

            size_t index = 0;
            for (size_t r = 0; r < num_r; r++) {
                for (size_t p = index_p; p < index_p + num_p; p++) {
                    for (size_t q = index_q; q < index_q + num_q; q++) {
                        tempp[r][p * nbf + q] = tempp[r][q * nbf + p] = df_buff[thread][index++];
                    }
                }
            }
        }

        // Stitch it together
        for (size_t i = 0; i < densities.size(); i++) {
            C_DGEMV('N', num_r, nbf2, 1.0, tempp[0], nbf2, densities[i]->pointer()[0], 1, 0.0,
                    (aux_dens[i]->pointer() + index_r), 1);
        }

    }  // End Rshell

    // Clear a few temps
    collapse_temp.clear();
    df_buff.clear();
    df_ints.clear();
    df_pairs.clear();

    // ==> Contract (S|R)^-1 R -> S <== //
    std::vector<SharedVector> aux_dens_inv;
    for (size_t i = 0; i < densities.size(); i++) {
        aux_dens_inv.push_back(std::make_shared<Vector>(naux));
        aux_dens_inv[i]->gemv(false, 1.0, *metric_inv_, *aux_dens[i], 0.0);
    }

    // ==> Contract (PQS) S -> PQ <== //
    std::vector<std::shared_ptr<ThreeCenterOverlapInt>> aux_ints(nthread);
    std::vector<const double*> aux_buff(nthread);

    for (size_t i = 0; i < nthread; i++) {
        aux_ints[i] = std::shared_ptr<ThreeCenterOverlapInt>(
            new ThreeCenterOverlapInt(auxiliary_, auxiliary_, auxiliary_));
        aux_buff[i] = aux_ints[i]->buffers()[0];
    }

    // Pack the Aux pairs
    std::vector<std::pair<size_t, size_t>> aux_pairs;
    for (size_t M = 0; M < auxiliary_->nshell(); M++) {
        for (size_t N = 0; N <= M; N++) {
            aux_pairs.push_back(std::pair<size_t, size_t>(M, N));
        }
    }

    // Build result and temp vectors
    std::vector<SharedMatrix> ret;
    for (size_t i = 0; i < densities.size(); i++) {
        ret.push_back(std::make_shared<Matrix>(naux, naux));
    }

    size_t max_func = auxiliary_->max_function_per_shell();
    for (size_t i = 0; i < nthread; i++) {
        collapse_temp.push_back(std::make_shared<Matrix>(max_func * max_func, naux));
    }

#pragma omp parallel for schedule(dynamic) num_threads(nthread)
    for (size_t PQi = 0; PQi < aux_pairs.size(); PQi++) {
        size_t thread = 0;
#ifdef _OPENMP
        thread = omp_get_thread_num();
#endif

        size_t Pshell = aux_pairs[PQi].first;
        size_t Qshell = aux_pairs[PQi].second;

        size_t num_p = auxiliary_->shell(Pshell).nfunction();
        size_t index_p = auxiliary_->shell(Pshell).function_index();

        size_t num_q = auxiliary_->shell(Qshell).nfunction();
        size_t index_q = auxiliary_->shell(Qshell).function_index();

        double** tempp = collapse_temp[thread]->pointer();

        // Build aux temp
        for (size_t Rshell = 0; Rshell < auxiliary_->nshell(); Rshell++) {
            size_t num_r = auxiliary_->shell(Rshell).nfunction();
            size_t index_r = auxiliary_->shell(Rshell).function_index();

            aux_ints[thread]->compute_shell(Pshell, Qshell, Rshell);
            aux_buff[thread] = aux_ints[thread]->buffers()[0];

            size_t index = 0;
            for (size_t p = 0; p < num_p; p++) {
                for (size_t q = 0; q < num_q; q++) {
                    for (size_t r = index_r; r < num_r + index_r; r++) {
                        tempp[p * num_q + q][r] = aux_buff[thread][index++];
                    }
                }
            }
        }

        // Contract back to aux
        for (size_t i = 0; i < densities.size(); i++) {
            double** retp = ret[i]->pointer();
            double* dens = aux_dens_inv[i]->pointer();
            for (size_t p = 0; p < num_p; p++) {
                size_t abs_p = index_p + p;
                for (size_t q = 0; q < num_q; q++) {
                    size_t abs_q = index_q + q;

                    retp[abs_p][abs_q] = retp[abs_q][abs_p] = 2.0 * C_DDOT(naux, tempp[p * num_q + q], 1, dens, 1);
                }
            }
        }
    }

    return ret;
}

SharedMatrix FDDS_Dispersion::form_unc_amplitude(std::string monomer, double omega) {
    check_monomer(monomer);
    if (!std::isfinite(omega) || omega < 0.0)
        throw PSIEXCEPTION("FDDS: imaginary frequency must be finite and nonnegative.");
    // ==> Configuration <==
    SharedVector eps_occ, eps_vir;
    std::string ovQ_tensor_name;

    if (monomer == "A") {
        eps_occ = vector_cache_["eps_occ_A"];
        eps_vir = vector_cache_["eps_vir_A"];
        ovQ_tensor_name = "arQ";
    } else if (monomer == "B") {
        eps_occ = vector_cache_["eps_occ_B"];
        eps_vir = vector_cache_["eps_vir_B"];
        ovQ_tensor_name = "bsQ";

    } else {
        throw PSIEXCEPTION("FDDS_Dispersion::form_unc_amplitude: Monomer must be A or B!");
    }

    // Sizes
    size_t nocc = eps_occ->dim(0);
    size_t nvir = eps_vir->dim(0);
    size_t naux = naux_;

    // Check on memory real quick
    size_t doubles = budget_doubles();
    // Row pointers of ret, amp and the tmp slice are charged on the declared budget only.
    const size_t rp = work_doubles_ ? kRowPointer : 0;
    size_t mem_size = 2 * naux * nvir + naux * naux + nvir * nocc + rp * (2 * nvir + naux + nocc);
    if (mem_size > doubles) {
        std::stringstream message;
        double mem_gb = ((double)(mem_size) / 0.8 * sizeof(double));
        message << "FDDS Dispersion requires at least naux * nvir + naux * naux of memory." << std::endl;
        message << "       After taxes this is " << std::setprecision(2) << mem_gb << " GB of memory.";
        throw PSIEXCEPTION(message.str());
    }

    // ==> Uncoupled Amplitudes <==
    auto amp = std::make_shared<Matrix>(nocc, nvir);

    double** ampp = amp->pointer();
    double* eoccp = eps_occ->pointer();
    double* evirp = eps_vir->pointer();

#pragma omp parallel for
    for (size_t i = 0; i < nocc; i++) {
        for (size_t a = 0; a < nvir; a++) {
            double val = -1.0 * (eoccp[i] - evirp[a]);
            double tmp = 4.0 * val / (val * val + omega * omega);
            // Lets see how stable this is, should be fine
            if (tmp < 1.e-14) {
                ampp[i][a] = 0.0;
            } else {
                ampp[i][a] = std::pow(tmp, 0.5);
            }
        }
    }

    // amp->print();

    // ==> Contract <==

    size_t dmem = doubles - naux * naux - nvir * nocc - rp * (naux + nocc);
    size_t bsize = dmem / (nvir * (naux + rp));
    if (bsize > nocc) {
        bsize = nocc;
    }
    size_t nblocks = 1 + ((nocc - 1) / bsize);

    // printf("dmem:    %zu\n", dmem);
    // printf("bsize:   %zu\n", bsize);


    auto ret = std::make_shared<Matrix>("UNC Amplitude", naux, naux);
    auto tmp = std::make_shared<Matrix>("arQ tmp", bsize * nvir, naux);

    double** tmpp = tmp->pointer();

    ret->zero();
    size_t rstat, osize;
    for (size_t block = 0, bcount = 0; block < nblocks; block++) {
        // printf("Block %zu\n", block);
        tmp->zero();
        if (((block + 1) * bsize) > nocc) {
            osize = nocc - block * bsize;
        } else {
            osize = bsize;
        }

        dfh_->fill_tensor(ovQ_tensor_name, tmp, {bcount, bcount + osize});
        size_t shift_i = block * bsize;

#pragma omp parallel for collapse(2)
        for (size_t i = 0; i < osize; i++) {
            for (size_t a = 0; a < nvir; a++) {
                double val = ampp[i + shift_i][a];
#pragma omp simd
                for (size_t Q = 0; Q < naux; Q++) {
                    tmpp[i * nvir + a][Q] *= val;
                }
            }
        }

        ret->gemm(true, false, 1.0, tmp, tmp, 1.0);
        bcount += osize;
    }

    return ret;
}

std::map<std::string, SharedMatrix> FDDS_Dispersion::form_aux_matrices(std::string monomer, double omega){
    check_monomer(monomer, true);
    if (!std::isfinite(omega) || omega < 0.0)
        throw PSIEXCEPTION("FDDS: imaginary frequency must be finite and nonnegative.");

    // => Configuration <= //
    SharedVector eps_occ, eps_vir;
    std::string arQ_name, XarQ_name, YarQ_name, QXarQ_name, QYarQ_name;

    if (monomer == "A") {
        eps_occ = vector_cache_["eps_occ_A"];
        eps_vir = vector_cache_["eps_vir_A"];
        arQ_name = "arQ";
        XarQ_name = "XarQ";
        QXarQ_name = "QXarQ";
        YarQ_name = "YarQ";
        QYarQ_name = "QYarQ";
    } else if (monomer == "B") {
        eps_occ = vector_cache_["eps_occ_B"];
        eps_vir = vector_cache_["eps_vir_B"];
        arQ_name = "bsQ";
        XarQ_name = "XbsQ";
        QXarQ_name = "QXbsQ";
        YarQ_name = "YbsQ";
        QYarQ_name = "QYbsQ";
    } else {
        throw PSIEXCEPTION("FDDS_Dispersion::form_aux_matrices: Monomer must be A or B!");
    }


    // => Sizing <= //

    size_t nocc = eps_occ->dim(0);
    size_t nvir = eps_vir->dim(0);
    size_t naux = naux_;

    // => Blocking <= //

    size_t doubles = budget_doubles();
    // Row pointers of Lar/LDar, the targets and the seven slices are charged on the declared budget only.
    const size_t rp = work_doubles_ ? kRowPointer : 0;
    long long int rem = doubles - 2 * nocc * nvir - 6 * naux * naux - rp * (2 * nocc + 6 * naux);
    if (rem < 0)
        throw PSIEXCEPTION("Too little static memory for FDDS_Dispersion::form_aux_matrices()");

    size_t maxo = rem / (7 * nvir * (naux + rp));
    maxo = (maxo > nocc ? nocc : maxo);
    if (maxo < 1)
        throw PSIEXCEPTION("Too little static memory for FDDS_Dispersion::form_aux_matrices()");

    // => Scalars <= //

    auto Lar = std::make_shared<Matrix>(nocc, nvir); // lambda
    auto LDar = std::make_shared<Matrix>(nocc, nvir); // lambda * d

    double** Larp = Lar->pointer();
    double** LDarp = LDar->pointer();
    double* eoccp = eps_occ->pointer();
    double* evirp = eps_vir->pointer();

#pragma omp parallel for
    for (size_t a = 0; a < nocc; a++) {
        for (size_t r = 0; r < nvir; r++) {
            double val = evirp[r] - eoccp[a]; // d
            double ll = -4.0 / (val * val + omega * omega); // lambda
            double ld = val * ll; // lambda * d
            Larp[a][r] = ll;
            LDarp[a][r] = ld;
        }
    }

    // => Tensor Slices <= //

    auto arQ = std::make_shared<Matrix>("arQ", maxo * nvir, naux);
    auto arQLD = std::make_shared<Matrix>("arQLD", maxo * nvir, naux);
    auto XarQ = std::make_shared<Matrix>("XarQ", maxo * nvir, naux);
    auto QXarQ = std::make_shared<Matrix>("QXarQ", maxo * nvir, naux);
    auto YarQ = std::make_shared<Matrix>("YarQ", maxo * nvir, naux);
    auto QYarQ = std::make_shared<Matrix>("QYarQ", maxo * nvir, naux);
    auto YarQL = std::make_shared<Matrix>("YarQL", maxo * nvir, naux);

    // => Target <= //    
    std::map<std::string, SharedMatrix> ret;
    ret["amp"] = std::make_shared<Matrix>("amp", naux, naux); // Uncoupled amplitude
    ret["K1LD"] = std::make_shared<Matrix>("K1LD", naux, naux);
    ret["K2LD"] = std::make_shared<Matrix>("K2LD", naux, naux);
    ret["K2L"] = std::make_shared<Matrix>("K2L", naux, naux);
    ret["K21L"] = std::make_shared<Matrix>("K21L", naux, naux);

    // Zero out matrices
    
    for (auto const &mat : ret)
        mat.second->zero();

    // => Pointers <= //

    double** arQp = arQ->pointer();
    double** arQLDp = arQLD->pointer();
    double** XarQp = XarQ->pointer();
    double** YarQp = YarQ->pointer();
    double** YarQLp = YarQL->pointer();
    double** QXarQp = QXarQ->pointer();
    double** QYarQp = QYarQ->pointer();

    double** ampp = ret["amp"]->pointer();
    double** K1LDp = ret["K1LD"]->pointer();
    double** K2LDp = ret["K2LD"]->pointer();
    double** K2Lp = ret["K2L"]->pointer();
    double** K21Lp = ret["K21L"]->pointer();

    // => Master Loop <= //

    for (size_t astart = 0; astart < nocc; astart += maxo) {
        size_t nablock = (astart + maxo >= nocc ? nocc - astart : maxo);

        size_t navir = nablock * nvir;

        dfh_->fill_tensor(arQ_name, arQ, {astart, astart + nablock});
        dfh_->fill_tensor(XarQ_name, XarQ, {astart, astart + nablock});
        dfh_->fill_tensor(QXarQ_name, QXarQ, {astart, astart + nablock});
        dfh_->fill_tensor(YarQ_name, YarQ, {astart, astart + nablock});
        dfh_->fill_tensor(QYarQ_name, QYarQ, {astart, astart + nablock});
        
        // X <- X + Y, Y <- Y - X
        XarQ->axpy(1.0, YarQ);
        YarQ->scale(2.0);
        YarQ->axpy(-1.0, XarQ);

        QXarQ->axpy(1.0, QYarQ);
        QYarQ->scale(2.0);
        QYarQ->axpy(-1.0, QXarQ);

        arQLD->copy(arQ);
        YarQL->copy(YarQ);

#pragma omp parallel for collapse(2)
        for (size_t a = 0; a < nablock; a++) {
            for (size_t r = 0; r < nvir; r++) {
                double ll = Larp[astart + a][r];
                double ld = LDarp[astart + a][r];
#pragma omp simd
                for (size_t Q = 0; Q < naux; Q++) {
                    arQLDp[a * nvir + r][Q] *= ld;
                    YarQLp[a * nvir + r][Q] *= ll;
                }
            }
        }

        C_DGEMM('T', 'N', naux, naux, navir, 1.0, arQLDp[0], naux, arQp[0], naux, 1.0, ampp[0], naux);
        C_DGEMM('T', 'N', naux, naux, navir, 1.0, arQLDp[0], naux, QXarQp[0], naux, 1.0, K1LDp[0], naux);
        C_DGEMM('T', 'N', naux, naux, navir, 1.0, arQLDp[0], naux, QYarQp[0], naux, 1.0, K2LDp[0], naux);
        C_DGEMM('T', 'N', naux, naux, navir, 1.0, arQp[0], naux, YarQLp[0], naux, 1.0, K2Lp[0], naux);
        C_DGEMM('T', 'N', naux, naux, navir, 1.0, YarQLp[0], naux, QXarQp[0], naux, 1.0, K21Lp[0], naux);

    }
    
    return ret;
}

void FDDS_Dispersion::form_X(std::string monomer) {
    check_monomer(monomer, true);

    // => Configuration <= //

    std::string arQ_name, arR_name, QarQ_name, XarQ_name, QXarQ_name;
    SharedVector eps_occ, eps_vir;
    
    if (monomer == "A") {
        arQ_name = "arQ";
        arR_name = "arR";
        QarQ_name = "QarQ";
        XarQ_name = "XarQ";
        QXarQ_name = "QXarQ";
        eps_occ = vector_cache_["eps_occ_A"];
        eps_vir = vector_cache_["eps_vir_A"];
    } else if (monomer == "B") {
        arQ_name = "bsQ";
        arR_name = "bsR";
        QarQ_name = "QbsQ";
        XarQ_name = "XbsQ";
        QXarQ_name = "QXbsQ";
        eps_occ = vector_cache_["eps_occ_B"];
        eps_vir = vector_cache_["eps_vir_B"];
    } else {
        throw PSIEXCEPTION("FDDS_Dispersion::form_X: Monomer must be A or B!");
    }

    exchange_X(arR_name, {arQ_name, QarQ_name}, {XarQ_name, QXarQ_name}, eps_occ->dim(0), eps_vir->dim(0));
}

void FDDS_Dispersion::exchange_X(const std::string& arR_name, const std::array<std::string, 2>& in,
                                 const std::array<std::string, 2>& out, size_t nocc, size_t nvir) {
    const std::string &arQ_name = in[0], &QarQ_name = in[1], &XarQ_name = out[0], &QXarQ_name = out[1];

    // => Sizing <= //

    size_t naux = naux_;

    // => Blocking <= //
    
    int nthread = 1;
#ifdef _OPENMP
    nthread = blocking_threads();
#endif

    size_t doubles = budget_doubles();
    // Row pointers of Vrr and the six slices are charged on the declared budget only.
    const size_t rp = work_doubles_ ? kRowPointer : 0;
    long long int rem = doubles - nthread * nvir * nvir - rp * nvir;
    if (rem < 0) 
        throw PSIEXCEPTION("Too little static memory for FDDS_Dispersion::form_X()");

    size_t maxo = rem / (6 * nvir * (naux + rp));
    maxo = (maxo > nocc ? nocc : maxo);
    if (maxo < 1) 
        throw PSIEXCEPTION("Too little static memory for FDDS_Dispersion::form_X()");

    // => Tensor Slices <= //

    auto arR = std::make_shared<Matrix>("arR", maxo * nvir, naux); // ([a]r'|R)
    auto aprR = std::make_shared<Matrix>("aprR", maxo * nvir, naux); // ([a']r|R)
    auto arQ = std::make_shared<Matrix>("arQ", maxo * nvir, naux); // ([a']r'|Q)
    auto QarQ = std::make_shared<Matrix>("QarQ", maxo * nvir, naux); // ([a']r'|Q|Q)
    auto Vrr = std::make_shared<Matrix>("Vrr", nvir, nvir); // ([a']r|[a]r') single layer

    // => Target <= //

    dfh_->add_disk_tensor(XarQ_name, std::make_tuple(nocc, nvir, naux)); // (ar|X|Q) to disk
    auto XarQ = std::make_shared<Matrix>("XarQ", maxo * nvir, naux); // ([a]r|X|Q)

    dfh_->add_disk_tensor(QXarQ_name, std::make_tuple(nocc, nvir, naux)); // (ar|QX|Q) to disk
    auto QXarQ = std::make_shared<Matrix>("QXarQ", maxo * nvir, naux); // ([a]r|QX|Q)

    // => Pointers <= //

    double** arRp = arR->pointer();
    double** aprRp = aprR->pointer();
    double** arQp = arQ->pointer();
    double** QarQp = QarQ->pointer();
    double** Vrrp = Vrr->pointer();
    double** XarQp = XarQ->pointer();
    double** QXarQp = QXarQ->pointer();

    // => Master Loop <= //
    
    for (size_t astart = 0; astart < nocc; astart += maxo) {
        size_t nablock = (astart + maxo >= nocc ? nocc - astart : maxo);

        arR->zero();
        dfh_->fill_tensor(arR_name, arR, {astart, astart + nablock});
        XarQ->zero();
        QXarQ->zero();

        for (size_t apstart = 0; apstart < nocc; apstart += maxo) {
            size_t napblock = (apstart + maxo >= nocc ? nocc - apstart : maxo);        

            aprR->zero();   
            arQ->zero();
            QarQ->zero();

            dfh_->fill_tensor(arR_name, aprR, {apstart, apstart + napblock});
            dfh_->fill_tensor(arQ_name, arQ, {apstart, apstart + napblock});
            dfh_->fill_tensor(QarQ_name, QarQ, {apstart, apstart + napblock});

            size_t naap = nablock * napblock;

            for (size_t aap = 0; aap < naap; aap++) {
                size_t a = aap / napblock;
                size_t ap = aap % napblock;

                // => Contraction ((a')r|R) (R|(a)r') ((a')r'|Q) -> ((a)r|X|Q) for a, a' <= // This one is not correct. a (a') and r (r') have to be in the same bra/ket, because (ar|a'r') != (ar'|a'r). (ar|a'r') = (ar|R) (R|a'r') defines the ordering of 3-index integrals.
                // => Contraction [((a')r'|R) (R|(a)r)]^T ((a')r'|Q) -> ((a)r|X|Q) for a, a' <= // Correct expression in terms of (a'r'|R) and (ar|R).  
                // => Similar for ((a)r|QX|Q) <= //

                C_DGEMM('N', 'T', nvir, nvir, naux, 1.0, aprRp[ap * nvir], naux, arRp[a * nvir], naux, 0.0, Vrrp[0], nvir);
                C_DGEMM('N', 'N', nvir, naux, nvir, 1.0, Vrrp[0], nvir, arQp[ap * nvir], naux, 1.0, XarQp[a * nvir], naux);
                C_DGEMM('N', 'N', nvir, naux, nvir, 1.0, Vrrp[0], nvir, QarQp[ap * nvir], naux, 1.0, QXarQp[a * nvir], naux);
            }
        }

        dfh_->write_disk_tensor(XarQ_name, XarQ, {astart, astart + nablock});
        dfh_->write_disk_tensor(QXarQ_name, QXarQ, {astart, astart + nablock});
    }
}

void FDDS_Dispersion::form_Y(std::string monomer) {
    check_monomer(monomer, true);

    // => Configuration <= //

    std::string raQ_name, aaR_name, QraQ_name, rrR_name, YarQ_name, QYarQ_name;
    SharedVector eps_occ, eps_vir;
    
    if (monomer == "A") {
        raQ_name = "raQ";
        aaR_name = "aaR";
        rrR_name = "rrR";
        QraQ_name = "QraQ";
        YarQ_name = "YarQ";
        QYarQ_name = "QYarQ";
        eps_occ = vector_cache_["eps_occ_A"];
        eps_vir = vector_cache_["eps_vir_A"];
    } else if (monomer == "B") {
        raQ_name = "sbQ";
        aaR_name = "bbR";
        rrR_name = "ssR";
        QraQ_name = "QsbQ";
        YarQ_name = "YbsQ";
        QYarQ_name = "QYbsQ";
        eps_occ = vector_cache_["eps_occ_B"];
        eps_vir = vector_cache_["eps_vir_B"];
    } else {
        throw PSIEXCEPTION("FDDS_Dispersion::form_Y: Monomer must be A or B!");
    }

    exchange_Y(aaR_name, rrR_name, {raQ_name, QraQ_name}, {YarQ_name, QYarQ_name}, eps_occ->dim(0),
               eps_vir->dim(0));
}

void FDDS_Dispersion::exchange_Y(const std::string& aaR_name, const std::string& rrR_name,
                                 const std::array<std::string, 2>& in_ra, const std::array<std::string, 2>& out,
                                 size_t nocc, size_t nvir) {
    const std::string &raQ_name = in_ra[0], &QraQ_name = in_ra[1], &YarQ_name = out[0], &QYarQ_name = out[1];

    // => Sizing <= //

    size_t naux = naux_;

    // => Blocking <= //
    
    int nthread = 1;
#ifdef _OPENMP
    nthread = blocking_threads();
#endif

    size_t doubles = budget_doubles();
    // Row pointers of Vra and the six slices are charged on the declared budget only; the slices
    // hold at most (kov^2 + 4 kov + 1) nocc rows per occupied index, as their storage does.
    const size_t rp = work_doubles_ ? kRowPointer : 0;
    long long int rem = doubles - nthread * nocc * nvir - rp * nvir;
    if (rem < 0) 
        throw PSIEXCEPTION("Too little static memory for FDDS_Dispersion::form_Y()");

    size_t kov = std::max(size_t{1}, nvir / nocc) + 1; // Keep virtual blocks nonempty when v < o.
    size_t maxo = rem / ((kov * kov + 4 * kov + 1) * nocc * (naux + rp));
    size_t maxv = maxo * (kov - 1); 
    maxo = (maxo > nocc ? nocc : maxo);
    maxv = (maxv > nvir ? nvir : maxv);
    if (maxo < 1) 
        throw PSIEXCEPTION("Too little static memory for FDDS_Dispersion::form_Y()");

    // => Tensor Slices <= //

    auto aaR = std::make_shared<Matrix>("aaR", maxo * nocc, naux); // ([a]a'|R)
    auto rrR = std::make_shared<Matrix>("rrR", maxv * nvir, naux); // ([r']r|R)
    auto raQ = std::make_shared<Matrix>("raQ", maxv * nocc, naux); // ([r']a'|Q)
    auto QraQ = std::make_shared<Matrix>("QraQ", maxv * nocc, naux); // ([r']a'|Q|Q)
    auto Vra = std::make_shared<Matrix>("Vra", nvir, nocc); // ([r']r|[a]a') single layer

    // => Target <= //

    dfh_->add_disk_tensor(YarQ_name, std::make_tuple(nocc, nvir, naux)); // (ar|Y|Q) to disk
    auto YarQ = std::make_shared<Matrix>("YarQ", maxo * nvir, naux); // ([a]r|Y|Q)

    dfh_->add_disk_tensor(QYarQ_name, std::make_tuple(nocc, nvir, naux)); // (ar|QY|Q) to disk
    auto QYarQ = std::make_shared<Matrix>("QYarQ", maxo * nvir, naux); // ([a]r|QY|Q)

    // => Pointers <= //

    double** aaRp = aaR->pointer();
    double** rrRp = rrR->pointer();
    double** raQp = raQ->pointer();
    double** QraQp = QraQ->pointer();
    double** Vrap = Vra->pointer();
    double** YarQp = YarQ->pointer();
    double** QYarQp = QYarQ->pointer();

    // => Master Loop <= //
    
    for (size_t astart = 0; astart < nocc; astart += maxo) {
        size_t nablock = (astart + maxo >= nocc ? nocc - astart : maxo);
        
        aaR->zero();
        dfh_->fill_tensor(aaR_name, aaR, {astart, astart + nablock});
        YarQ->zero();
        QYarQ->zero();

        for (size_t rstart = 0; rstart < nvir; rstart += maxv) {
            size_t nrblock = (rstart + maxv >= nvir ? nvir - rstart : maxv);        

            rrR->zero();
            raQ->zero();
            QraQ->zero();

            dfh_->fill_tensor(rrR_name, rrR, {rstart, rstart + nrblock});
            dfh_->fill_tensor(raQ_name, raQ, {rstart, rstart + nrblock});
            dfh_->fill_tensor(QraQ_name, QraQ, {rstart, rstart + nrblock});

            size_t nar = nablock * nrblock;

            for (size_t ar = 0; ar < nar; ar++) {
                size_t a = ar / nrblock;
                size_t r = ar % nrblock;

                // => Contraction ((r')r|R) (R|(a)a') ((r')a'|Q) -> ((a)r|Y|Q) for a, r' <= //
                // Similar for ((a)r|QY|Q)

                C_DGEMM('N', 'T', nvir, nocc, naux, 1.0, rrRp[r * nvir], naux, aaRp[a * nocc], naux, 0.0, Vrap[0], nocc);
                C_DGEMM('N', 'N', nvir, naux, nocc, 1.0, Vrap[0], nocc, raQp[r * nocc], naux, 1.0, YarQp[a * nvir], naux);
                C_DGEMM('N', 'N', nvir, naux, nocc, 1.0, Vrap[0], nocc, QraQp[r * nocc], naux, 1.0, QYarQp[a * nvir], naux);
            }
        }

        dfh_->write_disk_tensor(YarQ_name, YarQ, {astart, astart + nablock});
        dfh_->write_disk_tensor(QYarQ_name, QYarQ, {astart, astart + nablock});
    }
}

SharedMatrix FDDS_Dispersion::QR(std::string monomer) {
    check_monomer(monomer, true);

    // => Configuration <= //

    std::string Qar_name, QarQ_name, QraQ_name;
    SharedVector eps_occ, eps_vir;
    
    if (monomer == "A") {
        Qar_name = "Qar";
        QarQ_name = "QarQ";
        QraQ_name = "QraQ";
        eps_occ = vector_cache_["eps_occ_A"];
        eps_vir = vector_cache_["eps_vir_A"];
    } else if (monomer == "B") {
        Qar_name = "Qbs";
        QarQ_name = "QbsQ";
        QraQ_name = "QsbQ";
        eps_occ = vector_cache_["eps_occ_B"];
        eps_vir = vector_cache_["eps_vir_B"];
    } else {
        throw PSIEXCEPTION("FDDS_Dispersion::QR: Monomer must be A or B!");
    }

    // => Sizing <= //

    size_t nocc = eps_occ->dim(0);
    size_t nvir = eps_vir->dim(0);
    size_t nbf = primary_->nbf();
    size_t naux = naux_;
    size_t nov = nocc * nvir;
    size_t nq = std::min(nov, naux);

    // => Meomry Check <= //

    size_t doubles = budget_doubles();
    const size_t rp = work_doubles_ ? kRowPointer : 0;  // Qar, Q and R row pointers, declared budget only
    size_t req_mem = 2 * nov * naux + naux * naux + naux + rp * (2 * naux + nov);
    if (doubles < req_mem) 
        throw PSIEXCEPTION("Too little static memory for FDDS_Dispersion::QR()");

    // => Tensor Slices <= //

    auto Qar = std::make_shared<Matrix>("Qar", naux, nov); 
    auto Q = std::make_shared<Matrix>("Q", nov, naux);
    if (qr_from_arQ_) {
        // Declared path: only the (ar|Q) stream exists; transpose it through the Q buffer.
        dfh_->fill_tensor("arQ", Q, {0, nocc});
        double** Qsrc = Q->pointer();
        double** Qdst = Qar->pointer();
        for (size_t row = 0; row < nov; row++)
            for (size_t col = 0; col < naux; col++) Qdst[col][row] = Qsrc[row][col];
    } else {
        dfh_->fill_tensor(Qar_name, Qar, {0, naux});
    }

    // => Target <= //
    auto tau = std::make_shared<Vector>("tau", nq);
    auto R = std::make_shared<Matrix>("R", naux, naux);

    // => Pointers <= //

    double** Qarp = Qar->pointer();
    double* taup = tau->pointer();
    double** Qp = Q->pointer();
    double** Rp = R->pointer();

    // => Work Buffer <= //

    double lwork_tmp, qwork_tmp;
    if (C_DGEQRF(nov, naux, Qarp[0], nov, taup, &lwork_tmp, -1) != 0)
        throw PSIEXCEPTION("FDDS: DGEQRF workspace query failed.");
    if (C_DORGQR(nov, nq, nq, Qarp[0], nov, taup, &qwork_tmp, -1) != 0)
        throw PSIEXCEPTION("FDDS: DORGQR workspace query failed.");
    size_t lwork = (size_t) std::max(lwork_tmp, qwork_tmp);
    auto work = std::make_shared<Vector>("work", lwork);
    double* workp = work->pointer();

    // Householder QR
    if (C_DGEQRF(nov, naux, Qarp[0], nov, taup, workp, lwork) != 0)
        throw PSIEXCEPTION("FDDS: DGEQRF failed.");

    // Pad the economy factors with zeros: (ar|P) = Q R for tall and wide inputs.
    R->zero();
    for (size_t row = 0; row < nq; row++)
        for (size_t col = row; col < naux; col++) {
            Rp[row][col] = Qarp[col][row]; 
        }

    // Return R uninverted; the driver applies its existing pseudoinverse policy.
    if (C_DORGQR(nov, nq, nq, Qarp[0], nov, taup, workp, lwork) != 0)
        throw PSIEXCEPTION("FDDS: DORGQR failed.");
    size_t nar, nra;
    Q->zero();
    for (size_t nQ = 0; nQ < nq; nQ++)
        for (size_t na = 0; na < nocc; na++) 
            for (size_t nr = 0; nr < nvir; nr++) {
                nar = na * nvir + nr;
                Qp[nar][nQ] = Qarp[nQ][nar]; 
            }

    dfh_->add_disk_tensor(QarQ_name, std::make_tuple(nocc, nvir, naux)); 
    dfh_->write_disk_tensor(QarQ_name, Q, {0, nocc});

    // Save transposed Q (ra|Q|Q)
    
    for (size_t nQ = 0; nQ < nq; nQ++)
        for (size_t na = 0; na < nocc; na++) 
            for (size_t nr = 0; nr < nvir; nr++) {
                nar = na * nvir + nr;
                nra = nr * nocc + na;
                Qp[nra][nQ] = Qarp[nQ][nar]; 
            }

    dfh_->add_disk_tensor(QraQ_name, std::make_tuple(nvir, nocc, naux)); 
    dfh_->write_disk_tensor(QraQ_name, Q, {0, nvir});

    return R;

}

SharedMatrix FDDS_Dispersion::get_tensor_pqQ(std::string name, std::tuple<size_t, size_t, size_t> dimensions) {

    // Debug helper
    // Returns as (pq|Q)
    size_t np = std::get<0>(dimensions);
    size_t nq = std::get<1>(dimensions);
    size_t nQ = std::get<2>(dimensions);

    auto M = std::make_shared<Matrix>(name, np * nq, nQ);

    dfh_->fill_tensor(name, M, {0, np});

    return M;
}


void FDDS_Dispersion::print_tensor_pqQ(std::string tensor_name, std::string file_name, std::tuple<size_t, size_t, size_t> dimensions) {

    // Debug helper
    size_t np = std::get<0>(dimensions);
    size_t nq = std::get<1>(dimensions);
    size_t nQ = std::get<2>(dimensions);

    auto M = std::make_shared<Matrix>(tensor_name, np * nq, nQ);
    dfh_->fill_tensor(tensor_name, M, {0, np});
    double** Mp = M->pointer();
    auto Mout = std::make_shared<PsiOutStream>(file_name);
    for (size_t row = 0; row < np * nq; row++)
        for (size_t col = 0; col < nQ; col++)
            Mout->Printf("%.15e\n", Mp[row][col]); 
}

// ==> Declared-basis FDDS_Monomer path <== //

namespace {

// LAPACK workspace queries use dimensions only; the dummy arrays are never read.
size_t lwork_syev(size_t n) {
    double w, dummy = 0.0, a = 0.0;
    C_DSYEV('V', 'U', n, &a, n, &dummy, &w, -1);
    return (size_t)w;
}
size_t lwork_qr(size_t m, size_t n) {
    double w1 = 0.0, w2 = 0.0, a = 0.0, tau = 0.0;
    size_t k = std::min(m, n);
    C_DGEQRF(m, n, &a, m, &tau, &w1, -1);
    C_DORGQR(m, k, k, &a, m, &tau, &w2, -1);
    return (size_t)std::max(w1, w2);
}
size_t lwork_gesdd(char jobz, size_t n) {
    double w = 0.0, a = 0.0, s = 0.0, u = 0.0, vt = 0.0;
    std::vector<int> iwork(8 * n);
    C_DGESDD(jobz, n, n, &a, n, &s, &u, n, &vt, n, &w, -1, iwork.data());
    return (size_t)w;
}

// Singular values of a square row-major matrix, descending; U and V (= V^T rows) on request.
SharedVector square_svd(const SharedMatrix& A, SharedMatrix U = nullptr, SharedMatrix V = nullptr) {
    int n = A->rowspi()[0];
    auto work_A = A->clone();
    auto S = std::make_shared<Vector>(n);
    std::vector<int> iwork(8L * n);
    char jobz = U ? 'S' : 'N';
    double* Up = U ? U->pointer()[0] : nullptr;
    double* Vp = V ? V->pointer()[0] : nullptr;
    double lwork;
    // The column-major view of A is A^T, so LAPACK's U and VT are the row-major V and U.
    C_DGESDD(jobz, n, n, work_A->pointer()[0], n, S->pointer(), Vp, n, Up, n, &lwork, -1, iwork.data());
    std::vector<double> work((size_t)lwork);
    int info = C_DGESDD(jobz, n, n, work_A->pointer()[0], n, S->pointer(), Vp, n, Up, n, work.data(),
                        (int)lwork, iwork.data());
    if (info != 0) throw PSIEXCEPTION("FDDS: DGESDD failed.");
    return S;
}

bool all_finite(const Matrix& M) {
    for (int i = 0; i < M.rowspi()[0]; i++)
        for (int j = 0; j < M.colspi()[0]; j++)
            if (!std::isfinite(M.get(i, j))) return false;
    return true;
}

SharedMatrix symmetrized(const SharedMatrix& M) {
    auto S = M->clone();
    S->add(M->transpose());
    S->scale(0.5);
    return S;
}

// Numerical storage in doubles for one declared monomer; see sapt.rst for the stage list.
// ==> Integral-object storage, bytes <== //
// Psi4-owned storage of the Libint2 integral objects on the declared path, from basis dimensions
// and primitive counts only. A vector grown by push_back/emplace_back/resize(size() + 1) is charged
// at capacity <= 2 * count: libstdc++ and libc++ double the capacity and MSVC grows it by half; copies
// and resizes from empty allocate exactly. A growth step briefly also holds the old buffer (fewer
// than count elements). Objects are built one at a time, so each pass adds its largest such buffer.
// Not bounded here: libint2::Engine scratch, allocator headers and names.
size_t grown(size_t count, size_t elem) { return cmul(cmul(2, count), elem); }
size_t tri(size_t n) { return cmul(n, cadd({n, 1})) / 2; }
size_t ncart_of(size_t l) { return (l + 1) * (l + 2) / 2; }
constexpr size_t kPair = sizeof(std::pair<int, int>);
constexpr size_t kPrim = sizeof(libint2::ShellPair::PrimPairData);
// make_shared<ShellPair> adds a control block of a vtable pointer and two counters (libstdc++,
// libc++, MSVC), padded to the object's alignment: within four pointers.
constexpr size_t kShellPair = sizeof(libint2::ShellPair) + 4 * sizeof(void*);
// Per listed pair: its single-pair ShellPairBlock and the ShellPair pointer.
constexpr size_t kListed = sizeof(ShellPairBlock) + kPair + sizeof(std::shared_ptr<libint2::ShellPair>);

// TwoBodyAOInt::create_sieve_pair_info on one basis, CSAM (the largest screening): function and
// shell pair values, exchange values and function roots, both reverse maps, the grown function and
// shell pair lists and adjacency rows, and the shell_pairs_ copy.
size_t sieve_bytes(const BasisSet& b) {
    const size_t N = b.nbf(), S = b.nshell();
    return cadd({cmul(cadd({cmul(N, N), cmul(2, cmul(S, S)), N}), sizeof(double)),
                 cmul(cadd({tri(N), tri(S)}), sizeof(long int)), grown(tri(N), kPair),
                 cmul(cadd({N, S}), sizeof(std::vector<int>)), grown(cadd({cmul(N, N), cmul(S, S)}), sizeof(int)),
                 grown(tri(S), kPair), cmul(tri(S), kPair)});
}

// ShellPair data (primitive pairs <= nprim1 * nprim2, grown) over the triangular shell pairs of b,
// or over b's shells paired with the unit shell; and the largest single primitive-pair list.
size_t shellpairs_bytes(const BasisSet& b, bool unit) {
    size_t r = 0;
    for (int P = 0; P < b.nshell(); P++)
        for (int Q = 0; Q <= (unit ? 0 : P); Q++) {
            const size_t np = cmul(b.l2_shell(P).nprim(), unit ? 1 : b.l2_shell(Q).nprim());
            r = cadd({r, kShellPair, grown(np, kPrim)});
        }
    return r;
}
size_t max_primpairs_bytes(const BasisSet& b) {
    size_t m = 0;
    for (int P = 0; P < b.nshell(); P++) m = std::max(m, (size_t)b.l2_shell(P).nprim());
    return cmul(cmul(m, m), kPrim);
}

// One Libint2ERI: (mn|mn) ('p') sieves the bra and copies it to the ket; (Q0|mn) ('q') grows a unit
// bra list and sieves the ket; (Q0|P0) ('m') grows two unit lists. Clones copy everything but share the ShellPairs.
struct EriBytes {
    size_t own;    // sieve, pair lists, blocks, pointers, zero_vec_ and buffer list; also a clone
    size_t pairs;  // ShellPair data, built once by the constructed object
};
EriBytes eri_bytes(const BasisSet& p, const BasisSet& a, char kind) {
    const size_t sp = tri(p.nshell()), sa = a.nshell(), mf = p.max_function_per_shell();
    const size_t mq = a.max_function_per_shell(), mf2 = cmul(mf, mf), mf4 = cmul(mf2, mf2);
    if (kind == 'p')
        return {cadd({sieve_bytes(p), cmul(sp, kPair), cmul(cmul(2, sp), kListed), cmul(mf4, sizeof(double)),
                      sizeof(double*)}),
                cmul(2, shellpairs_bytes(p, false))};
    if (kind == 'q')
        return {cadd({sieve_bytes(p), grown(sa, kPair), cmul(cadd({sa, sp}), kListed),
                      cmul(cmax({cmul(mq, mf2), cmul(mq, mq), mf4}), sizeof(double)), sizeof(double*)}),
                cadd({shellpairs_bytes(a, true), shellpairs_bytes(p, false)})};
    if (kind != 'm') throw PSIEXCEPTION("FDDS: unknown integral kind.");
    return {cadd({cmul(2, grown(sa, kPair)), cmul(cmul(2, sa), kListed), cmul(cmul(mq, mq), sizeof(double)),
                  sizeof(double*)}),
            cmul(2, shellpairs_bytes(a, true))};
}

// IntegralFactory::set_basis builds SphericalTransform and ISphericalTransform tables for l = 0..8.
// Charged: both tables (copied entries hold exactly their components); the outer vectors grown to
// nine entries, plus one reallocation's old entries and component copies; the temporary being
// appended (grown components with an old buffer) and its copy; and the l = 8 ISphericalTransform
// scratch: two ncart^2 blocks with row pointers, invert_matrix's column and ludcmp's scale vector,
// and the pivot index.
constexpr size_t kFactoryMaxAm = 8;
size_t factory_bytes() {
    const size_t csz = sizeof(SphericalTransformComponent);
    const size_t esz = std::max(sizeof(SphericalTransform), sizeof(ISphericalTransform));
    size_t comps = 0, top = 0;
    for (size_t l = 0; l <= kFactoryMaxAm; l++) {
        comps += (2 * l + 1) * ncart_of(l);
        top = std::max(top, (2 * l + 1) * ncart_of(l));
    }
    const size_t nc = ncart_of(kFactoryMaxAm), entries = kFactoryMaxAm + 1;
    return cadd({cmul(cmul(2, comps), csz), cmul(2, grown(entries, esz)), cmul(entries, esz), cmul(comps, csz),
                 grown(top, csz), cmul(cmul(2, top), csz), cmul(cmul(2, cadd({cmul(nc, nc), nc})), sizeof(double)),
                 cmul(cmul(2, nc), sizeof(double)), cmul(nc, sizeof(int))});
}

// The aux overlap OneBodyAOInt: tformbuf_ and target_ (ncart_of(L)^2 each) and its shell pair list.
// build_shell_pair_list_no_spdata keeps every pair (threshold 0): per-thread lists totalling P pairs
// (<= 2P capacity), thread 0 merging all P (<= 2P plus an old buffer < P), and the returned copy.
size_t overlap_bytes(const BasisSet& a) {
    const size_t nc = ncart_of(a.max_am());
    return cadd({factory_bytes(), cmul(cmul(2, cmul(nc, nc)), sizeof(double)), cmul(cmul(6, tri(a.nshell())), kPair),
                 sizeof(double*)});
}

// BasisSet::zero_ao_basis_set() (basisset.cc BasisSet()): seven one-entry or xyz double arrays, nine
// one-entry index arrays, one libint2::Shell (small vectors inline, else one contraction and three
// doubles), and the dummy atom's CartesianEntry and three NumberValue coordinates, each shared_ptr
// owned. The BasisSet and Molecule objects hold scalars, names, pointers and empty containers only.
size_t dummy_basis_bytes() {
    return cadd({7 * sizeof(double), 9 * sizeof(int),
                 sizeof(libint2::Shell) + sizeof(libint2::Shell::Contraction) + 3 * sizeof(double),
                 sizeof(CartesianEntry) + 4 * sizeof(void*), 3 * (sizeof(NumberValue) + 4 * sizeof(void*))});
}

// The declared metric pass: one factory and dummy basis, nthread separately constructed (Q0|P0) objects.
size_t coulomb_pass_bytes(const BasisSet& p, const BasisSet& a, size_t t) {
    const auto e = eri_bytes(p, a, 'm');
    return cadd({factory_bytes(), dummy_basis_bytes(), cmul(t, cadd({e.own, e.pairs})),
                 cmax({cmul(a.nshell(), kPair), max_primpairs_bytes(a)}), kListed});
}

// Raw DFHelper: prepare_sparsity's (mn|mn) object and its clone (two screening threads when t > 1),
// then prepare_AO_core's (INCORE) and transform()'s (Q0|mn) object with t - 1 clones and a dummy
// basis; each set is released before the next is built.
size_t dfhelper_pass_bytes(const BasisSet& p, const BasisSet& a, size_t t) {
    const auto m = eri_bytes(p, a, 'p'), q = eri_bytes(p, a, 'q');
    const size_t screen = t == 1 ? 1 : 2;
    const size_t N = p.nbf();
    return cadd({factory_bytes(),
                 cmax({cadd({m.pairs, cmul(screen, m.own)}),
                       cadd({q.pairs, cmul(t, q.own), dummy_basis_bytes()})}),
                 cmax({cmul(tri(N), kPair), cmul(tri(p.nshell()), kPair), cmul(a.nshell(), kPair),
                       cmul(N, sizeof(int)), max_primpairs_bytes(p), max_primpairs_bytes(a)}),
                 kListed});
}

struct Ledger {
    size_t resident = 0, memory = 0, disk = 0;
    std::map<std::string, size_t> stages;     // transient working sets, doubles
    std::map<std::string, size_t> files;      // disk, doubles
    std::map<std::string, size_t> integrals;  // integral-object storage inside stages, doubles
    size_t dfh_extra = 0;                      // raw-DFHelper storage it does not count in its own memory
    size_t dfh_retained = 0;                   // the part of dfh_extra still held after transform()
    size_t dfh_integral = 0;                   // raw-DFHelper integral objects, also outside its memory
};

Ledger declared_ledger(const BasisSet& primary, const BasisSet& auxiliary, size_t o, size_t v, size_t pd, bool hyb,
                       const std::string& subalgo, size_t t) {
    const size_t N = primary.nbf(), pr = auxiliary.nbf(), n = cmul(o, v), pr2 = cmul(pr, pr);
    const size_t big = std::max(o, v), eig_d = lwork_syev(pd), eig_r = lwork_syev(pr);
    const size_t Md = mat(pd, pd), Mr = mat(pr, pr), MT = mat(pd, pr);
    Ledger L;
    // Orbitals, energies, metric/overlap/inverse/LU and pivots (R and pinv(R)^T when hybrid), and the
    // declared DFHelper's shell offsets, which it holds from the declared pass to destruction.
    L.resident = cadd({mat(N, o), mat(N, v), o, v, cmul(4, Md), pd, hyb ? cmul(2, Md) : 0,
                       dfh_offsets(primary, auxiliary)});

    // J_r, S_r, T J_r with the metric or overlap integral pass; then eigen check / Matrix::power (two
    // n^2 copies, eigenvalues, work); dgecon. J_r is held for the raw DFHelper throughout.
    L.integrals["metric"] = doubles_for(cmax({coulomb_pass_bytes(primary, auxiliary, t), overlap_bytes(auxiliary)}));
    L.stages["metric"] = cadd({MT, cmax({cadd({cmul(2, Mr), MT, L.integrals["metric"]}),
                                         cadd({Mr, cmul(3, Md), pd, eig_d}), cadd({Mr, cmul(5, pd)})})});

    // Raw DIRECT_iaQ DFHelper at metric power 0: its memory share must admit the metric, one
    // auxiliary shell of dense AOs with the worst half/final transforms, one metric-contraction
    // row, and the second copy of each thread's C buffer while it is assigned; DFHelper
    // additionally holds C buffers, sparsity masks, shell offsets, the supplied J_r, the power(0)
    // transient and its integral objects.
    const size_t wtmp = hyb ? big : std::min(o, v), wfinal = hyb ? cmul(big, big) : n;
    const size_t qmax = auxiliary.max_function_per_shell(), nshell = primary.nshell();
    size_t dfh_min = cmax({pr2, cmul(qmax, cadd({cmul(N, N), cmul(wtmp, N), cmul(2, wfinal)})),
                           cadd({pr2, cmul(cmul(2, pr), big)}), cmul(t, cmul(N, wtmp))});
    if (subalgo == "INCORE")
        dfh_min = cmax({dfh_min, cadd({cmul(pr, cmul(N, N)), pr2, cmul(t, cmul(N, N)),
                                       cmul(3, cmul(qmax, cmul(N, N)))})});
    // After transform() it keeps the Schwarz shell mask/function index, skip arrays and shell offsets.
    L.dfh_retained = cadd({cmul(nshell, nshell), cmul(N, N), cmul(5, N), 3, dfh_offsets(primary, auxiliary)});
    L.dfh_extra = cadd({MT, cmul(t, cmul(N, wtmp)), cmul(nshell, nshell), cmul(N, N), L.dfh_retained, Mr,
                        cmul(3, Mr), pr, eig_r});
    L.dfh_integral = L.integrals["dfhelper"] = doubles_for(dfhelper_pass_bytes(primary, auxiliary, t));
    L.stages["raw_dfhelper"] = cadd({L.dfh_extra, L.dfh_integral, dfh_min});

    // Declared pass, beside the still-live raw DFHelper: T, J_d^-1/2 (or its power() transient),
    // one first-index block of raw/T/out rows with their row pointers.
    L.stages["declared_pass"] = cadd({L.dfh_retained, MT, Md,
                                      cmax({cmul(big, cadd({pr, cmul(2, pd), cmul(3, kRowPointer)})),
                                            cadd({cmul(2, Md), pd, eig_d})})});

    const size_t ov = n;
    if (hyb) {
        const size_t k = std::max(size_t{1}, v / o) + 1;
        // Minimum blocks (one occupied index) of the runtime blocking in exchange_X/exchange_Y and
        // form_aux_matrices, each slice and single-layer matrix with its row pointers.
        L.stages["qr"] = cmax({cadd({mat(pd, n), mat(n, pd), std::min(n, pd), lwork_qr(n, pd)}),
                               cadd({mat(pd, n), mat(n, pd), Md, pd}),
                               cadd({cmul(3, Md), cmul(9, pd), lwork_gesdd('S', pd)})});
        L.stages["form_X"] = cadd({cmul(t, cmul(v, v)), cmul(v, kRowPointer), cmul(6, mat(v, pd))});
        L.stages["form_Y"] = cadd({cmul(t, ov), cmul(v, kRowPointer),
                                   cmul(cadd({cmul(k, k), cmul(4, k), 1}), mat(o, pd))});
        L.stages["dyson_diagnostic"] = cmax({cadd({cmul(2, mat(o, v)), cmul(6, Md), cmul(7, mat(v, pd))}),
                                            cmul(7, Md), cadd({cmul(2, Md), cmul(9, pd), lwork_gesdd('N', pd)})});
    } else {
        L.stages["dyson_diagnostic"] = cmax({cadd({cmul(2, mat(v, pd)), Md, mat(o, v)}), cmul(3, Md),
                                            cadd({cmul(2, Md), cmul(9, pd), lwork_gesdd('N', pd)})});
    }
    // S2 blocks; then its J - J A admission SVD (chi0, A, the matrix and its copy); then the solve.
    L.stages["s2_response"] = cmax({cadd({cmul(2, ov), cmul(5, Md), cmul(hyb ? 8 : 3, mat(v, pd))}),
                                    cadd({cmul(4, Md), cmul(9, pd), lwork_gesdd('N', pd)}), cadd({cmul(5, Md), pd})});

    // Disk: DIRECT_iaQ keeps a pre-metric and a final file per transform plus two metric files.
    const size_t npr = cmul(n, pr), npd = cmul(n, pd);
    size_t raw = cadd({cmul(2, npr), cmul(2, pr2)});
    size_t pass = cmul(3, npd);
    size_t later = 0;
    if (hyb) {
        raw = cadd({raw, cmul(2, npr), cmul(2, cmul(cmul(o, o), pr)), cmul(2, cmul(cmul(v, v), pr))});
        pass = cadd({cmul(7, npd), cmul(cmul(o, o), pd), cmul(cmul(v, v), pd)});
        later = cmul(10, npd);  // QarQ, QraQ, X/Y of (B, Q) and of (b, b_t)
    }
    L.files["raw_peak"] = raw;
    L.files["steady"] = cadd({pass, later});
    L.disk = cmax({cadd({raw, pass}), cadd({pass, later})});

    size_t peak = 0;
    for (const auto& kv : L.stages) peak = std::max(peak, kv.second);
    L.memory = cadd({L.resident, peak});
    return L;
}


}  // namespace

std::map<std::string, size_t> FDDS_Monomer::requirement(std::shared_ptr<BasisSet> primary,
                                                        std::shared_ptr<BasisSet> auxiliary, size_t nocc, size_t nvir,
                                                        size_t naux, bool is_hybrid, const std::string& subalgo,
                                                        size_t nthread) {
    if (!primary || !auxiliary || !nocc || !nvir || !naux || !nthread)
        throw PSIEXCEPTION("FDDS: requirement needs bases and nonzero dimensions and threads.");
    if (subalgo != "INCORE" && subalgo != "OUT_OF_CORE")
        throw PSIEXCEPTION("FDDS: subalgo must be INCORE or OUT_OF_CORE.");
    auto L = declared_ledger(*primary, *auxiliary, nocc, nvir, naux, is_hybrid, subalgo, nthread);
    std::map<std::string, size_t> ret{{"memory_bytes", cmul(L.memory, sizeof(double))},
                                      {"disk_bytes", cmul(L.disk, sizeof(double))},
                                      {"resident_bytes", cmul(L.resident, sizeof(double))}};
    for (const auto& kv : L.stages) ret["stage:" + kv.first] = cmul(kv.second, sizeof(double));
    for (const auto& kv : L.files) ret["disk:" + kv.first] = cmul(kv.second, sizeof(double));
    for (const auto& kv : L.integrals) ret["integral:" + kv.first] = cmul(kv.second, sizeof(double));
    return ret;
}

FDDS_Monomer::FDDS_Monomer(std::shared_ptr<BasisSet> primary, std::shared_ptr<BasisSet> auxiliary,
                           SharedMatrix Cocc, SharedMatrix Cvir, SharedVector eps_occ, SharedVector eps_vir,
                           bool is_hybrid, const FDDSResources& resources, SharedMatrix aux_transform)
    : FDDS_Dispersion(primary, auxiliary, {{"Cocc_A", Cocc}, {"Cvir_A", Cvir}},
                      {{"eps_occ_A", eps_occ}, {"eps_vir_A", eps_vir}}, is_hybrid, Deferred{}) {
    // The inputs are borrowed until prepare_declared has validated and admitted them, then copied.
    declared_ = true;
    prepare_declared(resources, aux_transform);
}

const FDDSModel& FDDS_Monomer::model() const {
    if (!declared_) throw PSIEXCEPTION("FDDS: model() is available on the declared path only.");
    return model_;
}

std::vector<SharedMatrix> FDDS_Monomer::project_densities(std::vector<SharedMatrix> dens) {
    // A raw-auxiliary (SAPT gridless-kernel) operation outside the declared storage ledger.
    if (declared_) throw PSIEXCEPTION("FDDS: project_densities is unavailable on the declared path.");
    return FDDS_Dispersion::project_densities(dens);
}

void FDDS_Monomer::prepare_declared(const FDDSResources& res, SharedMatrix T) {
    // ==> Validate: nothing below allocates beyond the copies before admission <== //
    const size_t pr = auxiliary_->nbf();
    const auto eo = vector_cache_.at("eps_occ_A"), ev = vector_cache_.at("eps_vir_A");
    const size_t nocc = eo->dim(0), nvir = ev->dim(0);
    if (T) {
        if (T->nirrep() != 1 || T->colspi()[0] != (int)pr || T->rowspi()[0] < 1)
            throw PSIEXCEPTION("FDDS: aux_transform must be C1 with naux_declared >= 1 rows and raw naux columns.");
        if (!all_finite(*T)) throw PSIEXCEPTION("FDDS: aux_transform must be finite.");
    }
    has_transform_ = (bool)T;
    naux_ = T ? T->rowspi()[0] : pr;
    const size_t pd = naux_;
    if (res.subalgo != "INCORE" && res.subalgo != "OUT_OF_CORE")
        throw PSIEXCEPTION("FDDS: subalgo must be INCORE or OUT_OF_CORE.");
    if (!writable_directory(res.scratch_dir))
        throw PSIEXCEPTION("FDDS: scratch_dir must be an existing writable directory.");
    double homo = eo->get(0), lumo = ev->get(0);
    for (size_t i = 0; i < nocc; i++) homo = std::max(homo, eo->get(i));
    for (size_t a = 0; a < nvir; a++) lumo = std::min(lumo, ev->get(a));
    if (!(lumo - homo > 0.0))
        throw PSIEXCEPTION("FDDS: the declared path requires every virtual energy above every occupied energy.");

    if (res.nthread < 1) throw PSIEXCEPTION("FDDS: nthread must be at least 1.");
    nthread_ = res.nthread;

    // ==> Admission, in bytes, before any integral or file <== //
    const auto L = declared_ledger(*primary_, *auxiliary_, nocc, nvir, pd, is_hybrid_, res.subalgo, nthread_);
    model_.naux_raw = pr;
    model_.naux = pd;
    model_.nocc = nocc;
    model_.nthread = nthread_;
    model_.nvir = nvir;
    model_.is_hybrid = is_hybrid_;
    model_.memory_bytes = res.memory_bytes;
    model_.disk_bytes = res.disk_bytes;
    model_.required_memory_bytes = cmul(L.memory, sizeof(double));
    model_.required_disk_bytes = cmul(L.disk, sizeof(double));
    if (res.memory_bytes < model_.required_memory_bytes)
        throw PSIEXCEPTION("FDDS: memory_bytes " + std::to_string(res.memory_bytes) + " is below the required " +
                           std::to_string(model_.required_memory_bytes) + " bytes.");
    if (res.disk_bytes < model_.required_disk_bytes)
        throw PSIEXCEPTION("FDDS: disk_bytes " + std::to_string(res.disk_bytes) + " is below the required " +
                           std::to_string(model_.required_disk_bytes) + " bytes.");
    work_doubles_ = res.memory_bytes / sizeof(double) - L.resident;

    // Admitted: own the orbital data and T (both are in the ledger).
    for (const auto& key : {"Cocc_A", "Cvir_A"}) matrix_cache_[key] = matrix_cache_[key]->clone();
    for (const auto& key : {"eps_occ_A", "eps_vir_A"})
        vector_cache_[key] = std::make_shared<Vector>(*vector_cache_[key]);
    if (T) T = T->clone();

    // ==> Declared metric: T is applied before every power, factorization and QR <== //
    // J_r also replaces the raw DFHelper's FittingMetric, whose integrals use the global thread count.
    auto Jr = form_coulomb_metric(auxiliary_, nthread_, true);
    auto J = Jr;
    auto S = form_aux_overlap(auxiliary_);
    if (T) {
        J = symmetrized(linalg::triplet(T, Jr, T, false, false, true));
        S = symmetrized(linalg::triplet(T, S, T, false, false, true));
    }
    metric_ = J;
    aux_overlap_ = S;

    // Native T1/T2 rule on J_d: |lambda| < 1e-12 max|lambda| is dropped; a kept negative lambda
    // would make power(-1/2)^2 differ from power(-1), so it is refused.
    {
        auto evecs = std::make_shared<Matrix>(pd, pd);
        auto evals = std::make_shared<Vector>(pd);
        J->diagonalize(evecs, evals, ascending);
        double lmax = std::max(std::fabs(evals->get(0)), std::fabs(evals->get(pd - 1)));
        model_.metric_min_kept = 1.0;
        for (size_t i = 0; i < pd; i++) {
            double ratio = std::fabs(evals->get(i)) / lmax;
            if (ratio < metric_cutoff) {
                model_.metric_dropped++;
                model_.metric_max_dropped = std::max(model_.metric_max_dropped, ratio);
            } else if (evals->get(i) < 0.0) {
                throw PSIEXCEPTION("FDDS: the declared metric has a kept negative eigenvalue.");
            } else {
                model_.metric_min_kept = std::min(model_.metric_min_kept, ratio);
            }
        }
        if (!(lmax > 0.0) || !std::isfinite(lmax))
            throw PSIEXCEPTION("FDDS: the declared metric is zero or nonfinite.");
    }
    metric_inv_ = J->clone();
    metric_inv_->power(-1.0, metric_cutoff);

    // Full J_d^-1 enters only through LU solves (b = B_d J_d^-1), never an explicit inverse.
    metric_lu_ = J->clone();
    metric_ipiv_.assign(pd, 0);
    double anorm = 0.0;
    for (size_t j = 0; j < pd; j++) {
        double col = 0.0;
        for (size_t i = 0; i < pd; i++) col += std::fabs(J->get(i, j));
        anorm = std::max(anorm, col);
    }
    if (C_DGETRF(pd, pd, metric_lu_->pointer()[0], pd, metric_ipiv_.data()) != 0)
        throw PSIEXCEPTION("FDDS: the declared metric is singular to working precision.");
    {
        std::vector<double> work(4 * pd);
        std::vector<int> iwork(pd);
        C_DGECON('1', pd, metric_lu_->pointer()[0], pd, anorm, &model_.metric_lu_rcond, work.data(), iwork.data());
    }

    // ==> Raw integrals in a private scratch path, then the declared streams <== //
    std::vector<SharedMatrix> C = {matrix_cache_["Cocc_A"], matrix_cache_["Cvir_A"]};
    auto raw = std::make_shared<DFHelper>(primary_, auxiliary_);
    raw->set_scratch_path(res.scratch_dir);
    raw->set_subalgo(res.subalgo);
    raw->set_memory(work_doubles_ - L.dfh_extra - L.dfh_integral);
    raw->set_fitting_metric(Jr);
    Jr.reset();  // the DFHelper releases it once the metric file is written
    raw->set_libint2_eri(true);
    raw->set_method("DIRECT_iaQ");
    raw->set_nthreads(nthread_);
    raw->set_metric_pow(0.0);
    raw->initialize();
    raw->print_header();
    raw->add_space("a", C[0]);
    raw->add_space("r", C[1]);
    raw->add_transformation("arQ", "a", "r", "pqQ");
    if (is_hybrid_) {
        raw->add_transformation("raQ", "r", "a", "pqQ");
        raw->add_transformation("aaQ", "a", "a", "pqQ");
        raw->add_transformation("rrQ", "r", "r", "pqQ");
    }
    raw->set_release_core_AO_before_metric(true);
    raw->transform();

    dfh_ = std::make_shared<DFHelper>(primary_, auxiliary_);
    dfh_->set_scratch_path(res.scratch_dir);
    SharedMatrix half_inv;
    if (is_hybrid_) {
        half_inv = J->clone();
        half_inv->power(-0.5, metric_cutoff);
    }
    declared_pass(raw, T, half_inv, L.dfh_retained);
    raw.reset();  // removes the raw files
    T.reset();
    half_inv.reset();

    if (is_hybrid_) {
        qr_from_arQ_ = true;
        timer_on("FDDS: QR");
        R_A_ = QR("A");
        timer_off("FDDS: QR");
        // numpy.linalg.pinv(R, rcond=1e-13).T = U S^+ V^T for R = U S V^T. The SVD factors are
        // scoped so they are released before form_X/form_Y block from the whole work budget.
        {
            auto U = std::make_shared<Matrix>(pd, pd);
            auto V = std::make_shared<Matrix>(pd, pd);
            auto sv = square_svd(R_A_, U, V);
            double cut = r_rcond * sv->get(0);
            for (size_t k = 0; k < pd; k++) {
                double s = sv->get(k);
                bool keep = s > cut;
                model_.qr_rank += keep;
                U->scale_column(0, k, keep ? 1.0 / s : 0.0);
            }
            Rtinv_ = linalg::doublet(U, V);
        }

        timer_on("FDDS: Form X");
        form_X("A");
        exchange_X("arR", {"b_ar", "bt_ar"}, {"Xb", "Xbt"}, nocc, nvir);
        timer_off("FDDS: Form X");
        timer_on("FDDS: Form Y");
        form_Y("A");
        exchange_Y("aaR", "rrR", {"b_ra", "bt_ra"}, {"Yb", "Ybt"}, nocc, nvir);
        timer_off("FDDS: Form Y");
    }
}

void FDDS_Monomer::declared_pass(std::shared_ptr<DFHelper> raw, const SharedMatrix& T, const SharedMatrix& half_inv,
                                 size_t raw_retained) {
    // Streams raw (xy|P_r) blocks of the first index into declared tensors: B_d = B_r T^T, then
    // B_d, B_d J_d^-1 (LU solve), B_d J^+_d or B_d J_d^-1/2 (DFHelper's metric_pow -1/2 on J_d).
    enum Kind { Declared, Solve, Plus, Half };
    struct Job {
        std::string raw;
        size_t n1, n2;
        std::vector<std::pair<std::string, Kind>> out;
    };
    const size_t o = model_.nocc, v = model_.nvir, pr = model_.naux_raw, pd = naux_;
    std::vector<Job> jobs = {{"arQ", o, v, {{"arQ", Declared}, {"b_ar", Solve}, {"bt_ar", Plus}}}};
    if (is_hybrid_) {
        jobs[0].out.push_back({"arR", Half});
        jobs.push_back({"raQ", v, o, {{"raQ", Declared}, {"b_ra", Solve}, {"bt_ra", Plus}}});
        jobs.push_back({"aaQ", o, o, {{"aaR", Half}}});
        jobs.push_back({"rrQ", v, v, {{"rrR", Half}}});
    }
    const size_t fixed = cadd({raw_retained, mat(pd, pr), mat(pd, pd)});
    for (const auto& job : jobs) {
        size_t per = cmul(job.n2, cadd({pr, cmul(2, pd), cmul(3, kRowPointer)}));
        size_t block = std::min(job.n1, (work_doubles_ - fixed) / per);
        if (block < 1) throw PSIEXCEPTION("FDDS: declared pass exceeds the admitted work budget.");
        auto in = std::make_shared<Matrix>("raw", block * job.n2, pr);
        auto Bd = T ? std::make_shared<Matrix>("declared", block * job.n2, pd) : in;
        auto out = std::make_shared<Matrix>("out", block * job.n2, pd);
        for (const auto& target : job.out) dfh_->add_disk_tensor(target.first, std::make_tuple(job.n1, job.n2, pd));
        auto& seen = model_.declared_pass_blocks[job.raw];
        for (size_t start = 0; start < job.n1; start += block) {
            size_t nb = std::min(block, job.n1 - start), rows = nb * job.n2;
            seen[0]++;
            seen[1] = std::max(seen[1], rows);
            raw->fill_tensor(job.raw, in, {start, start + nb});
            if (T) C_DGEMM('N', 'T', rows, pd, pr, 1.0, in->pointer()[0], pr, T->pointer()[0], pr, 0.0,
                           Bd->pointer()[0], pd);
            for (const auto& target : job.out) {
                SharedMatrix dst = out;
                if (target.second == Declared) {
                    dst = Bd;
                } else if (target.second == Solve) {
                    C_DCOPY(rows * pd, Bd->pointer()[0], 1, out->pointer()[0], 1);
                    // The row-major rows are column-major right-hand sides of the symmetric J_d.
                    if (C_DGETRS('N', pd, rows, metric_lu_->pointer()[0], pd, metric_ipiv_.data(), out->pointer()[0],
                                 pd) != 0)
                        throw PSIEXCEPTION("FDDS: DGETRS failed.");
                } else {
                    const auto& M = target.second == Plus ? metric_inv_ : half_inv;
                    C_DGEMM('N', 'N', rows, pd, pd, 1.0, Bd->pointer()[0], pd, M->pointer()[0], pd, 0.0,
                            out->pointer()[0], pd);
                }
                dfh_->write_disk_tensor(target.first, dst, {start, start + nb});
            }
        }
    }
}

double FDDS_Monomer::native_dyson_ratio(double omega, double x_alpha, const SharedMatrix& W) {
    // J_d - XSW in the verbatim fdds_coupled_amplitudes order; the SVD is only a diagnostic.
    SharedMatrix X, KRS;
    if (is_hybrid_) {
        auto aux = FDDS_Dispersion::form_aux_matrices("A", omega);
        X = aux["amp"]->clone();
        X->axpy(-x_alpha, aux["K2L"]);
        auto K = aux["K1LD"]->clone();
        K->scale(-x_alpha);
        K->axpy(-x_alpha, aux["K2LD"]);
        K->axpy(x_alpha * x_alpha, aux["K21L"]);
        aux.clear();
        KRS = linalg::doublet(linalg::doublet(K, Rtinv_), metric_);
    } else {
        X = FDDS_Dispersion::form_unc_amplitude("A", omega);
        X->scale(-1.0);
    }
    auto XSW = linalg::doublet(linalg::doublet(X, metric_inv_), W);
    X.reset();
    if (KRS) XSW->axpy(0.25, KRS);
    KRS.reset();
    auto M = metric_->clone();
    M->subtract(XSW);
    XSW.reset();
    if (!all_finite(*M)) return std::numeric_limits<double>::quiet_NaN();
    auto sv = square_svd(M);
    return sv->get(naux_ - 1) / sv->get(0);
}

FDDSResponse FDDS_Monomer::form_coefficient_response(double omega, double x_alpha, SharedMatrix W) {
    if (!declared_) throw PSIEXCEPTION("FDDS: form_coefficient_response requires the declared constructor.");
    if (!std::isfinite(omega) || omega < 0.0)
        throw PSIEXCEPTION("FDDS: imaginary frequency must be finite and nonnegative.");
    if (!std::isfinite(x_alpha) || (!is_hybrid_ && x_alpha != 0.0))
        throw PSIEXCEPTION("FDDS: x_alpha must be finite, and zero unless hybrid.");
    const size_t p = naux_, o = model_.nocc, v = model_.nvir;
    if (!W || W->nirrep() != 1 || W->rowspi()[0] != (int)p || W->colspi()[0] != (int)p || !all_finite(*W))
        throw PSIEXCEPTION("FDDS: kernel must be a finite C1 naux by naux matrix in the declared basis.");

    FDDSResponse ret;
    ret.omega = omega;
    ret.x_alpha = x_alpha;
    // Dual admission (empirical, not an accuracy or sign certificate): both formations of J_d - XSW
    // must keep sigma_min/sigma_max > 2e-13. Either check refuses; the native one runs first.
    auto refuse = [&](const std::string& check, double ratio, const std::string& other) {
        std::stringstream msg;
        msg << "FDDS: Dyson admission refused by the " << check << " check: sigma_min/sigma_max = "
            << std::setprecision(5) << ratio << " <= " << dyson_refusal << " at omega = " << omega << "; " << other
            << ".";
        throw PSIEXCEPTION(msg.str());
    };
    ret.native_dyson_ratio = native_dyson_ratio(omega, x_alpha, W);
    if (!(ret.native_dyson_ratio > dyson_refusal))
        refuse("native J - XSW", ret.native_dyson_ratio, "the S2 J - J*A check was not evaluated");

    // ==> S2: chi0, chi0_in and Kc streamed over occupied blocks <== //
    const double* eoccp = vector_cache_["eps_occ_A"]->pointer();
    const double* evirp = vector_cache_["eps_vir_A"]->pointer();
    std::vector<double> Lar(o * v), LDar(o * v);
    for (size_t i = 0; i < o; i++)
        for (size_t a = 0; a < v; a++) {
            double val = evirp[a] - eoccp[i];
            double ll = -4.0 / (val * val + omega * omega);
            Lar[i * v + a] = ll;
            LDar[i * v + a] = val * ll;
            // The native non-hybrid amplitude zeroes 4D/(D^2+w^2) < 1e-14 (form_unc_amplitude).
            if (!is_hybrid_ && 4.0 * val / (val * val + omega * omega) < 1.e-14) {
                LDar[i * v + a] = 0.0;
                ret.masked_transitions++;
            }
        }
    const size_t nstream = is_hybrid_ ? 8 : 3;
    const size_t fixed = cadd({cmul(2, o * v), cmul(5, mat(p, p))});
    size_t maxo = std::min(o, (work_doubles_ - fixed) / cmul(nstream, mat(v, p)));
    if (maxo < 1) throw PSIEXCEPTION("FDDS: response exceeds the admitted work budget.");
    std::vector<SharedMatrix> buf(nstream);
    for (auto& m : buf) m = std::make_shared<Matrix>(maxo * v, p);
    auto &b = buf[0], &bt = buf[1], &lDb = buf[2];
    auto chi0 = std::make_shared<Matrix>("chi0", p, p);
    auto chi0in = std::make_shared<Matrix>("chi0_in", p, p);
    SharedMatrix K1, K2, K21;
    if (is_hybrid_) {
        K1 = std::make_shared<Matrix>(p, p);
        K2 = std::make_shared<Matrix>(p, p);
        K21 = std::make_shared<Matrix>(p, p);
    }
    auto gemm_tn = [p](size_t rows, double alpha, const SharedMatrix& A, const SharedMatrix& B, SharedMatrix& C) {
        C_DGEMM('T', 'N', p, p, rows, alpha, A->pointer()[0], p, B->pointer()[0], p, 1.0, C->pointer()[0], p);
    };
    auto scale_rows = [p, v](SharedMatrix& M, const std::vector<double>& f, size_t astart, size_t rows) {
        double** Mp = M->pointer();
        for (size_t r = 0; r < rows; r++) C_DSCAL(p, f[astart * v + r], Mp[r], 1);
    };
    for (size_t astart = 0; astart < o; astart += maxo) {
        size_t nb = std::min(maxo, o - astart), rows = nb * v;
        dfh_->fill_tensor("b_ar", b, {astart, astart + nb});
        dfh_->fill_tensor("bt_ar", bt, {astart, astart + nb});
        lDb->copy(b);
        scale_rows(lDb, LDar, astart, rows);
        gemm_tn(rows, 1.0, lDb, b, chi0);
        gemm_tn(rows, 1.0, lDb, bt, chi0in);
        if (!is_hybrid_) continue;
        auto &Ymb = buf[3], &Ymbt = buf[4], &Xtmp = buf[5], &EpQ = buf[6], &EmQ = buf[7];
        // E_- b = (Y - X) b; E_+Q = (X + Y)Q and E_-Q = (Y - X)Q exactly as form_aux_matrices combines them.
        dfh_->fill_tensor("Yb", Ymb, {astart, astart + nb});
        dfh_->fill_tensor("Xb", Xtmp, {astart, astart + nb});
        Ymb->axpy(-1.0, Xtmp);
        dfh_->fill_tensor("Ybt", Ymbt, {astart, astart + nb});
        dfh_->fill_tensor("Xbt", Xtmp, {astart, astart + nb});
        Ymbt->axpy(-1.0, Xtmp);
        dfh_->fill_tensor("QXarQ", EpQ, {astart, astart + nb});
        dfh_->fill_tensor("QYarQ", EmQ, {astart, astart + nb});
        EpQ->axpy(1.0, EmQ);
        EmQ->scale(2.0);
        EmQ->axpy(-1.0, EpQ);
        gemm_tn(rows, 1.0, lDb, EpQ, K1);
        gemm_tn(rows, 1.0, lDb, EmQ, K2);
        scale_rows(Ymb, Lar, astart, rows);   // l E_- b
        scale_rows(Ymbt, Lar, astart, rows);  // l E_- b_t
        gemm_tn(rows, -x_alpha, b, Ymb, chi0);
        gemm_tn(rows, -x_alpha, b, Ymbt, chi0in);
        gemm_tn(rows, 1.0, Ymb, EpQ, K21);
    }
    buf.clear();
    std::vector<double>().swap(Lar);
    std::vector<double>().swap(LDar);

    // A = chi0_in W + 1/4 Kc pinv(R)^T J_d, Kc = -a(K1 + K2) + a^2 K21 (= J_d^-1 Kx). Kc is
    // combined first so that at most four p x p matrices are live from here on.
    if (is_hybrid_) {
        K1->add(K2);
        K1->scale(-x_alpha);
        K1->axpy(x_alpha * x_alpha, K21);
        K2.reset();
        K21.reset();
    }
    auto A = linalg::doublet(chi0in, W);
    chi0in.reset();
    if (is_hybrid_) {
        auto KR = linalg::doublet(K1, Rtinv_);
        K1.reset();
        A->gemm(false, false, 0.25, KR, metric_, 1.0);
    }
    {
        auto JA = linalg::doublet(metric_, A);
        JA->scale(-1.0);
        JA->add(metric_);
        ret.s2_dyson_ratio = std::numeric_limits<double>::quiet_NaN();
        if (all_finite(*JA)) {
            auto sv = square_svd(JA);
            ret.s2_dyson_ratio = sv->get(p - 1) / sv->get(0);
        }
    }
    if (!(ret.s2_dyson_ratio > dyson_refusal)) {
        std::stringstream other;
        other << "the native J - XSW check passed with " << std::setprecision(5) << ret.native_dyson_ratio;
        refuse("S2 J - J*A", ret.s2_dyson_ratio, other.str());
    }
    A->scale(-1.0);
    for (size_t i = 0; i < p; i++) A->add(0, i, i, 1.0);  // I - A

    // (I - A) chi = chi0: the row-major I - A is column-major (I - A)^T, so solve with 'T' and a
    // column-major chi0 (row-major chi0^T); the result is chi in column-major, i.e. chi^T here.
    auto lu = A->clone();
    std::vector<int> ipiv(p);
    if (C_DGETRF(p, p, lu->pointer()[0], p, ipiv.data()) != 0)
        throw PSIEXCEPTION("FDDS: I - A is singular to working precision.");
    auto chiT = chi0->transpose();
    if (C_DGETRS('T', p, p, lu->pointer()[0], p, ipiv.data(), chiT->pointer()[0], p) != 0)
        throw PSIEXCEPTION("FDDS: DGETRS failed.");
    lu.reset();
    {
        auto resid = linalg::doublet(A, chiT, false, true);
        resid->subtract(chi0);
        double norm0 = chi0->rms();
        ret.solve_residual = norm0 > 0.0 ? resid->rms() / norm0 : resid->rms();
    }
    chi0.reset();
    A.reset();
    ret.response = symmetrized(chiT);
    ret.response->set_name("FDDS coefficient response");
    return ret;
}

}  // namespace sapt
}  // namespace psi
