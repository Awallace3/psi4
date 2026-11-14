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

#include "dfhelper.h"

#include <algorithm>
#include <cstdlib>
#ifdef _MSC_VER
#include <process.h>
#define SYSTEM_GETPID ::_getpid
#else
#include <unistd.h>
#define SYSTEM_GETPID ::getpid
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <libint2/engine.h>
#include <libint2/shell.h>
#include "psi4/psi4-dec.h"
#include "psi4/liboptions/liboptions.h"
#include "psi4/libfock/jk.h"
#include "psi4/libmints/integral.h"
#include "psi4/libmints/vector.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/basisset.h"
#include "psi4/libmints/twobody.h"
#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libqt/qt.h"
#include "psi4/libpsi4util/process.h"
#include "psi4/libpsio/psio.hpp"
#include "psi4/libpsio/psio.h"
#include "psi4/libpsio/aiohandler.h"

#include "dftensor.h"

// Use the configured provider and its official ABI, not compiler-specific weak
// symbols (which are not portable to Windows or macOS).
#ifdef USING_LAPACK_MKL
#include <mkl_service.h>
#endif

namespace psi {

DFHelper::DFHelper(std::shared_ptr<BasisSet> primary, std::shared_ptr<BasisSet> aux)
    : primary_(primary), aux_(aux) {
    if(Process::environment.options["SCF_SUBTYPE"].has_changed()) {
        subalgo_ = Process::environment.options.get_str("SCF_SUBTYPE");
    }
    else {
        // Default to in-core for MemDF b/c out-of-core is so slow we don't want it to be hit w/o explicit user SCF_SUBTYPE.
        // By setting this, checks below will crash if memory is insufficient.
        // Needed b/c the enough_mem ? mem : disk memory logic is slightly different from the DFHelper memory logic, so one can hit MemDF+Disk_algorithm by accident.
        subalgo_ = "INCORE";
    }
    
    nbf_ = primary_->nbf();
    naux_ = aux_->nbf();
    prepare_blocking();
}

DFHelper::~DFHelper() { clear_all(); }

void DFHelper::prepare_blocking() {
    Qshells_ = aux_->nshell();
    pshells_ = primary_->nshell();

    Qshell_aggs_.resize(Qshells_ + 1);
    pshell_aggs_.resize(pshells_ + 1);

    // Aux shell blocking
    Qshell_max_ = aux_->max_function_per_shell();
    Qshell_aggs_[0] = 0;
    for (size_t i = 0; i < Qshells_; i++) {
        Qshell_aggs_[i + 1] = Qshell_aggs_[i] + aux_->shell(i).nfunction();
    }

    // AO shell blocking
    pshell_aggs_[0] = 0;
    for (size_t i = 0; i < pshells_; i++) {
        pshell_aggs_[i + 1] = pshell_aggs_[i] + primary_->shell(i).nfunction();
    }
}

void DFHelper::AO_filename_maker(size_t i) {
    auto name = start_filename("dfh.AO" + std::to_string(i));
    AO_names_.push_back(name);
}

std::string DFHelper::start_filename(std::string start) {
    std::string name = PSIOManager::shared_object()->get_default_path();
    std::string safe_mol = primary_->molecule()->name();
    safe_mol.erase(std::remove(safe_mol.begin(), safe_mol.end(), '/'), safe_mol.end());
    name += start + "." + std::to_string(SYSTEM_GETPID());
    name += "." + safe_mol + ".";
    name += std::to_string(rand()) + "." + ".dat";
    return name;
}

void DFHelper::filename_maker(std::string name, size_t Q, size_t p, size_t q, size_t op) {
    auto pfilename = start_filename("dfh.p" + name);
    auto filename = start_filename("dfh" + name);

    std::tuple<std::string, std::string> files(pfilename, filename);
    files_[name] = files;

    bool is_transf = transf_.count(name);

    // direct_iaQ is special, because it has two different sizes
    if (direct_iaQ_ && is_transf) {
        sizes_[pfilename] = std::make_tuple(Q, p, q);
        sizes_[filename] = std::make_tuple(p, q, Q);

    } else {
        std::tuple<size_t, size_t, size_t> sizes;
        if (op == 0) {
            sizes = std::make_tuple(Q, p, q);
        } else if (op == 1) {
            sizes = std::make_tuple(p, Q, q);
        } else {
            sizes = std::make_tuple(p, q, Q);
        }

        sizes_[pfilename] = sizes;
        sizes_[filename] = sizes;
    }
}

void DFHelper::initialize() {
    if (debug_) {
        outfile->Printf("Entering DFHelper::initialize\n");
    }
    timer_on("DFH: initialize()");

    // have the algorithm specified before init
    if (method_.compare("DIRECT") && method_.compare("STORE") && method_.compare("DIRECT_iaQ")) {
        std::stringstream error;
        error << "DFHelper:initialize: specified method (" << method_ << ") is incorrect";
        throw PSIEXCEPTION(error.str().c_str());
    }

    // workflow holders
    direct_iaQ_ = (!method_.compare("DIRECT_iaQ") ? true : false);
    direct_ = (!method_.compare("DIRECT") ? true : false);

    // did we get enough memory for at least the metric?
    if (naux_ * naux_ > memory_) {
        std::stringstream error;
        error << "DFHelper: The Coulomb metric requires at least " << naux_ * naux_ * 8 / (1024 * 1024 * 1024.0)
              << "[GiB].  We need that plus some more, but we only got " << memory_ * 8 / (1024 * 1024 * 1024.0)
              << "[GiB].";
        throw PSIEXCEPTION(error.str().c_str());
    }

    // if metric power is not zero, prepare it
    if (!(std::fabs(mpower_ - 0.0) < 1e-13)) (hold_met_ ? prepare_metric_core() : prepare_metric());

    // if metric power for omega integrals is not zero, prepare its metric
    if (do_wK_) {
        if (!(std::fabs(wmpower_ - 0.0) < 1e-13) && (std::fabs(mpower_ - 0.0) < 1e-13))
            (hold_met_ ? prepare_metric_core() : prepare_metric());
    }

    // prepare sparsity masks
    prepare_sparsity();

    // figure out AO_core
    AO_core(true);

    // prepare AOs for STORE method
    if (AO_core_) {
        if (do_wK_) {
            prepare_AO_wK_core();
        } else {  // It is possible to reformulate the expression for the
            //   coulomb matrix to save memory in case do_wK_ is
            //   is true, but do_K_ is false. This code isn't written
            prepare_AO_core();
        }
    } else if (!direct_ && !direct_iaQ_) {
        prepare_AO();
        if (do_wK_) {
            std::stringstream error;
            error << "DFHelper: not equipped to do wK out of core. \nPlease supply more memory or remove scf_type Mem_DF from the imput file";
            throw PSIEXCEPTION(error.str().c_str());
            //prepare_AO_wK();
        }
    }

    built_ = true;
    timer_off("DFH: initialize()");

    if (debug_) {
        outfile->Printf("Exiting DFHelper::initialize\n");
    }
}
void DFHelper::AO_core(bool set_AO_core=true) {
    prepare_sparsity();

    if (direct_iaQ_) {
        // the direct_iaQ method does not use sparse storage
        // if do_wK added to code, the following will need to be changed to match
        required_core_size_ = naux_ * nbf_ * nbf_;
    } else {
        // total size of sparse AOs.
        // yes, a nested ternary operator
        required_core_size_ = (do_wK_ ? ( wcombine_ ? 2 * big_skips_[nbf_] : 3 * big_skips_[nbf_] ) : big_skips_[nbf_]);
    }

    // Auxiliary metric(s): prepare_AO_core holds one naux^2 metric; prepare_AO_wK_core holds two.
    required_core_size_ += ((do_wK_ ? 2 : 1) + (hold_met_ ? 1 : 0)) * naux_ * naux_;

    // Transients on top of the AOs and the metric: the JK build's C_buffers (conservative estimate
    // since I do not have max_nocc TODO) and tmp buffers, or, while the AOs are built, the metric
    // contraction's panels plus at least one p-shell AO block. They are never live together.
    // The half-of-spare heuristic caps wider panels, but does not require duplicating
    // the minimum panel allowance: its floor may use more than half the spare if
    // the panels and the largest individual p-shell block still fit together.
    // Retain the historical conservative bound, but also cover padding and
    // allocation floors. For tiny bases an IQ/FP32 occupied row padded to eight
    // can be much larger than nbf; nthreads*naux can dominate the J scratch too.
    const size_t padded_nocc = (nbf_ + 7) / 8 * 8;
    const size_t q_shell_max = static_cast<size_t>(Qshell_max_);
    const size_t tmp_floor = std::max({nbf_ * nbf_, nthreads_ * naux_,
                                      nbf_ * padded_nocc * q_shell_max});
    const size_t jk_tmp = std::max(nthreads_ * nbf_ * nbf_ + 3 * nbf_ * nbf_ * q_shell_max,
                                  nthreads_ * nbf_ * padded_nocc + 2 * tmp_floor);
    const size_t s_min = AO_core_init_scratch(true);
    const size_t panels = (direct_ || direct_iaQ_) ? 0 : s_min + min_AO_core_block();
    required_core_size_ += std::max(jk_tmp, panels);

    // set AO_core_ if requested
    if (set_AO_core) {
        if (print_lvl_ > 0) {
            outfile->Printf("  DFHelper Memory: AOs need %.3f GiB; user supplied %.3f GiB. \n",
                            (required_core_size_ * 8 / (1024 * 1024 * 1024.0)), (memory_ * 8 / (1024 * 1024 * 1024.0)));
        }

        // determine AO_core_ either automatically...
        if (subalgo_ == "AUTO") {
            // a fraction of memory to use, do we want it as an option?
            AO_core_ = true;
            if (memory_ < required_core_size_) AO_core_ = false;

        // .. or forcibly disable AO_core_ if user specifies ...
        } else if (subalgo_ == "OUT_OF_CORE") {
            AO_core_ = false;

            if (print_lvl_ > 0) {
                outfile->Printf("  SCF_SUBTYPE = OUT_OF_CORE selected. Out-of-core MEM_DF algorithm will be used.\n");
            }
        // .. or force AO_core_ if user specifies
        } else if (subalgo_ == "INCORE") {
            if (memory_ < required_core_size_) {
                throw PSIEXCEPTION("SCF_SUBTYPE=INCORE was specified, but there is not enough memory to do in-core! Increase the amount of memory allocated to Psi4 or allow for out-of-core to be used.\n");
            } else {
                AO_core_ = true;

                if (print_lvl_ > 0) {
                    outfile->Printf("  SCF_SUBTYPE=INCORE selected. In-core MEM_DF algorithm will be used.\n");
                }
            }
        } else {
            throw PSIEXCEPTION("Invalid SCF_SUBTYPE option! The choices for SCF_SUBTYPE are AUTO, INCORE, and OUT_OF_CORE.");
        }

        if (print_lvl_ > 0) {
            outfile->Printf("  %s in-core AOs.\n\n", AO_core_ ? "Using" : "Turning off");
        }
    }
}
void DFHelper::print_header() {
    // Preps any required metadata, safe to call multiple times
    get_core_size();

    outfile->Printf("  ==> DFHelper <==\n");
    outfile->Printf("    NBF:                     %11ld\n", nbf_);
    outfile->Printf("    NAux:                    %11ld\n", naux_);
    outfile->Printf("    Schwarz Cutoff:          %11.0E\n", cutoff_);
    outfile->Printf("    Mask sparsity (%%):       %11.0f\n", 100. * ao_sparsity());
    outfile->Printf("    DFH Avail. Memory [GiB]: %11.3f\n", (memory_ * 8L) / ((double)(1024L * 1024L * 1024L)));
    outfile->Printf("    OpenMP threads:          %11zu\n", nthreads_);
    outfile->Printf("    Algorithm:               %11s\n", method_.c_str());
    outfile->Printf("    AO Core:                 %11s\n", (AO_core_ ? "True" : "False"));
    outfile->Printf("    MO Core:                 %11s\n", (MO_core_ ? "True" : "False"));
    outfile->Printf("    Hold Metric:             %11s\n", (hold_met_ ? "True" : "False"));
    outfile->Printf("    Metric Power:            %11.3f\n", mpower_);
    outfile->Printf("    Fitting Condition:       %11.0E\n", condition_);
    outfile->Printf("    Q Shell Max:             %11d\n", (int)Qshell_max_);
    outfile->Printf("\n\n");
}

void DFHelper::prepare_sparsity() {
    if (sparsity_prepared_) return;
    timer_on("DFH: sparsity prep");

    // => Initialize vectors <=
    std::vector<double> shell_max_vals(pshells_ * pshells_, 0.0);
    std::vector<double> fun_max_vals(nbf_ * nbf_, 0.0);
    schwarz_shell_mask_.resize(pshells_ * pshells_);
    schwarz_fun_index_.resize(nbf_ * nbf_);
    symm_ignored_columns_.resize(nbf_);
    symm_big_skips_.resize(nbf_ + 1);
    symm_small_skips_.resize(nbf_);
    small_skips_.resize(nbf_ + 1);
    big_skips_.resize(nbf_ + 1);

    // => For each shell pair and basis pair, store the max (mn|mn)-type integral for screening. <=
    double max_val = 0.0;
    const bool use_simint = (Process::environment.options.get_str("INTEGRAL_PACKAGE") == "SIMINT");
    if (!use_simint) {
        const int max_am = primary_->max_am();
        const int max_nprim = primary_->max_nprimitive();
        std::vector<libint2::Engine> engs;
        engs.reserve(nthreads_);
        for (size_t t = 0; t < nthreads_; ++t) {
            engs.emplace_back(libint2::Operator::coulomb, max_nprim, max_am, 0, 0.0,
                              libint2::operator_traits<libint2::Operator::coulomb>::default_params(),
                              libint2::BraKet::xx_xx);
        }
#pragma omp parallel for num_threads(nthreads_) schedule(guided) reduction(max : max_val)
        for (size_t MU = 0; MU < pshells_; ++MU) {
            int rank = 0;
#ifdef _OPENMP
            rank = omp_get_thread_num();
#endif
            auto& eng = engs[rank];
            const auto& sh_mu = primary_->l2_shell(MU);
            size_t nmu = primary_->shell(MU).nfunction();
            size_t bmu = primary_->shell(MU).function_index();
            for (size_t NU = 0; NU <= MU; ++NU) {
                const auto& sh_nu = primary_->l2_shell(NU);
                size_t nnu = primary_->shell(NU).nfunction();
                size_t bnu = primary_->shell(NU).function_index();
                eng.compute(sh_mu, sh_nu, sh_mu, sh_nu);
                const double* buffer = eng.results()[0];
                double sh_max = 0.0;
                if (buffer != nullptr) {
                    for (size_t mu = 0; mu < nmu; ++mu) {
                        size_t omu = bmu + mu;
                        for (size_t nu = 0; nu < nnu; ++nu) {
                            size_t onu = bnu + nu;
                            if (omu >= onu) {
                                size_t index = mu * (nnu * nmu * nnu + nnu) + nu * (nmu * nnu + 1);
                                double val = std::fabs(buffer[index]);
                                sh_max = std::max(sh_max, val);
                                fun_max_vals[omu * nbf_ + onu] = val;
                                fun_max_vals[onu * nbf_ + omu] = val;
                            }
                        }
                    }
                }
                max_val = std::max(max_val, sh_max);
                shell_max_vals[MU * pshells_ + NU] = sh_max;
                shell_max_vals[NU * pshells_ + MU] = sh_max;
            }
        }
    } else {
        auto rifactory = std::make_shared<IntegralFactory>(primary_, primary_, primary_, primary_);
        std::vector<std::shared_ptr<TwoBodyAOInt>> eri(nthreads_);
        eri[0] = std::shared_ptr<TwoBodyAOInt>(rifactory->eri(0, false));
        for (size_t rank = 1; rank < nthreads_; rank++) {
            eri[rank] = std::shared_ptr<TwoBodyAOInt>(eri.front()->clone());
        }
#pragma omp parallel for num_threads(nthreads_) schedule(guided) reduction(max : max_val)
        for (size_t MU = 0; MU < pshells_; ++MU) {
            int rank = 0;
#ifdef _OPENMP
            rank = omp_get_thread_num();
#endif
            const auto& buffers = eri[rank]->buffers();
            size_t nmu = primary_->shell(MU).nfunction();
            for (size_t NU = 0; NU <= MU; ++NU) {
                size_t nnu = primary_->shell(NU).nfunction();
                eri[rank]->compute_shell(MU, NU, MU, NU);
                const auto* buffer = buffers[0];
                for (size_t mu = 0; mu < nmu; ++mu) {
                    size_t omu = primary_->shell(MU).function_index() + mu;
                    for (size_t nu = 0; nu < nnu; ++nu) {
                        size_t onu = primary_->shell(NU).function_index() + nu;
                        if (omu >= onu) {
                            size_t index = mu * (nnu * nmu * nnu + nnu) + nu * (nmu * nnu + 1);
                            double val = std::fabs(buffer[index]);
                            max_val = std::max(val, max_val);
                            if (shell_max_vals[MU * pshells_ + NU] <= val) {
                                shell_max_vals[MU * pshells_ + NU] = val;
                                shell_max_vals[NU * pshells_ + MU] = val;
                            }
                            if (fun_max_vals[omu * nbf_ + onu] <= val) {
                                fun_max_vals[omu * nbf_ + onu] = val;
                                fun_max_vals[onu * nbf_ + omu] = val;
                            }
                        }
                    }
                }
            }
        }
    }

    // => Prepare screening/indexing data <=
    double tolerance = cutoff_ * cutoff_ / max_val;

    // ==> Is this shell pair significant? <==
    for (size_t i = 0; i < pshells_ * pshells_; i++) schwarz_shell_mask_[i] = (shell_max_vals[i] >= tolerance);

    // ==> Is this basis function pair significant? Also start storing non-symmetric indexing & symmetric counts. <==
#pragma omp parallel for num_threads(nthreads_) schedule(static) if (nbf_ >= 256)
    for (size_t i = 0; i < nbf_; i++) {
        size_t count = 0;
        size_t symm_size = 0;
        size_t symm_skip = 0;
        const double* frow = &fun_max_vals[i * nbf_];
        size_t* srow = &schwarz_fun_index_[i * nbf_];
        for (size_t j = 0; j < nbf_; j++) {
            if (frow[j] >= tolerance) {
                count++;
                srow[j] = count;
                if (j >= i)
                    symm_size++;
                else
                    symm_skip++;
            } else {
                srow[j] = 0;
            }
        }
        small_skips_[i] = count;
        symm_small_skips_[i] = symm_size;
        symm_ignored_columns_[i] = symm_skip;
    }

    // ==> Non-symmetric indexing. <==
    big_skips_[0] = 0;
    size_t coltots = 0;
    for (size_t j = 0; j < nbf_; j++) {
        size_t cols = small_skips_[j];
        size_t size = cols * naux_;
        coltots += cols;
        big_skips_[j + 1] = size + big_skips_[j];
    }
    small_skips_[nbf_] = coltots;

    // ==> Symmetric indexing. <==
    symm_big_skips_[0] = 0;
    for (size_t i = 1; i < nbf_ + 1; i++) {
        symm_big_skips_[i] = symm_big_skips_[i - 1] + symm_small_skips_[i - 1] * naux_;
    }

    // ==> Compact significant column index list per row p (avoids O(nbf^2) mask scans in J and K). <==
    sig_q_row_off_.resize(nbf_ + 1);
    sig_q_row_off_[0] = 0;
    for (size_t i = 0; i < nbf_; ++i) {
        sig_q_row_off_[i + 1] = sig_q_row_off_[i] + small_skips_[i];
    }
    sig_q_.resize(sig_q_row_off_[nbf_]);
#pragma omp parallel for num_threads(nthreads_) schedule(static) if (nbf_ >= 256)
    for (size_t i = 0; i < nbf_; ++i) {
        uint32_t* dst = &sig_q_[sig_q_row_off_[i]];
        const size_t* srow = &schwarz_fun_index_[i * nbf_];
        size_t pos = 0;
        for (size_t j = 0; j < nbf_; ++j) {
            if (srow[j]) dst[pos++] = static_cast<uint32_t>(j);
        }
    }

    sparsity_prepared_ = true;
    timer_off("DFH: sparsity prep");
}

void DFHelper::prepare_AO() {
    // prepare eris
    std::shared_ptr<BasisSet> zero = BasisSet::zero_ao_basis_set();
    auto rifactory = std::make_shared<IntegralFactory>(aux_, zero, primary_, primary_);
    std::vector<std::shared_ptr<TwoBodyAOInt>> eri(nthreads_);
    eri[0] = std::shared_ptr<TwoBodyAOInt>(rifactory->eri(0, false));
    for(int rank = 1; rank < nthreads_; rank++) {
        eri[rank] = std::shared_ptr<TwoBodyAOInt>(eri.front()->clone());
    }

    // gather blocking info
    std::vector<std::pair<size_t, size_t>> psteps;
    std::pair<size_t, size_t> plargest = pshell_blocks_for_AO_build(memory_, 0, psteps);

    // declare largest necessary
    std::unique_ptr<double[]> M(new double[std::get<0>(plargest) / 2]);  // there was a factor of two built in
    std::unique_ptr<double[]> F(new double[std::get<0>(plargest) / 2]);
    std::unique_ptr<double[]> metric;
    double* Mp = M.get();
    double* Fp = F.get();

    // grab metric
    double* metp;
    if (!hold_met_) {
        metric = std::unique_ptr<double[]>(new double[naux_ * naux_]);
        metp = metric.get();
        std::string filename = return_metfile(mpower_);
        get_tensor_(std::get<0>(files_[filename]), metp, 0, naux_ - 1, 0, naux_ - 1);

    } else
        metp = metric_prep_core(mpower_);

    // prepare files
    AO_filename_maker(1);
    AO_filename_maker(2);
    std::string putf = AO_names_[1];
    std::string op = "ab";

    // Contract metric according to previously calculated scheme
    size_t count = 0;
    for (size_t i = 0; i < psteps.size(); i++) {
        // setup
        size_t start = std::get<0>(psteps[i]);
        size_t stop = std::get<1>(psteps[i]);
        size_t begin = pshell_aggs_[start];
        size_t end = pshell_aggs_[stop + 1] - 1;
        size_t block_size = end - begin + 1;
        size_t size = big_skips_[end + 1] - big_skips_[begin];

        // compute
        timer_on("DFH: Total Workflow");
        timer_on("DFH: AO Construction");
        compute_sparse_pQq_blocking_p(start, stop, Mp, eri);
        timer_off("DFH: AO Construction");

        // loop and contract
        timer_on("DFH: AO-Met. Contraction");
#pragma omp parallel for num_threads(nthreads_) schedule(guided)
        for (size_t j = 0; j < block_size; j++) {
            size_t mi = small_skips_[begin + j];
            size_t skips = big_skips_[begin + j] - big_skips_[begin];
            C_DGEMM('N', 'N', naux_, mi, naux_, 1.0, metp, naux_, &Mp[skips], mi, 0.0, &Fp[skips], mi);
        }
        timer_off("DFH: AO-Met. Contraction");
        timer_off("DFH: Total Workflow");

        // put
        put_tensor_AO(putf, Fp, size, count, op);
        count += size;
    }
}
void DFHelper::prepare_AO_wK() {
    // prepare eris
    std::shared_ptr<BasisSet> zero = BasisSet::zero_ao_basis_set();
    auto rifactory = std::make_shared<IntegralFactory>(aux_, zero, primary_, primary_);
    std::vector<std::shared_ptr<TwoBodyAOInt>> eri(nthreads_);
    eri[0] = std::shared_ptr<TwoBodyAOInt>(rifactory->eri(0, false));
#pragma omp parallel for num_threads(nthreads_) schedule(static)
    for (size_t rank = 0; rank < nthreads_; ++rank) {
        if (rank) eri[rank] = std::shared_ptr<TwoBodyAOInt>(eri.front()->clone());
    }

    // gather blocking info
    std::vector<std::pair<size_t, size_t>> psteps;
    std::pair<size_t, size_t> plargest = pshell_blocks_for_AO_build(memory_, 0, psteps);
}
// Grow-only scratch: reallocate (freeing the old buffer first) only when a build needs more.
static double* grow_scratch(DFHelperScratch& buf, size_t& size, size_t need) {
    if (size < need) {
        buf.reset();
        size = 0;  // stays consistent if the allocation below throws
        buf.resize(need);
        size = need;
    }
    return buf.get();
}
size_t DFHelper::metric_panel_cols(bool narrowest) const {
    // Width of the dense (naux x ncol) panels contract_metric_AO_core_symm gathers runs of p into:
    // ~8192 columns (or nbf), but never more than the significant (p, q >= p) columns that exist.
    // A tight budget may cap it (panel_cols_cap_), down to nbf, which any single p fits in;
    // narrowest gives that floor.
    const size_t total = (naux_ && symm_big_skips_.size() == nbf_ + 1) ? symm_big_skips_[nbf_] / naux_ : nbf_;
    size_t cols = narrowest ? nbf_ : std::max<size_t>(8192, nbf_);
    if (!narrowest && panel_cols_cap_) cols = std::min(cols, std::max(panel_cols_cap_, nbf_));
    return std::max<size_t>(1, std::min(cols, total));
}
size_t DFHelper::AO_core_init_scratch(bool narrowest) const {
    // two (naux x panel) panels and the significant-q index lists in contract_metric_AO_core_symm,
    // plus the copy of J the Cholesky route factors when prepare_AO_core holds only one naux^2 metric
    size_t s = 2 * naux_ * metric_panel_cols(narrowest) + small_skips_[nbf_];
    if (metric_cholesky_route() && hold_met_ && !do_wK_) s += naux_ * naux_;
    return s;
}
bool DFHelper::metric_cholesky_route() const {
    if (!metric_cholesky_ || std::fabs(mpower_ + 0.5) >= 1e-13) return false;
    if (!do_wK_) return true;
    return !wcombine_ && std::fabs(wmpower_ + 1.0) < 1e-13;
}
size_t DFHelper::min_AO_core_block() const {
    if (direct_ || direct_iaQ_) return 0;
    size_t max_pshell = 0;
    for (size_t i = 0; i < pshells_; ++i) {
        const size_t begin = pshell_aggs_[i];
        const size_t end = pshell_aggs_[i + 1] - 1;
        max_pshell = std::max(max_pshell, symm_big_skips_[end + 1] - symm_big_skips_[begin]);
    }
    return (do_wK_ && wcombine_) ? 2 * max_pshell : max_pshell;
}
size_t DFHelper::reserve_metric_panels(size_t held) {
    // Under a tight budget give the metric scratch at most half of what the held tensors (AOs,
    // metrics) leave, and always leave at least min_AO_core_block() for the largest single
    // p-shell AO block. Returns the doubles reserved for the metric contraction scratch.
    panel_cols_cap_ = 0;
    const size_t q_min = min_AO_core_block();
    const size_t spare = memory_ > held ? memory_ - held : 0;
    const size_t max_for_panels = std::min(spare / 2, spare > q_min ? spare - q_min : size_t(0));
    if (AO_core_init_scratch() > max_for_panels) {
        const size_t fixed_scratch =
            small_skips_[nbf_] + (metric_cholesky_route() && hold_met_ && !do_wK_ ? naux_ * naux_ : 0);
        const size_t panel_doubles = max_for_panels > fixed_scratch ? max_for_panels - fixed_scratch : 0;
        panel_cols_cap_ = std::max<size_t>(1, panel_doubles / (2 * naux_));
    }
    const size_t s = AO_core_init_scratch();  // with the cap applied (floored at narrowest)
    if (memory_ < held + q_min || s > memory_ - held - q_min) {
        throw PSIEXCEPTION("DFHelper: not enough memory for the metric contraction panels.");
    }
    return s;
}
void DFHelper::prepare_AO_core() {
    const bool use_simint = (Process::environment.options.get_str("INTEGRAL_PACKAGE") == "SIMINT");
    // Integral engines retain a raw factory pointer for spherical transforms.
    // Keep the factory alive through all of the AO construction below.
    std::shared_ptr<IntegralFactory> rifactory;
    std::vector<std::shared_ptr<TwoBodyAOInt>> eri;
    if (use_simint || direct_ || direct_iaQ_) {
        std::shared_ptr<BasisSet> zero = BasisSet::zero_ao_basis_set();
        rifactory = std::make_shared<IntegralFactory>(aux_, zero, primary_, primary_);
        eri.resize(nthreads_);
        eri[0] = std::shared_ptr<TwoBodyAOInt>(rifactory->eri(0, false));
#pragma omp parallel for num_threads(nthreads_) schedule(static)
        for (size_t rank = 0; rank < nthreads_; ++rank) {
            if (rank) eri[rank] = std::shared_ptr<TwoBodyAOInt>(eri.front()->clone());
        }
    }

    // determine blocking, leaving room for the metric contraction's panels (and Cholesky copy)
    // next to Ppq_ and the naux^2 metric; the direct modes neither contract panels nor block Qpq
    std::vector<std::pair<size_t, size_t>> psteps;
    std::pair<size_t, size_t> plargest{0, 0};
    if (!direct_ && !direct_iaQ_) {
        const size_t init_scratch = reserve_metric_panels(big_skips_[nbf_] + (hold_met_ ? 2 : 1) * naux_ * naux_);
        plargest = pshell_blocks_for_AO_build(memory_ - init_scratch, 1, psteps);
    }

    // allocate final AO vector
    if (direct_iaQ_) {
        Ppq_ = std::unique_ptr<double[]>(new double[naux_ * nbf_ * nbf_]);
    } else {
        Ppq_ = std::unique_ptr<double[]>(new double[big_skips_[nbf_]]);
        // Parallel first-touch by p-block so NUMA placement matches J and K first-transform loops.
#pragma omp parallel for schedule(guided) num_threads(nthreads_) if (big_skips_[nbf_] >= 524288)
        for (size_t p = 0; p < nbf_; ++p) {
            const size_t b0 = big_skips_[p];
            const size_t b1 = big_skips_[p + 1];
            for (size_t idx = b0; idx < b1; idx += 512) {
                Ppq_[idx] = 0.0;
            }
        }
    }

    double* ppq = Ppq_.get();

    // outfile->Printf("\n    ==> Begin AO Blocked Construction <==\n\n");
    if (direct_iaQ_ || direct_) {
        timer_on("DFH: AO Construction");
        if (direct_iaQ_) {
            compute_dense_Qpq_blocking_Q(0, Qshells_ - 1, &Ppq_[0], eri);
        } else {
            compute_sparse_pQq_blocking_p(0, pshells_ - 1, &Ppq_[0], eri);
        }
        timer_off("DFH: AO Construction");

    } else {
        // declare sparse buffer
        std::unique_ptr<double[]> Qpq(new double[std::get<0>(plargest)]);
        double* Mp = Qpq.get();
        std::unique_ptr<double[]> metric;
        double* metp;

        // Cholesky route: B = L^-1 (Q|pq) with J = L L^T has the same B^T B as J^-1/2 (Q|pq),
        // and a triangular solve costs half the GEMM. Skips the metric eigendecomposition too.
        bool chol = metric_cholesky_route();
        if (chol) {
            timer_on("DFH: metric Cholesky");
            metric = std::unique_ptr<double[]>(new double[naux_ * naux_]);
            metp = metric.get();
            if (!hold_met_) {
                std::string filename = return_metfile(1.0);
                get_tensor_(std::get<0>(files_[filename]), metp, 0, naux_ - 1, 0, naux_ - 1);
            } else {
                double* Jp = metric_prep_core(1.0);
                std::copy(Jp, Jp + naux_ * naux_, metp);
            }
            // 1-norm of J (symmetric: largest absolute column sum), for the condition estimate
            double anorm = 0.0;
            for (size_t Q = 0; Q < naux_; Q++) {
                double s = 0.0;
                for (size_t P = 0; P < naux_; P++) s += std::fabs(metp[Q * naux_ + P]);
                anorm = std::max(anorm, s);
            }
            // column-major upper == row-major lower L
            int info = C_DPOTRF('U', naux_, metp, naux_);
            double rcond = 0.0;
            if (info == 0) {
                std::vector<double> work(3 * naux_);
                std::vector<int> iwork(naux_);
                C_DPOCON('U', naux_, metp, naux_, anorm, &rcond, work.data(), iwork.data());
            }
            timer_off("DFH: metric Cholesky");
            // Cholesky does not truncate small metric eigenvalues as J^-1/2 does (DF_FITTING_CONDITION);
            // at the conditioning of standard JKFIT metrics that moves energies far less than the DF
            // error. Fall back only when the metric is singular to working precision.
            constexpr double rcond_min = 1.0e-13;
            if (info != 0 || rcond < rcond_min) {
                outfile->Printf("    DFHelper: metric Cholesky not used (info %d, rcond %.2E < %.2E); using J^-1/2.\n",
                                info, rcond, rcond_min);
                chol = false;
            } else if (print_lvl_ > 0) {
                outfile->Printf("    DFHelper: metric Cholesky, rcond %.2E\n", rcond);
            }
        }
        if (chol) {
            // metp holds L
        } else if (!hold_met_) {
            metric = std::unique_ptr<double[]>(new double[naux_ * naux_]);
            metp = metric.get();
            std::string filename = return_metfile(mpower_);
            get_tensor_(std::get<0>(files_[filename]), metp, 0, naux_ - 1, 0, naux_ - 1);
        } else
            metp = metric_prep_core(mpower_);

        for (size_t i = 0; i < psteps.size(); i++) {
            size_t start = std::get<0>(psteps[i]);
            size_t stop = std::get<1>(psteps[i]);
            size_t begin = pshell_aggs_[start];
            size_t end = pshell_aggs_[stop + 1] - 1;

            // compute
            timer_on("DFH: AO Construction");
            compute_sparse_pQq_blocking_p_symm(start, stop, Mp, eri);
            timer_off("DFH: AO Construction");

            // contract metric
            timer_on("DFH: AO-Met. Contraction");
            contract_metric_AO_core_symm(Mp, ppq, metp, begin, end, chol);
            timer_off("DFH: AO-Met. Contraction");
        }
        // no more need for metrics or metric contraction panels
        metric_panel_buf_.reset();
        metric_out_buf_.reset();
        metric_panel_size_ = metric_out_size_ = 0;
        if (hold_met_) metrics_.clear();
    }
    // outfile->Printf("\n    ==> End AO Blocked Construction <==");
}
void DFHelper::prepare_AO_wK_core() {
    const bool use_simint = (Process::environment.options.get_str("INTEGRAL_PACKAGE") == "SIMINT");
    std::shared_ptr<IntegralFactory> rifactory;  // outlive the engines' spherical transforms
    std::vector<std::shared_ptr<TwoBodyAOInt>> eri;
    std::vector<std::shared_ptr<TwoBodyAOInt>> weri;
    if (use_simint) {
        std::shared_ptr<BasisSet> zero = BasisSet::zero_ao_basis_set();
        rifactory = std::make_shared<IntegralFactory>(aux_, zero, primary_, primary_);
        eri.resize(nthreads_);
        eri[0] = std::shared_ptr<TwoBodyAOInt>(rifactory->eri(0, false));
        // Both separate and combined wK builds need the erf engine in the SIMINT fallback.
        weri.resize(nthreads_);
        weri[0] = std::shared_ptr<TwoBodyAOInt>(rifactory->erf_eri(omega_, 0, false));
#pragma omp parallel for num_threads(nthreads_) schedule(static)
        for (size_t rank = 0; rank < nthreads_; ++rank) {
            if (rank) {
                if (!eri.empty()) eri[rank] = std::shared_ptr<TwoBodyAOInt>(eri.front()->clone());
                if (!weri.empty()) weri[rank] = std::shared_ptr<TwoBodyAOInt>(weri.front()->clone());
            }
        }
    }

    // use blocking as for prepare_AO_core, leaving room for the metric contraction's panels
    // next to the held AOs (wPpq, m1Ppq and, unless combined, Ppq) and the two naux^2 metrics
    std::vector<std::pair<size_t, size_t>> psteps;
    const size_t held = (wcombine_ ? 2 : 3) * big_skips_[nbf_] + (hold_met_ ? 3 : 2) * naux_ * naux_;
    const size_t init_scratch = reserve_metric_panels(held);
    std::pair<size_t, size_t> plargest = pshell_blocks_for_AO_build(memory_ - init_scratch, 1, psteps);

    wPpq_ = std::make_unique<double[]>(big_skips_[nbf_]);
    m1Ppq_ = std::make_unique<double[]>(big_skips_[nbf_]);
    if (!wcombine_) Ppq_ = std::make_unique<double[]>(big_skips_[nbf_]);

    // Parallel first-touch by p-block across NUMA nodes so J, K, and wK stream interleaved pages
#pragma omp parallel for schedule(guided) num_threads(nthreads_) if (big_skips_[nbf_] >= 524288)
    for (size_t p = 0; p < nbf_; ++p) {
        const size_t b0 = big_skips_[p];
        const size_t b1 = big_skips_[p + 1];
        for (size_t idx = b0; idx < b1; idx += 512) {
            wPpq_[idx] = 0.0;
            m1Ppq_[idx] = 0.0;
            if (!wcombine_) Ppq_[idx] = 0.0;
        }
    }

    double* wppq = wPpq_.get();
    double* ppq = !wcombine_ ? Ppq_.get() : nullptr;
    double* m1ppq = m1Ppq_.get();

    std::unique_ptr<double[]> Qpq(new double[std::get<0>(plargest)]);
    std::unique_ptr<double[]> Qpq2 = wcombine_ ? std::make_unique<double[]>(std::get<0>(plargest)) : nullptr;
    double* M1p = Qpq.get();
    double* M2p = Qpq2.get();
    std::unique_ptr<double[]> metric1;
    double* met1p = nullptr;
    std::unique_ptr<double[]> metric;
    double* metp = nullptr;

    // Cholesky route for !wcombine_ (J = L L^T): B = L^-1 (Q|mn) for J/K and M1 = L^-T B = J^-1 (Q|mn) for wK,
    // replacing two O(naux^3) eigensolves and two dense DGEMMs with one DPOTRF, one gather, and two DTRSMs.
    bool chol = metric_cholesky_route();
    if (chol) {
        timer_on("DFH: metric Cholesky");
        metric = std::make_unique<double[]>(naux_ * naux_);
        metp = metric.get();
        if (!hold_met_) {
            std::string filename = return_metfile(1.0);
            get_tensor_(std::get<0>(files_[filename]), metp, 0, naux_ - 1, 0, naux_ - 1);
        } else {
            double* Jp = metric_prep_core(1.0);
            std::copy(Jp, Jp + naux_ * naux_, metp);
        }
        double anorm = 0.0;
        for (size_t Q = 0; Q < naux_; Q++) {
            double s = 0.0;
            for (size_t P = 0; P < naux_; P++) s += std::fabs(metp[Q * naux_ + P]);
            anorm = std::max(anorm, s);
        }
        int info = C_DPOTRF('U', naux_, metp, naux_);
        double rcond = 0.0;
        if (info == 0) {
            std::vector<double> work(3 * naux_);
            std::vector<int> iwork(naux_);
            C_DPOCON('U', naux_, metp, naux_, anorm, &rcond, work.data(), iwork.data());
        }
        timer_off("DFH: metric Cholesky");
        constexpr double rcond_min = 1.0e-13;
        if (info != 0 || rcond < rcond_min) {
            outfile->Printf("    DFHelper: metric Cholesky not used (info %d, rcond %.2E < %.2E); using EIGEN.\n",
                            info, rcond, rcond_min);
            metric.reset();
            metp = nullptr;
            chol = false;
        } else {
            if (hold_met_) metrics_.clear();
            if (print_lvl_ > 0) {
                outfile->Printf("    DFHelper: metric Cholesky (wK), rcond %.2E\n", rcond);
            }
        }
    }

    if (!chol) {
        if (!hold_met_) {
            metric1 = std::make_unique<double[]>(naux_ * naux_);
            met1p = metric1.get();
            std::string filename = return_metfile(wmpower_);
            get_tensor_(std::get<0>(files_[filename]), met1p, 0, naux_ - 1, 0, naux_ - 1);
            metric = std::make_unique<double[]>(naux_ * naux_);
            metp = metric.get();
            filename = return_metfile(mpower_);
            get_tensor_(std::get<0>(files_[filename]), metp, 0, naux_ - 1, 0, naux_ - 1);
        } else {
            met1p = metric_prep_core(wmpower_);
            metp = metric_prep_core(mpower_);
        }
    }

    for (size_t i = 0; i < psteps.size(); i++) {
        size_t start = std::get<0>(psteps[i]);
        size_t stop = std::get<1>(psteps[i]);
        size_t begin = pshell_aggs_[start];
        size_t end = pshell_aggs_[stop + 1] - 1;

        if (wcombine_) {
            // computes (Q|mn) and (Q|w|mn)
            timer_on("DFH: AO Construction");
            compute_sparse_pQq_blocking_p_symm_abw(start, stop, M1p, M2p, eri, weri);
            timer_off("DFH: AO Construction");
            // computes  [J^{-1.0}](Q|mn)
            timer_on("DFH: AO-Met. Contraction");
            contract_metric_AO_core_symm(M1p, m1ppq, met1p, begin, end);
            timer_off("DFH: AO-Met. Contraction");

            copy_upper_lower_wAO_core_symm(M2p, wppq, begin, end);
        } else {
            timer_on("DFH: AO Construction");
            compute_sparse_pQq_blocking_p_symm(start, stop, M1p, eri);
            timer_off("DFH: AO Construction");

            timer_on("DFH: AO-Met. Contraction");
            if (chol) {
                contract_metric_AO_wK_cholesky_symm(M1p, ppq, m1ppq, metp, begin, end);
            } else {
                // contract full metric inverse computes  [J^{-1.0}](Q|mn)
                contract_metric_AO_core_symm(M1p, m1ppq, met1p, begin, end);
                // contract half metric inverse computes [J^{-1/2}](Q|mn)
                contract_metric_AO_core_symm(M1p, ppq, metp, begin, end);
            }
            timer_off("DFH: AO-Met. Contraction");

            // compute (A|w|mn) directly via thread-local libint2::Engine(erf_coulomb) when !use_simint
            timer_on("DFH: wAO Construction");
            compute_sparse_pQq_blocking_p_symm(start, stop, M1p, weri, omega_);
            timer_off("DFH: wAO Construction");

            timer_on("DFH: wAO Copy");
            copy_upper_lower_wAO_core_symm(M1p, wppq, begin, end);
            timer_off("DFH: wAO Copy");
        }
    }
    // no more need for metrics or metric contraction panels
    metric_panel_buf_.reset();
    metric_out_buf_.reset();
    metric_panel_size_ = metric_out_size_ = 0;
    if (hold_met_) metrics_.clear();
}
std::pair<size_t, size_t> DFHelper::pshell_blocks_for_AO_build(const size_t mem, size_t symm,
                                                               std::vector<std::pair<size_t, size_t>>& b) {
    size_t full_3index = (symm ? (do_wK_ ? (wcombine_ ? 2 : 3) * big_skips_[nbf_] : big_skips_[nbf_]) : 0);
    size_t symm_metrics = (symm ? ((do_wK_ ? 2 : 1) + (hold_met_ ? 1 : 0)) * naux_ * naux_ : 0);
    size_t constraint, end, begin, current, block_size, tmpbs, total, count, largest;
    block_size = tmpbs = total = count = largest = 0;
    for (size_t i = 0; i < pshells_; i++) {
        count++;
        begin = pshell_aggs_[i];
        end = pshell_aggs_[i + 1] - 1;
        tmpbs += end - begin + 1;

        if (symm) {
            // in-core symmetric: Qpq holds one block (plus Qpq2 when do_wK_ && wcombine_)
            current = symm_big_skips_[end + 1] - symm_big_skips_[begin];
            total += current;
        } else {
            // on-disk
            // get current cost of this block of AOs and add it to the total
            // count current twice, for both pre and post contracted buffers
            current = big_skips_[end + 1] - big_skips_[begin];
            if (do_wK_) {
                current *= 3;
            }
            total += 2 * current;
        }

        if (symm) {
            constraint = full_3index + symm_metrics + ((do_wK_ && wcombine_) ? 2 * total : total);
        } else {
            constraint = total + full_3index + (hold_met_ ? 2 * naux_ * naux_ : total);
        }
        if (constraint > mem || i == pshells_ - 1) {
            if (count == 1 && constraint > mem) {
                std::stringstream error;
                error << "DFHelper: not enough memory for (p shell) AO blocking!"
                      << " required memory: " << constraint * 8 / (1024 * 1024 * 1024.0) << " [GiB].";
                throw PSIEXCEPTION(error.str().c_str());
            }
            if (constraint > mem) {
                total -= current;
                tmpbs -= end - begin + 1;
                b.push_back(std::make_pair(i - count + 1, i - 1));
                i--;
            } else if (i == pshells_ - 1)
                b.push_back(std::make_pair(i - count + 1, i));
            if (largest < total) {
                largest = total;
                block_size = tmpbs;
            }
            count = 0;
            total = 0;
            tmpbs = 0;
        }
    }
    // returns pair(largest buffer size, largest block size)
    return std::make_pair(largest, block_size);
}
std::pair<size_t, size_t> DFHelper::Qshell_blocks_for_transform(const size_t mem, size_t wtmp, size_t wfinal,
                                                                std::vector<std::pair<size_t, size_t>>& b) {
    size_t extra = (hold_met_ ? 2 * naux_ * naux_ : 0);
    size_t end, begin, current, block_size, tmpbs, total, count, largest;
    block_size = tmpbs = total = count = largest = 0;
    for (size_t i = 0; i < Qshells_; i++) {
        count++;
        begin = Qshell_aggs_[i];
        end = Qshell_aggs_[i + 1] - 1;
        tmpbs += end - begin + 1;

        if (direct_iaQ_) {
            // the direct_iaQ method does not use sparse storage
            current = (end - begin + 1) * nbf_ * nbf_;
            total += current;
            total = (AO_core_ ? naux_ * nbf_ * nbf_ : total);
        } else {
            current = (end - begin + 1) * small_skips_[nbf_];
            total += current;
            total = (AO_core_ ? big_skips_[nbf_] : total);
        }

        size_t constraint = total + (wtmp * nbf_ + 2 * wfinal) * tmpbs + extra;
        // AOs + worst half transformed + worst final
        if (constraint > mem || i == Qshells_ - 1) {
            if (count == 1 && constraint > mem) {
                std::stringstream error;
                error << "DFHelper: not enough memory for transformation blocking!";
                throw PSIEXCEPTION(error.str().c_str());
            }
            if (constraint > mem) {
                if (!AO_core_) total -= current;
                tmpbs -= end - begin + 1;
                b.push_back(std::make_pair(i - count + 1, i - 1));
                i--;
            } else if (i == Qshells_ - 1)
                b.push_back(std::make_pair(i - count + 1, i));
            if (block_size < tmpbs) {
                block_size = tmpbs;
                largest = total;
            }
            count = 0;
            total = 0;
            tmpbs = 0;
        }
    }

    return std::make_pair(largest, block_size);
}
std::tuple<size_t, size_t> DFHelper::Qshell_blocks_for_JK_build(std::vector<std::pair<size_t, size_t>>& b,
                                                                size_t max_nocc, bool lr_symmetric) {
    // Charge retained high-water marks, all resident tensors (three for separate wK),
    // and the same scratch floors as the actual allocations. The resident charge
    // applies to every block, not just the first one.
    const size_t floor = std::max(nbf_ * nbf_, nthreads_ * naux_);
    const size_t c_need = nbf_ * std::max(max_nocc, nbf_);
    const size_t core_size = JK_core_memory();
    auto footprint = [&](size_t nq) {
        const size_t tmp = std::max(floor, nbf_ * std::max<size_t>(max_nocc, 1) * nq);
        size_t size = core_size + std::max(jk_T1_size_, tmp) +
                      std::max(jk_T2_size_, lr_symmetric ? floor : tmp);
        for (size_t r = 0; r < nthreads_; ++r) {
            size += r < jk_C_buffers_.size() ? std::max(c_need, jk_C_buffers_[r].capacity()) : c_need;
        }
        if (!AO_core_) size += nq * small_skips_[nbf_];
        return size;
    };
    size_t largest_shell = 0;
    for (size_t i = 0; i < Qshells_; ++i) {
        largest_shell = std::max(largest_shell, Qshell_aggs_[i + 1] - Qshell_aggs_[i]);
    }
    if (footprint(largest_shell) > memory_) {
        // A changed occupation/symmetry or memory allowance can make old scratch
        // too large. No raw scratch pointers have been taken yet, so replan fresh.
        clear_JK_scratch();
        if (footprint(largest_shell) > memory_) {
            throw PSIEXCEPTION("DFHelper: not enough memory for JK blocking!");
        }
    }

    b.clear();
    size_t start = 0, nq = 0, block_size = 0;
    for (size_t i = 0; i < Qshells_; ++i) {
        const size_t shell_size = Qshell_aggs_[i + 1] - Qshell_aggs_[i];
        if (nq && footprint(nq + shell_size) > memory_) {
            b.emplace_back(start, i - 1);
            block_size = std::max(block_size, nq);
            start = i;
            nq = 0;
        }
        nq += shell_size;
    }
    if (nq) {
        b.emplace_back(start, Qshells_ - 1);
        block_size = std::max(block_size, nq);
    }
    // Prefer the selected contraction blocks to the optional mirror. Evict it
    // before growing any scratch, and let prepare_K_fp32 use only genuine spare.
    const size_t held = footprint(block_size);
    if (held > memory_ || K_fp32_memory() > memory_ - held) Ppq_f_.reset();

    return std::make_tuple(AO_core_ ? big_skips_[nbf_] : block_size * small_skips_[nbf_], block_size);
}

FILE* DFHelper::stream_check(std::string filename, std::string op) {
    if (file_streams_.count(filename) == 0) {
        file_streams_[filename] = std::make_shared<Stream>(filename, op);
    }

    return file_streams_[filename]->get_stream(op);
}

DFHelper::StreamStruct::StreamStruct(std::string filename, std::string op, bool activate) {
    op_ = op;
    filename_ = filename;
    if (activate) {
        fp_ = fopen(filename.c_str(), op_.c_str());
        open_ = true;
    }
}

DFHelper::StreamStruct::StreamStruct() {}

DFHelper::StreamStruct::~StreamStruct() {
    fflush(fp_);
    fclose(fp_);
    std::remove(filename_.c_str());
}

FILE* DFHelper::StreamStruct::get_stream(std::string op) {
    if (op.compare(op_)) {
        change_stream(op);
    } else {
        if (!open_) {
            fp_ = fopen(filename_.c_str(), op_.c_str());
            open_ = true;
        }
    }

    return fp_;
}

void DFHelper::StreamStruct::change_stream(std::string op) {
    if (open_) {
        close_stream();
    }
    op_ = op;
    fp_ = fopen(filename_.c_str(), op_.c_str());
}

void DFHelper::StreamStruct::close_stream() {
    fflush(fp_);
    fclose(fp_);
}

void DFHelper::put_tensor(std::string file, double* b, std::pair<size_t, size_t> i0, std::pair<size_t, size_t> i1,
                          std::pair<size_t, size_t> i2, std::string op) {
    // collapse to 2D, assume file has form (i1 | i2 i3)
    size_t A2 = std::get<2>(sizes_[file]);

    size_t sta0 = std::get<0>(i0);
    size_t sto0 = std::get<1>(i0);
    size_t sta1 = std::get<0>(i1);
    size_t sto1 = std::get<1>(i1);
    size_t sta2 = std::get<0>(i2);
    size_t sto2 = std::get<1>(i2);

    size_t a0 = sto0 - sta0 + 1;
    size_t a1 = sto1 - sta1 + 1;
    size_t a2 = sto2 - sta2 + 1;

    // check contiguity (a2)
    if (A2 == a2) {
        put_tensor(file, b, sta0, sto0, a2 * sta1, a2 * (sto1 + 1) - 1, op);
    } else {  // loop (a0, a1)
        for (size_t j = 0; j < a0; j++) {
            for (size_t i = 0; i < a1; i++) {
                put_tensor(file, &b[j * (a1 * a2) + i * a2], sta0 + j, sta0 + j, (i + sta1) * A2 + sta2,
                           (i + sta1) * A2 + sta2 + a2 - 1, op);
            }
        }
    }
}
void DFHelper::put_tensor(std::string file, double* Mp, const size_t start1, const size_t stop1, const size_t start2,
                          const size_t stop2, std::string op) {
    size_t a0 = stop1 - start1 + 1;
    size_t a1 = stop2 - start2 + 1;
    size_t A0 = std::get<0>(sizes_[file]);
    size_t A1 = std::get<1>(sizes_[file]) * std::get<2>(sizes_[file]);
    size_t st = A1 - a1;

    // begin stream
    FILE* fp = stream_check(file, op);

    // adjust position
    fseek(fp, (start1 * A1 + start2) * sizeof(double), SEEK_SET);

    // is everything contiguous?
    if (st == 0) {
        size_t s = fwrite(&Mp[0], sizeof(double), a0 * a1, fp);
        if (!s) {
            std::stringstream error;
            error << "DFHelper:put_tensor: write error";
            throw PSIEXCEPTION(error.str().c_str());
        }
    } else {
        for (size_t i = start1; i < stop1; i++) {
            // write
            size_t s = fwrite(&Mp[i * a1], sizeof(double), a1, fp);
            if (!s) {
                std::stringstream error;
                error << "DFHelper:put_tensor: write error";
                throw PSIEXCEPTION(error.str().c_str());
            }
            // advance stream
            fseek(fp, st * sizeof(double), SEEK_CUR);
        }
        // manual last one
        size_t s = fwrite(&Mp[(a0 - 1) * a1], sizeof(double), a1, fp);
        if (!s) {
            std::stringstream error;
            error << "DFHelper:put_tensor: write error";
            throw PSIEXCEPTION(error.str().c_str());
        }
    }
}
void DFHelper::put_tensor_AO(std::string file, double* Mp, size_t size, size_t start, std::string op) {
    // begin stream
    FILE* fp = stream_check(file, op);

    // adjust position
    fseek(fp, start, SEEK_SET);

    // everything is contiguous
    size_t s = fwrite(&Mp[0], sizeof(double), size, fp);
    if (!s) {
        std::stringstream error;
        error << "DFHelper:put_tensor_AO: write error";
        throw PSIEXCEPTION(error.str().c_str());
    }
}
void DFHelper::get_tensor_AO(std::string file, double* Mp, size_t size, size_t start) {
    // begin stream
    FILE* fp = stream_check(file, "rb");

    // adjust position
    fseek(fp, start * sizeof(double), SEEK_SET);

    // everything is contiguous
    size_t s = fread(&Mp[0], sizeof(double), size, fp);
    if (!s) {
        std::stringstream error;
        error << "DFHelper:get_tensor_AO: read error";
        throw PSIEXCEPTION(error.str().c_str());
    }
}
void DFHelper::get_tensor_(std::string file, double* b, std::pair<size_t, size_t> i0, std::pair<size_t, size_t> i1,
                           std::pair<size_t, size_t> i2) {
    // has this integral been transposed?
    std::tuple<size_t, size_t, size_t> sizes;
    sizes = (tsizes_.find(file) != tsizes_.end() ? tsizes_[file] : sizes_[file]);

    // collapse to 2D, assume file has form (i1 | i2 i3)
    size_t A2 = std::get<2>(sizes);

    size_t sta0 = std::get<0>(i0);
    size_t sto0 = std::get<1>(i0);
    size_t sta1 = std::get<0>(i1);
    size_t sto1 = std::get<1>(i1);
    size_t sta2 = std::get<0>(i2);
    size_t sto2 = std::get<1>(i2);

    size_t a0 = sto0 - sta0 + 1;
    size_t a1 = sto1 - sta1 + 1;
    size_t a2 = sto2 - sta2 + 1;

    // check contiguity (a2)
    if (A2 == a2) {
        get_tensor_(file, b, sta0, sto0, a2 * sta1, a2 * (sto1 + 1) - 1);
    } else {  // loop (a0, a1)
        for (size_t j = 0; j < a0; j++) {
            for (size_t i = 0; i < a1; i++) {
                get_tensor_(file, &b[j * (a1 * a2) + i * a2], sta0 + j, sta0 + j, (i + sta1) * A2 + sta2,
                            (i + sta1) * A2 + sta2 + a2 - 1);
            }
        }
    }
}
void DFHelper::get_tensor_(std::string file, double* b, const size_t start1, const size_t stop1, const size_t start2,
                           const size_t stop2) {
    size_t a0 = stop1 - start1 + 1;
    size_t a1 = stop2 - start2 + 1;

    // has this integral been transposed?
    std::tuple<size_t, size_t, size_t> sizes;
    sizes = (tsizes_.find(file) != tsizes_.end() ? tsizes_[file] : sizes_[file]);

    size_t A0 = std::get<0>(sizes);
    size_t A1 = std::get<1>(sizes) * std::get<2>(sizes);
    size_t st = A1 - a1;

    // check stream
    FILE* fp = stream_check(file, "rb");

    // adjust position
    fseek(fp, (start1 * A1 + start2) * sizeof(double), SEEK_SET);

    // is everything contiguous?
    if (st == 0) {
        size_t s = fread(&b[0], sizeof(double), a0 * a1, fp);
        if (!s) {
            std::stringstream error;
            error << "DFHelper:get_tensor: read error";
            throw PSIEXCEPTION(error.str().c_str());
        }
    } else {
        for (size_t i = 0; i < a0 - 1; i++) {
            // read
            size_t s = fread(&b[i * a1], sizeof(double), a1, fp);
            if (!s) {
                std::stringstream error;
                error << "DFHelper:get_tensor: read error";
                throw PSIEXCEPTION(error.str().c_str());
            }
            // advance stream
            s = fseek(fp, st * sizeof(double), SEEK_CUR);
            if (s) {
                std::stringstream error;
                error << "DFHelper:get_tensor: read error";
                throw PSIEXCEPTION(error.str().c_str());
            }
        }
        // manual last one
        size_t s = fread(&b[(a0 - 1) * a1], sizeof(double), a1, fp);
        if (!s) {
            std::stringstream error;
            error << "DFHelper:get_tensor: read error";
            throw PSIEXCEPTION(error.str().c_str());
        }
    }
}

void DFHelper::compute_dense_Qpq_blocking_Q(const size_t start, const size_t stop, double* Mp,
                                            std::vector<std::shared_ptr<TwoBodyAOInt>> eri) {
    // Here, we compute dense AO integrals in the Qpq memory layout.
    // Sparsity and permutational symmetry are used in the computation,
    // but not in the resulting tensor.

    size_t begin = Qshell_aggs_[start];
    size_t end = Qshell_aggs_[stop + 1] - 1;
    size_t block_size = end - begin + 1;

    // stripe the buffer
    fill(Mp, block_size * nbf_ * nbf_, 0.0);

    // prepare eri buffers
    size_t nthread = nthreads_;
    if (eri.size() != nthreads_) nthread = eri.size();
    std::vector<const double*> buffer(nthread);
#pragma omp parallel num_threads(nthread)
    {
        int rank = 0;
#ifdef _OPENMP
        rank = omp_get_thread_num();
#endif
        buffer[rank] = eri[rank]->buffer();
    }

#pragma omp parallel for schedule(guided) num_threads(nthreads_)
    for (size_t MU = 0; MU < pshells_; MU++) {
        int rank = 0;
#ifdef _OPENMP
        rank = omp_get_thread_num();
#endif
        size_t nummu = primary_->shell(MU).nfunction();
        for (size_t NU = 0; NU < pshells_; NU++) {
            size_t numnu = primary_->shell(NU).nfunction();
            if (!schwarz_shell_mask_[MU * pshells_ + NU]) {
                continue;
            }
            for (size_t Pshell = start; Pshell <= stop; Pshell++) {
                size_t PHI = aux_->shell(Pshell).function_index();
                size_t numP = aux_->shell(Pshell).nfunction();
                eri[rank]->compute_shell(Pshell, 0, MU, NU);
                buffer[rank] = eri[rank]->buffer();
                for (size_t mu = 0; mu < nummu; mu++) {
                    size_t omu = primary_->shell(MU).function_index() + mu;
                    for (size_t nu = 0; nu < numnu; nu++) {
                        size_t onu = primary_->shell(NU).function_index() + nu;
                        if (!schwarz_fun_index_[omu * nbf_ + onu]) {
                            continue;
                        }
                        for (size_t P = 0; P < numP; P++) {
                            Mp[(PHI + P - begin) * nbf_ * nbf_ + omu * nbf_ + onu] =
                                Mp[(PHI + P - begin) * nbf_ * nbf_ + onu * nbf_ + omu] =
                                    buffer[rank][P * nummu * numnu + mu * numnu + nu];
                        }
                    }
                }
            }
        }
    }
}

void DFHelper::compute_sparse_pQq_blocking_Q(const size_t start, const size_t stop, double* Mp,
                                             std::vector<std::shared_ptr<TwoBodyAOInt>> eri) {
    size_t begin = Qshell_aggs_[start];
    size_t end = Qshell_aggs_[stop + 1] - 1;
    size_t block_size = end - begin + 1;

    // prepare eri buffers
    size_t nthread = nthreads_;
    if (eri.size() != nthreads_) nthread = eri.size();

    std::vector<const double*> buffer(nthread);
#pragma omp parallel num_threads(nthread)
    {
        int rank = 0;
#ifdef _OPENMP
        rank = omp_get_thread_num();
#endif
        buffer[rank] = eri[rank]->buffer();
    }

#pragma omp parallel for schedule(guided) num_threads(nthreads_)
    for (size_t MU = 0; MU < pshells_; MU++) {
        int rank = 0;
#ifdef _OPENMP
        rank = omp_get_thread_num();
#endif
        size_t nummu = primary_->shell(MU).nfunction();
        for (size_t NU = 0; NU < pshells_; NU++) {
            size_t numnu = primary_->shell(NU).nfunction();
            if (!schwarz_shell_mask_[MU * pshells_ + NU]) {
                continue;
            }
            for (size_t Pshell = start; Pshell <= stop; Pshell++) {
                size_t PHI = aux_->shell(Pshell).function_index();
                size_t numP = aux_->shell(Pshell).nfunction();
                eri[rank]->compute_shell(Pshell, 0, MU, NU);
                buffer[rank] = eri[rank]->buffer();
                for (size_t mu = 0; mu < nummu; mu++) {
                    size_t omu = primary_->shell(MU).function_index() + mu;
                    for (size_t nu = 0; nu < numnu; nu++) {
                        size_t onu = primary_->shell(NU).function_index() + nu;
                        if (!schwarz_fun_index_[omu * nbf_ + onu]) {
                            continue;
                        }
                        for (size_t P = 0; P < numP; P++) {
                            Mp[(big_skips_[omu] * block_size) / naux_ + (PHI + P - begin) * small_skips_[omu] +
                               schwarz_fun_index_[omu * nbf_ + onu] - 1] =
                                buffer[rank][P * nummu * numnu + mu * numnu + nu];
                        }
                    }
                }
            }
        }
    }
}
void DFHelper::compute_sparse_pQq_blocking_p(const size_t start, const size_t stop, double* Mp,
                                             std::vector<std::shared_ptr<TwoBodyAOInt>> eri) {
    size_t begin = pshell_aggs_[start];
    size_t end = pshell_aggs_[stop + 1] - 1;
    size_t startind = big_skips_[begin];
    //    outfile->Printf("      MU shell: (%zu, %zu)", start, stop);
    //    outfile->Printf(", nbf_ index: (%zu, %zu), size: %zu\n", begin, end, block_size);

    // prepare eri buffers
    size_t nthread = nthreads_;
    if (eri.size() != nthreads_) nthread = eri.size();

    std::vector<const double*> buffer(nthread);
#pragma omp parallel num_threads(nthread)
    {
        int rank = 0;
#ifdef _OPENMP
        rank = omp_get_thread_num();
#endif
        buffer[rank] = eri[rank]->buffer();
    }

#pragma omp parallel for schedule(guided) num_threads(nthread)
    for (size_t MU = start; MU <= stop; MU++) {
        int rank = 0;
#ifdef _OPENMP
        rank = omp_get_thread_num();
#endif
        size_t nummu = primary_->shell(MU).nfunction();
        for (size_t NU = 0; NU < pshells_; NU++) {
            size_t numnu = primary_->shell(NU).nfunction();
            if (!schwarz_shell_mask_[MU * pshells_ + NU]) {
                continue;
            }
            for (size_t Pshell = 0; Pshell < Qshells_; Pshell++) {
                size_t PHI = aux_->shell(Pshell).function_index();
                size_t numP = aux_->shell(Pshell).nfunction();
                eri[rank]->compute_shell(Pshell, 0, MU, NU);
                buffer[rank] = eri[rank]->buffer();
                for (size_t mu = 0; mu < nummu; mu++) {
                    size_t omu = primary_->shell(MU).function_index() + mu;
                    for (size_t nu = 0; nu < numnu; nu++) {
                        size_t onu = primary_->shell(NU).function_index() + nu;
                        if (!schwarz_fun_index_[omu * nbf_ + onu]) {
                            continue;
                        }
                        for (size_t P = 0; P < numP; P++) {
                            Mp[big_skips_[omu] - startind + (PHI + P) * small_skips_[omu] +
                               schwarz_fun_index_[omu * nbf_ + onu] - 1] =
                                buffer[rank][P * nummu * numnu + mu * numnu + nu];
                        }
                    }
                }
            }
        }
    }
}
void DFHelper::compute_sparse_pQq_blocking_p_symm(const size_t start, const size_t stop, double* Mp,
                                                  std::vector<std::shared_ptr<TwoBodyAOInt>> eri,
                                                  std::optional<double> omega) {
    size_t begin = pshell_aggs_[start];
    size_t startind = symm_big_skips_[begin];

    struct PairSlot {
        size_t mn_idx;
        size_t base;
        size_t stride;
    };

    if (eri.empty()) {
        // No omega means Coulomb; an explicitly supplied zero still means erf-Coulomb.
        const bool is_erf = omega.has_value();
        const int max_am = std::max(aux_->max_am(), primary_->max_am());
        const int max_nprim = std::max(aux_->max_nprimitive(), primary_->max_nprimitive());
        const double ln_prec = -std::numeric_limits<double>::max();
        const auto& unit_sh = libint2::Shell::unit();

        std::vector<libint2::ShellPair> aux_sp(Qshells_);
#pragma omp parallel for schedule(static) num_threads(nthreads_) if (Qshells_ >= 64)
        for (size_t Pshell = 0; Pshell < Qshells_; ++Pshell) {
            aux_sp[Pshell].init(aux_->l2_shell(Pshell), unit_sh, ln_prec);
        }

        std::vector<libint2::Engine> engs;
        engs.reserve(nthreads_);
        for (size_t t = 0; t < nthreads_; ++t) {
            if (is_erf) {
                engs.emplace_back(libint2::Operator::erf_coulomb, max_nprim, max_am, 0, 0.0, *omega,
                                  libint2::BraKet::xs_xx);
            } else {
                engs.emplace_back(libint2::Operator::coulomb, max_nprim, max_am, 0, 0.0,
                                  libint2::operator_traits<libint2::Operator::coulomb>::default_params(),
                                  libint2::BraKet::xs_xx);
            }
        }

#pragma omp parallel num_threads(nthreads_)
        {
            int rank = 0;
#ifdef _OPENMP
            rank = omp_get_thread_num();
#endif
            auto& eng = engs[rank];
            libint2::ShellPair ket_sp;
            std::vector<PairSlot> slots;
            slots.reserve(256);

#pragma omp for schedule(guided)
            for (size_t MU = start; MU <= stop; MU++) {
                const auto& sh_mu = primary_->l2_shell(MU);
                size_t nummu = primary_->shell(MU).nfunction();
                size_t bmu = primary_->shell(MU).function_index();
                for (size_t NU = MU; NU < pshells_; NU++) {
                    if (!schwarz_shell_mask_[MU * pshells_ + NU]) continue;
                    const auto& sh_nu = primary_->l2_shell(NU);
                    size_t numnu = primary_->shell(NU).nfunction();
                    size_t bnu = primary_->shell(NU).function_index();

                    slots.clear();
                    for (size_t mu = 0; mu < nummu; mu++) {
                        size_t omu = bmu + mu;
                        const size_t upper_start = 1 + symm_ignored_columns_[omu];
                        size_t row_base = symm_big_skips_[omu] - startind;
                        size_t row_stride = symm_small_skips_[omu];
                        for (size_t nu = 0; nu < numnu; nu++) {
                            size_t onu = bnu + nu;
                            size_t fidx = schwarz_fun_index_[omu * nbf_ + onu];
                            if (!fidx || omu > onu) continue;
                            slots.push_back({mu * numnu + nu, row_base + (fidx - upper_start), row_stride});
                        }
                    }
                    if (slots.empty()) continue;

                    ket_sp.init(sh_mu, sh_nu, ln_prec);
                    const size_t nmn = nummu * numnu;
                    const size_t nslots = slots.size();

                    for (size_t Pshell = 0; Pshell < Qshells_; Pshell++) {
                        const auto& sh_P = aux_->l2_shell(Pshell);
                        size_t PHI = aux_->shell(Pshell).function_index();
                        size_t numP = aux_->shell(Pshell).nfunction();
                        if (is_erf) {
                            eng.compute2<libint2::Operator::erf_coulomb, libint2::BraKet::xs_xx, 0>(
                                sh_P, unit_sh, sh_mu, sh_nu, &aux_sp[Pshell], &ket_sp);
                        } else {
                            eng.compute2<libint2::Operator::coulomb, libint2::BraKet::xs_xx, 0>(
                                sh_P, unit_sh, sh_mu, sh_nu, &aux_sp[Pshell], &ket_sp);
                        }
                        const double* buf = eng.results()[0];
                        if (buf != nullptr) {
                            for (size_t s = 0; s < nslots; ++s) {
                                const size_t mn = slots[s].mn_idx;
                                const size_t stride = slots[s].stride;
                                size_t dst = slots[s].base + PHI * stride;
                                for (size_t P = 0; P < numP; P++, dst += stride) {
                                    Mp[dst] = buf[P * nmn + mn];
                                }
                            }
                        } else {
                            for (size_t s = 0; s < nslots; ++s) {
                                const size_t stride = slots[s].stride;
                                size_t dst = slots[s].base + PHI * stride;
                                for (size_t P = 0; P < numP; P++, dst += stride) {
                                    Mp[dst] = 0.0;
                                }
                            }
                        }
                    }
                }
            }
        }
        return;
    }

    size_t nthread = (eri.size() != nthreads_) ? eri.size() : nthreads_;
#pragma omp parallel num_threads(nthread)
    {
        int rank = 0;
#ifdef _OPENMP
        rank = omp_get_thread_num();
#endif
        std::vector<PairSlot> slots;
        slots.reserve(256);

#pragma omp for schedule(guided)
        for (size_t MU = start; MU <= stop; MU++) {
            size_t nummu = primary_->shell(MU).nfunction();
            size_t bmu = primary_->shell(MU).function_index();
            for (size_t NU = MU; NU < pshells_; NU++) {
                if (!schwarz_shell_mask_[MU * pshells_ + NU]) continue;
                size_t numnu = primary_->shell(NU).nfunction();
                size_t bnu = primary_->shell(NU).function_index();

                slots.clear();
                for (size_t mu = 0; mu < nummu; mu++) {
                    size_t omu = bmu + mu;
                    const size_t upper_start = 1 + symm_ignored_columns_[omu];
                    size_t row_base = symm_big_skips_[omu] - startind;
                    size_t row_stride = symm_small_skips_[omu];
                    for (size_t nu = 0; nu < numnu; nu++) {
                        size_t onu = bnu + nu;
                        size_t fidx = schwarz_fun_index_[omu * nbf_ + onu];
                        if (!fidx || omu > onu) continue;
                        slots.push_back({mu * numnu + nu, row_base + (fidx - upper_start), row_stride});
                    }
                }
                if (slots.empty()) continue;

                const size_t nmn = nummu * numnu;
                const size_t nslots = slots.size();
                for (size_t Pshell = 0; Pshell < Qshells_; Pshell++) {
                    size_t PHI = aux_->shell(Pshell).function_index();
                    size_t numP = aux_->shell(Pshell).nfunction();
                    eri[rank]->compute_shell(Pshell, 0, MU, NU);
                    const double* buf = eri[rank]->buffer();
                    for (size_t s = 0; s < nslots; ++s) {
                        const size_t mn = slots[s].mn_idx;
                        const size_t stride = slots[s].stride;
                        size_t dst = slots[s].base + PHI * stride;
                        for (size_t P = 0; P < numP; P++, dst += stride) {
                            Mp[dst] = buf[P * nmn + mn];
                        }
                    }
                }
            }
        }
    }
}

void DFHelper::compute_sparse_pQq_blocking_p_symm_abw(const size_t start, const size_t stop, double* just_Mp, double* param_Mp,
                                                  std::vector<std::shared_ptr<TwoBodyAOInt>> eri,
                                                  std::vector<std::shared_ptr<TwoBodyAOInt>> weri
) {
    size_t begin = pshell_aggs_[start];
    size_t startind = symm_big_skips_[begin];

    struct PairSlot {
        size_t mn_idx;
        size_t base;
        size_t stride;
    };

    if (eri.empty()) {
        const int max_am = std::max(aux_->max_am(), primary_->max_am());
        const int max_nprim = std::max(aux_->max_nprimitive(), primary_->max_nprimitive());
        const double ln_prec = -std::numeric_limits<double>::max();
        const auto& unit_sh = libint2::Shell::unit();

        std::vector<libint2::ShellPair> aux_sp(Qshells_);
#pragma omp parallel for schedule(static) num_threads(nthreads_) if (Qshells_ >= 64)
        for (size_t Pshell = 0; Pshell < Qshells_; ++Pshell) {
            aux_sp[Pshell].init(aux_->l2_shell(Pshell), unit_sh, ln_prec);
        }

        std::vector<libint2::Engine> c_engs, w_engs;
        c_engs.reserve(nthreads_);
        w_engs.reserve(nthreads_);
        for (size_t t = 0; t < nthreads_; ++t) {
            c_engs.emplace_back(libint2::Operator::coulomb, max_nprim, max_am, 0, 0.0,
                                libint2::operator_traits<libint2::Operator::coulomb>::default_params(),
                                libint2::BraKet::xs_xx);
            w_engs.emplace_back(libint2::Operator::erf_coulomb, max_nprim, max_am, 0, 0.0, omega_,
                                libint2::BraKet::xs_xx);
        }

#pragma omp parallel num_threads(nthreads_)
        {
            int rank = 0;
#ifdef _OPENMP
            rank = omp_get_thread_num();
#endif
            auto& c_eng = c_engs[rank];
            auto& w_eng = w_engs[rank];
            libint2::ShellPair ket_sp;
            std::vector<PairSlot> slots;
            slots.reserve(256);

#pragma omp for schedule(guided)
            for (size_t MU = start; MU <= stop; MU++) {
                const auto& sh_mu = primary_->l2_shell(MU);
                size_t nummu = primary_->shell(MU).nfunction();
                size_t bmu = primary_->shell(MU).function_index();
                for (size_t NU = MU; NU < pshells_; NU++) {
                    if (!schwarz_shell_mask_[MU * pshells_ + NU]) continue;
                    const auto& sh_nu = primary_->l2_shell(NU);
                    size_t numnu = primary_->shell(NU).nfunction();
                    size_t bnu = primary_->shell(NU).function_index();

                    slots.clear();
                    for (size_t mu = 0; mu < nummu; mu++) {
                        size_t omu = bmu + mu;
                        const size_t upper_start = 1 + symm_ignored_columns_[omu];
                        size_t row_base = symm_big_skips_[omu] - startind;
                        size_t row_stride = symm_small_skips_[omu];
                        for (size_t nu = 0; nu < numnu; nu++) {
                            size_t onu = bnu + nu;
                            size_t fidx = schwarz_fun_index_[omu * nbf_ + onu];
                            if (!fidx || omu > onu) continue;
                            slots.push_back({mu * numnu + nu, row_base + (fidx - upper_start), row_stride});
                        }
                    }
                    if (slots.empty()) continue;

                    ket_sp.init(sh_mu, sh_nu, ln_prec);
                    const size_t nmn = nummu * numnu;
                    const size_t nslots = slots.size();

                    for (size_t Pshell = 0; Pshell < Qshells_; Pshell++) {
                        const auto& sh_P = aux_->l2_shell(Pshell);
                        size_t PHI = aux_->shell(Pshell).function_index();
                        size_t numP = aux_->shell(Pshell).nfunction();
                        c_eng.compute2<libint2::Operator::coulomb, libint2::BraKet::xs_xx, 0>(
                            sh_P, unit_sh, sh_mu, sh_nu, &aux_sp[Pshell], &ket_sp);
                        w_eng.compute2<libint2::Operator::erf_coulomb, libint2::BraKet::xs_xx, 0>(
                            sh_P, unit_sh, sh_mu, sh_nu, &aux_sp[Pshell], &ket_sp);
                        const double* cbuf = c_eng.results()[0];
                        const double* wbuf = w_eng.results()[0];
                        for (size_t s = 0; s < nslots; ++s) {
                            const size_t mn = slots[s].mn_idx;
                            const size_t stride = slots[s].stride;
                            size_t dst = slots[s].base + PHI * stride;
                            for (size_t P = 0; P < numP; P++, dst += stride) {
                                const double cv = (cbuf != nullptr) ? cbuf[P * nmn + mn] : 0.0;
                                const double wv = (wbuf != nullptr) ? wbuf[P * nmn + mn] : 0.0;
                                just_Mp[dst] = cv;
                                param_Mp[dst] = omega_alpha_ * cv + omega_beta_ * wv;
                            }
                        }
                    }
                }
            }
        }
        return;
    }

    size_t nthread = (eri.size() != nthreads_) ? eri.size() : nthreads_;
#pragma omp parallel for schedule(guided) num_threads(nthread)
    for (size_t MU = start; MU <= stop; MU++) {
        int rank = 0;
#ifdef _OPENMP
        rank = omp_get_thread_num();
#endif
        size_t nummu = primary_->shell(MU).nfunction();
        for (size_t NU = MU; NU < pshells_; NU++) {
            size_t numnu = primary_->shell(NU).nfunction();
            if (!schwarz_shell_mask_[MU * pshells_ + NU]) {
                continue;
            }
            for (size_t Pshell = 0; Pshell < Qshells_; Pshell++) {
                size_t PHI = aux_->shell(Pshell).function_index();
                size_t numP = aux_->shell(Pshell).nfunction();
                eri[rank]->compute_shell(Pshell, 0, MU, NU);
                weri[rank]->compute_shell(Pshell, 0, MU, NU);
                const double* buf = eri[rank]->buffers()[0];
                const double* wbuf = weri[rank]->buffers()[0];
                for (size_t mu = 0; mu < nummu; mu++) {
                    size_t omu = primary_->shell(MU).function_index() + mu;
                    for (size_t nu = 0; nu < numnu; nu++) {
                        size_t onu = primary_->shell(NU).function_index() + nu;
                        if (!schwarz_fun_index_[omu * nbf_ + onu] || omu > onu) continue;
                        for (size_t P = 0; P < numP; P++) {
                            size_t jump = schwarz_fun_index_[omu * nbf_ + onu] - 1 - symm_ignored_columns_[omu];
                            size_t ind1 = symm_big_skips_[omu] - startind + (PHI + P) * symm_small_skips_[omu] + jump;
                            just_Mp[ind1] = buf[P * nummu * numnu + mu * numnu + nu];
                            param_Mp[ind1] = omega_alpha_ * buf[P * nummu * numnu + mu * numnu + nu] +
                                             omega_beta_ * wbuf[P * nummu * numnu + mu * numnu + nu];
                        }
                    }
                }
            }
        }
    }
}
void DFHelper::grab_AO(const size_t start, const size_t stop, double* Mp) {
    size_t begin = Qshell_aggs_[start];
    size_t end = Qshell_aggs_[stop + 1] - 1;
    size_t block_size = end - begin + 1;
    auto getf = AO_names_[1];

    // presumably not thread safe or inherently sequential, but could revisit
    for (size_t i = 0, sta = 0; i < nbf_; i++) {
        size_t size = block_size * small_skips_[i];
        size_t jump = begin * small_skips_[i];
        get_tensor_AO(getf, &Mp[sta], size, big_skips_[i] + jump);
        sta += size;
    }
}
void DFHelper::prepare_metric_core() {
    timer_on("DFH: metric construction");
    FittingMetric J(aux_, true);
    J.form_fitting_metric();
    metrics_[1.0] = J.get_metric();
    timer_off("DFH: metric construction");
}
double* DFHelper::metric_prep_core(double m_pow) {
    bool on = false;
    double power;
    for (auto& kv : metrics_) {
        if (!(std::fabs(m_pow - kv.first) > 1e-13)) {
            on = true;
            power = kv.first;
            break;
        }
    }
    if (!on) {
        power = m_pow;
        SharedMatrix J = metrics_.at(1.0);
        if ( fabs(m_pow - 1.0) < 1e-13 ) {
            return J->pointer()[0];
        } else {
            J = J->clone();
            J->power(power, condition_);
        }
        metrics_[power] = J;
    }
    return metrics_[power]->pointer()[0];
}
void DFHelper::prepare_metric() {
    // construct metric
    FittingMetric J(aux_, true);
    J.form_fitting_metric();
    auto metric = J.get_metric();
    auto Mp = metric->pointer()[0];

    // create file
    std::string filename = "metric.1.0";
    filename_maker(filename, naux_, naux_, 1);
    metric_keys_.emplace_back(1.0, filename);
    // store
    std::string putf = std::get<0>(files_[filename]);
    put_tensor(putf, Mp, 0, naux_ - 1, 0, naux_ - 1, "wb");
}
std::string DFHelper::return_metfile(double m_pow) {
    bool on = 0;
    std::string key;
    for (size_t i = 0; i < metric_keys_.size() && !on; i++) {
        double power = std::get<0>(metric_keys_[i]);
        if (std::fabs(power - m_pow) < 1e-12) {
            key = std::get<1>(metric_keys_[i]);
            on = 1;
        }
    }

    if (!on) key = compute_metric(m_pow);
    return key;
}
std::string DFHelper::compute_metric(double m_pow) {
    // ensure J
    if (std::fabs(m_pow - 1.0) < 1e-13)
        prepare_metric();
    else {
        // get metric
        auto metric = std::make_shared<Matrix>("met", naux_, naux_);
        double* metp = metric->pointer()[0];
        std::string filename = return_metfile(1.0);

        // get and compute
        get_tensor_(std::get<0>(files_[filename]), metp, 0, naux_ - 1, 0, naux_ - 1);
        metric->power(m_pow, condition_);

        // make new file
        std::string name = "metric";
        name.append(".");
        name.append(std::to_string(m_pow));
        filename_maker(name, naux_, naux_, 1);
        metric_keys_.push_back(std::make_pair(m_pow, name));

        // store
        std::string putf = std::get<0>(files_[name]);
        put_tensor(putf, metp, 0, naux_ - 1, 0, naux_ - 1, "wb");
    }
    return return_metfile(m_pow);
}
double* DFHelper::metric_inverse_prep_core() {
    bool on = false;
    double power;
    for (auto& kv : metrics_) {
        if ((std::fabs(-1.0 - kv.first) < 1e-13)) {
            on = true;
            power = kv.first;
            break;
        }
    }
    if (!on) {
        power = -1.0;
        timer_on("DFH: metric power");
        SharedMatrix J = metrics_.at(1.0)->clone();
        J->invert();
        metrics_[power] = J;
        timer_off("DFH: metric power");
    }

    return metrics_[power]->pointer()[0];
}
void DFHelper::metric_contraction_blocking(std::vector<std::pair<size_t, size_t>>& steps, size_t blocking_index,
                                           size_t block_sizes, size_t total_mem, size_t memory_factor,
                                           size_t memory_bump) {
    for (size_t i = 0, count = 1; i < blocking_index; i++, count++) {
        if (total_mem < count * block_sizes || i == blocking_index - 1) {
            if (count == 1 && i != blocking_index - 1) {
                std::stringstream error;
                error << "DFHelper:contract_metric: not enough memory, ";
                error << "needs at least "
                      << ((count * block_sizes) * memory_factor + memory_bump) / (1024 * 1024 * 1024.0) * 8. << "[GiB]";
                throw PSIEXCEPTION(error.str().c_str());
            }
            if (total_mem < count * block_sizes) {
                steps.push_back(std::make_pair(i - count + 1, i - 1));
                i--;
            } else {
                steps.push_back(std::make_pair(i - count + 1, i));
            }
            count = 0;
        }
    }
}

void DFHelper::contract_metric_Qpq(std::string file, double* metp, double* Mp, double* Fp, const size_t total_mem) {
    std::string getf = std::get<0>(files_[file]);
    std::string putf = std::get<1>(files_[file]);

    size_t Q = std::get<0>(sizes_[getf]);
    size_t l = std::get<1>(sizes_[getf]);
    size_t r = std::get<2>(sizes_[getf]);

    std::string op = "wb";
    std::vector<std::pair<size_t, size_t>> steps;
    metric_contraction_blocking(steps, l, Q * r, total_mem, 2, naux_ * naux_);

    for (size_t i = 0; i < steps.size(); i++) {
        size_t begin = std::get<0>(steps[i]);
        size_t end = std::get<1>(steps[i]);
        size_t bs = end - begin + 1;

        get_tensor_(getf, Mp, 0, Q - 1, begin * r, (end + 1) * r - 1);
        timer_on("DFH: Total Workflow");
        C_DGEMM('T', 'N', bs * r, Q, Q, 1.0, Mp, bs * r, metp, Q, 0.0, Fp, Q);
        timer_off("DFH: Total Workflow");
        put_tensor(putf, Fp, begin, end, 0, r * Q - 1, op);
    }
}

void DFHelper::contract_metric(std::string file, double* metp, double* Mp, double* Fp, const size_t total_mem) {
    std::string getf = std::get<0>(files_[file]);
    std::string putf = std::get<1>(files_[file]);
    size_t a0 = std::get<0>(sizes_[getf]);
    size_t a1 = std::get<1>(sizes_[getf]);
    size_t a2 = std::get<2>(sizes_[getf]);

    std::string op = "wb";
    std::vector<std::pair<size_t, size_t>> steps;

    // contract in steps
    if (std::get<2>(transf_[file])) {
        // determine blocking
        // both pqQ and pQq formats block through p, which is index 0
        metric_contraction_blocking(steps, a0, a1 * a2, total_mem, 2, naux_ * naux_);

        // grab val, the inner contractions are different depending on the form
        size_t val = std::get<2>(transf_[file]);
        for (size_t i = 0; i < steps.size(); i++) {
            size_t begin = std::get<0>(steps[i]);
            size_t end = std::get<1>(steps[i]);
            size_t bs = end - begin + 1;

            get_tensor_(getf, Mp, begin, end, 0, a1 * a2 - 1);
            timer_on("DFH: Total Workflow");

            if (val == 2) {
                C_DGEMM('N', 'N', bs * a1, a2, a2, 1.0, Mp, a2, metp, a2, 0.0, Fp, a2);
            } else {
#pragma omp parallel for num_threads(nthreads_)
                for (size_t i = 0; i < bs; i++) {
                    C_DGEMM('N', 'N', a1, a2, a1, 1.0, metp, a1, &Mp[i * a1 * a2], a2, 0.0, &Fp[i * a1 * a2], a2);
                }
            }
            timer_off("DFH: Total Workflow");
            put_tensor(putf, Fp, begin, end, 0, a1 * a2 - 1, op);
        }

    } else {
        // determine blocking
        // the Qpq format blocks through p, which is index 1
        metric_contraction_blocking(steps, a1, a0 * a2, total_mem, 2, naux_ * naux_);

        for (size_t i = 0; i < steps.size(); i++) {
            size_t begin = std::get<0>(steps[i]);
            size_t end = std::get<1>(steps[i]);
            size_t bs = end - begin + 1;

            get_tensor_(getf, Mp, 0, a0 - 1, begin * a2, (end + 1) * a2 - 1);
            timer_on("DFH: Total Workflow");
            C_DGEMM('N', 'N', a0, bs * a2, a0, 1.0, metp, a0, Mp, bs * a2, 0.0, Fp, bs * a2);
            timer_off("DFH: Total Workflow");
            put_tensor(putf, Fp, 0, a0 - 1, begin * a2, (end + 1) * a2 - 1, op);
        }
    }
}

void DFHelper::contract_metric_AO_core(double* Qpq, double* metp) {
// loop and contract
#pragma omp parallel for num_threads(nthreads_) schedule(guided)
    for (size_t j = 0; j < nbf_; j++) {
        size_t mi = small_skips_[j];
        size_t skips = big_skips_[j];
        C_DGEMM('N', 'N', naux_, mi, naux_, 1.0, metp, naux_, &Qpq[skips], mi, 0.0, &Ppq_[skips], mi);
    }
}

void DFHelper::contract_metric_AO_core_symm(double* Qpq, double* Ppq, double* metp, size_t begin, size_t end,
                                            bool cholesky) {
    // Qpq holds, for each p in [begin, end], a (naux x mi) block of the significant q >= p.
    // Gather runs of p into a dense (naux x ncol) panel, contract it with a single threaded
    // DTRSM/DGEMM, and scatter the result row-contiguously into both upper (p, q) and lower (q, p)
    // blocks of Ppq so each thread writes exclusively to its own p/q block.
    size_t startind = symm_big_skips_[begin];
    const size_t target_cols = metric_panel_cols();

    std::vector<size_t> col_off;
    std::vector<uint32_t> q_cnt(nbf_, 0);
    std::vector<uint32_t> q_off(nbf_ + 1, 0);
    std::vector<uint32_t> q_cstart(nbf_, 0);
    std::vector<uint32_t> q_fill(nbf_, 0);
    std::vector<uint32_t> active_q;
    active_q.reserve(nbf_);
    std::vector<uint32_t> lower_src;
    lower_src.reserve(target_cols);

    // Precompute flat upper-q list for p in [begin, end] once per psteps chunk
    size_t total_upper = 0;
    for (size_t p = begin; p <= end; ++p) total_upper += symm_small_skips_[p];
    std::vector<uint32_t> upper_q(total_upper);
    std::vector<size_t> upper_p_off(end - begin + 2, 0);
    for (size_t p = begin; p <= end; ++p) {
        upper_p_off[p - begin + 1] = upper_p_off[p - begin] + symm_small_skips_[p];
    }
#pragma omp parallel for num_threads(nthreads_) schedule(guided) if (end - begin >= 64)
    for (size_t p = begin; p <= end; ++p) {
        uint32_t* dst = &upper_q[upper_p_off[p - begin]];
        size_t pos = 0;
        const size_t* srow = &schwarz_fun_index_[p * nbf_];
        for (size_t q = p; q < nbf_; ++q) {
            if (srow[q]) dst[pos++] = static_cast<uint32_t>(q);
        }
    }

    for (size_t p0 = begin; p0 <= end;) {
        size_t ncol = 0, p1 = p0;
        col_off.clear();
        while (p1 <= end && (ncol == 0 || ncol + symm_small_skips_[p1] <= target_cols)) {
            col_off.push_back(ncol);
            ncol += symm_small_skips_[p1];
            p1++;
        }
        if (ncol == 0) {
            p0 = p1;
            continue;
        }
        const size_t need = naux_ * ncol;
        double* panel_ptr = grow_scratch(metric_panel_buf_, metric_panel_size_, need);
        double* out_ptr = cholesky ? panel_ptr : grow_scratch(metric_out_buf_, metric_out_size_, need);

        timer_on("DFH: AO-Met. gather");
#pragma omp parallel for num_threads(nthreads_) schedule(static)
        for (size_t Q = 0; Q < naux_; Q++) {
            double* prow = &panel_ptr[Q * ncol];
            for (size_t p = p0; p < p1; p++) {
                size_t mi = symm_small_skips_[p];
                const double* src = &Qpq[symm_big_skips_[p] - startind + Q * mi];
                std::copy(src, src + mi, prow + col_off[p - p0]);
            }
        }
        timer_off("DFH: AO-Met. gather");

        timer_on("DFH: AO-Met. GEMM");
        if (cholesky) {
            // panel <- L^-1 panel in-place (row-major lower L)
            C_DTRSM('L', 'L', 'N', 'N', naux_, ncol, 1.0, metp, naux_, panel_ptr, ncol);
        } else {
            C_DGEMM('N', 'N', naux_, ncol, naux_, 1.0, metp, naux_, panel_ptr, ncol, 0.0, out_ptr, ncol);
        }
        timer_off("DFH: AO-Met. GEMM");

        timer_on("DFH: AO-Met. scatter");
        // Build compact CSR of panel source columns per destination q > p (strictly contiguous columns in q's row).
        active_q.clear();
        size_t total_lower = 0;
        for (size_t p = p0; p < p1; ++p) {
            const uint32_t* qs = &upper_q[upper_p_off[p - begin]];
            const size_t mi = symm_small_skips_[p];
            for (size_t k = 0; k < mi; ++k) {
                const uint32_t q = qs[k];
                if (q == p) continue;
                if (q_cnt[q] == 0) {
                    active_q.push_back(q);
                    q_cstart[q] = static_cast<uint32_t>(schwarz_fun_index_[(size_t)q * nbf_ + p] - 1);
                }
                q_cnt[q]++;
                total_lower++;
            }
        }
        lower_src.resize(total_lower);
        size_t running = 0;
        for (uint32_t q : active_q) {
            q_off[q] = static_cast<uint32_t>(running);
            q_fill[q] = static_cast<uint32_t>(running);
            running += q_cnt[q];
        }
        for (size_t p = p0; p < p1; ++p) {
            const uint32_t* qs = &upper_q[upper_p_off[p - begin]];
            const size_t mi = symm_small_skips_[p];
            const uint32_t base_col = static_cast<uint32_t>(col_off[p - p0]);
            for (size_t k = 0; k < mi; ++k) {
                const uint32_t q = qs[k];
                if (q == p) continue;
                lower_src[q_fill[q]++] = base_col + static_cast<uint32_t>(k);
            }
        }

        const size_t np_chunk = p1 - p0;
        const size_t nq_active = active_q.size();
#pragma omp parallel num_threads(nthreads_)
        {
            // Note: upper writes columns [symm_ignored_columns_[p], small_skips_[p]) for p in [p0, p1),
            // while lower writes columns [c_start, c_start + cnt) < symm_ignored_columns_[q] for q in active_q.
            // Even when q in [p0, p1), upper and lower column ranges are disjoint; running with a barrier or
            // separate non-overlapping cache lines is completely race-free. Run sequentially with barrier to avoid
            // any cache-line sharing at the symm_ignored_columns_[q] boundary.
#pragma omp for schedule(dynamic, 2)
            for (size_t idx = 0; idx < np_chunk; ++idx) {
                const size_t p = p0 + idx;
                const size_t mi = symm_small_skips_[p];
                const size_t si = small_skips_[p];
                const size_t ign = symm_ignored_columns_[p];
                const size_t coff = col_off[idx];
                double* dst_base = &Ppq[big_skips_[p] + ign];
                for (size_t Q = 0; Q < naux_; ++Q) {
                    const double* src = &out_ptr[Q * ncol + coff];
                    std::copy(src, src + mi, dst_base + Q * si);
                }
            }
#pragma omp for schedule(dynamic, 8)
            for (size_t qi = 0; qi < nq_active; ++qi) {
                const uint32_t q = active_q[qi];
                const uint32_t cnt = q_cnt[q];
                const uint32_t cstart = q_cstart[q];
                const uint32_t* src_cols = &lower_src[q_off[q]];
                const size_t sq = small_skips_[q];
                double* dst_base = &Ppq[big_skips_[q] + cstart];
                for (size_t Q = 0; Q < naux_; ++Q) {
                    const double* r = &out_ptr[Q * ncol];
                    double* dst = dst_base + Q * sq;
#pragma omp simd
                    for (uint32_t c = 0; c < cnt; ++c) {
                        dst[c] = r[src_cols[c]];
                    }
                }
                q_cnt[q] = 0;
            }
        }
        timer_off("DFH: AO-Met. scatter");
        p0 = p1;
    }
}
void DFHelper::contract_metric_AO_wK_cholesky_symm(double* Qpq, double* Ppq, double* m1Ppq, double* metp, size_t begin,
                                                   size_t end) {
    // Fused Cholesky dual-metric contraction for !wcombine_ range-separated builds:
    //   1) Gather sparse Qpq panel once
    //   2) Solve L * B = Qpq in-place -> scatter B into Ppq (for J and short-range K)
    //   3) Solve L^T * M1 = B in-place -> scatter M1 = (L L^T)^-1 Qpq = J^-1 Qpq into m1Ppq (for wK)
    size_t startind = symm_big_skips_[begin];
    const size_t target_cols = metric_panel_cols();

    std::vector<size_t> col_off;
    std::vector<uint32_t> q_cnt(nbf_, 0);
    std::vector<uint32_t> q_off(nbf_ + 1, 0);
    std::vector<uint32_t> q_cstart(nbf_, 0);
    std::vector<uint32_t> q_fill(nbf_, 0);
    std::vector<uint32_t> active_q;
    active_q.reserve(nbf_);
    std::vector<uint32_t> lower_src;
    lower_src.reserve(target_cols);

    size_t total_upper = 0;
    for (size_t p = begin; p <= end; ++p) total_upper += symm_small_skips_[p];
    std::vector<uint32_t> upper_q(total_upper);
    std::vector<size_t> upper_p_off(end - begin + 2, 0);
    for (size_t p = begin; p <= end; ++p) {
        upper_p_off[p - begin + 1] = upper_p_off[p - begin] + symm_small_skips_[p];
    }
#pragma omp parallel for num_threads(nthreads_) schedule(guided) if (end - begin >= 64)
    for (size_t p = begin; p <= end; ++p) {
        uint32_t* dst = &upper_q[upper_p_off[p - begin]];
        size_t pos = 0;
        const size_t* srow = &schwarz_fun_index_[p * nbf_];
        for (size_t q = p; q < nbf_; ++q) {
            if (srow[q]) dst[pos++] = static_cast<uint32_t>(q);
        }
    }

    auto scatter_panel = [&](const double* out_ptr, double* dst_tensor, size_t p0, size_t p1, size_t ncol,
                             bool clear_q_cnt) {
        const size_t np_chunk = p1 - p0;
        const size_t nq_active = active_q.size();
#pragma omp parallel num_threads(nthreads_)
        {
#pragma omp for schedule(dynamic, 2)
            for (size_t idx = 0; idx < np_chunk; ++idx) {
                const size_t p = p0 + idx;
                const size_t mi = symm_small_skips_[p];
                const size_t si = small_skips_[p];
                const size_t ign = symm_ignored_columns_[p];
                const size_t coff = col_off[idx];
                double* dst_base = &dst_tensor[big_skips_[p] + ign];
                for (size_t Q = 0; Q < naux_; ++Q) {
                    const double* src = &out_ptr[Q * ncol + coff];
                    std::copy(src, src + mi, dst_base + Q * si);
                }
            }
#pragma omp for schedule(dynamic, 8)
            for (size_t qi = 0; qi < nq_active; ++qi) {
                const uint32_t q = active_q[qi];
                const uint32_t cnt = q_cnt[q];
                const uint32_t cstart = q_cstart[q];
                const uint32_t* src_cols = &lower_src[q_off[q]];
                const size_t sq = small_skips_[q];
                double* dst_base = &dst_tensor[big_skips_[q] + cstart];
                for (size_t Q = 0; Q < naux_; ++Q) {
                    const double* r = &out_ptr[Q * ncol];
                    double* dst = dst_base + Q * sq;
#pragma omp simd
                    for (uint32_t c = 0; c < cnt; ++c) {
                        dst[c] = r[src_cols[c]];
                    }
                }
                if (clear_q_cnt) q_cnt[q] = 0;
            }
        }
    };

    for (size_t p0 = begin; p0 <= end;) {
        size_t ncol = 0, p1 = p0;
        col_off.clear();
        while (p1 <= end && (ncol == 0 || ncol + symm_small_skips_[p1] <= target_cols)) {
            col_off.push_back(ncol);
            ncol += symm_small_skips_[p1];
            p1++;
        }
        if (ncol == 0) {
            p0 = p1;
            continue;
        }
        const size_t need = naux_ * ncol;
        double* panel_ptr = grow_scratch(metric_panel_buf_, metric_panel_size_, need);

        timer_on("DFH: AO-Met. gather");
#pragma omp parallel for num_threads(nthreads_) schedule(static)
        for (size_t Q = 0; Q < naux_; Q++) {
            double* prow = &panel_ptr[Q * ncol];
            for (size_t p = p0; p < p1; p++) {
                size_t mi = symm_small_skips_[p];
                const double* src = &Qpq[symm_big_skips_[p] - startind + Q * mi];
                std::copy(src, src + mi, prow + col_off[p - p0]);
            }
        }
        timer_off("DFH: AO-Met. gather");

        // Build CSR lower-scatter map once for both B and M1 scatters
        timer_on("DFH: AO-Met. scatter");
        active_q.clear();
        size_t total_lower = 0;
        for (size_t p = p0; p < p1; ++p) {
            const uint32_t* qs = &upper_q[upper_p_off[p - begin]];
            const size_t mi = symm_small_skips_[p];
            for (size_t k = 0; k < mi; ++k) {
                const uint32_t q = qs[k];
                if (q == p) continue;
                if (q_cnt[q] == 0) {
                    active_q.push_back(q);
                    q_cstart[q] = static_cast<uint32_t>(schwarz_fun_index_[(size_t)q * nbf_ + p] - 1);
                }
                q_cnt[q]++;
                total_lower++;
            }
        }
        lower_src.resize(total_lower);
        size_t running = 0;
        for (uint32_t q : active_q) {
            q_off[q] = static_cast<uint32_t>(running);
            q_fill[q] = static_cast<uint32_t>(running);
            running += q_cnt[q];
        }
        for (size_t p = p0; p < p1; ++p) {
            const uint32_t* qs = &upper_q[upper_p_off[p - begin]];
            const size_t mi = symm_small_skips_[p];
            const uint32_t base_col = static_cast<uint32_t>(col_off[p - p0]);
            for (size_t k = 0; k < mi; ++k) {
                const uint32_t q = qs[k];
                if (q == p) continue;
                lower_src[q_fill[q]++] = base_col + static_cast<uint32_t>(k);
            }
        }
        timer_off("DFH: AO-Met. scatter");

        // Step 1: B = L^-1 * Qpq -> Ppq
        timer_on("DFH: AO-Met. GEMM");
        C_DTRSM('L', 'L', 'N', 'N', naux_, ncol, 1.0, metp, naux_, panel_ptr, ncol);
        timer_off("DFH: AO-Met. GEMM");

        timer_on("DFH: AO-Met. scatter");
        scatter_panel(panel_ptr, Ppq, p0, p1, ncol, false);
        timer_off("DFH: AO-Met. scatter");

        // Step 2: M1 = L^-T * B = J^-1 * Qpq -> m1Ppq
        timer_on("DFH: AO-Met. GEMM");
        C_DTRSM('L', 'L', 'T', 'N', naux_, ncol, 1.0, metp, naux_, panel_ptr, ncol);
        timer_off("DFH: AO-Met. GEMM");

        timer_on("DFH: AO-Met. scatter");
        scatter_panel(panel_ptr, m1Ppq, p0, p1, ncol, true);
        timer_off("DFH: AO-Met. scatter");

        p0 = p1;
    }
}

void DFHelper::copy_upper_lower_wAO_core_symm(double* Qpq, double* Ppq, size_t begin, size_t end) {
    // Parallel upper copy + conflict-free CSR lower scatter partitioned by destination row q
    const size_t startind = symm_big_skips_[begin];
    const size_t np = end - begin + 1;

    struct LowerEntry {
        size_t qpq_off;
        uint32_t mi;
        uint32_t dst_col;
    };

    std::vector<uint32_t> q_cnt(nbf_, 0);
    std::vector<uint32_t> active_q;
    active_q.reserve(nbf_);
    size_t total_lower = 0;

    for (size_t p = begin; p <= end; ++p) {
        const size_t* srow = &schwarz_fun_index_[p * nbf_];
        for (size_t q = p + 1; q < nbf_; ++q) {
            if (!srow[q]) continue;
            if (q_cnt[q] == 0) active_q.push_back(static_cast<uint32_t>(q));
            q_cnt[q]++;
            total_lower++;
        }
    }

    std::vector<size_t> q_off(nbf_ + 1, 0);
    std::vector<size_t> q_fill(nbf_, 0);
    size_t running = 0;
    for (uint32_t q : active_q) {
        q_off[q] = running;
        q_fill[q] = running;
        running += q_cnt[q];
    }

    std::vector<LowerEntry> lower_entries(total_lower);
    for (size_t p = begin; p <= end; ++p) {
        const size_t mi = symm_small_skips_[p];
        const size_t p_base = symm_big_skips_[p] - startind;
        const size_t* srow = &schwarz_fun_index_[p * nbf_];
        const size_t upper_start = 1 + symm_ignored_columns_[p];
        for (size_t q = p + 1; q < nbf_; ++q) {
            const size_t fidx = srow[q];
            if (!fidx) continue;
            const size_t k = fidx - upper_start;
            const uint32_t dcol = static_cast<uint32_t>(schwarz_fun_index_[q * nbf_ + p] - 1);
            lower_entries[q_fill[q]++] = {p_base + k, static_cast<uint32_t>(mi), dcol};
        }
    }

    const size_t nq_active = active_q.size();
#pragma omp parallel num_threads(nthreads_)
    {
#pragma omp for schedule(dynamic, 2)
        for (size_t idx = 0; idx < np; ++idx) {
            const size_t j = begin + idx;
            const size_t mi = symm_small_skips_[j];
            const size_t si = small_skips_[j];
            const size_t jump = symm_ignored_columns_[j];
            const size_t skip1 = big_skips_[j] + jump;
            const size_t skip2 = symm_big_skips_[j] - startind;
            for (size_t Q = 0; Q < naux_; ++Q) {
                std::copy(&Qpq[skip2 + mi * Q], &Qpq[skip2 + mi * Q + mi], &Ppq[skip1 + si * Q]);
            }
        }
#pragma omp for schedule(dynamic, 8)
        for (size_t qi = 0; qi < nq_active; ++qi) {
            const uint32_t q = active_q[qi];
            const uint32_t cnt = q_cnt[q];
            const LowerEntry* entries = &lower_entries[q_off[q]];
            const size_t sq = small_skips_[q];
            double* dst_base = &Ppq[big_skips_[q]];
            constexpr size_t QB = 64;
            for (size_t q0 = 0; q0 < naux_; q0 += QB) {
                const size_t q1 = std::min(q0 + QB, naux_);
                for (uint32_t c = 0; c < cnt; ++c) {
                    const size_t src_off = entries[c].qpq_off;
                    const size_t src_mi = entries[c].mi;
                    const size_t dcol = entries[c].dst_col;
                    for (size_t Q = q0; Q < q1; ++Q) {
                        dst_base[Q * sq + dcol] = Qpq[src_off + Q * src_mi];
                    }
                }
            }
        }
    }
}

void DFHelper::add_space(std::string key, SharedMatrix M) {
    size_t a0 = M->rowspi()[0];
    size_t a1 = M->colspi()[0];

    if (!built_) {
        throw PSIEXCEPTION("DFHelper:add_space: call initialize() before adding spaces!");
    } else if (a0 != nbf_) {
        std::stringstream error;
        error << "DFHelper:add_space: illegal space (" << key << "), primary axis is not nbf_";
        throw PSIEXCEPTION(error.str().c_str());
    } else if (spaces_.find(key) != spaces_.end()) {
        if (a1 != std::get<1>(spaces_[key])) {
            std::stringstream error;
            error << "DFHelper:add_space: illegal space (" << key << "), new space has incorrect dimension!";
            throw PSIEXCEPTION(error.str().c_str());
        }
    }
    sorted_spaces_.push_back(std::make_pair(key, a1));
    spaces_[key] = std::make_tuple(M, a1);
}
void DFHelper::add_transformation(std::string name, std::string key1, std::string key2, std::string order) {
    if (spaces_.find(key1) == spaces_.end()) {
        std::stringstream error;
        error << "DFHelper:add_transformation: first space (" << key1 << "), is not in space list!";
        throw PSIEXCEPTION(error.str().c_str());
    } else if (spaces_.find(key2) == spaces_.end()) {
        std::stringstream error;
        error << "DFHelper:add_transformation: second space (" << key2 << "), is not in space list!";
        throw PSIEXCEPTION(error.str().c_str());
    }

    int op;
    if (!order.compare("Qpq")) {
        op = 0;
    } else if (!order.compare("pQq")) {
        op = 1;
    } else if (!order.compare("pqQ")) {
        op = 2;
    } else {
        throw PSIEXCEPTION("DF_Hepler:add_transformation: incorrect integral format, use 'Qpq', 'pQq', or 'pqQ'");
    }
    transf_[name] = std::make_tuple(key1, key2, op);

    size_t a1 = std::get<1>(spaces_[key1]);
    size_t a2 = std::get<1>(spaces_[key2]);
    filename_maker(name, naux_, a1, a2, op);
}
void DFHelper::clear_spaces() {
    // clear spaces
    spaces_.clear();
    sorted_spaces_.clear();
    order_.clear();
    bspace_.clear();
    strides_.clear();

    // no ordering
    ordered_ = false;
    transformed_ = false;
}

void DFHelper::clear_transformations() {
    // clears transformations

    transf_.clear();
    transf_core_.clear();
}

void DFHelper::clear_all() {
    // invokes destructors, eliminating all files.
    file_streams_.clear();

    // clears all info
    clear_spaces();
    files_.clear();
    sizes_.clear();
    tsizes_.clear();
    clear_transformations();
}

std::pair<size_t, size_t> DFHelper::identify_order() {
    // Identify order of transformations to use strategic intermediates
    std::sort(sorted_spaces_.begin(), sorted_spaces_.end(),
              [](const std::pair<std::string, size_t>& left, const std::pair<std::string, size_t>& right) {
                  return left.second < right.second;
              });

    // copy transf_ keys into a list of needs
    std::list<std::string> needs;
    for (auto const& itr : transf_) needs.push_back(itr.first);

    // construct best transformation order
    size_t largest = 0, maximum = 0, small, large, op;
    for (size_t i = 0; i < sorted_spaces_.size(); i++) {
        bool on = false;
        size_t st = 0;
        std::string str = sorted_spaces_[i].first;

        auto itr = needs.begin();
        while (itr != needs.end()) {
            op = 0;
            op = (!(std::get<0>(transf_[*itr]).compare(str)) ? 1 : op);
            op = (!(std::get<1>(transf_[*itr]).compare(str)) ? 2 : op);
            if (op != 0) {
                if (!on) {
                    bspace_.push_back(str);
                    on = true;
                }
                small = (op == 1 ? std::get<1>(spaces_[std::get<0>(transf_[*itr])])
                                 : std::get<1>(spaces_[std::get<1>(transf_[*itr])]));
                large = (op == 1 ? std::get<1>(spaces_[std::get<1>(transf_[*itr])])
                                 : std::get<1>(spaces_[std::get<0>(transf_[*itr])]));
                maximum = (maximum < small * large ? small * large : maximum);
                largest = (largest < small ? small : largest);
                order_.push_back(*itr);
                st++;
                itr = needs.erase(itr);
            } else
                itr++;
        }
        if (st > 0) {
            strides_.push_back(st);
        }
    }
    // print_order();
    ordered_ = true;
    return std::make_pair(largest, maximum);
}
void DFHelper::print_order() {
    size_t o = order_.size();
    size_t b = bspace_.size();
    outfile->Printf("\n     ==> DFHelper:--Begin Transformations Information <==\n\n");
    outfile->Printf("   Transformation order:\n");
    for (size_t i = 0; i < o; i++) {
        outfile->Printf("         %s: (%s, %s)\n", order_[i].c_str(), std::get<0>(transf_[order_[i]]).c_str(),
                        std::get<1>(transf_[order_[i]]).c_str());
    }
    outfile->Printf("\n    Best Spaces:\n");
    for (size_t i = 0; i < b; i++) {
        outfile->Printf("         (space: %s, size: %zu)\n", bspace_[i].c_str(), std::get<1>(spaces_[bspace_[i]]));
    }
    outfile->Printf("\n    Transformation strides: ");
    for (size_t i = 0; i < b; i++) {
        outfile->Printf("%zu", strides_[i]);
        if (i < b - 1) outfile->Printf(", ");
    }
    outfile->Printf("\n\n     ==> DFHelper:--End Transformations Information <==\n\n");
}

void DFHelper::transform() {
    if (debug_) {
        outfile->Printf("Entering DFHelper::transform\n");
    }

    timer_on("DFH: transform()");
    // outfile->Printf("\n     ==> DFHelper:--Begin Transformations <==\n\n");


    // reset tranposes (in case the transpose() function was called)
    tsizes_.clear();

    // get optimal path and info
    if (!ordered_) info_ = identify_order();
    size_t wtmp = std::get<0>(info_);
    size_t wfinal = std::get<1>(info_);

    // prep AO file stream if STORE + !AO_core_
    if (!direct_iaQ_ && !direct_ && !AO_core_) stream_check(AO_names_[1], "rb");

    // get Q blocking scheme
    std::vector<std::pair<size_t, size_t>> Qsteps;
    std::pair<size_t, size_t> Qlargest = Qshell_blocks_for_transform(memory_, wtmp, wfinal, Qsteps);
    size_t max_block = std::get<1>(Qlargest);

    // prepare eri and C buffers per thread
    size_t nthread = nthreads_;
    std::vector<DFHelperScratch> C_buffers(nthreads_);
    std::shared_ptr<BasisSet> zero = BasisSet::zero_ao_basis_set();
    auto rifactory = std::make_shared<IntegralFactory>(aux_, zero, primary_, primary_);
    std::vector<std::shared_ptr<TwoBodyAOInt>> eri(nthread);
    eri[0] = std::shared_ptr<TwoBodyAOInt>(rifactory->eri(0, false));
#pragma omp parallel for num_threads(nthreads_) schedule(static)
    for (size_t rank = 0; rank < nthreads_; ++rank) {
        C_buffers[rank].resize(nbf_ * wtmp);
        if (rank) {
            eri[rank] = std::shared_ptr<TwoBodyAOInt>(eri.front()->clone());
        }
    }

    // allocate in-core transformed integrals if necessary
    if (MO_core_) {
        for (auto& kv : transf_) {
            size_t size = std::get<1>(spaces_[std::get<0>(kv.second)]) * std::get<1>(spaces_[std::get<1>(kv.second)]);
            transf_core_[kv.first] = std::unique_ptr<double[]>(new double[size * naux_]);
        }
    }

    // scope buffer declarations
    {
        // declare buffers
        std::unique_ptr<double[]> T(new double[max_block * nbf_ * wtmp]);
        std::unique_ptr<double[]> F(new double[max_block * wfinal]);
        std::unique_ptr<double[]> N;
        double* Tp = T.get();
        double* Fp = F.get();
        double* Np;
        if (!MO_core_) {
            N = std::unique_ptr<double[]>(new double[max_block * wfinal]);
            Np = N.get();
        }

        // AO buffer, allocate if not in-core, else point to in-core
        std::unique_ptr<double[]> M;
        double* Mp;
        if (!AO_core_) {
            M = std::unique_ptr<double[]>(new double[std::get<0>(Qlargest)]);
            Mp = M.get();
        } else {
            Mp = Ppq_.get();
        }

        // transform in steps, blocking over the auxiliary basis (Q blocks)
        for (size_t j = 0, bcount = 0, block_size; j < Qsteps.size(); j++, bcount += block_size) {
            // Qshell step info
            size_t start = std::get<0>(Qsteps[j]);
            size_t stop = std::get<1>(Qsteps[j]);
            size_t begin = Qshell_aggs_[start];
            size_t end = Qshell_aggs_[stop + 1] - 1;
            block_size = end - begin + 1;

            // print step info
            // outfile->Printf("      Qshell: (%zu, %zu)", start, stop);
            // outfile->Printf(", PHI: (%zu, %zu), size: %zu\n", begin, end, block_size);

            // get AO chunk according to directives
            if (AO_core_) {
                ;  // pass
            } else if (direct_iaQ_) {
                timer_on("DFH: Total Workflow");
                compute_dense_Qpq_blocking_Q(start, stop, Mp, eri);
                timer_off("DFH: Total Workflow");
            } else if (direct_) {
                timer_on("DFH: Total Workflow");
                compute_sparse_pQq_blocking_Q(start, stop, Mp, eri);
                timer_off("DFH: Total Workflow");
            } else {
                timer_on("DFH: Grabbing AOs");
                grab_AO(start, stop, Mp);
                timer_off("DFH: Grabbing AOs");
            }

            // stride through best spaces
            for (size_t i = 0, count = 0; i < bspace_.size(); count += strides_[i], i++) {
                // grab best space
                std::string bspace = bspace_[i];
                size_t bsize = std::get<1>(spaces_[bspace]);
                double* Bp = std::get<0>(spaces_[bspace])->pointer()[0];

                // index bump if AOs are in core
                size_t bump = (AO_core_ ? bcount * nbf_ * nbf_ : 0);

                // perform first contraction
                timer_on("DFH: Total Workflow");
                timer_on("DFH: Total Transform");
                timer_on("DFH: 1st Contraction");
                if (direct_iaQ_) {
                    // (qb)(Q|pq)->(Q|pb)
                    C_DGEMM('N', 'N', block_size * nbf_, bsize, nbf_, 1.0, &Mp[bump], nbf_, Bp, bsize, 0.0, Tp, bsize);
                } else {
                    // (bq)(p|Qq)->(p|Qb)
                    first_transform_pQq(bsize, bcount, block_size, Mp, Tp, Bp, C_buffers);
                }
                timer_off("DFH: 1st Contraction");
                timer_off("DFH: Total Transform");
                timer_off("DFH: Total Workflow");

                // to completion per transformation
                for (size_t k = 0; k < strides_[i]; k++) {
                    // get transformation info
                    auto transf_name = order_[count + k];
                    auto left = std::get<0>(transf_[transf_name]);
                    auto right = std::get<1>(transf_[transf_name]);
                    bool bleft = (bspace.compare(left) == 0 ? true : false);

                    // get worst space
                    std::tuple<SharedMatrix, size_t> I = (bleft ? spaces_[right] : spaces_[left]);
                    double* Wp = std::get<0>(I)->pointer()[0];
                    size_t wsize = std::get<1>(I);

                    // grab in-core pointer
                    if (direct_iaQ_ && MO_core_) {
                        Fp = transf_core_[order_[count + k]].get();
                    } else if (MO_core_) {
                        Np = transf_core_[order_[count + k]].get();
                    }

                    // perform final contraction
                    // (wp)(p|Qb)->(w|Qb)
                    timer_on("DFH: Total Workflow");
                    timer_on("DFH: Total Transform");
                    timer_on("DFH: 2nd Contraction");
                    if (direct_iaQ_) {
                        size_t bump = (MO_core_ ? begin * wsize * bsize : 0);
                        // (pw)(Q|pb)->(Q|bw)
                        if (bleft) {
#pragma omp parallel for num_threads(nthreads_)
                            for (size_t i = 0; i < block_size; i++) {
                                C_DGEMM('T', 'N', bsize, wsize, nbf_, 1.0, &Tp[i * nbf_ * bsize], bsize, Wp, wsize, 0.0,
                                        &Fp[bump + i * wsize * bsize], wsize);
                            }
                        } else {
// (pw)(Q|pb)->(Q|wb)
#pragma omp parallel for num_threads(nthreads_)
                            for (size_t i = 0; i < block_size; i++) {
                                C_DGEMM('T', 'N', wsize, bsize, nbf_, 1.0, Wp, wsize, &Tp[i * nbf_ * bsize], bsize, 0.0,
                                        &Fp[bump + i * wsize * bsize], bsize);
                            }
                        }
                    } else {
                        // (pw)(p|Qb)->(w|Qb)
                        C_DGEMM('T', 'N', wsize, block_size * bsize, nbf_, 1.0, Wp, wsize, Tp, block_size * bsize, 0.0,
                                Fp, block_size * bsize);
                    }
                    timer_off("DFH: 2nd Contraction");
                    timer_off("DFH: Total Transform");
                    timer_off("DFH: Total Workflow");

                    // put the transformations away
                    timer_on("DFH: MO to disk, " + transf_name);
                    if (direct_iaQ_) {
                        put_transformations_Qpq(begin, end, wsize, bsize, Fp, count + k, bleft);
                    } else {
                        put_transformations_pQq(begin, end, block_size, bcount, wsize, bsize, Np, Fp, count + k, bleft);
                    }
                    timer_off("DFH: MO to disk, " + transf_name);
                }
            }
        }
    }  // buffers destroyed with std housekeeping

    // outfile->Printf("\n     ==> DFHelper:--End Transformations (disk)<==\n\n");

    // transformations complete, time for metric contractions

    timer_on("DFH: Metric Contractions");

    // release in-core AO 
    if (AO_core_ && release_core_AO_before_metric_)  {
        Ppq_.reset();
    }

    if (direct_iaQ_ || direct_) {
        // prepare metric
        std::unique_ptr<double[]> metric;
        double* metp;
        if (!hold_met_) {
            metric = std::unique_ptr<double[]>(new double[naux_ * naux_]);
            metp = metric.get();
            std::string filename = return_metfile(mpower_);
            get_tensor_(std::get<0>(files_[filename]), metp, 0, naux_ - 1, 0, naux_ - 1);
        } else
            metp = metric_prep_core(mpower_);

        if (direct_iaQ_) {
            if (MO_core_) {
                std::unique_ptr<double[]> N(new double[naux_ * wfinal]);
                double* Np = N.get();

                for (auto& kv : transf_core_) {
                    size_t l = std::get<0>(sizes_[std::get<1>(files_[kv.first])]);
                    size_t r = std::get<1>(sizes_[std::get<1>(files_[kv.first])]);
                    size_t Q = std::get<2>(sizes_[std::get<1>(files_[kv.first])]);

                    double* Lp = kv.second.get();
                    C_DCOPY(l * r * Q, Lp, 1, Np, 1);

                    // (Q|ia) (PQ) -> (ia|Q)
                    C_DGEMM('T', 'N', l * r, Q, Q, 1.0, Np, l * r, metp, Q, 0.0, Lp, Q);
                }

            } else {
                // total size allowed, in doubles
                size_t rem_mem = memory_ - (AO_core_ && !release_core_AO_before_metric_ ? naux_ * nbf_ * nbf_ : 0);
                size_t total_mem =
                    (rem_mem > wfinal * naux_ * 2 + naux_ * naux_ ? wfinal * naux_ : (rem_mem - naux_ * naux_) / 2);

                std::unique_ptr<double[]> M(new double[total_mem]);
                std::unique_ptr<double[]> F(new double[total_mem]);
                double* Mp = M.get();
                double* Fp = F.get();
                for (std::vector<std::string>::iterator itr = order_.begin(); itr != order_.end(); itr++)
                    contract_metric_Qpq(*itr, metp, Mp, Fp, total_mem);
            }

        } else if (direct_) {
            if (!MO_core_) {
                // total size allowed, in doubles.
                // note that memory - naux_2 cannot be negative (handled in init)
                size_t rem_mem = memory_ - (AO_core_ && !release_core_AO_before_metric_ ? big_skips_[nbf_] : 0);
                size_t total_mem =
                    (rem_mem > wfinal * naux_ * 2 + naux_ * naux_ ? wfinal * naux_ : (rem_mem - naux_ * naux_) / 2);

                std::unique_ptr<double[]> M(new double[total_mem]);
                std::unique_ptr<double[]> F(new double[total_mem]);
                double* Mp = M.get();
                double* Fp = F.get();

                for (std::vector<std::string>::iterator itr = order_.begin(); itr != order_.end(); itr++)
                    contract_metric(*itr, metp, Mp, Fp, total_mem);

            } else {
                std::unique_ptr<double[]> N(new double[naux_ * wfinal]);
                double* Np = N.get();

                for (auto& kv : transf_core_) {
                    size_t a0 = std::get<0>(sizes_[std::get<1>(files_[kv.first])]);
                    size_t a1 = std::get<1>(sizes_[std::get<1>(files_[kv.first])]);
                    size_t a2 = std::get<2>(sizes_[std::get<1>(files_[kv.first])]);

                    double* Lp = kv.second.get();
                    C_DCOPY(a0 * a1 * a2, Lp, 1, Np, 1);

                    // the following differs depending on the form being outputted
                    // 0 : Qpq -- 1 : pQq -- 2 : pqQ

                    if (std::get<2>(transf_[kv.first]) == 2) {
                        C_DGEMM('N', 'N', a0 * a1, a2, a2, 1.0, Np, a2, metp, a2, 0.0, Lp, a2);
                    } else if (std::get<2>(transf_[kv.first]) == 0) {
                        C_DGEMM('N', 'N', a0, a1 * a2, a0, 1.0, metp, naux_, Np, a1 * a2, 0.0, Lp, a1 * a2);
                    } else {
#pragma omp parallel for num_threads(nthreads_)
                        for (size_t i = 0; i < a0; i++) {
                            C_DGEMM('N', 'N', a1, a2, a1, 1.0, metp, naux_, &Np[i * a1 * a2], a2, 0.0, &Lp[i * a1 * a2],
                                    a2);
                        }
                    }
                }
            }
        }
    }
    timer_off("DFH: Metric Contractions");
    timer_off("DFH: transform()");
    transformed_ = true;

    if (debug_) {
        outfile->Printf("Exiting DFHelper::transform\n");
    }
}

void DFHelper::first_transform_pQq(size_t bsize, size_t bcount, size_t block_size, double* Mp, double* Tp, double* Bp,
                                   std::vector<DFHelperScratch>& C_buffers) {
// perform first contraction on pQq, thread over p.
#pragma omp parallel for schedule(guided) num_threads(nthreads_)
    for (size_t k = 0; k < nbf_; k++) {
        size_t sp_size = small_skips_[k];
        if (sp_size == 0) {
            std::fill_n(&Tp[k * block_size * bsize], block_size * bsize, 0.0);
            continue;
        }
        size_t jump = (AO_core_ ? big_skips_[k] + bcount * sp_size : (big_skips_[k] * block_size) / naux_);

        int rank = 0;
#ifdef _OPENMP
        rank = omp_get_thread_num();
#endif
        double* Cb = C_buffers[rank].data();
        const uint32_t* sig_m = &sig_q_[sig_q_row_off_[k]];
        for (size_t c = 0; c < sp_size; ++c) {
            std::copy_n(&Bp[static_cast<size_t>(sig_m[c]) * bsize], bsize, &Cb[c * bsize]);
        }

        // (Qm)(mb)->(Qb)
        C_DGEMM('N', 'N', block_size, bsize, sp_size, 1.0, &Mp[jump], sp_size, Cb, bsize, 0.0,
                &Tp[k * block_size * bsize], bsize);
    }
}

// For row p of the in-core AOs: C^T restricted to p's significant m, as an (nocc_pad x sp_size)
// block whose padding rows are zero. Tiled in 16x16 blocks so writes stay L1-cache resident.
template <typename T>
static void gather_Ct_pad(T* Ct, const double* Bp, const uint32_t* sig_m, size_t nocc, size_t nocc_pad,
                          size_t sp_size) {
    constexpr size_t TB = 16;
    for (size_t c0 = 0; c0 < sp_size; c0 += TB) {
        const size_t cb = std::min(TB, sp_size - c0);
        for (size_t i0 = 0; i0 < nocc; i0 += TB) {
            const size_t ib = std::min(TB, nocc - i0);
            for (size_t c = 0; c < cb; ++c) {
                const double* src = &Bp[static_cast<size_t>(sig_m[c0 + c]) * nocc + i0];
                for (size_t i = 0; i < ib; ++i) {
                    Ct[(i0 + i) * sp_size + (c0 + c)] = static_cast<T>(src[i]);
                }
            }
        }
    }
    if (nocc_pad > nocc) {
        std::fill(Ct + nocc * sp_size, Ct + nocc_pad * sp_size, T(0));
    }
}

void DFHelper::first_transform_pQq_K(size_t nocc, size_t nocc_pad, size_t bcount, size_t block_size, double* Mp,
                                     double* Tp, double* Bp, std::vector<DFHelperScratch>& C_buffers) {
    // K-build variant of first_transform_pQq: T[p] is stored (i x Q) with i padded to nocc_pad
    // (zero rows). The K contraction only needs both factors in the same layout, and this
    // orientation with a multiple-of-8 row count runs ~3x faster in MKL on AMD Zen CPUs.
#pragma omp parallel for schedule(guided) num_threads(nthreads_)
    for (size_t k = 0; k < nbf_; k++) {
        size_t sp_size = small_skips_[k];
        if (sp_size == 0) {
            std::fill_n(&Tp[k * block_size * nocc_pad], block_size * nocc_pad, 0.0);
            continue;
        }
        size_t jump = (AO_core_ ? big_skips_[k] + bcount * sp_size : (big_skips_[k] * block_size) / naux_);

        int rank = 0;
#ifdef _OPENMP
        rank = omp_get_thread_num();
#endif
        double* Ct = C_buffers[rank].data();
        gather_Ct_pad(Ct, Bp, &sig_q_[sig_q_row_off_[k]], nocc, nocc_pad, sp_size);

        // (im)(Qm)^T -> (iQ)
        C_DGEMM('N', 'T', nocc_pad, block_size, sp_size, 1.0, Ct, sp_size, &Mp[jump], sp_size, 0.0,
                &Tp[k * block_size * nocc_pad], block_size);
    }
}

void DFHelper::put_transformations_Qpq(int begin, int end, int wsize, int bsize, double* Fp, int ind, bool bleft) {
    // incoming transformed integrals to this function are in a Qpq format.
    // if MO_core is on, do nothing
    // else, the buffers are put to disk.

    if (!MO_core_) {
        // "ab" is great here since we are actually appending.
        std::string putf = std::get<0>(files_[order_[ind]]);
        std::string op = "ab";

        if (bleft) {
            put_tensor(putf, Fp, std::make_pair(begin, end), std::make_pair(0, bsize - 1), std::make_pair(0, wsize - 1),
                       op);
        } else {
            put_tensor(putf, Fp, std::make_pair(begin, end), std::make_pair(0, wsize - 1), std::make_pair(0, bsize - 1),
                       op);
        }
    }
}

void DFHelper::put_transformations_pQq(int begin, int end, int rblock_size, int bcount, int wsize, int bsize,
                                       double* Np, double* Fp, int ind, bool bleft) {
    // incoming transformed integrals to this function are in a pQq format.
    // first, the integrals are tranposed to the desired format specified in add_transformation().
    // if MO_core is on, then the LHS buffers are final destinations.
    // else, the buffers are put to disk.

    // setup ~
    int lblock_size = rblock_size;
    std::string putf, op;
    if (!MO_core_) {
        putf = (!direct_ ? std::get<1>(files_[order_[ind]]) : std::get<0>(files_[order_[ind]]));
        op = "wb";
        bcount = 0;
    } else {
        lblock_size = naux_;
    }

    if (bleft) {
        // result is in pqQ format
        if (std::get<2>(transf_[order_[ind]]) == 2) {
// (w|Qb)->(bw|Q)
#pragma omp parallel for num_threads(nthreads_)
            for (size_t z = 0; z < wsize; z++) {
                for (size_t y = 0; y < bsize; y++) {
                    for (size_t x = 0; x < rblock_size; x++) {
                        Np[y * wsize * lblock_size + z * lblock_size + (bcount + x)] =
                            Fp[z * bsize * rblock_size + x * bsize + y];
                    }
                }
            }
            if (!MO_core_) {
                put_tensor(putf, Np, std::make_pair(0, bsize - 1), std::make_pair(0, wsize - 1),
                           std::make_pair(begin, end), op);
            }

            // result is in Qpq format
        } else if (std::get<2>(transf_[order_[ind]]) == 0) {
// (w|Qb)->(Q|bw)
#pragma omp parallel for num_threads(nthreads_)
            for (size_t x = 0; x < rblock_size; x++) {
                for (size_t z = 0; z < wsize; z++) {
                    for (size_t y = 0; y < bsize; y++) {
                        Np[(bcount + x) * bsize * wsize + y * wsize + z] = Fp[z * bsize * rblock_size + x * bsize + y];
                    }
                }
            }
            if (!MO_core_) {
                put_tensor(putf, Np, std::make_pair(begin, end), std::make_pair(0, bsize - 1),
                           std::make_pair(0, wsize - 1), op);
            }

            // result is in pQq format
        } else {
// (w|Qb)->(bQw)
#pragma omp parallel for num_threads(nthreads_)
            for (size_t x = 0; x < rblock_size; x++) {
                for (size_t y = 0; y < bsize; y++) {
                    for (size_t z = 0; z < wsize; z++) {
                        Np[y * lblock_size * wsize + (bcount + x) * wsize + z] =
                            Fp[z * bsize * rblock_size + x * bsize + y];
                    }
                }
            }
            if (!MO_core_) {
                put_tensor(putf, Np, std::make_pair(0, bsize - 1), std::make_pair(begin, end),
                           std::make_pair(0, wsize - 1), op);
            }
        }

    } else {
        // result is in pqQ format
        if (std::get<2>(transf_[order_[ind]]) == 2) {
// (w|Qb)->(wbQ)
#pragma omp parallel for num_threads(nthreads_)
            for (size_t z = 0; z < wsize; z++) {
                for (size_t x = 0; x < rblock_size; x++) {
                    for (size_t y = 0; y < bsize; y++) {
                        Np[z * lblock_size * bsize + y * lblock_size + (bcount + x)] =
                            Fp[z * rblock_size * bsize + x * bsize + y];
                    }
                }
            }
            if (!MO_core_) {
                put_tensor(putf, Np, std::make_pair(0, wsize - 1), std::make_pair(0, bsize - 1),
                           std::make_pair(begin, end), op);
            }

            // result is in Qpq format
        } else if (std::get<2>(transf_[order_[ind]]) == 0) {
// (w|Qb)->(Q|wb)
#pragma omp parallel for num_threads(nthreads_)
            for (size_t x = 0; x < rblock_size; x++) {
                for (size_t z = 0; z < wsize; z++) {
                    C_DCOPY(bsize, &Fp[z * rblock_size * bsize + x * bsize], 1,
                            &Np[(bcount + x) * wsize * bsize + z * bsize], 1);
                }
            }
            if (!MO_core_) {
                put_tensor(putf, Np, std::make_pair(begin, end), std::make_pair(0, wsize - 1),
                           std::make_pair(0, bsize - 1), op);
            }

            // result is in pQq format
        } else {
            // (w|Qb)
            if (!MO_core_) {
                put_tensor(putf, Fp, std::make_pair(0, wsize - 1), std::make_pair(begin, end),
                           std::make_pair(0, bsize - 1), op);
            } else {
// we have to copy over the buffer
#pragma omp parallel for num_threads(nthreads_)
                for (size_t x = 0; x < wsize; x++) {
                    for (size_t y = 0; y < rblock_size; y++) {
                        C_DCOPY(bsize, &Fp[x * rblock_size * bsize + y * bsize], 1,
                                &Np[x * lblock_size * bsize + (bcount + y) * bsize], 1);
                    }
                }
            }
        }
    }
}

// Fill using a pointer, be cautious of bounds!!
void DFHelper::fill_tensor(std::string name, double* b) {
    check_file_key(name);
    std::string filename = std::get<1>(files_[name]);
    std::tuple<size_t, size_t, size_t> sizes;
    sizes = (tsizes_.find(filename) != tsizes_.end() ? tsizes_[filename] : sizes_[filename]);

    fill_tensor(name, b, {0, std::get<0>(sizes)}, {0, std::get<1>(sizes)}, {0, std::get<2>(sizes)});
}
void DFHelper::fill_tensor(std::string name, double* b, std::vector<size_t> a1) {
    check_file_key(name);
    std::string filename = std::get<1>(files_[name]);
    std::tuple<size_t, size_t, size_t> sizes;
    sizes = (tsizes_.find(filename) != tsizes_.end() ? tsizes_[filename] : sizes_[filename]);

    fill_tensor(name, b, a1, {0, std::get<1>(sizes)}, {0, std::get<2>(sizes)});
}
void DFHelper::fill_tensor(std::string name, double* b, std::vector<size_t> a1, std::vector<size_t> a2) {
    check_file_key(name);
    std::string filename = std::get<1>(files_[name]);
    std::tuple<size_t, size_t, size_t> sizes;
    sizes = (tsizes_.find(filename) != tsizes_.end() ? tsizes_[filename] : sizes_[filename]);

    fill_tensor(name, b, a1, a2, {0, std::get<2>(sizes)});
}
void DFHelper::fill_tensor(std::string name, double* b, std::vector<size_t> a1, std::vector<size_t> a2,
                           std::vector<size_t> a3) {
    if (a1.size() != 2) {
        std::stringstream error;
        error << "DFHelper:fill_tensor:  axis 0 tensor indexing vector has " << a1.size() << " elements!";
        throw PSIEXCEPTION(error.str().c_str());
    }
    if (a2.size() != 2) {
        std::stringstream error;
        error << "DFHelper:fill_tensor:  axis 1 tensor indexing vector has " << a2.size() << " elements!";
        throw PSIEXCEPTION(error.str().c_str());
    }
    if (a3.size() != 2) {
        std::stringstream error;
        error << "DFHelper:fill_tensor:  axis 2 tensor indexing vector has " << a3.size() << " elements!";
        throw PSIEXCEPTION(error.str().c_str());
    }

    check_file_key(name);
    std::string filename = std::get<1>(files_[name]);

    // being pythonic ;)
    std::pair<size_t, size_t> i0 = std::make_pair(a1[0], a1[1] - 1);
    std::pair<size_t, size_t> i1 = std::make_pair(a2[0], a2[1] - 1);
    std::pair<size_t, size_t> i2 = std::make_pair(a3[0], a3[1] - 1);

    get_tensor_(filename, b, i0, i1, i2);
}

// Fill using a pre-allocated SharedMatrix
void DFHelper::fill_tensor(std::string name, SharedMatrix M) {
    std::string filename = std::get<1>(files_[name]);
    std::tuple<size_t, size_t, size_t> sizes;
    sizes = (tsizes_.find(filename) != tsizes_.end() ? tsizes_[filename] : sizes_[filename]);

    fill_tensor(name, M, {0, std::get<0>(sizes)}, {0, std::get<1>(sizes)}, {0, std::get<2>(sizes)});
}
void DFHelper::fill_tensor(std::string name, SharedMatrix M, std::vector<size_t> a1) {
    std::string filename = std::get<1>(files_[name]);
    std::tuple<size_t, size_t, size_t> sizes;
    sizes = (tsizes_.find(filename) != tsizes_.end() ? tsizes_[filename] : sizes_[filename]);

    fill_tensor(name, M, a1, {0, std::get<1>(sizes)}, {0, std::get<2>(sizes)});
}
void DFHelper::fill_tensor(std::string name, SharedMatrix M, std::vector<size_t> a1, std::vector<size_t> a2) {
    std::string filename = std::get<1>(files_[name]);
    std::tuple<size_t, size_t, size_t> sizes;
    sizes = (tsizes_.find(filename) != tsizes_.end() ? tsizes_[filename] : sizes_[filename]);

    fill_tensor(name, M, a1, a2, {0, std::get<2>(sizes)});
}
void DFHelper::fill_tensor(std::string name, SharedMatrix M, std::vector<size_t> t0, std::vector<size_t> t1,
                           std::vector<size_t> t2) {
    std::string filename = std::get<1>(files_[name]);
    // has this integral been transposed?
    std::tuple<size_t, size_t, size_t> sizes;
    sizes = (tsizes_.find(filename) != tsizes_.end() ? tsizes_[filename] : sizes_[filename]);

    if (t0.size() != 2) {
        std::stringstream error;
        error << "DFHelper:fill_tensor:  axis 0 tensor indexing vector has " << t0.size() << " elements!";
        throw PSIEXCEPTION(error.str().c_str());
    }
    if (t1.size() != 2) {
        std::stringstream error;
        error << "DFHelper:fill_tensor:  axis 1 tensor indexing vector has " << t1.size() << " elements!";
        throw PSIEXCEPTION(error.str().c_str());
    }
    if (t2.size() != 2) {
        std::stringstream error;
        error << "DFHelper:fill_tensor:  axis 2 tensor indexing vector has " << t2.size() << " elements!";
        throw PSIEXCEPTION(error.str().c_str());
    }

    // be pythonic - adjust stops
    size_t sta0 = t0[0];
    size_t sto0 = t0[1] - 1;
    size_t sta1 = t1[0];
    size_t sto1 = t1[1] - 1;
    size_t sta2 = t2[0];
    size_t sto2 = t2[1] - 1;

    std::pair<size_t, size_t> i0 = std::make_pair(sta0, sto0);
    std::pair<size_t, size_t> i1 = std::make_pair(sta1, sto1);
    std::pair<size_t, size_t> i2 = std::make_pair(sta2, sto2);

    check_file_key(name);
    check_file_tuple(name, i0, i1, i2);
    check_matrix_size(name, M, i0, i1, i2);

    size_t A0 = (sto0 - sta0 + 1);
    size_t A1 = (sto1 - sta1 + 1);
    size_t A2 = (sto2 - sta2 + 1);

    double* Mp = M->pointer()[0];
    if (MO_core_) {
        size_t a0 = std::get<0>(sizes);
        size_t a1 = std::get<1>(sizes);
        size_t a2 = std::get<2>(sizes);

        double* Fp = transf_core_[name].get();
#pragma omp parallel for num_threads(nthreads_)
        for (size_t i = 0; i < A0; i++) {
            for (size_t j = 0; j < A1; j++) {
#pragma omp simd
                for (size_t k = 0; k < A2; k++) {
                    Mp[i * A1 * A2 + j * A2 + k] = Fp[(sta0 + i) * a1 * a2 + (sta1 + j) * a2 + (sta2 + k)];
                }
            }
        }
    } else {
        get_tensor_(filename, Mp, i0, i1, i2);
    }
    M->set_numpy_shape({(int)A0, (int)A1, (int)A2});
}

// Return a SharedMatrix
SharedMatrix DFHelper::get_tensor(std::string name) {
    std::string filename = std::get<1>(files_[name]);
    std::tuple<size_t, size_t, size_t> sizes;
    sizes = (tsizes_.find(filename) != tsizes_.end() ? tsizes_[filename] : sizes_[filename]);

    return get_tensor(name, {0, std::get<0>(sizes)}, {0, std::get<1>(sizes)}, {0, std::get<2>(sizes)});
}
SharedMatrix DFHelper::get_tensor(std::string name, std::vector<size_t> a1) {
    std::string filename = std::get<1>(files_[name]);
    std::tuple<size_t, size_t, size_t> sizes;
    sizes = (tsizes_.find(filename) != tsizes_.end() ? tsizes_[filename] : sizes_[filename]);

    return get_tensor(name, a1, {0, std::get<1>(sizes)}, {0, std::get<2>(sizes)});
}
SharedMatrix DFHelper::get_tensor(std::string name, std::vector<size_t> a1, std::vector<size_t> a2) {
    std::string filename = std::get<1>(files_[name]);
    std::tuple<size_t, size_t, size_t> sizes;
    sizes = (tsizes_.find(filename) != tsizes_.end() ? tsizes_[filename] : sizes_[filename]);

    return get_tensor(name, a1, a2, {0, std::get<2>(sizes)});
}
SharedMatrix DFHelper::get_tensor(std::string name, std::vector<size_t> t0, std::vector<size_t> t1,
                                  std::vector<size_t> t2) {
    // has this integral been transposed?
    std::string filename = std::get<1>(files_[name]);
    std::tuple<size_t, size_t, size_t> sizes;
    sizes = (tsizes_.find(filename) != tsizes_.end() ? tsizes_[filename] : sizes_[filename]);

    if (t0.size() != 2) {
        std::stringstream error;
        error << "DFHelper:fill_tensor:  axis 0 tensor indexing vector has " << t0.size() << " elements!";
        throw PSIEXCEPTION(error.str().c_str());
    }
    if (t1.size() != 2) {
        std::stringstream error;
        error << "DFHelper:fill_tensor:  axis 1 tensor indexing vector has " << t1.size() << " elements!";
        throw PSIEXCEPTION(error.str().c_str());
    }
    if (t2.size() != 2) {
        std::stringstream error;
        error << "DFHelper:fill_tensor:  axis 2 tensor indexing vector has " << t2.size() << " elements!";
        throw PSIEXCEPTION(error.str().c_str());
    }

    // be pythonic - adjust stops
    size_t sta0 = t0[0];
    size_t sto0 = t0[1] - 1;
    size_t sta1 = t1[0];
    size_t sto1 = t1[1] - 1;
    size_t sta2 = t2[0];
    size_t sto2 = t2[1] - 1;

    std::pair<size_t, size_t> i0 = std::make_pair(sta0, sto0);
    std::pair<size_t, size_t> i1 = std::make_pair(sta1, sto1);
    std::pair<size_t, size_t> i2 = std::make_pair(sta2, sto2);

    check_file_key(name);
    check_file_tuple(name, i0, i1, i2);

    size_t A0 = (sto0 - sta0 + 1);
    size_t A1 = (sto1 - sta1 + 1);
    size_t A2 = (sto2 - sta2 + 1);

    auto M = std::make_shared<Matrix>("M", A0, A1 * A2);
    double* Mp = M->pointer()[0];

    if (MO_core_) {
        size_t a0 = std::get<0>(sizes);
        size_t a1 = std::get<1>(sizes);
        size_t a2 = std::get<2>(sizes);

        double* Fp = transf_core_[name].get();
#pragma omp parallel for num_threads(nthreads_)
        for (size_t i = 0; i < A0; i++) {
            for (size_t j = 0; j < A1; j++) {
#pragma omp simd
                for (size_t k = 0; k < A2; k++) {
                    Mp[i * A1 * A2 + j * A2 + k] = Fp[(sta0 + i) * a1 * a2 + (sta1 + j) * a2 + (sta2 + k)];
                }
            }
        }
    } else {
        get_tensor_(filename, Mp, i0, i1, i2);
    }
    M->set_numpy_shape({(int)A0, (int)A1, (int)A2});
    return M;
}

// Add a disk tensor
void DFHelper::add_disk_tensor(std::string key, std::tuple<size_t, size_t, size_t> dimensions) {
    if (files_.count(key)) {
        std::stringstream error;
        error << "DFHelper:add_disk_tensor:  tensor already exists: (" << key << "!";
        throw PSIEXCEPTION(error.str().c_str());
    }

    filename_maker(key, std::get<0>(dimensions), std::get<1>(dimensions), std::get<2>(dimensions));
}

// Write to a disk tensor from Sharedmatrix
void DFHelper::write_disk_tensor(std::string key, SharedMatrix M) {
    check_file_key(key);
    std::string filename = std::get<1>(files_[key]);
    std::tuple<size_t, size_t, size_t> sizes;
    sizes = (tsizes_.find(filename) != tsizes_.end() ? tsizes_[filename] : sizes_[filename]);
    write_disk_tensor(key, M, {0, std::get<0>(sizes)}, {0, std::get<1>(sizes)}, {0, std::get<2>(sizes)});
}
void DFHelper::write_disk_tensor(std::string key, SharedMatrix M, std::vector<size_t> a1) {
    check_file_key(key);
    std::string filename = std::get<1>(files_[key]);
    std::tuple<size_t, size_t, size_t> sizes;
    sizes = (tsizes_.find(filename) != tsizes_.end() ? tsizes_[filename] : sizes_[filename]);
    write_disk_tensor(key, M, a1, {0, std::get<1>(sizes)}, {0, std::get<2>(sizes)});
}
void DFHelper::write_disk_tensor(std::string key, SharedMatrix M, std::vector<size_t> a1, std::vector<size_t> a2) {
    check_file_key(key);
    std::string filename = std::get<1>(files_[key]);
    std::tuple<size_t, size_t, size_t> sizes;
    sizes = (tsizes_.find(filename) != tsizes_.end() ? tsizes_[filename] : sizes_[filename]);
    write_disk_tensor(key, M, a1, a2, {0, std::get<2>(sizes)});
}
void DFHelper::write_disk_tensor(std::string key, SharedMatrix M, std::vector<size_t> a0, std::vector<size_t> a1,
                                 std::vector<size_t> a2) {
    // being pythonic ;)
    std::pair<size_t, size_t> i0 = std::make_pair(a0[0], a0[1] - 1);
    std::pair<size_t, size_t> i1 = std::make_pair(a1[0], a1[1] - 1);
    std::pair<size_t, size_t> i2 = std::make_pair(a2[0], a2[1] - 1);

    check_file_key(key);
    check_file_tuple(key, i0, i1, i2);
    check_matrix_size(key, M, i0, i1, i2);

    // "wb" is the way to go. the stream will change when when the tensor is read,
    // but this should be extendible to back-and-forth read/writes.
    std::string op = "wb";
    put_tensor(std::get<1>(files_[key]), M->pointer()[0], i0, i1, i2, op);
}

// Write to a disk tensor from pointer, be careful!
void DFHelper::write_disk_tensor(std::string key, double* b) {
    check_file_key(key);
    std::string filename = std::get<1>(files_[key]);
    std::tuple<size_t, size_t, size_t> sizes;
    sizes = (tsizes_.find(filename) != tsizes_.end() ? tsizes_[filename] : sizes_[filename]);
    write_disk_tensor(key, b, {0, std::get<0>(sizes)}, {0, std::get<1>(sizes)}, {0, std::get<2>(sizes)});
}
void DFHelper::write_disk_tensor(std::string key, double* b, std::vector<size_t> a0) {
    check_file_key(key);
    std::string filename = std::get<1>(files_[key]);
    std::tuple<size_t, size_t, size_t> sizes;
    sizes = (tsizes_.find(filename) != tsizes_.end() ? tsizes_[filename] : sizes_[filename]);
    write_disk_tensor(key, b, a0, {0, std::get<1>(sizes)}, {0, std::get<2>(sizes)});
}
void DFHelper::write_disk_tensor(std::string key, double* b, std::vector<size_t> a0, std::vector<size_t> a1) {
    check_file_key(key);
    std::string filename = std::get<1>(files_[key]);
    std::tuple<size_t, size_t, size_t> sizes;
    sizes = (tsizes_.find(filename) != tsizes_.end() ? tsizes_[filename] : sizes_[filename]);
    write_disk_tensor(key, b, a0, a1, {0, std::get<2>(sizes)});
}
void DFHelper::write_disk_tensor(std::string key, double* b, std::vector<size_t> a0, std::vector<size_t> a1,
                                 std::vector<size_t> a2) {
    // being pythonic ;)
    std::pair<size_t, size_t> i0 = std::make_pair(a0[0], a0[1] - 1);
    std::pair<size_t, size_t> i1 = std::make_pair(a1[0], a1[1] - 1);
    std::pair<size_t, size_t> i2 = std::make_pair(a2[0], a2[1] - 1);

    check_file_key(key);
    check_file_tuple(key, i0, i1, i2);

    // "wb" is the way to go. the stream will change when when the tensor is read,
    // but this should be extendible to back-and-forth read/writes.
    std::string op = "wb";
    put_tensor(std::get<1>(files_[key]), b, i0, i1, i2, op);
}

void DFHelper::check_file_key(std::string name) {
    if (files_.find(name) == files_.end()) {
        std::stringstream error;
        error << "DFHelper:get_tensor OR write_tensor: " << name << " not found.";
        throw PSIEXCEPTION(error.str().c_str());
    }
}
void DFHelper::check_matrix_size(std::string name, SharedMatrix M, std::pair<size_t, size_t> t0,
                                 std::pair<size_t, size_t> t1, std::pair<size_t, size_t> t2) {
    size_t A0 = std::get<1>(t0) - std::get<0>(t0) + 1;
    size_t A1 = (std::get<1>(t1) - std::get<0>(t1) + 1) * (std::get<1>(t2) - std::get<0>(t2) + 1);

    size_t a0 = M->rowspi()[0];
    size_t a1 = M->colspi()[0];

    if (A0 * A1 > a0 * a1) {
        std::stringstream error;
        error << "DFHelper:get_tensor: your matrix contridicts your tuple sizes when obtaining the (" << name
              << ") integral.  ";
        error << "you gave me a matrix of size: (" << a0 << "," << a1 << "), but tuple sizes give:(" << A0 << "," << A1
              << ")";
        throw PSIEXCEPTION(error.str().c_str());
    }
}
void DFHelper::check_file_tuple(std::string name, std::pair<size_t, size_t> t0, std::pair<size_t, size_t> t1,
                                std::pair<size_t, size_t> t2) {
    size_t sta0 = std::get<0>(t0);
    size_t sto0 = std::get<1>(t0);
    size_t sta1 = std::get<0>(t1);
    size_t sto1 = std::get<1>(t1);
    size_t sta2 = std::get<0>(t2);
    size_t sto2 = std::get<1>(t2);
    std::string filename = std::get<1>(files_[name]);

    // has this integral been transposed?
    std::tuple<size_t, size_t, size_t> sizes;
    sizes = (tsizes_.find(filename) != tsizes_.end() ? tsizes_[filename] : sizes_[filename]);

    if (sta0 > sto0) {
        std::stringstream error;
        error << "when getting integral: (" << name << ")"
              << " your axis 0 tuple has a larger start index: " << sta0 << " than its stop index: " << sto0;
        throw PSIEXCEPTION(error.str().c_str());
    }
    if (sta1 > sto1) {
        std::stringstream error;
        error << "when getting integral: (" << name << ")"
              << " your axis 1 tuple has a larger start index: " << sta1 << " than its stop index: " << sto1;
        throw PSIEXCEPTION(error.str().c_str());
    }
    if (sta2 > sto2) {
        std::stringstream error;
        error << "when getting integral: (" << name << ")"
              << " your axis 2 tuple has a larger start index: " << sta2 << " than its stop index: " << sto2;
        throw PSIEXCEPTION(error.str().c_str());
    }
    size_t M0 = std::get<0>(sizes);
    if (sto0 > M0 - 1) {
        std::stringstream error;
        error << "your axis 0 tuple goes out of bounds when getting integral: " << name;
        error << ". you entered (" << sto0 << "), but bounds is (" << M0 - 1 << ").";
        throw PSIEXCEPTION(error.str().c_str());
    }
    size_t M1 = std::get<1>(sizes);
    if (sto1 > M1 - 1) {
        std::stringstream error;
        error << "your axis 1 tuple goes out of bounds when getting integral: " << name;
        error << ". you entered (" << sto1 << "), but bounds is (" << M1 - 1 << ").";
        throw PSIEXCEPTION(error.str().c_str());
    }
    size_t M2 = std::get<2>(sizes);
    if (sto2 > M2 - 1) {
        std::stringstream error;
        error << "your axis 2 tuple goes out of bounds when getting integral: " << name;
        error << ". you entered (" << sto2 << "), but bounds is (" << M2 - 1 << ").";
        throw PSIEXCEPTION(error.str().c_str());
    }
}
void DFHelper::transpose(std::string name, std::tuple<size_t, size_t, size_t> order) {
    if (!files_.count(name)) {
        std::stringstream error;
        error << "DFHelper::transpose(): cannot transpose input (" << name << "), name doe not exist!";
        throw PSIEXCEPTION(error.str().c_str());
    }

    (MO_core_ ? transpose_core(name, order) : transpose_disk(name, order));
}
void DFHelper::transpose_core(std::string name, std::tuple<size_t, size_t, size_t> order) {
    size_t a0 = std::get<0>(order);
    size_t a1 = std::get<1>(order);
    size_t a2 = std::get<2>(order);

    std::string filename = std::get<1>(files_[name]);
    size_t M0 = std::get<0>(sizes_[filename]);
    size_t M1 = std::get<1>(sizes_[filename]);
    size_t M2 = std::get<2>(sizes_[filename]);
    std::tuple<size_t, size_t, size_t> sizes;

    std::unique_ptr<double[]> M(new double[M0 * M1 * M2]);
    double* Mp = M.get();
    double* Fp = transf_core_[name].get();
    C_DCOPY(M0 * M1 * M2, Fp, 1, Mp, 1);

    bool on = false;
    if (a0 == 0) {
        if (a1 == 2) {
            sizes = std::make_tuple(M0, M2, M1);
            on = true;
        }
    } else if (a0 == 1) {
        if (a1 == 0) {
            sizes = std::make_tuple(M1, M0, M2);
            on = true;
        } else if (a1 == 2) {
            sizes = std::make_tuple(M1, M2, M0);
            on = true;
        }
    } else {
        if (a1 == 0) {
            sizes = std::make_tuple(M2, M0, M1);
            on = true;
        } else if (a1 == 1) {
            sizes = std::make_tuple(M2, M1, M0);
            on = true;
        }
    }

    if (!on) throw PSIEXCEPTION("you transposed all wrong!");

    if (a0 == 0) {
        if (a1 == 2) {  // (0|12) -> (0|21)
#pragma omp parallel for num_threads(nthreads_)
            for (size_t i = 0; i < M0; i++) {
                for (size_t j = 0; j < M1; j++) {
                    for (size_t k = 0; k < M2; k++) {
                        Fp[i * M1 * M2 + k * M1 + j] = Mp[i * M1 * M2 + j * M2 + k];
                    }
                }
            }
        }
    } else if (a0 == 1) {
        if (a1 == 0) {  // (0|12) -> (1|02)
#pragma omp parallel for num_threads(nthreads_)
            for (size_t i = 0; i < M0; i++) {
                for (size_t j = 0; j < M1; j++) {
#pragma omp simd
                    for (size_t k = 0; k < M2; k++) {
                        Fp[j * M0 * M2 + i * M2 + k] = Mp[i * M1 * M2 + j * M2 + k];
                    }
                }
            }
        } else if (a1 == 2) {  // (0|12) -> (1|20)
#pragma omp parallel for num_threads(nthreads_)
            for (size_t i = 0; i < M0; i++) {
                for (size_t j = 0; j < M1; j++) {
                    for (size_t k = 0; k < M2; k++) {
                        Fp[j * M0 * M2 + k * M0 + i] = Mp[i * M1 * M2 + j * M2 + k];
                    }
                }
            }
        }
    } else if (a0 == 2) {
        if (a1 == 0) {  // (0|12) -> (2|01)
#pragma omp parallel for num_threads(nthreads_)
            for (size_t i = 0; i < M0; i++) {
                for (size_t j = 0; j < M1; j++) {
                    for (size_t k = 0; k < M2; k++) {
                        Fp[k * M1 * M0 + i * M1 + j] = Mp[i * M1 * M2 + j * M2 + k];
                    }
                }
            }
        } else if (a1 == 1) {  // (0|12) -> (2|10)
#pragma omp parallel for num_threads(nthreads_)
            for (size_t i = 0; i < M0; i++) {
                for (size_t j = 0; j < M1; j++) {
                    for (size_t k = 0; k < M2; k++) {
                        Fp[k * M1 * M0 + j * M0 + i] = Mp[i * M1 * M2 + j * M2 + k];
                    }
                }
            }
        }
    }
    // keep tsizes_ separate and do not ovwrt sizes_ in case of STORE directive
    tsizes_[filename] = sizes;
}
void DFHelper::transpose_disk(std::string name, std::tuple<size_t, size_t, size_t> order) {
    size_t a0 = std::get<0>(order);
    size_t a1 = std::get<1>(order);
    size_t a2 = std::get<2>(order);

    // determine blocking
    std::string filename = std::get<1>(files_[name]);
    size_t M0 = std::get<0>(sizes_[filename]);
    size_t M1 = std::get<1>(sizes_[filename]);
    size_t M2 = std::get<2>(sizes_[filename]);

    size_t current = 0, count = 0, largest = 0;
    std::vector<std::pair<size_t, size_t>> steps;
    for (size_t i = 0; i < M0; i++) {
        current += M1 * M2;
        count++;
        if ((current * 2 > memory_) || (i == M0 - 1)) {  //
            if (count == 1 && i != M0 - 1) {
                std::stringstream error;
                error << "DFHelper:transpose_disk: not enough memory.";
                throw PSIEXCEPTION(error.str().c_str());
            }
            if (i == M0 - 1)
                steps.push_back(std::make_pair(i - count + 1, i));
            else {
                current -= M1 * M2;
                steps.push_back(std::make_pair(i - count + 1, i - 1));
                i--;
            }
            if (largest < current) largest = current;
            count = 0;
            current = 0;
        }
    }

    // declare
    std::unique_ptr<double[]> M(new double[largest]);
    std::unique_ptr<double[]> F(new double[largest]);
    double* Mp = M.get();
    double* Fp = F.get();
    std::tuple<size_t, size_t, size_t> sizes;

    bool on = false;
    if (a0 == 0) {
        if (a1 == 2) {
            sizes = std::make_tuple(M0, M2, M1);
            on = true;
        }
    } else if (a0 == 1) {
        if (a1 == 0) {
            sizes = std::make_tuple(M1, M0, M2);
            on = true;
        } else if (a1 == 2) {
            sizes = std::make_tuple(M1, M2, M0);
            on = true;
        }
    } else {
        if (a1 == 0) {
            sizes = std::make_tuple(M2, M0, M1);
            on = true;
        } else if (a1 == 1) {
            sizes = std::make_tuple(M2, M1, M0);
            on = true;
        }
    }

    if (!on) throw PSIEXCEPTION("you transposed all wrong!");

    std::string new_file = "newfilefortransposition";
    filename_maker(new_file, std::get<0>(sizes), std::get<1>(sizes), std::get<2>(sizes));
    std::string new_filename = std::get<1>(files_[new_file]);

    for (size_t m = 0; m < steps.size(); m++) {
        std::string op = (m ? "r+b" : "wb");
        size_t start = std::get<0>(steps[m]);
        size_t stop = std::get<1>(steps[m]);
        M0 = stop - start + 1;

        // grab
        get_tensor_(filename, Mp, start, stop, 0, M1 * M2 - 1);

        if (a0 == 0) {
            if (a1 == 2) {  // (0|12) -> (0|21)
#pragma omp parallel for num_threads(nthreads_)
                for (size_t i = 0; i < M0; i++) {
                    for (size_t j = 0; j < M1; j++) {
                        for (size_t k = 0; k < M2; k++) {
                            Fp[i * M1 * M2 + k * M1 + j] = Mp[i * M1 * M2 + j * M2 + k];
                        }
                    }
                }
                put_tensor(new_filename, Fp, std::make_pair(start, stop), std::make_pair(0, M2 - 1),
                           std::make_pair(0, M1 - 1), op);
            }
        } else if (a0 == 1) {
            if (a1 == 0) {  // (0|12) -> (1|02)
#pragma omp parallel for num_threads(nthreads_)
                for (size_t i = 0; i < M0; i++) {
                    for (size_t j = 0; j < M1; j++) {
#pragma omp simd
                        for (size_t k = 0; k < M2; k++) {
                            Fp[j * M0 * M2 + i * M2 + k] = Mp[i * M1 * M2 + j * M2 + k];
                        }
                    }
                }
                put_tensor(new_filename, Fp, std::make_pair(0, M1 - 1), std::make_pair(start, stop),
                           std::make_pair(0, M2 - 1), op);
            } else if (a1 == 2) {  // (0|12) -> (1|20)
#pragma omp parallel for num_threads(nthreads_)
                for (size_t i = 0; i < M0; i++) {
                    for (size_t j = 0; j < M1; j++) {
                        for (size_t k = 0; k < M2; k++) {
                            Fp[j * M0 * M2 + k * M0 + i] = Mp[i * M1 * M2 + j * M2 + k];
                        }
                    }
                }
                put_tensor(new_filename, Fp, std::make_pair(0, M1 - 1), std::make_pair(0, M2 - 1),
                           std::make_pair(start, stop), op);
            }
        } else if (a0 == 2) {
            if (a1 == 0) {  // (0|12) -> (2|01)
#pragma omp parallel for num_threads(nthreads_)
                for (size_t i = 0; i < M0; i++) {
                    for (size_t j = 0; j < M1; j++) {
                        for (size_t k = 0; k < M2; k++) {
                            Fp[k * M1 * M0 + i * M1 + j] = Mp[i * M1 * M2 + j * M2 + k];
                        }
                    }
                }
                put_tensor(new_filename, Fp, std::make_pair(0, M2 - 1), std::make_pair(start, stop),
                           std::make_pair(0, M1 - 1), op);
            } else if (a1 == 1) {  // (0|12) -> (2|10)
#pragma omp parallel for num_threads(nthreads_)
                for (size_t i = 0; i < M0; i++) {
                    for (size_t j = 0; j < M1; j++) {
                        for (size_t k = 0; k < M2; k++) {
                            Fp[k * M1 * M0 + j * M0 + i] = Mp[i * M1 * M2 + j * M2 + k];
                        }
                    }
                }
                put_tensor(new_filename, Fp, std::make_pair(0, M2 - 1), std::make_pair(0, M1 - 1),
                           std::make_pair(start, stop), op);
            }
        }
    }
    // better be careful
    remove(filename.c_str());
    rename(new_filename.c_str(), filename.c_str());
    file_streams_[filename] = file_streams_[new_filename];
    stream_check(filename, "rb");
    file_streams_.erase(new_filename);

    // keep tsizes_ separate and do not ovwrt sizes_ in case of STORE directive
    files_.erase(new_file);
    tsizes_[filename] = sizes;
}
size_t DFHelper::get_space_size(std::string name) {
    if (spaces_.find(name) == spaces_.end()) {
        std::stringstream error;
        error << "DFHelper:get_space_size: " << name << " not found.";
        throw PSIEXCEPTION(error.str().c_str());
    }
    return std::get<1>(spaces_[name]);
}
size_t DFHelper::get_tensor_size(std::string name) {
    if (transf_.find(name) == transf_.end()) {
        std::stringstream error;
        error << "DFHelper:get_tensor_size: " << name << " not found.";
        throw PSIEXCEPTION(error.str().c_str());
    }
    std::tuple<size_t, size_t, size_t> s = sizes_[std::get<1>(files_[name])];
    return std::get<0>(s) * std::get<1>(s) * std::get<2>(s);
}
std::tuple<size_t, size_t, size_t> DFHelper::get_tensor_shape(std::string name) {
    if (transf_.find(name) == transf_.end()) {
        std::stringstream error;
        error << "DFHelper:get_tensor_size: " << name << " not found.";
        throw PSIEXCEPTION(error.str().c_str());
    }
    return sizes_[std::get<1>(files_[name])];
}
void DFHelper::build_JK(std::vector<SharedMatrix> Cleft, std::vector<SharedMatrix> Cright, std::vector<SharedMatrix> D,
                        std::vector<SharedMatrix> J, std::vector<SharedMatrix> K, std::vector<SharedMatrix> wK,
                        size_t max_nocc, bool do_J, bool do_K, bool do_wK, bool lr_symmetric) {
    if (debug_) {
        outfile->Printf("Entering DFHelper::build_JK\n");
    }
    bool store_k = do_K;
    if ( do_wK && wcombine_ ) { do_K = false; } 

    // This was an if-else statement. Presumably, we could manage J construction
    //   to more effectively manage memory, so I think that was what was going on.
    if (do_J || do_K) {
        timer_on("DFH: compute_JK()");
        compute_JK(Cleft, Cright, D, J, K, max_nocc, do_J, do_K, do_wK, lr_symmetric);
        timer_off("DFH: compute_JK()");
    }

    if (do_wK_) {
        timer_on("DFH: compute_wK()");
        compute_wK(Cleft, Cright, wK, max_nocc, do_J, do_K, do_wK);
        timer_off("DFH: compute_wK()");
    }

    if (store_k) { do_K = true;}

    if (debug_) {
        outfile->Printf("Exiting DFHelper::build_JK\n");
    }
}
void DFHelper::ensure_jk_C_buffers(size_t elem_per_thread) {
    if (jk_C_buffers_.size() != nthreads_) {
        jk_C_buffers_.resize(nthreads_);
    }
    bool need_grow = false;
    for (size_t r = 0; r < nthreads_; ++r) {
        if (jk_C_buffers_[r].size() < elem_per_thread) {
            need_grow = true;
            break;
        }
    }
    if (need_grow) {
#pragma omp parallel for num_threads(nthreads_) schedule(static)
        for (size_t rank = 0; rank < nthreads_; ++rank) {
            if (jk_C_buffers_[rank].size() < elem_per_thread) {
                // As with grow_scratch, release before allocating and avoid
                // vector's geometric capacity growth exceeding the planned size.
                jk_C_buffers_[rank].resize(elem_per_thread);
            }
        }
    }
}

namespace {
template <typename T>
void symm_k_final_accumulate(size_t N, size_t K_inner, const T* T1, double* Kp,
                             std::vector<DFHelperScratch>& C_buffers, size_t nthreads) {
    if (N == 0 || K_inner == 0) return;
    const size_t active_nt = std::min(nthreads, std::max<size_t>(1, K_inner / 64));
    // Reduction threads read every active buffer. Start their scalar array
    // lifetimes serially, including buffers unused by the first transform.
    for (size_t t = 0; t < active_nt; ++t) C_buffers[t].template data_as<T>();

    auto run_syrk = [](int n, int k, const T* a, int lda, T* c, int ldc) {
        if constexpr (sizeof(T) == 8) {
            C_DSYRK('U', 'N', n, k, 1.0, const_cast<double*>(reinterpret_cast<const double*>(a)), lda, 0.0,
                    reinterpret_cast<double*>(c), ldc);
        } else {
            C_SSYRK('U', 'N', n, k, 1.0f, const_cast<float*>(reinterpret_cast<const float*>(a)), lda, 0.0f,
                    reinterpret_cast<float*>(c), ldc);
        }
    };
    auto run_gemm = [](int m, int n, int k, const T* a, int lda, const T* b, int ldb, T* c, int ldc) {
        if constexpr (sizeof(T) == 8) {
            C_DGEMM('N', 'T', m, n, k, 1.0, const_cast<double*>(reinterpret_cast<const double*>(a)), lda,
                    const_cast<double*>(reinterpret_cast<const double*>(b)), ldb, 0.0,
                    reinterpret_cast<double*>(c), ldc);
        } else {
            C_SGEMM('N', 'T', m, n, k, 1.0f, const_cast<float*>(reinterpret_cast<const float*>(a)), lda,
                    const_cast<float*>(reinterpret_cast<const float*>(b)), ldb, 0.0f,
                    reinterpret_cast<float*>(c), ldc);
        }
    };

    if (active_nt == 1) {
        T* Cb = C_buffers[0].template data_as<T>();
        if (N <= 512) {
            run_syrk(static_cast<int>(N), static_cast<int>(K_inner), T1, static_cast<int>(K_inner), Cb,
                     static_cast<int>(N));
        } else {
            constexpr size_t B = 192;
            for (size_t i0 = 0; i0 < N; i0 += B) {
                const size_t ib = std::min(B, N - i0);
                run_syrk(static_cast<int>(ib), static_cast<int>(K_inner), T1 + i0 * K_inner,
                         static_cast<int>(K_inner), Cb + i0 * N + i0, static_cast<int>(N));
                if (i0 + ib < N) {
                    run_gemm(static_cast<int>(ib), static_cast<int>(N - (i0 + ib)), static_cast<int>(K_inner),
                             T1 + i0 * K_inner, static_cast<int>(K_inner), T1 + (i0 + ib) * K_inner,
                             static_cast<int>(K_inner), Cb + i0 * N + (i0 + ib), static_cast<int>(N));
                }
            }
        }
        for (size_t i = 0; i < N; ++i) {
            Kp[i * N + i] += static_cast<double>(Cb[i * N + i]);
            for (size_t j = i + 1; j < N; ++j) {
                const double v = static_cast<double>(Cb[i * N + j]);
                Kp[i * N + j] += v;
                Kp[j * N + i] += v;
            }
        }
        return;
    }

#pragma omp parallel num_threads(nthreads)
    {
        int tid = 0;
        size_t team_active = 1;
#ifdef _OPENMP
        tid = omp_get_thread_num();
        team_active = std::min(active_nt, static_cast<size_t>(omp_get_num_threads()));
#endif
        if (static_cast<size_t>(tid) < team_active) {
#ifdef USING_LAPACK_MKL
            const int previous_mkl_threads = mkl_set_num_threads_local(1);
#endif
            const size_t k0 = (K_inner * static_cast<size_t>(tid)) / team_active;
            const size_t kn = (K_inner * (static_cast<size_t>(tid) + 1)) / team_active - k0;
            T* Cb = C_buffers[tid].template data_as<T>();
            if (N <= 512) {
                run_syrk(static_cast<int>(N), static_cast<int>(kn), T1 + k0, static_cast<int>(K_inner), Cb,
                         static_cast<int>(N));
            } else {
                constexpr size_t B = 192;
                for (size_t i0 = 0; i0 < N; i0 += B) {
                    const size_t ib = std::min(B, N - i0);
                    run_syrk(static_cast<int>(ib), static_cast<int>(kn), T1 + i0 * K_inner + k0,
                             static_cast<int>(K_inner), Cb + i0 * N + i0, static_cast<int>(N));
                    if (i0 + ib < N) {
                        run_gemm(static_cast<int>(ib), static_cast<int>(N - (i0 + ib)), static_cast<int>(kn),
                                 T1 + i0 * K_inner + k0, static_cast<int>(K_inner),
                                 T1 + (i0 + ib) * K_inner + k0, static_cast<int>(K_inner),
                                 Cb + i0 * N + (i0 + ib), static_cast<int>(N));
                    }
                }
            }
#ifdef USING_LAPACK_MKL
            mkl_set_num_threads_local(previous_mkl_threads);
#endif
        }
#pragma omp barrier
#pragma omp for schedule(dynamic, 16)
        for (size_t i = 0; i < N; ++i) {
            double* dst = Kp + i * N;
            const T* s0 = C_buffers[0].template data_as<T>() + i * N;
#pragma omp simd
            for (size_t j = i; j < N; ++j) {
                dst[j] += static_cast<double>(s0[j]);
            }
            for (size_t t = 1; t < team_active; ++t) {
                const T* st = C_buffers[t].template data_as<T>() + i * N;
#pragma omp simd
                for (size_t j = i; j < N; ++j) {
                    dst[j] += static_cast<double>(st[j]);
                }
            }
        }
#pragma omp for schedule(static)
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < i; ++j) {
                Kp[i * N + j] = Kp[j * N + i];
            }
        }
    }
}
}  // namespace

void DFHelper::compute_JK(std::vector<SharedMatrix> Cleft, std::vector<SharedMatrix> Cright,
                          std::vector<SharedMatrix> D, std::vector<SharedMatrix> J, std::vector<SharedMatrix> K,
                          size_t max_nocc, bool do_J, bool do_K, bool do_wK, bool lr_symmetric) {
    // Each element of Qsteps specifies the endpoints of a batch of auxiliary shells.
    // We'll treat all (PN|Q) for Q in this batch at once.
    // (i,Q) K intermediates hold nocc padded to a multiple of 8 (see first_transform_pQq_K).
    if (K_layout_iQ_ || K_fp32_) max_nocc = (max_nocc + 7) / 8 * 8;
    std::vector<std::pair<size_t, size_t>> Qsteps;
    // Separate wK needs two transform buffers later in this same build. Reserve
    // both now so it cannot grow T2 underneath an admitted FP32 AO mirror.
    const bool one_transform = lr_symmetric && !do_wK_;
    std::tuple<size_t, size_t> info = Qshell_blocks_for_JK_build(Qsteps, max_nocc, one_transform);
    size_t tots = std::get<0>(info);
    size_t totsb = std::get<1>(info);

    // prep stream, blocking
    if (!direct_ && !AO_core_) stream_check(AO_names_[1], "rb");

    ensure_jk_C_buffers(nbf_ * std::max(max_nocc, nbf_));
    auto& C_buffers = jk_C_buffers_;

    // declare bufs
    std::unique_ptr<double[]> M;   // AOs

    // allocate first Ktmp
    size_t Ktmp_size = (!max_nocc ? totsb * 1 : totsb * max_nocc);
    Ktmp_size = std::max(Ktmp_size * nbf_, nthreads_ * naux_);  // max necessary
    Ktmp_size = std::max(Ktmp_size, nbf_ * nbf_);
    // Scratch persists across builds: a fresh ~GiB buffer per SCF iteration is mostly page faults.
    double* T1p = grow_scratch(jk_T1_, jk_T1_size_, Ktmp_size);

    // if lr_symmetric, we can be more clever with mem usage. T2 is used for both the
    // second tmp in the K build, as well as the completed, pruned J build.
    if (one_transform) {
        Ktmp_size = nbf_ * nbf_;  // size for pruned J build
    } else {
        Ktmp_size = std::max(nbf_ * nbf_, Ktmp_size);  // max necessary
    }

    // necessary for wcombine case for small molecules
    Ktmp_size = std::max(Ktmp_size, nbf_ * nbf_);
    Ktmp_size = std::max(Ktmp_size, nthreads_ * naux_);

    double* T2p = grow_scratch(jk_T2_, jk_T2_size_, Ktmp_size);

    double* Mp = nullptr;
    if (!AO_core_) {
        M = std::make_unique<double[]>(tots);
        Mp = M.get();
    } else if (!wcombine_) {
        Mp = Ppq_.get();
    }

    double* M1p = nullptr;
    if (wcombine_) {
        M1p = m1Ppq_.get();
    }

    // Low-precision builds read Ppq_f_ when it fits in memory, or tile Mp in L2 cache with zero extra RAM.
    const bool use_fp32 = K_fp32_ && (do_J || do_K) && prepare_K_fp32();

    // Transform a single batch of integrals
    size_t bcount = 0;
    for (const auto& Qstep : Qsteps) {
        // Qshell step info
        auto start = std::get<0>(Qstep);
        auto stop = std::get<1>(Qstep);
        auto begin = Qshell_aggs_[start];
        auto end = Qshell_aggs_[stop + 1] - 1;
        auto block_size = end - begin + 1;

        // get AO chunk according to directive
        timer_on("DFH: Grabbing AOs");
        if (!AO_core_) {
            grab_AO(start, stop, Mp);
        }
        timer_off("DFH: Grabbing AOs");
        // A preceding FP32 block ended the double arrays' lifetimes. Reacquire
        // typed pointers before the J contraction writes double scratch.
        T1p = jk_T1_.data();
        T2p = jk_T2_.data();
        if (do_J) {
            timer_on("DFH: compute_J");
            if (wcombine_ && do_wK_) {
                compute_J_combined(D, J, M1p, T1p, T2p, C_buffers, bcount, block_size);
            } else {
                if (lr_symmetric && use_fp32 && Ppq_f_) {
                    compute_J_symm_fp32(D, J, T1p, T2p, C_buffers, bcount, block_size);
                } else if (lr_symmetric) {
                    compute_J_symm(D, J, Mp, T1p, T2p, C_buffers, bcount, block_size);
                } else {
                    compute_J(D, J, Mp, T1p, T2p, C_buffers, bcount, block_size);
                }
            }
            timer_off("DFH: compute_J");
        }

        if (do_K) {
            timer_on("DFH: compute_K");
            if (use_fp32) {
                compute_K_fp32(Cleft, Cright, K, T1p, T2p, Mp, bcount, block_size, C_buffers, lr_symmetric);
            } else {
                compute_K(Cleft, Cright, K, T1p, T2p, Mp, bcount, block_size, C_buffers, lr_symmetric);
            }
            timer_off("DFH: compute_K");
        }

        bcount += block_size;
    }
}
void DFHelper::compute_J_symm(std::vector<SharedMatrix> D, std::vector<SharedMatrix> J, double* Mp, double* T1p,
                              double* T2p, std::vector<DFHelperScratch>& D_buffers, size_t bcount,
                              size_t block_size) {
    for (size_t i = 0; i < J.size(); i++) {
        double* Dp = D[i]->pointer()[0];
        double* Jp = J[i]->pointer()[0];

        fill(T1p, nthreads_ * naux_, 0.0);

#pragma omp parallel for schedule(guided) num_threads(nthreads_)
        for (size_t k = 0; k < nbf_; k++) {
            size_t mi = symm_small_skips_[k];
            if (mi == 0) continue;
            size_t si = small_skips_[k];
            size_t skip = symm_ignored_columns_[k];
            size_t jump = (AO_core_ ? big_skips_[k] + bcount * si : (big_skips_[k] * block_size) / naux_);

            int rank = 0;
#ifdef _OPENMP
            rank = omp_get_thread_num();
#endif
            double* Db = D_buffers[rank].data();
            const uint32_t* sig_m = &sig_q_[sig_q_row_off_[k] + skip];
            const double* Drow = &Dp[nbf_ * k];
            for (size_t c = 0; c < mi; ++c) {
                const size_t m = sig_m[c];
                Db[c] = (m == k ? Drow[m] : 2.0 * Drow[m]);
            }

            // (Qm)(m) -> (Q)
            C_DGEMV('N', block_size, mi, 1.0, &Mp[jump + skip], si, Db, 1, 1.0, &T1p[rank * naux_], 1);
        }

        // reduce
        for (size_t k = 1; k < nthreads_; k++) {
            for (size_t l = 0; l < naux_; l++) T1p[l] += T1p[k * naux_ + l];
        }

// complete pruned J
#pragma omp parallel for schedule(guided) num_threads(nthreads_)
        for (size_t k = 0; k < nbf_; k++) {
            size_t mi = symm_small_skips_[k];
            if (mi == 0) continue;
            size_t si = small_skips_[k];
            size_t skip = symm_ignored_columns_[k];
            size_t jump = (AO_core_ ? big_skips_[k] + bcount * si : (big_skips_[k] * block_size) / naux_);
            C_DGEMV('T', block_size, mi, 1.0, &Mp[jump + skip], si, T1p, 1, 0.0, &T2p[k * nbf_], 1);
        }

        // unpack from sparse to dense using compact sig_q_ indices
        for (size_t k = 0; k < nbf_; k++) {
            const size_t mi = symm_small_skips_[k];
            const uint32_t* sig_m = &sig_q_[sig_q_row_off_[k] + symm_ignored_columns_[k]];
            const double* row = &T2p[k * nbf_];
            for (size_t c = 0; c < mi; ++c) {
                const size_t m = sig_m[c];
                Jp[k * nbf_ + m] += row[c];
                if (m != k) Jp[m * nbf_ + k] += row[c];
            }
        }
    }
}
void DFHelper::compute_J_symm_fp32(std::vector<SharedMatrix> D, std::vector<SharedMatrix> J, double* T1p, double* T2p,
                                   std::vector<DFHelperScratch>& D_buffers, size_t bcount, size_t block_size) {
    // compute_J_symm reading the single-precision tensor; products and sums stay in double.
    const float* Mf = Ppq_f_.get();
    for (size_t i = 0; i < J.size(); i++) {
        double* Dp = D[i]->pointer()[0];
        double* Jp = J[i]->pointer()[0];
        fill(T1p, nthreads_ * naux_, 0.0);

#pragma omp parallel for schedule(guided) num_threads(nthreads_)
        for (size_t k = 0; k < nbf_; k++) {
            size_t mi = symm_small_skips_[k];
            if (mi == 0) continue;
            size_t si = small_skips_[k];
            size_t skip = symm_ignored_columns_[k];
            size_t jump = big_skips_[k] + bcount * si;
            int rank = 0;
#ifdef _OPENMP
            rank = omp_get_thread_num();
#endif
            double* Db = D_buffers[rank].data();
            const uint32_t* sig_m = &sig_q_[sig_q_row_off_[k] + skip];
            const double* Drow = &Dp[nbf_ * k];
            for (size_t c = 0; c < mi; ++c) {
                const size_t m = sig_m[c];
                Db[c] = (m == k ? Drow[m] : 2.0 * Drow[m]);
            }
            double* T = &T1p[rank * naux_];
            for (size_t Q = 0; Q < block_size; Q++) {
                const float* row = &Mf[jump + Q * si + skip];
                double sum = 0.0;
#pragma omp simd reduction(+ : sum)
                for (size_t m = 0; m < mi; m++) sum += row[m] * Db[m];
                T[Q] += sum;
            }
        }
        for (size_t k = 1; k < nthreads_; k++) {
            for (size_t l = 0; l < naux_; l++) T1p[l] += T1p[k * naux_ + l];
        }

#pragma omp parallel for schedule(guided) num_threads(nthreads_)
        for (size_t k = 0; k < nbf_; k++) {
            size_t mi = symm_small_skips_[k];
            if (mi == 0) continue;
            size_t si = small_skips_[k];
            size_t skip = symm_ignored_columns_[k];
            size_t jump = big_skips_[k] + bcount * si;
            double* out = &T2p[k * nbf_];
            std::fill(out, out + mi, 0.0);
            for (size_t Q = 0; Q < block_size; Q++) {
                const float* row = &Mf[jump + Q * si + skip];
                const double t = T1p[Q];
#pragma omp simd
                for (size_t m = 0; m < mi; m++) out[m] += row[m] * t;
            }
        }

        for (size_t k = 0; k < nbf_; k++) {
            const size_t mi = symm_small_skips_[k];
            const uint32_t* sig_m = &sig_q_[sig_q_row_off_[k] + symm_ignored_columns_[k]];
            const double* row = &T2p[k * nbf_];
            for (size_t c = 0; c < mi; ++c) {
                const size_t m = sig_m[c];
                Jp[k * nbf_ + m] += row[c];
                if (m != k) Jp[m * nbf_ + k] += row[c];
            }
        }
    }
}
void DFHelper::fill(double* b, size_t count, double value) {
#pragma omp parallel for simd num_threads(nthreads_) schedule(static)
    for (size_t i = 0; i < count; i++) {
        b[i] = value;
    }
}
void DFHelper::compute_J(const std::vector<SharedMatrix> D, std::vector<SharedMatrix> J, double* Mp, double* T1p, double* T2p,
                         std::vector<DFHelperScratch>& D_buffers, size_t bcount, size_t block_size) {
    for (size_t i = 0; i < J.size(); i++) {
        auto Dp = D[i]->pointer()[0];
        auto Jp = J[i]->pointer()[0];

        fill(T1p, nthreads_ * naux_, 0.0);

#pragma omp parallel for schedule(guided) num_threads(nthreads_)
        for (size_t k = 0; k < nbf_; k++) {
            size_t sp_size = small_skips_[k];
            if (sp_size == 0) continue;
            size_t jump = (AO_core_ ? big_skips_[k] + bcount * sp_size : (big_skips_[k] * block_size) / naux_);

            int rank = 0;
#ifdef _OPENMP
            rank = omp_get_thread_num();
#endif
            double* Db = D_buffers[rank].data();
            const uint32_t* sig_m = &sig_q_[sig_q_row_off_[k]];
            const double* Drow = &Dp[nbf_ * k];
            for (size_t c = 0; c < sp_size; ++c) {
                Db[c] = Drow[sig_m[c]];
            }
            // (Qm)(m) -> (Q)
            C_DGEMV('N', block_size, sp_size, 1.0, &Mp[jump], sp_size, Db, 1, 1.0, &T1p[rank * naux_], 1);
        }

        // reduce
        for (size_t k = 1; k < nthreads_; k++) {
            for (size_t l = 0; l < naux_; l++) T1p[l] += T1p[k * naux_ + l];
        }

// complete pruned J
#pragma omp parallel for schedule(guided) num_threads(nthreads_)
        for (size_t k = 0; k < nbf_; k++) {
            size_t sp_size = small_skips_[k];
            if (sp_size == 0) continue;
            size_t jump = (AO_core_ ? big_skips_[k] + bcount * sp_size : (big_skips_[k] * block_size) / naux_);
            C_DGEMV('T', block_size, sp_size, 1.0, &Mp[jump], sp_size, T1p, 1, 0.0, &T2p[k * nbf_], 1);
        }

        // unpack from sparse to dense
#pragma omp parallel for schedule(static) num_threads(nthreads_) if (nbf_ >= 256)
        for (size_t k = 0; k < nbf_; k++) {
            const size_t sp_size = small_skips_[k];
            const uint32_t* sig_m = &sig_q_[sig_q_row_off_[k]];
            const double* row = &T2p[k * nbf_];
            double* Jrow = &Jp[k * nbf_];
            for (size_t c = 0; c < sp_size; ++c) {
                Jrow[sig_m[c]] += row[c];
            }
        }
    }
}
void DFHelper::compute_J_combined(std::vector<SharedMatrix> D, std::vector<SharedMatrix> J, double* Mp, double* T1p, double* T2p, std::vector<DFHelperScratch>& D_buffers, size_t bcount, size_t block_size) {
    double* coul_metp;
    std::unique_ptr<double[]> metric;

    if (!hold_met_) {
        metric = std::make_unique<double[]>(naux_ * naux_);
        coul_metp = metric.get();
        std::string filename = return_metfile(1.0);
        get_tensor_(std::get<0>(files_[filename]), coul_metp, 0, naux_ - 1, 0, naux_ - 1);
    } else
        coul_metp = metric_prep_core(1.0);

    for (size_t i = 0; i < J.size(); i++) {
        double* Dp = D[i]->pointer()[0];
        double* Jp = J[i]->pointer()[0];
        memset(T1p, 0, naux_ * nthreads_ * sizeof(double));
        memset(T2p, 0, naux_ * sizeof(double));

#pragma omp parallel for schedule(guided) num_threads(nthreads_)
        for (size_t k = 0; k < nbf_; k++) {
            size_t sp_size = small_skips_[k];
            if (sp_size == 0) continue;
            size_t jump = (AO_core_ ? big_skips_[k] + bcount * sp_size : (big_skips_[k] * block_size) / naux_);
            int rank = 0;
#ifdef _OPENMP
            rank = omp_get_thread_num();
#endif
            const uint32_t* sig_m = &sig_q_[sig_q_row_off_[k]];
            double* Db = D_buffers[rank].data();
            for (size_t c = 0; c < sp_size; ++c) {
                Db[c] = Dp[nbf_ * sig_m[c] + k];
            }
            C_DGEMV('N', block_size, sp_size, 1.0, &Mp[jump], sp_size, Db, 1, 1.0, &T1p[rank * naux_], 1);
        }
        // reduce
        for (size_t k = 1; k < nthreads_; k++) {
            for (size_t l = 0; l < naux_; l++) {
                T1p[l] += T1p[k * naux_ + l];
            }
        }

        // metric contraction
        C_DGEMV('N', naux_, naux_, 1.0, coul_metp, naux_, T1p, 1, 0.0, T2p, 1);

        memset(T1p, 0, nbf_ * nbf_ * sizeof(double));
        // compute pruned J
#pragma omp parallel for schedule(guided) num_threads(nthreads_)
        for (size_t k = 0; k < nbf_; k++) {
            size_t sp_size = small_skips_[k];
            if (sp_size == 0) continue;
            size_t jump = (AO_core_ ? big_skips_[k] + bcount * sp_size : (big_skips_[k] * block_size) / naux_);
            C_DGEMV('T', block_size, sp_size, 1.0, &Mp[jump], sp_size, T2p, 1, 0.0, &T1p[k * nbf_], 1);
        }

        for (size_t k = 0; k < nbf_; k++) {
            const size_t sp_size = small_skips_[k];
            const uint32_t* sig_m = &sig_q_[sig_q_row_off_[k]];
            for (size_t c = 0; c < sp_size; ++c) {
                Jp[k * nbf_ + sig_m[c]] += T1p[k * nbf_ + c];
            }
        }
    }
}
void DFHelper::compute_K(std::vector<SharedMatrix> Cleft, std::vector<SharedMatrix> Cright, std::vector<SharedMatrix> K,
                         double* T1p, double* T2p, double* Mp, size_t bcount, size_t block_size,
                         std::vector<DFHelperScratch>& C_buffers, bool lr_symmetric) {
    for (size_t i = 0; i < K.size(); i++) {
        size_t nocc = Cleft[i]->colspi()[0];
        if (!nocc) {
            continue;
        }

        double* Clp = Cleft[i]->pointer()[0];
        double* Crp = Cright[i]->pointer()[0];
        double* Kp = K[i]->pointer()[0];

        // (i,Q) layout pads nocc to a multiple of 8; the (Q,i) layout is the original one.
        const size_t nocc_pad = K_layout_iQ_ ? (nocc + 7) / 8 * 8 : nocc;
        const size_t K_inner = nocc_pad * block_size;

        // compute first tmp
        timer_on("DFH: K first transform");
        if (K_layout_iQ_) {
            first_transform_pQq_K(nocc, nocc_pad, bcount, block_size, Mp, T1p, Clp, C_buffers);
        } else {
            first_transform_pQq(nocc, bcount, block_size, Mp, T1p, Clp, C_buffers);
        }

        // compute second tmp
        if (lr_symmetric) {
            T2p = T1p;
        } else if (K_layout_iQ_) {
            first_transform_pQq_K(nocc, nocc_pad, bcount, block_size, Mp, T2p, Crp, C_buffers);
        } else {
            first_transform_pQq(nocc, bcount, block_size, Mp, T2p, Crp, C_buffers);
        }
        timer_off("DFH: K first transform");

        // compute K; when lr_symmetric, exploit K = T1 * T1^T symmetry to halve FLOPs
        timer_on("DFH: K final GEMM");
        if (lr_symmetric) {
            symm_k_final_accumulate<double>(nbf_, K_inner, T1p, Kp, C_buffers, nthreads_);
        } else {
            C_DGEMM('N', 'T', nbf_, nbf_, K_inner, 1.0, T1p, K_inner, T2p, K_inner, 1.0, Kp, nbf_);
        }
        timer_off("DFH: K final GEMM");
    }
}
size_t DFHelper::JK_scratch_memory() const {
    size_t size = jk_T1_size_ + jk_T2_size_;
    for (const auto& buffer : jk_C_buffers_) size += buffer.capacity();
    return size;
}
size_t DFHelper::JK_core_memory() const {
    const size_t n = direct_iaQ_ ? naux_ * nbf_ * nbf_ : big_skips_[nbf_];
    size_t metric_size = 0;
    // metric_prep_core can index one allocation by both 1.0 and its inverse
    // power. Those aliases survive out-of-core builds with hold_met enabled.
    for (auto it = metrics_.begin(); it != metrics_.end(); ++it) {
        bool alias = false;
        for (auto prev = metrics_.begin(); prev != it; ++prev) {
            if (prev->second == it->second) {
                alias = true;
                break;
            }
        }
        if (!alias) metric_size += naux_ * naux_;
    }
    return (Ppq_ ? n : 0) + (m1Ppq_ ? n : 0) + (wPpq_ ? n : 0) +
           metric_size + metric_panel_size_ + metric_out_size_;
}
bool DFHelper::prepare_K_fp32() {
    if (!AO_core_ || wcombine_ || direct_iaQ_ || !Ppq_) {
        Ppq_f_.reset();
        K_fp32_ = false;
        return false;
    }
    const size_t n = big_skips_[nbf_];
    const size_t held = JK_core_memory() + JK_scratch_memory();
    const size_t extra = n / 2 + n % 2;
    if (held > memory_ || extra > memory_ - held) {
        // Zero-extra-memory fp32 mode: keep Ppq_ in double and downcast tiles in L2 cache during K_fp32.
        Ppq_f_.reset();
        return true;
    }
    if (Ppq_f_) return true;
    timer_on("DFH: K fp32 tensor");
    Ppq_f_ = std::unique_ptr<float[]>(new float[n]);
    float* f = Ppq_f_.get();
    const double* d = Ppq_.get();
#pragma omp parallel for simd num_threads(nthreads_) schedule(static)
    for (size_t x = 0; x < n; x++) f[x] = static_cast<float>(d[x]);
    timer_off("DFH: K fp32 tensor");
    return true;
}
void DFHelper::compute_K_fp32(std::vector<SharedMatrix> Cleft, std::vector<SharedMatrix> Cright,
                              std::vector<SharedMatrix> K, double* T1p, double* T2p, const double* Mp, size_t bcount,
                              size_t block_size, std::vector<DFHelperScratch>& C_buffers, bool lr_symmetric) {
    // Respects the active K_layout_iQ_ orientation (IQ on AMD, QI on Intel), uses Ppq_f_ when available
    // or L2-tiled on-the-fly downcast of Mp when memory is tight (0 extra RAM), and contracts symmetric K
    // with K-sliced SSYRK + SGEMM directly into Kp without allocating an nbf^2 Kf matrix.
    const float* Mf = Ppq_f_.get();
    const bool use_iq = K_layout_iQ_;

    auto transform = [&](size_t nocc, size_t nocc_pad, const double* Bp, float* Tf) {
        const size_t k_stride = block_size * nocc_pad;
#pragma omp parallel for schedule(guided) num_threads(nthreads_)
        for (size_t k = 0; k < nbf_; k++) {
            const size_t sp_size = small_skips_[k];
            if (sp_size == 0) {
                std::fill_n(&Tf[k * k_stride], k_stride, 0.0f);
                continue;
            }
            const size_t jump = big_skips_[k] + bcount * sp_size;
            int rank = 0;
#ifdef _OPENMP
            rank = omp_get_thread_num();
#endif
            float* Cbuf_f = C_buffers[rank].float_data();
            const uint32_t* sig_m = &sig_q_[sig_q_row_off_[k]];

            if (use_iq) {
                gather_Ct_pad(Cbuf_f, Bp, sig_m, nocc, nocc_pad, sp_size);
                if (Mf != nullptr) {
                    C_SGEMM('N', 'T', nocc_pad, block_size, sp_size, 1.0f, Cbuf_f, sp_size,
                            const_cast<float*>(&Mf[jump]), sp_size, 0.0f, &Tf[k * k_stride], block_size);
                } else {
                    float* Mtile = Cbuf_f + nocc_pad * sp_size;
                    const size_t max_tile_floats = C_buffers[rank].size() * 2 - nocc_pad * sp_size;
                    const size_t qb_max = std::max<size_t>(1, std::min<size_t>(256, max_tile_floats / sp_size));
                    for (size_t q0 = 0; q0 < block_size; q0 += qb_max) {
                        const size_t qb = std::min(qb_max, block_size - q0);
                        const double* msrc = &Mp[jump + q0 * sp_size];
                        const size_t nel = qb * sp_size;
#pragma omp simd
                        for (size_t x = 0; x < nel; ++x) Mtile[x] = static_cast<float>(msrc[x]);
                        C_SGEMM('N', 'T', nocc_pad, qb, sp_size, 1.0f, Cbuf_f, sp_size, Mtile, sp_size, 0.0f,
                                &Tf[k * k_stride + q0], block_size);
                    }
                }
            } else {
                for (size_t c = 0; c < sp_size; ++c) {
                    const double* src = &Bp[static_cast<size_t>(sig_m[c]) * nocc];
                    float* dst = &Cbuf_f[c * nocc];
#pragma omp simd
                    for (size_t o = 0; o < nocc; ++o) dst[o] = static_cast<float>(src[o]);
                }
                if (Mf != nullptr) {
                    C_SGEMM('N', 'N', block_size, nocc, sp_size, 1.0f, const_cast<float*>(&Mf[jump]), sp_size, Cbuf_f,
                            nocc, 0.0f, &Tf[k * k_stride], nocc);
                } else {
                    float* Mtile = Cbuf_f + nocc * sp_size;
                    const size_t max_tile_floats = C_buffers[rank].size() * 2 - nocc * sp_size;
                    const size_t qb_max = std::max<size_t>(1, std::min<size_t>(256, max_tile_floats / sp_size));
                    for (size_t q0 = 0; q0 < block_size; q0 += qb_max) {
                        const size_t qb = std::min(qb_max, block_size - q0);
                        const double* msrc = &Mp[jump + q0 * sp_size];
                        const size_t nel = qb * sp_size;
#pragma omp simd
                        for (size_t x = 0; x < nel; ++x) Mtile[x] = static_cast<float>(msrc[x]);
                        C_SGEMM('N', 'N', qb, nocc, sp_size, 1.0f, Mtile, sp_size, Cbuf_f, nocc, 0.0f,
                                &Tf[(k * block_size + q0) * nocc], nocc);
                    }
                }
            }
        }
    };

    for (size_t i = 0; i < K.size(); i++) {
        size_t nocc = Cleft[i]->colspi()[0];
        if (!nocc) continue;
        const size_t nocc_pad = use_iq ? (nocc + 7) / 8 * 8 : nocc;
        const size_t K_inner = nocc_pad * block_size;
        float* T1f = jk_T1_.float_data();
        float* T2f = lr_symmetric ? T1f : jk_T2_.float_data();

        timer_on("DFH: K first transform");
        transform(nocc, nocc_pad, Cleft[i]->pointer()[0], T1f);
        if (!lr_symmetric) transform(nocc, nocc_pad, Cright[i]->pointer()[0], T2f);
        timer_off("DFH: K first transform");

        double* Kp = K[i]->pointer()[0];
        timer_on("DFH: K final GEMM");
        if (lr_symmetric) {
            symm_k_final_accumulate<float>(nbf_, K_inner, T1f, Kp, C_buffers, nthreads_);
        } else {
            // The coefficient buffers are idle after the transforms and hold
            // at least nbf^2 doubles. Do not allocate unbudgeted GEMM output.
            float* Kf = C_buffers[0].float_data();
            C_SGEMM('N', 'T', nbf_, nbf_, K_inner, 1.0f, T1f, K_inner, T2f, K_inner, 0.0f, Kf, nbf_);
#pragma omp parallel for simd num_threads(nthreads_) schedule(static)
            for (size_t x = 0; x < nbf_ * nbf_; x++) Kp[x] += Kf[x];
        }
        timer_off("DFH: K final GEMM");
    }
}
void DFHelper::compute_wK(std::vector<SharedMatrix> Cleft, std::vector<SharedMatrix> Cright,
                          std::vector<SharedMatrix> wK, size_t max_nocc, bool do_J, bool do_K, bool do_wK) {
    if (K_layout_iQ_ || K_fp32_) max_nocc = (max_nocc + 7) / 8 * 8;
    std::vector<std::pair<size_t, size_t>> Qsteps;
    std::tuple<size_t, size_t> info = Qshell_blocks_for_JK_build(Qsteps, max_nocc, false);
    size_t totsb = std::get<1>(info);

    double* wMp = wPpq_.get();
    double* M1p = m1Ppq_.get();
    ensure_jk_C_buffers(nbf_ * std::max(max_nocc, nbf_));
    auto& C_buffers = jk_C_buffers_;

    size_t Ktmp_size = (!max_nocc ? totsb * 1 : totsb * max_nocc);
    Ktmp_size = std::max(Ktmp_size * nbf_, nthreads_ * naux_);
    double* T1p = grow_scratch(jk_T1_, jk_T1_size_, Ktmp_size);
    double* T2p = grow_scratch(jk_T2_, jk_T2_size_, Ktmp_size);

    const bool use_fp32 = K_fp32_ && AO_core_ && !wcombine_ && !direct_iaQ_;
    const bool use_iq = K_layout_iQ_ && !wcombine_ && !direct_iaQ_;

    for (size_t bind = 0, bcount = 0; bind < Qsteps.size(); bind++) {
        size_t start = std::get<0>(Qsteps[bind]);
        size_t stop = std::get<1>(Qsteps[bind]);
        size_t begin = Qshell_aggs_[start];
        size_t end = Qshell_aggs_[stop + 1] - 1;
        size_t block_size = end - begin + 1;

        if (use_fp32) {
            auto transform_f32 = [&](size_t nocc, size_t nocc_pad, const double* Mp_src, const double* Bp, float* Tf) {
                const size_t k_stride = block_size * nocc_pad;
#pragma omp parallel for schedule(guided) num_threads(nthreads_)
                for (size_t k = 0; k < nbf_; k++) {
                    const size_t sp_size = small_skips_[k];
                    if (sp_size == 0) {
                        std::fill_n(&Tf[k * k_stride], k_stride, 0.0f);
                        continue;
                    }
                    const size_t jump = big_skips_[k] + bcount * sp_size;
                    int rank = 0;
#ifdef _OPENMP
                    rank = omp_get_thread_num();
#endif
                    float* Cbuf_f = C_buffers[rank].float_data();
                    const uint32_t* sig_m = &sig_q_[sig_q_row_off_[k]];
                    if (use_iq) {
                        gather_Ct_pad(Cbuf_f, Bp, sig_m, nocc, nocc_pad, sp_size);
                        float* Mtile = Cbuf_f + nocc_pad * sp_size;
                        const size_t max_tile_floats = C_buffers[rank].size() * 2 - nocc_pad * sp_size;
                        const size_t qb_max = std::max<size_t>(1, std::min<size_t>(256, max_tile_floats / sp_size));
                        for (size_t q0 = 0; q0 < block_size; q0 += qb_max) {
                            const size_t qb = std::min(qb_max, block_size - q0);
                            const double* msrc = &Mp_src[jump + q0 * sp_size];
                            const size_t nel = qb * sp_size;
#pragma omp simd
                            for (size_t x = 0; x < nel; ++x) Mtile[x] = static_cast<float>(msrc[x]);
                            C_SGEMM('N', 'T', nocc_pad, qb, sp_size, 1.0f, Cbuf_f, sp_size, Mtile, sp_size, 0.0f,
                                    &Tf[k * k_stride + q0], block_size);
                        }
                    } else {
                        for (size_t c = 0; c < sp_size; ++c) {
                            const double* src = &Bp[static_cast<size_t>(sig_m[c]) * nocc];
                            float* dst = &Cbuf_f[c * nocc];
#pragma omp simd
                            for (size_t o = 0; o < nocc; ++o) dst[o] = static_cast<float>(src[o]);
                        }
                        float* Mtile = Cbuf_f + nocc * sp_size;
                        const size_t max_tile_floats = C_buffers[rank].size() * 2 - nocc * sp_size;
                        const size_t qb_max = std::max<size_t>(1, std::min<size_t>(256, max_tile_floats / sp_size));
                        for (size_t q0 = 0; q0 < block_size; q0 += qb_max) {
                            const size_t qb = std::min(qb_max, block_size - q0);
                            const double* msrc = &Mp_src[jump + q0 * sp_size];
                            const size_t nel = qb * sp_size;
#pragma omp simd
                            for (size_t x = 0; x < nel; ++x) Mtile[x] = static_cast<float>(msrc[x]);
                            C_SGEMM('N', 'N', qb, nocc, sp_size, 1.0f, Mtile, sp_size, Cbuf_f, nocc, 0.0f,
                                    &Tf[(k * block_size + q0) * nocc], nocc);
                        }
                    }
                }
            };

            for (size_t i = 0; i < wK.size(); i++) {
                size_t nocc = Cleft[i]->colspi()[0];
                if (!nocc) continue;
                const size_t nocc_pad = use_iq ? (nocc + 7) / 8 * 8 : nocc;
                const size_t K_inner = nocc_pad * block_size;
                float* T1f = jk_T1_.float_data();
                float* T2f = jk_T2_.float_data();

                transform_f32(nocc, nocc_pad, M1p, Cleft[i]->pointer()[0], T1f);
                transform_f32(nocc, nocc_pad, wMp, Cright[i]->pointer()[0], T2f);

                // Reuse C_buffers[0] (>= nbf^2 doubles = 2*nbf^2 floats) for zero-allocation SGEMM output
                float* wKf = C_buffers[0].float_data();
                C_SGEMM('N', 'T', nbf_, nbf_, K_inner, 1.0f, T1f, K_inner, T2f, K_inner, 0.0f, wKf, nbf_);
                double* wKp = wK[i]->pointer()[0];
#pragma omp parallel for simd num_threads(nthreads_) schedule(static)
                for (size_t x = 0; x < nbf_ * nbf_; ++x) {
                    wKp[x] += static_cast<double>(wKf[x]);
                }
            }
        } else {
            for (size_t i = 0; i < wK.size(); i++) {
                size_t nocc = Cleft[i]->colspi()[0];
                if (!nocc) {
                    continue;
                }
                const size_t nocc_pad = use_iq ? (nocc + 7) / 8 * 8 : nocc;
                const size_t K_inner = nocc_pad * block_size;
                double* Clp = Cleft[i]->pointer()[0];
                double* Crp = Cright[i]->pointer()[0];
                double* wKp = wK[i]->pointer()[0];

                if (use_iq) {
                    first_transform_pQq_K(nocc, nocc_pad, bcount, block_size, M1p, T1p, Clp, C_buffers);
                    first_transform_pQq_K(nocc, nocc_pad, bcount, block_size, wMp, T2p, Crp, C_buffers);
                } else {
                    first_transform_pQq(nocc, bcount, block_size, M1p, T1p, Clp, C_buffers);
                    first_transform_pQq(nocc, bcount, block_size, wMp, T2p, Crp, C_buffers);
                }

                C_DGEMM('N', 'T', nbf_, nbf_, K_inner, 1.0, T1p, K_inner, T2p, K_inner, 1.0, wKp, nbf_);
            }
        }
        bcount += block_size;
    }
}

}  // End namespaces
