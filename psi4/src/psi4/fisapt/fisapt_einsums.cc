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

#include "fisapt.h"

#include <Einsums/TensorAlgebra/Detail/Index.hpp>
#include <TensorAlgebra/TensorAlgebra.hpp>
#include <algorithm>
#include <ctime>
#include <functional>
#include <set>
#include "psi4/mcscf/block_matrix.h"
#ifdef _OPENMP
#include <omp.h>
#endif

#include "psi4/psi4-dec.h"
#include "psi4/physconst.h"

#include "psi4/lib3index/dfhelper.h"
#include "psi4/libcubeprop/csg.h"
#include "psi4/libdiis/diismanager.h"
#include "psi4/libfock/jk.h"
#include "psi4/libmints/basisset.h"
#include "psi4/libmints/extern.h"
#include "psi4/libmints/factory.h"
#include "psi4/libmints/integral.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libmints/potential.h"
#include "psi4/libmints/vector.h"
#include "psi4/liboptions/liboptions.h"
#include "psi4/libpsi4util/process.h"
#include "psi4/libqt/qt.h"
#include "psi4/libmints/mintshelper.h"

#include "local2.h"

// Einsums includes
#include <Einsums/TensorAlgebra.hpp>
#include <Einsums/Tensor/DiskTensor.hpp>

namespace psi {

namespace fisapt {

// Compute fragment-fragment partitioning of electrostatic contribution
void FISAPT::felst_einsums() {
    using namespace einsums;
    using namespace einsums::tensor_algebra;
    using namespace einsums::index;

    outfile->Printf("  ==> F-SAPT Electrostatics (Einsums) <==\n\n");

    // => Sizing <= //

    std::string link_assignment = options_.get_str("FISAPT_LINK_ASSIGNMENT");
    std::shared_ptr<Molecule> mol = primary_->molecule();
    int nn = primary_->nbf();
    int nA = mol->natom();
    int nB = mol->natom();

// Some special considerations for the SAOn/SIAOn variants of ISAPT below
    int na = matrices_["Locc0A"]->colspi()[0];
    int nb = matrices_["Locc0B"]->colspi()[0];
    if (link_assignment == "SAO0" || link_assignment == "SAO1" || link_assignment == "SAO2" || link_assignment == "SIAO0" || link_assignment == "SIAO1" || link_assignment == "SIAO2") {
        na = matrices_["LLocc0A"]->colspi()[0];
        nb = matrices_["LLocc0B"]->colspi()[0];
    }
    std::shared_ptr<Matrix> L0A = std::make_shared<Matrix>("L0A", nn, na);
    std::shared_ptr<Matrix> L0B = std::make_shared<Matrix>("L0B", nn, nb);
    if (link_assignment == "SAO0" || link_assignment == "SAO1" || link_assignment == "SAO2" || link_assignment == "SIAO0" || link_assignment == "SIAO1" || link_assignment == "SIAO2") {
        L0A->copy(matrices_["LLocc0A"]);
        L0B->copy(matrices_["LLocc0B"]);
    }
    else {
        L0A->copy(matrices_["Locc0A"]);
        L0B->copy(matrices_["Locc0B"]);
    }
// for the SAOn/SIAOn variants, we need to do the same things but with the Locc0A, Locc0B matrices augmented by the link orbital. 
// The local matrices L0A/L0B hold the correct form, augmented or not, as needed.

    // => Targets <= //

    double Elst10 = 0.0;
    std::vector<double> Elst10_terms;
    Elst10_terms.resize(4);

    // This matrix contains the decomposition of the F-SAPT electrostatics energy to nuclear, orbital, and external
    // point charges contributions. nA and nB are the total number of atoms in subsystems A and B. na(nb) is the number
    // of occupied oritals for subsystem A(B). One additional entry is added to include the effect of external 
    // point charges in A and B (if present; otherwise it will be zero columns/rows). This matrix is saved
    // in the Elst.dat file generated after the F-SAPT analysis. The fsapt.py script reads this file and the
    // user-defined functional group partitioning and analyzes the F-SAPT interaction energy in terms of functional
    // group contributions.
    //
    // For the nuclei-nuclei interactions (entries [0:nA-1, 0:nB-1]), the total number of A-B atoms is used for the rows
    // and columns but only the entries where nuclei A interact with nuclei B are actually nonzero. Entries
    // [nA:nA+na-1, nB:nB+nb-1] represent the interactions between the local occupied orbitals in A and the local occupied
    // orbitals in B. Lastly, entry [nA, nB] represent the interaction between the external point charges in A and B. 
    // Cross terms represent the interaction between nuclei and electrons, nuclei and external point charges, 
    // and electrons and external point charges.
    //
    // Similar matrices are used for the other SAPT components.
    matrices_["Elst_AB"] = std::make_shared<Matrix>("Elst_AB", nA + na + 1, nB + nb + 1); // Add one entry for external
                                                                                          // potentials in A and B
    double** Ep = matrices_["Elst_AB"]->pointer();

    // => A <-> B <= //

    double* ZAp = vectors_["ZA"]->pointer();
    double* ZBp = vectors_["ZB"]->pointer();
    for (int A = 0; A < nA; A++) {
        for (int B = 0; B < nB; B++) {
            if (A == B) continue;
            double E = ZAp[A] * ZBp[B] / mol->xyz(A).distance(mol->xyz(B));
            Ep[A][B] += E;
            Elst10_terms[3] += E;
        }
    }

    // Extern A-atom B interaction
    // Compute the interaction between the external potenrial and each atom in the fragment
    if (reference_->has_potential_variable("A")) {
        double conv = 1;
        if (mol->units() == Molecule::Angstrom) {
            conv *= pc_bohr2angstroms;
        }
        double E = 0.0;
        for (int B = 0; B < nB; B++) {
            auto atom = std::make_shared<Molecule>();
            atom->set_units(mol->units());
            atom->add_atom(ZBp[B], mol->x(B)*conv, mol->y(B)*conv, mol->z(B)*conv);
            double interaction = reference_->potential_variable("A")->computeNuclearEnergy(atom);
            Ep[nA + na][B] = interaction;
            E += interaction;
        }
        Elst10_terms[3] += E;
    }

    // Extern B-atom A interaction
    // Compute the interaction between the external potenrial and each atom in the fragment
    if (reference_->has_potential_variable("B")) {
        double conv = 1;
        if (mol->units() == Molecule::Angstrom) {
            conv *= pc_bohr2angstroms;
        }
        double E = 0.0;
        for (int A = 0; A < nA; A++) {
            auto atom = std::make_shared<Molecule>();
            atom->set_units(mol->units());
            atom->add_atom(ZAp[A], mol->x(A)*conv, mol->y(A)*conv, mol->z(A)*conv);
            double interaction = reference_->potential_variable("B")->computeNuclearEnergy(atom);
            Ep[A][nB + nb] = interaction;
            E += interaction;
        }
        Elst10_terms[3] += E;
    }

    // => a <-> b <= //

    int nT = 1;
#ifdef _OPENMP
    nT = Process::environment.get_n_threads();
#endif

    // => Get integrals from DFHelper <= //
    dfh_ = std::make_shared<DFHelper>(primary_, reference_->get_basisset("DF_BASIS_SCF"));
    dfh_->set_memory(doubles_);
    dfh_->set_method("DIRECT_iaQ");
    dfh_->set_nthreads(nT);
    dfh_->initialize();
    dfh_->print_header();

    dfh_->add_space("a", L0A);
    dfh_->add_space("b", L0B);

    dfh_->add_transformation("Aaa", "a", "a");
    dfh_->add_transformation("Abb", "b", "b");

    dfh_->transform();

    size_t nQ = dfh_->get_naux();
    auto QaC = std::make_shared<Matrix>("QaC", na, nQ);
    double** QaCp = QaC->pointer();
    for (size_t a = 0; a < na; a++) {
        dfh_->fill_tensor("Aaa", QaCp[a], {a, a + 1}, {a, a + 1});
    }

    auto QbC = std::make_shared<Matrix>("QbC", nb, nQ);
    double** QbCp = QbC->pointer();
    for (size_t b = 0; b < nb; b++) {
        dfh_->fill_tensor("Abb", QbCp[b], {b, b + 1}, {b, b + 1});
    }

    std::shared_ptr<Matrix> Elst10_3 = linalg::doublet(QaC, QbC, false, true);
    // need to initialize Elst10_3 based on QaC and QbC sizes for einsums compatibility
    // std::shared_ptr<Matrix> Elst10_3 = std::make_shared<Matrix>("Elst10_3", na, nb);
    // einsum(Indices{a, b}, Elst10_3.get(), Indices{a, c}, QaC.get(), Indices{b, c}, QbC.get());
    double** Elst10_3p = Elst10_3->pointer();
    for (int a = 0; a < na; a++) {
        for (int b = 0; b < nb; b++) {
            double E = 4.0 * Elst10_3p[a][b];
            Elst10_terms[2] += E;
            Ep[a + nA][b + nB] += E;
        }
    }

    matrices_["Vlocc0A"] = QaC;
    matrices_["Vlocc0B"] = QbC;

    // => Nuclear Part (PITA) <= //
    auto Vfact2 = std::make_shared<IntegralFactory>(primary_);
    std::shared_ptr<PotentialInt> Vint2(static_cast<PotentialInt*>(Vfact2->ao_potential().release()));
    auto Vtemp2 = std::make_shared<Matrix>("Vtemp2", nn, nn);


    // => A <-> b <= //

    for (int A = 0; A < nA; A++) {
        if (ZAp[A] == 0.0) continue;
        Vtemp2->zero();
        Vint2->set_charge_field({{ZAp[A], {mol->x(A), mol->y(A), mol->z(A)}}});
        Vint2->compute(Vtemp2);
        std::shared_ptr<Matrix> Vbb =
            linalg::triplet(L0B, Vtemp2, L0B, true, false, false);
        double** Vbbp = Vbb->pointer();
        for (int b = 0; b < nb; b++) {
            double E = 2.0 * Vbbp[b][b];
            Elst10_terms[1] += E;
            Ep[A][b + nB] += E;
        }
    }

    // Add Extern-A - Orbital b interaction
    if (reference_->has_potential_variable("A")) {
        std::shared_ptr<Matrix> Vbb =
            linalg::triplet(L0B, matrices_["VA_extern"], L0B, true, false, false);
        double** Vbbp = Vbb->pointer();
        for (int b = 0; b < nb; b++) {
            double E = 2.0 * Vbbp[b][b];
            Elst10_terms[1] += E;
            Ep[nA + na][b + nB] += E;
        }
    }

    // => a <-> B <= //

    for (int B = 0; B < nB; B++) {
        if (ZBp[B] == 0.0) continue;
        Vtemp2->zero();
        Vint2->set_charge_field({{ZBp[B], {mol->x(B), mol->y(B), mol->z(B)}}});
        Vint2->compute(Vtemp2);
        std::shared_ptr<Matrix> Vaa =
            linalg::triplet(L0A, Vtemp2, L0A, true, false, false);
        double** Vaap = Vaa->pointer();
        for (int a = 0; a < na; a++) {
            double E = 2.0 * Vaap[a][a];
            Elst10_terms[0] += E;
            Ep[a + nA][B] += E;
        }
    }

    // Add Extern-B - Orbital a interaction
    if (reference_->has_potential_variable("B")) {
        std::shared_ptr<Matrix> Vaa =
            linalg::triplet(L0A, matrices_["VB_extern"], L0A, true, false, false);
        double** Vaap = Vaa->pointer();
        for (int a = 0; a < na; a++) {
            double E = 2.0 * Vaap[a][a];
            Elst10_terms[0] += E;
            Ep[a + nA][nB + nb] += E;
        }
    }


    // Prepare DFHelper object for the next module
    dfh_->clear_spaces();

    // => Summation <= //

    for (int k = 0; k < Elst10_terms.size(); k++) {
        Elst10 += Elst10_terms[k];
    }
    // for (int k = 0; k < Elst10_terms.size(); k++) {
    //    outfile->Printf("    Elst10,r (%1d)        = %18.12lf [Eh]\n",k+1,Elst10_terms[k]);
    //}
    // scalars_["Elst10,r"] = Elst10;
    outfile->Printf("    Elst10,r            = %18.12lf [Eh]\n", Elst10);

    // Add extern-extern contribution
    if (reference_->has_potential_variable("A") && reference_->has_potential_variable("B")) {
        // Add the interaction between the external potenitals in A and B
        // Multiply by 2 to get the full A-B + B-A interaction energy
        double ext = matrices_["extern_extern_IE"]->get(0, 1)*2.0; 
        Ep[nA + na][nB + nb] += ext;
        outfile->Printf("    Extern-Extern       = %18.12lf [Eh]\n", ext);
    }

    outfile->Printf("\n");

    // fflush(outfile);
}

// Compute fragment-fragment partitioning of exchange contribution
void FISAPT::fexch_einsums() {
    outfile->Printf("  ==> F-SAPT Exchange <==\n\n");

    // => Sizing <= //

    std::shared_ptr<Molecule> mol = primary_->molecule();
    int nn = primary_->nbf();
    int nA = mol->natom();
    int nB = mol->natom();
    int na = matrices_["Locc0A"]->colspi()[0];
    int nb = matrices_["Locc0B"]->colspi()[0];
    int nr = matrices_["Cvir0A"]->colspi()[0];
    int ns = matrices_["Cvir0B"]->colspi()[0];

//for the SAOn/SIAOn variants, we sometimes need na1 = na+1 (with link orbital) and sometimes na (without) - be careful with this!
    std::string link_assignment = options_.get_str("FISAPT_LINK_ASSIGNMENT");
    int na1 = na;
    int nb1 = nb;
    if (link_assignment == "SAO0" || link_assignment == "SAO1" || link_assignment == "SAO2" || link_assignment == "SIAO0" || link_assignment == "SIAO1" || link_assignment == "SIAO2") {
        na1 = na + 1;
        nb1 = nb + 1;
    }

    // => Targets <= //

    double Exch10_2 = 0.0;
    std::vector<double> Exch10_2_terms;
    Exch10_2_terms.resize(3);

    matrices_["Exch_AB"] = std::make_shared<Matrix>("Exch_AB", nA + na1 + 1, nB + nb1 + 1); // Add one entry for external
                                                                                          // potentials in A and B
    double** Ep = matrices_["Exch_AB"]->pointer();

    // ==> Stack Variables <== //

    std::shared_ptr<Matrix> S = matrices_["S"];
    std::shared_ptr<Matrix> V_A = matrices_["V_A"];
    std::shared_ptr<Matrix> J_A = matrices_["J_A"];
    std::shared_ptr<Matrix> V_B = matrices_["V_B"];
    std::shared_ptr<Matrix> J_B = matrices_["J_B"];

    std::shared_ptr<Matrix> LoccA = matrices_["Locc0A"];
    std::shared_ptr<Matrix> LoccB = matrices_["Locc0B"];
    std::shared_ptr<Matrix> CvirA = matrices_["Cvir0A"];
    std::shared_ptr<Matrix> CvirB = matrices_["Cvir0B"];

    // ==> DF ERI Setup (JKFIT Type, in Full Basis) <== //

    int nT = 1;
#ifdef _OPENMP
    nT = Process::environment.get_n_threads();
#endif

    std::vector<std::shared_ptr<Matrix> > Cs;
    Cs.push_back(LoccA);
    Cs.push_back(CvirA);
    Cs.push_back(LoccB);
    Cs.push_back(CvirB);

    size_t max_MO = 0;
    for (auto& mat : Cs) max_MO = std::max(max_MO, (size_t)mat->ncol());

    dfh_->add_space("a", Cs[0]);
    dfh_->add_space("r", Cs[1]);
    dfh_->add_space("b", Cs[2]);
    dfh_->add_space("s", Cs[3]);

    dfh_->add_transformation("Aar", "a", "r");
    dfh_->add_transformation("Abs", "b", "s");

    dfh_->transform();

    // ==> Electrostatic Potentials <== //

    std::shared_ptr<Matrix> W_A(J_A->clone());
    W_A->copy(J_A);
    W_A->scale(2.0);
    W_A->add(V_A);

    std::shared_ptr<Matrix> W_B(J_B->clone());
    W_B->copy(J_B);
    W_B->scale(2.0);
    W_B->add(V_B);

    std::shared_ptr<Matrix> WAbs = linalg::triplet(LoccB, W_A, CvirB, true, false, false);
    std::shared_ptr<Matrix> WBar = linalg::triplet(LoccA, W_B, CvirA, true, false, false);
    double** WBarp = WBar->pointer();
    double** WAbsp = WAbs->pointer();

    W_A.reset();
    W_B.reset();

    // ==> Exchange S^2 Computation <== //

    std::shared_ptr<Matrix> Sab = linalg::triplet(LoccA, S, LoccB, true, false, false);
    std::shared_ptr<Matrix> Sba = linalg::triplet(LoccB, S, LoccA, true, false, false);
    std::shared_ptr<Matrix> Sas = linalg::triplet(LoccA, S, CvirB, true, false, false);
    std::shared_ptr<Matrix> Sbr = linalg::triplet(LoccB, S, CvirA, true, false, false);
    double** Sabp = Sab->pointer();
    double** Sbap = Sba->pointer();
    double** Sasp = Sas->pointer();
    double** Sbrp = Sbr->pointer();
    // LoccA->set_name("LoccA");
    // LoccA->print();

    // Sas->set_name("Sas");
    // Sbr->set_name("Sbr");
    // Sab->set_name("Sab");
    // Sas->print();
    // Sab->print();

    auto WBab = std::make_shared<Matrix>("WBab", na, nb);
    double** WBabp = WBab->pointer();
    auto WAba = std::make_shared<Matrix>("WAba", nb, na);
    double** WAbap = WAba->pointer();

    C_DGEMM('N', 'T', na, nb, nr, 1.0, WBarp[0], nr, Sbrp[0], nr, 0.0, WBabp[0], nb);
    C_DGEMM('N', 'T', nb, na, ns, 1.0, WAbsp[0], ns, Sasp[0], ns, 0.0, WAbap[0], na);

    auto E_exch1 = std::make_shared<Matrix>("E_exch [a <x- b]", na, nb);
    double** E_exch1p = E_exch1->pointer();
    auto E_exch2 = std::make_shared<Matrix>("E_exch [a -x> b]", na, nb);
    double** E_exch2p = E_exch2->pointer();

    for (int a = 0; a < na; a++) {
        for (int b = 0; b < nb; b++) {
            E_exch1p[a][b] -= 2.0 * Sabp[a][b] * WBabp[a][b];
            E_exch2p[a][b] -= 2.0 * Sbap[b][a] * WAbap[b][a];
        }
    }

    // E_exch1->print();
    // E_exch2->print();

    size_t nQ = dfh_->get_naux();
    auto TrQ = std::make_shared<Matrix>("TrQ", nr, nQ);
    double** TrQp = TrQ->pointer();
    auto TsQ = std::make_shared<Matrix>("TsQ", ns, nQ);
    double** TsQp = TsQ->pointer();
    auto TbQ = std::make_shared<Matrix>("TbQ", nb, nQ);
    double** TbQp = TbQ->pointer();
    auto TaQ = std::make_shared<Matrix>("TaQ", na, nQ);
    double** TaQp = TaQ->pointer();

    dfh_->add_disk_tensor("Bab", std::make_tuple(na, nb, nQ));

    for (size_t a = 0; a < na; a++) {
        dfh_->fill_tensor("Aar", TrQ, {a, a + 1});
        C_DGEMM('N', 'N', nb, nQ, nr, 1.0, Sbrp[0], nr, TrQp[0], nQ, 0.0, TbQp[0], nQ);
        dfh_->write_disk_tensor("Bab", TbQ, {a, a + 1});
    }

    dfh_->add_disk_tensor("Bba", std::make_tuple(nb, na, nQ));

    for (size_t b = 0; b < nb; b++) {
        dfh_->fill_tensor("Abs", TsQ, {b, b + 1});
        C_DGEMM('N', 'N', na, nQ, ns, 1.0, Sasp[0], ns, TsQp[0], nQ, 0.0, TaQp[0], nQ);
        dfh_->write_disk_tensor("Bba", TaQ, {b, b + 1});
    }

    auto E_exch3 = std::make_shared<Matrix>("E_exch [a <x-x> b]", na, nb);
    double** E_exch3p = E_exch3->pointer();

    for (size_t a = 0; a < na; a++) {
        dfh_->fill_tensor("Bab", TbQ, {a, a + 1});
        for (size_t b = 0; b < nb; b++) {
            dfh_->fill_tensor("Bba", TaQ, {b, b + 1}, {a, a + 1});
            E_exch3p[a][b] -= 2.0 * C_DDOT(nQ, TbQp[b], 1, TaQp[0], 1);
        }
    }

    // E_exch3->print();

    // => Totals <= //

    for (int a = 0; a < na; a++) {
        for (int b = 0; b < nb; b++) {
            Ep[a + nA][b + nB] = E_exch1p[a][b] + E_exch2p[a][b] + E_exch3p[a][b];
            Exch10_2_terms[0] += E_exch1p[a][b];
            Exch10_2_terms[1] += E_exch2p[a][b];
            Exch10_2_terms[2] += E_exch3p[a][b];
        }
    }

    for (int k = 0; k < Exch10_2_terms.size(); k++) {
        Exch10_2 += Exch10_2_terms[k];
    }
    // for (int k = 0; k < Exch10_2_terms.size(); k++) {
    //    outfile->Printf("    Exch10(S^2) (%1d)     = %18.12lf [Eh]\n",k+1,Exch10_2_terms[k]);
    //}
    // scalars_["Exch10(S^2)"] = Exch10_2;
    outfile->Printf("    Exch10(S^2)         = %18.12lf [Eh]\n", Exch10_2);
    outfile->Printf("\n");
    // fflush(outfile);

    // => Exchange scaling <= //

    if (options_.get_bool("FISAPT_FSAPT_EXCH_SCALE")) {
    // KP: the following change in the scaling (see the commented out line) is essential for ISAPT(SAOn) and ISAPT(SIAOn),
    // and should not mess up the other variants. The problem is that exch() calculates E(10)exch(S^2) in the DM formalism
    // (good for SAO/SIAO because no virtual orbitals in formulas, bad for FSAPT), while fexch() calculates this term
    // in the second quantization formalism (good for FSAPT, bad for SAO/SIAO). In effect, the scaling for SAO/SIAO accounts
    // for two effects at once: terms beyond S^2 and linking terms computed by exch() but not here.
    //
    // Not changing the scaling for sSAPT0 - if you try using sSAPT0 with ISAPT(SAO/SIAO), you've only got yourself to blame.
    //  double scale = scalars_["Exch10"] / scalars_["Exch10(S^2)"];
        double scale = scalars_["Exch10"] / Exch10_2;
        matrices_["Exch_AB"]->scale(scale);
        outfile->Printf("    Scaling F-SAPT Exch10(S^2) by %11.3E to match Exch10\n\n", scale);
    }
    if (options_.get_bool("SSAPT0_SCALE")) {
        sSAPT0_scale_ = scalars_["Exch10"] / scalars_["Exch10(S^2)"];
        sSAPT0_scale_ = pow(sSAPT0_scale_, 3.0);
        outfile->Printf("    Scaling F-SAPT Exch-Ind and Exch-Disp by %11.3E \n\n", sSAPT0_scale_);
    }

    // Prepare DFHelper object for the next module
    dfh_->clear_spaces();
}

// Compute fragment-fragment partitioning of induction contribution
void FISAPT::find_einsums() {
    outfile->Printf("  ==> F-SAPT Induction <==\n\n");

    // => Options <= //

    bool ind_resp = options_.get_bool("FISAPT_FSAPT_IND_RESPONSE");
    bool ind_scale = options_.get_bool("FISAPT_FSAPT_IND_SCALE");
    std::string link_assignment = options_.get_str("FISAPT_LINK_ASSIGNMENT");

    // => Sizing <= //

    std::shared_ptr<Molecule> mol = primary_->molecule();
    int nn = primary_->nbf();
    int nA = mol->natom();
    int nB = mol->natom();
    int na = matrices_["Locc0A"]->colspi()[0];
    int nb = matrices_["Locc0B"]->colspi()[0];
    int nr = matrices_["Cvir0A"]->colspi()[0];
    int ns = matrices_["Cvir0B"]->colspi()[0];

//for the SAOn/SIAOn variants, we sometimes need na1 = na+1 (with link orbital) and sometimes na (without) - be careful with this!
    int na1 = na;
    int nb1 = nb;
    if (link_assignment == "SAO0" || link_assignment == "SAO1" || link_assignment == "SAO2" || link_assignment == "SIAO0" || link_assignment == "SIAO1" || link_assignment == "SIAO2") {
        na1 = na + 1;
        nb1 = nb + 1;
    }

    // => Pointers <= //

    std::shared_ptr<Matrix> Locc_A = matrices_["Locc0A"];
    std::shared_ptr<Matrix> Locc_B = matrices_["Locc0B"];

    std::shared_ptr<Matrix> Uocc_A = matrices_["Uocc0A"];
    std::shared_ptr<Matrix> Uocc_B = matrices_["Uocc0B"];

    std::shared_ptr<Matrix> Cocc_A = matrices_["Cocc0A"];
    std::shared_ptr<Matrix> Cocc_B = matrices_["Cocc0B"];
    std::shared_ptr<Matrix> Cvir_A = matrices_["Cvir0A"];
    std::shared_ptr<Matrix> Cvir_B = matrices_["Cvir0B"];

    std::shared_ptr<Vector> eps_occ_A = vectors_["eps_occ0A"];
    std::shared_ptr<Vector> eps_occ_B = vectors_["eps_occ0B"];
    std::shared_ptr<Vector> eps_vir_A = vectors_["eps_vir0A"];
    std::shared_ptr<Vector> eps_vir_B = vectors_["eps_vir0B"];

    // => DFHelper = DF + disk tensors <= //

    int nT = 1;
#ifdef _OPENMP
    nT = Process::environment.get_n_threads();
#endif

    size_t nQ = dfh_->get_naux();

    // => ESPs <= //

    dfh_->add_disk_tensor("WBar", std::make_tuple(nB + nb1 + 1, na, nr)); // add entry for external potentials
    dfh_->add_disk_tensor("WAbs", std::make_tuple(nA + na1 + 1, nb, ns)); // add entry for external potentials

    // => Nuclear Part (PITA) <= //

    auto Vfact2 = std::make_shared<IntegralFactory>(primary_);
    std::shared_ptr<PotentialInt> Vint2(static_cast<PotentialInt*>(Vfact2->ao_potential().release()));
    auto Vtemp2 = std::make_shared<Matrix>("Vtemp2", nn, nn);

    double* ZAp = vectors_["ZA"]->pointer();
    for (size_t A = 0; A < nA; A++) {
        Vtemp2->zero();
        Vint2->set_charge_field({{ZAp[A], {mol->x(A), mol->y(A), mol->z(A)}}});
        Vint2->compute(Vtemp2);
        std::shared_ptr<Matrix> Vbs = linalg::triplet(Cocc_B, Vtemp2, Cvir_B, true, false, false);
        dfh_->write_disk_tensor("WAbs", Vbs, {A, A + 1});
    }

    double* ZBp = vectors_["ZB"]->pointer();
    for (size_t B = 0; B < nB; B++) {
        Vtemp2->zero();
        Vint2->set_charge_field({{ZBp[B], {mol->x(B), mol->y(B), mol->z(B)}}});
        Vint2->compute(Vtemp2);
        std::shared_ptr<Matrix> Var = linalg::triplet(Cocc_A, Vtemp2, Cvir_A, true, false, false);
        dfh_->write_disk_tensor("WBar", Var, {B, B + 1});
    }

    // ==> DFHelper Setup (JKFIT Type, in Full Basis) <== //

    std::vector<std::shared_ptr<Matrix> > Cs;
    Cs.push_back(Cocc_A);
    Cs.push_back(Cvir_A);
    Cs.push_back(Cocc_B);
    Cs.push_back(Cvir_B);

    size_t max_MO = 0;
    for (auto& mat : Cs) max_MO = std::max(max_MO, (size_t)mat->ncol());

    dfh_->add_space("a", Cs[0]);
    dfh_->add_space("r", Cs[1]);
    dfh_->add_space("b", Cs[2]);
    dfh_->add_space("s", Cs[3]);

    dfh_->add_transformation("Aar", "a", "r");
    dfh_->add_transformation("Abs", "b", "s");

    dfh_->transform();

    // => Electronic Part (Massive PITA) <= //

    double** RaCp = matrices_["Vlocc0A"]->pointer();
    double** RbDp = matrices_["Vlocc0B"]->pointer();
    // for SAOn/SIAOn, the above two matrices contain the link orbital contribution

    auto TsQ = std::make_shared<Matrix>("TsQ", ns, nQ);
    auto T1As = std::make_shared<Matrix>("T1As", na1, ns);
    double** TsQp = TsQ->pointer();
    double** T1Asp = T1As->pointer();
    for (size_t b = 0; b < nb; b++) {
        dfh_->fill_tensor("Abs", TsQ, {b, b + 1});
        C_DGEMM('N', 'T', na1, ns, nQ, 2.0, RaCp[0], nQ, TsQp[0], nQ, 0.0, T1Asp[0], ns);
        for (size_t a = 0; a < na1; a++) {
            dfh_->write_disk_tensor("WAbs", T1Asp[a], {nA + a, nA + a + 1}, {b, b + 1});
        }
    }

    auto TrQ = std::make_shared<Matrix>("TrQ", nr, nQ);
    auto T1Br = std::make_shared<Matrix>("T1Br", nb1, nr);
    double** TrQp = TrQ->pointer();
    double** T1Brp = T1Br->pointer();
    for (size_t a = 0; a < na; a++) {
        dfh_->fill_tensor("Aar", TrQ, {a, a + 1});
        C_DGEMM('N', 'T', nb1, nr, nQ, 2.0, RbDp[0], nQ, TrQp[0], nQ, 0.0, T1Brp[0], nr);
        for (size_t b = 0; b < nb1; b++) {
            dfh_->write_disk_tensor("WBar", T1Brp[b], {nB + b, nB + b + 1}, {a, a + 1});
        }
    }

    // ==> Stack Variables <== //

    double* eap = eps_occ_A->pointer();
    double* ebp = eps_occ_B->pointer();
    double* erp = eps_vir_A->pointer();
    double* esp = eps_vir_B->pointer();

    std::shared_ptr<Matrix> S = matrices_["S"];
    std::shared_ptr<Matrix> D_A = matrices_["D_A"];
    std::shared_ptr<Matrix> V_A = matrices_["V_A"];
    std::shared_ptr<Matrix> J_A = matrices_["J_A"];
    std::shared_ptr<Matrix> K_A = matrices_["K_A"];
    std::shared_ptr<Matrix> D_B = matrices_["D_B"];
    std::shared_ptr<Matrix> V_B = matrices_["V_B"];
    std::shared_ptr<Matrix> J_B = matrices_["J_B"];
    std::shared_ptr<Matrix> K_B = matrices_["K_B"];
    std::shared_ptr<Matrix> J_O = matrices_["J_O"];
    std::shared_ptr<Matrix> K_O = matrices_["K_O"];
    std::shared_ptr<Matrix> J_P_A = matrices_["J_P_A"];
    std::shared_ptr<Matrix> J_P_B = matrices_["J_P_B"];

    // ==> MO Amplitudes/Sources (by source atom) <== //

    auto xA = std::make_shared<Matrix>("xA", na, nr);
    auto xB = std::make_shared<Matrix>("xB", nb, ns);
    double** xAp = xA->pointer();
    double** xBp = xB->pointer();

    auto wB = std::make_shared<Matrix>("wB", na, nr);
    auto wA = std::make_shared<Matrix>("wA", nb, ns);
    double** wBp = wB->pointer();
    double** wAp = wA->pointer();

    auto uAT = std::make_shared<Matrix>("uAT", nb, ns);
    auto wAT = std::make_shared<Matrix>("wAT", nb, ns);
    auto uBT = std::make_shared<Matrix>("uBT", na, nr);
    auto wBT = std::make_shared<Matrix>("wBT", na, nr);

// Now come the second-order exchange-induction energy expressions appropriate for the SAOn/SIAOn link assignments.
// Here, we compute only the averaged spin coupling, like when FISAPT_EXCH_PARPERP == false.
// See https://doi.org/10.1021/acs.jpca.2c06465 for details - equation numbers below refer to this paper and its SI.
    if (link_assignment == "SAO0" || link_assignment == "SAO1" || link_assignment == "SAO2" || link_assignment == "SIAO0" || link_assignment == "SIAO1" || link_assignment == "SIAO2" ) {

//here we need the link-only parts of density matrices D_X,D_Y and their corresponding Coulomb and exchange matrices (computed earlier)
        std::shared_ptr<Matrix> D_X(D_A->clone());
        std::shared_ptr<Matrix> D_Y(D_A->clone());
        D_X = linalg::doublet(matrices_["thislinkA"], matrices_["thislinkA"], false, true);
        D_Y = linalg::doublet(matrices_["thislinkB"], matrices_["thislinkB"], false, true);
        auto J_X(matrices_["JLA"]->clone());
        auto K_X(matrices_["KLA"]->clone());
        auto J_Y(matrices_["JLB"]->clone());
        auto K_Y(matrices_["KLB"]->clone());
       
        // ==> Generalized ESP (Flat and Exchange) <== //
        std::shared_ptr<Matrix> K_AOY = matrices_["K_AOY"];
        std::shared_ptr<Matrix> K_XOB = matrices_["K_XOB"];
        K_XOB->transpose_this();
        std::shared_ptr<Matrix> J_P_YAY = matrices_["J_P_YAY"];
        std::shared_ptr<Matrix> J_P_XBX = matrices_["J_P_XBX"];
       
        std::map<std::string, std::shared_ptr<Matrix> > mapA;
        mapA["Cocc_A"] = Locc_A;
        mapA["Cvir_A"] = Cvir_A;
        mapA["S"] = S;
        mapA["D_A"] = D_A;
        mapA["V_A"] = V_A;
        mapA["J_A"] = J_A;
        mapA["K_A"] = K_A;
        mapA["D_B"] = D_B;
        mapA["V_B"] = V_B;
        mapA["J_B"] = J_B;
        mapA["K_B"] = K_B;
        mapA["D_X"] = D_X;
        mapA["J_X"] = J_X;
        mapA["K_X"] = K_X;
        mapA["D_Y"] = D_Y;
        mapA["J_Y"] = J_Y;
        mapA["K_Y"] = K_Y;
        mapA["J_O"] = J_O;
        mapA["K_O"] = K_O;
        mapA["K_AOY"] = K_AOY;
        mapA["J_P"] = J_P_A;
        mapA["J_PYAY"] = J_P_YAY;

        wBT = build_ind_pot(mapA);
        uBT = build_exch_ind_pot_avg(mapA);

        K_O->transpose_this();

        std::map<std::string, std::shared_ptr<Matrix> > mapB;
        mapB["Cocc_A"] = Locc_B;
        mapB["Cvir_A"] = Cvir_B;
        mapB["S"] = S;
        mapB["D_A"] = D_B;
        mapB["V_A"] = V_B;
        mapB["J_A"] = J_B;
        mapB["K_A"] = K_B;
        mapB["D_B"] = D_A;
        mapB["V_B"] = V_A;
        mapB["J_B"] = J_A;
        mapB["K_B"] = K_A;
        mapB["D_X"] = D_Y;
        mapB["J_X"] = J_Y;
        mapB["K_X"] = K_Y;
        mapB["D_Y"] = D_X;
        mapB["J_Y"] = J_X;
        mapB["K_Y"] = K_X;
        mapB["J_O"] = J_O;
        mapB["K_O"] = K_O;
        mapB["K_AOY"] = K_XOB;
        mapB["J_P"] = J_P_B;
        mapB["J_PYAY"] = J_P_XBX;

        wAT = build_ind_pot(mapB);
        uAT = build_exch_ind_pot_avg(mapB);

        K_O->transpose_this();

    }
    else {

    // ==> Generalized ESP (Flat and Exchange) <== //

        std::map<std::string, std::shared_ptr<Matrix> > mapA;
        mapA["Cocc_A"] = Locc_A;
        mapA["Cvir_A"] = Cvir_A;
        mapA["Cocc_B"] = Locc_B;
        mapA["Cvir_B"] = Cvir_B;
        mapA["S"] = S;
        mapA["D_A"] = D_A;
        mapA["V_A"] = V_A;
        mapA["J_A"] = J_A;
        mapA["K_A"] = K_A;
        mapA["D_B"] = D_B;
        mapA["V_B"] = V_B;
        mapA["J_B"] = J_B;
        mapA["K_B"] = K_B;
        mapA["J_O"] = J_O;
        mapA["K_O"] = K_O;
        mapA["J_P"] = J_P_A;

        wBT = build_ind_pot(mapA);
        uBT = build_exch_ind_pot(mapA);

        K_O->transpose_this();

        std::map<std::string, std::shared_ptr<Matrix> > mapB;
        mapB["Cocc_A"] = Locc_B;
        mapB["Cvir_A"] = Cvir_B;
        mapB["Cocc_B"] = Locc_A;
        mapB["Cvir_B"] = Cvir_A;
        mapB["S"] = S;
        mapB["D_A"] = D_B;
        mapB["V_A"] = V_B;
        mapB["J_A"] = J_B;
        mapB["K_A"] = K_B;
        mapB["D_B"] = D_A;
        mapB["V_B"] = V_A;
        mapB["J_B"] = J_A;
        mapB["K_B"] = K_A;
        mapB["J_O"] = J_O;
        mapB["K_O"] = K_O;
        mapB["J_P"] = J_P_B;

        wAT = build_ind_pot(mapB);
        uAT = build_exch_ind_pot(mapB);

        K_O->transpose_this();
    }
    // log the link assignment
    // outfile->Printf("    F-SAPT Induction with link assignment: %s\n\n", link_assignment.c_str());

    // V_A->set_name("V_A");
    // V_A->print();
    // Locc_B     ->set_name("Cocc_A") ;
    // Cvir_B     ->set_name("Cvir_A") ;
    // Locc_A     ->set_name("Cocc_B") ;
    // Cvir_A     ->set_name("Cvir_B") ;
    // S          ->set_name("S")      ;
    // D_B        ->set_name("D_A")    ;
    // V_B        ->set_name("V_A")    ;
    // J_B        ->set_name("J_A")    ;
    // K_B        ->set_name("K_A")    ;
    // D_A        ->set_name("D_B")    ;
    // V_A        ->set_name("V_B")    ;
    // J_A        ->set_name("J_B")    ;
    // K_A        ->set_name("K_B")    ;
    // J_O        ->set_name("J_O")    ;
    // K_O        ->set_name("K_O")    ;
    // J_P_B      ->set_name("J_P_B")  ;
    // J_P_A      ->set_name("J_P_A")  ;
    // Locc_B     ->print()            ;
    // Cvir_B     ->print()            ;
    // Locc_A     ->print()            ;
    // Cvir_A     ->print()            ;
    // S          ->print()            ;
    // D_B        ->print()            ;
    // V_B        ->print()            ;
    // J_B        ->print()            ;
    // K_B        ->print()            ;
    // D_A        ->print()            ;
    // V_A        ->print()            ;
    // J_A        ->print()            ;
    // K_A        ->print()            ;
    // J_O        ->print()            ;
    // K_O        ->print()            ;
    // J_P_B      ->print()            ;
    // J_P_A      ->print()            ;

    // Remove prints after debugging
    // wBT->set_name("wBT");
    // uBT->set_name("uBT");
    // wAT->set_name("wAT");
    // uAT->set_name("uAT");
    // wBT->print();
    // uBT->print();
    // wAT->print();
    // uAT->print();

    double** wATp = wAT->pointer();
    double** uATp = uAT->pointer();
    double** wBTp = wBT->pointer();
    double** uBTp = uBT->pointer();

    // ==> Uncoupled Targets <== //

    auto Ind20u_AB_terms = std::make_shared<Matrix>("Ind20 [A<-B] (a x B)", na, nB + nb1 + 1); // add one entry for external potential 
    auto Ind20u_BA_terms = std::make_shared<Matrix>("Ind20 [B<-A] (A x b)", nA + na1 + 1, nb); // add one entry for external potential
    double** Ind20u_AB_termsp = Ind20u_AB_terms->pointer();
    double** Ind20u_BA_termsp = Ind20u_BA_terms->pointer();

    double Ind20u_AB = 0.0;
    double Ind20u_BA = 0.0;

    auto ExchInd20u_AB_terms = std::make_shared<Matrix>("ExchInd20 [A<-B] (a x B)", na, nB + nb1 + 1); // add one for external potential 
    auto ExchInd20u_BA_terms = std::make_shared<Matrix>("ExchInd20 [B<-A] (A x b)", nA + na1 + 1, nb); // add one for external potential
    double** ExchInd20u_AB_termsp = ExchInd20u_AB_terms->pointer();
    double** ExchInd20u_BA_termsp = ExchInd20u_BA_terms->pointer();

    double ExchInd20u_AB = 0.0;
    double ExchInd20u_BA = 0.0;

    int sna = 0;
    int snB = 0;
    int snb = 0;
    int snA = 0;

    if (options_.get_bool("SSAPT0_SCALE")) {
    // This will NOT work with ISAPT(SAOn/SIAOn) and I (Konrad) think that's OK.
        sna = na;
        snB = nB;
        snb = nb;
        snA = nA;
    }

    std::shared_ptr<Matrix> sExchInd20u_AB_terms =
        std::make_shared<Matrix>("sExchInd20 [A<-B] (a x B)", sna, snB + snb + 1); // add one entry for external potential
    std::shared_ptr<Matrix> sExchInd20u_BA_terms =
        std::make_shared<Matrix>("sExchInd20 [B<-A] (A x b)", snA + sna + 1, snb); // add one entry for external potential
    double** sExchInd20u_AB_termsp = sExchInd20u_AB_terms->pointer();
    double** sExchInd20u_BA_termsp = sExchInd20u_BA_terms->pointer();

    double sExchInd20u_AB = 0.0;
    double sExchInd20u_BA = 0.0;

    auto Indu_AB_terms = std::make_shared<Matrix>("Ind [A<-B] (a x B)", na, nB + nb1 + 1); // add one entry for external potential
    auto Indu_BA_terms = std::make_shared<Matrix>("Ind [B<-A] (A x b)", nA + na1 + 1, nb); // add one entry for external potential
    double** Indu_AB_termsp = Indu_AB_terms->pointer();
    double** Indu_BA_termsp = Indu_BA_terms->pointer();

    double Indu_AB = 0.0;
    double Indu_BA = 0.0;

    auto sIndu_AB_terms = std::make_shared<Matrix>("sInd [A<-B] (a x B)", sna, snB + snb + 1); // add one entry for external potential 
    auto sIndu_BA_terms = std::make_shared<Matrix>("sInd [B<-A] (A x b)", snA + sna + 1, snb); // add one entry for external potential
    double** sIndu_AB_termsp = sIndu_AB_terms->pointer();
    double** sIndu_BA_termsp = sIndu_BA_terms->pointer();

    double sIndu_AB = 0.0;
    double sIndu_BA = 0.0;

    // ==> A <- B Uncoupled <== //

    // Add the external potential
    if (reference_->has_potential_variable("B")) {
        std::shared_ptr<Matrix> Var = linalg::triplet(Cocc_A, matrices_["VB_extern"], Cvir_A, true, false, false);
        dfh_->write_disk_tensor("WBar", Var, {(size_t) nB + nb1, (size_t) nB + nb1 + 1});
    }

    else { // Add empty matrix
        std::shared_ptr<Matrix> Var = std::make_shared<Matrix>("zero", na, nr);
        Var->zero();
        dfh_->write_disk_tensor("WBar", Var, {(size_t) nB + nb1, (size_t) nB + nb1 + 1});
    }

    for (size_t B = 0; B < nB + nb1 + 1; B++) { // add one for external potential
        // ESP
        dfh_->fill_tensor("WBar", wB, {B, B + 1});

        // Uncoupled amplitude
        for (int a = 0; a < na; a++) {
            for (int r = 0; r < nr; r++) {
                xAp[a][r] = wBp[a][r] / (eap[a] - erp[r]);
            }
        }

        // Backtransform the amplitude to LO
        std::shared_ptr<Matrix> x2A = linalg::doublet(Uocc_A, xA, true, false);
        double** x2Ap = x2A->pointer();

        // Zip up the Ind20 contributions
        for (int a = 0; a < na; a++) {
            double Jval = 2.0 * C_DDOT(nr, x2Ap[a], 1, wBTp[a], 1);
            double Kval = 2.0 * C_DDOT(nr, x2Ap[a], 1, uBTp[a], 1);
            Ind20u_AB_termsp[a][B] = Jval;
            Ind20u_AB += Jval;
            ExchInd20u_AB_termsp[a][B] = Kval;
            ExchInd20u_AB += Kval;
            if (options_.get_bool("SSAPT0_SCALE")) {
                sExchInd20u_AB_termsp[a][B] = Kval;
                sExchInd20u_AB += Kval;
                sIndu_AB_termsp[a][B] = Jval + Kval;
                sIndu_AB += Jval + Kval;
            }

            Indu_AB_termsp[a][B] = Jval + Kval;
            Indu_AB += Jval + Kval;
        }
    }

    // ==> B <- A Uncoupled <== //

    // Add the external potential
    if (reference_->has_potential_variable("A")) {
        std::shared_ptr<Matrix> Vbs = linalg::triplet(Cocc_B, matrices_["VA_extern"], Cvir_B, true, false, false);
        dfh_->write_disk_tensor("WAbs", Vbs, {(size_t) nA + na1, (size_t) nA + na1 + 1});
    }

    else { // add empty matrix
        std::shared_ptr<Matrix> Vbs = std::make_shared<Matrix>("zero", nb, ns);
        Vbs->zero();
        dfh_->write_disk_tensor("WAbs", Vbs, {(size_t) nA + na1, (size_t) nA + na1 + 1});
    }


    for (size_t A = 0; A < nA + na1 + 1; A++) { // add one for extenral potential
        // ESP
        dfh_->fill_tensor("WAbs", wA, {A, A + 1});

        // Uncoupled amplitude
        for (int b = 0; b < nb; b++) {
            for (int s = 0; s < ns; s++) {
                xBp[b][s] = wAp[b][s] / (ebp[b] - esp[s]);
            }
        }

        // Backtransform the amplitude to LO
        std::shared_ptr<Matrix> x2B = linalg::doublet(Uocc_B, xB, true, false);
        double** x2Bp = x2B->pointer();

        // Zip up the Ind20 contributions
        for (int b = 0; b < nb; b++) {
            double Jval = 2.0 * C_DDOT(ns, x2Bp[b], 1, wATp[b], 1);
            double Kval = 2.0 * C_DDOT(ns, x2Bp[b], 1, uATp[b], 1);
            Ind20u_BA_termsp[A][b] = Jval;
            Ind20u_BA += Jval;
            ExchInd20u_BA_termsp[A][b] = Kval;
            ExchInd20u_BA += Kval;
            if (options_.get_bool("SSAPT0_SCALE")) {
                sExchInd20u_BA_termsp[A][b] = Kval;
                sExchInd20u_BA += Kval;
                sIndu_BA_termsp[A][b] = Jval + Kval;
                sIndu_BA += Jval + Kval;
            }
            Indu_BA_termsp[A][b] = Jval + Kval;
            Indu_BA += Jval + Kval;
        }
    }

    double Ind20u = Ind20u_AB + Ind20u_BA;
    outfile->Printf("    Ind20,u (A<-B)      = %18.12lf [Eh]\n", Ind20u_AB);
    outfile->Printf("    Ind20,u (B<-A)      = %18.12lf [Eh]\n", Ind20u_BA);
    outfile->Printf("    Ind20,u             = %18.12lf [Eh]\n", Ind20u);
    // fflush(outfile);

    double ExchInd20u = ExchInd20u_AB + ExchInd20u_BA;
    outfile->Printf("    Exch-Ind20,u (A<-B) = %18.12lf [Eh]\n", ExchInd20u_AB);
    outfile->Printf("    Exch-Ind20,u (B<-A) = %18.12lf [Eh]\n", ExchInd20u_BA);
    outfile->Printf("    Exch-Ind20,u        = %18.12lf [Eh]\n", ExchInd20u);
    outfile->Printf("\n");
    // fflush(outfile);
    if (options_.get_bool("SSAPT0_SCALE")) {
        double sExchInd20u = sExchInd20u_AB + sExchInd20u_BA;
        outfile->Printf("    sExch-Ind20,u (A<-B) = %18.12lf [Eh]\n", sExchInd20u_AB);
        outfile->Printf("    sExch-Ind20,u (B<-A) = %18.12lf [Eh]\n", sExchInd20u_BA);
        outfile->Printf("    sExch-Ind20,u        = %18.12lf [Eh]\n", sExchInd20u);
        outfile->Printf("\n");
    }

    double Ind = Ind20u + ExchInd20u;
    std::shared_ptr<Matrix> Ind_AB_terms = Indu_AB_terms;
    std::shared_ptr<Matrix> Ind_BA_terms = Indu_BA_terms;
    std::shared_ptr<Matrix> sInd_AB_terms = sIndu_AB_terms;
    std::shared_ptr<Matrix> sInd_BA_terms = sIndu_BA_terms;

    if (ind_resp) {
        outfile->Printf("  COUPLED INDUCTION (You asked for it!):\n\n");

        // ==> Coupled Targets <== //

        auto Ind20r_AB_terms = std::make_shared<Matrix>("Ind20 [A<-B] (a x B)", na, nB + nb1 + 1); // add one for external potential 
        auto Ind20r_BA_terms = std::make_shared<Matrix>("Ind20 [B<-A] (A x b)", nA + na1 + 1, nb); // add one for external potential
        double** Ind20r_AB_termsp = Ind20r_AB_terms->pointer();
        double** Ind20r_BA_termsp = Ind20r_BA_terms->pointer();

        double Ind20r_AB = 0.0;
        double Ind20r_BA = 0.0;

        auto ExchInd20r_AB_terms = std::make_shared<Matrix>("ExchInd20 [A<-B] (a x B)", na, nB + nb1 + 1); // add one for external
                                                                                                          // potential
        auto ExchInd20r_BA_terms = std::make_shared<Matrix>("ExchInd20 [B<-A] (A x b)", nA + na1 + 1, nb); // add one for external
                                                                                                          // potential
        double** ExchInd20r_AB_termsp = ExchInd20r_AB_terms->pointer();
        double** ExchInd20r_BA_termsp = ExchInd20r_BA_terms->pointer();

        double ExchInd20r_AB = 0.0;
        double ExchInd20r_BA = 0.0;

        auto Indr_AB_terms = std::make_shared<Matrix>("Ind [A<-B] (a x B)", na, nB + nb1 + 1); // add one entry for external potential 
        auto Indr_BA_terms = std::make_shared<Matrix>("Ind [B<-A] (A x b)", nA + na1 + 1, nb); // add one entry for external potential
        double** Indr_AB_termsp = Indr_AB_terms->pointer();
        double** Indr_BA_termsp = Indr_BA_terms->pointer();

        double Indr_AB = 0.0;
        double Indr_BA = 0.0;

        // => JK Object <= //

        // TODO: Account for 2-index overhead in memory
        auto nso = primary_->nbf();
        auto jk_memory = (long int)doubles_;
        jk_memory -= 24 * nso * nso;
        jk_memory -= 4 * na * nso;
        jk_memory -= 4 * nb * nso;
        if (jk_memory < 0L) {
            throw PSIEXCEPTION("Too little static memory for FISAPT::induction");
        }

        std::shared_ptr<JK> jk =
            JK::build_JK(primary_, reference_->get_basisset("DF_BASIS_SCF"), options_, false, (size_t)jk_memory);

        jk->set_memory((size_t)jk_memory);
        jk->set_do_J(true);
        jk->set_do_K(true);
        jk->initialize();
        jk->print_header();

        // ==> Master Loop over perturbing atoms <== //

        int nC = std::max(nA + na1 + 1, nB + nb1 + 1); // add one for external potential

        for (size_t C = 0; C < nC; C++) {
            if (C < nB + nb1) dfh_->fill_tensor("WBar", wB, {C, C + 1});
            if (C < nA + na1) dfh_->fill_tensor("WAbs", wB, {C, C + 1});

            outfile->Printf("    Responses for (A <- Source B = %3zu) and (B <- Source A = %3zu)\n\n",
                            (C < nB + nb1 ? C : nB + nb1), (C < nA + na1 ? C : nA + na1));

            auto cphf = std::make_shared<CPHF_FISAPT>();

            // Effective constructor
            cphf->delta_ = options_.get_double("CPHF_R_CONVERGENCE");
            cphf->maxiter_ = options_.get_int("MAXITER");
            cphf->jk_ = jk;

            cphf->w_A_ = wB;  // Reversal of convention
            cphf->Cocc_A_ = Cocc_A;
            cphf->Cvir_A_ = Cvir_A;
            cphf->eps_occ_A_ = eps_occ_A;
            cphf->eps_vir_A_ = eps_vir_A;

            cphf->w_B_ = wA;  // Reversal of convention
            cphf->Cocc_B_ = Cocc_B;
            cphf->Cvir_B_ = Cvir_B;
            cphf->eps_occ_B_ = eps_occ_B;
            cphf->eps_vir_B_ = eps_vir_B;

            // Gogo CPKS
            cphf->compute_cphf();

            xA = cphf->x_A_;
            xB = cphf->x_B_;

            xA->scale(-1.0);
            xB->scale(-1.0);

            if (C < nB + nb1 + 1) {
                // Backtransform the amplitude to LO
                std::shared_ptr<Matrix> x2A = linalg::doublet(Uocc_A, xA, true, false);
                double** x2Ap = x2A->pointer();

                // Zip up the Ind20 contributions
                for (int a = 0; a < na; a++) {
                    double Jval = 2.0 * C_DDOT(nr, x2Ap[a], 1, wBTp[a], 1);
                    double Kval = 2.0 * C_DDOT(nr, x2Ap[a], 1, uBTp[a], 1);
                    Ind20r_AB_termsp[a][C] = Jval;
                    Ind20r_AB += Jval;
                    ExchInd20r_AB_termsp[a][C] = Kval;
                    ExchInd20r_AB += Kval;
                    Indr_AB_termsp[a][C] = Jval + Kval;
                    Indr_AB += Jval + Kval;
                }
            }

            if (C < nA + na1 + 1) {
                // Backtransform the amplitude to LO
                std::shared_ptr<Matrix> x2B = linalg::doublet(Uocc_B, xB, true, false);
                double** x2Bp = x2B->pointer();

                // Zip up the Ind20 contributions
                for (int b = 0; b < nb; b++) {
                    double Jval = 2.0 * C_DDOT(ns, x2Bp[b], 1, wATp[b], 1);
                    double Kval = 2.0 * C_DDOT(ns, x2Bp[b], 1, uATp[b], 1);
                    Ind20r_BA_termsp[C][b] = Jval;
                    Ind20r_BA += Jval;
                    ExchInd20r_BA_termsp[C][b] = Kval;
                    ExchInd20r_BA += Kval;
                    Indr_BA_termsp[C][b] = Jval + Kval;
                    Indr_BA += Jval + Kval;
                }
            }
        }

        double Ind20r = Ind20r_AB + Ind20r_BA;
        outfile->Printf("    Ind20,r (A<-B)      = %18.12lf [Eh]\n", Ind20r_AB);
        outfile->Printf("    Ind20,r (B<-A)      = %18.12lf [Eh]\n", Ind20r_BA);
        outfile->Printf("    Ind20,r             = %18.12lf [Eh]\n", Ind20r);
        // fflush(outfile);

        double ExchInd20r = ExchInd20r_AB + ExchInd20r_BA;
        outfile->Printf("    Exch-Ind20,r (A<-B) = %18.12lf [Eh]\n", ExchInd20r_AB);
        outfile->Printf("    Exch-Ind20,r (B<-A) = %18.12lf [Eh]\n", ExchInd20r_BA);
        outfile->Printf("    Exch-Ind20,r        = %18.12lf [Eh]\n", ExchInd20r);
        outfile->Printf("\n");
        // fflush(outfile);

        Ind = Ind20r + ExchInd20r;
        Ind_AB_terms = Indr_AB_terms;
        Ind_BA_terms = Indr_BA_terms;
    }

    // => Induction scaling <= //

    if (ind_scale) {
        double dHF = 0.0;
        if (scalars_["HF"] != 0.0) {
            dHF = scalars_["HF"] - scalars_["Elst10,r"] - scalars_["Exch10"] - scalars_["Ind20,r"] -
                  scalars_["Exch-Ind20,r"];
        }
        double IndHF = scalars_["Ind20,r"] + scalars_["Exch-Ind20,r"] + dHF;
        double IndSAPT0 = scalars_["Ind20,r"] + scalars_["Exch-Ind20,r"];

        double Sdelta = IndHF / IndSAPT0;
        double SrAB = (ind_resp ? 1.0
                                : (scalars_["Ind20,r (A<-B)"] + scalars_["Exch-Ind20,r (A<-B)"]) /
                                      (scalars_["Ind20,u (A<-B)"] + scalars_["Exch-Ind20,u (A<-B)"]));
        double SrBA = (ind_resp ? 1.0
                                : (scalars_["Ind20,r (B<-A)"] + scalars_["Exch-Ind20,r (B<-A)"]) /
                                      (scalars_["Ind20,u (B<-A)"] + scalars_["Exch-Ind20,u (B<-A)"]));

        double sIndHF = scalars_["Ind20,r"] + scalars_["sExch-Ind20,r"] + dHF;
        double sIndSAPT0 = scalars_["Ind20,r"] + scalars_["sExch-Ind20,r"];

        double sSdelta = sIndHF / IndSAPT0;

        double sSrAB = (ind_resp ? 1.0
                                 : (scalars_["Ind20,r (A<-B)"] + scalars_["sExch-Ind20,r (A<-B)"]) /
                                       (scalars_["Ind20,u (A<-B)"] + scalars_["sExch-Ind20,u (A<-B)"]));
        double sSrBA = (ind_resp ? 1.0
                                 : (scalars_["Ind20,r (B<-A)"] + scalars_["sExch-Ind20,r (B<-A)"]) /
                                       (scalars_["Ind20,u (B<-A)"] + scalars_["sExch-Ind20,u (B<-A)"]));

        outfile->Printf("    Scaling for delta HF        = %11.3E\n", Sdelta);
        outfile->Printf("    Scaling for response (A<-B) = %11.3E\n", SrAB);
        outfile->Printf("    Scaling for response (B<-A) = %11.3E\n", SrBA);
        outfile->Printf("    Scaling for total (A<-B)    = %11.3E\n", Sdelta * SrAB);
        outfile->Printf("    Scaling for total (B<-A)    = %11.3E\n", Sdelta * SrBA);
        outfile->Printf("\n");

        Ind_AB_terms->scale(Sdelta * SrAB);
        Ind_BA_terms->scale(Sdelta * SrBA);
        Ind20u_AB_terms->scale(Sdelta * SrAB);
        ExchInd20u_AB_terms->scale(Sdelta * SrAB);
        Ind20u_BA_terms->scale(Sdelta * SrBA);
        ExchInd20u_BA_terms->scale(Sdelta * SrBA);
        sInd_AB_terms->scale(sSdelta * SrAB);
        sInd_BA_terms->scale(sSdelta * SrBA);
    }

    matrices_["IndAB_AB"] = std::make_shared<Matrix>("IndAB_AB", nA + na1 + 1, nB + nb1 + 1); // add one entry for external
                                                                                            // potentials in A and B
    matrices_["IndBA_AB"] = std::make_shared<Matrix>("IndBA_AB", nA + na1 + 1, nB + nb1 + 1); // add one entry for external
                                                                                            // potentials in A and B
    matrices_["Ind20u_AB_terms"] = std::make_shared<Matrix>("Ind20uAB_AB", nA + na1 + 1, nB + nb1 + 1); // add one for external pot
    matrices_["ExchInd20u_AB_terms"] = std::make_shared<Matrix>("ExchInd20uAB_AB", nA + na1 + 1, nB + nb1 + 1); // add one for external
    matrices_["Ind20u_BA_terms"] = std::make_shared<Matrix>("Ind20uBA_AB", nA + na1 + 1, nB + nb1 + 1); // add one for external
    matrices_["ExchInd20u_BA_terms"] = std::make_shared<Matrix>("ExchInd20uBA_AB", nA + na1 + 1, nB + nb1 + 1); // add one for external
    double** EABp = matrices_["IndAB_AB"]->pointer();
    double** EBAp = matrices_["IndBA_AB"]->pointer();
    double** Ind20ABp = matrices_["Ind20u_AB_terms"]->pointer();
    double** ExchInd20ABp = matrices_["ExchInd20u_AB_terms"]->pointer();
    double** Ind20BAp = matrices_["Ind20u_BA_terms"]->pointer();
    double** ExchInd20BAp = matrices_["ExchInd20u_BA_terms"]->pointer();
    double** EAB2p = Ind_AB_terms->pointer();
    double** EBA2p = Ind_BA_terms->pointer();
    double** Ind20AB2p = Ind20u_AB_terms->pointer();
    double** ExchInd20AB2p = ExchInd20u_AB_terms->pointer();
    double** Ind20BA2p = Ind20u_BA_terms->pointer();
    double** ExchInd20BA2p = ExchInd20u_BA_terms->pointer();

    for (int a = 0; a < na; a++) {
        for (int B = 0; B < nB + nb1 + 1; B++) { // add one for external potential
            EABp[a + nA][B] = EAB2p[a][B];
            Ind20ABp[a + nA][B] = Ind20AB2p[a][B];
            ExchInd20ABp[a + nA][B] = ExchInd20AB2p[a][B];
        }
    }

    for (int A = 0; A < nA + na1 + 1; A++) { // add one for external potential
        for (int b = 0; b < nb; b++) {
            EBAp[A][b + nB] = EBA2p[A][b];
            Ind20BAp[A][b + nB] = Ind20BA2p[A][b];
            ExchInd20BAp[A][b + nB] = ExchInd20BA2p[A][b];
        }
    }

    matrices_["sIndAB_AB"] = std::make_shared<Matrix>("sIndAB_AB", snA + sna + 1, snB + snb + 1); // add one entry for external
                                                                                                  // potentials in A and B
    matrices_["sIndBA_AB"] = std::make_shared<Matrix>("sIndBA_AB", snA + sna + 1, snB + snb + 1); // add one entry for external
                                                                                                  // potentials in A and B
    double** sEABp = matrices_["sIndAB_AB"]->pointer();
    double** sEBAp = matrices_["sIndBA_AB"]->pointer();
    double** sEAB2p = sInd_AB_terms->pointer();
    double** sEBA2p = sInd_BA_terms->pointer();

    for (int a = 0; a < sna; a++) {
        for (int B = 0; B < snB + snb + 1; B++) { // add one for external potential
            sEABp[a + snA][B] = sEAB2p[a][B];
        }
    }

    for (int A = 0; A < snA + sna + 1; A++) { // add one for external potential
        for (int b = 0; b < snb; b++) {
            sEBAp[A][b + snB] = sEBA2p[A][b];
        }
    }
    // We're done with dfh_'s integrals
    dfh_->clear_all();
}

// Compute fragment-fragment partitioning of dispersion contribution
// This is also where the modified exchange-dispersion expressions for the SAOn/SIAOn ISAPT algorithms are coded.
void FISAPT::fdisp_einsums() {
    outfile->Printf("  ==> F-SAPT Dispersion <==\n\n");

    // => Auxiliary Basis Set <= //

    std::shared_ptr<BasisSet> auxiliary = reference_->get_basisset("DF_BASIS_SAPT");

    // => Sizing <= //

    std::shared_ptr<Molecule> mol = primary_->molecule();
    int nn = primary_->nbf();
    int nA = mol->natom();
    int nB = mol->natom();
    int na = matrices_["Laocc0A"]->colspi()[0];
    int nb = matrices_["Laocc0B"]->colspi()[0];
    int nr = matrices_["Cvir0A"]->colspi()[0];
    int ns = matrices_["Cvir0B"]->colspi()[0];
    int nQ = auxiliary->nbf();
    size_t naQ = na * (size_t)nQ;
    size_t nbQ = nb * (size_t)nQ;

//for the SAOn/SIAOn variants, we sometimes need na1 = na+1 (with link orbital) and sometimes na (without) - be careful with this!
    std::string link_assignment = options_.get_str("FISAPT_LINK_ASSIGNMENT");
    int na1 = na;
    int nb1 = nb;
    if (link_assignment == "SAO0" || link_assignment == "SAO1" || link_assignment == "SAO2" || link_assignment == "SIAO0" || link_assignment == "SIAO1" || link_assignment == "SIAO2") {
        na1 = na + 1;
        nb1 = nb + 1;
    }

    int nfa = matrices_["Lfocc0A"]->colspi()[0];
    int nfb = matrices_["Lfocc0B"]->colspi()[0];

    int nT = 1;
#ifdef _OPENMP
    nT = Process::environment.get_n_threads();
#endif

    // => Targets <= //

    matrices_["Disp_AB"] = std::make_shared<Matrix>("Disp_AB", nA + nfa + na1 + 1, nB + nfb + nb1 + 1); // add one entry for external
                                                                                                      // potentials in A and B
    double** Ep = matrices_["Disp_AB"]->pointer();

    int snA = 0;
    int snfa = 0;
    int sna = 0;
    int snB = 0;
    int snfb = 0;
    int snb = 0;

    if (options_.get_bool("SSAPT0_SCALE")) {
        snA = nA;
        snfa = nfa;
        sna = na;
        snB = nB;
        snfb = nfb;
        snb = nb;
    }

    matrices_["sDisp_AB"] = std::make_shared<Matrix>("Disp_AB", snA + snfa + sna + 1, snB + snfb + snb + 1); // add one entry for
                                                                                                             // external potenrials
                                                                                                             // in A and B
    double** sEp = matrices_["sDisp_AB"]->pointer();

    // => Stashed Variables <= //

    std::shared_ptr<Matrix> S = matrices_["S"];
    std::shared_ptr<Matrix> D_A = matrices_["D_A"];
    std::shared_ptr<Matrix> P_A = matrices_["P_A"];
    std::shared_ptr<Matrix> V_A = matrices_["V_A"];
    std::shared_ptr<Matrix> J_A = matrices_["J_A"];
    std::shared_ptr<Matrix> K_A = matrices_["K_A"];
    std::shared_ptr<Matrix> D_B = matrices_["D_B"];
    std::shared_ptr<Matrix> P_B = matrices_["P_B"];
    std::shared_ptr<Matrix> V_B = matrices_["V_B"];
    std::shared_ptr<Matrix> J_B = matrices_["J_B"];
    std::shared_ptr<Matrix> K_B = matrices_["K_B"];
    std::shared_ptr<Matrix> K_O = matrices_["K_O"];

    bool parperp = options_.get_bool("FISAPT_EXCH_PARPERP");

    std::shared_ptr<Matrix> Caocc_A = matrices_["Caocc0A"];
    std::shared_ptr<Matrix> Caocc_B = matrices_["Caocc0B"];
    std::shared_ptr<Matrix> Cavir_A = matrices_["Cvir0A"];
    std::shared_ptr<Matrix> Cavir_B = matrices_["Cvir0B"];

    std::shared_ptr<Vector> eps_aocc_A = vectors_["eps_aocc0A"];
    std::shared_ptr<Vector> eps_aocc_B = vectors_["eps_aocc0B"];
    std::shared_ptr<Vector> eps_avir_A = vectors_["eps_vir0A"];
    std::shared_ptr<Vector> eps_avir_B = vectors_["eps_vir0B"];

    std::shared_ptr<Matrix> Uaocc_A = matrices_["Uaocc0A"];
    std::shared_ptr<Matrix> Uaocc_B = matrices_["Uaocc0B"];

    // => Auxiliary C matrices <= //

    std::shared_ptr<Matrix> Cr1 = linalg::triplet(D_B, S, Cavir_A);
    Cr1->scale(-1.0);
    Cr1->add(Cavir_A);
    std::shared_ptr<Matrix> Cs1 = linalg::triplet(D_A, S, Cavir_B);
    Cs1->scale(-1.0);
    Cs1->add(Cavir_B);
    std::shared_ptr<Matrix> Ca2 = linalg::triplet(D_B, S, Caocc_A);
    std::shared_ptr<Matrix> Cb2 = linalg::triplet(D_A, S, Caocc_B);

    std::shared_ptr<Matrix> Cr3 = linalg::triplet(D_B, S, Cavir_A);
    std::shared_ptr<Matrix> CrX = linalg::triplet(linalg::triplet(D_A, S, D_B), S, Cavir_A);
    Cr3->subtract(CrX);
    Cr3->scale(2.0);
    std::shared_ptr<Matrix> Cs3 = linalg::triplet(D_A, S, Cavir_B);
    std::shared_ptr<Matrix> CsX = linalg::triplet(linalg::triplet(D_B, S, D_A), S, Cavir_B);
    Cs3->subtract(CsX);
    Cs3->scale(2.0);

    std::shared_ptr<Matrix> Ca4 = linalg::triplet(linalg::triplet(D_A, S, D_B), S, Caocc_A);
    Ca4->scale(-2.0);
    std::shared_ptr<Matrix> Cb4 = linalg::triplet(linalg::triplet(D_B, S, D_A), S, Caocc_B);
    Cb4->scale(-2.0);

    // => Auxiliary V matrices <= //

    std::shared_ptr<Matrix> Jbr = linalg::triplet(Caocc_B, J_A, Cavir_A, true, false, false);
    Jbr->scale(2.0);
    std::shared_ptr<Matrix> Kbr = linalg::triplet(Caocc_B, K_A, Cavir_A, true, false, false);
    Kbr->scale(-1.0);

    std::shared_ptr<Matrix> Jas = linalg::triplet(Caocc_A, J_B, Cavir_B, true, false, false);
    Jas->scale(2.0);
    std::shared_ptr<Matrix> Kas = linalg::triplet(Caocc_A, K_B, Cavir_B, true, false, false);
    Kas->scale(-1.0);

    std::shared_ptr<Matrix> KOas = linalg::triplet(Caocc_A, K_O, Cavir_B, true, false, false);
    KOas->scale(1.0);
    std::shared_ptr<Matrix> KObr = linalg::triplet(Caocc_B, K_O, Cavir_A, true, true, false);
    KObr->scale(1.0);

    std::shared_ptr<Matrix> JBas = linalg::triplet(linalg::triplet(Caocc_A, S, D_B, true, false, false), J_A, Cavir_B);
    JBas->scale(-2.0);
    std::shared_ptr<Matrix> JAbr = linalg::triplet(linalg::triplet(Caocc_B, S, D_A, true, false, false), J_B, Cavir_A);
    JAbr->scale(-2.0);

    std::shared_ptr<Matrix> Jbs = linalg::triplet(Caocc_B, J_A, Cavir_B, true, false, false);
    Jbs->scale(4.0);
    std::shared_ptr<Matrix> Jar = linalg::triplet(Caocc_A, J_B, Cavir_A, true, false, false);
    Jar->scale(4.0);

    std::shared_ptr<Matrix> JAas = linalg::triplet(linalg::triplet(Caocc_A, J_B, D_A, true, false, false), S, Cavir_B);
    JAas->scale(-2.0);
    std::shared_ptr<Matrix> JBbr = linalg::triplet(linalg::triplet(Caocc_B, J_A, D_B, true, false, false), S, Cavir_A);
    JBbr->scale(-2.0);

    // Get your signs right Hesselmann!
    std::shared_ptr<Matrix> Vbs = linalg::triplet(Caocc_B, V_A, Cavir_B, true, false, false);
    Vbs->scale(2.0);
    std::shared_ptr<Matrix> Var = linalg::triplet(Caocc_A, V_B, Cavir_A, true, false, false);
    Var->scale(2.0);
    std::shared_ptr<Matrix> VBas = linalg::triplet(linalg::triplet(Caocc_A, S, D_B, true, false, false), V_A, Cavir_B);
    VBas->scale(-1.0);
    std::shared_ptr<Matrix> VAbr = linalg::triplet(linalg::triplet(Caocc_B, S, D_A, true, false, false), V_B, Cavir_A);
    VAbr->scale(-1.0);
    std::shared_ptr<Matrix> VRas = linalg::triplet(linalg::triplet(Caocc_A, V_B, P_A, true, false, false), S, Cavir_B);
    VRas->scale(1.0);
    std::shared_ptr<Matrix> VSbr = linalg::triplet(linalg::triplet(Caocc_B, V_A, P_B, true, false, false), S, Cavir_A);
    VSbr->scale(1.0);

    std::shared_ptr<Matrix> Sas = linalg::triplet(Caocc_A, S, Cavir_B, true, false, false);
    std::shared_ptr<Matrix> Sbr = linalg::triplet(Caocc_B, S, Cavir_A, true, false, false);

    std::shared_ptr<Matrix> Qbr(Jbr->clone());

    // Jbr->set_name("Jbr");
    // Jbr->print();

    Qbr->zero();
    Qbr->add(Jbr);
    Qbr->add(Kbr);
    Qbr->add(KObr);
    Qbr->add(JAbr);
    Qbr->add(JBbr);
    Qbr->add(VAbr);
    Qbr->add(VSbr);

    std::shared_ptr<Matrix> Qas(Jas->clone());
    Qas->zero();
    Qas->add(Jas);
    Qas->add(Kas);
    Qas->add(KOas);
    Qas->add(JAas);
    Qas->add(JBas);
    Qas->add(VBas);
    Qas->add(VRas);

    std::shared_ptr<Matrix> SBar = linalg::triplet(linalg::triplet(Caocc_A, S, D_B, true, false, false), S, Cavir_A);
    std::shared_ptr<Matrix> SAbs = linalg::triplet(linalg::triplet(Caocc_B, S, D_A, true, false, false), S, Cavir_B);

    std::shared_ptr<Matrix> Qar(Jar->clone());
    Qar->zero();
    Qar->add(Jar);
    Qar->add(Var);

    std::shared_ptr<Matrix> Qbs(Jbs->clone());
    Qbs->zero();
    Qbs->add(Jbs);
    Qbs->add(Vbs);

    std::shared_ptr<Matrix> KXOYas(Jas->clone());
    std::shared_ptr<Matrix> KXOYbr(Jbr->clone());
    double** KXOYasp = KXOYas->pointer();
    double** KXOYbrp = KXOYbr->pointer();

    Jbr.reset();
    Kbr.reset();
    Jas.reset();
    Kas.reset();
    KOas.reset();
    KObr.reset();
    JBas.reset();
    JAbr.reset();
    Jbs.reset();
    Jar.reset();
    JAas.reset();
    JBbr.reset();
    Vbs.reset();
    Var.reset();
    VBas.reset();
    VAbr.reset();
    VRas.reset();
    VSbr.reset();

    // => Integrals from DFHelper <= //

    std::vector<std::shared_ptr<Matrix> > Cs;
    Cs.push_back(Caocc_A);
    Cs.push_back(Cavir_A);
    Cs.push_back(Caocc_B);
    Cs.push_back(Cavir_B);
    Cs.push_back(Cr1);
    Cs.push_back(Cs1);
    Cs.push_back(Ca2);
    Cs.push_back(Cb2);
    Cs.push_back(Cr3);
    Cs.push_back(Cs3);
    Cs.push_back(Ca4);
    Cs.push_back(Cb4);

    size_t max_MO = 0, ncol = 0;
    for (auto& mat : Cs) {
        max_MO = std::max(max_MO, (size_t)mat->ncol());
        ncol += (size_t)mat->ncol();
    }

    auto dfh(std::make_shared<DFHelper>(primary_, auxiliary));
    dfh->set_memory(doubles_ - Cs[0]->nrow() * ncol);
    dfh->set_method("DIRECT_iaQ");
    dfh->set_nthreads(nT);
    dfh->initialize();
    dfh->print_header();

    dfh->add_space("a", Cs[0]);
    dfh->add_space("r", Cs[1]);
    dfh->add_space("b", Cs[2]);
    dfh->add_space("s", Cs[3]);
    dfh->add_space("r1", Cs[4]);
    dfh->add_space("s1", Cs[5]);
    dfh->add_space("a2", Cs[6]);
    dfh->add_space("b2", Cs[7]);
    dfh->add_space("r3", Cs[8]);
    dfh->add_space("s3", Cs[9]);
    dfh->add_space("a4", Cs[10]);
    dfh->add_space("b4", Cs[11]);

    // print all space Cs
    // Cs[0]->set_name("Caocc_A");
    // Cs[1]->set_name("Cavir_A");
    // Cs[2]->set_name("Caocc_B");
    // Cs[3]->set_name("Cavir_B");
    // Cs[4]->set_name("Cr1");
    // Cs[5]->set_name("Cs1");
    // Cs[6]->set_name("Ca2");
    // Cs[7]->set_name("Cb2");
    // Cs[8]->set_name("Cr3");
    // Cs[9]->set_name("Cs3");
    // Cs[10]->set_name("Ca4");
    // Cs[11]->set_name("Cb4");
    // 
    // Cs[0]->print();
    // Cs[1]->print();
    // Cs[2]->print();
    // Cs[3]->print();
    // Cs[4]->print();
    // Cs[5]->print();
    // Cs[6]->print();
    // Cs[7]->print();
    // Cs[8]->print();
    // Cs[9]->print();
    // Cs[10]->print();
    // Cs[11]->print();

    dfh->add_transformation("Aar", "r", "a");
    dfh->add_transformation("Abs", "s", "b");
    dfh->add_transformation("Bas", "s1", "a");
    dfh->add_transformation("Bbr", "r1", "b");
    dfh->add_transformation("Cas", "s", "a2");
    dfh->add_transformation("Cbr", "r", "b2");
    dfh->add_transformation("Dar", "r3", "a");
    dfh->add_transformation("Dbs", "s3", "b");
    dfh->add_transformation("Ear", "r", "a4");
    dfh->add_transformation("Ebs", "s", "b4");

    // => Additional quantities needed for parallel/perpendicular link orbital spin coupling (but not for their average)  <= //

    if ((link_assignment == "SAO0" || link_assignment == "SAO1" || link_assignment == "SAO2" || link_assignment == "SIAO0" || link_assignment == "SIAO1" || link_assignment == "SIAO2") && parperp) {
        std::shared_ptr<Matrix> K_XOY = matrices_["K_XOY"];
        std::shared_ptr<Matrix> D_X = matrices_["D_X"];
        std::shared_ptr<Matrix> J_X = matrices_["J_X"];
        std::shared_ptr<Matrix> K_X = matrices_["K_X"];
        std::shared_ptr<Matrix> D_Y = matrices_["D_Y"];
        std::shared_ptr<Matrix> J_Y = matrices_["J_Y"];
        std::shared_ptr<Matrix> K_Y = matrices_["K_Y"];
        std::shared_ptr<Matrix> Cx = matrices_["thislinkA"];
        std::shared_ptr<Matrix> Cy = matrices_["thislinkB"];
        std::shared_ptr<Matrix> Cx1 = linalg::triplet(D_Y, S, Cavir_A);
        Cx1->scale(-1.0);
        std::shared_ptr<Matrix> Cy1 = linalg::triplet(D_X, S, Cavir_B);
        Cy1->scale(-1.0);
        std::shared_ptr<Matrix> Cx2 = linalg::triplet(D_Y, S, Caocc_A);
        std::shared_ptr<Matrix> Cy2 = linalg::triplet(D_X, S, Caocc_B);
        std::shared_ptr<Matrix> Cx3 = linalg::triplet(linalg::triplet(D_X, S, D_Y), S, Cavir_A);
        Cx3->scale(-2.0);
        std::shared_ptr<Matrix> Cy3 = linalg::triplet(linalg::triplet(D_Y, S, D_X), S, Cavir_B);
        Cy3->scale(-2.0);
        std::shared_ptr<Matrix> Cx4 = linalg::triplet(linalg::triplet(D_X, S, D_Y), S, Caocc_A);
        Cx4->scale(-2.0);
        std::shared_ptr<Matrix> Cy4 = linalg::triplet(linalg::triplet(D_Y, S, D_X), S, Caocc_B);
        Cy4->scale(-2.0);
        KXOYas = linalg::triplet(Caocc_A, K_XOY, Cavir_B, true, false, false);
        KXOYas->scale(1.0);
        KXOYbr = linalg::triplet(Caocc_B, K_XOY, Cavir_A, true, true, false);
        KXOYbr->scale(1.0);

        Cs.push_back(Cx1);
        Cs.push_back(Cy1);
        Cs.push_back(Cx2);
        Cs.push_back(Cy2);
        Cs.push_back(Cx3);
        Cs.push_back(Cy3);
        Cs.push_back(Cx4);
        Cs.push_back(Cy4);

        dfh->add_space("x1", Cs[12]);
        dfh->add_space("y1", Cs[13]);
        dfh->add_space("x2", Cs[14]);
        dfh->add_space("y2", Cs[15]);
        dfh->add_space("x3", Cs[16]);
        dfh->add_space("y3", Cs[17]);
        dfh->add_space("x4", Cs[18]);
        dfh->add_space("y4", Cs[19]);

        dfh->add_transformation("BYas", "y1", "a");
        dfh->add_transformation("BXbr", "x1", "b");
        dfh->add_transformation("CXas", "s", "x2");
        dfh->add_transformation("CYbr", "r", "y2");
        dfh->add_transformation("DXar", "x3", "a");
        dfh->add_transformation("DYbs", "y3", "b");
        dfh->add_transformation("EXar", "r", "x4");
        dfh->add_transformation("EYbs", "s", "y4");

// now ready for DF transformation in both cases
        dfh->transform();

        Cx1.reset();
        Cy1.reset();
        Cx2.reset();
        Cy2.reset();
        Cx3.reset();
        Cy3.reset();
        Cx4.reset();
        Cy4.reset();

    }
    else {
        dfh->transform();
 
    }

    Cr1.reset();
    Cs1.reset();
    Ca2.reset();
    Cb2.reset();
    Cr3.reset();
    Cs3.reset();
    Ca4.reset();
    Cb4.reset();
    Cs.clear();
    dfh->clear_spaces();

    // => Blocking ... figure out how big a tensor slice to handle at a time <= //

    long int overhead = 0L;
    overhead += 5L * nT * na * nb; // Tab, Vab, T2ab, V2ab, and Iab work arrays below
    if ((link_assignment == "SAO0" || link_assignment == "SAO1" || link_assignment == "SAO2" || link_assignment == "SIAO0" || link_assignment == "SIAO1" || link_assignment == "SIAO2") && parperp) {
        overhead += 4L * na * ns + 4L * nb * nr + 4L * na * nr + 4L * nb * ns; // Sas, Sbr, sBar, sAbs, Qas, Qbr, Qar, Qbs
    }
    else {
        overhead += 2L * na * ns + 2L * nb * nr + 2L * na * nr + 2L * nb * ns; // Sas, Sbr, sBar, sAbs, Qas, Qbr, Qar, Qbs
    }
    // the next few matrices allocated here don't take too much room (but might if large numbers of threads)
    overhead += 2L * na * nb * (nT + 1); // E_disp20 and E_exch_disp20 thread work and final matrices
    overhead += 1L * sna * snb * (nT + 1); // sE_exch_disp20 thread work and final matrices
    overhead += 1L * (nA + nfa + na) * (nB + nfb + nb); // Disp_AB
    overhead += 1L * (snA + snfa + sna) * (snB + snfb + snb); // sDisp_AB
    // account for a few of the smaller matrices already defined, but not exhaustively
    overhead += 12L * nn * nn; // D, V, J, K, P, and C matrices for A and B (neglecting C)
    long int rem = doubles_ - overhead;

    outfile->Printf("    %ld doubles - %ld overhead leaves %ld for dispersion\n", doubles_, overhead, rem);
    
    if (rem < 0L) {
        throw PSIEXCEPTION("Too little static memory for DFTSAPT::mp2_terms");
    }

    // cost_r is how much mem for Aar, Bbr, Cbr, Dar for a single r
    // cost_s would be the same value, and is the mem requirement for Abs, Bas, Cas, and Dbs for single s
    long int cost_r = 0L;
    if ((link_assignment == "SAO0" || link_assignment == "SAO1" || link_assignment == "SAO2" || link_assignment == "SIAO0" || link_assignment == "SIAO1" || link_assignment == "SIAO2") && parperp) {
        cost_r = 4L * na * nQ + 4L * nb * nQ; 
    }
    else {
        cost_r = 2L * na * nQ + 2L * nb * nQ;
    } 
    long int max_r_l = rem / (2L * cost_r); // 2 b/c need to hold both an r and an s
    long int max_s_l = max_r_l;
    int max_r = (max_r_l > nr ? nr : (int) max_r_l);
    int max_s = (max_s_l > ns ? ns : (int) max_s_l);
    if (max_r < 1 || max_s < 1) {
        throw PSIEXCEPTION("Too little dynamic memory for DFTSAPT::mp2_terms");
    }
    int nrblocks = (nr / max_r);
    if (nr % max_r) nrblocks++;
    int nsblocks = (ns / max_s);
    if (ns % max_s) nsblocks++;
    outfile->Printf("    Processing a single (r,s) pair requires %ld doubles\n", cost_r * 2L);
    outfile->Printf("    %d values of r processed in %d blocks of %d\n", nr, nrblocks, max_r);
    outfile->Printf("    %d values of s processed in %d blocks of %d\n\n", ns, nsblocks, max_s);

    // => Tensor Slices <= //

    auto Aar = std::make_shared<Matrix>("Aar", max_r * na, nQ);
    auto Abs = std::make_shared<Matrix>("Abs", max_s * nb, nQ);
    auto Bas = std::make_shared<Matrix>("Bas", max_s * na, nQ);
    auto Bbr = std::make_shared<Matrix>("Bbr", max_r * nb, nQ);
    auto Cas = std::make_shared<Matrix>("Cas", max_s * na, nQ);
    auto Cbr = std::make_shared<Matrix>("Cbr", max_r * nb, nQ);
    auto Dar = std::make_shared<Matrix>("Dar", max_r * na, nQ);
    auto Dbs = std::make_shared<Matrix>("Dbs", max_s * nb, nQ);

    // => Thread Work Arrays <= //

    std::vector<std::shared_ptr<Matrix> > Tab;
    std::vector<std::shared_ptr<Matrix> > Vab;
    std::vector<std::shared_ptr<Matrix> > T2ab;
    std::vector<std::shared_ptr<Matrix> > V2ab;
    std::vector<std::shared_ptr<Matrix> > Iab;
    for (int t = 0; t < nT; t++) {
        Tab.push_back(std::make_shared<Matrix>("Tab", na, nb));
        Vab.push_back(std::make_shared<Matrix>("Vab", na, nb));
        T2ab.push_back(std::make_shared<Matrix>("T2ab", na, nb));
        V2ab.push_back(std::make_shared<Matrix>("V2ab", na, nb));
        Iab.push_back(std::make_shared<Matrix>("Iab", na, nb));
    }

    // => Pointers <= //

    double** Aarp = Aar->pointer();
    double** Absp = Abs->pointer();
    double** Basp = Bas->pointer();
    double** Bbrp = Bbr->pointer();
    double** Casp = Cas->pointer();
    double** Cbrp = Cbr->pointer();
    double** Darp = Dar->pointer();
    double** Dbsp = Dbs->pointer();

    double** Sasp = Sas->pointer();
    double** Sbrp = Sbr->pointer();
    double** SBarp = SBar->pointer();
    double** SAbsp = SAbs->pointer();

    double** Qasp = Qas->pointer();
    double** Qbrp = Qbr->pointer();
    double** Qarp = Qar->pointer();
    double** Qbsp = Qbs->pointer();

    double* eap = eps_aocc_A->pointer();
    double* ebp = eps_aocc_B->pointer();
    double* erp = eps_avir_A->pointer();
    double* esp = eps_avir_B->pointer();

    // => Slice D + E -> D <= //

    dfh->add_disk_tensor("Far", std::make_tuple(nr, na, nQ));

    for (size_t rstart = 0; rstart < nr; rstart += max_r) {
        size_t nrblock = (rstart + max_r >= nr ? nr - rstart : max_r);

        dfh->fill_tensor("Dar", Dar, {rstart, rstart + nrblock});
        dfh->fill_tensor("Ear", Aar, {rstart, rstart + nrblock});

        double* D2p = Darp[0];
        double* A2p = Aarp[0];
        for (long int arQ = 0L; arQ < nrblock * naQ; arQ++) {
            (*D2p++) += (*A2p++);
        }
        dfh->write_disk_tensor("Far", Dar, {rstart, rstart + nrblock});
    }

    dfh->add_disk_tensor("Fbs", std::make_tuple(ns, nb, nQ));

    for (size_t sstart = 0; sstart < ns; sstart += max_s) {
        size_t nsblock = (sstart + max_s >= ns ? ns - sstart : max_s);

        dfh->fill_tensor("Dbs", Dbs, {sstart, sstart + nsblock});
        dfh->fill_tensor("Ebs", Abs, {sstart, sstart + nsblock});

        double* D2p = Dbsp[0];
        double* A2p = Absp[0];
        for (long int bsQ = 0L; bsQ < nsblock * nbQ; bsQ++) {
            (*D2p++) += (*A2p++);
        }
        dfh->write_disk_tensor("Fbs", Dbs, {sstart, sstart + nsblock});
    }

    // => Targets <= //

    double Disp20 = 0.0;
    double ExchDisp20 = 0.0;
    double sExchDisp20 = 0.0;
    double par_ExchDisp20 = 0.0;

    // => Local Targets <= //

    std::vector<std::shared_ptr<Matrix> > E_disp20_threads;
    std::vector<std::shared_ptr<Matrix> > E_exch_disp20_threads;
    std::vector<std::shared_ptr<Matrix> > sE_exch_disp20_threads;
    std::vector<std::shared_ptr<Matrix> > par_E_exch_disp20_threads;
    for (int t = 0; t < nT; t++) {
        E_disp20_threads.push_back(std::make_shared<Matrix>("E_disp20", na, nb));
        E_exch_disp20_threads.push_back(std::make_shared<Matrix>("E_exch_disp20", na, nb));
        sE_exch_disp20_threads.push_back(std::make_shared<Matrix>("sE_exch_disp20", sna, snb));
        par_E_exch_disp20_threads.push_back(std::make_shared<Matrix>("par_E_exch_disp20", na, nb));
    }

    // => MO => LO Transform <= //

    double** UAp = Uaocc_A->pointer();
    double** UBp = Uaocc_B->pointer();

    double scale = 1.0;
    if (options_.get_bool("SSAPT0_SCALE")) {
        scale = sSAPT0_scale_;
    }

    // ==> Master Loop <== //

// For the SAOn/SIAOn algorithms if parallel and perpendicular spin couplings are requested for E(20)exch-disp, 
// we need to compute the expression from https://doi.org/10.1021/acs.jpca.2c06465, Eq. (8) in the SI.
// If only averaged spin coupling is requested, the standard SAPT0 formula works and we don't do anything extra.
    if ((link_assignment == "SAO0" || link_assignment == "SAO1" || link_assignment == "SAO2" || link_assignment == "SIAO0" || link_assignment == "SIAO1" || link_assignment == "SIAO2") && parperp) {

        auto BYas = std::make_shared<Matrix>("BYas", max_s * na, nQ);
        auto BXbr = std::make_shared<Matrix>("BXbr", max_r * nb, nQ);
        auto CXas = std::make_shared<Matrix>("CXas", max_s * na, nQ);
        auto CYbr = std::make_shared<Matrix>("CYbr", max_r * nb, nQ);
        auto DXar = std::make_shared<Matrix>("DXar", max_r * na, nQ);
        auto DYbs = std::make_shared<Matrix>("DYbs", max_s * nb, nQ);
        double** BYasp = BYas->pointer();
        double** BXbrp = BXbr->pointer();
        double** CXasp = CXas->pointer();
        double** CYbrp = CYbr->pointer();
        double** DXarp = DXar->pointer();
        double** DYbsp = DYbs->pointer();
        double** KXOYasp = KXOYas->pointer();
        double** KXOYbrp = KXOYbr->pointer();

        dfh->add_disk_tensor("FXar", std::make_tuple(nr, na, nQ));

        for (size_t rstart = 0; rstart < nr; rstart += max_r) {
            size_t nrblock = (rstart + max_r >= nr ? nr - rstart : max_r);

            dfh->fill_tensor("DXar", DXar, {rstart, rstart + nrblock});
            dfh->fill_tensor("EXar", Aar, {rstart, rstart + nrblock});

            double* DX2p = DXarp[0];
            double* A2p = Aarp[0];
            for (long int arQ = 0L; arQ < nrblock * naQ; arQ++) {
                (*DX2p++) += (*A2p++);
            }
            dfh->write_disk_tensor("FXar", DXar, {rstart, rstart + nrblock});
        }

        dfh->add_disk_tensor("FYbs", std::make_tuple(ns, nb, nQ));

        for (size_t sstart = 0; sstart < ns; sstart += max_s) {
            size_t nsblock = (sstart + max_s >= ns ? ns - sstart : max_s);

            dfh->fill_tensor("DYbs", DYbs, {sstart, sstart + nsblock});
            dfh->fill_tensor("EYbs", Abs, {sstart, sstart + nsblock});

            double* DY2p = DYbsp[0];
            double* A2p = Absp[0];
            for (long int bsQ = 0L; bsQ < nsblock * nbQ; bsQ++) {
                (*DY2p++) += (*A2p++);
            }
            dfh->write_disk_tensor("FYbs", DYbs, {sstart, sstart + nsblock});
        }
    
        for (size_t rstart = 0; rstart < nr; rstart += max_r) {
            size_t nrblock = (rstart + max_r >= nr ? nr - rstart : max_r);
    
            dfh->fill_tensor("Aar", Aar, {rstart, rstart + nrblock});
            dfh->fill_tensor("Far", Dar, {rstart, rstart + nrblock});
            dfh->fill_tensor("Bbr", Bbr, {rstart, rstart + nrblock});
            dfh->fill_tensor("Cbr", Cbr, {rstart, rstart + nrblock});
            dfh->fill_tensor("FXar", DXar, {rstart, rstart + nrblock});
            dfh->fill_tensor("BXbr", BXbr, {rstart, rstart + nrblock});
            dfh->fill_tensor("CYbr", CYbr, {rstart, rstart + nrblock});
    
            for (size_t sstart = 0; sstart < ns; sstart += max_s) {
                size_t nsblock = (sstart + max_s >= ns ? ns - sstart : max_s);
    
                dfh->fill_tensor("Abs", Abs, {sstart, sstart + nsblock});
                dfh->fill_tensor("Fbs", Dbs, {sstart, sstart + nsblock});
                dfh->fill_tensor("Bas", Bas, {sstart, sstart + nsblock});
                dfh->fill_tensor("Cas", Cas, {sstart, sstart + nsblock});
                dfh->fill_tensor("FYbs", DYbs, {sstart, sstart + nsblock});
                dfh->fill_tensor("BYas", BYas, {sstart, sstart + nsblock});
                dfh->fill_tensor("CXas", CXas, {sstart, sstart + nsblock});

                long int nrs = nrblock * nsblock;

#pragma omp parallel for schedule(dynamic) reduction(+ : Disp20, ExchDisp20, sExchDisp20, par_ExchDisp20)
                for (long int rs = 0L; rs < nrs; rs++) {
                    int r = rs / nsblock;
                    int s = rs % nsblock;

                    int thread = 0;
#ifdef _OPENMP
                    thread = omp_get_thread_num();
#endif

                    double** E_disp20Tp = E_disp20_threads[thread]->pointer();
                    double** E_exch_disp20Tp = E_exch_disp20_threads[thread]->pointer();
                    double** sE_exch_disp20Tp = sE_exch_disp20_threads[thread]->pointer();
                    double** par_E_exch_disp20Tp = par_E_exch_disp20_threads[thread]->pointer();
 
                    double** Tabp = Tab[thread]->pointer();
                    double** Vabp = Vab[thread]->pointer();
                    double** T2abp = T2ab[thread]->pointer();
                    double** V2abp = V2ab[thread]->pointer();
                    double** Iabp = Iab[thread]->pointer();
 
                    // => Amplitudes, Disp20 <= //
 
                    C_DGEMM('N', 'T', na, nb, nQ, 1.0, Aarp[(r)*na], nQ, Absp[(s)*nb], nQ, 0.0, Vabp[0], nb);
                    for (int a = 0; a < na; a++) {
                        for (int b = 0; b < nb; b++) {
                            Tabp[a][b] = Vabp[a][b] / (eap[a] + ebp[b] - erp[r + rstart] - esp[s + sstart]);
                        }
                    }
 
                    C_DGEMM('N', 'N', na, nb, nb, 1.0, Tabp[0], nb, UBp[0], nb, 0.0, Iabp[0], nb);
                    C_DGEMM('T', 'N', na, nb, na, 1.0, UAp[0], na, Iabp[0], nb, 0.0, T2abp[0], nb);
                    C_DGEMM('N', 'N', na, nb, nb, 1.0, Vabp[0], nb, UBp[0], nb, 0.0, Iabp[0], nb);
                    C_DGEMM('T', 'N', na, nb, na, 1.0, UAp[0], na, Iabp[0], nb, 0.0, V2abp[0], nb);
 
                    for (int a = 0; a < na; a++) {
                        for (int b = 0; b < nb; b++) {
                            E_disp20Tp[a][b] += 4.0 * T2abp[a][b] * V2abp[a][b];
                            Disp20 += 4.0 * T2abp[a][b] * V2abp[a][b];
                        }
                    }

                    // => Exch-Disp20 <= //

                    // > Q1-Q3 < //

                    C_DGEMM('N', 'T', na, nb, nQ, 1.0, Basp[(s)*na], nQ, Bbrp[(r)*nb], nQ, 0.0, Vabp[0], nb);
                    C_DGEMM('N', 'T', na, nb, nQ, 1.0, Casp[(s)*na], nQ, Cbrp[(r)*nb], nQ, 1.0, Vabp[0], nb);
                    C_DGEMM('N', 'T', na, nb, nQ, 1.0, Aarp[(r)*na], nQ, Dbsp[(s)*nb], nQ, 1.0, Vabp[0], nb);
                    C_DGEMM('N', 'T', na, nb, nQ, 1.0, Darp[(r)*na], nQ, Absp[(s)*nb], nQ, 1.0, Vabp[0], nb);

                    // > V,J,K < //

                    C_DGER(na, nb, 1.0, &Sasp[0][s + sstart], ns, &Qbrp[0][r + rstart], nr, Vabp[0], nb);
                    C_DGER(na, nb, 1.0, &Qasp[0][s + sstart], ns, &Sbrp[0][r + rstart], nr, Vabp[0], nb);
                    C_DGER(na, nb, 1.0, &Qarp[0][r + rstart], nr, &SAbsp[0][s + sstart], ns, Vabp[0], nb);
                    C_DGER(na, nb, 1.0, &SBarp[0][r + rstart], nr, &Qbsp[0][s + sstart], ns, Vabp[0], nb);

                    C_DGEMM('N', 'N', na, nb, nb, 1.0, Vabp[0], nb, UBp[0], nb, 0.0, Iabp[0], nb);
                    C_DGEMM('T', 'N', na, nb, na, 1.0, UAp[0], na, Iabp[0], nb, 0.0, V2abp[0], nb);

                    for (int a = 0; a < na; a++) {
                        for (int b = 0; b < nb; b++) {
                            E_exch_disp20Tp[a][b] -= 2.0 * T2abp[a][b] * V2abp[a][b];
                            if (options_.get_bool("SSAPT0_SCALE"))
                                sE_exch_disp20Tp[a][b] -= scale * 2.0 * T2abp[a][b] * V2abp[a][b];
                            ExchDisp20 -= 2.0 * T2abp[a][b] * V2abp[a][b];
                            sExchDisp20 -= scale * 2.0 * T2abp[a][b] * V2abp[a][b];
                        }
                    }

                    // now, additional term for parallel/perpendicular spin coupling
                    // > Q1-Q3 < //

                    C_DGEMM('N', 'T', na, nb, nQ, 1.0, BYasp[(s)*na], nQ, BXbrp[(r)*nb], nQ, 0.0, Vabp[0], nb);
                    C_DGEMM('N', 'T', na, nb, nQ, 1.0, CXasp[(s)*na], nQ, CYbrp[(r)*nb], nQ, 1.0, Vabp[0], nb);
                    C_DGEMM('N', 'T', na, nb, nQ, 1.0, Aarp[(r)*na], nQ, DYbsp[(s)*nb], nQ, 1.0, Vabp[0], nb);
                    C_DGEMM('N', 'T', na, nb, nQ, 1.0, DXarp[(r)*na], nQ, Absp[(s)*nb], nQ, 1.0, Vabp[0], nb);

                    // > V,J,K < //

                    C_DGER(na, nb, 1.0, &Sasp[0][s + sstart], ns, &KXOYbrp[0][r + rstart], nr, Vabp[0], nb);
                    C_DGER(na, nb, 1.0, &KXOYasp[0][s + sstart], ns, &Sbrp[0][r + rstart], nr, Vabp[0], nb);

                    C_DGEMM('N', 'N', na, nb, nb, 1.0, Vabp[0], nb, UBp[0], nb, 0.0, Iabp[0], nb);
                    C_DGEMM('T', 'N', na, nb, na, 1.0, UAp[0], na, Iabp[0], nb, 0.0, V2abp[0], nb);

                    for (int a = 0; a < na; a++) {
                        for (int b = 0; b < nb; b++) {
                            par_E_exch_disp20Tp[a][b] -= 2.0 * T2abp[a][b] * V2abp[a][b];
                            par_ExchDisp20 -= 2.0 * T2abp[a][b] * V2abp[a][b];
                        }
                    }
                }
            }
        }
    }
    else {
        for (size_t rstart = 0; rstart < nr; rstart += max_r) {
            size_t nrblock = (rstart + max_r >= nr ? nr - rstart : max_r);

            dfh->fill_tensor("Aar", Aar, {rstart, rstart + nrblock});
            dfh->fill_tensor("Far", Dar, {rstart, rstart + nrblock});
            dfh->fill_tensor("Bbr", Bbr, {rstart, rstart + nrblock});
            dfh->fill_tensor("Cbr", Cbr, {rstart, rstart + nrblock});

            for (size_t sstart = 0; sstart < ns; sstart += max_s) {
                size_t nsblock = (sstart + max_s >= ns ? ns - sstart : max_s);

                dfh->fill_tensor("Abs", Abs, {sstart, sstart + nsblock});
                dfh->fill_tensor("Fbs", Dbs, {sstart, sstart + nsblock});
                dfh->fill_tensor("Bas", Bas, {sstart, sstart + nsblock});
                dfh->fill_tensor("Cas", Cas, {sstart, sstart + nsblock});

                long int nrs = nrblock * nsblock;

#pragma omp parallel for schedule(dynamic) reduction(+ : Disp20, ExchDisp20, sExchDisp20, par_ExchDisp20)
                for (long int rs = 0L; rs < nrs; rs++) {
                    int r = rs / nsblock;
                    int s = rs % nsblock;

                    int thread = 0;
#ifdef _OPENMP
                    thread = omp_get_thread_num();
#endif

                    double** E_disp20Tp = E_disp20_threads[thread]->pointer();
                    double** E_exch_disp20Tp = E_exch_disp20_threads[thread]->pointer();
                    double** sE_exch_disp20Tp = sE_exch_disp20_threads[thread]->pointer();
                    double** par_E_exch_disp20Tp = par_E_exch_disp20_threads[thread]->pointer();

                    double** Tabp = Tab[thread]->pointer();
                    double** Vabp = Vab[thread]->pointer();
                    double** T2abp = T2ab[thread]->pointer();
                    double** V2abp = V2ab[thread]->pointer();
                    double** Iabp = Iab[thread]->pointer();

                    // => Amplitudes, Disp20 <= //

                    C_DGEMM('N', 'T', na, nb, nQ, 1.0, Aarp[(r)*na], nQ, Absp[(s)*nb], nQ, 0.0, Vabp[0], nb);
                    for (int a = 0; a < na; a++) {
                        for (int b = 0; b < nb; b++) {
                            Tabp[a][b] = Vabp[a][b] / (eap[a] + ebp[b] - erp[r + rstart] - esp[s + sstart]);
                        }
                    }

                    C_DGEMM('N', 'N', na, nb, nb, 1.0, Tabp[0], nb, UBp[0], nb, 0.0, Iabp[0], nb);
                    C_DGEMM('T', 'N', na, nb, na, 1.0, UAp[0], na, Iabp[0], nb, 0.0, T2abp[0], nb);
                    C_DGEMM('N', 'N', na, nb, nb, 1.0, Vabp[0], nb, UBp[0], nb, 0.0, Iabp[0], nb);
                    C_DGEMM('T', 'N', na, nb, na, 1.0, UAp[0], na, Iabp[0], nb, 0.0, V2abp[0], nb);

                    for (int a = 0; a < na; a++) {
                        for (int b = 0; b < nb; b++) {
                            E_disp20Tp[a][b] += 4.0 * T2abp[a][b] * V2abp[a][b];
                            Disp20 += 4.0 * T2abp[a][b] * V2abp[a][b];
                        }
                    }

                    // => Exch-Disp20 <= //

                    // > Q1-Q3 < //

                    C_DGEMM('N', 'T', na, nb, nQ, 1.0, Basp[(s)*na], nQ, Bbrp[(r)*nb], nQ, 0.0, Vabp[0], nb);
                    C_DGEMM('N', 'T', na, nb, nQ, 1.0, Casp[(s)*na], nQ, Cbrp[(r)*nb], nQ, 1.0, Vabp[0], nb);
                    C_DGEMM('N', 'T', na, nb, nQ, 1.0, Aarp[(r)*na], nQ, Dbsp[(s)*nb], nQ, 1.0, Vabp[0], nb);
                    C_DGEMM('N', 'T', na, nb, nQ, 1.0, Darp[(r)*na], nQ, Absp[(s)*nb], nQ, 1.0, Vabp[0], nb);

                    // > V,J,K < //

                    C_DGER(na, nb, 1.0, &Sasp[0][s + sstart], ns, &Qbrp[0][r + rstart], nr, Vabp[0], nb);
                    C_DGER(na, nb, 1.0, &Qasp[0][s + sstart], ns, &Sbrp[0][r + rstart], nr, Vabp[0], nb);
                    C_DGER(na, nb, 1.0, &Qarp[0][r + rstart], nr, &SAbsp[0][s + sstart], ns, Vabp[0], nb);
                    C_DGER(na, nb, 1.0, &SBarp[0][r + rstart], nr, &Qbsp[0][s + sstart], ns, Vabp[0], nb);

                    C_DGEMM('N', 'N', na, nb, nb, 1.0, Vabp[0], nb, UBp[0], nb, 0.0, Iabp[0], nb);
                    C_DGEMM('T', 'N', na, nb, na, 1.0, UAp[0], na, Iabp[0], nb, 0.0, V2abp[0], nb);

                    for (int a = 0; a < na; a++) {
                        for (int b = 0; b < nb; b++) {
                            E_exch_disp20Tp[a][b] -= 2.0 * T2abp[a][b] * V2abp[a][b];
                            if (options_.get_bool("SSAPT0_SCALE"))
                                sE_exch_disp20Tp[a][b] -= scale * 2.0 * T2abp[a][b] * V2abp[a][b];
                            ExchDisp20 -= 2.0 * T2abp[a][b] * V2abp[a][b];
                            sExchDisp20 -= scale * 2.0 * T2abp[a][b] * V2abp[a][b];
                        }
                    }
                }
            }
        }
    }

    auto E_disp20 = std::make_shared<Matrix>("E_disp20", na, nb);
    auto E_exch_disp20 = std::make_shared<Matrix>("E_exch_disp20", na, nb);
    double** E_disp20p = E_disp20->pointer();
    double** E_exch_disp20p = E_exch_disp20->pointer();

    for (int t = 0; t < nT; t++) {
        E_disp20->add(E_disp20_threads[t]);
        E_exch_disp20->add(E_exch_disp20_threads[t]);
    }

    for (int a = 0; a < na; a++) {
        for (int b = 0; b < nb; b++) {
            Ep[a + nfa + nA][b + nfb + nB] = E_disp20p[a][b] + E_exch_disp20p[a][b];
        }
    }

    if (options_.get_bool("SSAPT0_SCALE")) {
        auto sE_exch_disp20 = std::make_shared<Matrix>("sE_exch_disp20", na, nb);
        sE_exch_disp20->copy(E_exch_disp20);
        double** sE_exch_disp20p = sE_exch_disp20->pointer();
        sE_exch_disp20->scale(sSAPT0_scale_);

        for (int a = 0; a < na; a++) {
            for (int b = 0; b < nb; b++) {
                sEp[a + nfa + nA][b + nfb + nB] = E_disp20p[a][b] + sE_exch_disp20p[a][b];
            }
        }
    }

    // E_disp20->print();
    // E_exch_disp20->print();

    scalars_["Disp20"] = Disp20;
    scalars_["Exch-Disp20"] = ExchDisp20;
    if (options_.get_bool("SSAPT0_SCALE")) scalars_["sExch-Disp20"] = sExchDisp20;
    outfile->Printf("    Disp20              = %18.12lf [Eh]\n", Disp20);
    outfile->Printf("    Exch-Disp20         = %18.12lf [Eh]\n", ExchDisp20);
    if ((link_assignment == "SAO0" || link_assignment == "SAO1" || link_assignment == "SAO2" || link_assignment == "SIAO0" || link_assignment == "SIAO1" || link_assignment == "SIAO2") && parperp) {
        outfile->Printf("    Exch-Disp20 (PAR)   = %18.12lf [Eh]\n", ExchDisp20 + par_ExchDisp20);
        outfile->Printf("    Exch-Disp20 (PERP)  = %18.12lf [Eh]\n", ExchDisp20 - par_ExchDisp20);
    }
    if (options_.get_bool("SSAPT0_SCALE")) outfile->Printf("    sExch-Disp20         = %18.12lf [Eh]\n", sExchDisp20);
    outfile->Printf("\n");
    // fflush(outfile);
}

}  // namespace fisapt
}  // namespace psi
