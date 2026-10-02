/*
 * @BEGIN LICENSE
 *
 * Psi4: an open-source quantum chemistry software package
 *
 * Copyright (c) 2007-2025 The Psi4 Developers.
 *
 * This file is part of Psi4.
 *
 * Psi4 is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * @END LICENSE
 */
// Adapted from construct_restricted_c1_primitives_impl in
// camcasp_psi4 5449bd1a01c73f45c307b36b006e264c1e43b994.
// Adaptation copyright 2026 Psi4 Developers. No FrozenResponseContext/seals ported.
#include "native_response.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <vector>
#include <libint2/shell.h>  // snapshot checks read shell data; no engines here
#include "psi4/libmints/basisset.h"
#include "psi4/libmints/coordentry.h"
#include "psi4/libmints/gshell.h"
#include "psi4/libmints/integral.h"
#include "psi4/libmints/onebody.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libmints/vector.h"
#include "psi4/libmints/wavefunction.h"

namespace psi { namespace isapol {
namespace {
void require_native(bool ok, const std::string& message) {
    if (!ok) throw std::invalid_argument("NativeRestrictedState: " + message);
}
std::size_t mul(std::size_t a, std::size_t b) {
    require_native(!b || a <= std::numeric_limits<std::size_t>::max()/b, "resource size overflow");
    return a*b;
}
std::size_t add(std::size_t a, std::size_t b) {
    require_native(a <= std::numeric_limits<std::size_t>::max()-b, "resource size overflow");
    return a+b;
}
void matrix_ok(const SharedMatrix& m, int rows, int cols, const char* name) {
    require_native(m && m->nirrep()==1 && m->nrow()==rows && m->ncol()==cols,
                   std::string(name)+" dimensions/C1 symmetry mismatch");
    for (int i=0;i<rows;++i) for (int j=0;j<cols;++j)
        require_native(std::isfinite(m->get(i,j)), std::string(name)+" must be finite");
}
// Validate both representations before any engine/collocation buffer access.
// BasisSet uses one global puream for Libint shells; mixed high-AM Gaussian
// shells cannot be repaired here without changing the supplied AO context.
void validate_shells(const std::shared_ptr<BasisSet>& basis) {
    require_native(basis->nshell()>0 && basis->nshell()<=basis->nbf(), "invalid shell count");
    int offset=0, high_am_purity=-1;
    for(int s=0;s<basis->nshell();++s) {
        const auto& sh=basis->shell(s);
        require_native(sh.am()>=0 && sh.am()<=4 && sh.am()<=basis->max_am() &&
                       sh.nprimitive()>0 && sh.nprimitive()<=64 &&
                       sh.nprimitive()<=basis->max_nprimitive() &&
                       sh.ncenter()>=0 && sh.ncenter()<basis->molecule()->natom() &&
                       sh.function_index()==offset && sh.nfunction()>0 &&
                       sh.nfunction()<=basis->nbf()-offset, "invalid Gaussian shell bounds");
        const auto& l2=basis->l2_shell(s);
        require_native(l2.contr.size()==1, "Libint shell must have exactly one contraction");
        const auto& contraction=l2.contr[0];
        require_native(contraction.l==sh.am() && l2.size()==static_cast<std::size_t>(sh.nfunction()) &&
                       l2.alpha.size()==static_cast<std::size_t>(sh.nprimitive()) &&
                       contraction.coeff.size()==l2.alpha.size(), "inconsistent Libint shell dimensions");
        if(sh.am()>=2) {
            const int purity=sh.is_pure()?1:0;
            require_native(contraction.pure==sh.is_pure() &&
                           (high_am_purity<0 || high_am_purity==purity),
                           "mixed or inconsistent high-AM shell purity unsupported");
            high_am_purity=purity;
        }
        for(int axis=0;axis<3;++axis)
            require_native(std::isfinite(sh.coord(axis)) && l2.O[axis]==sh.coord(axis),
                           "stale or nonfinite Libint shell origin");
        for(int p=0;p<sh.nprimitive();++p)
            require_native(std::isfinite(sh.exp(p)) && sh.exp(p)>0 && l2.alpha[p]==sh.exp(p) &&
                           std::isfinite(contraction.coeff[p]), "invalid Libint shell primitives");
        offset+=sh.nfunction();
    }
    require_native(offset==basis->nbf(), "incomplete Gaussian shell function coverage");
}
// BasisSet is noncopyable. Reconstruct a private basis from original coefficients
// and a deep molecule clone, then verify effective coefficients. Both are needed:
// BasisFunctions uses effective coefficients, BasisSet::update_l2_shells uses
// original coefficients with Libint normalization. No basis name resolution.
std::shared_ptr<BasisSet> snapshot_basis(const std::shared_ptr<BasisSet>& source,
                                       bool bounded_geometry = false) {
    require_native(!source->has_ECP(), "ECP basis snapshots are not supported");
    validate_shells(source);
    std::shared_ptr<Molecule> mol;
    if (bounded_geometry) {
        // Do not clone unbounded comments, variable maps, provenance, labels,
        // connectivity or dummy centres. Only basis-bearing physical/ghost
        // centres in their original order and bohr coordinates are needed.
        const auto original = source->molecule();
        mol = std::make_shared<Molecule>();
        mol->set_units(Molecule::Bohr);
        mol->set_com_fixed(true); mol->set_orientation_fixed(true);
        mol->set_molecular_charge(original->molecular_charge());
        mol->set_multiplicity(original->multiplicity());
        for (int atom=0; atom<original->natom(); ++atom) {
            const auto& symbol = original->atom_entry(atom)->symbol();
            require_native(!symbol.empty() && symbol.size() <= 3,
                           "bounded state requires a short atomic symbol");
            require_native(std::isfinite(original->x(atom)) && std::isfinite(original->y(atom)) &&
                           std::isfinite(original->z(atom)), "molecular coordinates must be finite");
            mol->add_atom(original->Z(atom), original->x(atom), original->y(atom), original->z(atom),
                          symbol, original->mass(atom), original->charge(atom),
                          "NATIVE_STATE_"+std::to_string(atom));
        }
    } else {
        mol = std::make_shared<Molecule>(source->molecule()->clone());
    }
    for (int atom=0;atom<mol->natom();++atom)
        require_native(std::isfinite(mol->x(atom)) && std::isfinite(mol->y(atom)) && std::isfinite(mol->z(atom)),
                       "molecular coordinates must be finite");
    std::map<std::string,std::map<std::string,std::vector<ShellInfo>>> shells, ecps;
    for (int atom=0;atom<mol->natom();++atom) {
        const std::string key="NATIVE_RESPONSE_"+std::to_string(atom);
        mol->set_basis_by_number(atom,key,"BASIS");
        auto& list=shells[key][mol->label(atom)];
        for (int s=0;s<source->nshell();++s) {
            const auto& sh=source->shell(s);
            if (sh.ncenter()!=atom) continue;
            std::vector<double> exponents, coefficients;
            for (int p=0;p<sh.nprimitive();++p) {
                require_native(std::isfinite(sh.exp(p)) && sh.exp(p)>0 && std::isfinite(sh.coef(p)),
                               "invalid basis primitive");
                require_native(std::isfinite(sh.original_coef(p)),"invalid original basis coefficient");
                exponents.push_back(sh.exp(p)); coefficients.push_back(sh.original_coef(p));
            }
            ShellInfo info(sh.am(),coefficients,exponents,sh.is_pure()?Pure:Cartesian,Unnormalized);
            list.push_back(info);
        }
    }
    auto result=std::make_shared<BasisSet>("BASIS",mol,shells,ecps);
    require_native(result->nbf()==source->nbf() && result->nshell()==source->nshell(),
                   "basis reconstruction changed dimensions");
    validate_shells(result);
    for (int s=0;s<source->nshell();++s) {
        const auto& a=source->shell(s); const auto& b=result->shell(s);
        const auto& la=source->l2_shell(s); const auto& lb=result->l2_shell(s);
        require_native(a.am()==b.am() && a.ncenter()==b.ncenter() && a.is_pure()==b.is_pure() &&
                       a.nprimitive()==b.nprimitive() && a.function_index()==b.function_index(),
                       "basis reconstruction changed shell/function order");
        for (int axis=0;axis<3;++axis)
            require_native(std::isfinite(a.coord(axis)) && a.coord(axis)==b.coord(axis) &&
                           source->l2_shell(s).O[axis]==result->l2_shell(s).O[axis],
                           "basis/molecule geometry mismatch or stale integral shell");
        for (int p=0;p<a.nprimitive();++p)
            require_native(a.exp(p)==b.exp(p) && a.original_coef(p)==b.original_coef(p) &&
                           a.coef(p)==b.coef(p) && la.alpha[p]==lb.alpha[p] &&
                           la.contr[0].coeff[p]==lb.contr[0].coeff[p],
                           "basis reconstruction changed primitives (custom normalization unsupported)");
    }
    return result;
}
// Restricted-C1 admission in three steps: dimensions first, then the numerical
// state, then the owned deep snapshot.
struct RestrictedDims { int nbf=0,nmo=0,nocc=0,nvir=0; std::size_t nov=0; };
RestrictedDims restricted_dims(const std::shared_ptr<Wavefunction>& wfn, bool caller_converged) {
    require_native(wfn && caller_converged,"explicit wavefunction and caller convergence declaration required");
    require_native(wfn->nirrep()==1 && wfn->same_a_b_orbs() && wfn->same_a_b_dens() &&
                   wfn->nalpha()==wfn->nbeta() && wfn->soccpi()[0]==0,
                   "only restricted closed-shell C1 wavefunctions are supported");
    auto source=wfn->basisset();
    require_native(source && source->molecule() && source->nbf()>0,"missing orbital basis/molecule");
    RestrictedDims d;
    d.nbf=source->nbf(); d.nmo=wfn->nmo(); d.nocc=wfn->nalpha(); d.nvir=d.nmo-d.nocc;
    require_native(d.nocc>0 && d.nvir>0 && d.nmo<=d.nbf && wfn->doccpi()[0]==d.nocc,
                   "invalid integer Aufbau occupations or empty OV space");
    d.nov=mul(d.nocc,d.nvir);
    return d;
}
void validate_restricted_state(const std::shared_ptr<Wavefunction>& wfn, const RestrictedDims& d) {
    matrix_ok(wfn->Ca(),d.nbf,d.nmo,"Ca"); matrix_ok(wfn->Cb(),d.nbf,d.nmo,"Cb");
    matrix_ok(wfn->Da(),d.nbf,d.nbf,"Da"); matrix_ok(wfn->Db(),d.nbf,d.nbf,"Db");
    auto ea=wfn->epsilon_a(), eb=wfn->epsilon_b();
    require_native(ea && eb && ea->nirrep()==1 && eb->nirrep()==1 && ea->dim()==d.nmo && eb->dim()==d.nmo,
                   "orbital energy dimension/C1 mismatch");
    for(int p=0;p<d.nmo;++p) {
        require_native(std::isfinite(ea->get(p)) && ea->get(p)==eb->get(p),"energies must be finite and restricted");
        for(int mu=0;mu<d.nbf;++mu)
            require_native(wfn->Ca()->get(mu,p)==wfn->Cb()->get(mu,p),"alpha/beta orbitals differ");
    }
    for(int mu=0;mu<d.nbf;++mu) for(int nu=0;nu<d.nbf;++nu) {
        double density=0;
        for(int i=0;i<d.nocc;++i) density+=wfn->Ca()->get(mu,i)*wfn->Ca()->get(nu,i);
        require_native(std::isfinite(density) && wfn->Da()->get(mu,nu)==wfn->Db()->get(mu,nu) &&
                       std::abs(density-wfn->Da()->get(mu,nu))<=1.e-9*std::max(1.0,std::abs(density)),
                       "density inconsistent with integer occupied orbitals (fractional occupations unsupported)");
    }
    for(int a=0;a<d.nvir;++a) for(int i=0;i<d.nocc;++i) {
        double gap=ea->get(d.nocc+a)-ea->get(i);
        require_native(std::isfinite(gap) && gap>0,"occupied-virtual gaps must be finite and positive");
    }
}
struct OwnedState { std::shared_ptr<BasisSet> basis; SharedMatrix c, da; SharedVector eps; };
OwnedState own_restricted_state(const std::shared_ptr<Wavefunction>& wfn, const RestrictedDims& d,
                               bool bounded_geometry = false) {
    OwnedState s;
    if (bounded_geometry) {
        // Copy numerical state only: caller-controlled matrix/vector names
        // and Dimension metadata are not bounded by scientific dimensions.
        s.c=std::make_shared<Matrix>(d.nbf,d.nmo);
        s.da=std::make_shared<Matrix>(d.nbf,d.nbf);
        s.eps=std::make_shared<Vector>(d.nmo);
        for(int mu=0;mu<d.nbf;++mu) {
            for(int p=0;p<d.nmo;++p) s.c->set(mu,p,wfn->Ca()->get(mu,p));
            for(int nu=0;nu<d.nbf;++nu) s.da->set(mu,nu,wfn->Da()->get(mu,nu));
        }
        for(int p=0;p<d.nmo;++p) s.eps->set(p,wfn->epsilon_a()->get(p));
    } else {
        s.c=wfn->Ca()->clone(); s.da=wfn->Da()->clone();
        s.eps=std::make_shared<Vector>(*wfn->epsilon_a());
    }
    s.basis=snapshot_basis(wfn->basisset(), bounded_geometry);
    auto overlap=std::make_shared<Matrix>(d.nbf,d.nbf);
    // Native OverlapInt over the ordinary (normalized, source-puream) snapshot,
    // driven serially shell pair by shell pair: no Process threads, and one-body
    // integrals consult no tolerance option. Not the explicit-AUX raw bridge.
    // snapshot_basis has validated all shell dimensions and AO offsets before
    // these ordered shell-pair buffers are indexed.
    IntegralFactory factory(s.basis);
    auto overlap_int=factory.ao_overlap();
    for(int s0=0;s0<s.basis->nshell();++s0) for(int s1=0;s1<s.basis->nshell();++s1) {
        const auto& sh0=s.basis->shell(s0); const auto& sh1=s.basis->shell(s1);
        overlap_int->compute_shell(s0,s1);
        const double* buf=overlap_int->buffers()[0];
        if(!buf) continue;
        std::size_t index=0;
        for(int m=0;m<sh0.nfunction();++m) for(int n=0;n<sh1.nfunction();++n,++index) {
            require_native(std::isfinite(buf[index]), "nonfinite overlap integral");
            overlap->set(sh0.function_index()+m,sh1.function_index()+n,buf[index]);
        }
    }
    auto gram=linalg::triplet(s.c,overlap,s.c,true,false,false);
    for(int p=0;p<d.nmo;++p) for(int q=0;q<d.nmo;++q)
        require_native(std::isfinite(gram->get(p,q)) && std::abs(gram->get(p,q)-(p==q?1.0:0.0))<=1.e-8,
                       "orbitals must be AO-overlap orthonormal");
    return s;
}
} // namespace

NativeRestrictedState::NativeRestrictedState(std::shared_ptr<Wavefunction> wfn,
        bool caller_converged, std::size_t max_bytes) {
    auto d = restricted_dims(wfn, caller_converged);
    require_native(max_bytes > 0, "state byte budget must be positive");
    auto source = wfn->basisset();
    planned_bytes_ = add(mul(mul(64, mul(d.nbf, d.nbf)), sizeof(double)),
                         16ULL*1024*1024);
    planned_bytes_ = add(planned_bytes_, mul(source->nshell(), 16*1024));
    planned_bytes_ = add(planned_bytes_, mul(source->molecule()->natom(), 4*1024));
    require_native(planned_bytes_ <= max_bytes, "restricted state byte resource limit exceeded");
    validate_restricted_state(wfn, d);
    require_native(std::isfinite(wfn->energy()), "nonfinite wavefunction energy");
    auto owned = own_restricted_state(wfn, d, true);
    basis_ = std::move(owned.basis);
    c_ = std::move(owned.c); da_ = std::move(owned.da); eps_ = std::move(owned.eps);
    nbf_ = d.nbf; nmo_ = d.nmo; nocc_ = d.nocc; nvir_ = d.nvir; nov_ = d.nov;
}
SharedMatrix NativeRestrictedState::orbitals() const { return c_->clone(); }
SharedVector NativeRestrictedState::energies() const { return std::make_shared<Vector>(*eps_); }
SharedMatrix NativeRestrictedState::density_alpha() const { return da_->clone(); }
std::shared_ptr<BasisSet> NativeRestrictedState::basis_snapshot() const {
    return snapshot_basis(basis_, true);
}

} } // namespace psi::isapol
