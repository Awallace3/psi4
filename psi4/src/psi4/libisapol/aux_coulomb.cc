/* Psi4: Copyright (c) 2007-2026 The Psi4 Developers.
 * SPDX-License-Identifier: LGPL-3.0-only
 */
#include "aux_coulomb.h"
#include "harmonic_transform.h"
#include "psi4/libmints/basisset.h"
#include "psi4/libmints/eri.h"
#include "psi4/libmints/gshell.h"
#include "psi4/libmints/integral.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/molecule.h"
#include <libint2/cgshell_ordering.h>
#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>
#include <string>
namespace psi { namespace isapol {
namespace {
void aux_int_require(bool ok, const char* message) {
    if (!ok) throw std::invalid_argument(message);
}
double aux_df(int n) { double v=1.; for (;n>0;n-=2) v*=n; return v; }
double aux_factor(int l, const std::array<int,3>& p) {
    return std::sqrt(aux_df(2*l-1)/(aux_df(2*p[0]-1)*aux_df(2*p[1]-1)*aux_df(2*p[2]-1)));
}
}
IsaAuxCoulomb::NativeBasis IsaAuxCoulomb::native_basis(const IsaExplicitBasis& explicit_basis) {
    const int ncentre=explicit_basis.ncentre();
    auto molecule=std::make_shared<Molecule>();
    molecule->set_units(Molecule::Bohr);
    // add_atom refuses centres closer than 0.05 bohr, but distinct explicit
    // centres may coincide: add them apart, then assign the bohr coordinates.
    Matrix geometry(ncentre,3);
    for (int c=0;c<ncentre;++c) {
        molecule->add_atom(0.,10.*c,0.,0.,"GH",0.,0.,"GH"+std::to_string(c+1));
        for (int axis=0;axis<3;++axis) geometry.set(c,axis,explicit_basis.centres_[c][axis]);
    }
    molecule->set_geometry(geometry);
    molecule->set_basis_all_atoms("ISAPOL-EXPLICIT","ISAPOL");
    std::map<std::string,std::map<std::string,std::vector<ShellInfo>>> shells,ecp;
    auto& by_centre=shells["ISAPOL-EXPLICIT"];
    for (int c=0;c<ncentre;++c) by_centre["GH"+std::to_string(c+1)];
    std::vector<std::vector<int>> members(ncentre);
    for (size_t s=0;s<explicit_basis.shells_.size();++s) {
        const auto& shell=explicit_basis.shells_[s];
        members[shell.centre].push_back(static_cast<int>(s));
        // Unnormalized only populates every construction array; its normalized
        // copies are replaced by the stored coefficients below.
        by_centre["GH"+std::to_string(shell.centre+1)].emplace_back(
            shell.l,shell.coefficients,shell.exponents,Cartesian,Unnormalized);
    }
    NativeBasis result{std::make_shared<BasisSet>("ISAPOL",molecule,shells,ecp),
                       std::vector<int>(explicit_basis.shells_.size())};
    result.basis->use_original_coefficients_as_effective();
    int next=0;
    for (const auto& centre:members) for (int s:centre) result.shells[s]=next++;
    return result;
}
IsaAuxCoulomb::IsaAuxCoulomb(const IsaExplicitBasis& auxiliary) : basis_(auxiliary) {
    aux_int_require(basis_.role_==IsaBasisRole::MolecularAux,"Coulomb provider requires molecular AUX role");
    aux_int_require(basis_.representation_==IsaBasisRepresentation::Cartesian ||
                    basis_.representation_==IsaBasisRepresentation::Spherical,
                    "Coulomb provider requires a Cartesian GAMINT or spherical DALTON AUX");
}
std::vector<double> IsaAuxCoulomb::charges() const {
    const bool pure=basis_.representation_==IsaBasisRepresentation::Spherical;
    std::vector<double> q;
    const double pi=std::acos(-1.);
    for (const auto& shell:basis_.shells_) {
        const auto& powers=IsaExplicitBasis::cartesian_powers(shell.l);
        // Raw monomial integrals first, in Libint standard indexing, so a spherical
        // row is the same combination the grid evaluation uses. A harmonic row of
        // l>0 cancels to roundoff rather than to an imposed zero; that residue is
        // the integral this basis actually has and is never clipped.
        std::vector<double> raw(pure ? powers.size() : 0,0.);
        for (size_t k=0;k<powers.size();++k) {
            const auto& power=powers[k];
            double value=0.;
            if (!(power[0]%2 || power[1]%2 || power[2]%2)) {
                for (size_t p=0;p<shell.exponents.size();++p) {
                    const double a=shell.exponents[p];
                    double moment=std::pow(pi/a,1.5);
                    for (int axis=0;axis<3;++axis)
                        moment*=aux_df(power[axis]-1)/std::pow(2*a,power[axis]/2);
                    value+=shell.coefficients[p]*moment;
                }
                if (!pure) value*=aux_factor(shell.l,power);
            }
            aux_int_require(std::isfinite(value),"Nonfinite AUX charge integral");
            if (pure) raw[libint2::INT_CARTINDEX(shell.l,power[0],power[1])]=value;
            else q.push_back(value);
        }
        if (!pure) continue;
        for (const auto& row:isa_dalton_transform(shell.l)) {
            double value=0.;
            for (const auto& e:row) value+=e.second*raw[e.first];
            aux_int_require(std::isfinite(value),"Nonfinite AUX charge integral");
            q.push_back(value);
        }
    }
    return q;
}
std::shared_ptr<Matrix> IsaAuxCoulomb::metric() const {
    const bool pure=basis_.representation_==IsaBasisRepresentation::Spherical;
    std::vector<int> offsets;
    std::vector<IsaHarmonicTransform> transforms;
    int offset=0;
    for (const auto& s:basis_.shells_) {
        offsets.push_back(offset);
        offset+=IsaExplicitBasis::shell_size(s.l,basis_.representation_);
        transforms.push_back(pure ? isa_dalton_transform(s.l) : IsaHarmonicTransform{});
    }
    // Direct unscreened Libint2ERI, independent of INTEGRAL_PACKAGE and SCREENING:
    // compute_shell runs the zero-precision engine with no precomputed shell pairs
    // and no sieve. Standard Cartesian components.
    const auto aux=native_basis(basis_);
    const auto zero=BasisSet::zero_ao_basis_set();
    IntegralFactory factory(aux.basis,zero,aux.basis,zero);
    Libint2ERI eri(&factory,0.,0,false,false,false);
    auto result=std::make_shared<Matrix>("Native explicit AUX Coulomb metric",offset,offset);
    for (size_t a=0;a<basis_.shells_.size();++a) for (size_t b=0;b<=a;++b) {
        if (!eri.compute_shell(aux.shells[a],0,aux.shells[b],0)) continue;
        const double* block=eri.buffer();
        const int la=basis_.shells_[a].l,lb=basis_.shells_[b].l;
        const auto& pa=IsaExplicitBasis::cartesian_powers(la);
        const auto& pb=IsaExplicitBasis::cartesian_powers(lb);
        if (pure) {
            // Both indices carry raw monomials into their own harmonic rows; the
            // GAMINT factor is a Cartesian function's own normalization and has no
            // place here. Pair blocks are small, so contract them directly.
            for (size_t i=0;i<transforms[a].size();++i) for (size_t j=0;j<transforms[b].size();++j) {
                if (a==b && j>i) continue;
                double value=0.;
                for (const auto& x:transforms[a][i]) for (const auto& y:transforms[b][j])
                    value+=x.second*y.second*block[x.first*pb.size()+y.first];
                aux_int_require(std::isfinite(value),"Nonfinite AUX Coulomb integral");
                result->set(offsets[a]+i,offsets[b]+j,value);
                result->set(offsets[b]+j,offsets[a]+i,value);
            }
            continue;
        }
        for (size_t i=0;i<pa.size();++i) for (size_t j=0;j<pb.size();++j) {
            if (a==b && j>i) continue;
            const int ii=libint2::INT_CARTINDEX(la,pa[i][0],pa[i][1]);
            const int jj=libint2::INT_CARTINDEX(lb,pb[j][0],pb[j][1]);
            // Standard engine components are raw monomials. Apply the GAMINT
            // mixed-component factor exactly once, with configured indexing.
            const double value=block[ii*pb.size()+jj]*aux_factor(la,pa[i])*aux_factor(lb,pb[j]);
            aux_int_require(std::isfinite(value),"Nonfinite AUX Coulomb integral");
            result->set(offsets[a]+i,offsets[b]+j,value);
            result->set(offsets[b]+j,offsets[a]+i,value);
        }
    }
    return result;
}
std::pair<std::shared_ptr<BasisSet>, std::shared_ptr<Matrix>> IsaAuxCoulomb::native_auxiliary() const {
    const bool pure=basis_.representation_==IsaBasisRepresentation::Spherical;
    auto aux=native_basis(basis_);
    auto map=std::make_shared<Matrix>("Declared AUX map T (declared x raw)",basis_.nfunction_,aux.basis->nbf());
    int offset=0;
    for (size_t s=0;s<basis_.shells_.size();++s) {
        const int l=basis_.shells_[s].l;
        const int first=aux.basis->shell_to_basis_function(aux.shells[s]);
        if (pure) {
            const auto rows=isa_dalton_transform(l);
            for (size_t i=0;i<rows.size();++i)
                for (const auto& e:rows[i]) map->set(offset+static_cast<int>(i),first+e.first,e.second);
            offset+=static_cast<int>(rows.size());
        } else {
            const auto& powers=IsaExplicitBasis::cartesian_powers(l);
            for (size_t i=0;i<powers.size();++i)
                map->set(offset+static_cast<int>(i),first+libint2::INT_CARTINDEX(l,powers[i][0],powers[i][1]),
                         aux_factor(l,powers[i]));
            offset+=static_cast<int>(powers.size());
        }
    }
    aux_int_require(offset==basis_.nfunction_,"Declared AUX map does not cover the basis");
    return {aux.basis,map};
}
} }
