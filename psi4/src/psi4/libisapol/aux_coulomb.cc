/* Psi4: Copyright (c) 2007-2026 The Psi4 Developers.
 * SPDX-License-Identifier: LGPL-3.0-only
 */
#include "aux_coulomb.h"
#include "harmonic_transform.h"
#include "psi4/libmints/matrix.h"
#include <libint2.hpp>
#include <libint2/cgshell_ordering.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>
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
    std::vector<libint2::Shell> shells;
    std::vector<int> offsets;
    std::vector<IsaHarmonicTransform> transforms;
    size_t max_nprim=0;
    int max_l=0, offset=0;
    for (const auto& s:basis_.shells_) {
        libint2::svector<double> exponents(s.exponents.begin(),s.exponents.end());
        libint2::svector<double> coefficients(s.coefficients.begin(),s.coefficients.end());
        // Fourth argument false preserves the already-effective coefficients.
        shells.emplace_back(exponents,libint2::svector<libint2::Shell::Contraction>{{s.l,false,coefficients}},
                            basis_.centres_[s.centre],false);
        offsets.push_back(offset);
        offset+=IsaExplicitBasis::shell_size(s.l,basis_.representation_);
        transforms.push_back(pure ? isa_dalton_transform(s.l) : IsaHarmonicTransform{});
        max_nprim=std::max(max_nprim,s.exponents.size()); max_l=std::max(max_l,s.l);
    }
    constexpr auto op=libint2::Operator::coulomb;
    constexpr auto braket=libint2::BraKet::xs_xs;
    libint2::Engine engine(op,max_nprim,max_l,0,0.0,
                          libint2::operator_traits<op>::default_params(),braket);
    engine.set(libint2::CartesianShellNormalization::standard);
    const auto& unit=libint2::Shell::unit();
    auto result=std::make_shared<Matrix>("Native explicit AUX Coulomb metric",offset,offset);
    for (size_t a=0;a<shells.size();++a) for (size_t b=0;b<=a;++b) {
        engine.compute2<op,braket,0>(shells[a],unit,shells[b],unit);
        const double* block=engine.results()[0];
        if (!block) continue;
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
} }
