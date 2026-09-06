/* Psi4: Copyright (c) 2007-2026 The Psi4 Developers.
 * SPDX-License-Identifier: LGPL-3.0-only
 */
#include "aux_coulomb.h"
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
    aux_int_require(basis_.representation_==IsaBasisRepresentation::Cartesian,
                    "Initial native Coulomb provider requires Cartesian GAMINT AUX");
}
std::vector<double> IsaAuxCoulomb::charges() const {
    std::vector<double> q;
    const double pi=std::acos(-1.);
    for (const auto& shell:basis_.shells_) {
        for (const auto& power:IsaExplicitBasis::cartesian_powers(shell.l)) {
            double value=0.;
            if (!(power[0]%2 || power[1]%2 || power[2]%2)) {
                for (size_t p=0;p<shell.exponents.size();++p) {
                    const double a=shell.exponents[p];
                    double moment=std::pow(pi/a,1.5);
                    for (int axis=0;axis<3;++axis)
                        moment*=aux_df(power[axis]-1)/std::pow(2*a,power[axis]/2);
                    value+=shell.coefficients[p]*moment;
                }
                value*=aux_factor(shell.l,power);
            }
            aux_int_require(std::isfinite(value),"Nonfinite AUX charge integral");
            q.push_back(value);
        }
    }
    return q;
}
std::shared_ptr<Matrix> IsaAuxCoulomb::metric() const {
    std::vector<libint2::Shell> shells;
    std::vector<int> offsets;
    size_t max_nprim=0;
    int max_l=0, offset=0;
    for (const auto& s:basis_.shells_) {
        libint2::svector<double> exponents(s.exponents.begin(),s.exponents.end());
        libint2::svector<double> coefficients(s.coefficients.begin(),s.coefficients.end());
        // Fourth argument false preserves the already-effective coefficients.
        shells.emplace_back(exponents,libint2::svector<libint2::Shell::Contraction>{{s.l,false,coefficients}},
                            basis_.centres_[s.centre],false);
        offsets.push_back(offset);
        offset+=(s.l+1)*(s.l+2)/2;
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
