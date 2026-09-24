/* Psi4: Copyright (c) 2007-2026 The Psi4 Developers.
 * SPDX-License-Identifier: LGPL-3.0-only
 */
#include "aux_coulomb.h"
#include "harmonic_transform.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libqt/qt.h"
#include <libint2.hpp>
#include <libint2/cgshell_ordering.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
namespace psi { namespace isapol {
namespace {
void orbital_require(bool ok,const char* message) { if (!ok) throw std::invalid_argument(message); }
double orbital_df(int n) { double v=1.; for (;n>0;n-=2) v*=n; return v; }
libint2::Shell raw_shell(const IsaGaussianShell& s,const std::array<double,3>& centre) {
    libint2::svector<double> a(s.exponents.begin(),s.exponents.end()),c(s.coefficients.begin(),s.coefficients.end());
    return libint2::Shell(a,{{s.l,false,c}},centre,false);
}
}
std::shared_ptr<Matrix> IsaAuxCoulomb::three_center_shell_block(const IsaExplicitBasis& orbital,
        std::size_t first_shell, std::size_t shell_count, std::size_t max_bytes) const {
    orbital_require(orbital.role_==IsaBasisRole::Orbital &&
                    orbital.representation_==IsaBasisRepresentation::Spherical,
                    "Three-centre MAIN requires DALTON spherical Orbital basis");
    orbital_require(first_shell < basis_.shells_.size() && shell_count > 0 &&
                    shell_count <= basis_.shells_.size()-first_shell,
                    "Invalid AUX shell block range");
    std::size_t rows=0;
    for (std::size_t s=first_shell;s<first_shell+shell_count;++s) {
        rows+=IsaExplicitBasis::shell_size(basis_.shells_[s].l,basis_.representation_);
        orbital_require(rows<=512,"AUX shell block resource limit (maximum 512 functions)");
    }
    const int n=orbital.nfunction();
    orbital_require(n>0 && n<=std::numeric_limits<int>::max()/n,"MAIN pair dimension overflow");
    orbital_require(max_bytes>0 && max_bytes<=512UL*1024*1024 &&
                    rows<=max_bytes/sizeof(double)/n/n,
                    "AUX shell block matrix byte resource limit");
    // Reuse the exact integral/transform kernel with only selected AUX shells.
    // This neither computes discarded AUX rows nor changes arithmetic within
    // a retained shell triple; all MAIN functions and both pair orders remain.
    std::vector<IsaGaussianShell> selected(basis_.shells_.begin()+first_shell,
                                          basis_.shells_.begin()+first_shell+shell_count);
    IsaExplicitBasis subset(IsaBasisRole::MolecularAux,basis_.representation_,basis_.centres_,selected);
    return IsaAuxCoulomb(subset).three_center(orbital);
}
std::shared_ptr<Matrix> IsaAuxCoulomb::three_center(const IsaExplicitBasis& orbital) const {
    orbital_require(orbital.role_==IsaBasisRole::Orbital && orbital.representation_==IsaBasisRepresentation::Spherical,
                    "Three-centre MAIN requires DALTON spherical Orbital basis");
    const int n=orbital.nfunction();
    orbital_require(n<=std::numeric_limits<int>::max()/n,"MAIN pair dimension overflow");
    std::vector<libint2::Shell> aux,main;
    std::vector<int> ao,mo;
    std::vector<IsaHarmonicTransform> transforms;
    // A spherical molecular AUX is a DIFFERENT declared basis from the Cartesian
    // GAMINT one built from the same exponents, never a normalization variant of
    // it: it spans 2l+1 rather than (l+1)(l+2)/2 functions per shell.
    const bool pure=basis_.representation_==IsaBasisRepresentation::Spherical;
    std::vector<IsaHarmonicTransform> aux_transforms;
    size_t max_prim=0; int max_l=0,offset=0;
    for (const auto& s:basis_.shells_) {
        ao.push_back(offset); offset+=IsaExplicitBasis::shell_size(s.l,basis_.representation_);
        aux.push_back(raw_shell(s,basis_.centres_[s.centre]));
        aux_transforms.push_back(pure ? isa_dalton_transform(s.l) : IsaHarmonicTransform{});
        max_prim=std::max(max_prim,s.exponents.size()); max_l=std::max(max_l,s.l);
    }
    offset=0;
    for (const auto& s:orbital.shells_) {
        mo.push_back(offset); offset+=2*s.l+1;
        main.push_back(raw_shell(s,orbital.centres_[s.centre]));
        transforms.push_back(isa_dalton_transform(s.l));
        max_prim=std::max(max_prim,s.exponents.size()); max_l=std::max(max_l,s.l);
    }
    constexpr auto op=libint2::Operator::coulomb;
    constexpr auto bk=libint2::BraKet::xs_xx;
    libint2::Engine engine(op,max_prim,max_l,0,0.,libint2::operator_traits<op>::default_params(),bk);
    engine.set(libint2::CartesianShellNormalization::standard);
    const auto& unit=libint2::Shell::unit();
    auto result=std::make_shared<Matrix>("Native AUX MAIN MAIN Coulomb",basis_.nfunction(),n*n);
    for (size_t a=0;a<aux.size();++a) for (size_t b=0;b<main.size();++b) for (size_t c=0;c<=b;++c) {
        engine.compute2<op,bk,0>(aux[a],unit,main[b],main[c]);
        const double* block=engine.results()[0];
        if (!block) continue;
        const int la=basis_.shells_[a].l;
        const auto& powers=IsaExplicitBasis::cartesian_powers(la);
        const int nb=main[b].cartesian_size(),nc=main[c].cartesian_size();
        const size_t ri=transforms[b].size(),rj=transforms[c].size();
        // Raw AUX monomial index -> (spherical row, coefficient). Built per shell
        // triple so the Cartesian path allocates nothing and keeps its arithmetic.
        std::vector<std::vector<std::pair<int,double>>> rows_of;
        std::vector<double> acc;
        if (pure) {
            rows_of.assign(powers.size(),{});
            for (size_t r=0;r<aux_transforms[a].size();++r)
                for (const auto& e:aux_transforms[a][r]) rows_of[e.first].emplace_back(int(r),e.second);
            acc.assign(aux_transforms[a].size()*ri*rj,0.);
        }
        for (size_t k=0;k<powers.size();++k) {
            const auto& p=powers[k];
            const int raw=libint2::INT_CARTINDEX(la,p[0],p[1]);
            // The GAMINT mixed-component factor belongs to a Cartesian AUX function
            // alone; a harmonic row is a combination of RAW monomials.
            const double factor=pure ? 1. :
                std::sqrt(orbital_df(2*la-1)/(orbital_df(2*p[0]-1)*orbital_df(2*p[1]-1)*orbital_df(2*p[2]-1)));
            for (size_t i=0;i<ri;++i) for (size_t j=0;j<rj;++j) {
                if (b==c && j>i) continue;
                double value=0.;
                for (const auto& x:transforms[b][i]) for (const auto& y:transforms[c][j])
                    value+=x.second*y.second*block[(raw*nb+x.first)*nc+y.first];
                value*=factor;
                orbital_require(std::isfinite(value),"Nonfinite native three-centre integral");
                if (pure) {
                    for (const auto& e:rows_of[raw]) acc[(e.first*ri+i)*rj+j]+=e.second*value;
                    continue;
                }
                const int mu=mo[b]+i,nu=mo[c]+j;
                result->set(ao[a]+k,mu*n+nu,value);
                result->set(ao[a]+k,nu*n+mu,value);
            }
        }
        if (!pure) continue;
        for (size_t r=0;r<aux_transforms[a].size();++r)
            for (size_t i=0;i<ri;++i) for (size_t j=0;j<rj;++j) {
                if (b==c && j>i) continue;
                const double value=acc[(r*ri+i)*rj+j];
                orbital_require(std::isfinite(value),"Nonfinite native three-centre integral");
                const int mu=mo[b]+i,nu=mo[c]+j;
                result->set(ao[a]+r,mu*n+nu,value);
                result->set(ao[a]+r,nu*n+mu,value);
            }
    }
    return result;
}
std::vector<double> IsaAuxCoulomb::closed_shell_rhs(const IsaExplicitBasis& orbital,const Matrix& occupied) const {
    const int n=orbital.nfunction();
    orbital_require(occupied.nirrep()==1 && occupied.nrow()==n && occupied.ncol()>0 && occupied.ncol()<=n,
                    "Invalid occupied coefficient dimensions");
    for (int mu=0;mu<n;++mu) for (int i=0;i<occupied.ncol();++i)
        orbital_require(std::isfinite(occupied.get(mu,i)),"Nonfinite occupied coefficient");
    const auto b=three_center(orbital);
    std::vector<double> rhs(basis_.nfunction(),0.);
    for (int k=0;k<basis_.nfunction();++k) for (int i=0;i<occupied.ncol();++i) {
        double diagonal=0.;
        for (int mu=0;mu<n;++mu) {
            double value=0.;
            for (int nu=0;nu<n;++nu) value+=b->get(k,mu*n+nu)*occupied.get(nu,i);
            diagonal+=occupied.get(mu,i)*value;
        }
        rhs[k]+=2*diagonal;
        orbital_require(std::isfinite(rhs[k]),"Nonfinite occupied-trace RHS");
    }
    return rhs;
}
IsaDrhoCResult IsaAuxCoulomb::fit_drho_c(const IsaExplicitBasis& orbital,const Matrix& occupied,double penalty) const {
    orbital_require(std::isfinite(penalty) && penalty>0.,"Drho-C charge penalty must be finite and positive");
    IsaDrhoCResult result;
    result.charge_penalty=penalty;
    result.raw_rhs=closed_shell_rhs(orbital,occupied);  // also validates inputs
    result.charges=charges();
    result.coulomb_metric=metric();
    const int n=basis_.nfunction(),nm=orbital.nfunction();
    result.metric=std::make_shared<Matrix>("Native Drho-C constrained metric",n,n);
    result.rhs.assign(n,0.);
    // Retain explicit pair-diagonal penalty ordering from the reference.
    // Recompute fresh B rather than retaining mutable integral views in provider.
    const auto b=three_center(orbital);
    for (int k=0;k<n;++k) {
        const double penalty_q=penalty*result.charges[k];
        for (int j=0;j<n;++j) {
            const double value=result.coulomb_metric->get(k,j)+penalty_q*result.charges[j];
            orbital_require(std::isfinite(value),"Nonfinite constrained metric");
            result.metric->set(k,j,value); // no silent symmetrization
        }
        for (int i=0;i<occupied.ncol();++i) {
            double diagonal=0.;
            for (int mu=0;mu<nm;++mu) {
                double value=0.;
                for (int nu=0;nu<nm;++nu) value+=b->get(k,mu*nm+nu)*occupied.get(nu,i);
                diagonal+=occupied.get(mu,i)*value;
            }
            result.rhs[k]+=2*(diagonal+penalty_q);
        }
        orbital_require(std::isfinite(result.rhs[k]),"Nonfinite constrained RHS");
    }
    // Pack column-major explicitly; constrained A has roundoff-level asymmetry.
    std::vector<double> lu(static_cast<size_t>(n)*n);
    std::vector<int> pivots(n);
    for (int i=0;i<n;++i) for (int j=0;j<n;++j) lu[i+static_cast<size_t>(j)*n]=result.metric->get(i,j);
    result.coefficients=result.rhs;
    const int info=C_DGESV(n,1,lu.data(),n,pivots.data(),result.coefficients.data(),n);
    orbital_require(info==0,"Native Drho-C LU solve failed");
    long double residual=0.,anorm=0.,dnorm=0.,bnorm=0.,electrons=0.;
    for (int i=0;i<n;++i) {
        orbital_require(std::isfinite(result.coefficients[i]),"Nonfinite Drho-C coefficient");
        dnorm=std::max(dnorm,std::abs(static_cast<long double>(result.coefficients[i])));
        bnorm=std::max(bnorm,std::abs(static_cast<long double>(result.rhs[i])));
        electrons+=static_cast<long double>(result.charges[i])*result.coefficients[i];
        long double ad=0.,row=0.;
        for (int j=0;j<n;++j) {
            const long double value=result.metric->get(i,j);
            ad+=value*result.coefficients[j]; row+=std::abs(value);
        }
        anorm=std::max(anorm,row);
        residual=std::max(residual,std::abs(ad-result.rhs[i]));
    }
    const long double denominator=anorm*dnorm+bnorm;
    result.relative_residual=denominator>0.?static_cast<double>(residual/denominator):0.;
    result.fitted_electrons=static_cast<double>(electrons);
    orbital_require(std::isfinite(result.relative_residual) && std::isfinite(result.fitted_electrons),
                    "Nonfinite Drho-C diagnostics");
    return result;
}
} }
