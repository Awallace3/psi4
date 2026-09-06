/* Psi4: Copyright (c) 2007-2026 The Psi4 Developers.
 * SPDX-License-Identifier: LGPL-3.0-only
 */
#include "aux_coulomb.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libqt/qt.h"
#include <libint2.hpp>
#include <libint2/cgshell_ordering.h>
#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <map>
#include <stdexcept>
namespace psi { namespace isapol {
namespace {
using Powers=std::array<int,3>;
using Polynomial=std::map<Powers,std::complex<double>>;
using Transform=std::vector<std::vector<std::pair<int,double>>>;
void orbital_require(bool ok,const char* message) { if (!ok) throw std::invalid_argument(message); }
double orbital_df(int n) { double v=1.; for (;n>0;n-=2) v*=n; return v; }
double orbital_fac(int n) { double v=1.; for (;n>1;--n) v*=n; return v; }
Polynomial shift(const Polynomial& p,int axis) {
    Polynomial out;
    for (const auto& term:p) { auto key=term.first; ++key[axis]; out[key]+=term.second; }
    return out;
}
Polynomial combine(const Polynomial& a,std::complex<double> x,const Polynomial& b,std::complex<double> y) {
    Polynomial out;
    for (const auto& term:a) out[term.first]+=x*term.second;
    for (const auto& term:b) out[term.first]+=y*term.second;
    return out;
}
Polynomial times_r2(const Polynomial& p) {
    Polynomial out;
    for (int axis=0;axis<3;++axis) out=combine(out,1.,shift(shift(p,axis),axis),1.);
    return out;
}
Transform dalton_transform(int l) {
    // Symbolic regular-harmonic recurrence, not Libint's runtime SH transform.
    // No Condon--Shortley phase; rows sin(l..1), m0, cos(1..l), with p=x,y,z.
    std::vector<Polynomial> h(l+1);
    for (int m=0;m<=l;++m) {
        Polynomial prev{{Powers{{0,0,0}},orbital_df(2*m-1)}};
        for (int k=0;k<m;++k) prev=combine(shift(prev,0),1.,shift(prev,1),{0.,1.});
        Polynomial value=prev;
        if (l>m) {
            value=combine(shift(prev,2),double(2*m+1),{},0.);
            for (int n=m+2;n<=l;++n) {
                auto next=combine(shift(value,2),double(2*n-1)/double(n-m),
                                  times_r2(prev),-double(n+m-1)/double(n-m));
                prev=value; value=next;
            }
        }
        const double scale=std::sqrt((m?2.:1.)*orbital_fac(l-m)/orbital_fac(l+m));
        h[m]=combine(value,scale,{},0.);
    }
    Transform rows;
    auto append=[&](int m,bool imaginary) {
        rows.emplace_back();
        for (const auto& term:h[m]) {
            const double c=imaginary?term.second.imag():term.second.real();
            if (c!=0.) rows.back().emplace_back(libint2::INT_CARTINDEX(l,term.first[0],term.first[1]),c);
        }
    };
    if (l==1) { append(1,false); append(1,true); append(0,false); }
    else {
        for (int m=l;m>0;--m) append(m,true);
        append(0,false);
        for (int m=1;m<=l;++m) append(m,false);
    }
    return rows;
}
libint2::Shell raw_shell(const IsaGaussianShell& s,const std::array<double,3>& centre) {
    libint2::svector<double> a(s.exponents.begin(),s.exponents.end()),c(s.coefficients.begin(),s.coefficients.end());
    return libint2::Shell(a,{{s.l,false,c}},centre,false);
}
}
std::shared_ptr<Matrix> IsaAuxCoulomb::three_center(const IsaExplicitBasis& orbital) const {
    orbital_require(orbital.role_==IsaBasisRole::Orbital && orbital.representation_==IsaBasisRepresentation::Spherical,
                    "Three-centre MAIN requires DALTON spherical Orbital basis");
    const int n=orbital.nfunction();
    orbital_require(n<=std::numeric_limits<int>::max()/n,"MAIN pair dimension overflow");
    std::vector<libint2::Shell> aux,main;
    std::vector<int> ao,mo;
    std::vector<Transform> transforms;
    size_t max_prim=0; int max_l=0,offset=0;
    for (const auto& s:basis_.shells_) {
        ao.push_back(offset); offset+=(s.l+1)*(s.l+2)/2;
        aux.push_back(raw_shell(s,basis_.centres_[s.centre]));
        max_prim=std::max(max_prim,s.exponents.size()); max_l=std::max(max_l,s.l);
    }
    offset=0;
    for (const auto& s:orbital.shells_) {
        mo.push_back(offset); offset+=2*s.l+1;
        main.push_back(raw_shell(s,orbital.centres_[s.centre]));
        transforms.push_back(dalton_transform(s.l));
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
        for (size_t k=0;k<powers.size();++k) {
            const auto& p=powers[k];
            const int raw=libint2::INT_CARTINDEX(la,p[0],p[1]);
            const double factor=std::sqrt(orbital_df(2*la-1)/(orbital_df(2*p[0]-1)*orbital_df(2*p[1]-1)*orbital_df(2*p[2]-1)));
            for (size_t i=0;i<transforms[b].size();++i) for (size_t j=0;j<transforms[c].size();++j) {
                if (b==c && j>i) continue;
                double value=0.;
                for (const auto& x:transforms[b][i]) for (const auto& y:transforms[c][j])
                    value+=x.second*y.second*block[(raw*nb+x.first)*nc+y.first];
                value*=factor;
                orbital_require(std::isfinite(value),"Nonfinite native three-centre integral");
                const int mu=mo[b]+i,nu=mo[c]+j;
                result->set(ao[a]+k,mu*n+nu,value);
                result->set(ao[a]+k,nu*n+mu,value);
            }
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
