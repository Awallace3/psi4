/* Psi4: Copyright (c) 2007-2026 The Psi4 Developers.
 * SPDX-License-Identifier: LGPL-3.0-only
 * CamCASP-compatible conventions; independent solid-harmonic recurrence.
 */
#include "harmonic_transform.h"
#include <libint2.hpp>
#include <libint2/cgshell_ordering.h>
#include <cmath>
#include <array>
#include <complex>
#include <map>
namespace psi { namespace isapol {
namespace {
using Powers=std::array<int,3>;
using Polynomial=std::map<Powers,std::complex<double>>;
double harmonic_df(int n) { double v=1.; for (;n>0;n-=2) v*=n; return v; }
double harmonic_fac(int n) { double v=1.; for (;n>1;--n) v*=n; return v; }
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
}
IsaHarmonicTransform isa_dalton_transform(int l) {
    // Symbolic regular-harmonic recurrence, not Libint's runtime SH transform.
    // No Condon--Shortley phase; rows sin(l..1), m0, cos(1..l), with p=x,y,z.
    std::vector<Polynomial> h(l+1);
    for (int m=0;m<=l;++m) {
        Polynomial prev{{Powers{{0,0,0}},harmonic_df(2*m-1)}};
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
        const double scale=std::sqrt((m?2.:1.)*harmonic_fac(l-m)/harmonic_fac(l+m));
        h[m]=combine(value,scale,{},0.);
    }
    IsaHarmonicTransform rows;
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
} }
