/* Psi4: Copyright (c) 2007-2026 The Psi4 Developers.
 * SPDX-License-Identifier: LGPL-3.0-only
 * Constructed from Legendre-polynomial identities, not external translation tables.
 */
#include "multipole_transform.h"
#include "psi4/libmints/matrix.h"
#include <algorithm>
#include <cmath>
#include <complex>
#include <map>
#include <stdexcept>
#include <vector>
namespace psi { namespace isapol {
namespace {
using MPower = std::array<int,3>;
using MComplex = std::complex<long double>;
using MPoly = std::map<MPower, MComplex>;
void transform_require(bool ok, const char* message) {
    if (!ok) throw std::invalid_argument(message);
}
long double transform_factorial(int n) {
    long double out=1.; for (int k=2;k<=n;++k) out*=k; return out;
}
MPoly poly_product(const MPoly& a, const MPoly& b) {
    MPoly out;
    for (const auto& x:a) for (const auto& y:b) {
        MPower p; for (int d=0;d<3;++d) p[d]=x.first[d]+y.first[d];
        out[p]+=x.second*y.second;
    }
    return out;
}
MPoly poly_power(const MPoly& a,int n) {
    MPoly out{{{0,0,0},1.L}};
    for (int k=0;k<n;++k) out=poly_product(out,a);
    return out;
}
std::vector<MPoly> regular_polynomials(int rank) {
    // P_l(z/r)*r^l; associated harmonics follow by differentiating P_l,
    // multiplying (x+iy)^m and the real Racah normalization, with no CS phase.
    constexpr long double legendre[5][5] = {
        {1,0,0,0,0}, {0,1,0,0,0}, {-0.5L,0,1.5L,0,0},
        {0,-1.5L,0,2.5L,0}, {0.375L,0,-3.75L,0,4.375L}};
    const MPoly xy{{{1,0,0},1.L},{{0,1,0},MComplex(0.L,1.L)}};
    const MPoly r2{{{2,0,0},1.L},{{0,2,0},1.L},{{0,0,2},1.L}};
    std::vector<MPoly> out;
    for (int l=0;l<=rank;++l) for (int m=0;m<=l;++m) {
        MPoly radial;
        for (int k=m;k<=l;++k) if (legendre[l][k]!=0.L) {
            const auto radial_power=poly_power(r2,(l-k)/2);
            const long double coefficient=legendre[l][k]*transform_factorial(k)/transform_factorial(k-m);
            for (const auto& term:radial_power) {
                auto p=term.first; p[2]+=k-m; radial[p]+=coefficient*term.second;
            }
        }
        auto complex_poly=poly_product(poly_power(xy,m),radial);
        const long double scale=std::sqrt((m?2.L:1.L)*transform_factorial(l-m)/transform_factorial(l+m));
        MPoly real,imag;
        for (const auto& term:complex_poly) {
            if (term.second.real()!=0.L) real[term.first]=scale*term.second.real();
            if (term.second.imag()!=0.L) imag[term.first]=scale*term.second.imag();
        }
        out.push_back(real);
        if (m) out.push_back(imag);
    }
    return out;
}
MPoly substitute(const MPoly& p,const std::array<MPoly,3>& coordinates) {
    MPoly result;
    for (const auto& term:p) {
        MPoly product{{{0,0,0},term.second}};
        for (int d=0;d<3;++d) product=poly_product(product,poly_power(coordinates[d],term.first[d]));
        for (const auto& v:product) result[v.first]+=v.second;
    }
    return result;
}
long double coefficient(const MPoly& p,const MPower& power) {
    auto found=p.find(power); return found==p.end()?0.L:found->second.real();
}
std::vector<long double> small_solve(std::vector<std::vector<long double>> a,std::vector<long double> b) {
    const int n=b.size();
    for (int k=0;k<n;++k) {
        int pivot=k;
        for (int i=k+1;i<n;++i) if (std::abs(a[i][k])>std::abs(a[pivot][k])) pivot=i;
        transform_require(a[pivot][k]!=0.L,"Singular harmonic coefficient basis");
        std::swap(a[k],a[pivot]); std::swap(b[k],b[pivot]);
        for (int i=k+1;i<n;++i) {
            const long double f=a[i][k]/a[k][k];
            for (int j=k;j<n;++j) a[i][j]-=f*a[k][j];
            b[i]-=f*b[k];
        }
    }
    std::vector<long double> x(n);
    for (int i=n-1;i>=0;--i) {
        long double sum=b[i];
        for (int j=i+1;j<n;++j) sum-=a[i][j]*x[j];
        x[i]=sum/a[i][i];
    }
    return x;
}
std::shared_ptr<Matrix> transform_polynomials(int rank,const std::array<MPoly,3>& coordinates) {
    transform_require(rank>=0 && rank<=4,"Multipole transform rank must be in [0,4]");
    const auto harmonics=regular_polynomials(rank);
    const int n=harmonics.size();
    auto out=std::make_shared<Matrix>("Real Racah multipole transformation",n,n);
    for (int t=0;t<n;++t) {
        const auto shifted=substitute(harmonics[t],coordinates);
        for (int l=0;l<=rank;++l) {
            std::vector<MPower> monomials;
            for (int x=0;x<=l;++x) for (int y=0;y<=l-x;++y) monomials.push_back({x,y,l-x-y});
            const int count=2*l+1, offset=l*l;
            std::vector<std::vector<long double>> gram(count,std::vector<long double>(count));
            std::vector<long double> rhs(count);
            for (int i=0;i<count;++i) for (const auto& p:monomials) {
                const long double h=coefficient(harmonics[offset+i],p);
                rhs[i]+=h*coefficient(shifted,p);
                for (int j=0;j<count;++j) gram[i][j]+=h*coefficient(harmonics[offset+j],p);
            }
            const auto row=small_solve(gram,rhs);
            long double scale=1.L, residual=0.L;
            for (const auto& p:monomials) {
                long double reconstructed=0.L;
                for (int i=0;i<count;++i) reconstructed+=row[i]*coefficient(harmonics[offset+i],p);
                scale=std::max(scale,std::abs(coefficient(shifted,p)));
                residual=std::max(residual,std::abs(reconstructed-coefficient(shifted,p)));
            }
            transform_require(std::isfinite(residual) && residual<=1.e-12L*scale,
                              "Harmonic polynomial reconstruction failed");
            for (int i=0;i<count;++i) {
                const double value=static_cast<double>(row[i]);
                transform_require(std::isfinite(value),"Nonfinite multipole transform");
                out->set(t,offset+i,value);
            }
        }
    }
    return out;
}
}
std::shared_ptr<Matrix> isa_multipole_translation(int rank,const std::array<double,3>& displacement) {
    std::array<MPoly,3> coordinates;
    for (int d=0;d<3;++d) {
        transform_require(std::isfinite(displacement[d]),"Multipole displacement must be finite");
        MPower p{0,0,0}; p[d]=1;
        coordinates[d][p]=1.L; coordinates[d][{0,0,0}]=displacement[d];
    }
    return transform_polynomials(rank,coordinates);
}
std::shared_ptr<Matrix> isa_multipole_rotation(int rank,const std::array<std::array<double,3>,3>& frame) {
    for (const auto& row:frame) for (double x:row) transform_require(std::isfinite(x),"Multipole frame must be finite");
    for (int i=0;i<3;++i) for (int j=0;j<3;++j) {
        long double dot=0.L;
        for (int k=0;k<3;++k) dot+=static_cast<long double>(frame[i][k])*frame[j][k];
        transform_require(std::abs(dot-(i==j?1.L:0.L))<=1.e-12L,"Multipole frame must be orthogonal");
    }
    const long double determinant=static_cast<long double>(frame[0][0])*(frame[1][1]*frame[2][2]-frame[1][2]*frame[2][1])-
        static_cast<long double>(frame[0][1])*(frame[1][0]*frame[2][2]-frame[1][2]*frame[2][0])+
        static_cast<long double>(frame[0][2])*(frame[1][0]*frame[2][1]-frame[1][1]*frame[2][0]);
    transform_require(std::abs(determinant-1.L)<=1.e-12L,"Multipole frame must be a proper rotation");
    std::array<MPoly,3> coordinates;
    for (int i=0;i<3;++i) for (int j=0;j<3;++j) {
        MPower p{0,0,0}; p[j]=1; coordinates[i][p]=frame[i][j];
    }
    return transform_polynomials(rank,coordinates);
}
} }
