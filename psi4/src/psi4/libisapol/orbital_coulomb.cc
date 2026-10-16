/* Psi4: Copyright (c) 2007-2026 The Psi4 Developers.
 * SPDX-License-Identifier: LGPL-3.0-only
 */
#include "aux_coulomb.h"
#include "harmonic_transform.h"
#include "psi4/libmints/basisset.h"
#include "psi4/libmints/eri.h"
#include "psi4/libmints/integral.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libqt/qt.h"
#include <libint2/cgshell_ordering.h>
#include <algorithm>
#include <array>
#include <cfenv>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
namespace psi { namespace isapol {
namespace {
void orbital_require(bool ok,const char* message) { if (!ok) throw std::invalid_argument(message); }
double orbital_df(int n) { double v=1.; for (;n>0;n-=2) v*=n; return v; }

// Exact residual r = round_RN(b - A x): an original integer long-accumulator
// ("superaccumulator", cf. Kulisch; Neal arXiv:1505.05571) implementation.
// Every finite binary64 is (-1)^s m 2^e with integer 0 <= m < 2^53 and
// e in [-1074, 971]; each product m_a m_x < 2^106 is formed exactly from four
// 32x32-bit unsigned products and added, with its sign, into signed radix-2^32
// digits whose bit 0 has weight 2^-2148 (the smallest product exponent).
// Rounding to binary64 is done once, by integer logic. No FP arithmetic is used.
constexpr int kDrhocMantissaBits=53;
constexpr int kDrhocMinExponent=-1074;  // weight of a subnormal's last bit
constexpr int kDrhocMaxExponent=971;    // weight of DBL_MAX's last bit
constexpr int kDrhocAccumulatorExponent=2*kDrhocMinExponent;  // -2148
constexpr int kDrhocTermLog2=30;        // fewer than 2^30 terms per row (checked)
constexpr int kDrhocRadixBits=32;
constexpr int kDrhocDigits=134;
static_assert(std::numeric_limits<double>::is_iec559 &&
              std::numeric_limits<double>::radix==2 &&
              std::numeric_limits<double>::digits==kDrhocMantissaBits &&
              std::numeric_limits<double>::min_exponent==-1021 &&
              std::numeric_limits<double>::max_exponent==1024 && sizeof(double)==sizeof(std::uint64_t),
              "exact Drho-C residual requires IEEE-754 binary64");
// The largest product is < 2^(2*971+106) = 2^2048, and a row sums fewer than
// 2^30 terms, so |sum| < 2^2078: bits 0..4225 above 2^-2148. The digits span
// 32*134 = 4288 bits (top weight 2^2139), leaving room for the sign.
constexpr int kDrhocSumBits=2*kDrhocMaxExponent+2*kDrhocMantissaBits+kDrhocTermLog2-kDrhocAccumulatorExponent;
static_assert(kDrhocSumBits==4226, "sum bound");
static_assert(kDrhocAccumulatorExponent+kDrhocRadixBits*kDrhocDigits-1==2139, "top digit weight");
static_assert(kDrhocRadixBits*kDrhocDigits>kDrhocSumBits+kDrhocRadixBits, "sign headroom");
// A product at offset pos <= 2*(971+1074) touches digits pos/32 .. pos/32+4
// (106 product bits plus a shift below 32 fit in 5 digits).
static_assert((2*(kDrhocMaxExponent-kDrhocMinExponent))/kDrhocRadixBits+4<kDrhocDigits, "term digit range");
// Each term changes a digit by less than 2^32; fewer than 2^30 terms keep every
// digit below 2^62, so carries (< 2^31) never overflow int64_t.
static_assert(kDrhocRadixBits+kDrhocTermLog2<=62, "digit magnitude");
using DrhocAccumulator=std::array<std::int64_t,kDrhocDigits>;
struct DrhocParts { std::uint64_t mantissa; int exponent; bool negative; };
DrhocParts drhoc_split(double v) {  // v finite
    std::uint64_t bits;
    std::memcpy(&bits,&v,sizeof bits);
    const int biased=static_cast<int>((bits>>52)&0x7ffu);
    DrhocParts p;
    p.negative=(bits>>63)!=0;
    p.mantissa=bits&((std::uint64_t(1)<<52)-1);
    if (biased==0) {
        p.exponent=kDrhocMinExponent;
    } else {
        p.mantissa|=std::uint64_t(1)<<52;
        p.exponent=biased-1075;
    }
    return p;
}
// acc += (negative ? -1 : 1) * ma * mx * 2^exponent, exactly.
void drhoc_add(DrhocAccumulator& acc,std::uint64_t ma,std::uint64_t mx,int exponent,bool negative) {
    if (ma==0 || mx==0) return;
    const std::uint64_t mask=0xffffffffu;
    const std::uint64_t a0=ma&mask,a1=ma>>32,x0=mx&mask,x1=mx>>32;
    const std::uint64_t p00=a0*x0,p01=a0*x1,p10=a1*x0,p11=a1*x1;  // a1, x1 < 2^21
    const std::uint64_t mid=p01+p10+(p00>>32);                     // < 2^55
    const std::uint64_t top=p11+(mid>>32);                         // < 2^43
    const std::uint64_t limbs[4]={p00&mask,mid&mask,top&mask,top>>32};
    const int offset=exponent-kDrhocAccumulatorExponent;          // >= 0
    const int first=offset/kDrhocRadixBits,shift=offset%kDrhocRadixBits;
    std::uint64_t carry=0;
    for (int k=0;k<5;++k) {
        const std::uint64_t value=k<4 ? limbs[k]<<shift : 0;        // < 2^63
        const std::int64_t digit=static_cast<std::int64_t>((value&mask)|carry);  // < 2^32
        carry=value>>32;
        acc[first+k]+=negative ? -digit : digit;
    }
}
// Carry-normalize so digits 0..n-2 lie in [0, 2^32); the top digit keeps the sign.
void drhoc_normalize(DrhocAccumulator& acc) {
    const std::int64_t radix=std::int64_t(1)<<kDrhocRadixBits;
    for (int k=0;k+1<kDrhocDigits;++k) {
        std::int64_t carry=acc[k]/radix,rest=acc[k]-carry*radix;
        if (rest<0) { rest+=radix; --carry; }
        acc[k]=rest;
        acc[k+1]+=carry;
    }
}
bool drhoc_bit(const DrhocAccumulator& acc,int i) {
    return ((static_cast<std::uint64_t>(acc[i/kDrhocRadixBits])>>(i%kDrhocRadixBits))&1u)!=0;
}
bool drhoc_any_below(const DrhocAccumulator& acc,int i) {  // any set bit at index < i
    if (i<=0) return false;
    const int digit=i/kDrhocRadixBits,bit=i%kDrhocRadixBits;
    for (int k=0;k<digit;++k) if (acc[k]!=0) return true;
    return bit>0 && (static_cast<std::uint64_t>(acc[digit])&((std::uint64_t(1)<<bit)-1))!=0;
}
// Round the exact accumulated value half-to-even; false on binary64 overflow.
bool drhoc_round(DrhocAccumulator& acc,double& out) {
    drhoc_normalize(acc);
    const bool negative=acc[kDrhocDigits-1]<0;
    if (negative) {
        for (auto& digit:acc) digit=-digit;
        drhoc_normalize(acc);
    }
    int top_digit=kDrhocDigits-1;
    while (top_digit>=0 && acc[top_digit]==0) --top_digit;
    if (top_digit<0) { out=0.; return true; }  // exact zero is +0
    int top=top_digit*kDrhocRadixBits+kDrhocRadixBits-1;
    while (!drhoc_bit(acc,top)) --top;
    const int subnormal_bit=kDrhocMinExponent-kDrhocAccumulatorExponent;  // index of 2^-1074
    int low=std::max(top-(kDrhocMantissaBits-1),subnormal_bit);
    std::uint64_t kept=0;
    for (int i=top;i>=low;--i) kept=(kept<<1)|(drhoc_bit(acc,i) ? 1u : 0u);
    const bool round=low>0 && drhoc_bit(acc,low-1);
    const bool sticky=drhoc_any_below(acc,low-1);
    if (round && (sticky || (kept&1u))) {
        ++kept;
        if (kept==(std::uint64_t(1)<<kDrhocMantissaBits)) { kept>>=1; ++low; }
    }
    const int exponent=low+kDrhocAccumulatorExponent;  // weight of kept's last bit
    std::uint64_t bits;
    if (kept>=(std::uint64_t(1)<<52)) {
        const int biased=exponent+1075;
        if (biased>=0x7ff) return false;
        bits=(static_cast<std::uint64_t>(biased)<<52)|(kept-(std::uint64_t(1)<<52));
    } else {
        bits=kept;  // subnormal (exponent == -1074) or rounded to a signed zero
    }
    if (negative) bits|=std::uint64_t(1)<<63;
    std::memcpy(&out,&bits,sizeof out);
    return true;
}
// r = round_RN(b - A x) for finite inputs; false if some row overflows.
bool drhoc_residual(const double* a,const double* x,const double* b,std::size_t n,double* r) {
    DrhocAccumulator acc;
    for (std::size_t i=0;i<n;++i) {
        acc.fill(0);
        const DrhocParts bi=drhoc_split(b[i]);
        drhoc_add(acc,bi.mantissa,1,bi.exponent,bi.negative);
        for (std::size_t j=0;j<n;++j) {
            const DrhocParts aij=drhoc_split(a[i*n+j]),xj=drhoc_split(x[j]);
            drhoc_add(acc,aij.mantissa,xj.mantissa,aij.exponent+xj.exponent,aij.negative==xj.negative);
        }
        if (!drhoc_round(acc,r[i])) return false;
    }
    return true;
}
void drhoc_require_size(std::size_t n) {
    const std::size_t limit=std::min<std::size_t>(INT_MAX,(std::size_t(1)<<kDrhocTermLog2)-2);
    orbital_require(n>=1 && n<=limit && n<=std::numeric_limits<std::size_t>::max()/sizeof(double)/n,
                    "Drho-C solve dimension must be in [1, 2^30-2]");
}
void drhoc_require_finite(const double* v,std::size_t count,const char* message) {
    for (std::size_t i=0;i<count;++i) orbital_require(std::isfinite(v[i]),message);
}
// Round-to-nearest with gradual underflow on the calling thread (FTZ/DAZ off).
bool drhoc_fp_environment_ok() {
    if (std::fegetround()!=FE_TONEAREST) return false;
    volatile double smallest=std::numeric_limits<double>::min();
    volatile double half=smallest/2.;
    volatile double restored=half*2.;
    return half!=0. && restored==smallest;
}
}
std::vector<double> isa_exact_residual(const double* a,const double* x,const double* b,std::size_t n) {
    drhoc_require_size(n);
    drhoc_require_finite(a,n*n,"Exact residual requires finite A");
    drhoc_require_finite(x,n,"Exact residual requires finite x");
    drhoc_require_finite(b,n,"Exact residual requires finite b");
    std::vector<double> r(n);
    orbital_require(drhoc_residual(a,x,b,n,r.data()),"Exact residual overflow");
    return r;
}
IsaRefinedSolve isa_refined_lu_solve(const double* a,const double* b,std::size_t n,int max_iterations) {
    drhoc_require_size(n);
    orbital_require(max_iterations>=0 && max_iterations<=32,"Drho-C refinement iterations must be in [0, 32]");
    drhoc_require_finite(a,n*n,"Nonfinite constrained metric");
    drhoc_require_finite(b,n,"Nonfinite constrained RHS");
    orbital_require(max_iterations==0 || drhoc_fp_environment_ok(),
                    "Drho-C refinement requires round-to-nearest with gradual underflow");
    const int ni=static_cast<int>(n);
    // Pack column-major explicitly; constrained A has roundoff-level asymmetry.
    std::vector<double> lu(n*n);
    std::vector<int> pivots(n);
    for (std::size_t i=0;i<n;++i) for (std::size_t j=0;j<n;++j) lu[i+j*n]=a[i*n+j];
    IsaRefinedSolve solve;
    solve.x.assign(b,b+n);
    const int info=C_DGESV(ni,1,lu.data(),ni,pivots.data(),solve.x.data(),ni);
    orbital_require(info==0,"Native Drho-C LU solve failed");
    if (max_iterations==0) return solve;
    drhoc_require_finite(solve.x.data(),n,"Nonfinite Drho-C coefficient");
    const std::vector<double> plain(solve.x);
    std::vector<double> correction(n);
    const double unit=std::ldexp(1.,-53);
    double previous=0.;
    bool converged=false;
    for (int it=1;it<=max_iterations && !converged;++it) {
        orbital_require(drhoc_residual(a,solve.x.data(),b,n,correction.data()),
                        "Drho-C refinement residual overflow");
        const int solved=C_DGETRS('N',ni,1,lu.data(),ni,pivots.data(),correction.data(),ni);
        orbital_require(solved==0,"Drho-C refinement correction solve failed");
        double step=0.,size=0.;
        for (std::size_t i=0;i<n;++i) {
            orbital_require(std::isfinite(correction[i]),"Nonfinite Drho-C refinement correction");
            solve.x[i]+=correction[i];
            orbital_require(std::isfinite(solve.x[i]),"Drho-C refinement update overflow");
            step=std::max(step,std::abs(correction[i]));
            size=std::max(size,std::abs(solve.x[i]));
        }
        solve.iterations=it;
        converged=step<=2.*unit*size;
        orbital_require(converged || it<2 || step<=.5*previous,"Drho-C refinement stagnated");
        previous=step;
    }
    orbital_require(converged,"Drho-C refinement did not converge within the iteration cap");
    double moved=0.,size=0.;
    for (std::size_t i=0;i<n;++i) {
        moved=std::max(moved,std::abs(solve.x[i]-plain[i]));
        size=std::max(size,std::abs(solve.x[i]));
    }
    orbital_require(size>0. || moved==0.,"Drho-C refinement reached zero from a nonzero LU solution");
    solve.displacement=size>0. ? moved/size : 0.;
    orbital_require(std::isfinite(solve.displacement),"Nonfinite Drho-C refinement displacement");
    return solve;
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
    orbital_require(max_bytes>0 &&
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
    std::vector<int> ao,mo;
    std::vector<IsaHarmonicTransform> transforms;
    // A spherical molecular AUX is a DIFFERENT declared basis from the Cartesian
    // GAMINT one built from the same exponents, never a normalization variant of
    // it: it spans 2l+1 rather than (l+1)(l+2)/2 functions per shell.
    const bool pure=basis_.representation_==IsaBasisRepresentation::Spherical;
    std::vector<IsaHarmonicTransform> aux_transforms;
    int offset=0;
    for (const auto& s:basis_.shells_) {
        ao.push_back(offset); offset+=IsaExplicitBasis::shell_size(s.l,basis_.representation_);
        aux_transforms.push_back(pure ? isa_dalton_transform(s.l) : IsaHarmonicTransform{});
    }
    offset=0;
    for (const auto& s:orbital.shells_) {
        mo.push_back(offset); offset+=2*s.l+1;
        transforms.push_back(isa_dalton_transform(s.l));
    }
    // Both bases enter as raw Cartesian native twins; the DALTON MAIN transforms
    // act on raw monomials below. Zero-precision compute_shell, as in metric().
    const auto aux=native_basis(basis_),main=native_basis(orbital);
    IntegralFactory factory(aux.basis,BasisSet::zero_ao_basis_set(),main.basis,main.basis);
    Libint2ERI eri(&factory,0.,0,false,false,false);
    auto result=std::make_shared<Matrix>("Native AUX MAIN MAIN Coulomb",basis_.nfunction(),n*n);
    for (size_t a=0;a<basis_.shells_.size();++a) for (size_t b=0;b<orbital.shells_.size();++b) for (size_t c=0;c<=b;++c) {
        if (!eri.compute_shell(aux.shells[a],0,main.shells[b],main.shells[c])) continue;
        const double* block=eri.buffer();
        const int la=basis_.shells_[a].l;
        const auto& powers=IsaExplicitBasis::cartesian_powers(la);
        const int nb=(orbital.shells_[b].l+1)*(orbital.shells_[b].l+2)/2;
        const int nc=(orbital.shells_[c].l+1)*(orbital.shells_[c].l+2)/2;
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
IsaDrhoCResult IsaAuxCoulomb::fit_drho_c(const IsaExplicitBasis& orbital,const Matrix& occupied,double penalty,
                                         int max_refinement_iterations) const {
    orbital_require(std::isfinite(penalty) && penalty>0.,"Drho-C charge penalty must be finite and positive");
    orbital_require(max_refinement_iterations>=0 && max_refinement_iterations<=32,
                    "Drho-C refinement iterations must be in [0, 32]");
    orbital_require(max_refinement_iterations==0 || drhoc_fp_environment_ok(),
                    "Drho-C refinement requires round-to-nearest with gradual underflow");
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
    // Plain DGESV for K = 0 (the default); otherwise refined on the same factors.
    // The metric's storage is one contiguous row-major n x n block.
    auto solve=isa_refined_lu_solve(result.metric->get_const_pointer(),result.rhs.data(),
                                    static_cast<std::size_t>(n),max_refinement_iterations);
    result.coefficients=std::move(solve.x);
    result.refinement_iterations=solve.iterations;
    result.refinement_displacement=solve.displacement;
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
