/* Psi4: Copyright (c) 2007-2026 The Psi4 Developers.
 * SPDX-License-Identifier: LGPL-3.0-only
 */
#include "ov_fit.h"
#include "aux_coulomb.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libqt/qt.h"
#include <cmath>
#include <limits>
#include <new>
#include <stdexcept>
namespace psi { namespace isapol {
namespace {
void ov_require(bool ok, const char* message) {
    if (!ok) throw std::invalid_argument(message);
}
void ov_finite(const Matrix& m) {
    for (int i=0; i<m.nrow(); ++i) for (int j=0; j<m.ncol(); ++j)
        ov_require(std::isfinite(m.get(i,j)), "OV fit: nonfinite matrix input/intermediate/result");
}
// Bound every Matrix product to INT_MAX as well as size_t. Conservative dense
// workspace admission cap (1 GiB), NOT a conditioning threshold. Includes output
// getter copies and LU buffers, but not Libint's internal engine allocations.
size_t ov_product(size_t a, size_t b) {
    ov_require(b && a <= static_cast<size_t>(std::numeric_limits<int>::max())/b,
               "OV fit: dimension/product overflow");
    return a*b;
}
}
std::shared_ptr<Matrix> IsaOvFitResult::coulomb_metric() const { return j_->clone(); }
std::shared_ptr<Matrix> IsaOvFitResult::metric() const { return a_->clone(); }
std::shared_ptr<Matrix> IsaOvFitResult::rhs() const { return t_->clone(); }
std::shared_ptr<Matrix> IsaOvFitResult::coefficients() const { return d_->clone(); }
IsaOvFitResult IsaAuxCoulomb::fit_ov(const IsaExplicitBasis& orbital, const Matrix& occupied,
        const Matrix& virtuals, const std::string& provenance, double penalty) const {
    ov_require(basis_.role_==IsaBasisRole::MolecularAux &&
               basis_.representation_==IsaBasisRepresentation::Cartesian,
               "OV fit: requires Cartesian molecular AUX");
    ov_require(orbital.role_==IsaBasisRole::Orbital &&
               orbital.representation_==IsaBasisRepresentation::Spherical,
               "OV fit: MAIN requires DALTON spherical Orbital basis");
    ov_require(!provenance.empty() && provenance.find_first_not_of(" \t\r\n")!=std::string::npos,
               "OV fit: explicit supplied-orbital provenance required");
    ov_require(std::isfinite(penalty) && penalty>=0., "OV fit: penalty must be finite and nonnegative");
    ov_require(occupied.nirrep()==1 && virtuals.nirrep()==1,
               "OV fit: coefficients require a single symmetry block");
    const int n=orbital.nfunction(), m=basis_.nfunction(), o=occupied.ncol(), v=virtuals.ncol();
    ov_require(n>0 && m>0 && occupied.nrow()==n && virtuals.nrow()==n &&
               o>0 && v>0 && o<=n && v<=n && o<=n-v,
               "OV fit: invalid MAIN/occupied/virtual coefficient dimensions");
    const size_t nov=ov_product(o,v), nn=ov_product(n,n), mm=ov_product(m,m);
    const size_t mn2=ov_product(m,nn), tm=ov_product(nov,m), on=ov_product(o,n);
    size_t bytes=0;
    auto reserve=[&](size_t count, size_t width) {
        constexpr size_t cap=size_t(1)<<30;
        ov_require(count <= (cap-bytes)/width, "OV fit: dense workspace resource limit (1 GiB)");
        bytes+=count*width;
    };
    reserve(mn2,sizeof(double)); reserve(nn,sizeof(double)); reserve(on,sizeof(double));
    reserve(nov,sizeof(double));
    for (int i=0;i<5;++i) reserve(mm,sizeof(double)); // J,A,LU plus clones
    for (int i=0;i<5;++i) reserve(tm,sizeof(double)); // T,D,RHS plus clones
    reserve(m,sizeof(double)+sizeof(int));
    ov_finite(occupied); ov_finite(virtuals);
    try {
        IsaOvFitResult result;
        result.nmain_=n; result.naux_=m; result.noccupied_=o; result.nvirtual_=v;
        result.penalty_=penalty; result.provenance_=provenance;
        result.charges_=charges(); result.j_=metric();
        const auto b=three_center(orbital);
        ov_finite(*result.j_); ov_finite(*b);
        result.a_=std::make_shared<Matrix>("OV A = J + (lambda q) q^T",m,m);
        result.t_=std::make_shared<Matrix>("OV T occupied-fast",static_cast<int>(nov),m);
        result.d_=std::make_shared<Matrix>("OV fitted density coefficients",static_cast<int>(nov),m);
        Matrix block("OV B[k]",n,n), left("OV Cocc^T B[k]",o,n), pair("OV L[k] Cvir",o,v);
        for (int k=0;k<m;++k) {
            const double pq=penalty*result.charges_[k];
            ov_require(std::isfinite(pq), "OV fit: nonfinite penalty charge product");
            for (int l=0;l<m;++l) result.a_->set(k,l,result.j_->get(k,l)+pq*result.charges_[l]);
            for (int mu=0;mu<n;++mu) for (int nu=0;nu<n;++nu) block.set(mu,nu,b->get(k,mu*n+nu));
            // Two explicit GEMMs. Never reassociate as B[k] @ Cvir first.
            left.gemm(true,false,1.,occupied,block,0.);
            ov_finite(left);
            pair.gemm(false,false,1.,left,virtuals,0.);
            ov_finite(pair);
            for (int r=0;r<v;++r) for (int a=0;a<o;++a) result.t_->set(a+o*r,k,pair.get(a,r));
        }
        ov_finite(*result.a_);
        std::vector<double> lu(mm), rhs(tm);
        std::vector<int> pivots(m);
        for (int k=0;k<m;++k) for (int l=0;l<m;++l) lu[k+size_t(l)*m]=result.a_->get(k,l);
        for (size_t p=0;p<nov;++p) for (int k=0;k<m;++k) rhs[k+p*m]=result.t_->get(p,k);
        result.info_=C_DGESV(m,static_cast<int>(nov),lu.data(),m,pivots.data(),rhs.data(),m);
        if (result.info_!=0) throw std::runtime_error("OV fit: general LU C_DGESV failed, INFO="+
            std::to_string(result.info_)+(result.info_>0 ? " (singular pivot; no fallback)" : " (invalid LAPACK argument)"));
        for (double x:lu) ov_require(std::isfinite(x), "OV fit: nonfinite LU factor");
        for (size_t p=0;p<nov;++p) for (int k=0;k<m;++k) result.d_->set(p,k,rhs[k+p*m]);
        ov_finite(*result.d_);
        long double an=0.,dn=0.,tn=0.,rn=0.;
        for (int k=0;k<m;++k) for (int l=0;l<m;++l) an=std::hypot(an,static_cast<long double>(result.a_->get(k,l)));
        for (size_t p=0;p<nov;++p) for (int k=0;k<m;++k) {
            dn=std::hypot(dn,static_cast<long double>(result.d_->get(p,k)));
            tn=std::hypot(tn,static_cast<long double>(result.t_->get(p,k)));
            long double ad=0.;
            for (int l=0;l<m;++l) ad+=static_cast<long double>(result.a_->get(k,l))*result.d_->get(p,l);
            rn=std::hypot(rn,ad-result.t_->get(p,k));
        }
        const long double denom=an*dn+tn;
        ov_require(std::isfinite(denom) && std::isfinite(rn), "OV fit: nonfinite backward diagnostic");
        result.residual_=denom>0. ? static_cast<double>(rn/denom) : 0.;
        return result;
    } catch (const std::bad_alloc&) {
        throw std::runtime_error("OV fit: workspace allocation failed");
    } catch (const std::length_error&) {
        throw std::runtime_error("OV fit: workspace allocation length exceeded");
    }
}
} }
