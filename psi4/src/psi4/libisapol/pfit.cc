/* Psi4: Copyright (c) 2007-2026 The Psi4 Developers.
 * SPDX-License-Identifier: LGPL-3.0-only
 */
#include "pfit.h"
#include "psi4/libqt/qt.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>

namespace psi { namespace isapol {
namespace {
void require(bool ok, const std::string& what) {
    if (!ok) throw std::invalid_argument("PFIT: " + what);
}
double finite(double x) {
    require(std::isfinite(x), "nonfinite input or numerical intermediate (overflow)");
    return x;
}
size_t add(size_t a, size_t b) {
    require(b <= std::numeric_limits<size_t>::max() - a, "size addition overflow"); return a+b;
}
size_t mul(size_t a, size_t b) {
    require(!a || b <= std::numeric_limits<size_t>::max()/a, "size multiplication overflow"); return a*b;
}
void values(const std::vector<double>& v) { for (double x : v) finite(x); }
void vector_shape(const std::vector<double>& v, size_t n) {
    require(v.size()==n, "vector dimension mismatch"); values(v);
}
void matrix_shape(const IsaPfitMatrix& a, size_t n, size_t m) {
    require(a.rows==n && a.cols==m && a.values.size()==mul(n,m), "matrix dimension mismatch"); values(a.values);
}
void symmetric(const IsaPfitMatrix& a) {
    for (size_t i=0;i<a.rows;++i) for (size_t j=0;j<i;++j)
        require(a.values[i*a.cols+j]==a.values[j*a.cols+i], "matrix must be exactly symmetric (strict policy)");
}
void labels(const std::vector<std::string>& s) {
    std::set<std::string> seen;
    for (const auto& x:s) require(x.find_first_not_of(" \t\n\r")!=std::string::npos && seen.insert(x).second,
                               "labels must be nonblank and unique");
}
void text(const std::string& s) { require(s.find_first_not_of(" \t\n\r")!=std::string::npos, "missing provenance/units"); }
IsaPfitMatrix zeros(size_t n) { return {n,n,std::vector<double>(mul(n,n),0.)}; }
// Every LAPACK buffer is explicitly column-major. Symmetric row-major inputs
// have the same bytes, but conversion here also handles nonsymmetric R.
std::vector<double> column(const IsaPfitMatrix& a) {
    std::vector<double> c(a.values.size());
    for (size_t i=0;i<a.rows;++i) for (size_t j=0;j<a.cols;++j) c[i+j*a.rows]=a.values[i*a.cols+j];
    return c;
}
struct Budget {
    size_t base, limit;
    size_t* peak = nullptr;
    int workspace(double query) const {
        finite(query); require(query>=1 && query<=std::numeric_limits<int>::max(), "invalid LAPACK workspace query");
        size_t n=static_cast<size_t>(std::ceil(query));
        size_t bytes=add(base,mul(n,sizeof(double)));
        require(bytes<=limit, "LAPACK workspace exceeds maximum_work_bytes");
        if(peak) *peak=std::max(*peak,bytes);
        return static_cast<int>(n);
    }
};
void info_ok(int info) { require(info==0, "LAPACK diagnostic/workspace failure: info="+std::to_string(info)); }
std::vector<double> eigen(std::vector<double>& a, int n, const Budget& budget) {
    std::vector<double> w(n); double q=0;
    info_ok(C_DSYEV('V','L',n,a.data(),n,w.data(),&q,-1));
    int l=budget.workspace(q); std::vector<double> work(l);
    info_ok(C_DSYEV('V','L',n,a.data(),n,w.data(),work.data(),l)); values(w); values(a); return w;
}
double dot(const std::vector<double>& a, const std::vector<double>& b) {
    double s=0; for(size_t i=0;i<a.size();++i) s=finite(s+finite(a[i]*b[i])); return s;
}
// One canonical physical pair generator. No row weights, cross-batch pairs or pruning.
template<class F> void data_rows(const IsaPfitProblem& p, F consume) {
    size_t nc=p.model.channel_labels.size(), np=p.model.parameter_labels.size();
    std::vector<double> row(np);
    for(size_t b=0;b<p.batches.size();++b) {
        const auto& batch=p.batches[b]; size_t packed=0;
        for(size_t i=0;i<batch.points_bohr.size();++i) for(size_t j=0;j<=i;++j) {
            std::fill(row.begin(),row.end(),0.);
            for(size_t k=0;k<np;++k) for(size_t u=0;u<nc;++u) for(size_t v=0;v<nc;++v)
                row[k]=finite(row[k]+finite(finite(batch.fields.values[i*nc+u]*p.model.parameter_tensors[k].values[u*nc+v])*
                                               batch.fields.values[j*nc+v]));
            consume(b,packed,row,batch.targets[packed]); ++packed;
        }
    }
}
// Stable row-insertion Givens QR. Chunking only buffers computational rows;
// it never defines physical clouds. R has nonnegative diagonal. No normal-H
// arithmetic is used to solve the QR system.
struct Givens {
    size_t n, chunk, used=0;
    std::vector<double> r,z,buffer;
    double discarded=0;
    Givens(size_t n_,size_t chunk_):n(n_),chunk(chunk_),r(n*n),z(n),buffer(chunk*(n+1)) {}
    void flush() {
        for(size_t i=0;i<used;++i) {
            double* a=buffer.data()+i*(n+1); double y=a[n];
            for(size_t j=0;j<n;++j) {
                double h=finite(std::hypot(r[j*n+j],a[j])); if(h==0) continue;
                double c=r[j*n+j]/h, s=a[j]/h; r[j*n+j]=h;
                for(size_t k=j+1;k<n;++k) {
                    double old=r[j*n+k]; r[j*n+k]=finite(c*old+s*a[k]); a[k]=finite(-s*old+c*a[k]);
                }
                double old=z[j]; z[j]=finite(c*old+s*y); y=finite(-s*old+c*y);
            }
            discarded=finite(discarded+finite(y*y));
        }
        used=0;
    }
    void push(const std::vector<double>& a,double y) {
        std::copy(a.begin(),a.end(),buffer.begin()+used*(n+1)); buffer[used*(n+1)+n]=y;
        if(++used==chunk) flush();
    }
};
}
std::vector<double> IsaPfitResult::parameters() const {
    if(status_!=IsaPfitStatus::Solved && status_!=IsaPfitStatus::AllFixed)
        throw std::runtime_error("PFIT: parameters unavailable for failed fit");
    return parameters_;
}
IsaPfitMatrix IsaPfitResult::effective_penalty_matrix() const {
    auto v=penalty_; for(size_t i=0;i<v.values.size();++i) v.values[i]=finite(v.values[i]+lc_.values[i]); return v;
}
std::vector<double> IsaPfitResult::effective_penalty_rhs() const {
    auto v=matrix_rhs_; for(size_t i=0;i<v.size();++i) v[i]=finite(v[i]+lc_rhs_[i]); return v;
}
IsaPfitResult isa_pfit_solve(const IsaPfitProblem& p,const IsaPfitOptions& o) {
    require(o.solver==IsaPfitSolver::StreamingQR || o.solver==IsaPfitSolver::NormalEquationsDSYSV,"unknown solver");
    require(o.qr_chunk_rows>0 && o.qr_chunk_rows<=static_cast<size_t>(std::numeric_limits<int>::max()),"invalid QR chunk rows");
    require(finite(o.rank_relative_tolerance)>0 && o.rank_relative_tolerance<1,"invalid rank tolerance");
    require(finite(o.minimum_solver_rcond)>=0 && o.minimum_solver_rcond<=1,"invalid rcond threshold");
    require(finite(p.frequency_au)>=0,"frequency must be nonnegative");
    const auto& provenance=p.target_provenance;
    require(provenance.origin==IsaPfitTargetOrigin::SuppliedActualPointResponse ||
            provenance.origin==IsaPfitTargetOrigin::SuppliedFittedPropagatorPointResponse ||
            provenance.origin==IsaPfitTargetOrigin::NativeDirectActualPointResponse ||
            provenance.origin==IsaPfitTargetOrigin::SyntheticAnalyticTest,"target origin must be declared");
    require(provenance.convention==IsaPfitTargetConvention::NegativeInducedPotentialPerUnitSourceChargeAtomicUnits,"wrong/unspecified target convention");
    text(provenance.source_id); text(provenance.generation_record); text(p.model.provenance);
    if(provenance.origin==IsaPfitTargetOrigin::SuppliedFittedPropagatorPointResponse) {
        require(provenance.response_representation=="fitted_density_coefficients","fitted target requires fitted_density_coefficients representation");
        text(provenance.auxiliary_basis_id);
    }
    // Native direct-OV point response carries no auxiliary fit, so an auxiliary
    // basis identifier would be a false provenance claim rather than metadata.
    if(provenance.origin==IsaPfitTargetOrigin::NativeDirectActualPointResponse) {
        require(provenance.response_representation=="native_point_charge_ov_operators","native direct point target requires native_point_charge_ov_operators representation");
        require(provenance.auxiliary_basis_id.empty(),"native direct point target must not declare an auxiliary basis");
    }
    size_t np=p.model.parameter_labels.size(), nc=p.model.channel_labels.size();
    require(np>0 && nc>0 && np<=static_cast<size_t>(std::numeric_limits<int>::max()/8),"invalid parameter/channel count");
    labels(p.model.parameter_labels); labels(p.model.channel_labels);
    require(p.model.parameter_units.size()==np,"parameter units dimension mismatch"); for(const auto& s:p.model.parameter_units) text(s);
    require(p.model.parameter_tensors.size()==np && p.model.fixed.size()==np,"model dimension mismatch");
    vector_shape(p.model.fixed_values,np); vector_shape(p.penalty.anchor,np);
    matrix_shape(p.penalty.matrix,np,np); symmetric(p.penalty.matrix);
    for(const auto& k:p.model.parameter_tensors) { matrix_shape(k,nc,nc); symmetric(k); }
    for(const auto& l:p.linear_penalties) {
        vector_shape(l.coefficients,np); finite(l.target); require(finite(l.strength)>=0,"negative LC strength");
    }
    size_t rows=0; std::set<std::string> batch_names;
    for(const auto& b:p.batches) {
        text(b.label); require(batch_names.insert(b.label).second,"duplicate batch label");
        size_t n=b.points_bohr.size(); require(n>0,"empty physical batch");
        size_t pairs=mul(n,add(n,1))/2;
        require(b.targets.size()==pairs,"wrong packed triangular target count"); values(b.targets);
        matrix_shape(b.fields,n,nc); for(const auto& xyz:b.points_bohr) for(double x:xyz) finite(x);
        rows=add(rows,pairs);
    }
    require(rows>0,"empty data not supported");
    size_t nf=0; for(bool fixed:p.model.fixed) if(!fixed) ++nf;
    // Conservative numeric payload budget: all persistent result buffers, small
    // LAPACK copies/factors/diagnostics, integer arrays (charged as doubles), and
    // QR buffer. No input numerical copy is made; input is consumed synchronously.
    // Additional LAPACK queried workspace must fit on top of this base. Not RSS.
    size_t scalars=add(mul(40,mul(np,np)),mul(64,np));
    if(o.solver==IsaPfitSolver::StreamingQR && nf) scalars=add(scalars,mul(o.qr_chunk_rows,add(nf,1)));
    if(o.retain_pair_predictions) scalars=add(scalars,rows);
    scalars=add(scalars,mul(16,p.batches.size()));
    Budget budget{mul(scalars,sizeof(double)),o.maximum_work_bytes};
    require(budget.base<=budget.limit,"kernel numerical buffers exceed maximum_work_bytes");
    IsaPfitResult out; auto& d=out.diagnostics_; out.settings_=o; out.frequency_=p.frequency_au;
    out.provenance_=provenance; out.model_provenance_=p.model.provenance;
    out.parameter_labels_=p.model.parameter_labels; out.parameter_units_=p.model.parameter_units;
    out.channel_labels_=p.model.channel_labels;
    for(const auto& b:p.batches) out.batch_labels_.push_back(b.label);
    d.data_rows=rows; d.augmented_rows=add(add(rows,np),p.linear_penalties.size());
    d.work_budget_bytes=budget.base; budget.peak=&d.work_budget_bytes;
    for(size_t i=0;i<np;++i) if(!p.model.fixed[i]) d.free_indices.push_back(i);
    for(const auto& b:p.batches) d.batches.push_back({b.points_bohr.size(),b.targets.size(),0,0,0});
    out.normal_=zeros(nf); out.rhs_.assign(nf,0); out.penalty_=zeros(np); out.lc_=zeros(np);
    out.matrix_rhs_.assign(np,0); out.lc_rhs_.assign(np,0);
    auto eigvec=column(p.penalty.matrix); auto eigval=eigen(eigvec,static_cast<int>(np),budget);
    d.penalty_min_eigenvalue=eigval.front(); require(eigval.front()>=0,"penalty is not PSD under strict eigenvalue policy");
    // Square-root rows preserve every positive mode, however small. Both paths
    // use exactly these rows, and the reported effective matrix is their Gram.
    std::vector<double> root(mul(np,np));
    for(size_t k=0;k<np;++k) for(size_t j=0;j<np;++j) root[k*np+j]=finite(std::sqrt(eigval[k])*eigvec[j+k*np]);
    for(size_t i=0;i<np;++i) for(size_t j=0;j<np;++j) {
        double x=0; for(size_t k=0;k<np;++k) x=finite(x+finite(root[k*np+i]*root[k*np+j]));
        out.penalty_.values[i*np+j]=x;
        d.penalty_correction_max=std::max(d.penalty_correction_max,std::abs(finite(x-p.penalty.matrix.values[i*np+j])));
    }
    for(size_t i=0;i<np;++i) for(size_t j=0;j<np;++j)
        out.matrix_rhs_[i]=finite(out.matrix_rhs_[i]+finite(out.penalty_.values[i*np+j]*p.penalty.anchor[j]));
    std::vector<double> a(np);
    // Use identical scaled LC rows for assembly, effective diagnostics and
    // residuals. Multiplying strength first can underflow and even produce an
    // asymmetric reported Gram matrix for otherwise finite scaled rows.
    auto lc_row=[&](const IsaPfitLinearPenalty& l) {
        double s=std::sqrt(l.strength);
        for(size_t k=0;k<np;++k) a[k]=finite(s*l.coefficients[k]);
        return finite(s*l.target);
    };
    for(const auto& l:p.linear_penalties) {
        double y=lc_row(l);
        for(size_t i=0;i<np;++i) {
            out.lc_rhs_[i]=finite(out.lc_rhs_[i]+finite(a[i]*y));
            for(size_t j=0;j<np;++j)
                out.lc_.values[i*np+j]=finite(out.lc_.values[i*np+j]+finite(a[i]*a[j]));
        }
    }
    Givens qr(o.solver==IsaPfitSolver::StreamingQR?nf:0, o.solver==IsaPfitSolver::StreamingQR && nf?o.qr_chunk_rows:1);
    std::vector<double> free_row(nf);
    auto append=[&](const std::vector<double>& a,double target) {
        double y=target;
        for(size_t k=0;k<np;++k) if(p.model.fixed[k]) y=finite(y-finite(a[k]*p.model.fixed_values[k]));
        for(size_t i=0;i<nf;++i) free_row[i]=a[d.free_indices[i]];
        for(size_t i=0;i<nf;++i) {
            out.rhs_[i]=finite(out.rhs_[i]+finite(free_row[i]*y));
            for(size_t j=0;j<nf;++j) out.normal_.values[i*nf+j]=finite(out.normal_.values[i*nf+j]+finite(free_row[i]*free_row[j]));
        }
        if(nf && o.solver==IsaPfitSolver::StreamingQR) qr.push(free_row,y);
    };
    data_rows(p,[&](size_t,size_t,const std::vector<double>& a,double y){append(a,y);});
    for(size_t k=0;k<np;++k) {
        std::copy(root.begin()+k*np,root.begin()+(k+1)*np,a.begin()); append(a,dot(a,p.penalty.anchor));
    }
    for(const auto& l:p.linear_penalties) {
        double y=lc_row(l); append(a,y);
    }
    std::vector<double> x;
    if(!nf) out.status_=IsaPfitStatus::AllFixed;
    else {
        int n=static_cast<int>(nf); double rcond=0;
        std::vector<double> spectrum, factor;
        if(o.solver==IsaPfitSolver::StreamingQR) {
            qr.flush(); d.qr_discarded_rhs_sse=qr.discarded;
            factor=column({nf,nf,qr.r}); auto copy=factor; spectrum.resize(nf);
            std::vector<int> iw(8*nf); double dummy=0, query=0;
            info_ok(C_DGESDD('N',n,n,copy.data(),n,spectrum.data(),&dummy,1,&dummy,1,&query,-1,iw.data()));
            int l=budget.workspace(query); std::vector<double> work(l);
            info_ok(C_DGESDD('N',n,n,copy.data(),n,spectrum.data(),&dummy,1,&dummy,1,work.data(),l,iw.data()));
            values(spectrum); d.rank_method="singular values of streaming Givens R (diagnostic only)";
            d.rank_largest=spectrum.front(); d.rank_smallest=spectrum.back();
            std::vector<double> cw(3*nf); std::vector<int> ci(nf);
            info_ok(C_DTRCON('1','U','N',n,factor.data(),n,&rcond,cw.data(),ci.data()));
            d.qr_r_rcond=finite(rcond); d.condition_estimate_available=true;
        } else {
            auto copy=column(out.normal_); spectrum=eigen(copy,n,budget);
            d.rank_method="normal H eigenvalues (squared design conditioning)";
            d.rank_largest=spectrum.back(); d.rank_smallest=spectrum.front();
        }
        for(double s:spectrum) if(s>o.rank_relative_tolerance*d.rank_largest) ++d.numerical_rank;
        if(d.numerical_rank<nf) { out.status_=IsaPfitStatus::RankDeficient; return out; }
        if(o.solver==IsaPfitSolver::NormalEquationsDSYSV) {
            factor=column(out.normal_); x=out.rhs_; std::vector<int> piv(nf); double query=0;
            info_ok(C_DSYSV('L',n,1,factor.data(),n,piv.data(),x.data(),n,&query,-1));
            int l=budget.workspace(query); std::vector<double> work(l);
            d.lapack_info=C_DSYSV('L',n,1,factor.data(),n,piv.data(),x.data(),n,work.data(),l);
            require(d.lapack_info>=0,"DSYSV illegal argument");
            if(d.lapack_info>0) {out.status_=IsaPfitStatus::RankDeficient; return out;}
            double norm=0;
            for(size_t j=0;j<nf;++j) { double sum=0; for(size_t i=0;i<nf;++i) sum=finite(sum+std::abs(out.normal_.values[i*nf+j])); norm=std::max(norm,sum); }
            d.normal_h_norm1=norm;
            std::vector<double> cw(2*nf); std::vector<int> ci(nf);
            info_ok(C_DSYCON('L',n,factor.data(),n,piv.data(),norm,&rcond,cw.data(),ci.data()));
            d.normal_h_rcond=finite(rcond); d.condition_estimate_available=true;
        }
        if(rcond<o.minimum_solver_rcond || rcond<=0) {out.status_=IsaPfitStatus::IllConditioned; return out;}
        if(o.solver==IsaPfitSolver::StreamingQR) {
            x=qr.z; d.lapack_info=C_DTRTRS('U','N','N',n,1,factor.data(),n,x.data(),n);
            require(d.lapack_info>=0,"DTRTRS illegal argument");
            if(d.lapack_info>0) {out.status_=IsaPfitStatus::RankDeficient; return out;}
        }
        values(x); out.status_=IsaPfitStatus::Solved;
        double hn=0,xn=0,bn=0;
        for(size_t i=0;i<nf;++i) {
            double residual=-out.rhs_[i], sum=0;
            for(size_t j=0;j<nf;++j) { residual=finite(residual+finite(out.normal_.values[i*nf+j]*x[j])); sum=finite(sum+std::abs(out.normal_.values[i*nf+j])); }
            hn=std::max(hn,sum); xn=std::max(xn,std::abs(x[i])); bn=std::max(bn,std::abs(out.rhs_[i]));
            d.stationarity_inf=std::max(d.stationarity_inf,std::abs(residual));
        }
        double denom=finite(finite(hn*xn)+bn); d.backward_residual=denom?d.stationarity_inf/denom:0;
    }
    out.parameters_=p.model.fixed_values;
    for(size_t i=0;i<nf;++i) out.parameters_[d.free_indices[i]]=x[i];
    if(o.retain_pair_predictions) {out.predictions_.resize(p.batches.size()); for(size_t b=0;b<p.batches.size();++b) out.predictions_[b].resize(p.batches[b].targets.size());}
    data_rows(p,[&](size_t b,size_t k,const std::vector<double>& row,double target) {
        double pred=dot(row,out.parameters_), residual=finite(pred-target);
        auto& bd=d.batches[b]; bd.sse=finite(bd.sse+finite(residual*residual)); bd.max_residual=std::max(bd.max_residual,std::abs(residual));
        if(o.retain_pair_predictions) out.predictions_[b][k]=pred;
    });
    for(auto& bd:d.batches) {bd.rms=std::sqrt(bd.sse/bd.rows); d.data_sse=finite(d.data_sse+bd.sse); d.data_max_residual=std::max(d.data_max_residual,bd.max_residual);}
    d.data_rms=std::sqrt(d.data_sse/rows);
    std::vector<double> delta(np); for(size_t i=0;i<np;++i) delta[i]=finite(out.parameters_[i]-p.penalty.anchor[i]);
    for(size_t k=0;k<np;++k) {
        std::copy(root.begin()+k*np,root.begin()+(k+1)*np,a.begin()); double r=dot(a,delta);
        d.matrix_objective=finite(d.matrix_objective+finite(r*r));
    }
    for(const auto& l:p.linear_penalties) {
        double y=lc_row(l), r=finite(dot(a,out.parameters_)-y);
        d.lc_objective=finite(d.lc_objective+finite(r*r));
    }
    d.total_objective=finite(finite(d.data_sse+d.matrix_objective)+d.lc_objective); d.objective_available=true;
    return out;
}
}} // namespace psi::isapol
