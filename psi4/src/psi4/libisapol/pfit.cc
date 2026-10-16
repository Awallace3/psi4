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
[[noreturn]] void fail(const std::string& what) {
    throw std::invalid_argument("PFIT: " + what);
}
// Hot path of `data_rows`: take `const char*` so a passing check never builds a
// std::string message.
inline void require(bool ok, const char* what) {
    if (!ok) fail(what);
}
inline void require(bool ok, const std::string& what) {
    if (!ok) fail(what);
}
inline double finite(double x) {
    if (!std::isfinite(x)) fail("nonfinite input or numerical intermediate (overflow)");
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
            consume(b,packed,1,row.data(),&batch.targets[packed]); ++packed;
        }
    }
}
// Stable row-insertion Givens QR. Chunking only buffers computational rows;
// it never defines physical clouds. R has nonnegative diagonal. No normal-H
// arithmetic is used to solve the QR system. Each of the nt right-hand sides
// receives exactly the rotations, and so the arithmetic, of a lone one.
struct Givens {
    size_t n, chunk, nt, used=0;
    std::vector<double> r,z,buffer,discarded;
    Givens(size_t n_,size_t chunk_,size_t nt_=1):n(n_),chunk(chunk_),nt(nt_),r(n*n),z(n*nt),buffer(chunk*(n+nt)),discarded(nt) {}
    void flush() {
        for(size_t i=0;i<used;++i) {
            double* a=buffer.data()+i*(n+nt); double* y=a+n;
            for(size_t j=0;j<n;++j) {
                double h=finite(std::hypot(r[j*n+j],a[j])); if(h==0) continue;
                double c=r[j*n+j]/h, s=a[j]/h; r[j*n+j]=h;
                for(size_t k=j+1;k<n;++k) {
                    double old=r[j*n+k]; r[j*n+k]=finite(c*old+s*a[k]); a[k]=finite(-s*old+c*a[k]);
                }
                for(size_t t=0;t<nt;++t) {
                    double old=z[j*nt+t]; z[j*nt+t]=finite(c*old+s*y[t]); y[t]=finite(-s*old+c*y[t]);
                }
            }
            for(size_t t=0;t<nt;++t) discarded[t]=finite(discarded[t]+finite(y[t]*y[t]));
        }
        used=0;
    }
    void push(const double* a,const double* y) {
        std::copy(a,a+n,buffer.begin()+used*(n+nt)); std::copy(y,y+nt,buffer.begin()+used*(n+nt)+n);
        if(++used==chunk) flush();
    }
    // Single right-hand-side continuation of column t; requires a flushed state.
    Givens column(size_t t) const {
        Givens g(n,chunk); g.r=r; g.discarded[0]=discarded[t];
        for(size_t j=0;j<n;++j) g.z[j]=z[j*nt+t];
        return g;
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
namespace detail {
struct IsaPfitSolverImpl {
    // A consumer receives `rows` consecutive packed rows of one batch starting
    // at `start`: row-major design[rows][np] and targets[rows][nt], all finite,
    // where column t of the targets belongs to problem t.
    using Consumer = std::function<void(size_t,size_t,size_t,const double*,const double*)>;
    using Replay = std::function<void(size_t,const Consumer&)>;
    template<class Problem>
    static void check(const Problem& p) {
    require(finite(p.frequency_au)>=0,"frequency must be nonnegative");
    const auto& provenance=p.target_provenance;
    require(provenance.origin==IsaPfitTargetOrigin::SuppliedActualPointResponse ||
            provenance.origin==IsaPfitTargetOrigin::SuppliedFittedPropagatorPointResponse ||
            provenance.origin==IsaPfitTargetOrigin::NativeDirectActualPointResponse ||
            provenance.origin==IsaPfitTargetOrigin::NativeFittedPointResponse ||
            provenance.origin==IsaPfitTargetOrigin::SyntheticAnalyticTest,"target origin must be declared");
    require(provenance.convention==IsaPfitTargetConvention::NegativeInducedPotentialPerUnitSourceChargeAtomicUnits,"wrong/unspecified target convention");
    text(provenance.source_id); text(provenance.generation_record); text(p.model.provenance);
    if(provenance.origin==IsaPfitTargetOrigin::SuppliedFittedPropagatorPointResponse ||
       provenance.origin==IsaPfitTargetOrigin::NativeFittedPointResponse) {
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
    require(p.model.fixed.size()==np,"model dimension mismatch");
    vector_shape(p.model.fixed_values,np); vector_shape(p.penalty.anchor,np);
    matrix_shape(p.penalty.matrix,np,np); symmetric(p.penalty.matrix);
    for(const auto& l:p.linear_penalties) {
        vector_shape(l.coefficients,np); finite(l.target); require(finite(l.strength)>=0,"negative LC strength");
    }
    }
    // Problems share one design: the same model rows, only targets, penalties,
    // frequencies and provenance differ. Every problem's fit is exactly what it
    // would be alone, except that the data rhs of nt>1 is one DGEMM.
    template<class Problem>
    static std::vector<IsaPfitResult> solve(const std::vector<const Problem*>& ps, const IsaPfitOptions& o,
                              const std::vector<IsaPfitCloudRows>& clouds,
                              const Replay& replay, size_t source_bytes, size_t block_rows) {
    require(o.solver==IsaPfitSolver::StreamingQR || o.solver==IsaPfitSolver::NormalEquationsDSYSV,"unknown solver");
    require(o.qr_chunk_rows>0 && o.qr_chunk_rows<=static_cast<size_t>(std::numeric_limits<int>::max()),"invalid QR chunk rows");
    require(finite(o.rank_relative_tolerance)>0 && o.rank_relative_tolerance<1,"invalid rank tolerance");
    require(finite(o.minimum_solver_rcond)>=0 && o.minimum_solver_rcond<=1,"invalid rcond threshold");
    require(!ps.empty() && ps.size()<=static_cast<size_t>(std::numeric_limits<int>::max()),"invalid problem count");
    const size_t nt=ps.size();
    for(const auto* p:ps) check(*p);
    const auto& model=ps.front()->model;
    for(const auto* p:ps)
        require(p->model.channel_labels==model.channel_labels && p->model.parameter_labels==model.parameter_labels &&
                p->model.parameter_units==model.parameter_units && p->model.fixed==model.fixed &&
                p->model.fixed_values==model.fixed_values,"problems sharing rows must share one model");
    size_t np=model.parameter_labels.size();
    size_t rows=0; std::set<std::string> batch_names;
    for(const auto& b:clouds) {
        text(b.label); require(batch_names.insert(b.label).second,"duplicate batch label");
        require(b.points>0 && b.full_row_count>0,"empty physical cloud");
        rows=add(rows,b.full_row_count);
    }
    require(rows>0,"empty data not supported");
    size_t nf=0; for(bool fixed:model.fixed) if(!fixed) ++nf;
    std::vector<size_t> free_indices, fixed_indices;
    for(size_t i=0;i<np;++i) (model.fixed[i]?fixed_indices:free_indices).push_back(i);
    // Conservative numeric payload budget: all persistent result buffers, small
    // LAPACK copies/factors/diagnostics, integer arrays (charged as doubles), and
    // QR buffer. No input numerical copy is made; input is consumed synchronously.
    // Additional LAPACK queried workspace must fit on top of this base. Not RSS.
    size_t each=add(mul(40,mul(np,np)),mul(64,np));
    if(o.retain_pair_predictions) each=add(each,rows);
    each=add(each,mul(16,clouds.size()));
    size_t scalars=mul(each,nt);
    // StreamingQR keeps the shared chunk buffer alive while one per-problem
    // column(t) copy solves, and that copy has its own chunk*(nf+1) buffer; its
    // r and z are inside the 40*np^2 slack above, its buffer is not.
    if(o.solver==IsaPfitSolver::StreamingQR && nf)
        scalars=add(scalars,add(mul(o.qr_chunk_rows,add(nf,nt)),mul(o.qr_chunk_rows,add(nf,1))));
    require(block_rows>0,"row block capacity must be positive");
    scalars=add(scalars,mul(block_rows,add(nf,mul(2,nt))));
    Budget budget{add(mul(scalars,sizeof(double)),source_bytes),o.maximum_work_bytes};
    require(budget.base<=budget.limit,"kernel numerical buffers exceed maximum_work_bytes");
    size_t peak=budget.base; budget.peak=&peak;
    std::vector<IsaPfitResult> outs(nt);
    std::vector<std::vector<double>> roots(nt);
    std::vector<double> a(np);
    // Use identical scaled LC rows for assembly, effective diagnostics and
    // residuals. Multiplying strength first can underflow and even produce an
    // asymmetric reported Gram matrix for otherwise finite scaled rows.
    auto lc_row=[&](const IsaPfitLinearPenalty& l) {
        double s=std::sqrt(l.strength);
        for(size_t k=0;k<np;++k) a[k]=finite(s*l.coefficients[k]);
        return finite(s*l.target);
    };
    for(size_t t=0;t<nt;++t) {
    const auto& p=*ps[t]; auto& out=outs[t]; auto& d=out.diagnostics_;
    out.settings_=o; out.frequency_=p.frequency_au;
    out.provenance_=p.target_provenance; out.model_provenance_=p.model.provenance;
    out.parameter_labels_=p.model.parameter_labels; out.parameter_units_=p.model.parameter_units;
    out.channel_labels_=p.model.channel_labels;
    for(const auto& b:clouds) out.batch_labels_.push_back(b.label);
    d.data_rows=rows; d.augmented_rows=add(add(rows,np),p.linear_penalties.size());
    d.free_indices=free_indices;
    for(const auto& b:clouds) d.batches.push_back({b.points,b.full_row_count,0,0,0});
    out.penalty_=zeros(np); out.lc_=zeros(np);
    out.matrix_rhs_.assign(np,0); out.lc_rhs_.assign(np,0);
    auto eigvec=column(p.penalty.matrix); auto eigval=eigen(eigvec,static_cast<int>(np),budget);
    d.penalty_min_eigenvalue=eigval.front(); require(eigval.front()>=0,"penalty is not PSD under strict eigenvalue policy");
    // Square-root rows preserve every positive mode, however small. Both paths
    // use exactly these rows, and the reported effective matrix is their Gram.
    auto& root=roots[t]; root.resize(mul(np,np));
    for(size_t k=0;k<np;++k) for(size_t j=0;j<np;++j) root[k*np+j]=finite(std::sqrt(eigval[k])*eigvec[j+k*np]);
    for(size_t i=0;i<np;++i) for(size_t j=0;j<np;++j) {
        double x=0; for(size_t k=0;k<np;++k) x=finite(x+finite(root[k*np+i]*root[k*np+j]));
        out.penalty_.values[i*np+j]=x;
        d.penalty_correction_max=std::max(d.penalty_correction_max,std::abs(finite(x-p.penalty.matrix.values[i*np+j])));
    }
    for(size_t i=0;i<np;++i) for(size_t j=0;j<np;++j)
        out.matrix_rhs_[i]=finite(out.matrix_rhs_[i]+finite(out.penalty_.values[i*np+j]*p.penalty.anchor[j]));
    for(const auto& l:p.linear_penalties) {
        double y=lc_row(l);
        for(size_t i=0;i<np;++i) {
            out.lc_rhs_[i]=finite(out.lc_rhs_[i]+finite(a[i]*y));
            for(size_t j=0;j<np;++j)
                out.lc_.values[i*np+j]=finite(out.lc_.values[i*np+j]+finite(a[i]*a[j]));
        }
    }
    }
    const bool streaming=o.solver==IsaPfitSolver::StreamingQR && nf;
    Givens qr(streaming?nf:0, streaming?o.qr_chunk_rows:1, nt);
    // Blocks are accumulated with BLAS; overflow anywhere in a product
    // propagates to a nonfinite sum, so checking the accumulators per block
    // enforces the same finiteness policy as checking every operation.
    std::vector<double> free_rows(mul(block_rows,nf)), y(mul(block_rows,nt));
    // DSYRK fills only the lower triangle; it is mirrored once all rows are in.
    auto accumulate=[&](std::vector<double>& normal,double* rhs,size_t width,size_t count) {
        int n=static_cast<int>(nf), m=static_cast<int>(count), w=static_cast<int>(width);
        C_DSYRK('L','T',n,m,1.,free_rows.data(),n,1.,normal.data(),n);
        if(width==1) C_DGEMV('T',m,n,1.,free_rows.data(),n,y.data(),1,1.,rhs,1);
        else C_DGEMM('T','N',n,w,m,1.,free_rows.data(),n,y.data(),w,1.,rhs,w);
        values(normal); for(size_t i=0;i<mul(nf,width);++i) finite(rhs[i]);
    };
    auto gather=[&](const double* design,const double* targets,size_t count,size_t width) {
        require(count<=block_rows,"row block exceeds declared capacity");
        for(size_t r=0;r<count;++r) {
            const double* row=design+r*np; double* f=free_rows.data()+r*nf;
            // Only fixed parameters shift the targets, in ascending order as before.
            for(size_t t=0;t<width;++t) {
                double& v=y[r*width+t]; v=targets[r*width+t];
                for(size_t k:fixed_indices) v=finite(v-finite(row[k]*model.fixed_values[k]));
            }
            if(nf==np) std::copy(row,row+np,f);
            else for(size_t i=0;i<nf;++i) f[i]=row[free_indices[i]];
        }
    };
    std::vector<double> normal(mul(nf,nf),0.), rhs(mul(nf,nt),0.);
    replay(0,[&](size_t,size_t,size_t count,const double* design,const double* targets) {
        gather(design,targets,count,nt);
        if(!nf) return;
        accumulate(normal,rhs.data(),nt,count);
        if(streaming) for(size_t r=0;r<count;++r) qr.push(free_rows.data()+r*nf,y.data()+r*nt);
    });
    if(streaming) qr.flush();
    std::vector<size_t> active;
    for(size_t t=0;t<nt;++t) {
    const auto& p=*ps[t]; auto& out=outs[t]; auto& d=out.diagnostics_; const auto& root=roots[t];
    out.normal_={nf,nf,normal}; out.rhs_.resize(nf);
    for(size_t i=0;i<nf;++i) out.rhs_[i]=rhs[i*nt+t];
    Givens single=streaming?qr.column(t):Givens(0,1);
    auto append=[&](double target) {
        gather(a.data(),&target,1,1);
        if(!nf) return;
        accumulate(out.normal_.values,out.rhs_.data(),1,1);
        if(streaming) single.push(free_rows.data(),y.data());
    };
    for(size_t k=0;k<np;++k) {
        std::copy(root.begin()+k*np,root.begin()+(k+1)*np,a.begin()); append(dot(a,p.penalty.anchor));
    }
    for(const auto& l:p.linear_penalties) append(lc_row(l));
    for(size_t i=0;i<nf;++i) for(size_t j=0;j<i;++j) out.normal_.values[j*nf+i]=out.normal_.values[i*nf+j];
    std::vector<double> x;
    if(!nf) out.status_=IsaPfitStatus::AllFixed;
    else {
        int n=static_cast<int>(nf); double rcond=0;
        std::vector<double> spectrum, factor;
        if(streaming) {
            single.flush(); d.qr_discarded_rhs_sse=single.discarded[0];
            factor=column({nf,nf,single.r}); auto copy=factor; spectrum.resize(nf);
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
        if(d.numerical_rank<nf) { out.status_=IsaPfitStatus::RankDeficient; continue; }
        if(!streaming) {
            factor=column(out.normal_); x=out.rhs_; std::vector<int> piv(nf); double query=0;
            info_ok(C_DSYSV('L',n,1,factor.data(),n,piv.data(),x.data(),n,&query,-1));
            int l=budget.workspace(query); std::vector<double> work(l);
            d.lapack_info=C_DSYSV('L',n,1,factor.data(),n,piv.data(),x.data(),n,work.data(),l);
            require(d.lapack_info>=0,"DSYSV illegal argument");
            if(d.lapack_info>0) {out.status_=IsaPfitStatus::RankDeficient; continue;}
            double norm=0;
            for(size_t j=0;j<nf;++j) { double sum=0; for(size_t i=0;i<nf;++i) sum=finite(sum+std::abs(out.normal_.values[i*nf+j])); norm=std::max(norm,sum); }
            d.normal_h_norm1=norm;
            std::vector<double> cw(2*nf); std::vector<int> ci(nf);
            info_ok(C_DSYCON('L',n,factor.data(),n,piv.data(),norm,&rcond,cw.data(),ci.data()));
            d.normal_h_rcond=finite(rcond); d.condition_estimate_available=true;
        }
        if(rcond<o.minimum_solver_rcond || rcond<=0) {out.status_=IsaPfitStatus::IllConditioned; continue;}
        if(streaming) {
            x=single.z; d.lapack_info=C_DTRTRS('U','N','N',n,1,factor.data(),n,x.data(),n);
            require(d.lapack_info>=0,"DTRTRS illegal argument");
            if(d.lapack_info>0) {out.status_=IsaPfitStatus::RankDeficient; continue;}
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
    for(size_t i=0;i<nf;++i) out.parameters_[free_indices[i]]=x[i];
    if(o.retain_pair_predictions) {out.predictions_.resize(clouds.size()); for(size_t b=0;b<clouds.size();++b) out.predictions_[b].resize(clouds[b].full_row_count);}
    active.push_back(t);
    }
    for(auto& out:outs) out.diagnostics_.work_budget_bytes=peak;
    // Failed fits keep the single-problem early return: no residual traversal
    // unless some problem has parameters.
    if(active.empty()) return outs;
    const size_t na=active.size();
    std::vector<double> parameters(mul(np,na));
    for(size_t c=0;c<na;++c) for(size_t k=0;k<np;++k) parameters[k*na+c]=outs[active[c]].parameters_[k];
    replay(1,[&](size_t b,size_t start,size_t count,const double* design,const double* targets) {
        require(count<=block_rows,"row block exceeds declared capacity");
        int m=static_cast<int>(count), n=static_cast<int>(np), w=static_cast<int>(na);
        if(na==1) C_DGEMV('N',m,n,1.,const_cast<double*>(design),n,parameters.data(),1,0.,y.data(),1);
        else C_DGEMM('N','N',m,w,n,1.,const_cast<double*>(design),n,parameters.data(),w,0.,y.data(),w);
        for(size_t c=0;c<na;++c) {
            auto& out=outs[active[c]]; auto& bd=out.diagnostics_.batches[b];
            for(size_t r=0;r<count;++r) {
                double pred=finite(y[r*na+c]), residual=finite(pred-targets[r*nt+active[c]]);
                bd.sse=finite(bd.sse+finite(residual*residual)); bd.max_residual=std::max(bd.max_residual,std::abs(residual));
                if(o.retain_pair_predictions) out.predictions_[b][start+r]=pred;
            }
        }
    });
    for(size_t t:active) {
    const auto& p=*ps[t]; auto& out=outs[t]; auto& d=out.diagnostics_; const auto& root=roots[t];
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
    }
    return outs;
}
};
} // namespace detail

IsaPfitResult isa_pfit_solve(const IsaPfitProblem& p,const IsaPfitOptions& o) {
    const size_t np=p.model.parameter_labels.size(), nc=p.model.channel_labels.size();
    require(p.model.parameter_tensors.size()==np,"model dimension mismatch");
    for(const auto& k:p.model.parameter_tensors) { matrix_shape(k,nc,nc); symmetric(k); }
    std::vector<IsaPfitCloudRows> clouds;
    for(const auto& b:p.batches) {
        size_t n=b.points_bohr.size();
        require(n>0,"empty physical batch");
        size_t pairs=mul(n,add(n,1))/2;
        require(b.targets.size()==pairs,"wrong packed triangular target count"); values(b.targets);
        matrix_shape(b.fields,n,nc);
        for(const auto& xyz:b.points_bohr) for(double x:xyz) finite(x);
        clouds.push_back({b.label,n,pairs,0});
    }
    return detail::IsaPfitSolverImpl::solve(std::vector<const IsaPfitProblem*>{&p},o,clouds,
        [&](size_t,const detail::IsaPfitSolverImpl::Consumer& consume){data_rows(p,consume);},0,1).front();
}

std::vector<IsaPfitResult> isa_pfit_solve_rows_multi(const std::vector<IsaPfitRowProblem>& input,
                                                   const IsaPfitRowSource& input_source,const IsaPfitOptions& options) {
    // Native callbacks receive the same immutability guarantee as Python
    // producers, even when they retain references to the caller's declarations.
    const auto ps=input;
    const auto o=options;
    const auto source=input_source;
    require(!ps.empty(),"no row problems");
    const auto& p=ps.front();
    for(const auto& q:ps)
        require(q.cloud.label==p.cloud.label && q.cloud.points==p.cloud.points &&
                q.cloud.full_row_count==p.cloud.full_row_count &&
                q.cloud.maximum_block_rows==p.cloud.maximum_block_rows,"problems sharing rows must share one cloud");
    const size_t nt=ps.size();
    size_t n=p.cloud.points, next_n=add(n,1);
    require(n>0,"empty row cloud");
    const size_t rows=(n%2)?mul(n,next_n/2):mul(n/2,next_n);
    require(rows==p.cloud.full_row_count,"wrong complete-cloud packed row count");
    const size_t capacity=p.cloud.maximum_block_rows, np=p.model.parameter_labels.size();
    require(capacity>0,"row block capacity must be positive");
    require(source.begin_pass && source.next && source.finish_pass,"incomplete row source");
    size_t payload=mul(capacity,add(np,nt));
    size_t bytes=mul(add(payload,np),sizeof(double));
    std::array<unsigned char,32> first_digest{};
    auto replay=[&](size_t pass,const detail::IsaPfitSolverImpl::Consumer& consume) {
        // The common solver admits this workspace before invoking the source.
        std::vector<double> design(mul(capacity,np)), targets(mul(capacity,nt));
        source.begin_pass(pass);
        size_t expected=0;
        while(true) {
            auto block=source.next(capacity,np,design.data(),targets.data());
            require(block.packed_start==expected,"row source gap, duplicate or reordered start");
            if(!block.rows) {
                require(expected==rows,"row source ended before complete cloud");
                break;
            }
            require(block.rows<=capacity && block.rows<=rows-expected,"excess/oversized row block");
            for(size_t i=0;i<mul(block.rows,np);++i) finite(design[i]);
            for(size_t i=0;i<mul(block.rows,nt);++i) finite(targets[i]);
            consume(0,expected,block.rows,design.data(),targets.data());
            expected=add(expected,block.rows);
        }
        auto digest=source.finish_pass();
        if(!pass) first_digest=digest;
        else require(digest==first_digest,"row source replay content changed");
    };
    std::vector<const IsaPfitRowProblem*> pointers;
    for(const auto& q:ps) pointers.push_back(&q);
    return detail::IsaPfitSolverImpl::solve(pointers,o,{p.cloud},replay,bytes,capacity);
}

IsaPfitResult isa_pfit_solve_rows(const IsaPfitRowProblem& p,const IsaPfitRowSource& source,
                                const IsaPfitOptions& options) {
    return isa_pfit_solve_rows_multi({p},source,options).front();
}
}} // namespace psi::isapol
