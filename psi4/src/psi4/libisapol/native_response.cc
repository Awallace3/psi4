/*
 * @BEGIN LICENSE
 *
 * Psi4: an open-source quantum chemistry software package
 *
 * Copyright (c) 2007-2025 The Psi4 Developers.
 *
 * This file is part of Psi4.
 *
 * Psi4 is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * @END LICENSE
 */
// Adapted from construct_restricted_c1_primitives_impl, restricted ALDA
// collocation, and assemble_restricted_singlet_hessian in
// camcasp_psi4 5449bd1a01c73f45c307b36b006e264c1e43b994.
// Adaptation copyright 2026 Psi4 Developers. No FrozenResponseContext/seals ported.
#include "native_response.h"
#include "parallel_work.h"
#include <algorithm>
#include <functional>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <vector>
#include <libint2.hpp>
#include "psi4/libfock/cubature.h"
#include "psi4/libfock/points.h"
#include "psi4/libfunctional/LibXCfunctional.h"
#include "psi4/libfunctional/superfunctional.h"
#include "psi4/libmints/basisset.h"
#include "psi4/libmints/gshell.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libmints/vector.h"
#include "psi4/libmints/wavefunction.h"

namespace psi { namespace isapol {
namespace {
void require_native(bool ok, const std::string& message) {
    if (!ok) throw std::invalid_argument("NativeResponseProvider: " + message);
}
std::size_t mul(std::size_t a, std::size_t b) {
    require_native(!b || a <= std::numeric_limits<std::size_t>::max()/b, "resource size overflow");
    return a*b;
}
std::size_t add(std::size_t a, std::size_t b) {
    require_native(a <= std::numeric_limits<std::size_t>::max()-b, "resource size overflow");
    return a+b;
}
void matrix_ok(const SharedMatrix& m, int rows, int cols, const char* name) {
    require_native(m && m->nirrep()==1 && m->nrow()==rows && m->ncol()==cols,
                   std::string(name)+" dimensions/C1 symmetry mismatch");
    for (int i=0;i<rows;++i) for (int j=0;j<cols;++j)
        require_native(std::isfinite(m->get(i,j)), std::string(name)+" must be finite");
}
void symmetric(const SharedMatrix& m, const char* name) {
    matrix_ok(m,m->nrow(),m->nrow(),name);
    for (int i=0;i<m->nrow();++i) for (int j=0;j<i;++j)
        require_native(std::abs(m->get(i,j)-m->get(j,i)) <=
                       1.e-10*std::max({1.0,std::abs(m->get(i,j)),std::abs(m->get(j,i))}),
                       std::string(name)+" reciprocity failure (not symmetrized)");
}
// Validate both representations before any engine/collocation buffer access.
// BasisSet uses one global puream for Libint shells; mixed high-AM Gaussian
// shells cannot be repaired here without changing the supplied AO context.
void validate_shells(const std::shared_ptr<BasisSet>& basis) {
    require_native(basis->nshell()>0 && basis->nshell()<=basis->nbf(), "invalid shell count");
    int offset=0, high_am_purity=-1;
    for(int s=0;s<basis->nshell();++s) {
        const auto& sh=basis->shell(s);
        require_native(sh.am()>=0 && sh.am()<=4 && sh.am()<=basis->max_am() &&
                       sh.nprimitive()>0 && sh.nprimitive()<=64 &&
                       sh.nprimitive()<=basis->max_nprimitive() &&
                       sh.ncenter()>=0 && sh.ncenter()<basis->molecule()->natom() &&
                       sh.function_index()==offset && sh.nfunction()>0 &&
                       sh.nfunction()<=basis->nbf()-offset, "invalid Gaussian shell bounds");
        const auto& l2=basis->l2_shell(s);
        require_native(l2.contr.size()==1, "Libint shell must have exactly one contraction");
        const auto& contraction=l2.contr[0];
        require_native(contraction.l==sh.am() && l2.size()==static_cast<std::size_t>(sh.nfunction()) &&
                       l2.alpha.size()==static_cast<std::size_t>(sh.nprimitive()) &&
                       contraction.coeff.size()==l2.alpha.size(), "inconsistent Libint shell dimensions");
        if(sh.am()>=2) {
            const int purity=sh.is_pure()?1:0;
            require_native(contraction.pure==sh.is_pure() &&
                           (high_am_purity<0 || high_am_purity==purity),
                           "mixed or inconsistent high-AM shell purity unsupported");
            high_am_purity=purity;
        }
        for(int axis=0;axis<3;++axis)
            require_native(std::isfinite(sh.coord(axis)) && l2.O[axis]==sh.coord(axis),
                           "stale or nonfinite Libint shell origin");
        for(int p=0;p<sh.nprimitive();++p)
            require_native(std::isfinite(sh.exp(p)) && sh.exp(p)>0 && l2.alpha[p]==sh.exp(p) &&
                           std::isfinite(contraction.coeff[p]), "invalid Libint shell primitives");
        offset+=sh.nfunction();
    }
    require_native(offset==basis->nbf(), "incomplete Gaussian shell function coverage");
}
// BasisSet is noncopyable. Reconstruct a private basis from original coefficients
// and a deep molecule clone, then verify effective coefficients. Both are needed:
// BasisFunctions uses effective coefficients, BasisSet::update_l2_shells uses
// original coefficients with Libint normalization. No basis name resolution.
std::shared_ptr<BasisSet> snapshot_basis(const std::shared_ptr<BasisSet>& source) {
    require_native(!source->has_ECP(), "ECP basis snapshots are not supported");
    validate_shells(source);
    auto mol = std::make_shared<Molecule>(source->molecule()->clone());
    for (int atom=0;atom<mol->natom();++atom)
        require_native(std::isfinite(mol->x(atom)) && std::isfinite(mol->y(atom)) && std::isfinite(mol->z(atom)),
                       "molecular coordinates must be finite");
    std::map<std::string,std::map<std::string,std::vector<ShellInfo>>> shells, ecps;
    for (int atom=0;atom<mol->natom();++atom) {
        const std::string key="NATIVE_RESPONSE_"+std::to_string(atom);
        mol->set_basis_by_number(atom,key,"BASIS");
        auto& list=shells[key][mol->label(atom)];
        for (int s=0;s<source->nshell();++s) {
            const auto& sh=source->shell(s);
            if (sh.ncenter()!=atom) continue;
            std::vector<double> exponents, coefficients;
            for (int p=0;p<sh.nprimitive();++p) {
                require_native(std::isfinite(sh.exp(p)) && sh.exp(p)>0 && std::isfinite(sh.coef(p)),
                               "invalid basis primitive");
                require_native(std::isfinite(sh.original_coef(p)),"invalid original basis coefficient");
                exponents.push_back(sh.exp(p)); coefficients.push_back(sh.original_coef(p));
            }
            ShellInfo info(sh.am(),coefficients,exponents,sh.is_pure()?Pure:Cartesian,Unnormalized);
            list.push_back(info);
        }
    }
    auto result=std::make_shared<BasisSet>("BASIS",mol,shells,ecps);
    require_native(result->nbf()==source->nbf() && result->nshell()==source->nshell(),
                   "basis reconstruction changed dimensions");
    validate_shells(result);
    for (int s=0;s<source->nshell();++s) {
        const auto& a=source->shell(s); const auto& b=result->shell(s);
        const auto& la=source->l2_shell(s); const auto& lb=result->l2_shell(s);
        require_native(a.am()==b.am() && a.ncenter()==b.ncenter() && a.is_pure()==b.is_pure() &&
                       a.nprimitive()==b.nprimitive() && a.function_index()==b.function_index(),
                       "basis reconstruction changed shell/function order");
        for (int axis=0;axis<3;++axis)
            require_native(std::isfinite(a.coord(axis)) && a.coord(axis)==b.coord(axis) &&
                           source->l2_shell(s).O[axis]==result->l2_shell(s).O[axis],
                           "basis/molecule geometry mismatch or stale integral shell");
        for (int p=0;p<a.nprimitive();++p)
            require_native(a.exp(p)==b.exp(p) && a.original_coef(p)==b.original_coef(p) &&
                           a.coef(p)==b.coef(p) && la.alpha[p]==lb.alpha[p] &&
                           la.contr[0].coeff[p]==lb.contr[0].coeff[p],
                           "basis reconstruction changed primitives (custom normalization unsupported)");
    }
    return result;
}
// Retain every AO in every block. BasisExtents is only a BlockOPoints constructor
// dependency, not a density-dependent screening policy.
class NativeCompleteBlock final : public BlockOPoints {
 public:
    NativeCompleteBlock(SharedVector x,SharedVector y,SharedVector z,SharedVector w,
                        std::shared_ptr<BasisExtents> extents,std::shared_ptr<BasisSet> basis)
        : BlockOPoints(x,y,z,w,extents) {
        shells_local_to_global_.clear(); functions_local_to_global_.clear();
        for(int s=0;s<basis->nshell();++s) shells_local_to_global_.push_back(s);
        for(int mu=0;mu<basis->nbf();++mu) functions_local_to_global_.push_back(mu);
        local_nbf_=basis->nbf();
    }
};
// One ALDA collocation policy, shared by the local primitive and by its row
// screen. Rows are visited in the caller's original order in fixed 128-row
// blocks; the callback sees this block's orbital values, the LibXC row weights
// w*fxc and the same inclusion mask the primitive itself uses. Nothing here
// renormalizes weights, drops AOs or alters the caller's quadrature.
template <class Visit>
void alda_rows(const std::shared_ptr<BasisSet>& basis, const SharedMatrix& c, int no,
               const SharedMatrix& grid, const std::string& kernel, double cutoff, Visit visit) {
    const int blocksize=128;
    auto f=SuperFunctional::blank();
    // This setter propagates to existing components: call before insertion so
    // LibXC retains the positive half-density threshold below.
    f->set_density_tolerance(cutoff);
    auto x=std::make_shared<LibXCFunctional>("XC_LDA_X",true);
    x->set_alpha(1.0); x->set_density_cutoff(std::nextafter(0.5*cutoff,0.0));
    f->add_x_functional(x);
    if (kernel!="alda_slater") {
        auto corr=std::make_shared<LibXCFunctional>(kernel=="alda_slater_pw92" ? "XC_LDA_C_PW" : "XC_LDA_C_VWN",true);
        corr->set_alpha(1.0); corr->set_density_cutoff(std::nextafter(0.5*cutoff,0.0));
        f->add_c_functional(corr);
    }
    f->set_max_points(blocksize); f->set_deriv(2); f->allocate();
    auto extents=std::make_shared<BasisExtents>(basis,1.e-12);
    BasisFunctions collocation(basis,blocksize,basis->nbf());
    for(int start=0;start<grid->nrow();start+=blocksize) {
        int count=std::min(blocksize,grid->nrow()-start);
        auto gx=std::make_shared<Vector>(count), gy=std::make_shared<Vector>(count);
        auto gz=std::make_shared<Vector>(count), gw=std::make_shared<Vector>(count);
        for(int p=0;p<count;++p) {
            gx->set(p,grid->get(start+p,0)); gy->set(p,grid->get(start+p,1));
            gz->set(p,grid->get(start+p,2)); gw->set(p,grid->get(start+p,3));
        }
        auto block=std::make_shared<NativeCompleteBlock>(gx,gy,gz,gw,extents,basis);
        collocation.compute_functions(block);
        auto orbitals=linalg::doublet(collocation.basis_value("PHI"),c,false,false);
        auto rho=std::make_shared<Vector>(count);
        std::vector<double> density(count);
        for(int p=0;p<count;++p) {
            double d=0;
            for(int i=0;i<no;++i) d+=2*std::pow(orbitals->get(p,i),2);
            require_native(std::isfinite(d),"nonfinite collocated density");
            density[p]=d; rho->set(p,std::max(d,cutoff));
        }
        std::map<std::string,SharedVector> input{{"RHO_A",rho}};
        auto& values=f->compute_functional(input,count,true);
        auto fxc=values.at("V_RHO_A_RHO_A");
        std::vector<double> factors(count);
        std::vector<unsigned char> included(count,0);
        for(int p=0;p<count;++p) {
            if(density[p]<cutoff || gw->get(p)==0) continue;
            factors[p]=gw->get(p)*fxc->get(p);
            require_native(std::isfinite(factors[p]),"nonfinite local kernel weight");
            included[p]=1;
        }
        visit(start,count,*orbitals,factors,included);
    }
}
SharedMatrix local_kernel(std::shared_ptr<BasisSet> basis, SharedMatrix c, int no,
                          SharedMatrix grid, const std::string& kernel, double cutoff,
                          bool blocked_blas3) {
    const int nv=c->ncol()-no, nov=no*nv;
    auto local=std::make_shared<Matrix>(nov,nov);
    alda_rows(basis,c,no,grid,kernel,cutoff,[&](int,int count,const Matrix& orbitals,
              const std::vector<double>& factors,const std::vector<unsigned char>& included) {
        auto transitions=std::make_shared<Matrix>(count,nov);
        for(int p=0;p<count;++p)
            for(int a=0;a<nv;++a) for(int i=0;i<no;++i)
                transitions->set(p,a*no+i,orbitals.get(p,i)*orbitals.get(p,no+a));
        auto tr=transitions->pointer();
        if(blocked_blas3) {
            // Same rows, same global block order, same inclusion mask and the
            // same left-associated factor*tr(p,t) scaling of the LEFT factor as
            // the accumulator below; the excluded rows are exact zeros so they
            // contribute nothing. One rank-count update per block, accumulating
            // into the same L. Only BLAS's summation order over the points
            // inside a block differs from the accumulator, so this agrees with
            // it to rounding and is NOT claimed to be bitwise identical.
            auto scaled=std::make_shared<Matrix>(count,nov);
            auto sc=scaled->pointer();
            for(int p=0;p<count;++p) {
                if(!included[p]) continue;
                const double factor=factors[p];
                for(int t=0;t<nov;++t) sc[p][t]=factor*tr[p][t];
            }
            local->gemm(true,false,1.0,scaled,transitions,1.0);
            return;
        }
        // LibXC, BasisFunctions and BLAS stay outside worker regions. Neither
        // triangular mirroring nor a reduction: every ordered (t,u) retains
        // global point order and the original left-associated product.
        auto output=local->pointer();
        detail::parallel_work(static_cast<size_t>(nov),8,[&](size_t t) {
            for(int p=0;p<count;++p) {
                if(!included[p]) continue;
                const double factor=factors[p];
                for(int u=0;u<nov;++u)
                    output[t][u] += factor*tr[p][t]*tr[p][u];
            }
        });
    });
    symmetric(local,"local kernel");
    return local;
}
// Shared restricted-C1 admission, split exactly where NativeResponseProvider
// already interleaved its resource guards: dimensions first, then the numerical
// state, then the owned deep snapshot. Messages are unchanged so both callers
// report the identical native contract.
struct RestrictedDims { int nbf=0,nmo=0,nocc=0,nvir=0; std::size_t nov=0; };
RestrictedDims restricted_dims(const std::shared_ptr<Wavefunction>& wfn, bool caller_converged) {
    require_native(wfn && caller_converged,"explicit wavefunction and caller convergence declaration required");
    require_native(wfn->nirrep()==1 && wfn->same_a_b_orbs() && wfn->same_a_b_dens() &&
                   wfn->nalpha()==wfn->nbeta() && wfn->soccpi()[0]==0,
                   "only restricted closed-shell C1 wavefunctions are supported");
    auto source=wfn->basisset();
    require_native(source && source->molecule() && source->nbf()>0,"missing orbital basis/molecule");
    RestrictedDims d;
    d.nbf=source->nbf(); d.nmo=wfn->nmo(); d.nocc=wfn->nalpha(); d.nvir=d.nmo-d.nocc;
    require_native(d.nocc>0 && d.nvir>0 && d.nmo<=d.nbf && wfn->doccpi()[0]==d.nocc,
                   "invalid integer Aufbau occupations or empty OV space");
    d.nov=mul(d.nocc,d.nvir);
    return d;
}
void validate_restricted_state(const std::shared_ptr<Wavefunction>& wfn, const RestrictedDims& d) {
    matrix_ok(wfn->Ca(),d.nbf,d.nmo,"Ca"); matrix_ok(wfn->Cb(),d.nbf,d.nmo,"Cb");
    matrix_ok(wfn->Da(),d.nbf,d.nbf,"Da"); matrix_ok(wfn->Db(),d.nbf,d.nbf,"Db");
    auto ea=wfn->epsilon_a(), eb=wfn->epsilon_b();
    require_native(ea && eb && ea->nirrep()==1 && eb->nirrep()==1 && ea->dim()==d.nmo && eb->dim()==d.nmo,
                   "orbital energy dimension/C1 mismatch");
    for(int p=0;p<d.nmo;++p) {
        require_native(std::isfinite(ea->get(p)) && ea->get(p)==eb->get(p),"energies must be finite and restricted");
        for(int mu=0;mu<d.nbf;++mu)
            require_native(wfn->Ca()->get(mu,p)==wfn->Cb()->get(mu,p),"alpha/beta orbitals differ");
    }
    for(int mu=0;mu<d.nbf;++mu) for(int nu=0;nu<d.nbf;++nu) {
        double density=0;
        for(int i=0;i<d.nocc;++i) density+=wfn->Ca()->get(mu,i)*wfn->Ca()->get(nu,i);
        require_native(std::isfinite(density) && wfn->Da()->get(mu,nu)==wfn->Db()->get(mu,nu) &&
                       std::abs(density-wfn->Da()->get(mu,nu))<=1.e-9*std::max(1.0,std::abs(density)),
                       "density inconsistent with integer occupied orbitals (fractional occupations unsupported)");
    }
    for(int a=0;a<d.nvir;++a) for(int i=0;i<d.nocc;++i) {
        double gap=ea->get(d.nocc+a)-ea->get(i);
        require_native(std::isfinite(gap) && gap>0,"occupied-virtual gaps must be finite and positive");
    }
}
struct OwnedState { std::shared_ptr<BasisSet> basis; SharedMatrix c, da; SharedVector eps; };
OwnedState own_restricted_state(const std::shared_ptr<Wavefunction>& wfn, const RestrictedDims& d) {
    OwnedState s;
    s.c=wfn->Ca()->clone(); s.da=wfn->Da()->clone();
    s.eps=std::make_shared<Vector>(*wfn->epsilon_a());
    s.basis=snapshot_basis(wfn->basisset());
    auto overlap=std::make_shared<Matrix>(d.nbf,d.nbf);
    // Local serial engine: no OneBodyAOInt/global tolerance or Process threads.
    // snapshot_basis has validated all shell dimensions and AO offsets before
    // these ordered shell-pair buffers (and the ERI buffers below) are indexed.
    libint2::Engine overlap_engine(libint2::Operator::overlap,s.basis->max_nprimitive(),s.basis->max_am(),0);
    overlap_engine.set_precision(1.e-15);
    for(int s0=0;s0<s.basis->nshell();++s0) for(int s1=0;s1<s.basis->nshell();++s1) {
        const auto& sh0=s.basis->shell(s0); const auto& sh1=s.basis->shell(s1);
        overlap_engine.compute(s.basis->l2_shell(s0),s.basis->l2_shell(s1));
        const double* buf=overlap_engine.results()[0];
        if(!buf) continue;
        std::size_t index=0;
        for(int m=0;m<sh0.nfunction();++m) for(int n=0;n<sh1.nfunction();++n,++index) {
            require_native(std::isfinite(buf[index]), "nonfinite overlap integral");
            overlap->set(sh0.function_index()+m,sh1.function_index()+n,buf[index]);
        }
    }
    auto gram=linalg::triplet(s.c,overlap,s.c,true,false,false);
    for(int p=0;p<d.nmo;++p) for(int q=0;q<d.nmo;++q)
        require_native(std::isfinite(gram->get(p,q)) && std::abs(gram->get(p,q)-(p==q?1.0:0.0))<=1.e-8,
                       "orbitals must be AO-overlap orthonormal");
    return s;
}
} // namespace

NativeResponseProvider::NativeResponseProvider(std::shared_ptr<Wavefunction> wfn,bool caller_converged,
        const std::string& kernel,double exact_exchange,double local_scale,SharedMatrix grid,
        double density_cutoff,std::size_t max_bytes,std::size_t max_nov,const std::string& algorithm)
    : kernel_(kernel),algorithm_(algorithm),a_(exact_exchange),b_(local_scale),cutoff_(density_cutoff) {
    require_native(wfn && caller_converged,"explicit wavefunction and caller convergence declaration required");
    require_native(algorithm=="ordered_pairwise" || algorithm=="shared_sweep",
                   "unsupported named response algorithm (no inference)");
    const bool shared_sweep = algorithm=="shared_sweep";
    require_native(kernel=="no_local" || kernel=="alda_slater" || kernel=="alda_slater_pw92" ||
                   kernel=="alda_slater_vwn","unsupported kernel (no functional inference)");
    require_native(std::isfinite(a_) && a_>=0 && a_<=1 && std::isfinite(b_) && b_>=0 && b_<=1,
                   "exact_exchange and local_scale must be finite in [0,1]");
    require_native(std::isfinite(cutoff_) && cutoff_>0 && std::nextafter(0.5*cutoff_,0.0)>0,
                   "density cutoff must be finite and positive with a positive LibXC half-cutoff");
    require_native(kernel!="no_local" || (b_==0 && !grid),"no_local requires local_scale=0 and no grid");
    const auto dims=restricted_dims(wfn,caller_converged);
    auto source=wfn->basisset();
    const int nbf=dims.nbf, nmo=dims.nmo;
    nocc_=dims.nocc; nvir_=dims.nvir;
    const std::size_t nov=dims.nov;
    require_native(max_nov>0 && nov<=max_nov && nov<=512,"dense OV resource limit (maximum 512)");
    require_native(source->max_am()<=4 && source->max_nprimitive()<=64 &&
                   nbf<=256 && mul(nov,mul(mul(nbf,nbf),mul(nbf,nbf)))<=64000000000ULL,
                   "direct JK work resource limit");
    int np=0;
    if(kernel!="no_local") {
        require_native(grid && grid->nirrep()==1 && grid->ncol()==4 && grid->nrow()>0,
                       "ALDA requires explicit grid rows [x,y,z,weight] in bohr");
        np=grid->nrow();
        // Two separately calibrated limits on the SAME np*nov^2 accumulation:
        // 2e9 for the hand accumulator it was measured against, 6.4e10 for the
        // blocked BLAS3 primitive, which performs the identical flop count at a
        // measured order-of-magnitude higher rate. Neither authorizes the other
        // and no caller option raises either.
        require_native(np<=1000000 && mul(np,mul(nov,nov))<=(shared_sweep?64000000000ULL:2000000000ULL),
                       "ALDA work resource limit");
    }
    // Before snapshots, integral engines, dense operators or AO JK allocation.
    // Conservative tensor envelope: outputs/copies, one-transition JK scratch,
    // block collocation and basis metadata allowance. Engine-private allocations
    // are backend-dependent; this is not an OS total-memory guarantee.
    auto elements=add(mul(12,mul(nov,nov)),mul(64,mul(nbf,nbf)));
    elements=add(elements,mul(16,mul(128,add(nbf,add(nmo,nov)))));
    elements=add(elements,mul(4,np));
    // shared_sweep holds one J and one K AO operator per (b,j) transition at
    // once; that is exactly what buys the single ordered quartet sweep.
    if(shared_sweep) elements=add(elements,mul(2,mul(nov,mul(nbf,nbf))));
    planned_bytes_=add(mul(elements,sizeof(double)),16ULL*1024*1024);
    if (kernel!="no_local") {
        planned_bytes_=add(planned_bytes_,mul(128,sizeof(double)+sizeof(unsigned char)));
        planned_bytes_=add(planned_bytes_,mul(nov,sizeof(std::exception_ptr)+sizeof(size_t)));
    }
    require_native(max_bytes>0 && planned_bytes_<=max_bytes,"dense workspace byte resource limit");
    validate_restricted_state(wfn,dims);
    if(grid) {
        matrix_ok(grid,np,4,"grid");
        for(int p=0;p<np;++p) require_native(grid->get(p,3)>=0,"grid weights must be nonnegative");
    }
    require_native(std::isfinite(wfn->energy()),"wavefunction energy must be finite");
    // Complete owned scientific input snapshot before any integrals/collocation.
    auto owned=own_restricted_state(wfn,dims);
    c_=owned.c; da_=owned.da; eps_=owned.eps; basis_=owned.basis;
    if(grid) grid_=grid->clone();
    auto co=std::make_shared<Matrix>(nbf,nocc_), cv=std::make_shared<Matrix>(nbf,nvir_);
    for(int mu=0;mu<nbf;++mu) {
        for(int i=0;i<nocc_;++i) co->set(mu,i,c_->get(mu,i));
        for(int a=0;a<nvir_;++a) cv->set(mu,a,c_->get(mu,nocc_+a));
    }

    v_=std::make_shared<Matrix>(nov,nov); x_=std::make_shared<Matrix>(nov,nov);
    y_=std::make_shared<Matrix>(nov,nov);
    // Current DirectJK constructs IntegralFactory::eri() using ambient global
    // SCREENING/INTS_TOLERANCE even with local JK Options. Its old standard-only
    // backend switch is also unavailable. Use the SAME native Libint shell data
    // and engine as libmints/eribase.cc, but explicitly pin precision and visit
    // all ordered quartets. No global option mutation, sieve or alternate backend.
    const int ns=basis_->nshell();
    if(!shared_sweep) {
    libint2::Engine engine(libint2::Operator::coulomb,basis_->max_nprimitive(),basis_->max_am(),0);
    engine.set_precision(1.e-15);
    for(int b=0;b<nvir_;++b) for(int j=0;j<nocc_;++j) {
        auto J=std::make_shared<Matrix>(nbf,nbf), K=std::make_shared<Matrix>(nbf,nbf);
        for(int s0=0;s0<ns;++s0) for(int s1=0;s1<ns;++s1)
        for(int s2=0;s2<ns;++s2) for(int s3=0;s3<ns;++s3) {
            engine.compute(basis_->l2_shell(s0),basis_->l2_shell(s1),basis_->l2_shell(s2),basis_->l2_shell(s3));
            const double* buf=engine.results()[0];
            if(!buf) continue; // engine precision zero; no external shell screening
            const auto& sh0=basis_->shell(s0); const auto& sh1=basis_->shell(s1);
            const auto& sh2=basis_->shell(s2); const auto& sh3=basis_->shell(s3);
            std::size_t index=0;
            for(int m=0;m<sh0.nfunction();++m) for(int n=0;n<sh1.nfunction();++n)
            for(int r=0;r<sh2.nfunction();++r) for(int s=0;s<sh3.nfunction();++s,++index) {
                int mu=sh0.function_index()+m, nu=sh1.function_index()+n;
                int rho=sh2.function_index()+r, sigma=sh3.function_index()+s;
                double eri=buf[index];
                require_native(std::isfinite(eri),"nonfinite ERI");
                // J_mn=(mn|rs) C_rj C_sb; K_mr=(mn|rs) C_nj C_sb.
                J->add(mu,nu,eri*co->get(rho,j)*cv->get(sigma,b));
                K->add(mu,rho,eri*co->get(nu,j)*cv->get(sigma,b));
            }
        }
        auto j_ov=linalg::triplet(co,J,cv,true,false,false);
        auto k_ov=linalg::triplet(co,K,cv,true,false,false);
        auto k_vo=linalg::triplet(cv,K,co,true,false,false);
        for(int a=0;a<nvir_;++a) for(int i=0;i<nocc_;++i) {
            int t=a*nocc_+i,u=b*nocc_+j;
            v_->set(t,u,j_ov->get(i,a)); x_->set(t,u,k_ov->get(i,a)); y_->set(t,u,k_vo->get(a,i));
        }
    }
    } else {
    // The identical ordered quartet sweep, identical engine and pinned
    // precision, visited ONCE with every (b,j) transition updated inside it.
    // For each element of each J_(b,j) and K_(b,j) the addends still arrive in
    // the original s0,s1,s2,s3 then m,n,r,s order, each one the same
    // left-associated eri*co(rho,j)*cv(sigma,b), so V, X and Y come out
    // bitwise identical to the ordered_pairwise arrangement above. Work is
    // split over s0 only: shell s0 owns rows mu of BOTH J and K, so workers
    // write disjoint output rows and nothing is reduced or reordered. Each
    // worker builds its own engine; no global option or sieve is touched.
    std::vector<SharedMatrix> js(nov), ks(nov);
    std::vector<double**> jp(nov), kp(nov);
    for(std::size_t u=0;u<nov;++u) {
        js[u]=std::make_shared<Matrix>(nbf,nbf); ks[u]=std::make_shared<Matrix>(nbf,nbf);
        jp[u]=js[u]->pointer(); kp[u]=ks[u]->pointer();
    }
    double** cop=co->pointer(); double** cvp=cv->pointer();
    detail::parallel_work(static_cast<std::size_t>(ns),2,[&](std::size_t shell0) {
        const int s0=static_cast<int>(shell0);
        libint2::Engine engine(libint2::Operator::coulomb,basis_->max_nprimitive(),basis_->max_am(),0);
        engine.set_precision(1.e-15);
        const auto& sh0=basis_->shell(s0);
        for(int s1=0;s1<ns;++s1) for(int s2=0;s2<ns;++s2) for(int s3=0;s3<ns;++s3) {
            engine.compute(basis_->l2_shell(s0),basis_->l2_shell(s1),basis_->l2_shell(s2),basis_->l2_shell(s3));
            const double* buf=engine.results()[0];
            if(!buf) continue; // engine precision zero; no external shell screening
            const auto& sh1=basis_->shell(s1); const auto& sh2=basis_->shell(s2);
            const auto& sh3=basis_->shell(s3);
            std::size_t index=0;
            for(int m=0;m<sh0.nfunction();++m) for(int n=0;n<sh1.nfunction();++n)
            for(int r=0;r<sh2.nfunction();++r) for(int s=0;s<sh3.nfunction();++s,++index) {
                const int mu=sh0.function_index()+m, nu=sh1.function_index()+n;
                const int rho=sh2.function_index()+r, sigma=sh3.function_index()+s;
                const double eri=buf[index];
                require_native(std::isfinite(eri),"nonfinite ERI");
                // J_mn=(mn|rs) C_rj C_sb; K_mr=(mn|rs) C_nj C_sb.
                const double* co_rho=cop[rho]; const double* co_nu=cop[nu];
                const double* cv_sigma=cvp[sigma];
                for(int b=0;b<nvir_;++b) {
                    const double cvb=cv_sigma[b]; const int base=b*nocc_;
                    for(int j=0;j<nocc_;++j) {
                        double* jrow=jp[base+j][mu]; double* krow=kp[base+j][mu];
                        jrow[nu]+=eri*co_rho[j]*cvb;
                        krow[rho]+=eri*co_nu[j]*cvb;
                    }
                }
            }
        }
    });
    for(int b=0;b<nvir_;++b) for(int j=0;j<nocc_;++j) {
        const int u=b*nocc_+j;
        auto j_ov=linalg::triplet(co,js[u],cv,true,false,false);
        auto k_ov=linalg::triplet(co,ks[u],cv,true,false,false);
        auto k_vo=linalg::triplet(cv,ks[u],co,true,false,false);
        for(int a=0;a<nvir_;++a) for(int i=0;i<nocc_;++i) {
            int t=a*nocc_+i;
            v_->set(t,u,j_ov->get(i,a)); x_->set(t,u,k_ov->get(i,a)); y_->set(t,u,k_vo->get(a,i));
        }
    }
    }
    symmetric(v_,"Coulomb"); symmetric(x_,"exchange direct"); symmetric(y_,"exchange transpose");
    local_=kernel=="no_local" ? std::make_shared<Matrix>(nov,nov) :
        local_kernel(basis_,c_,nocc_,grid_,kernel,cutoff_,shared_sweep);
    h1_=std::make_shared<Matrix>(nov,nov); h2_=std::make_shared<Matrix>(nov,nov);
    for(int t=0;t<static_cast<int>(nov);++t) for(int u=0;u<static_cast<int>(nov);++u) {
        double gap=t==u ? eps_->get(nocc_+t/nocc_)-eps_->get(t%nocc_) : 0;
        h1_->set(t,u,gap+4*v_->get(t,u)-a_*(x_->get(t,u)+y_->get(t,u))+4*b_*local_->get(t,u));
        h2_->set(t,u,gap-a_*(x_->get(t,u)-y_->get(t,u)));
    }
    symmetric(h1_,"H1"); symmetric(h2_,"H2"); // validate, never average
}
SharedMatrix NativeResponseProvider::h1() const { return h1_->clone(); }
SharedMatrix NativeResponseProvider::h2() const { return h2_->clone(); }
SharedMatrix NativeResponseProvider::coulomb() const { return v_->clone(); }
SharedMatrix NativeResponseProvider::exchange_direct() const { return x_->clone(); }
SharedMatrix NativeResponseProvider::exchange_transpose() const { return y_->clone(); }
SharedMatrix NativeResponseProvider::local_primitive() const { return local_->clone(); }
SharedMatrix NativeResponseProvider::orbitals() const { return c_->clone(); }
SharedVector NativeResponseProvider::energies() const { return std::make_shared<Vector>(*eps_); }
SharedMatrix NativeResponseProvider::density_alpha() const { return da_->clone(); }

IsaAldaGridScreen::IsaAldaGridScreen(std::shared_ptr<Wavefunction> wfn,bool caller_converged,
        const std::string& kernel,SharedMatrix grid,double density_cutoff,std::size_t max_bytes)
    : kernel_(kernel),cutoff_(density_cutoff) {
    require_native(wfn && caller_converged,"explicit wavefunction and caller convergence declaration required");
    require_native(kernel=="alda_slater" || kernel=="alda_slater_pw92" || kernel=="alda_slater_vwn",
                   "row screening needs a named local kernel (no_local has no grid rows)");
    require_native(std::isfinite(cutoff_) && cutoff_>0 && std::nextafter(0.5*cutoff_,0.0)>0,
                   "density cutoff must be finite and positive with a positive LibXC half-cutoff");
    require_native(grid && grid->nirrep()==1 && grid->ncol()==4 && grid->nrow()>0,
                   "ALDA requires explicit grid rows [x,y,z,weight] in bohr");
    const auto dims=restricted_dims(wfn,caller_converged);
    auto source=wfn->basisset();
    np_=grid->nrow();
    // Screening is collocation only: grid_rows*nbf*nmo, with no nov^2 term at
    // all. That absence is the whole reason it can look at a grid the primitive
    // cannot yet afford; it is a separate gate, not a waiver of the other one.
    require_native(source->max_am()<=4 && source->max_nprimitive()<=64 && dims.nbf<=256,
                   "ALDA screen basis resource limit");
    require_native(np_<=1000000 && mul(np_,mul(dims.nbf,dims.nmo))<=64000000000ULL,
                   "ALDA screen work resource limit");
    auto elements=add(mul(16,mul(128,add(dims.nbf,dims.nmo))),mul(6,np_));
    elements=add(elements,mul(64,mul(dims.nbf,dims.nbf)));
    planned_bytes_=add(mul(elements,sizeof(double)),16ULL*1024*1024);
    require_native(max_bytes>0 && planned_bytes_<=max_bytes,"dense workspace byte resource limit");
    validate_restricted_state(wfn,dims);
    matrix_ok(grid,np_,4,"grid");
    for(int p=0;p<np_;++p) require_native(grid->get(p,3)>=0,"grid weights must be nonnegative");
    require_native(std::isfinite(wfn->energy()),"wavefunction energy must be finite");
    auto owned=own_restricted_state(wfn,dims);
    grid_=grid->clone();
    value_.assign(np_,0.);
    const int no=dims.nocc, nv=dims.nvir;
    alda_rows(owned.basis,owned.c,no,grid_,kernel_,cutoff_,[&](int start,int count,
              const Matrix& orbitals,const std::vector<double>& factors,
              const std::vector<unsigned char>& included) {
        for(int p=0;p<count;++p) {
            if(!included[p]) continue; // exactly the rows the primitive itself skips
            double o=0,u=0;
            for(int i=0;i<no;++i) o+=std::pow(orbitals.get(p,i),2);
            for(int a=0;a<nv;++a) u+=std::pow(orbitals.get(p,no+a),2);
            const double v=std::abs(factors[p])*o*u;
            require_native(std::isfinite(v),"nonfinite ALDA row screen value");
            value_[start+p]=v;
        }
    });
    for(int p=0;p<np_;++p) {
        total_+=value_[p];
        maximum_=std::max(maximum_,value_[p]);
        if(value_[p]==0.) ++zeros_;
    }
    require_native(std::isfinite(total_),"ALDA row screen total overflowed");
    sorted_=value_;
    std::sort(sorted_.begin(),sorted_.end(),std::greater<double>());
}
void IsaAldaGridScreen::check_threshold(double threshold) const {
    require_native(std::isfinite(threshold) && threshold>=0,
                   "ALDA row screen threshold must be finite and nonnegative");
}
SharedVector IsaAldaGridScreen::values() const {
    auto v=std::make_shared<Vector>(np_);
    for(int p=0;p<np_;++p) v->set(p,value_[p]);
    return v;
}
std::vector<int> IsaAldaGridScreen::retained_rows(double threshold) const {
    check_threshold(threshold);
    std::vector<int> keep;
    for(int p=0;p<np_;++p) if(value_[p]>threshold) keep.push_back(p);
    return keep;
}
SharedMatrix IsaAldaGridScreen::retained(double threshold) const {
    const auto keep=retained_rows(threshold);
    require_native(!keep.empty(),"ALDA row screen retained no rows at this threshold");
    auto out=std::make_shared<Matrix>(static_cast<int>(keep.size()),4);
    for(std::size_t r=0;r<keep.size();++r)
        for(int c=0;c<4;++c) out->set(static_cast<int>(r),c,grid_->get(keep[r],c));
    return out;
}
double IsaAldaGridScreen::omitted_bound(double threshold) const {
    check_threshold(threshold);
    double bound=0;
    for(int p=0;p<np_;++p) if(!(value_[p]>threshold)) bound+=value_[p];
    return bound;
}
int IsaAldaGridScreen::omitted_count(double threshold) const {
    return np_-static_cast<int>(retained_rows(threshold).size());
}
double IsaAldaGridScreen::threshold_for_rows(int max_rows) const {
    require_native(max_rows>0,"ALDA row screen row target must be positive");
    if(max_rows>=np_-zeros_) return 0.;
    return sorted_[max_rows]; // largest omitted value; strict ">" keeps at most max_rows
}
} } // namespace psi::isapol
