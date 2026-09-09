/* CamCASP casimir.f90:262-435,486-649 conventions, not executable source extraction.
 * Alston J. Misquitta and Anthony J. Stone; Copyright (c) 2019 Anthony Stone.
 * MIT: see RECOUPLED_CAMCASP_LICENSE. Source hashes: realcg_manifest.txt.
 * Psi4 additions: Copyright (c) 2026 The Psi4 Developers. LGPL-3.0-only.
 */
#include "recoupled_dispersion.h"
#include "realcg_tables.h"
#include "recoupling_tables.h"
#include "psi4/libmints/matrix.h"
#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>
namespace psi { namespace isapol {
namespace recoupled_detail {
using Quad = std::array<int,4>;
void charge(std::size_t& total, std::size_t a, std::size_t b, std::size_t cap) {
    if (a && b > (cap - total) / a)
        throw std::invalid_argument("recoupled: resource budget exceeded");
    total += a*b;
}
void finite(std::complex<double> z) {
    if (!std::isfinite(z.real()) || !std::isfinite(z.imag()))
        throw std::invalid_argument("recoupled: nonfinite arithmetic/overflow");
}
bool has(const IsaAnisotropicSite& s, int l) {
    return std::find(s.ranks.begin(), s.ranks.end(), l) != s.ranks.end();
}
bool present(const IsaAnisotropicSite& a, const IsaAnisotropicSite& b, const Quad& q) {
    return has(a,q[0]) && has(a,q[1]) && has(b,q[2]) && has(b,q[3]);
}
Quad quad(const RecouplingTerm& t) { return {{t.la,t.lap,t.lb,t.lbp}}; }
std::size_t count(const RecouplingBlock& block, const IsaAnisotropicSite& a,
                  const IsaAnisotropicSite& b) {
    std::size_t n = 0;
    for (const auto& t : block) if (present(a,b,quad(t))) ++n;
    return n;
}
}
std::vector<std::string> IsaRecoupledBlock::components() const {
    std::vector<std::string> out;
    for (int t=first_component; t<=last_component; ++t) out.push_back(component_label(t));
    return out;
}
IsaRecoupledModel::IsaRecoupledModel(const IsaAnisotropicModel& local) : source_(local) {
    using namespace recoupled_detail;
    const auto& input = source_.sites_;
    const auto freq = frequencies();
    nfrequency_ = freq.size();
    std::size_t elements=0, work=0;
    for (const auto& site : input) {
        for (int l : site.ranks) if (l > 3)
            throw std::invalid_argument("recoupled: declared rank 4 unsupported; no silent truncation");
        for (int l : site.ranks) for (int p : site.ranks) {
            charge(elements, freq.size(), (l+p+1)*(l+p+1)-(l-p)*(l-p), 8000000);
            charge(work, freq.size(), realcg_terms(l,p).size(), 100000000);
        }
    }
    sites_.reserve(input.size());
    for (const auto& site : input) {
        std::vector<IsaRecoupledBlock> blocks;
        int offset_l=0;
        for (int l : site.ranks) {
            int offset_p=0;
            for (int p : site.ranks) {
                IsaRecoupledBlock block;
                block.la=l; block.lap=p;
                block.first_component=(l-p)*(l-p)+1;
                block.last_component=(l+p+1)*(l+p+1);
                const auto width=block.last_component-block.first_component+1;
                block.values.resize(freq.size()*width);
                const auto terms=realcg_terms(l,p);
                for (std::size_t f=0; f<freq.size(); ++f) {
                    for (const auto& t : terms) {
                        auto& z=block.values[f*width+t.v+1-block.first_component];
                        z += t.value()*site.responses[f]->get(offset_l+t.k,offset_p+t.q);
                        finite(z);
                    }
                }
                blocks.push_back(std::move(block));
                offset_p+=2*p+1;
            }
            offset_l+=2*l+1;
        }
        sites_.push_back(std::move(blocks));
    }
}
std::complex<double> IsaRecoupledModel::value(std::size_t site, std::size_t f,
                                             int l, int p, int t) const {
    if (site>=sites_.size() || f>=nfrequency_ || l<1 || l>3 || p<1 || p>3 || t<1 || t>81)
        throw std::invalid_argument("recoupled: invalid tensor index");
    for (const auto& b : sites_[site]) if (b.la==l && b.lap==p) {
        if (t<b.first_component || t>b.last_component) return {};
        return b.values[f*(b.last_component-b.first_component+1)+t-b.first_component];
    }
    return {};
}
double IsaRecoupledPair::coefficient(int n, int t, int u, int J) const {
    if (n<6 || n>12 || t<1 || t>81 || u<1 || u>81 || J<0 || J>10)
        throw std::invalid_argument("recoupled: invalid coefficient index");
    for (const auto& c : coefficients) if (c.order==n && c.t==t && c.u==u && c.J==J) return c.value;
    return 0.;
}
IsaRecoupledDispersionResult isa_recoupled_dispersion(const IsaRecoupledModel& a,
    const IsaRecoupledModel& b, const std::vector<double>& weights, int max_order) {
    using namespace recoupled_detail;
    if (max_order<6 || max_order>12) throw std::invalid_argument("recoupled: max_order must be 6..12");
    const auto freq=a.frequencies();
    if (freq!=b.frequencies() || weights.size()!=freq.size())
        throw std::invalid_argument("recoupled: exact matching frequency grids and CP weights required");
    bool positive=false;
    for (std::size_t f=0; f<freq.size(); ++f) {
        if (!std::isfinite(weights[f]) || (freq[f]==0. ? weights[f]!=0. : weights[f]<=0.))
            throw std::invalid_argument("recoupled: static weights must be zero; dynamic weights positive finite");
        positive = positive || weights[f]>0.;
    }
    if (!positive) throw std::invalid_argument("recoupled: empty positive quadrature");
    const auto& sa=a.source_.sites_;
    const auto& sb=b.source_.sites_;
    std::size_t pairs=0, records=0, work=0;
    charge(pairs,sa.size(),sb.size(),4096);
    // Preflight all pairs before allocating results. No dense [n,81,81,J] tensor.
    for (const auto& x : sa) for (const auto& y : sb)
        for (int i=0; i<num_recoupling_blocks(); ++i) {
            int n,L1,L2,J;
            const auto block=recoupling_block_at(i,&n,&L1,&L2,&J);
            if (n>max_order) continue;
            const auto terms=count(block,x,y);
            if (!terms) continue;
            const std::size_t size=(2*L1+1)*(2*L2+1);
            charge(records,size,1,2000000);
            charge(work,size*terms,freq.size(),100000000);
        }
    IsaRecoupledDispersionResult result(a,b);
    result.frequencies=freq; result.cp_weights=weights; result.max_order=max_order;
    result.pairs.reserve(pairs);
    for (std::size_t ia=0; ia<sa.size(); ++ia) for (std::size_t ib=0; ib<sb.size(); ++ib) {
        IsaRecoupledPair pair; pair.site_a=ia; pair.site_b=ib;
        for (int n=6; n<=max_order; ++n) {
            IsaRecoupledCoverage cov; cov.order=n;
            std::set<Quad> table;
            for (int i=0; i<num_recoupling_blocks(); ++i) {
                int nn,L1,L2,J;
                const auto block=recoupling_block_at(i,&nn,&L1,&L2,&J);
                if (nn!=n) continue;
                for (const auto& term : block) table.insert(quad(term));
                if (!count(block,sa[ia],sb[ib])) continue;
                for (int t=component_first(L1); t<=component_last(L1); ++t)
                    for (int u=component_first(L2); u<=component_last(L2); ++u) {
                        double value=0.;
                        for (const auto& term : block) {
                            if (!present(sa[ia],sb[ib],quad(term))) continue;
                            std::complex<double> integral{};
                            for (std::size_t f=0; f<freq.size(); ++f) {
                                if (weights[f]==0.) continue; // BEFORE multiplication, especially static nodes
                                integral += weights[f]*a.value(ia,f,term.la,term.lap,t)*b.value(ib,f,term.lb,term.lbp,u);
                            }
                            for (int ip=0; ip<term.ipow; ++ip) integral*=std::complex<double>(0.,1.);
                            finite(integral);
                            if (!(std::abs(integral.imag())<1e-8))
                                throw std::invalid_argument("recoupled: phased CP integral imaginary residue >= 1e-8");
                            value+=term.coefficient()*integral.real();
                            finite({value,0.});
                        }
                        pair.coefficients.push_back({n,t,u,J,value});
                    }
            }
            for (const auto& q : table) {
                if (present(sa[ia],sb[ib],q)) cov.included_rank_quadruples.push_back(q);
                else cov.missing_table_rank_quadruples.push_back(q);
            }
            // Unrestricted positive ranks sum to n-2, including ranks absent from tables.
            for (int l=1; l<=n-5; ++l) for (int p=1; p<=n-5; ++p)
                for (int k=1; k<=n-5; ++k) {
                    const int q=n-2-l-p-k;
                    if (q<1) continue;
                    const Quad ranks{{l,p,k,q}};
                    if (!present(sa[ia],sb[ib],ranks) || !table.count(ranks))
                        cov.missing_unrestricted_rank_quadruples.push_back(ranks);
                }
            cov.table_complete=cov.missing_table_rank_quadruples.empty();
            cov.unrestricted_complete=cov.missing_unrestricted_rank_quadruples.empty();
            pair.coverage.push_back(std::move(cov));
        }
        result.pairs.push_back(std::move(pair));
    }
    return result;
}
} }
