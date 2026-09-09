/* Numerical conventions/data: CamCASP, Alston J. Misquitta and Anthony J. Stone.
 * Copyright (c) 2019 Anthony Stone. MIT: see RECOUPLED_CAMCASP_LICENSE.
 * Psi4 additions: Copyright (c) 2026 The Psi4 Developers. LGPL-3.0-only.
 */
#ifndef PSI4_LIBISAPOL_REALCG_TABLES_H
#define PSI4_LIBISAPOL_REALCG_TABLES_H
#include <complex>
#include <vector>
namespace psi { namespace isapol {
/// k,q are rank-local zero-based; v is global coupled zero-based.
struct RealCGTerm {
    int la, lap, k, q, v, p, denominator, r, s;
    std::complex<double> value() const;
};
/// True for the ordered rank pairs upstream actually defines: 1<=la,lap<=4 with
/// la+lap<=6.  casimir.f90 read_cg and recouple both execute
/// "if (j1+j2>6) cycle", so realcg_3_4/4_3/4_4 are never read and alpha_c is
/// left uninitialized for (3,4),(4,3),(4,4).  Those three pairs are therefore
/// structurally absent here, never silently zero-valued data.
bool realcg_defined(int la, int lap);
/// Shipped numerical records, sorted by (la,lap,v,k,q) for deterministic sums.
/// Throws unless realcg_defined(la,lap).
std::vector<RealCGTerm> realcg_terms(int la, int lap);
} }
#endif
