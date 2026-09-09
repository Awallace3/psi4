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
/// Shipped numerical records, sorted by (la,lap,v,k,q) for deterministic sums.
/// Only declared ranks 1..3; rank four is explicitly rejected.
std::vector<RealCGTerm> realcg_terms(int la, int lap);
} }
#endif
