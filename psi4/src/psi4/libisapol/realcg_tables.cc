/* CamCASP numerical data: Alston J. Misquitta and Anthony J. Stone.
 * Copyright (c) 2019 Anthony Stone. MIT: see RECOUPLED_CAMCASP_LICENSE.
 * Psi4 additions: Copyright (c) 2026 The Psi4 Developers. LGPL-3.0-only.
 */
#include "realcg_tables.h"
#include <cmath>
#include <stdexcept>
namespace psi { namespace isapol {
namespace realcg_detail {
const RealCGTerm records[] = {
#include "realcg_data.inc"
};
}
std::complex<double> RealCGTerm::value() const {
    const double x = (double(p) / denominator) * std::sqrt(double(std::abs(r)) / s);
    return r > 0 ? std::complex<double>(x, 0.) : std::complex<double>(0., x);
}
bool realcg_defined(int la, int lap) {
    return la >= 1 && la <= 4 && lap >= 1 && lap <= 4 && la + lap <= 6;
}
std::vector<RealCGTerm> realcg_terms(int la, int lap) {
    if (!realcg_defined(la, lap))
        throw std::invalid_argument("realcg: ordered pair undefined upstream; ranks 1..4 with "
                                    "la+lap<=6 only (casimir.f90 skips j1+j2>6)");
    std::vector<RealCGTerm> out;
    for (const auto& t : realcg_detail::records)
        if (t.la == la && t.lap == lap) out.push_back(t);
    return out;
}
} }
