#ifndef PSI4_CPSCF_EXCHANGE_H
#define PSI4_CPSCF_EXCHANGE_H

#include <algorithm>
#include <cmath>
#include "psi4/libfock/jk.h"
#include "psi4/libfunctional/superfunctional.h"
#include "psi4/libpsi4util/exception.h"

namespace psi {
namespace scf {
namespace cpscf {

/// Convert a preweighted combined K to the functional's response operator.
/// SAPT lends global-hybrid CPKS a bare-K builder: scale its result, do not
/// mutate/rebuild the shared plan (later SAPT exchange still needs bare K).
inline double exchange_scale(JK& jk, SuperFunctional& functional) {
    const double alpha = functional.x_alpha();
    const double beta = functional.x_omega() == 0.0 ? 0.0 : functional.x_beta();
    const double plan_alpha = jk.get_omega_alpha();
    const double plan_beta = jk.get_omega() == 0.0 ? 0.0 : jk.get_omega_beta();
    const double scale = plan_alpha != 0.0 ? alpha / plan_alpha :
                         plan_beta != 0.0 ? beta / plan_beta : 0.0;
    const auto same = [](double a, double b) {
        return std::abs(a - b) <= 1.e-14 + 1.e-12 * std::max(std::abs(a), std::abs(b));
    };
    // Different full-/long-range mixtures cannot be recovered from one K.
    if (!same(scale * plan_alpha, alpha) || !same(scale * plan_beta, beta) ||
        (beta != 0.0 && !same(jk.get_omega(), functional.x_omega()))) {
        throw PSIEXCEPTION("CPSCF: the combined-exchange JK plan is incompatible with the functional. "
                           "Use a response JK initialized with its x_alpha, x_beta, and x_omega; "
                           "a different range-separated mixture cannot be recovered from one K.");
    }
    return scale;
}

}  // namespace cpscf
}  // namespace scf
}  // namespace psi
#endif
