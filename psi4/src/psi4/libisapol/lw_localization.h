/* Psi4: Copyright (c) 2007-2026 The Psi4 Developers.
 * SPDX-License-Identifier: LGPL-3.0-only
 * Graph code ported from atomic_polarizability.{h,cc},
 * Copyright (c) 2007-2025 The Psi4 Developers, LGPL version 3.
 */
#ifndef PSI4_LIBISAPOL_LW_LOCALIZATION_H
#define PSI4_LIBISAPOL_LW_LOCALIZATION_H

#include <array>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>
#include "psi4/psi4-dec.h"
#include "psi4/libmints/matrix.h"

namespace psi { namespace isapol {
/// Explicit undirected, unweighted graph; zero-based sites, no geometry or bond inference.
/// Isolated sites are valid. This POD is not a response model or a localizer.
struct PSI_API IsaBondGraph {
    std::size_t site_count;
    std::vector<std::array<std::size_t, 2>> bonds;
};
/// Resource policy for dense O(N^3) validation; checked before matrix allocation.
constexpr std::size_t kIsaLwGraphMaxSites = 256;
/// Absolute maximum residuals, unscaled; tolerances are 1e-10 times component/global size.
struct PSI_API IsaLwGraphDiagnostics {
    std::size_t component_count = 0;
    double eigen_norm = 0.0, eigen_orthogonality = 0.0, eigen_residual = 0.0;
    double inverse_symmetry = 0.0;
    // A P A - A, P A P - P, A P - (A P)^T, P A - (P A)^T.
    std::array<double, 4> moore_penrose{};
};
/// Dimensionless negative Laplacian: off-diagonal +1, diagonal -degree.
/// Reject N outside [1,256], self-loops, out-of-range or duplicate/reversed edges.
PSI_API Matrix isa_lw_graph_operator(const IsaBondGraph& graph);
/// Owned matrix/eigenvalue copies; component BFS order, ascending modes within each.
/// Per-component LAPACK with |lambda| < 1e-4 omitted; fail closed on algebra checks.
/// Optional diagnostics are copied only on success. No localization/parity claim.
PSI_API std::pair<Matrix, std::vector<double>> isa_lw_graph_pseudoinverse(
    const IsaBondGraph& graph, IsaLwGraphDiagnostics* diagnostics = nullptr);
using IsaLwLocalMatrix = std::array<std::array<double, 15>, 15>;
using IsaLwWorkingMatrix = std::array<std::array<double, 16>, 16>;
using IsaLwPosition = std::array<double, 3>;
/// Supplied atomic-unit response, not a native producer or an externally refined model.
/// Explicit finite nonnegative frequency; NaN sentinel makes missing identity fail closed.
/// Positions in bohr; N*N ordered blocks: response coordinate first, potential second.
/// Real Racah order 00,10,11c,11s,... through rank 3.
struct PSI_API IsaSitePairResponse {
    double frequency = std::numeric_limits<double>::quiet_NaN();
    std::vector<IsaLwPosition> positions;
    std::vector<IsaLwWorkingMatrix> blocks;
};
struct PSI_API IsaBondTransfer {
    std::size_t first, second, first_component, second_component, fixed_site;
    double amount;
};
/// Two distinct classes of number live here, and they must not be conflated.
///
/// Algorithm-controlled (LW is responsible, always gated at residual_tolerance):
///   off_site       - off-site blocks LW is required to annihilate
///   reciprocity    - alpha(a,b;t,u) == alpha(b,a;u,t) after redistribution
///   molecular_sum  - the origin-translated molecular response LW must conserve
///   charge_sum_transport - change in the charge-flow sums sum_b alpha(a,b;t,00)
///                    and sum_b alpha(b,a;00,t) caused by localization. LW's bond
///                    transfers are antisymmetric in the two site slots, so these
///                    sums are invariants: measured at <= 2.2e-16 over the eleven
///                    recorded water frequencies.
///
/// Supplied-input quality (LW cannot influence these; gated separately, see
/// input_sum_rule_tolerance):
///   input_sum_rule - max charge-flow sum-rule defect of the SUPPLIED pair data,
///                    max over (a,t) of |sum_b alpha(a,b;t,00)| and |sum_b alpha(b,a;00,t)|.
///   charge_sum, local_charge - the same defect after localization. Because
///                    charge_sum_transport is at machine epsilon and off-site
///                    blocks are annihilated, both reproduce input_sum_rule; they
///                    are reports on the input, not measures of LW's accuracy.
struct PSI_API IsaLocalizationResiduals {
    double off_site, charge_sum, reciprocity, molecular_sum, local_charge;
    double input_sum_rule, charge_sum_transport;
};
/// Owned LW result; local drops rank 0. refined_pairs is the LW workspace,
/// NOT externally PFIT-refined data. No native-upstream or production acceptance claim.
///
/// localization_rank_limit is the rank the localization was DECLARED at (see
/// isa_localize_lw's rank_limit). Every component of rank above it is identically
/// zero in local and refined_pairs, in both index slots, because it was removed
/// from the supplied input before any transfer and never re-entered the algebra.
/// truncated_input_maxabs is the largest absolute supplied value so removed, i.e.
/// how much of the caller's own data the declared limit discarded. It is a report
/// on the declaration, not a residual, and it is deliberately NOT gated: choosing
/// a limit below the rank of the supplied data is the caller's declaration.
struct PSI_API IsaLocalizedResponse {
    double frequency;
    std::vector<IsaLwPosition> positions;
    std::vector<IsaLwLocalMatrix> local;
    std::vector<IsaBondTransfer> transfers;
    IsaLocalizationResiduals residuals;
    std::vector<IsaLwWorkingMatrix> refined_pairs;
    std::vector<std::array<std::size_t, 2>> omitted_component_pairs;
    std::size_t omitted_transfer_count;
    std::size_t localization_rank_limit = 3;
    double truncated_input_maxabs = 0.0;
};
/// Resource policy: at most 256 sites, 1,000,000 retained transfers (including pending),
/// and a conservative 768 MiB native workspace budget. Exceeding a cap throws;
/// no transfers are silently discarded. Caller-owned inputs/getter copies are additional.
constexpr std::size_t kIsaLwMaxTransfers = 1000000;
constexpr std::size_t kIsaLwMaxWorkspaceBytes = 768 * 1024 * 1024;
/// Shared preallocation guard for the POD entry point and bounded Python conversion.
PSI_API void isa_lw_validate_workspace(std::size_t site_count, std::size_t bond_count);
/// Explicit graph, no fallback or symmetrization. 1e-6 is a postcondition gate,
/// not an iterative criterion; callers may explicitly select synthetic-test tolerances.
///
/// residual_tolerance always gates the algorithm-controlled residuals above.
///
/// input_sum_rule_tolerance gates the SUPPLIED data's charge-flow sum-rule defect
/// (input_sum_rule, and equivalently the transported charge_sum/local_charge):
///   negative (default) - inherit residual_tolerance, i.e. one combined gate.
///   positive finite    - explicit separate admission threshold.
///   infinity           - measure and report the defect without gating it. Required
///                        to process supplied pair data whose sum rule is only
///                        approximately satisfied, which is the case for every
///                        recorded reference file: LW transports such a defect
///                        exactly and cannot repair it, so rejecting on it gates
///                        the producer of the input, not this routine.
/// Reporting the defect is not waiving it: the caller receives the measured value
/// and every algorithm-controlled residual is still held to residual_tolerance.
///
/// rank_limit DECLARES the rank the localization is performed at, in 1..3, and
/// defaults to 3, the full working space, for which this routine is unchanged.
/// A declared limit L restricts the whole algorithm to the first (L+1)^2 real
/// Racah components: the supplied blocks are truncated to that range in both
/// index slots first (reported through truncated_input_maxabs), and the
/// component-pair loop, the translated transfer application, the molecular-sum
/// conservation check and the local output all run inside it.
///
/// That restriction is exact, not approximate, and relaxes no tolerance. The
/// multipole translation matrix is rank-raising -- its (row, column) entry
/// vanishes unless rank(row) >= rank(column) -- so on the leading (L+1)^2 index
/// range the composition T_first * T(-d) = T_second still holds identically.
/// Off-site annihilation, reciprocity, charge-sum transport and the molecular
/// sum are therefore each conserved within the declared space, and every
/// residual above is gated at exactly the same threshold as at rank 3.
///
/// A limited localization is a DIFFERENT MODEL from the rank-3 one and the two
/// must never be quoted as agreeing. They are, however, exactly CONSISTENT, and
/// that is a theorem about this algorithm rather than a numerical observation:
///
///   For any declared L, the localized blocks equal the rank-3 localized blocks
///   restricted to the leading (L+1)^2 components, bitwise.
///
/// The component-pair loop is ordered first_component <= second_component, and a
/// transfer for the pair (t, u) writes only into the u-th slot, with the target
/// weight delta(target, t) + T(+-d)[target][t]. Translation is rank-raising, so
/// that weight vanishes for rank(target) < rank(t); a pair with t outside the
/// declared space therefore writes only outside it, and t <= u puts u outside
/// too. No higher-rank pair can reach a component below the limit, and the
/// screening decisions for the lower pairs read only lower components, so they
/// are unchanged as well. Measured at 0.0 over the eleven recorded water
/// frequencies and over random reciprocal input on unrelated graphs.
///
/// The practical consequence is a negative one and must be reported as such: a
/// declared localization rank limit cannot change any rank <= L observable, so
/// it cannot explain a disagreement in one. What the limit does buy is an
/// honest, cheaper model whose higher components are absent by declaration
/// rather than dropped afterwards, and a limit a consumer can check against.
PSI_API IsaLocalizedResponse isa_localize_lw(const IsaSitePairResponse& response,
    const IsaBondGraph& graph, double residual_tolerance = 1.0e-6,
    double input_sum_rule_tolerance = -1.0, int rank_limit = 3);
} }
#endif
