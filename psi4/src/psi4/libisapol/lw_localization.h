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
/// Optional diagnostics are copied only on success.
PSI_API std::pair<Matrix, std::vector<double>> isa_lw_graph_pseudoinverse(
    const IsaBondGraph& graph, IsaLwGraphDiagnostics* diagnostics = nullptr);
/// Storage widths, not the declared rank: real Racah components through rank 4,
/// (4+1)^2 = 25 working and 24 local (rank 0 dropped). Components above the
/// declared rank_limit are identically zero, so rank 3 uses the leading 16/15.
constexpr std::size_t kIsaLwMaxRank = 4;
constexpr std::size_t kIsaLwWorkingComponents = (kIsaLwMaxRank + 1) * (kIsaLwMaxRank + 1);
constexpr std::size_t kIsaLwLocalComponents = kIsaLwWorkingComponents - 1;
using IsaLwLocalMatrix = std::array<std::array<double, kIsaLwLocalComponents>, kIsaLwLocalComponents>;
using IsaLwWorkingMatrix = std::array<std::array<double, kIsaLwWorkingComponents>, kIsaLwWorkingComponents>;
using IsaLwPosition = std::array<double, 3>;
/// Supplied atomic-unit response. Explicit finite nonnegative frequency; the NaN
/// sentinel makes a missing frequency fail closed.
/// Positions in bohr; N*N ordered blocks: response coordinate first, potential second.
/// Real Racah order 00,10,11c,11s,... through rank 4.
struct PSI_API IsaSitePairResponse {
    double frequency = std::numeric_limits<double>::quiet_NaN();
    std::vector<IsaLwPosition> positions;
    std::vector<IsaLwWorkingMatrix> blocks;
};
struct PSI_API IsaBondTransfer {
    std::size_t first, second, first_component, second_component, fixed_site;
    double amount;
};
/// Algorithm-controlled, always gated at residual_tolerance:
///   off_site       - off-site blocks LW must annihilate
///   reciprocity    - alpha(a,b;t,u) == alpha(b,a;u,t) after redistribution
///   molecular_sum  - the origin-translated molecular response LW must conserve
///   charge_sum_transport - change in sum_b alpha(a,b;t,00) and sum_b alpha(b,a;00,t)
///                    caused by localization; an invariant, since bond transfers
///                    are antisymmetric in the two site slots.
///
/// Supplied-input quality, gated by input_sum_rule_tolerance:
///   input_sum_rule - max over (a,t) of |sum_b alpha(a,b;t,00)| and
///                    |sum_b alpha(b,a;00,t)| in the SUPPLIED pair data.
///   charge_sum, local_charge - the same defect after localization; they
///                    reproduce input_sum_rule and do not measure LW accuracy.
struct PSI_API IsaLocalizationResiduals {
    double off_site, charge_sum, reciprocity, molecular_sum, local_charge;
    double input_sum_rule, charge_sum_transport;
};
/// Owned LW result; local drops rank 0. refined_pairs is the LW workspace,
/// NOT externally PFIT-refined data.
///
/// localization_rank_limit is the declared rank_limit; components above it are
/// identically zero in local and refined_pairs, in both index slots.
/// truncated_input_maxabs is the largest supplied value that limit discarded. It
/// is reported, not gated: the limit is the caller's declaration.
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
/// The 25x25 working blocks cost about 20128*N^2 bytes, so the budget admits about
/// 174 sites, below the graph cap. Do not raise the budget to recover them.
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
/// rank_limit (1..4, default 3) declares the localization rank L. Supplied
/// blocks are truncated to the first (L+1)^2 real Racah components in both slots
/// before any transfer, and the whole algorithm then runs inside that range.
/// This is exact: translation is rank-raising (entry (row, column) vanishes
/// unless rank(row) >= rank(column)), so T_first * T(-d) = T_second holds on the
/// leading range and every residual keeps the same gate.
///
/// Different limits are different models, but consistent: for L < L' the blocks
/// at L equal those at L' restricted to (L+1)^2 components, bitwise. A transfer
/// for component pair (t <= u) writes only slot u with weight
/// delta(target, t) + T(+-d)[target][t], which vanishes for rank(target) < rank(t).
/// Hence a limit cannot change any rank <= L observable.
PSI_API IsaLocalizedResponse isa_localize_lw(const IsaSitePairResponse& response,
    const IsaBondGraph& graph, double residual_tolerance = 1.0e-6,
    double input_sum_rule_tolerance = -1.0, int rank_limit = 3);
} }
#endif
