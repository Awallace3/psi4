/* Psi4: Copyright (c) 2007-2026 The Psi4 Developers.
 * SPDX-License-Identifier: LGPL-3.0-only
 * Graph code ported from atomic_polarizability.cc,
 * Copyright (c) 2007-2025 The Psi4 Developers, LGPL version 3.
 */
#include "lw_localization.h"
#include "multipole_transform.h"
#include <sstream>
#include "psi4/libmints/vector.h"
#include "psi4/libpsi4util/exception.h"
#include <algorithm>
#include <cmath>
#include <queue>
#include <string>

namespace psi { namespace isapol {
namespace { namespace lw_graph_private {
using DenseMatrix = std::vector<std::vector<double>>;
constexpr double kGraphEigenvalueCutoff = 1.0e-4;
constexpr double kLinearAlgebraTolerance = 1.0e-10;
// Dense helpers below are private: every argument is a nonempty N x N
// matrix created here after validating N <= kIsaLwGraphMaxSites.
void require_finite(double value, const char* context) {
    if (!std::isfinite(value)) throw PSIEXCEPTION(std::string(context) + ": derived value is not finite");
}


double finite_absolute(double value, const char* context) {
    require_finite(value, context);
    const double result = std::abs(value);
    require_finite(result, context);
    return result;
}


DenseMatrix make_graph_operator(const IsaBondGraph& graph) {
    if (graph.site_count == 0) throw PSIEXCEPTION("isa_lw_graph: bond graph must contain at least one site");
    if (graph.site_count > kIsaLwGraphMaxSites)
        throw PSIEXCEPTION("isa_lw_graph: site_count exceeds dense graph limit (256)");
    const auto max_bonds = graph.site_count * (graph.site_count - 1) / 2;
    if (graph.bonds.size() > max_bonds)
        throw PSIEXCEPTION("isa_lw_graph: bond count exceeds simple graph capacity");
    DenseMatrix result(graph.site_count, std::vector<double>(graph.site_count, 0.0));
    for (const auto& bond : graph.bonds) {
        const std::size_t a = bond[0], b = bond[1];
        if (a >= graph.site_count || b >= graph.site_count || a == b)
            throw PSIEXCEPTION("isa_lw_graph: bond graph contains an invalid bond");
        if (result[a][b] != 0.0) throw PSIEXCEPTION("isa_lw_graph: bond graph contains a duplicate bond");
        result[a][b] = result[b][a] = 1.0;
        result[a][a] -= 1.0;
        result[b][b] -= 1.0;
    }
    return result;
}

std::vector<std::vector<std::size_t>> graph_components(const DenseMatrix& graph_operator) {
    std::vector<std::vector<std::size_t>> components;
    std::vector<bool> visited(graph_operator.size(), false);
    for (std::size_t root = 0; root < graph_operator.size(); ++root) {
        if (visited[root]) continue;
        components.emplace_back();
        std::queue<std::size_t> pending;
        visited[root] = true;
        pending.push(root);
        while (!pending.empty()) {
            const auto current = pending.front();
            pending.pop();
            components.back().push_back(current);
            for (std::size_t next = 0; next < graph_operator.size(); ++next) {
                if (next != current && graph_operator[current][next] != 0.0 && !visited[next]) {
                    visited[next] = true;
                    pending.push(next);
                }
            }
        }
    }
    return components;
}

DenseMatrix dense_multiply(const DenseMatrix& first, const DenseMatrix& second) {
    DenseMatrix result(first.size(), std::vector<double>(second[0].size(), 0.0));
    for (std::size_t row = 0; row < first.size(); ++row) {
        for (std::size_t column = 0; column < second[0].size(); ++column) {
            for (std::size_t k = 0; k < second.size(); ++k) {
                const double term = first[row][k] * second[k][column];
                require_finite(term, "graph linear algebra product");
                result[row][column] += term;
                require_finite(result[row][column], "graph linear algebra accumulation");
            }
        }
    }
    return result;
}

DenseMatrix dense_transpose(const DenseMatrix& matrix) {
    DenseMatrix result(matrix[0].size(), std::vector<double>(matrix.size(), 0.0));
    for (std::size_t row = 0; row < matrix.size(); ++row) {
        for (std::size_t column = 0; column < matrix[row].size(); ++column) {
            result[column][row] = matrix[row][column];
        }
    }
    return result;
}

double dense_max_difference(const DenseMatrix& first, const DenseMatrix& second, const char* context) {
    double result = 0.0;
    for (std::size_t row = 0; row < first.size(); ++row) {
        for (std::size_t column = 0; column < first[row].size(); ++column) {
            const double difference = first[row][column] - second[row][column];
            result = std::max(result, finite_absolute(difference, context));
        }
    }
    return result;
}

DenseMatrix graph_pseudoinverse(const DenseMatrix& graph_operator, std::vector<double>* eigenvalues_out,
                                IsaLwGraphDiagnostics& diagnostics) {
    const auto components = graph_components(graph_operator);
    diagnostics.component_count = components.size();
    DenseMatrix result(graph_operator.size(), std::vector<double>(graph_operator.size(), 0.0));
    std::vector<double> all_eigenvalues;
    for (const auto& component : components) {
        const std::size_t count = component.size();
        Matrix block(count, count);
        for (std::size_t row = 0; row < count; ++row) {
            for (std::size_t column = 0; column < count; ++column) {
                block(row, column) = graph_operator[component[row]][component[column]];
            }
        }
        Matrix eigenvectors(count, count);
        Vector eigenvalues(count);
        block.diagonalize(eigenvectors, eigenvalues, ascending);
        const double validation_tolerance = kLinearAlgebraTolerance * std::max<std::size_t>(1, count);
        for (std::size_t mode = 0; mode < count; ++mode) {
            const double eigenvalue = eigenvalues(mode);
            require_finite(eigenvalue, "graph eigenvalue");
            all_eigenvalues.push_back(eigenvalue);
            double norm = 0.0;
            for (std::size_t row = 0; row < count; ++row) {
                require_finite(eigenvectors(row, mode), "graph eigenvector");
                norm += eigenvectors(row, mode) * eigenvectors(row, mode);
            }
            require_finite(norm, "graph eigenvector norm");
            diagnostics.eigen_norm = std::max(diagnostics.eigen_norm, std::abs(norm - 1.0));
            if (std::abs(norm - 1.0) > validation_tolerance) {
                throw PSIEXCEPTION("isa_lw_graph: graph eigensolver returned non-orthonormal vectors");
            }
            for (std::size_t other = 0; other < mode; ++other) {
                double dot = 0.0;
                for (std::size_t row = 0; row < count; ++row) {
                    dot += eigenvectors(row, mode) * eigenvectors(row, other);
                }
                require_finite(dot, "graph eigenvector orthogonality");
                diagnostics.eigen_orthogonality = std::max(diagnostics.eigen_orthogonality, std::abs(dot));
                if (std::abs(dot) > validation_tolerance) {
                    throw PSIEXCEPTION("isa_lw_graph: graph eigensolver returned non-orthogonal vectors");
                }
            }
            for (std::size_t row = 0; row < count; ++row) {
                double residual = -eigenvalue * eigenvectors(row, mode);
                for (std::size_t column = 0; column < count; ++column) {
                    residual += graph_operator[component[row]][component[column]] *
                                eigenvectors(column, mode);
                }
                diagnostics.eigen_residual = std::max(diagnostics.eigen_residual,
                    finite_absolute(residual, "graph eigen residual"));
                if (finite_absolute(residual, "graph eigen residual") > validation_tolerance) {
                    throw PSIEXCEPTION("isa_lw_graph: graph eigensolver residual exceeds tolerance");
                }
            }
            if (std::abs(eigenvalue) < kGraphEigenvalueCutoff) continue;
            for (std::size_t row = 0; row < count; ++row) {
                for (std::size_t column = 0; column < count; ++column) {
                    const double contribution = eigenvectors(row, mode) * eigenvectors(column, mode) / eigenvalue;
                    require_finite(contribution, "graph pseudoinverse contribution");
                    result[component[row]][component[column]] += contribution;
                    require_finite(result[component[row]][component[column]],
                                   "graph pseudoinverse accumulation");
                }
            }
        }
    }

    const double validation_tolerance =
        kLinearAlgebraTolerance * std::max<std::size_t>(1, graph_operator.size());
    for (std::size_t row = 0; row < result.size(); ++row) {
        for (std::size_t column = 0; column < result.size(); ++column) {
            require_finite(result[row][column], "graph pseudoinverse");
            diagnostics.inverse_symmetry = std::max(diagnostics.inverse_symmetry,
                finite_absolute(result[row][column] - result[column][row], "graph inverse symmetry"));
            if (std::abs(result[row][column] - result[column][row]) > validation_tolerance) {
                throw PSIEXCEPTION("isa_lw_graph: graph pseudoinverse is not symmetric");
            }
        }
    }
    const auto operator_inverse = dense_multiply(graph_operator, result);
    const auto inverse_operator = dense_multiply(result, graph_operator);
    // Evaluate all four identities, even when an earlier identity fails.
    diagnostics.moore_penrose = {{
        dense_max_difference(dense_multiply(operator_inverse, graph_operator), graph_operator,
                             "graph Moore-Penrose residual"),
        dense_max_difference(dense_multiply(inverse_operator, result), result,
                             "graph Moore-Penrose residual"),
        dense_max_difference(operator_inverse, dense_transpose(operator_inverse), "graph projector symmetry"),
        dense_max_difference(inverse_operator, dense_transpose(inverse_operator), "graph projector symmetry")
    }};
    for (double residual : diagnostics.moore_penrose) {
        if (residual > validation_tolerance)
            throw PSIEXCEPTION("isa_lw_graph: graph pseudoinverse failed Moore-Penrose validation");
    }
    if (eigenvalues_out) *eigenvalues_out = std::move(all_eigenvalues);
    return result;
}

Matrix to_psi_matrix(const DenseMatrix& values) {
    Matrix result(values.size(), values.size());
    for (std::size_t row = 0; row < values.size(); ++row)
        for (std::size_t column = 0; column < values.size(); ++column) result(row, column) = values[row][column];
    return result;
}

} }  // anonymous / lw_graph_private

Matrix isa_lw_graph_operator(const IsaBondGraph& graph) {
    return lw_graph_private::to_psi_matrix(lw_graph_private::make_graph_operator(graph));
}

std::pair<Matrix, std::vector<double>> isa_lw_graph_pseudoinverse(
    const IsaBondGraph& graph, IsaLwGraphDiagnostics* diagnostics) {
    const auto graph_operator = lw_graph_private::make_graph_operator(graph);
    std::vector<double> eigenvalues;
    IsaLwGraphDiagnostics checked;
    auto inverse = lw_graph_private::graph_pseudoinverse(graph_operator, &eigenvalues, checked);
    auto matrix = lw_graph_private::to_psi_matrix(inverse);
    if (diagnostics) *diagnostics = checked;
    return {std::move(matrix), std::move(eigenvalues)};
}

namespace { namespace lw_transfer_private {
using lw_graph_private::require_finite;
using lw_graph_private::finite_absolute;
using lw_graph_private::make_graph_operator;
using lw_graph_private::graph_components;
using lw_graph_private::graph_pseudoinverse;
using lw_graph_private::kLinearAlgebraTolerance;
constexpr double kElementTransferThreshold = 1.0e-7;

// The sole translation seam, including molecular origin shifts.
IsaLwWorkingMatrix translation_matrix(const IsaLwPosition& displacement) {
    for (double value : displacement) require_finite(value, "translation displacement");
    const auto matrix = isa_multipole_translation(3, displacement);
    IsaLwWorkingMatrix result{};
    for (std::size_t row = 0; row < 16; ++row)
        for (std::size_t column = 0; column < 16; ++column) {
            result[row][column] = (*matrix)(row, column);
            require_finite(result[row][column], "translation matrix");
        }
    return result;
}

// `components` is the declared working width (L+1)^2. Both the molecular index
// range and the site-local one are restricted to it: translation is rank-raising,
// so an unrestricted molecular row would receive contributions from inside the
// declared space that the truncated transfer application never balances, and the
// conservation law would then be measured on a quantity the algorithm does not own.
IsaLwWorkingMatrix molecular_response(const IsaSitePairResponse& response, std::size_t components) {
    const std::size_t count = response.positions.size();
    std::vector<IsaLwWorkingMatrix> translations;
    for (const auto& position : response.positions) translations.push_back(translation_matrix(position));
    IsaLwWorkingMatrix result{};
    for (std::size_t a = 0; a < count; ++a) for (std::size_t b = 0; b < count; ++b) {
        const auto& block = response.blocks[a * count + b];
        for (std::size_t row = 0; row < components; ++row) for (std::size_t column = 0; column < components; ++column)
            for (std::size_t local_row = 0; local_row < components; ++local_row)
                for (std::size_t local_column = 0; local_column < components; ++local_column) {
                    const double term = translations[a][row][local_row] * block[local_row][local_column] *
                                        translations[b][column][local_column];
                    require_finite(term, "molecular response product");
                    result[row][column] += term;
                    require_finite(result[row][column], "molecular response accumulation");
                }
    }
    return result;
}

double matrix_max_difference(const IsaLwWorkingMatrix& first, const IsaLwWorkingMatrix& second,
                             std::size_t components) {
    double result = 0.0;
    for (std::size_t row = 0; row < components; ++row) for (std::size_t column = 0; column < components; ++column)
        result = std::max(result, finite_absolute(first[row][column] - second[row][column],
                                                   "molecular response residual"));
    return result;
}

// `before` must already be the truncated input: every residual here is a
// before/after comparison, and the algorithm's input is what it was given.
IsaLocalizationResiduals localization_residuals(const IsaSitePairResponse& before,
                                                const IsaSitePairResponse& after, std::size_t components) {
    const std::size_t count = after.positions.size();
    IsaLocalizationResiduals residuals{};
    for (std::size_t a = 0; a < count; ++a) {
        const auto& local = after.blocks[a * count + a];
        for (std::size_t component = 0; component < components; ++component) {
            residuals.local_charge = std::max(
                residuals.local_charge,
                std::max(finite_absolute(local[0][component], "local charge residual"),
                         finite_absolute(local[component][0], "local charge residual")));
            double first_sum = 0.0, second_sum = 0.0;
            double first_input = 0.0, second_input = 0.0;
            for (std::size_t b = 0; b < count; ++b) {
                first_sum += after.blocks[a * count + b][component][0];
                second_sum += after.blocks[b * count + a][0][component];
                require_finite(first_sum, "charge sum residual");
                require_finite(second_sum, "charge sum residual");
                first_input += before.blocks[a * count + b][component][0];
                second_input += before.blocks[b * count + a][0][component];
                require_finite(first_input, "input charge sum-rule defect");
                require_finite(second_input, "input charge sum-rule defect");
            }
            residuals.charge_sum = std::max(
                residuals.charge_sum,
                std::max(finite_absolute(first_sum, "charge sum residual"),
                         finite_absolute(second_sum, "charge sum residual")));
            // The supplied data's own sum-rule defect, measured before any transfer.
            residuals.input_sum_rule = std::max(
                residuals.input_sum_rule,
                std::max(finite_absolute(first_input, "input charge sum-rule defect"),
                         finite_absolute(second_input, "input charge sum-rule defect")));
            // LW's bond transfers are antisymmetric in the site slots, so these sums
            // are invariants. This is the algorithm-controlled part of charge_sum.
            residuals.charge_sum_transport = std::max(
                residuals.charge_sum_transport,
                std::max(finite_absolute(first_sum - first_input, "charge sum transport"),
                         finite_absolute(second_sum - second_input, "charge sum transport")));
        }
        for (std::size_t b = 0; b < count; ++b) {
            const auto& block = after.blocks[a * count + b];
            const auto& reciprocal = after.blocks[b * count + a];
            for (std::size_t row = 0; row < components; ++row)
                for (std::size_t column = 0; column < components; ++column) {
                if (a != b) {
                    residuals.off_site = std::max(
                        residuals.off_site,
                        finite_absolute(block[row][column], "off-site residual"));
                }
                residuals.reciprocity = std::max(
                    residuals.reciprocity,
                    finite_absolute(block[row][column] - reciprocal[column][row],
                                    "reciprocity residual"));
            }
        }
    }
    residuals.molecular_sum = matrix_max_difference(molecular_response(before, components),
                                                    molecular_response(after, components), components);
    return residuals;
}

} }  // anonymous / lw_transfer_private

void isa_lw_validate_workspace(std::size_t count, std::size_t edges) {
    // Bound factors before any products: these products cannot overflow size_t.
    if (count == 0 || count > kIsaLwGraphMaxSites)
        throw PSIEXCEPTION("localize_lw: site count must be in 1..256 (resource limit)");
    if (edges > count * (count - 1) / 2)
        throw PSIEXCEPTION("localize_lw: bond count exceeds simple graph capacity");
    // Input conversion + refined + result blocks, edge/origin translations, local
    // output, generous dense graph temporaries, and geometrically growing queues.
    const std::size_t bytes = 3 * count * count * sizeof(IsaLwWorkingMatrix) +
        (2 * edges + count) * sizeof(IsaLwWorkingMatrix) +
        count * (sizeof(IsaLwLocalMatrix) + sizeof(IsaLwPosition)) +
        16 * count * count * sizeof(double) +
        4 * kIsaLwMaxTransfers * sizeof(IsaBondTransfer);
    if (bytes > kIsaLwMaxWorkspaceBytes)
        throw PSIEXCEPTION("localize_lw: native workspace exceeds 768 MiB resource limit");
}

IsaLocalizedResponse isa_localize_lw(const IsaSitePairResponse& response, const IsaBondGraph& graph,
                              double residual_tolerance, double input_sum_rule_tolerance, int rank_limit) {
    using namespace lw_transfer_private;
    if (rank_limit < 1 || rank_limit > 3)
        throw PSIEXCEPTION(
            "localize_lw: declared rank_limit must be 1, 2 or 3; rank 4 needs a rank-4 working matrix");
    // The declared working width. rank_limit 3 gives 16, the historical behaviour.
    const std::size_t working_components = static_cast<std::size_t>((rank_limit + 1) * (rank_limit + 1));
    if (!std::isfinite(response.frequency))
        throw PSIEXCEPTION("localize_lw: response frequency must be finite");
    if (response.frequency < 0.0)
        throw PSIEXCEPTION("localize_lw: response frequency must be nonnegative");
    if (!std::isfinite(residual_tolerance) || residual_tolerance <= 0.0) {
        throw PSIEXCEPTION("localize_lw: residual tolerance must be finite and positive");
    }
    if (std::isnan(input_sum_rule_tolerance))
        throw PSIEXCEPTION("localize_lw: input sum-rule tolerance must not be NaN");
    if (input_sum_rule_tolerance == 0.0)
        throw PSIEXCEPTION("localize_lw: input sum-rule tolerance must be positive, or negative to inherit");
    const std::size_t count = response.positions.size();
    isa_lw_validate_workspace(count, graph.bonds.size());
    if (count == 0 || graph.site_count != count || response.blocks.size() != count * count) {
        throw PSIEXCEPTION("localize_lw: inconsistent site-pair response and bond graph dimensions");
    }
    for (const auto& position : response.positions) {
        for (double value : position) {
            if (!std::isfinite(value)) throw PSIEXCEPTION("localize_lw: site positions must be finite");
        }
    }
    for (const auto& block : response.blocks) {
        for (const auto& row : block) {
            for (double value : row) {
                if (!std::isfinite(value)) throw PSIEXCEPTION("localize_lw: response values must be finite");
            }
        }
    }

    const auto graph_operator = make_graph_operator(graph);
    const auto components = graph_components(graph_operator);
    IsaLwGraphDiagnostics graph_diagnostics;
    const auto pseudoinverse = graph_pseudoinverse(graph_operator, nullptr, graph_diagnostics);

    double input_reciprocity = 0.0;
    for (std::size_t a = 0; a < count; ++a) {
        for (std::size_t b = 0; b < count; ++b) {
            for (std::size_t row = 0; row < 16; ++row) {
                for (std::size_t column = 0; column < 16; ++column) {
                    input_reciprocity = std::max(
                        input_reciprocity,
                        finite_absolute(response.blocks[a * count + b][row][column] -
                                            response.blocks[b * count + a][column][row],
                                        "input reciprocity"));
                }
            }
        }
    }
    if (input_reciprocity > residual_tolerance) {
        throw PSIEXCEPTION("localize_lw: input reciprocity exceeds residual tolerance");
    }

    // Truncate to the declared space BEFORE any transfer, exactly as the reference
    // protocol does (`Limit all rank {LIMIT}` precedes `Localise ... Limit {LIMIT}`).
    // Truncation is symmetric in the two index slots, so the reciprocity just
    // checked over the full space is inherited by the truncated data.
    IsaSitePairResponse truncated = response;
    double truncated_input_maxabs = 0.0;
    if (working_components < 16) {
        for (auto& block : truncated.blocks) {
            for (std::size_t row = 0; row < 16; ++row) {
                for (std::size_t column = 0; column < 16; ++column) {
                    if (row < working_components && column < working_components) continue;
                    truncated_input_maxabs = std::max(
                        truncated_input_maxabs,
                        finite_absolute(block[row][column], "truncated supplied component"));
                    block[row][column] = 0.0;
                }
            }
        }
    }

    IsaSitePairResponse refined = truncated;
    std::vector<IsaLwWorkingMatrix> positive_translations;
    std::vector<IsaLwWorkingMatrix> negative_translations;
    positive_translations.reserve(graph.bonds.size());
    negative_translations.reserve(graph.bonds.size());
    for (const auto& bond : graph.bonds) {
        IsaLwPosition displacement{};
        for (std::size_t axis = 0; axis < 3; ++axis) {
            displacement[axis] = response.positions[bond[0]][axis] - response.positions[bond[1]][axis];
            require_finite(displacement[axis], "bond displacement");
        }
        positive_translations.push_back(translation_matrix(displacement));
        for (double& value : displacement) value = -value;
        negative_translations.push_back(translation_matrix(displacement));
    }

    struct PendingTransfer {
        std::size_t edge;
        std::size_t fixed_site;
        double amount;
    };
    IsaLocalizedResponse result{};
    result.frequency = response.frequency;
    result.positions = response.positions;
    result.localization_rank_limit = static_cast<std::size_t>(rank_limit);
    result.truncated_input_maxabs = truncated_input_maxabs;
    for (std::size_t first_component = 0; first_component < working_components; ++first_component) {
        for (std::size_t second_component = first_component; second_component < working_components; ++second_component) {
            double largest_candidate = 0.0;
            for (std::size_t a = 0; a < count; ++a) {
                for (std::size_t b = 0; b < count; ++b) {
                    if (a == b) continue;
                    largest_candidate = std::max(
                        largest_candidate,
                        finite_absolute(refined.blocks[a * count + b][first_component][second_component],
                                        "candidate pair magnitude"));
                }
            }
            if (largest_candidate < kElementTransferThreshold) {
                result.omitted_component_pairs.push_back({first_component, second_component});
                continue;
            }
            std::vector<PendingTransfer> pending;
            const double symmetry_factor = first_component == second_component ? 0.5 : 1.0;
            for (std::size_t fixed_site = 0; fixed_site < count; ++fixed_site) {
                std::vector<double> unwanted(count, 0.0);
                double offsite_sum = 0.0;
                for (std::size_t site = 0; site < count; ++site) {
                    if (site == fixed_site) continue;
                    unwanted[site] = symmetry_factor *
                                     refined.blocks[site * count + fixed_site][first_component][second_component];
                    require_finite(unwanted[site], "balanced unwanted response");
                    offsite_sum += unwanted[site];
                    require_finite(offsite_sum, "balanced unwanted response sum");
                }
                unwanted[fixed_site] = -offsite_sum;
                require_finite(unwanted[fixed_site], "balanced unwanted response diagonal");
                for (const auto& component : components) {
                    double component_sum = 0.0;
                    double component_scale = 1.0;
                    for (std::size_t site : component) {
                        component_sum += unwanted[site];
                        require_finite(component_sum, "component unwanted response sum");
                        component_scale = std::max(
                            component_scale,
                            finite_absolute(unwanted[site], "component unwanted response scale"));
                    }
                    const double component_tolerance = kLinearAlgebraTolerance * component_scale;
                    require_finite(component_tolerance, "component unwanted response tolerance");
                    if (std::abs(component_sum) > component_tolerance) {
                        throw PSIEXCEPTION(
                            "localize_lw: graph component unwanted response does not have zero sum");
                    }
                }

                std::vector<double> potential(count, 0.0);
                for (std::size_t row = 0; row < count; ++row) {
                    for (std::size_t column = 0; column < count; ++column) {
                        const double term = pseudoinverse[row][column] * unwanted[column];
                        require_finite(term, "graph potential product");
                        potential[row] += term;
                        require_finite(potential[row], "graph potential accumulation");
                    }
                }
                double range_residual = 0.0;
                for (std::size_t row = 0; row < count; ++row) {
                    double projected = 0.0;
                    for (std::size_t column = 0; column < count; ++column) {
                        projected += graph_operator[row][column] * potential[column];
                        require_finite(projected, "graph range projection");
                    }
                    range_residual = std::max(
                        range_residual,
                        finite_absolute(projected - unwanted[row], "graph solve residual"));
                }
                if (range_residual > residual_tolerance) {
                    throw PSIEXCEPTION("localize_lw: graph solve exceeds residual tolerance");
                }
                for (std::size_t edge = 0; edge < graph.bonds.size(); ++edge) {
                    const auto& bond = graph.bonds[edge];
                    const double amount = 0.5 * (potential[bond[1]] - potential[bond[0]]);
                    require_finite(amount, "bond transfer amount");
                    if (std::abs(amount) <= kElementTransferThreshold) {
                        ++result.omitted_transfer_count;
                    } else {
                        if (result.transfers.size() + pending.size() >= kIsaLwMaxTransfers)
                            throw PSIEXCEPTION("localize_lw: retained transfer resource limit (1000000) exceeded");
                        pending.push_back({edge, fixed_site, amount});
                    }
                }
            }

            for (const auto& transfer : pending) {
                const auto& bond = graph.bonds[transfer.edge];
                const std::size_t first = bond[0];
                const std::size_t second = bond[1];
                const std::size_t fixed = transfer.fixed_site;
                const double amount = transfer.amount;
                // Restricted to the declared space with the same exactness: the
                // translation entries reached here have rank(target) <= rank(first_component)
                // whenever target is dropped, and those are structurally zero.
                for (std::size_t target = 0; target < working_components; ++target) {
                    const double at_first = (target == first_component ? 1.0 : 0.0) +
                                            negative_translations[transfer.edge][target][first_component];
                    const double at_second = (target == first_component ? 1.0 : 0.0) +
                                             positive_translations[transfer.edge][target][first_component];
                    require_finite(at_first, "translated transfer coefficient");
                    require_finite(at_second, "translated transfer coefficient");
                    const double first_update = amount * at_first;
                    const double second_update = amount * at_second;
                    require_finite(first_update, "translated bond update");
                    require_finite(second_update, "translated bond update");
                    auto& first_column = refined.blocks[first * count + fixed][target][second_component];
                    auto& second_column = refined.blocks[second * count + fixed][target][second_component];
                    auto& first_row = refined.blocks[fixed * count + first][second_component][target];
                    auto& second_row = refined.blocks[fixed * count + second][second_component][target];
                    first_column -= first_update;
                    second_column += second_update;
                    first_row -= first_update;
                    second_row += second_update;
                    require_finite(first_column, "refined first-index update");
                    require_finite(second_column, "refined first-index update");
                    require_finite(first_row, "refined reciprocal update");
                    require_finite(second_row, "refined reciprocal update");
                }
                const double canonical_amount = first <= second ? amount : -amount;
                result.transfers.push_back(
                    {std::min(first, second), std::max(first, second), first_component,
                     second_component, fixed, canonical_amount});
            }
        }
    }

    result.residuals = localization_residuals(truncated, refined, working_components);
    // Algorithm-controlled residuals: always held to residual_tolerance.
    const std::array<double, 4> owned_values{
        result.residuals.off_site, result.residuals.reciprocity,
        result.residuals.molecular_sum, result.residuals.charge_sum_transport,
    };
    // Supplied-input sum-rule defect, transported unchanged into these two.
    const std::array<double, 3> supplied_values{
        result.residuals.input_sum_rule, result.residuals.charge_sum,
        result.residuals.local_charge,
    };
    double owned_residual = 0.0, supplied_residual = 0.0;
    for (double residual : owned_values) {
        require_finite(residual, "localization residual candidate");
        owned_residual = std::max(owned_residual, residual);
    }
    for (double residual : supplied_values) {
        require_finite(residual, "supplied input sum-rule candidate");
        supplied_residual = std::max(supplied_residual, residual);
    }
    // Negative selects the historical single combined gate; infinity reports only.
    const double supplied_tolerance =
        input_sum_rule_tolerance < 0.0 ? residual_tolerance : input_sum_rule_tolerance;
    if (owned_residual > residual_tolerance || supplied_residual > supplied_tolerance) {
        std::ostringstream message;
        message << "localize_lw: postcondition exceeds residual tolerance (off-site="
                << result.residuals.off_site << ", charge-sum=" << result.residuals.charge_sum
                << ", reciprocity=" << result.residuals.reciprocity
                << ", molecular-sum=" << result.residuals.molecular_sum
                << ", local-charge=" << result.residuals.local_charge
                << ", input-sum-rule=" << result.residuals.input_sum_rule
                << ", charge-sum-transport=" << result.residuals.charge_sum_transport << ")";
        throw PSIEXCEPTION(message.str());
    }

    result.refined_pairs = refined.blocks;
    result.local.resize(count);
    for (std::size_t site = 0; site < count; ++site) {
        const auto& working = refined.blocks[site * count + site];
        // Components above the declared limit stay at the value-initialized zero.
        for (std::size_t row = 1; row < working_components; ++row) {
            for (std::size_t column = 1; column < working_components; ++column) {
                result.local[site][row - 1][column - 1] = working[row][column];
            }
        }
    }
    return result;
}
} }  // psi::isapol
