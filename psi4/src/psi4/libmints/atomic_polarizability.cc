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

#include "psi4/libmints/atomic_polarizability.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <queue>
#include <sstream>
#include <string>
#include <utility>

#include "psi4/libmints/matrix.h"
#include "psi4/libmints/wavefunction.h"
#include "psi4/libpsi4util/exception.h"

namespace psi {
namespace {

constexpr std::size_t kTensorDimension = 3;
constexpr double kValidationTolerance = 1.0e-10;

void require_three_by_three(const Matrix& matrix, const char* context) {
    if (matrix.nirrep() != 1 || matrix.nrow() != kTensorDimension || matrix.ncol() != kTensorDimension) {
        throw PSIEXCEPTION(std::string(context) + ": expected a 3 by 3 matrix");
    }
}

void require_finite_symmetric(const Matrix& matrix, const char* context) {
    require_three_by_three(matrix, context);
    for (std::size_t row = 0; row < kTensorDimension; ++row) {
        for (std::size_t column = 0; column < kTensorDimension; ++column) {
            if (!std::isfinite(matrix(row, column))) {
                throw PSIEXCEPTION(std::string(context) + ": expected a finite symmetric tensor");
            }
        }
    }
    for (std::size_t row = 0; row < kTensorDimension; ++row) {
        for (std::size_t column = row + 1; column < kTensorDimension; ++column) {
            const double scale = std::max({1.0, std::abs(matrix(row, column)), std::abs(matrix(column, row))});
            if (std::abs(matrix(row, column) - matrix(column, row)) > kValidationTolerance * scale) {
                throw PSIEXCEPTION(std::string(context) + ": expected a finite symmetric tensor");
            }
        }
    }
}

void require_rotation(const Matrix& rotation) {
    require_three_by_three(rotation, "rotate_tensor rotation");
    for (std::size_t row = 0; row < kTensorDimension; ++row) {
        for (std::size_t column = 0; column < kTensorDimension; ++column) {
            if (!std::isfinite(rotation(row, column))) {
                throw PSIEXCEPTION("rotate_tensor: rotation must be an orthonormal right-handed frame");
            }
            double dot = 0.0;
            for (std::size_t k = 0; k < kTensorDimension; ++k) {
                dot += rotation(row, k) * rotation(column, k);
            }
            const double expected = row == column ? 1.0 : 0.0;
            if (std::abs(dot - expected) > kValidationTolerance) {
                throw PSIEXCEPTION("rotate_tensor: rotation must be an orthonormal right-handed frame");
            }
        }
    }

    const double determinant =
        rotation(0, 0) * (rotation(1, 1) * rotation(2, 2) - rotation(1, 2) * rotation(2, 1)) -
        rotation(0, 1) * (rotation(1, 0) * rotation(2, 2) - rotation(1, 2) * rotation(2, 0)) +
        rotation(0, 2) * (rotation(1, 0) * rotation(2, 1) - rotation(1, 1) * rotation(2, 0));
    if (!(determinant > 0.0)) {
        throw PSIEXCEPTION("rotate_tensor: rotation must be an orthonormal right-handed frame");
    }
}

}  // namespace

namespace {

using DenseMatrix = std::vector<std::vector<double>>;
using Complex = std::complex<double>;
using ComplexVector = std::array<Complex, 16>;
constexpr double kGraphEigenvalueCutoff = 1.0e-4;
constexpr double kJacobiTolerance = 1.0e-14;

std::size_t real_cosine_index(unsigned int rank, unsigned int order) { return rank * rank + 2 * order - 1; }
std::size_t real_sine_index(unsigned int rank, unsigned int order) { return rank * rank + 2 * order; }
std::size_t complex_index(unsigned int rank, int order) {
    return rank * rank + static_cast<std::size_t>(static_cast<int>(rank) + order);
}

double binomial(unsigned int n, unsigned int k) {
    if (k > n) return 0.0;
    if (k > n - k) k = n - k;
    double result = 1.0;
    for (unsigned int i = 1; i <= k; ++i) result *= static_cast<double>(n - k + i) / i;
    return result;
}

L3WorkingVector regular_harmonics(const SitePosition& d) {
    const double x = d[0], y = d[1], z = d[2];
    const double rho2 = x * x + y * y + z * z;
    return {
        1.0, z, x, y,
        (3.0 * z * z - rho2) / 2.0, std::sqrt(3.0) * x * z, std::sqrt(3.0) * y * z,
        std::sqrt(3.0) * (x * x - y * y) / 2.0, std::sqrt(3.0) * x * y,
        (5.0 * z * z * z - 3.0 * z * rho2) / 2.0,
        std::sqrt(3.0 / 8.0) * x * (5.0 * z * z - rho2),
        std::sqrt(3.0 / 8.0) * y * (5.0 * z * z - rho2),
        std::sqrt(15.0) * z * (x * x - y * y) / 2.0, std::sqrt(15.0) * x * y * z,
        std::sqrt(10.0) * x * (x * x - 3.0 * y * y) / 4.0,
        std::sqrt(10.0) * y * (3.0 * x * x - y * y) / 4.0,
    };
}

ComplexVector real_to_complex(const L3WorkingVector& real) {
    ComplexVector result{};
    const double inverse_root2 = 1.0 / std::sqrt(2.0);
    for (unsigned int rank = 0; rank <= 3; ++rank) {
        result[complex_index(rank, 0)] = real[rank * rank];
        for (unsigned int order = 1; order <= rank; ++order) {
            const double cosine = real[real_cosine_index(rank, order)];
            const double sine = real[real_sine_index(rank, order)];
            const double phase = order % 2 == 0 ? 1.0 : -1.0;
            result[complex_index(rank, static_cast<int>(order))] =
                phase * inverse_root2 * Complex(cosine, sine);
            result[complex_index(rank, -static_cast<int>(order))] =
                inverse_root2 * Complex(cosine, -sine);
        }
    }
    return result;
}

L3WorkingVector complex_to_real(const ComplexVector& values) {
    L3WorkingVector result{};
    const double root2 = std::sqrt(2.0);
    for (unsigned int rank = 0; rank <= 3; ++rank) {
        result[rank * rank] = values[complex_index(rank, 0)].real();
        for (unsigned int order = 1; order <= rank; ++order) {
            const double phase = order % 2 == 0 ? 1.0 : -1.0;
            const Complex positive = values[complex_index(rank, static_cast<int>(order))];
            result[real_cosine_index(rank, order)] = phase * root2 * positive.real();
            result[real_sine_index(rank, order)] = phase * root2 * positive.imag();
        }
    }
    return result;
}

ComplexVector complex_regular_harmonics(const SitePosition& d) { return real_to_complex(regular_harmonics(d)); }

L3WorkingMatrix translation_matrix(const SitePosition& displacement) {
    L3WorkingMatrix result{};
    for (std::size_t source_component = 0; source_component < 16; ++source_component) {
        L3WorkingVector source{};
        source[source_component] = 1.0;
        const auto translated = translate_l3_multipoles(source, displacement);
        for (std::size_t target = 0; target < 16; ++target) result[target][source_component] = translated[target];
    }
    return result;
}

DenseMatrix make_graph_operator(const BondGraph& graph) {
    if (graph.site_count == 0) throw PSIEXCEPTION("localize_lw: bond graph must contain at least one site");
    DenseMatrix result(graph.site_count, std::vector<double>(graph.site_count, 0.0));
    for (const auto& bond : graph.bonds) {
        const std::size_t a = bond[0], b = bond[1];
        if (a >= graph.site_count || b >= graph.site_count || a == b)
            throw PSIEXCEPTION("localize_lw: bond graph contains an invalid bond");
        if (result[a][b] != 0.0) throw PSIEXCEPTION("localize_lw: bond graph contains a duplicate bond");
        result[a][b] = result[b][a] = 1.0;
        result[a][a] -= 1.0;
        result[b][b] -= 1.0;
    }
    return result;
}

bool graph_is_connected(const DenseMatrix& graph_operator) {
    std::vector<bool> visited(graph_operator.size(), false);
    std::queue<std::size_t> pending;
    visited[0] = true;
    pending.push(0);
    while (!pending.empty()) {
        const auto current = pending.front();
        pending.pop();
        for (std::size_t next = 0; next < graph_operator.size(); ++next) {
            if (next != current && graph_operator[current][next] != 0.0 && !visited[next]) {
                visited[next] = true;
                pending.push(next);
            }
        }
    }
    return std::all_of(visited.begin(), visited.end(), [](bool value) { return value; });
}

std::pair<std::vector<double>, DenseMatrix> symmetric_eigendecomposition(DenseMatrix matrix) {
    const std::size_t count = matrix.size();
    DenseMatrix vectors(count, std::vector<double>(count, 0.0));
    for (std::size_t i = 0; i < count; ++i) vectors[i][i] = 1.0;
    for (std::size_t iteration = 0; iteration < std::max<std::size_t>(1, 100 * count * count); ++iteration) {
        std::size_t p = 0, q = 0;
        double largest = 0.0;
        for (std::size_t i = 0; i < count; ++i) {
            for (std::size_t j = i + 1; j < count; ++j) {
                if (std::abs(matrix[i][j]) > largest) {
                    largest = std::abs(matrix[i][j]);
                    p = i;
                    q = j;
                }
            }
        }
        if (largest <= kJacobiTolerance) break;
        const double angle = 0.5 * std::atan2(2.0 * matrix[p][q], matrix[q][q] - matrix[p][p]);
        const double cosine = std::cos(angle), sine = std::sin(angle);
        const double app = matrix[p][p], aqq = matrix[q][q], apq = matrix[p][q];
        matrix[p][p] = cosine * cosine * app - 2.0 * sine * cosine * apq + sine * sine * aqq;
        matrix[q][q] = sine * sine * app + 2.0 * sine * cosine * apq + cosine * cosine * aqq;
        matrix[p][q] = matrix[q][p] = 0.0;
        for (std::size_t k = 0; k < count; ++k) {
            if (k == p || k == q) continue;
            const double akp = matrix[k][p], akq = matrix[k][q];
            matrix[k][p] = matrix[p][k] = cosine * akp - sine * akq;
            matrix[k][q] = matrix[q][k] = sine * akp + cosine * akq;
        }
        for (std::size_t k = 0; k < count; ++k) {
            const double vkp = vectors[k][p], vkq = vectors[k][q];
            vectors[k][p] = cosine * vkp - sine * vkq;
            vectors[k][q] = sine * vkp + cosine * vkq;
        }
    }
    std::vector<double> eigenvalues(count);
    for (std::size_t i = 0; i < count; ++i) eigenvalues[i] = matrix[i][i];
    return {eigenvalues, vectors};
}

DenseMatrix graph_pseudoinverse(const DenseMatrix& graph_operator, std::vector<double>* eigenvalues_out) {
    const auto decomposition = symmetric_eigendecomposition(graph_operator);
    const auto& eigenvalues = decomposition.first;
    const auto& vectors = decomposition.second;
    DenseMatrix result(graph_operator.size(), std::vector<double>(graph_operator.size(), 0.0));
    for (std::size_t mode = 0; mode < eigenvalues.size(); ++mode) {
        if (std::abs(eigenvalues[mode]) < kGraphEigenvalueCutoff) continue;
        for (std::size_t row = 0; row < result.size(); ++row)
            for (std::size_t column = 0; column < result.size(); ++column)
                result[row][column] += vectors[row][mode] * vectors[column][mode] / eigenvalues[mode];
    }
    if (eigenvalues_out) *eigenvalues_out = eigenvalues;
    return result;
}

Matrix to_psi_matrix(const DenseMatrix& values) {
    Matrix result(values.size(), values.size());
    for (std::size_t row = 0; row < values.size(); ++row)
        for (std::size_t column = 0; column < values.size(); ++column) result(row, column) = values[row][column];
    return result;
}

L3WorkingMatrix molecular_response(const SitePairResponse& response) {
    const std::size_t count = response.positions.size();
    std::vector<L3WorkingMatrix> translations;
    for (const auto& position : response.positions) translations.push_back(translation_matrix(position));
    L3WorkingMatrix result{};
    for (std::size_t a = 0; a < count; ++a) for (std::size_t b = 0; b < count; ++b) {
        const auto& block = response.blocks[a * count + b];
        for (std::size_t row = 0; row < 16; ++row) for (std::size_t column = 0; column < 16; ++column)
            for (std::size_t local_row = 0; local_row < 16; ++local_row)
                for (std::size_t local_column = 0; local_column < 16; ++local_column)
                    result[row][column] += translations[a][row][local_row] * block[local_row][local_column] *
                                           translations[b][column][local_column];
    }
    return result;
}

double matrix_max_difference(const L3WorkingMatrix& first, const L3WorkingMatrix& second) {
    double result = 0.0;
    for (std::size_t row = 0; row < 16; ++row) for (std::size_t column = 0; column < 16; ++column)
        result = std::max(result, std::abs(first[row][column] - second[row][column]));
    return result;
}

LocalizationResiduals localization_residuals(const SitePairResponse& before, const SitePairResponse& after) {
    const std::size_t count = after.positions.size();
    LocalizationResiduals residuals{};
    for (std::size_t a = 0; a < count; ++a) {
        const auto& local = after.blocks[a * count + a];
        for (std::size_t component = 0; component < 16; ++component) {
            residuals.local_charge = std::max(residuals.local_charge,
                std::max(std::abs(local[0][component]), std::abs(local[component][0])));
            double first_sum = 0.0, second_sum = 0.0;
            for (std::size_t b = 0; b < count; ++b) {
                first_sum += after.blocks[a * count + b][component][0];
                second_sum += after.blocks[b * count + a][0][component];
            }
            residuals.charge_sum = std::max(residuals.charge_sum,
                                             std::max(std::abs(first_sum), std::abs(second_sum)));
        }
        for (std::size_t b = 0; b < count; ++b) {
            const auto& block = after.blocks[a * count + b];
            const auto& reciprocal = after.blocks[b * count + a];
            for (std::size_t row = 0; row < 16; ++row) for (std::size_t column = 0; column < 16; ++column) {
                if (a != b) residuals.off_site = std::max(residuals.off_site, std::abs(block[row][column]));
                residuals.reciprocity = std::max(residuals.reciprocity,
                                                  std::abs(block[row][column] - reciprocal[column][row]));
            }
        }
    }
    residuals.molecular_sum = matrix_max_difference(molecular_response(before), molecular_response(after));
    return residuals;
}

}  // namespace

Matrix lw_graph_operator(const BondGraph& graph) { return to_psi_matrix(make_graph_operator(graph)); }

std::pair<Matrix, std::vector<double>> lw_graph_pseudoinverse(const BondGraph& graph) {
    const auto graph_operator = make_graph_operator(graph);
    std::vector<double> eigenvalues;
    auto pseudoinverse = graph_pseudoinverse(graph_operator, &eigenvalues);
    return {to_psi_matrix(pseudoinverse), eigenvalues};
}

L3WorkingVector translate_l3_multipoles(const L3WorkingVector& source,
                                         const SitePosition& source_minus_target) {
    for (double value : source) {
        if (!std::isfinite(value)) {
            throw PSIEXCEPTION("translate_l3_multipoles: expected finite multipole values");
        }
    }
    for (double value : source_minus_target) {
        if (!std::isfinite(value)) {
            throw PSIEXCEPTION("translate_l3_multipoles: expected a finite displacement");
        }
    }

    const auto complex_source = real_to_complex(source);
    const auto harmonics = complex_regular_harmonics(source_minus_target);
    ComplexVector translated{};
    for (unsigned int target_rank = 0; target_rank <= 3; ++target_rank) {
        for (int target_order = -static_cast<int>(target_rank);
             target_order <= static_cast<int>(target_rank); ++target_order) {
            Complex value = 0.0;
            for (unsigned int source_rank = 0; source_rank <= target_rank; ++source_rank) {
                for (int source_order = -static_cast<int>(source_rank);
                     source_order <= static_cast<int>(source_rank); ++source_order) {
                    const int difference_order = target_order - source_order;
                    const unsigned int difference_rank = target_rank - source_rank;
                    if (std::abs(difference_order) > static_cast<int>(difference_rank)) continue;
                    const int lower_first = static_cast<int>(source_rank) - source_order;
                    const int lower_second = static_cast<int>(source_rank) + source_order;
                    if (lower_first < 0 || lower_second < 0) continue;
                    const double coefficient = std::sqrt(
                        binomial(target_rank - target_order, static_cast<unsigned int>(lower_first)) *
                        binomial(target_rank + target_order, static_cast<unsigned int>(lower_second)));
                    value += coefficient * complex_source[complex_index(source_rank, source_order)] *
                             harmonics[complex_index(difference_rank, difference_order)];
                }
            }
            translated[complex_index(target_rank, target_order)] = value;
        }
    }
    return complex_to_real(translated);
}

LocalizedResponse localize_lw(const SitePairResponse& response, const BondGraph& graph,
                              double residual_tolerance) {
    if (!std::isfinite(residual_tolerance) || residual_tolerance <= 0.0) {
        throw PSIEXCEPTION("localize_lw: residual tolerance must be finite and positive");
    }
    const std::size_t count = response.positions.size();
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
    if (!graph_is_connected(graph_operator)) {
        throw PSIEXCEPTION("localize_lw: expected a connected bond graph");
    }
    const auto pseudoinverse = graph_pseudoinverse(graph_operator, nullptr);

    double input_reciprocity = 0.0;
    for (std::size_t a = 0; a < count; ++a) {
        for (std::size_t b = 0; b < count; ++b) {
            for (std::size_t row = 0; row < 16; ++row) {
                for (std::size_t column = 0; column < 16; ++column) {
                    input_reciprocity =
                        std::max(input_reciprocity,
                                 std::abs(response.blocks[a * count + b][row][column] -
                                          response.blocks[b * count + a][column][row]));
                }
            }
        }
    }
    if (input_reciprocity > residual_tolerance) {
        throw PSIEXCEPTION("localize_lw: input reciprocity exceeds residual tolerance");
    }

    SitePairResponse refined = response;
    std::vector<L3WorkingMatrix> positive_translations;
    std::vector<L3WorkingMatrix> negative_translations;
    positive_translations.reserve(graph.bonds.size());
    negative_translations.reserve(graph.bonds.size());
    for (const auto& bond : graph.bonds) {
        SitePosition displacement{};
        for (std::size_t axis = 0; axis < 3; ++axis) {
            displacement[axis] = response.positions[bond[0]][axis] - response.positions[bond[1]][axis];
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
    LocalizedResponse result;
    for (std::size_t first_component = 0; first_component < 16; ++first_component) {
        for (std::size_t second_component = first_component; second_component < 16; ++second_component) {
            std::vector<PendingTransfer> pending;
            const double symmetry_factor = first_component == second_component ? 0.5 : 1.0;
            for (std::size_t fixed_site = 0; fixed_site < count; ++fixed_site) {
                std::vector<double> unwanted(count, 0.0);
                double offsite_sum = 0.0;
                for (std::size_t site = 0; site < count; ++site) {
                    if (site == fixed_site) continue;
                    unwanted[site] = symmetry_factor *
                                     refined.blocks[site * count + fixed_site][first_component][second_component];
                    offsite_sum += unwanted[site];
                }
                unwanted[fixed_site] = -offsite_sum;

                std::vector<double> potential(count, 0.0);
                for (std::size_t row = 0; row < count; ++row) {
                    for (std::size_t column = 0; column < count; ++column) {
                        potential[row] += pseudoinverse[row][column] * unwanted[column];
                    }
                }
                double range_residual = 0.0;
                for (std::size_t row = 0; row < count; ++row) {
                    double projected = 0.0;
                    for (std::size_t column = 0; column < count; ++column) {
                        projected += graph_operator[row][column] * potential[column];
                    }
                    range_residual = std::max(range_residual, std::abs(projected - unwanted[row]));
                }
                if (range_residual > residual_tolerance) {
                    throw PSIEXCEPTION("localize_lw: graph solve exceeds residual tolerance");
                }
                for (std::size_t edge = 0; edge < graph.bonds.size(); ++edge) {
                    const auto& bond = graph.bonds[edge];
                    const double amount = 0.5 * (potential[bond[1]] - potential[bond[0]]);
                    if (amount != 0.0) pending.push_back({edge, fixed_site, amount});
                }
            }

            for (const auto& transfer : pending) {
                const auto& bond = graph.bonds[transfer.edge];
                const std::size_t first = bond[0];
                const std::size_t second = bond[1];
                const std::size_t fixed = transfer.fixed_site;
                const double amount = transfer.amount;
                for (std::size_t target = 0; target < 16; ++target) {
                    const double at_first = (target == first_component ? 1.0 : 0.0) +
                                            negative_translations[transfer.edge][target][first_component];
                    const double at_second = (target == first_component ? 1.0 : 0.0) +
                                             positive_translations[transfer.edge][target][first_component];
                    refined.blocks[first * count + fixed][target][second_component] -= amount * at_first;
                    refined.blocks[second * count + fixed][target][second_component] += amount * at_second;
                    refined.blocks[fixed * count + first][second_component][target] -= amount * at_first;
                    refined.blocks[fixed * count + second][second_component][target] += amount * at_second;
                }
                result.transfers.push_back(
                    {std::min(first, second), std::max(first, second), first_component,
                     second_component, fixed, amount});
            }
        }
    }

    result.residuals = localization_residuals(response, refined);
    const double maximum_residual =
        std::max({result.residuals.off_site, result.residuals.charge_sum,
                  result.residuals.reciprocity, result.residuals.molecular_sum,
                  result.residuals.local_charge});
    if (maximum_residual > residual_tolerance) {
        std::ostringstream message;
        message << "localize_lw: postcondition exceeds residual tolerance (off-site="
                << result.residuals.off_site << ", charge-sum=" << result.residuals.charge_sum
                << ", reciprocity=" << result.residuals.reciprocity
                << ", molecular-sum=" << result.residuals.molecular_sum
                << ", local-charge=" << result.residuals.local_charge << ")";
        throw PSIEXCEPTION(message.str());
    }

    result.local.resize(count);
    for (std::size_t site = 0; site < count; ++site) {
        const auto& working = refined.blocks[site * count + site];
        for (std::size_t row = 1; row < 16; ++row) {
            for (std::size_t column = 1; column < 16; ++column) {
                result.local[site][row - 1][column - 1] = working[row][column];
            }
        }
    }
    return result;
}

FrequencyGrid make_casimir_grid(unsigned int nonzero_count, double scale) {
    if (nonzero_count != 10) {
        throw PSIEXCEPTION("make_casimir_grid: protocol requires exactly ten nonzero frequencies");
    }
    if (!std::isfinite(scale) || scale <= 0.0) {
        throw PSIEXCEPTION("make_casimir_grid: scale must be finite and positive");
    }

    // Reviewed default-scale values are embedded so production never depends on
    // generated reference files. Other positive scales preserve the same nodes.
    static constexpr std::array<double, 11> reviewed_frequencies{
        0.0,
        0.0066096015960872435,
        0.03617481199863096,
        0.09544736369034827,
        0.1976442118453127,
        0.3704172128053672,
        0.6749146404580301,
        1.264899172436498,
        2.619244684547324,
        6.910885950408292,
        37.82376235021415,
    };
    static constexpr std::array<double, 10> nodes{
        -0.9739065285171717, -0.8650633666889845, -0.6794095682990244, -0.4333953941292472,
        -0.1488743389816312, 0.1488743389816312,  0.4333953941292472,  0.6794095682990244,
        0.8650633666889845,  0.9739065285171717,
    };
    static constexpr std::array<double, 10> legendre_weights{
        0.06667134430868814, 0.1494513491505806, 0.2190863625159820, 0.2692667193099964,
        0.2955242247147529,  0.2955242247147529, 0.2692667193099964, 0.2190863625159820,
        0.1494513491505806,  0.06667134430868814,
    };

    FrequencyGrid grid;
    grid.frequencies.reserve(reviewed_frequencies.size());
    grid.weights.reserve(reviewed_frequencies.size());
    const double scale_ratio = scale / 0.5;
    if (!std::isfinite(scale_ratio)) {
        throw PSIEXCEPTION("make_casimir_grid: scale must be finite and positive at every grid point");
    }
    grid.frequencies.push_back(0.0);
    grid.weights.push_back(0.0);
    for (std::size_t point = 0; point < nodes.size(); ++point) {
        const double frequency = reviewed_frequencies[point + 1] * scale_ratio;
        const double denominator = 1.0 - nodes[point];
        const double weight = legendre_weights[point] * 2.0 * scale / (denominator * denominator);
        if (!std::isfinite(frequency) || frequency <= 0.0 || !std::isfinite(weight) || weight <= 0.0) {
            throw PSIEXCEPTION("make_casimir_grid: scale must be finite and positive at every grid point");
        }
        grid.frequencies.push_back(frequency);
        grid.weights.push_back(weight);
    }
    return grid;
}

Matrix local_spherical_dipole_to_cartesian(const L3Matrix& spherical) {
    for (const auto& row : spherical) {
        for (double value : row) {
            if (!std::isfinite(value)) {
                throw PSIEXCEPTION(
                    "local_spherical_dipole_to_cartesian: expected finite real-spherical values");
            }
        }
    }

    // Real-spherical dipoles are ordered (10, 11c, 11s) = (z, x, y).
    static constexpr std::array<std::size_t, 3> cartesian_to_spherical{1, 2, 0};
    Matrix cartesian(3, 3);
    for (std::size_t row = 0; row < kTensorDimension; ++row) {
        for (std::size_t column = 0; column < kTensorDimension; ++column) {
            cartesian(row, column) = spherical[cartesian_to_spherical[row]][cartesian_to_spherical[column]];
        }
    }
    require_finite_symmetric(cartesian, "local_spherical_dipole_to_cartesian");
    return cartesian;
}

Matrix rotate_tensor(const Matrix& local, const Matrix& local_to_global) {
    require_finite_symmetric(local, "rotate_tensor local tensor");
    require_rotation(local_to_global);

    Matrix global(3, 3);
    for (std::size_t row = 0; row < kTensorDimension; ++row) {
        for (std::size_t column = 0; column < kTensorDimension; ++column) {
            double value = 0.0;
            for (std::size_t local_row = 0; local_row < kTensorDimension; ++local_row) {
                for (std::size_t local_column = 0; local_column < kTensorDimension; ++local_column) {
                    value += local_to_global(row, local_row) * local(local_row, local_column) *
                             local_to_global(column, local_column);
                }
            }
            global(row, column) = value;
        }
    }
    require_finite_symmetric(global, "rotate_tensor result");
    return global;
}

std::array<double, 6> pack_symmetric_tensor(const Matrix& tensor) {
    require_finite_symmetric(tensor, "pack_symmetric_tensor");
    return {tensor(0, 0), tensor(0, 1), tensor(0, 2), tensor(1, 1), tensor(1, 2), tensor(2, 2)};
}

AtomicPolarizabilityCalculator::AtomicPolarizabilityCalculator(std::shared_ptr<Wavefunction> wfn)
    : wfn_(std::move(wfn)) {
    if (!wfn_) {
        throw PSIEXCEPTION("AtomicPolarizabilityCalculator: wavefunction is null");
    }
}

void AtomicPolarizabilityCalculator::validate_wavefunction_prerequisites() const {
    bool has_orbital_response_data = false;
    try {
        has_orbital_response_data =
            wfn_->molecule() && wfn_->basisset() && wfn_->Ca() && wfn_->Da() && wfn_->epsilon_a();
    } catch (const PsiException&) {
        // Some Wavefunction accessors reject incomplete, safely constructed wavefunctions.
    }

    if (!has_orbital_response_data) {
        throw PSIEXCEPTION(
            "AtomicPolarizabilityCalculator: unsupported wavefunction is missing required orbital response data");
    }
}

void AtomicPolarizabilityCalculator::compute() {
    // Output arrays must not be allocated or published until every native response
    // prerequisite has been validated. The response provider is added in a later stage.
    validate_wavefunction_prerequisites();
    throw PSIEXCEPTION("AtomicPolarizabilityCalculator: required native response data are unavailable");
}

}  // namespace psi
