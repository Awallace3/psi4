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
#include <iomanip>
#include <initializer_list>
#include <limits>
#include <queue>
#include <sstream>
#include <string>
#include <utility>

#include "psi4/libfunctional/LibXCfunctional.h"
#include "psi4/libfunctional/functional.h"
#include "psi4/libfunctional/superfunctional.h"
#include "psi4/libfock/cubature.h"
#include "psi4/libfock/jk.h"
#include "psi4/libfock/points.h"
#include "psi4/libmints/basisset.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/mintshelper.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libmints/pointgrp.h"
#include "psi4/libmints/vector.h"
#include "psi4/libmints/wavefunction.h"
#include "psi4/liboptions/liboptions.h"
#include "psi4/libscf_solver/hf.h"
#include "psi4/libscf_solver/rhf.h"
#include "psi4/libscf_solver/uhf.h"
#include "psi4/libpsi4util/exception.h"
#include "psi4/libpsi4util/process.h"
#include "psi4/libqt/qt.h"

namespace psi {
namespace {

constexpr std::size_t kTensorDimension = 3;
constexpr double kValidationTolerance = 1.0e-10;
constexpr double kDenseMinimumReciprocalCondition = 1.0e-12;
constexpr double kDenseMinimumReciprocalPivotGrowth = 1.0e-12;
constexpr double kDenseMaximumForwardError = 1.0e-8;
constexpr double kDenseMaximumBackwardError = 1.0e-11;
constexpr double kDenseMaximumScaledResidual = 1.0e-11;

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

void require_dense_response_operator(const Matrix& matrix, const char* name) {
    if (matrix.nirrep() != 1 || matrix.nrow() == 0 || matrix.nrow() != matrix.ncol()) {
        throw PSIEXCEPTION(std::string("dense restricted response: ") + name +
                           " must be a nonempty square matrix");
    }
    for (int row = 0; row < matrix.nrow(); ++row) {
        for (int column = 0; column < matrix.ncol(); ++column) {
            if (!std::isfinite(matrix(row, column))) {
                throw PSIEXCEPTION(std::string("dense restricted response: ") + name +
                                   " must contain only finite values");
            }
        }
    }
    for (int row = 0; row < matrix.nrow(); ++row) {
        for (int column = row + 1; column < matrix.ncol(); ++column) {
            const double scale = std::max({1.0, std::abs(matrix(row, column)),
                                           std::abs(matrix(column, row))});
            if (std::abs(matrix(row, column) - matrix(column, row)) >
                kValidationTolerance * scale) {
                throw PSIEXCEPTION(std::string("dense restricted response: ") + name +
                                   " must be symmetric");
            }
        }
    }
}

void require_restricted_hessian_primitive(const Matrix& matrix, std::size_t transition_count,
                                          const char* name) {
    const std::string prefix = "restricted singlet Hessian: ";
    if (matrix.nirrep() != 1 || matrix.nrow() != transition_count ||
        matrix.ncol() != transition_count) {
        throw PSIEXCEPTION(prefix + name + " dimensions must match the orbital gaps");
    }
    for (int row = 0; row < matrix.nrow(); ++row) {
        for (int column = 0; column < matrix.ncol(); ++column) {
            if (!std::isfinite(matrix(row, column)))
                throw PSIEXCEPTION(prefix + name + " must contain only finite values");
        }
    }
    for (int row = 0; row < matrix.nrow(); ++row) {
        for (int column = row + 1; column < matrix.ncol(); ++column) {
            const double scale = std::max({1.0, std::abs(matrix(row, column)),
                                           std::abs(matrix(column, row))});
            if (std::abs(matrix(row, column) - matrix(column, row)) >
                kValidationTolerance * scale)
                throw PSIEXCEPTION(prefix + name + " must be symmetric");
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
constexpr double kElementTransferThreshold = 1.0e-7;
constexpr double kLinearAlgebraTolerance = 1.0e-10;

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

void require_finite(double value, const char* context) {
    if (!std::isfinite(value)) throw PSIEXCEPTION(std::string(context) + ": derived value is not finite");
}

void require_finite(const Complex& value, const char* context) {
    require_finite(value.real(), context);
    require_finite(value.imag(), context);
}

double finite_absolute(double value, const char* context) {
    require_finite(value, context);
    const double result = std::abs(value);
    require_finite(result, context);
    return result;
}

L3WorkingVector regular_harmonics(const SitePosition& d) {
    const double x = d[0], y = d[1], z = d[2];
    const double rho2 = x * x + y * y + z * z;
    L3WorkingVector result{
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
    for (double value : result) require_finite(value, "regular_harmonics");
    return result;
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
    for (const auto& value : result) require_finite(value, "real_to_complex");
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
    for (double value : result) require_finite(value, "complex_to_real");
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

DenseMatrix graph_pseudoinverse(const DenseMatrix& graph_operator, std::vector<double>* eigenvalues_out) {
    const auto components = graph_components(graph_operator);
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
            if (std::abs(norm - 1.0) > validation_tolerance) {
                throw PSIEXCEPTION("localize_lw: graph eigensolver returned non-orthonormal vectors");
            }
            for (std::size_t other = 0; other < mode; ++other) {
                double dot = 0.0;
                for (std::size_t row = 0; row < count; ++row) {
                    dot += eigenvectors(row, mode) * eigenvectors(row, other);
                }
                require_finite(dot, "graph eigenvector orthogonality");
                if (std::abs(dot) > validation_tolerance) {
                    throw PSIEXCEPTION("localize_lw: graph eigensolver returned non-orthogonal vectors");
                }
            }
            for (std::size_t row = 0; row < count; ++row) {
                double residual = -eigenvalue * eigenvectors(row, mode);
                for (std::size_t column = 0; column < count; ++column) {
                    residual += graph_operator[component[row]][component[column]] *
                                eigenvectors(column, mode);
                }
                if (finite_absolute(residual, "graph eigen residual") > validation_tolerance) {
                    throw PSIEXCEPTION("localize_lw: graph eigensolver residual exceeds tolerance");
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
            if (std::abs(result[row][column] - result[column][row]) > validation_tolerance) {
                throw PSIEXCEPTION("localize_lw: graph pseudoinverse is not symmetric");
            }
        }
    }
    const auto operator_inverse = dense_multiply(graph_operator, result);
    const auto inverse_operator = dense_multiply(result, graph_operator);
    if (dense_max_difference(dense_multiply(operator_inverse, graph_operator), graph_operator,
                             "graph Moore-Penrose residual") > validation_tolerance ||
        dense_max_difference(dense_multiply(inverse_operator, result), result,
                             "graph Moore-Penrose residual") > validation_tolerance ||
        dense_max_difference(operator_inverse, dense_transpose(operator_inverse),
                             "graph projector symmetry") > validation_tolerance ||
        dense_max_difference(inverse_operator, dense_transpose(inverse_operator),
                             "graph projector symmetry") > validation_tolerance) {
        throw PSIEXCEPTION("localize_lw: graph pseudoinverse failed Moore-Penrose validation");
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

L3WorkingMatrix molecular_response(const SitePairResponse& response) {
    const std::size_t count = response.positions.size();
    std::vector<L3WorkingMatrix> translations;
    for (const auto& position : response.positions) translations.push_back(translation_matrix(position));
    L3WorkingMatrix result{};
    for (std::size_t a = 0; a < count; ++a) for (std::size_t b = 0; b < count; ++b) {
        const auto& block = response.blocks[a * count + b];
        for (std::size_t row = 0; row < 16; ++row) for (std::size_t column = 0; column < 16; ++column)
            for (std::size_t local_row = 0; local_row < 16; ++local_row)
                for (std::size_t local_column = 0; local_column < 16; ++local_column) {
                    const double term = translations[a][row][local_row] * block[local_row][local_column] *
                                        translations[b][column][local_column];
                    require_finite(term, "molecular response product");
                    result[row][column] += term;
                    require_finite(result[row][column], "molecular response accumulation");
                }
    }
    return result;
}

double matrix_max_difference(const L3WorkingMatrix& first, const L3WorkingMatrix& second) {
    double result = 0.0;
    for (std::size_t row = 0; row < 16; ++row) for (std::size_t column = 0; column < 16; ++column)
        result = std::max(result, finite_absolute(first[row][column] - second[row][column],
                                                   "molecular response residual"));
    return result;
}

LocalizationResiduals localization_residuals(const SitePairResponse& before, const SitePairResponse& after) {
    const std::size_t count = after.positions.size();
    LocalizationResiduals residuals{};
    for (std::size_t a = 0; a < count; ++a) {
        const auto& local = after.blocks[a * count + a];
        for (std::size_t component = 0; component < 16; ++component) {
            residuals.local_charge = std::max(
                residuals.local_charge,
                std::max(finite_absolute(local[0][component], "local charge residual"),
                         finite_absolute(local[component][0], "local charge residual")));
            double first_sum = 0.0, second_sum = 0.0;
            for (std::size_t b = 0; b < count; ++b) {
                first_sum += after.blocks[a * count + b][component][0];
                second_sum += after.blocks[b * count + a][0][component];
                require_finite(first_sum, "charge sum residual");
                require_finite(second_sum, "charge sum residual");
            }
            residuals.charge_sum = std::max(
                residuals.charge_sum,
                std::max(finite_absolute(first_sum, "charge sum residual"),
                         finite_absolute(second_sum, "charge sum residual")));
        }
        for (std::size_t b = 0; b < count; ++b) {
            const auto& block = after.blocks[a * count + b];
            const auto& reciprocal = after.blocks[b * count + a];
            for (std::size_t row = 0; row < 16; ++row) for (std::size_t column = 0; column < 16; ++column) {
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
    residuals.molecular_sum = matrix_max_difference(molecular_response(before), molecular_response(after));
    return residuals;
}

}  // namespace

namespace {

constexpr const char* kProtocolGRACX = "XC_GGA_X_LB";
constexpr const char* kProtocolGRACC = "XC_LDA_C_VWN";
// Molecule coordinates are stored in Bohr. This absolute tolerance admits only
// serialization/orientation roundoff from independently built vertical states.
constexpr double kGeometryToleranceBohr = 1.0e-12;

bool same_coordinate_bohr(double first, double second) {
    return std::abs(first - second) <= kGeometryToleranceBohr;
}

bool same_coordinate_vector(const std::vector<double>& first, const std::vector<double>& second) {
    return first.size() == second.size() &&
           std::equal(first.begin(), first.end(), second.begin(), same_coordinate_bohr);
}

bool same_vertical_basis_structure(const BasisSetStructuralSnapshot& first,
                                   const BasisSetStructuralSnapshot& second) {
    if (first.shells.size() != second.shells.size() ||
        first.ecp_shells.size() != second.ecp_shells.size() ||
        !same_coordinate_vector(first.centers, second.centers)) return false;

    auto coordinate_adjusted = second;
    coordinate_adjusted.centers = first.centers;
    for (std::size_t shell = 0; shell < first.shells.size(); ++shell) {
        if (!same_coordinate_vector(first.shells[shell].coordinates,
                                    second.shells[shell].coordinates)) return false;
        coordinate_adjusted.shells[shell].coordinates = first.shells[shell].coordinates;
    }
    for (std::size_t shell = 0; shell < first.ecp_shells.size(); ++shell) {
        if (!same_coordinate_vector(first.ecp_shells[shell].coordinates,
                                    second.ecp_shells[shell].coordinates)) return false;
        coordinate_adjusted.ecp_shells[shell].coordinates = first.ecp_shells[shell].coordinates;
    }
    return first == coordinate_adjusted;
}

bool close_enough(double first, double second) {
    const double scale = std::max({1.0, std::abs(first), std::abs(second)});
    return std::abs(first - second) <= kValidationTolerance * scale;
}

std::shared_ptr<scf::HF> require_converged_scf(const std::shared_ptr<Wavefunction>& wfn,
                                               const char* role) {
    const auto hf = std::dynamic_pointer_cast<scf::HF>(wfn);
    if (!hf) throw PSIEXCEPTION(std::string("FrozenResponseContext: ") + role + " must be an scf::HF state");
    if (!hf->response_state_sealed() || !hf->response_provenance()) {
        throw PSIEXCEPTION(std::string("FrozenResponseContext: ") + role + " SCF state has no finalized provenance seal");
    }
    return hf;
}

bool same_nuclei_and_geometry(const Molecule& first, const Molecule& second) {
    if (first.natom() != second.natom()) return false;
    for (int atom = 0; atom < first.natom(); ++atom) {
        if (first.Z(atom) != second.Z(atom) ||
            !same_coordinate_bohr(first.x(atom), second.x(atom)) ||
            !same_coordinate_bohr(first.y(atom), second.y(atom)) ||
            !same_coordinate_bohr(first.z(atom), second.z(atom))) return false;
    }
    return true;
}

}  // namespace

void detail::validate_vertical_protocol(bool cation_state_valid, bool complete_basis_valid) {
    if (!cation_state_valid) {
        throw PSIEXCEPTION("FrozenResponseContext: vertical cation must be a charge +1 doublet UKS state");
    }
    if (!complete_basis_valid) {
        throw PSIEXCEPTION("FrozenResponseContext: precursor/cation complete basis structure is inconsistent");
    }
}

void detail::validate_dense_response_diagnostics(
    double reciprocal_condition, double reciprocal_pivot_growth,
    const std::vector<double>& forward_error, const std::vector<double>& backward_error,
    const std::vector<double>& scaled_residual) {
    if (forward_error.empty() || backward_error.size() != forward_error.size() ||
        scaled_residual.size() != forward_error.size()) {
        throw PSIEXCEPTION("dense restricted response: diagnostic cardinalities are inconsistent");
    }
    if (!std::isfinite(reciprocal_condition) ||
        reciprocal_condition < kDenseMinimumReciprocalCondition) {
        throw PSIEXCEPTION("dense restricted response: reciprocal condition estimate is below 1e-12");
    }
    if (!std::isfinite(reciprocal_pivot_growth) ||
        reciprocal_pivot_growth < kDenseMinimumReciprocalPivotGrowth) {
        throw PSIEXCEPTION("dense restricted response: reciprocal pivot growth is below 1e-12");
    }
    for (std::size_t column = 0; column < forward_error.size(); ++column) {
        if (!std::isfinite(forward_error[column]) || forward_error[column] < 0.0 ||
            forward_error[column] > kDenseMaximumForwardError) {
            throw PSIEXCEPTION("dense restricted response: forward error estimate exceeds 1e-8");
        }
        if (!std::isfinite(backward_error[column]) || backward_error[column] < 0.0 ||
            backward_error[column] > kDenseMaximumBackwardError) {
            throw PSIEXCEPTION("dense restricted response: backward error estimate exceeds 1e-11");
        }
        if (!std::isfinite(scaled_residual[column]) || scaled_residual[column] < 0.0 ||
            scaled_residual[column] > kDenseMaximumScaledResidual) {
            throw PSIEXCEPTION("dense restricted response: recomputed scaled residual exceeds 1e-11");
        }
    }
}

detail::DenseRestrictedResponse::DenseRestrictedResponse(
    SharedMatrix P, SharedMatrix Q, double reciprocal_condition,
    double reciprocal_pivot_growth, std::vector<double> forward_error,
    std::vector<double> backward_error, std::vector<double> scaled_residual,
    std::vector<double> solution_column_scales)
    : P_(std::move(P)),
      Q_(std::move(Q)),
      reciprocal_condition_(reciprocal_condition),
      reciprocal_pivot_growth_(reciprocal_pivot_growth),
      forward_error_(std::move(forward_error)),
      backward_error_(std::move(backward_error)),
      scaled_residual_(std::move(scaled_residual)),
      solution_column_scales_(std::move(solution_column_scales)) {}

SharedMatrix detail::DenseRestrictedResponse::P_clone() const { return P_->clone(); }
SharedMatrix detail::DenseRestrictedResponse::Q_clone() const { return Q_->clone(); }
std::size_t detail::DenseRestrictedResponse::transition_count() const {
    return static_cast<std::size_t>(P_->nrow());
}

detail::DenseRestrictedResponse detail::solve_dense_restricted_response(
    const Matrix& H1, const Matrix& H2, double omega, const Matrix& rhs) {
    require_dense_response_operator(H1, "H1");
    require_dense_response_operator(H2, "H2");
    if (H2.nrow() != H1.nrow())
        throw PSIEXCEPTION("dense restricted response: H1 and H2 must have the same dimension");
    if (!std::isfinite(omega))
        throw PSIEXCEPTION("dense restricted response: omega must be finite and nonnegative");
    if (omega < 0.0)
        throw PSIEXCEPTION("dense restricted response: omega must be nonnegative");
    if (rhs.nirrep() != 1 || rhs.nrow() != H1.nrow() || rhs.ncol() == 0)
        throw PSIEXCEPTION("dense restricted response: RHS dimensions must be n by one-or-more columns");
    for (int row = 0; row < rhs.nrow(); ++row)
        for (int column = 0; column < rhs.ncol(); ++column)
            if (!std::isfinite(rhs(row, column)))
                throw PSIEXCEPTION("dense restricted response: RHS must contain only finite values");

    const int transition_count = H1.nrow();
    if (omega != 0.0 && transition_count > std::numeric_limits<int>::max() / 2)
        throw PSIEXCEPTION("dense restricted response: doubled dimension exceeds LAPACK limits");
    const int order = omega == 0.0 ? transition_count : 2 * transition_count;
    const int rhs_count = rhs.ncol();
    const auto dimension = static_cast<std::size_t>(order);
    const auto column_count = static_cast<std::size_t>(rhs_count);
    if (dimension > std::numeric_limits<std::size_t>::max() / dimension ||
        dimension > std::numeric_limits<std::size_t>::max() / column_count)
        throw PSIEXCEPTION("dense restricted response: allocation size overflow");

    // Column-major LAPACK storage. At exactly zero frequency the decoupled
    // H2 equation is omitted, making this precisely the H1 P = rhs solve.
    std::vector<double> coefficients(dimension * dimension, 0.0);
    for (int row = 0; row < transition_count; ++row) {
        for (int column = 0; column < transition_count; ++column) {
            coefficients[row + static_cast<std::size_t>(column) * dimension] = H1(row, column);
            if (omega != 0.0)
                coefficients[row + transition_count +
                             static_cast<std::size_t>(column + transition_count) * dimension] = H2(row, column);
        }
        if (omega != 0.0) {
            coefficients[row + static_cast<std::size_t>(row + transition_count) * dimension] = omega;
            coefficients[row + transition_count + static_cast<std::size_t>(row) * dimension] = -omega;
        }
    }
    const auto original_coefficients = coefficients;
    std::vector<double> lapack_rhs(dimension * column_count, 0.0);
    for (int column = 0; column < rhs_count; ++column)
        for (int row = 0; row < transition_count; ++row)
            lapack_rhs[row + static_cast<std::size_t>(column) * dimension] = rhs(row, column);
    const auto original_rhs = lapack_rhs;

    std::vector<double> solution(dimension * column_count, 0.0);
    std::vector<double> factors(dimension * dimension, 0.0);
    std::vector<int> pivots(dimension, 0), integer_work(dimension, 0);
    std::vector<double> row_scale(dimension, 0.0), column_scale(dimension, 0.0);
    std::vector<double> forward_error(column_count, 0.0), backward_error(column_count, 0.0);
    std::vector<double> work(4 * dimension, 0.0);
    double reciprocal_condition = 0.0;
    const int info = C_DGESVX('N', 'N', order, rhs_count, coefficients.data(), order,
                              factors.data(), order, pivots.data(), 'N', row_scale.data(),
                              column_scale.data(), lapack_rhs.data(), order, solution.data(), order,
                              &reciprocal_condition, forward_error.data(), backward_error.data(),
                              work.data(), integer_work.data());
    if (info != 0)
        throw PSIEXCEPTION("dense restricted response: LAPACK reported singular or invalid equations; info=" +
                           std::to_string(info));

    std::vector<double> scaled_residual(column_count, 0.0);
    for (int column = 0; column < rhs_count; ++column) {
        for (int row = 0; row < order; ++row) {
            const auto rhs_index = row + static_cast<std::size_t>(column) * dimension;
            double product = 0.0;
            double scale = std::abs(original_rhs[rhs_index]);
            for (int inner = 0; inner < order; ++inner) {
                const double a = original_coefficients[row + static_cast<std::size_t>(inner) * dimension];
                const double x = solution[inner + static_cast<std::size_t>(column) * dimension];
                product += a * x;
                scale += std::abs(a) * std::abs(x);
            }
            const double residual = product - original_rhs[rhs_index];
            double relative_residual = std::numeric_limits<double>::infinity();
            if (std::isfinite(product) && std::isfinite(scale) && std::isfinite(residual)) {
                relative_residual = scale == 0.0 ? 0.0 : std::abs(residual) / scale;
            }
            scaled_residual[column] = std::max(scaled_residual[column], relative_residual);
        }
    }
    // With FACT='N', DGESVX does not equilibrate: WORK(1) is norm(A)/norm(U)
    // for the original doubled operator, so no conversion from scaled units is needed.
    detail::validate_dense_response_diagnostics(reciprocal_condition, work[0], forward_error,
                                                backward_error, scaled_residual);

    auto P = std::make_shared<Matrix>(transition_count, rhs_count);
    auto Q = std::make_shared<Matrix>(transition_count, rhs_count);
    std::vector<double> solution_column_scales(column_count, 0.0);
    for (int column = 0; column < rhs_count; ++column) {
        for (int row = 0; row < transition_count; ++row) {
            const double p = solution[row + static_cast<std::size_t>(column) * dimension];
            const double q = omega == 0.0 ? 0.0 :
                solution[row + transition_count + static_cast<std::size_t>(column) * dimension];
            if (!std::isfinite(p) || !std::isfinite(q))
                throw PSIEXCEPTION("dense restricted response: solution amplitudes are not finite");
            (*P)(row, column) = p;
            (*Q)(row, column) = q;
            solution_column_scales[column] = std::max(
                {solution_column_scales[column], std::abs(p), std::abs(q)});
        }
    }
    return DenseRestrictedResponse(
        std::move(P), std::move(Q), reciprocal_condition, work[0],
        std::move(forward_error), std::move(backward_error),
        std::move(scaled_residual), std::move(solution_column_scales));
}

ResponseKernel::ResponseKernel(double chf_exchange, double alda_kernel)
    : chf_exchange_(chf_exchange), alda_kernel_(alda_kernel) {
    if (!std::isfinite(chf_exchange_) || chf_exchange_ != 0.25)
        throw PSIEXCEPTION("ResponseKernel: CHF exchange coefficient must be exactly 0.25");
    if (!std::isfinite(alda_kernel_) || alda_kernel_ != 0.75)
        throw PSIEXCEPTION("ResponseKernel: ALDA coefficient must be exactly 0.75");
}

namespace detail {
RestrictedSingletHessian assemble_restricted_singlet_hessian(
    const std::vector<double>& orbital_gaps, const Matrix& coulomb,
    const Matrix& exchange_direct, const Matrix& exchange_transpose,
    const Matrix& full_alda, const ResponseKernel& kernel) {
    if (orbital_gaps.empty())
        throw PSIEXCEPTION("restricted singlet Hessian: orbital gaps must be nonempty");
    for (double gap : orbital_gaps) {
        if (!std::isfinite(gap))
            throw PSIEXCEPTION("restricted singlet Hessian: orbital gaps must be finite");
        if (!(gap > 0.0))
            throw PSIEXCEPTION("restricted singlet Hessian: orbital gaps must be positive");
    }
    const auto transition_count = orbital_gaps.size();
    require_restricted_hessian_primitive(coulomb, transition_count, "Coulomb J");
    require_restricted_hessian_primitive(exchange_direct, transition_count, "K_direct");
    require_restricted_hessian_primitive(exchange_transpose, transition_count, "K_transpose");
    require_restricted_hessian_primitive(full_alda, transition_count, "full ALDA kernel");

    const double a = kernel.chf_exchange();
    const double b = kernel.alda_kernel();
    auto H1 = std::make_shared<Matrix>(transition_count, transition_count);
    auto H2 = std::make_shared<Matrix>(transition_count, transition_count);
    for (std::size_t row = 0; row < transition_count; ++row) {
        for (std::size_t column = 0; column < transition_count; ++column) {
            const double gap = row == column ? orbital_gaps[row] : 0.0;
            const double h1 = gap + 4.0 * coulomb(row, column) -
                              a * (exchange_direct(row, column) +
                                   exchange_transpose(row, column)) +
                              4.0 * b * full_alda(row, column);
            const double h2 = gap - a * exchange_direct(row, column) +
                              a * exchange_transpose(row, column);
            if (!std::isfinite(h1) || !std::isfinite(h2))
                throw PSIEXCEPTION("restricted singlet Hessian: assembled values must be finite");
            (*H1)(row, column) = h1;
            (*H2)(row, column) = h2;
        }
    }
    // The formulas preserve symmetry because each distinctly indexed transition
    // primitive was independently required to be symmetric above.
    require_dense_response_operator(*H1, "assembled H1");
    require_dense_response_operator(*H2, "assembled H2");
    return {std::move(H1), std::move(H2)};
}
}  // namespace detail

FrozenResponseContext::FrozenResponseContext(
    SharedMatrix Ca, SharedMatrix Cb, SharedVector epsilon_a, SharedVector epsilon_b,
    SharedVector occupation_a, SharedVector occupation_b, SharedMatrix Da, SharedMatrix Db,
    double energy, std::shared_ptr<const Molecule> molecule, std::shared_ptr<const BasisSet> basis,
    std::shared_ptr<const BasisSetStructuralSnapshot> basis_snapshot,
    std::shared_ptr<const SuperFunctional> functional, std::vector<SitePosition> sites,
    std::vector<double> grid_points, std::vector<double> grid_weights,
    std::vector<FrozenGridBlock> grid_blocks, GRACProvenance grac, std::string functional_name,
    std::string grac_x_name, std::string grac_c_name, double functional_density_tolerance)
    : Ca_(std::move(Ca)), Cb_(std::move(Cb)), epsilon_a_(std::move(epsilon_a)),
      epsilon_b_(std::move(epsilon_b)), occupation_a_(std::move(occupation_a)),
      occupation_b_(std::move(occupation_b)), Da_(std::move(Da)), Db_(std::move(Db)), energy_(energy),
      molecule_(std::move(molecule)), basis_(std::move(basis)), basis_snapshot_(std::move(basis_snapshot)),
      functional_(std::move(functional)),
      sites_(std::move(sites)), grid_points_(std::move(grid_points)), grid_weights_(std::move(grid_weights)),
      grid_blocks_(std::move(grid_blocks)), grac_(std::move(grac)),
      functional_name_(std::move(functional_name)), grac_x_name_(std::move(grac_x_name)),
      grac_c_name_(std::move(grac_c_name)), functional_density_tolerance_(functional_density_tolerance) {}

void FrozenResponseContext::verify_basis_unchanged() const {
    if (!basis_ || !basis_snapshot_ || basis_->structural_snapshot() != *basis_snapshot_)
        throw PSIEXCEPTION("FrozenResponseContext: retained basis changed after provenance sealing");
}

namespace {
std::size_t checked_c1_product(std::size_t first, std::size_t second, const std::string& prefix) {
    if (first != 0 && second > std::numeric_limits<std::size_t>::max() / first)
        throw PSIEXCEPTION(prefix + "allocation size overflow");
    return first * second;
}

std::size_t checked_c1_sum(std::size_t first, std::size_t second, const std::string& prefix) {
    if (second > std::numeric_limits<std::size_t>::max() - first)
        throw PSIEXCEPTION(prefix + "allocation size overflow");
    return first + second;
}
}  // namespace

namespace detail {
PointResponsePlan plan_point_response(
    std::size_t frequency_count, std::size_t nbf, std::size_t nocc,
    std::size_t nvir, std::size_t point_count, bool has_dynamic_frequency,
    std::size_t memory_bytes) {
    const std::string prefix = "point response: ";
    constexpr std::size_t max_point_count = 500;
    constexpr std::size_t max_frequency_count = 64;
    constexpr std::size_t max_transition_count = 512;
    if (frequency_count == 0)
        throw PSIEXCEPTION(prefix + "frequency list requires at least one value");
    if (frequency_count > max_frequency_count)
        throw PSIEXCEPTION(prefix +
                           "frequency count exceeds the canonical 64-frequency envelope");
    if (nbf == 0 || nocc == 0 || nvir == 0)
        throw PSIEXCEPTION(prefix + "resource estimate requires nonzero orbital dimensions");
    if (point_count == 0)
        throw PSIEXCEPTION(prefix + "requires at least one point");
    if (point_count > max_point_count)
        throw PSIEXCEPTION(prefix + "point count exceeds the canonical 500-point envelope");
    const auto nov = checked_c1_product(nocc, nvir, prefix);
    if (nov > max_transition_count)
        throw PSIEXCEPTION(prefix + "transition count exceeds the canonical 512-transition envelope");
    if (nbf > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        nov > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        point_count > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw PSIEXCEPTION(prefix + "dimensions exceed native matrix limits");

    const auto reserved_memory_bytes = memory_bytes / 2;
    const auto bytes = [&prefix](std::size_t count) {
        return checked_c1_product(count, sizeof(double), prefix);
    };
    const auto ao_matrix_bytes = bytes(checked_c1_product(nbf, nbf, prefix));
    const auto transition_potential_bytes = bytes(
        checked_c1_product(nov, point_count, prefix));
    const auto point_square = checked_c1_product(point_count, point_count, prefix);
    const auto output_bytes = bytes(checked_c1_product(
        frequency_count, point_square, prefix));
    // The underscored Python carrier export clones every response matrix while
    // the immutable carrier remains live, so reserve a second full payload.
    const auto output_clone_bytes = output_bytes;
    const auto retained_frequency_bytes = bytes(frequency_count);
    const auto retained_points_bytes = checked_c1_product(
        point_count, sizeof(SitePosition), prefix);
    const auto native_diagnostic_record_bytes = sizeof(PointResponseDiagnostics);
    const auto native_diagnostics_bytes = checked_c1_product(
        frequency_count, native_diagnostic_record_bytes, prefix);

    // Native vector/shared-pointer/Matrix objects and their row-pointer arrays
    // are retained alongside the numeric payloads. Count carrier and Python
    // response clones plus carrier/clone/transpose transition matrices.
    auto container_overhead_bytes = checked_c1_sum(
        sizeof(std::vector<double>), sizeof(std::vector<SitePosition>), prefix);
    container_overhead_bytes = checked_c1_sum(
        container_overhead_bytes,
        checked_c1_product(2, sizeof(std::vector<SharedMatrix>), prefix), prefix);
    container_overhead_bytes = checked_c1_sum(
        container_overhead_bytes,
        sizeof(std::vector<PointResponseDiagnostics>), prefix);
    const auto response_matrix_objects = checked_c1_product(
        checked_c1_product(2, frequency_count, prefix),
        sizeof(SharedMatrix) + sizeof(Matrix), prefix);
    const auto transition_matrix_objects = checked_c1_product(
        3, sizeof(SharedMatrix) + sizeof(Matrix), prefix);
    const auto transient_matrix_objects = checked_c1_product(
        4, sizeof(SharedMatrix) + sizeof(Matrix), prefix);
    const auto response_row_pointers = checked_c1_product(
        checked_c1_product(checked_c1_product(2, frequency_count, prefix),
                           point_count, prefix), sizeof(double*), prefix);
    const auto transition_row_pointers = checked_c1_product(
        checked_c1_sum(checked_c1_product(2, nov, prefix), point_count, prefix),
        sizeof(double*), prefix);
    const auto transient_row_pointers = checked_c1_product(
        checked_c1_sum(checked_c1_product(3, nov, prefix), point_count, prefix),
        sizeof(double*), prefix);
    container_overhead_bytes = checked_c1_sum(
        container_overhead_bytes, response_matrix_objects, prefix);
    container_overhead_bytes = checked_c1_sum(
        container_overhead_bytes, transition_matrix_objects, prefix);
    container_overhead_bytes = checked_c1_sum(
        container_overhead_bytes, transient_matrix_objects, prefix);
    container_overhead_bytes = checked_c1_sum(
        container_overhead_bytes, response_row_pointers, prefix);
    container_overhead_bytes = checked_c1_sum(
        container_overhead_bytes, transition_row_pointers, prefix);
    container_overhead_bytes = checked_c1_sum(
        container_overhead_bytes, transient_row_pointers, prefix);
    auto retained_metadata_bytes = checked_c1_sum(
        retained_frequency_bytes, retained_points_bytes, prefix);
    retained_metadata_bytes = checked_c1_sum(
        retained_metadata_bytes, native_diagnostics_bytes, prefix);
    retained_metadata_bytes = checked_c1_sum(
        retained_metadata_bytes, container_overhead_bytes, prefix);

    constexpr std::size_t python_scalar_diagnostic_bytes_per_frequency = 512;
    constexpr std::size_t python_fixed_metadata_bytes = 1024;
    constexpr std::size_t python_frequency_object_bytes = 32;
    constexpr std::size_t python_point_object_bytes = 128;
    const auto python_scalar_diagnostic_overhead_bytes = checked_c1_product(
        frequency_count, python_scalar_diagnostic_bytes_per_frequency, prefix);
    auto python_metadata_overhead_bytes = checked_c1_sum(
        python_fixed_metadata_bytes,
        checked_c1_product(frequency_count, python_frequency_object_bytes, prefix),
        prefix);
    python_metadata_overhead_bytes = checked_c1_sum(
        python_metadata_overhead_bytes,
        checked_c1_product(point_count, python_point_object_bytes, prefix), prefix);
    const auto python_export_overhead_bytes = checked_c1_sum(
        python_scalar_diagnostic_overhead_bytes,
        python_metadata_overhead_bytes, prefix);

    const auto order = checked_c1_product(has_dynamic_frequency ? 2 : 1, nov, prefix);
    const auto order_square = checked_c1_product(order, order, prefix);
    const auto order_rhs = checked_c1_product(order, point_count, prefix);
    auto dense_solve_peak_bytes = bytes(checked_c1_product(3, order_square, prefix));
    dense_solve_peak_bytes = checked_c1_sum(
        dense_solve_peak_bytes, bytes(checked_c1_product(3, order_rhs, prefix)), prefix);
    dense_solve_peak_bytes = checked_c1_sum(
        dense_solve_peak_bytes,
        bytes(checked_c1_product(2, checked_c1_product(nov, point_count, prefix), prefix)),
        prefix);
    dense_solve_peak_bytes = checked_c1_sum(
        dense_solve_peak_bytes,
        bytes(checked_c1_sum(checked_c1_product(12, order, prefix),
                             checked_c1_product(4, point_count, prefix), prefix)), prefix);

    const auto operator_bytes = bytes(checked_c1_product(
        2, checked_c1_product(nov, nov, prefix), prefix));
    const auto scratch_bytes = bytes(point_square);
    auto estimated_bytes = checked_c1_sum(output_bytes, output_clone_bytes, prefix);
    estimated_bytes = checked_c1_sum(estimated_bytes, transition_potential_bytes, prefix);
    estimated_bytes = checked_c1_sum(estimated_bytes, ao_matrix_bytes, prefix);
    estimated_bytes = checked_c1_sum(estimated_bytes, dense_solve_peak_bytes, prefix);
    estimated_bytes = checked_c1_sum(estimated_bytes, scratch_bytes, prefix);
    estimated_bytes = checked_c1_sum(estimated_bytes, operator_bytes, prefix);
    estimated_bytes = checked_c1_sum(
        estimated_bytes, retained_metadata_bytes, prefix);
    estimated_bytes = checked_c1_sum(
        estimated_bytes, python_export_overhead_bytes, prefix);
    if (estimated_bytes > reserved_memory_bytes)
        throw PSIEXCEPTION(prefix + "estimated storage exceeds reserved memory");

    PointResponsePlan plan;
    plan.frequency_count = frequency_count;
    plan.nbf = nbf;
    plan.nocc = nocc;
    plan.nvir = nvir;
    plan.transition_count = nov;
    plan.point_count = point_count;
    plan.max_point_count = max_point_count;
    plan.max_frequency_count = max_frequency_count;
    plan.configured_memory_bytes = memory_bytes;
    plan.reserved_memory_bytes = reserved_memory_bytes;
    plan.ao_matrix_bytes = ao_matrix_bytes;
    plan.transition_potential_bytes = transition_potential_bytes;
    plan.output_bytes = output_bytes;
    plan.output_clone_bytes = output_clone_bytes;
    plan.retained_frequency_bytes = retained_frequency_bytes;
    plan.retained_points_bytes = retained_points_bytes;
    plan.native_diagnostic_record_bytes = native_diagnostic_record_bytes;
    plan.native_diagnostics_bytes = native_diagnostics_bytes;
    plan.container_overhead_bytes = container_overhead_bytes;
    plan.retained_metadata_bytes = retained_metadata_bytes;
    plan.python_scalar_diagnostic_overhead_bytes =
        python_scalar_diagnostic_overhead_bytes;
    plan.python_metadata_overhead_bytes = python_metadata_overhead_bytes;
    plan.python_export_overhead_bytes = python_export_overhead_bytes;
    plan.dense_solve_peak_bytes = dense_solve_peak_bytes;
    plan.scratch_bytes = scratch_bytes;
    plan.hessian_bytes = operator_bytes;
    plan.estimated_bytes = estimated_bytes;
    plan.integral_work_terms = checked_c1_product(point_count,
                                                   checked_c1_product(nbf, nbf, prefix), prefix);
    plan.algorithm = "CALLER_POINTS_NATIVE_ORDER0_AO_POTENTIAL_DENSE_RESPONSE";
    plan.memory_semantics =
        "SIMULTANEOUS_LIVE_STORAGE_HARD_GATE_HALF_PROCESS_MEMORY_PYTHON_CLONES_SCALAR_DIAGNOSTICS_FREQ64";
    return plan;
}

PointResponsePlan plan_point_response_provider(
    std::size_t frequency_count, std::size_t nbf, std::size_t nocc,
    std::size_t nvir, std::size_t point_count,
    const std::vector<FrozenGridBlock>& blocks, bool has_dynamic_frequency,
    std::size_t memory_bytes, double density_cutoff) {
    const std::string prefix = "point response: ";
    // Acquire overflow-checked point-stage components without applying the
    // standalone all-live gate; the aggregate stage peaks below are the sole
    // production point-response memory claim.
    const auto point_plan = plan_point_response(
        frequency_count, nbf, nocc, nvir, point_count,
        has_dynamic_frequency, std::numeric_limits<std::size_t>::max());
    if (blocks.empty())
        throw PSIEXCEPTION(prefix + "canonical ALDA grid blocks are unavailable");
    std::size_t grid_point_count = 0;
    for (const auto& block : blocks)
        grid_point_count = checked_c1_sum(grid_point_count, block.point_count, prefix);
    if (grid_point_count == 0)
        throw PSIEXCEPTION(prefix + "canonical ALDA grid is empty");

    const auto c1_plan = plan_restricted_c1_jk(nbf, nocc, nvir, memory_bytes);
    const auto alda_plan = plan_restricted_alda(
        nbf, nocc, nvir, grid_point_count, blocks, memory_bytes, false,
        density_cutoff);
    const auto nov = point_plan.transition_count;
    const auto matrix_bytes = checked_c1_product(
        checked_c1_product(nov, nov, prefix), sizeof(double), prefix);
    const auto retained_c1_bytes = checked_c1_product(3, matrix_bytes, prefix);
    const auto retained_alda_bytes = matrix_bytes;
    const auto hessian_bytes = checked_c1_product(2, matrix_bytes, prefix);
    const auto transition_metadata_bytes = checked_c1_product(
        nov, 5 * sizeof(std::size_t) + sizeof(double), prefix);
    constexpr std::size_t conservative_overhead_bytes = 1024ULL * 1024ULL;
    const auto persistent = checked_c1_sum(
        transition_metadata_bytes, conservative_overhead_bytes, prefix);
    const auto add_stage = [&prefix](std::initializer_list<std::size_t> terms) {
        std::size_t value = 0;
        for (const auto term : terms) value = checked_c1_sum(value, term, prefix);
        return value;
    };
    const auto retained_operator = add_stage(
        {retained_c1_bytes, retained_alda_bytes, hessian_bytes, persistent});
    const auto c1_stage_peak_bytes = c1_plan.estimated_bytes;
    const auto alda_stage_peak_bytes = add_stage(
        {retained_c1_bytes, alda_plan.estimated_bytes, persistent});
    const auto point_potential_stage_peak_bytes = add_stage(
        {retained_operator, point_plan.retained_metadata_bytes,
         point_plan.ao_matrix_bytes, point_plan.transition_potential_bytes});
    const auto dense_solve_stage_peak_bytes = add_stage(
        {retained_operator, point_plan.retained_metadata_bytes,
         point_plan.transition_potential_bytes, point_plan.output_bytes,
         point_plan.dense_solve_peak_bytes, point_plan.scratch_bytes});
    // The underscored export retains the carrier transition matrix while a
    // clone and its point-major transpose are both live.
    const auto output_clone_stage_peak_bytes = add_stage(
        {retained_operator, point_plan.retained_metadata_bytes,
         checked_c1_product(3, point_plan.transition_potential_bytes, prefix),
         point_plan.output_bytes, point_plan.output_clone_bytes,
         point_plan.python_export_overhead_bytes});
    const auto estimated_bytes = std::max(
        {c1_stage_peak_bytes, alda_stage_peak_bytes,
         point_potential_stage_peak_bytes, dense_solve_stage_peak_bytes,
         output_clone_stage_peak_bytes});
    const auto reserved_memory_bytes = memory_bytes / 2;
    if (estimated_bytes > reserved_memory_bytes)
        throw PSIEXCEPTION(prefix +
                           "aggregate stage peak exceeds the half-memory reservation");

    auto plan = point_plan;
    plan.configured_memory_bytes = memory_bytes;
    plan.reserved_memory_bytes = reserved_memory_bytes;
    plan.c1_plan_estimated_bytes = c1_plan.estimated_bytes;
    plan.alda_plan_estimated_bytes = alda_plan.estimated_bytes;
    plan.retained_c1_bytes = retained_c1_bytes;
    plan.retained_alda_bytes = retained_alda_bytes;
    plan.hessian_bytes = hessian_bytes;
    plan.transition_metadata_bytes = transition_metadata_bytes;
    plan.conservative_overhead_bytes = conservative_overhead_bytes;
    plan.c1_stage_peak_bytes = c1_stage_peak_bytes;
    plan.alda_stage_peak_bytes = alda_stage_peak_bytes;
    plan.point_potential_stage_peak_bytes = point_potential_stage_peak_bytes;
    plan.dense_solve_stage_peak_bytes = dense_solve_stage_peak_bytes;
    plan.output_clone_stage_peak_bytes = output_clone_stage_peak_bytes;
    plan.estimated_bytes = estimated_bytes;
    plan.algorithm = "CANONICAL_C1_ALDA_HESSIAN_CALLER_POINTS_ORDER0_DENSE_RESPONSE";
    plan.memory_semantics =
        "KNOWN_STORAGE_HARD_GATE_DIRECT_JK_WORKSPACE_ADVISORY_PYTHON_CLONES_SCALAR_DIAGNOSTICS_FREQ64";
    return plan;
}

RestrictedC1JKPlan plan_restricted_c1_jk(std::size_t nbf, std::size_t nocc,
                                         std::size_t nvir, std::size_t memory_bytes) {
    const std::string prefix = "restricted C1 transition primitives: ";
    constexpr std::size_t max_supported_nov = 512;
    constexpr double integral_cutoff = 1.0e-15;
    if (nbf == 0 || nocc == 0 || nvir == 0)
        throw PSIEXCEPTION(prefix + "memory estimate requires nonzero orbital dimensions");
    const auto nov = checked_c1_product(nocc, nvir, prefix);
    if (nov > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw PSIEXCEPTION(prefix + "transition-space dimension exceeds native matrix limits");
    if (nov > max_supported_nov)
        throw PSIEXCEPTION(prefix + "supported transition envelope is at most " +
                           std::to_string(max_supported_nov) + " occupied-virtual pairs");

    // Reserve at most half of configured process memory because the frozen
    // context and wavefunction remain live. Only the unavoidable retained
    // payload is a hard gate: DirectJK exposes no supported peak estimator.
    const auto reserved_memory_bytes = memory_bytes / 2;
    const auto retained_elements = checked_c1_product(checked_c1_product(3, nov, prefix), nov, prefix);
    const auto retained_payload_bytes = checked_c1_product(retained_elements, sizeof(double), prefix);
    if (retained_payload_bytes > reserved_memory_bytes)
        throw PSIEXCEPTION(prefix + "retained dense transition payload exceeds reserved memory (" +
                           std::to_string(retained_payload_bytes) + " bytes required, " +
                           std::to_string(reserved_memory_bytes) + " bytes reserved)");

    // Batch and integral threads are fixed at one. The following components
    // deliberately expose the storage model used for protocol diagnostics.
    // The integral-engine term allows eight nbf^2 arrays plus one nbf vector;
    // it is advisory because the integral backend provides no allocation API.
    const auto ao_square = checked_c1_product(nbf, nbf, prefix);
    const auto metadata_bytes = checked_c1_product(nov, 2 * sizeof(std::size_t) + sizeof(double), prefix);
    const auto orbital_count = checked_c1_sum(nocc, nvir, prefix);
    const auto coefficient_bytes = checked_c1_product(
        checked_c1_product(nbf, orbital_count, prefix), sizeof(double), prefix);
    const auto row_pointer_count = checked_c1_sum(checked_c1_product(8, nbf, prefix),
                                                  checked_c1_product(3, nov, prefix), prefix);
    const auto matrix_overhead_bytes = checked_c1_sum(
        checked_c1_product(16, sizeof(Matrix), prefix),
        checked_c1_product(row_pointer_count, sizeof(double*), prefix), prefix);
    const auto jk_coefficient_bytes = checked_c1_product(
        checked_c1_product(2, nbf, prefix), sizeof(double), prefix);
    const auto jk_ao_bytes = checked_c1_product(
        checked_c1_product(3, ao_square, prefix), sizeof(double), prefix);
    // DirectJK.cc allocates 2*max_task^2 J and 8*max_task^2 K scratch for a
    // nonsymmetric density. max_task <= nbf gives this conservative term.
    const auto direct_jk_scratch_bytes = checked_c1_product(
        checked_c1_product(10, ao_square, prefix), sizeof(double), prefix);
    const auto integral_engine_elements = checked_c1_sum(
        checked_c1_product(8, ao_square, prefix), nbf, prefix);
    const auto integral_engine_allowance_bytes = checked_c1_product(
        integral_engine_elements, sizeof(double), prefix);
    // Three projected nov outputs coexist; one triplet intermediate is bounded
    // by nbf*max(nocc,nvir) for the fixed one-entry pass.
    auto projection_elements = checked_c1_product(3, nov, prefix);
    projection_elements = checked_c1_sum(
        projection_elements, checked_c1_product(nbf, std::max(nocc, nvir), prefix), prefix);
    const auto projection_bytes = checked_c1_product(projection_elements, sizeof(double), prefix);

    std::size_t estimated_bytes = retained_payload_bytes;
    for (const auto component : {metadata_bytes, coefficient_bytes, matrix_overhead_bytes,
                                 jk_coefficient_bytes, jk_ao_bytes, direct_jk_scratch_bytes,
                                 integral_engine_allowance_bytes, projection_bytes})
        estimated_bytes = checked_c1_sum(estimated_bytes, component, prefix);

    RestrictedC1JKPlan plan;
    plan.nbf = nbf;
    plan.nocc = nocc;
    plan.nvir = nvir;
    plan.nov = nov;
    plan.batch_size = 1;
    plan.jk_threads = 1;
    plan.max_supported_nov = max_supported_nov;
    plan.configured_memory_bytes = memory_bytes;
    plan.reserved_memory_bytes = reserved_memory_bytes;
    plan.retained_payload_bytes = retained_payload_bytes;
    plan.metadata_bytes = metadata_bytes;
    plan.coefficient_bytes = coefficient_bytes;
    plan.matrix_overhead_bytes = matrix_overhead_bytes;
    plan.jk_coefficient_bytes = jk_coefficient_bytes;
    plan.jk_ao_bytes = jk_ao_bytes;
    plan.direct_jk_scratch_bytes = direct_jk_scratch_bytes;
    plan.integral_engine_allowance_bytes = integral_engine_allowance_bytes;
    plan.projection_bytes = projection_bytes;
    plan.estimated_bytes = estimated_bytes;
    plan.integral_cutoff = integral_cutoff;
    plan.incfock = false;
    plan.screening = "NONE";
    plan.memory_semantics = "RETAINED_PAYLOAD_HARD_GATE_WORKSPACE_ADVISORY";
    plan.algorithm = "DIRECT_JK_CANONICAL_NONSYMMETRIC";
    return plan;
}
}  // namespace detail

namespace {
detail::RestrictedC1Primitives construct_restricted_c1_primitives_impl(
    const std::shared_ptr<const FrozenResponseContext>& context, const Matrix& Ca, const Matrix& Cb,
    const Vector& epsilon_a, const Vector& epsilon_b, const Vector& occupation_a,
    const Vector& occupation_b) {
    const std::string prefix = "restricted C1 transition primitives: ";
    if (!context) throw PSIEXCEPTION(prefix + "frozen response context is null");
    context->verify_basis_unchanged();
    const auto& basis = context->basis();
    if (!basis) throw PSIEXCEPTION(prefix + "retained orbital basis is unavailable");
    const int nbf = basis->nbf();
    if (Ca.nirrep() != 1 || Cb.nirrep() != 1 || epsilon_a.nirrep() != 1 ||
        epsilon_b.nirrep() != 1 || occupation_a.nirrep() != 1 || occupation_b.nirrep() != 1)
        throw PSIEXCEPTION(prefix + "only C1 orbital states are supported");
    if (nbf <= 0 || Ca.nrow() != nbf || Cb.nrow() != nbf || Ca.ncol() == 0 ||
        Cb.ncol() != Ca.ncol())
        throw PSIEXCEPTION(prefix + "orbital coefficient dimensions must match the retained basis");
    const int nmo = Ca.ncol();
    if (epsilon_a.dim(0) != nmo || epsilon_b.dim(0) != nmo ||
        occupation_a.dim(0) != nmo || occupation_b.dim(0) != nmo)
        throw PSIEXCEPTION(prefix + "orbital energy and occupation dimensions are inconsistent");

    for (int mu = 0; mu < nbf; ++mu) {
        for (int orbital = 0; orbital < nmo; ++orbital) {
            const double alpha = Ca(mu, orbital);
            const double beta = Cb(mu, orbital);
            if (!std::isfinite(alpha) || !std::isfinite(beta))
                throw PSIEXCEPTION(prefix + "orbital coefficients must be finite");
            if (alpha != beta)
                throw PSIEXCEPTION(prefix + "restricted Ca and Cb orbitals must match exactly");
        }
    }

    std::vector<int> occupied, virtuals;
    occupied.reserve(nmo);
    virtuals.reserve(nmo);
    for (int orbital = 0; orbital < nmo; ++orbital) {
        const double energy_a = epsilon_a.get(0, orbital);
        const double energy_b = epsilon_b.get(0, orbital);
        const double occ_a = occupation_a.get(0, orbital);
        const double occ_b = occupation_b.get(0, orbital);
        if (!std::isfinite(energy_a) || !std::isfinite(energy_b))
            throw PSIEXCEPTION(prefix + "orbital energies must be finite");
        if (energy_a != energy_b)
            throw PSIEXCEPTION(prefix + "restricted alpha and beta orbital energies must match exactly");
        if (!std::isfinite(occ_a) || !std::isfinite(occ_b) ||
            (occ_a != 0.0 && occ_a != 1.0) || (occ_b != 0.0 && occ_b != 1.0))
            throw PSIEXCEPTION(prefix + "closed-shell C1 requires integer occupations of zero or one");
        if (occ_a != occ_b)
            throw PSIEXCEPTION(prefix + "closed-shell alpha and beta occupations must match");
        (occ_a == 1.0 ? occupied : virtuals).push_back(orbital);
    }
    if (occupied.empty() || virtuals.empty())
        throw PSIEXCEPTION(prefix + "at least one occupied and one virtual orbital are required");
    const std::size_t nocc = occupied.size();
    const std::size_t nvir = virtuals.size();
    if (nocc > std::numeric_limits<std::size_t>::max() / nvir ||
        nocc * nvir > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw PSIEXCEPTION(prefix + "transition-space dimension exceeds native matrix limits");
    const std::size_t nov = nocc * nvir;
    // Fail before allocating any dense transition primitive or AO JK batch.
    const auto jk_plan = detail::plan_restricted_c1_jk(
        static_cast<std::size_t>(nbf), nocc, nvir, Process::environment.get_memory());

    detail::RestrictedC1Primitives result;
    result.jk_plan = jk_plan;
    result.transitions.reserve(nov);
    result.orbital_gaps.reserve(nov);
    // The transition order is explicit: occupied orbital first, virtual orbital second.
    // Both source-index lists are ascending, hence (i,a) occupied-major/virtual-minor.
    for (int i : occupied) {
        for (int a : virtuals) {
            const double gap = epsilon_a.get(0, a) - epsilon_a.get(0, i);
            if (!std::isfinite(gap) || !(gap > 0.0))
                throw PSIEXCEPTION(prefix + "all occupied-virtual orbital gaps must be finite and positive");
            result.transitions.emplace_back(i, a);
            result.orbital_gaps.push_back(gap);
        }
    }

    auto Co = std::make_shared<Matrix>(nbf, static_cast<int>(nocc));
    auto Cv = std::make_shared<Matrix>(nbf, static_cast<int>(nvir));
    for (int mu = 0; mu < nbf; ++mu) {
        for (std::size_t i = 0; i < nocc; ++i) (*Co)(mu, i) = Ca(mu, occupied[i]);
        for (std::size_t a = 0; a < nvir; ++a) (*Cv)(mu, a) = Ca(mu, virtuals[a]);
    }

    result.coulomb = std::make_shared<Matrix>(nov, nov);
    result.exchange_direct = std::make_shared<Matrix>(nov, nov);
    result.exchange_transpose = std::make_shared<Matrix>(nov, nov);

    // Use a fresh local Options registry, not a shallow copy of or reference to
    // Process options. SCREENING=NONE is a supported DirectJK value and disables
    // shell-quartet screening; the pinned 1e-15 integral-engine threshold is a
    // numeric protocol setting, not a claim of mathematical exactness. INCFOCK
    // is disabled and both batching and integral OpenMP execution are fixed at
    // one for deterministic storage and results. Caller options are never read
    // or mutated by this JK/integral factory path.
    //
    // JK defines D_ls=C_left_li C_right_si, J_mn=(mn|ls)D_ls, and
    // K_mn=(ml|ns)D_ls. Thus C_left=C_j and C_right=C_b yields
    // Co^T J Cv=(ia|jb), Co^T K Cv=(ij|ab), and
    // Cv^T K Co=(aj|ib)=(aj|bi).
    Options canonical_options;
    canonical_options.set_current_module("SCF");
    canonical_options.add_str("SCREENING", jk_plan.screening, "SCHWARZ CSAM DENSITY NONE");
    canonical_options.add_double("INTS_TOLERANCE", jk_plan.integral_cutoff);
    canonical_options.add_bool("INCFOCK", jk_plan.incfock);
    canonical_options.add_int("INCFOCK_FULL_FOCK_EVERY", 1);
    canonical_options.add_str("INTEGRAL_PACKAGE", "LIBINT2", "LIBINT2");
    auto mutable_basis = std::const_pointer_cast<BasisSet>(basis);
    auto jk = std::make_shared<DirectJK>(mutable_basis, canonical_options);
    jk->set_standard_integral_backend_only(true);
    jk->set_cutoff(jk_plan.integral_cutoff);
    jk->set_csam(false);
    jk->set_df_ints_num_threads(static_cast<int>(jk_plan.jk_threads));
    jk->set_do_J(true);
    jk->set_do_K(true);
    jk->set_do_wK(false);
    jk->set_print(0);
    jk->initialize();

    auto& left = jk->C_left();
    auto& right = jk->C_right();
    for (std::size_t start = 0; start < nov; start += jk_plan.batch_size) {
        const auto count = std::min(jk_plan.batch_size, nov - start);
        left.clear();
        right.clear();
        left.reserve(count);
        right.reserve(count);
        for (std::size_t entry = 0; entry < count; ++entry) {
            const auto source = start + entry;
            const auto j = source / nvir;
            const auto b = source % nvir;
            auto Cj = std::make_shared<Matrix>(nbf, 1);
            auto Cb_source = std::make_shared<Matrix>(nbf, 1);
            for (int mu = 0; mu < nbf; ++mu) {
                (*Cj)(mu, 0) = (*Co)(mu, j);
                (*Cb_source)(mu, 0) = (*Cv)(mu, b);
            }
            left.push_back(std::move(Cj));
            right.push_back(std::move(Cb_source));
        }
        jk->compute();
        const auto& J = jk->J();
        const auto& K = jk->K();
        if (J.size() != count || K.size() != count)
            throw PSIEXCEPTION(prefix + "direct-JK result cardinality is inconsistent");
        for (std::size_t entry = 0; entry < count; ++entry) {
            const auto source = start + entry;
            const auto J_ov = linalg::triplet(Co, J[entry], Cv, true, false, false);
            const auto Kd_ov = linalg::triplet(Co, K[entry], Cv, true, false, false);
            const auto Kt_vo = linalg::triplet(Cv, K[entry], Co, true, false, false);
            for (std::size_t i = 0; i < nocc; ++i) {
                for (std::size_t a = 0; a < nvir; ++a) {
                    const auto row = i * nvir + a;
                    (*result.coulomb)(row, source) = (*J_ov)(i, a);
                    (*result.exchange_direct)(row, source) = (*Kd_ov)(i, a);
                    (*result.exchange_transpose)(row, source) = (*Kt_vo)(a, i);
                }
            }
        }
        // JK vector interfaces are explicitly renewed on every bounded chunk.
        left.clear();
        right.clear();
    }
    result.integral_engine_thread_count = jk->integral_engine_thread_count();
    jk->finalize();
    require_restricted_hessian_primitive(*result.coulomb, nov, "Coulomb J");
    require_restricted_hessian_primitive(*result.exchange_direct, nov, "K_direct");
    require_restricted_hessian_primitive(*result.exchange_transpose, nov, "K_transpose");
    context->verify_basis_unchanged();
    return result;
}
}  // namespace

namespace detail {
RestrictedC1Primitives construct_restricted_c1_primitives(
    const std::shared_ptr<const FrozenResponseContext>& context) {
    if (!context)
        throw PSIEXCEPTION("restricted C1 transition primitives: frozen response context is null");
    return construct_restricted_c1_primitives_impl(
        context, *context->Ca(), *context->Cb(), *context->epsilon_a(), *context->epsilon_b(),
        *context->occupation_a(), *context->occupation_b());
}

RestrictedC1Primitives construct_restricted_c1_primitives_test_only(
    const std::shared_ptr<const FrozenResponseContext>& context, const Matrix& Ca, const Matrix& Cb,
    const Vector& epsilon_a, const Vector& epsilon_b, const Vector& occupation_a,
    const Vector& occupation_b) {
    return construct_restricted_c1_primitives_impl(
        context, Ca, Cb, epsilon_a, epsilon_b, occupation_a, occupation_b);
}
}  // namespace detail

namespace {
constexpr std::size_t kRestrictedALDAMaxNOV = 512;
constexpr const char* kRestrictedALDACutoffSource = "FROZEN_FUNCTIONAL_DENSITY_TOLERANCE";
// Conservative aggregate operation envelope covering the supported canonical
// closed-shell water/aug-cc-pVTZ response protocol, not only its DGEMM term.
constexpr std::size_t kRestrictedALDAMaxWorkTerms = 64ULL * 1024ULL * 1024ULL * 1024ULL;
constexpr const char* kALDAX = "XC_LDA_X";
constexpr const char* kALDAC = "XC_LDA_C_VWN";

class CompleteBlockOPoints final : public BlockOPoints {
   public:
    CompleteBlockOPoints(const SharedVector& x, const SharedVector& y, const SharedVector& z,
                         const SharedVector& w, const std::shared_ptr<BasisExtents>& extents,
                         const std::shared_ptr<BasisSet>& basis)
        : BlockOPoints(x, y, z, w, extents) {
        shells_local_to_global_.clear();
        functions_local_to_global_.clear();
        for (int shell = 0; shell < basis->nshell(); ++shell) shells_local_to_global_.push_back(shell);
        for (int function = 0; function < basis->nbf(); ++function)
            functions_local_to_global_.push_back(function);
        local_nbf_ = functions_local_to_global_.size();
    }
};

std::pair<std::shared_ptr<SuperFunctional>, detail::RestrictedALDADiagnostics>
build_restricted_alda_functional(std::size_t max_points, bool include_correlation,
                                  double density_cutoff) {
    if (max_points == 0 || max_points > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw PSIEXCEPTION("restricted ALDA kernel: invalid functional point capacity");
    if (!std::isfinite(density_cutoff) || !(density_cutoff > 0.0))
        throw PSIEXCEPTION("restricted ALDA kernel: frozen functional density tolerance must be finite and positive");
    auto exchange = std::make_shared<LibXCFunctional>(kALDAX, true);
    auto correlation = std::make_shared<LibXCFunctional>(kALDAC, true);
    exchange->set_alpha(1.0);
    correlation->set_alpha(1.0);
    auto functional = SuperFunctional::blank();
    functional->set_name("RESPONSE_LDA_X_VWN");
    functional->add_x_functional(exchange);
    if (include_correlation) functional->add_c_functional(correlation);
    functional->set_density_tolerance(density_cutoff);
    // RV skips total rho < cutoff, while LibXC suppresses each unpolarized spin
    // channel at <= its internal threshold. Preserve the exact positive frozen
    // tolerance on the dedicated SuperFunctional, but set only the component
    // implementation threshold just below cutoff/2 so total rho == cutoff is
    // evaluated at exactly rho. Production still explicitly skips rho < cutoff.
    const double libxc_threshold = std::nextafter(0.5 * density_cutoff, 0.0);
    exchange->set_density_cutoff(libxc_threshold);
    correlation->set_density_cutoff(libxc_threshold);
    functional->set_max_points(static_cast<int>(max_points));
    functional->set_deriv(2);
    functional->allocate();

    detail::RestrictedALDADiagnostics diagnostics;
    diagnostics.exchange_component = kALDAX;
    diagnostics.correlation_component = include_correlation ? kALDAC : "";
    diagnostics.exchange_libxc_id = exchange->libxc_id();
    diagnostics.correlation_libxc_id = include_correlation ? correlation->libxc_id() : 0;
    diagnostics.exchange_libxc_canonical_name = exchange->libxc_canonical_name();
    diagnostics.correlation_libxc_canonical_name =
        include_correlation ? correlation->libxc_canonical_name() : "";
    diagnostics.exchange_effective_parameters = exchange->effective_parameter_map();
    if (include_correlation)
        diagnostics.correlation_effective_parameters = correlation->effective_parameter_map();
    diagnostics.exchange_coefficient = exchange->alpha();
    diagnostics.correlation_coefficient = include_correlation ? correlation->alpha() : 0.0;
    diagnostics.derivative_order = functional->deriv();
    diagnostics.density_cutoff = density_cutoff;
    diagnostics.density_cutoff_source = kRestrictedALDACutoffSource;
    diagnostics.restricted_normalization =
        "rho=Da+Db; LibXC unpolarized d2E_xc/drho2; rho<cutoff skipped; no internal spin factor; C2a applies 4*b once";
    return {std::move(functional), std::move(diagnostics)};
}

void require_restricted_alda_work_bound(std::size_t work_terms) {
    if (work_terms > kRestrictedALDAMaxWorkTerms)
        throw PSIEXCEPTION("restricted ALDA kernel: canonical-water total work bound exceeded");
}

double restricted_alda_policy_density(double density, double cutoff) {
    if (!std::isfinite(density))
        throw PSIEXCEPTION("restricted ALDA kernel: density must be finite");
    if (!std::isfinite(cutoff) || !(cutoff > 0.0))
        throw PSIEXCEPTION("restricted ALDA kernel: density cutoff must be finite and positive");
    return density < cutoff ? cutoff : density;
}

std::pair<std::size_t, std::size_t> validate_restricted_alda_orbitals(
    const FrozenResponseContext& context) {
    const auto& Ca = *context.Ca();
    const auto& Cb = *context.Cb();
    const auto& occ_a = *context.occupation_a();
    const auto& occ_b = *context.occupation_b();
    const int nbf = context.basis()->nbf();
    if (Ca.nirrep() != 1 || Cb.nirrep() != 1 || occ_a.nirrep() != 1 || occ_b.nirrep() != 1 ||
        Ca.nrow() != nbf || Cb.nrow() != nbf || Ca.ncol() == 0 || Cb.ncol() != Ca.ncol() ||
        occ_a.dim(0) != Ca.ncol() || occ_b.dim(0) != Ca.ncol())
        throw PSIEXCEPTION("restricted ALDA kernel: malformed restricted orbital state");
    std::size_t nocc = 0, nvir = 0;
    for (int orbital = 0; orbital < Ca.ncol(); ++orbital) {
        for (int mu = 0; mu < nbf; ++mu)
            if (!std::isfinite(Ca(mu, orbital)) || Ca(mu, orbital) != Cb(mu, orbital))
                throw PSIEXCEPTION("restricted ALDA kernel: restricted Ca and Cb must be finite and identical");
        const double oa = occ_a.get(0, orbital), ob = occ_b.get(0, orbital);
        if (!std::isfinite(oa) || oa != ob || (oa != 0.0 && oa != 1.0))
            throw PSIEXCEPTION("restricted ALDA kernel: closed-shell occupations must be identical zero or one");
        (oa == 1.0 ? nocc : nvir)++;
    }
    if (nocc == 0 || nvir == 0)
        throw PSIEXCEPTION("restricted ALDA kernel: at least one occupied and virtual orbital are required");
    return {nocc, nvir};
}

std::vector<std::pair<std::size_t, std::size_t>> make_restricted_alda_transitions(
    const FrozenResponseContext& context, std::size_t nov) {
    std::vector<std::size_t> occupied, virtuals;
    occupied.reserve(context.Ca()->ncol());
    virtuals.reserve(context.Ca()->ncol());
    for (int orbital = 0; orbital < context.Ca()->ncol(); ++orbital)
        (context.occupation_a()->get(0, orbital) == 1.0 ? occupied : virtuals)
            .push_back(static_cast<std::size_t>(orbital));
    std::vector<std::pair<std::size_t, std::size_t>> transitions;
    transitions.reserve(nov);
    for (auto i : occupied)
        for (auto a : virtuals) transitions.emplace_back(i, a);
    return transitions;
}

// Allocation-free production preflight: only existing sealed storage is read.
// Duplicate detection is deliberately deferred until after the resource gate.
void preflight_restricted_alda_grid(std::size_t nbf, std::size_t point_count,
                                    const std::vector<double>& weights,
                                    const std::vector<FrozenGridBlock>& blocks) {
    if (nbf == 0 || point_count == 0 || weights.size() != point_count || blocks.empty())
        throw PSIEXCEPTION("restricted ALDA kernel: sealed grid dimensions are inconsistent");
    std::size_t expected_offset = 0;
    for (const auto& block : blocks) {
        if (block.point_offset != expected_offset || block.point_count == 0 ||
            expected_offset > point_count || block.point_count > point_count - expected_offset)
            throw PSIEXCEPTION("restricted ALDA kernel: malformed sealed block offsets");
        if (block.functions_local_to_global.empty() || block.functions_local_to_global.size() > nbf)
            throw PSIEXCEPTION("restricted ALDA kernel: sealed block function map is empty or oversized");
        for (int function : block.functions_local_to_global)
            if (function < 0 || static_cast<std::size_t>(function) >= nbf)
                throw PSIEXCEPTION("restricted ALDA kernel: malformed sealed local-to-global function map");
        expected_offset += block.point_count;
    }
    if (expected_offset != point_count)
        throw PSIEXCEPTION("restricted ALDA kernel: sealed block cardinality is inconsistent");
    for (double weight : weights)
        if (!std::isfinite(weight) || weight < 0.0)
            throw PSIEXCEPTION("restricted ALDA kernel: sealed weights must be finite and nonnegative");
}

// Called only after planning has reserved max(map size) * sizeof(int).
void validate_restricted_alda_duplicate_maps(const std::vector<FrozenGridBlock>& blocks) {
    for (const auto& block : blocks) {
        auto ordered = block.functions_local_to_global;
        std::sort(ordered.begin(), ordered.end());
        if (std::adjacent_find(ordered.begin(), ordered.end()) != ordered.end())
            throw PSIEXCEPTION("restricted ALDA kernel: malformed sealed local-to-global function map");
    }
}

void validate_restricted_alda_grid(std::size_t nbf, std::size_t point_count,
                                   const std::vector<double>& weights,
                                   const std::vector<FrozenGridBlock>& blocks) {
    preflight_restricted_alda_grid(nbf, point_count, weights, blocks);
    validate_restricted_alda_duplicate_maps(blocks);
}

std::shared_ptr<CompleteBlockOPoints> make_complete_alda_block(
    const FrozenResponseContext& context, const FrozenGridBlock& sealed,
    const std::shared_ptr<BasisExtents>& extents, const std::shared_ptr<BasisSet>& basis) {
    auto x = std::make_shared<Vector>("sealed x", sealed.point_count);
    auto y = std::make_shared<Vector>("sealed y", sealed.point_count);
    auto z = std::make_shared<Vector>("sealed z", sealed.point_count);
    auto w = std::make_shared<Vector>("sealed w", sealed.point_count);
    for (std::size_t local_point = 0; local_point < sealed.point_count; ++local_point) {
        const auto point = sealed.point_offset + local_point;
        x->set(local_point, context.grid_points()[3 * point]);
        y->set(local_point, context.grid_points()[3 * point + 1]);
        z->set(local_point, context.grid_points()[3 * point + 2]);
        w->set(local_point, context.grid_weights()[point]);
    }
    return std::make_shared<CompleteBlockOPoints>(x, y, z, w, extents, basis);
}

// Restricted normalization, derived directly from RV::compute_Vx_full:
// RKSFunctions forms rho=2*phi*Da*phi = phi*(Da+Db)*phi. For the nonsymmetric
// transition density D_ia=C_i C_a^T, compute_Vx_full forms
// rho_k=0.5*phi*(D_ia+D_ia^T)*phi=phi_i*phi_a. Its LDA intermediate has a
// 0.5 multiplier, then the explicit adjoint doubles it. Therefore the projected
// primitive is exactly integral w*f_xc(rho)*phi_i*phi_a*phi_j*phi_b, with no
// internal spin factor. The existing C2a formula alone applies 4*b (b=0.75).
SharedMatrix contract_restricted_alda(const std::vector<double>& weights,
                                      const Matrix& transition_values,
                                      const std::vector<double>& densities,
                                      const std::vector<double>& fxc,
                                      double density_cutoff) {
    const std::string prefix = "restricted ALDA contraction: ";
    if (!std::isfinite(density_cutoff) || !(density_cutoff > 0.0))
        throw PSIEXCEPTION(prefix + "density cutoff must be finite and positive");
    if (transition_values.nirrep() != 1 || transition_values.nrow() != weights.size() ||
        transition_values.ncol() == 0 || densities.size() != weights.size() || fxc.size() != weights.size())
        throw PSIEXCEPTION(prefix + "point arrays have inconsistent dimensions");
    const auto nov = static_cast<std::size_t>(transition_values.ncol());
    auto result = std::make_shared<Matrix>(nov, nov);
    for (std::size_t point = 0; point < weights.size(); ++point) {
        const double weight = weights[point], rho = densities[point];
        if (!std::isfinite(weight) || weight < 0.0)
            throw PSIEXCEPTION(prefix + "weights must be finite and nonnegative");
        if (!std::isfinite(rho)) throw PSIEXCEPTION(prefix + "density must be finite");
        if (weight == 0.0 || rho < density_cutoff) continue;
        const double kernel = fxc[point];
        if (!std::isfinite(kernel)) throw PSIEXCEPTION(prefix + "LibXC kernel values must be finite");
        for (std::size_t row = 0; row < nov; ++row) {
            const double left = transition_values(point, row);
            if (!std::isfinite(left)) throw PSIEXCEPTION(prefix + "transition values must be finite");
            for (std::size_t column = row; column < nov; ++column) {
                const double right = transition_values(point, column);
                if (!std::isfinite(right)) throw PSIEXCEPTION(prefix + "transition values must be finite");
                const double increment = weight * kernel * left * right;
                if (!std::isfinite(increment)) throw PSIEXCEPTION(prefix + "contraction overflowed");
                (*result)(row, column) += increment;
            }
        }
    }
    for (std::size_t row = 0; row < nov; ++row)
        for (std::size_t column = row + 1; column < nov; ++column)
            (*result)(column, row) = (*result)(row, column);
    require_restricted_hessian_primitive(*result, nov, "full ALDA");
    return result;
}
}  // namespace

namespace detail {
RestrictedALDAPlan plan_restricted_alda(std::size_t nbf, std::size_t nocc,
                                        std::size_t nvir, std::size_t point_count,
                                        const std::vector<FrozenGridBlock>& blocks,
                                        std::size_t memory_bytes,
                                        bool retain_test_diagnostics,
                                        double density_cutoff) {
    const std::string prefix = "restricted ALDA kernel: ";
    if (nbf == 0 || nocc == 0 || nvir == 0 || point_count == 0 || blocks.empty())
        throw PSIEXCEPTION(prefix + "plan dimensions/block metadata must be nonzero");
    if (!std::isfinite(density_cutoff) || !(density_cutoff > 0.0))
        throw PSIEXCEPTION(prefix + "frozen functional density tolerance must be finite and positive");
    if (nbf > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw PSIEXCEPTION(prefix + "plan dimensions exceed native integer limits");
    const auto nmo = checked_c1_sum(nocc, nvir, prefix);
    if (nmo > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw PSIEXCEPTION(prefix + "orbital dimension exceeds native integer limits");
    const auto nov = checked_c1_product(nocc, nvir, prefix);
    if (nov > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw PSIEXCEPTION(prefix + "transition dimension exceeds native matrix limits");
    if (nov > kRestrictedALDAMaxNOV)
        throw PSIEXCEPTION(prefix + "supported transition envelope is at most " +
                           std::to_string(kRestrictedALDAMaxNOV) + " occupied-virtual pairs");
    const auto nov_square = checked_c1_product(nov, nov, prefix);

    std::size_t covered_points = 0, max_block_points = 0, max_map_entries = 0, total_map_entries = 0;
    std::size_t density_work = 0, mo_transition_work = 0, ao_work = 0, libxc_work = 0, dgemm_work = 0;
    for (const auto& block : blocks) {
        const auto points = block.point_count;
        const auto map_size = block.functions_local_to_global.size();
        if (points == 0 || map_size == 0 || map_size > nbf ||
            points > static_cast<std::size_t>(std::numeric_limits<int>::max()))
            throw PSIEXCEPTION(prefix + "block metadata exceeds supported dimensions");
        covered_points = checked_c1_sum(covered_points, points, prefix);
        total_map_entries = checked_c1_sum(total_map_entries, map_size, prefix);
        max_block_points = std::max(max_block_points, points);
        max_map_entries = std::max(max_map_entries, map_size);
        density_work = checked_c1_sum(
            density_work, checked_c1_product(points, checked_c1_product(map_size, map_size, prefix), prefix), prefix);
        const auto mo_bound = std::max(nmo, nov);
        const auto per_point_mo = checked_c1_sum(
            checked_c1_product(map_size, mo_bound, prefix), nov, prefix);
        mo_transition_work = checked_c1_sum(
            mo_transition_work, checked_c1_product(points, per_point_mo, prefix), prefix);
        ao_work = checked_c1_sum(ao_work, checked_c1_product(points, nbf, prefix), prefix);
        libxc_work = checked_c1_sum(libxc_work, points, prefix);
        dgemm_work = checked_c1_sum(
            dgemm_work, checked_c1_product(points, nov_square, prefix), prefix);
    }
    if (covered_points != point_count)
        throw PSIEXCEPTION(prefix + "block metadata does not match point count");
    std::size_t work_terms = 0;
    for (const auto value : {density_work, mo_transition_work, ao_work, libxc_work, dgemm_work})
        work_terms = checked_c1_sum(work_terms, value, prefix);
    require_restricted_alda_work_bound(work_terms);

    const auto retained = checked_c1_product(nov_square, sizeof(double), prefix);
    const auto block_transition = checked_c1_product(
        checked_c1_product(max_block_points, nov, prefix), sizeof(double), prefix);
    const auto block_mo = checked_c1_product(
        checked_c1_product(max_block_points, nmo, prefix), sizeof(double), prefix);
    const auto collocation = checked_c1_product(
        checked_c1_product(checked_c1_product(2, max_block_points, prefix), nbf, prefix),
        sizeof(double), prefix);
    const auto coordinate_weight = checked_c1_product(
        checked_c1_product(4, max_block_points, prefix), sizeof(double), prefix);
    const auto density_kernel = checked_c1_product(
        checked_c1_product(8, max_block_points, prefix), sizeof(double), prefix);
    const auto functional_workspace = checked_c1_product(
        checked_c1_product(8, max_block_points, prefix), sizeof(double), prefix);
    const auto point_scratch = checked_c1_sum(
        checked_c1_sum(coordinate_weight, density_kernel, prefix), functional_workspace, prefix);
    // Conservatively reserve retained map entries plus complete block function/shell
    // maps, BasisExtents shell storage, and transition index pairs simultaneously.
    std::size_t metadata = checked_c1_product(total_map_entries, sizeof(int), prefix);
    metadata = checked_c1_sum(metadata, checked_c1_product(4 * sizeof(int), nbf, prefix), prefix);
    metadata = checked_c1_sum(
        metadata, checked_c1_product(checked_c1_product(2, nov, prefix), sizeof(std::size_t), prefix), prefix);
    metadata = checked_c1_sum(
        metadata, checked_c1_product(blocks.size(), sizeof(FrozenGridBlock), prefix), prefix);
    const auto validation_scratch = checked_c1_product(max_map_entries, sizeof(int), prefix);
    // Fixed reserve for Matrix/Vector objects, LibXC/SuperFunctional state,
    // BasisExtents engines, maps, and allocator bookkeeping not modeled above.
    constexpr std::size_t conservative_overhead = 1024ULL * 1024ULL;
    std::size_t streaming = retained;
    for (const auto value : {block_transition, block_transition, block_mo, collocation,
                             coordinate_weight, density_kernel, functional_workspace,
                             metadata, validation_scratch, conservative_overhead})
        streaming = checked_c1_sum(streaming, value, prefix);
    const auto reserved = memory_bytes / 2;
    if (streaming > reserved)
        throw PSIEXCEPTION(prefix + "conservative simultaneous live storage exceeds reserved memory");
    std::size_t diagnostic = 0;
    if (retain_test_diagnostics) {
        const auto diagnostic_elements = checked_c1_product(
            point_count, checked_c1_sum(nov, 2, prefix), prefix);
        diagnostic = checked_c1_product(diagnostic_elements, sizeof(double), prefix);
        if (diagnostic > reserved - streaming)
            throw PSIEXCEPTION(prefix + "diagnostic retention exceeds reserved memory");
    }
    RestrictedALDAPlan plan;
    plan.nbf = nbf;
    plan.nocc = nocc;
    plan.nvir = nvir;
    plan.nov = nov;
    plan.point_count = point_count;
    plan.max_block_points = max_block_points;
    plan.max_supported_nov = kRestrictedALDAMaxNOV;
    plan.configured_memory_bytes = memory_bytes;
    plan.reserved_memory_bytes = reserved;
    plan.retained_payload_bytes = retained;
    plan.block_transition_bytes = block_transition;
    plan.block_weighted_transition_bytes = block_transition;
    plan.block_mo_scratch_bytes = block_mo;
    plan.collocation_bytes = collocation;
    plan.block_coordinate_weight_bytes = coordinate_weight;
    plan.block_density_kernel_bytes = density_kernel;
    plan.functional_workspace_bytes = functional_workspace;
    plan.point_scratch_bytes = point_scratch;
    plan.metadata_bytes = metadata;
    plan.validation_scratch_bytes = validation_scratch;
    plan.conservative_overhead_bytes = conservative_overhead;
    plan.diagnostics_payload_bytes = diagnostic;
    plan.estimated_bytes = checked_c1_sum(streaming, diagnostic, prefix);
    plan.density_work_terms = density_work;
    plan.mo_transition_work_terms = mo_transition_work;
    plan.ao_collocation_work_terms = ao_work;
    plan.libxc_work_terms = libxc_work;
    plan.dgemm_work_terms = dgemm_work;
    plan.work_terms = work_terms;
    plan.max_work_terms = kRestrictedALDAMaxWorkTerms;
    plan.density_cutoff = density_cutoff;
    plan.retain_test_diagnostics = retain_test_diagnostics;
    plan.density_cutoff_source = kRestrictedALDACutoffSource;
    plan.algorithm = "SEALED_BLOCK_DGEMM";
    plan.memory_semantics = "CONSERVATIVE_SIMULTANEOUS_LIVE_RESERVATION";
    return plan;
}

void validate_restricted_alda_grid_test_only(std::size_t nbf, std::size_t point_count,
                                             const std::vector<double>& weights,
                                             const std::vector<FrozenGridBlock>& blocks) {
    validate_restricted_alda_grid(nbf, point_count, weights, blocks);
}

std::size_t validate_restricted_alda_work_bound_test_only(std::size_t work_terms) {
    require_restricted_alda_work_bound(work_terms);
    return work_terms;
}

std::pair<std::vector<double>, RestrictedALDADiagnostics> evaluate_restricted_alda_fxc_test_only(
    const std::vector<double>& densities, bool include_correlation,
    double density_cutoff) {
    if (densities.empty()) throw PSIEXCEPTION("restricted ALDA kernel: density array is empty");
    auto built = build_restricted_alda_functional(densities.size(), include_correlation, density_cutoff);
    auto rho = std::make_shared<Vector>("policy density", densities.size());
    for (std::size_t point = 0; point < densities.size(); ++point) {
        rho->set(point, restricted_alda_policy_density(densities[point], density_cutoff));
    }
    std::map<std::string, SharedVector> input{{"RHO_A", rho}};
    auto& values = built.first->compute_functional(input, static_cast<int>(densities.size()), true);
    const auto kernel = values.at("V_RHO_A_RHO_A");
    std::vector<double> result(densities.size());
    for (std::size_t point = 0; point < result.size(); ++point) {
        if (densities[point] < density_cutoff) continue;
        result[point] = kernel->get(point);
        if (!std::isfinite(result[point]))
            throw PSIEXCEPTION("restricted ALDA kernel: LibXC returned a nonfinite kernel");
    }
    built.second.point_count = densities.size();
    return {std::move(result), std::move(built.second)};
}

SharedMatrix contract_restricted_alda_test_only(
    const std::vector<double>& weights, const Matrix& transition_values,
    const std::vector<double>& densities, const std::vector<double>& fxc,
    double density_cutoff) {
    return contract_restricted_alda(weights, transition_values, densities, fxc, density_cutoff);
}

RestrictedALDAPrimitive construct_restricted_alda_kernel(
    const std::shared_ptr<const FrozenResponseContext>& context, bool retain_test_diagnostics) {
    const std::string prefix = "restricted ALDA kernel: ";
    if (!context) throw PSIEXCEPTION(prefix + "frozen response context is null");
    const auto basis_const = context->basis();
    if (!basis_const) throw PSIEXCEPTION(prefix + "retained basis is unavailable");
    const auto basis = std::const_pointer_cast<BasisSet>(basis_const);
    const int nbf = basis->nbf();
    if (nbf <= 0 || context->Da()->nirrep() != 1 || context->Db()->nirrep() != 1 ||
        context->Da()->nrow() != nbf || context->Da()->ncol() != nbf ||
        context->Db()->nrow() != nbf || context->Db()->ncol() != nbf)
        throw PSIEXCEPTION(prefix + "frozen density dimensions do not match the retained basis");
    const auto counts = validate_restricted_alda_orbitals(*context);
    const auto npoints = context->grid_point_count();
    if (npoints == 0 || npoints > std::numeric_limits<std::size_t>::max() / 3 ||
        context->grid_points().size() != 3 * npoints)
        throw PSIEXCEPTION(prefix + "sealed grid dimensions are inconsistent");
    preflight_restricted_alda_grid(static_cast<std::size_t>(nbf), npoints,
                                   context->grid_weights(), context->grid_blocks());
    for (double coordinate : context->grid_points())
        if (!std::isfinite(coordinate)) throw PSIEXCEPTION(prefix + "sealed coordinates must be finite");
    // Hard gates precede every dense output, block-transition, or diagnostic allocation.
    const auto plan = plan_restricted_alda(
        static_cast<std::size_t>(nbf), counts.first, counts.second, npoints,
        context->grid_blocks(), Process::environment.get_memory(), retain_test_diagnostics,
        context->functional_density_tolerance());
    // Structural snapshot construction is basis-sized, so it follows the
    // resource gate but still precedes duplicate validation and scientific work.
    context->verify_basis_unchanged();
    // The copied/sorted per-block duplicate scratch is bounded by the gated
    // validation_scratch_bytes; no nbf-sized seen vector is used.
    validate_restricted_alda_duplicate_maps(context->grid_blocks());
    const auto max_block_points = plan.max_block_points;
    auto transitions = make_restricted_alda_transitions(*context, plan.nov);
    auto built = build_restricted_alda_functional(max_block_points, true, plan.density_cutoff);
    built.second.plan = plan;
    built.second.point_count = npoints;
    auto extents = std::make_shared<BasisExtents>(basis, 1.0e-12);
    BasisFunctions collocation(basis, static_cast<int>(max_block_points), nbf);
    auto full_alda = std::make_shared<Matrix>(plan.nov, plan.nov);
    RestrictedALDAPrimitive result;
    result.full_alda = full_alda;
    result.diagnostics = built.second;
    if (retain_test_diagnostics) {
        result.densities.resize(npoints);
        result.fxc.resize(npoints);
        result.transition_values.resize(checked_c1_product(npoints, plan.nov, prefix));
    }
    const auto& Ca = *context->Ca();
    const auto& Da = *context->Da();
    const auto& Db = *context->Db();
    for (const auto& sealed : context->grid_blocks()) {
        auto block = make_complete_alda_block(*context, sealed, extents, basis);
        collocation.compute_functions(block);
        const auto phi = collocation.basis_value("PHI");
        Matrix transition_values(sealed.point_count, plan.nov);
        Matrix weighted_transition_values(sealed.point_count, plan.nov);
        Matrix orbital_values(sealed.point_count, Ca.ncol());
        std::vector<double> densities(sealed.point_count);
        auto functional_rho = std::make_shared<Vector>("policy density", sealed.point_count);
        for (std::size_t local_point = 0; local_point < sealed.point_count; ++local_point) {
            double rho = 0.0;
            for (int mu : sealed.functions_local_to_global)
                for (int nu : sealed.functions_local_to_global)
                    rho += (*phi)(local_point, mu) * (Da(mu, nu) + Db(mu, nu)) * (*phi)(local_point, nu);
            if (!std::isfinite(rho)) throw PSIEXCEPTION(prefix + "sealed-grid density is nonfinite");
            densities[local_point] = rho;
            functional_rho->set(
                local_point, restricted_alda_policy_density(rho, plan.density_cutoff));
            for (int orbital = 0; orbital < Ca.ncol(); ++orbital) {
                double value = 0.0;
                for (int mu : sealed.functions_local_to_global)
                    value += (*phi)(local_point, mu) * Ca(mu, orbital);
                if (!std::isfinite(value))
                    throw PSIEXCEPTION(prefix + "orbital collocation is nonfinite");
                orbital_values(local_point, orbital) = value;
            }
            for (std::size_t transition = 0; transition < plan.nov; ++transition) {
                const auto i = transitions[transition].first;
                const auto a = transitions[transition].second;
                transition_values(local_point, transition) =
                    orbital_values(local_point, i) * orbital_values(local_point, a);
            }
        }
        std::map<std::string, SharedVector> input{{"RHO_A", functional_rho}};
        auto& values = built.first->compute_functional(input, static_cast<int>(sealed.point_count), true);
        const auto block_fxc = values.at("V_RHO_A_RHO_A");
        for (std::size_t local_point = 0; local_point < sealed.point_count; ++local_point) {
            const auto point = sealed.point_offset + local_point;
            const bool active = densities[local_point] >= plan.density_cutoff &&
                                context->grid_weights()[point] != 0.0;
            const double kernel = active ? block_fxc->get(local_point) : 0.0;
            if (!std::isfinite(kernel)) throw PSIEXCEPTION(prefix + "LibXC returned a nonfinite kernel");
            const double factor = context->grid_weights()[point] * kernel;
            for (std::size_t transition = 0; transition < plan.nov; ++transition)
                weighted_transition_values(local_point, transition) =
                    factor * transition_values(local_point, transition);
            if (retain_test_diagnostics) {
                result.densities[point] = densities[local_point];
                result.fxc[point] = kernel;
                for (std::size_t transition = 0; transition < plan.nov; ++transition)
                    result.transition_values[point * plan.nov + transition] =
                        transition_values(local_point, transition);
            }
        }
        C_DGEMM('T', 'N', static_cast<int>(plan.nov), static_cast<int>(plan.nov),
                static_cast<int>(sealed.point_count), 1.0,
                weighted_transition_values.pointer()[0], static_cast<int>(plan.nov),
                transition_values.pointer()[0], static_cast<int>(plan.nov), 1.0,
                full_alda->pointer()[0], static_cast<int>(plan.nov));
    }
    for (std::size_t row = 0; row < plan.nov; ++row) {
        if (!std::isfinite((*full_alda)(row, row)))
            throw PSIEXCEPTION(prefix + "streamed contraction produced a nonfinite value");
        for (std::size_t column = row + 1; column < plan.nov; ++column) {
            const double value = 0.5 * ((*full_alda)(row, column) + (*full_alda)(column, row));
            if (!std::isfinite(value))
                throw PSIEXCEPTION(prefix + "streamed contraction produced a nonfinite value");
            (*full_alda)(row, column) = (*full_alda)(column, row) = value;
        }
    }
    require_restricted_hessian_primitive(*full_alda, plan.nov, "full ALDA");
    result.transitions = std::move(transitions);
    context->verify_basis_unchanged();
    return result;
}

RestrictedALDACollocationTestResult collocate_restricted_alda_ao_target_test_only(
    const std::shared_ptr<const FrozenResponseContext>& context) {
    const std::string prefix = "restricted ALDA AO parity target: ";
    if (!context || !context->basis()) throw PSIEXCEPTION(prefix + "frozen context/basis is unavailable");
    context->verify_basis_unchanged();
    const auto basis = std::const_pointer_cast<BasisSet>(context->basis());
    if (basis->nbf() <= 0) throw PSIEXCEPTION(prefix + "retained basis is empty");
    const auto nbf = static_cast<std::size_t>(basis->nbf());
    const auto npoints = context->grid_point_count();
    if (npoints == 0 || npoints > std::numeric_limits<std::size_t>::max() / 3 ||
        context->grid_points().size() != 3 * npoints)
        throw PSIEXCEPTION(prefix + "sealed coordinate dimensions are inconsistent");
    validate_restricted_alda_grid(nbf, npoints, context->grid_weights(), context->grid_blocks());
    std::size_t max_block_points = 0;
    for (const auto& block : context->grid_blocks())
        max_block_points = std::max(max_block_points, block.point_count);
    if (nbf > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        max_block_points > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw PSIEXCEPTION(prefix + "basis/block dimension exceeds native integer limits");
    const auto elements = checked_c1_product(npoints, nbf, prefix);
    const auto output_bytes = checked_c1_product(elements, sizeof(double), prefix);
    const auto block_elements = checked_c1_product(
        checked_c1_product(2, max_block_points, prefix), nbf, prefix);
    const auto block_bytes = checked_c1_product(block_elements, sizeof(double), prefix);
    const auto estimated_bytes = checked_c1_sum(output_bytes, block_bytes, prefix);
    if (estimated_bytes > Process::environment.get_memory() / 2)
        throw PSIEXCEPTION(prefix + "explicit test diagnostics exceed reserved memory");
    RestrictedALDACollocationTestResult result;
    result.point_count = npoints;
    result.nbf = nbf;
    result.ao_values.resize(elements);
    auto extents = std::make_shared<BasisExtents>(basis, 1.0e-12);
    BasisFunctions collocation(basis, static_cast<int>(max_block_points), static_cast<int>(nbf));
    for (const auto& sealed : context->grid_blocks()) {
        auto block = make_complete_alda_block(*context, sealed, extents, basis);
        collocation.compute_functions(block);
        const auto phi = collocation.basis_value("PHI");
        for (std::size_t local_point = 0; local_point < sealed.point_count; ++local_point)
            for (std::size_t function = 0; function < nbf; ++function)
                result.ao_values[(sealed.point_offset + local_point) * nbf + function] =
                    (*phi)(local_point, function);
    }
    context->verify_basis_unchanged();
    return result;
}

TransitionMultipoleProjectionPlan plan_transition_multipole_projection(
    std::size_t point_count, std::size_t site_count, std::size_t transition_count,
    std::size_t max_block_points, std::size_t nbf, std::size_t nmo,
    std::size_t memory_bytes) {
    const std::string prefix = "transition multipole projection: ";
    constexpr std::size_t max_sites = 64;
    constexpr std::size_t max_transitions = 512;
    constexpr std::size_t max_work = 64ULL * 1024ULL * 1024ULL * 1024ULL;
    constexpr std::size_t overhead = 1024ULL * 1024ULL;
    if (point_count == 0 || site_count == 0 || transition_count == 0 ||
        max_block_points == 0 || max_block_points > point_count)
        throw PSIEXCEPTION(prefix + "dimensions must be nonzero and block-bounded");
    if ((nbf == 0) != (nmo == 0))
        throw PSIEXCEPTION(prefix + "basis/orbital planning dimensions are inconsistent");
    if (site_count > max_sites)
        throw PSIEXCEPTION(prefix + "site count exceeds the supported canonical-molecule envelope");
    if (transition_count > max_transitions)
        throw PSIEXCEPTION(prefix + "transition count exceeds the supported response envelope");
    if (transition_count > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        checked_c1_product(site_count, 16, prefix) >
            static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw PSIEXCEPTION(prefix + "dimensions exceed native matrix limits");
    const auto output_elements = checked_c1_product(
        checked_c1_product(site_count, 16, prefix), transition_count, prefix);
    const auto output_bytes = checked_c1_product(output_elements, sizeof(double), prefix);
    auto block_elements = checked_c1_product(max_block_points, transition_count, prefix);
    if (nbf != 0) {
        block_elements = checked_c1_sum(
            block_elements, checked_c1_product(max_block_points, nmo, prefix), prefix);
        block_elements = checked_c1_sum(
            block_elements,
            checked_c1_product(checked_c1_product(2, max_block_points, prefix), nbf, prefix),
            prefix);
    }
    const auto block_scratch_bytes = checked_c1_product(block_elements, sizeof(double), prefix);
    auto estimated = checked_c1_sum(output_bytes, block_scratch_bytes, prefix);
    estimated = checked_c1_sum(estimated, overhead, prefix);
    const auto work = checked_c1_product(
        checked_c1_product(checked_c1_product(point_count, site_count, prefix), 16, prefix),
        transition_count, prefix);
    if (work > max_work) throw PSIEXCEPTION(prefix + "work bound exceeded");
    if (estimated > memory_bytes / 2)
        throw PSIEXCEPTION(prefix + "conservative simultaneous storage exceeds reserved memory");
    TransitionMultipoleProjectionPlan plan;
    plan.point_count = point_count;
    plan.site_count = site_count;
    plan.transition_count = transition_count;
    plan.max_block_points = max_block_points;
    plan.output_bytes = output_bytes;
    plan.block_scratch_bytes = block_scratch_bytes;
    plan.estimated_bytes = estimated;
    plan.work_terms = work;
    plan.max_work_terms = max_work;
    plan.max_site_count = max_sites;
    plan.algorithm = nbf == 0 ? "PURE_POINT_STREAM" : "SEALED_BLOCK_TAU_STREAM";
    return plan;
}

TransitionMultipoleProjection project_transition_multipoles(
    const std::vector<SitePosition>& points, const std::vector<double>& weights,
    const std::vector<double>& partition, const std::vector<SitePosition>& sites,
    const Matrix& transition_values) {
    const std::string prefix = "transition multipole projection: ";
    const auto npoints = points.size();
    const auto nsites = sites.size();
    if (npoints == 0 || nsites == 0 || weights.size() != npoints ||
        npoints > std::numeric_limits<std::size_t>::max() / nsites ||
        partition.size() != npoints * nsites || transition_values.nirrep() != 1 ||
        transition_values.nrow() != npoints || transition_values.ncol() == 0)
        throw PSIEXCEPTION(prefix + "input dimensions are inconsistent");
    const auto nov = static_cast<std::size_t>(transition_values.ncol());
    const auto plan = plan_transition_multipole_projection(
        npoints, nsites, nov, npoints, 0, 0, Process::environment.get_memory());
    for (const auto& point : points)
        for (double coordinate : point)
            if (!std::isfinite(coordinate)) throw PSIEXCEPTION(prefix + "points must be finite");
    for (const auto& site : sites)
        for (double coordinate : site)
            if (!std::isfinite(coordinate)) throw PSIEXCEPTION(prefix + "sites must be finite");
    for (std::size_t point = 0; point < npoints; ++point) {
        if (!std::isfinite(weights[point]) || weights[point] < 0.0)
            throw PSIEXCEPTION(prefix + "quadrature weights must be finite and nonnegative");
        double unity = 0.0;
        for (std::size_t site = 0; site < nsites; ++site) {
            const double value = partition[point * nsites + site];
            if (!std::isfinite(value) || value < 0.0)
                throw PSIEXCEPTION(prefix + "partition must be finite and nonnegative");
            unity += value;
        }
        if (!std::isfinite(unity) || std::abs(unity - 1.0) > kValidationTolerance)
            throw PSIEXCEPTION(prefix + "pointwise partition unity failed");
        for (std::size_t transition = 0; transition < nov; ++transition)
            if (!std::isfinite(transition_values(point, transition)))
                throw PSIEXCEPTION(prefix + "transition values must be finite");
    }
    auto values = std::make_shared<Matrix>(nsites * 16, nov);
    for (std::size_t point = 0; point < npoints; ++point) {
        for (std::size_t site = 0; site < nsites; ++site) {
            const double factor = weights[point] * partition[point * nsites + site];
            if (!std::isfinite(factor)) throw PSIEXCEPTION(prefix + "weight contraction overflowed");
            const SitePosition displacement{
                points[point][0] - sites[site][0], points[point][1] - sites[site][1],
                points[point][2] - sites[site][2]};
            const auto harmonics = regular_harmonics(displacement);
            for (std::size_t component = 0; component < 16; ++component) {
                const auto row = site * 16 + component;
                for (std::size_t transition = 0; transition < nov; ++transition) {
                    const double increment = factor * harmonics[component] *
                                             transition_values(point, transition);
                    if (!std::isfinite(increment))
                        throw PSIEXCEPTION(prefix + "multipole contraction overflowed");
                    (*values)(row, transition) += increment;
                    if (!std::isfinite((*values)(row, transition)))
                        throw PSIEXCEPTION(prefix + "multipole accumulation overflowed");
                }
            }
        }
    }
    TransitionMultipoleProjection result;
    result.values = std::move(values);
    result.plan = plan;
    return result;
}

ISAPolResponsePlan plan_isapol_response_provider(
    std::size_t frequency_count, std::size_t site_count,
    std::size_t nbf, std::size_t nocc, std::size_t nvir,
    std::size_t point_count, const std::vector<FrozenGridBlock>& blocks,
    bool has_dynamic_frequency, std::size_t memory_bytes,
    double density_cutoff) {
    const std::string prefix = "ISAPolResponseProvider: ";
    if (frequency_count == 0 || site_count == 0 || nbf == 0 || nocc == 0 ||
        nvir == 0 || point_count == 0 || blocks.empty())
        throw PSIEXCEPTION(prefix + "resource-plan dimensions must be nonzero");
    const auto transition_count = checked_c1_product(nocc, nvir, prefix);
    const auto nmo = checked_c1_sum(nocc, nvir, prefix);
    std::size_t covered_points = 0;
    std::size_t max_block_points = 0;
    for (const auto& block : blocks) {
        covered_points = checked_c1_sum(covered_points, block.point_count, prefix);
        max_block_points = std::max(max_block_points, block.point_count);
    }
    if (covered_points != point_count)
        throw PSIEXCEPTION(prefix + "resource-plan blocks do not cover the sealed point count");

    // Pure stage planners run before any physical stage output is constructed.
    // DirectJK's backend workspace is necessarily advisory, but its estimate is
    // still included in the aggregate C1 peak. All known retained allocations
    // below are hard-gated against the documented half-memory reservation.
    const auto c1_plan = plan_restricted_c1_jk(nbf, nocc, nvir, memory_bytes);
    const auto alda_plan = plan_restricted_alda(
        nbf, nocc, nvir, point_count, blocks, memory_bytes, false, density_cutoff);
    const auto projection_plan = plan_transition_multipole_projection(
        point_count, site_count, transition_count, max_block_points,
        nbf, nmo, memory_bytes);
    const auto contraction_plan = plan_site_pair_response_contraction(
        site_count, transition_count, memory_bytes);

    const auto component_count = checked_c1_product(site_count, 16, prefix);
    const auto transition_square = checked_c1_product(
        transition_count, transition_count, prefix);
    const auto transition_matrix_bytes = checked_c1_product(
        transition_square, sizeof(double), prefix);
    const auto retained_c1_bytes = checked_c1_product(
        3, transition_matrix_bytes, prefix);
    const auto retained_alda_bytes = transition_matrix_bytes;
    const auto hessian_bytes = checked_c1_product(
        2, transition_matrix_bytes, prefix);
    const auto identity_bytes = transition_matrix_bytes;
    const auto retained_projection_bytes = projection_plan.output_bytes;
    const auto response_carrier_bytes = checked_c1_product(
        2, transition_matrix_bytes, prefix);

    const auto component_square = checked_c1_product(
        component_count, component_count, prefix);
    const auto response_block_bytes = checked_c1_product(
        component_square, sizeof(double), prefix);
    const auto response_position_bytes = checked_c1_product(
        checked_c1_product(site_count, 3, prefix), sizeof(double), prefix);
    const auto retained_output_bytes = checked_c1_product(
        frequency_count,
        checked_c1_sum(response_block_bytes, response_position_bytes, prefix), prefix);

    // solve_dense_restricted_response retains at most twenty nov-square double
    // payloads for a doubled dynamic solve (eight for static), plus its bounded
    // diagnostic/scaling vectors. P/Q are part of this peak; only their two-map
    // carrier remains live when contraction starts.
    const std::size_t dense_matrix_count = has_dynamic_frequency ? 20 : 8;
    auto dense_solve_peak_bytes = checked_c1_product(
        dense_matrix_count, transition_matrix_bytes, prefix);
    dense_solve_peak_bytes = checked_c1_sum(
        dense_solve_peak_bytes,
        checked_c1_product(
            checked_c1_product(16, transition_count, prefix), sizeof(double), prefix), prefix);

    // C1 carries (i,a) and gaps; ALDA and projection each carry their own
    // ordered (i,a) vector. This is deliberately conservative even where an
    // individual stage planner already includes some of this metadata.
    const auto transition_metadata_entries = checked_c1_product(
        transition_count, 6 * sizeof(std::size_t) + sizeof(double), prefix);
    const auto transition_metadata_bytes = transition_metadata_entries;
    constexpr std::size_t conservative_overhead_bytes = 1024ULL * 1024ULL;
    const auto persistent_metadata_bytes = checked_c1_sum(
        transition_metadata_bytes, conservative_overhead_bytes, prefix);

    const auto add_stage = [&prefix](std::initializer_list<std::size_t> terms) {
        std::size_t value = 0;
        for (const auto term : terms) value = checked_c1_sum(value, term, prefix);
        return value;
    };
    const auto c1_stage_peak_bytes = c1_plan.estimated_bytes;
    const auto alda_stage_peak_bytes = add_stage(
        {retained_c1_bytes, alda_plan.estimated_bytes, persistent_metadata_bytes});
    const auto projection_stage_peak_bytes = add_stage(
        {retained_c1_bytes, retained_alda_bytes, hessian_bytes,
         projection_plan.estimated_bytes, persistent_metadata_bytes});
    const auto common_solve_retained = add_stage(
        {retained_c1_bytes, retained_alda_bytes, hessian_bytes,
         retained_projection_bytes, identity_bytes, retained_output_bytes,
         persistent_metadata_bytes});
    const auto dense_solve_stage_peak_bytes = checked_c1_sum(
        common_solve_retained, dense_solve_peak_bytes, prefix);
    const auto contraction_stage_peak_bytes = add_stage(
        {common_solve_retained, response_carrier_bytes,
         contraction_plan.estimated_bytes});
    const auto estimated_bytes = std::max(
        {c1_stage_peak_bytes, alda_stage_peak_bytes,
         projection_stage_peak_bytes, dense_solve_stage_peak_bytes,
         contraction_stage_peak_bytes});
    const auto reserved_memory_bytes = memory_bytes / 2;
    if (estimated_bytes > reserved_memory_bytes)
        throw PSIEXCEPTION(prefix +
                           "aggregate stage peak exceeds the half-memory reservation");

    ISAPolResponsePlan plan;
    plan.frequency_count = frequency_count;
    plan.site_count = site_count;
    plan.nbf = nbf;
    plan.nocc = nocc;
    plan.nvir = nvir;
    plan.transition_count = transition_count;
    plan.point_count = point_count;
    plan.max_block_points = max_block_points;
    plan.component_count = component_count;
    plan.configured_memory_bytes = memory_bytes;
    plan.reserved_memory_bytes = reserved_memory_bytes;
    plan.c1_plan_estimated_bytes = c1_plan.estimated_bytes;
    plan.alda_plan_estimated_bytes = alda_plan.estimated_bytes;
    plan.projection_plan_estimated_bytes = projection_plan.estimated_bytes;
    plan.contraction_plan_estimated_bytes = contraction_plan.estimated_bytes;
    plan.retained_c1_bytes = retained_c1_bytes;
    plan.retained_alda_bytes = retained_alda_bytes;
    plan.hessian_bytes = hessian_bytes;
    plan.retained_projection_bytes = retained_projection_bytes;
    plan.identity_bytes = identity_bytes;
    plan.retained_output_bytes = retained_output_bytes;
    plan.dense_solve_peak_bytes = dense_solve_peak_bytes;
    plan.response_carrier_bytes = response_carrier_bytes;
    plan.transition_metadata_bytes = transition_metadata_bytes;
    plan.conservative_overhead_bytes = conservative_overhead_bytes;
    plan.c1_stage_peak_bytes = c1_stage_peak_bytes;
    plan.alda_stage_peak_bytes = alda_stage_peak_bytes;
    plan.projection_stage_peak_bytes = projection_stage_peak_bytes;
    plan.dense_solve_stage_peak_bytes = dense_solve_stage_peak_bytes;
    plan.contraction_stage_peak_bytes = contraction_stage_peak_bytes;
    plan.estimated_bytes = estimated_bytes;
    plan.algorithm = "UPFRONT_C1_ALDA_ISA_DENSE_STAGE_MAX";
    plan.memory_semantics =
        "KNOWN_STORAGE_HARD_GATE_DIRECT_JK_WORKSPACE_ADVISORY";
    return plan;
}

SitePairResponseContractionPlan plan_site_pair_response_contraction(
    std::size_t site_count, std::size_t transition_count, std::size_t memory_bytes) {
    const std::string prefix = "site-pair response contraction: ";
    constexpr std::size_t max_sites = 64;
    constexpr std::size_t max_transitions = 512;
    constexpr std::size_t max_work = 64ULL * 1024ULL * 1024ULL * 1024ULL;
    if (site_count == 0) throw PSIEXCEPTION(prefix + "site count must be nonzero");
    if (transition_count == 0)
        throw PSIEXCEPTION(prefix + "transition count must be nonzero");
    const auto component_count = checked_c1_product(site_count, 16, prefix);
    const auto output_elements = checked_c1_product(component_count, component_count, prefix);
    const auto projected_response_elements =
        checked_c1_product(component_count, transition_count, prefix);
    const auto response_map_elements =
        checked_c1_product(transition_count, transition_count, prefix);
    const auto response_map_scratch_elements =
        checked_c1_product(2, response_map_elements, prefix);
    const auto scratch_elements =
        checked_c1_sum(projected_response_elements, response_map_scratch_elements, prefix);
    const auto output_bytes = checked_c1_product(output_elements, sizeof(double), prefix);
    const auto scratch_bytes = checked_c1_product(scratch_elements, sizeof(double), prefix);
    const auto estimated_bytes = checked_c1_sum(output_bytes, scratch_bytes, prefix);
    const auto first_work =
        checked_c1_product(projected_response_elements, transition_count, prefix);
    const auto second_work = checked_c1_product(output_elements, transition_count, prefix);
    const auto work_terms = checked_c1_sum(first_work, second_work, prefix);
    if (site_count > max_sites)
        throw PSIEXCEPTION(prefix + "site count exceeds the supported canonical-molecule envelope");
    if (transition_count > max_transitions)
        throw PSIEXCEPTION(prefix + "transition count exceeds the supported response envelope");
    if (component_count > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        transition_count > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw PSIEXCEPTION(prefix + "dimensions exceed native matrix limits");
    if (work_terms > max_work) throw PSIEXCEPTION(prefix + "work bound exceeded");
    if (estimated_bytes > memory_bytes)
        throw PSIEXCEPTION(prefix + "incremental numeric payload exceeds available memory");
    SitePairResponseContractionPlan plan;
    plan.site_count = site_count;
    plan.transition_count = transition_count;
    plan.component_count = component_count;
    plan.output_bytes = output_bytes;
    plan.scratch_bytes = scratch_bytes;
    plan.estimated_bytes = estimated_bytes;
    plan.work_terms = work_terms;
    plan.max_work_terms = max_work;
    plan.max_site_count = max_sites;
    plan.algorithm = "SYMMETRIC_SITE_COMPONENT_OUTER_PRODUCT";
    plan.memory_semantics =
        "INCREMENTAL_NUMERIC_PAYLOAD_CALLER_B_AND_DENSE_RESPONSE_EXCLUDED";
    return plan;
}

namespace {
std::vector<double> response_solution_column_scales(
    const Matrix& response_map, const Matrix& conjugate_map,
    const std::string& prefix) {
    if (response_map.nirrep() != 1 || response_map.nrow() <= 0 ||
        response_map.nrow() != response_map.ncol() || conjugate_map.nirrep() != 1 ||
        conjugate_map.nrow() != response_map.nrow() ||
        conjugate_map.ncol() != response_map.ncol())
        throw PSIEXCEPTION(prefix + "dense response carrier dimensions are inconsistent");
    std::vector<double> scales(static_cast<std::size_t>(response_map.ncol()), 0.0);
    for (int column = 0; column < response_map.ncol(); ++column) {
        for (int row = 0; row < response_map.nrow(); ++row) {
            const double p = response_map(row, column);
            const double q = conjugate_map(row, column);
            if (!std::isfinite(p) || !std::isfinite(q))
                throw PSIEXCEPTION(prefix + "dense response maps must contain only finite values");
            scales[column] = std::max({scales[column], std::abs(p), std::abs(q)});
        }
    }
    return scales;
}

ResponseMapSymmetryDiagnostics analyze_response_map_data(
    const Matrix& response_map, const std::vector<double>& forward_error,
    const std::vector<double>& solution_column_scales, const std::string& prefix) {
    const auto count = static_cast<std::size_t>(response_map.ncol());
    if (forward_error.size() != count || solution_column_scales.size() != count)
        throw PSIEXCEPTION(prefix + "response-map diagnostic cardinalities are inconsistent");
    const double dimension = static_cast<double>(count);
    double solution_scale = 0.0;
    double max_allowed_antisymmetry = 0.0;
    double max_symmetry_residual = 0.0;
    double max_normalized_antisymmetry = 0.0;
    for (std::size_t column = 0; column < count; ++column) {
        if (!std::isfinite(solution_column_scales[column]) ||
            solution_column_scales[column] < 0.0)
            throw PSIEXCEPTION(prefix + "response-map solution scales must be finite and nonnegative");
        solution_scale = std::max(solution_scale, solution_column_scales[column]);
    }
    for (std::size_t row = 0; row < count; ++row) {
        for (std::size_t column = row + 1; column < count; ++column) {
            const double machine_scale =
                std::max({1.0, solution_column_scales[row], solution_column_scales[column]});
            const double machine_roundoff =
                64.0 * std::numeric_limits<double>::epsilon() * machine_scale * dimension;
            // P(row,column) is bounded by FERR[column] and P(column,row) by
            // FERR[row], each with its own full [P;Q] solution-column envelope.
            const double allowed = forward_error[column] * solution_column_scales[column] +
                                   forward_error[row] * solution_column_scales[row] +
                                   machine_roundoff;
            const double residual =
                std::abs(response_map(row, column) - response_map(column, row));
            if (!std::isfinite(allowed) || !std::isfinite(residual))
                throw PSIEXCEPTION(prefix + "derived response-map symmetry diagnostics overflowed");
            const double normalized = residual / allowed;
            max_allowed_antisymmetry = std::max(max_allowed_antisymmetry, allowed);
            max_symmetry_residual = std::max(max_symmetry_residual, residual);
            max_normalized_antisymmetry = std::max(max_normalized_antisymmetry, normalized);
            if (normalized > 1.0)
                throw PSIEXCEPTION(prefix +
                                   "response map must be symmetric within its derived solver-error bound");
        }
    }
    return {solution_scale, max_allowed_antisymmetry, max_symmetry_residual,
            max_normalized_antisymmetry};
}

ResponseMapSymmetryDiagnostics analyze_response_map(
    const DenseRestrictedResponse& response, const Matrix& response_map) {
    const std::string prefix = "site-pair response contraction: ";
    validate_dense_response_diagnostics(
        response.reciprocal_condition(), response.reciprocal_pivot_growth(),
        response.forward_error(), response.backward_error(), response.scaled_residual());
    if (response_map.nirrep() != 1 || response_map.nrow() <= 0 ||
        response_map.nrow() != response_map.ncol() ||
        static_cast<std::size_t>(response_map.nrow()) != response.transition_count())
        throw PSIEXCEPTION(prefix + "dense response carrier dimensions are inconsistent");
    return analyze_response_map_data(response_map, response.forward_error(),
                                     response.solution_column_scales(), prefix);
}
}  // namespace

ResponseMapSymmetryDiagnostics validate_response_map_symmetry_test_only(
    const Matrix& response_map, const Matrix& conjugate_map,
    const std::vector<double>& forward_error) {
    const std::string prefix = "site-pair response symmetry test: ";
    std::vector<double> zero_error(forward_error.size(), 0.0);
    validate_dense_response_diagnostics(1.0, 1.0, forward_error, zero_error, zero_error);
    const auto scales = response_solution_column_scales(response_map, conjugate_map, prefix);
    return analyze_response_map_data(response_map, forward_error, scales, prefix);
}

SitePairResponseContraction contract_site_pair_response(
    std::size_t site_count, const Matrix& projection,
    const DenseRestrictedResponse& response) {
    const std::string prefix = "site-pair response contraction: ";
    constexpr double restricted_factor = 4.0;
    if (site_count == 0) throw PSIEXCEPTION(prefix + "site count must be nonzero");
    if (projection.nirrep() != 1 || projection.nrow() <= 0 || projection.ncol() <= 0)
        throw PSIEXCEPTION(prefix + "input dimensions are inconsistent");
    const auto transition_count = static_cast<std::size_t>(projection.ncol());
    const auto plan = plan_site_pair_response_contraction(
        site_count, transition_count, Process::environment.get_memory());
    if (static_cast<std::size_t>(projection.nrow()) != plan.component_count)
        throw PSIEXCEPTION(prefix + "projection dimensions do not match the site count");
    if (response.transition_count() != transition_count)
        throw PSIEXCEPTION(prefix + "response-map and projection dimensions are inconsistent");
    for (int row = 0; row < projection.nrow(); ++row)
        for (int transition = 0; transition < projection.ncol(); ++transition)
            if (!std::isfinite(projection(row, transition)))
                throw PSIEXCEPTION(prefix + "projection must contain only finite values");

    // Resource planning and caller projection checks precede the detached P copy.
    auto response_map_copy = response.P_clone();
    const auto& response_map = *response_map_copy;
    const auto symmetry = analyze_response_map(response, response_map);
    auto symmetric_response_map = std::make_shared<Matrix>(response_map.nrow(), response_map.ncol());
    for (int row = 0; row < response_map.nrow(); ++row) {
        (*symmetric_response_map)(row, row) = response_map(row, row);
        for (int column = row + 1; column < response_map.ncol(); ++column) {
            const double value = 0.5 * (response_map(row, column) + response_map(column, row));
            if (!std::isfinite(value))
                throw PSIEXCEPTION(prefix + "response-map roundoff averaging overflowed");
            (*symmetric_response_map)(row, column) = value;
            (*symmetric_response_map)(column, row) = value;
        }
    }

    Matrix projected_response(plan.component_count, transition_count);
    for (std::size_t component = 0; component < plan.component_count; ++component) {
        for (std::size_t column = 0; column < transition_count; ++column) {
            double value = 0.0;
            for (std::size_t transition = 0; transition < transition_count; ++transition) {
                const double term = projection(component, transition) *
                                    (*symmetric_response_map)(transition, column);
                if (!std::isfinite(term))
                    throw PSIEXCEPTION(prefix + "projection-response product overflowed");
                value += term;
                if (!std::isfinite(value))
                    throw PSIEXCEPTION(prefix + "projection-response accumulation overflowed");
            }
            projected_response(component, column) = value;
        }
    }
    auto values = std::make_shared<Matrix>(plan.component_count, plan.component_count);
    for (std::size_t row = 0; row < plan.component_count; ++row) {
        for (std::size_t column = row; column < plan.component_count; ++column) {
            double value = 0.0;
            for (std::size_t transition = 0; transition < transition_count; ++transition) {
                const double term = restricted_factor * projected_response(row, transition) *
                                    projection(column, transition);
                if (!std::isfinite(term))
                    throw PSIEXCEPTION(prefix + "response-matrix product overflowed");
                value += term;
                if (!std::isfinite(value))
                    throw PSIEXCEPTION(prefix + "response-matrix result is nonfinite");
            }
            (*values)(row, column) = value;
            (*values)(column, row) = value;
        }
    }

    SitePairResponseContraction result;
    result.values = std::move(values);
    result.plan = plan;
    result.restricted_factor = restricted_factor;
    result.response_map_forward_error_bound = *std::max_element(
        response.forward_error().begin(), response.forward_error().end());
    result.response_map_solution_scale = symmetry.solution_scale;
    result.response_map_allowed_antisymmetry = symmetry.allowed_antisymmetry;
    result.response_map_symmetry_residual = symmetry.symmetry_residual;
    result.response_map_max_normalized_antisymmetry =
        symmetry.max_normalized_antisymmetry;
    result.reciprocity_enforced = true;
    return result;
}

struct LeastSquaresSVD {
    std::size_t rows{};
    std::size_t columns{};
    std::size_t economy_rank{};
    std::vector<double> u;
    std::vector<double> singular_values;
    std::vector<double> vt;
};

std::size_t checked_lsq_elements(std::size_t first, std::size_t second, const char* name) {
    if (second != 0 && first > std::numeric_limits<std::size_t>::max() / second)
        throw PSIEXCEPTION(std::string("constrained least squares: ") + name + " allocation overflow");
    return first * second;
}

void require_lapack_dimensions(std::size_t rows, std::size_t columns) {
    if (rows == 0 || columns == 0 ||
        rows > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        columns > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw PSIEXCEPTION("constrained least squares: invalid LAPACK SVD dimensions");
}

int validated_lapack_workspace(double query, std::size_t maximum_elements,
                               const char* context) {
    if (!std::isfinite(query) || query < 1.0 ||
        query > static_cast<double>(std::numeric_limits<int>::max()))
        throw PSIEXCEPTION(std::string("constrained least squares: ") + context +
                           " workspace query failed");
    const auto rounded = static_cast<std::size_t>(std::ceil(query));
    if (rounded > maximum_elements)
        throw PSIEXCEPTION(std::string("constrained least squares: ") + context +
                           " workspace exceeds the explicit allocation cap");
    return static_cast<int>(rounded);
}

std::vector<double> lsq_column_major(const std::vector<double>& row_major,
                                     std::size_t rows, std::size_t columns) {
    const auto elements = checked_lsq_elements(rows, columns, "SVD input");
    if (row_major.size() != elements)
        throw PSIEXCEPTION("constrained least squares: internal SVD input dimension mismatch");
    std::vector<double> result(elements);
    for (std::size_t row = 0; row < rows; ++row)
        for (std::size_t column = 0; column < columns; ++column)
            result[column * rows + row] = row_major[row * columns + column];
    return result;
}

/** Economy U plus full right singular vectors for equality null-space elimination. */
LeastSquaresSVD constraint_null_space_svd(const std::vector<double>& row_major,
                                           std::size_t rows, std::size_t columns,
                                           std::size_t maximum_workspace_elements) {
    require_lapack_dimensions(rows, columns);
    LeastSquaresSVD result;
    result.rows = rows;
    result.columns = columns;
    result.economy_rank = std::min(rows, columns);
    const auto u_elements = checked_lsq_elements(rows, result.economy_rank, "constraint U");
    const auto vt_elements = checked_lsq_elements(columns, columns, "constraint VT");
    auto values = lsq_column_major(row_major, rows, columns);
    std::vector<double> column_major_u(u_elements);
    std::vector<double> column_major_vt(vt_elements);
    result.singular_values.resize(result.economy_rank);
    double workspace_query = 0.0;
    const int m = static_cast<int>(rows);
    const int n = static_cast<int>(columns);
    int info = C_DGESVD('S', 'A', m, n, values.data(), m, result.singular_values.data(),
                        column_major_u.data(), m, column_major_vt.data(), n,
                        &workspace_query, -1);
    if (info != 0)
        throw PSIEXCEPTION("constrained least squares: constraint SVD workspace query failed");
    const int workspace_size = validated_lapack_workspace(
        workspace_query, maximum_workspace_elements, "constraint SVD");
    std::vector<double> workspace(static_cast<std::size_t>(workspace_size));
    info = C_DGESVD('S', 'A', m, n, values.data(), m, result.singular_values.data(),
                    column_major_u.data(), m, column_major_vt.data(), n,
                    workspace.data(), workspace_size);
    if (info != 0) throw PSIEXCEPTION("constrained least squares: constraint SVD failed to converge");
    result.u.resize(u_elements);
    result.vt.resize(vt_elements);
    for (std::size_t row = 0; row < rows; ++row)
        for (std::size_t column = 0; column < result.economy_rank; ++column)
            result.u[row * result.economy_rank + column] = column_major_u[column * rows + row];
    for (std::size_t row = 0; row < columns; ++row)
        for (std::size_t column = 0; column < columns; ++column)
            result.vt[row * columns + column] = column_major_vt[column * columns + row];
    for (double value : result.singular_values)
        if (!std::isfinite(value))
            throw PSIEXCEPTION("constrained least squares: constraint SVD produced nonfinite diagnostics");
    return result;
}

/** Economy divide-and-conquer SVD for the tall reduced augmented fit. */
LeastSquaresSVD reduced_fit_svd(const std::vector<double>& row_major,
                                std::size_t rows, std::size_t columns,
                                std::size_t maximum_workspace_elements) {
    require_lapack_dimensions(rows, columns);
    LeastSquaresSVD result;
    result.rows = rows;
    result.columns = columns;
    result.economy_rank = std::min(rows, columns);
    const auto u_elements = checked_lsq_elements(rows, result.economy_rank, "fit U");
    const auto vt_elements = checked_lsq_elements(result.economy_rank, columns, "fit VT");
    const auto iwork_elements = checked_lsq_elements(8, result.economy_rank, "fit integer workspace");
    auto values = lsq_column_major(row_major, rows, columns);
    std::vector<double> column_major_u(u_elements);
    std::vector<double> column_major_vt(vt_elements);
    std::vector<int> integer_workspace(iwork_elements);
    result.singular_values.resize(result.economy_rank);
    double workspace_query = 0.0;
    const int m = static_cast<int>(rows);
    const int n = static_cast<int>(columns);
    const int ldvt = static_cast<int>(result.economy_rank);
    int info = C_DGESDD('S', m, n, values.data(), m, result.singular_values.data(),
                        column_major_u.data(), m, column_major_vt.data(), ldvt,
                        &workspace_query, -1, integer_workspace.data());
    if (info != 0)
        throw PSIEXCEPTION("constrained least squares: fit SVD workspace query failed");
    const int workspace_size = validated_lapack_workspace(
        workspace_query, maximum_workspace_elements, "fit SVD");
    std::vector<double> workspace(static_cast<std::size_t>(workspace_size));
    info = C_DGESDD('S', m, n, values.data(), m, result.singular_values.data(),
                    column_major_u.data(), m, column_major_vt.data(), ldvt,
                    workspace.data(), workspace_size, integer_workspace.data());
    if (info != 0) throw PSIEXCEPTION("constrained least squares: fit SVD failed to converge");
    result.u.resize(u_elements);
    result.vt.resize(vt_elements);
    for (std::size_t row = 0; row < rows; ++row)
        for (std::size_t column = 0; column < result.economy_rank; ++column)
            result.u[row * result.economy_rank + column] = column_major_u[column * rows + row];
    for (std::size_t row = 0; row < result.economy_rank; ++row)
        for (std::size_t column = 0; column < columns; ++column)
            result.vt[row * columns + column] = column_major_vt[column * result.economy_rank + row];
    for (double value : result.singular_values)
        if (!std::isfinite(value))
            throw PSIEXCEPTION("constrained least squares: fit SVD produced nonfinite diagnostics");
    return result;
}

ConstrainedLeastSquaresResult solve_constrained_least_squares(
    const Matrix& design, const std::vector<double>& observations,
    const std::vector<double>& row_weights, double lambda,
    const std::vector<double>& diagonal_anchor, const std::vector<double>& reference,
    const Matrix& constraints, const std::vector<double>& constraint_targets,
    const ConstrainedLeastSquaresOptions& options) {
    const std::string prefix = "constrained least squares: ";
    if (design.nirrep() != 1 || design.nrow() <= 0 || design.ncol() <= 0)
        throw PSIEXCEPTION(prefix + "design must be a nonempty dense matrix");
    const std::size_t row_count = static_cast<std::size_t>(design.nrow());
    const std::size_t column_count = static_cast<std::size_t>(design.ncol());
    if (observations.size() != row_count || row_weights.size() != row_count)
        throw PSIEXCEPTION(prefix + "observation and row-weight dimensions must match design rows");
    if (diagonal_anchor.size() != column_count || reference.size() != column_count)
        throw PSIEXCEPTION(prefix + "anchor and reference dimensions must match design columns");
    if (constraints.nirrep() != 1 || static_cast<std::size_t>(constraints.ncol()) != column_count ||
        constraint_targets.size() != static_cast<std::size_t>(constraints.nrow()))
        throw PSIEXCEPTION(prefix + "constraint dimensions must be C(rows, design columns) and d(rows)");
    if (!std::isfinite(lambda) || lambda < 0.0)
        throw PSIEXCEPTION(prefix + "lambda must be finite and nonnegative");
    if (!std::isfinite(options.column_cutoff) || options.column_cutoff < 0.0 ||
        !std::isfinite(options.maximum_condition_number) || options.maximum_condition_number < 1.0 ||
        !std::isfinite(options.rank_tolerance) || options.rank_tolerance <= 0.0 ||
        options.rank_tolerance >= 1.0)
        throw PSIEXCEPTION(prefix + "cutoff and numerical thresholds are invalid");
    const auto require_finite = [&prefix](double value, const char* name) {
        if (!std::isfinite(value)) throw PSIEXCEPTION(prefix + name + " must contain only finite values");
    };
    for (std::size_t row = 0; row < row_count; ++row) {
        require_finite(observations[row], "observations");
        require_finite(row_weights[row], "row weights");
        if (row_weights[row] < 0.0) throw PSIEXCEPTION(prefix + "row weights must be nonnegative");
        for (std::size_t column = 0; column < column_count; ++column)
            require_finite(design(row, column), "design");
    }
    for (std::size_t column = 0; column < column_count; ++column) {
        require_finite(diagonal_anchor[column], "diagonal anchor");
        require_finite(reference[column], "reference");
        if (diagonal_anchor[column] < 0.0)
            throw PSIEXCEPTION(prefix + "diagonal anchor weights must be nonnegative");
    }
    const std::size_t constraint_count = static_cast<std::size_t>(constraints.nrow());
    for (std::size_t row = 0; row < constraint_count; ++row) {
        require_finite(constraint_targets[row], "constraint targets");
        for (std::size_t column = 0; column < column_count; ++column)
            require_finite(constraints(row, column), "constraints");
    }

    ConstrainedLeastSquaresResult pending;
    pending.lambda = lambda;
    const auto weight_bounds = std::minmax_element(row_weights.begin(), row_weights.end());
    pending.row_weight_min = *weight_bounds.first;
    pending.row_weight_max = *weight_bounds.second;
    pending.row_weight_source = "caller_explicit";
    pending.full_to_reduced.assign(column_count, -1);
    pending.column_weighted_norms.assign(column_count, 0.0);
    for (std::size_t column = 0; column < column_count; ++column) {
        double norm = 0.0;
        for (std::size_t row = 0; row < row_count; ++row)
            norm = std::hypot(norm, row_weights[row] * design(row, column));
        if (!std::isfinite(norm)) throw PSIEXCEPTION(prefix + "weighted column norm overflowed");
        pending.column_weighted_norms[column] = norm;
        if (norm < options.column_cutoff) {
            if (!options.prune_below_cutoff)
                throw PSIEXCEPTION(prefix + "column " + std::to_string(column) + " is below cutoff");
            pending.pruned_columns.push_back(column);
        } else {
            pending.full_to_reduced[column] = static_cast<int>(pending.kept_columns.size());
            pending.kept_columns.push_back(column);
        }
    }
    const std::size_t reduced_count = pending.kept_columns.size();

    const auto numerical_rank = [&options](const std::vector<double>& singular_values) {
        if (singular_values.empty() || singular_values.front() == 0.0) return std::size_t{0};
        const double threshold = options.rank_tolerance * singular_values.front();
        return static_cast<std::size_t>(std::count_if(
            singular_values.begin(), singular_values.end(),
            [threshold](double value) { return value > threshold; }));
    };

    std::vector<double> particular(reduced_count, 0.0);
    std::vector<double> null_space(reduced_count * reduced_count, 0.0);
    std::size_t constraint_rank = 0;
    if (constraint_count == 0) {
        for (std::size_t column = 0; column < reduced_count; ++column)
            null_space[column * reduced_count + column] = 1.0;
    } else if (reduced_count == 0) {
        double residual = 0.0;
        for (double target : constraint_targets) residual = std::hypot(residual, target);
        if (residual > options.rank_tolerance)
            throw PSIEXCEPTION(prefix + "constraints are inconsistent after cutoff pruning");
        throw PSIEXCEPTION(prefix + "constraints are ambiguous after cutoff pruning");
    } else {
        std::vector<double> reduced_constraints(constraint_count * reduced_count);
        for (std::size_t row = 0; row < constraint_count; ++row)
            for (std::size_t column = 0; column < reduced_count; ++column)
                reduced_constraints[row * reduced_count + column] =
                    constraints(row, pending.kept_columns[column]);
        const auto constraint_svd = constraint_null_space_svd(
            reduced_constraints, constraint_count, reduced_count,
            options.maximum_workspace_elements);
        pending.allocation_plan.constraint_rows = constraint_count;
        pending.allocation_plan.constraint_columns = reduced_count;
        pending.allocation_plan.constraint_u_elements = constraint_svd.u.size();
        pending.allocation_plan.constraint_vt_elements = constraint_svd.vt.size();
        constraint_rank = numerical_rank(constraint_svd.singular_values);
        for (std::size_t mode = 0; mode < constraint_rank; ++mode) {
            double projection = 0.0;
            for (std::size_t row = 0; row < constraint_count; ++row)
                projection += constraint_svd.u[row * constraint_svd.economy_rank + mode] *
                              constraint_targets[row];
            projection /= constraint_svd.singular_values[mode];
            for (std::size_t column = 0; column < reduced_count; ++column)
                particular[column] += constraint_svd.vt[mode * reduced_count + column] * projection;
        }
        double residual = 0.0;
        double target_norm = 0.0;
        for (std::size_t row = 0; row < constraint_count; ++row) {
            double value = -constraint_targets[row];
            for (std::size_t column = 0; column < reduced_count; ++column)
                value += reduced_constraints[row * reduced_count + column] * particular[column];
            residual = std::hypot(residual, value);
            target_norm = std::hypot(target_norm, constraint_targets[row]);
        }
        const double feasibility_tolerance = options.rank_tolerance *
            static_cast<double>(std::max(constraint_count, reduced_count)) * (1.0 + target_norm);
        if (!std::isfinite(residual) || residual > feasibility_tolerance)
            throw PSIEXCEPTION(prefix + "constraints are inconsistent");
        if (constraint_rank < constraint_count)
            throw PSIEXCEPTION(prefix + "constraints are ambiguous (linearly dependent)");
        const std::size_t free_count = reduced_count - constraint_rank;
        null_space.assign(reduced_count * free_count, 0.0);
        for (std::size_t mode = 0; mode < free_count; ++mode)
            for (std::size_t column = 0; column < reduced_count; ++column)
                null_space[column * free_count + mode] =
                    constraint_svd.vt[(constraint_rank + mode) * reduced_count + column];
    }

    pending.constraint_rank = constraint_rank;
    pending.free_dimension = reduced_count - constraint_rank;
    std::vector<double> reduced_solution = particular;
    if (pending.free_dimension > 0) {
        const std::size_t augmented_rows = row_count + reduced_count;
        std::vector<double> reduced_design(augmented_rows * pending.free_dimension, 0.0);
        std::vector<double> reduced_target(augmented_rows, 0.0);
        for (std::size_t row = 0; row < row_count; ++row) {
            double offset = observations[row];
            for (std::size_t column = 0; column < reduced_count; ++column)
                offset -= design(row, pending.kept_columns[column]) * particular[column];
            reduced_target[row] = row_weights[row] * offset;
            for (std::size_t mode = 0; mode < pending.free_dimension; ++mode) {
                double value = 0.0;
                for (std::size_t column = 0; column < reduced_count; ++column)
                    value += design(row, pending.kept_columns[column]) *
                             null_space[column * pending.free_dimension + mode];
                reduced_design[row * pending.free_dimension + mode] = row_weights[row] * value;
            }
        }
        const double penalty_scale = std::sqrt(lambda);
        for (std::size_t anchor_row = 0; anchor_row < reduced_count; ++anchor_row) {
            const std::size_t full_column = pending.kept_columns[anchor_row];
            const double scale = penalty_scale * diagonal_anchor[full_column];
            reduced_target[(row_count + anchor_row)] = scale * (reference[full_column] - particular[anchor_row]);
            for (std::size_t mode = 0; mode < pending.free_dimension; ++mode)
                reduced_design[(row_count + anchor_row) * pending.free_dimension + mode] =
                    scale * null_space[anchor_row * pending.free_dimension + mode];
        }
        for (double value : reduced_design)
            if (!std::isfinite(value)) throw PSIEXCEPTION(prefix + "augmented design overflowed");
        for (double value : reduced_target)
            if (!std::isfinite(value)) throw PSIEXCEPTION(prefix + "augmented target overflowed");
        const auto fit_svd = reduced_fit_svd(
            reduced_design, augmented_rows, pending.free_dimension,
            options.maximum_workspace_elements);
        pending.allocation_plan.fit_rows = augmented_rows;
        pending.allocation_plan.fit_columns = pending.free_dimension;
        pending.allocation_plan.fit_u_elements = fit_svd.u.size();
        pending.allocation_plan.fit_vt_elements = fit_svd.vt.size();
        pending.singular_values = fit_svd.singular_values;
        pending.rank = numerical_rank(pending.singular_values);
        if (pending.rank != pending.free_dimension)
            throw PSIEXCEPTION(prefix + "reduced objective is rank deficient");
        pending.condition_number = pending.singular_values.front() /
                                   pending.singular_values[pending.free_dimension - 1];
        if (!std::isfinite(pending.condition_number) ||
            pending.condition_number > options.maximum_condition_number)
            throw PSIEXCEPTION(prefix + "condition number exceeds explicit threshold");
        std::vector<double> free_solution(pending.free_dimension, 0.0);
        for (std::size_t mode = 0; mode < pending.free_dimension; ++mode) {
            double projection = 0.0;
            for (std::size_t row = 0; row < augmented_rows; ++row)
                projection += fit_svd.u[row * fit_svd.economy_rank + mode] * reduced_target[row];
            projection /= pending.singular_values[mode];
            for (std::size_t column = 0; column < pending.free_dimension; ++column)
                free_solution[column] += fit_svd.vt[mode * pending.free_dimension + column] * projection;
        }
        for (std::size_t column = 0; column < reduced_count; ++column)
            for (std::size_t mode = 0; mode < pending.free_dimension; ++mode)
                reduced_solution[column] += null_space[column * pending.free_dimension + mode] * free_solution[mode];
    } else {
        pending.rank = 0;
        pending.condition_number = 1.0;
    }

    pending.solution.assign(column_count, 0.0);
    for (std::size_t column = 0; column < reduced_count; ++column)
        pending.solution[pending.kept_columns[column]] = reduced_solution[column];
    for (double value : pending.solution)
        if (!std::isfinite(value)) throw PSIEXCEPTION(prefix + "solution is nonfinite");
    for (std::size_t row = 0; row < row_count; ++row) {
        double residual = -observations[row];
        for (std::size_t column = 0; column < column_count; ++column)
            residual += design(row, column) * pending.solution[column];
        pending.weighted_residual_norm = std::hypot(
            pending.weighted_residual_norm, row_weights[row] * residual);
    }
    for (std::size_t column = 0; column < column_count; ++column)
        pending.anchor_residual_norm = std::hypot(
            pending.anchor_residual_norm,
            diagonal_anchor[column] * (pending.solution[column] - reference[column]));
    for (std::size_t row = 0; row < constraint_count; ++row) {
        double residual = -constraint_targets[row];
        for (std::size_t column = 0; column < column_count; ++column)
            residual += constraints(row, column) * pending.solution[column];
        pending.constraint_residual_norm = std::hypot(pending.constraint_residual_norm, residual);
    }
    pending.objective_residual_norm = std::hypot(
        pending.weighted_residual_norm, std::sqrt(lambda) * pending.anchor_residual_norm);
    if (!std::isfinite(pending.weighted_residual_norm) || !std::isfinite(pending.anchor_residual_norm) ||
        !std::isfinite(pending.constraint_residual_norm) || !std::isfinite(pending.objective_residual_norm))
        throw PSIEXCEPTION(prefix + "residual diagnostics are nonfinite");
    return pending;
}
}  // namespace detail

std::shared_ptr<FrozenResponseContext> FrozenResponseContext::create(
    const std::shared_ptr<Wavefunction>& grac_wfn,
    const std::shared_ptr<Wavefunction>& neutral_precursor_wfn,
    const std::shared_ptr<Wavefunction>& cation_wfn) {
    const auto grac_hf = require_converged_scf(grac_wfn, "GRAC neutral");
    const auto precursor_hf = require_converged_scf(neutral_precursor_wfn, "neutral precursor");
    const auto cation_hf = require_converged_scf(cation_wfn, "cation");
    const auto grac_rhf = std::dynamic_pointer_cast<scf::RHF>(grac_hf);
    const auto precursor_rhf = std::dynamic_pointer_cast<scf::RHF>(precursor_hf);
    const auto cation_uhf = std::dynamic_pointer_cast<scf::UHF>(cation_hf);
    const auto& grac_seal = *grac_hf->response_provenance();
    const auto& precursor_seal = *precursor_hf->response_provenance();
    const auto& cation_seal = *cation_hf->response_provenance();
    if (!grac_rhf || !precursor_rhf || grac_seal.reference != "RKS" || precursor_seal.reference != "RKS" ||
        grac_seal.charge != 0 || precursor_seal.charge != 0 || grac_seal.multiplicity != 1 ||
        precursor_seal.multiplicity != 1 || grac_seal.nalpha != grac_seal.nbeta ||
        precursor_seal.nalpha != precursor_seal.nbeta || !grac_seal.functional.unpolarized ||
        !precursor_seal.functional.unpolarized) {
        throw PSIEXCEPTION("FrozenResponseContext: neutral and precursor must be neutral restricted singlet RKS states");
    }
    const bool cation_state_valid = cation_uhf && cation_seal.reference == "UKS" &&
                                    cation_seal.charge == 1 && cation_seal.multiplicity == 2 &&
                                    cation_seal.nalpha == cation_seal.nbeta + 1 &&
                                    !cation_seal.functional.unpolarized;
    // Validate this fact here and the basis fact at its existing point below so
    // factory failure ordering remains unchanged.
    detail::validate_vertical_protocol(cation_state_valid, true);
    const auto functional = grac_seal.sealed_functional;
    const auto precursor_functional = precursor_seal.sealed_functional;
    if (!functional || !precursor_functional || !functional->needs_xc() ||
        !grac_seal.potential_grac_initialized || !grac_seal.functional.needs_grac) {
        throw PSIEXCEPTION("FrozenResponseContext: neutral must contain an actual applied GRAC RKS state with needs_grac");
    }
    if (precursor_seal.functional.needs_grac ||
        !grac_seal.functional.same_ground_state(precursor_seal.functional)) {
        throw PSIEXCEPTION("FrozenResponseContext: neutral precursor must be the same unshifted ground-state functional");
    }
    const double functional_density_tolerance = grac_seal.functional.density_tolerance;
    if (!std::isfinite(functional_density_tolerance) || !(functional_density_tolerance > 0.0) ||
        functional->density_tolerance() != functional_density_tolerance ||
        precursor_functional->density_tolerance() != functional_density_tolerance) {
        throw PSIEXCEPTION("FrozenResponseContext: functional density tolerance provenance is invalid");
    }
    const LibXCFunctional expected_x(kProtocolGRACX, true);
    const LibXCFunctional expected_c(kProtocolGRACC, true);
    const auto& actual_x = grac_seal.functional.grac_x;
    const auto& actual_c = grac_seal.functional.grac_c;
    if (actual_x.libxc_id != expected_x.libxc_id() || actual_c.libxc_id != expected_c.libxc_id() ||
        actual_x.libxc_canonical_name != expected_x.libxc_canonical_name() ||
        actual_c.libxc_canonical_name != expected_c.libxc_canonical_name()) {
        throw PSIEXCEPTION("FrozenResponseContext: GRAC immutable LibXC identity does not match the intended X/C components");
    }
    if (actual_x.effective_parameters != expected_x.effective_parameter_map() ||
        actual_c.effective_parameters != expected_c.effective_parameter_map()) {
        throw PSIEXCEPTION("FrozenResponseContext: GRAC functional parameter map does not match the intended full map");
    }
    const auto same_component_settings = [](const scf::ResponseFunctionalComponentState& actual,
                                            const LibXCFunctional& expected) {
        return actual.omega == expected.omega() && actual.lsda_cutoff == expected.lsda_cutoff() &&
               actual.meta_cutoff == expected.meta_cutoff() &&
               actual.density_cutoff == expected.density_cutoff() && actual.gga == expected.is_gga() &&
               actual.meta == expected.is_meta() && actual.lrc == expected.is_lrc() &&
               actual.unpolarized == expected.is_unpolarized();
    };
    if (actual_x.alpha != 1.0 - grac_seal.functional.x_alpha || actual_c.alpha != 1.0 ||
        !same_component_settings(actual_x, expected_x) || !same_component_settings(actual_c, expected_c) ||
        grac_seal.functional.grac_alpha != 0.5 || grac_seal.functional.grac_beta != 40.0) {
        throw PSIEXCEPTION("FrozenResponseContext: GRAC alpha/beta, cutoff, polarization, or scaling is inconsistent");
    }
    if (grac_seal.functional_workers.empty())
        throw PSIEXCEPTION("FrozenResponseContext: GRAC worker functional provenance is unavailable");
    for (const auto& worker : grac_seal.functional_workers) {
        if (!worker.needs_grac || worker.grac_shift != grac_seal.functional.grac_shift ||
            worker.grac_alpha != grac_seal.functional.grac_alpha ||
            worker.grac_beta != grac_seal.functional.grac_beta ||
            worker.density_tolerance != functional_density_tolerance || !(worker.grac_x == actual_x) ||
            !(worker.grac_c == actual_c)) {
            throw PSIEXCEPTION("FrozenResponseContext: GRAC master/worker effective state is inconsistent");
        }
    }

    const auto grac_molecule = grac_seal.sealed_molecule;
    const auto precursor_molecule = precursor_seal.sealed_molecule;
    const auto cation_molecule = cation_seal.sealed_molecule;
    if (!grac_molecule || !precursor_molecule || !cation_molecule ||
        !same_nuclei_and_geometry(*grac_molecule, *precursor_molecule) ||
        !same_nuclei_and_geometry(*grac_molecule, *cation_molecule) ||
        cation_seal.nalpha + cation_seal.nbeta != grac_seal.nalpha + grac_seal.nbeta - 1) {
        throw PSIEXCEPTION("FrozenResponseContext: cation calculation geometry/electron identity is inconsistent");
    }
    const bool complete_basis_valid =
        grac_seal.basis && precursor_seal.basis && cation_seal.basis &&
        same_vertical_basis_structure(*grac_seal.basis, *precursor_seal.basis) &&
        same_vertical_basis_structure(*grac_seal.basis, *cation_seal.basis) &&
        grac_hf->basisset()->structural_snapshot() == *grac_seal.basis;
    detail::validate_vertical_protocol(true, complete_basis_valid);
    if (!cation_seal.sealed_functional ||
        cation_seal.sealed_functional->density_tolerance() != functional_density_tolerance ||
        !precursor_seal.functional.same_ground_state(cation_seal.functional) ||
        cation_seal.functional.needs_grac) {
        throw PSIEXCEPTION("FrozenResponseContext: cation must use the same unshifted ground-state functional");
    }

    const double homo = precursor_seal.occupied_homo;
    const double ip = cation_seal.energy - precursor_seal.energy;
    const double derived_shift = ip + homo;
    const double applied_shift = grac_seal.functional.grac_shift;
    if (!std::isfinite(ip) || ip <= 0.0 || !std::isfinite(derived_shift) || derived_shift < 0.0 ||
        !close_enough(applied_shift, derived_shift)) {
        throw PSIEXCEPTION("FrozenResponseContext: actual applied GRAC shift must equal IP plus HOMO energy");
    }

    if (grac_seal.grid_weights.empty() || grac_seal.grid_points.size() != grac_seal.grid_weights.size() * 3 ||
        grac_seal.grid_blocks.empty())
        throw PSIEXCEPTION("FrozenResponseContext: sealed exact ordered RKS grid is unavailable");
    std::vector<FrozenGridBlock> grid_blocks;
    grid_blocks.reserve(grac_seal.grid_blocks.size());
    std::size_t grid_offset = 0;
    for (const auto& block : grac_seal.grid_blocks) {
        if (block.offset != grid_offset || block.point_count == 0)
            throw PSIEXCEPTION("FrozenResponseContext: sealed RKS grid block ordering is inconsistent");
        grid_blocks.push_back({block.offset, block.point_count, block.functions_local_to_global});
        grid_offset += block.point_count;
    }
    if (grid_offset != grac_seal.grid_weights.size())
        throw PSIEXCEPTION("FrozenResponseContext: sealed RKS grid block cardinality is inconsistent");

    std::vector<SitePosition> sites;
    sites.reserve(grac_molecule->natom());
    for (int atom = 0; atom < grac_molecule->natom(); ++atom)
        sites.push_back({grac_molecule->x(atom), grac_molecule->y(atom), grac_molecule->z(atom)});
    GRACProvenance provenance{precursor_seal.energy, cation_seal.energy, homo, ip, applied_shift,
                              cation_seal.reference, cation_seal.charge, cation_seal.multiplicity};
    auto molecule_copy = std::make_shared<Molecule>(grac_molecule->clone());
    return std::shared_ptr<FrozenResponseContext>(new FrozenResponseContext(
        grac_seal.Ca->clone(), grac_seal.Cb->clone(),
        std::make_shared<Vector>(grac_seal.epsilon_a->clone()),
        std::make_shared<Vector>(grac_seal.epsilon_b->clone()),
        std::make_shared<Vector>(grac_seal.occupation_a->clone()),
        std::make_shared<Vector>(grac_seal.occupation_b->clone()),
        grac_seal.Da->clone(), grac_seal.Db->clone(), grac_seal.energy, std::move(molecule_copy),
        grac_hf->basisset(), grac_seal.basis, functional, std::move(sites), grac_seal.grid_points,
        grac_seal.grid_weights, std::move(grid_blocks), std::move(provenance), grac_seal.functional.name,
        kProtocolGRACX, kProtocolGRACC, functional_density_tolerance));
}

ISAWeights::ISAWeights(std::shared_ptr<const FrozenResponseContext> context,
                       std::vector<double> partition_weights, ISADiagnostics diagnostics)
    : context_(std::move(context)), partition_weights_(std::move(partition_weights)),
      diagnostics_(std::move(diagnostics)) {}

ISAWeights ISAWeights::create_test_only(std::shared_ptr<const FrozenResponseContext> context,
                                        std::vector<double> partition_weights) {
    if (!context) throw PSIEXCEPTION("ISAWeights: frozen response context is null");
    const auto point_count = context->grid_point_count();
    const auto site_count = context->sites().size();
    if (point_count == 0 || site_count == 0 || point_count > std::numeric_limits<std::size_t>::max() / site_count ||
        partition_weights.size() != point_count * site_count)
        throw PSIEXCEPTION("ISAWeights: partition weights do not match the frozen ordered grid/sites");
    for (std::size_t point = 0; point < point_count; ++point) {
        double sum = 0.0;
        for (std::size_t site = 0; site < site_count; ++site) {
            const double value = partition_weights[point * site_count + site];
            if (!std::isfinite(value) || value < 0.0)
                throw PSIEXCEPTION("ISAWeights: partition weights must be finite and nonnegative");
            sum += value;
        }
        if (!std::isfinite(sum) || std::abs(sum - 1.0) > kValidationTolerance)
            throw PSIEXCEPTION("ISAWeights: partition unity failed on the frozen ordered grid");
    }
    return ISAWeights(std::move(context), std::move(partition_weights), ISADiagnostics{});
}

std::size_t ISAWeights::point_count() const { return context_->grid_point_count(); }
std::size_t ISAWeights::site_count() const { return context_->sites().size(); }

namespace detail {
TransitionMultipoleProjection TransitionMultipoleProjector::project(
    const std::shared_ptr<const FrozenResponseContext>& context,
    const ISAWeights& isa_weights) {
    const std::string prefix = "transition multipole projection: ";
    if (!context) throw PSIEXCEPTION(prefix + "frozen response context is null");
    if (isa_weights.context_.get() != context.get())
        throw PSIEXCEPTION(prefix + "ISA weights must belong to the same frozen response context");
    const auto basis_const = context->basis();
    if (!basis_const) throw PSIEXCEPTION(prefix + "retained basis is unavailable");
    const int nbf_int = basis_const->nbf();
    if (nbf_int <= 0) throw PSIEXCEPTION(prefix + "retained basis is empty");
    const auto nbf = static_cast<std::size_t>(nbf_int);
    const auto counts = validate_restricted_alda_orbitals(*context);
    const auto npoints = context->grid_point_count();
    const auto nsites = context->sites().size();
    if (npoints == 0 || nsites == 0 ||
        npoints > std::numeric_limits<std::size_t>::max() / 3 ||
        context->grid_points().size() != 3 * npoints)
        throw PSIEXCEPTION(prefix + "sealed grid/site dimensions are inconsistent");
    if (npoints > std::numeric_limits<std::size_t>::max() / nsites ||
        isa_weights.partition_weights_.size() != npoints * nsites)
        throw PSIEXCEPTION(prefix + "ISA partition dimensions are inconsistent");
    preflight_restricted_alda_grid(nbf, npoints, context->grid_weights(), context->grid_blocks());
    std::size_t max_block_points = 0;
    for (const auto& block : context->grid_blocks())
        max_block_points = std::max(max_block_points, block.point_count);
    const auto nov = checked_c1_product(counts.first, counts.second, prefix);
    const auto plan = plan_transition_multipole_projection(
        npoints, nsites, nov, max_block_points, nbf,
        static_cast<std::size_t>(context->Ca()->ncol()), Process::environment.get_memory());
    context->verify_basis_unchanged();
    validate_restricted_alda_duplicate_maps(context->grid_blocks());
    for (double coordinate : context->grid_points())
        if (!std::isfinite(coordinate))
            throw PSIEXCEPTION(prefix + "sealed coordinates must be finite");
    for (const auto& site : context->sites())
        for (double coordinate : site)
            if (!std::isfinite(coordinate)) throw PSIEXCEPTION(prefix + "sites must be finite");
    for (std::size_t point = 0; point < npoints; ++point) {
        double unity = 0.0;
        for (std::size_t site = 0; site < nsites; ++site) {
            const double value = isa_weights.partition_weights_[point * nsites + site];
            if (!std::isfinite(value) || value < 0.0)
                throw PSIEXCEPTION(prefix + "ISA partition must be finite and nonnegative");
            unity += value;
        }
        if (!std::isfinite(unity) || std::abs(unity - 1.0) > kValidationTolerance)
            throw PSIEXCEPTION(prefix + "pointwise ISA partition unity failed");
    }

    const auto basis = std::const_pointer_cast<BasisSet>(basis_const);
    auto transitions = make_restricted_alda_transitions(*context, nov);
    auto values = std::make_shared<Matrix>(nsites * 16, nov);
    auto extents = std::make_shared<BasisExtents>(basis, 1.0e-12);
    BasisFunctions collocation(basis, static_cast<int>(max_block_points), nbf_int);
    const auto& Ca = *context->Ca();
    for (const auto& sealed : context->grid_blocks()) {
        auto block = make_complete_alda_block(*context, sealed, extents, basis);
        collocation.compute_functions(block);
        const auto phi = collocation.basis_value("PHI");
        Matrix orbital_values(sealed.point_count, Ca.ncol());
        Matrix block_tau(sealed.point_count, nov);
        for (std::size_t local_point = 0; local_point < sealed.point_count; ++local_point) {
            for (int orbital = 0; orbital < Ca.ncol(); ++orbital) {
                double value = 0.0;
                for (int mu : sealed.functions_local_to_global)
                    value += (*phi)(local_point, mu) * Ca(mu, orbital);
                if (!std::isfinite(value))
                    throw PSIEXCEPTION(prefix + "orbital collocation is nonfinite");
                orbital_values(local_point, orbital) = value;
            }
            for (std::size_t transition = 0; transition < nov; ++transition) {
                const double value = orbital_values(local_point, transitions[transition].first) *
                                     orbital_values(local_point, transitions[transition].second);
                if (!std::isfinite(value))
                    throw PSIEXCEPTION(prefix + "transition collocation is nonfinite");
                block_tau(local_point, transition) = value;
            }
        }
        for (std::size_t local_point = 0; local_point < sealed.point_count; ++local_point) {
            const auto point = sealed.point_offset + local_point;
            const SitePosition position{context->grid_points()[3 * point],
                                        context->grid_points()[3 * point + 1],
                                        context->grid_points()[3 * point + 2]};
            for (std::size_t site = 0; site < nsites; ++site) {
                const double factor = context->grid_weights()[point] *
                    isa_weights.partition_weights_[point * nsites + site];
                if (!std::isfinite(factor))
                    throw PSIEXCEPTION(prefix + "weight contraction overflowed");
                const SitePosition displacement{position[0] - context->sites()[site][0],
                                                position[1] - context->sites()[site][1],
                                                position[2] - context->sites()[site][2]};
                const auto harmonics = regular_harmonics(displacement);
                for (std::size_t component = 0; component < 16; ++component) {
                    const auto row = site * 16 + component;
                    for (std::size_t transition = 0; transition < nov; ++transition) {
                        const double increment = factor * harmonics[component] *
                                                 block_tau(local_point, transition);
                        if (!std::isfinite(increment))
                            throw PSIEXCEPTION(prefix + "multipole contraction overflowed");
                        (*values)(row, transition) += increment;
                        if (!std::isfinite((*values)(row, transition)))
                            throw PSIEXCEPTION(prefix + "multipole accumulation overflowed");
                    }
                }
            }
        }
    }
    TransitionMultipoleProjection result;
    result.transitions = std::move(transitions);
    result.values = std::move(values);
    result.plan = plan;
    context->verify_basis_unchanged();
    return result;
}
}  // namespace detail

detail::TransitionMultipoleProjection project_transition_multipoles(
    const std::shared_ptr<const FrozenResponseContext>& context,
    const ISAWeights& isa_weights) {
    return detail::TransitionMultipoleProjector::project(context, isa_weights);
}

namespace {
struct ISAPolResponsePreflight {
    std::size_t nbf{};
    std::size_t nocc{};
    std::size_t nvir{};
    std::size_t nov{};
    std::size_t nmo{};
    std::size_t point_count{};
    std::size_t max_block_points{};
};

// Allocation-light C4 context preflight. It reads sealed arrays with checked
// loops and creates no transition vector, dense matrix, grid block, or output.
ISAPolResponsePreflight preflight_isapol_response_provider(
    const std::shared_ptr<const FrozenResponseContext>& context,
    bool verify_basis_snapshot = true) {
    const std::string prefix = "ISAPolResponseProvider: ";
    if (!context || !context->basis())
        throw PSIEXCEPTION(prefix + "frozen response context/basis is unavailable");
    if (verify_basis_snapshot) context->verify_basis_unchanged();
    const int nbf_int = context->basis()->nbf();
    if (nbf_int <= 0)
        throw PSIEXCEPTION(prefix + "retained basis is empty");
    const auto nbf = static_cast<std::size_t>(nbf_int);
    const auto counts = validate_restricted_alda_orbitals(*context);
    const auto nov = checked_c1_product(counts.first, counts.second, prefix);
    const auto nmo = checked_c1_sum(counts.first, counts.second, prefix);
    if (context->Ca()->ncol() <= 0 ||
        static_cast<std::size_t>(context->Ca()->ncol()) != nmo ||
        context->epsilon_a()->nirrep() != 1 || context->epsilon_b()->nirrep() != 1 ||
        context->epsilon_a()->dim(0) != context->Ca()->ncol() ||
        context->epsilon_b()->dim(0) != context->Ca()->ncol())
        throw PSIEXCEPTION(prefix + "orbital energy dimensions are inconsistent");
    for (std::size_t orbital = 0; orbital < nmo; ++orbital) {
        const double alpha = context->epsilon_a()->get(0, orbital);
        const double beta = context->epsilon_b()->get(0, orbital);
        if (!std::isfinite(alpha) || alpha != beta)
            throw PSIEXCEPTION(prefix +
                               "restricted orbital energies must be finite and identical");
    }
    for (std::size_t occupied = 0; occupied < nmo; ++occupied) {
        if (context->occupation_a()->get(0, occupied) != 1.0) continue;
        for (std::size_t virtual_orbital = 0; virtual_orbital < nmo;
             ++virtual_orbital) {
            if (context->occupation_a()->get(0, virtual_orbital) != 0.0) continue;
            const double gap = context->epsilon_a()->get(0, virtual_orbital) -
                               context->epsilon_a()->get(0, occupied);
            if (!std::isfinite(gap) || !(gap > 0.0))
                throw PSIEXCEPTION(prefix +
                                   "occupied-virtual gaps must be finite and positive");
        }
    }
    if (context->Da()->nirrep() != 1 || context->Db()->nirrep() != 1 ||
        context->Da()->nrow() != nbf_int || context->Da()->ncol() != nbf_int ||
        context->Db()->nrow() != nbf_int || context->Db()->ncol() != nbf_int)
        throw PSIEXCEPTION(prefix +
                           "frozen density dimensions do not match the retained basis");
    if (!std::isfinite(context->functional_density_tolerance()) ||
        !(context->functional_density_tolerance() > 0.0))
        throw PSIEXCEPTION(prefix + "functional density tolerance is invalid");

    const auto point_count = context->grid_point_count();
    if (point_count == 0 ||
        point_count > std::numeric_limits<std::size_t>::max() / 3 ||
        context->grid_points().size() != 3 * point_count)
        throw PSIEXCEPTION(prefix + "sealed grid dimensions are inconsistent");
    preflight_restricted_alda_grid(
        nbf, point_count, context->grid_weights(), context->grid_blocks());
    for (double coordinate : context->grid_points())
        if (!std::isfinite(coordinate))
            throw PSIEXCEPTION(prefix + "sealed coordinates must be finite");
    for (const auto& site : context->sites())
        for (double coordinate : site)
            if (!std::isfinite(coordinate))
                throw PSIEXCEPTION(prefix + "site coordinates must be finite");
    std::size_t max_block_points = 0;
    for (const auto& block : context->grid_blocks())
        max_block_points = std::max(max_block_points, block.point_count);
    return {nbf, counts.first, counts.second, nov, nmo,
            point_count, max_block_points};
}
}  // namespace

ISAPolResponseProvider::ISAPolResponseProvider(std::shared_ptr<const FrozenResponseContext> context,
                                               ResponseKernel kernel, ISAWeights isa_weights)
    : context_(std::move(context)), kernel_(std::move(kernel)), isa_weights_(std::move(isa_weights)) {
    if (!context_) throw PSIEXCEPTION("ISAPolResponseProvider: frozen response context is null");
    if (isa_weights_.context_.get() != context_.get())
        throw PSIEXCEPTION("ISAPolResponseProvider: ISA weights belong to a different frozen response context");
}

std::size_t ISAPolResponseProvider::expected_response_count(const FrequencyGrid& frequencies) const {
    context_->verify_basis_unchanged();
    if (frequencies.frequencies.empty())
        throw PSIEXCEPTION("ISAPolResponseProvider: frequency grid requires at least one point");
    if (frequencies.frequencies.size() != frequencies.weights.size())
        throw PSIEXCEPTION("ISAPolResponseProvider: frequency grid has inconsistent dimensions");
    const double static_frequency = frequencies.frequencies.front();
    const double static_weight = frequencies.weights.front();
    if (!std::isfinite(static_frequency) || !std::isfinite(static_weight))
        throw PSIEXCEPTION("ISAPolResponseProvider: frequency grid values must be finite");
    if (static_frequency < 0.0)
        throw PSIEXCEPTION("ISAPolResponseProvider: static frequency must be nonnegative");
    if (static_frequency != 0.0)
        throw PSIEXCEPTION("ISAPolResponseProvider: frequency grid must start at exactly zero");
    if (static_weight != 0.0)
        throw PSIEXCEPTION("ISAPolResponseProvider: static frequency weight must be exactly zero");
    for (std::size_t point = 1; point < frequencies.frequencies.size(); ++point) {
        const double frequency = frequencies.frequencies[point];
        const double weight = frequencies.weights[point];
        if (!std::isfinite(frequency) || !std::isfinite(weight))
            throw PSIEXCEPTION("ISAPolResponseProvider: frequency grid values must be finite");
        if (frequency < 0.0)
            throw PSIEXCEPTION("ISAPolResponseProvider: every nonstatic frequency must be positive");
        if (frequency <= frequencies.frequencies[point - 1])
            throw PSIEXCEPTION("ISAPolResponseProvider: frequencies must be strictly increasing");
        if (weight <= 0.0)
            throw PSIEXCEPTION("ISAPolResponseProvider: every nonzero-frequency weight must be positive");
    }
    return frequencies.frequencies.size();
}

std::vector<SitePairResponse> ISAPolResponseProvider::compute_isapol_response(
    const FrequencyGrid& frequencies) const {
    const std::string prefix = "ISAPolResponseProvider: ";
    const auto frequency_count = expected_response_count(frequencies);
    if (kernel_.chf_exchange() != 0.25 || kernel_.alda_kernel() != 0.75)
        throw PSIEXCEPTION(prefix + "response-kernel metadata is inconsistent");
    if (isa_weights_.context_.get() != context_.get())
        throw PSIEXCEPTION(prefix + "ISA weights belong to a different frozen response context");
    const auto site_count = context_->sites().size();
    const auto point_count = context_->grid_point_count();
    if (site_count == 0 || isa_weights_.site_count() != site_count ||
        isa_weights_.point_count() != point_count)
        throw PSIEXCEPTION(prefix + "ISA dimensions do not match the frozen response context");
    const auto isa_element_count = checked_c1_product(
        point_count, site_count, prefix);
    if (isa_weights_.partition_weights_.size() != isa_element_count)
        throw PSIEXCEPTION(prefix + "ISA partition dimensions are inconsistent");
    for (std::size_t point = 0; point < point_count; ++point) {
        double unity = 0.0;
        for (std::size_t site = 0; site < site_count; ++site) {
            const double value =
                isa_weights_.partition_weights_[point * site_count + site];
            if (!std::isfinite(value) || value < 0.0)
                throw PSIEXCEPTION(prefix +
                                   "ISA partition must be finite and nonnegative");
            unity += value;
        }
        if (!std::isfinite(unity) ||
            std::abs(unity - 1.0) > kValidationTolerance)
            throw PSIEXCEPTION(prefix + "pointwise ISA partition unity failed");
    }

    // Allocation-light validation and all pure resource planners precede C1's
    // transition vectors, dense primitives, or DirectJK construction.
    const auto preflight = preflight_isapol_response_provider(context_);
    const bool has_dynamic_frequency = frequency_count > 1;
    const auto provider_plan = detail::plan_isapol_response_provider(
        frequency_count, site_count, preflight.nbf, preflight.nocc,
        preflight.nvir, preflight.point_count, context_->grid_blocks(),
        has_dynamic_frequency, Process::environment.get_memory(),
        context_->functional_density_tolerance());
    if (provider_plan.frequency_count != frequency_count ||
        provider_plan.site_count != site_count ||
        provider_plan.transition_count != preflight.nov ||
        provider_plan.point_count != preflight.point_count ||
        provider_plan.max_block_points != preflight.max_block_points ||
        checked_c1_sum(preflight.nocc, preflight.nvir, prefix) != preflight.nmo)
        throw PSIEXCEPTION(prefix + "preflight and aggregate resource plan differ");

    // The four reviewed physical primitives are each constructed exactly once.
    const auto c1 = detail::construct_restricted_c1_primitives(context_);
    const auto transition_count = c1.transitions.size();
    if (transition_count != provider_plan.transition_count ||
        c1.orbital_gaps.size() != transition_count)
        throw PSIEXCEPTION(prefix + "restricted C1 transition metadata is inconsistent");

    const auto alda = detail::construct_restricted_alda_kernel(context_, false);
    if (alda.transitions != c1.transitions || !alda.full_alda ||
        alda.full_alda->nirrep() != 1 ||
        static_cast<std::size_t>(alda.full_alda->nrow()) != transition_count ||
        alda.full_alda->ncol() != alda.full_alda->nrow())
        throw PSIEXCEPTION(prefix +
                           "restricted C1 and full ALDA transition ordering/dimensions differ");
    const auto hessian = detail::assemble_restricted_singlet_hessian(
        c1.orbital_gaps, *c1.coulomb, *c1.exchange_direct,
        *c1.exchange_transpose, *alda.full_alda, kernel_);

    const auto projection = project_transition_multipoles(context_, isa_weights_);
    if (projection.transitions != c1.transitions || !projection.values ||
        projection.values->nirrep() != 1 ||
        static_cast<std::size_t>(projection.values->nrow()) != provider_plan.component_count ||
        static_cast<std::size_t>(projection.values->ncol()) != transition_count)
        throw PSIEXCEPTION(prefix +
                           "ISA projection transition ordering/dimensions differ from the response Hessian");

    Matrix identity(static_cast<int>(transition_count),
                    static_cast<int>(transition_count));
    for (std::size_t transition = 0; transition < transition_count; ++transition)
        identity(transition, transition) = 1.0;

    // Results remain function-local until every solve, contraction, conversion,
    // and final basis check has succeeded: an exception publishes no prefix.
    std::vector<SitePairResponse> complete;
    complete.reserve(frequency_count);
    for (double omega : frequencies.frequencies) {
        const auto response = detail::solve_dense_restricted_response(
            *hessian.H1, *hessian.H2, omega, identity);
        const auto contracted = detail::contract_site_pair_response(
            site_count, *projection.values, response);
        if (!contracted.values || contracted.values->nirrep() != 1 ||
            static_cast<std::size_t>(contracted.values->nrow()) !=
                provider_plan.component_count ||
            contracted.values->ncol() != contracted.values->nrow())
            throw PSIEXCEPTION(prefix + "site-pair contraction dimensions are inconsistent");

        SitePairResponse result;
        result.frequency = omega;
        result.positions = context_->sites();
        result.blocks.resize(checked_c1_product(site_count, site_count, prefix));
        for (std::size_t response_site = 0; response_site < site_count;
             ++response_site) {
            for (std::size_t source_site = 0; source_site < site_count;
                 ++source_site) {
                auto& block = result.blocks[response_site * site_count + source_site];
                for (std::size_t response_component = 0; response_component < 16;
                     ++response_component) {
                    for (std::size_t source_component = 0; source_component < 16;
                         ++source_component) {
                        const double value = (*contracted.values)(
                            response_site * 16 + response_component,
                            source_site * 16 + source_component);
                        if (!std::isfinite(value))
                            throw PSIEXCEPTION(prefix +
                                               "site-pair response contains a nonfinite value");
                        block[response_component][source_component] = value;
                    }
                }
            }
        }
        complete.push_back(std::move(result));
    }
    context_->verify_basis_unchanged();
    return complete;
}

PointResponseData::PointResponseData(
    std::vector<SitePosition> points, std::vector<double> frequencies,
    std::vector<SharedMatrix> responses,
    std::vector<PointResponseDiagnostics> diagnostics, PointResponsePlan plan,
    SharedMatrix transition_potentials)
    : points_(std::move(points)),
      frequencies_(std::move(frequencies)),
      responses_(std::move(responses)),
      transition_potentials_test_only_(std::move(transition_potentials)),
      diagnostics_(std::move(diagnostics)),
      plan_(std::move(plan)) {
    if (points_.empty() || frequencies_.empty() ||
        responses_.size() != frequencies_.size() ||
        diagnostics_.size() != frequencies_.size() ||
        plan_.frequency_count != frequencies_.size() ||
        plan_.point_count != points_.size())
        throw PSIEXCEPTION("PointResponseData: carrier dimensions are inconsistent");
    for (std::size_t index = 0; index < frequencies_.size(); ++index) {
        const auto& diagnostic = diagnostics_[index];
        if (!std::isfinite(frequencies_[index]) ||
            diagnostic.frequency != frequencies_[index] ||
            !std::isfinite(diagnostic.reciprocal_condition) ||
            !std::isfinite(diagnostic.reciprocal_pivot_growth) ||
            !std::isfinite(diagnostic.max_forward_error) ||
            !std::isfinite(diagnostic.max_backward_error) ||
            !std::isfinite(diagnostic.max_scaled_residual) ||
            !std::isfinite(diagnostic.max_solution_scale) ||
            !std::isfinite(diagnostic.allowed_antisymmetry) ||
            !std::isfinite(diagnostic.symmetry_residual) ||
            !std::isfinite(diagnostic.max_normalized_antisymmetry) ||
            diagnostic.max_forward_error < 0.0 ||
            diagnostic.max_backward_error < 0.0 ||
            diagnostic.max_scaled_residual < 0.0 ||
            diagnostic.max_solution_scale < 0.0 ||
            !responses_[index] || responses_[index]->nirrep() != 1 ||
            static_cast<std::size_t>(responses_[index]->nrow()) != points_.size() ||
            responses_[index]->ncol() != responses_[index]->nrow())
            throw PSIEXCEPTION("PointResponseData: frequency-major payload is inconsistent");
    }
}

namespace detail {
struct PointResponseBuilder {
    static PointResponseData create(
        std::vector<SitePosition> points, std::vector<double> frequencies,
        std::vector<SharedMatrix> responses,
        std::vector<PointResponseDiagnostics> diagnostics, PointResponsePlan plan,
        SharedMatrix transition_potentials) {
        return PointResponseData(
            std::move(points), std::move(frequencies), std::move(responses),
            std::move(diagnostics), std::move(plan),
            std::move(transition_potentials));
    }
};
}  // namespace detail

SharedMatrix PointResponseData::response_clone(std::size_t frequency) const {
    if (frequency >= responses_.size())
        throw PSIEXCEPTION("point response: frequency index is out of range");
    return responses_[frequency]->clone();
}

std::vector<SharedMatrix> PointResponseData::response_clones() const {
    std::vector<SharedMatrix> clones;
    clones.reserve(responses_.size());
    for (const auto& response : responses_) clones.push_back(response->clone());
    return clones;
}

SharedMatrix PointResponseData::transition_potentials_clone_test_only() const {
    if (!transition_potentials_test_only_)
        throw PSIEXCEPTION("point response: test transition potentials are unavailable");
    return transition_potentials_test_only_->clone();
}

namespace {
bool preflight_point_response_request(
    const std::shared_ptr<const FrozenResponseContext>& context,
    const std::vector<double>& frequencies,
    const std::vector<SitePosition>& points,
    double minimum_site_distance_bohr) {
    const std::string prefix = "point response: ";
    if (!context || !context->basis())
        throw PSIEXCEPTION(prefix + "frozen response context/basis is unavailable");
    if (frequencies.empty())
        throw PSIEXCEPTION(prefix + "frequency list requires at least one value");
    bool has_dynamic_frequency = false;
    for (double frequency : frequencies) {
        if (!std::isfinite(frequency))
            throw PSIEXCEPTION(prefix + "frequencies must be finite and nonnegative");
        if (frequency < 0.0)
            throw PSIEXCEPTION(prefix + "frequencies must be nonnegative");
        has_dynamic_frequency = has_dynamic_frequency || frequency != 0.0;
    }
    if (points.empty()) throw PSIEXCEPTION(prefix + "requires at least one point");
    for (const auto& point : points)
        for (double coordinate : point)
            if (!std::isfinite(coordinate))
                throw PSIEXCEPTION(prefix + "point coordinates must be finite");
    if (!std::isfinite(minimum_site_distance_bohr) ||
        minimum_site_distance_bohr < 0.0)
        throw PSIEXCEPTION(prefix +
                           "minimum site distance must be finite and nonnegative");
    return has_dynamic_frequency;
}

void validate_point_response_locations(
    const FrozenResponseContext& context,
    const std::vector<SitePosition>& points,
    double minimum_site_distance_bohr) {
    const std::string prefix = "point response: ";
    for (std::size_t first = 0; first < points.size(); ++first) {
        for (std::size_t second = first + 1; second < points.size(); ++second)
            if (points[first] == points[second])
                throw PSIEXCEPTION(prefix +
                                   "caller-supplied points must be distinct");
        if (minimum_site_distance_bohr > 0.0) {
            for (const auto& site : context.sites()) {
                double distance_squared = 0.0;
                for (std::size_t axis = 0; axis < 3; ++axis) {
                    const double displacement = points[first][axis] - site[axis];
                    distance_squared += displacement * displacement;
                }
                if (distance_squared <
                    minimum_site_distance_bohr * minimum_site_distance_bohr)
                    throw PSIEXCEPTION(prefix +
                                       "point violates the requested minimum site distance");
            }
        }
    }
}

PointResponseData evaluate_point_response_with_operator(
    const std::shared_ptr<const FrozenResponseContext>& context,
    const Matrix& H1, const Matrix& H2, const std::vector<double>& frequencies,
    const std::vector<SitePosition>& points,
    const std::vector<std::pair<std::size_t, std::size_t>>& transitions,
    const PointResponsePlan& plan) {
    const std::string prefix = "point response: ";
    const auto nbf = plan.nbf;
    const auto nov = plan.transition_count;
    const int nbf_int = static_cast<int>(nbf);
    require_dense_response_operator(H1, "H1");
    require_dense_response_operator(H2, "H2");
    if (H1.nrow() != H2.nrow() || static_cast<std::size_t>(H1.nrow()) != nov)
        throw PSIEXCEPTION(prefix +
                           "response operator dimension does not match occupied-virtual transitions");
    if (transitions.size() != nov)
        throw PSIEXCEPTION(prefix +
                           "occupied-major transition metadata are inconsistent");
    auto transition_potentials = std::make_shared<Matrix>(
        static_cast<int>(nov), static_cast<int>(points.size()));
    // MintsHelper aliases the already sealed basis only to construct native
    // one-electron integral engines; no basis mutation is performed.
    MintsHelper mints(std::const_pointer_cast<BasisSet>(context->basis()));
    const auto& coefficients = *context->Ca();
    for (std::size_t point_index = 0; point_index < points.size(); ++point_index) {
        const auto potentials = mints.ao_multipole_potential(0, {
            points[point_index][0], points[point_index][1], points[point_index][2]});
        if (potentials.size() != 1 || !potentials[0] || potentials[0]->nirrep() != 1 ||
            potentials[0]->nrow() != nbf_int || potentials[0]->ncol() != nbf_int)
            throw PSIEXCEPTION(prefix + "native order-zero AO potential dimensions are inconsistent");
        const auto& ao_potential = *potentials[0];
        for (std::size_t transition = 0; transition < nov; ++transition) {
            const auto occupied = transitions[transition].first;
            const auto virtual_orbital = transitions[transition].second;
            double value = 0.0;
            for (std::size_t mu = 0; mu < nbf; ++mu) {
                for (std::size_t nu = 0; nu < nbf; ++nu) {
                    const double integral = ao_potential(mu, nu);
                    if (!std::isfinite(integral))
                        throw PSIEXCEPTION(prefix + "native order-zero AO potential is nonfinite");
                    value += coefficients(mu, occupied) * integral *
                             coefficients(nu, virtual_orbital);
                }
            }
            if (!std::isfinite(value))
                throw PSIEXCEPTION(prefix + "MO transition potential is nonfinite");
            (*transition_potentials)(transition, point_index) = value;
        }
    }

    std::vector<SharedMatrix> complete_responses;
    std::vector<PointResponseDiagnostics> complete_diagnostics;
    complete_responses.reserve(frequencies.size());
    complete_diagnostics.reserve(frequencies.size());
    for (double frequency : frequencies) {
        const auto solved = detail::solve_dense_restricted_response(
            H1, H2, frequency, *transition_potentials);
        const auto amplitudes = solved.P_clone();
        auto raw = std::make_shared<Matrix>(static_cast<int>(points.size()),
                                            static_cast<int>(points.size()));
        for (std::size_t response_point = 0; response_point < points.size(); ++response_point) {
            for (std::size_t source_point = 0; source_point < points.size(); ++source_point) {
                double value = 0.0;
                for (std::size_t transition = 0; transition < nov; ++transition)
                    value += (*transition_potentials)(transition, response_point) *
                             (*amplitudes)(transition, source_point);
                value *= 4.0;
                if (!std::isfinite(value))
                    throw PSIEXCEPTION(prefix + "contracted response is nonfinite");
                (*raw)(response_point, source_point) = value;
            }
        }

        double allowed_antisymmetry = 0.0;
        double symmetry_residual = 0.0;
        double max_normalized_antisymmetry = 0.0;
        for (std::size_t first = 0; first < points.size(); ++first) {
            double first_l1 = 0.0;
            for (std::size_t transition = 0; transition < nov; ++transition)
                first_l1 += std::abs((*transition_potentials)(transition, first));
            for (std::size_t second = first + 1; second < points.size(); ++second) {
                double second_l1 = 0.0;
                for (std::size_t transition = 0; transition < nov; ++transition)
                    second_l1 += std::abs((*transition_potentials)(transition, second));
                const double scale = std::max(
                    {1.0, std::abs((*raw)(first, second)), std::abs((*raw)(second, first))});
                const double allowed =
                    4.0 * (first_l1 * solved.forward_error()[second] *
                               solved.solution_column_scales()[second] +
                           second_l1 * solved.forward_error()[first] *
                               solved.solution_column_scales()[first]) +
                    128.0 * std::numeric_limits<double>::epsilon() * scale *
                        std::max<std::size_t>(1, nov);
                const double residual = std::abs(
                    (*raw)(first, second) - (*raw)(second, first));
                if (!std::isfinite(allowed) || residual > allowed)
                    throw PSIEXCEPTION(prefix + "point response failed the solver-derived symmetry policy");
                allowed_antisymmetry = std::max(allowed_antisymmetry, allowed);
                symmetry_residual = std::max(symmetry_residual, residual);
                max_normalized_antisymmetry = std::max(
                    max_normalized_antisymmetry, allowed == 0.0 ? 0.0 : residual / allowed);
            }
        }

        auto symmetric = std::make_shared<Matrix>(static_cast<int>(points.size()),
                                                  static_cast<int>(points.size()));
        for (std::size_t row = 0; row < points.size(); ++row) {
            (*symmetric)(row, row) = (*raw)(row, row);
            for (std::size_t column = row + 1; column < points.size(); ++column) {
                const double average = 0.5 * ((*raw)(row, column) + (*raw)(column, row));
                (*symmetric)(row, column) = average;
                (*symmetric)(column, row) = average;
            }
        }

        PointResponseDiagnostics diagnostic;
        diagnostic.frequency = frequency;
        diagnostic.reciprocal_condition = solved.reciprocal_condition();
        diagnostic.reciprocal_pivot_growth = solved.reciprocal_pivot_growth();
        diagnostic.max_forward_error = *std::max_element(
            solved.forward_error().begin(), solved.forward_error().end());
        diagnostic.max_backward_error = *std::max_element(
            solved.backward_error().begin(), solved.backward_error().end());
        diagnostic.max_scaled_residual = *std::max_element(
            solved.scaled_residual().begin(), solved.scaled_residual().end());
        diagnostic.max_solution_scale = *std::max_element(
            solved.solution_column_scales().begin(),
            solved.solution_column_scales().end());
        diagnostic.allowed_antisymmetry = allowed_antisymmetry;
        diagnostic.symmetry_residual = symmetry_residual;
        diagnostic.max_normalized_antisymmetry = max_normalized_antisymmetry;
        diagnostic.reciprocity_enforced = true;
        complete_responses.push_back(std::move(symmetric));
        complete_diagnostics.push_back(std::move(diagnostic));
    }
    context->verify_basis_unchanged();
    return detail::PointResponseBuilder::create(
        points, frequencies, std::move(complete_responses),
        std::move(complete_diagnostics), plan,
        std::move(transition_potentials));
}
}  // namespace

PointResponseData detail::evaluate_raw_point_response_test_only(
    const std::shared_ptr<const FrozenResponseContext>& context,
    const Matrix& H1, const Matrix& H2, const std::vector<double>& frequencies,
    const std::vector<SitePosition>& points,
    const std::vector<std::size_t>& transition_permutation,
    double minimum_site_distance_bohr) {
    const std::string prefix = "test-only raw point response: ";
    const bool has_dynamic_frequency = preflight_point_response_request(
        context, frequencies, points, minimum_site_distance_bohr);
    const auto preflight = preflight_isapol_response_provider(context, false);
    const auto plan = plan_point_response(
        frequencies.size(), preflight.nbf, preflight.nocc, preflight.nvir,
        points.size(), has_dynamic_frequency,
        Process::environment.get_memory());
    context->verify_basis_unchanged();
    validate_point_response_locations(
        *context, points, minimum_site_distance_bohr);

    auto transitions = make_restricted_alda_transitions(*context, preflight.nov);
    if (!transition_permutation.empty()) {
        if (transition_permutation.size() != transitions.size())
            throw PSIEXCEPTION(prefix +
                               "transition permutation has the wrong dimension");
        std::vector<bool> seen(transitions.size(), false);
        std::vector<std::pair<std::size_t, std::size_t>> permuted;
        permuted.reserve(transitions.size());
        for (const auto source : transition_permutation) {
            if (source >= transitions.size() || seen[source])
                throw PSIEXCEPTION(prefix +
                                   "transition order must be a permutation");
            seen[source] = true;
            permuted.push_back(transitions[source]);
        }
        transitions = std::move(permuted);
    }
    return evaluate_point_response_with_operator(
        context, H1, H2, frequencies, points, transitions, plan);
}

PointResponseData evaluate_point_response(
    const std::shared_ptr<const FrozenResponseContext>& context,
    const ResponseKernel& kernel, const std::vector<double>& frequencies,
    const std::vector<SitePosition>& points,
    double minimum_site_distance_bohr) {
    const std::string prefix = "point response: ";
    const bool has_dynamic_frequency = preflight_point_response_request(
        context, frequencies, points, minimum_site_distance_bohr);

    // Allocation-light context inspection and every aggregate stage estimate
    // precede snapshot verification and all canonical physical construction.
    const auto preflight = preflight_isapol_response_provider(context, false);
    const auto plan = detail::plan_point_response_provider(
        frequencies.size(), preflight.nbf, preflight.nocc, preflight.nvir,
        points.size(), context->grid_blocks(), has_dynamic_frequency,
        Process::environment.get_memory(),
        context->functional_density_tolerance());
    context->verify_basis_unchanged();
    validate_point_response_locations(
        *context, points, minimum_site_distance_bohr);

    const auto c1 = detail::construct_restricted_c1_primitives(context);
    const auto alda = detail::construct_restricted_alda_kernel(context, false);
    if (c1.transitions != alda.transitions ||
        c1.transitions.size() != plan.transition_count ||
        c1.orbital_gaps.size() != c1.transitions.size() || !alda.full_alda ||
        alda.full_alda->nirrep() != 1 ||
        static_cast<std::size_t>(alda.full_alda->nrow()) !=
            c1.transitions.size() ||
        alda.full_alda->ncol() != alda.full_alda->nrow())
        throw PSIEXCEPTION(prefix +
                           "canonical C1 and full-ALDA transition identities differ");
    const auto hessian = detail::assemble_restricted_singlet_hessian(
        c1.orbital_gaps, *c1.coulomb, *c1.exchange_direct,
        *c1.exchange_transpose, *alda.full_alda, kernel);
    return evaluate_point_response_with_operator(
        context, *hessian.H1, *hessian.H2, frequencies, points,
        c1.transitions, plan);
}

namespace {

constexpr std::size_t kWSMComponents = 15;
constexpr std::size_t kWSMVariablesPerSite = 120;
constexpr std::size_t kWSMMaximumPoints = 500;
constexpr std::size_t kWSMMaximumVariables = 360;

std::size_t wsm_upper_index(std::size_t first, std::size_t second) {
    return first * kWSMComponents - first * (first - 1) / 2 + (second - first);
}

L3WorkingVector irregular_harmonics(const SitePosition& point, const SitePosition& site) {
    SitePosition displacement{};
    double radius_squared = 0.0;
    for (std::size_t axis = 0; axis < 3; ++axis) {
        if (!std::isfinite(point[axis]) || !std::isfinite(site[axis]))
            throw PSIEXCEPTION("WSM refinement: point and site positions must be finite");
        displacement[axis] = point[axis] - site[axis];
        radius_squared += displacement[axis] * displacement[axis];
    }
    if (!std::isfinite(radius_squared) || radius_squared <= 1.0e-24)
        throw PSIEXCEPTION("WSM refinement: external point is singular or near a refinement site");
    const auto regular = regular_harmonics(displacement);
    L3WorkingVector result{};
    result[0] = 1.0 / std::sqrt(radius_squared);
    double denominator = radius_squared * std::sqrt(radius_squared);
    for (std::size_t rank = 1; rank <= 3; ++rank) {
        for (std::size_t component = rank * rank; component < (rank + 1) * (rank + 1); ++component) {
            result[component] = regular[component] / denominator;
            require_finite(result[component], "WSM irregular harmonics");
        }
        denominator *= radius_squared;
    }
    return result;
}

void validate_wsm_policy(const RefinementOptions& options) {
    if (options.wsm_rank != 3 || options.hydrogen_rank != 3 || options.weight_type != 4 ||
        options.weight_coefficient != 0.001 || options.cutoff != 1.0e-4)
        throw PSIEXCEPTION("WSM refinement: only the exact rank-3/rank-3 weight-type-4 physical policy is supported");
    if (!std::isfinite(options.maximum_condition_number) || options.maximum_condition_number < 1.0)
        throw PSIEXCEPTION("WSM refinement: maximum condition number must be finite and at least one");
}

RefinedL3Model refine_wsm_frequency(const LocalizedResponse& localized,
                                    const PointResponseData& point_response,
                                    std::size_t frequency_index,
                                    const PDefConstraints& constraints,
                                    const RefinementOptions& options) {
    const std::string prefix = "WSM refinement: ";
    validate_wsm_policy(options);
    if (frequency_index >= point_response.frequency_count())
        throw PSIEXCEPTION(prefix + "frequency index is out of range");
    const auto& points = point_response.points();
    const std::size_t site_count = localized.local.size();
    if (site_count == 0 || localized.positions.size() != site_count)
        throw PSIEXCEPTION(prefix + "localized tensors and physical site positions are inconsistent");
    const auto full_variable_count = checked_c1_product(
        site_count, kWSMVariablesPerSite, prefix);
    if (full_variable_count > kWSMMaximumVariables)
        throw PSIEXCEPTION(prefix + "variable envelope is at most 360 (three sites)");
    if (!constraints.active_variables.empty() &&
        constraints.active_variables.size() != full_variable_count)
        throw PSIEXCEPTION(prefix + "active-variable mask must contain 120 entries per site");
    std::vector<bool> active = constraints.active_variables;
    if (active.empty()) active.assign(full_variable_count, true);
    std::vector<std::size_t> active_to_full;
    active_to_full.reserve(full_variable_count);
    for (std::size_t variable = 0; variable < full_variable_count; ++variable)
        if (active[variable]) active_to_full.push_back(variable);
    if (active_to_full.empty()) throw PSIEXCEPTION(prefix + "at least one variable must be active");

    std::size_t input_constraint_rows = 0;
    if (constraints.equality) {
        if (constraints.equality->nirrep() != 1 ||
            static_cast<std::size_t>(constraints.equality->ncol()) != full_variable_count ||
            constraints.equality_targets.size() != static_cast<std::size_t>(constraints.equality->nrow()))
            throw PSIEXCEPTION(prefix + "equality constraints must be C(rows, 120*sites) and d(rows)");
        input_constraint_rows = static_cast<std::size_t>(constraints.equality->nrow());
        if (input_constraint_rows > full_variable_count)
            throw PSIEXCEPTION(prefix + "equality constraint rows exceed the variable envelope");
    } else if (!constraints.equality_targets.empty()) {
        throw PSIEXCEPTION(prefix + "equality targets require an equality matrix");
    }
    std::size_t effective_constraint_rows = 0;
    for (std::size_t row = 0; row < input_constraint_rows; ++row) {
        if (!std::isfinite(constraints.equality_targets[row]))
            throw PSIEXCEPTION(prefix + "equality constraints must be finite");
        bool active_nonzero = false;
        for (std::size_t variable = 0; variable < full_variable_count; ++variable) {
            const double value = (*constraints.equality)(row, variable);
            if (!std::isfinite(value))
                throw PSIEXCEPTION(prefix + "equality constraints must be finite");
            active_nonzero = active_nonzero || (active[variable] && value != 0.0);
        }
        if (active_nonzero) ++effective_constraint_rows;
        else if (constraints.equality_targets[row] != 0.0)
            throw PSIEXCEPTION(prefix + "equality constraints are inconsistent with inactive zero variables");
    }
    if (effective_constraint_rows > active_to_full.size())
        throw PSIEXCEPTION(prefix + "effective equality constraints exceed active variables");

    const double localized_frequency = localized.frequency;
    const double point_frequency = point_response.frequencies()[frequency_index];
    const double frequency_tolerance = 16.0 * std::numeric_limits<double>::epsilon() *
        std::max({1.0, std::abs(localized_frequency), std::abs(point_frequency)});
    if (!std::isfinite(localized_frequency) ||
        std::abs(localized_frequency - point_frequency) > frequency_tolerance)
        throw PSIEXCEPTION(prefix + "localized and point-response frequencies do not align");

    for (const auto& block : localized.local) {
        for (std::size_t row = 0; row < kWSMComponents; ++row) {
            for (std::size_t column = 0; column < kWSMComponents; ++column) {
                if (!std::isfinite(block[row][column]))
                    throw PSIEXCEPTION(prefix + "localized reference must be finite");
                const double scale = 1.0 + std::max(std::abs(block[row][column]),
                                                    std::abs(block[column][row]));
                if (std::abs(block[row][column] - block[column][row]) > 1.0e-12 * scale)
                    throw PSIEXCEPTION(prefix + "localized reference must be symmetric");
            }
        }
    }

    const auto plan = detail::plan_wsm_refinement(
        points.size(), site_count, active_to_full.size(), effective_constraint_rows,
        Process::environment.get_memory());
    // All dimensional, numeric, frequency, and resource gates precede dense allocations.
    std::vector<L3WorkingVector> irregular(points.size() * site_count);
    for (std::size_t point = 0; point < points.size(); ++point)
        for (std::size_t site = 0; site < site_count; ++site)
            irregular[point * site_count + site] =
                irregular_harmonics(points[point], localized.positions[site]);

    struct VariableIdentity { std::size_t site, first, second; };
    std::vector<VariableIdentity> identities;
    identities.reserve(active_to_full.size());
    for (const auto full : active_to_full) {
        const std::size_t site = full / kWSMVariablesPerSite;
        const std::size_t within = full % kWSMVariablesPerSite;
        bool found = false;
        for (std::size_t first = 0; first < kWSMComponents && !found; ++first) {
            for (std::size_t second = first; second < kWSMComponents; ++second) {
                if (wsm_upper_index(first, second) == within) {
                    identities.push_back({site, first, second});
                    found = true;
                    break;
                }
            }
        }
        if (!found) throw PSIEXCEPTION(prefix + "internal variable ordering is inconsistent");
    }

    const auto response = point_response.response_clone(frequency_index);
    double response_scale = 0.0;
    for (std::size_t row = 0; row < points.size(); ++row) {
        for (std::size_t column = 0; column < points.size(); ++column) {
            if (!std::isfinite((*response)(row, column)))
                throw PSIEXCEPTION(prefix + "point response must be finite");
            response_scale = std::max(response_scale, std::abs((*response)(row, column)));
        }
    }
    for (std::size_t row = 0; row < points.size(); ++row)
        for (std::size_t column = row + 1; column < points.size(); ++column)
            if (std::abs((*response)(row, column) - (*response)(column, row)) >
                1.0e-10 * (1.0 + response_scale))
                throw PSIEXCEPTION(prefix + "PointResponseData response must be symmetric");

    Matrix design(static_cast<int>(plan.pair_rows), static_cast<int>(active_to_full.size()));
    std::vector<double> observations(plan.pair_rows);
    std::vector<double> row_weights(plan.pair_rows, 1.0);
    std::size_t pair = 0;
    for (std::size_t first_point = 0; first_point < points.size(); ++first_point) {
        for (std::size_t second_point = first_point; second_point < points.size(); ++second_point, ++pair) {
            observations[pair] = (*response)(first_point, second_point);
            if (first_point != second_point) row_weights[pair] = std::sqrt(2.0);
            for (std::size_t variable = 0; variable < identities.size(); ++variable) {
                const auto identity = identities[variable];
                const auto& first_irregular = irregular[first_point * site_count + identity.site];
                const auto& second_irregular = irregular[second_point * site_count + identity.site];
                double coefficient = first_irregular[identity.first + 1] *
                                     second_irregular[identity.second + 1];
                if (identity.first != identity.second)
                    coefficient += first_irregular[identity.second + 1] *
                                   second_irregular[identity.first + 1];
                if (!std::isfinite(coefficient))
                    throw PSIEXCEPTION(prefix + "design coefficient is nonfinite");
                design(pair, variable) = coefficient;
            }
        }
    }

    // The WSM policy cutoff is a RELATIVE rank threshold, not an atomic-unit
    // magnitude. solve_constrained_least_squares compares against an absolute
    // weighted column norm, so the protocol value is scaled here by the largest
    // weighted column norm of this design matrix.
    //
    // This matters physically, not just numerically. The irregular harmonics fall
    // off as r^-(2l+1), so every column norm shrinks as the fit points move
    // outward, and an absolute cutoff silently makes the retained rank a function
    // of the shell radii. Under the absolute reading the reviewed protocol's own
    // point grid (4.63 to 11.46 bohr from the nearest nucleus) prunes the rank-3
    // block and then fails closed in the constraint elimination with "constraints
    // are ambiguous (linearly dependent)" -- that is, the absolute reading cannot
    // express the reviewed protocol at all. The relative reading admits it.
    double maximum_weighted_column_norm = 0.0;
    for (std::size_t variable = 0; variable < active_to_full.size(); ++variable) {
        double norm = 0.0;
        for (std::size_t row = 0; row < plan.pair_rows; ++row)
            norm = std::hypot(norm, row_weights[row] * design(row, variable));
        if (!std::isfinite(norm))
            throw PSIEXCEPTION(prefix + "weighted design column norm overflowed");
        maximum_weighted_column_norm = std::max(maximum_weighted_column_norm, norm);
    }
    if (!std::isfinite(maximum_weighted_column_norm) || !(maximum_weighted_column_norm > 0.0))
        throw PSIEXCEPTION(prefix + "design matrix has no nonzero weighted column");

    std::vector<double> anchor(active_to_full.size(), 0.0);
    std::vector<double> reference(active_to_full.size(), 0.0);
    std::size_t anchor_count = 0;
    for (std::size_t variable = 0; variable < identities.size(); ++variable) {
        const auto identity = identities[variable];
        reference[variable] = localized.local[identity.site][identity.first][identity.second];
        if (identity.first == identity.second && identity.first < 3) {
            anchor[variable] = 1.0;
            ++anchor_count;
        }
    }

    std::vector<std::size_t> effective_rows;
    effective_rows.reserve(effective_constraint_rows);
    for (std::size_t row = 0; row < input_constraint_rows; ++row) {
        bool active_nonzero = false;
        for (const auto variable : active_to_full)
            active_nonzero = active_nonzero || (*constraints.equality)(row, variable) != 0.0;
        if (active_nonzero) effective_rows.push_back(row);
    }
    Matrix reduced_constraints(static_cast<int>(effective_rows.size()),
                               static_cast<int>(active_to_full.size()));
    std::vector<double> reduced_targets(effective_rows.size());
    for (std::size_t row = 0; row < effective_rows.size(); ++row) {
        reduced_targets[row] = constraints.equality_targets[effective_rows[row]];
        for (std::size_t variable = 0; variable < active_to_full.size(); ++variable)
            reduced_constraints(row, variable) =
                (*constraints.equality)(effective_rows[row], active_to_full[variable]);
    }

    detail::ConstrainedLeastSquaresOptions solve_options;
    solve_options.column_cutoff = options.cutoff * maximum_weighted_column_norm;
    solve_options.prune_below_cutoff = true;
    solve_options.maximum_condition_number = options.maximum_condition_number;
    solve_options.maximum_workspace_elements = plan.workspace_elements;
    const auto solved = detail::solve_constrained_least_squares(
        design, observations, row_weights, options.weight_coefficient, anchor, reference,
        reduced_constraints, reduced_targets, solve_options);

    RefinedL3Model pending;
    pending.frequency = point_response.frequencies()[frequency_index];
    pending.positions = localized.positions;
    pending.tensors.resize(site_count);
    pending.diagnostics.point_count = points.size();
    pending.diagnostics.pair_rows = plan.pair_rows;
    pending.diagnostics.variable_count = full_variable_count;
    pending.diagnostics.active_variable_count = active_to_full.size();
    pending.diagnostics.anchor_variable_count = anchor_count;
    pending.diagnostics.solution.assign(full_variable_count, 0.0);
    for (std::size_t variable = 0; variable < active_to_full.size(); ++variable)
        pending.diagnostics.solution[active_to_full[variable]] = solved.solution[variable];
    for (const auto variable : solved.kept_columns)
        pending.diagnostics.kept_variables.push_back(active_to_full[variable]);
    for (const auto variable : solved.pruned_columns)
        pending.diagnostics.pruned_variables.push_back(active_to_full[variable]);
    pending.diagnostics.condition_number = solved.condition_number;
    pending.diagnostics.weighted_residual_norm = solved.weighted_residual_norm;
    pending.diagnostics.anchor_residual_norm = solved.anchor_residual_norm;
    pending.diagnostics.constraint_residual_norm = solved.constraint_residual_norm;
    pending.diagnostics.objective_residual_norm = solved.objective_residual_norm;
    pending.diagnostics.maximum_weighted_column_norm = maximum_weighted_column_norm;
    pending.diagnostics.applied_column_cutoff = solve_options.column_cutoff;
    pending.diagnostics.row_weight_source = "full_symmetric_frobenius";
    pending.diagnostics.plan = plan;
    for (std::size_t site = 0; site < site_count; ++site) {
        for (std::size_t first = 0; first < kWSMComponents; ++first) {
            for (std::size_t second = first; second < kWSMComponents; ++second) {
                const double value = pending.diagnostics.solution[
                    site * kWSMVariablesPerSite + wsm_upper_index(first, second)];
                pending.tensors[site][first][second] = value;
                pending.tensors[site][second][first] = value;
                pending.diagnostics.max_output_asymmetry = std::max(
                    pending.diagnostics.max_output_asymmetry,
                    std::abs(pending.tensors[site][first][second] -
                             pending.tensors[site][second][first]));
            }
        }
    }
    for (std::size_t row = 0; row < plan.pair_rows; ++row) {
        double predicted = 0.0;
        for (std::size_t variable = 0; variable < active_to_full.size(); ++variable)
            predicted += design(row, variable) * solved.solution[variable];
        pending.diagnostics.max_point_residual = std::max(
            pending.diagnostics.max_point_residual, std::abs(predicted - observations[row]));
    }
    return pending;
}

}  // namespace

namespace detail {

WSMRefinementPlan plan_wsm_refinement(std::size_t point_count, std::size_t site_count,
                                      std::size_t active_variable_count,
                                      std::size_t constraint_rows,
                                      std::size_t memory_bytes) {
    const std::string prefix = "WSM refinement plan: ";
    if (point_count == 0 || point_count > kWSMMaximumPoints)
        throw PSIEXCEPTION(prefix + "point envelope is 1 through 500");
    if (site_count == 0) throw PSIEXCEPTION(prefix + "at least one site is required");
    const auto variable_count = checked_c1_product(site_count, kWSMVariablesPerSite, prefix);
    if (variable_count > kWSMMaximumVariables)
        throw PSIEXCEPTION(prefix + "variable envelope is at most 360 (three sites)");
    if (active_variable_count == 0 || active_variable_count > variable_count)
        throw PSIEXCEPTION(prefix + "active-variable count is outside the variable envelope");
    if (constraint_rows > active_variable_count)
        throw PSIEXCEPTION(prefix + "constraint rows exceed active variables");
    const auto adjacent = checked_c1_sum(point_count, 1, prefix);
    const auto doubled_pairs = checked_c1_product(point_count, adjacent, prefix);
    const auto pair_rows = doubled_pairs / 2;
    const auto design_elements = checked_c1_product(pair_rows, active_variable_count, prefix);
    const auto irregular_elements = checked_c1_product(
        checked_c1_product(point_count, site_count, prefix), std::size_t{16}, prefix);
    const auto response_elements = checked_c1_product(point_count, point_count, prefix);
    const auto constraint_elements = checked_c1_product(constraint_rows, active_variable_count, prefix);
    const auto square_variables = checked_c1_product(active_variable_count, active_variable_count, prefix);
    const auto bytes = [&prefix](std::size_t elements) {
        return checked_c1_product(elements, sizeof(double), prefix);
    };
    const auto base_elements = checked_c1_sum(
        checked_c1_sum(design_elements, irregular_elements, prefix), response_elements, prefix);
    const auto null_space_elements = square_variables;
    const auto solver_base_elements = checked_c1_sum(base_elements, null_space_elements, prefix);
    // DGESVD and DGESDD work arrays are sequential, so one enforceable cap is live.
    // 64 times all quadratic/constraint/row drivers is deliberately above documented
    // LAPACK minima while remaining overflow-checked and charged to the peak.
    const auto workspace_drivers = checked_c1_sum(
        square_variables, checked_c1_sum(constraint_elements,
            checked_c1_sum(pair_rows, active_variable_count, prefix), prefix), prefix);
    const auto workspace_elements = checked_c1_product(
        workspace_drivers, std::size_t{64}, prefix);

    std::size_t constraint_peak_elements = solver_base_elements;
    if (constraint_rows != 0) {
        const auto three_constraint_matrices = checked_c1_product(constraint_elements, std::size_t{3}, prefix);
        const auto two_u = checked_c1_product(constraint_elements, std::size_t{2}, prefix);
        const auto two_vt = checked_c1_product(square_variables, std::size_t{2}, prefix);
        constraint_peak_elements = checked_c1_sum(
            constraint_peak_elements,
            checked_c1_sum(three_constraint_matrices,
                checked_c1_sum(two_u, checked_c1_sum(two_vt,
                    checked_c1_sum(workspace_elements, constraint_rows, prefix), prefix), prefix), prefix), prefix);
    }

    const auto free_variable_count = active_variable_count - constraint_rows;
    const auto augmented_rows = checked_c1_sum(pair_rows, active_variable_count, prefix);
    const auto augmented_elements = checked_c1_product(augmented_rows, free_variable_count, prefix);
    const auto square_free = checked_c1_product(free_variable_count, free_variable_count, prefix);
    std::size_t fit_peak_elements = solver_base_elements;
    std::size_t fit_integer_workspace_bytes = 0;
    if (free_variable_count != 0) {
        const auto four_augmented = checked_c1_product(augmented_elements, std::size_t{4}, prefix);
        fit_peak_elements = checked_c1_sum(
            solver_base_elements, checked_c1_sum(four_augmented,
                checked_c1_sum(checked_c1_product(square_free, std::size_t{2}, prefix),
                               workspace_elements, prefix), prefix), prefix);
        fit_integer_workspace_bytes = checked_c1_product(
            checked_c1_product(free_variable_count, std::size_t{8}, prefix), sizeof(int), prefix);
    }
    const auto fit_peak_bytes = checked_c1_sum(bytes(fit_peak_elements), fit_integer_workspace_bytes, prefix);
    const auto constraint_peak_bytes = bytes(constraint_peak_elements);
    const auto estimated_bytes = std::max(fit_peak_bytes, constraint_peak_bytes);
    const auto reserved = memory_bytes / 2;
    if (estimated_bytes > reserved)
        throw PSIEXCEPTION(prefix + "dense economy-SVD design/constraint storage exceeds half the reserved memory");
    WSMRefinementPlan plan;
    plan.point_count = point_count;
    plan.pair_rows = pair_rows;
    plan.site_count = site_count;
    plan.variable_count = variable_count;
    plan.active_variable_count = active_variable_count;
    plan.constraint_rows = constraint_rows;
    plan.irregular_elements = irregular_elements;
    plan.response_clone_bytes = bytes(response_elements);
    plan.design_elements = design_elements;
    plan.design_bytes = bytes(design_elements);
    plan.constraint_matrix_bytes = bytes(constraint_elements);
    plan.null_space_elements = null_space_elements;
    plan.null_space_bytes = bytes(null_space_elements);
    plan.workspace_elements = workspace_elements;
    plan.workspace_bytes = bytes(workspace_elements);
    plan.constraint_svd_peak_bytes = constraint_peak_bytes;
    plan.fit_svd_peak_bytes = fit_peak_bytes;
    plan.estimated_bytes = estimated_bytes;
    plan.configured_memory_bytes = memory_bytes;
    plan.reserved_memory_bytes = reserved;
    plan.algorithm = "upper-point-pairs/site-major-upper-triangle/direct-economy-SVD";
    plan.memory_semantics = "hard half-memory gate over response clone, design, explicit active^2 null space, reduced constraint matrix, and one enforceable sequential LAPACK workspace cap with DGESVD/DGESDD copies/U/VT";
    return plan;
}

L3WorkingVector irregular_harmonics_test_only(const SitePosition& point,
                                               const SitePosition& site) {
    return irregular_harmonics(point, site);
}

std::vector<RefinedL3Model> refine_wsm_test_only(
    const std::vector<SitePosition>& points, const std::vector<double>& frequencies,
    const std::vector<SharedMatrix>& responses, const std::vector<SitePosition>& sites,
    const std::vector<std::vector<L3Matrix>>& localized,
    const std::vector<double>& localized_frequencies,
    const PDefConstraints& constraints, const RefinementOptions& options) {
    if (frequencies.size() != responses.size() || frequencies.size() != localized.size() ||
        frequencies.size() != localized_frequencies.size())
        throw PSIEXCEPTION("WSM refinement: frequency-major responses, localized responses, and frequencies must have equal counts");
    std::vector<PointResponseDiagnostics> diagnostics(frequencies.size());
    for (std::size_t index = 0; index < frequencies.size(); ++index) {
        diagnostics[index].frequency = frequencies[index];
        diagnostics[index].reciprocal_condition = 1.0;
        diagnostics[index].reciprocal_pivot_growth = 1.0;
        diagnostics[index].reciprocity_enforced = true;
    }
    PointResponsePlan carrier_plan{};
    carrier_plan.frequency_count = frequencies.size();
    carrier_plan.point_count = points.size();
    auto carrier = PointResponseBuilder::create(
        points, frequencies, responses, std::move(diagnostics), std::move(carrier_plan), nullptr);
    std::vector<LocalizedResponse> localized_responses(frequencies.size());
    for (std::size_t index = 0; index < frequencies.size(); ++index) {
        localized_responses[index].frequency = localized_frequencies[index];
        localized_responses[index].positions = sites;
        localized_responses[index].local = localized[index];
    }
    if (frequencies.size() == 1)
        return {refine_wsm(localized_responses.front(), carrier, constraints, options)};
    return refine_wsm(localized_responses, carrier, constraints, options);
}

}  // namespace detail

RefinedL3Model refine_wsm(const LocalizedResponse& localized,
                          const PointResponseData& point_response,
                          const PDefConstraints& constraints,
                          const RefinementOptions& options) {
    if (point_response.frequency_count() != 1)
        throw PSIEXCEPTION("WSM refinement: one-frequency API requires exactly one PointResponseData response");
    return refine_wsm_frequency(localized, point_response, 0, constraints, options);
}

std::vector<RefinedL3Model> refine_wsm(
    const std::vector<LocalizedResponse>& localized,
    const PointResponseData& point_response,
    const PDefConstraints& constraints,
    const RefinementOptions& options) {
    if (localized.size() != point_response.frequency_count())
        throw PSIEXCEPTION("WSM refinement: localized responses and frequency-major PointResponseData must have equal counts");
    std::vector<RefinedL3Model> result;
    result.reserve(localized.size());
    for (std::size_t frequency = 0; frequency < localized.size(); ++frequency)
        result.push_back(refine_wsm_frequency(
            localized[frequency], point_response, frequency, constraints, options));
    return result;
}

namespace {

/** Exact geometry tolerance for site-symmetry classification, in bohr; matches libmints. */
constexpr double kSiteSymmetryTolerance = DEFAULT_SYM_TOL;
/** Largest deviation tolerated before a local axis frame is rejected as non-diagonalizing. */
constexpr double kSiteAxisTolerance = 1.0e-10;
/**
 * Bounded derivation envelope. Sixteen sites cap the equality matrix at 1920 columns and
 * 1920 rows; refine_wsm applies its own stricter three-site variable envelope downstream.
 */
constexpr std::size_t kMaximumConstraintSites = 16;

/**
 * Monomial parity (px, py, pz) of the real solid harmonics in the 15-component L3 order
 * 10, 11c, 11s, 20, 21c, 21s, 22c, 22s, 30, 31c, 31s, 32c, 32s, 33c, 33s.
 *
 * Every real solid harmonic in this basis is a sum of monomials sharing one parity per axis,
 * so a diagonal sign operation multiplies it by an exact integer character.
 */
constexpr std::array<std::array<int, 3>, kWSMComponents> kL3MonomialParity = {{
    {{0, 0, 1}}, {{1, 0, 0}}, {{0, 1, 0}},
    {{0, 0, 0}}, {{1, 0, 1}}, {{0, 1, 1}}, {{0, 0, 0}}, {{1, 1, 0}},
    {{0, 0, 1}}, {{1, 0, 0}}, {{0, 1, 0}}, {{0, 0, 1}}, {{1, 1, 1}}, {{1, 0, 0}}, {{0, 1, 0}},
}};

/** Exact integer character of one L3 component under a diagonal sign operation. */
int l3_character(std::size_t component, const std::array<int, 3>& signs) {
    int character = 1;
    for (std::size_t axis = 0; axis < 3; ++axis)
        if (kL3MonomialParity[component][axis] != 0) character *= signs[axis];
    return character;
}

/** One libmints D2h generator reduced to its exact diagonal signs. */
struct DiagonalOperation {
    unsigned char bit{};
    std::array<int, 3> signs{{1, 1, 1}};
};

/**
 * Read the eight D2h generators from libmints and keep only their diagonal signs.
 * libmints builds C2(z) from a finite rotation whose off-diagonal sines vanish only to
 * roundoff; the diagonal is the exact signed representation, as libmints itself relies on.
 */
std::vector<DiagonalOperation> d2h_diagonal_operations() {
    CharacterTable table(static_cast<unsigned char>(PointGroups::D2h));
    std::vector<DiagonalOperation> result;
    result.reserve(static_cast<std::size_t>(table.order()));
    for (int index = 0; index < table.order(); ++index) {
        const SymmetryOperation& operation = table.symm_operation(index);
        DiagonalOperation entry;
        entry.bit = operation.bit();
        for (int axis = 0; axis < 3; ++axis) {
            const double diagonal = operation(axis, axis);
            if (!std::isfinite(diagonal) || std::abs(std::abs(diagonal) - 1.0) > kSiteAxisTolerance)
                throw PSIEXCEPTION("PDef constraints: D2h generators must have unit diagonal signs");
            entry.signs[static_cast<std::size_t>(axis)] = diagonal < 0.0 ? -1 : 1;
        }
        result.push_back(entry);
    }
    return result;
}

/** Distinguishing nuclear identity used when matching symmetry images. */
struct SiteIdentity {
    int atomic_number{};
    double nuclear_charge{};
    double mass{};
};

/** The D2h subgroup exactly realized by the supplied frame, with its site permutations. */
struct FrameSymmetry {
    std::vector<DiagonalOperation> operations;
    std::vector<std::vector<std::size_t>> images;
    unsigned char bits{};
};

FrameSymmetry detect_frame_symmetry(const std::vector<SitePosition>& sites,
                                    const std::vector<SiteIdentity>& identities) {
    const std::string prefix = "PDef constraints: ";
    FrameSymmetry result;
    const auto candidates = d2h_diagonal_operations();
    result.operations.reserve(candidates.size());
    result.images.reserve(candidates.size());
    for (const auto& candidate : candidates) {
        std::vector<std::size_t> images(sites.size(), sites.size());
        bool complete = true;
        for (std::size_t site = 0; site < sites.size() && complete; ++site) {
            SitePosition imaged{};
            for (std::size_t axis = 0; axis < 3; ++axis)
                imaged[axis] = candidate.signs[axis] * sites[site][axis];
            for (std::size_t other = 0; other < sites.size(); ++other) {
                if (identities[other].atomic_number != identities[site].atomic_number) continue;
                if (std::abs(identities[other].nuclear_charge - identities[site].nuclear_charge) >
                        kSiteSymmetryTolerance ||
                    std::abs(identities[other].mass - identities[site].mass) > kSiteSymmetryTolerance)
                    continue;
                double deviation = 0.0;
                for (std::size_t axis = 0; axis < 3; ++axis)
                    deviation = std::max(deviation, std::abs(imaged[axis] - sites[other][axis]));
                if (deviation > kSiteSymmetryTolerance) continue;
                if (images[site] != sites.size())
                    throw PSIEXCEPTION(prefix + "sites coincide within the symmetry tolerance");
                images[site] = other;
            }
            if (images[site] == sites.size()) complete = false;
        }
        if (!complete) continue;
        result.operations.push_back(candidate);
        result.images.push_back(std::move(images));
        result.bits = static_cast<unsigned char>(result.bits | candidate.bit);
    }
    if (result.operations.empty())
        throw PSIEXCEPTION(prefix + "the identity is not a symmetry of the supplied geometry");
    return result;
}

void validate_site_axes(const SiteAxes& axes, std::size_t site) {
    const std::string prefix =
        "PDef constraints: site " + std::to_string(site) + " local axes ";
    for (std::size_t row = 0; row < 3; ++row)
        for (std::size_t column = 0; column < 3; ++column)
            if (!std::isfinite(axes[row][column])) throw PSIEXCEPTION(prefix + "must be finite");
    for (std::size_t first = 0; first < 3; ++first) {
        for (std::size_t second = 0; second < 3; ++second) {
            double overlap = 0.0;
            for (std::size_t axis = 0; axis < 3; ++axis)
                overlap += axes[axis][first] * axes[axis][second];
            const double expected = first == second ? 1.0 : 0.0;
            if (std::abs(overlap - expected) > kSiteAxisTolerance)
                throw PSIEXCEPTION(prefix + "must be orthonormal");
        }
    }
    const double determinant =
        axes[0][0] * (axes[1][1] * axes[2][2] - axes[1][2] * axes[2][1]) -
        axes[0][1] * (axes[1][0] * axes[2][2] - axes[1][2] * axes[2][0]) +
        axes[0][2] * (axes[1][0] * axes[2][1] - axes[1][1] * axes[2][0]);
    if (determinant <= 0.0) throw PSIEXCEPTION(prefix + "must be right-handed");
}

/**
 * Express diag(signs) in the target and source local frames as target^T diag(signs) source.
 * The result must be an exact diagonal sign matrix; anything else means the local axes do
 * not diagonalize the operation and the integer character arithmetic would be invalid.
 */
std::array<int, 3> local_operation_signs(const SiteAxes& target, const SiteAxes& source,
                                         const std::array<int, 3>& signs,
                                         const std::string& context) {
    std::array<int, 3> result{{1, 1, 1}};
    for (std::size_t row = 0; row < 3; ++row) {
        for (std::size_t column = 0; column < 3; ++column) {
            double value = 0.0;
            for (std::size_t axis = 0; axis < 3; ++axis)
                value += target[axis][row] * signs[axis] * source[axis][column];
            const double rounded = std::round(value);
            const bool diagonal = row == column;
            if (!std::isfinite(value) || std::abs(value - rounded) > kSiteAxisTolerance ||
                (diagonal ? std::abs(rounded) != 1.0 : rounded != 0.0))
                throw PSIEXCEPTION("PDef constraints: " + context +
                                   " local axes must diagonalize every site-group operation "
                                   "with unit signs");
            if (diagonal) result[row] = rounded < 0.0 ? -1 : 1;
        }
    }
    return result;
}

/** Group the 15 L3 components by their exact character tuple over the site-group operations. */
void classify_components(const std::vector<std::array<int, 3>>& operation_signs,
                         SiteSymmetry& record) {
    std::vector<std::vector<int>> tuples(kWSMComponents);
    for (std::size_t component = 0; component < kWSMComponents; ++component) {
        tuples[component].reserve(operation_signs.size());
        for (const auto& signs : operation_signs)
            tuples[component].push_back(l3_character(component, signs));
    }
    std::vector<std::size_t> representatives;
    for (std::size_t component = 0; component < kWSMComponents; ++component) {
        std::size_t label = representatives.size();
        for (std::size_t candidate = 0; candidate < representatives.size(); ++candidate) {
            if (tuples[component] == tuples[representatives[candidate]]) {
                label = candidate;
                break;
            }
        }
        if (label == representatives.size()) representatives.push_back(component);
        record.component_class[component] = label;
    }
    record.class_count = representatives.size();
    record.active_pairs.clear();
    for (std::size_t first = 0; first < kWSMComponents; ++first)
        for (std::size_t second = first; second < kWSMComponents; ++second)
            if (record.component_class[first] == record.component_class[second])
                record.active_pairs.push_back({first, second});
}

/** Label the connected components of an undirected site graph in deterministic order. */
std::pair<std::size_t, std::vector<std::size_t>> label_graph_components(const BondGraph& graph) {
    std::vector<std::vector<std::size_t>> neighbors(graph.site_count);
    for (const auto& bond : graph.bonds) {
        neighbors[bond[0]].push_back(bond[1]);
        neighbors[bond[1]].push_back(bond[0]);
    }
    const std::size_t unlabeled = graph.site_count;
    std::vector<std::size_t> labels(graph.site_count, unlabeled);
    std::size_t count = 0;
    for (std::size_t seed = 0; seed < graph.site_count; ++seed) {
        if (labels[seed] != unlabeled) continue;
        std::queue<std::size_t> pending;
        pending.push(seed);
        labels[seed] = count;
        while (!pending.empty()) {
            const std::size_t site = pending.front();
            pending.pop();
            for (const auto neighbor : neighbors[site]) {
                if (labels[neighbor] != unlabeled) continue;
                labels[neighbor] = count;
                pending.push(neighbor);
            }
        }
        ++count;
    }
    return {count, std::move(labels)};
}

}  // namespace

PDefDerivation derive_pdef_constraints(const Molecule& molecule,
                                      const std::vector<SiteAxes>& site_axes) {
    const std::string prefix = "PDef constraints: ";
    const int atom_count = molecule.natom();
    if (atom_count <= 0) throw PSIEXCEPTION(prefix + "at least one site is required");
    const std::size_t site_count = static_cast<std::size_t>(atom_count);
    if (site_count > kMaximumConstraintSites)
        throw PSIEXCEPTION(prefix + "derivation envelope is at most sixteen sites");
    if (!site_axes.empty() && site_axes.size() != site_count)
        throw PSIEXCEPTION(prefix + "supply either no local axes or one local axis frame per site");

    SiteAxes molecular_frame{};
    for (std::size_t axis = 0; axis < 3; ++axis) molecular_frame[axis][axis] = 1.0;
    std::vector<SiteAxes> axes(site_count, molecular_frame);
    if (!site_axes.empty()) axes = site_axes;
    for (std::size_t site = 0; site < site_count; ++site) validate_site_axes(axes[site], site);

    // Symmetry operations act about the molecular symmetry center, so the scan is translation
    // invariant: only the frame's rotational alignment can hide a symmetry element.
    const Vector3 center = molecule.center_of_mass();
    std::vector<SitePosition> sites(site_count);
    std::vector<SiteIdentity> identities(site_count);
    for (std::size_t site = 0; site < site_count; ++site) {
        const int atom = static_cast<int>(site);
        for (std::size_t axis = 0; axis < 3; ++axis) {
            sites[site][axis] = molecule.xyz(atom, static_cast<int>(axis)) - center[axis];
            if (!std::isfinite(sites[site][axis]))
                throw PSIEXCEPTION(prefix + "site coordinates must be finite");
        }
        identities[site] = {molecule.true_atomic_number(atom), molecule.Z(atom),
                            molecule.mass(atom)};
    }

    const auto symmetry = detect_frame_symmetry(sites, identities);
    // libmints locates the true symmetry axes independently of the current frame; counting the
    // operations realized there is the frame-independent reference for the gate below.
    Molecule probe = molecule.clone();
    const auto frame = probe.symmetry_frame(kSiteSymmetryTolerance);
    if (!frame || frame->nirrep() != 1 || frame->nrow() != 3 || frame->ncol() != 3)
        throw PSIEXCEPTION(prefix + "libmints returned an unusable symmetry frame");
    std::vector<SitePosition> canonical(site_count);
    for (std::size_t site = 0; site < site_count; ++site)
        for (std::size_t axis = 0; axis < 3; ++axis) {
            double value = 0.0;
            for (std::size_t source = 0; source < 3; ++source)
                value += sites[site][source] *
                         (*frame)(static_cast<int>(source), static_cast<int>(axis));
            canonical[site][axis] = value;
        }
    const auto reference = detect_frame_symmetry(canonical, identities);
    if (symmetry.operations.size() < reference.operations.size())
        throw PSIEXCEPTION(prefix + "the molecular frame does not realize the detected " +
                           std::string(PointGroup::bits_to_full_name(reference.bits)) +
                           " point group; rotate the geometry into its symmetry frame before "
                           "deriving constraints");

    const auto variable_count = checked_c1_product(site_count, kWSMVariablesPerSite, prefix);

    PDefDerivation derivation;
    derivation.geometry_tolerance = kSiteSymmetryTolerance;
    derivation.molecular_point_group = PointGroup::bits_to_full_name(symmetry.bits);
    derivation.variable_count = variable_count;
    derivation.sites.resize(site_count);

    for (std::size_t site = 0; site < site_count; ++site) {
        auto& record = derivation.sites[site];
        record.symmetry_source = site;
        for (std::size_t operation = 0; operation < symmetry.operations.size(); ++operation) {
            if (symmetry.images[operation][site] != site) {
                record.symmetry_source =
                    std::min(record.symmetry_source, symmetry.images[operation][site]);
                continue;
            }
            record.point_group_bits = static_cast<unsigned char>(
                record.point_group_bits | symmetry.operations[operation].bit);
            record.operation_signs.push_back(local_operation_signs(
                axes[site], axes[site], symmetry.operations[operation].signs,
                "site " + std::to_string(site)));
        }
        record.point_group = PointGroup::bits_to_full_name(record.point_group_bits);
        classify_components(record.operation_signs, record);
    }

    for (std::size_t site = 0; site < site_count; ++site) {
        auto& record = derivation.sites[site];
        if (record.symmetry_source == site) continue;
        const std::size_t source = record.symmetry_source;
        std::size_t carrier = symmetry.operations.size();
        for (std::size_t operation = 0; operation < symmetry.operations.size(); ++operation)
            if (symmetry.images[operation][source] == site) {
                carrier = operation;
                break;
            }
        if (carrier == symmetry.operations.size())
            throw PSIEXCEPTION(prefix + "site orbits are inconsistent with the detected symmetry");
        record.copy_signs = local_operation_signs(
            axes[site], axes[source], symmetry.operations[carrier].signs,
            "site " + std::to_string(site) + " and " + std::to_string(source));
        if (record.active_pairs != derivation.sites[source].active_pairs)
            throw PSIEXCEPTION(prefix + "symmetry-related sites disagree on their active pairs");
    }

    // Every count is fixed before the mask and the equality matrix are allocated.
    std::size_t equality_rows = 0;
    std::size_t independent = 0;
    std::size_t active_variables = 0;
    for (std::size_t site = 0; site < site_count; ++site) {
        const auto& record = derivation.sites[site];
        active_variables += record.active_pairs.size();
        if (record.symmetry_source == site) independent += record.active_pairs.size();
        else equality_rows += derivation.sites[record.symmetry_source].active_pairs.size();
    }
    if (equality_rows > variable_count)
        throw PSIEXCEPTION(prefix + "equality rows exceed the variable envelope");
    derivation.active_variable_count = active_variables;
    derivation.equality_row_count = equality_rows;
    derivation.independent_variable_count = independent;

    derivation.constraints.active_variables.assign(variable_count, false);
    for (std::size_t site = 0; site < site_count; ++site)
        for (const auto& pair : derivation.sites[site].active_pairs)
            derivation.constraints.active_variables[
                site * kWSMVariablesPerSite + wsm_upper_index(pair[0], pair[1])] = true;

    derivation.constraints.equality = std::make_shared<Matrix>(
        "PDef symmetry copies", static_cast<int>(equality_rows),
        static_cast<int>(variable_count));
    derivation.constraints.equality_targets.assign(equality_rows, 0.0);
    std::size_t row = 0;
    for (std::size_t site = 0; site < site_count; ++site) {
        const auto& record = derivation.sites[site];
        if (record.symmetry_source == site) continue;
        const std::size_t source = record.symmetry_source;
        for (const auto& pair : derivation.sites[source].active_pairs) {
            const std::size_t within = wsm_upper_index(pair[0], pair[1]);
            const int character =
                l3_character(pair[0], record.copy_signs) * l3_character(pair[1], record.copy_signs);
            (*derivation.constraints.equality)(
                static_cast<int>(row), static_cast<int>(site * kWSMVariablesPerSite + within)) = 1.0;
            (*derivation.constraints.equality)(
                static_cast<int>(row),
                static_cast<int>(source * kWSMVariablesPerSite + within)) =
                -static_cast<double>(character);
            ++row;
        }
    }
    if (row != equality_rows)
        throw PSIEXCEPTION(prefix + "equality row accounting is inconsistent");
    return derivation;
}

namespace detail {

BondGraphDerivation derive_bond_graph(const std::vector<SitePosition>& sites,
                                     const std::vector<int>& atomic_numbers,
                                     double covalent_scale) {
    const std::string prefix = "Bond graph: ";
    if (sites.empty()) throw PSIEXCEPTION(prefix + "at least one site is required");
    if (atomic_numbers.size() != sites.size())
        throw PSIEXCEPTION(prefix + "atomic numbers do not match sites");
    if (!std::isfinite(covalent_scale) || covalent_scale <= 0.0)
        throw PSIEXCEPTION(prefix + "covalent scale must be finite and positive");

    BondGraphDerivation derivation;
    derivation.covalent_scale = covalent_scale;
    derivation.radius_table = "Slater-1964-bohr-v1";
    derivation.radii.resize(sites.size());
    for (std::size_t site = 0; site < sites.size(); ++site) {
        for (std::size_t axis = 0; axis < 3; ++axis)
            if (!std::isfinite(sites[site][axis]))
                throw PSIEXCEPTION(prefix + "site coordinates must be finite");
        derivation.radii[site] = slater_radius(atomic_numbers[site]);
    }

    // The complete-graph upper triangle bounds every allocation below up front.
    const std::size_t maximum_bonds = sites.size() * (sites.size() - 1) / 2;
    derivation.graph.site_count = sites.size();
    derivation.graph.bonds.reserve(maximum_bonds);
    derivation.bond_distances.reserve(maximum_bonds);
    derivation.bond_thresholds.reserve(maximum_bonds);
    for (std::size_t first = 0; first < sites.size(); ++first) {
        for (std::size_t second = first + 1; second < sites.size(); ++second) {
            double separation_squared = 0.0;
            for (std::size_t axis = 0; axis < 3; ++axis) {
                const double delta = sites[first][axis] - sites[second][axis];
                separation_squared += delta * delta;
            }
            const double separation = std::sqrt(separation_squared);
            if (!std::isfinite(separation) || separation <= kValidationTolerance)
                throw PSIEXCEPTION(prefix + "sites must be distinct");
            const double threshold =
                covalent_scale * (derivation.radii[first] + derivation.radii[second]);
            if (!std::isfinite(threshold))
                throw PSIEXCEPTION(prefix + "bond threshold is nonfinite");
            if (separation > threshold) continue;
            derivation.graph.bonds.push_back({first, second});
            derivation.bond_distances.push_back(separation);
            derivation.bond_thresholds.push_back(threshold);
        }
    }

    auto components = label_graph_components(derivation.graph);
    derivation.component_count = components.first;
    derivation.component_labels = std::move(components.second);
    if (derivation.component_count != 1) {
        std::ostringstream message;
        message << prefix << "the derived covalent graph is disconnected with "
                << derivation.component_count << " components at scale " << covalent_scale
                << "; LW localization requires one connected component";
        throw PSIEXCEPTION(message.str());
    }
    return derivation;
}

}  // namespace detail

BondGraphDerivation derive_bond_graph(const Molecule& molecule, double covalent_scale) {
    const std::string prefix = "Bond graph: ";
    const int atom_count = molecule.natom();
    if (atom_count <= 0) throw PSIEXCEPTION(prefix + "at least one site is required");
    const std::size_t site_count = static_cast<std::size_t>(atom_count);
    std::vector<SitePosition> sites(site_count);
    std::vector<int> atomic_numbers(site_count);
    for (std::size_t site = 0; site < site_count; ++site) {
        const int atom = static_cast<int>(site);
        for (std::size_t axis = 0; axis < 3; ++axis)
            sites[site][axis] = molecule.xyz(atom, static_cast<int>(axis));
        atomic_numbers[site] = molecule.true_atomic_number(atom);
    }
    return detail::derive_bond_graph(sites, atomic_numbers, covalent_scale);
}

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
                    const Complex term = coefficient *
                                         complex_source[complex_index(source_rank, source_order)] *
                                         harmonics[complex_index(difference_rank, difference_order)];
                    require_finite(term, "translate_l3_multipoles product");
                    value += term;
                    require_finite(value, "translate_l3_multipoles accumulation");
                }
            }
            translated[complex_index(target_rank, target_order)] = value;
        }
    }
    return complex_to_real(translated);
}

LocalizedResponse localize_lw(const SitePairResponse& response, const BondGraph& graph,
                              double residual_tolerance) {
    if (!std::isfinite(response.frequency))
        throw PSIEXCEPTION("localize_lw: response frequency must be finite");
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
    const auto components = graph_components(graph_operator);
    const auto pseudoinverse = graph_pseudoinverse(graph_operator, nullptr);

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

    SitePairResponse refined = response;
    std::vector<L3WorkingMatrix> positive_translations;
    std::vector<L3WorkingMatrix> negative_translations;
    positive_translations.reserve(graph.bonds.size());
    negative_translations.reserve(graph.bonds.size());
    for (const auto& bond : graph.bonds) {
        SitePosition displacement{};
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
    LocalizedResponse result{};
    result.frequency = response.frequency;
    result.positions = response.positions;
    for (std::size_t first_component = 0; first_component < 16; ++first_component) {
        for (std::size_t second_component = first_component; second_component < 16; ++second_component) {
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
                for (std::size_t target = 0; target < 16; ++target) {
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

    result.residuals = localization_residuals(response, refined);
    const std::array<double, 5> residual_values{
        result.residuals.off_site, result.residuals.charge_sum,
        result.residuals.reciprocity, result.residuals.molecular_sum,
        result.residuals.local_charge,
    };
    double maximum_residual = 0.0;
    for (double residual : residual_values) {
        require_finite(residual, "localization residual candidate");
        maximum_residual = std::max(maximum_residual, residual);
    }
    if (maximum_residual > residual_tolerance) {
        std::ostringstream message;
        message << "localize_lw: postcondition exceeds residual tolerance (off-site="
                << result.residuals.off_site << ", charge-sum=" << result.residuals.charge_sum
                << ", reciprocity=" << result.residuals.reciprocity
                << ", molecular-sum=" << result.residuals.molecular_sum
                << ", local-charge=" << result.residuals.local_charge << ")";
        throw PSIEXCEPTION(message.str());
    }

    result.refined_pairs = refined.blocks;
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

namespace {

// Standard ten-point Gauss-Legendre nodes and weights on [-1, 1], stored in the
// ascending order of the mapped half-line frequencies: the mapped point at index
// p uses t = -kHalfLineNodes[p], so that
//   frequency(p) = scale * (1 + kHalfLineNodes[p]) / (1 - kHalfLineNodes[p]),
//   weight(p)    = kHalfLineWeights[p] * 2 * scale / (1 - kHalfLineNodes[p])^2.
// Both tables are symmetric, so the reversal is invisible in the weights.
constexpr std::array<double, 10> kHalfLineNodes{
    -0.9739065285171717, -0.8650633666889845, -0.6794095682990244, -0.4333953941292472,
    -0.1488743389816312, 0.1488743389816312,  0.4333953941292472,  0.6794095682990244,
    0.8650633666889845,  0.9739065285171717,
};
constexpr std::array<double, 10> kHalfLineWeights{
    0.06667134430868814, 0.1494513491505806, 0.2190863625159820, 0.2692667193099964,
    0.2955242247147529,  0.2955242247147529, 0.2692667193099964, 0.2190863625159820,
    0.1494513491505806,  0.06667134430868814,
};

}  // namespace

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

    FrequencyGrid grid;
    grid.frequencies.reserve(reviewed_frequencies.size());
    grid.weights.reserve(reviewed_frequencies.size());
    const double scale_ratio = scale / 0.5;
    if (!std::isfinite(scale_ratio)) {
        throw PSIEXCEPTION("make_casimir_grid: scale must be finite and positive at every grid point");
    }
    grid.frequencies.push_back(0.0);
    grid.weights.push_back(0.0);
    for (std::size_t point = 0; point < kHalfLineNodes.size(); ++point) {
        const double frequency = reviewed_frequencies[point + 1] * scale_ratio;
        const double denominator = 1.0 - kHalfLineNodes[point];
        const double weight = kHalfLineWeights[point] * 2.0 * scale / (denominator * denominator);
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

namespace {

constexpr std::size_t kDispersionRankCount = 3;
constexpr std::size_t kDispersionCoefficientCount = 4;
constexpr std::size_t kDispersionMaximumSites = 256;
constexpr std::size_t kDispersionMaximumFrequencies = 64;
constexpr std::size_t kDispersionMaximumWorkTerms = 100000000;
constexpr std::size_t kProtocolGridPoints = 11;
constexpr double kDispersionGridTolerance = 1.0e-10;
constexpr double kDispersionPositionTolerance = 1.0e-10;

/** Real-spherical row/column offset of the rank-l diagonal block of an L3Matrix. */
std::size_t dispersion_rank_offset(unsigned int rank) {
    return static_cast<std::size_t>(rank) * static_cast<std::size_t>(rank) - 1;
}

std::size_t dispersion_rank_dimension(unsigned int rank) {
    return 2 * static_cast<std::size_t>(rank) + 1;
}

void require_dispersion_rank(unsigned int rank, const std::string& prefix) {
    if (rank < 1 || rank > kDispersionRankCount) {
        throw PSIEXCEPTION(prefix + "rank must lie inside the L3 range 1 through 3");
    }
}

double dispersion_deviation(double actual, double expected) {
    return std::abs(actual - expected) / std::max(1.0, std::abs(expected));
}

/**
 * Self-check the ordered recoupling table before it is used. The isotropic
 * 00 00 0 component only needs these closed-form prefactors, so this replaces a
 * real Clebsch-Gordan contraction loader for the published coefficients.
 */
void validate_dispersion_rank_pairs(const std::vector<DispersionRankPair>& pairs) {
    const std::string prefix = "dispersion recoupling table: ";
    if (pairs.empty()) throw PSIEXCEPTION(prefix + "the ordered rank-pair table is empty");
    for (const auto& pair : pairs) {
        require_dispersion_rank(pair.first_rank, prefix);
        require_dispersion_rank(pair.second_rank, prefix);
        if (pair.coefficient_order != 2 * (pair.first_rank + pair.second_rank + 1))
            throw PSIEXCEPTION(prefix + "every ordered pair must satisfy n = 2 (la + lb + 1)");
        if (pair.coefficient_order < 6 || pair.coefficient_order > 12 ||
            pair.coefficient_order % 2 != 0)
            throw PSIEXCEPTION(prefix + "coefficient order must be one of 6, 8, 10, or 12");
        const double expected = binomial(2 * (pair.first_rank + pair.second_rank),
                                        2 * pair.first_rank) / (2.0 * M_PI);
        if (!std::isfinite(pair.prefactor) || pair.prefactor <= 0.0 || pair.prefactor != expected)
            throw PSIEXCEPTION(prefix + "every prefactor must be binom(2 la + 2 lb, 2 la)/(2 pi)");
        const auto identical = std::count_if(
            pairs.begin(), pairs.end(), [&pair](const DispersionRankPair& other) {
                return other.first_rank == pair.first_rank && other.second_rank == pair.second_rank;
            });
        if (identical != 1) throw PSIEXCEPTION(prefix + "ordered pairs must be unique");
        const auto exchanged = std::count_if(
            pairs.begin(), pairs.end(), [&pair](const DispersionRankPair& other) {
                return other.first_rank == pair.second_rank &&
                       other.second_rank == pair.first_rank && other.prefactor == pair.prefactor;
            });
        if (exchanged != 1)
            throw PSIEXCEPTION(prefix + "ordered pairs must be closed under rank exchange");
    }
    for (unsigned int order = 6; order <= 12; order += 2) {
        if (std::none_of(pairs.begin(), pairs.end(), [order](const DispersionRankPair& pair) {
                return pair.coefficient_order == order;
            }))
            throw PSIEXCEPTION(prefix + "every published coefficient needs at least one ordered pair");
    }
}

std::size_t dispersion_coefficient_index(unsigned int coefficient_order) {
    return (static_cast<std::size_t>(coefficient_order) - 6) / 2;
}

/**
 * Validate the caller-supplied quadrature and return its total weight. Both the
 * protocol and convergence paths require an ascending non-negative grid whose
 * static zero frequency carries no quadrature weight.
 */
double validate_dispersion_grid(const FrequencyGrid& grid, const std::string& prefix) {
    double weight_sum = 0.0;
    for (std::size_t point = 0; point < grid.frequencies.size(); ++point) {
        const double frequency = grid.frequencies[point];
        const double weight = grid.weights[point];
        require_finite(frequency, "dispersion grid frequency");
        require_finite(weight, "dispersion grid weight");
        if (frequency < 0.0)
            throw PSIEXCEPTION(prefix + "imaginary-frequency grid points must be non-negative");
        if (point != 0 && frequency <= grid.frequencies[point - 1])
            throw PSIEXCEPTION(prefix + "grid frequencies must be strictly ascending");
        if (weight < 0.0) throw PSIEXCEPTION(prefix + "quadrature weights must be non-negative");
        if (frequency == 0.0 && weight != 0.0)
            throw PSIEXCEPTION(prefix + "the static zero frequency must carry no quadrature weight");
        weight_sum += weight;
        require_finite(weight_sum, "dispersion quadrature weight sum");
    }
    if (weight_sum <= 0.0)
        throw PSIEXCEPTION(prefix + "at least one grid point must carry positive quadrature weight");
    return weight_sum;
}

/**
 * Confirm the grid is the eleven-point protocol grid at one positive scale and
 * report the inferred scale plus the largest relative deviation. The scale is
 * recovered from the first mapped node and then every remaining frequency and
 * all ten weights are checked against the closed-form mapping.
 */
std::pair<double, double> validate_protocol_dispersion_grid(const FrequencyGrid& grid,
                                                            const std::string& prefix) {
    if (grid.frequencies.size() != kProtocolGridPoints)
        throw PSIEXCEPTION(prefix +
                           "protocol requires the static point plus ten mapped imaginary "
                           "frequencies, that is exactly eleven grid points");
    if (grid.frequencies[0] != 0.0 || grid.weights[0] != 0.0)
        throw PSIEXCEPTION(prefix +
                           "the protocol grid must begin with the static zero frequency at zero weight");
    const double leading = 1.0 - kHalfLineNodes[0];
    const double inferred_scale = grid.frequencies[1] * leading / (1.0 + kHalfLineNodes[0]);
    require_finite(inferred_scale, "dispersion protocol grid scale");
    if (inferred_scale <= 0.0)
        throw PSIEXCEPTION(prefix + "the inferred protocol grid scale must be positive");

    double max_deviation = 0.0;
    for (std::size_t point = 0; point < kHalfLineNodes.size(); ++point) {
        const double denominator = 1.0 - kHalfLineNodes[point];
        const double frequency = inferred_scale * (1.0 + kHalfLineNodes[point]) / denominator;
        const double weight =
            kHalfLineWeights[point] * 2.0 * inferred_scale / (denominator * denominator);
        require_finite(frequency, "dispersion protocol grid frequency");
        require_finite(weight, "dispersion protocol grid weight");
        max_deviation = std::max(
            max_deviation, dispersion_deviation(grid.frequencies[point + 1], frequency));
        max_deviation =
            std::max(max_deviation, dispersion_deviation(grid.weights[point + 1], weight));
    }
    if (max_deviation > kDispersionGridTolerance)
        throw PSIEXCEPTION(prefix +
                           "grid must match make_casimir_grid at the inferred protocol scale");
    return {inferred_scale, max_deviation};
}

DispersionMatrices compute_dispersion_impl(const std::vector<RefinedL3Model>& models,
                                           const FrequencyGrid& grid,
                                           bool require_protocol_grid) {
    const std::string prefix = "compute_dispersion: ";
    if (models.empty()) throw PSIEXCEPTION(prefix + "at least one refined L3 model is required");
    if (grid.frequencies.size() != grid.weights.size())
        throw PSIEXCEPTION(prefix + "grid frequencies and weights must have equal counts");
    if (models.size() != grid.frequencies.size())
        throw PSIEXCEPTION(prefix + "expected exactly one model per grid frequency");

    const double weight_sum = validate_dispersion_grid(grid, prefix);
    double inferred_scale = 0.0;
    double max_grid_deviation = 0.0;
    if (require_protocol_grid) {
        const auto protocol = validate_protocol_dispersion_grid(grid, prefix);
        inferred_scale = protocol.first;
        max_grid_deviation = protocol.second;
    }

    const std::size_t site_count = models.front().tensors.size();
    for (std::size_t point = 0; point < models.size(); ++point) {
        const auto& model = models[point];
        if (model.tensors.size() != site_count || model.positions.size() != site_count)
            throw PSIEXCEPTION(prefix +
                               "every model must supply one L3 tensor and one position per site");
        require_finite(model.frequency, "refined model frequency");
        if (dispersion_deviation(model.frequency, grid.frequencies[point]) >
            kDispersionGridTolerance)
            throw PSIEXCEPTION(prefix + "model frequency does not match its grid frequency");
        for (std::size_t site = 0; site < site_count; ++site)
            for (std::size_t axis = 0; axis < kTensorDimension; ++axis)
                if (std::abs(model.positions[site][axis] -
                             models.front().positions[site][axis]) > kDispersionPositionTolerance)
                    throw PSIEXCEPTION(prefix + "site positions must agree across all frequencies");
    }

    const auto plan = detail::plan_dispersion(models.size(), site_count,
                                              Process::environment.get_memory());
    const auto& pairs = detail::dispersion_rank_pairs();

    std::vector<double> isotropic(plan.isotropic_elements, 0.0);
    std::size_t nonpositive_count = 0;
    double minimum_isotropic = std::numeric_limits<double>::infinity();
    double maximum_isotropic = -std::numeric_limits<double>::infinity();
    std::size_t weighted_frequency_count = 0;
    for (std::size_t point = 0; point < models.size(); ++point) {
        if (grid.weights[point] > 0.0) ++weighted_frequency_count;
        for (std::size_t site = 0; site < site_count; ++site) {
            const auto& tensor = models[point].tensors[site];
            for (unsigned int rank = 1; rank <= kDispersionRankCount; ++rank) {
                const auto offset = dispersion_rank_offset(rank);
                const auto dimension = dispersion_rank_dimension(rank);
                bool populated = false;
                for (std::size_t index = 0; index < dimension; ++index)
                    if (tensor[offset + index][offset + index] != 0.0) populated = true;
                if (!populated) {
                    std::ostringstream message;
                    message << prefix << "site " << site << " at frequency index " << point
                            << " supplies an empty rank " << rank
                            << " block; a rank-complete L3 model must supply ranks 1 through 3";
                    throw PSIEXCEPTION(message.str());
                }
                const double value = detail::isotropic_rank_polarizability(tensor, rank);
                if (value <= 0.0) ++nonpositive_count;
                minimum_isotropic = std::min(minimum_isotropic, value);
                maximum_isotropic = std::max(maximum_isotropic, value);
                isotropic[(point * site_count + site) * kDispersionRankCount + (rank - 1)] = value;
            }
        }
    }

    std::vector<double> contributions(plan.contribution_elements, 0.0);
    for (std::size_t term = 0; term < pairs.size(); ++term) {
        const auto& pair = pairs[term];
        for (std::size_t first = 0; first < site_count; ++first) {
            for (std::size_t second = 0; second < site_count; ++second) {
                double integral = 0.0;
                for (std::size_t point = 0; point < models.size(); ++point) {
                    if (grid.weights[point] == 0.0) continue;
                    const double first_value = isotropic[(point * site_count + first) *
                                                             kDispersionRankCount +
                                                         (pair.first_rank - 1)];
                    const double second_value = isotropic[(point * site_count + second) *
                                                              kDispersionRankCount +
                                                          (pair.second_rank - 1)];
                    // The product of the two isotropic factors is parenthesized so
                    // that exchanging the ordered pair reproduces the same value bit
                    // for bit and pair symmetry is exact.
                    integral += grid.weights[point] * (first_value * second_value);
                }
                const double contribution = pair.prefactor * integral;
                require_finite(contribution, "dispersion rank-pair contribution");
                contributions[(term * site_count + first) * site_count + second] = contribution;
            }
        }
    }

    const auto matrix_dimension = static_cast<int>(site_count);
    std::array<SharedMatrix, kDispersionCoefficientCount> coefficients{
        std::make_shared<Matrix>("Isotropic C6", matrix_dimension, matrix_dimension),
        std::make_shared<Matrix>("Isotropic C8", matrix_dimension, matrix_dimension),
        std::make_shared<Matrix>("Isotropic C10", matrix_dimension, matrix_dimension),
        std::make_shared<Matrix>("Isotropic C12", matrix_dimension, matrix_dimension),
    };
    for (std::size_t first = 0; first < site_count; ++first) {
        for (std::size_t second = first; second < site_count; ++second) {
            std::array<double, kDispersionCoefficientCount> totals{};
            for (std::size_t term = 0; term < pairs.size(); ++term)
                totals[dispersion_coefficient_index(pairs[term].coefficient_order)] +=
                    contributions[(term * site_count + first) * site_count + second];
            for (std::size_t coefficient = 0; coefficient < totals.size(); ++coefficient) {
                require_finite(totals[coefficient], "isotropic dispersion coefficient");
                // Assigning both triangles from one sum keeps C_n[A][B] exactly
                // equal to C_n[B][A] even though single ordered terms are not symmetric.
                (*coefficients[coefficient])(first, second) = totals[coefficient];
                (*coefficients[coefficient])(second, first) = totals[coefficient];
            }
        }
    }

    DispersionMatrices result;
    result.c6 = coefficients[0];
    result.c8 = coefficients[1];
    result.c10 = coefficients[2];
    result.c12 = coefficients[3];
    result.diagnostics.frequency_count = models.size();
    result.diagnostics.weighted_frequency_count = weighted_frequency_count;
    result.diagnostics.site_count = site_count;
    result.diagnostics.quadrature_weight_sum = weight_sum;
    result.diagnostics.min_isotropic_polarizability = minimum_isotropic;
    result.diagnostics.max_isotropic_polarizability = maximum_isotropic;
    // Non-positive isotropic blocks are reported, never repaired or dropped: a
    // reviewed L3 model may contain non-positive-definite higher-rank blocks.
    result.diagnostics.nonpositive_isotropic_count = nonpositive_count;
    result.diagnostics.inferred_scale = inferred_scale;
    result.diagnostics.max_protocol_grid_deviation = max_grid_deviation;
    result.diagnostics.protocol_grid_enforced = require_protocol_grid;
    result.diagnostics.rank_pair_terms = pairs;
    result.diagnostics.rank_pair_contributions = std::move(contributions);
    result.diagnostics.plan = plan;
    return result;
}

}  // namespace

namespace detail {

DispersionPlan plan_dispersion(std::size_t frequency_count, std::size_t site_count,
                               std::size_t memory_bytes) {
    const std::string prefix = "dispersion plan: ";
    if (frequency_count == 0 || frequency_count > kDispersionMaximumFrequencies)
        throw PSIEXCEPTION(prefix + "frequency envelope is 1 through 64");
    if (site_count == 0 || site_count > kDispersionMaximumSites)
        throw PSIEXCEPTION(prefix + "site envelope is 1 through 256");
    const auto& pairs = dispersion_rank_pairs();
    const auto bytes = [&prefix](std::size_t elements) {
        return checked_c1_product(elements, sizeof(double), prefix);
    };
    const auto site_pairs = checked_c1_product(site_count, site_count, prefix);
    const auto isotropic_elements = checked_c1_product(
        checked_c1_product(frequency_count, site_count, prefix), kDispersionRankCount, prefix);
    const auto coefficient_elements =
        checked_c1_product(site_pairs, kDispersionCoefficientCount, prefix);
    const auto contribution_elements = checked_c1_product(site_pairs, pairs.size(), prefix);
    const auto rank_pair_table_bytes =
        checked_c1_product(pairs.size(), sizeof(DispersionRankPair), prefix);
    const auto work_terms = checked_c1_product(contribution_elements, frequency_count, prefix);
    if (work_terms > kDispersionMaximumWorkTerms)
        throw PSIEXCEPTION(prefix + "isotropic recoupling work exceeds the supported envelope");
    // One conservative megabyte covers the matrix objects, plan strings, and the
    // Python export of the diagnostics record.
    constexpr std::size_t conservative_overhead_bytes = 1024ULL * 1024ULL;
    const auto estimated_bytes = checked_c1_sum(
        checked_c1_sum(bytes(isotropic_elements), bytes(coefficient_elements), prefix),
        checked_c1_sum(bytes(contribution_elements),
                       checked_c1_sum(rank_pair_table_bytes, conservative_overhead_bytes, prefix),
                       prefix),
        prefix);
    const auto reserved = memory_bytes / 2;
    if (estimated_bytes > reserved)
        throw PSIEXCEPTION(prefix +
                           "isotropic/coefficient/rank-pair storage exceeds half the reserved memory");

    DispersionPlan plan;
    plan.frequency_count = frequency_count;
    plan.site_count = site_count;
    plan.max_frequency_count = kDispersionMaximumFrequencies;
    plan.max_site_count = kDispersionMaximumSites;
    plan.coefficient_count = kDispersionCoefficientCount;
    plan.rank_pair_count = pairs.size();
    plan.isotropic_elements = isotropic_elements;
    plan.isotropic_bytes = bytes(isotropic_elements);
    plan.coefficient_elements = coefficient_elements;
    plan.coefficient_bytes = bytes(coefficient_elements);
    plan.contribution_elements = contribution_elements;
    plan.contribution_bytes = bytes(contribution_elements);
    plan.rank_pair_table_bytes = rank_pair_table_bytes;
    plan.metadata_bytes = conservative_overhead_bytes;
    plan.estimated_bytes = estimated_bytes;
    plan.configured_memory_bytes = memory_bytes;
    plan.reserved_memory_bytes = reserved;
    plan.work_terms = work_terms;
    plan.max_work_terms = kDispersionMaximumWorkTerms;
    plan.algorithm = "isotropic-rank-trace/ordered-rank-pair/half-line-quadrature-sum";
    plan.memory_semantics =
        "hard half-memory gate over the frequency-major isotropic ranks, four site-pair "
        "coefficient matrices, ordered rank-pair contributions, and the validated rank-pair table";
    return plan;
}

double isotropic_rank_polarizability(const L3Matrix& tensor, unsigned int rank) {
    const std::string prefix = "isotropic rank polarizability: ";
    require_dispersion_rank(rank, prefix);
    const auto offset = dispersion_rank_offset(rank);
    const auto dimension = dispersion_rank_dimension(rank);
    double trace = 0.0;
    for (std::size_t index = 0; index < dimension; ++index) {
        const double value = tensor[offset + index][offset + index];
        require_finite(value, "isotropic rank polarizability");
        trace += value;
    }
    // Only the trace of the diagonal rank block enters the isotropic 00 00 0
    // component; rank-mixing blocks and all anisotropic components drop out.
    const double mean = trace / static_cast<double>(dimension);
    require_finite(mean, "isotropic rank polarizability");
    return mean;
}

double dispersion_rank_prefactor(unsigned int first_rank, unsigned int second_rank) {
    const std::string prefix = "dispersion rank prefactor: ";
    require_dispersion_rank(first_rank, prefix);
    require_dispersion_rank(second_rank, prefix);
    const double prefactor =
        binomial(2 * (first_rank + second_rank), 2 * first_rank) / (2.0 * M_PI);
    require_finite(prefactor, "dispersion rank prefactor");
    return prefactor;
}

const std::vector<DispersionRankPair>& dispersion_rank_pairs() {
    static const std::vector<DispersionRankPair> table = [] {
        // Ordered rank pairs with n = 2 (la + lb + 1) <= 12 that an L3 model can
        // supply. (1,4) and (4,1) also give n = 12 but rank 4 is absent, which is
        // why C12 is reviewed-model parity rather than rank-complete dispersion.
        static constexpr std::array<std::array<unsigned int, 2>, 8> ordered{{
            {{1, 1}}, {{1, 2}}, {{2, 1}}, {{1, 3}}, {{3, 1}}, {{2, 2}}, {{2, 3}}, {{3, 2}},
        }};
        std::vector<DispersionRankPair> pairs;
        pairs.reserve(ordered.size());
        for (const auto& entry : ordered) {
            DispersionRankPair pair;
            pair.first_rank = entry[0];
            pair.second_rank = entry[1];
            pair.coefficient_order = 2 * (entry[0] + entry[1] + 1);
            pair.prefactor = dispersion_rank_prefactor(entry[0], entry[1]);
            pairs.push_back(pair);
        }
        validate_dispersion_rank_pairs(pairs);
        return pairs;
    }();
    return table;
}

DispersionMatrices compute_dispersion_test_only(const std::vector<RefinedL3Model>& models,
                                                const FrequencyGrid& frequencies) {
    return compute_dispersion_impl(models, frequencies, false);
}

}  // namespace detail

DispersionMatrices compute_dispersion(const std::vector<RefinedL3Model>& models,
                                      const FrequencyGrid& frequencies) {
    return compute_dispersion_impl(models, frequencies, true);
}

ISAOptions isa_options_from(Options& options) {
    const std::string prefix = "ISA options: ";
    const int radial = options.get_int("ATOMIC_POLARIZABILITY_ISA_RADIAL_POINTS");
    const int polar = options.get_int("ATOMIC_POLARIZABILITY_ISA_ANGULAR_POLAR_POINTS");
    const int azimuthal = options.get_int("ATOMIC_POLARIZABILITY_ISA_ANGULAR_AZIMUTHAL_POINTS");
    const int iterations = options.get_int("ATOMIC_POLARIZABILITY_ISA_MAX_ITERATIONS");
    if (radial <= 0 || polar <= 0 || azimuthal <= 0 || iterations <= 0)
        throw ATOMIC_POLARIZABILITY_PREREQUISITE(prefix + "ISA grid and iteration keywords must be positive");
    const double convergence = options.get_double("ATOMIC_POLARIZABILITY_ISA_CONVERGENCE");
    if (!(convergence > 0.0) || !std::isfinite(convergence))
        throw ATOMIC_POLARIZABILITY_PREREQUISITE(prefix + "ISA convergence must be finite and positive");
    ISAOptions defaults;
    return ISAOptions(static_cast<std::size_t>(radial), static_cast<std::size_t>(polar),
                      static_cast<std::size_t>(azimuthal), static_cast<std::size_t>(iterations),
                      convergence, defaults.mix_fraction(), defaults.initial_alpha(),
                      defaults.tail_join_factor(), defaults.tail_activation_iteration(),
                      defaults.tail_activation_convergence(), defaults.electron_count_tolerance());
}

ResponseKernel reviewed_response_kernel() { return ResponseKernel(0.25, 0.75); }

namespace {

constexpr double kReferenceGeometryTolerance = 1.0e-10;

/**
 * The WSM design matrix is built from molecular-frame harmonics, so every site's
 * local-to-global frame is the identity and the packing rotation is the identity too.
 * Keeping the rotation explicit means rotate_tensor still enforces orthonormality and
 * det(R) = 1 on the frame actually used.
 */
Matrix identity_site_frame() {
    Matrix rotation("atomic polarizability local-to-global frame", 3, 3);
    for (int axis = 0; axis < 3; ++axis) rotation.set(axis, axis, 1.0);
    return rotation;
}

/** Fail closed unless the auxiliary SCF describes exactly the reference's structure. */
void require_matching_structure(const Wavefunction& reference, const Wavefunction& other,
                                const std::string& role) {
    const std::string prefix = "atomic polarizability: the " + role + " wavefunction ";
    const auto reference_molecule = reference.molecule();
    const auto other_molecule = other.molecule();
    if (!reference_molecule || !other_molecule)
        throw ATOMIC_POLARIZABILITY_PREREQUISITE(prefix + "has no molecule");
    if (reference_molecule->natom() != other_molecule->natom())
        throw ATOMIC_POLARIZABILITY_PREREQUISITE(prefix + "has a different atom count");
    for (int atom = 0; atom < reference_molecule->natom(); ++atom) {
        if (reference_molecule->true_atomic_number(atom) != other_molecule->true_atomic_number(atom))
            throw ATOMIC_POLARIZABILITY_PREREQUISITE(prefix + "has a different nuclear framework");
        for (int axis = 0; axis < 3; ++axis) {
            const double displacement =
                reference_molecule->xyz(atom, axis) - other_molecule->xyz(atom, axis);
            if (!std::isfinite(displacement) ||
                std::abs(displacement) > kReferenceGeometryTolerance)
                throw ATOMIC_POLARIZABILITY_PREREQUISITE(
                    prefix + "is not at the reference geometry; the vertical protocol requires "
                             "all three SCFs at one geometry");
        }
    }
    const auto reference_basis = reference.basisset();
    const auto other_basis = other.basisset();
    if (!reference_basis || !other_basis)
        throw ATOMIC_POLARIZABILITY_PREREQUISITE(prefix + "has no orbital basis");
    if (reference_basis->name() != other_basis->name() ||
        reference_basis->nbf() != other_basis->nbf())
        throw ATOMIC_POLARIZABILITY_PREREQUISITE(
            prefix + "does not use the reference orbital basis; the vertical protocol requires "
                     "one complete basis for all three SCFs");
}

}  // namespace

AtomicPolarizabilityCalculator::AtomicPolarizabilityCalculator(
    std::shared_ptr<Wavefunction> grac_wfn, std::shared_ptr<Wavefunction> neutral_precursor_wfn,
    std::shared_ptr<Wavefunction> cation_wfn)
    : wfn_(std::move(grac_wfn)),
      neutral_precursor_wfn_(std::move(neutral_precursor_wfn)),
      cation_wfn_(std::move(cation_wfn)) {
    if (!wfn_)
        throw ATOMIC_POLARIZABILITY_PREREQUISITE(
            "the GRAC-corrected reference wavefunction is missing");
    if (!neutral_precursor_wfn_)
        throw ATOMIC_POLARIZABILITY_PREREQUISITE(
            "the neutral precursor wavefunction is missing; it fixes the applied GRAC shift");
    if (!cation_wfn_)
        throw ATOMIC_POLARIZABILITY_PREREQUISITE(
            "the cation wavefunction is missing; it fixes the vertical ionization potential");
}

AtomicPolarizabilityCalculator::AtomicPolarizabilityCalculator(std::shared_ptr<Wavefunction> wfn)
    : wfn_(std::move(wfn)) {
    if (!wfn_)
        throw ATOMIC_POLARIZABILITY_PREREQUISITE(
            "the GRAC-corrected reference wavefunction is missing");
}

void AtomicPolarizabilityCalculator::validate_wavefunction_prerequisites() const {
    bool has_orbital_response_data = false;
    try {
        has_orbital_response_data =
            wfn_->molecule() && wfn_->basisset() && wfn_->Ca() && wfn_->Da() && wfn_->epsilon_a();
    } catch (const PsiException&) {
        // Some Wavefunction accessors reject incomplete, safely constructed wavefunctions.
    }

    if (!has_orbital_response_data)
        throw ATOMIC_POLARIZABILITY_PREREQUISITE(
            "the reference wavefunction is missing required orbital response data");

    // The bare single-wavefunction path lands here. It must never publish partial output.
    if (!neutral_precursor_wfn_ || !cation_wfn_)
        throw ATOMIC_POLARIZABILITY_PREREQUISITE(
            "the native pipeline needs the GRAC-corrected reference together with its neutral "
            "precursor and cation wavefunctions, which a bare OEProp call cannot supply. Run "
            "psi4.driver.procrouting.atomic_polarizability.atomic_polarizabilities instead, or "
            "hand the triple to OEProp with set_atomic_polarizability_references");

    require_matching_structure(*wfn_, *neutral_precursor_wfn_, "neutral precursor");
    require_matching_structure(*wfn_, *cation_wfn_, "cation");
}

AtomicPolarizabilityPublication AtomicPolarizabilityCalculator::run() const {
    const std::string prefix = "atomic polarizability: ";
    // Nothing downstream allocates output until this gate passes.
    validate_wavefunction_prerequisites();
    Options& options = Process::environment.options;

    const int nonzero = options.get_int("ATOMIC_POLARIZABILITY_N_FREQUENCIES");
    if (nonzero <= 0)
        throw ATOMIC_POLARIZABILITY_PREREQUISITE(prefix + "the nonzero frequency count must be positive");
    const double scale = options.get_double("ATOMIC_POLARIZABILITY_FREQUENCY_SCALE");
    AtomicPolarizabilityPublication result;
    result.grid = make_casimir_grid(static_cast<unsigned int>(nonzero), scale);
    const std::size_t frequency_count = result.grid.frequencies.size();

    // Stage 1: the frozen GRAC response context, which revalidates the SCF triple itself.
    auto context = FrozenResponseContext::create(wfn_, neutral_precursor_wfn_, cation_wfn_);
    if (!context) throw ATOMIC_POLARIZABILITY_PREREQUISITE(prefix + "the frozen response context is null");
    const auto molecule = context->molecule();
    if (!molecule) throw ATOMIC_POLARIZABILITY_PREREQUISITE(prefix + "the frozen context has no molecule");
    const std::size_t site_count = context->sites().size();
    if (site_count == 0 || site_count != static_cast<std::size_t>(molecule->natom()))
        throw ATOMIC_POLARIZABILITY_PREREQUISITE(prefix + "frozen sites and molecular atoms disagree");
    // refine_wsm pairs fit points, which are generated from the molecule, against site
    // positions, which are inherited from the frozen context. The two must therefore be the
    // same points in the same order. They agree today, but a divergence would corrupt the
    // fitted anisotropy without tripping any dimensional check, so it is enforced here.
    for (std::size_t site = 0; site < site_count; ++site)
        for (int axis = 0; axis < 3; ++axis)
            if (std::abs(context->sites()[site][static_cast<std::size_t>(axis)] -
                         molecule->xyz(static_cast<int>(site), axis)) >
                kReferenceGeometryTolerance)
                throw ATOMIC_POLARIZABILITY_PREREQUISITE(
                    prefix + "frozen site positions and molecular coordinates disagree");

    // Stage 2: the ISA partition of the frozen density.
    const auto kernel = reviewed_response_kernel();
    auto isa_weights = compute_isa_weights(context, isa_options_from(options));
    result.isa = isa_weights.diagnostics();
    if (!result.isa.converged)
        throw ATOMIC_POLARIZABILITY_PREREQUISITE(prefix + "the ISA partition did not converge");
    if (isa_weights.site_count() != site_count)
        throw ATOMIC_POLARIZABILITY_PREREQUISITE(prefix + "ISA sites and frozen sites disagree");

    // Stage 3: the frequency-dependent site-pair response.
    const ISAPolResponseProvider provider(context, kernel, std::move(isa_weights));
    const auto responses = provider.compute_isapol_response(result.grid);
    if (responses.size() != frequency_count)
        throw ATOMIC_POLARIZABILITY_PREREQUISITE(prefix + "the response provider returned the wrong frequency count");

    // Stage 4: the covalent bond graph, which fails closed when disconnected.
    result.bond_graph = derive_bond_graph(*molecule,
                                          options.get_double("ATOMIC_POLARIZABILITY_COVALENT_SCALE"));
    if (result.bond_graph.component_count != 1 || result.bond_graph.graph.site_count != site_count)
        throw ATOMIC_POLARIZABILITY_PREREQUISITE(prefix + "the derived bond graph is not a single connected component over the sites");

    // Stage 5: LW localization at every frequency.
    const double localization_tolerance =
        options.get_double("ATOMIC_POLARIZABILITY_LOCALIZATION_TOLERANCE");
    if (!(localization_tolerance > 0.0) || !std::isfinite(localization_tolerance))
        throw ATOMIC_POLARIZABILITY_PREREQUISITE(prefix + "the localization tolerance must be finite and positive");
    std::vector<LocalizedResponse> localized;
    localized.reserve(frequency_count);
    result.localization_residuals.reserve(frequency_count);
    for (std::size_t frequency = 0; frequency < frequency_count; ++frequency) {
        if (responses[frequency].frequency != result.grid.frequencies[frequency])
            throw ATOMIC_POLARIZABILITY_PREREQUISITE(prefix + "the response frequencies do not match the protocol grid");
        auto one = localize_lw(responses[frequency], result.bond_graph.graph, localization_tolerance);
        if (one.local.size() != site_count || one.positions.size() != site_count)
            throw ATOMIC_POLARIZABILITY_PREREQUISITE(prefix + "localization returned the wrong site count");
        for (std::size_t site = 0; site < site_count; ++site)
            for (std::size_t axis = 0; axis < 3; ++axis)
                if (std::abs(one.positions[site][axis] - context->sites()[site][axis]) >
                    kReferenceGeometryTolerance)
                    throw ATOMIC_POLARIZABILITY_PREREQUISITE(
                        prefix + "localized site positions left the frozen site frame");
        result.localization_residuals.push_back(one.residuals);
        localized.push_back(std::move(one));
    }

    // Stage 6: the symmetry-faithful fit points and the external-point response on them.
    const auto fit_points = generate_wsm_fit_points(*molecule, options);
    result.fit_points = fit_points.plan;
    if (fit_points.points.empty())
        throw ATOMIC_POLARIZABILITY_PREREQUISITE(prefix + "no WSM fit points were generated");
    const auto point_response =
        evaluate_point_response(context, kernel, result.grid.frequencies, fit_points.points);
    if (point_response.frequency_count() != frequency_count ||
        point_response.points().size() != fit_points.points.size())
        throw ATOMIC_POLARIZABILITY_PREREQUISITE(prefix + "the point response does not span the requested grid and points");

    // Stage 7: the PDef active-variable mask.
    //
    // FRAME CONTRACT: derive_pdef_constraints must be called with EMPTY site axes. The
    // returned mask indexes variables in whichever frame site_axes selects, and refine_wsm's
    // design matrix uses molecular-frame harmonics, so any non-identity local frame here
    // yields a mask indexed in the wrong frame and plausible-looking but wrong anisotropy.
    const std::vector<SiteAxes> molecular_frame_axes{};
    if (!molecular_frame_axes.empty())
        throw ATOMIC_POLARIZABILITY_PREREQUISITE(prefix + "the PDef mask must be derived in the molecular frame");
    result.pdef = derive_pdef_constraints(*molecule, molecular_frame_axes);
    if (result.pdef.active_variable_count == 0 || result.pdef.independent_variable_count == 0)
        throw ATOMIC_POLARIZABILITY_PREREQUISITE(prefix + "the PDef derivation left no fit variables");

    // Stage 8: the constrained L3 WSM refinement at every frequency.
    RefinementOptions refinement;
    refinement.maximum_condition_number =
        options.get_double("ATOMIC_POLARIZABILITY_MAX_CONDITION_NUMBER");
    const auto models = refine_wsm(localized, point_response, result.pdef.constraints, refinement);
    if (models.size() != frequency_count)
        throw ATOMIC_POLARIZABILITY_PREREQUISITE(prefix + "refinement returned the wrong frequency count");
    result.refinement.reserve(frequency_count);
    for (const auto& model : models) {
        if (model.tensors.size() != site_count)
            throw ATOMIC_POLARIZABILITY_PREREQUISITE(prefix + "a refined model has the wrong site count");
        result.refinement.push_back(model.diagnostics);
    }

    // Stage 9: isotropic dispersion recoupling on the same protocol grid.
    result.dispersion = compute_dispersion(models, result.grid);
    if (!result.dispersion.c6 || !result.dispersion.c8 || !result.dispersion.c10 ||
        !result.dispersion.c12)
        throw ATOMIC_POLARIZABILITY_PREREQUISITE(prefix + "dispersion recoupling produced no coefficients");

    // Stage 10: pack the dipole blocks into the global Cartesian frame.
    const auto frame = identity_site_frame();
    const int rows = static_cast<int>(frequency_count * site_count);
    auto dynamic = std::make_shared<Matrix>("ATOMIC DYNAMIC POLARIZABILITIES", rows, 6);
    auto statics = std::make_shared<Matrix>("ATOMIC POLARIZABILITIES",
                                            static_cast<int>(site_count), 6);
    auto frequencies = std::make_shared<Matrix>("ATOMIC POLARIZABILITY FREQUENCIES",
                                                static_cast<int>(frequency_count), 1);
    for (std::size_t frequency = 0; frequency < frequency_count; ++frequency) {
        if (models[frequency].frequency != result.grid.frequencies[frequency])
            throw ATOMIC_POLARIZABILITY_PREREQUISITE(prefix + "a refined model frequency left the protocol grid");
        frequencies->set(static_cast<int>(frequency), 0, result.grid.frequencies[frequency]);
        for (std::size_t site = 0; site < site_count; ++site) {
            const auto local = local_spherical_dipole_to_cartesian(models[frequency].tensors[site]);
            const auto packed = pack_symmetric_tensor(rotate_tensor(local, frame));
            const int row = static_cast<int>(frequency * site_count + site);
            for (int component = 0; component < 6; ++component) {
                if (!std::isfinite(packed[static_cast<std::size_t>(component)]))
                    throw ATOMIC_POLARIZABILITY_PREREQUISITE(prefix + "a packed global tensor is not finite");
                dynamic->set(row, component, packed[static_cast<std::size_t>(component)]);
                if (frequency == 0)
                    statics->set(static_cast<int>(site), component,
                                 packed[static_cast<std::size_t>(component)]);
            }
        }
    }
    // The static tensor is the zero-frequency block, exactly.
    for (std::size_t site = 0; site < site_count; ++site)
        for (int component = 0; component < 6; ++component)
            if (statics->get(static_cast<int>(site), component) !=
                dynamic->get(static_cast<int>(site), component))
                throw ATOMIC_POLARIZABILITY_PREREQUISITE(prefix + "the static tensor is not the zero-frequency block");

    result.static_polarizabilities = statics;
    result.dynamic_polarizabilities = dynamic;
    result.frequencies = frequencies;
    return result;
}

void AtomicPolarizabilityCalculator::compute() {
    // Every stage gate lives in run(). Publication below is unconditional precisely because
    // run() either returns a complete result or throws, so partial output cannot escape.
    const auto result = run();

    const std::vector<std::pair<std::string, SharedMatrix>> published{
        {"ATOMIC POLARIZABILITIES", result.static_polarizabilities},
        {"ATOMIC DYNAMIC POLARIZABILITIES", result.dynamic_polarizabilities},
        {"ATOMIC POLARIZABILITY FREQUENCIES", result.frequencies},
        {"ATOMIC C6", result.dispersion.c6},
        {"ATOMIC C8", result.dispersion.c8},
        {"ATOMIC C10", result.dispersion.c10},
        {"ATOMIC C12", result.dispersion.c12}};
    for (const auto& entry : published)
        if (!entry.second)
            throw ATOMIC_POLARIZABILITY_PREREQUISITE("atomic polarizability: '" + entry.first +
                                                     "' was not produced");
    for (const auto& entry : published) {
        Process::environment.arrays[entry.first] = entry.second;
        wfn_->set_array_variable(entry.first, entry.second);
    }
}

}  // namespace psi
