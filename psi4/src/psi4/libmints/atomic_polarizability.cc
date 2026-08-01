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
#include "psi4/libmints/molecule.h"
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
    for (int column = 0; column < rhs_count; ++column) {
        for (int row = 0; row < transition_count; ++row) {
            const double p = solution[row + static_cast<std::size_t>(column) * dimension];
            const double q = omega == 0.0 ? 0.0 :
                solution[row + transition_count + static_cast<std::size_t>(column) * dimension];
            if (!std::isfinite(p) || !std::isfinite(q))
                throw PSIEXCEPTION("dense restricted response: solution amplitudes are not finite");
            (*P)(row, column) = p;
            (*Q)(row, column) = q;
        }
    }
    return {std::move(P), std::move(Q), reciprocal_condition, work[0],
            *std::max_element(forward_error.begin(), forward_error.end()),
            *std::max_element(backward_error.begin(), backward_error.end()),
            *std::max_element(scaled_residual.begin(), scaled_residual.end())};
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
    const auto scratch_elements =
        checked_c1_sum(projected_response_elements, response_map_elements, prefix);
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
ResponseMapSymmetryDiagnostics analyze_response_map(
    const DenseRestrictedResponse& response) {
    const std::string prefix = "site-pair response contraction: ";
    if (!response.P || !response.Q)
        throw PSIEXCEPTION(prefix + "dense response carrier is incomplete");
    const auto& response_map = *response.P;
    const auto& conjugate_map = *response.Q;
    if (response_map.nirrep() != 1 || response_map.nrow() <= 0 ||
        response_map.nrow() != response_map.ncol() || conjugate_map.nirrep() != 1 ||
        conjugate_map.nrow() != response_map.nrow() ||
        conjugate_map.ncol() != response_map.ncol())
        throw PSIEXCEPTION(prefix + "dense response carrier dimensions are inconsistent");
    validate_dense_response_diagnostics(
        response.reciprocal_condition, response.reciprocal_pivot_growth,
        {response.max_forward_error}, {response.max_backward_error},
        {response.max_scaled_residual});

    // DGESVX FERR applies to each full doubled solution column [P;Q]. Its
    // infinity-norm envelope therefore includes both maps. A separate unit-floor
    // machine scale prevents exact or tiny maps from eliminating arithmetic noise.
    double solution_scale = 0.0;
    for (int row = 0; row < response_map.nrow(); ++row) {
        for (int column = 0; column < response_map.ncol(); ++column) {
            const double p = response_map(row, column);
            const double q = conjugate_map(row, column);
            if (!std::isfinite(p) || !std::isfinite(q))
                throw PSIEXCEPTION(prefix + "dense response maps must contain only finite values");
            solution_scale = std::max({solution_scale, std::abs(p), std::abs(q)});
        }
    }
    const double dimension = static_cast<double>(response_map.nrow());
    const double machine_scale = std::max(1.0, solution_scale);
    const double machine_antisymmetry =
        64.0 * std::numeric_limits<double>::epsilon() * machine_scale * dimension;
    // Two independently FERR-bounded P entries form each antisymmetric difference.
    const double solver_antisymmetry =
        2.0 * response.max_forward_error * solution_scale;
    const double allowed_antisymmetry = std::max(machine_antisymmetry, solver_antisymmetry);
    if (!std::isfinite(allowed_antisymmetry))
        throw PSIEXCEPTION(prefix + "derived response-map symmetry bound overflowed");
    double symmetry_residual = 0.0;
    for (int row = 0; row < response_map.nrow(); ++row) {
        for (int column = row + 1; column < response_map.ncol(); ++column) {
            const double residual =
                std::abs(response_map(row, column) - response_map(column, row));
            symmetry_residual = std::max(symmetry_residual, residual);
            if (!std::isfinite(residual) || residual > allowed_antisymmetry)
                throw PSIEXCEPTION(prefix +
                                   "response map must be symmetric within its derived solver-error bound");
        }
    }
    return {solution_scale, allowed_antisymmetry, symmetry_residual};
}
}  // namespace

ResponseMapSymmetryDiagnostics validate_response_map_symmetry_test_only(
    const Matrix& response_map, const Matrix& conjugate_map,
    double response_map_forward_error_bound) {
    DenseRestrictedResponse synthetic{response_map.clone(), conjugate_map.clone(), 1.0, 1.0,
                                      response_map_forward_error_bound, 0.0, 0.0};
    return analyze_response_map(synthetic);
}

SitePairResponseContraction contract_site_pair_response(
    std::size_t site_count, const Matrix& projection,
    const DenseRestrictedResponse& response) {
    const std::string prefix = "site-pair response contraction: ";
    constexpr double restricted_factor = 4.0;
    if (site_count == 0) throw PSIEXCEPTION(prefix + "site count must be nonzero");
    if (projection.nirrep() != 1 || projection.nrow() <= 0 || projection.ncol() <= 0)
        throw PSIEXCEPTION(prefix + "input dimensions are inconsistent");
    const auto symmetry = analyze_response_map(response);
    const auto& response_map = *response.P;
    const auto transition_count = static_cast<std::size_t>(projection.ncol());
    const auto plan = plan_site_pair_response_contraction(
        site_count, transition_count, Process::environment.get_memory());
    if (static_cast<std::size_t>(projection.nrow()) != plan.component_count)
        throw PSIEXCEPTION(prefix + "projection dimensions do not match the site count");
    if (response_map.nrow() != projection.ncol())
        throw PSIEXCEPTION(prefix + "response-map and projection dimensions are inconsistent");
    for (int row = 0; row < projection.nrow(); ++row)
        for (int transition = 0; transition < projection.ncol(); ++transition)
            if (!std::isfinite(projection(row, transition)))
                throw PSIEXCEPTION(prefix + "projection must contain only finite values");

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
    result.response_map_forward_error_bound = response.max_forward_error;
    result.response_map_solution_scale = symmetry.solution_scale;
    result.response_map_allowed_antisymmetry = symmetry.allowed_antisymmetry;
    result.response_map_symmetry_residual = symmetry.symmetry_residual;
    result.reciprocity_enforced = true;
    return result;
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
    (void)expected_response_count(frequencies);
    throw PSIEXCEPTION("ISAPolResponseProvider: native point-response execution is not implemented; no response was published");
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
    throw PSIEXCEPTION(
        "AtomicPolarizabilityCalculator: required native response data are unavailable: missing GRAC "
        "provenance and ISA weights");
}

}  // namespace psi
