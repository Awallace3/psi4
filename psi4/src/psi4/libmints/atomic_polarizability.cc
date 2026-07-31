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
#include "psi4/libmints/basisset.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libmints/vector.h"
#include "psi4/libmints/wavefunction.h"
#include "psi4/libscf_solver/hf.h"
#include "psi4/libscf_solver/rhf.h"
#include "psi4/libscf_solver/uhf.h"
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

ResponseKernel::ResponseKernel(double chf_exchange, double alda_kernel)
    : chf_exchange_(chf_exchange), alda_kernel_(alda_kernel) {
    if (!std::isfinite(chf_exchange_) || chf_exchange_ != 0.25)
        throw PSIEXCEPTION("ResponseKernel: CHF exchange coefficient must be exactly 0.25");
    if (!std::isfinite(alda_kernel_) || alda_kernel_ != 0.75)
        throw PSIEXCEPTION("ResponseKernel: ALDA coefficient must be exactly 0.75");
}

FrozenResponseContext::FrozenResponseContext(
    SharedMatrix Ca, SharedMatrix Cb, SharedVector epsilon_a, SharedVector epsilon_b,
    SharedVector occupation_a, SharedVector occupation_b, SharedMatrix Da, SharedMatrix Db,
    double energy, std::shared_ptr<const Molecule> molecule, std::shared_ptr<const BasisSet> basis,
    std::shared_ptr<const BasisSetStructuralSnapshot> basis_snapshot,
    std::shared_ptr<const SuperFunctional> functional, std::vector<SitePosition> sites,
    std::vector<double> grid_points, std::vector<double> grid_weights,
    std::vector<FrozenGridBlock> grid_blocks, GRACProvenance grac, std::string functional_name,
    std::string grac_x_name, std::string grac_c_name)
    : Ca_(std::move(Ca)), Cb_(std::move(Cb)), epsilon_a_(std::move(epsilon_a)),
      epsilon_b_(std::move(epsilon_b)), occupation_a_(std::move(occupation_a)),
      occupation_b_(std::move(occupation_b)), Da_(std::move(Da)), Db_(std::move(Db)), energy_(energy),
      molecule_(std::move(molecule)), basis_(std::move(basis)), basis_snapshot_(std::move(basis_snapshot)),
      functional_(std::move(functional)),
      sites_(std::move(sites)), grid_points_(std::move(grid_points)), grid_weights_(std::move(grid_weights)),
      grid_blocks_(std::move(grid_blocks)), grac_(std::move(grac)),
      functional_name_(std::move(functional_name)), grac_x_name_(std::move(grac_x_name)),
      grac_c_name_(std::move(grac_c_name)) {}

void FrozenResponseContext::verify_basis_unchanged() const {
    if (!basis_ || !basis_snapshot_ || basis_->structural_snapshot() != *basis_snapshot_)
        throw PSIEXCEPTION("FrozenResponseContext: retained basis changed after provenance sealing");
}

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
            worker.grac_beta != grac_seal.functional.grac_beta || !(worker.grac_x == actual_x) ||
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
        kProtocolGRACX, kProtocolGRACC));
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
