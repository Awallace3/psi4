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

#include "psi4/libmints/basisset.h"
#include "psi4/libmints/gshell.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libmints/vector.h"
#include "psi4/libmints/wavefunction.h"
#include "psi4/liboptions/liboptions.h"
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

bool has_content(const std::string& value) {
    return value.find_first_not_of(" \t\r\n") != std::string::npos;
}

std::string make_basis_fingerprint(const BasisSet& basis) {
    std::ostringstream out;
    out << std::hexfloat << basis.name().size() << ':' << basis.name() << '|' << basis.nbf() << '|'
        << basis.nao() << '|' << basis.nshell() << '|' << basis.n_ecp_shell() << '|'
        << basis.has_puream();
    for (int shell_index = 0; shell_index < basis.nshell(); ++shell_index) {
        const auto& shell = basis.shell(shell_index);
        out << "|S:" << shell.ncenter() << ':' << shell.am() << ':' << shell.is_pure() << ':'
            << shell.nprimitive();
        for (int primitive = 0; primitive < shell.nprimitive(); ++primitive) {
            out << ':' << shell.exp(primitive) << ':' << shell.original_coef(primitive);
        }
    }
    for (int shell_index = 0; shell_index < basis.n_ecp_shell(); ++shell_index) {
        const auto& shell = basis.ecp_shell(shell_index);
        out << "|E:" << shell.ncenter() << ':' << shell.am() << ':' << shell.nprimitive();
        for (int primitive = 0; primitive < shell.nprimitive(); ++primitive) {
            out << ':' << shell.exp(primitive) << ':' << shell.original_coef(primitive) << ':'
                << shell.nval(primitive);
        }
    }
    return out.str();
}

std::string make_grid_fingerprint(const DFTGridIdentity& grid) {
    std::ostringstream out;
    out << std::hexfloat << grid.spherical_points << '|' << grid.radial_points << '|'
        << grid.block_max_points << '|' << grid.block_min_points << '|' << grid.bs_radius_alpha << '|'
        << grid.basis_tolerance << '|' << grid.weights_tolerance << '|' << grid.density_tolerance << '|'
        << grid.pruning_alpha << '|' << grid.block_max_radius << '|' << grid.remove_distant_points << '|'
        << grid.grid_name.size() << ':' << grid.grid_name << '|' << grid.radial_scheme.size() << ':'
        << grid.radial_scheme << '|' << grid.spherical_scheme.size() << ':' << grid.spherical_scheme << '|'
        << grid.nuclear_scheme.size() << ':' << grid.nuclear_scheme << '|' << grid.pruning_scheme.size()
        << ':' << grid.pruning_scheme << '|' << grid.block_scheme.size() << ':' << grid.block_scheme;
    return out.str();
}

WavefunctionIdentity provider_identity(const std::shared_ptr<Wavefunction>& wfn) {
    if (!wfn) throw PSIEXCEPTION("ISAPolResponseProvider: wavefunction is null");
    return WavefunctionIdentity::from_wavefunction(wfn);
}

}  // namespace

bool DFTGridIdentity::operator==(const DFTGridIdentity& other) const {
    return spherical_points == other.spherical_points && radial_points == other.radial_points &&
           block_max_points == other.block_max_points && block_min_points == other.block_min_points &&
           bs_radius_alpha == other.bs_radius_alpha && basis_tolerance == other.basis_tolerance &&
           weights_tolerance == other.weights_tolerance && density_tolerance == other.density_tolerance &&
           pruning_alpha == other.pruning_alpha && block_max_radius == other.block_max_radius &&
           remove_distant_points == other.remove_distant_points && grid_name == other.grid_name &&
           radial_scheme == other.radial_scheme && spherical_scheme == other.spherical_scheme &&
           nuclear_scheme == other.nuclear_scheme && pruning_scheme == other.pruning_scheme &&
           block_scheme == other.block_scheme && fingerprint == other.fingerprint;
}

WavefunctionIdentity WavefunctionIdentity::from_wavefunction(const std::shared_ptr<Wavefunction>& wfn) {
    if (!wfn) throw PSIEXCEPTION("WavefunctionIdentity: wavefunction is null");
    const auto molecule = wfn->molecule();
    const auto basis = wfn->basisset();
    if (!molecule || !basis) {
        throw PSIEXCEPTION("WavefunctionIdentity: wavefunction must own a molecule and orbital basis");
    }

    WavefunctionIdentity identity;
    identity.nuclear_charges.reserve(molecule->natom());
    identity.geometry.reserve(molecule->natom());
    for (int atom = 0; atom < molecule->natom(); ++atom) {
        identity.nuclear_charges.push_back(molecule->Z(atom));
        identity.geometry.push_back({molecule->x(atom), molecule->y(atom), molecule->z(atom)});
    }
    identity.molecular_charge = molecule->molecular_charge();
    identity.multiplicity = molecule->multiplicity();
    identity.basis_name = basis->name();
    identity.basis_nbf = basis->nbf();
    identity.basis_nao = basis->nao();
    identity.basis_nshell = basis->nshell();
    identity.basis_necp_shell = basis->n_ecp_shell();
    identity.basis_has_puream = basis->has_puream();
    identity.basis_fingerprint = make_basis_fingerprint(*basis);
    identity.method = wfn->module() + "/" + wfn->name();
    identity.reference = wfn->options().get_str("REFERENCE");
    identity.functional = wfn->functional_identity();
    identity.functional_fingerprint = wfn->functional_fingerprint();

    auto& options = wfn->options();
    auto& grid = identity.grid;
    grid.spherical_points = options.get_int("DFT_SPHERICAL_POINTS");
    grid.radial_points = options.get_int("DFT_RADIAL_POINTS");
    grid.block_max_points = options.get_int("DFT_BLOCK_MAX_POINTS");
    grid.block_min_points = options.get_int("DFT_BLOCK_MIN_POINTS");
    grid.bs_radius_alpha = options.get_double("DFT_BS_RADIUS_ALPHA");
    grid.basis_tolerance = options.get_double("DFT_BASIS_TOLERANCE");
    grid.weights_tolerance = options.get_double("DFT_WEIGHTS_TOLERANCE");
    grid.density_tolerance = options.get_double("DFT_DENSITY_TOLERANCE");
    grid.pruning_alpha = options.get_double("DFT_PRUNING_ALPHA");
    grid.block_max_radius = options.get_double("DFT_BLOCK_MAX_RADIUS");
    grid.remove_distant_points = options.get_bool("DFT_REMOVE_DISTANT_POINTS");
    grid.grid_name = options.get_str("DFT_GRID_NAME");
    grid.radial_scheme = options.get_str("DFT_RADIAL_SCHEME");
    grid.spherical_scheme = options.get_str("DFT_SPHERICAL_SCHEME");
    grid.nuclear_scheme = options.get_str("DFT_NUCLEAR_SCHEME");
    grid.pruning_scheme = options.get_str("DFT_PRUNING_SCHEME");
    grid.block_scheme = options.get_str("DFT_BLOCK_SCHEME");
    grid.fingerprint = make_grid_fingerprint(grid);
    identity.validate();
    return identity;
}

void WavefunctionIdentity::validate() const {
    if (nuclear_charges.empty() || nuclear_charges.size() != geometry.size() || multiplicity <= 0) {
        throw PSIEXCEPTION("WavefunctionIdentity: molecular dimensions and multiplicity are invalid");
    }
    for (std::size_t site = 0; site < nuclear_charges.size(); ++site) {
        if (!std::isfinite(nuclear_charges[site]) || nuclear_charges[site] < 0.0 ||
            !std::all_of(geometry[site].begin(), geometry[site].end(),
                         [](double value) { return std::isfinite(value); })) {
            throw PSIEXCEPTION("WavefunctionIdentity: geometry and nuclear charges must be finite");
        }
    }
    const auto slash = method.find('/');
    if (!has_content(basis_name) || !has_content(basis_fingerprint) || !has_content(method) ||
        slash == std::string::npos || slash == 0 || slash + 1 == method.size() || !has_content(reference) ||
        !has_content(functional) || !has_content(functional_fingerprint) ||
        !has_content(grid.radial_scheme) ||
        !has_content(grid.spherical_scheme) || !has_content(grid.nuclear_scheme) ||
        !has_content(grid.pruning_scheme) || !has_content(grid.block_scheme) ||
        !has_content(grid.fingerprint)) {
        throw PSIEXCEPTION(
            "WavefunctionIdentity: all basis, method, functional, and grid identity fields are required");
    }
    if (basis_nbf == 0 || basis_nao == 0 || basis_nshell == 0 || grid.spherical_points <= 0 ||
        grid.radial_points <= 0 || grid.block_max_points <= 0 || grid.block_min_points <= 0) {
        throw PSIEXCEPTION("WavefunctionIdentity: basis and grid dimensions must be positive");
    }
    const std::array<double, 6> grid_values{grid.bs_radius_alpha, grid.basis_tolerance,
                                            grid.weights_tolerance, grid.density_tolerance,
                                            grid.pruning_alpha, grid.block_max_radius};
    if (!std::all_of(grid_values.begin(), grid_values.end(),
                     [](double value) { return std::isfinite(value); })) {
        throw PSIEXCEPTION("WavefunctionIdentity: grid settings must be finite");
    }
}

bool WavefunctionIdentity::operator==(const WavefunctionIdentity& other) const {
    return nuclear_charges == other.nuclear_charges && geometry == other.geometry &&
           molecular_charge == other.molecular_charge && multiplicity == other.multiplicity &&
           basis_name == other.basis_name && basis_nbf == other.basis_nbf && basis_nao == other.basis_nao &&
           basis_nshell == other.basis_nshell && basis_necp_shell == other.basis_necp_shell &&
           basis_has_puream == other.basis_has_puream && basis_fingerprint == other.basis_fingerprint &&
           method == other.method && reference == other.reference && functional == other.functional &&
           functional_fingerprint == other.functional_fingerprint && grid == other.grid;
}

ResponseKernel::ResponseKernel(double chf_exchange, double alda_kernel)
    : chf_exchange_(chf_exchange), alda_kernel_(alda_kernel) {
    if (!std::isfinite(chf_exchange_) || chf_exchange_ != 0.25) {
        throw PSIEXCEPTION("ResponseKernel: CHF exchange coefficient must be exactly 0.25");
    }
    if (!std::isfinite(alda_kernel_) || alda_kernel_ != 0.75) {
        throw PSIEXCEPTION("ResponseKernel: ALDA coefficient must be exactly 0.75");
    }
}

GRACProvenance::GRACProvenance(WavefunctionIdentity identity, double neutral_energy,
                               double cation_energy, double homo_energy, double ionization_potential,
                               double shift)
    : identity_(std::move(identity)),
      neutral_energy_(neutral_energy),
      cation_energy_(cation_energy),
      homo_energy_(homo_energy),
      ionization_potential_(ionization_potential),
      shift_(shift) {
    identity_.validate();
    const std::array<double, 5> values{
        neutral_energy_, cation_energy_, homo_energy_, ionization_potential_, shift_};
    if (!std::all_of(values.begin(), values.end(), [](double value) { return std::isfinite(value); })) {
        throw PSIEXCEPTION("GRACProvenance: all energy metadata must be finite");
    }
    if (!(cation_energy_ > neutral_energy_)) {
        throw PSIEXCEPTION("GRACProvenance: cation energy must be greater than neutral energy");
    }
    if (!(ionization_potential_ > 0.0)) {
        throw PSIEXCEPTION("GRACProvenance: ionization potential must be positive");
    }
    if (!(homo_energy_ < 0.0)) {
        throw PSIEXCEPTION("GRACProvenance: HOMO energy must be negative");
    }
    if (shift_ < 0.0) {
        throw PSIEXCEPTION("GRACProvenance: GRAC shift must be nonnegative");
    }

    const double derived_ionization_potential = cation_energy_ - neutral_energy_;
    const double derived_shift = ionization_potential_ + homo_energy_;
    if (!std::isfinite(derived_ionization_potential) || !std::isfinite(derived_shift)) {
        throw PSIEXCEPTION("GRACProvenance: derived ionization potential and shift must be finite");
    }
    const auto consistent = [](double supplied, double derived) {
        const double scale = std::max({1.0, std::abs(supplied), std::abs(derived)});
        return std::abs(supplied - derived) <= kValidationTolerance * scale;
    };
    if (!consistent(ionization_potential_, derived_ionization_potential)) {
        throw PSIEXCEPTION(
            "GRACProvenance: ionization potential must equal cation energy minus neutral energy");
    }
    if (!consistent(shift_, derived_shift)) {
        throw PSIEXCEPTION("GRACProvenance: GRAC shift must equal ionization potential plus HOMO energy");
    }
}

ISAWeights::ISAWeights(WavefunctionIdentity identity, std::size_t point_count,
                       std::size_t grid_dimension, std::size_t site_count, std::vector<double> points,
                       std::vector<double> quadrature_weights, std::vector<double> partition_weights)
    : identity_(std::move(identity)),
      point_count_(point_count),
      grid_dimension_(grid_dimension),
      site_count_(site_count),
      points_(std::move(points)),
      quadrature_weights_(std::move(quadrature_weights)),
      partition_weights_(std::move(partition_weights)) {
    identity_.validate();
    if (point_count_ == 0) throw PSIEXCEPTION("ISAWeights: point count must be positive");
    if (grid_dimension_ != 3) throw PSIEXCEPTION("ISAWeights: grid dimension must be exactly 3");
    if (site_count_ == 0) throw PSIEXCEPTION("ISAWeights: site count must be positive");
    if (point_count_ > std::numeric_limits<std::size_t>::max() / grid_dimension_ ||
        points_.size() != point_count_ * grid_dimension_) {
        throw PSIEXCEPTION("ISAWeights: point coordinates do not match declared dimensions");
    }
    if (quadrature_weights_.size() != point_count_) {
        throw PSIEXCEPTION("ISAWeights: quadrature weights do not match declared dimensions");
    }
    if (point_count_ > std::numeric_limits<std::size_t>::max() / site_count_ ||
        partition_weights_.size() != point_count_ * site_count_) {
        throw PSIEXCEPTION("ISAWeights: partition weights do not match declared dimensions");
    }
    if (!std::all_of(points_.begin(), points_.end(), [](double value) { return std::isfinite(value); }) ||
        !std::all_of(quadrature_weights_.begin(), quadrature_weights_.end(),
                     [](double value) { return std::isfinite(value) && value > 0.0; }) ||
        !std::all_of(partition_weights_.begin(), partition_weights_.end(),
                     [](double value) { return std::isfinite(value) && value >= 0.0; })) {
        throw PSIEXCEPTION(
            "ISAWeights: coordinates and weights must be finite; quadrature weights must be positive "
            "and partition weights must be nonnegative");
    }
    for (std::size_t point = 0; point < point_count_; ++point) {
        double sum = 0.0;
        for (std::size_t site = 0; site < site_count_; ++site) {
            sum += partition_weights_[point * site_count_ + site];
        }
        if (!std::isfinite(sum) || std::abs(sum - 1.0) > kValidationTolerance) {
            throw PSIEXCEPTION("ISAWeights: partition unity failed at a grid point");
        }
    }
}

ISAPolResponseProvider::ISAPolResponseProvider(std::shared_ptr<Wavefunction> wfn, ResponseKernel kernel,
                                               GRACProvenance grac, ISAWeights isa_weights)
    : wfn_(std::move(wfn)),
      identity_snapshot_(provider_identity(wfn_)),
      kernel_(std::move(kernel)),
      grac_(std::move(grac)),
      isa_weights_(std::move(isa_weights)) {
    if (grac_.identity() != identity_snapshot_) {
        throw PSIEXCEPTION("ISAPolResponseProvider: GRAC provenance wavefunction identity mismatch");
    }
    if (isa_weights_.identity() != identity_snapshot_) {
        throw PSIEXCEPTION("ISAPolResponseProvider: ISA weights wavefunction identity mismatch");
    }
}

void ISAPolResponseProvider::validate_current_wavefunction_identity() const {
    if (WavefunctionIdentity::from_wavefunction(wfn_) != identity_snapshot_) {
        throw PSIEXCEPTION(
            "ISAPolResponseProvider: wavefunction identity changed since provider construction");
    }
}

std::size_t ISAPolResponseProvider::expected_response_count(const FrequencyGrid& frequencies) const {
    validate_current_wavefunction_identity();
    if (frequencies.frequencies.empty() || frequencies.frequencies.size() != frequencies.weights.size()) {
        throw PSIEXCEPTION("ISAPolResponseProvider: frequency grid has inconsistent dimensions");
    }
    for (std::size_t point = 0; point < frequencies.frequencies.size(); ++point) {
        const double frequency = frequencies.frequencies[point];
        const double weight = frequencies.weights[point];
        if (!std::isfinite(frequency) || frequency < 0.0 || !std::isfinite(weight) || weight < 0.0 ||
            (point > 0 && frequency <= frequencies.frequencies[point - 1])) {
            throw PSIEXCEPTION(
                "ISAPolResponseProvider: frequency grid must be finite, nonnegative, and increasing");
        }
    }
    return frequencies.frequencies.size();
}

std::vector<SitePairResponse> ISAPolResponseProvider::compute_isapol_response(
    const FrequencyGrid& frequencies) const {
    const auto response_count = expected_response_count(frequencies);
    (void)response_count;
    throw PSIEXCEPTION(
        "ISAPolResponseProvider: native point-response execution is not implemented; no response was published");
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
