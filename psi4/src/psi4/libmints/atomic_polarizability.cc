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
