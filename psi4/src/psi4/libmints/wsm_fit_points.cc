/*
 * @BEGIN LICENSE
 *
 * Psi4: an open-source quantum chemistry software package
 *
 * Copyright (c) 2007-2025 The Psi4 Developers.
 *
 * The copyrights for code used from other parties are included in
 * the corresponding files.
 *
 * This file is part of Psi4.
 *
 * Psi4 is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * Psi4 is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License along
 * with Psi4; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * @END LICENSE
 */

/*
 * Deterministic, symmetry-faithful external fit points for WSM refinement.
 *
 * Construction. For each shell offset s_k the retained set is the discrete
 * sampling of the equidistant (offset) surface
 *
 *     { p : min_A |p - R_A| / rho_A = s_k },
 *
 * obtained by placing a Lebedev unit-sphere grid of radius s_k * rho_A around
 * every nucleus A and discarding any node that lies strictly inside another
 * nucleus' sphere at the same shell. Shell offsets are equally spaced across the
 * closed interval [inner_limit, outer_limit]. The Lebedev tables Psi4 already
 * uses for its DFT grids are reused verbatim through lebedev_spherical_grid;
 * no new angular quadrature is defined here.
 *
 * Symmetry. Each Lebedev grid is a union of complete octahedral orbits, so its
 * node set is invariant under the full O_h matrix group, and the keep predicate
 * depends only on internuclear geometry. The construction is therefore exactly
 * invariant under any molecular point group whose operations are signed
 * coordinate permutations in the angular frame -- which is every abelian D2h
 * subgroup Psi4 uses, since Psi4 expresses a molecule in its symmetry frame.
 * That condition is checked rather than assumed, and the invariance of the
 * finished set is then verified as a postcondition. An arbitrary point set
 * produces arbitrary fitted anisotropy, so a set that cannot be shown invariant
 * fails closed instead of quietly injecting symmetry-violating residuals.
 *
 * Determinism. Candidates are enumerated shell-major, then atom, then Lebedev
 * node index, and duplicates are merged first-come-first-kept. There is no RNG,
 * no hashing, and no iteration-order dependence, so identical input geometry
 * yields bit-identical output. Because that enumeration is purely index driven,
 * rotating the centers, the operations and the angular frame by R reproduces the
 * same sequence of points transformed by R.
 */

#include "psi4/libmints/atomic_polarizability.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <vector>

#include "psi4/libfock/cubature.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libmints/pointgrp.h"
#include "psi4/liboptions/liboptions.h"
#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libpsi4util/exception.h"
#include "psi4/physconst.h"

namespace psi {

namespace {

const char* const kPrefix = "WSM fit points: ";

/* Storage ceiling on the pre-pruning candidate set, far above any useful grid. */
constexpr std::size_t kMaximumCandidatePoints = 200000;
constexpr double kOrthogonalityTolerance = 1.0e-10;
constexpr double kInvarianceTolerance = 1.0e-9;

/*
 * Bondi, J. Phys. Chem. 68 (1964) 441, extended by Mantina et al.,
 * J. Phys. Chem. A 113 (2009) 5806, in angstrom and indexed by atomic number.
 * Zero marks an element with no tabulated radius, which fails closed.
 */
constexpr double kBondiRadiiAngstrom[] = {
    0.00,                                                              // placeholder
    1.20, 1.40,                                                        // H  He
    1.82, 1.53, 1.92, 1.70, 1.55, 1.52, 1.47, 1.54,                    // Li - Ne
    2.27, 1.73, 1.84, 2.10, 1.80, 1.80, 1.75, 1.88,                    // Na - Ar
    2.75, 2.31, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,              // K  - Co
    0.00, 0.00, 1.39, 1.87, 2.11, 1.85, 1.90, 1.85, 2.02,              // Ni - Kr
    3.03, 2.49, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,              // Rb - Rh
    0.00, 1.72, 1.58, 1.93, 2.17, 2.06, 2.06, 1.98, 2.16,              // Pd - Xe
};
constexpr int kBondiMaxZ = static_cast<int>(sizeof(kBondiRadiiAngstrom) / sizeof(double)) - 1;

void require_finite(double value, const char* name) {
    if (!std::isfinite(value)) throw PSIEXCEPTION(std::string(kPrefix) + name + " must be finite");
}

std::size_t checked_product(std::size_t first, std::size_t second) {
    if (first != 0 && second > std::numeric_limits<std::size_t>::max() / first)
        throw PSIEXCEPTION(std::string(kPrefix) + "candidate count overflows");
    return first * second;
}

std::string radial_units_name(FitPointRadialUnits units) {
    return units == FitPointRadialUnits::Bohr ? "BOHR" : "VDW";
}

/** Equally spaced shell offsets across the closed limit interval. */
std::vector<double> shell_offsets(const FitPointOptions& options) {
    std::vector<double> offsets(options.radial_shells);
    if (options.radial_shells == 1) {
        offsets[0] = options.inner_limit;
        return offsets;
    }
    const double span = options.outer_limit - options.inner_limit;
    const auto last = options.radial_shells - 1;
    for (std::size_t shell = 0; shell < options.radial_shells; ++shell)
        offsets[shell] = options.inner_limit +
                         span * static_cast<double>(shell) / static_cast<double>(last);
    offsets[last] = options.outer_limit;
    return offsets;
}

void validate_options(const FitPointOptions& options) {
    require_finite(options.inner_limit, "inner shell limit");
    require_finite(options.outer_limit, "outer shell limit");
    require_finite(options.merge_tolerance_bohr, "merge tolerance");
    if (options.radial_shells == 0)
        throw PSIEXCEPTION(std::string(kPrefix) + "at least one radial shell is required");
    if (options.maximum_points == 0)
        throw PSIEXCEPTION(std::string(kPrefix) + "maximum point count must be positive");
    if (options.inner_limit <= 0.0)
        throw PSIEXCEPTION(std::string(kPrefix) + "inner shell limit must be positive");
    if (options.radial_shells > 1 && !(options.outer_limit > options.inner_limit))
        throw PSIEXCEPTION(std::string(kPrefix) +
                           "outer shell limit must exceed the inner shell limit when more "
                           "than one shell is requested");
    if (options.radial_shells == 1 && options.outer_limit < options.inner_limit)
        throw PSIEXCEPTION(std::string(kPrefix) +
                           "outer shell limit must not be below the inner shell limit");
    if (!(options.merge_tolerance_bohr > 0.0))
        throw PSIEXCEPTION(std::string(kPrefix) + "merge tolerance must be positive");
}

SitePosition apply_operation(const FitPointOperation& operation, const SitePosition& point) {
    SitePosition image{};
    for (std::size_t row = 0; row < 3; ++row)
        image[row] = operation[3 * row] * point[0] + operation[3 * row + 1] * point[1] +
                     operation[3 * row + 2] * point[2];
    return image;
}

double squared_distance(const SitePosition& first, const SitePosition& second) {
    double total = 0.0;
    for (std::size_t axis = 0; axis < 3; ++axis) {
        const double delta = first[axis] - second[axis];
        total += delta * delta;
    }
    return total;
}

void require_orthogonal(const FitPointOperation& operation, const std::string& what) {
    for (const double entry : operation) require_finite(entry, what.c_str());
    for (std::size_t row = 0; row < 3; ++row) {
        for (std::size_t column = 0; column < 3; ++column) {
            double product = 0.0;
            for (std::size_t axis = 0; axis < 3; ++axis)
                product += operation[3 * axis + row] * operation[3 * axis + column];
            const double expected = (row == column) ? 1.0 : 0.0;
            if (std::abs(product - expected) > kOrthogonalityTolerance)
                throw PSIEXCEPTION(std::string(kPrefix) + what + " is not orthogonal");
        }
    }
}

double determinant(const FitPointOperation& operation) {
    return operation[0] * (operation[4] * operation[8] - operation[5] * operation[7]) -
           operation[1] * (operation[3] * operation[8] - operation[5] * operation[6]) +
           operation[2] * (operation[3] * operation[7] - operation[4] * operation[6]);
}

/** Return F^T S F, the operation expressed in the angular frame. */
FitPointOperation in_angular_frame(const FitPointOperation& frame,
                                   const FitPointOperation& operation) {
    FitPointOperation product{};  // S F
    FitPointOperation result{};   // F^T (S F)
    for (std::size_t row = 0; row < 3; ++row)
        for (std::size_t column = 0; column < 3; ++column)
            for (std::size_t axis = 0; axis < 3; ++axis)
                product[3 * row + column] += operation[3 * row + axis] * frame[3 * axis + column];
    for (std::size_t row = 0; row < 3; ++row)
        for (std::size_t column = 0; column < 3; ++column)
            for (std::size_t axis = 0; axis < 3; ++axis)
                result[3 * row + column] += frame[3 * axis + row] * product[3 * axis + column];
    return result;
}

/**
 * Largest departure of a matrix from a signed coordinate permutation, i.e. from
 * membership of the 48-element O_h matrix group that every Lebedev node set is
 * invariant under.
 */
double octahedral_deviation(const FitPointOperation& operation) {
    double deviation = 0.0;
    for (const double entry : operation) {
        const double magnitude = std::abs(entry);
        deviation = std::max(deviation, std::min(magnitude, std::abs(magnitude - 1.0)));
    }
    return deviation;
}

/** Reject anything that is not an exact molecular symmetry of the input frame. */
double validate_symmetry_operations(const std::vector<int>& atomic_numbers,
                                    const std::vector<SitePosition>& centers,
                                    const std::vector<FitPointOperation>& operations,
                                    const FitPointOperation& frame, double geometry_scale) {
    if (operations.empty())
        throw PSIEXCEPTION(std::string(kPrefix) +
                           "at least the identity symmetry operation is required");
    require_orthogonal(frame, "angular frame");
    if (determinant(frame) < 0.0)
        throw PSIEXCEPTION(std::string(kPrefix) + "angular frame must be a proper rotation");
    const double framework_tolerance = 1.0e-8 * std::max(1.0, geometry_scale);
    double worst_octahedral = 0.0;
    for (std::size_t index = 0; index < operations.size(); ++index) {
        const auto& operation = operations[index];
        require_orthogonal(operation, "symmetry operation " + std::to_string(index));
        for (std::size_t atom = 0; atom < centers.size(); ++atom) {
            const auto image = apply_operation(operation, centers[atom]);
            bool matched = false;
            for (std::size_t other = 0; other < centers.size() && !matched; ++other)
                matched = atomic_numbers[other] == atomic_numbers[atom] &&
                          std::sqrt(squared_distance(image, centers[other])) <= framework_tolerance;
            if (!matched)
                throw PSIEXCEPTION(std::string(kPrefix) + "symmetry operation " +
                                   std::to_string(index) +
                                   " does not map the nuclear framework onto itself");
        }
        const double deviation = octahedral_deviation(in_angular_frame(frame, operation));
        if (deviation > kOrthogonalityTolerance)
            throw PSIEXCEPTION(
                std::string(kPrefix) + "symmetry operation " + std::to_string(index) +
                " is not a signed coordinate permutation in the angular frame, so a Lebedev "
                "node set cannot be invariant under it; supply the molecule's symmetry frame "
                "as the angular frame");
        worst_octahedral = std::max(worst_octahedral, deviation);
    }
    return worst_octahedral;
}

}  // namespace

FitPointOperation identity_fit_point_frame() {
    return FitPointOperation{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
}

double bondi_vdw_radius_bohr(int atomic_number) {
    if (atomic_number < 1 || atomic_number > kBondiMaxZ ||
        kBondiRadiiAngstrom[atomic_number] == 0.0)
        throw PSIEXCEPTION(std::string(kPrefix) + "no tabulated van der Waals radius for " +
                           "atomic number " + std::to_string(atomic_number) +
                           "; use the bohr radial convention instead");
    return kBondiRadiiAngstrom[atomic_number] / pc_bohr2angstroms;
}

FitPointPlan plan_fit_points(std::size_t atom_count, const FitPointOptions& options) {
    validate_options(options);
    if (atom_count == 0) throw PSIEXCEPTION(std::string(kPrefix) + "at least one atom is required");

    FitPointPlan plan;
    plan.atom_count = atom_count;
    plan.spherical_points = options.spherical_points;
    plan.radial_shells = options.radial_shells;
    // Reuse of Psi4's DFT Lebedev tables; this throws for unsupported sizes.
    plan.lebedev_order = static_cast<std::size_t>(
        lebedev_spherical_grid_order(static_cast<int>(options.spherical_points)));
    plan.candidate_count = checked_product(
        checked_product(atom_count, options.spherical_points), options.radial_shells);
    plan.maximum_points = options.maximum_points;
    plan.shell_offsets = shell_offsets(options);
    plan.radial_units = radial_units_name(options.radial_units);
    plan.algorithm = "nested_equidistant_lebedev_surfaces";
    plan.candidate_bytes = checked_product(plan.candidate_count, sizeof(SitePosition));
    plan.retained_metadata_bytes = checked_product(
        plan.candidate_count, sizeof(double) + 2 * sizeof(std::size_t));
    plan.estimated_bytes = plan.candidate_bytes + plan.retained_metadata_bytes;
    // The candidate set is the only allocation the generator makes; the retained
    // count depends on geometry, so maximum_points is enforced after pruning.
    if (plan.candidate_count > kMaximumCandidatePoints)
        throw PSIEXCEPTION(std::string(kPrefix) + "candidate set of " +
                           std::to_string(plan.candidate_count) + " points exceeds the " +
                           std::to_string(kMaximumCandidatePoints) +
                           "-point storage ceiling; reduce the spherical point count or the "
                           "shell count");
    return plan;
}

FitPointSet generate_fit_points(const std::vector<int>& atomic_numbers,
                                const std::vector<SitePosition>& centers,
                                const std::vector<FitPointOperation>& symmetry_operations,
                                const FitPointOperation& angular_frame,
                                const FitPointOptions& options) {
    if (atomic_numbers.size() != centers.size())
        throw PSIEXCEPTION(std::string(kPrefix) + "expected one atomic number per center");
    auto plan = plan_fit_points(centers.size(), options);

    double geometry_scale = 0.0;
    for (const auto& center : centers)
        for (const double coordinate : center) {
            require_finite(coordinate, "nuclear coordinate");
            geometry_scale = std::max(geometry_scale, std::abs(coordinate));
        }
    for (std::size_t atom = 0; atom < centers.size(); ++atom)
        for (std::size_t other = atom + 1; other < centers.size(); ++other)
            if (std::sqrt(squared_distance(centers[atom], centers[other])) <
                options.merge_tolerance_bohr)
                throw PSIEXCEPTION(std::string(kPrefix) + "nuclei " + std::to_string(atom) +
                                   " and " + std::to_string(other) + " are coincident");
    const double octahedral = validate_symmetry_operations(
        atomic_numbers, centers, symmetry_operations, angular_frame, geometry_scale);
    plan.symmetry_operation_count = symmetry_operations.size();

    std::vector<double> radii(centers.size(), 1.0);
    if (options.radial_units == FitPointRadialUnits::VanDerWaals)
        for (std::size_t atom = 0; atom < centers.size(); ++atom)
            radii[atom] = bondi_vdw_radius_bohr(atomic_numbers[atom]);

    // Reuse of Psi4's DFT Lebedev tables; only the node directions are needed, and
    // they are carried into the angular frame once for the whole enumeration.
    std::vector<SitePosition> directions;
    directions.reserve(options.spherical_points);
    for (const auto& node : lebedev_spherical_grid(static_cast<int>(options.spherical_points)))
        directions.push_back(
            apply_operation(angular_frame, SitePosition{node.x, node.y, node.z}));

    FitPointSet result;
    result.scaling_radii = radii;
    result.points.reserve(plan.candidate_count);
    result.nearest_offsets.reserve(plan.candidate_count);
    result.shell_index.reserve(plan.candidate_count);
    result.generator_atom.reserve(plan.candidate_count);

    const double merge_squared = options.merge_tolerance_bohr * options.merge_tolerance_bohr;
    const auto already_present = [&result, merge_squared](const SitePosition& point) {
        for (const auto& kept : result.points)
            if (squared_distance(kept, point) < merge_squared) return true;
        return false;
    };
    const auto append = [&result](const SitePosition& point, double offset, std::size_t shell,
                                  std::size_t atom) {
        result.points.push_back(point);
        result.nearest_offsets.push_back(offset);
        result.shell_index.push_back(shell);
        result.generator_atom.push_back(atom);
    };

    // Candidate enumeration is shell-major, then atom, then Lebedev node index.
    for (std::size_t shell = 0; shell < plan.shell_offsets.size(); ++shell) {
        const double offset = plan.shell_offsets[shell];
        for (std::size_t atom = 0; atom < centers.size(); ++atom) {
            const double radius = offset * radii[atom];
            for (const auto& direction : directions) {
                SitePosition candidate{centers[atom][0] + radius * direction[0],
                                       centers[atom][1] + radius * direction[1],
                                       centers[atom][2] + radius * direction[2]};
                // Keep only nodes on the equidistant surface, i.e. no closer to any
                // other nucleus than that nucleus' own radius at this shell. The
                // tolerance keeps exact ties, which are then merged as duplicates.
                bool on_surface = true;
                for (std::size_t other = 0; other < centers.size() && on_surface; ++other) {
                    if (other == atom) continue;
                    const double allowed = offset * radii[other];
                    on_surface = squared_distance(candidate, centers[other]) >=
                                 allowed * allowed - 2.0 * allowed * options.merge_tolerance_bohr;
                }
                if (!on_surface) continue;
                if (already_present(candidate)) continue;
                append(candidate, offset, shell, atom);
            }
        }
    }

    plan.point_count = result.points.size();
    if (plan.point_count == 0)
        throw PSIEXCEPTION(std::string(kPrefix) + "the requested shells retained no points");
    if (plan.point_count > options.maximum_points)
        throw PSIEXCEPTION(std::string(kPrefix) + "generated " + std::to_string(plan.point_count) +
                           " points, which exceeds the maximum point count of " +
                           std::to_string(options.maximum_points));

    for (std::size_t index = 0; index < result.points.size(); ++index) {
        double nearest = std::numeric_limits<double>::max();
        for (std::size_t atom = 0; atom < centers.size(); ++atom)
            nearest = std::min(nearest,
                               std::sqrt(squared_distance(result.points[index], centers[atom])) /
                                   radii[atom]);
        if (!std::isfinite(nearest) || nearest < options.inner_limit - 1.0e-9)
            throw PSIEXCEPTION(std::string(kPrefix) +
                               "a generated point fell inside the inner shell limit");
        result.nearest_offsets[index] = nearest;
    }

    // Verified postcondition: every operation maps the finished set onto itself.
    for (const auto& operation : symmetry_operations) {
        for (const auto& point : result.points) {
            const auto image = apply_operation(operation, point);
            double nearest = std::numeric_limits<double>::max();
            for (const auto& other : result.points)
                nearest = std::min(nearest, squared_distance(image, other));
            result.max_symmetry_deviation =
                std::max(result.max_symmetry_deviation, std::sqrt(nearest));
        }
    }
    if (!(result.max_symmetry_deviation <= kInvarianceTolerance))
        throw PSIEXCEPTION(std::string(kPrefix) +
                           "the generated set is not invariant under the molecular point group");
    result.max_octahedral_deviation = octahedral;
    result.plan = plan;
    return result;
}

FitPointOptions fit_point_options_from(Options& options) {
    FitPointOptions policy;
    const int spherical = options.get_int("ATOMIC_POLARIZABILITY_FIT_SPHERICAL_POINTS");
    const int shells = options.get_int("ATOMIC_POLARIZABILITY_FIT_RADIAL_SHELLS");
    const int maximum = options.get_int("ATOMIC_POLARIZABILITY_FIT_MAX_POINTS");
    if (spherical < 0 || shells < 0 || maximum < 0)
        throw PSIEXCEPTION(std::string(kPrefix) + "point-count keywords must be nonnegative");
    policy.spherical_points = static_cast<std::size_t>(spherical);
    policy.radial_shells = static_cast<std::size_t>(shells);
    policy.maximum_points = static_cast<std::size_t>(maximum);
    policy.inner_limit = options.get_double("ATOMIC_POLARIZABILITY_FIT_INNER_LIMIT");
    policy.outer_limit = options.get_double("ATOMIC_POLARIZABILITY_FIT_OUTER_LIMIT");
    const std::string units = options.get_str("ATOMIC_POLARIZABILITY_FIT_RADIAL_UNITS");
    if (units == "BOHR")
        policy.radial_units = FitPointRadialUnits::Bohr;
    else if (units == "VDW")
        policy.radial_units = FitPointRadialUnits::VanDerWaals;
    else
        throw PSIEXCEPTION(std::string(kPrefix) + "unsupported radial units '" + units + "'");
    return policy;
}

FitPointSet generate_wsm_fit_points(const Molecule& molecule, Options& options) {
    const auto policy = fit_point_options_from(options);
    const int atom_count = molecule.natom();
    if (atom_count <= 0) throw PSIEXCEPTION(std::string(kPrefix) + "at least one atom is required");

    const auto group = molecule.point_group();
    if (!group) throw PSIEXCEPTION(std::string(kPrefix) + "the molecule has no point group");
    // Psi4's symmetry operations act about the point group's own origin and in the
    // molecule's symmetry frame, so shift the geometry onto that origin and shift
    // the generated points back afterwards.
    const auto& origin = group->origin();

    std::vector<int> atomic_numbers(static_cast<std::size_t>(atom_count));
    std::vector<SitePosition> centers(static_cast<std::size_t>(atom_count));
    for (int atom = 0; atom < atom_count; ++atom) {
        atomic_numbers[atom] = static_cast<int>(std::lround(molecule.Z(atom)));
        centers[atom] = {molecule.x(atom) - origin[0], molecule.y(atom) - origin[1],
                         molecule.z(atom) - origin[2]};
    }

    const auto table = group->char_table();
    std::vector<FitPointOperation> operations(static_cast<std::size_t>(table.order()));
    for (int index = 0; index < table.order(); ++index) {
        const auto& symop = table.symm_operation(index);
        for (std::size_t row = 0; row < 3; ++row)
            for (std::size_t column = 0; column < 3; ++column)
                operations[index][3 * row + column] =
                    symop(static_cast<int>(row), static_cast<int>(column));
    }

    auto result = generate_fit_points(atomic_numbers, centers, operations,
                                      identity_fit_point_frame(), policy);
    for (auto& point : result.points)
        for (std::size_t axis = 0; axis < 3; ++axis) point[axis] += origin[axis];
    return result;
}

}  // namespace psi
