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

#ifndef PSI4_SRC_PSI4_LIBMINTS_ATOMIC_POLARIZABILITY_H
#define PSI4_SRC_PSI4_LIBMINTS_ATOMIC_POLARIZABILITY_H

#include <array>
#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "psi4/psi4-dec.h"
#include "psi4/libmints/typedefs.h"

namespace psi {

class Wavefunction;

/** Imaginary-frequency points and transformed Gauss-Legendre weights. */
struct PSI_API FrequencyGrid {
    std::vector<double> frequencies;
    std::vector<double> weights;
};

/** Complete rank-3 real-spherical response matrix (3 + 5 + 7 components). */
using L3Matrix = std::array<std::array<double, 15>, 15>;

/** Rank-0-through-rank-3 real-spherical workspace (1 + 3 + 5 + 7 components). */
using L3WorkingVector = std::array<double, 16>;
using L3WorkingMatrix = std::array<std::array<double, 16>, 16>;
using SitePosition = std::array<double, 3>;

/** Dense site-pair response; blocks use row-major (response site, potential site) order. */
struct PSI_API SitePairResponse {
    std::vector<SitePosition> positions;
    std::vector<L3WorkingMatrix> blocks;
};

/** Unweighted undirected graph over the polarizable sites. */
struct PSI_API BondGraph {
    std::size_t site_count;
    std::vector<std::array<std::size_t, 2>> bonds;
};

struct PSI_API BondTransfer {
    std::size_t first;
    std::size_t second;
    std::size_t first_component;
    std::size_t second_component;
    std::size_t fixed_site;
    double amount;
};

struct PSI_API LocalizationResiduals {
    double off_site;
    double charge_sum;
    double reciprocity;
    double molecular_sum;
    double local_charge;
};

/** Fully localized rank-1-through-rank-3 response and deterministic diagnostics. */
struct PSI_API LocalizedResponse {
    std::vector<L3Matrix> local;
    std::vector<BondTransfer> transfers;
    LocalizationResiduals residuals;
};

PSI_API Matrix lw_graph_operator(const BondGraph& graph);
PSI_API std::pair<Matrix, std::vector<double>> lw_graph_pseudoinverse(const BondGraph& graph);
PSI_API L3WorkingVector translate_l3_multipoles(const L3WorkingVector& source,
                                                 const SitePosition& source_minus_target);
PSI_API LocalizedResponse localize_lw(const SitePairResponse& response, const BondGraph& graph,
                                      double residual_tolerance);

/** Build the required static plus ten-point imaginary-frequency grid. */
PSI_API FrequencyGrid make_casimir_grid(unsigned int nonzero_count, double scale);

/** Extract and reorder the real-spherical dipole block as Cartesian x, y, z. */
PSI_API Matrix local_spherical_dipole_to_cartesian(const L3Matrix& spherical);

/** Rotate a symmetric tensor from a right-handed local frame into the global frame. */
PSI_API Matrix rotate_tensor(const Matrix& local, const Matrix& local_to_global);

/** Pack a symmetric Cartesian tensor as xx, xy, xz, yy, yz, zz. */
PSI_API std::array<double, 6> pack_symmetric_tensor(const Matrix& tensor);

/** Native atomic-polarizability pipeline entry point. */
class PSI_API AtomicPolarizabilityCalculator {
   public:
    explicit AtomicPolarizabilityCalculator(std::shared_ptr<Wavefunction> wfn);

    /** Compute and publish the atomic polarizability and dispersion arrays. */
    void compute();

   private:
    void validate_wavefunction_prerequisites() const;

    std::shared_ptr<Wavefunction> wfn_;
};

}  // namespace psi

#endif  // PSI4_SRC_PSI4_LIBMINTS_ATOMIC_POLARIZABILITY_H
