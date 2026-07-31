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
#include <string>
#include <utility>
#include <vector>

#include "psi4/psi4-dec.h"
#include "psi4/libmints/typedefs.h"

namespace psi {

class BasisSet;
struct BasisSetStructuralSnapshot;
class Matrix;
class Molecule;
class SuperFunctional;
class Vector;
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

/** Immutable protocol response-kernel selection, independent of the ground-state functional. */
class PSI_API ResponseKernel {
   public:
    ResponseKernel(double chf_exchange, double alda_kernel);

    double chf_exchange() const { return chf_exchange_; }
    double alda_kernel() const { return alda_kernel_; }

   private:
    double chf_exchange_;
    double alda_kernel_;
};

/** Exact ordered effective DFT-grid state retained by a frozen response context. */
struct PSI_API FrozenGridBlock {
    std::size_t point_offset{};
    std::size_t point_count{};
    std::vector<int> functions_local_to_global;
};

/** Verified GRAC provenance from actual converged neutral, precursor, and cation SCFs. */
struct PSI_API GRACProvenance {
    double neutral_precursor_energy{};
    double cation_energy{};
    double homo_energy{};
    double ionization_potential{};
    double applied_shift{};
    std::string cation_reference;
    int cation_charge{};
    int cation_multiplicity{};
};

/**
 * Frozen single-thread response state with cloned electronic/functional/molecular/grid data.
 * The orbital BasisSet is deliberately retained by const alias under an explicit no-mutation
 * contract and rechecked before future use. Production compute must resolve exclusive ownership
 * across check/use; this alias is not a current response-success claim.
 */
class PSI_API FrozenResponseContext {
   public:
    static std::shared_ptr<FrozenResponseContext> create(
        const std::shared_ptr<Wavefunction>& grac_wfn,
        const std::shared_ptr<Wavefunction>& neutral_precursor_wfn,
        const std::shared_ptr<Wavefunction>& cation_wfn);

    const std::shared_ptr<const Matrix>& Ca() const { return Ca_; }
    const std::shared_ptr<const Matrix>& Cb() const { return Cb_; }
    const std::shared_ptr<const Vector>& epsilon_a() const { return epsilon_a_; }
    const std::shared_ptr<const Vector>& epsilon_b() const { return epsilon_b_; }
    const std::shared_ptr<const Vector>& occupation_a() const { return occupation_a_; }
    const std::shared_ptr<const Vector>& occupation_b() const { return occupation_b_; }
    const std::shared_ptr<const Matrix>& Da() const { return Da_; }
    const std::shared_ptr<const Matrix>& Db() const { return Db_; }
    double energy() const { return energy_; }
    const std::shared_ptr<const Molecule>& molecule() const { return molecule_; }
    const std::shared_ptr<const BasisSet>& basis() const { return basis_; }
    const std::shared_ptr<const SuperFunctional>& functional() const { return functional_; }
    const std::vector<SitePosition>& sites() const { return sites_; }
    const std::vector<double>& grid_points() const { return grid_points_; }
    const std::vector<double>& grid_weights() const { return grid_weights_; }
    const std::vector<FrozenGridBlock>& grid_blocks() const { return grid_blocks_; }
    const GRACProvenance& grac() const { return grac_; }
    const std::string& functional_name() const { return functional_name_; }
    const std::string& grac_x_name() const { return grac_x_name_; }
    const std::string& grac_c_name() const { return grac_c_name_; }
    std::size_t grid_point_count() const { return grid_weights_.size(); }
    /** Enforce the documented single-thread/no-mutation contract for the retained basis alias. */
    void verify_basis_unchanged() const;

   private:
    FrozenResponseContext(SharedMatrix Ca, SharedMatrix Cb, SharedVector epsilon_a,
                          SharedVector epsilon_b, SharedVector occupation_a,
                          SharedVector occupation_b, SharedMatrix Da, SharedMatrix Db,
                          double energy, std::shared_ptr<const Molecule> molecule,
                          std::shared_ptr<const BasisSet> basis,
                          std::shared_ptr<const BasisSetStructuralSnapshot> basis_snapshot,
                          std::shared_ptr<const SuperFunctional> functional,
                          std::vector<SitePosition> sites, std::vector<double> grid_points,
                          std::vector<double> grid_weights, std::vector<FrozenGridBlock> grid_blocks,
                          GRACProvenance grac, std::string functional_name,
                          std::string grac_x_name, std::string grac_c_name);

    std::shared_ptr<const Matrix> Ca_;
    std::shared_ptr<const Matrix> Cb_;
    std::shared_ptr<const Vector> epsilon_a_;
    std::shared_ptr<const Vector> epsilon_b_;
    std::shared_ptr<const Vector> occupation_a_;
    std::shared_ptr<const Vector> occupation_b_;
    std::shared_ptr<const Matrix> Da_;
    std::shared_ptr<const Matrix> Db_;
    double energy_{};
    std::shared_ptr<const Molecule> molecule_;
    // Deliberate alias: safe only under the single-thread/no-mutation contract checked before response.
    std::shared_ptr<const BasisSet> basis_;
    std::shared_ptr<const BasisSetStructuralSnapshot> basis_snapshot_;
    std::shared_ptr<const SuperFunctional> functional_;
    std::vector<SitePosition> sites_;
    std::vector<double> grid_points_;
    std::vector<double> grid_weights_;
    std::vector<FrozenGridBlock> grid_blocks_;
    GRACProvenance grac_{};
    std::string functional_name_;
    std::string grac_x_name_;
    std::string grac_c_name_;
};

/** Actual ISA data structurally bound to one exact frozen context and its ordered grid/sites. */
class PSI_API ISAWeights {
   public:
    static ISAWeights create(std::shared_ptr<const FrozenResponseContext> context,
                             std::vector<double> partition_weights);

    std::size_t point_count() const;
    std::size_t site_count() const;
    const std::vector<double>& partition_weights() const { return partition_weights_; }

   private:
    friend class ISAPolResponseProvider;
    ISAWeights(std::shared_ptr<const FrozenResponseContext> context,
               std::vector<double> partition_weights);

    std::shared_ptr<const FrozenResponseContext> context_;
    std::vector<double> partition_weights_;
};

/** Production response interface; no mutable Wavefunction is retained or revalidated. */
class PSI_API ISAPolResponseProvider {
   public:
    ISAPolResponseProvider(std::shared_ptr<const FrozenResponseContext> context,
                           ResponseKernel kernel, ISAWeights isa_weights);

    std::size_t expected_response_count(const FrequencyGrid& frequencies) const;
    std::vector<SitePairResponse> compute_isapol_response(const FrequencyGrid& frequencies) const;

   private:
    std::shared_ptr<const FrozenResponseContext> context_;
    ResponseKernel kernel_;
    ISAWeights isa_weights_;
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
    /** Deterministic diagnostics consumed only by the underscored math-test seam. */
    std::vector<L3WorkingMatrix> refined_pairs;
    std::vector<std::array<std::size_t, 2>> omitted_component_pairs;
    std::size_t omitted_transfer_count;
};

PSI_API Matrix lw_graph_operator(const BondGraph& graph);
PSI_API std::pair<Matrix, std::vector<double>> lw_graph_pseudoinverse(const BondGraph& graph);
PSI_API L3WorkingVector translate_l3_multipoles(const L3WorkingVector& source,
                                                 const SitePosition& source_minus_target);
/** Localize independently within every graph component; reject component-inconsistent flow. */
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
