/* Psi4: Copyright (c) 2007-2026 The Psi4 Developers.
 * SPDX-License-Identifier: LGPL-3.0-only
 */
#ifndef PSI4_LIBISAPOL_EXPLICIT_BASIS_H
#define PSI4_LIBISAPOL_EXPLICIT_BASIS_H
#include <array>
#include <memory>
#include <vector>
namespace psi {
class Matrix;
namespace isapol {
/// Racah regular real multipoles through rank 4, ordered 00,10,11c,11s,...
std::vector<double> isa_regular_multipoles(int rank, const std::array<double,3>& displacement);
enum class IsaBasisRole { MolecularAux, AtomAux, Shape, Orbital };
enum class IsaBasisRepresentation { Cartesian, Spherical };
/// Zero-based centre; effective coefficients include ALL normalization factors.
struct IsaGaussianShell {
    int centre = 0;
    int l = 0;
    std::vector<double> exponents, coefficients;
};
/// Immutable owned snapshot. Coordinates in bohr; exponents in bohr^-2.
/// Shell order is retained; Cartesian GAMINT / spherical DALTON (p=x,y,z).
class IsaExplicitBasis {
   public:
    IsaExplicitBasis(IsaBasisRole role, IsaBasisRepresentation representation,
                     const std::vector<std::array<double, 3>>& centres,
                     const std::vector<IsaGaussianShell>& shells);
    int nfunction() const { return nfunction_; }
    int ncentre() const { return static_cast<int>(centres_.size()); }
    IsaBasisRole role() const { return role_; }
    /// Co-centred AtomAux/Shape metric, before damping/ridge. No grid exponent cap.
    /// All primitive pairs must be integrable under the selected weighting.
    std::shared_ptr<Matrix> overlap(double w_eps = 0.0, bool s_block_only = true) const;
    std::shared_ptr<Matrix> evaluate(const std::vector<std::array<double, 3>>& points) const;
    std::shared_ptr<Matrix> evaluate_screened(const std::vector<std::array<double, 3>>& points,
                                            const std::vector<int>& sites) const;
   private:
    friend class IsaShapeMap;
    friend class IsaAFitProvider;
    friend class IsaGaussianShape;
    friend class IsaAuxCoulomb;
    static const std::vector<std::array<int,3>>& cartesian_powers(int l);
    IsaBasisRole role_;
    IsaBasisRepresentation representation_;
    std::vector<std::array<double, 3>> centres_;
    std::vector<IsaGaussianShell> shells_;
    int nfunction_ = 0;
};
/// Validated shape-shell -> AtomAux-shell map. Indices are zero-based, no padding.
/// Each shape shell must exactly match its unique s-shell target (centre, ordered
/// exponents and effective coefficients). An explicit subset/permutation is allowed.
class IsaShapeMap {
   public:
    IsaShapeMap(const IsaExplicitBasis& atomic, const IsaExplicitBasis& shape,
                const std::vector<int>& shell_map);
    std::vector<int> function_indices() const { return columns_; }
    /// Select raw coefficients only: no normalization, DIIS, mixing or tail policy.
    std::vector<double> project(const std::vector<double>& atomic_coefficients) const;
   private:
    int atomic_nfunction_;
    std::vector<int> columns_;
};
/// Supplied molecular AUX expansion, NOT AO density or a native Drho-C fit.
/// The caller owns provenance (e.g. Drho-C). No clipping or charge rescaling.
class IsaFixedDensity {
   public:
    IsaFixedDensity(const IsaExplicitBasis& basis, const std::vector<double>& coefficients);
    /// Explicit zero-based unique active sites, no padding. Empty means all screened out.
    std::vector<double> evaluate(const std::vector<std::array<double, 3>>& points,
                                 const std::vector<int>& sites) const;
   private:
    IsaExplicitBasis basis_;
    std::vector<double> coefficients_;
};
struct IsaAFitOptions;
struct IsaAFitData;
struct IsaAFitResult;
/// Caller-owned sampling policy: shapes are already screened and tail-corrected.
/// Density sites are unique zero-based active sites for this entire point batch.
struct IsaAFitSamples {
    std::vector<std::array<double, 3>> points;
    std::vector<double> weights, shape, shape_sum, previous;
    std::vector<int> density_sites;
};
/// Owned explicit-input provider, not a native basis/DF recipe or ISA controller.
class IsaAFitProvider {
   public:
    IsaAFitProvider(const IsaExplicitBasis& atomic, const IsaFixedDensity& density);
    /// Fresh inspectable data; use the same W-Eps/s-block settings in a later solve.
    IsaAFitData assemble(const IsaAFitSamples& samples, const IsaAFitOptions& options) const;
    /// Assembly and frozen solve with identical options; no retained mutable samples.
    IsaAFitResult fit(const IsaAFitSamples& samples, const IsaAFitOptions& options) const;
   private:
    friend class IsaASweep;
    IsaExplicitBasis atomic_;
    IsaFixedDensity density_;
    std::array<double, 3> centre_;
    std::vector<int> angular_;
    std::vector<double> exponents_;
};
} }
#endif
