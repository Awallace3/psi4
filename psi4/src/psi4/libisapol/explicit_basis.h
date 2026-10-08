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
    static const std::vector<std::array<int,3>>& cartesian_powers(int l);
    IsaBasisRole role_;
    IsaBasisRepresentation representation_;
    std::vector<std::array<double, 3>> centres_;
    std::vector<IsaGaussianShell> shells_;
    int nfunction_ = 0;
};
} }
#endif
