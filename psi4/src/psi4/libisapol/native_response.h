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
// Adapted from libmints/atomic_polarizability.{h,cc}, camcasp_psi4
// 5449bd1a01c73f45c307b36b006e264c1e43b994. Adaptation copyright 2026 Psi4 Developers.
#pragma once
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace psi {
class BasisSet;
class Wavefunction;
class Matrix;
class Vector;
namespace isapol {
/** Immutable, eager, restricted C1 native full-OV producer; no frequency solver.
 * Real AO/MO context, atomic units, t=a*nocc+i (occupied fast).
 * V=(ia|jb), X=(ij|ab), Y=(ib|aj), L=int w fxc(rho) ia jb.
 * H1=Delta+4V-a(X+Y)+4bL; H2=Delta-a(X-Y).
 * The caller declares convergence. Metadata/density/orthonormality are checked,
 * but neither SCF convergence nor a GRAC/grid provenance seal is asserted.
 * All getters return copies. No mutable Wavefunction or Options is retained.
 */
class NativeResponseProvider {
 public:
    NativeResponseProvider(std::shared_ptr<Wavefunction> wfn, bool caller_converged,
                           const std::string& kernel, double exact_exchange,
                           double local_scale, std::shared_ptr<Matrix> grid, double density_cutoff,
                           std::size_t max_bytes, std::size_t max_nov);
    std::shared_ptr<Matrix> h1() const;
    std::shared_ptr<Matrix> h2() const;
    std::shared_ptr<Matrix> coulomb() const;
    std::shared_ptr<Matrix> exchange_direct() const;
    std::shared_ptr<Matrix> exchange_transpose() const;
    std::shared_ptr<Matrix> local_primitive() const;
    std::shared_ptr<Matrix> orbitals() const;
    std::shared_ptr<Vector> energies() const;
    std::shared_ptr<Matrix> density_alpha() const;
    int nocc() const { return nocc_; }
    int nvir() const { return nvir_; }
    const std::string& kernel() const { return kernel_; }
    double exact_exchange() const { return a_; }
    double local_scale() const { return b_; }
    double density_cutoff() const { return cutoff_; }
    std::size_t planned_bytes() const { return planned_bytes_; }
 private:
    std::shared_ptr<BasisSet> basis_; // private deep reconstruction, never exposed
    std::shared_ptr<Matrix> c_, da_, grid_, v_, x_, y_, local_, h1_, h2_;
    std::shared_ptr<Vector> eps_;
    int nocc_ = 0, nvir_ = 0;
    std::string kernel_;
    double a_, b_, cutoff_;
    std::size_t planned_bytes_ = 0;
};

/** Exact per-row contribution bound for the ALDA local primitive, so a caller
 * can screen its own quadrature rows instead of coarsening the quadrature.
 *
 * Row p adds factor(p)*tr_p tr_p^T to L, with factor(p)=w(p)*fxc(p) and
 * tr_p(t)=phi_i(p)*phi_a(p). Because tr_p is the outer product of the occupied
 * and virtual orbital values at that point,
 *     ||factor(p) tr_p tr_p^T||_F = |factor(p)| * o(p) * u(p),
 *     o(p)=sum_i phi_i(p)^2,  u(p)=sum_a phi_a(p)^2,
 * and every single element obeys the same bound. value(p) is exactly that
 * number, so omitting any set of rows perturbs L by at most the sum of their
 * values in BOTH maxabs and Frobenius norm. Rows the primitive itself already
 * skips (rho<cutoff, or zero weight) have value exactly 0, so threshold 0 is
 * lossless pruning with a bound of exactly zero.
 *
 * Retained rows keep their original coordinates, weights and order: no weight
 * renormalization, no radial/angular reduction, no AO screening. This class has
 * its OWN gate on grid_rows*nbf*nmo collocation work and deliberately carries no
 * nov^2 term; it therefore never authorizes NativeResponseProvider's nov^2 ALDA
 * limit, which still applies in full to whatever row subset is finally passed.
 */
class IsaAldaGridScreen {
 public:
    IsaAldaGridScreen(std::shared_ptr<Wavefunction> wfn, bool caller_converged,
                      const std::string& kernel, std::shared_ptr<Matrix> grid,
                      double density_cutoff, std::size_t max_bytes);
    /// Per input row, in input order.
    std::shared_ptr<Vector> values() const;
    /// Zero-based indices of rows with value strictly above threshold.
    std::vector<int> retained_rows(double threshold) const;
    /// Those rows as [x,y,z,weight], original values, original order.
    std::shared_ptr<Matrix> retained(double threshold) const;
    /// Sum of the omitted rows' values: an upper bound on maxabs and Frobenius
    /// deviation of the resulting local primitive. Summed in ascending row order.
    double omitted_bound(double threshold) const;
    int omitted_count(double threshold) const;
    /// Smallest threshold retaining at most max_rows rows; 0 when the exact-zero
    /// rows alone already suffice. Ties retain fewer rows, never more.
    double threshold_for_rows(int max_rows) const;
    double total() const { return total_; }
    double maximum() const { return maximum_; }
    int rows() const { return np_; }
    int exact_zero_rows() const { return zeros_; }
    const std::string& kernel() const { return kernel_; }
    double density_cutoff() const { return cutoff_; }
    std::size_t planned_bytes() const { return planned_bytes_; }
 private:
    void check_threshold(double threshold) const;
    std::shared_ptr<Matrix> grid_;
    std::vector<double> value_;
    std::vector<double> sorted_; // descending, for threshold_for_rows only
    std::string kernel_;
    int np_ = 0, zeros_ = 0;
    double cutoff_ = 0., total_ = 0., maximum_ = 0.;
    std::size_t planned_bytes_ = 0;
};
} // namespace isapol
} // namespace psi
