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
} // namespace isapol
} // namespace psi
