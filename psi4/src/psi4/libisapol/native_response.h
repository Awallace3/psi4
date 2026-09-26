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
/** Owned restricted-C1 state admission without dense response construction.
 * Reuses the native occupation, finite-energy/gap, density, basis and
 * independently evaluated overlap checks. Caller convergence is a declaration,
 * not an SCF or correction-policy seal. No Wavefunction/Options is retained;
 * All getters return copies and the retained reconstructed basis is private.
 * Basis copies retain scientific centre/shell data but not comments, arbitrary
 * labels, variables, dummy centres or provenance metadata from the molecule.
 * Concurrent mutation during construction is unsupported.
 *
 * The state-only envelope is 64*nbf^2 doubles, 16 KiB per shell (bounded
 * primitives), 4 KiB per atom and 16 MiB fixed engine allowance, within max_bytes.
 * It excludes caller storage and is not a process-RSS guarantee. No nov^2
 * operators, ERIs, grids or frequency solver are constructed. Admission here
 * does not authorize any response backend's independent work/resource limits.
 */
class NativeRestrictedState {
 public:
    NativeRestrictedState(std::shared_ptr<Wavefunction> wfn, bool caller_converged,
                          std::size_t max_bytes);
    std::shared_ptr<Matrix> orbitals() const;
    std::shared_ptr<Vector> energies() const;
    std::shared_ptr<Matrix> density_alpha() const;
    std::shared_ptr<BasisSet> basis_snapshot() const;
    int nbf() const { return nbf_; }
    int nmo() const { return nmo_; }
    int nocc() const { return nocc_; }
    int nvir() const { return nvir_; }
    std::size_t nov() const { return nov_; }
    std::size_t planned_bytes() const { return planned_bytes_; }
 private:
    std::shared_ptr<BasisSet> basis_;
    std::shared_ptr<Matrix> c_, da_;
    std::shared_ptr<Vector> eps_;
    int nbf_ = 0, nmo_ = 0, nocc_ = 0, nvir_ = 0;
    std::size_t nov_ = 0, planned_bytes_ = 0;
};

} // namespace isapol
} // namespace psi
