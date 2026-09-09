/*
 * @BEGIN LICENSE
 *
 * Psi4: an open-source quantum chemistry software package
 *
 * Copyright (c) 2007-2026 The Psi4 Developers.
 *
 * This file is part of Psi4.
 *
 * Psi4 is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * @END LICENSE
 */
#pragma once
#include <cstddef>
#include <memory>
#include <string>

namespace psi {
class Wavefunction;
class Matrix;
namespace isapol {
/** Immutable, eager, restricted C1 native point-charge OV coupling operators.
 *
 * W(t,p) = + integral phi_i(r) phi_a(r) / |r - R_p| dr, occupied fast
 * t = a*nocc+i, real AO/MO context, atomic units (Eh/e per unit charge).
 * This is the POSITIVE Coulomb kernel, i.e. exactly minus the charge-inclusive
 * electron electrostatic-potential operator that oeprop obtains from
 * ElectrostaticInt::compute(result, C). Both conventions are admissible for a
 * response leg because W enters the contraction twice; mixing them within one
 * leg is a sign error, so the single convention is fixed and published here.
 *
 * This class performs NO response solve, carries NO frequency, and applies NO
 * fitted auxiliary metric, charge model, multipole expansion, energy 1/2 or
 * bare electrostatics. The only integrals dropped are the shell pairs libint2
 * itself reports as precision zero; nothing here screens on distance, magnitude
 * or geometry. Diagnostics are recorded only: no point screening, no
 * distance repair, no symmetrization and no conditioning policy.
 *
 * The caller declares convergence; metadata, density and AO-overlap
 * orthonormality are checked, but no SCF/GRAC/grid provenance seal is
 * asserted. All getters return copies. No Wavefunction, BasisSet, Molecule or
 * Options object is retained; the supplied inputs must not be mutated
 * concurrently during construction.
 */
class IsaPointChargeOperators {
   public:
    IsaPointChargeOperators(std::shared_ptr<Wavefunction> wfn, bool caller_converged,
                            std::shared_ptr<Matrix> points_bohr, std::size_t max_bytes,
                            std::size_t max_points);
    /// (nocc*nvir) x npoint positive-kernel OV point-charge operators.
    std::shared_ptr<Matrix> operators() const;
    /// npoint x 3 owned copy of the accepted source points, in bohr.
    std::shared_ptr<Matrix> points() const;
    int nocc() const { return nocc_; }
    int nvir() const { return nvir_; }
    int ntransition() const { return nocc_ * nvir_; }
    int npoint() const { return npoint_; }
    std::string ov_order() const { return "occupied_fast: t=a*nocc+i"; }
    std::string representation() const { return "native_point_charge_ov_operators"; }
    std::string convention() const {
        return "W(t,p)=+int phi_i phi_a/|r-R_p| dr; atomic units; positive Coulomb kernel; "
               "minus the charge-inclusive oeprop electron ESP operator; no 1/2, no nuclear term";
    }
    /// Recorded geometry diagnostics. Never used to filter, screen or repair.
    double minimum_nuclear_distance_bohr() const { return min_nuclear_; }
    /// Smallest distance between distinct supplied points; 0 for a single point.
    double minimum_point_separation_bohr() const { return min_separation_; }
    double maximum_absolute_element() const { return max_element_; }
    std::size_t planned_bytes() const { return planned_bytes_; }

   private:
    std::shared_ptr<Matrix> w_, points_;
    int nocc_ = 0, nvir_ = 0, npoint_ = 0;
    double min_nuclear_ = 0., min_separation_ = 0., max_element_ = 0.;
    std::size_t planned_bytes_ = 0;
};
}  // namespace isapol
}  // namespace psi
