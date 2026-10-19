/*
 * @BEGIN LICENSE
 *
 * Psi4: an open-source quantum chemistry software package
 *
 * Copyright (c) 2007-2026 The Psi4 Developers.
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

#ifndef PSI4_SRC_PSI4_LIBXDM_XDM_H
#define PSI4_SRC_PSI4_LIBXDM_XDM_H

#include "psi4/libmints/typedefs.h"
#include "psi4/pragma.h"

#include <memory>
#include <string>
#include <vector>

namespace psi {

class Wavefunction;
class Molecule;
class Matrix;

namespace xdm {

/// Integrated atomic properties from Hirshfeld-weighted density integration.
struct AtomicData {
    double mm1 = 0.0;     ///< Second moment of exchange-hole dipole, <d_xh^2>
    double mm2 = 0.0;     ///< Second moment of exchange-hole quadrupole
    double mm3 = 0.0;     ///< Second moment of exchange-hole octupole
    double vol = 0.0;     ///< Effective atomic volume
    double charge = 0.0;  ///< Integrated electron count (for verification)
};

/// Columns of the ``XDM ATOMIC PROPERTIES`` array, one row per centre (ghosts included).
enum XDMProperty {
    XDM_PROP_Z = 0,               ///< true atomic number (also for ghosts)
    XDM_PROP_REAL = 1,            ///< 1 for real atoms, 0 for ghosts (only real atoms enter the pair sum)
    XDM_PROP_CHARGE = 2,          ///< Hirshfeld-integrated electron count
    XDM_PROP_VOLUME = 3,          ///< Hirshfeld volume, <r^3>
    XDM_PROP_M1 = 4,              ///< <M1^2>
    XDM_PROP_M2 = 5,              ///< <M2^2>
    XDM_PROP_M3 = 6,              ///< <M3^2>
    XDM_PROP_POLARIZABILITY = 7,  ///< volume-scaled atomic polarizability
    XDM_NPROP = 8
};

/// XDM dispersion correction with Becke-Johnson damping.
///
/// Computes the exchange-hole dipole moment (XDM) dispersion energy from a
/// converged wavefunction. Unlike D1-D4 corrections, XDM requires the
/// electron density (not just molecular geometry).
///
/// Reference: A.D. Becke and E.R. Johnson, J. Chem. Phys. 127, 154108 (2007)
class PSI_API XDMDispersion {
   public:
    /// Construct with BJ damping parameters and functional name.
    /// @param a1             dimensionless BJ damping parameter
    /// @param a2_bohr        BJ damping parameter in bohr
    /// @param functional_name  DFT functional name (for free-atom volume lookup)
    XDMDispersion(double a1, double a2_bohr, const std::string& functional_name);

    /// Build from functional name with explicit a1 and a2 (a2 in angstrom, converted internally).
    static std::shared_ptr<XDMDispersion> build(const std::string& functional, double a1, double a2_angstrom);

    /// Compute XDM dispersion energy from a converged wavefunction.
    double compute_energy(std::shared_ptr<Wavefunction> wfn, double hf_fraction);

    /// Pairwise XDM energy from per-centre properties, without printing or setting variables.
    /// @param coords  (natom, 3) Cartesian coordinates [bohr]
    /// @param props   (natom, XDM_NPROP) properties as in ``XDM ATOMIC PROPERTIES``; only rows
    ///                flagged real enter the sum, so a counterpoise driver can add a partner
    ///                fragment's ghost-centre moments to a monomer and re-evaluate it.
    double energy_from_properties(SharedMatrix coords, SharedMatrix props) const;

    double a1() const { return a1_; }
    double a2() const { return a2_; }
    std::string functional_name() const { return functional_name_; }

   private:
    double a1_;
    double a2_;
    std::string functional_name_;

    /// Integrate Hirshfeld-weighted atomic properties from the wavefunction.
    std::vector<AtomicData> integrate_properties(std::shared_ptr<Wavefunction> wfn);

    /// Compute pairwise BJ-damped dispersion energy; stores the property and pair arrays on wfn.
    double pairwise_energy(std::shared_ptr<Wavefunction> wfn, const std::vector<AtomicData>& atoms, double hf_fraction);

    /// BJ-damped pair sum over the real rows of ``props``. A non-null ``wfn`` prints the pair table
    /// and stores the coefficient arrays on it.
    double pair_sum(const Matrix& coords, const Matrix& props, Wavefunction* wfn) const;
};

}  // namespace xdm
}  // namespace psi

#endif
