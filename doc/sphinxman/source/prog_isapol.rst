.. #
.. # @BEGIN LICENSE
.. #
.. # Psi4: an open-source quantum chemistry software package
.. #
.. # Copyright (c) 2007-2026 The Psi4 Developers.
.. #
.. # The copyrights for code used from other parties are included in
.. # the corresponding files.
.. #
.. # This file is part of Psi4.
.. #
.. # Psi4 is free software; you can redistribute it and/or modify
.. # it under the terms of the GNU Lesser General Public License as published by
.. # the Free Software Foundation, version 3.
.. #
.. # Psi4 is distributed in the hope that it will be useful,
.. # but WITHOUT ANY WARRANTY; without even the implied warranty of
.. # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
.. # GNU Lesser General Public License for more details.
.. #
.. # You should have received a copy of the GNU Lesser General Public License along
.. # with Psi4; if not, write to the Free Software Foundation, Inc.,
.. # 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
.. #
.. # @END LICENSE
.. #


.. _`sec:prog_isapol`:

ISA-Pol distribution building blocks (libisapol)
================================================

``libisapol`` ports parts of CamCASP 6.0 (A. J. Misquitta and A. J. Stone,
https://gitlab.com/anthonyjstone/camcasp) with permission. Most of it
reproduces CamCASP's arithmetic bit for bit, so the library builds with
``-ffp-contract=off``. That agreement was verified only on the local non-FMA
build; cross-platform bitwise identity is not claimed. This section covers the
building blocks used for distributed moments. No end-user property keyword uses them yet.
Units are bohr and atomic units throughout, on global Cartesian axes.

Explicit bases
   ``psi4.driver.procrouting.isapol_basis`` holds the immutable
   ``ShellRecipe``/``BasisRecipe`` descriptors, whose shell coefficients already
   include normalization. ``BasisRecipe.build(role)`` returns an owned
   ``psi4.core.IsaExplicitBasis``, Cartesian GAMINT or spherical DALTON,
   S through G. Its methods are ``evaluate``, ``evaluate_screened`` and the
   co-centred ``overlap``. ``adapt_main(wfn, caller_converged=True)`` converts
   a restricted closed-shell C1 wavefunction's MAIN basis and occupied orbitals
   into that convention and checks the result.
   It fits no density and renormalizes nothing.

Atom tables and integration grid
   ``psi4.core.IsaGrid`` (with ``IsaGridOptions``) is CamCASP's
   atom-centred Becke/Euler--MacLaurin/Lebedev grid. It uses unrotated
   Lebedev spheres, which come from ``lebedev_npoints_at_least`` and
   ``lebedev_sphere`` in ``libfock/cubature.h``. The shared Lebedev generator
   now rounds each product before subtracting (``sub_no_fma``). ``libfock`` is
   not built with ``-ffp-contract=off``, so on FMA-contracting builds this can
   move every Psi4 DFT/``DFTGrid`` Lebedev coordinate by up to 1 ulp relative to
   previous builds. The ``isapol_*`` element functions (Slater, Bondi, Grimme
   and covalent radii, Grimme C6, symbol) reproduce CamCASP's ``AtomProp`` table.

Distributed moments (Q)
   Q maps fitted-density coefficients of a declared response auxiliary basis to
   site multipoles. Rows are site-major real Racah components 00, 10, 11c, 11s,
   ..., so dipoles are ordered z, x, y. Q carries no electronic sign and no
   neutrality correction.

   * ``psi4.core.IsaPartitionedMultipoles`` integrates Q on
     caller-supplied quadrature. For each site the caller gives points, weights,
     a ``shape``, a ``shape_sum`` and ``auxiliary_sites`` (via
     ``IsaMultipoleSamples`` and ``IsaMultipoleSite``). ``auxiliary_sites`` lists
     the zero-based AUX centres to collocate and is required: an empty list
     deliberately gives all-zero columns. The stockholder ratio
     ``shape/shape_sum`` is formed natively, and points whose denominator is at
     or below ``denominator_cutoff`` are excluded and counted. This is
     independent of how the shapes were obtained.
   * ``isapol_df_multipoles.analytic_df_centre_multipoles`` produces CamCASP's
     DF-centre rule in closed form: every auxiliary function belongs wholly to
     its own centre.
   * ``isapol_distribution.DistributedMoments`` is the owned, immutable,
     validated Q contract. ``analytic_df_moments`` wraps the analytic DF-centre
     producer without changing any element. ``validate_for`` checks the site,
     rank and auxiliary identity. ``anchor_legs(C)`` is the only supported
     contraction, ``C @ Q.T``.

The ISA and MBIS partitions (density partitions, not orbital rotations),
response, Lamb--Wilkinson localization, point-response fitting and dispersion
are built on top of these blocks and are added separately.

Deferred to later stages
   These candidate APIs have no consumer here. Each is added, from candidate
   ``4189ded9cc8f319c8bc21bbe78960f0fab7e2eff``, with the stage that uses and
   tests it.

   * Response: the orbital constructor
     ``IsaPartitionedMultipoles(orbital, sites, provenance, denominator_cutoff,
     orbitals, nocc)``, which gives direct occupied-fast (``a*nocc+i``) OV
     columns with representation ``'direct_ov'``, and ``'direct_ov'`` as a
     supplied-Q representation. Also ``IsaExplicitBasis.screening_s_overlap``
     (CamCASP's signed ``screening_s_ovr`` shell surrogate for ALDA screening)
     and ``shell_layout``.
   * Point-response fitting: ``isapol_vdw_radius`` (``vdw_radius``), the
     double-precision ``MODULE radii`` Bondi table that the fit-point lattice
     reads. ``isapol_vdw_radius_bondi`` is the float32 ``AtomProp`` copy.
