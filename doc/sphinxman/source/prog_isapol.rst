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
``-ffp-contract=off``. This section covers the building blocks
used for distributed moments. No end-user property keyword uses them yet.
Units are bohr and atomic units throughout, on global Cartesian axes.

Explicit bases
   ``psi4.driver.procrouting.isapol_basis`` holds the immutable
   ``ShellRecipe``/``BasisRecipe`` descriptors, whose shell coefficients already
   include normalization. ``BasisRecipe.build(role)`` returns an owned
   ``psi4.core.IsaExplicitBasis``, Cartesian GAMINT or spherical DALTON,
   S through G. Its methods include ``evaluate``, ``evaluate_screened``, the
   co-centred ``overlap`` and ``shell_layout``. ``adapt_main(wfn,
   caller_converged=True)`` converts a restricted closed-shell C1 wavefunction's
   MAIN basis and occupied orbitals into that convention and checks the result.
   It fits no density and renormalizes nothing.

Atom tables and integration grid
   ``psi4.core.IsaGrid`` (with ``IsaGridOptions``) is CamCASP's
   atom-centred Becke/Euler--MacLaurin/Lebedev grid. It uses unrotated
   Lebedev spheres, which come from ``lebedev_npoints_at_least`` and
   ``lebedev_sphere`` in ``libfock/cubature.h``. The ``isapol_*`` element
   functions (Slater, Bondi, Grimme and covalent radii, Grimme C6, symbol)
   reproduce CamCASP's ``AtomProp`` table.

Distributed moments (Q)
   Q maps fitted-density coefficients of a declared response auxiliary basis to
   site multipoles. Rows are site-major real Racah components 00, 10, 11c, 11s,
   ..., so dipoles are ordered z, x, y. Q carries no electronic sign and no
   neutrality correction.

   * ``psi4.core.IsaPartitionedMultipoles`` integrates Q on
     caller-supplied quadrature. For each site the caller gives points, weights,
     a ``shape`` and a ``shape_sum`` (via ``IsaMultipoleSamples`` and
     ``IsaMultipoleSite``). The stockholder ratio ``shape/shape_sum`` is formed
     natively, and points whose denominator is at or below ``denominator_cutoff``
     are excluded and counted. This is independent of how the shapes were
     obtained.
   * ``isapol_df_multipoles.df_centre_multipoles`` produces CamCASP's DF-centre
     rule, in which every auxiliary function belongs wholly to its own centre.
     ``'df_centre_analytic'`` is the closed form; ``'df_centre_grid'`` uses the
     caller's quadrature.
   * ``isapol_distribution.DistributedMoments`` is the owned, immutable,
     validated Q contract. ``analytic_df_moments`` wraps the analytic DF-centre
     producer without changing any element. ``validate_for`` checks the site,
     rank and auxiliary identity. ``anchor_legs(C)`` is the only supported
     contraction, ``C @ Q.T``.

The ISA and MBIS partitions (density partitions, not orbital rotations),
response, Lamb--Wilkinson localization, point-response fitting and dispersion
are built on top of these blocks and are added separately.
