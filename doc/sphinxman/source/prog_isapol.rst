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
     its own centre. Its provenance string reads "no stockholder fixed point";
     the candidate's "no ISA-A fixed point" was reworded with the same meaning.
   * ``isapol_distribution.DistributedMoments`` is the owned, immutable,
     validated Q contract. ``analytic_df_moments`` wraps the analytic DF-centre
     producer without changing any element. ``validate_for`` checks the site,
     rank and auxiliary identity. ``anchor_legs(C)`` is the only supported
     contraction, ``C @ Q.T``.

LW localization, multipole transforms and frequency grid
   These consume supplied tensors only. No response, partition or external
   CamCASP/ORIENT program is involved. LW is the Lillestolen--Wheatley
   localization (Lillestolen and Wheatley, *J. Phys. Chem. A* **111**, 11141
   (2007); CamCASP user's guide section 8.2.1, where LS is the older
   Le Sueur--Stone method). The implementation follows ORIENT's LW routine
   and reproduces its output.

   * ``psi4.core.isa_multipole_translation(rank, d)`` returns T(d) with
     R(x+d) = T(d) R(x), where d is the source-minus-target displacement in
     bohr. ``psi4.core.isa_multipole_rotation(rank, F)`` returns D(F) with
     R(F x) = D(F) R(x), where the columns of F are the local axes in global
     coordinates; F must be proper orthogonal to 1e-12. Both work in real
     Racah order, ranks 0 to 4.
   * ``psi4.core.isa_localize_lw`` localizes one frequency of a supplied
     ordered site-pair response, given as N*N blocks (16x16 for rank 3 or
     25x25 for rank 4), over an explicit zero-based bond graph. The
     frequency must be finite and nonnegative. It returns an owned
     ``IsaLocalizedResponse``, whose local tensors exclude rank 0.
     ``residual_tolerance`` (1e-6) is a postcondition gate on the residuals
     LW controls, not an iteration criterion. ``input_sum_rule_tolerance``
     gates the supplied charge-flow sum-rule defect: negative inherits the
     residual tolerance, positive sets its own threshold, and infinity
     reports the defect without gating it. Nothing is repaired, symmetrized
     or retried. ``isa_lw_graph_math`` exposes the graph Laplacian and its
     pseudoinverse.

     .. warning:: LW localization is not rotationally covariant. Rotating the
        whole molecule (origins and tensors together) can change the local LW
        tensors and the rank 2 and higher isotropic polarizabilities that
        later feed C8 and C10. ORIENT behaves the same way. Rank-1
        isotropic values were unchanged in the cases sampled, but no
        orientation independence of C6 is guaranteed. Passing the
        ``production`` residual gate means the input and residual checks
        passed; it is not a guarantee of physical accuracy or of rotational
        invariance. The multipole translations and rotations themselves are
        exact.
   * ``psi4.driver.procrouting.isapol_lw.supplied_nonlocal_properties`` is the
     typed driver. It requires an explicit ``Provenance``, an explicit
     truncation for rank-4 input, and strictly increasing imaginary-axis frequencies;
     frames are optional. It returns an immutable ``LocalProperties``. The
     residual policies are:

     - ``production`` (the default): one combined 1e-6 gate.
     - ``reported_input_sum_rule``: 1e-6 on what LW controls. The supplied
       sum-rule defect is measured and reported, not gated.

     ``localization_rank_limit`` (1 to 4, default 3) declares the
     localization rank. Different limits give different models, but the
     results agree exactly on the ranks they share.
   * ``psi4.core.CasimirGrid(n_freq, omega0=0.3)`` is CamCASP's
     Gauss--Legendre imaginary-frequency quadrature. ``n_freq`` is even, from 2
     to 10. Index 0 is the static point (zero frequency, zero weight).
     ``cp_weight`` includes the mapping Jacobian and the 1/(2 pi) of the
     Casimir--Polder integral. The ISA-Pol protocols use ``omega0=0.5``.

Bounded DF response
   ``psi4.driver.procrouting.isapol_bounded_response.BoundedResponse`` computes
   the plain-DF PBE0 response of one sealed, restricted, closed-shell C1
   wavefunction at the declared quadrature nodes, under explicit
   ``BoundedResources`` (bytes, arithmetic work, cumulative I/O). Use it as a
   context manager: enter (seal, state, complete preflight, private scratch),
   ``prepare(provider)``, ``solve(omega)`` per node, then ``release()``. Leaving
   the context removes the private scratch directory, on failure too. Each
   ``FrequencyResponse`` holds owned arrays: ``target_response`` (p x p, in
   response-AUX coefficient space) and ``nonlocal_response`` (site-major real
   Racah, rank 4). ``provider(ledger)`` returns ``DistributedMoments``; the
   default is the analytic DF-centre Q.

   * ``response='reference_h2h1'`` (the default) is the reference model:
     constrained OV fits, the fitted-density Slater/PW92 ALDA kernel, H1/H2 with
     reciprocity/stability gates and the H2H1 residual gate.
   * ``response='native_fdds'`` must be explicit and needs
     ``fdds=NativeFDDSOptions(nthread, disk_bytes, subalgo)``. Each node is solved
     by the native declared-AUX ``FDDS_Monomer`` (:ref:`sec:sapt`) on the Psi4
     AO-order orbitals, with W = J_d + 0.75 K and x_alpha = 0.25. K is the same
     kernel, grid, smoothing and shell mask as the reference. J_d is the native
     instance's own metric; its maximum difference from ``IsaAuxCoulomb.metric()``
     is recorded in ``provenance``. The target is chi and the nonlocal response
     is -Q chi Q^T. Both are symmetrized, but negative or positive
     semidefiniteness is not enforced. ``residual`` is native ``solve_residual``.
     Q, the native memory and W stay reserved for the whole sweep; native
     scratch is capped by ``disk_bytes`` and lives in the private directory. The
     cumulative native I/O and arithmetic charged to the budget are
     source-derived conservative estimates (native has no counters). Libint
     integral generation has no arithmetic estimate.
   * Before any response, an instance whose ``metric_lu_rcond`` is below
     ``FDDS_METRIC_RCOND_CUTOFF`` (1e-14) is refused, and the value and cutoff are
     reported. This heuristic separated the sampled epsilon-level metrics
     (<= 1.13e-15, 10-100% wrong) from the declared recipes (>= 5.77e-14). It is
     not an accuracy or forward-error bound. The native dual Dyson admission
     (> 2e-13) is unchanged, and its refusals propagate.
   * There is no molecular accuracy proof for the FDDS route. The toy recipes
     (PBE0/sto-3g, Cartesian aug-cc-pVTZ-RI) are admitted at every node. The
     declared molecular water recipe (PBE0/aug-cc-pVTZ) passes the metric guard,
     but every node is refused by the dual Dyson admission on the MKL paths
     tested.
   * For the downstream LW stage the runner exposes ``lw_residual_policy``:
     ``production`` for the reference, and ``reported_input_sum_rule`` for FDDS.
     With FDDS, ``local_charge`` is report-only and the rank-0 remainder does not
     enter local tensors, PFIT or C6. The LW orientation dependence above
     applies to both.
   * ``IsaAuxCoulomb.native_auxiliary()`` returns fresh copies of the raw
     Cartesian twin and the declared map T (J_d = T J_raw T^T). The returned basis
     is a read-only integral input; ``MintsHelper(basis)`` empties its ghost-centre
     molecule.
   * Every restricted C1 SCF now records a convergence seal (the stopping
     diagnostics and a SHA-256 of the final state, streamed without full-matrix
     copies). The seal changes no SCF arithmetic.

The ISA and MBIS partitions (density partitions, not orbital rotations),
point-response fitting and dispersion are built on top of these blocks and
are added separately.

Deferred to later stages
   These candidate APIs have no consumer here. Each is added, from candidate
   ``4189ded9cc8f319c8bc21bbe78960f0fab7e2eff``, with the stage that uses and
   tests it.

   * Response: the orbital constructor
     ``IsaPartitionedMultipoles(orbital, sites, provenance, denominator_cutoff,
     orbitals, nocc)``, which gives direct occupied-fast (``a*nocc+i``) OV
     columns with representation ``'direct_ov'``, and the unsampled supplied-Q
     constructor ``IsaPartitionedMultipoles(values, sites, representation,
     provenance)``, which takes a finished Q (``'fitted_density_coefficients'``
     or ``'direct_ov'`` columns) and sites without samples, as given.
   * Dispersion: the ``isapol_lw.Coefficient`` and ``DispersionPair``
     result records.
   * Point-response fitting: ``isapol_vdw_radius`` (``vdw_radius``), the
     double-precision ``MODULE radii`` Bondi table that the fit-point lattice
     reads. ``isapol_vdw_radius_bondi`` is the float32 ``AtomProp`` copy.
