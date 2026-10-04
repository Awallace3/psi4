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
   context manager: enter (seal, state, preflight, private scratch),
   ``prepare(provider)``, ``solve(omega)`` per node, then ``release()``. The
   entry preflight mirrors the byte admissions of the frequency sweep (operator
   formation and every node for the reference; native admission, W and every
   node for FDDS) and the planned total work; ``prepare`` stages are admitted as
   they are reached, so a budget can still be refused during ``prepare``. A
   failed ``prepare`` is final: ``solve`` then refuses. Leaving
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
     source-derived estimates of the modeled tensor payload and dense/streamed
     operations, charged at admission (native has no counters). They are not
     measured, and not proven bounds. Libint integral generation has no
     arithmetic estimate.
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
     is a read-only integral input (e.g. ``FDDS_Monomer`` or ``IntegralFactory``),
     not a general ``BasisSet``: ``MintsHelper(basis)`` empties its ghost-centre
     molecule and then fails. Only that caller-owned copy is affected. Each call
     builds a new twin, and the runner uses its own private copy.
   * Every restricted C1 SCF now records a convergence seal (the stopping
     diagnostics and a SHA-256 of the final state, streamed without full-matrix
     copies). The seal changes no SCF arithmetic.

Point-response fitting (PFIT)
   PFIT refines supplied local polarizability anchors against a supplied
   point-to-point response, CamCASP's ``pfit``. Targets follow its sign
   convention, v_pq = -d(phi_induced at R_p)/d(q at R_q) in Eh/e^2 with no
   energy 1/2, packed at ``i*(i+1)//2 + j`` for every ``j <= i``. Every problem
   declares where its targets came from (``IsaPfitTargetOrigin``); the solver
   checks that declaration's form but cannot verify it.

   * ``psi4.core.FitPoints(molecule, FitPointsOptions)`` is CamCASP's
     ``RANDOM`` lattice: candidates drawn uniformly in a cube about the
     unweighted nuclear centroid and kept between ``lolim`` and ``hilim``
     van der Waals radii (defaults 2000 points, 2 and 4, seed 1). It draws
     from ``MaclarenRng``, CamCASP's ``sdprnd``/``dprand``, and reads the
     double-precision ``MODULE radii`` table (``isapol_vdw_radius``), not the
     float32 ``AtomProp`` copy (``isapol_vdw_radius_bondi``). Both the stream
     and the clouds reproduce CamCASP bit for bit on the local build.
   * ``psi4.core.isa_t_functions(rank, point, site, frame, damping=0)`` is one
     row of pfit's T matrix: the irregular solid harmonics
     (``isa_irregular_solid_harmonics``; rank 0 is 1/r) of the point in the
     site's local axes, in the Racah order above. ``frame`` follows
     ``isa_multipole_rotation``. Tang--Toennies damping
     (``isa_t_function_damping``) is optional and off by default.
   * ``isapol_refine.refinement_model(sites, anchor_tensors, ...)`` builds
     CamCASP's ``.pdef`` variable list and ``Penalties`` block: one set of
     variables per ``site_type`` (the COPY equivalence, read off the first site
     of the type), upper-triangle components whose reference anchor exceeds
     ``cutoff``, and penalty strengths from ``weight_type`` and
     ``weight_coefficient``. ``declared_variables`` replays a supplied variable
     list instead. ``refine`` solves one target densely (``core.isa_pfit_solve``,
     at most 512 points) and returns per-site refined tensors; components
     outside the model are exactly zero.
   * ``isapol_pfit_stream.refine_streamed(models, points, packed, ...)`` fits
     several targets (typically one per frequency) over one complete cloud.
     ``PackedDesignRows`` replays the shared design rows in bounded blocks, and
     ``core.isa_pfit_solve_rows_multi`` traverses them twice in total and
     refuses a second pass whose content differs. Each result equals the dense
     fit of its column up to rounding.
   * ``isapol_pfit_stream.fitted_point_targets(auxiliary, points, C)`` turns
     p x p fitted AUX coefficient responses (for example a bounded response
     node's ``target_response``) into packed targets -P^T C P, where
     ``IsaAuxCoulomb.point_potentials`` gives the exact AUX Coulomb potentials
     P at the points. Such targets are declared ``NativeFittedPointResponse``
     with ``fitted_density_coefficients`` and the AUX identity.
   * Resources: ``PackedDesignRows`` and ``fitted_point_targets`` check their
     planned numerical buffers against ``max_bytes`` before allocating, and
     the native solver checks its own against
     ``IsaPfitOptions.maximum_work_bytes``. These are plans, not measured RSS;
     caller-owned inputs, Libint workspace and interpreter overhead are not
     charged. A fit that is not ``Solved`` withholds its parameters, so
     ``refine`` and ``refine_streamed`` raise.
   * The fit is not iterated, and anchors are neither symmetrized nor repaired.
     ``copy_anchor_discrepancy`` reports, rather than repairs, COPY-equivalent
     sites whose local anchors disagree. A refined tensor is a different model
     from the unrefined LW tensor of the same site.

The ISA and MBIS partitions (density partitions, not orbital rotations) and
dispersion are built on top of these blocks and are added separately.

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
     result records, and the refined-tensor reduction
     ``isapol_refine.isotropic_scalars``.
