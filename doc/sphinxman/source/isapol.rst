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


.. include:: autodoc_abbr_options_c.rst

.. index::
   single: ISA-Pol
   single: polarizability; distributed
   single: dispersion; distributed coefficients
   pair: ISA-Pol; theory

.. _`sec:isapol`:

Distributed Polarizabilities and Dispersion Coefficients |w---w| ISA-Pol
========================================================================

|PSIfour| can form the frequency-dependent distributed polarizability of each
atomic site of an already converged closed-shell PBE0 wavefunction, localize
it with the Le Sueur |w--w| Stone (LW) procedure, refine the local site tensors
against the molecular point-to-point response (PFIT), and contract pairs of
refined site polarizabilities over a Casimir |w--w| Polder quadrature into
site |w--w| site dispersion coefficients.  The arrangement follows the ISA-Pol
protocol of CamCASP 6.0, but this is a native implementation: it reads no
reference file or external program. A property request computes its own
response from the supplied live wavefunction, but does not run SCF or apply an
asymptotic correction to that wavefunction.

The default distributed response uses the analytic DF-centre rule: each
auxiliary fitting function is charged wholly to its centre. Explicit
``distribution='isa'`` instead runs native ISA-A from this wavefunction's
density, with a caller-declared partition recipe and integration grid.
Neither route imports external orbitals, partition products or response data.

.. warning:: Every setting below **names a model, not a tolerance.**  The
   auxiliary basis, the response grid, the kernel smoothing, the localization
   and refinement ranks, the refinement lattice and its weight coefficient are
   all model declarations: two values are two different models, and numbers
   recorded under any two of them may not be quoted as agreeing with one
   another, averaged together, or treated as a converged and an unconverged
   version of the same quantity.  If a number matters, record the full
   declaration alongside it.


Requesting a computation
^^^^^^^^^^^^^^^^^^^^^^^^

The refined tasks are passed to :py:func:`~psi4.driver.oeprop` together
with an explicit backend and a molecule preset.  They share one full chain;
any other ``ATOMIC_*`` request, or one without ``atomic_backend='BOUNDED_DF'``,
is refused.

.. _`table:isapol_tasks`:

.. table:: Atomic property tasks

   +-----------------------------------+--------------------------------------------------------------------------------+
   | Keyword                           | What it produces                                                               |
   +===================================+================================================================================+
   | ATOMIC_REFINED_POLARIZABILITIES   | PFIT-refined local site polarizabilities at every Casimir |w--w| Polder node.   |
   +-----------------------------------+--------------------------------------------------------------------------------+
   | ATOMIC_REFINED_DISPERSION         | The refined isotropic site-pair :math:`C_n` contracted from them.              |
   +-----------------------------------+--------------------------------------------------------------------------------+

The additional opt-in task ``ATOMIC_REFINED_RECOUPLED_DISPERSION`` contracts
the full PFIT-refined tensors into local-axis :math:`C_n(t,u,J)` (including
odd orders), without dimer placement. Combined requests execute response,
LW and PFIT only once. Ordinary requests do not construct a recoupled model.

A minimal |PSIfour| input::

    molecule water {
    0 1
    O 0.0 0.0 0.0
    H -1.45365196 0.0 -1.12168732
    H  1.45365196 0.0 -1.12168732
    units bohr
    symmetry c1
    no_com
    no_reorient
    }

    set {
      basis            aug-cc-pvtz
      reference        rks
      e_convergence    1e-10
      d_convergence    1e-10
    }

    e, wfn = energy('pbe0', return_wfn=True)
    oeprop(wfn, 'ATOMIC_REFINED_POLARIZABILITIES', 'ATOMIC_REFINED_DISPERSION',
           atomic_backend='BOUNDED_DF', preset='water')   # or 'benzene'

The incoming wavefunction must come from a fresh successful restricted,
closed-shell, :math:`C_1` PBE0 SCF; a symmetrized or open-shell wavefunction,
or one whose successful-SCF seal is absent or stale, is refused rather than
silently accepted.

The presets require O,H,H atom order for water, and cyclic C1..C6 followed by
their H1..H6 partners for benzene.  Coordinates come from the live
wavefunction.  A preset declares only molecule topology: bond frames, bonds and
COPY-equivalent site types.  These constraints are not inferred from arbitrary
distorted geometries; use explicit ``sites`` and ``bonds`` keyword arguments to
change them.


Retrieving the results
^^^^^^^^^^^^^^^^^^^^^^

:py:func:`~psi4.driver.oeprop` itself returns ``None``, as it does for every
other property, and this route publishes no QCVariables.  The owned
``BoundedProperties`` record is taken from the wavefunction afterwards:

.. autofunction:: psi4.atomic_property_result(wfn)
   :noindex:

::

    r = psi4.atomic_property_result(wfn)
    r.frequencies          # Casimir-Polder nodes actually used
    r.local_tensors        # unrefined LW local tensors, per node and site
    r.refined_tensors      # PFIT-refined local tensors, per node and site
    r.dispersion.pairs     # individually labelled refined site-pair C_n
    r.diagnostics          # per-node response/localization/PFIT diagnostics
    r.resources            # the cumulative resource ledger

Local-axis recoupled dispersion (explicit opt-in)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    psi4.oeprop(wfn, 'ATOMIC_REFINED_POLARIZABILITIES',
                'ATOMIC_REFINED_DISPERSION', 'ATOMIC_REFINED_RECOUPLED_DISPERSION',
                atomic_backend='BOUNDED_DF', preset='water')
    r = psi4.atomic_property_result(wfn)
    d = r.recoupled_dispersion  # None unless explicitly requested
    d.frequencies, d.cp_weights  # CP includes Jacobian and 1/(2*pi), once
    d.pair_labels              # ordered labelled site pairs, self pairs included
    d.model.source.sites       # owned labels, origins, frames, ranks, full tensors
    d.model.sites              # complex recoupled rank-pair blocks and t labels
    d.pairs[0].coefficients    # order, one-based t/u, J, value
    d.pairs[0].coverage        # original core rank-quadruple inventories
    d.declared_model_coverage  # additional per-pair/order declared-rank inventory

``r.dispersion`` remains isotropic, unchanged even on combined requests.
Both ``r.local_tensors`` and ``r.refined_tensors`` retain full tensors, not
traces. Components are real Racah/no-Condon--Shortley
``00,10,11c,11s,20,...`` (dipoles z,x,y); frames are proper local-to-global
Cartesian columns, not rotations applied by the recoupling adapter. The core
source tensors omit only the identically zero charge ``00`` row/column.
Nonzero charge response, rank-zero-only sites and any failure of exact
reciprocity are refused, not repaired. Signed symmetric tensors are allowed.
Every frequency retains its own tensor, with zero static CP weight and positive
dynamic weights. Origins are bohr, imaginary frequencies atomic units, tensor
rank-(l,l') blocks have units :math:`a_0^{l+l'+1}`, and coefficients
:math:`E_h a_0^n`. The adapter uses only this run's native PFIT results.

``t`` and ``u`` are one-based real angular-component indices, labelled
``00,10,11c,11s,...`` in the recoupled blocks; ``J`` is angular momentum.
These are **not** oriented scalar coefficients or :math:`-C_n/R^n` energies.
The scalar ``t=u=1,J=0`` C6/C8 equals the existing isotropic coefficient with
no extra prefactor, but this identity is not anisotropic completeness.

Completeness is reported separately for each ordered site pair and order:
``coverage`` preserves ``included_rank_quadruples``,
``missing_table_rank_quadruples``, ``missing_unrestricted_rank_quadruples``,
``table_complete`` and ``unrestricted_complete`` from the core.
``declared_model_coverage`` separately reports ``declared_model_complete`` and
``missing_declared_rank_quadruples`` within the actual per-site rank limits.
The core tables omit rank pairs (3,4), (4,3), (4,4); these are not zero physics.
Rank-2 models are not unrestricted-complete for anisotropic C8/C10. No partial
coefficient is relabelled full, and coefficient zeros do not certify coverage.
The optional conversion/contraction is charged to the shared bounded ledger,
in addition to the unchanged native core ceilings.

Refined and unrefined tensors remain distinct under ``refined_tensors`` and
``local_tensors``.  The narrative written to the output file is controlled by
|globals__atomic_property_print| |w---w| 0 silent, 1 each stage with its
parameters and the final tables, 2 iteration tables and per-stage diagnostics,
3 per-frequency detail.  It is reporting only and changes no number.


Preset settings
^^^^^^^^^^^^^^^

Every numerical setting is shared by both presets and is the CamCASP default:

* Cartesian ``aug-cc-pVTZ-RI`` AUX and rank-4 distributed response.
* A 100/200 radial/spherical response grid, rounded up to a Lebedev order as
  CamCASP does.
* ALDA FD smoothing with rho-epsilon 1e-8, F-max 1000, FD-delta 0.01 and
  FD-alpha 1.0, and a 1e-8 kernel-integral cutoff.
* A constrained transition-density fit with charge penalty 1000; the LW anchor
  legs additionally use off-centre metric damping 0.0005.
* A random 2000-point fit lattice between 2 and 4 vdW radii, with seed 1.
* The ``localize.py`` refinement defaults: weight type 3, coefficient 1e-3,
  cutoff 1e-4 and rank limit 2 for every site, hydrogen included.
* Fit variables derived from that cutoff on the first site of each type at the
  static node and reused at every frequency.  This matches how CamCASP
  ``process`` writes a missing ``.pdef`` on the first call.
* 10-plus-static Gauss |w--w| Legendre Casimir nodes at beta 0.5, and
  ``max_order`` 10.

The SCF correction defaults to ``AUTO``.  As CamCASP attaches GRAC only for a
positive IP+HOMO shift, ``AUTO`` selects ``FIXED_GRAC`` at the shift attached to
the wavefunction's functional, and ``NONE`` otherwise; it never computes a
shift.  The declared 17/16-variable, weight 4/1e-5, L2/H1 models used for
reference parity are not defaults.  Request them with ``weight_type``,
``weight_coefficient``, ``hydrogen_rank_limit`` and ``declared_variables``.

Override precedence is explicit keyword arguments, then explicitly changed Psi4
options, then preset values.  Options left at their defaults never overwrite a
preset.  For example, ``psi4.set_options({'atomic_refinement_points': 600})``
changes the point count, while ``npoints=500`` on the call takes precedence.
The supported option mapping is ``isapol_bounded_oeprop.OPTION_MAP``; the
keywords are listed below.

Additional keyword arguments accept ``auxiliary_recipe``, ``response_grid``,
``quadrature``, ``lattice_options``, ``smoothing``, ``shell_cutoff``,
``resources``, ``max_order``, ``scratch_directory`` and ``log``.  Object
arguments replace the corresponding generated object; scalar generation options
then have no effect on that object.  Unknown keyword arguments are rejected.
The explicit distribution/response declarations are ``distribution``,
``partition_recipe``, ``partition_grid``, ``response`` and ``fdds``.
``max_refinement_parameters`` opts into a declared allowance up to 72 without
changing the default of 64.


Explicit native ISA workflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The installed package contains a declared water regression recipe constructed
only from its own basis data. It is not a general ISA default or a
CamCASP-matched recipe: its tails use an absolute 1.5-bohr cutoff, not a
Bragg-Slater multiplier. The following small acceptance calculation owns its SCF,
partition, response, localization and fit:

.. code-block:: python

    import numpy as np
    import psi4
    from psi4.driver.procrouting.isapol_recipes import native_water_recipe
    from psi4.driver.procrouting.isapol_native import Quadrature

    mol = psi4.geometry("""
    0 1
    O 0 0 0
    H -1.45365196 0 -1.12168732
    H  1.45365196 0 -1.12168732
    units bohr
    symmetry c1
    no_com
    no_reorient
    """)
    psi4.set_options({"basis": "aug-cc-pvtz", "reference": "rks",
                      "scf_type": "pk", "puream": True})
    energy, wfn = psi4.energy("pbe0", molecule=mol, return_wfn=True)
    recipe = native_water_recipe(mol)
    options = psi4.core.IsaGridOptions()
    options.radial_points, options.spherical_points = 200, 590
    grid = psi4.core.IsaGrid(mol.clone(), options)
    qgrid = np.column_stack((grid.x(), grid.y(), grid.z(), grid.w()))
    psi4.oeprop(
        wfn, "ATOMIC_REFINED_DISPERSION", atomic_backend="BOUNDED_DF",
        preset="water", distribution="isa", partition_recipe=recipe,
        partition_grid=qgrid, auxiliary_recipe=recipe.auxiliary,
        quadrature=Quadrature.from_casimir(psi4.core.CasimirGrid(2, .5)),
        npoints=64, rank_limit=1, hydrogen_rank_limit=1,
        localization_rank_limit=2, max_order=6, scf_correction="NONE",
        weight_type=4, weight_coefficient=1e-5)
    result = psi4.atomic_property_result(wfn)

This is a small C6 acceptance model, not production C8/C10 or quadrature/grid
convergence evidence. Production ranks, quadrature, fitting lattice and
resource choices must be declared and validated separately.

``response='reference_h2h1'`` names a formulation, not a replay source: its
integrals, fitted-density kernel and response operands are generated from the
current wavefunction. ``response='native_fdds'`` is a distinct opt-in model with
its own native stability/dual-Dyson admission. Molecular recipes which fail
that admission are refused; changing the backend or loosening its gate to
obtain a plot does not validate the original model.


SCF asymptotic correction
^^^^^^^^^^^^^^^^^^^^^^^^^

Distributed response at long range is sensitive to the asymptotic behavior of
the exchange-correlation potential, so the request declares what correction the
incoming orbitals already carry.  |globals__atomic_scf_asymptotic_correction|
is an *acceptance* policy, never a producer: no value of it runs an SCF or adds
a response kernel.

``NONE``
   Admits unmodified canonical orbitals, and refuses any GRAC state.

``FIXED_GRAC``
   Admits orbitals that were converged with |PSIfour|'s own gradient-regulated
   asymptotic correction at the shift declared in
   |globals__atomic_scf_expected_grac_shift| (or, under ``AUTO``, the shift
   attached to the SCF functional).

The Fermi |w--w| Amaldi multipole / Tozer |w--w| Handy correction described by
the ``ATOMIC_AC_*`` keywords has its own explicit producer, whose policy name is
``DECLARED_MULTPOLE_AC``.  The bounded route does not admit its orbitals.

.. autofunction:: psi4.atomic_asymptotic_correction(wfn)
   :noindex:

.. warning:: :py:func:`~psi4.driver.atomic_asymptotic_correction` **mutates its
   argument in place.**  The wavefunction's orbitals, density, Fock matrix,
   eigenvalues and energy are replaced, and its successful-SCF seal is
   deliberately invalidated, because the resulting state is no longer the one
   SCF converged: the reported energy is the plain functional evaluated at the
   corrected density, which lies above the SCF minimum and is not variational.
   Do not quote that energy as an SCF energy.


Explicit bounded DF-centre workflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The presets are a shortcut to the Python entry point
``psi4.driver.procrouting.isapol_bounded.bounded_properties``, which connects
native response, production LW localization, complete-cloud streamed PFIT and
refined isotropic dispersion.  Its scientific scope is restricted C1 canonical
PBE0 with either ``NONE`` or ``FIXED_GRAC`` correction, plain Coulomb-DF
two-electron operators, and a Slater/PW92 ALDA kernel evaluated on the plain
fitted density.  The kernel and point targets use the eta-zero constrained fit;
localization anchors use the separately declared ``anchor_metric_damping``.
Every response frequency solves the original :math:`H_2 H_1 + \omega^2 I`
equation; no symmetric-part replacement, charge repair or rank reduction is
used to pass a gate.

The caller supplies a live successful SCF wavefunction, an explicit
``BasisRecipe`` for AUX, ``RefinementSite`` objects (including local frames and
COPY types), bonds, the full response grid, a ``Quadrature``, a
``KernelSmoothing`` declaration, fit-point options and fit parameters.
Geometry and AUX/site centres must agree exactly, in bohr and in atom order.
The driver does not read CamCASP outputs, run an external executable, derive a
GRAC shift, or manufacture convergence evidence.

Stage reporting is enabled by default in the Psi4 output file.  Pass a
``psi4.driver.procrouting.isapol_logging.StageLog`` as ``log`` to customize the
writer or verbosity (``StageLog(0)`` suppresses the stage narrative; resource
admission messages remain in the native output).  Reports include stage
timings, input dimensions, stability checks, per-frequency
response/localization residuals, PFIT status and final resource totals.

For an already converged ``wfn`` and those explicit declarations:

.. code-block:: python

   from psi4 import core
   from psi4.driver.procrouting.isapol_bounded import (
       bounded_properties, BoundedResources,
   )
   from psi4.driver.procrouting.isapol_native import Quadrature
   from psi4.driver.procrouting.isapol_native_propagator import KernelSmoothing

   lattice = core.FitPointsOptions()
   lattice.npoints, lattice.seed = 2000, 1
   lattice.lolim, lattice.hilim = 2.0, 4.0  # multiples of vdW radii
   result = bounded_properties(
       wfn, auxiliary_recipe,
       caller_converged=True, distribution="df_centre_analytic",
       sites=sites, bonds=bonds,
       quadrature=Quadrature.from_casimir(core.CasimirGrid(10, 0.5)),
       response_grid=response_grid,  # float64 (npoint,4): x,y,z,weight
       smoothing=KernelSmoothing(1e-8, 400.0, 0.1, 0.1, "FD"),
       shell_cutoff=1e-8, charge_penalty=1000.0,
       anchor_metric_damping=0.0005,
       lattice_options=lattice, localization_rank_limit=2,
       weight_type=4, weight_coefficient=1e-5, cutoff=1e-4,
       declared_variables=declared_variables,
       resources=BoundedResources(
           max_bytes=int(core.get_memory()),
           max_work=6_000_000_000_000,
           max_io_bytes=64 * 1024**3,
       ),
       scf_correction="FIXED_GRAC", expected_grac_shift=declared_shift,
       max_order=10,
   )
   refined = result.refined_tensors  # node, site; local-axis tensors
   pairs = result.dispersion.pairs  # individually labelled C_n coefficients

The numerical settings above are a declaration, not transferable defaults or a
claim of agreement for an arbitrary molecule. In particular, different kernel
smoothing settings define different calculations. If variables are declared
explicitly, use the same complete variable inventory at every node; the cutoff
still decides which of those variables carry anchor penalties. Refined and
unrefined tensors remain distinct under ``refined_tensors`` and
``local_tensors``.

The bounded route permits at most 2000 fit points and 64 sites, with 64
fit variables by default and an explicit allowance up to 72, subject to its
stricter numerical/work admission checks. It
includes every packed point pair and both solver passes. Its private
disk-staged operands are created in a fresh temporary directory, hash-checked
when read and removed on success or failure. ``scratch_directory`` optionally
selects the parent directory; it is never an existing checkpoint to resume.
Only small local/refined results and diagnostics are retained in the result.

``BoundedResources`` is an explicit cumulative opt-in ceiling on planned
numerical storage (caller-sized; the oeprop presets use the Psi4 memory
setting), arithmetic work (at most 6 trillion units in the default
``capacity='standard'``, or 100 trillion in explicitly selected
``capacity='large'``) and checkpoint I/O (at most 64 GiB). Array lifetimes,
retained output allowances and LW
workspace reservations are charged together; work and I/O are not reset per
frequency or block. These are not process-RSS or elapsed-time guarantees:
caller-owned wavefunction/basis metadata and implementation-specific
BLAS/Libint workspaces are outside the explicit numerical plan. The same Psi4 memory
setting also sizes SCF. Insufficient resources, changed checkpoints,
unstable/nonreciprocal operators, response residuals, production localization
failures and unsolved fits all raise instead of returning an accepted result.

The result records state/AUX/grid/lattice identities, correction and model
declarations, per-node response/localization diagnostics, the resource ledger
and the refined dispersion model. Completion alone does not certify reference
agreement; comparisons must match the full declarations and test each required
atomic/pair quantity independently.


Keywords
^^^^^^^^

All of the feature's keywords begin with ``ATOMIC_`` and live in the global
keyword group; their descriptions, types and defaults are in the options
appendix, :ref:`apdx:options_c_module`.  The bounded route reads each one only
when it has been explicitly set.  The groups are:

* **Auxiliary basis and response grid** |w---w|
  |globals__atomic_property_auxiliary_basis|,
  |globals__atomic_response_radial_points|,
  |globals__atomic_response_spherical_points|.

* **Transition-density fit** |w---w| |globals__atomic_ov_charge_penalty|,
  |globals__atomic_ov_metric_damping|.

* **Localization** |w---w| |globals__atomic_localization_rank_limit|.

* **Refinement** |w---w| the ``ATOMIC_REFINEMENT_*`` family: lattice points,
  seed and bounds, weight type and coefficient, cutoff, and the site and
  hydrogen rank limits.

* **SCF asymptotic correction** |w---w|
  |globals__atomic_scf_asymptotic_correction|,
  |globals__atomic_scf_expected_grac_shift|, and the ``ATOMIC_AC_*`` family
  read by :py:func:`~psi4.driver.atomic_asymptotic_correction`.

* **Reporting** |w---w| |globals__atomic_property_print|.
