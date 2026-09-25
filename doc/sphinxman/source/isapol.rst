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

|PSIfour| can partition an already converged closed-shell density into
iterated-stockholder atoms (ISA-A), form the frequency-dependent
distributed polarizability of each atomic site, and contract pairs of
those site polarizabilities over a Casimir |w--w| Polder quadrature into
site |w--w| site dispersion coefficients :math:`C_6`, :math:`C_8`,
:math:`C_{10}` and, at site rank 4, :math:`C_{12}`.  The arrangement
follows the ISA-Pol protocol of CamCASP, but this is a native
implementation: it reads no reference file, writes no scratch interchange
file, and never runs an SCF, a response iteration or an asymptotic
correction of its own inside a property request.

The whole feature is requested through :ref:`oeprop <sec:oeprop>` on a
wavefunction the caller has already converged, and the large labeled
tensors it produces are returned through an owned Python record rather
than through padded QCVariables.

.. warning:: Every keyword below **names a model, not a tolerance.**  The
   grid resolutions, the auxiliary basis, the site rank, the localization
   limit, the multipole distribution rule, the refinement lattice and its
   weight coefficient are all model declarations: two values are two
   different models, and numbers recorded under any two of them may not be
   quoted as agreeing with one another, averaged together, or treated as a
   converged and an unconverged version of the same quantity.  An inherited
   default is not a declaration |w---w| if a number matters, set the keyword
   that produced it explicitly and record its value alongside the number.

.. note:: The shipped recipe |globals__atomic_property_recipe| =
   ``GENERATED_JKFIT_ISA_A`` is a deliberately modest, self-contained
   demonstration recipe for hydrogen and oxygen, built from |PSIfour|'s own
   shipped JKFIT sets and closed-form even-tempered radial functions.  It is
   **not** an alias for, or a reproduction of, a modern CamCASP ISA-Pol basis
   declaration, and it should not be described as one.


Requesting a computation
^^^^^^^^^^^^^^^^^^^^^^^^

The five task names below are passed to :py:func:`~psi4.driver.oeprop`
exactly like any other one-electron property.  Each name is a separate
request: asking for the partition alone never forms a response, and asking
for the unrefined coefficients never runs the refinement.

.. _`table:isapol_tasks`:

.. table:: Native atomic property tasks

   +-----------------------------------+--------------------------------------------------------------------------------+
   | Keyword                           | What it produces                                                               |
   +===================================+================================================================================+
   | ATOMIC_PARTITION                  | The converged Drho-C ISA-A density partition only. No response is formed.      |
   +-----------------------------------+--------------------------------------------------------------------------------+
   | ATOMIC_POLARIZABILITIES           | Static (:math:`\omega = 0`) distributed site polarizabilities.                 |
   +-----------------------------------+--------------------------------------------------------------------------------+
   | ATOMIC_DISPERSION                 | Imaginary-frequency site polarizabilities on the 11-node Casimir |w--w| Polder |
   |                                   | quadrature, and the site-pair :math:`C_n` contracted from them.                |
   +-----------------------------------+--------------------------------------------------------------------------------+
   | ATOMIC_REFINED_POLARIZABILITIES   | The above, plus the reference protocol's PFIT refinement of the localized site |
   |                                   | tensors against the molecular response on a random point cloud.                |
   +-----------------------------------+--------------------------------------------------------------------------------+
   | ATOMIC_REFINED_DISPERSION         | The refined site-pair :math:`C_n`. A **different model** from                  |
   |                                   | ATOMIC_DISPERSION, not a better-converged version of it.                       |
   +-----------------------------------+--------------------------------------------------------------------------------+

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
      basis            cc-pvdz
      reference        rks
      puream           true
      e_convergence    1e-10
      d_convergence    1e-10
    }

    e, wfn = energy('pbe0', return_wfn=True)
    oeprop(wfn, 'ATOMIC_DISPERSION')

The incoming wavefunction must be an actual converged, restricted,
closed-shell, :math:`C_1` wavefunction; a symmetrized or open-shell
wavefunction, or one whose successful-SCF seal is absent, is refused
rather than silently accepted.  Only hydrogen and oxygen nuclei are
supported by the shipped recipe.


Retrieving the results
^^^^^^^^^^^^^^^^^^^^^^

:py:func:`~psi4.driver.oeprop` itself returns ``None``, as it does for
every other property.  The full record of the request |w---w| the
partition, the distributed response, the dispersion coefficients, the
refinement, and the per-stage numerical diagnostics |w---w| is taken from
the wavefunction afterwards:

.. autofunction:: psi4.atomic_property_result(wfn)
   :noindex:

::

    r = psi4.atomic_property_result(wfn)
    r.partition.converged                  # ISA-A convergence flag
    r.atomic_scalars                       # (nfreq, nsite, 3) isotropic responses by rank
    r.properties.frequencies               # the quadrature nodes actually used
    r.dispersion.pairs                     # unrefined site-pair coefficients
    r.refined_dispersion                   # refined coefficients, or None

Taking the accessor transfers that request's result to the caller, and a
later request never mutates an earlier record.  ``dispersion`` and
``refined_dispersion`` are deliberately separate accessors, because the
refined and the unrefined coefficient of the same order on the same site
pair are numbers about two different models and neither is a correction to
the other.

Scalar summaries are additionally published as wavefunction variables, to
be read with ``wfn.variable(...)``.  Site labels are the recipe's own
(``O1``, ``H1``, ``H2`` for water):

.. _`table:isapol_variables`:

.. table:: Representative wavefunction variables

   +----------------------------------------------------+-------------------------------------------------------------+
   | Variable                                           | Meaning                                                     |
   +====================================================+=============================================================+
   | :samp:`ISA CONVERGED`, :samp:`ISA MAX DELTA`       | ISA-A partition convergence                                 |
   +----------------------------------------------------+-------------------------------------------------------------+
   | :samp:`ISA DRHO RELATIVE RESIDUAL`                 | Drho-C density-fit residual                                 |
   +----------------------------------------------------+-------------------------------------------------------------+
   | :samp:`ATOM {label} DIPOLE POLARIZABILITY`         | Static isotropic rank-1 site polarizability                 |
   +----------------------------------------------------+-------------------------------------------------------------+
   | :samp:`ATOMIC POLARIZABILITY SITE SUM ISOTROPIC`   | Trace/3 of the summed site dipole polarizabilities          |
   +----------------------------------------------------+-------------------------------------------------------------+
   | :samp:`ATOMIC DISPERSION C{n} {a} {b}`             | Unrefined site-pair coefficient of order *n*                |
   +----------------------------------------------------+-------------------------------------------------------------+
   | :samp:`ATOMIC REFINED DISPERSION C{n} {a} {b}`     | Refined site-pair coefficient, under its own name           |
   +----------------------------------------------------+-------------------------------------------------------------+

A site-pair coefficient whose rank pairs are not all available at the
declared site rank is published under the same name with ``INCOMPLETE``
appended, rather than being silently reported as though it were complete.
At the default |globals__atomic_multipole_rank| of 3 this affects the
highest orders, and :math:`C_{12}` does not exist at all: the rank pairs
(1,4) and (4,1) that create it require a declared site rank of 4.

The narrative written to the output file is controlled by
|globals__atomic_property_print| |w---w| 0 silent, 1 each stage with its
parameters and the final tables, 2 iteration tables and per-stage
diagnostics, 3 per-frequency detail.  It is reporting only: it selects
nothing and changes no number.


Declared asymptotic correction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Distributed response at long range is sensitive to the asymptotic behavior
of the exchange-correlation potential, so the property request requires the
caller to **declare** what correction the incoming orbitals already carry.
|globals__atomic_scf_asymptotic_correction| is an *acceptance* policy, never
a producer: no value of it runs an SCF or adds a response kernel.  Its three
values are three different models.

``NONE``
   Admits unmodified canonical orbitals, and refuses any orbitals that carry
   a declared correction or a GRAC state.

``FIXED_GRAC``
   Admits orbitals that were converged with |PSIfour|'s own gradient-regulated
   asymptotic correction at the shift declared in
   |globals__atomic_scf_expected_grac_shift|.

``DECLARED_MULTPOLE_AC``
   Admits orbitals produced by the multipole-expansion asymptotic correction
   described by the ``ATOMIC_AC_*`` keywords.  Those orbitals do not come into
   existence by themselves: they must be produced first, by the explicit
   producer below, and the declaration read at the property request must be the
   same one that produced them.

.. autofunction:: psi4.atomic_asymptotic_correction(wfn)
   :noindex:

.. warning:: :py:func:`~psi4.driver.atomic_asymptotic_correction` **mutates its
   argument in place.**  The wavefunction's orbitals, density, Fock matrix,
   eigenvalues and energy are replaced, and its successful-SCF seal is
   deliberately invalidated, because the resulting state is no longer the one
   SCF converged: the reported energy is the plain functional evaluated at the
   corrected density, which lies above the SCF minimum and is not variational.
   Do not quote that energy as an SCF energy.


PFIT refinement
^^^^^^^^^^^^^^^

The two ``ATOMIC_REFINED_*`` tasks run the reference protocol's PFIT stage on
top of the localized site tensors: the rank-restricted site parameters are
fitted to the molecular response evaluated on a random point cloud in a shell
around the molecule, with each rank-1 parameter held to its unrefined anchor by
a penalty.  Its inputs are declarations throughout, and two of them deserve
particular care:

* |globals__atomic_refinement_points| and |globals__atomic_refinement_seed|
  define the point cloud itself, draw for draw.  The cloud is a reproducible
  *model input*, not a sample size to be converged, so a run at a different
  count or seed is a differently declared model whose numbers may not be
  averaged with or compared against another's.

* |globals__atomic_refinement_weight_coefficient| scales how strongly each
  refined rank-1 variable is held to its anchor, and therefore how
  underdetermined the remaining variables are.  The corresponding default in
  CamCASP itself changed between releases, and this coefficient alone shifts
  site-pair :math:`C_6` by tens of percent.  Record it with every refined
  number; it is not a convergence knob.

|globals__atomic_refinement_rank_limit| and
|globals__atomic_refinement_hydrogen_rank_limit| are separate per-element
declarations, and neither may exceed
|globals__atomic_localization_rank_limit|: a variable cannot be refined at a
rank the local tensors were never localized at.

Refined coefficients are published under their own variable names and returned
through their own accessor, so a refined and an unrefined number can never
overwrite or be mistaken for one another.

Explicit fitted-density targets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Without the explicit ``atomic_backend='BOUNDED_DF'`` selection described below,
the public ``ATOMIC_REFINED_*`` tasks use direct-OV point-charge
targets from the original context provider's operators, not from an optional
rebuilt propagator. Do not interpret that default as rebuilt-propagator target
equivalence. A separate Python producer accepts an explicitly selected native
fitted-AUX calculation:

.. code-block:: python

   from psi4.driver.procrouting.isapol_native_fitted_point_response import (
       native_fitted_point_response,
   )
   targets = native_fitted_point_response(
       target_properties, wfn, points_bohr,
       charge_penalty=1000.0, metric_damping=0.0,
   )

Here ``target_properties`` is the result of ``native_properties`` with
``response_basis='fitted_auxiliary'``. The required penalty and damping must
match that calculation exactly; the producer does not refit. If the anchors
use a different damping, compute the target calculation separately.

The producer evaluates exact AUX Coulomb potentials and contracts
:math:`v=-B^T C_{\mathrm{AUX}} B` at every retained frequency. It uses the
retained coefficient responses, including any declared propagator rebuild,
not the original context provider's operators. Targets carry the distinct
``NativeFittedPointResponse`` PFIT origin and an AUX identity. Neither this
origin nor the fitted representation certifies agreement with CamCASP.

The Python refinement orchestrator can select that target calculation
separately from the accepted localization used as anchors:

.. code-block:: python

   from psi4.driver.procrouting.isapol_native_refinement import native_refinement
   refined = native_refinement(
       anchor_properties, wfn,
       site_types=("O", "H", "H"), rank_limits={"O": 2, "H": 1},
       fitted_target_properties=target_properties,
       target_charge_penalty=1000.0, target_metric_damping=0.0,
   )

Target and anchor results must share the exact wavefunction state, response
policy and frequency grid. Their fit damping may differ. The anchor
localization must pass its production gates; target localization is not an
operand of the fit and need not have passed. Supplying fit declarations
without a target calculation is refused. Omitting all three new arguments
preserves the direct-OV target route.


Explicit bounded DF-centre workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The opt-in Python entry point
``psi4.driver.procrouting.isapol_bounded.bounded_properties`` connects native
response, production LW localization, complete-cloud streamed PFIT and refined
isotropic dispersion. It is a separate explicitly declared model, not another
spelling of the ISA-A or ``ATOMIC_REFINED_*`` defaults. No existing legacy
admission gate is widened. Carbon stockholder defaults are neither required nor
invented: this route explicitly uses the analytic DF-centre distribution.

For the explicit water and benzene COPY/reflection models, the wavefunction-first
shortcut constructs the AUX basis, local axes, quadrature, response grid and fit
declarations:

.. code-block:: python

   # wfn must come from a fresh successful restricted C1 PBE0 SCF.
   # A GRAC shift attached to that SCF is followed automatically.
   psi4.oeprop(wfn, 'ATOMIC_REFINED_POLARIZABILITIES', 'ATOMIC_REFINED_DISPERSION',
               atomic_backend='BOUNDED_DF', preset='water')  # or 'benzene'
   result = psi4.atomic_property_result(wfn)
   tensors, dispersion = result.refined_tensors, result.dispersion

This explicit backend returns an owned ``BoundedProperties`` record through the
accessor; ``oeprop`` itself still returns ``None``. Only the two refined tasks
are supported, each running the common full chain. Ordinary and legacy atomic
tasks cannot be mixed into this request. Omitting ``atomic_backend`` preserves
the legacy route. No hidden SCF or automatic reference comparison is performed.

The presets require O,H,H atom order for water, and cyclic C1..C6 followed by
their H1..H6 partners for benzene. Coordinates come from the live wavefunction.
A preset declares only molecule topology: bond frames, bonds and COPY-equivalent
site types. These constraints are not inferred from arbitrary distorted
geometries. Use explicit ``sites`` and ``bonds`` kwargs to change them.

Every numerical setting is shared by both presets and is the CamCASP default:

* Cartesian ``aug-cc-pVTZ-RI`` AUX and rank-4 distributed response.
* A 100/200 radial/spherical response grid, rounded up to a Lebedev order as
  CamCASP does.
* ALDA FD smoothing with rho-epsilon 1e-8, F-max 1000, FD-delta 0.01 and
  FD-alpha 1.0, and a 1e-8 kernel-integral cutoff.
* A random 2000-point fit lattice between 2 and 4 vdW radii, with seed 1.
* The ``localize.py`` refinement defaults: weight type 3, coefficient 1e-3,
  cutoff 1e-4 and rank limit 2 for every site, hydrogen included.
* Fit variables derived from that cutoff on the first site of each type at the
  static node and reused at every frequency. This matches how CamCASP
  ``process`` writes a missing ``.pdef`` on the first call.
* 10-plus-static Gauss-Legendre Casimir nodes at beta 0.5, and ``max_order`` 10.

The SCF correction defaults to ``AUTO``. As CamCASP attaches GRAC only for a
positive IP+HOMO shift, ``AUTO`` selects ``FIXED_GRAC`` at the shift attached to
the wavefunction's functional, and ``NONE`` otherwise; it never computes a
shift. The declared 17/16-variable, weight 4/1e-5, L2/H1 models used for
reference parity are not defaults. Request them with ``weight_type``,
``weight_coefficient``, ``hydrogen_rank_limit`` and ``declared_variables``.

Override precedence is explicit kwargs, then explicitly changed Psi4 options,
then preset defaults. Unchanged legacy defaults do not overwrite a preset.
For example, ``psi4.set_options({'atomic_refinement_points': 600})`` changes the
point count, while ``npoints=500`` on the call takes precedence. The supported
option mapping is ``isapol_bounded_oeprop.OPTION_MAP``: response radial/spherical
points; refinement points, seed, bounds, weight type/coefficient and cutoff;
localization and refinement ranks; AUX basis; OV charge penalty/anchor damping;
SCF correction; and verbosity. An explicitly conflicting distribution, response
basis, localization policy or distributed rank is refused.

Additional kwargs accept ``auxiliary_recipe``, ``response_grid``, ``quadrature``,
``lattice_options``, ``smoothing``, ``shell_cutoff``, ``resources``, ``max_order``,
``scratch_directory`` and ``log``. Object kwargs replace the corresponding
generated object; scalar generation options then have no effect on that object.
The default numeric budget is the Psi4 memory setting (``psi4.set_memory``);
work and I/O default to 6e12 operations and 64 GiB. None of these are
process-RSS limits. Changed model inputs require a newly
matched reference before making a parity claim. Unknown kwargs are rejected.

Its current scientific scope is restricted C1 canonical PBE0, with either
``NONE`` or ``FIXED_GRAC`` correction, plain Coulomb-DF two-electron operators,
and a Slater/PW92 ALDA kernel evaluated on the plain fitted density. The kernel
and point targets use the eta-zero constrained fit. Localization anchors use
the separately declared ``anchor_metric_damping``. Every response frequency
solves the original :math:`H_2 H_1 + \omega^2 I` equation; no symmetric-part
replacement, charge repair or rank reduction is used to pass a gate.

The caller supplies a live successful SCF wavefunction, an explicit
``BasisRecipe`` for AUX, ``RefinementSite`` objects (including local frames and
COPY types), bonds, the full response grid, a ``Quadrature``, a
``KernelSmoothing`` declaration, fit-point options and fit parameters. Geometry
and AUX/site centres must agree exactly, in bohr and in atom order. A serialized
wavefunction without a current successful-SCF seal is not admitted. The driver
does not read CamCASP outputs, run an external executable, derive a GRAC shift,
or manufacture convergence evidence.

Stage reporting is enabled by default in the Psi4 output file. Pass a
``psi4.driver.procrouting.isapol_logging.StageLog`` as ``log`` to customize the
writer or verbosity (``StageLog(0)`` suppresses the stage narrative; resource
admission messages remain in the native output). Reports include stage timings,
input dimensions, stability checks, per-frequency response/localization
residuals, PFIT status and final resource totals. A writer may tee the narrative
to a terminal and a separate log file. These reports do not alter numerical
policies or convergence decisions.

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

The bounded route currently permits at most 2000 fit points, 64 sites and 64
fit variables, subject to its stricter numerical/work admission checks. It
includes every packed point pair and both solver passes. Its private
disk-staged operands are created in a fresh temporary directory, hash-checked
when read and removed on success or failure. ``scratch_directory`` optionally
selects the parent directory; it is never an existing checkpoint to resume.
Only small local/refined results and diagnostics are retained in the result.

``BoundedResources`` is an explicit cumulative opt-in ceiling on planned
numerical storage (caller-sized; the oeprop presets use the Psi4 memory
setting), arithmetic work (at most 6 trillion units) and checkpoint I/O (at
most 64 GiB). Array lifetimes, retained output allowances and LW
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
keyword group; their complete, authoritative descriptions, types and defaults
are in the options appendix, :ref:`apdx:options_c_module`, and are not restated
here.  The groups are:

* **Recipe and partition** |w---w| |globals__atomic_property_recipe|,
  |globals__atomic_property_auxiliary_basis|, |globals__partition_scheme|,
  |globals__atomic_property_radial_points|,
  |globals__atomic_property_spherical_points|.  Only
  |globals__partition_scheme| ``= ISA_A`` has a native continuous-partition
  adapter here; a request under any other value is refused rather than
  approximated by ISA-A.

* **Response** |w---w| |globals__atomic_response_basis|,
  |globals__atomic_response_propagator|,
  |globals__atomic_response_algorithm|,
  |globals__atomic_response_radial_points|,
  |globals__atomic_response_spherical_points|,
  |globals__atomic_ov_charge_penalty|, |globals__atomic_ov_metric_damping|.

* **Distribution and rank** |w---w| |globals__atomic_multipole_distribution|,
  |globals__atomic_multipole_rank|,
  |globals__atomic_response_localization|,
  |globals__atomic_localization_rank_limit|.

* **Declared asymptotic correction** |w---w|
  |globals__atomic_scf_asymptotic_correction|,
  |globals__atomic_scf_expected_grac_shift|, and the ``ATOMIC_AC_*`` family
  that describes the multipole correction itself.

* **Refinement** |w---w| the ``ATOMIC_REFINEMENT_*`` family.

* **Reporting** |w---w| |globals__atomic_property_print|.

Explicit bond-defined local frames
---------------------------------

The lower-level ``isapol_geometry.frames_from_axis_pairs(origins, declarations)``
helper constructs local-to-global column frames ``[x, y, z]``. Each declaration
is ``(site, (z_from, z_to), (x_from, x_to))`` with zero-based site indices.
Every site must be declared exactly once; neither bonds nor symmetry are
inferred. The declared x direction is projected perpendicular to z.
Coincident endpoints and numerically collinear directions (sine of the angle
at most 64 times float64 epsilon) are rejected without fallback axes.

Pass these frames explicitly to the native or supplied-nonlocal property API.
The supplied-local file reader also accepts
``site z from A to B x from C to D`` in its narrow axes grammar, in addition
to the existing global-Z form. Sites without axes still require explicit
global-frame declarations in the manifest.

Frame support alone does not enable native benzene calculations through the
legacy ISA-A driver: carbon partition policy and its response/refinement
resource limits remain unchanged. The explicit bounded DF-centre workflow
above is a separate opt-in calculation with its own complete declarations.

Independent model declarations
------------------------------

Two distinctions are easy to confuse and are genuinely independent:

* The ISA quadrature (``ATOMIC_PROPERTY_*_POINTS``), the response quadrature
  (``ATOMIC_RESPONSE_*_POINTS``) and the SCF quadrature
  (|scf__dft_radial_points|, |scf__dft_spherical_points|) are three
  separate grids with three separate declarations.  None of them inherits from
  another.

* |globals__atomic_multipole_distribution| chooses the *rule* by which the
  molecular response is distributed onto sites |w---w| the ISA-A stockholder
  shape partition, or CamCASP's ``DistPolAlgorithm = DF`` rule, which charges
  each auxiliary fitting function wholly to the center it sits on and forms no
  stockholder weight at all.  These answer different questions, and site
  multipoles, site polarizabilities or dispersion coefficients from the two
  rules may never be quoted as agreeing or disagreeing with each other.
