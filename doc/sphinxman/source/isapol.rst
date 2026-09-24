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

The public ``ATOMIC_REFINED_*`` tasks still use direct-OV point-charge
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

Frame support alone does not enable native benzene calculations: carbon
partition policy and response/refinement resource limits remain unchanged.

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
