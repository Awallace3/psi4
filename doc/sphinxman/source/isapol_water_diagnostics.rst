Explicit water comparison diagnostics
=====================================

``psi4.driver.procrouting.isapol_density_reference.finite_density_reference``
is an explicitly requested, supplied-array reference diagnostic. The module
imports NumPy and the standard library, not Psi4. It solves the finite model
``(J + lambda q q.T)c = b + lambda N q`` with fixed 80-digit arithmetic,
one J factorization and two right-hand sides, for at most 256 functions.
Inputs must be finite binary64 arrays; singular J, singular/near-cancelling
rank-one denominators and failed residual sanity guards are refused.

This is not a production solver selection or a density-model/default change.
It performs no symmetrization, clipping, charge normalization, hard-charge
replacement or quantum calculation. Its immutable binary64 coefficients and
reference/rounded residual diagnostics are not a general correct-rounding,
interval, field, kernel or property accuracy certificate. No production
kernel caller selects this diagnostic or the older direct finite helper
automatically. The older binary64 direct helper is not an accuracy remedy.

The portable native-only fixture and 16 offline tests are under
``tests/pytests/data_isapol/native_density_reference`` and
``test_isapol_density_reference.py``. They exercise analytic/refusal cases
and all 246 water coefficient bit patterns against the accepted rounded
110-digit reference, without quantum imports. Run from this checkout with
a fresh, nonexistent output path::

    PYTHON=/path/to/python3.13 bash tests/pytests/run_stable_density_offline.bash /new/output

The fixture is fixed-input evidence, not fresh validation of this assembled
stack's compiled response/property path. Earlier binary64 field differences
were not reevaluated with corrected coefficients.

A new matched-comparison procedure must label independent choices explicitly:

* C engine grid: ``RELATIVE-RADII SLATER`` (C-SLATER), not the old C-default;
* angular ordering: the frozen 230-entry correspondence, applied uniformly
  to all atom/radial blocks, never nearest-point or fitted matching;
* density model: plain, finite penalty, or hard charge, separately identified;
* numerical solve: original binary64, direct finite comparison, or explicit
  fixed-80 reference diagnostic, separately identified;
* refinement selection: legacy direct current-node cutoff, explicit subset,
  or explicitly retained static reference, separately identified.

Preserve the original ``water_df_variant_approved_006`` and all old HOLDs.
The 506 C-SLATER weight print-interval violations remain unresolved; grid
ordering agreement is not weight, field, property or CamCASP parity.
Never rerun a sealed lab harness into its historical output ID.

Refinement selection metadata (PFIT layer)
-----------------------------------------

The published direct-helper API ``refinement_model(...,
penalized_variables=...)`` is retained, including an empty explicit subset.
With no new keyword, no subset means current-node cutoff selection; a supplied
subset means exactly that subset, with the published current-node declaration
completeness check. Current tensors still supply all anchors/weights; declared
extras outside selection remain free. No existing call changes numerically.

An opt-in ``penalty_selection='static_reference'`` instead requires
``static_reference_tensors`` in the same site order/shapes/axes. Selection uses
the first site's strict ``abs(value)>cutoff`` scan within each COPY type;
current tensors still supply selected anchors, including small/zero values.
References under any other policy, a subset with explicit ``'current'`` or
``'static_reference'``, or ``'explicit_subset'`` without a subset are refused.
There is no silent precedence between two selection declarations.

Models gain ``penalty_selection``, ``static_reference_sha256`` and
``selection_sha256``. Legacy subset calls are labelled ``explicit_subset``;
their canonical subset hash is independent of order/frequency. Static hashes
snapshot all reference tensors as little-endian binary64 bytes, not site/frame
metadata; current ``anchor_sha256`` is separate. Reference policy provenance
adds these labels/hashes. Dataclass repr/equality and reference-mode provenance
identities therefore migrate; compare explicit scientific declarations, not
cross-version repr/hash identity, to assess numerical compatibility. Legacy
current/subset provenance strings are unchanged.
