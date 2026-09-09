# Native fixed-GRAC SCF-input acceptance

This is an **opt-in prerequisite**, not a GRAC response kernel or matched
CamCASP protocol. The generated H/O cc-pVDZ demonstration remains no-GRAC by
default. Its recipe, quadratures, rank policy, numerical operations and strict
LW `1e-6` postcondition are unchanged.

## Declaration and supported profile

Public `psi4.oeprop` atomic tasks use two new global options:

- `ATOMIC_SCF_ASYMPTOTIC_CORRECTION`: `NONE` (default) or `FIXED_GRAC`.
- `ATOMIC_SCF_EXPECTED_GRAC_SHIFT`: `0.0` (default, undeclared sentinel), or an
  explicitly supplied positive finite shift in Hartree for `FIXED_GRAC`.

`NONE` rejects attached GRAC, a nonzero actual shift, changed GRAC alpha/beta,
or a nonzero expected-shift declaration. `FIXED_GRAC` requires exact equality
of the declared shift and the already-converged wavefunction's actual shift.
Zero, negative, nonfinite, missing and mismatched declarations fail. No matching
tolerance or method-name inference is used.

Only this fixed canonical profile is supported:

- Restricted closed-shell C1, underlying unmodified canonical full-LibXC PBE0
  (`XC_HYB_GGA_XC_PBEH`), 25% exact exchange, no dispersion or range separation.
- Actual GRAC alpha `0.5`, beta `40.0`.
- Actual unpolarized `XC_GGA_X_LB` exchange, scale `0.75`.
- Actual unpolarized `XC_LDA_C_VWN` correlation, scale `1.0`.
- Canonical component definitions, including omega, flags, LibXC mix/CAM data,
  no explicit parameter tweaks, and the freshly constructed components' density
  cutoff controls. Explicit tweaks are conservatively rejected even if a caller
  believes them equivalent to defaults.

The exchange scale follows existing `VBase::set_grac_shift`: for global PBE0,
`1 - x_alpha = 0.75`; correlation retains its factory scale. Admission compares
all underlying PBE0 scalar/component definitions to an independent core factory.
It substitutes **only the explicitly declared correction fields** in the expected
profile; it does not discard GRAC fields from the comparison. Actual attachments
are inspected independently of `needs_grac` and of ambient SCF DFT options.

## Usage with an already-converged wavefunction

```python
# The caller, not oeprop, has already performed and converged fixed-GRAC SCF.
# This illustrative value is an explicit caller choice, never an API default.
fixed_shift = 0.06490004527520865
psi4.set_options({
    'atomic_scf_asymptotic_correction': 'FIXED_GRAC',
    'atomic_scf_expected_grac_shift': fixed_shift,
})
psi4.oeprop(wfn, 'ATOMIC_POLARIZABILITIES')  # returns None
from psi4.driver.procrouting.isapol_oeprop import atomic_property_result
result = atomic_property_result(wfn)
print(result.correction_provenance)
```

Declaring these options does not configure, run or repair SCF. A caller preparing
SCF separately can set `SCF/DFT_GRAC_SHIFT` before its explicit PBE0 energy call;
changing that option after convergence does not change the admitted state.
Setting correction controls on the wavefunction after convergence invalidates
its seal, even if they now match the declaration.

Expert `native_properties` and `native_response_from_wavefunction` take keyword
arguments `scf_correction='NONE'` and `expected_grac_shift=None`, independent of
the public global options. Opt in with `scf_correction='FIXED_GRAC'` and an explicit
positive `expected_grac_shift`. FIXED_GRAC requires a supported explicit ALDA
kernel; its exchange/local scales remain explicitly supplied response policy,
not inferred GRAC or PBE0 derivatives. NONE retains expert non-PBE0/no-local
support for actual SCF wavefunctions. The C++ primitive `NativeResponseProvider`
remains a lower-level caller-declared orbital/operator builder, not the Python
atomic admission boundary.

## Seal, provenance and lifecycle

Successful restricted C1 SCF seals now include actual GRAC X/C component
definitions (including parameter overrides) as well as shift/alpha/beta and the
existing state. Missing, stale, externally reconstructed or pre-change seals do
not establish eligibility. No seal is manufactured by native property code.
Public atomic admission retains the existing stopping-diagnostic and stationarity
checks. FIXED_GRAC expert admission also requires the current successful seal.

`CorrectionProvenance` is a frozen owned value of strings, floats and nested
immutable tuples. It is present on public results, native property results,
native responses and response contexts. Functional state contributes to the
wavefunction fingerprint; correction provenance contributes to the response
policy fingerprint. Reuse rejects changed state, policy or correction before
partitioning. The actual component getters expose the existing components for
inspection, with no new attachment setter; mutation through existing Functional
methods is detected by the next seal check. Provenance holds no such live objects.

Public `oeprop` still returns `None`; the accessor returns the owned latest result.
New requests invalidate the accessor on entry, including failing requests, and
never mutate previously returned results. Callers must not concurrently mutate
wavefunction state during construction.

The metadata explicitly says:

> ALDA response using fixed-GRAC SCF orbitals; no GRAC kernel derivative

The producer uses the actual supplied orbitals and energies. Existing
`SuperFunctional::compute_functional` applies GRAC only for derivative 1;
derivative 2 is **not** a GRAC kernel. This patch makes no derivative or
performance-kernel changes, runs no hidden SCF/IP/HOMO evaluation, and applies no
post-SCF correction, manual energy shift, orbital repair or table input.

## Remaining limits and verification boundary

The referenced H2O psi4 deck uses O at zero and H at
`(+/-1.45365196, 0, -1.12168732)` bohr, the `aVTZ` alias, IP `12.62063` eV,
HOMO `-0.3989`, and the audited fixed shift `0.06490004527520865` Hartree.
That reference is not selected automatically. Exact MAIN/AUX/AtomAux expansion
is not established. Its O-limit2/H-limit1, LW, weight3 `.001`, cutoff `.0001`,
SVD0/refinement settings are not implemented as a matched preset here. Neither
the generated recipe nor the uniform-rank3 route is that preset. PFIT, full aVTZ
experiments, GRAC response derivatives and full protocol parity remain out of scope.

Dense limits remain `nov <= 512` and `np * nov^2 <= 2e9`, with existing byte/work
caps and configured thread policy unchanged. No tolerance is relaxed. A corrected
SCF passing admission does not guarantee partition or strict LW acceptance.

New source tests cover helper rejection/ownership and one fresh small water
fixed-GRAC SCF plus native ALDA endpoint. The small response grid is a construction
test, not a converged atomic-property or matched-grid benchmark. Public lifecycle
coverage doubles partition only and does not claim a real atomic LW endpoint pass.
Rebuild core and use matching Python sources before running core tests. This
worker checks were offline only. Parent subsequently built/staged the core and
passed **1847 integrated tests in35.78s**, including a real underlying LibXC
density-cutoff mutation/restoration regression. Canonical functional admission
keeps numerical cutoffs separate, but actual underlying cutoffs now enter both
SCF seals and response-context fingerprints.

Parent also ran the original water input with caller-configured fixed-GRAC SCF
and the explicit atomic declaration: all public tasks completed, including real
strict-LW polarizabilities and9 ordered dispersion pairs, in30.2651s/578396KiB,
30 ISA iterations, energy-76.33871950327045Eh. Shift0.06490004527520865Eh was
explicitly supplied, not inferred. This is still the generated cc-pVDZ recipe,
not the aVTZ/refined reference. Evidence `.pi/audit/native-fixed-grac-water.json`.
The no-GRAC input remains byte-unchanged and all saved outputs bitwise baseline
in29.6120s/582816KiB (`native-water-post-grac-comparison.json`). Regression log:
`native-fixed-grac-regressions-v2.log`. Separate molecular/SAPT reruns also PASS:3 tests99.05s and4 tests113.29s,
respectively (`native-fixed-grac-{molecular,sapt}-regressions.log`).
