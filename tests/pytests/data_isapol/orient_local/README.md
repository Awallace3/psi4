# Retained LW reference declarations

These numerical declarations originate from an external, untracked historical
H2O working directory from Psi4 ISA-Pol port development, identified here and
in the JSON files as `historical-H2O-track`. That directory is not shipped and
cannot be reproduced from this repository; the recorded sha256 values pin the
original file identities.
Attribution: external ORIENT 5.0.10 (d8d8610) localization and CamCASP/PFIT
refinement. No ORIENT algorithm source is included; numerical-output use is not
blanket licensing clearance for other source material.

The supplied ORIENT import/placement product has been removed. Retained files:

- `manifest.json` and `manifest.json.sha256`: external authority for
  geometry, frames, units and historical source identity. A hash establishes
  identity, not numerical correctness. Entries for removed refined tensor
  snapshots describe their provenance, not files required by this test suite.
- `H2O.sites`, `H2O.axes`: bohr sites and proper local-to-global Cartesian frames
  (columns): O/H2 identity, H1 diag(-1,-1,1).
- `H2O.ornt`: input recipe provenance for the historical ORIENT run.
- `frequency_header_excerpt.json`: literal NL4/L3 frequency headers and hashes.
  L3 headers at nodes 7–10 cannot identify canonical nodes at their printed
  precision; retained tests check this rejection rather than alter frequencies.

Multipoles use real Racah order `00,10,11c,11s,...`; dipoles are z,x,y.
NEW-format L3 has no CARTSPHER field: its real spherical representation is
declared here by provenance. NL4 explicitly declares `CARTSPHER S`.
Local tensor conversion is `D(F)^T A D(F)`.

The `../oracle/extract_lw_{hermetic,dynamic}.py` parsers are not runtime
dependencies. They are tested on synthetic documents (the dynamic capture CLI
was removed; see `../oracle/README.md`); the grid and rotation tests use
tracked declarations and actual core operations. Full historical LW comparison
captures remain external development evidence, not a claim made by these tests.
The historical supplied charge-flow defect (~7.011e-4) exceeds the unchanged
production 1e-6 gate; it is never repaired or automatically retried.

Packaging migration (byte-level only): `H2O.sites` lost its trailing blank
line, and private absolute paths in `manifest.json` and
`frequency_header_excerpt.json` were replaced by `historical-H2O-track`. No
numerical, label, unit, frame or geometry value changed. The manifest's
provenance hashes and `manifest.json.sha256` were updated to match, and
`manifest.json` records the original (pre-normalization) hashes under
`packaging_migration`.
