# HF FISAPT GPU expansion

Canonical HF dispersion has passed the bounded signed physical GPU gate
(`af31e4a7f`, whole stage05). Spherical SPDF conventions and observable PSD
relative-cutoff support are qualified. Arbitrary negative-metric injection,
native effective-rank queries, general basis admission and speedups are not.

Reuse existing GPU SCF/elst/exch/induction wherever applicable. Canonical
coverage does not itself certify the separate fragment-resolved routines.
Keep CPU/non-HF defaults unchanged and production factors DFHelper-free.

## Actual seams and remaining coverage

* `sapt_jk_terms_ein.flocalization` builds IBO rotations and explicitly separates
  frozen/active blocks. `Uaocc0A/B` are square active rotations. Localization
  itself is not yet certified GPU work.
* `felst` builds the atom/orbital/external-potential matrix. Its current fragment
  implementation uses DFHelper with **JKFIT**, diagonal `(aa|Q)/(bb|Q)` factors,
  NumPy reductions and atomwise CPU ExternalPotential matrices. It also supplies
  `Vlocc0A/B` to fragment induction. This is distinct from canonical GPU elst.
  A port must preserve the actual JK fitting basis, not substitute RIFIT.
* `fexch` and `find`: inventory actual localized JK calls, response algebra,
  fragment layouts, scaling and link conventions before claiming coverage.
  Reuse the existing GPU JK/induction seams rather than replacing canonical
  algorithms. E2 graph placement alone is not fail-closed GPU-residency proof.
* `fdisp0`: canonical factor/correction equations are shared, but localized
  occupied-pair energies require **canonical denominator division before LO
  rotation**, then signed contractions with rotated V/X/crossed W.
  The current fragment path still uses host DFHelper and rejects `_gpu=True`.
* `saptdft_fisapt.setup_fisapt_object`, `fisapt_proc.fisapt_compute_energy` and
  native `FISAPT::{felst,fexch,find,fdisp}` define output/cache conventions.
  Native fragment/link contract inspection is running independently.

## First implementation increment

`sapt_fisapt_cuda.py` adds GPU occupied-only rotations, canonical amplitude
division and signed localized-pair reduction. Validation checks orthogonality
on GPU and downloads scalar errors only. CPU serial-shim tests cover unequal
spaces, edge blocks, crossed layout, changed inputs, repeated accumulation and
the deliberately wrong "divide after localization" alternative.

This increment is **not** public FISAPT dispatch or atom/fragment/link
assignment. It is not proof of GPU localization or other FISAPT terms.
Real standalone GPU feedback must precede native-factor/physical LO testing.

## Next gates

1. Wire localized dispersion to real device DFMO and compare full localized
   pair matrices and canonical/LO global conservation with native FISAPT.
   Preserve frozen offsets; reject unimplemented link/external/S-inf cases.
2. Qualify actual JKFIT angular momenta and atomwise nuclear/external-potential
   native interfaces. Port fragment elst/exch with no host factor provider.
3. Port fragment induction/response and localization/partition numerical work,
   then SAO/SIAO/link/external-potential cases. Check every signed term, matrix
   layout, group sum, repeated call, lifetime and final total independently.

No automatic CUDA option or full-FISAPT GPU claim is enabled by this document.
