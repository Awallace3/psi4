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

Actual standalone CUDA feedback passes32checks/145DGEMMs in3.440s, including
parallel reductions and nonorthogonal-rotation rejection. Maxabs7.958e-13.
This is primitive evidence, not native-factor/full-FISAPT execution.

The next source increment wires these rotations into the device-input DFMO
block loop and adds explicit `fdisp0_cuda` for unscaled two-fragment HF/C-link
assignment without external potentials. It returns a fresh cache and places
the GPU-combined active pair block at native frozen offsets. Link/scaling/
external/S-inf cases fail closed. Its141CPU/mock/serial executions pass;
whole-stage/native-factor/physical LO tests are pending.
This is not automatic FISAPT dispatch, GPU localization or other-term coverage.

UPDATE: whole stage06 atd8710cf97 and actual localized physical gate now pass:
water-Ne5x5, shifted/frozenwater-Ne4x4, frozenwater-water4x4, blocks500/5.
NativeCppPAIR/BLOCKED/Python reference matrices agree. GPU fullmatrix maxerror
6.800e-18Eh, signedcomponent/pair max3.193e-17Eh,208nativeDFMOcalls,
coefficientH2D0 and cleanup0bytes/owners. Upstream IBO localization remains CPU
reference input; it is not full-FISAPT GPU coverage.

## Next gates

1. Localized dispersion native-factor/full-matrix gate above is complete for
   the stated unscaled/C-assignment/no-external scope. Keep unsupported cases
   fail-closed; localization itself remains outside that coverage.
2. Qualify actual JKFIT angular momenta and atomwise nuclear/external-potential
   native interfaces. Port fragment elst/exch with no host factor provider.
3. Port fragment induction/response and localization/partition numerical work,
   then SAO/SIAO/link/external-potential cases. Check every signed term, matrix
   layout, group sum, repeated call, lifetime and final total independently.

No automatic CUDA option or full-FISAPT GPU claim is enabled by this document.

## Fragment electrostatics groundwork

`sapt_fisapt_elst_cuda.py` supplies device diagonal extraction, nuclear
repulsion, weighted atom/orbital block placement, cuBLAS electronic pair and
AO-potential contractions, and GPU scalar reduction. It includes all ordinary
occupied orbitals (frozen included) and preserves the final external zero slot.
Diagonal factors stay on device for later shared-JKFIT fragment induction.
Distinct coincident charged centers produce nonfinite output, never a clamp;
a public caller must reject nonfinite reported values.

151 CPU/mock/serial executions pass, including changed inputs, unequal spaces,
all matrix slots, retained factors and layout/session/alias guards. Actual GPU
primitives pass32checks/392DGEMMs in3.550s, maxabs6.821e-13 and zero live bytes.
Receipt: sibling `runs/gpu-fisapt-elst-primitives-01`, source9de9a88e2.
Synthetic factors/potentials were test inputs, not native-provider coverage.
Native point-charge qualification is pending. There is not yet a coupled
`felst_cuda` or physical fragment-elst coverage claim.
