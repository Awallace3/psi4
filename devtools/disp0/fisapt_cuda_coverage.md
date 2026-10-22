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

UPDATE: native FP64 point potentials pass attraction/repulsion/mixed/zero
SPDF matrices, maxabs8.382e-15, complete cleanup
(`runs/potential-device-02`). The new explicit `CuESTPointPotential` retains
OE plan/FP64 parameter/fixed-count workspace objects in the existing owner
graph and rejects borrowed precision drift without silently restoring it.

`felst_device` now connects native JKFIT DFMO and point potentials to the
device algebra. It returns borrowed device diagonal factors tied to the
original provider/metric gauge, plus full matrix and GPU sum; it does not
download factors, close their enclosing scope, or overwrite canonical Elst.
154 CPU/mock/serial executions pass. Whole-stage installation and independent
physical/native-C++ matrix comparison are pending. No default driver dispatch,
GPU IBO localization, external/link/scaled or remaining-term claim.

UPDATE: whole stage07/source551c06038 passes physical native fragment elst for
water-Ne, shifted/frozenwater-Ne, frozenwater-water, two calls each. Full-matrix
maxerror1.288e-13Eh, GPUglobal1.434e-12Eh;6nativeDFMO/56point-potential calls,
coefficientH2D0, cleanup0bytes/owners (`runs/fragment-elst-gpu-01` through `-03`).
The independent oracle explicitly builds spherical JKFIT; its first attempt
correctly rejected an inherited Cartesian set. No runtime conversion/fallback.

## Fragment exchange groundwork

`fexch_device` implements the three unscaled localized first-order S^2 terms
from existing raw device S/V/J, localized occupied and virtual inputs. It does
not reimplement upstream SCF/JK. Linearity projects virtual spaces on GPU before
native JKFIT DFMO, avoiding host Bab/Bba disk tensors. Crossed factors keep
their distinct `(Q,b,a)` layout; a custom CUDA kernel reduces the signed pairs.
The original live electrostatic state identifies the shared provider/gauge.
Frozen occupied orbitals participate; atom/external output slots remain zero.

163 CPU/mock/serial executions pass, including nonsymmetric test inputs to
discriminate transpose conventions and unequal/changed spaces. Actual RTX3090
primitive feedback passes48checks/112DGEMMs in3.620s, maxabs3.469e-17 and zero
live bytes (`runs/gpu-fisapt-exch-primitives-01`, sourcebc1b3aeb7). Synthetic
factors were explicitly uploaded test inputs, not native-provider coverage.
Whole installation and physical native-factor exchange comparison remain
pending. This is not scaled Exch10, Exch-Disp20, or complete FISAPT coverage.

UPDATE: whole stage08/source8aebb7b16 and physical shared-provider elst/exchange
gate pass all three systems twice. Unscaled S^2 exchange fullmatrix maxerror
2.641e-16Eh/global3.619e-16Eh. Simultaneous elst matrix1.296e-13/global1.421e-12.
Twelve native DFMO/56point-potential calls, coefficientH2D0, cleanup0bytes/owners.
Receipts: `runs/fragment-exchange-gpu-01` through `-03`. Frozen occupied orbitals
remain included; no scaled/coupled/link/external/default workflow claim.

## Uncoupled induction primitives

`sapt_fisapt_induction_cuda.uncoupled_source` divides a canonical occupied/
virtual source by single-excitation denominators BEFORE full occupied
localization, then reduces signed Ind20u and optional Exch-Ind20u target
contractions into caller-owned directional source columns. It includes frozen
occupied targets and performs no intermediate download. This does not yet
construct native nuclear/electronic sources or exchange-induction targets.

172 CPU/mock/serial executions pass, including nontrivial rotation,
denominator-order discrimination, optional signed exchange, source columns,
reserved zero slots, changed inputs and lifetime/layout/alias guards.
Actual GPU primitive feedback passes24checks/68DGEMMs in3.560s,
maxabs4.974e-14, zero live bytes (`runs/gpu-fisapt-induction-primitives-01`,
source0cc4e1ba8). Synthetic source/target inputs only, not native assembly.
No physical fragment-induction, coupled-response or full FISAPT claim.

Next source increment adds explicit `ind20u_device`: resident shared-JKFIT
canonical occupied/virtual factors plus existing localized diagonal factors,
native atomwise attractive potentials, GPU full-occupied rotation extraction/
span/orthogonality/gap validation, and separate directional Ind20u matrices
with native atom/orbital/external slots. The elst state now retains the exact
localized occupied inputs used by its diagonal factors.

174 CPU/mock/serial executions pass, including complete synthetic source-sum
closure and both directional matrices. Native-factor physical Ind20u, source
assembly CUDA qualification, and all Exch-Ind20u targets remain pending.

Exchange-induction groundwork now includes a GPU Q-wise density-factor trace,
projected J_P contraction (same DFMO metric gauge, no AO J_P/CPU factors), and
the localized S^2 target operator for both directions, with nonsymmetric K_O
orientation explicit.175CPU/mock/serial executions pass against an independent
fourteen-term expanded AO expression and wrong-K_O-transpose discriminator.
The targets are not yet wired to native generalized-density DFMO in the source
loop; actual GPU target primitives and physical Exch-Ind20u remain pending.

UPDATE: actual projected-JP/target primitives pass52checks/216DGEMMs in3.705s,
maxabs1.137e-13, zero live bytes (`gpu-fisapt-induction-primitives-03`,
sourceaad7376e2). The new explicit `find20u_device` wires generalized-density
coefficient products and six same-gauge native DFMO pairs into both signed
directional components and GPU-combined matrices/totals.175CPU/mock/serial
executions pass, including independent full synthetic component/combined
matrices. Whole installation and physical native Exch-Ind20u remain pending.

UPDATE: native directional Ind20u gate passes all three systems twice on whole
stage09/source99bbbcb9f: matrix1.084e-16/global8.272e-16Eh,12DFMO/112potential
calls, coefficientH2D0, cleanup0bytes/owners (`fragment-ind20u-gpu-01`–`03`).
Both nuclear/electronic sources, all occupied targets and reserved zero slots
are checked. Whole stage10/source1d87a0643 and independent signed CPU/native
references now pass; the shared-source signed GPU gate (response disabled)
is running. This is not CPSCF response, localization or full-workflow coverage.

## Resident HF response groundwork

`sapt_fisapt_response_cuda.HFResponseOperator` uses same-gauge native
occupied-virtual, occupied-occupied and virtual-virtual factors for the positive
singlet operator gap+4J-K-K^T. Both exchange contractions are distinct.
Resident PCG uses orbital-gap preconditioning and returns physical x=-H+^-1 W.
Only scalar convergence/curvature values reach host control; a true-residual
check follows recurrence convergence. Zero/tiny RHS, negative curvature,
nonfinite values, nonconvergence and a bounded fused-exchange work cap are
explicit. No AO JK/CPU solver fallback exists.

184CPU/mock/serial executions pass against independent four-index Hessians and
dense CPU solutions, changed RHS, zero RHS, sign and lifetime/failure guards.
Actual GPU primitive feedback is next. Native-factor response Hx/solutions,
fragment source integration and physical coupled-response qualification remain
pending. The known native find coupled-source bug is not used as an oracle.

UPDATE: signed uncoupled native-factor induction gate passes three systems
twice on whole stage10/source1d87a0643 (`fragment-find20u-gpu-01`–`03`).
Ind20u matrixmax1.084e-16Eh, Exch-Ind20u5.329e-17, combined7.357e-17;
all-component globalmax8.271e-16.12DFMO/112potentials, coefficientH2D0,
cleanup0bytes/owners. Together with earlier localized dispersion/elst/exchange
gates, this covers the stated bounded UNCOUPLED/unscaled term-family scope
only. Coupled response, GPU localization, upstream resident handoff and complete
default/link/external/scaled FISAPT remain unqualified.
