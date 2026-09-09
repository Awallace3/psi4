# Supplied-input property stages (developer API)

These stages are **not** the ordinary wavefunction-first ISA property interface.
They do not generate a partition, Drho-C, transition fits or a response kernel.
The LW stage localizes explicitly supplied nonlocal tensors; the PFIT stage fits
parameters only from explicit supplied inputs.
No new `oeprop` task or energy method is registered.
See `SPEC.md` for the still-open native and end-to-end acceptance gates.

## Supplied nonlocal tensors through Psi4 LW

`psi4.driver.procrouting.isapol_lw` adds a separate expert workflow. Localization
is computed by C++ LW, not imported or inferred from nonlocal diagonal blocks:

```python
from psi4.driver.procrouting import isapol_lw as lw

local = lw.supplied_nonlocal_properties(
    labels=labels, origins=origins_bohr, frames=local_to_global_frames,
    bonds=explicit_bonds, frequencies=imaginary_frequencies,
    tensors=global_pair_tensors, input_rank=4,
    truncation=lw.TRUNCATE_RANK4,
    provenance=lw.Provenance(source_name, source_sha256, producer, description),
)
alpha = local.atomic_scalars.array  # (frequency, site, rank1/2/3)
tensors = local.raw_local.array     # (frequency, site, 15, 15)
dipoles = local.global_dipoles.array # global Cartesian xyz

# Separate optional stage: requires dynamic data and actual CP quadrature weights.
cn = lw.isotropic_dispersion(
    local_a, local_b, cp_weights=cp_weights,
    quadrature_provenance=quadrature_provenance,
)
```

Input axes are `(frequency, response_site, potential_site, response_component,
potential_component)` in real Racah order `00,10,11c,11s,...`. Origins are bohr,
frames are proper local-to-global matrices with axes as columns (`None` explicitly
means identity). Graph edges are caller-supplied zero-based site pairs. Rank4
requires the explicit truncation declaration; all25x25 entries must be finite,
then exactly the top16x16 is localized. Rank3 input needs no truncation. Returned
local ranks1–3 omit rank0; input snapshots retain the original arrays. Array
getters return copies. Diagnostics preserve asymmetry and indefinite tensors;
no repair, clipping or implicit symmetrization occurs. Static-only calls require
no weights, partner or PFIT. Dispersion weights include the Jacobian and1/(2*pi)
once, static weights are zero, and A/B grids match exactly. Local rank3 C12 is
partial even if supplied nonlocal tensors had rank4.

Production calls retain the C++1e-6 postcondition. The named
`residual_policy="historical_water_diagnostic"` authorizes1e-3 ONLY for the exact
pinned static water input, source-hash declaration, geometry, frames and graph;
modified data are rejected. It performs no reference-tree or fixture I/O and no
automatic retry. Its result retains `production_postcondition_passed=False` for
this input's7.011e-4 charge residual. The675-entry historical output comparison
passes1e-11, independently of that failed production postcondition. This is
supplied-nonlocal/Psi4-LW output, not a new wavefunction prediction or PFIT model.
The driver also exposes a separate explicit-placement anisotropic adapter; it
still registers no oeprop task:

```python
import numpy as np
oriented = lw.anisotropic_dispersion(
    local_a, local_b,
    placement_a=lw.Placement(np.eye(3), [0., 0., 0.]),
    placement_b=lw.Placement(np.eye(3), [0., 0., 10.]),
    cp_weights=cp_weights, quadrature_provenance=quadrature_provenance,
    max_order=12,
)
```

This requires genuine positive-frequency data and exactly reciprocal `raw_global`
tensors. Even roundoff asymmetry is rejected, without repair or isotropic fallback.
The historical static example cannot supply this dispersion request. Placements
act on whole models as `r_placed=R*r+t` (bohr); C++ receives unchanged global
response tensors with component frame R, never a local/global tensor roundtrip.
Placed source frames R*F are metadata only. Coincident A/B sites are errors.
Owned results retain both source models, placements, geometry, quadrature authority,
per-order C6–C12 including odd orders, energies and ordered-rank coverage.
Rank1–3 tensors provide unrestricted completeness only through C8; higher orders
explicitly retain missing quadruples. These are orientation-resolved scalar
coefficients, not recoupled coefficients or molecular/native acceptance.
Independent review found no must-fix; final parent installed-import tests passed52
and the combined suite passed1387 in14.71s, including a deterministic raw-local
getter-access regression. No numerical policy changed. Bounded evidence:
`tests/pytests/data_isapol/psi4_lw_anisotropic_driver_evidence.json`.
Independent driver review found no must-fix for factory-produced results; public
result dataclass constructors are trusted containers, not an alternative validated
input boundary. Use `supplied_nonlocal_properties` to construct results.
Parent verified source/staged module identity, regenerated the water artifact
through the installed module, and passed1335 combined ISA/FDDS tests in14.40s.
Artifact: `.pi/audit/lw-driver-water-static-parent-v1.json`. Its static dipole
trace scalars are O6.127427953741659, H1 1.7289320737992027,
H2 1.7289324071325358 bohr^3. These are LW-localized unrefined results, distinct
from the imported PFIT-refined values below. This static fixture provides no
molecular C_n prediction: dynamic nonlocal tensors and actual weights are still
required. The dispersion adapter is validated with supplied synthetic frequency
models, not by inventing a dynamic model from static water.

## Working supplied-ORIENT property workflow

The original imported-localization workflow remains available unchanged. The Python bridge
`psi4.driver.procrouting.isapol_supplied` reads explicitly supplied local ORIENT
outputs and produces atomic tensor/scalar polarizabilities and dispersion:

```python
from psi4.driver.procrouting import isapol_supplied as b
model = b.read_orient_local_response(input_directory, manifest_path)
result = b.supplied_local_properties(model)  # all isotropic site pairs
# Optional explicit placement for directional coefficients/energies:
placement = b.Placement((0, 0, 10), ((1, 0, 0), (0, 1, 0), (0, 0, 1)))
directional = b.supplied_local_properties(model, model,
    placement_b=placement, anisotropic=True)
```

The manifest pins sources, units, ranks, geometry, local frames and frequency
mapping. The supplied refined water headers incorrectly label dynamic tensors as
static: explicit manifest authority maps eleven sections to CasimirGrid(10,0.5),
while preserving the header conflicts. Unsupported/distributed/off-site formats,
malformed or unterminated sections, missing data and hash mismatches are errors.
This is a bounded NEW-format reader, not a general ORIENT interpreter.

**Atomic tensors are imported**, including external PFIT refinement when declared.
Psi4 derives trace scalars/global dipole tensors and computes isotropic and
orientation-resolved anisotropic coefficients. Raw tensors are retained without
clipping or symmetrization. Slight asymmetry can prevent anisotropic conversion
without suppressing otherwise valid isotropic output. Optional reference type
mapping failures produce unavailable comparisons, not an output-aborting error.
Numerical comparison failure does not suppress valid results.

Portable example (with the staged Psi4 package on PYTHONPATH):

```sh
python -P tests/pytests/data_isapol/oracle/run_supplied_local_properties.py \
  --manifest tests/pytests/data_isapol/orient_local/manifest.json \
  --model-a tests/pytests/data_isapol/orient_local \
  --placement-b tests/pytests/data_isapol/orient_local/placement_b_example.json \
  --output water-supplied-properties.json
```

Outputs are exclusively created and carry raw tensors, provenance, computed
values, warnings, comparison availability and correction TODOs. The example B
translation is a demonstration, not reference dimer geometry. Rank3 isotropic C12
is partial; directional coefficients are not recoupled CamCASP components.

Validated water outputs: static scalar dipole alpha O6.129740498218001 and
H1/H2 1.7335580596856666 bohr³; O–O C6=17.25558575556027 Eh·bohr⁶.
All12 bounded archived isotropic comparisons pass at printed precision, while
four unrefined frequency-header comparisons fail and remain visible. Both reviewed
boundary bugs are fixed; 54 focused and1152 combined ISA/FDDS tests pass.
This is a working **external-local-response → properties** workflow, not a native
wavefunction-first calculation or full upstream agreement.

## Native-integral, supplied-orbital OV fitting

`core.IsaAuxCoulomb(aux).fit_ov(main, Cocc, Cvir, provenance, charge_penalty=1.0)`
returns owned original J/A/T, fitted D, q and diagnostics. It uses two native
integral contractions `(Cocc.T @ B[k]) @ Cvir`, occupied-fast row index
`a+noccupied*r`, and one general-LU solve of `A D.T=T.T` with
`A[k,l]=J[k,l]+(lambda*q[k])*q[l]`. No spin factor, rescaling or neutrality repair
is added. Inputs remain supplied orbitals, not an independently generated SCF.

The water C++ coefficients match the same-core NumPy control bitwise and pass
provisional1e-3 reference comparison (scaled2.3019798321950356e-5); strict parity
remains TODO11. Native H1/H2/kernel generation and downstream native-response
acceptance are separate. Matrix getters clone; storage admission is bounded, not
process-RSS or runtime guaranteed. No property registration is implied.

## Partitioned molecular-AUX multipoles

`core.IsaPartitionedMultipoles(auxiliary, sites, provenance, denominator_cutoff=1e-36)`
uses an `IsaExplicitBasis` with role `MolecularAux`. Every `IsaMultipoleSite` supplies:

- `label`: unique nonempty identifier, not just an element symbol;
- `origin`: three Cartesian coordinates in bohr;
- `rank`: 0–4;
- `samples`: an `IsaMultipoleSamples` object with `points`, `weights`, `shape`,
  `shape_sum`, and `auxiliary_sites`.

Shapes must already reflect the caller's screening and tail policy. For every
point with `abs(shape_sum) > denominator_cutoff`, the stage integrates

```
Q[a,t,k] += weight * R_t(point-origin[a]) * shape/shape_sum * auxiliary[k](point)
```

Signed values are retained. No density clipping, population normalization, or
charge rescaling occurs. AUX neighbours are unique zero-based indices, with no
Fortran padding. An empty list screens every AUX function, not none of them.
One neighbour list applies to the entire site's sample batch. Separate site grids
are supported; the caller must ensure their partitions and quadratures are consistent.

`values` is a fresh Matrix with concatenated `(site,component)` rows and AUX columns.
Use `offsets`, `ranks`, `components`, `labels`, and `origins` to interpret it.
Components are Racah-normalized regular real harmonics in
`00,10,11c,11s,20,21c,21s,22c,22s,...` order; dipoles are **z,x,y**.
All sites use the global Cartesian axes. `excluded_denominators` and
`negative_ratios` are per-site diagnostics, not convergence certificates.

## Distributed response

```
response = core.IsaDistributedResponse(
    partition, frequencies, coefficient_responses,
    "fitted_density_coefficients", response_provenance)
alpha = response.at_index(0)
```

Every input response is an AUX-by-AUX Matrix in fitted-density coefficient
coordinates. The stage computes **`-Q C_DF Q.T`**. It does not transform a Psi4
Coulomb-auxiliary response into those coordinates. Constrained fitting and kernel
policy differences cannot be resolved by a representation label.

Frequency values are nonnegative imaginary-axis magnitudes in atomic units; input
order is retained. `at_index` returns an independent matrix, not a view into stored
state. `partition` supplies both labeled site/component axes. `reciprocity_errors`
reports `max(abs(alpha-alpha.T))/max(1,max(abs(alpha)))`; raw tensors are not
symmetrized. Large tensors are not stored in padded QCVariables.

Molecular dipoles generally require translating charge-flow components before
summing. The sum of local dipole blocks alone is not molecular-response recovery.
This object is **distributed**, not automatically localized.

## Isotropic dispersion from explicit local scalar models

An `IsaIsotropicSite` supplies a unique `label`, bohr `origin`, explicit increasing
`ranks` from 1–4, and a `polarizabilities` Matrix. Rows are frequencies and columns
are the listed ranks. Each scalar is

```
alpha_l = trace(local_Racah_alpha_ll) / (2*l+1)
```

This definition does not authorize treating arbitrary diagonal site blocks of a
multisite distributed tensor as a localized model. Localization and frame/model
selection remain separate transformations.

```
model_a = core.IsaIsotropicModel(frequencies, sites_a, provenance_a)
model_b = core.IsaIsotropicModel(frequencies, sites_b, provenance_b)
result = core.isa_isotropic_dispersion(model_a, model_b, cp_weights, max_order=12)
```

Models must have exactly matching, strictly increasing frequency lists. There is
no interpolation. Weights must already include the frequency-mapping Jacobian and
**`1/(2*pi)` exactly once**. A static frequency has zero weight. For the existing
`CasimirGrid`, use its `cp_weight(k)`, not its raw Gauss–Legendre `weight(k)`.
At least one positive integration weight is required. Signed scalar responses are
retained; the API does not certify passivity or positivity.

For every A/B site pair, each coefficient includes its `order`, within-model
`value`, `included_rank_pairs`, `missing_rank_pairs`, and `complete` flag.
The contribution of `(la,lb)` is

```
n = 2*la + 2*lb + 2
C_n += binomial(2*la+2*lb, 2*la) * sum(cp_weights * alpha_A_la * alpha_B_lb)
```

Rank-2 models give complete C6/C8, **partial C10**, and no included C12 terms.
Rank-3 models give partial C12; rank-4 covers complete local isotropic C12.
A zero value with missing rank pairs is not a complete physical zero.
This API implements no anisotropic dispersion or intersite energy contraction.

## Orientation-resolved anisotropic dispersion

An `IsaAnisotropicSite` supplies a label, bohr origin, proper local-to-global
Cartesian frame, explicit increasing ranks within1–4, and one reciprocal real
local response Matrix per frequency. All blocks among declared ranks must be
present. Input matrices must be exactly symmetric; signed/indefinite values are
allowed. An omitted frame is invalid, not an implicit identity.

```
a = core.IsaAnisotropicModel(frequencies, sites_a, "supplied_local_response", provenance_a)
b = core.IsaAnisotropicModel(frequencies, sites_b, "supplied_local_response", provenance_b)
result = core.isa_anisotropic_dispersion(a, b, cp_weights, max_order=12)
```

Frames rotate local responses as `D alpha D.T`. The direct Coulomb interaction
uses displacement B−A and the existing real Racah convention. CP weights already
include1/(2*pi). The result retains owned model/grid snapshots and all site pairs;
each pair exposes distance/direction, scalar orientation-resolved coefficients,
per-order energies `-C_n/R^n`, and their explicitly truncated sum. Orders6–12
include **odd7/9/11**, which mixed-rank response blocks can generate.

Included/missing ordered rank quadruples distinguish declared-model completeness
from unrestricted completeness: full rank4 covers general C6–C9, **not general
C10–C12**. Explicit zero blocks are not missing blocks. These scalars are not
CamCASP recoupled `C_n(t,u,J)` tensors. No implicit localization, damping,
retardation, chargeflow, intersite response or PFIT adapter is performed.

`isa_anisotropic_interaction(rank_a, rank_b, displacement)` exposes an owned
physical electrostatic T matrix for expert sign/normalization checks. Its ranks
are each1–4; axes within a rank are `0,1c,1s,...`.

See `ANISOTROPIC_CONTRACT.md` for the formula, numerical/resource limits and exact
validation scope. Canonical build/staging and binary identity checks passed,
with120 focused tests and1002 combined ISA/FDDS tests. Independent review found no
must-fix. High-rank off-axis element evidence remains indirect; the tested
orientation-average cubature is explicitly state-specific on B. This is not
native molecular or historical-reference anisotropic parity.

## PFIT from explicit point-response targets

`core.isa_pfit_solve(problem, options)` consumes an `IsaPfitProblem` at one
nonnegative imaginary frequency. Its value-owned model supplies channel labels,
full symmetric parameter tensors `K[k]`, parameter labels/units, and fixed values.
Each `IsaPfitBatch` supplies bohr coordinates, channel fields, and packed targets
in `(i,j<=i)` order. Physical batches remain separate: no cross-batch pairs,
off-diagonal factor two, or induction-energy factor one-half is introduced.

The objective is

```
sum_batches sum_j<=i (sum_k (f_i.T K[k] f_j)*p[k] - v[i,j])**2
+ (p-h).T P (p-h) + sum_l strength[l]*(t[l].T p-target[l])**2
```

Targets mean **`-d(phi_induced)/dq`**, in `Eh/e^2`, not permanent or bare
source-source electrostatics. Target origin, convention, source and generation
metadata are mandatory caller declarations. Fitted-propagator targets additionally
require the fitted-density representation and AUX basis identity. Synthetic tests
are explicitly labeled; `native_verified` remains false. The stage does not create
targets, select/localize a model, or reconstruct targets from a truncated model.

`IsaPfitOptions.solver` selects `StreamingQR` (default, bounded Givens row insertion)
or `NormalEquationsDSYSV`. Both use the same augmented rows and fixed elimination,
including correlated-prior fixed terms and actual LC targets. Matrix priors require
exact symmetry and nonnegative computed eigenvalues; no tiny-negative projection
or hidden ridge is applied. Their effective matrix is `B.T B` from the accepted
square-root rows, with reconstruction correction reported. LC diagnostics, solver
input, and residuals consistently use `sqrt(strength)*t` and
`sqrt(strength)*target`, avoiding strength-first underflow discrepancies.

Results expose original reduced H/b, separate and combined penalty matrices/RHSs,
optional per-batch predictions, objective components, residuals and diagnostics.
Normal rank/condition diagnostics concern H; QR diagnostics concern R, not H.
Rank/condition failures do not expose fitted parameters and have
`objective_available=false`. All-fixed problems still validate and evaluate the
full objective. No pseudoinverse, automatic pruning, or solver fallback occurs.

`maximum_work_bytes` bounds a conservative kernel numerical-buffer budget, **not
process RSS**; caller inputs, strings/container overhead and later getter copies
are excluded. `qr_chunk_rows` changes computational buffering, not physical batches.
All nested Python container properties are snapshots: modify then assign back to
update an input. Result getters likewise return independent values.

PFIT's 102 focused tests include independent dense augmented-system oracles,
fixed/LC penalties, analytic point-response penetration residuals, failure/resource
contracts, and the six reproduced-and-fixed scaled-LC regressions. Canonical
installation matched the built core byte-for-byte; the combined ISA/FDDS suite
passed **882 tests in 5.69 s**. This establishes supplied-input numerical behavior,
not native molecular or historical-reference PFIT parity.

## Ownership and evidence

Model constructors copy inputs. Matrix getters return copies; modifying an input
or returned Matrix does not change later evaluations. Result lists and coordinates
are inspectable snapshots. Invalid dimensions, symmetry blocks, nonfinite inputs,
unsupported ranks/representations, and nonfinite computed results fail explicitly.

Tests in `test_isapol_partitioned_response.py`,
`test_isapol_isotropic_dispersion.py`, and `test_isapol_supplied_chain.py` establish
analytic/invariant and supplied-operator checks. The composed chain uses synthetic
Gaussian p AUX and supplied Lorentz operators, then checks analytic C6. It is **not**
a native wavefunction-to-property water calculation or a molecular parity fixture.
