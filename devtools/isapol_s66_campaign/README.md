# Native S66x8 monomer campaign preparation

This directory prepares **raw geometry inputs only** for a future native
PBE0/aug-cc-pVTZ ISA/response/refinement campaign. It does not submit work,
execute Psi4, or load CamCASP molecular products.

```console
python -m pytest --noconftest -c /dev/null -p no:cacheprovider -q \
  devtools/isapol_s66_campaign/test_inventory.py
python devtools/isapol_s66_campaign/inventory.py \
  --data /path/to/los-ii --output /new/campaign/inventory-001
```

The inventory covers all528 S66x8 dimers and1056 fragment occurrences.
It groups monomers only after matching ordered elements, charge, multiplicity
and a **proper** rigid placement with an explicitly recorded input-roundoff
tolerance. Reflections, different conformations and unverified atom
permutations are not reused. Every dimer/fragment retains its representative,
source atom indices, rotation, translation and placement residual.

The default tolerance is1e-7 bohr; a separate1e-10-bohr grouping count exposes
sensitivity. This is a geometry bookkeeping declaration, not a claim of
rotational covariance or response accuracy. In particular, the native LW
algorithm's known covariance limitation remains; any future anisotropic
evaluation needs its own frame and placement tests.

The first inventory of the supplied LoS-II files found131 ordered monomer
geometries covering1056 fragment occurrences. The largest actual reused
placement residual was1.053e-8 bohr; tightening the criterion to1e-10 bohr
split the inventory into562 groups, showing the effect of source-coordinate
roundoff. These are13 molecular formulas but **not**13 interchangeable
geometries. All were neutral CHNO singlets, with3–17 atoms and up to612
independent noncharge rank2 upper-triangle fit parameters.

## Submission gates

The user's queue policy is **free EMBERS only**, never implicit/default QoS
or an inferno fallback. Use bounded QCFractal-managed records, a dedicated
campaign tag and an exact pinned Psi4 build, not an ad hoc large SLURM array.

Before queuing molecular calculations:

1. Implement and test an explicit generic CHNO recipe/declaration, retaining
   the existing molecule-specific water and benzene regressions.
2. Give chemically inequivalent sites independent parameters and frames;
   formula/element equality is **not** site-level COPY symmetry.
3. Admit the actual finite parameter/resource inventory rather than
   truncating it to the existing72-variable special-case allowance.
4. Add a shipped serializable QCSchema native-property/export path and prove
   its QCFractal submit/retrieve/checksum round trip.
5. Qualify representative new monomers and their scientific/artifact/input
   acceptance inside embers allocations before expanding concurrency.

The existing rank2 model supplies complete isotropic C6/C8 but only partial
C10(2,2). This preparation does not promise full C10 or convergence.

## Implemented generic source seam

`psi4.driver.procrouting.isapol_molecular` accepts a versioned
`MolecularDeclaration` containing explicit proper local-to-global frames and
bonds. It gives **each atom its own fit type**, declares all36 noncharge
rank2 upper-triangle variables per atom, and calls the existing public
`psi4.oeprop(..., preset="explicit")` route. It neither starts SCF nor imports
external molecular products. It requires the existing successful-SCF seal;
an attempted new request clears stale owned atomic results.

The separate `s66-chno-isa-a-v1` recipe is experimental: native Cartesian
aug-cc-pVTZ-RI, C/N/O s2**[8..-3], H s2**[5..-2], zero atomic initialization
and uniform1.5×Bragg–Slater tails. Nitrogen's sequence and campaign water's
tail policy are declared model choices, not newly established accuracy
results. Existing water/benzene regression functions remain unchanged.

The legacy default64/standard72-parameter cap and64-billion replay-work
default remain. The new finite `molecular` opt-in admits up to1024 parameters
only under explicit `large` resources; packed replay can explicitly admit up
to2e12 work units and is still charged to the cumulative shared ledger.
Retained-frequency memory scales with the declared parameter² inventory.
The shared refinement ledger also charges per-frequency cubic dense solver
work (penalty/normal eigendiagnostics, penalty assembly and DSYSV); a small
fit cloud does not make those operations free. Its conservative arithmetic
allowance is not a vendor-library iteration-count or wall-time guarantee.
`plan_molecular_refinement()` allocates no design buffer and is explicitly
**refinement-only**, not a complete SCF/ISA/response or wall-time guarantee.

`export_molecular_result()` requires the actual wavefunction-owned result,
complete11-node independent-site fit inventory and finite numeric exports.
It retains partial-C10/solver-certification qualifications and reports
nonpositive scalar coefficients rather than clipping them into usable BJ
inputs. Source-under-test unit checks are not installed scientific acceptance.

## QCSchema transport seam under development

`isapol_schema.py` defines a strict `native-monomer-request/1` JSON input:
the explicit declaration, finite resources/replay allowance, and required
SHA256 pins for the imported native core, all installed `isapol_*.py`
sources and the shipped schema/driver routing files. Preflight requires
PBE0/aug-cc-pVTZ, neutral singlet C1 and fixed
COM/orientation, so declared local frames cannot silently follow a geometry
reorientation. These pins identify implementation bytes, not scientific
accuracy or an independent runtime audit.

The owned producer can package its result metadata and complete numeric
NPZ into a namespaced JSON artifact envelope. Binary data are bounded,
base64-encoded and checksum-bound; decoding rejects object/nonfinite tensors,
unexpected members/layouts and forged huge array headers before allocation.
Temporary exports live only under Psi4's configured scratch directory.
`schema_wrapper.py` now consumes the request before ordinary properties
dispatch and never forwards it through the planner or SCF kwargs. It retains
the normal dipole result, verifies admitted geometry before and after SCF,
and publishes the checked envelope in `extras["psi4:native_monomer"]` only
after successful native execution. Its narrow initial input surface requires
explicit spherical RHF/PK, fresh SAD and tight stopping thresholds; it rejects
alternate functionals/bases, embedding, extra input files and ambient options.
Malformed requests are `ValidationError`, not silent ordinary calculations.

Source tests exercise v1/v2 schema conversion and artifact preservation with
an explicitly **synthetic stub** at the quantum boundary; they do not compute
molecular data or establish installed QCEngine collection, server acceptance
or a QCArchive submit/retrieve round trip. Both current draft entry/declaration
preparers include C1 in the bound molecule identity; drafts made before this
field was introduced must be preserved but not submitted.

## Explicit frame and bond inputs

```console
python devtools/isapol_s66_campaign/prepare_declarations.py \
  /new/campaign/inventory-001/inventory.json \
  --output /new/campaign/declaration-preparation-001.json
```

The raw supplied geometries have no connectivity. `declarations.py` records
a geometry-only CHNO preprocessing policy: edges shorter than1.20 times
the sum of H/C/N/O covalent radii0.31/0.76/0.71/0.66angstrom, a0.02angstrom
ambiguity guard, connectedness and finite degree checks. These are explicit
input conventions, not inferred bond orders or a response-accuracy result.

Each proper local-to-global frame has z toward the first bonded atom in
source order and x toward the most noncollinear other atom's projection.
Near-linear sites below sine1e-5 use a disclosed laboratory-axis azimuth;
the choice is not presented as covariant. Frame inputs and policy evidence
are bound to each representative's full molecule hash, including atom order.
No atom sharing or COPY is introduced, and no molecular quantum computation
or QCArchive registration is performed by this preparation.

## Dataset entry preparation

```console
python devtools/isapol_s66_campaign/prepare_entries.py \
  /new/campaign/inventory-001/inventory.json \
  --output /new/campaign/entry-preparation-001.json
```

As requested, each singlepoint entry retains every parent dimer/fragment
context and supplied SAPT reference component **in attributes only**. Missing
reference values remain explicit nulls. They are not monomer targets and
never appear in the molecule or `additional_keywords`. This command does
not connect to the existing QCArchive deployment, register a specification,
or submit a job. Runtime declarations and the installed QCSchema artifact
round trip must be qualified first.
