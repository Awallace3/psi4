# Complete benzene PFIT declaration candidate

**Local engine-free preparation only. NO NEW ACTIVATION AUTHORITY.**
Job14397036 failed; its single-run approval is consumed. Neither these files,
the earlier angular instructions, nor a future scheduler test-only authorizes
another allocation. Explicit new user approval is required.

## New scientific identity

`pdef_plan.json`, `pdef_profile.py`, `pdef_driver.py`, `pdef_job.py` and
`pdef_full.sbatch` are separate opt-ins. Old defaults, angular files and every
production `psi4/` byte remain unchanged; no shared seam changes are needed.
The new driver uses the existing integrated own-SCF/ISA/response/LW/PFIT/export
pipeline and real pre-LW diagnostic sink. It is not a capture-only driver.

The full effective config is frozen `pilot.json` (SHA256
`19170c907137c7883dfbceaa4a23e4b919ec2d97dac38343dc4559b8bffb0f0b`)
plus Q200/974, numerical25GiB26843545600, and explicit
`refinement.declared_variables`. Relative to the frozen angular effective
config, **only the declaration is new**. Relative to the original pilot there
are three overrides, not two. This is a **new effective scientific model**,
not parity/equivalence with the angular model: an additional fitted free
parameter can change tensors and dispersion.

The generator is:

```python
tuple(f"{ref}_{COMPONENT_NAMES[i]}_{COMPONENT_NAMES[j]}_A"
      for ref in ("C1", "H1")
      for i in range(1, 9) for j in range(i, 9))
```

The frozen rank0..2 component order is
`00,10,11c,11s,20,21c,21s,22c,22s`; tests compare it with the real source.
There are exactly36/reference,72total, no00, all noncharge rank1/2 upper
triangle components. This is not the missing label alone or a four-node union.
The plan stores the complete list; profile validation requires equality to the
generator. Identity binds list/hash, generator source, effective config/hash,
plan/hash and driver/runtime sources. Future source binding additionally binds
commit/tree and all tools/routing hashes.

Allowance72 alone never defines variables. The static cutoff still selects
only penalty inventory, using C1/H1 reference ownership and unchanged COPY
equivalences. Current anchors/exact weights remain in use even on downward
crossings. Static-unselected declared extras remain **fitted FREE**, with
penalty anchor/strength zero, not fixed-zero coefficients. An upward crossing
is not newly penalized. Columns never adapt per node. The intended
current-node completeness guard and cutoff remain unchanged, through
`isapol_bounded.refinement_models`/`_refinement_node`, not the different direct
static-reference helper API.

All original physical settings remain: one numerical thread, ISA100/200,
Q200/974, response100/200, SCF75/302, PBE0/noGRAC, MAIN414sph/AUX1146cart,
original bases/frame/fingerprints, strict rank2 LW1e-6, work6.4e13, I/O64GiB,
maxparameters72, all11nodes, PFIT2000points, both all-pair passes, static
penalty mask, isotropic and full-tensor recoupled properties. Both actual
Psi4 memory and BoundedResources calls receive25GiB26843545600.
Original probe provenance is retained separately from derivative admission;
no private molecular geometry is committed.

## Acceptance is all-or-nothing

The inherited strict response, LW, native PFIT Solved, reciprocity, property
and final loader/source/runtime closure gates remain authoritative. The new
pre-export validator retains the angular completeness checks (all11records,
strict LW diagnostics, common static-reference hash,144ordered isotropic and
recoupled pairs, resource receipts) and verifies **actual**72labels, complete
COPY entries, reference ownership/ranks, declaration equality and identical
static penalty masks across all nodes, not just a metadata count.

Any scientific/closure exception remains nonzero with original traceback and
retained pre-LW captures; a separate closure error cannot erase the primary
failure. TERM/failure cannot mark complete. Partial capture is never acceptance.
Rank2 C10/unrestricted angular coverage remains honestly limited.

Job14397036 returned from strict LW at four reached nodes, but failed at node4
model construction. Logged first3 residuals were4.326238e-7,4.326151e-7,
4.323548e-7 below1e-6; node4's maximum was not logged. Actual failing post-LW
anchors were **not captured**. Four pre-LW bundles cannot resume PFIT/point
targets or prove remaining-node success. This candidate must start its own
complete calculation, only after fresh approval.

## Engine-free red/green and resources

Use the existing SciPy-equipped interpreter; no installation:

```bash
export PYTHONDONTWRITEBYTECODE=1
P=/home/awallace43/miniconda3/envs/p4_ci/bin/python
"$P" -B devtools/ccasp_phoenix_benzene/pdef_boundary.py --driver angular
# EXPECT EXIT1: intended omission H1_11s_21c_A, no declaration supplied.
"$P" -B devtools/ccasp_phoenix_benzene/pdef_boundary.py --driver pdef
# EXPECT EXIT0: synthetic constructor path only, NOT native acceptance.
"$P" -B -m unittest discover -s devtools/ccasp_phoenix_benzene -p 'test_*.py' -v
PYTHONPATH=tests/pytests "$P" -B -m unittest \
  test_isapol_bounded_campaign_source test_isapol_selection_source \
  test_isapol_scheme_source test_isapol_property_water_source -v
bash -n devtools/ccasp_phoenix_benzene/pdef_full.sbatch
```

The adapter reaches **actual wrapper/shared-driver kwargs**, then forwards
them to the real source-loaded bounded constructor; it never injects a
declaration. Synthetic12sites/11nodes include71static penalties, below/exact
cutoff→upward H1[3,5], downward crossing, and COPY disagreement. Tests check
same72labels/entries, surviving-current subset, exact anchors/weights/hash,
FREE nonzero design column and real streamed native declaration `fixed=False`.
Native PFIT dispatch is replaced by a refusing synthetic boundary, not a solve.
Malformed/duplicate/incomplete/out-of-rank/nonreference/lower-triangle inputs
still refuse. All old suites/refusal tests remain.

Resource code executes real `_pfit_plan(2000,108,72,11)` and source ISA
planner ASTs, and the unmodified real design-admission prefix with the actual
72-column/12site model. No large field allocation, large replay or native work:

| Numerical/source plan | Value |
|---|---:|
| Standalone ISA | 26333117936 bytes |
| Held buffers | 237797760 bytes |
| Total | 26570915696 bytes |
| 25GiB modeled headroom | 272629904 bytes |
| Actual72 PFIT stage | 326608384 bytes /94490544000 work |
| Packed used noncharge channels | 96 (108 including charge slots) |
| Two-pass replay bound | 60047184000 <64000000000 |
| Packed admission bytes | 198039872 |
| Fullchain source-work bound | 62337903183272 <64000000000000 |
| Checkpoint I/O | 6700030544 bytes |
| Diagnostic I/O allowance | 268435456 bytes |
| Total I/O | 6968466000 <68719476736 |

Fullchain work is recomputed from actual source ledger work expressions,
including ISA, factor/fits/kernel/assembly/stability, H2H1 operator/all11solves,
LW, targets, actual72 PFIT and diagnostic plan. The recoupling contribution
uses the conservative sum of its two enforced100M work ceilings (200M);
no native table query is performed. I/O recounts328writes/331reads with actual
`_Store` charge expressions and512-column factor tiles. These results happen
to equal earlier conservative fullchain constants, but are independently
recalculated here with72, not inherited from a71-column observation.
Tests cover default64 refusal, explicit72 admission,73 refusal, 24GiB
refusal/25GiB admission and exact byte/work/I/O/replay boundaries.
All memory claims are numerical plans, **not measured RSS**, scheduler memory,
timing, walltime or physical-accuracy guarantees.

## Future-only preparation, not executed

Only after independent review and separately authorized publication/staging:
stage the reviewed clean commit/tree in the new source path, never old sources.
Use concrete reviewed C/T:

```text
/storage/project/r-cs207-0/awallace43/miniconda/envs/p4cuest_sapt/bin/python3.13
  -I -S -B
  /storage/project/r-cs207-0/awallace43/gits/psi4.ccasp-benzene-full-pdef-20261025/devtools/ccasp_phoenix_benzene/pdef_job.py
  prepare --commit C --tree T
```

Exclusive new root:
`/storage/project/r-cs207-0/awallace43/runs/ccasp-benzene-full-pdef-20261025`,
`runtime-001`, `full-001`; scratch:
`/storage/scratch1/3/awallace43/ccasp-benzene-full-pdef-20261025-full001`.
Preparation creates log parents before scheduler access and validates original
probe plus derivative/source/runtime identities. Inspect the resulting
`full-001/metadata/preparation.json`; use **only its exact `test_only_argv`**:

```text
/opt/slurm/current/bin/sbatch --test-only --export=NONE --chdir=NEW_FULL_RUN
  NEW_SOURCE/devtools/ccasp_phoenix_benzene/pdef_full.sbatch
  C T GENERATED_RUNTIME_SHA GENERATED_DERIVATIVE_SHA
```

No fabricated manifest pins. No overwrite/cleanup/retry if preparation fails.
No prepare, test-only, staging, SSH or network command was executed here.
Proposed route remains gts-cs207-chemx/cpu-small/free embers,
8CPU48G6h, nodeatl1-1-02-007-23-1. All accepted native/dependency/interpreter/
Einsums preload/retained-handle and original probe/fixture pins are inherited.
This route is a proposal only; no fresh allocation is approved.

## Precise native acceptance gaps and next plan

**Unrun**: any native PFIT/LW test, actual native Solved status, full-cloud
two-pass execution, all11physical nodes, final tensors/properties, runtime
composition/admission, measured RSS and six-hour sufficiency. Unit tests do not
establish any of these.

After explicit authorization for native testing, first perform a focused
**synthetic** acceptance test in an independently admitted environment: use
the same12synthetic COPY sites/72declarations/11node crossing anchors, supplied
deterministic finite108-channel fields on a small24-or-more-point cloud with
verified full-column rank/conditioning, all packed pairs and two streamed
passes. Choose a nonzero known free-extra coefficient; form synthetic targets
from the design. Compare the unchanged default native solver to an independent
dense penalized reference with the exact current anchors/strengths, verify
every node `Solved`, 72unfixed variables, free-extra recovery, shared
static-reference hash/mask, COPY tensors and replay identity. Include refusal
cases and compare small dense/streamed routes within conditioning-aware
tolerances, without claiming molecular parity. This test needs **new explicit
approval**; no extra allocation or build is implicitly granted.

Only a separately approved complete new end-to-end run can then test original
basis/frame/closure, own PBE0 SCF, ISA/Q, all11strict response/LW nodes,
2000point/two-pass PFIT Solved and full isotropic/recoupled acceptance. Retain
real pre-LW captures, actual model inventories/masks/current penalties and
honest nonzero terminal receipts on any failure; do not resume old four-node
captures or retry automatically. Scientific acceptance requires all original
gates, additional validator, attributable exports and unsignalled zero
payload/child/batch plus final closure, not scheduler `COMPLETED` alone.
