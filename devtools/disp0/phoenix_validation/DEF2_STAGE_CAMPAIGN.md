# Matched def2-basis F-SAPT comparisons

The user authorized def2 basis sets to unblock protein comparisons. Start with
**Protein83/def2-SVP**, then evaluate def2-TZVP only after measured headroom.
Protein157 remains excluded. This changes the benchmark's basis, not its
accuracy guards or the existing aug-cc-pVDZ results.

## Explicit basis profiles

| Profile | Primary | SCF/SAPT fitting | MP2 fitting |
|---|---|---|---|
| aug-cc-pvdz (legacy) | aug-cc-pvdz | aug-cc-pvdz-jkfit | aug-cc-pvdz-ri |
| def2-svp | def2-svp | def2-universal-jkfit | def2-svp-ri |
| def2-tzvp | def2-tzvp | def2-universal-jkfit | def2-tzvp-ri |

Counts below come from the actual repository spherical basis files and the
frozen timing-suite compositions, not retained-MO-rank assumptions:

| System | Atoms | aug-cc-pVDZ NBF | def2-SVP NBF | def2-TZVP NBF | def2 JKFIT functions |
|---|---:|---:|---:|---:|---:|
| peptide | 24 | 356 | 210 | 394 | 1010 |
| nanotube | 42 | 742 | 444 | 902 | 2238 |
| protein83 | 83 | 1293 | 766 | 1473 | 3735 |

For Protein83 the SVP RI fit has2488 functions; TZVP RI has3624.
**def2-TZVP has more primary functions than aug-cc-pVDZ here**. Its smaller
JKFIT basis alone does not establish lower peak memory or admission.
Native queried workspaces and actual device-free-memory checks remain decisive.

## Frozen inputs and matching

`prepare_inputs.py --basis-profile def2-svp|def2-tzvp` writes a new schema2
manifest containing the profile, literal geometries, hashes of primary and
both fitting files, counts and unchanged fragment/electron metadata.
Legacy default inputs keep schema1 and the original basis names/counts.
`full_inputs.py` rejects unsupported profiles, mixed fitting names, changed
filenames and changed per-case dimensions. Before scientific execution,
the worker checks every installed basis file against its frozen hash.

`fragment_gate.py` selects basis options from that admitted manifest for both
native CPU and cuEST GPU arms. No functional-group assignment files are
needed. The independent fixed-GPU-gauge CPU oracle additionally requires
identical scientific options, including all basis names. Source pin, runtime
closure, retained ranks, matrix comparisons, signed components, repeat
checks and scratch/retirement requirements remain active.

The method is still **scaled HF-reference F-SAPT**, FP64, exact coupled
response. GRAC and DFT-reference stages remain unexecuted. No energies
or speedup ratios are compared across different basis profiles.

## Launch and acceptance

Changes to authenticated execution helpers require a new clean frozen source
and exact whole-stage CPU build/runtime-admission proof. Do not overlay
`fragment_gate.py`, advance an old frozen checkout, relabel an old build
proof or modify the shared environment. `stage_scaling.py` now binds to the
exact full source pin from that newly qualified runtime config, and checks
the clean source and proof against that pin at submission and execution.

Start with one Protein83/def2-SVP H200 producer, on
`gts-cs207-chemx / gpu-h200 / embers`, oneH200,8hostCPUs,160GiB host,
eight-hour cap. Keep96GiB Psi4 host workspace,120GiB declared device ceiling,
12GiB native allowance, block24 and64MiB cache. Run the original small
physical gate, three fresh baseline processes with strict repeats, then the
separate synchronized stage trace. Failure stops the producer without retry.

Only after the producer and cleanup pass, run three independent matched
24-core CPU reference/oracle allocations. The def2-SVP Protein83 CPU pilot
is bounded at eight hours per allocation; actual completion is not promised.
The known over-eight-hour aug-cc-pVDZ CPU route stays blocked, as does the
larger def2-TZVP Protein83 CPU route pending a separate resource plan.
No inferno submission is authorized.

Qualification records the basis profile in the result and checks actual
worker primary/fitting options against the frozen inputs. The separate
traced GPU sample never enters the ratio of three per-arm baseline medians.

Local generated input candidates:
`/tmp/fsapt-def2-svp-inputs-01/inputs.json` and
`/tmp/fsapt-def2-tzvp-inputs-01/inputs.json`.
These are input preparation, not submitted jobs, native admission or timings.

## CPU prerequisite submitted

Frozen source/helper pin:`1cad631980626825918d058fd93d77455090b091`.
Phoenix checkout:
`/storage/project/r-cs207-0/awallace43/gits/psi4.disp0-def2-1cad631980`.
The committed `psi4/` and `psi4_runtime_support/` trees are unchanged from
the previously qualified927283f8ed source; this is still a new whole-build
proof, not reuse or relabeling of the old proof.

CPU-only build/qualification job **15317825** was submitted successfully on
`gts-cs207-chemx / cpu-small / embers`,8CPUs,48GiB, three-hour cap.
Controller readback and its own prolog confirm embers and RUNNING.
Test-only request15317824 is not the submitted job.
Root:
`/storage/project/r-cs207-0/awallace43/runs/saptdft-cuest-disp0/phoenix-def2-build-1cad631980`.
Manifest SHA256:
`42d5a906eae8de4d8e5af13ecfbe441f05bcc0bcc72d4f3f1d84550bd76de0ce`.
Hourly watcher:`b5300e303`.

The local basis/profile/source-pin helper suite passed142tests. These are
host-only checks, not native/GPU correctness evidence. The independent
launch-safety review remains separate from the CPU prerequisite. No def2
H200 producer or CPU protein reference has been submitted yet.

SVP frozen data was copied (not execution code) to:
`/storage/project/r-cs207-0/awallace43/runs/saptdft-cuest-disp0/benchmark-inputs-def2-svp-e93a40f2ca96/inputs.json`.
SHA256:
`e93a40f2ca961461cb98ba423b8eff5f199d7b201a0925185c80fff610dfe327`.

## Launch-review admission gap closed

Independent review found that the newly generalized stage launcher accepted
a matching PASS whole-build proof without explicitly requiring the separate
sealed CPU runtime-admission receipt. The CPU build already produces that
receipt, but its existence/success must be a launch requirement, not an
assumption inferred from the source pin.

The outer launcher now requires the finalized CPU prerequisite exit0/no-TERM,
including the build wrapper's `CPU_BUILD_VALIDATION_ONLY_CHECK_RESULT_JSON` marker,
reuses the frozen source's `require_cpu_admission` checks, binds its admitted
runtime-config SHA to the proof, and compares the entire runtime closure
against the requested config (only threads and per-run alias parent may
differ). It seals the admission receipt/config hashes into the submission
manifest and revalidates them at execution before the small H200 gate.
A regression exercises the actual submission path and confirms that even
`sbatch --test-only` is not called for a PASS build with FAILED admission.

This correction is a separately pinned **outer launcher** change. The
running15317825 source/build at1cad631980 stays frozen; no installed science,
build proof, worker or shared dependency is edited or relabeled.

Corrected outer launcher pin:
`387b36b92b7d05175c5a4c0ad90b0b3086f4fa6f`, frozen on Phoenix at
`/storage/project/r-cs207-0/awallace43/gits/psi4.disp0-def2-launch-387b36b92b`.
All143final host helper tests passed. A bounded read-only preflight against
the original927283f8ed CPU admission succeeded with receipt SHA256
`c1f1d57993c807f82a3a93ff63447fd24d7da89cb179ad1dd0f7be350f4c42fe`
and admitted config SHA256
`7ff292aae7a4e34b13653e735fa74e65a38957527bf85aef2cb10c899578d840`.
This verifies compatibility of the new admission requirement with an actual
finalized proof; it is not admission for the new1cad631980 build or def2 GPU
correctness evidence. H200 submission still waits for15317825's own completed
whole-build and sealed-runtime-admission artifacts.

## New prerequisite passed; Protein83/def2-SVP submitted

CPU15317825 completed job/batch/step0:0 in10min01s with no TERM and the
expected build-only marker. Its exact1cad631980 whole-build proof is PASS,
all383core CPU tests passed, and142helper tests passed with one explicit
optional skip (`Literal input suite not installed`). The separately frozen
SVP manifest was validated on Phoenix and binds Protein83 to83atoms,
766AOs, occupied78/68, frozen20/19 and active58/49. The sealed-runtime
admission receipt is PASS, with reaped workers and no CUDA session; this
is not GPU correctness evidence.

| CPU prerequisite artifact (under `job-15317825/`) | SHA256 |
|---|---|
| `result.json` | `277e50b8461aeb952890b3a868ef8356681a521ec3bf477ad1de0ac748d36a6a` |
| `gpu-runtime/preparation-result.json` | `d858eadf5c35d60e5dc68178421d5d20da18706bf71372b32909440490e4c121` |
| `cpu-sealed-admission/preflight-result.json` | `45b1da425583628e92909026f286f6ab4f1e045d4ce8de528368630972836c04` |
| `cpu-sealed-admission/runtime-admission.json` | `881f35c052dd7d46b44c47ec4d3adb633f534a42c958c0bca892fe662035de79` |
| `cpu-sealed-admission/runtime-config.json` | `894a91bca74747fb633b888d7e33d6759417a8f9dd2cf93318de1732a883ca44` |

The corrected387b36b92b outer launcher accepted and sealed that admission,
then submitted **15318800** successfully. Root:
`/storage/project/r-cs207-0/awallace43/runs/saptdft-cuest-disp0/def2-stage-scaling-1cad631980-01/protein83-def2-svp-gpu`.
Effective controller route:
`gts-cs207-chemx / gpu-h200 / embers`, one typedH200 and H200 constraint,
8hostCPUs,160GiB host allocation, eight-hour cap. Initially PENDING for
Priority;15318799 is only the scheduler test-only request.

- Manifest SHA256:
  `e3f2d34098ebde4b2562019ae4d29605d2e3c62a8a4104b7457ba315e0449f3e`.
- GPU runtime config SHA256:
  `c58337bf988b01132f1679a53da53452c0d59adaa1bd729ec05427885f057f0a`.
- Hourly official watcher:`bd3d5e3a5`.

The full small physical gate precedes three fresh strict-repeat baselines
and one separate traced process. No CPU protein reference is submitted
until this producer passes and cleans up. There is no def2 timing, GPU
correctness PASS or accepted protein speedup at submission. No old worktree,
proof, memory budget or correctness tolerance was changed; no automatic retry.
