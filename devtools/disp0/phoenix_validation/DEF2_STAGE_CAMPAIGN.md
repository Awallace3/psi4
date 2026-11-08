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
