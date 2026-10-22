# Phoenix-only benzene P pilot — prepared, NOT authorized to execute

**Current correction:** see `BASIS_ADMISSION.md` and `SCIENCE_JOB.md`. The
readiness-only description below is historical preparation context. Native
build15cc/v3 later passed five small smokes; P002 converged its own SCF then
refused AUX admission before ISA. The corrected driver/helper now have a
separate source pin from that unchanged accepted native stage. A native
basis-only probe and its reviewed exact-shell manifest are required before
any separately granted P003. No old build review is claimed for this correction.

Candidate base: `6f147f4829ffc83c868b9e78618ebe7c4f17fbdb`.
This directory changes no inherited scientific source, fixture or default.
The new adapter is statically inspected only: **no scientific imports, builds,
calculations or submissions have been performed for this pilot**.

## Binding and scope

`pilot.json` binds the recovered `bz-alg0` scientific profile (SHA256
`8b421b5a85d6c3b42676e5f724d89d3206aa18874300366ddf6493645dc13ff1`).
The historical recipe56o SHA256 is
`b6d0bb1c500ebb1f3a86f6cfa5aa0957be3163dabd7331aa75ddb1b6d256df38`.
Recipe/controller declarations are transcribed into a small current-API adapter,
not imported from that scratch tree. Its published `always` tail semantics are
retained, NOT the historical proposed `when_enabled` experiment.

First science: **one P-only native ISA benzene monomer pipeline**, own SCF and
state, current integrated source/core, production LW tolerance `1e-6`, PFIT
explicit allowance 72 (package default 64 unchanged). Current integrated
static-reference PFIT selection is disclosed, not overridden: this cannot be
compared as the old campaign1 PFIT algorithm. No full CamCASP arm, synthetic
benzene-pair test, old SAPT study, gate relaxation or automatic retry.

The exact external fixture is `benzene_s66x8_24_monA.json`, SHA256
`dbfd1bfcb832ed28c94b486962c253f1e3fb8295617557d11e5cdf17631bb8dd`,
from S66x8 index 3806, monomer A of `24_Benzene-Benzene_pi-pi_1.00`.
It contains public dataset coordinates and local source attribution;
redistribution permission has not been independently verified. **It is
deliberately not committed.** Selectively stage only this small hash-pinned input after
coordinator permission; never transfer the dataframe, archived arrays, reports,
old runtime, source bundles or credentials. The supplied bohr dimer frame and
the C/H permutation are exact. Nearly-D6h geometry is not idealized.

## Outputs and known missing adapter

`pilot.py` uses public `psi4.oeprop` / `psi4.atomic_property_result` and current
recipe declarations. Before imports, it checks the Phoenix identity, requires
a SLURM job ID, rejects login hostnames, and requires the executing hostname
to match `SLURMD_NODENAME`. These are accidental-mislaunch checks, not proof of
authorization; the eventual scheduler wrapper must verify its allocation.
The eventual job must initialize PsiAPI from its **own** staged executable and
bind p4qca Python explicitly, after independent runtime smoke admission.
Required arguments pin source commit, stage, core SHA256, fixture and new
output/scratch directories; the parent scratch directory must already exist.

On success: `local-response.npz` (all unrefined LW and refined local response
blocks, mixed rank included), `properties.json` (frequencies/CP weights in NPZ,
sites/frames/ranks/units, 144 ordered isotropic pairs per C6/C8/partial C10,
COPY types, actual static-reference hashes, optional native recoupled
Cn(t,u,J) and all coverage records), SCF record, Psi4 log and final result.
Recoupled output is explicitly requested through the existing public keyword
and charged to the same ledger. It is NOT an oriented energy, full C10, or
anisotropic parity with the scalar historical comparator. Keep per-site values
and their intra-type spread; never collapse them to exact D6h.

On a production refusal: nonzero driver exit, `result.json` with the original
error/traceback, `psi4.out` and own SCF record if reached. A refusal is useful
pilot evidence, **not science PASS**. Missing terminal result after scheduler
termination is incomplete evidence, not completion.

**Genuine missing adapter:** the all-or-nothing public oeprop result does not
expose ISA/Q/fit/H2H1 or pre-throw LW tensor state on failure. This minimal
driver does not claim to capture it, monkeypatch internals, or resurrect the
NOT READY Phase-0 capture/watchdog framework. Full pre-admission capture would
need a separately reviewed observer/export seam. No second reporting-only ISA
run is made outside the cumulative ledger.

## Proposed resources — grant still required

Two separate allocations, no automatic dependency chain:

1. Build/smoke: `gts-cs207-chemx`, explicit `qos=embers`, CPU-only
   `cpu-small`, 1 task / 8 allocated CPUs / 48 GiB / up to 8 h,
   private build and stage. This is a conservative unmeasured build allowance,
   not a timing promise. Recheck exact dependency/runtime isolation first.
2. Benchmark: same account/QoS/CPU shape, **6 h**, only **one numerical thread**,
   24 GiB Psi4 numerical memory, 48 GiB total including dirty page cache;
   work ceiling `6.4e13`, checkpoint I/O 64 GiB, 200 GiB scratch budget,
   require 400 GiB free before starting. Eight allocated CPUs keep 48 GiB near
   6 GiB/core (profile's credible cpu-small route); they do not authorize eight
   numerical threads.

Historical P: 6398 s to static-node-1 refusal, 22.08 GB RSS, 61.82 GB scratch.
There are no native P timings beyond node 1. Full eleven-node completion in
six hours is unmeasured. Do not enlarge limits or relax gates to finish.
The scratch budget here is a declared operational budget, not a filesystem
quota or a newly implemented polling watchdog.

Before any granted run: commit reviewed SBATCH wrappers here, based on the
skill TERM-safe template, scheduler walltime only; absolute output paths and
precreated parents; no watchdog/signals experiments or afterok chain.
First build smoke must identify package/core and actual mapped dependencies
from Phoenix, never assume workstation ABI/backend equivalence.
Then inspect smoke artifacts before separately authorizing/submitting science.
No `sbatch --test-only` or submission has been executed under readiness scope.

Static-only checks (safe without scientific imports):

```
python3 -I -B devtools/ccasp_phoenix_benzene/test_source_contract.py -v
```

These five source-contract checks do not execute the adapter and are not
runtime validation. Read-only static review found and corrected the public
coefficient completeness field name, the LW rank-3 padding/metadata mismatch,
and the allocation-host guard. Engine-free JSON comparison also checked the
recovered profile and exact fixture permutation; the response AUX spelling
was made byte-equal to the profile (basis names are case-insensitive).
