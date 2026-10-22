# Phoenix benzene: corrected driver, accepted native build, no automatic retry

## Preserved history

- P-001/job14256310 stopped before SCF on an overly broad deleted-map guard.
  The reviewed narrow parser and `SystemExit` failure receipts remain intact.
- P-002/job14277082 ran 17m14 and converged its own spherical
  PBE0/aug-cc-pVTZ SCF: `-232.04232740170460 Eh`, 414 main functions. It then
  refused the AUX count before ISA/response. No actual AUX count was printed.
  Its source, run, scratch, own SCF/error and runtime/batch evidence are
  immutable. There is no new queued retry.
- Accepted native source is **15cc9a7a9a2bd60e43ebc2533ec99686bca58374**, tree
  **8f0aa8fabd476aafc6bc49b824c60ac935a86753**. Its immutable build-smoke-v3
  stage core SHA is
  **b283901ddac01e6bf7210a7461fb71cd5b2333cf9e42b99d60271959b40ac7eb**.
  Job14254599 passed five native water/angular/property tests. That acceptance
  does **not** cover the new adapter correction.

## Profile-faithful basis correction

Changed global `PUREAM=True` overrides `BasisSet.build(...puream=0)` in
`export_mints.cc`; the public Python helper documents this precedence.
Historical recipe56o constructed its AUX before applying SCF options. Our
P002 driver constructed it afterward, so the declared Cartesian AUX was
not protected from the spherical SCF option.

The new driver applies the original SCF options, admits the main basis, then
uses `p4util.OptionsState(["PUREAM"])` and `try/finally` to set global PUREAM
false **only around AUX construction**. It restores both value and changed
flag, even on error. It keeps that admitted AUX/recipe before the one SCF;
there is no post-SCF recipe rebuild. MAIN must remain spherical414; AUX must
be Cartesian1146/nao1146/252shells/maxAM4. Counts alone are insufficient.
No acceptance of912, nao substitution, new default, basis-file edit, native
package patch, tolerance change or rebuild is involved. `pilot.json` is
byte-identical to the prior scientific profile.

`basis_admission.py` records canonical source/stage GBS hashes and raw C/H
contraction tokens separately from ordered native normalized shell records.
Native values use exact binary64 `float.hex`; no normalization reconstruction
or float tolerance is invented. Source/core/fixture/config/driver identities
and both basis fingerprints must match a reviewed private native-probe
manifest **before SCF**. The actual wavefunction's MAIN fingerprint must then
match the admitted MAIN before any property call.

## Explicit two-source provenance

The corrected driver/helper live in a new exact-commit **science-launch-v3**
source. That is also the new dedicated basis-probe source. The native stage
and Python routing remain in **build-smoke-v3 at 15cc**, unchanged. Required
driver root/commit/tree and native root/commit/tree/core pins are distinct.
No old pilot monkeypatch, fake stage symlink or alteration of accepted build.
`build_smoke_plan.json` and its historical accepted-adapter hashes describe
the old build acceptance, not this correction; they are not rewritten and
no future build is authorized.

## Basis-only probe first — NOT executed by preparation

See `BASIS_ADMISSION.md`. One separate free embers probe, max10min,
2 allocated CPUs/8G, one thread, Psi4 memory4GiB, on the **same build node
atl1-1-02-007-23-1** because of `-march=native`. It imports the exact accepted
runtime but constructs only molecule/basis metadata. No SCF, density,
integrals, ISA iterations, response, PFIT or benzene properties. The expected
unscoped912 is a static prediction until measured there.

## P003 barriers

`benzene.sbatch` now describes **future** new P-003/science003 paths and a
corrected driver, but it is not authorized or prepared/submitted here. Its
four arguments are driver commit, tree, reviewed private probe-manifest path
and SHA256. The driver refuses to start without this baseline. Before P003:
review the native probe's exact shell/file/option/runtime evidence, accept
its hash-pinned manifest, then separately grant/preflight the full pilot.
No automatic continuation or resubmission.

Future scientific resources/science stay one own SCF/one ISA property chain,
6h/48G aggregate/24GiB numerical, one thread, same build node, unchanged
production LW/PFIT gates, optional rank<=2 mixed/recoupled exports. The
200GiB scratch budget is a declaration, not a fake hard quota/watchdog.
Failure stays nonzero with error/log/SCF and identity receipts; no gate
relaxation. Full pre-throw tensor capture and external anisotropic parity
remain absent, not blockers for the separately scoped metadata probe.
