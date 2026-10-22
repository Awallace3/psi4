# Benzene mixed-basis admission — prepared, no native execution

The basis-only source/probe must be reviewed and separately granted. Source/
mock checks are not native measurements.

## Static evidence

Canonical `aug-cc-pvtz-ri.gbs`:
`b657d18ba2a0931a337afd8e8bc7d11b047456d2871db9973d16aa5402c7cb41`.
C shells9s7p6d4f2g (136cart/106sph), H5s4p3d2f (55cart/46sph);
6C6H gives **1146cart / 912sph / 252shells**, maxAM4.

Canonical `aug-cc-pvtz.gbs`:
`235b70809d1f205815c72046ab7319b291c4fe826b93f83dea876502dbb0a53c`.
C5s4p3d2f and H4s3p2d give **414sph / 480cart / 138shells**, maxAM3.
These are contracted-shell combinatorial counts from source declarations.
They do not measure the actual P002 AUX (that was not printed).

The same source/stage GBS file hashes were verified read-only on Phoenix.
Changed global PUREAM trumps explicit puream0 in native construction. The
new AUX scope uses `OptionsState(["PUREAM"])`, finally restoration and exact
before/during/after option-state receipts. The scientific JSON is unchanged.

## Manifest and admission

`basis-admission.json` is a **private output of the future native probe**,
not a committed invented reference. It binds:

- native source15cc/tree8f0a/stage and accepted b283 core;
- separate corrected driver root/commit/tree and actual pilot/helper/contract/
  scientific-config file hashes;
- exact private fixture and source/stage GBS hashes;
- raw GBS tokens (not normalized) and ordered native shell/centre records
  (GaussianShell.coef normalized binary64 `float.hex`), representation/counts
  and exact fingerprints;
- the observed unscoped globalTrue/requested0 result and scoped1146 result;
- PUREAM value/changed-flag restoration, including controlled callback-error
  cases. No production function is monkeypatched.

The full driver requires its externally supplied SHA256 and matches exact
context/files/native shell snapshots before the sole SCF. Matching counts
with different coefficients/order/centres fail. It retains the admitted
recipe and compares the post-SCF MAIN exactly before the single property
chain. A refusal preserves useful actual metadata and option receipts.

## One proposed probe, not a full run

Source/root:
`/storage/project/r-cs207-0/awallace43/gits/psi4.ccasp-phoenix-benzene-pilot-20261021-science-launch-v3`.
New run:
`/storage/project/r-cs207-0/awallace43/runs/ccasp-phoenix-benzene-pilot-20261021/basis-probe-001`.
New scratch parent:
`/storage/scratch1/3/awallace43/ccasp-phoenix-benzene-basis-probe-001`.

`basis_probe.py prepare --commit COMMIT --tree TREE` creates only exclusive
log/metadata/scratch parents and metadata receipts. It reads the existing
private P002 fixture by exact SHA; there is no geometry publication/transfer.
`basis_probe.sbatch COMMIT TREE` is a future **00:10:00**, 2CPU/8G,
gts-cs207-chemx/cpu-small/explicit-embers job pinned to
**atl1-1-02-007-23-1**. No GPU, chain, retry, watchdog or lifecycle test.
TERM-safe live-child/wait status, absolute logs, job-owned scratch, numerical
threads1 and Psi4 memory4GiB.

The probe uses the accepted build-smoke-003 dependency-only prefix, exact
Einsums1.1.3 CDLL/hash, single libomp/GNU-MKL maps and isolated
QCElemental0.50.4/QCEngine0.50.0/OptKing0.5.0 with p4cuest_sapt Python3.13.11.
It reuses the reviewed narrow deleted-map parser and failure receipts.
No local/import/build fallback, package change or native rebuild.

After future execution, a complete manifest plus runtime/batch/probe receipts
requires independent review. Only then may a separate P003 grant be requested.
No manifest exists and no native probe or full pilot executes during this
preparation. Metadata admission is not binary/source attestation or parity.

## Engine-free checks

```
python3 -I -S -B -m unittest discover -s devtools/ccasp_phoenix_benzene -p 'test_*.py' -v
```

The original24 tests remain. Added tests use stdlib source parsing, fictitious
mock centres/bases and the actual OptionsState class logic on a fake core;
they do not import Psi4/NumPy, normalize real primitives, run native basis
construction or any molecular calculation. They exercise actual pre-SCF
driver statements with mocked energy and require energy never called on
count/representation/file/hash/binding/shell mismatch.
