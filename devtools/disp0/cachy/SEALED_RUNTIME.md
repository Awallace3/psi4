# Opt-in sealed runtime — small-water gate passed, performance still incomplete

The latest accepted restricted small-water energy calls take26.2–26.7s;
a fresh whole worker takes37.631s including startup and cleanup.
This is **RTX3090 host-overhead debugging**, not an H200 speedup measurement.
The previous lease owner took42–44s per energy call and made163,164 complete
17-file lease checks. See [the historical ledger](HOT_PATH_ADMISSION.md).

## Contract

`process-owned-kernel-sealed-snapshots-v1` is a separate adapter, not a weakened
lease checker. Fresh-byte and kernel-read-lease selections remain available.

- A normal import of the coherently installed, stdlib-only
  `psi4_runtime_support` package prepares snapshots **before** Psi4/core/vendor
  imports. No scientific module is injected into `sys.modules` or overlaid.
- Preparation copies opened source descriptors, checks source stability/SHA256,
  installs irreversible WRITE/GROW/SHRINK/SEAL/EXEC seals, then hashes the sealed
  destination again. The complete snapshot allocation is bounded to2GiB.
- The parent creates a fresh owned0700 diagnostic directory, **excluded** from
  the loader path along with original SDK/toolkit directories. First loads use
  only `/proc/self/fd/N` for the verified snapshots, never mutable aliases.
  Bounded ELF parsing obtains DT_NEEDED/SONAME from sealed bytes. Pinned
  dependencies are loaded recursively and must bind their exact SONAME through
  `RTLD_NOLOAD`; unknown non-base dependencies and non-ORIGIN search paths reject.
  The small readonly CPU/OS base remains outside the toolkit snapshot contract.
- Snapshot descriptors and loaded handles remain owned until process exit,
  independently of session/plan cleanup. There is no close, transfer, renewal,
  fresh-hash fallback or contract switching.
- Once loaded, the backing inode is immutable even if its original file, alias
  or descriptor is removed. `check_live()` therefore checks caller/lifetime
  state in constant time, not filesystem metadata or revocable SIGIO leases.
  Unloaded snapshots still require descriptor/seal validation before loading.
- The scientific selector adopts only the exact prepared object retained by
  the support package. Both preparation and selection share the admission lock;
  legacy-session selection during preparation/handoff is rejected.
- Published proof failures terminate the dedicated worker without native
  unwinding. Ordinary pre-publication input/copy/seal failures roll back.

## Mapping obligations are retained

Every existing factory/provider mapping audit still runs. The sealed parser
translates only exact admitted kernel-name/inode/device triples to logical
source locators; no stale parsed rows survive a call. Foreign/unsealed toolkit
images, including cuDNN/CUPTI, and unrelated deleted shared libraries reject.
The legacy parser's deleted-library rejection is unchanged.

Bootstrap checks all25 canonical native/factory symbol owners before core import
and again after it. Later native context/version/FP64, device, buffer, numerical
and cleanup checks remain in place. No loader epoch or cached audit substitutes
for the retained fresh scans.

The whole-build proof includes the complete support-package inventory in
addition to scientific Python/core/dependency inventories. Parent-created
per-worker aliases and cache homes are separate from durable SDK/build inputs.
Sealed mode needs no reusable tmpfs dependency copy to work around Btrfs mapping
identities: its executable inodes are the newly verified memfds.

## Validation and remaining gates

CPU fixtures exercise real kernel seals and tiny ELF images, source changes,
bounded copying, rollback, foreign mappings, normal early support import/adoption,
contract/thread/fork misuse and absence of file IO on loaded-image hot checks.
These fixtures are not CUDA/cuEST/device evidence.

The complete host run passed373+33+109=**515 tests**, without skips.
Review reproduced a first-load alias defect in the initial candidate: replacing
a dependency alias executed a foreign constructor before any mapping rejection.
That regression was run red, then green after descriptor-bound loading; primary
alias substitution is also covered. The alias-based candidate is not accepted.
The separate descriptor-only vendor CPU probe passed on Cachy with all nine
sealed preloads and NVRTC compilation, without a GPU call:
`sealed-descriptor-cpu-01/result.json`, SHA256
`d30c4a723fe0aa81ef44cbdb5283c6d741e7b188c9c7d7b6646125dc1a7496a8`.
Code is captured on `prototype/cuest-sealed-runtime` at`8fd153b61b`.
The bounded descriptor-only GPU repeat also passed one FP64 kernel (`2 -> 3`),
cuEST0.2.50 FP64 context construction and every explicit cleanup step:
`sealed-descriptor-gpu-01/result.json`, SHA256
`0e19acc6cc5cba9cfaa5ced72283a91b9a2048bfc3aec623b179d5dda82b2b38`.
Independent follow-up inspection closed the constructor-time finding with no
further concrete finding in that scope. The GPU repeat is still a throwaway
loader compatibility probe, not qualification of this whole installed adapter.

Before accepting a speed improvement: normally build/install a separate whole
Cachy prefix, freeze its CPU proof, run the same construction and ten-process
water CPU/GPU/oracle/displacement/repeat gate, and compare before/after artifacts
at unchanged tolerances. Record snapshot host memory/startup separately from
whole-energy time and tracked device memory. No Phoenix submission or H200
performance claim is implied.

The first whole prefix,`install-owned-sealed-a7ca4a86a9`, built successfully and
preserved every prior prefix's`BUILT_FROM`. Target CPU tests stopped before
qualification: conda's`cc` injected an absolute RPATH into the tiny test ELFs.
The fixtures now explicitly use`/usr/bin/cc`; production loader checks are not
relaxed. The corrected local fixture/bootstrap/argument suites passed37 tests.
A fresh coherent prefix/CPU proof is required for the corrected source pin.

`install-owned-sealed-7047c13de2` then passed68 target CPU tests and whole-prefix
CPU qualification (`CPU_RHF_Eh=-1.1167593073781577`). Configuration SHA256:
`33e11e7804acb6a93d8a582942c675e9c4d2a01bd4b1e19415ddc097484f3652`.
Both construction/JIT arms passed, but the first full-water GPU process stopped
at an additional point-potential symbol audit that still called the legacy
deleted-library parser. No whole-water result or speedup from that attempt is
accepted. Its outputs remain under`sealed-full-water-gate-01`.

Additional point-potential, overlap and cuSOLVER symbol binding now uses a
shared fresh audit with the exact selected owner, including mapped-inode and
symbol-address checks. Legacy behavior stays strict; sealed failures retire the
worker rather than unwinding native owners. Real sealed-ELF binder regressions
were run red for all three consumers, then green; a foreign function pointer
also verifies fail-stop retirement. The host suites passed375+33+109=**517 tests**.

## Completed small-water gate at`86b31abd59`

The fresh whole prefix`install-owned-sealed-86b31abd59` passed70 real target
CPU tests and CPU RHF qualification. `sealed-full-water-gate-02` then passed
all ten fresh CPU/GPU/oracle/displacement/repeat processes. All GPU cases
performed actual native/device work and completed cleanup.

| Diagnostic energy-call wall (s) | Previous lease owner | Sealed owner |
|---|---:|---:|
| Unscaled | 43.728 | 32.365 |
| Scaled | 42.472 | 32.259 |
| Displaced | 44.433 | 31.605 |
| Fresh scaled repeat | 43.482 | 31.381 |

These are **RTX3090 host-overhead diagnostics**, not H200 FP64 speedups or
cache-matched production benchmarks. New workers use private native cache homes
and disable CUDA caching. No numerical policy, workspace/reserve or factor-cache
budget changed. Tracked device peaks match the baseline exactly:88,794,624 bytes
for unscaled/scaled/repeat and88,793,600 for displaced geometry.

The separately timed, unprofiled scaled worker took**43.171s end to end**,
including interpreter, snapshot/bootstrap/import work, science, normal output
and native cleanup; its energy call took31.767s. Its strict fresh-repeat check
passed. There is no matched historical whole-worker measurement, so no
end-to-end ratio is inferred from the energy-call table.

Snapshot preparation costs2.887–3.290s per GPU worker, with1,756,549,808 host
bytes retained and34 whole-file hashes (source plus sealed destination).
All17 files have seal mask63. Startup preloading costs0.064–0.072s. The full
service's host cgroup peak was3,640,930,304 bytes under the existing12GiB limit.
Complete execution still makes about177,000 constant-time live checks and6,273
mapped-byte receipts; reducing repeated mapping work remains a measured target,
not permission to cache an unobserved loader state.

Construction/cleanup takes8.174/8.214s for JIT1/4 (previous9.695/9.906s).
Whole construction children take16.101/16.173s; preparation cannot be omitted
when assessing the remaining latency. Tracked device peak stays33,757,184 bytes,
with zero live bytes/native owners after teardown.

The offline before/after comparison passed all23 fields in each of four GPU
cases at unchanged1e-9 tolerances, with only orbital-label/sign matching.
Maximum field error was1.245e-11 Eh. It also checks complete installed Python
and support inventories, respective core proofs, unchanged CUDA/SDK bytes,
native policies, device/driver/PTX, actual work, device peaks and full cleanup.
Historical configurations are evidence only, never reactivated.

Under the private performance root's`runs/`:

| Artifact | SHA256 |
|---|---|
| `sealed-owned-cpu-03/runtime-config.json` | `25a2c2cefa68e4f507853e5c9dc454b0783082e6bcd7accccb32469a634e4256` |
| `construction-sealed-water-02/comparison.json` | `eca3927916510ccf6733e55d42b29e16416f171826dbfc999129edd477a67f7d` |
| `sealed-full-water-gate-02/gate-result.json` | `67a946394d7b1c90bd3aaa570ab6265e6bcbb95e443fa03dd241143e4809ad6a` |
| `sealed-gpu-launch-02.json` | `5c0fffa3c9db1d0374ed84628b08cbed429a7cd471b0a2229080bdd229d0a4c5` |
| `sealed-before-after-01.json` | `87217339b7b1fd70fe38bd4c2e157bb6dd54345105d0e86e8d661429399154a5` |

This is still restricted small-water evidence. Defaults are unchanged, broader
edge cases are not released, and no Phoenix job/H200 measurement was made.
An external caller profile is the next step before another optimization.

## Profile-guided follow-up: literal mapping names

`sealed-profile-water-01` passed its strict fresh-repeat comparison. Its
profiled durations are **not** accepted timing samples and overlap:

| Profiled scope | Calls | Inclusive seconds |
|---|---:|---:|
| Constant-time `check_live` | 177,277 | 0.292 |
| `mapped_rows` | 737 | 17.184 |
| `_audit_maps` | 692 | 23.629 |
| Python3.13 `Path.name` | 1,182,910 | 9.748 |

Direct callers attribute829,012 `Path.name` operations to the sealed parser
and323,844 to identity classification. This supports optimizing literal name
parsing, not deleting any mapping audit. Profile result SHA256:
`c8b7b51ec2b402ce9d9f6933cc862ba24d342d8fd4b547ca89d9c3873521eac4`.

The follow-up uses a stateless POSIX final-component operation for normal
kernel filenames, preserving `Path.name` fallback behavior for empty/dot final
components. Only sealed-mode classification changes; every fresh snapshot,
segment, mapped inode/device and symbol check remains. No resolved-path memo,
mapping receipt, filesystem or loader-state cache is introduced. Equivalence
and no-normal-case-Path tests join the full376+33+109=**518-test** host pass.

## Completed follow-up gate at`0433ec0452`

Science/helper pin:`0433ec0452679cfc2ec2f7226ae941c0fc6719ef`.
The separate whole prefix`install-owned-sealed-0433ec0452` passed71 target CPU
tests and CPU qualification. The ten-process water gate, paired construction
probe and extra inclusive scaled repeat all passed. No scientific overlays,
changed tolerances, omitted mapping audits or cache-budget changes were used.

| Diagnostic energy-call wall (s) | Lease owner | First gated sealed owner | Literal-name follow-up |
|---|---:|---:|---:|
| Unscaled | 43.728 | 32.365 | 26.444 |
| Scaled | 42.472 | 32.259 | 26.477 |
| Displaced | 44.433 | 31.605 | 26.725 |
| Fresh scaled repeat | 43.482 | 31.381 | 26.196 |

The separately timed fresh scaled worker took**37.631s end to end**, including
snapshot/bootstrap/import work, science, output and cleanup; the energy call
inside took26.405s. The preceding sealed iteration took43.171s end to end.
These are individual diagnostic observations, not a cache-matched statistical
benchmark. The latest inclusive repeat passed all23 fields at unchanged1e-9
tolerances; its largest field error was2.087e-12.

Construction/cleanup takes6.281/6.228s for JIT1/4, with whole children taking
14.282/14.200s. This does not establish a JIT-thread advantage. Construction
tracked device peak remains33,757,184 bytes. Whole-energy tracked peaks remain
88,794,624 bytes (88,793,600 for displaced geometry), exactly matching the lease
baseline; all native owners and live tracked bytes are released. These tracked
peaks do not measure opaque native internals or total desktop GPU usage.

The strict offline comparison against`3ea43ee66e` passed all23 fields in each
of four GPU cases, with orbital-label/sign matching only and no energy
rotations. Maximum field error was1.205e-11. Complete installed/support
inventories, respective core proofs, identical CUDA/SDK bytes, device/driver/PTX,
actual device work and full cleanup were also checked.

Under the private performance root's`runs/`:

| Artifact | SHA256 |
|---|---|
| `sealed-owned-cpu-04/runtime-config.json` | `4034046cb74140a7ade016e8cd290d61fa316f405cccccee1c238bd8f8d6b187` |
| `construction-sealed-water-03/comparison.json` | `80fd74d3611760e9c1da2445da90bbb03dbb13bf7ec64a640b1b75b4d4b7c39e` |
| `sealed-full-water-gate-03/gate-result.json` | `2ed367bc2940ffedf941c095405de2f6e35de038d79395dcb1c272c85f359b8c` |
| `sealed-gpu-launch-03.json` | `9c0fcfbacdf86c2343324a4e744cda55d065d843f4e43af3dad1d081be07cfc9` |
| `sealed-before-after-02.json` | `ae48fc4ffe32fe0ecd01c8ec34c65bd13300821f1dbff7a096a40fd38a63e9a1` |

Performance remains incomplete. Fresh mapping scans, canonical metadata
verification and startup remain costs; do not replace them with an unobserved
loader-state cache. This remains opt-in restricted-water evidence, not broader
edge-case qualification or a default change. No Phoenix job was submitted.
