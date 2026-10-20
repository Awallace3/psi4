# Bounded private developer INPUT provision

This is the already-cleared ONE input preparation attempt, not CPU submission
or GPU readiness. It does not replace approved CPU recipe
`89b6160126d876b2ad649e6cf2a22f01f4b300f7`.

**That clearance is now consumed:** tool8b34329bee failed before header
materialization at its zstd window guard. Preserve `...inputs-8b34329bee-01`.
The corrected window constant and real compressed-frame regression are
LOCAL preparation only. Do not run the command below again without fresh
explicit authorization and another absent prefix.

The separate input-tool commit must be immutable/pushed and staged by Git in a
fresh private source root. Run from that exact source checkout. The destination
must be absent beneath an owned, non-group/world-writable parent.

```bash
P=/storage/project/r-cs207-0/awallace43
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
timeout --signal=TERM --kill-after=5s 110s "$P/miniconda/bin/python" -B \
  devtools/disp0/provision_sdk_inputs.py \
  --source "$P/software/toolchains/cuda13" \
  --destination "$PWD/private-inputs" \
  --vendor-environment "$P/miniconda/envs/p4cuest_sapt" \
  --lock "$P/runs/saptdft-cuest-disp0/build-587e172a33afaadb00eb7f574a5089533f7fc755-14196330/environment-explicit.txt"
```

The helper enforces one CPU affinity,1GiB address-space,110s alarm/CPU bound;
the outer timeout additionally kills subprocesses after5s. No automatic
retry or fallback package/build is permitted. Preserve any partial prefix.
Actual provision must leave aggregate login work under120s; stop rather than
stretching the deadline if full hashing cannot fit.

Exactly two public developer archives are pinned in the helper (82569bytes
combined). It validates SHA/size/index, safely streams headers and checks
packaged genuine developer-link targets. No package manager/solve/install
is invoked. Complete existing include contents are copied, never hardlinked;
only missing package headers are added. Bin/NVVM/lib64 remain read-only
locators. New input `lib/` devlinks are separately bound in the provision
receipt: unchanged CPU89 `inspect_sdk` inventories `lib64`, not input `lib/`.
CPU89 later supplies its own six genuine private `lib64` aliases.

`sdk-manifest.json` comes from unchanged `runtime_contract.py inspect_sdk`;
`provision-receipt.json` binds source/package URLs, hashes, versions, archive
contents, new-header hashes, developer links, exact old dependency lock,
helper/tool pin, byte/wall limits and read-only dependency closure.
Neither artifact establishes compiler-discovery, installed runtime/API,
physical numerical or device correctness. In particular, symlinked nvcc
may report an original include root: the CPU89 hook will reject that root
if outside the copied input tree. Do not fabricate nvcc output or silently
patch a wrapper. Report this limitation in input-readiness review.

Return both manifest hashes, prefix/receipt, closure and exact lock digest
to parent for explicit **INPUT-READINESS CLEARANCE**. Conditional CPU resources
remain unconsumed until that clearance and subsequent exact source/route/
headroom/storage/quota/no-duplicate guards. No independent GPU request.

## Consumed input-only CPU approval — FAILED14215786

Exactly ONE CPU-only input-prep job:1node/1task/1CPU/2G/00:10:00,
free embers/gts-cs207-chemx, cpu-small subject exact-route test-only,
explicit no-requeue/no retry. Worst case1/6CPU-hour,1/3GiB-hour,zero
embers charge. New source/input/run/scratch roots; no reuse or modification
of failed sources/prefixes. Scope only the two exact developer archives,
private input tree, complete SDK metadata hashes/ELF/closure and explicit-lock
proof. No solve/E2/Psi4/scientific build/SCF/SAPT/device work.

Archive download82569bytes, newly written input data<=512MiB; corrected
decoder window128MiB and process address cap1GiB. Retain the existing
110s payload/5s kill bound (job10min includes bootstrap/evidence capture),
preserving any failure. This is separate from and does not consume/expand
the conditional8CPU/48G/2h CPU89 build approval.

Parent approved exactly this ONE input-only job, now consumed.
`input_prep.sbatch`/`input_prep_payload.sh` enforce route/resources,
TERM-safe status, fresh roots and no-requeue. At start the helper requires
actual zstandard0.19.0 and verifies an8MiB-window streaming frame: the old
131072 bound must reject; explicit128MiB bound must decode8192bytes.
This happens BEFORE any real package download/extraction, without version
override/unbounded decoder/fallback. The receipt binds that actual proof.
Local version-metadata injection tests are explicitly not remote proof.
Actual remote0.19 proof PASSED inside14215786; the job then failed at the
overly strict immediate-link/SONAME comparison. Both SDK manifest and receipt
are absent. Preserve the job/root/status as failed. No corrected provisioning
attempt/job/GPU step is approved. CPU89 remains unconsumed/NOT GO.

## LOCAL archive-layout regression and component bindings

Both exact public archives are validated in memory, without filesystem
extraction or runtime use:

```bash
/home/awallace43/miniconda3/envs/p4_einv2/bin/python -B \
  devtools/disp0/audit_sdk_archives.py
```

Exact official topology (each also has `targets/.../lib/UNVERSIONED` pointing
directly to the same full versioned filename):

```text
lib/libcusparse.so
  -> ../targets/x86_64-linux/lib/libcusparse.so.12.8.2.51
lib/libnvJitLink.so
  -> ../targets/x86_64-linux/lib/libnvJitLink.so.13.3.33
```

These terminal binaries are external to the developer archives. A raw link
target is not an ELF SONAME. The virtual resolver follows at most16 links,
normalizes only within an archive namespace, and rejects cycles, escapes,
broken chains, unexpected regular intermediates or an archive shadowing the
external terminal. It never dereferences archive links on the host.

Parent supplied approved component-file bindings from the frozen completed
14185006 manifest, SHA
`3030cecda5562f1716a8df262f4f05386427ed4aab70cbda7d65000dc25a1ad1`.
Canonical source root is
`P/miniconda/envs/p4cuest_sapt/targets/x86_64-linux/lib`.
File SHA/required ELF SONAME:

* `libcusparse.so.12.8.2.51`:
  `cadbe1c78feae59b7448dde354bc3c1a60c3da32e0f5902780842f14d475d405`,
  `libcusparse.so.12`.
* `libnvJitLink.so.13.3.33`:
  `f84abd240b428825c3ef16632deb1026b208252fe19cef225a9f4ad06a39d1aa`,
  `libnvJitLink.so.13`.

The local audit binds virtual terminals to those approved expectations; it
does NOT reverify remote bytes. Before any future use, the helper separately
requires an actual regular terminal at the exact approved canonical root,
matching file SHA and ELF SONAME. No host path is chosen from archive link
text. Component provenance is not full SDK/E2/runtime/device readiness.

## Narrow next-step proposal — NOT authorized

Only after final local guards, BOTH real-archive regression and narrow review
are green and committed/pushed: request fresh approval for exactly ONE
INPUT-ONLY CPU job with the SAME1node/1task/1CPU/2G/10min,
freeembers/gts-cs207-chemx/cpu-small subject fresh exact-route test-only,
no GPU/no-requeue/no retry. Maximum1/6CPU-hour and1/3GiB-hour; $0 embers.
Retain110s payload/5s kill/1GiB process/128MiB decoder bounds, same two
82569byte archives and<=512MiB new data. Fresh Git source/run/input/scratch
roots; preserve both failed input attempts and all failed builds.

At start require actual0.19 frame proof, exact lock, actual approved component
file SHA/ELF identities and correct SDK-selected canonical terminals. Then
validate BOTH real archive layouts, copy the private include tree, create
only genuine developer links and produce complete SDK identity/closure/
provision receipts. Recheck approved component identities against the final
SDK inventory. No dependency solve/install, scientific/native compilation,
binary patch, SCF/SAPT/device work or shared root mutation.

Fresh source/test-only/headroom/quota/inode/no-duplicate guards precede any
new submission. This proposal is not a retry authorization or CPU89/GPU GO.
Even complete input metadata will retain the unresolved nvcc/copied-header
discovery caveat; parent INPUT-READINESS clearance remains mandatory before
the conditionally approved CPU89 build. No new allocation has been requested.
