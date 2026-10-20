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

## Bounded alternative input-preparation proposal — NOT approved/submitted

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
the conditional8CPU/48G/2h CPU89 build approval. It requires fresh explicit
authorization, source pin and exact-route/storage/headroom/no-duplicate
checks. No new resource request has been submitted.
