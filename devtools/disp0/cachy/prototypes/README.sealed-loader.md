# THROWAWAY: kernel-sealed ELF loader feasibility

Question: can the exact pinned cuEST0.2.50/CUDA13.3 bytes be copied into
irreversibly sealed memfds, loaded through a private alias directory with working
`$ORIGIN`, and proven by actual mapped inode/device/function ownership?

This is a real Linux/ELF compatibility probe, not a state-model simulation.
It is deliberately kept on `prototype/cuest-sealed-runtime`, outside the
qualified implementation branch. No production owner or admission check is
changed. No Psi4 import, CUDA context creation, GPU kernel execution or cuEST
API call is performed.

```bash
# Small local loader/kernel-seal probe; needs cc and Linux>=6.3.
CUDA_VISIBLE_DEVICES='' python3 -B \
  devtools/disp0/cachy/prototypes/sealed_loader_probe.py \
  --toy-only --output /tmp/psi4-sealed-loader-toy-FRESH

# On Cachy, inside the established4-CPU/12-GiB/no-swap/128-task CPU scope:
CUDA_VISIBLE_DEVICES='' "$ENV/bin/python" -B \
  devtools/disp0/cachy/prototypes/sealed_loader_probe.py \
  --root "$HOME/gits/psi4-disp0-perf-cachy" \
  --output "$HOME/gits/psi4-disp0-perf-cachy/runs/sealed-prototype-cpu-FRESH"
```

Each output must be fresh. The toy parent depends on a child found only through
`$ORIGIN`, not LD_LIBRARY_PATH or an explicit child preload. Both are sealed
before loading; original toy files are then removed and the loaded function must
still return7. Writes, resizing, shared writable maps, reopened-FD writes and
further seal changes must fail with EPERM—even though owner write permission is
still set. An initial local run passed with kernel seal mask63.

The first Cachy attempt stopped at import before creating snapshots: its conda
Python omits the memfd/seal attributes despite kernel6.18 and libc support.
The probe uses libc's declared `memfd_create` entrypoint when necessary and
stable Linux UAPI constants verified against the host headers. No package,
driver, sysctl or environment installation is changed.

The vendor probe reads only existing private durable inputs. It copies/checks
all17 pinned libraries and preloads the same nine libraries as the qualified
bootstrap. It verifies every required cuEST native/factory symbol against the
sealed snapshot's mapped inode/device and exact kernel pathname.

It also runs the **CPU-only NVRTC compiler** and retrieves PTX without loading
or executing it. CUDA12.9+ NVRTC may otherwise invoke `cuInit` for caching, so
both `CUDA_CACHE_DISABLE=1` and `-no-cache` are set. This exercises builtins and
loader resource lookup without invoking CUDA driver/runtime APIs.

All memfds disappear at worker exit; retained aliases are dead experiment
artifacts, not reusable dependencies. Builds/installs/environments and the
qualified `3ea43ee66e` source remain untouched. A CPU PASS would establish only
loader/seal feasibility—not a production identity contract, cuEST JIT/context
compatibility, GPU numerical parity, cleanup behavior or a speedup.

Primary references:
- https://man7.org/linux/man-pages/man2/memfd_create.2.html
- https://man7.org/linux/man-pages/man2/F_ADD_SEALS.2const.html
- https://docs.nvidia.com/cuda/nvrtc/index.html

## CPU verdict: feasible, not integrated

`sealed-prototype-cpu-02` passed on Cachy with prototype code
`689388fd63e1bf4cf9e164d124d0eba16fae0927`.
Result SHA256:
`0ddca4f8e23a3cbf69aa5936e393b9b9602612befe6f94cc9f37ea9c185d1923`.

- All17 exact pinned vendor files (1,756,549,808 bytes), plus two toy ELFs,
  were sealed with mask63.
- All nine requested vendor preloads succeeded through private aliases.
- All25 required cuEST native/factory symbols belonged to the sealed SDK
  snapshot's actual mapped inode/device/name.
- The `$ORIGIN` dependency probe returned7 after original toy files were removed.
- Writes/resizing/shared writable maps/reopened-FD writes/seal changes were
  rejected by the kernel for every snapshot.
- NVRTC13.3 compiled528 bytes of PTX with its cache disabled, without loading
  or executing that kernel. Eight explicit CPU compiler calls were made;
  zero explicit CUDA driver/runtime or cuEST API calls were made.

This settles the CPU loader/kernel-seal feasibility question affirmatively.
It does not yet test Python CUDA-binding lazy loading, actual cuEST context/JIT
initialization, GPU module loading/execution, or integration with the qualified
Psi4 loader/admission checks. Those remain separate gates. The qualified
implementation and its default admission contract have not changed.

## Next bounded GPU compatibility probe

User approved the separate `--gpu` mode. It requires the established resource
scope and common nonblocking GPU lock/HOLD monitoring. A fresh child starts with
only the sealed aliases and existing shared environment on its loader path.
It imports the exact existing CUDA13.3 bindings, forces their runtime/NVRTC lazy
loading, executes a one-element FP64 kernel (`2 -> 3`), then creates one cuEST
context with a borrowed stream, FP64 math and one JIT compiler thread.
Its HOME/cache is private to this experiment. All explicit native/device resources
are released, without resetting or releasing the runtime's primary context.

After lazy loads, cuEST initialization and cleanup, it rejects every toolkit
mapping not backed by the exact sealed snapshots. This is loader compatibility,
not a production identity owner, full SCF, numerical gate, or performance result.
No installed scientific module or qualified runtime configuration is changed.

```bash
# Only after GPU permission, within <=4CPU/12GiB/no-swap/128-task scope.
"$ENV/bin/python" -B devtools/disp0/cachy/prototypes/sealed_loader_probe.py \
  --gpu --root "$HOME/gits/psi4-disp0-perf-cachy" \
  --output "$HOME/gits/psi4-disp0-perf-cachy/runs/sealed-prototype-gpu-FRESH"
```

## GPU compatibility verdict: passed, still not integrated

`sealed-prototype-gpu-01` passed with code pin
`57c6af5366a32723ae9471021d866b24e8b1e5a9`.
Result SHA256:
`0c20aa4b9aa07d0189683a8edaa1552b07b1be1e39d9458da1f220cb91e1d42b`.

- CUDA Python bindings13.3.0 imported from the existing selected environment.
  Runtime/NVRTC lazy-loading calls succeeded against the sealed toolkit.
- Exactly one one-element FP64 kernel ran and returned3.0 from input2.0.
- One cuEST0.2.50 context was created with the borrowed stream and one JIT
  compiler thread. Native version and FP64 math-mode1 readbacks succeeded.
- At all recorded mapping checkpoints (binding import, lazy calls, kernel,
  context creation, cleanup), all nine mapped toolkit/SDK objects belonged
  to the exact sealed snapshots; no second/unsealed toolkit was observed.
- Synchronization, cuEST handle/parameter destruction, device-buffer free,
  module unload and stream destruction all succeeded. The runtime primary
  context was neither replaced nor reset/released.
- The service exited0 under verified4CPU/12GiB/no-swap/128-task limits, with
  host cgroup peak2,042,314,752 bytes. Device free-memory endpoints differed
  by2MiB; these are **not** an allocation peak or proof of zero total caching.

This settles the narrow CUDA-binding, tiny-kernel and cuEST-context loader
compatibility question. Native basis/DFMO construction and its lazy JIT paths,
whole Psi4 integration, a reviewed immutable-owner contract, numerical parity,
memory comparison and whole-call performance remain untested in sealed mode.
No production implementation/default was changed and no Phoenix job was run.

### Reviewed repeat

Independent inspection found two probe-only defects: cuDNN/CUPTI were omitted
from the foreign-toolkit classifier, and relative output paths were not
canonicalized before the worker changed directory. Code`6997a49701` fixes both.
Fresh CPU subprocess probes now reject real unsealed ELF mappings named for
either omitted family; the relative-output toy probe also passes.

The corrected GPU repeat `sealed-prototype-gpu-02` passed with full code pin
`6997a497016170254f7f681ec9e4fcf636a49194`. Its parent deliberately started
from a different directory with relative `--output`, exercising that fix too.
Result SHA256:
`cced9283e58983183b1628a6e8a3ebcb5a922534a078a74ef08be11afc7085d3`.
This is the current compatibility result, superseding attempt01's narrower
mapping classifier. The kernel result, native version/FP64 readbacks and all six
explicit cleanup steps passed again. The same scope limits and all limitations
above remain in force; it is still not a reviewed production owner or speedup.
