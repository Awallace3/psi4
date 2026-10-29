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
