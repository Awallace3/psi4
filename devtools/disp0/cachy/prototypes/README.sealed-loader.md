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
