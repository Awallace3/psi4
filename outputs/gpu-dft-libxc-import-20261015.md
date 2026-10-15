# cuEST comparison build: LibXC import failure

Failed build14056196 compiled successfully but failed the staged-import gate.
Dependent GPU job14056197 exited before chemistry because no successful
build-manifest was produced. No shared-grid results exist from these jobs.

## Reproduction

Run the staged Python import using the build's exact environment:

```
PYTHONPATH=<failed-build>/build/stage/lib
LD_LIBRARY_PATH=<cuda-libxc>/lib64:<cuda13>/lib64:<p4cuest_sapt>/lib
<p4cuest_sapt>/bin/python -c 'import psi4'
```

Actual result, reproduced on 2026-10-15:
`ImportError: core.cpython-313-x86_64-linux-gnu.so: undefined symbol: xc_func_init_flags`

## Ranked hypotheses and observations

1. A different LibXC with the same SONAME is selected at runtime.
2. The intended CUDA LibXC lacks the required API.
3. CUDA headers were used but the host target was linked.

`readelf -d core*.so` reports NEEDED `libxc.so.15` and **DT_RPATH**:
`$ORIGIN/../:<p4cuest_sapt>/lib:<cuda-libxc>/lib64`.
`ldd` resolves LibXC to the conda host library even with CUDA LibXC first in
LD_LIBRARY_PATH. DT_RPATH precedes that environment search path.

`nm -D` shows:

- CUDA LibXC exports both `xc_func_init` and `xc_func_init_flags`.
- Conda host LibXC exports `xc_func_init` but not `xc_func_init_flags`.

Changing only the import process's LD_PRELOAD to the absolute intended CUDA
LibXC library makes the **same unmodified failed core** import successfully and
expose `VBase.export_cuest_grid`. This is a causal diagnostic, not a production
workaround or a modification of the failed build/shared environment.

## Regression and fix scope

`devtools/benchmarks/gpu_compare/test_libxc_rpath.py` builds tiny host/CUDA fake
LibXC libraries with the same SONAME and uses the **actual core RPATH construction
block** from `psi4/src/CMakeLists.txt`. The CUDA library provides a flags symbol
that the host library lacks. Loading the resulting core must select the
explicitly linked implementation rather than the earlier generic dependency
directory.

Before the fix, this local regression reproduces the exact undefined-symbol
failure in approximately one second, including an imported, configuration-
specific LibXC target. The applied CMake fix prepends the selected CUDA LibXC
target's directory in core's RPATH. The same regression then passes; the complete
local harness suite passes 104 tests after the protocol review amendments.
CPU/static builds retain their prior
runtime-path order. A new real build must also
pass the original import/linkage gate; the synthetic regression is not a
substitute for that validation.

The revised benchmark separately measures worker launch-to-reap including
imports and runtime shutdown. The successful diagnostic import also exhibited
Einsums initialization/shutdown, illustrating why post-import SCF timing is not
the requested full-process metric.

The user approved one new CPU build (2h) and one H200 job (4h) on inferno for
the corrected build and stored/native-DF, matched-rank/shared-grid protocol.
No further paid recovery jobs are implicitly authorized.
