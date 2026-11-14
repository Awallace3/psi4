# Native integral-clone lifecycle regression

Build this test-only module against a staged Psi4 package, using the same compiler
and Python environment as that package:

```sh
cmake -S tests/libint2-clone-workspace -B /tmp/clone-test \
  -Dpsi4_DIR=/path/to/stage/share/cmake/psi4 \
  -DCMAKE_PREFIX_PATH="/path/to/stage;$CONDA_PREFIX"
cmake --build /tmp/clone-test
ctest --test-dir /tmp/clone-test --output-on-failure
```

The probe compares original, cloned and twice-cloned engines before and after
manual sieve initialization. It covers two-, three- and four-center integrals,
screening modes, Coulomb derivatives, and parameterized operator families.
The capped retained-wavefunction benchmark is the memory regression test; this
small native test checks reconstruction correctness, not an RSS threshold.

The reversed three-center `xx_xs` cases are explicitly skipped: the existing
baseline aborts inside normal `compute_shell()` for that layout, before any
lazy reconstruction. They are not evidence of coverage for this change.
Yukawa cases are also skipped because the baseline constructor starts its
timer twice and throws before returning an engine. Its reconstruction recipe
is source-reviewed, not runtime-validated by this test.
