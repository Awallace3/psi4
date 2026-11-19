# Native DFHelper lifecycle regression

Build against a staged Psi4 package with its compiler and Python environment:

```sh
cmake -S tests/dfhelper-lifecycle -B /tmp/dfhelper-lifecycle \
  -Dpsi4_DIR=/path/to/stage/share/cmake/psi4 \
  -DCMAKE_PREFIX_PATH="/path/to/stage;$CONDA_PREFIX"
cmake --build /tmp/dfhelper-lifecycle
ctest --test-dir /tmp/dfhelper-lifecycle --output-on-failure
```

The test-only binding exercises `release_tensor()`, which is a C++ API.
Tests release one of two transformed tensors, transform the survivor again,
then release it and register a replacement. All three tensor layouts and
in-core/disk AO and MO storage combinations are covered.
The Python-exposed `release_AO()` cleanup regression is in
`tests/pytests/test_dfhelper_lifecycle.py`.
