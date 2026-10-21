# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""Fresh-output portable runner; no scientific package import or old evidence."""
import contextlib
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import unittest


def main():
    if len(sys.argv) != 2:
        raise SystemExit("one NEW output directory required")
    output = Path(sys.argv[1]).resolve()
    output.mkdir()  # Existing paths (including old receipts) must fail.
    assert sys.dont_write_bytecode and os.environ["PYTHONDONTWRITEBYTECODE"] == "1"
    assert resource.getrlimit(resource.RLIMIT_AS) == (8*1024**3,)*2
    assert resource.getrlimit(resource.RLIMIT_CORE) == (0, 0)
    assert all(os.environ[k] == "1" for k in (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"))
    with (output / "tests.log").open("x") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        import test_isapol_density_reference as tests
        suite = unittest.TestSuite()
        for case in (tests.FiniteDensityReferenceTests, tests.NativeWaterReferenceTests):
            suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(case))
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    source = Path(tests.reference.__file__).resolve()
    assert source == Path(__file__).resolve().parents[2] / "psi4/driver/procrouting/isapol_density_reference.py"
    assert not any(name.split(".")[0] in ("psi4", "adcc", "pylibxc") for name in sys.modules)
    fixture = Path(__file__).resolve().parent / "data_isapol/native_density_reference"
    paths = [source, Path(tests.__file__), Path(__file__), Path(__file__).with_name("run_stable_density_offline.bash"),
             fixture / "water.npz", fixture / "water.json"]
    record = dict(
        status="PASS" if result.wasSuccessful() else "FAIL", tests_run=result.testsRun,
        failures=len(result.failures), errors=len(result.errors), skipped=len(result.skipped),
        actual_numerical_module=str(source), no_quantum_imports=True,
        identity_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        interpreter=sys.executable, output_directory=str(output))
    with (output / "result.json").open("x") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
    print(json.dumps(record))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
