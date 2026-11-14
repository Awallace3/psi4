#!/usr/bin/env python3
"""Linux-only injected LAPACK workspace/failure regressions on a staged build."""
import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[2]
    tests = source / "tests/pytests/test_scf_portability.py"
    cases = [
        ("workspace", {"PSI_TEST_DSYEVD_MINIMUM": "1"}, "dsyevd_small"),
        ("eigen_fallback", {"PSI_TEST_DSYEVD_MINIMUM": "1", "PSI_TEST_DSYEVD_FAIL": "1"}, "dsyevd_small"),
        ("factorization_fallback", {"PSI_TEST_DPOTRF_FAIL": "1"}, "cholesky_matrix_equivalence"),
        ("conditioning_fallback", {"PSI_TEST_DPOCON_SMALL_RCOND": "1"}, "cholesky_matrix_equivalence"),
    ]
    # Python extension dependencies must be global for RTLD_NEXT in the probe.
    program = (
        "import os,sys; sys.setdlopenflags(os.RTLD_NOW|os.RTLD_GLOBAL); "
        "import psi4,pytest; psi4.set_num_threads(1); "
        "raise SystemExit(pytest.main(sys.argv[1:]))"
    )
    for name, variables, selection in cases:
        env = dict(os.environ, LD_PRELOAD=str(args.probe.resolve()), PYTHONSAFEPATH="1",
                   OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        for key in ("PSI_TEST_DSYEVD_MINIMUM", "PSI_TEST_DSYEVD_FAIL",
                    "PSI_TEST_DPOTRF_FAIL", "PSI_TEST_DPOCON_SMALL_RCOND"):
            env.pop(key, None)
        env.update(variables)
        subprocess.run(
            [sys.executable, "-c", program, str(tests), "-q", "-s", "-k", selection,
             f"--junitxml=scf-lapack-{name}.xml"], env=env, check=True,
        )


if __name__ == "__main__":
    main()
