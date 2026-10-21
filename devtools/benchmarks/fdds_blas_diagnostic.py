"""CPU SVD/thread-library diagnostic, NOT a physical or FDDS benchmark.

Run separately on frozen base/candidate stages through the exclusive helper.
No library/thread setting is modified. Periodic Python stacks bound attribution
at the Python/native boundary; they cannot establish a native CUDA root cause.
"""
import faulthandler
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np  # Preserve the accepted NumPy-before-Psi4 load order.
import psi4


def report(label, **data):
    print(json.dumps(dict(label=label, monotonic=time.monotonic(), **data)), flush=True)


def state(label):
    libraries = sorted({line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                        if "/" in line and any(key in line.split()[-1]
                                              for key in ("libmkl", "libgomp", "libiomp", "libomp",
                                                          "libopenblas", "libcusolver", "libcuest"))})
    try:
        from threadpoolctl import threadpool_info
        pools = threadpool_info()
    except ImportError:
        pools = "threadpoolctl unavailable; no package installation attempted"
    report(label, libraries=libraries, threadpools=pools)


def main():
    # package/hash/driver pin are supplied by the frozen owner controller.
    package, expected_hash, output = sys.argv[1:]
    assert Path(psi4.__file__).parent == Path(package)
    core = Path(psi4.core.__file__)
    assert hashlib.sha256(core.read_bytes()).hexdigest() == expected_hash
    assert os.environ["SLURM_JOB_ID"] == "14250361"
    target = Path(output)
    target.mkdir(mode=0o700)
    psi4.core.set_output_file(str(target / "psi4.out"), False)
    with (target / "python-stacks.log").open("x") as stacks:
        faulthandler.enable(file=stacks)
        faulthandler.dump_traceback_later(20, repeat=True, file=stacks)
        try:
            report("START", package=package, core_sha256=expected_hash,
                   job_id=os.environ["SLURM_JOB_ID"], affinity=sorted(os.sched_getaffinity(0)),
                   environment={k: v for k, v in os.environ.items()
                                if k.startswith(("MKL_", "OMP_", "OPENBLAS_", "KMP_"))
                                or k in ("LD_LIBRARY_PATH", "CONDA_PREFIX")})
            psi4.set_num_threads(8)
            state("BEFORE_GPU_INITIALIZATION")
            rng = np.random.default_rng(812)
            for initialized in (False, True):
                if initialized:
                    a = np.eye(2)
                    result = psi4.core._cuest_fdds_dot3(a, a, a, 0)
                    assert result is not None
                    np.testing.assert_array_equal(result, a)
                    state("AFTER_REAL_GPU_ENDPOINT")
                for n in (512, 2048):
                    matrix = rng.normal(size=(n, n))
                    report("CPU_PINV_ENTER", shape=[n, n], gpu_initialized=initialized)
                    start = time.monotonic()
                    inverse = np.linalg.pinv(matrix, rcond=1.e-13)
                    assert np.isfinite(inverse).all()
                    report("CPU_PINV_EXIT", shape=[n, n], gpu_initialized=initialized,
                           seconds=time.monotonic() - start)
            state("END")
            report("DIAGNOSTIC_COMPLETED", not_fdds_or_speedup_evidence=True)
        finally:
            faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    main()
