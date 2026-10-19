"""Stdlib-only module environment proof, never loads scientific/device code."""
import json
import os
import sys
import common as n


def main():
    before = dict(os.environ)
    n.check_env(os.environ)
    n.require(before == dict(os.environ), "guard must not mutate ENV")
    n.require(os.environ.get("CUDA_ROOT", "").startswith(
        "/usr/local/pace-apps/spack/packages/linux-rhel9-x86_64_v3/gcc-12.3.0/cuda-12.6.1-"),
        "observed site module CUDA_ROOT")
    n.require(os.environ.get("CUDA_HOME") == os.environ.get("CUDA_PATH") == n.CUDA,
              "explicit CUDA13 locators")
    cleared = ("GCC_ROOT", "CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH",
               "OBJC_INCLUDE_PATH", "LIBRARY_PATH", "PYTHONPATH", "PYTHONHOME")
    n.require(all(k not in os.environ for k in cleared), "original arm isolation")
    forbidden = [k for k in sys.modules if k.split(".")[0] in
                 ("numpy", "psi4", "torch", "cupy", "cuest")]
    n.require(not forbidden, "stdlib-only proof")
    print(json.dumps(dict(
        locators={k: os.environ[k] for k in ("CUDA_ROOT", "CUDA_HOME", "CUDA_PATH", "PATH")},
        checked_cuda_keys=sorted(k for k in os.environ if k.startswith("CUDA_")),
        loaded_modules=os.environ.get("LOADEDMODULES", ""),
        cleared_variables=list(cleared), numerical_overrides={}, environment_unchanged=True,
        forbidden_imports=forbidden, common_sha256=n.sha(n.HERE/"common.py"),
        module_only=True, no_psiapi=True, no_compile=True, no_device_work=True,
        actual_toolchain_identity="not inferred from module locators; allocation checks authoritative"
    ), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
