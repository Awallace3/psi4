"""New staged ABI/path/mapping probe; no energy, density, or GPU prewarming call."""
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from policy import E, LIBS, SDK, XC, require, sha, write_json


def maps():
    paths = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        value = line.split()[-1]
        if value.startswith("/") and Path(value).is_file():
            paths.add(value)
    return {p: sha(p) for p in sorted(paths)}


def main():
    stage = Path(os.environ["MAJOR2_EXPECTED_STAGE"]).resolve()
    sys.argv.append("--einsums:no-attach-debugger")
    import psi4
    sys.argv.pop()
    require(Path(psi4.__file__).resolve().is_relative_to(stage / "lib/psi4"), "wrong arm/package import")
    require(Path(psi4.core.__file__).resolve().is_relative_to(stage / "lib/psi4"), "wrong arm/core import")
    require(hasattr(psi4.core, "release_freed_memory"), "merged core binding absent")
    require(psi4.addons("cuest"), "cuEST addon unavailable")
    loaded = maps()
    for p, h in LIBS.items():
        actual = str(Path(p).resolve())
        require(actual in loaded and loaded[actual] == h, "wrong/missing SDK/LibXC/Einsums mapping " + p)
    for prefix, parent in [("libcudart.so", E + "/lib"), ("libmkl_rt.so", E + "/lib"),
                           ("libstdc++.so", E + "/lib")]:
        found = [p for p in loaded if Path(p).name.startswith(prefix)]
        require(found and all(Path(p).is_relative_to(parent) for p in found), "wrong runtime " + prefix)
    write_json(sys.argv[1], dict(pid=os.getpid(), module=psi4.__file__, core=psi4.core.__file__,
                               core_sha256=sha(psi4.core.__file__), maps=loaded,
                               env={k: os.getenv(k) for k in ("LD_LIBRARY_PATH", "PYTHONPATH", "OMP_PROC_BIND",
                                                            "OMP_PLACES", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}))


if __name__ == "__main__":
    main()
