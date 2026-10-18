"""Immutable campaign policy. No Psi4 imports, network, scheduler writes or submission."""
import ast
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET

BASE = "b1e49472686c4e5843688a27d85b808375b90852"
OBSERVED_CANDIDATE = "6171b5901dbb43de5f19223cd08b13e45f9af742"
P = "/storage/project/r-cs207-0/awallace43"
E = P + "/miniconda/envs/p4cuest_sapt"
DEPS = P + "/gits/psi4.saptdft-cuest-lifecycle-20261003/build_lifecycle/stage"
RUN = P + "/runs/saptdft-cuest-cpks/major2-merged-b1e4947268-20261018-einsums113-gpumap-toolchain-wallcap"
H0 = P + "/runs/saptdft-cuest-cpks/scal631-fc851a8346-20261016"
SOURCES = {
    "baseline": P + "/gits/psi4.saptdft-cuest-major2-baseline-20261017",
    "candidate": P + "/gits/psi4.saptdft-cuest-major2-gpu-20261017",
}
HARNESS = P + "/gits/psi4.saptdft-cuest-major2-campaign-20261018-einsums113-gpumap-toolchain-wallcap"
SDK = P + "/software/cuest-0.2.50-a06e8148"
XC = P + "/software/libxc-7.1.2-cuda-sm80-89-90-120"
CUDA = P + "/software/toolchains/cuda13"
INSPECTION_TOOL = dict(
    path="/usr/local/pace-apps/spack/packages/linux-rhel9-x86_64_v3/gcc-12.3.0/cuda-12.6.1-fgbhrk26npivybjj65zswagcf4zv6p2n/bin/cuobjdump",
    sha256="1a53d40230259b7904d068305c9c3b368746423e973df34fba55c5e2f1d35996",
    version="V12.6.68", role="ELF/SASS inspection only; NOT compiler/headers/runtime")
LIBS = {
    SDK + "/lib/13/libcuest.so.0": "65c9e559201ee589cb742b919cc1c5d80c45aa480b64aede17ac96aff9e4963c",
    XC + "/lib64/libxc.so.15": "b8ba011064d688748fca2676801602d6fd8a51712f13235ba2b4511d2976cc98",
    DEPS + "/lib/libEinsums.so.1": "cf1f411130bc902a3ca74280cf23b3e7f325843de8cf38ec4a95fcb6f917faf2",
}
EINSUMS_INPUTS = {
    "share/cmake/Einsums/EinsumsConfig.cmake": "d444049359b1f2c70914898b1178cfb1e0eec61a543bda775b2ae6d530991f4d",
    "share/cmake/Einsums/EinsumsConfigVersion.cmake": "54346a6d32bfe31f2fff43f0d17f4e9ee2218e704a3328973c2779e4875dff86",
    "share/cmake/Einsums/EinsumsTargets.cmake": "c3105b37555a03750875582200da34653619e9d5b1e3a99de048e478c91dd6d8",
    "share/cmake/Einsums/EinsumsTargets-release.cmake": "35e9f27642efa2265389c52625be265b7fcbb4de598f1f576b49c91cacbba7a5",
    "share/cmake/Einsums/einsums_internal_targets.cmake": "9f9932853f6aea503cc51d71567438ee145df45b4a660365f256ea0290d943ee",
    "share/cmake/Einsums/EinsumsCacheVariables.cmake": "68b9f1f176aa8497f27f0b29dc8f608a8a10a25de485df5bb0acf236fc6b4a13",
    "include/Einsums/Config/Version.hpp": "69fc93bcf4b6baf954d1ac850a675357f47bcf2c6282bb8565466649cb99c7c4",
}
RUNTIME_PATH = SDK + "/lib/13:" + XC + "/lib64:" + DEPS + "/lib:" + E + "/lib"
SUITE_SHA = "72668390d24f134316c8e779931a2f65efcd9289ecea8f1cf20a72733262e3bb"
RUNNER_SHA = "06f7b9d47ed5723eb52f54c9142a40d9311cfe0a85313d38a449a1d6baf20a16"
BUILD_SHA = "e2f2991967c7a70c63f929ee1ac63eba3da4f13dcd40285851da9dcde33991ff"
SYSTEMS = {
    "peptide": [24, 250, 1010, 1042, 125, 125, "14b4041b272a66bea7daf3d010ffd695a6340fd554906b0d7309de4c80b2acab"],
    "nanotube": [42, 548, 2238, 2240, 56, 492, "5010ea2807d2f43d06290cf41b99e4890911d810cf818678616dffce98cb4f94"],
    "protein83": [83, 922, 3735, 3820, 480, 442, "c4bc7429189d820d4fd8990317ad7e930383576a638adfc2eb70561e9805ab48"],
    "protein157": [157, 1786, 7265, 7384, 1344, 442, "a1e9c8949bb5950fdb17c07ab718cead826835f042a6b6e94c5cb7dcf75b6b60"],
}
ROUTE = dict(account="gts-cs207-chemx", qos="embers", partition="gpu-h200",
             constraint="H200", gres="gpu:h200:1", cpus=8, memory_mb=131072,
             seconds=14400)
GATES = dict(component_abs_eh=1e-8, residual_max=1e-8, rhf_atol=1e-10,
             rhf_norm_scaled_rtol=5e-13, iteration_changes="STOP_FOR_REVIEW")
RHF = "test_rhf_finalize_weighted.py"
PLAN = "test_cuest_plan_reuse.py"
RED = PLAN + "::test_cuest_unchanged_initialize_reuses_plan"
COMMON = [
    RHF, "test_scf_lifecycle.py", "test_cuest_jk.py",
    "test_cuest_cphf.py::test_cpscf_separate_operator_projection",
    "test_cuest_cphf.py::test_cpscf_mixed_symmetry_batch",
    "test_cuest_cphf.py::test_cpscf_mixed_symmetry_solver",
    "test_cuest_cphf.py::test_cuest_cpscf_solver_residual",
    "test_cuest_cphf.py::test_vx_cuest_reference_reuse",
    "test_cuest_cphf.py::test_vx_cuest_cpks_solve",
    "test_cuest_cphf.py::test_vx_grid_switched_saptdft_cpks_induction",
    "test_dft_grid_switch.py::test_cuest_switching_and_microtiming",
    "test_dft_grid_switch.py::test_full_threshold_minimum_two_matches_final_grid",
]
# Separate invocation: explicit exclusions, NOT "-k hf" (matches the file name).
FIXED = "test_cuest_cphf.py::test_cuest_cpscf_fixed_orbitals"
FIXED_K = "not cam-b3lyp and not lrc-wpbe and not svwn and not blyp"
OWNED_DIFF = {
    "psi4/src/psi4/libscf_solver/rhf.cc", "psi4/src/psi4/libfock/cuESTJK.cc",
    "psi4/src/psi4/libfock/cuESTJK.h",
    "tests/pytests/" + RHF, "tests/pytests/" + PLAN,
}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    """Exclusive create: receipts and partial evidence are never overwritten."""
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def full_pin(pin):
    require(isinstance(pin, str) and re.fullmatch("[0-9a-f]{40}", pin), "full immutable commit SHA required")
    return pin


def check_source(root, pin, files=None):
    full_pin(pin)
    require(git(root, "rev-parse", "HEAD") == pin, "source HEAD pin mismatch")
    require(not git(root, "status", "--porcelain", "--untracked-files=no"), "dirty tracked source")
    if files is not None:
        require(file_manifest(root) == files, "tracked file bytes/modes differ from frozen manifest")


def file_manifest(root):
    """All tracked files, with SHA256 and Git mode; no .git, untracked build, or old data."""
    root = Path(root)
    entries = subprocess.check_output(["git", "-C", str(root), "ls-files", "-s", "-z"]).split(b"\0")
    result = {}
    for entry in filter(None, entries):
        header, name = entry.split(b"\t", 1)
        mode, _, stage = header.decode().split()
        require(stage == "0" and mode in ("100644", "100755", "120000"), "unsupported Git entry")
        p = root / name.decode()
        data = str(p.readlink()).encode() if mode == "120000" else p.read_bytes()
        result[name.decode()] = dict(mode=mode, sha256=hashlib.sha256(data).hexdigest())
    return result


def tree_manifest(root):
    return {str(p.relative_to(root)): sha(p) for p in sorted(Path(root).rglob("*.py")) if p.is_file()}


def sequence():
    blocks = [
        [("peptide", "baseline"), ("nanotube", "candidate"), ("protein83", "baseline"), ("protein157", "candidate")],
        [("protein157", "baseline"), ("protein83", "candidate"), ("nanotube", "baseline"), ("peptide", "candidate")],
        [("nanotube", "candidate"), ("peptide", "baseline"), ("protein157", "baseline"), ("protein83", "candidate")],
    ]
    rows = []
    for repeat, block in enumerate(blocks, 1):
        for system, first in block:
            for arm in [first, "candidate" if first == "baseline" else "baseline"]:
                rows.append(dict(arm=arm, backend="gpu", system=system, repeat=repeat,
                                 label=f"major2-{arm}-{system}-r{repeat}", smoke=False))
    return rows


def selectors_ast(root):
    """Fail before pytest if the reviewed function names disappear/change file."""
    receipts = {}
    for selector in COMMON + [PLAN, RED, FIXED]:
        file, _, function = selector.partition("::")
        text = (Path(root) / "tests/pytests" / file).read_text()
        tree = ast.parse(text)
        functions = [n.name for n in tree.body if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")]
        require(function in functions if function else bool(functions), "missing selector " + selector)
        receipts[selector] = dict(sha256=hashlib.sha256(text.encode()).hexdigest(), functions=functions)
        if file == RHF:
            receipts[selector]["matrix_assertion_tolerances"] = rhf_tolerances(tree)
    return receipts


def rhf_tolerances(tree):
    """Flag any post-review test-tolerance edit BEFORE a manifest/job can run."""
    tolerances = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "assert_allclose":
            kwargs = {k.arg: ast.literal_eval(k.value) for k in node.keywords if k.arg in ("atol", "rtol")}
            require(set(kwargs) == {"atol", "rtol"}, "RHF tests must declare both tolerances")
            tolerances.append([kwargs["atol"], kwargs["rtol"]])
    expected = [[2e-13, 2e-13], [2e-13, 2e-13], [2e-13, 0], [1e-12, 0], [1e-10, 0]]
    require(sorted(tolerances) == sorted(expected),
            "RHF tolerance policy changed: report/review difference before job; no weakening")
    return tolerances


def build_argv(source, pin, out):
    return ["bash", str(Path(source) / "devtools/cpks_gpu_build.sh"),
            str(source), E, full_pin(pin), str(out), "cuda"]


def einsums_coherence(prefix=DEPS):
    """Read-only config/header/imported-library agreement; never configure/install."""
    root = Path(prefix).resolve()
    config = root / "share/cmake/Einsums"
    version = (config / "EinsumsConfigVersion.cmake").read_text()
    header = (root / "include/Einsums/Config/Version.hpp").read_text()
    target = (config / "EinsumsTargets-release.cmake").read_text()
    includes = (config / "einsums_internal_targets.cmake").read_text()
    require('set(PACKAGE_VERSION "1.1.3")' in version, "Einsums config version mismatch")
    for name, value in [("MAJOR", 1), ("MINOR", 1), ("PATCH", 3)]:
        require(re.search(rf"^#define EINSUMS_VERSION_{name}\s+{value}\s*$", header, re.M),
                "Einsums headers/runtime version mismatch")
    require('IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libEinsums.so.1.1.3"' in target and
            'IMPORTED_SONAME_RELEASE "libEinsums.so.1"' in target and
            'INTERFACE_INCLUDE_DIRECTORIES "${_IMPORT_PREFIX}/include"' in includes,
            "Einsums imported target/prefix mismatch")
    library = (root / "lib/libEinsums.so.1").resolve(strict=True)
    require(library == root / "lib/libEinsums.so.1.1.3", "Einsums resolved runtime prefix/version mismatch")
    require(sha(library) == LIBS[DEPS + "/lib/libEinsums.so.1"], "Einsums dependency hash changed")
    inputs = {name: sha(root / name) for name in EINSUMS_INPUTS}
    require(inputs == EINSUMS_INPUTS, "Einsums config/header dependency hash changed")
    return dict(prefix=str(root), version="1.1.3", library=str(library),
                library_sha256=sha(library), inputs=inputs)


def check_runtime_maps(loaded):
    for p, h in LIBS.items():
        actual = str(Path(p).resolve())
        require(actual in loaded and loaded[actual] == h, "wrong/missing SDK/LibXC/Einsums mapping " + p)
    copies = [p for p in loaded if Path(p).name.startswith("libEinsums.so")]
    require(copies == [str(Path(DEPS + "/lib/libEinsums.so.1").resolve())],
            "multiple/wrong Einsums SONAME copies")


def runtime_env(parent, scratch):
    env = dict(parent)
    for key in ("GCC_ROOT", "CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH",
                "OBJC_INCLUDE_PATH", "LIBRARY_PATH", "PYTHONPATH", "PYTHONHOME"):
        env.pop(key, None)
    env.update(PATH=E + "/bin:" + CUDA + "/bin:" + env.get("PATH", "/usr/bin:/bin"),
               CUDA_HOME=CUDA, CUDA_PATH=CUDA, PYTHONNOUSERSITE="1",
               PYTHONDONTWRITEBYTECODE="1", GIT_OPTIONAL_LOCKS="0",
               LD_LIBRARY_PATH=RUNTIME_PATH,
               OMP_NUM_THREADS="8", MKL_NUM_THREADS="8", OPENBLAS_NUM_THREADS="1",
               TMPDIR=str(scratch), SCRATCH=str(scratch), PSI_SCRATCH=str(scratch),
               CPKS_CUDA_ARCHS="80;89;90;120", CPKS_CUDA_LIBXC=XC,
               CPKS_CUEST_SDK=SDK, CPKS_BUILD_DIR="build_major2_cuda",
               CUDA_CACHE_DISABLE="0", CUDA_CACHE_MAXSIZE="1073741824")
    return env


def admission(remaining, needed):
    require(math.isfinite(remaining) and remaining >= needed + 300,
            f"insufficient remaining wall: {remaining:.0f}s; need {needed}s + 300s teardown")


def cmake_policy(outer, inner, stage):
    def parse(text):
        return {m[1]: m[2] for m in re.finditer(r"^([^#/\n][^:\n]*):[^=\n]+=([^\n]*)$", text, re.M)}
    o, i = parse(outer), parse(inner)
    expected = {
        "CMAKE_BUILD_TYPE": "Release", "CMAKE_C_COMPILER": P+"/software/toolchains/conda-gcc",
        "CMAKE_CXX_COMPILER": P+"/software/toolchains/conda-g++",
        "CMAKE_CUDA_COMPILER": CUDA+"/bin/nvcc", "CMAKE_CUDA_ARCHITECTURES": "OFF",
        "CUDAToolkit_ROOT": CUDA, "Python_EXECUTABLE": E+"/bin/python",
        "Libxc_DIR": XC+"/lib64/cmake/Libxc", "Libint2_DIR": E+"/lib/cmake/libint2",
        "Einsums_DIR": DEPS+"/share/cmake/Einsums",
        "ENABLE_Einsums": "ON", "ENABLE_cuEST": "ON", "ENABLE_OPENMP": "ON",
        "ENABLE_dkh": "OFF", "ENABLE_XHOST": "OFF", "psi4_CMAKE_UNITY_BUILD": "ON",
        "LAPACK_LIBRARIES": E+"/lib/libmkl_rt.so",
    }
    for key, value in expected.items():
        require(o.get(key) == value, "outer CMake configuration: " + key)
    require(i.get("CMAKE_INSTALL_PREFIX") == str(stage), "inner stage install prefix")
    require(i.get("PSI4_CUEST_HAS_NONSYMMETRIC_XC") == "1", "inner nonsymmetric XC SDK support")
    for key in ["CMAKE_C_COMPILER", "CMAKE_CXX_COMPILER", "CMAKE_CUDA_COMPILER", "CMAKE_BUILD_TYPE"]:
        require(i.get(key) == expected[key], "inner CMake configuration: " + key)
    for key in ["CMAKE_C_FLAGS", "CMAKE_CXX_FLAGS"]:
        for c in [o, i]:
            require(CUDA+"/targets/x86_64-linux/include" in c.get(key, "") and
                    "-DPSI4_CUDA13_HEADER_ENV=1" in c[key], "CUDA13 include isolation: " + key)
    for c in [o, i]:
        for arch in ["80", "89", "90", "120"]:
            require(f"-gencode=arch=compute_{arch},code=sm_{arch}" in c.get("CMAKE_CUDA_FLAGS", ""), "CMake multiarch flags")
    for old in ["build_cpks_cuda_multi", "extern-b9032fdf25", "scal631-fc851a8346-20261016"]:
        require(old not in outer+inner, "old build cache leaked")
    return expected


def validate_manifest(m, executable=True):
    require(m["schema"] == 1 and m["route"] == ROUTE and m["gates"] == GATES, "campaign policy mismatch")
    require(m["run_root"] == RUN and m["harness"]["path"] == HARNESS, "wrong/new root or harness path")
    require(m["scratch_parent"] == "/storage/scratch1/3/awallace43/cpks-major2", "scratch policy")
    require(m["sequence"] == sequence() and len({v["label"] for v in m["sequence"]}) == 24, "cohort/arm/order")
    require(m["systems"] == SYSTEMS and m["suite_sha256"] == SUITE_SHA and m["suite"] == H0 + "/geometries", "geometry policy")
    require(set(m["arms"]) == {"baseline", "candidate"}, "exactly two GPU arms")
    require(m["arms"]["baseline"]["pin"] == BASE, "baseline pin")
    for arm, item in m["arms"].items():
        full_pin(item["pin"])
        require(item["source"] == SOURCES[arm] and item["backend"] == "gpu", "wrong arm/source/backend")
        require(item["build"] == SOURCES[arm] + "/build_major2_cuda", "wrong build stage")
        require(item["files"] and digest(item["files"]) == item["files_digest"], "source files manifest")
        require(item["files"]["devtools/saptdft_cpks_suite_run.py"]["sha256"] == RUNNER_SHA, "runner pin")
        require(item["files"]["devtools/cpks_gpu_build.sh"]["sha256"] == BUILD_SHA, "build helper pin")
    full_pin(m["harness"]["pin"])
    require(digest(m["harness"]["files"]) == m["harness"]["files_digest"], "harness manifest")
    require(m["common_selectors"] == COMMON and m["fixed_selector"] == FIXED and m["fixed_k"] == FIXED_K, "test selectors")
    require(m["dependency_hashes"] == LIBS, "dependency hashes")
    require(m["inspection_tool"] == INSPECTION_TOOL, "inspection tool pin")
    require(m["runtime_policy"] == dict(threads=8, memory="100 GiB", cache="enabled-shared-job-1GiB",
                                      prewarm=False, print_override=False, affinity_override=False), "runtime policy")
    if executable:
        require(m["approval"]["state"] == "APPROVED" and m["approval"]["review_id"].strip(), "candidate review NOT approved")
        require(m["approval"]["candidate_pin"] == m["arms"]["candidate"]["pin"], "approval for another pin")


def check_junit(path, collected, exitcode, expected_red=False):
    """Every collected required case must execute; skips/xfails/errors cannot go green."""
    root = ET.parse(path).getroot()
    cases = root.findall(".//testcase")
    require(collected and len(collected) == len(set(collected)), "no/duplicate collection")
    require(len(cases) == len(collected), "case execution/collection mismatch")
    require(all(not c.findall("skipped") and not c.findall("error") for c in cases), "skip/import/setup error")
    failures = [f for c in cases for f in c.findall("failure")]
    # The plugin records phase outcomes under exact nodeids; JUnit names are cross-checked.
    expected_names = [x.rsplit("::", 1)[-1] for x in collected]
    require(sorted(c.attrib["name"] for c in cases) == sorted(expected_names), "JUnit identity mismatch")
    if expected_red:
        require(exitcode == 1 and len(cases) == len(failures) == 1, "baseline red must be exactly one assertion failure")
        require(expected_names == ["test_cuest_unchanged_initialize_reuses_plan"], "wrong expected-red selector")
        text = failures[0].text or ""
        require("_creations(trace) == 1" in text and "assert 2 == 1" in text, "not duplicate plan-creation red")
        require("ImportError" not in text and "ModuleNotFoundError" not in text, "import is not red")
    else:
        require(exitcode == 0 and not failures, "required compiled correctness failed")
    return dict(executed=len(cases), failures=len(failures), skips=0)
