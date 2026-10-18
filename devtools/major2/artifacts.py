"""Read-only adoption of the two explicitly approved builds, never a resume."""
from pathlib import Path
import re

from policy import *

REVIEW_ID = "parent-verified-artifact-pair-reviewID"
# Filled from actual read-only collection, before imports or any speedup view.
PAIR_DIGEST = "c83f605a6ca0a6249c98ef33b907105bd111b1336cfb8c6ab6a4930cbd145a6d"
OLD = P+"/runs/saptdft-cuest-cpks/major2-merged-b1e4947268-20261018-einsums113"
ALLOWLIST = {
    "baseline": dict(source=SOURCES["baseline"], pin=BASE,
        build=SOURCES["baseline"]+"/build_major2_cuda_armenv",
        metadata=OLD+"-gpumap-toolchain-wallcap-armenv/build-baseline", origin_job="14160948",
        core_sha256="f8155af7bbde540cb58a55806bdf1e10f4946da83f5d3bc0f38e9c1b4d45d3ff"),
    "candidate": dict(source=SOURCES["candidate"], pin=OBSERVED_CANDIDATE,
        build=SOURCES["candidate"]+"/build_major2_cuda_mapproof",
        metadata=OLD+"-gpumap-toolchain-wallcap-armenv-mapproof/build-candidate", origin_job="14161975",
        core_sha256="35ba91ed34da3cfa4e488d232858f7057871d0653376536e29c11f9be8c14fa1"),
}


def validate_pair(pair):
    require(pair["mode"] == "verified-artifact-pair" and pair["review_id"] == REVIEW_ID,
            "explicit reviewed artifact mode required")
    require(set(pair["arms"]) == set(ALLOWLIST), "artifact pair arms")
    for arm, allowed in ALLOWLIST.items():
        require(all(pair["arms"][arm].get(k) == v for k, v in allowed.items()),
                "artifact not the allowlisted build: "+arm)
    require(digest(pair) == PAIR_DIGEST, "immutable artifact pair digest")


def hashed_tree(root):
    return {str(p.relative_to(root)): sha(p) for p in sorted(Path(root).rglob("*")) if p.is_file()}


def normalized_configs(item):
    """Compare EVERY cache key/type/value; normalize ONLY exact source/build paths."""
    root = Path(item["build"])
    result = {}
    for name, file in [("outer", root/"CMakeCache.txt"),
                       ("inner", root/"psi4-core-prefix/src/psi4-core-build/CMakeCache.txt")]:
        text = file.read_text().replace(item["build"], "<BUILD>").replace(item["source"], "<SOURCE>")
        rows = re.findall(r"^([^#/\n][^:\n]*:[^=\n]+)=([^\n]*)$", text, re.M)
        require(rows and len(dict(rows)) == len(rows), "invalid/duplicate CMake cache keys")
        result[name] = dict(rows)
    return result


def normalized_compile_commands(item):
    file = Path(item["build"])/"psi4-core-prefix/src/psi4-core-build/compile_commands.json"
    text = file.read_text().replace(item["build"], "<BUILD>").replace(item["source"], "<SOURCE>")
    commands = json.loads(text)
    require(isinstance(commands, list) and commands, "artifact compile database")
    return sorted(commands, key=lambda row: json.dumps(row, sort_keys=True))


def verify_shared(pair):
    validate_pair(pair)
    for path, expected in pair["shared_inputs"].items():
        require(sha(path) == expected, "artifact shared toolchain input changed: "+path)
    for key in ["pyeinsums_core", "origin_host_receipt"]:
        require(sha(pair[key]["path"]) == pair[key]["sha256"], "artifact "+key+" changed")
    for path, expected in LIBS.items():
        require(sha(path) == expected, "artifact dependency changed")
    check_runtime_aliases()
    einsums_coherence()


def verify_artifact(pair, arm, files):
    validate_pair(pair)
    item = pair["arms"][arm]
    check_source(item["source"], item["pin"], files)
    require(digest(files) == item["source_files_digest"], "artifact whole-source digest")
    build, meta = Path(item["build"]), Path(item["metadata"])
    require(build.is_dir() and not build.is_symlink(), "approved build absent/retargeted")
    require(hashed_tree(meta) == item["metadata_files"], "artifact metadata bytes/file set changed")
    require((meta/"BUILD_OK").read_text().strip() == item["pin"]+" cuda", "artifact BUILD_OK")
    for path, expected in item["build_files"].items():
        require(sha(build/path) == expected, "artifact compiler config changed: "+path)
    stage = build/"stage"
    require(hashed_tree(stage) == item["stage_files"], "artifact stage bytes/file set changed")
    require(sha(stage/"lib/psi4/core.cpython-313-x86_64-linux-gnu.so") == item["core_sha256"],
            "artifact core SHA")
    drivers = check_installed_drivers(stage, item["source"], files)
    require(len(drivers) == 143, "approved installed driver count")
    inner = build/"psi4-core-prefix/src/psi4-core-build"
    cmake_policy((build/"CMakeCache.txt").read_text(), (inner/"CMakeCache.txt").read_text(), stage)
    return normalized_configs(item)


def verify_pair(pair, arms):
    verify_shared(pair)
    configs = {arm: verify_artifact(pair, arm, arms[arm]["files"]) for arm in ALLOWLIST}
    require(configs["baseline"] == configs["candidate"], "actual normalized CMake configs differ")
    commands = {arm: normalized_compile_commands(pair["arms"][arm]) for arm in ALLOWLIST}
    require(commands["baseline"] == commands["candidate"], "actual normalized compiler commands differ")
    a, b = [Path(pair["arms"][arm]["build"])/"stage/lib/psi4/driver" for arm in ALLOWLIST]
    require(tree_manifest(a) == tree_manifest(b), "artifact installed driver pair differs")
    return configs
