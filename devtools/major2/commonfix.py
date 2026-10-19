"""Exact 14165998 common-fixed binary pair; legacy artifacts.py stays unchanged."""
import sys
from pathlib import Path

import artifacts as a
import jkexception as j
import policy as p

MODE = "verified-commonfix-artifact-pair"
REVIEW_ID = "parent-fitcase-commonfix-review-required"
ORIGIN = j.RUN
PAIR_DIGEST = "180f8d96d764994b6ca5719056372aaf06bb7be34f17f6c54fe68949f98b3cf9"
CORE = {
    "baseline": "67c1e9021cc2fcaaf3b4ad3f98f6750ee39049c1aafe8cdfbe75b46c8335f224",
    "candidate": "11f4c3e83dc92f5482b499911156185bce45f07dd9a83abb4a4335f04545c696",
}
ALLOWLIST = {
    arm: dict(source=j.SOURCES[arm], pin=j.PINS[arm], build=j.SOURCES[arm]+"/"+j.BUILD,
              metadata=ORIGIN+"/build-"+arm, origin_job="14165998", core_sha256=CORE[arm])
    for arm in j.PINS
}
BUILD_FILES = ["CMakeCache.txt", "cpks-commit.txt",
               "psi4-core-prefix/src/psi4-core-build/CMakeCache.txt",
               "psi4-core-prefix/src/psi4-core-build/build.ninja",
               "psi4-core-prefix/src/psi4-core-build/compile_commands.json"]


def validate_pair(pair):
    p.require(pair["mode"] == MODE and pair["review_id"] == REVIEW_ID, "commonfix pair mode/review")
    p.require(set(pair["arms"]) == set(ALLOWLIST), "commonfix pair arms")
    for arm, allowed in ALLOWLIST.items():
        p.require(all(pair["arms"][arm].get(k) == v for k, v in allowed.items()),
                  "not the exact commonfix compiled origin: "+arm)
    p.require(p.digest(pair) == PAIR_DIGEST, "commonfix immutable pair digest")


def verify_shared(pair):
    validate_pair(pair)
    for group in ("shared_inputs", "origin_proofs"):
        for path, expected in pair[group].items():
            p.require(p.sha(path) == expected, "commonfix "+group+" changed: "+path)
    value = pair["pyeinsums_core"]
    p.require(p.sha(value["path"]) == value["sha256"], "commonfix PyEinsums changed")
    for path, expected in p.LIBS.items():
        p.require(p.sha(path) == expected, "commonfix dependency changed")
    p.check_runtime_aliases()
    p.einsums_coherence()


def verify_artifact(pair, arm, files):
    validate_pair(pair)
    item = pair["arms"][arm]
    p.check_source(item["source"], item["pin"], files)
    p.require(p.digest(files) == item["source_files_digest"], "commonfix compiled source digest")
    build, meta = Path(item["build"]), Path(item["metadata"])
    p.require(build.is_dir() and not build.is_symlink(), "commonfix build absent/retargeted")
    p.require(a.hashed_tree(meta) == item["metadata_files"], "commonfix origin metadata changed")
    p.require((meta/"BUILD_OK").read_text().strip() == item["pin"]+" cuda", "commonfix BUILD_OK")
    for name, expected in item["build_files"].items():
        p.require(p.sha(build/name) == expected, "commonfix compiler config changed")
    stage = build/"stage"
    p.require(a.hashed_tree(stage) == item["stage_files"], "commonfix stage bytes/file set changed")
    p.require(p.sha(stage/"lib/psi4/core.cpython-313-x86_64-linux-gnu.so") == item["core_sha256"],
              "commonfix core hash")
    p.require(len(p.check_installed_drivers(stage, item["source"], files)) == 143, "commonfix drivers")
    inner = build/"psi4-core-prefix/src/psi4-core-build"
    p.cmake_policy((build/"CMakeCache.txt").read_text(), (inner/"CMakeCache.txt").read_text(), stage)
    configs = a.normalized_configs(item)
    commands = a.normalized_compile_commands(item)
    p.require(p.digest(configs) == pair["normalized_configs_digest"], "commonfix full normalized configs")
    p.require(p.digest(commands) == pair["normalized_commands_digest"], "commonfix full compile database")
    return configs


def verify_pair(pair, arms):
    verify_shared(pair)
    return {arm: verify_artifact(pair, arm, arms[arm]["files"]) for arm in ALLOWLIST}


def capture():
    """One bounded foreground collection. No import, old writes, or full build walk.

    Reuse the allocation's complete stage manifests, cross-check the live core
    once. Full live stage/runtime checks remain mandatory in the new allocation.
    """
    root = Path(ORIGIN)
    m = p.read_json(root/"manifest.json")
    p.validate_manifest(m)
    p.require(m["harness"]["pin"] == "ede11975fd80a7f42bd2cff5a3029a2ea029ef71", "origin harness")
    p.require(p.read_json(root/"STARTED.json")["job"] == "14165998", "origin allocation")
    p.require(p.read_json(root/"ENGINE_FAILED.json")["production_records"] == 0, "unexpected production")
    pair = dict(mode=MODE, review_id=REVIEW_ID, arms={},
                shared_inputs=p.read_json(root/"raw/toolchain-inputs.json"), origin_proofs={})
    configs, commands, probes, drivers = {}, {}, {}, {}
    for arm, allowed in ALLOWLIST.items():
        item = dict(allowed)
        source, build, meta = Path(item["source"]), Path(item["build"]), Path(item["metadata"])
        p.check_source(source, item["pin"])
        p.require(item["pin"] == m["arms"][arm]["pin"], "origin scientific pin")
        item["source_files_digest"] = m["arms"][arm]["files_digest"]
        item["metadata_files"] = a.hashed_tree(meta)
        p.require(len(item["metadata_files"]) < 150, "unbounded metadata expansion")
        item["build_files"] = {name: p.sha(build/name) for name in BUILD_FILES}
        item["stage_files"] = p.read_json(root/("raw/"+arm+"-stage.json"))
        live = p.sha(build/"stage/lib/psi4/core.cpython-313-x86_64-linux-gnu.so")
        p.require(live == item["core_sha256"] == (meta/"core.sha256").read_text().split()[0],
                  "captured live core/origin hash")
        probes[arm] = p.read_json(root/("raw/"+arm+"-probe.json"))
        p.require(probes[arm]["core_sha256"] == live, "origin actual imported core")
        abi = p.read_json(meta/"abi-after-build/result.json")
        p.require(abi["ok"] and abi["no_gpu_calls"] and abi["cudart_version"] == 13030, "origin ABI")
        p.require(set((meta/"core_sass.txt").read_text().split()) == {"sm_80","sm_89","sm_90","sm_120"},
                  "origin SASS")
        drivers[arm] = p.read_json(root/("raw/"+arm+"-drivers.json"))
        expected = p.installed_drivers(source, m["arms"][arm]["files"])
        p.require(drivers[arm] == expected and len(expected) == 143, "origin staged drivers")
        configs[arm], commands[arm] = a.normalized_configs(item), a.normalized_compile_commands(item)
        p.cmake_policy((build/"CMakeCache.txt").read_text(),
                       (build/"psi4-core-prefix/src/psi4-core-build/CMakeCache.txt").read_text(), build/"stage")
        pair["arms"][arm] = item
        for name in [arm+"-probe.json", arm+"-stage.json", arm+"-drivers.json", arm+"-build-files.json",
                     arm+"-cmake-policy.json", arm+"-runtime-selection.json"]:
            path = root/"raw"/name
            pair["origin_proofs"][str(path)] = p.sha(path)
    p.require(configs["baseline"] == configs["candidate"], "actual normalized complete CMake mismatch")
    p.require(commands["baseline"] == commands["candidate"], "actual normalized complete compile DB mismatch")
    p.require(drivers["baseline"] == drivers["candidate"], "origin driver pair mismatch")
    for key in ("pyeinsums_core", "pyeinsums_core_sha256"):
        p.require(probes["baseline"][key] == probes["candidate"][key], "origin PyEinsums pair")
    pair.update(normalized_configs_digest=p.digest(configs["baseline"]),
                normalized_commands_digest=p.digest(commands["baseline"]),
                pyeinsums_core=dict(path=probes["baseline"]["pyeinsums_core"],
                                    sha256=probes["baseline"]["pyeinsums_core_sha256"]))
    for name in ["manifest.json", "STARTED.json", "ENGINE_FAILED.json", "raw/toolchain-inputs.json",
                 "raw/gpu-identity.json", "raw/environment.json"]:
        pair["origin_proofs"][str(root/name)] = p.sha(root/name)
    print(p.json.dumps(pair, indent=2, sort_keys=True))


if __name__ == "__main__":
    p.require(sys.argv[1:] == ["capture"], "capture only; no submission")
    capture()
