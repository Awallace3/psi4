"""Exact reused14165998 BASE + fresh constructor-fixed CAND, no arbitrary adoption."""
import hashlib
import subprocess
from pathlib import Path

import artifacts as a
import commonfix as c
import jkexception as j
import policy as p

MODE = "baseline14165998-fresh-nonc1"
REVIEW_ID = "parent-nonc1-constructor-hybrid-review-required"
REPLACEMENT_FOR = "14169328"
COMPILED_IN_NEW_JOB = True  # candidate only; explicit per-arm origin below
SUBMISSION_BUILD_ORIGINS = dict(baseline=dict(origin_job="14165998", compiled_in_this_job=False),
                                candidate=dict(origin_job="pending-new-allocation", compiled_in_this_job=True))
RUN = p.P+"/runs/saptdft-cuest-cpks/major2-nonc1-20261019"
HARNESS = p.P+"/gits/psi4.saptdft-cuest-major2-nonc1-20261019"
BASE = j.BASE
CANDIDATE = "21a7b4d9c922f293d6220cafd8143d70f7d85e18"
SOURCES = dict(baseline=j.SOURCES["baseline"],
               candidate=p.P+"/gits/psi4.saptdft-cuest-major2-nonc1-candidate-20261019")
BUILDS = dict(baseline=SOURCES["baseline"]+"/"+j.BUILD,
              candidate=SOURCES["candidate"]+"/build_major2_cuda_nonc1")
BUILD = "build_major2_cuda_nonc1"
PINS = dict(baseline=BASE, candidate=CANDIDATE)
CPP = "psi4/src/psi4/libfock/cuESTJK.cc"
CPP_SHA = "298d2998b65ae0a858e7d11b87c0dc1112af16356ee000e08e1873a963394838"
CTOR_PATCH = "9a16cf1de3f445cd7f886e3c40ade97c3d3c4cfe0c2a4fec0a86ac174d0e1d0c"
SCIENCE_PATCH = "6b1f9cd4dd7f416cecc8a35a85901cfef2c3dc819ca6bb0969177d82a0b848d2"
TEST_SOURCE = p.P+"/gits/psi4.saptdft-cuest-major2-nonc1-tests-20261019"
TEST_PIN = "85d2fa601856cd1e9ada04f071af1d5ff6676883"
TEST_TREE = "7ea4c57ed3e1c366ea609acec5ad44ce298dd31f"
TEST_FILE = "tests/pytests/"+p.PLAN
TEST_SHA = "9f9f23faf863f849d60eee0303dd4201ace85f7a5af4a3c27a86db3483bb1b0b"
TEST_PATCH = "ed094836b9b43d9550f4d1293da3233dca693f0f7f4037e85466adc701231378"
COUNTS = dict(common=48, fixed=4, baseline_red=1, candidate_plan=18)
ORIGIN_DIGEST = "2ec8cf8268a3eb772a7a52b4a36e2a08251637bb14a0c5a40b4273b73681c0ba"
PROOF = dict(original_optimization_sha256=j.EXPECTED_PROOF["optimization_diff_sha256"],
             constructor_patch_sha256=CTOR_PATCH, scientific_delta_sha256=SCIENCE_PATCH)
TEST_PROOF = dict(parent=CANDIDATE, diff_sha256=TEST_PATCH, test_sha256=TEST_SHA)


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args])


def proof(sources):
    for arm in PINS:
        p.check_source(sources[arm], PINS[arm])
    root = sources["candidate"]
    g = lambda *args: git(root, *args)
    p.require(g("rev-parse", CANDIDATE+"^").decode().strip() == j.CANDIDATE, "constructor direct child")
    p.require(g("diff", "--name-only", j.CANDIDATE, CANDIDATE).decode().splitlines() == [CPP],
              "constructor only C++ path")
    for lo, hi, expected in [(j.BASE, j.CANDIDATE, PROOF["original_optimization_sha256"]),
                              (j.CANDIDATE, CANDIDATE, CTOR_PATCH), (BASE, CANDIDATE, SCIENCE_PATCH)]:
        p.require(hashlib.sha256(g("diff", "--binary", lo, hi)).hexdigest() == expected, "scientific delta")
    # Original common JK fix and all five scientific paths remain attributable.
    for pin in PINS.values():
        p.require(hashlib.sha256(g("show", pin+":"+j.JK)).hexdigest() == j.JK_SHA, "shared JK drift")
        p.require(hashlib.sha256(g("show", pin+":"+j.RANGE_TEST)).hexdigest() == j.RANGE_SHA, "range test drift")
    p.require(set(g("diff", "--name-only", BASE, CANDIDATE).decode().splitlines()) == p.OWNED_DIFF,
              "five scientific paths")
    return PROOF


def test_proof(root):
    p.check_source(root, TEST_PIN)
    g = lambda *args: git(root, *args)
    p.require(g("rev-parse", TEST_PIN+"^").decode().strip() == CANDIDATE, "test direct new CAND child")
    p.require(g("rev-parse", TEST_PIN+"^{tree}").decode().strip() == TEST_TREE, "test tree")
    p.require(g("diff", "--name-only", CANDIDATE, TEST_PIN).decode().splitlines() == [TEST_FILE], "one test file")
    p.require(hashlib.sha256(g("diff", "--binary", CANDIDATE, TEST_PIN)).hexdigest() == TEST_PATCH, "test patch")
    p.require(p.sha(Path(root)/TEST_FILE) == TEST_SHA, "test bytes")
    return TEST_PROOF


def baseline_origin(pair):
    """Derive ONCE from frozen old pair; no old CAND artifact travels to new mode."""
    c.validate_pair(pair)
    return dict(mode=MODE, review_id=REVIEW_ID, arms=dict(baseline=pair["arms"]["baseline"]),
                shared_inputs=pair["shared_inputs"], pyeinsums_core=pair["pyeinsums_core"],
                normalized_configs_digest=pair["normalized_configs_digest"],
                normalized_commands_digest=pair["normalized_commands_digest"],
                origin_proofs={path: value for path, value in pair["origin_proofs"].items()
                               if "/candidate-" not in path})


def validate_origin(origin):
    p.require(origin["mode"] == MODE and origin["review_id"] == REVIEW_ID, "baseline mode/review")
    p.require(set(origin["arms"]) == {"baseline"}, "baseline ONLY adoption")
    item = origin["arms"]["baseline"]
    p.require(all(item.get(k) == v for k, v in c.ALLOWLIST["baseline"].items()), "exact14165998 BASE origin")
    p.require(p.digest(origin) == ORIGIN_DIGEST, "frozen BASE receipt/config/runtime digest")


def verify_shared(origin):
    validate_origin(origin)
    for group in ("shared_inputs", "origin_proofs"):
        for path, expected in origin[group].items():
            p.require(p.sha(path) == expected, "baseline original proof/input changed")
    item = origin["pyeinsums_core"]
    p.require(p.sha(item["path"]) == item["sha256"], "baseline PyEinsums changed")
    for path, expected in p.LIBS.items():
        p.require(p.sha(path) == expected, "dependency changed")
    p.check_runtime_aliases()
    p.einsums_coherence()


def verify_baseline(origin, files):
    validate_origin(origin)
    item = origin["arms"]["baseline"]
    p.check_source(item["source"], BASE, files)
    p.require(p.digest(files) == item["source_files_digest"], "BASE whole source")
    build, meta = Path(item["build"]), Path(item["metadata"])
    p.require(build.is_dir() and not build.is_symlink(), "BASE build retarget")
    p.require(a.hashed_tree(meta) == item["metadata_files"], "BASE origin receipts")
    p.require((meta/"BUILD_OK").read_text().strip() == BASE+" cuda", "BASE original BUILD_OK")
    for name, expected in item["build_files"].items():
        p.require(p.sha(build/name) == expected, "BASE compiler inputs")
    stage = build/"stage"
    p.require(a.hashed_tree(stage) == item["stage_files"], "BASE stage bytes")
    p.require(p.sha(stage/"lib/psi4/core.cpython-313-x86_64-linux-gnu.so") == item["core_sha256"], "BASE core")
    p.require(len(p.check_installed_drivers(stage, item["source"], files)) == 143, "BASE installed drivers")
    p.require(p.digest(a.normalized_configs(item)) == origin["normalized_configs_digest"], "BASE complete configs")
    p.require(p.digest(a.normalized_compile_commands(item)) == origin["normalized_commands_digest"], "BASE compile DB")


def validate(m):
    p.require(m["build_mode"] == MODE and "artifact_pair" not in m, "hybrid cannot adopt pair")
    validate_origin(m["baseline_origin"])
    p.require(m["build_budget_seconds"] == [0, 3600] and m["test_smoke_budget_seconds"] == 1800 and
              m["production_budget_seconds"] == 3600 and m["per_process_timeout_seconds"] == 600, "hybrid budgets")
    p.require(m["approval"]["review_id"] == REVIEW_ID and m["constructor_proof"] == PROOF, "constructor approval/proof")
    p.require(m["test_counts"] == COUNTS, "reviewed18 focused count")
    for arm, item in m["arms"].items():
        p.require(item["pin"] == PINS[arm] and item["source"] == SOURCES[arm] and item["build"] == BUILDS[arm],
                  "hybrid exact source/build")
        for file, expected in [(j.JK, j.JK_SHA), (j.RANGE_TEST, j.RANGE_SHA)]:
            p.require(item["files"][file]["sha256"] == expected, "unchanged common source")
    p.require(m["arms"]["candidate"]["files"][CPP]["sha256"] == CPP_SHA, "constructor file")
    p.require(m["arms"]["baseline"]["files_digest"] ==
              m["baseline_origin"]["arms"]["baseline"]["source_files_digest"], "BASE origin manifest")
    value = m["test_source"]
    p.require(value["source"] == TEST_SOURCE and value["pin"] == TEST_PIN and value["tree"] == TEST_TREE and
              value["proof"] == TEST_PROOF, "new test revision identity")
    p.require(p.digest(value["files"]) == value["files_digest"], "full test manifest")
    expected = dict(m["arms"]["candidate"]["files"])
    expected[TEST_FILE] = dict(mode="100644", sha256=TEST_SHA)
    p.require(value["files"] == expected, "test-only one file")
    p.require(m["selectors_ast"][p.PLAN]["sha256"] == TEST_SHA, "selector exact test source")
