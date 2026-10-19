"""Corrected test source is NOT a binary/scientific source revision."""
import hashlib
import subprocess
from pathlib import Path

import commonfix as c
import jkexception as j
import policy as p

MODE = c.MODE
REVIEW_ID = c.REVIEW_ID
RUN = p.P+"/runs/saptdft-cuest-cpks/major2-fitcase-20261019"
HARNESS = p.P+"/gits/psi4.saptdft-cuest-major2-fitcase-20261019"
SOURCES, BASE, CANDIDATE, BUILD = j.SOURCES, j.BASE, j.CANDIDATE, j.BUILD
TEST_SOURCE = p.P+"/gits/psi4.saptdft-cuest-major2-fitcase-tests-20261019"
TEST_PIN = "410136a70a22cae4f99c5cb6e3e8b911bc1897b1"
TEST_TREE = "f47510afe7a8ccfe6c955847244dbf882355f577"
TEST_FILE = "tests/pytests/"+p.PLAN
TEST_SHA = "446e0add13caec9fb6265c0061ab5cf9d563273e6529fc0e2af22746d3ed2912"
PATCH_SHA = "b06d5a83d126cccca4df23832757f15b532e6f6bdff8d725c115832696fd6876"
COUNTS = dict(common=48, fixed=4, baseline_red=1, candidate_plan=17)


def test_proof(root):
    p.check_source(root, TEST_PIN)
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args])
    p.require(git("rev-parse", TEST_PIN+"^").decode().strip() == CANDIDATE, "test direct CAND ancestry")
    p.require(git("rev-parse", TEST_PIN+"^{tree}").decode().strip() == TEST_TREE, "test tree pin")
    p.require(git("diff", "--name-only", CANDIDATE, TEST_PIN).decode().splitlines() == [TEST_FILE],
              "test source must differ by ONE test file only")
    patch = git("diff", "--binary", CANDIDATE, TEST_PIN)
    p.require(hashlib.sha256(patch).hexdigest() == PATCH_SHA, "corrected test exact patch")
    p.require(p.sha(Path(root)/TEST_FILE) == TEST_SHA, "corrected test bytes")
    return dict(parent=CANDIDATE, diff_sha256=PATCH_SHA, test_sha256=TEST_SHA)


def validate(m):
    p.require(m["build_mode"] == MODE, "commonfix build mode")
    c.validate_pair(m["artifact_pair"])
    p.require(m["build_budget_seconds"] == [0, 0] and m["test_smoke_budget_seconds"] == 1800 and
              m["production_budget_seconds"] == 3600 and m["per_process_timeout_seconds"] == 600,
              "commonfix reuse budgets")
    p.require(m["approval"]["review_id"] == REVIEW_ID, "commonfix review")
    p.require(m["test_counts"] == COUNTS, "corrected reviewed count guard")
    value = m["test_source"]
    p.require(value["source"] == TEST_SOURCE and value["pin"] == TEST_PIN and value["tree"] == TEST_TREE and
              value["proof"] == dict(parent=CANDIDATE, diff_sha256=PATCH_SHA, test_sha256=TEST_SHA),
              "corrected test provenance pin/diff")
    p.require(p.digest(value["files"]) == value["files_digest"], "test-source full manifest")
    expected = dict(m["arms"]["candidate"]["files"])
    expected[TEST_FILE] = dict(mode="100644", sha256=TEST_SHA)
    p.require(value["files"] == expected, "test source differs outside owned test file")
    p.require(m["selectors_ast"][p.PLAN]["sha256"] == TEST_SHA, "corrected selector AST revision")
    # Science provenance stays c567/c9, never the test-only child.
    for arm, pin in j.PINS.items():
        p.require(m["arms"][arm]["pin"] == pin and
                  m["arms"][arm]["files_digest"] == m["artifact_pair"]["arms"][arm]["source_files_digest"],
                  "compiled origin source manifest")
    proof = m["shared_fix"]
    p.require(proof == dict(origins=j.ORIGINS, pins=j.PINS, jk_sha256=j.JK_SHA,
                           range_test_sha256=j.RANGE_SHA, **j.EXPECTED_PROOF), "commonfix science proof")
