"""Host-only policy/source checks; not evidence of GPU recovery or numerical math."""
import ast
import copy
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace

import campaign
import handoff
import jkexception as j
import policy as p
from test_harness import manifest

HERE = Path(__file__).resolve().parent


def fresh_manifest():
    m = copy.deepcopy(manifest())
    m.update(build_mode=j.MODE, run_root=j.RUN,
             build_budget_seconds=[3600, 3600], test_smoke_budget_seconds=1800,
             production_budget_seconds=3600, per_process_timeout_seconds=600,
             shared_fix=dict(origins=j.ORIGINS, pins=j.PINS, jk_sha256=j.JK_SHA,
                             range_test_sha256=j.RANGE_SHA, **j.EXPECTED_PROOF))
    m["harness"]["path"] = j.HARNESS
    m["approval"].update(candidate_pin=j.CANDIDATE, review_id=j.REVIEW_ID)
    for arm, item in m["arms"].items():
        item.update(source=j.SOURCES[arm], pin=j.PINS[arm], build=j.SOURCES[arm]+"/"+j.BUILD)
        item["files"] = copy.deepcopy(item["files"])
        item["files"].update({j.JK: dict(mode="100644", sha256=j.JK_SHA),
                              j.RANGE_TEST: dict(mode="100644", sha256=j.RANGE_SHA)})
        item["files_digest"] = p.digest(item["files"])
    return m


class FreshTests(unittest.TestCase):
    def test_fresh_guards(self):
        p.validate_manifest(fresh_manifest())
        for mutate in [
            lambda m: m.update(build_mode="resume"),
            lambda m: m.update(artifact_pair={}),
            lambda m: m.update(build_budget_seconds=[0, 0]),
            lambda m: m.update(build_budget_seconds=[3600, 0]),
            lambda m: m.update(production_budget_seconds=300),
            lambda m: m["arms"]["baseline"].update(pin=p.BASE),
            lambda m: m["arms"]["candidate"].update(pin=p.OBSERVED_CANDIDATE),
            lambda m: m["arms"]["candidate"].update(build=j.SOURCES["candidate"]+"/build_major2_cuda_mapproof"),
            lambda m: m["arms"]["baseline"].update(source=p.SOURCES["baseline"]),
            lambda m: m["shared_fix"].update(shared_patch_sha256="0"*64),
            lambda m: m["shared_fix"].update(optimization_diff_sha256="0"*64),
            lambda m: m["arms"]["candidate"]["files"][j.RANGE_TEST].update(sha256="0"*64),
            lambda m: m["arms"]["baseline"]["files"][j.JK].update(sha256="0"*64),
            lambda m: m["approval"].update(review_id="another-review"),
        ]:
            with self.subTest(mutate=mutate), self.assertRaises(RuntimeError):
                m = fresh_manifest()
                mutate(m)
                p.validate_manifest(m)

    def test_actual_git_common_fix_and_original_tests(self):
        # Pin-based Git proof works in both local and remote checkouts.
        root = HERE.parents[1]
        patches = []
        for arm, pin in j.PINS.items():
            self.assertEqual(p.git(root, "rev-parse", pin+"^"), j.ORIGINS[arm])
            patches.append(p.git(root, "diff", "--binary", j.ORIGINS[arm], pin))
            self.assertEqual(p.git(root, "show", pin+":"+j.RANGE_TEST),
                             p.git(root, "show", j.ORIGINS[arm]+":"+j.RANGE_TEST))
        self.assertEqual(*patches)
        self.assertEqual(p.git(root, "diff", "--binary", p.BASE, p.OBSERVED_CANDIDATE),
                         p.git(root, "diff", "--binary", j.BASE, j.CANDIDATE))

    def test_original_53_test_functions_unchanged(self):
        root = HERE.parents[1]
        old = p.git(root, "show", "0a3bc60d3d59ab968d1deefbff3df7bb160c76eb:devtools/major2/test_harness.py")
        def tests(text):
            return {n.name: ast.dump(n) for n in ast.walk(ast.parse(text))
                    if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")}
        self.assertEqual(len(tests(old)), 53)
        self.assertEqual(tests(old), tests((HERE/"test_harness.py").read_text()))

    def test_extracted_native_backend_block_exception_semantics(self):
        # Compile the exact owned C++ block, not a reimplementation. Fake backend/timer
        # verifies control flow only; actual timer lock probe belongs to the parent.
        source = p.git(HERE.parents[1], "show", j.BASE+":"+j.JK)
        block = source[source.index('    timer_on("JK: JK");'):source.index('\n    if (C1()) {', source.index('    timer_on("JK: JK");'))]
        text = r'''
#include <cassert>
#include <stdexcept>
bool active = false, backend_error = false, cleanup_error = false;
int starts = 0, stops = 0;
void timer_on(const char*) { assert(!active); active = true; ++starts; }
void timer_off(const char*) {
    active = false; ++stops;
    if (cleanup_error) throw 99;
}
void compute_JK() { if (backend_error) throw 42; }
void compute() {
''' + block + r'''
}
int main() {
    compute(); assert(!active && starts == 1 && stops == 1);
    backend_error = true;
    try { compute(); assert(false); } catch (int v) { assert(v == 42); }
    assert(!active && starts == 2 && stops == 2);
    backend_error = false;
    compute(); assert(!active && starts == 3 && stops == 3);
    backend_error = cleanup_error = true;
    try { compute(); assert(false); } catch (int v) { assert(v == 42); }
}
'''
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root/"probe.cc").write_text(text)
            subprocess.run(["g++", "-std=c++17", "-Wall", "-Wextra", "-Werror",
                            str(root/"probe.cc"), "-o", str(root/"probe")], check=True, timeout=30)
            subprocess.run([str(root/"probe")], check=True, timeout=5)

    def test_fresh_batch_and_traceback_diagnostic(self):
        batch = campaign.batch_text("a"*64, fresh=True)
        self.assertIn(j.HARNESS+"/devtools/major2/supervise.sh "+j.RUN, batch)
        self.assertNotIn(p.HARNESS+"/", batch)
        text = (HERE/"run_pytest.py").read_text()
        self.assertIn("dump_traceback_later(540, repeat=False, exit=False)", text)
        self.assertNotIn("cancel_dump_traceback_later", text)
        self.assertNotIn("os._exit", text)

    def test_fresh_entry_point_rejects_legacy_artifact_manifest(self):
        import sys
        with patch.object(sys, "argv", ["campaign.py", "run", "--manifest", "/fake",
                                       "--manifest-sha256", "a"*64]), \
                patch.object(campaign, "sha", return_value="a"*64), \
                patch.object(campaign, "read_json", return_value=manifest()), \
                patch.object(campaign, "Engine") as engine:
            with self.assertRaisesRegex(RuntimeError, "fresh common-fixed"):
                campaign.main()
            engine.assert_not_called()

    def test_parent_submit_id_durable_before_acceptance_and_permanent_lock(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root/"logs").mkdir()
            gates = dict(harness_pin="pin", manifest_sha256="m", batch_sha256="b",
                         preflight_sha256="h", pristine_staging=True, old_run_unchanged=True,
                         old_build_sentinels_unchanged=True, test_only_stdout="gpu-h200", test_only_stderr="")
            p.write_json(root/"logs/pre-submit-gates.json", gates)
            def fail_acceptance(*args):
                self.assertEqual((root/"logs/submission.lock/job-id.txt").read_text(), "12345\n")
                raise RuntimeError("acceptance unavailable")
            with patch.object(j, "RUN", str(root)), patch.object(handoff, "verify_bundle", return_value=({}, {})), \
                    patch.object(handoff, "queue", return_value=[]), patch.object(p, "sha", return_value="h"), \
                    patch.object(handoff.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout="12345\n", stderr="")), \
                    patch.object(handoff, "output", side_effect=fail_acceptance):
                with self.assertRaisesRegex(RuntimeError, "acceptance unavailable"):
                    handoff.submit("pin", "m", "b")
                with self.assertRaises(FileExistsError):
                    handoff.submit("pin", "m", "b")


if __name__ == "__main__":
    unittest.main(verbosity=2)
