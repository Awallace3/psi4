"""Real captured startup regression; host/fake evidence, never GPU proof."""
import hashlib
import ast
import copy
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import fixed_jk as f
import fixed_jk_campaign as c
import fixed_jk_runner as runner
import observe as o
import policy as p
from test_fixed_jk import FakeJK

HERE = Path(__file__).resolve().parent
CAPTURE = HERE/"fixed_jk_data/job14174849-plan.out"
CAPTURE_SHA = "d7cd3bba1ae1977f788df11b53077c76eae30d10fa4961dae52560c8e8488777"


class TestWarning(unittest.TestCase):
    def test_captured_startup_zero_plans(self):
        raw = CAPTURE.read_bytes()
        self.assertEqual(len(raw), 329)
        self.assertEqual(hashlib.sha256(raw).hexdigest(), CAPTURE_SHA)
        for arm in ("baseline", "candidate"):
            self.assertEqual(f.traces(raw.decode("utf-8"), arm),
                             dict(created=0, reused=0, counters=[]))
        self.assertEqual(CAPTURE.read_bytes(), raw)

    def test_exact_old_parser_rejects_capture(self):
        old = subprocess.check_output(["git", "show", f.PARENT_PIN+":devtools/major2/fixed_jk.py"],
                                      cwd=HERE, text=True, timeout=10)
        tree = ast.parse(old)
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "traces")
        import re
        namespace = dict(p=p, re=re)
        exec(compile(ast.Module(body=[node], type_ignores=[]), "<pinned-old-parser>", "exec"), namespace)
        for arm in ("baseline", "candidate"):
            with self.assertRaisesRegex(RuntimeError, "backend fallback or SDK error"):
                namespace["traces"](CAPTURE.read_text(), arm)

    def test_native_like_suffix_counts_all_states_and_six_processes(self):
        total = 0
        for row in f.sequence():
            jk = FakeJK(row["arm"])
            jk.text = CAPTURE.read_text()
            outputs = []
            def snapshot(state, before, after, calls):
                self.assertEqual(after, f.expected_trace(row["arm"], state))
                outputs.append(state)
            _, calls = f.dispatch(jk, np.ones((2,1)), row["arm"],
                                  lambda:jk.text, snapshot)
            self.assertEqual(outputs, list(f.STATES))
            self.assertEqual(calls["compute"], 4)
            total += calls["compute"]
        self.assertEqual(total, 24)

    def test_warning_wrong_multiple_moved_and_whitespace_rejected(self):
        good = CAPTURE.read_text()
        wrong = [
            good.replace("defaulting", "falling back"),
            good.replace("           Please", "          Please"),
            good.replace("\n", "\r\n"),
            f.SIMPLE_DF_WARNING+good,
            "native text\n"+good,
            " \n"+good,
            good+f.SIMPLE_DF_WARNING,
            good.replace("simple constructor", "another constructor"),
            good.replace("workloads.", "workloads!"),
            good.replace("  Warning", "\tWarning"),
        ]
        for text in wrong:
            for arm in ("baseline", "candidate"):
                with self.subTest(text=text, arm=arm), self.assertRaises(RuntimeError):
                    f.traces(text, arm)
        self.assertEqual(f.traces("\n\n"+good, "baseline")["created"], 0)

    def test_exact_warning_does_not_hide_other_fallback_or_sdk_errors(self):
        for suffix in ("DiskDFJK", "MemDFJK", "DirectJK", "status code 8", "CUDA error"):
            for arm in ("baseline", "candidate"):
                with self.subTest(suffix=suffix, arm=arm), self.assertRaises(RuntimeError):
                    f.traces(CAPTURE.read_text()+suffix, arm)

    def test_valid_warning_never_substitutes_live_backend(self):
        for switch in ("start", "initialize", "compute"):
            jk = FakeJK("candidate")
            jk.text = CAPTURE.read_text()
            if switch == "start":
                jk.name = lambda:"DiskDFJK"
            else:
                original = getattr(jk, switch)
                def changed(original=original):
                    original()
                    jk.name = lambda:"DiskDFJK"
                setattr(jk, switch, changed)
            with self.subTest(switch=switch), self.assertRaisesRegex(RuntimeError, "backend"):
                f.dispatch(jk, None, "candidate", lambda:jk.text, lambda *args:None)
            if switch != "compute":
                self.assertNotIn("compute", jk.events)
            else:
                self.assertEqual(jk.events.count("compute"), 1)
            if switch != "start":
                self.assertEqual(jk.events[-1], "finalize")

    def test_native_failures_and_wrong_counts_still_fatal(self):
        for failure in ("sdk", "count"):
            jk = FakeJK("candidate")
            jk.text = CAPTURE.read_text()
            if failure == "sdk":
                def fail():
                    raise RuntimeError("native status code 8")
                jk.compute = fail
            else:
                original = jk.initialize
                def fail():
                    original()
                    jk.text += "Device memory, DF int plan:\ncuESTJK plan created #99\n"
                jk.initialize = fail
            with self.assertRaises(RuntimeError):
                f.dispatch(jk, None, "candidate", lambda:jk.text, lambda *args:None)
            self.assertEqual(jk.events[-1], "finalize")

    def test_executable_protocol_cannot_generate_factor(self):
        for file in ("fixed_jk_runner.py", "fixed_jk_campaign.py"):
            tree = ast.parse((HERE/file).read_text())
            calls = [ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)]
            self.assertFalse(any(x.endswith(("factor_once", "partial_cholesky_factorize")) for x in calls))
        self.assertNotIn('mode == "factor"', (HERE/"fixed_jk_runner.py").read_text())
        self.assertIn('"reuse-factor", "factor-reuse"', (HERE/"fixed_jk_campaign.py").read_text())

    def test_all_prior148_tests_byte_identical(self):
        for file in HERE.glob("test_*.py"):
            if file == Path(__file__):
                continue
            old = subprocess.check_output(["git", "show", f.PARENT_PIN+":devtools/major2/"+file.name],
                                          cwd=HERE, timeout=10)
            self.assertEqual(file.read_bytes(), old, file.name)

    def test_frozen_manifest_origin_negatives(self):
        path = os.environ.get("MAJOR2_FIXED_MANIFEST")
        if not path:
            self.fail("run reproducible run-tests.sh with frozen manifest")
        original = p.read_json(path)
        f.validate(original)
        for key, value in (("original_job","0"), ("root","wrong"), ("rank",464),
                           ("tolerance",1e-6), ("new_factor_calls",1),
                           ("baseline_core_sha256","0"*64), ("C_array_sha256","0"*64)):
            m = copy.deepcopy(original)
            m["frozen_factor"][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, "frozen_factor"):
                f.validate(m)

    def test_reuse_copies_bytes_and_original_receipt_without_factor_call(self):
        # Tiny fake files exercise the real exclusive copy/hash/origin code.
        # Real large file hashes/shapes are checked independently on Phoenix.
        with tempfile.TemporaryDirectory(dir=os.environ["MAJOR2_TEST_TMP"]) as tmp:
            src, dst = Path(tmp)/"original", Path(tmp)/"new"
            src.mkdir()
            for name, data in (("C.npy", b"frozen C"), ("D.npy", b"frozen D"),
                               ("FACTOR.json", b'{"original_receipt": true}\n')):
                (src/name).write_bytes(data)
                (src/name).chmod(0o444)
            frozen = dict(f.FROZEN_FACTOR, root=str(src),
                          files={file.name:p.sha(file) for file in src.iterdir()})
            with patch.object(f, "FROZEN_FACTOR", frozen), patch.object(f, "allocated"), \
                 patch.object(f, "load_factor", return_value=[None,None]), \
                 patch.object(f, "load_frozen_factor", side_effect=lambda root:(p.read_json(Path(root)/"FACTOR.json"), [])), \
                 patch.object(f, "INPUT", src/"C.npy"), patch.object(f, "INPUT_SHA", p.sha(src/"C.npy")), \
                 patch.dict(os.environ, SLURM_JOB_ID="999"), \
                 patch.object(f, "factor_once", side_effect=AssertionError("no new factor")):
                before = {name:(src/name).read_bytes() for name in frozen["files"]}
                f.reuse_factor(dst)
                f.verify_reused_factor(dst)
                self.assertEqual({name:(dst/name).read_bytes() for name in frozen["files"]}, before)
                self.assertEqual({name:(src/name).read_bytes() for name in frozen["files"]}, before)
                for name in frozen["files"]:
                    self.assertFalse((dst/name).stat().st_mode & 0o222)
                with self.assertRaises(FileExistsError):
                    f.reuse_factor(dst)
                (dst/"C.npy").chmod(0o644)
                with self.assertRaisesRegex(RuntimeError, "readonly"):
                    f.verify_reused_factor(dst)
                (dst/"C.npy").write_bytes(b"changed")
                (dst/"C.npy").chmod(0o444)
                with self.assertRaisesRegex(RuntimeError, "frozen factor"):
                    f.verify_reused_factor(dst)
                (dst/"C.npy").chmod(0o644)  # allow TemporaryDirectory cleanup

    def test_source_warning_exact_and_factory_selection_unchanged(self):
        from test_fixed_jk import BASE, CAND
        for root in (BASE, CAND):
            text = (root/"psi4/src/psi4/libfock/jk.cc").read_text()
            for line in f.SIMPLE_DF_WARNING.splitlines():
                self.assertIn(line+"\\n", text)
            self.assertLess(text.index("Warning: JK type"), text.index("std::make_shared<cuESTJK>"))
            self.assertLess(text.index("std::make_shared<cuESTJK>"), text.index("std::make_shared<DiskDFJK>"))


if __name__ == "__main__":
    unittest.main()
