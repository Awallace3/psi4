"""Offline stdlib AST/ordering/failure guards; these are not GPU tests."""
import ast
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import fdds_alda_profile as profile


EVENTS = []


def operand():
    EVENTS.append("operand")
    return 5


def target(a, *, factor):
    EVENTS.append(("target", a, factor))
    return a * factor


def original():
    return target(operand(), factor=3)


def broken():
    raise ValueError("original failure")


def failing():
    return broken()


class Guards(unittest.TestCase):
    def test_real_source_thirteen_sites(self):
        root = Path(__file__).resolve().parents[2]
        source = root / "psi4/driver/procrouting/sapt/sapt_mp2_terms.py"
        tree = ast.parse(source.read_text())
        for name, expected in (("df_fdds_dispersion", profile.DRIVER_SITES),
                               ("_compute_fxc", profile.FXC_SITES)):
            fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
            calls = [ast.unparse(n) for n in ast.walk(fn) if isinstance(n, ast.Call)]
            self.assertTrue(all(calls.count(s) == 1 for s in expected))
        self.assertEqual(len(profile.DRIVER_SITES) + len(profile.FXC_SITES), 13)

    def test_order_return_and_repeated_freshness(self):
        records = []
        copied, hashes = profile.instrument(original, records, {"target(operand(), factor=3)": "target"})
        EVENTS.clear()
        self.assertEqual(copied(), 15)
        self.assertEqual(copied(), 15)
        self.assertEqual(EVENTS, ["operand", ("target", 5, 3)] * 2)
        self.assertEqual(len(records), 2)
        self.assertTrue(all(r["succeeded"] for r in records))
        self.assertEqual(len(hashes["instrumented_ast_sha256"]), 64)

    def test_exception_propagates_and_is_recorded(self):
        records = []
        copied, _ = profile.instrument(failing, records, {"broken()": "failure"})
        with self.assertRaisesRegex(ValueError, "original failure"):
            copied()
        self.assertEqual(len(records), 1)
        self.assertFalse(records[0]["succeeded"])

    def test_changed_or_missing_site_rejected(self):
        with self.assertRaises(ValueError):
            profile.instrument(original, [], {"target(operand(), factor=4)": "changed"})
        with self.assertRaises(ValueError):
            profile.instrument(original, [], {"target(operand(), factor=3)": "target",
                                               "broken()": "missing"})

    def test_private_namespace_override(self):
        records = []
        calls = []
        replacement = lambda a, factor: calls.append((a, factor)) or 42
        copied, _ = profile.instrument(original, records,
                                       {"target(operand(), factor=3)": "target"},
                                       {"target": replacement})
        self.assertEqual(copied(), 42)
        self.assertEqual(calls, [(5, 3)])
        self.assertIs(original.__globals__["target"], target)

    def test_runner_failure_restores_driver_and_publishes_invalid_profile(self):
        fake = SimpleNamespace(__file__=__file__, df_fdds_dispersion=original, _compute_fxc=original)
        expected = {"target(operand(), factor=3)": "target"}
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "profile.json"
            sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
            with patch.object(profile, "load_driver", return_value=fake), \
                    patch.object(profile, "DRIVER_SITES", expected), \
                    patch.object(profile, "FXC_SITES", expected), \
                    patch.object(profile.sys, "argv", ["profile", "runner", sha, str(output)]), \
                    patch.object(profile.runpy, "run_path", side_effect=ValueError("runner failed")):
                with self.assertRaisesRegex(ValueError, "runner failed"):
                    profile.main()
            self.assertIs(fake.df_fdds_dispersion, original)
            self.assertIs(fake._compute_fxc, original)
            result = json.loads(output.read_text())
            self.assertFalse(result["valid"])
            self.assertEqual(result["records"], [])

    def test_success_marker_is_after_final_provenance_guards(self):
        script = Path(__file__).with_name("fdds_alda_profile.sh").read_text()
        self.assertNotIn('(out / "VERIFIED.json").open', script)
        publish = script.index('os.rename(out / "AUDIT_PENDING.json", out / "VERIFIED.json")')
        self.assertGreater(publish, script.rindex('test "$(git -C "$source" rev-parse HEAD)"'))
        self.assertGreater(publish, script.rindex('test -z "$(git -C "$source" status'))
        self.assertGreater(publish, script.rindex('test -z "$(git -C "$root" status'))


if __name__ == "__main__":
    unittest.main()
