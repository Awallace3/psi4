import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from references import REFERENCES, compare
from compare_results import assess


class ReferenceTests(unittest.TestCase):
    def test_both_original_runs_pass(self):
        for values in REFERENCES.values():
            self.assertTrue(compare(values)["passed"])

    def test_800_budget_vs_both_baselines(self):
        values = dict(REFERENCES["new-1200GB-job15521908"])
        values["SAPT DISP ENERGY"] += 1.9096088658271393e-8
        values["SAPT TOTAL ENERGY"] += 1.910284464640144e-8
        result = compare(values)
        self.assertTrue(result["passed"])
        self.assertFalse(compare(values, 1e-8)["passed"])
        old_delta = result["references"]["old-1200GB-job15521907"]["delta_Eh"]
        self.assertAlmostEqual(old_delta["SAPT DISP ENERGY"], 2.49031097618313e-8, places=15)

    def test_nonfinite_and_missing_rejected(self):
        values = dict(next(iter(REFERENCES.values())))
        values["SAPT TOTAL ENERGY"] = float("nan")
        with self.assertRaises(AssertionError):
            compare(values)
        with self.assertRaises(KeyError):
            compare({})

    def test_incomplete_result_is_not_success(self):
        config = json.loads((HERE.parent / "pr44-protein157/config.json").read_text())
        metadata = {
            "geometry_sha256": hashlib.sha256(
                (HERE.parent / "pr44-protein157/protein157.xyz").read_bytes()).hexdigest(),
            "options": config["options"], "config": config,
            "source_commit": "fixture", "harness_commit": "fixture", "core_sha256": "fixture",
            "mode": "full",
        }
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            (directory / "metadata.json").write_text(json.dumps(metadata))
            row = assess(directory)
            self.assertFalse(row["completed"])
            self.assertNotIn("comparison", row)
            metadata["mode"] = "replay"
            (directory / "metadata.json").write_text(json.dumps(metadata))
            with self.assertRaises(AssertionError):
                assess(directory)


if __name__ == "__main__":
    unittest.main()
