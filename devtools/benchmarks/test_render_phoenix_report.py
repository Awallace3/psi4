"""Regenerate the published table from its frozen raw evidence."""
from pathlib import Path
import tempfile
import unittest

from render_phoenix_report import build_paired_summary


EVIDENCE = Path(__file__).parent / "results/phoenix-a100-20260909"


class ReportTests(unittest.TestCase):
    def test_raw_merge_preserves_accuracy_misses(self):
        summary = build_paired_summary(EVIDENCE)
        self.assertEqual(len(summary["rows"]), 6)
        failures = [r for r in summary["rows"] if not r["accuracy_pass"]]
        self.assertEqual({(r["system"], r["basis"]) for r in failures},
                         {("benzene", "cc-pvdz"), ("benzene", "aug-cc-pvdz")})
        self.assertFalse(summary["source_campaigns"]["retry1"]["complete"])
        self.assertTrue(summary["source_campaigns"]["suite2"]["complete"])

    def test_monomer_counts_are_measured_not_half_the_dimer(self):
        summary = build_paired_summary(EVIDENCE)
        row = next(r for r in summary["rows"] if r["system"] == "nanotube")
        self.assertEqual((row["nbf_monomer_a"], row["nbf_monomer_b"], row["nbf"]), (56, 492, 548))

    def test_precomputed_summary_is_not_trusted(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence = Path(directory)
            (evidence / "raw").symlink_to((EVIDENCE / "raw").resolve(), target_is_directory=True)
            (evidence / "paired-summary.json").write_text('{"rows": [], "fabricated": true}')
            summary = build_paired_summary(evidence)
            self.assertEqual(len(summary["rows"]), 6)
            self.assertNotIn("fabricated", summary)


if __name__ == "__main__":
    unittest.main()
