"""Local, CPU-only publisher/regeneration regression checks; no Psi4 import."""
import contextlib
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import unittest

from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parent))
import publish_basis_matched as publisher
from publish_tiny_grid_pilot import iteration_sections

ROOT = Path(__file__).resolve().parents[2]
DOCS = Path(os.environ.get("CUEST_REPORT_DOCS", "/home/awallace43/docs/saptdft/cuest"))
RAW = ROOT / ".pi/validation/protein157-14096296-basis-matched/raw"
BASELINE = ROOT / ".pi/validation/report-refresh-20261016"
sys.path.insert(0, str(DOCS))
import build_breakdown_2026_10_04 as report
import operation_pairs


def payload(path):
    soup = BeautifulSoup(path.read_text(), "html.parser")
    return soup, json.loads(soup.select_one("#data, #measurements").string)


class AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.campaign = publisher.verify(RAW, DOCS)
        cls.row = json.loads((RAW / publisher.GPU / "result.json").read_text())
        cls.cpu = json.loads((RAW / "reference-13868775.json").read_text())

    def test_verified_basis_matched_schema_not_repeat3(self):
        c = self.campaign
        self.assertEqual(c["schema"], "basis-matched-whole-call-v1")
        self.assertAlmostEqual(c["ratio"], 40.315704521845575, places=10)
        self.assertEqual(c["iterations"], dict(initial=174, final=46, untagged=50, total=270))
        self.assertNotIn("groups", c)
        self.assertNotIn("repeats", c)
        self.assertEqual([p["final"] for p in c["proofs"]], [6, 10, 6, 7, 10, 7])
        self.assertAlmostEqual(c["sections"][0]["wall_s"], 123.053579420, places=8)

    def test_policy_and_identity_rejection(self):
        mutations = [
            ("nbf", 2171), ("basis", "heavy-aug-cc-pvdz"), ("threads", 24),
            ("source_commit", "unverified"), ("geometry_sha256", "bad"),
            ("reference_result_sha256", "bad"), ("cpu_builder_seen", True),
            ("direct_seen", True), ("slurm_step_id", "1"),
        ]
        for key, value in mutations:
            with self.subTest(key=key):
                row = copy.deepcopy(self.row)
                row[key] = value
                with self.assertRaises(AssertionError):
                    publisher.validate_policy(row, self.cpu)
        for key, value in (("SAPT_DFT_INDUCTION_TYPE", "CPKS"),
                           ("CUEST_MIXED_PRECISION", True),
                           ("DFT_GRID_SWITCH_FACTOR", 10),
                           ("DFT_RADIAL_POINTS", 99)):
            with self.subTest(option=key):
                row = copy.deepcopy(self.row)
                row["options"][key] = value
                with self.assertRaises(AssertionError):
                    publisher.validate_policy(row, self.cpu)
        for change in ("overlay", "minfinal", "proof"):
            row = copy.deepcopy(self.row)
            if change == "overlay":
                row["iterator_overlay"]["sha256"] = "bad"
            elif change == "minfinal":
                row["grid_policy"]["minimum_final_iterations"] = 1
            else:
                row["proofs"][0]["final_grid_after"] = False
            with self.subTest(change=change), self.assertRaises(AssertionError):
                publisher.validate_policy(row, self.cpu)

    def test_hash_rejection_before_parsing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = next(iter(publisher.HASHES))
            target = root / first
            target.parent.mkdir(parents=True)
            target.write_text("{}")
            with self.assertRaisesRegex(AssertionError, "Evidence hash mismatch"):
                publisher.verify(root, DOCS)

    def test_canonical_rows_include_sad_and_close(self):
        sections = iteration_sections((RAW / publisher.GPU / "psi4.out").read_text())
        self.assertEqual(sum(s["total"] for s in sections.values()), 270)
        for s in sections.values():
            self.assertEqual(s["total"], s["initial"] + s["final"] + s["untagged"])
        self.assertEqual(sections["gracA"]["total"], 114)
        with self.assertRaisesRegex(AssertionError, "Unassigned"):
            iteration_sections("  @RHF iter 1 unassigned")
        sample = " SAPT(DFT): DFT Monomer A\n    @RHF iter SAD XC-INITIAL\n @RHF iter 1 XC-FINAL\n"
        self.assertEqual(iteration_sections(sample)["dftA"]["total"], 2)

    def test_csv_units_and_whitespace(self):
        text = (RAW / "sessions/gpu-samples.csv").read_text()
        m = publisher.csv_memory(text)
        self.assertEqual(m["peak_gib"], 76.8056640625)
        self.assertEqual((m["samples"], m["interval_samples"]), (294, 263))
        self.assertEqual(m["max_gap_s"], 1.068)
        self.assertEqual(publisher.csv_memory(text.replace(", ", ",  ")), m)
        with self.assertRaises(AssertionError):
            publisher.csv_memory(text.replace("78649 MiB", "78649 GiB"))

    def test_timer_additivity_and_reject_broken_wall(self):
        for raw in (self.cpu, self.row):
            run = publisher.partition(raw, report, operation_pairs, "local")
            self.assertAlmostEqual(sum(operation_pairs.operation_values(run)), raw["wall_s"], places=7)
        row = copy.deepcopy(self.row)
        row["wall_s"] = 1.
        with self.assertRaises(AssertionError):
            publisher.partition(row, report, operation_pairs, "local")

    def test_idempotence_and_legacy_key_preservation(self):
        with tempfile.TemporaryDirectory() as directory:
            docs = Path(directory)
            # Resolve accounting modules before using a temporary publication destination.
            data = json.loads((DOCS / "grid-evidence.json").read_text())
            data.pop(publisher.KEY, None)
            target = docs / "grid-evidence.json"
            target.write_text(json.dumps(data))
            with contextlib.redirect_stdout(io.StringIO()):
                publisher.publish(RAW, docs)
                first = target.read_bytes()
                publisher.publish(RAW, docs)
            self.assertEqual(first, target.read_bytes())
            result = json.loads(first)
            result.pop(publisher.KEY)
            self.assertEqual(result, data)
            for name, digest in publisher.HASHES.items():
                self.assertEqual(publisher.sha(docs / publisher.SOURCE / name), digest)


class ReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.soup, cls.data = payload(DOCS / "breakdown-2026-10-04.html")
        cls.old_soup, cls.old_data = payload(
            BASELINE / "before/docs/saptdft/cuest/breakdown-2026-10-04.html")

    def test_historical_payload_and_all_48_selections_preserved(self):
        old, new = copy.deepcopy(self.old_data), copy.deepcopy(self.data)
        old.pop("gridEvidence")
        new.pop("gridEvidence")
        for d in (old, new):
            d["categories"]["functional"][1] = "qualified historical backend"
        self.assertEqual(old, new)
        oldgrid, grid = self.old_data["gridEvidence"], self.data["gridEvidence"]
        self.assertEqual({k: v for k, v in grid.items() if k not in (publisher.KEY, "status", "coupled_scaling_suite")},
                         {k: v for k, v in oldgrid.items() if k != "status"})
        legacy = sum(len(c["cases"]) for c in self.data["campaigns"])
        legacy += sum(len(c["cases"]) for key in ("cpu_gpu_grid_campaigns", "cpu_gpu_estimate_campaigns")
                      for c in grid[key])
        self.assertEqual(legacy, 48)
        self.assertEqual(sum(len(c["cases"]) for c in grid[publisher.KEY]), 1)
        # Neither the original historical report nor unrelated raw inputs may advance.
        baseline = json.loads((BASELINE / "baseline.json").read_text())
        for name, digest in baseline["files"].items():
            path = Path(name)
            if path.name in ("breakdown.html", "cuest-bottlenecks.html"):
                self.assertEqual(publisher.sha(path), digest)
        for name, digest in baseline["evidence"].items():
            self.assertEqual(publisher.sha(Path(name)), digest)

    def test_same_page_anchors_and_collapsed_history(self):
        # All static fragment IDs survive, including real source-line anchors.
        old_ids = {n["id"] for n in self.old_soup.select("[id]")}
        new_ids = {n["id"] for n in self.soup.select("[id]")}
        self.assertTrue(old_ids <= new_ids, sorted(old_ids-new_ids))
        appendix = self.soup.select_one("#historical-content")
        self.assertNotIn("open", appendix.attrs)
        for key in ("updates", "graph", "delta", "inside", "balance", "xc-code"):
            self.assertIsNotNone(appendix.select_one("#"+key))
        self.assertIsNotNone(self.soup.select_one("#evidence #python"))
        self.assertEqual(len(self.soup.select("main h2")), 11)
        self.assertIn("const heads = $$('main h2[id]');", str(self.soup))

    def test_truthful_current_claims_and_caption_metadata(self):
        text = self.soup.get_text(" ", strip=True)
        for phrase in ("1.037×", "1.425×", "32/32", "7–12", "76.805664 GiB",
                       "No completed full-aug", "static polarizability", "not CPKS evidence"):
            self.assertIn(phrase.lower(), text.lower(), phrase)
        self.assertNotIn("CPKS induction 18", self.soup.select_one("#cpks-title").get_text())
        self.assertNotIn("remains unsupported", self.data["gridEvidence"]["status"])
        scripts = "\n".join(s.string or "" for s in self.soup.select("script"))
        self.assertIn("findLast(c=>c.verified_basis_matched)", scripts)
        self.assertIn("label.textContent=`${pair.cpu.label}", scripts)
        self.assertNotIn("label.textContent='Protein157 · CPU", scripts)
        self.assertIn("group.arms.some(a=>a.arm==='fixed75')?'fixed75':'fixed99'", scripts)
        self.assertNotRegex(scripts, r"fetch\s*\(")
        self.assertLess(len(str(self.soup).encode()), 3*1024*1024)

    def test_render_idempotence_and_embedded_evidence(self):
        # Collect two short local renders; never invoke builder's unsafe default.
        page = DOCS / "breakdown-2026-10-04.html"
        before = page.read_bytes()
        with contextlib.redirect_stdout(io.StringIO()):
            publisher.publication.render_preserving_history(DOCS)
            middle = page.read_bytes()
            publisher.publication.render_preserving_history(DOCS)
        self.assertEqual(before, middle)
        self.assertEqual(middle, page.read_bytes())
        _, data = payload(page)
        self.assertEqual(data["gridEvidence"][publisher.KEY], [publisher.verify(RAW, DOCS)])


if __name__ == "__main__":
    unittest.main()
