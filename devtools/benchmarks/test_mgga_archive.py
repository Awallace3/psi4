import json
from pathlib import Path
import tempfile
import unittest

from mgga_archive import consolidate, replay
from mgga_ie import summarize
from mgga_probe import PROFILES


class ArchiveTests(unittest.TestCase):
    def fixture(self, directory, failed=False):
        records, results = [], {}
        for build in ("host", "cuda"):
            for route in ("cpu", "gpu-jk", "gpu-xc"):
                for fragment, energy in (("AB", -20.01), ("A", -10.), ("B", -10.)):
                    name = f"water-m06-{build}-{route}-{fragment}"
                    bad = failed and build == "host" and route == "gpu-xc" and fragment == "A"
                    code = 1 if bad else 0
                    result = dict(system="water", functional="m06", route=route,
                                  fragment=fragment, ok=True, nbf=20, energy_hartree=energy)
                    work = directory / name
                    work.mkdir()
                    (work / "result.json").write_text(json.dumps(result))
                    records.append(dict(name=name, returncode=code))
                    results[(build, "water", "m06", route, fragment)] = dict(result, ok=not bad)
        manifest = dict(records=records, packages={"host": "", "cuda": ""},
                        cases=[["water", "basis"]], functionals=["m06"])
        (directory / "manifest.json").write_text(json.dumps(manifest))
        rows = [r for r in summarize(results) if r["system"] == "water" and r["functional"] == "m06"]
        (directory / "comparisons.json").write_text(json.dumps(rows))

    def test_replay_validates_exit_codes(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self.fixture(directory, failed=True)
            rows, failures = replay(directory)
            self.assertEqual(sum(r["complete"] for r in rows), 3)
            self.assertEqual(len(failures), 1)
            self.assertTrue(failures[0]["reported_ok"])

    def test_replay_detects_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self.fixture(directory)
            (directory / "comparisons.json").write_text("[]")
            with self.assertRaises(AssertionError):
                replay(directory)

    def test_profiles_are_single_variable_probes(self):
        self.assertEqual(PROFILES["baseline"], {})
        self.assertTrue(all(len(changes) == 1 for label, changes in PROFILES.items()
                            if label != "baseline"))


if __name__ == "__main__":
    unittest.main()
