import json
from pathlib import Path
import tempfile
import unittest

from progress import snapshot


class ProgressTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.base = self.root / "preflight/campaign"
        self.base.mkdir(parents=True)

    def save(self, name, obj):
        (self.base / name).write_text(json.dumps(obj))

    def manifest(self):
        self.save("manifest.json", {"systems": ["protein157"],
                                   "config": {"functionals": {"pbe": {}}, "repeats": 1},
                                   "command": ["campaign"]})

    def test_missing_manifest_is_not_completion(self):
        self.assertEqual(snapshot(self.root)[0]["state"], "UNKNOWN")

    def test_empty_records_are_in_progress(self):
        self.manifest()
        item = snapshot(self.root)[0]
        self.assertEqual(item["state"], "IN_PROGRESS")
        self.assertEqual(item["planned_workers"], 2)
        self.assertEqual(item["finished_workers"], 0)

    def test_failed_process_remains_failed(self):
        self.manifest()
        self.save("records.json", [{"ok": True, "process_returncode": -9, "ended_unix_s": 0}])
        self.save("FAILED.json", {"interrupted": False})
        item = snapshot(self.root)[0]
        self.assertEqual(item["state"], "FAILED")
        self.assertEqual(item["failed_workers"], 1)

    def test_complete_requires_marker(self):
        self.manifest()
        self.save("records.json", [{"ok": True, "process_returncode": 0, "ended_unix_s": 0}] * 2)
        self.save("COMPLETE.json", {"complete": True, "final_build_hashes_unchanged": True})
        self.assertEqual(snapshot(self.root)[0]["state"], "COMPLETE")

    def test_completion_marker_cannot_hide_missing_workers(self):
        self.manifest()
        self.save("COMPLETE.json", {"complete": True, "final_build_hashes_unchanged": True})
        self.assertEqual(snapshot(self.root)[0]["state"], "INCOMPLETE")


if __name__ == "__main__":
    unittest.main()
