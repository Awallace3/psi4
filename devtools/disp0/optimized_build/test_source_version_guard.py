import copy
import unittest

import source_version_guard as guard


class SourceVersionGuard(unittest.TestCase):
    def fixture(self):
        return {"shallow": "false", "tag_object": guard.TAG_OBJECT,
                "tag_commit": guard.TAG_COMMIT, "tag_type": "tag",
                "ancestor_exit": 0, "describe": "v1.11-658-g613511b",
                "head": "613511b0dfe914ed4bd241040dc6fc1a25277570", "clean": True,
                "metadata": {"__version_last_release": "1.11",
                             "__version__": "1.12a1.dev658",
                             "__version_long": "1.12a1.dev658+613511b",
                             "__version_is_clean": "True"}}

    def test_matching_full_history_and_generated_identity(self):
        guard.validate(self.fixture())

    def test_missing_history_wrong_tags_and_dirty_source_reject(self):
        for key, value in (("shallow", "true"), ("tag_object", "0" * 40),
                           ("tag_commit", "0" * 40), ("tag_type", "commit"),
                           ("ancestor_exit", 1), ("describe", "613511b"), ("clean", False)):
            evidence = self.fixture()
            evidence[key] = value
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                guard.validate(evidence)

    def test_undefined_or_foreign_metadata_reject(self):
        for key, value in (("__version__", "undefined"),
                           ("__version_long", "1.12a1.dev658+0000000"),
                           ("__version_is_clean", "False"),
                           ("__version_last_release", "1.10")):
            evidence = copy.deepcopy(self.fixture())
            evidence["metadata"][key] = value
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                guard.validate(evidence)


if __name__ == "__main__":
    unittest.main()
