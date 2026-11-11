import copy
import unittest

from cachy_followup import require_clean_sanitizer, require_release


class FollowupAdmission(unittest.TestCase):
    def test_only_complete_same_source_release(self):
        release = dict(status="PASS_SMALL_CACHY_ONLY", source_pin="pin",
                       runtime_config_sha256="config",
                       gpu_small_case_correctness_verified=True,
                       cases=[dict(returncode=0) for _ in range(10)])
        require_release(release, "pin", "config")
        for key, value in (("status", "FAILED"), ("source_pin", "wrong"),
                           ("runtime_config_sha256", "wrong"),
                           ("gpu_small_case_correctness_verified", False),
                           ("cases", release["cases"][:9])):
            with self.assertRaises(RuntimeError):
                require_release({**release, key: value}, "pin", "config")
        failed = copy.deepcopy(release)
        failed["cases"][3]["returncode"] = 9
        with self.assertRaises(RuntimeError):
            require_release(failed, "pin", "config")

    def test_sanitizer_requires_all_zero_error_summaries(self):
        require_clean_sanitizer("========= ERROR SUMMARY: 0 errors\n")
        for text in ("", "ERROR SUMMARY: 1 error", "ERROR SUMMARY: 2 errors",
                     "ERROR SUMMARY: 0 errors\nERROR SUMMARY: 1 error"):
            with self.assertRaises(RuntimeError):
                require_clean_sanitizer(text)


if __name__ == "__main__":
    unittest.main()
