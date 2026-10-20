import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    "progress_watch", Path(__file__).with_name("cuest_shared_progress_watch.py"))
watch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watch)


class ProgressWatchTests(unittest.TestCase):
    def test_held_lease_cpu_baseline_not_misclassified_as_idle(self):
        self.assertEqual(watch.action(dict(manifest_status="allocated", lease="held")),
                         "ACTIVE_LEASE_CPU_CONTROL_OR_GPU_WORK")

    def test_free_lease_requests_reconciliation_not_automatic_job_action(self):
        self.assertEqual(watch.action(dict(manifest_status="allocated", lease="free")),
                         "ALERT_NO_ACTIVE_LEASE_RECONCILE_ACTUAL_STATE")

    def test_quarantine_alerts_even_with_held_lease(self):
        self.assertEqual(watch.action(dict(quarantined=True, lease="held")),
                         "ALERT_QUARANTINED")

    def test_unknown_lease_never_claims_work_or_idle(self):
        self.assertEqual(watch.action(dict(manifest_status="allocated", lease="unknown")),
                         "ALERT_LEASE_UNKNOWN")

    def test_pending_manifest_is_not_grant(self):
        self.assertEqual(watch.action(dict(manifest_status="pending", lease="free")),
                         "HOLD_MANIFEST_NOT_ALLOCATED")

    def test_held_lease_with_all_steps_terminal_needs_owner_attention(self):
        self.assertEqual(watch.action(dict(manifest_status="allocated", lease="held",
                                          steps=[dict(terminal=True)])),
                         "ALERT_HELD_LEASE_WITH_NO_UNFINISHED_STEP")

    def test_only_positive_official_terminal_event_ends_monitoring(self):
        for s in ["", "QUERY-FAILED x1", "14214205 RUNNING", "14214205 UNKNOWN"]:
            self.assertFalse(watch.official_terminal(s))
        self.assertTrue(watch.official_terminal("2026-10-21 ALL-TERMINAL 14214205 (SLURM state only)"))


if __name__ == "__main__":
    unittest.main()
