"""LOCAL standard-library regression fixtures. Never executes SSH or sbatch."""

import ast
from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import protocol_model as model


class Crash(BaseException):
    """Simulated interruption; not an actual process kill or power-loss test."""


def fixture_intent(parent):
    return model.Intent(
        artifact_parent=str(parent), attempt_name="attempt",
        source_pin="a" * 40, batch_sha256="b" * 64, payload_sha256="c" * 64,
        preflight_policy_sha256="d" * 64,
        authorization_decision="OFFLINE:new-explicit-decision",
        ambiguity_handling_decision="OFFLINE:prior-history-handled-externally",
        owner_turn="OFFLINE:new-single-turn")


def assert_order(test, events):
    if "rpc.begin" in events:
        for name in ("attempt.parent_fsync", "intent.json.file_fsync",
                     "intent.json.directory_fsync", "intent.revalidated",
                     "rpc-intent.json.file_fsync", "rpc-intent.json.directory_fsync"):
            test.assertLess(events.index(name), events.index("rpc.begin"), name)
    if "query.begin" in events:
        for name in ("rpc.stdout.directory_fsync", "rpc.stderr.directory_fsync",
                     "rpc-result.json.directory_fsync", "job-id.json.file_fsync",
                     "job-id.json.directory_fsync"):
            test.assertLess(events.index(name), events.index("query.begin"), name)


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="disp0-offline-model-")
        self.addCleanup(self.temp.cleanup)
        self.parent = Path(self.temp.name).resolve()
        self.intent = fixture_intent(self.parent)

    def run_fixture(self, name="attempt", scheduler=None, **kwargs):
        # Each distinct model fixture explicitly creates a distinct reviewed
        # intent. This helper is not a real approval or a replay mechanism.
        intent = replace(self.intent, attempt_name=name)
        scheduler = scheduler or model.OfflineScheduler()
        trace = []
        result = model.run_model_once(
            self.parent, name, intent, intent.digest(), scheduler,
            trace=trace, **kwargs)
        return result, scheduler, trace

    def test_one_call_durable_order_and_pinned_receipt(self):
        result, scheduler, trace = self.run_fixture()
        self.assertEqual(result.state, "COMPLETE")
        self.assertTrue(result.id_durable)
        self.assertEqual((scheduler.submit_calls, scheduler.query_calls), (1, 1))
        assert_order(self, trace)
        root = self.parent / "attempt"
        self.assertEqual(root.stat().st_mode & 0o777, 0o700)
        self.assertEqual((root / "intent.json").stat().st_mode & 0o777, 0o400)
        self.assertEqual((root / "intent.json").read_bytes(), self.intent.bytes())
        receipt = json.loads((root / "job-id.json").read_bytes())
        self.assertEqual(receipt["intent_sha256"], self.intent.digest())
        self.assertEqual(receipt["job_id"], result.candidate_id)

    def test_legacy_missing_directory_fsync_trace_is_red_capable(self):
        bad = ["attempt.parent_fsync", "intent.json.file_fsync", "intent.revalidated",
               "rpc-intent.json.file_fsync", "rpc.begin"]
        with self.assertRaises((AssertionError, ValueError)):
            assert_order(self, bad)

    def test_single_invocation_has_no_human_input_or_live_execution_adapter(self):
        tree = ast.parse(Path(model.__file__).read_text())
        forbidden = {"subprocess", "paramiko", "asyncssh", "requests", "socket"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                self.assertTrue(forbidden.isdisjoint(x.name.split(".")[0] for x in node.names))
            if isinstance(node, ast.ImportFrom):
                self.assertNotIn((node.module or "").split(".")[0], forbidden)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                self.assertNotIn(node.func.id, {"input", "exec", "eval"})
        with self.assertRaises(TypeError):
            model.run_model_once(self.parent, "none", self.intent, self.intent.digest(), object())

    def test_reviewed_digest_and_explicit_fixture_fields_fail_closed(self):
        with self.assertRaises(ValueError):
            model.run_model_once(self.parent, "attempt", self.intent, "0" * 64,
                                 model.OfflineScheduler())
        changes = [
            {"authorization_decision": ""}, {"ambiguity_handling_decision": ""},
            {"owner_turn": ""}, {"argv": ("sbatch", "--parsable")},
            {"gpus": 1}, {"requeue": True}, {"cpus": 2}, {"wall_seconds": 601},
            {"memory_mib": 2049}, {"cpus": True}, {"gpus": False}, {"requeue": 0}]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                altered = replace(self.intent, **change)
                model.run_model_once(self.parent, "attempt", altered, altered.digest(),
                                     model.OfflineScheduler())
        self.assertEqual(list(self.parent.iterdir()), [])

    def test_same_reviewed_intent_cannot_select_another_name(self):
        first, _, _ = self.run_fixture()
        self.assertEqual(first.state, "COMPLETE")
        other = model.OfflineScheduler()
        with self.assertRaises(ValueError):
            model.run_model_once(self.parent, "another-name", self.intent,
                                 self.intent.digest(), other)
        self.assertEqual(other.submit_calls, 0)

    def test_same_reviewed_intent_cannot_select_another_parent(self):
        first, _, _ = self.run_fixture()
        self.assertEqual(first.state, "COMPLETE")
        with tempfile.TemporaryDirectory(prefix="disp0-offline-other-") as temporary:
            other = model.OfflineScheduler()
            with self.assertRaises(ValueError):
                model.run_model_once(Path(temporary).resolve(), "attempt", self.intent,
                                     self.intent.digest(), other)
            self.assertEqual(other.submit_calls, 0)

    def test_other_cluster_is_rejected_even_when_fake_reply_matches(self):
        altered = replace(self.intent, cluster="OTHER_CLUSTER")
        scheduler = model.OfflineScheduler(reply=model.Reply(0, b"123;OTHER_CLUSTER\n"))
        with self.assertRaises(ValueError):
            model.run_model_once(self.parent, "attempt", altered, altered.digest(), scheduler)
        self.assertEqual(scheduler.submit_calls, 0)

    def test_failed_preflight_never_dispatches(self):
        result, scheduler, trace = self.run_fixture(scheduler=model.OfflineScheduler(preflight_ok=False))
        self.assertEqual(result.state, "PREFLIGHT_STOP")
        self.assertEqual((scheduler.submit_calls, scheduler.query_calls), (0, 0))
        self.assertNotIn("rpc.begin", trace)

    def test_stored_intent_tamper_is_detected_before_dispatch(self):
        def tamper(event):
            if event == "preflight.done":
                path = self.parent / "attempt/intent.json"
                path.chmod(0o600)
                path.write_bytes(b'{"changed":true}\n')
        result, scheduler, _ = self.run_fixture(checkpoint=tamper)
        self.assertEqual(result.state, "PREFLIGHT_STOP")
        self.assertIn("changed", result.error)
        self.assertEqual(scheduler.submit_calls, 0)

    def test_fsync_failures_never_query_or_retry(self):
        before_rpc = ["attempt.before_parent_fsync", "intent.json.before_file_fsync",
                      "intent.json.before_directory_fsync", "rpc-intent.json.before_file_fsync",
                      "rpc-intent.json.before_directory_fsync"]
        after_rpc = ["rpc.stdout.before_file_fsync", "rpc.stdout.before_directory_fsync",
                     "rpc.stderr.before_directory_fsync", "rpc-result.json.before_directory_fsync",
                     "job-id.json.before_file_fsync", "job-id.json.before_directory_fsync"]
        for index, point in enumerate(before_rpc + after_rpc):
            with self.subTest(point=point):
                def fail(event):
                    if event == point:
                        raise OSError("OFFLINE injected fsync failure")
                result, scheduler, trace = self.run_fixture(str(index), checkpoint=fail)
                self.assertEqual(scheduler.submit_calls, int(point in after_rpc))
                self.assertEqual(scheduler.query_calls, 0)
                self.assertNotEqual(result.state, "COMPLETE")
                assert_order(self, trace)

    def test_actual_fsync_call_failures_at_parent_and_id_directory(self):
        actual_fsync = os.fsync
        for index, point in enumerate(
                ["attempt.before_parent_fsync", "job-id.json.before_directory_fsync"]):
            with self.subTest(point=point):
                last = []
                failures = []
                def checkpoint(event):
                    last[:] = [event]
                def fail_syscall(fd):
                    if last == [point]:
                        failures.append(fd)
                        raise OSError("OFFLINE mocked os.fsync syscall failure")
                    return actual_fsync(fd)
                with mock.patch.object(model.os, "fsync", side_effect=fail_syscall):
                    result, scheduler, trace = self.run_fixture("syscall-" + str(index),
                                                               checkpoint=checkpoint)
                self.assertEqual(len(failures), 1)
                self.assertEqual(scheduler.submit_calls, index)
                self.assertEqual(scheduler.query_calls, 0)
                self.assertFalse(result.id_durable)
                self.assertNotEqual(result.state, "COMPLETE")
                assert_order(self, trace)

    def test_query_proof_persistence_failure_keeps_durable_identity(self):
        actual_fsync = os.fsync
        last = []
        failures = []
        def checkpoint(event):
            last[:] = [event]
        def fail_syscall(fd):
            if last == ["query.json.before_directory_fsync"]:
                failures.append(fd)
                raise OSError("OFFLINE query proof directory fsync failure")
            return actual_fsync(fd)
        with mock.patch.object(model.os, "fsync", side_effect=fail_syscall):
            result, scheduler, trace = self.run_fixture(checkpoint=checkpoint)
        self.assertEqual(len(failures), 1)
        self.assertEqual(result.state, "ACCEPTED_QUERY_UNRESOLVED")
        self.assertTrue(result.id_durable)
        self.assertTrue((self.parent / "attempt/job-id.json").is_file())
        self.assertEqual((scheduler.submit_calls, scheduler.query_calls), (1, 1))
        assert_order(self, trace)

    def test_every_success_trace_crash_point_refuses_reentry(self):
        _, _, baseline = self.run_fixture("baseline")
        for index, point in enumerate(baseline):
            with self.subTest(point=point):
                name = "crash-" + str(index)
                trace = []
                scheduler = model.OfflineScheduler()
                intended = replace(self.intent, attempt_name=name)
                def crash(event):
                    if event == point:
                        raise Crash(point)
                with self.assertRaises(Crash):
                    model.run_model_once(self.parent, name, intended, intended.digest(),
                                         scheduler, trace=trace, checkpoint=crash)
                self.assertLessEqual(scheduler.submit_calls, 1)
                self.assertLessEqual(scheduler.query_calls, 1)
                assert_order(self, trace)
                before = {p.name: p.read_bytes() for p in (self.parent / name).iterdir() if p.is_file()}
                rerun, fake, _ = self.run_fixture(name)
                self.assertEqual(rerun.state, "REFUSED_EXISTING_ATTEMPT")
                self.assertEqual((fake.submit_calls, fake.query_calls), (0, 0))
                self.assertEqual(before, {
                    p.name: p.read_bytes() for p in (self.parent / name).iterdir() if p.is_file()})

    def test_accepted_fake_rpc_with_lost_reply_stays_unresolved_no_query(self):
        result, scheduler, trace = self.run_fixture(scheduler=model.OfflineScheduler(lose_reply=True))
        self.assertEqual(result.state, "SUBMISSION_UNRESOLVED")
        self.assertEqual((scheduler.submit_calls, scheduler.fake_acceptances, scheduler.query_calls),
                         (1, 1, 0))
        self.assertIsNone(result.candidate_id)
        self.assertTrue((self.parent / "attempt/rpc-intent.json").is_file())
        assert_order(self, trace)

    def test_nonzero_malformed_wrong_cluster_and_oversized_replies_stop(self):
        replies = [model.Reply(1, b"", b"OFFLINE socket timeout"),
                   model.Reply(0, b"", b""), model.Reply(0, b"123\n456\n", b""),
                   model.Reply(0, b"123;WRONG_CLUSTER\n", b""),
                   model.Reply(0, b"x" * 65537, b"")]
        for index, reply in enumerate(replies):
            with self.subTest(reply=index):
                result, scheduler, trace = self.run_fixture(
                    "reply-" + str(index), scheduler=model.OfflineScheduler(reply=reply))
                self.assertEqual(result.state, "SUBMISSION_UNRESOLVED")
                self.assertEqual((scheduler.submit_calls, scheduler.query_calls), (1, 0))
                self.assertFalse(result.id_durable)
                assert_order(self, trace)

    def test_query_failure_preserves_durable_id_without_retry(self):
        result, scheduler, trace = self.run_fixture(scheduler=model.OfflineScheduler(query_fails=True))
        self.assertEqual(result.state, "ACCEPTED_QUERY_UNRESOLVED")
        self.assertTrue(result.id_durable)
        self.assertEqual((scheduler.submit_calls, scheduler.query_calls), (1, 1))
        self.assertTrue((self.parent / "attempt/job-id.json").is_file())
        assert_order(self, trace)

    def test_existing_namespace_and_path_escape_never_adopt(self):
        existing = self.parent / "existing"
        existing.mkdir()
        (existing / "original").write_bytes(b"do not change")
        result, fake, _ = self.run_fixture("existing")
        self.assertEqual(result.state, "REFUSED_EXISTING_ATTEMPT")
        self.assertEqual((existing / "original").read_bytes(), b"do not change")
        self.assertEqual(fake.submit_calls, 0)
        for name in ["../escape", "/absolute", ".", ".."]:
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.run_fixture(name)


if __name__ == "__main__":
    unittest.main()
