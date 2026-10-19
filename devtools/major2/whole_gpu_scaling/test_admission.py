"""Host/fake regressions only; scheduler calls are mocked or forbidden."""
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import admission as a
import config as s
import custody
import controller
import execute

UID = 3170273
XDM = dict(job_id="14179965", job_name="xdm-gpu-cases", state="PENDING",
           account="gts-cs207", qos="inferno", partition="gpu-h200",
           command=s.P+"/runs/xdm-gpu/20261019/job.sbatch",
           workdir=s.P+"/runs/xdm-gpu/20261019")
# Observed pending identities: preparation admission-blocker.json and user
# repair request. Live repair snapshot also observes these identities RUNNING.


def snapshot(jobs):
    return a.HEADER+"\n"+"".join("|".join(str(v) for v in (
        j["job_id"], a.OWNER, UID, j["state"], j["account"], j["qos"],
        j["job_name"], j["partition"], j["command"], j["workdir"]))+"\n" for j in jobs)


def legacy_output(*args, **kwargs):
    if args[0] == "squeue":
        return "|".join(XDM[k] for k in ("job_id", "qos", "job_name", "partition"))
    if args == ("scontrol", "show", "job", "-o", XDM["job_id"]):
        return (f"JobId={XDM['job_id']} UserId={a.OWNER}({UID}) "
                f"JobName={XDM['job_name']} JobState=PENDING Account=gts-cs207 QOS=inferno "
                f"Partition=gpu-h200 Command={XDM['command']} WorkDir={XDM['workdir']}")
    raise AssertionError("unexpected scheduler operation: "+str(args))


class AdmissionTests(unittest.TestCase):
    def call(self, jobs):
        with patch.object(a.os, "getuid", return_value=UID), \
             patch.object(a.pwd, "getpwuid", return_value=SimpleNamespace(pw_name=a.OWNER)), \
             patch.object(a, "query", return_value=snapshot(jobs)) as query, \
             patch.object(s.n, "output", side_effect=legacy_output), \
             patch.object(subprocess, "run", side_effect=AssertionError("scheduler write")):
            result = custody.queue()
        self.assertEqual(query.call_args.args, ("squeue", "-r", "-u", a.OWNER, "-o", a.FORMAT))
        return result

    def test_old_production_predicate_rejects_observed_pending_xdm(self):
        with patch.object(s.n, "output", side_effect=legacy_output):
            with self.assertRaisesRegex(RuntimeError, "active duplicate/H200 holder"):
                custody.legacy.queue()

    def test_production_custody_allows_observed_xdm(self):
        value = self.call([XDM])
        self.assertEqual(value["embers_used"], 0)
        self.assertEqual(value["observations"][0]["classification"], "unrelated")
        self.assertFalse(value["allocation_exclusivity_proven"])

    def test_both_observed_xdm_pending_or_running_permitted(self):
        for state in ("PENDING", "RUNNING"):
            jobs = [dict(XDM, state=state),
                    dict(XDM, job_id="14179966", job_name="xdm-gpu-tests", state=state)]
            self.assertEqual(self.call(jobs)["own_queue_rows"], 2)

    def test_same_campaign_name_rejected_including_partial(self):
        for state in ("PENDING", "RUNNING", "COMPLETING"):
            for name in (a.CAMPAIGN, a.CAMPAIGN+"-partial", a.CAMPAIGN+"-20261019-v2"):
                with self.assertRaisesRegex(RuntimeError, "campaign duplicate"):
                    self.call([dict(XDM, state=state, job_name=name)])

    def test_command_source_and_run_namespace_duplicates(self):
        for key, value in (("command", s.RUN+"/campaign.sbatch"),
                           ("command", s.SOURCE+"/job.sbatch"),
                           ("command", "/different/checkout/devtools/major2/whole_gpu_scaling/controller.py"),
                           ("command", s.RUN+"-partial/job.sbatch"),
                           ("workdir", s.RUN), ("workdir", s.SOURCE), ("workdir", s.RUN+"-v2")):
            with self.subTest(key=key, value=value):
                with self.assertRaisesRegex(RuntimeError, "campaign duplicate"):
                    self.call([dict(XDM, **{key:value})])

    def test_other_major2_campaign_not_duplicate(self):
        value = self.call([dict(XDM, job_name="major2-blocked-disp0",
                               command=s.P+"/runs/major2-other/job.sbatch",
                               workdir=s.P+"/runs/major2-other")])
        self.assertEqual(value["observations"][0]["classification"], "unrelated")

    def test_headroom_counts_every_expanded_embers_task(self):
        jobs = [dict(XDM, job_id="123_"+str(i), qos="embers") for i in range(49)]
        value = self.call(jobs)
        self.assertEqual((value["embers_used"], value["embers_available"]), (49, 1))
        jobs.append(dict(XDM, job_id="123_49", qos="embers"))
        with self.assertRaisesRegex(RuntimeError, "headroom"):
            self.call(jobs)

    def test_explicit_header_only_is_empty_not_bare_empty(self):
        self.assertEqual(self.call([])["embers_available"], 50)
        for text in ("", " ", "\n", a.HEADER, snapshot([XDM])[:-1]):
            with self.assertRaises(RuntimeError):
                a.parse(text, UID)

    def test_owner_mismatch_rejected(self):
        for text in (snapshot([XDM]).replace("|awallace43|", "|foreign|"),
                     snapshot([XDM]).replace("|3170273|", "|999|")):
            with self.assertRaisesRegex(RuntimeError, "owner"):
                a.parse(text, UID)
        with patch.object(a.pwd, "getpwuid", return_value=SimpleNamespace(pw_name="foreign")), \
             patch.object(a, "query") as query:
            with self.assertRaisesRegex(RuntimeError, "owner"):
                a.queue()
            query.assert_not_called()

    def test_malformed_metadata_rejected(self):
        for key, value in (("job_id", "123_[1-9]"), ("state", "UNKNOWN"),
                           ("command", "(null)"), ("workdir", "relative"),
                           ("command", "/a/../b"), ("job_name", ""),
                           ("qos", ""), ("partition", ""), ("job_name", "bad|delimiter")):
            with self.subTest(key=key):
                with self.assertRaises(RuntimeError):
                    a.parse(snapshot([dict(XDM, **{key:value})]), UID)
        for text in (snapshot([XDM, XDM]), snapshot([XDM]).replace("WORK_DIR", "UNKNOWN"),
                     snapshot([XDM]).replace("\n", "\r\n")):
            with self.assertRaises(RuntimeError):
                a.parse(text, UID)

    def test_row_and_byte_caps(self):
        with patch.object(a, "MAX_ROWS", 0):
            with self.assertRaises(RuntimeError):
                a.parse(snapshot([XDM]), UID)
        with patch.object(a, "MAX_BYTES", 5):
            with self.assertRaises(RuntimeError):
                a.parse(snapshot([XDM]), UID)

    def test_failed_query_never_returns_empty_queue(self):
        with patch.object(a.pwd, "getpwuid", return_value=SimpleNamespace(pw_name=a.OWNER)), \
             patch.object(a, "query", side_effect=subprocess.CalledProcessError(1, "squeue")):
            with self.assertRaises(subprocess.CalledProcessError):
                a.queue()

    def test_bounded_query_failure_stderr_and_output_cap(self):
        for code in ("raise SystemExit(2)", "import sys;sys.stderr.write('warning')",
                     "print('x'*100)"):
            with patch.object(a, "MAX_BYTES", 16):
                with self.assertRaises(RuntimeError):
                    a.query(sys.executable, "-c", code)
        self.assertEqual(a.query(sys.executable, "-c", "print('ok')"), "ok\n")

    def test_query_timeout_kills_child(self):
        # Force timeout without a wait/poll loop or a long-lived fake child.
        with patch.object(a.time, "monotonic", side_effect=[0, 36]):
            with self.assertRaisesRegex(RuntimeError, "timeout"):
                a.query(sys.executable, "-c", "pass")

    def test_guard_reuse_and_no_exclusivity_claim_from_queue(self):
        self.assertIs(controller.Engine.identify_gpu, controller.rc.Engine.identify_gpu)
        text = (s.HERE/"controller.py").read_text()
        self.assertIn('"no concurrent GPU holder"', text)
        self.assertIn("exclusive_pid", (s.HERE/"execute.py").read_text())
        for output in ("GPU-other, 123\n", "GPU-own, 456\n", "GPU-own, 123\nGPU-own, 456\n"):
            with patch.object(execute.os, "getpid", return_value=123), \
                 patch.object(execute.subprocess, "check_output", return_value=output):
                with self.assertRaises(RuntimeError):
                    execute.exclusive_pid("GPU-own")
        self.assertFalse(self.call([XDM])["allocation_exclusivity_proven"])

    def test_before_child_foreign_holder_rejected_without_launch(self):
        engine = controller.Engine.__new__(controller.Engine)
        engine.pins = Mock()
        engine.gpu_selector = "GPU-own"
        engine.query = Mock(return_value="GPU-own, 999\n")
        engine.run_command = Mock(side_effect=AssertionError("must not launch"))
        with self.assertRaisesRegex(RuntimeError, "concurrent GPU holder"):
            engine.case(s.SEQUENCE[0])
        engine.run_command.assert_not_called()
        self.assertIn("--id=GPU-own", engine.query.call_args.args[0])

    def test_actual_logical_to_physical_uuid_mapping_guard(self):
        rc = controller.rc
        uuid = "GPU-01234567-1234-1234-1234-0123456789ab"
        other = "GPU-11234567-1234-1234-1234-0123456789ab"
        proof = dict(count=1, logical_device=0, header_version=13030, runtime_version=13030,
                     uuid=uuid, name="NVIDIA H200", memory_bytes=141312*1048576)
        for changes, scheduler, nvml, accepted in (
                ({}, "1", uuid, True), ({}, other, uuid, False),
                ({"logical_device":1}, "1", uuid, False),
                ({"count":2}, "1", uuid, False), ({}, "1", other, False)):
            engine = controller.Engine.__new__(controller.Engine)
            engine.m = {"identity_helper":rc.IDENTITY}
            engine.raw = Path("/fake/raw")
            engine.env = {"CUDA_VISIBLE_DEVICES":"1"}
            engine.scheduler_gpu = scheduler
            engine.run_command = Mock()
            engine.query = Mock(return_value=f"NVIDIA H200, {nvml}, 141312, 580.95, 1")
            with patch.object(rc.p, "sha", return_value=rc.IDENTITY["sha256"]), \
                 patch.object(rc.p, "read_json", return_value=dict(proof, **changes)), \
                 patch.object(rc.o, "atomic"):
                if accepted:
                    engine.identify_gpu()
                    self.assertEqual(engine.gpu_selector, uuid)
                    self.assertIn("--id="+uuid, engine.query.call_args.args[0])
                else:
                    with self.assertRaises(RuntimeError):
                        engine.identify_gpu()

    def test_custody_binding_not_global_patch(self):
        self.assertIs(custody.queue, a.queue)
        self.assertIsNot(custody.legacy.queue, a.queue)
        for function in (custody.collect, custody.audit):
            self.assertIn("queue()", __import__("inspect").getsource(function))


if __name__ == "__main__":
    unittest.main()
