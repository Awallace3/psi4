"""Observed BatchFlag=1 CPU request; stdlib host fixtures, no scheduler writes."""
import hashlib
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import admission as a
from test_admission import UID, snapshot

CPU = dict(job_id="14182160", job_name="adiis", state="RUNNING",
           account="gts-cs207", qos="inferno", partition="cpu-small",
           command="(null)",
           workdir="/storage/project/r-cs207-0/awallace43/runs/psi4-vv10-split/grac-cation-dev")
# Exact selected fields of actual controller observation 2026-10-19; complete
# raw output/hash is preserved in command-admission/observed-controller.txt.
RAW = ("JobId=14182160 JobName=adiis UserId=awallace43(3170273) "
       "Account=gts-cs207 QOS=inferno JobState=RUNNING BatchFlag=1 "
       "Partition=cpu-small ReqTRES=cpu=16,mem=64G,node=1,billing=4864 "
       "AllocTRES=cpu=16,mem=64G,node=1,billing=4864 Command=(null) "
       "WorkDir="+CPU["workdir"]+" TresPerTask=cpu=16\n")


class CommandAdmissionTests(unittest.TestCase):
    def call(self, jobs=None, raw=RAW):
        calls = []
        def query(*args, **kwargs):
            calls.append((args, kwargs))
            if args == ("squeue", "-r", "-u", a.OWNER, "-o", a.FORMAT):
                return snapshot([CPU] if jobs is None else jobs)
            self.assertEqual(args, ("scontrol", "show", "job", "-o", CPU["job_id"]))
            self.assertEqual(kwargs, dict(timeout=5, max_bytes=65536))
            if isinstance(raw, BaseException):
                raise raw
            return raw
        with patch.object(a.os, "getuid", return_value=UID), \
             patch.object(a.pwd, "getpwuid", return_value=SimpleNamespace(pw_name=a.OWNER)), \
             patch.object(a, "query", side_effect=query), \
             patch.object(subprocess, "run", side_effect=AssertionError("scheduler write")):
            result = a.queue()
        self.assertEqual(len(calls), 2)
        return result["observations"][0]

    def test_observed_production_null_cpu(self):
        value = self.call()
        self.assertEqual(value["classification"], "unrelatedCPUproven-no-GPU")
        self.assertFalse(value["command_available"])
        self.assertEqual(value["command"], "(null)")
        self.assertFalse(value["allocation_exclusivity_proven"])
        proof = value["controller_proof"]
        self.assertEqual(proof["raw_sha256"], hashlib.sha256(RAW.encode()).hexdigest())
        self.assertEqual(proof["selected"]["BatchFlag"], "1")
        self.assertNotIn("interactive", str(value).lower())
        self.assertIn("Gres", proof["missing_fields"])

    def test_parse_without_proof_still_rejects(self):
        with self.assertRaises(RuntimeError):
            a.parse(snapshot([CPU]), UID)

    def test_controller_identity_and_state_fail_closed(self):
        for old, new in (("14182160", "14182161"), ("awallace43", "foreign"),
                         ("3170273", "1"), ("adiis", "other"), (CPU["workdir"], "/other"),
                         ("cpu-small", "gpu-h200"), ("RUNNING", "COMPLETED"),
                         ("RUNNING", "PENDING"), ("gts-cs207", "other"),
                         ("inferno", "embers"), ("Command=(null)", "Command=/known"),
                         ("BatchFlag=1", "")):
            with self.subTest(old=old), self.assertRaises(RuntimeError):
                self.call(raw=RAW.replace(old, new))

    def test_no_batchflag_zero_shortcut(self):
        with self.assertRaises(RuntimeError):
            self.call(raw=RAW.replace("BatchFlag=1", "BatchFlag=0").replace(
                "ReqTRES=cpu=16,mem=64G,node=1,billing=4864", "ReqTRES=(null)"))

    def test_resource_fail_closed(self):
        for raw in (RAW.replace("cpu=16,mem=64G,node=1,billing=4864", "cpu=16", 1),
                    RAW.replace("AllocTRES=cpu=16,mem=64G,node=1,billing=4864", ""),
                    RAW.replace("mem=64G", "mem=nan"),
                    RAW.replace("cpu=16", "cpu=0"),
                    RAW.replace("node=1", "node=-1"),
                    RAW.replace("cpu=16", "cpu=18446744073709551616"),
                    RAW.replace("billing=4864", "billing=4864,gres/gpu=1"),
                    RAW.replace("TresPerTask=cpu=16", "TresPerTask=gres/gpu:h200=1"),
                    RAW.rstrip()+" Gres=gpu:h200:1\n",
                    RAW.rstrip()+" TresPerNode=gres/gpu=1\n",
                    RAW.rstrip()+" Gres=unknown\n",
                    RAW.rstrip()+" ReqTRES=cpu=16\n",
                    RAW.replace("mem=64G", "mem=64G,mem=1G")):
            with self.subTest(raw=raw), self.assertRaises(RuntimeError):
                self.call(raw=raw)

    def test_invalid_rows_never_query_controller(self):
        for changes in (dict(partition="gpu-h200"), dict(partition="cpu-unknown"),
                        dict(workdir=""), dict(workdir="/a/../b"),
                        dict(command=""), dict(command="unknown"),
                        dict(job_name=a.CAMPAIGN), dict(workdir="/runs/"+a.CAMPAIGN)):
            with patch.object(a, "query") as query, self.assertRaises(RuntimeError):
                a.parse(snapshot([dict(CPU, **changes)]), UID, cpu_proof=query)
            query.assert_not_called()
        for text in (snapshot([CPU, CPU]),
                     snapshot([CPU]).replace("|awallace43|", "|foreign|"),
                     snapshot([dict(CPU, job_id=str(i+1), qos="embers") for i in range(50)]),
                     snapshot([dict(CPU, job_id=str(i+1)) for i in range(9)])):
            with patch.object(a, "query") as query, self.assertRaises(RuntimeError):
                a.parse(text, UID, cpu_proof=query)
            query.assert_not_called()

    def test_query_failure_ambiguous_or_capped(self):
        for raw in ("", RAW.rstrip(), RAW+RAW, RAW+"x"*65536,
                    RAW.replace("WorkDir=", "Missing="),
                    RuntimeError("controller disappeared/failed")):
            with self.subTest(raw=str(raw)[:40]), self.assertRaises(RuntimeError):
                self.call(raw=raw)

    def test_pending_needs_request_and_no_allocated_gpu(self):
        pending = dict(CPU, state="PENDING")
        raw = RAW.replace("RUNNING", "PENDING").replace(
            "AllocTRES=cpu=16,mem=64G,node=1,billing=4864 ", "")
        self.assertFalse(self.call([pending], raw)["command_available"])
        with self.assertRaises(RuntimeError):
            self.call([pending], raw.rstrip()+" AllocTRES=gres/gpu=1\n")

    def test_null_proof_deadline_and_byte_bounds(self):
        for code in ("print('x'*100)", "import sys;sys.stderr.write('warning')",
                     "raise SystemExit(2)"):
            import sys
            with self.assertRaises(RuntimeError):
                a.query(sys.executable, "-c", code, timeout=5, max_bytes=16)
        with patch.object(a.time, "monotonic", side_effect=[0, 6]):
            with self.assertRaisesRegex(RuntimeError, "timeout"):
                a.query(sys.executable, "-c", "pass", timeout=5, max_bytes=65536)


if __name__ == "__main__":
    unittest.main()
