"""Local mock guards. NEVER launches a scheduler, SSH, import or native build."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


submit = load("submit")
proof = load("proof")
PIN = "a" * 40


class SubmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.p = Path(self.temp.name)
        self.source = self.p / "gits/psi4.disp0-opt064-20261021-02/source"
        self.source.mkdir(parents=True, mode=0o700)
        self.source.parent.chmod(0o700)
        scripts = self.source / "devtools/disp0/optimized_build"
        scripts.mkdir(parents=True)
        for name in ("build.sbatch", "payload.sh"):
            (scripts / name).write_text(name)
        lock = self.p / f"runs/saptdft-cuest-disp0/build-{submit.SCIENCE}-14196330/environment-explicit.txt"
        lock.parent.mkdir(parents=True)
        lock.write_text("test-lock")
        self.attempt = self.source.parent / "submission-attempt"
        self.quota = self.source.parent / "quota.json"
        raw = self.source.parent / "quota.raw"
        raw.write_text("LOCAL FIXTURE")
        value = {"timestamp_unix": time.time(),
                 "scope": "HUMAN_REVIEWED_PROJECT_AND_SCRATCH_QUOTA"}
        for family in ("project", "scratch"):
            value[family] = {"free_gib": 1024, "free_inodes": 100000,
                             "raw_path": str(raw), "raw_sha256": submit.sha(raw)}
        self.quota.write_text(json.dumps(value))
        self.calls = []
        self.scheduler_environments = []
        self.actual = 0
        self.queue = ""
        self.route = "sbatch: Job 14217020 to start on partition cpu-small"
        self.response = "987654;phoenix-slurm\n"  # synthetic, never an actual ID
        self.reply_loss = False

    def fake(self, argv, **kwargs):
        self.calls.append(argv)
        stdout, stderr = "", ""
        if argv[0] == "git":
            if "rev-parse" in argv:
                stdout = PIN
        elif argv[0] == "squeue":
            stdout = self.queue
        elif "--test-only" in argv:
            self.scheduler_environments.append(kwargs.get("env"))
            stderr = self.route
        elif argv[0] == "sbatch":
            self.scheduler_environments.append(kwargs.get("env"))
            self.actual += 1
            self.assertTrue((self.attempt / "intent.json").exists())
            self.assertTrue((self.attempt / "dispatch-intent.json").exists())
            if self.reply_loss:
                raise subprocess.TimeoutExpired(argv, 20)
            stdout = self.response
        elif argv[0] == "scontrol":
            self.assertEqual((self.attempt / "accepted-job-id.txt").read_text(), "987654\n")
            stdout = "Account=gts-cs207-chemx QOS=embers Requeue=0"
        return {"argv": argv, "exit": 0, "stdout": stdout, "stderr": stderr}

    def invoke(self):
        fake_stat = type("Stat", (), {"f_bavail": 10**9, "f_frsize": 4096, "f_favail": 10**6})()
        with patch.object(submit, "P", self.p), patch.object(submit, "LOCK_SHA", submit.sha(
                self.p / f"runs/saptdft-cuest-disp0/build-{submit.SCIENCE}-14196330/environment-explicit.txt"
                )), patch.object(submit, "capture", self.fake), patch.object(
                submit.os, "statvfs", return_value=fake_stat), contextlib.redirect_stdout(io.StringIO()):
            submit.submit(self.source, PIN, self.attempt, self.quota, submit.sha(self.quota))

    def test_one_actual_call_durable_identity_before_query(self):
        self.invoke()
        self.assertEqual(self.actual, 1)
        self.assertEqual(json.loads((self.attempt / "test-only.json").read_text())["stderr"], self.route)
        self.assertEqual((self.attempt / "accepted-job-id.txt").read_text(), "987654\n")

    def test_existing_attempt_never_replays(self):
        self.invoke()
        with self.assertRaisesRegex(RuntimeError, "NEVER replay"):
            self.invoke()
        self.assertEqual(self.actual, 1)

    def test_polluted_sbatch_environment_is_removed_and_resources_explicit(self):
        with patch.dict(submit.os.environ, {"SBATCH_QOS": "inferno", "SBATCH_TIMELIMIT": "24:00:00",
                                           "SBATCH_GRES": "gpu:h200:1", "SALLOC_QOS": "inferno"}):
            self.invoke()
        self.assertEqual(len(self.scheduler_environments), 2)
        for env in self.scheduler_environments:
            self.assertFalse(any(name.startswith(("SBATCH_", "SALLOC_")) for name in env))
        for argv in [c for c in self.calls if c[0] == "sbatch"]:
            for resource in submit.RESOURCES:
                self.assertIn(resource, argv)

    def test_reply_loss_leaves_dispatch_intent_no_id_or_retry(self):
        self.reply_loss = True
        with self.assertRaises(subprocess.TimeoutExpired):
            self.invoke()
        self.assertTrue((self.attempt / "dispatch-intent.json").exists())
        self.assertFalse((self.attempt / "accepted-job-id.txt").exists())
        with self.assertRaisesRegex(RuntimeError, "NEVER replay"):
            self.invoke()
        self.assertEqual(self.actual, 1)

    def test_fsync_failure_prevents_rpc(self):
        with patch.object(submit.os, "fsync", side_effect=OSError("simulated fsync failure")):
            with self.assertRaises(OSError):
                self.invoke()
        self.assertEqual(self.actual, 0)

    def test_simulated_id_not_accepted_from_test_only(self):
        self.queue = "1|disp0-opt064|embers"
        with self.assertRaisesRegex(RuntimeError, "Duplicate"):
            self.invoke()
        self.assertFalse((self.attempt / "accepted-job-id.txt").exists())
        self.assertEqual(self.actual, 0)

    def test_wrong_route_no_rpc(self):
        self.route = "partition cpu-large"
        with self.assertRaisesRegex(RuntimeError, "CPU route"):
            self.invoke()
        self.assertEqual(self.actual, 0)

    def test_malformed_reply_never_queries_or_retries(self):
        self.response = "maybe accepted"
        with self.assertRaisesRegex(RuntimeError, "Unparseable"):
            self.invoke()
        self.assertEqual(self.actual, 1)
        self.assertFalse(any(c[0] == "scontrol" for c in self.calls))

    def test_stale_quota_prevents_rpc(self):
        data = json.loads(self.quota.read_text())
        data["timestamp_unix"] -= 601
        self.quota.write_text(json.dumps(data))
        with self.assertRaisesRegex(RuntimeError, "stale"):
            self.invoke()
        self.assertEqual(self.actual, 0)

    def test_real_fsync_order(self):
        events = []
        original = submit.durable
        def traced(path, data):
            original(path, data)
            events.append(Path(path).name)
        fake = self.fake
        def traced_fake(argv, **kwargs):
            if argv[0] == "sbatch" and "--test-only" not in argv:
                self.assertIn("dispatch-intent.json", events)
                events.append("RPC")
            if argv[0] == "scontrol":
                self.assertLess(events.index("accepted-job-id.txt"), len(events))
            return fake(argv, **kwargs)
        with patch.object(submit, "durable", traced):
            self.fake = traced_fake
            self.invoke()
        self.assertLess(events.index("intent.json"), events.index("dispatch-intent.json"))
        self.assertLess(events.index("dispatch-intent.json"), events.index("RPC"))


class BuildGuards(unittest.TestCase):
    def test_exact_world_and_reject_d17(self):
        world = {"git_commit": proof.E2, "version_major": 2, "compiler_major": 14}
        proof.validate_world(world)
        with self.assertRaises(RuntimeError):
            proof.validate_world({**world, "git_commit": "d17bb76e" + "0" * 32})

    def test_loader_paths_and_legacy_rejection(self):
        expected = ["$ORIGIN", "/private/cuda13/lib64", "/private/environment/lib"]
        good = {"rpath": [], "runpath": expected, "needed": ["libEinsums-abi1.so.1"]}
        proof.validate_paths(good, expected)
        for bad in ({**good, "rpath": ["/old/build"]},
                    {**good, "runpath": expected + [""]},
                    {**good, "needed": ["/old/build/libEinsums.so"]}):
            with self.assertRaises(RuntimeError):
                proof.validate_paths(bad, expected)

    def test_pins_and_no_old_packaging_hook(self):
        payload = (HERE / "payload.sh").read_text()
        self.assertIn(proof.E2, payload)
        self.assertIn("https://github.com/Awallace3/Einsums.git", payload)
        self.assertNotIn("revised_einsums_hook.cmake", payload)
        self.assertIn("-DEINSUMS_WITH_CUDA_ARCHITECTURES=90", payload)
        self.assertIn("-DEINSUMS_WITH_CCACHE=OFF", payload)
        self.assertIn("EINSUMS_PACKED_GEMM_BATCH_PROMOTION=1", payload)
        self.assertNotIn("-- -k 0", payload)
        hook = (HERE / "install_hook.cmake").read_text()
        self.assertNotIn('install(DIRECTORY "${CMAKE_BINARY_DIR}/lib/einsums/', hook)

    def test_exact_resource_shape(self):
        text = (HERE / "build.sbatch").read_text()
        for flag in ("--account=gts-cs207-chemx", "--qos=embers", "--cpus-per-task=8",
                     "--mem=48G", "--time=02:00:00", "--no-requeue"):
            self.assertIn("#SBATCH " + flag, text)
        self.assertNotIn("#SBATCH --gres", text)

    def test_executable_contract_is_tracked_despite_ignore_rules(self):
        result = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "devtools/disp0/optimized_build/build.sbatch",
             "devtools/disp0/optimized_build/payload.sh", "devtools/disp0/optimized_build/install_hook.cmake",
             "devtools/disp0/optimized_build/prepare_inputs.py",
             "devtools/disp0/optimized_build/interactive_acquire.py",
             "devtools/disp0/optimized_build/interactive_step.sh"],
            cwd=HERE.parents[2], capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
