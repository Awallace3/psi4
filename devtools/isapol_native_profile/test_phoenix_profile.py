"""Pure wrapper tests: fake srun and stdlib fake driver; NEVER imports Psi4."""
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import tempfile
import time
import unittest


WRAPPER = Path(__file__).resolve().with_name("phoenix_profile.sbatch")
SHA = "31c59282cafe0405468ea320e58039cb633acd873264d16ae38dfeac9fd60781"

FAKE_SRUN = r'''#!/usr/bin/python3
import json, os, signal, subprocess, sys
from pathlib import Path
state = Path(os.environ["WRAPPER_TEST_STATE"])
with (state / "srun-calls.jsonl").open("a") as f:
    f.write(json.dumps(sys.argv[1:]) + "\n")
args = sys.argv[1:]
child = subprocess.Popen(args[args.index("--")+1:])
def term(*unused):
    try: child.send_signal(signal.SIGTERM)
    except ProcessLookupError: pass
signal.signal(signal.SIGTERM, term)
signal.signal(signal.SIGINT, term)
sys.exit(child.wait())
'''

FAKE_DRIVER = r'''
import argparse, json, os, signal, sys, time
from pathlib import Path
p = argparse.ArgumentParser()
for name in ("fixtures", "build-manifest", "qualification", "qcengine-package",
             "qcengine-harness-sha256", "threads", "instrumentation", "sample-seconds",
             "scratch", "output"):
    p.add_argument("--"+name, required=True)
p.add_argument("--check-only", action="store_true")
p.add_argument("--run", action="store_true")
a = p.parse_args()
state = Path(os.environ["WRAPPER_TEST_STATE"])
behavior = os.environ.get("WRAPPER_TEST_BEHAVIOR", "")
out, scratch = Path(a.output), Path(a.scratch)
with (state / "driver-calls.jsonl").open("a") as f:
    f.write(json.dumps(dict(label=out.name, threads=int(a.threads),
                           mode=a.instrumentation, cwd=os.getcwd(),
                           omp=os.environ.get("OMP_NUM_THREADS"),
                           places=os.environ.get("OMP_PLACES"))) + "\n")
# A relative startup/finalizer write must not reach the caller's protected CWD.
Path("launch-relative.txt").write_text("fake startup only")
out.mkdir()
scratch.mkdir()
(scratch / "cwd").mkdir()
os.chdir(scratch / "cwd")
def write(name, data):
    (out / name).write_text(json.dumps(data))
write("manifest.json", dict(threads=int(a.threads), science_memory_gib=20,
      case_metadata={"monomer_id": "monomer-0001"},
      declarations=dict(partition_grid=[400,2030], response_grid=[100,200],
                        drho_refinement_iterations=32, response="reference_h2h1"),
      protocols_unchanged=True, instrumentation=a.instrumentation,
      status="ADMITTED_NOT_EXECUTED"))
write("wall-time-source.json", {"recognized_qcengine_wrapper_timer": True})
if a.check_only: sys.exit(0)
if out.name == "stages1" and behavior in ("term", "deadline", "cap"):
    (state / "payload-pid").write_text(str(os.getpid()))
    def term(*unused):
        (state / "TERM_RECEIVED").write_text("yes")
        sys.exit(143)
    signal.signal(signal.SIGTERM, term)
    if behavior == "cap":
        with (scratch / "sparse-byte-breach").open("wb") as f:
            f.truncate(256*1024**3 + 1)  # sparse; no large data/memory allocated
    (state / "READY").write_text("yes")
    while True: time.sleep(.1)
if out.name == "stages1" and behavior == "fail":
    sys.exit(7)
write("summary.json", dict(status="COMPUTED_NOT_ACCEPTED", acceptance=False,
      installed_science_unchanged=True, telemetry={"errors": []}))
for name in ("server-native-envelope.json", "server-tensors.npz", "native-result.json",
             "qcschema-result.json", "events.jsonl", "samples.jsonl", "hooks.json"):
    (out / name).write_text("synthetic test artifact, NOT scientific data")
if behavior == "missing" and out.name == "stages1":
    (out / "summary.json").unlink()
if behavior == "source-drift" and out.name == "off8":
    Path(os.environ["WRAPPER_TEST_SOURCE"]).joinpath("dirty-evidence.txt").write_text("test drift")
'''


class WrapperTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.repo = self.root / "source"
        self.source_dir = self.repo / "devtools/isapol_native_profile"
        self.source_dir.mkdir(parents=True)
        for name in (".gitignore", "README.md", "fixtures.json", "telemetry.py", "test_profile.py"):
            (self.source_dir / name).write_text("pure wrapper fixture\n")
        (self.source_dir / "profile.py").write_text(FAKE_DRIVER)
        (self.source_dir / WRAPPER.name).write_bytes(WRAPPER.read_bytes())
        self.env = dict(os.environ, GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL="/dev/null",
                        WRAPPER_TEST_STATE=str(self.root), WRAPPER_TEST_SOURCE=str(self.repo),
                        SLURM_JOB_ID="pure-test-not-a-real-job", SLURM_CPUS_PER_TASK="8")
        self.git("init", "-q")
        self.git("add", "-f", "devtools")
        self.git("-c", "user.name=Pure Wrapper Test", "-c", "user.email=pure@example.invalid",
                 "-c", "commit.gpgsign=false", "commit", "-qm", "fake fixture")
        self.commit = self.git("rev-parse", "HEAD").stdout.strip()
        self.run_root, self.scratch = self.root / "run", self.root / "scratch"
        self.run_root.mkdir()
        self.scratch.mkdir()
        self.protected = self.root / "protected-cwd"
        self.protected.mkdir()
        (self.protected / "sentinel").write_text("unchanged")
        self.fake_srun = self.root / "fake-srun"
        self.fake_srun.write_text(FAKE_SRUN)
        self.fake_srun.chmod(0o700)
        self.process = None

    def git(self, *args):
        return subprocess.run(["git", "-C", str(self.repo), *args], env=self.env,
                              text=True, capture_output=True, check=True)

    def tearDown(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        # Only a fake payload created by this test, never arbitrary processes.
        pid_path = self.root / "payload-pid"
        if pid_path.exists():
            pid = int(pid_path.read_text())
            try:
                cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
                if str(self.source_dir / "profile.py").encode() in cmdline:
                    os.killpg(os.getpgid(pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            except FileNotFoundError:
                pass
        self.temp.cleanup()

    def launch(self, behavior="", seconds=30):
        self.env["WRAPPER_TEST_BEHAVIOR"] = behavior
        # Source functions only: normal main's fixed science/tool paths are never overridden.
        assignments = dict(source=str(self.repo), commit=self.commit, run_root=str(self.run_root),
                           scratch_root=str(self.scratch), fixtures=str(self.root / "monomer-0001"),
                           harness_sha=SHA, srun=str(self.fake_srun), science_python=sys.executable,
                           build=str(self.root / "build"), qualification=str(self.root / "qualification"),
                           base=str(self.root / "base"), qce=str(self.root / "qce"))
        setup = "\n".join("local " + key + "=" + shlex.quote(value) for key, value in assignments.items())
        command = (f'source {shlex.quote(str(WRAPPER))}\n'
                   f'pure_test_main() {{\n{setup}\nlocal deadline=$((SECONDS+{seconds}))\n'
                   'execute_plan\nexit 0\n}\npure_test_main\n')
        self.process = subprocess.Popen(["bash", "-c", command], cwd=self.protected, env=self.env,
                                        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        start_new_session=True)
        return self.process

    def finish(self, process):
        stdout, stderr = process.communicate(timeout=20)
        self.assertEqual((self.protected / "sentinel").read_text(), "unchanged")
        self.assertEqual(sorted(p.name for p in self.protected.iterdir()), ["sentinel"])
        return process.returncode, stdout, stderr

    def calls(self):
        return [json.loads(line) for line in (self.root / "driver-calls.jsonl").read_text().splitlines()]

    def test_syntax_fixed_allocation_and_no_new_omp_affinity(self):
        subprocess.run(["bash", "-n", str(WRAPPER)], check=True)
        text = WRAPPER.read_text()
        for directive in ("--account=gts-cs207-chemx", "--partition=cpu-small", "--qos=embers",
                          "--nodes=1", "--ntasks=1", "--cpus-per-task=8", "--mem=32G",
                          "--time=02:00:00", "--no-requeue", "--signal=TERM@120"):
            self.assertIn("#SBATCH " + directive, text)
        self.assertNotIn("OMP_PROC_BIND=", text)
        self.assertNotIn("OMP_PLACES=", text)
        self.assertNotIn("131072", text)
        self.assertIn(SHA, text)

    def test_success_exact_sequence_fixed_eight_cpu_steps_and_owned_cwds(self):
        code, _, stderr = self.finish(self.launch())
        self.assertEqual(code, 0, stderr)
        self.assertTrue((self.run_root / "COMPLETED").exists())
        self.assertFalse((self.run_root / "FAILED").exists())
        calls = self.calls()
        self.assertEqual([(r["label"], r["threads"], r["mode"]) for r in calls],
                         [("check8", 8, "stages"), ("stages1", 1, "stages"),
                          ("stages2", 2, "stages"), ("stages4", 4, "stages"),
                          ("stages8", 8, "stages"), ("off8", 8, "off")])
        for row in calls:
            self.assertEqual(row["cwd"], str(self.scratch / ("launch-" + row["label"])))
            self.assertEqual(row["omp"], str(row["threads"]))
        for line in (self.root / "srun-calls.jsonl").read_text().splitlines():
            args = json.loads(line)
            self.assertIn("--cpus-per-task=8", args)
            self.assertIn("--cpu-bind=cores", args)
        self.assertEqual(self.git("status", "--porcelain").stdout, "")

    def test_child_failure_aborts_matrix_and_never_completes(self):
        code, _, _ = self.finish(self.launch("fail"))
        self.assertEqual(code, 7)
        self.assertEqual([r["label"] for r in self.calls()], ["check8", "stages1"])
        self.assertFalse((self.run_root / "COMPLETED").exists())
        self.assertTrue((self.run_root / "FAILED").exists())

    def test_zero_exit_with_missing_exports_is_failure(self):
        code, _, _ = self.finish(self.launch("missing"))
        self.assertNotEqual(code, 0)
        self.assertFalse((self.run_root / "COMPLETED").exists())

    def test_end_source_drift_is_failure_even_after_all_zero_exits(self):
        code, _, stderr = self.finish(self.launch("source-drift"))
        self.assertEqual(code, 43, stderr)
        self.assertFalse((self.run_root / "COMPLETED").exists())

    def test_term_forwarded_to_payload_group_no_false_completion(self):
        process = self.launch("term")
        limit = time.monotonic() + 10
        while not (self.root / "READY").exists() and time.monotonic() < limit:
            if process.poll() is not None:
                break
            time.sleep(.02)
        self.assertTrue((self.root / "READY").exists())
        process.send_signal(signal.SIGTERM)
        code, _, stderr = self.finish(process)
        self.assertEqual(code, 143, stderr)
        self.assertTrue((self.root / "TERM_RECEIVED").exists())
        self.assertFalse((self.run_root / "COMPLETED").exists())

    def test_scratch_byte_breach_preserves_sparse_evidence_and_fails(self):
        code, _, stderr = self.finish(self.launch("cap"))
        self.assertEqual(code, 42, stderr)
        report = json.loads((self.run_root / "stages1-scratch-monitor.json").read_text())
        self.assertEqual(report["failure"], "scratch_byte_cap")
        self.assertGreater(report["sampled_maxima"]["logical_bytes"], 256*1024**3)
        self.assertTrue((self.scratch / "stages1/sparse-byte-breach").exists())
        self.assertFalse((self.run_root / "COMPLETED").exists())

    def test_finite_deadline_fails_and_terminates_child(self):
        code, _, stderr = self.finish(self.launch("deadline", seconds=3))
        self.assertEqual(code, 42, stderr)
        reasons = [json.loads(p.read_text())["failure"]
                   for p in self.run_root.glob("*-scratch-monitor.json")]
        self.assertIn("workflow_deadline", reasons)
        self.assertFalse((self.run_root / "COMPLETED").exists())

    def test_allocation_validator_rejects_paid_qos_requeue_and_resource_drift(self):
        expected = dict(Account="gts-cs207-chemx", QOS="embers", Partition="cpu-small",
                        NumNodes="1", NumTasks="1", NumCPUs="8", **{"CPUs/Task": "8"},
                        MinMemoryNode="32G", TimeLimit="02:00:00", Requeue="0", Restarts="0")
        path = self.root / "allocation"
        for change in ({}, {"QOS": "inferno"}, {"Requeue": "1"}, {"NumCPUs": "4"}):
            with self.subTest(change=change):
                row = dict(expected, **change)
                path.write_text(" ".join(f"{k}={v}" for k, v in row.items()))
                result = subprocess.run(["bash", "-c",
                                         'source "$1"; check_allocation "$2"', "bash",
                                         str(WRAPPER), str(path)], capture_output=True)
                self.assertEqual(result.returncode == 0, not bool(change))


if __name__ == "__main__":
    unittest.main(verbosity=2)
