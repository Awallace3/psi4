"""Separated scheduling contracts: no GPU lease for a CPU reference/oracle."""
import json
import os
import signal
import subprocess
import sys
import threading
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fragment_gate import require_oracle_config
from gpu_runtime import digest
from split_jobs import validate_allocation,worker_command,sealed_gpu,admit_cpu_producer
from submit_split import command,prepare,submit_one
from split_scratch import OwnedScratch
from run_full import execute_owned


class SplitContracts(unittest.TestCase):
    def test_peer_config_failure_precedes_any_reference_or_producer_consumption(self):
        manifest={"gpu_config":"/changed-config","gpu_config_sha256":"old",
                  "gpu_job_root":"/producer","repeat":0}
        cpu={"threads":24,"source_pin":"science"}
        with patch("split_jobs.read_config",return_value=(
                {"threads":8,"source_pin":"different"},{})),patch(
                "split_jobs.sealed_gpu") as consume,patch("split_jobs.worker_command") as launch:
            with self.assertRaises(RuntimeError):admit_cpu_producer(manifest,cpu)
            consume.assert_not_called();launch.assert_not_called()

    def test_scratch_cleanup_follows_reaping_on_timeout_and_TERM(self):
        for terminate in (False,True):
            with self.subTest(terminate=terminate),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)/"work";root.mkdir(mode=0o700)
                output=Path(tmp)/"outputs";output.mkdir()
                keep=output/"result.json";keep.write_text("preserve")
                owner=OwnedScratch(root,str(root),output)
                groups=set()
                timer=threading.Timer(.2,lambda:os.kill(os.getpid(),signal.SIGTERM))
                def started(pid):
                    groups.add(pid)
                    if terminate:timer.start()
                try:
                    with tempfile.TemporaryFile("w+") as out,tempfile.TemporaryFile("w+") as err:
                        with self.assertRaises(InterruptedError if terminate else subprocess.TimeoutExpired):
                            execute_owned([sys.executable,"-c","import time; time.sleep(30)"],
                                dict(os.environ),out,err,30 if terminate else .1,started,
                                reaped=groups.discard)
                finally:
                    timer.cancel()
                    if terminate:timer.join()
                self.assertFalse(groups)
                self.assertTrue(owner.cleanup(groups)["complete"])
                self.assertFalse(root.exists());self.assertEqual(keep.read_text(),"preserve")

    def test_scratch_cannot_be_deleted_with_live_children_or_replaced_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"scratch";path.mkdir(mode=0o700)
            owner=OwnedScratch(path,str(path),Path(tmp)/"out")
            with self.assertRaises(RuntimeError):owner.cleanup({123})
            path.rename(Path(tmp)/"original")
            path.mkdir(mode=0o700)
            with self.assertRaises(RuntimeError):owner.cleanup(set())
            self.assertTrue(path.exists())

    def test_GPU_command_is_independent_and_CPU_command_reserves_no_GPU(self):
        gpu=command(Path("/source"),Path("/run"),"gpu","gts-cs207-chemx","inferno")
        self.assertIn("--gres=gpu:h200:1",gpu)
        self.assertIn("--cpus-per-task=8",gpu)
        self.assertFalse(any(a.startswith("--dependency") for a in gpu))
        with self.assertRaises(ValueError):
            command(Path("/source"),Path("/run"),"gpu","gts-cs207-chemx","inferno","123")
        cpu=command(Path("/source"),Path("/run"),"cpu","gts-cs207-chemx","embers","123")
        self.assertIn("--partition=cpu-small",cpu)
        self.assertIn("--cpus-per-task=24",cpu)
        self.assertIn("--dependency=afterany:123",cpu)
        self.assertIn("--kill-on-invalid-dep=yes",cpu)
        self.assertFalse(any(a.startswith("--gres") or "afterok" in a for a in cpu))
        late=command(Path("/source"),Path("/run"),"cpu","gts-cs207-chemx","embers",
                     completed_producer=True)
        self.assertFalse(any(a.startswith(("--gres","--dependency")) for a in late))
        with self.assertRaises(ValueError):
            command(Path("/source"),Path("/run"),"cpu","gts-cs207-chemx","embers")

    def test_GPU_worker_rejects_both_a_CPU_reference_and_oracle(self):
        config={"stage":"/stage","dependency_environment":"/env"}
        manifest={"kind":"gpu","config":"/cfg","config_sha256":"hash","inputs":"/input",
                  "inputs_sha256":"ih","system":"nanotube"}
        args=worker_command(Path("/helpers"),config,manifest,Path("/out"),"gpu")
        self.assertEqual(args[args.index("--arm")+1],"gpu")
        with self.assertRaises(RuntimeError):
            worker_command(Path("/helpers"),config,manifest,Path("/out"),"cpu")
        with self.assertRaises(RuntimeError):
            worker_command(Path("/helpers"),config,manifest,Path("/out"),"gpu",Path("/oracle"))
        with self.assertRaises(RuntimeError):
            worker_command(Path("/helpers"),config,{**manifest,"kind":"cpu"},Path("/out"),"gpu")

    def test_CPU_allocation_cannot_hide_an_actual_GPU_reservation(self):
        job={"Account":"gts-cs207-chemx","QOS":"embers","Partition":"cpu-small",
             "NumNodes":"1","NumCPUs":"24","CPUs/Task":"24","NumTasks":"1",
             "TimeLimit":"08:00:00","MinMemoryNode":"160G","AllocTRES":"cpu=24,mem=160G"}
        manifest={"kind":"cpu","account":"gts-cs207-chemx","qos":"embers"}
        with patch.dict(os.environ,{"SLURM_JOB_GPUS":""}):
            validate_allocation(job,manifest)
            for tres in ("gres/gpu=1","gres/gpu:h200=1"):
                with self.assertRaises(RuntimeError):
                    validate_allocation({**job,"AllocTRES":job["AllocTRES"]+","+tres},manifest)
        with patch.dict(os.environ,{"SLURM_JOB_GPUS":"0"}),self.assertRaises(RuntimeError):
            validate_allocation(job,manifest)

    def test_cross_role_oracle_only_changes_declared_thread_count(self):
        gpu={"source_pin":"science","threads":8,"host_memory":"96 GiB",
             "native_persistent_bytes":12,"optional_files":{"binding":"hash"}}
        cpu={**gpu,"threads":24}
        require_oracle_config(cpu,gpu)
        for change in ({"source_pin":"other"},{"host_memory":"128 GiB"},
                       {"native_persistent_bytes":13},{"threads":16},
                       {"optional_files":{"binding":"different"}}):
            with self.assertRaises(RuntimeError):require_oracle_config({**cpu,**change},gpu)

    def test_preparation_creates_only_two_GPU_and_six_CPU_plans_without_sbatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/"source";source.mkdir()
            original=Path(tmp)/"config.json";original.write_text("{}")
            inputs=Path(tmp)/"inputs.json";inputs.write_text("{}")
            release=Path(tmp)/"release.json";release.write_text("{}")
            args=SimpleNamespace(source=source,config=original,config_sha256="c",
                inputs=inputs,inputs_sha256="i",release=release,release_sha256="r",
                root=Path(tmp)/"plan")
            with patch("submit_split.source_pin",return_value="helper"),patch(
                    "submit_split.read_config",return_value=({"source_pin":"science"},{})),patch(
                    "submit_split.require_release"),patch("submit_split.load_point"),patch(
                    "submit_split.subprocess.run") as scheduler:
                prepare(args)
                scheduler.assert_not_called()
            plan=json.loads((args.root/"plan.json").read_text())
            self.assertEqual(len(plan["jobs"]),8)
            self.assertEqual([v["kind"] for v in plan["jobs"].values()].count("gpu"),2)
            for label,item in plan["jobs"].items():
                manifest=json.loads((Path(item["root"])/"draft-manifest.json").read_text())
                self.assertEqual(manifest["cpu_threads"],8 if item["kind"]=="gpu" else 24)
                self.assertFalse((Path(item["root"])/"job-id.txt").exists())
                self.assertNotIn("peptide",label)

    def test_explicit_completed_producer_submits_no_dependency_or_new_GPU_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/"source";source.mkdir()
            files=[]
            for name in ("config","inputs","release"):
                path=Path(tmp)/(name+".json");path.write_text("{}");files.append(path)
            args=SimpleNamespace(source=source,config=files[0],config_sha256="c",
                inputs=files[1],inputs_sha256="i",release=files[2],release_sha256="r",
                root=Path(tmp)/"plan",submit_one="nanotube-cpu-r0",
                completed_producer=True,account="gts-cs207-chemx",qos="embers")
            with patch("submit_split.source_pin",return_value="helper"),patch(
                    "submit_split.read_config",return_value=(
                        {"source_pin":"science","dependency_environment":"/env"},{})),patch(
                    "submit_split.require_release"),patch("submit_split.load_point"):
                prepare(args)
                (args.root/"nanotube-gpu/job-id.txt").write_text("123\n")
                responses=[SimpleNamespace(returncode=0,stdout="test-only",stderr=""),
                           SimpleNamespace(returncode=0,stdout="456\n",stderr="")]
                with patch("submit_split.admit_cpu_producer") as admit,patch(
                        "submit_split.subprocess.run",side_effect=responses) as scheduler:
                    submit_one(args)
                    admit.assert_called_once()
                    self.assertEqual(scheduler.call_count,2)
                    for call in scheduler.call_args_list:
                        argv=call.args[0]
                        self.assertEqual(argv[0],"sbatch")
                        self.assertIn("--partition=cpu-small",argv)
                        self.assertFalse(any(v.startswith(("--gres","--dependency")) for v in argv))
                self.assertEqual((args.root/"nanotube-gpu/job-id.txt").read_text(),"123\n")
                self.assertEqual((args.root/"nanotube-cpu-r0/job-id.txt").read_text(),"456\n")

    def test_CPU_consumes_only_clean_complete_hashed_GPU_producer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rows=[]
            for repeat in range(3):
                p=root/f"r{repeat}-gpu";p.mkdir()
                for name in ("raw-hf.npz","fragment-matrices.npz"):(p/name).write_bytes(b"raw")
                doc={"status":"EXECUTED_REQUIRES_INDEPENDENT_COMPARISON","system":"nanotube",
                     "arm":"gpu","source_pin":"science","runtime_config_sha256":"config",
                     "artifact_sha256":{n:digest(p/n) for n in ("raw-hf.npz","fragment-matrices.npz")}}
                (p/"result.json").write_text(json.dumps(doc))
                rows.append({"repeat":repeat,"result_sha256":digest(p/"result.json")})
            proof={"status":"PASS_GPU_EXECUTION_REPEATS_AWAITING_CPU","system":"nanotube",
                   "source_pin":"science","input_manifest_sha256":"inputs",
                   "runtime_config_sha256":"config","helper_pin":"helper","gpu_outputs":rows}
            (root/"split-result.json").write_text(json.dumps(proof))
            (root/"job-status.env").write_text("exit_code=0\nterm_received=0\n")
            manifest={"system":"nanotube","inputs_sha256":"inputs",
                      "helper_pin":"helper","science_pin":"science"}
            sealed_gpu(root,manifest,"config",1)
            (root/"job-status.env").write_text("exit_code=143\nterm_received=1\n")
            with self.assertRaises(RuntimeError):sealed_gpu(root,manifest,"config",1)
            (root/"job-status.env").write_text("exit_code=0\nterm_received=0\n")
            (root/"r1-gpu/raw-hf.npz").write_bytes(b"changed")
            with self.assertRaises(RuntimeError):sealed_gpu(root,manifest,"config",1)


if __name__=="__main__":unittest.main()
