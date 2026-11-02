"""CPU-only submission/provenance checks; not numerical or GPU evidence."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fragment_gate import require_oracle_config
from gpu_runtime import digest
import submit_sealed_cpu_qualification as recipe


DIMENSIONS = {"atoms":24,"ao":356,"occupied":[20,20],"frozen":[5,5],"active":[15,15]}


def save(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(root):
    config = {"schema":1,"device_family":"H200","threads":8,"source_pin":"pin",
              "identity_contract":"process-owned-kernel-sealed-snapshots-v1",
              "sealed_alias_parent":"/diagnostics","dependency_environment":"/env"}
    save(root/"runtime-config.json",config)
    (root/"job-status.env").write_text("exit_code=0\nterm_received=0\n")
    gate = root/"small-gate/gate-result.json"
    save(gate,{"status":"PASS_SMALL_H200_ONLY","gpu_small_case_correctness_verified":True,
               "source_pin":"pin"})
    folder = root/"benchmark/execution"
    rows = []
    for index in range(3):
        path = folder/f"peptide-r{index}"
        sample = {"status":"EXECUTED_REQUIRES_INDEPENDENT_COMPARISON",
                  "arm":"gpu","system":"peptide","source_pin":"pin","benchmark":True,
                  "scaled":True,"options":{"use_cuest":True},
                  "orbital_dimensions":DIMENSIONS,"pid":100+index,
                  "runtime_config_sha256":digest(root/"runtime-config.json"),
                  "benchmark_whole_energy_seconds":30.+index}
        save(path/"result.json",sample)
        rows.append({"path":str(path),"result_sha256":digest(path/"result.json"),"seconds":30.+index})
    benchmark = {"source_pin":"pin","helper_pin":"pin","scratch_cleanup":{"complete":True},
                 "cases":{"peptide":{"status":"PASS_REPEATS_AWAITING_OTHER_RESOURCE_CLASS",
                                    "samples":rows,"timing":{"n":3,"median_seconds":31.,
                                        "minimum_seconds":30.,"maximum_seconds":32.}}}}
    save(folder/"benchmark-result.json",benchmark)
    manifest = {"science_pin":"pin","helper_pin":"pin","source":"/frozen-source",
                "config_sha256":digest(root/"runtime-config.json"),
                "inputs":"/inputs","inputs_sha256":"input"}
    save(root/"benchmark/manifest.json",manifest)
    pilot = {"status":"PASS_GPU_PILOT_AWAITING_INDEPENDENT_CPU_QUALIFICATION",
             "source_pin":"pin","scratch_cleanup":{"complete":True},
             "benchmark_result":str(folder/"benchmark-result.json"),
             "benchmark_sha256":digest(folder/"benchmark-result.json"),
             "small_gate_sha256":digest(gate),
             "config_sha256":digest(root/"runtime-config.json")}
    save(root/"pilot-result.json",pilot)
    return benchmark,pilot


class CPUQualification(unittest.TestCase):
    def test_recipe_executes_the_frozen_cpu_driver_not_its_own_checkout(self):
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            source,gpu_root = base/"frozen-source",base/"gpu-job"
            source.mkdir();gpu_root.mkdir()
            config = {"schema":1,"device_family":"H200","threads":8,
                      "identity_contract":"process-owned-kernel-sealed-snapshots-v1",
                      "dependency_environment":"/env","sealed_alias_parent":"/diagnostics"}
            manifest = {"source":str(source),"inputs":"/inputs","inputs_sha256":"inputs",
                        "cpu_water_result":"/cpu-water","cpu_water_sha256":"water"}
            seals = {"gpu_job_root":"/producer/execution","gpu_config":"/producer/config",
                     "gpu_config_sha256":"gpu","gpu_manifest_sha256":"manifest"}
            root = base/"fresh-cpu"
            args = SimpleNamespace(source=source,pin="execution-pin",gpu_job_root=gpu_root,root=root)
            scheduler = Mock(side_effect=[
                SimpleNamespace(returncode=0,stdout="test-only\n",stderr=""),
                SimpleNamespace(returncode=0,stdout="456\n",stderr="")])
            with patch.object(recipe,"source_pin",side_effect=["execution-pin","recipe-pin"]), \
                    patch.object(recipe,"producer",return_value=(config,manifest,seals)), \
                    patch.object(recipe.subprocess,"run",scheduler):
                recipe.submit(args)
            saved = json.loads((root/"manifest.json").read_text())
            cpu = json.loads((root/"runtime-config.json").read_text())
            self.assertEqual(cpu,{**config,"threads":24})
            self.assertEqual(saved["helper_pin"],"execution-pin")
            self.assertEqual(saved["submission_recipe_pin"],"recipe-pin")
            env = scheduler.call_args.kwargs["env"]
            self.assertEqual(env["DISP0_SOURCE"],str(source))
            self.assertEqual(env["DISP0_PIN"],"execution-pin")
            self.assertEqual(env["DISP0_SPLIT_KIND"],"cpu")
            self.assertEqual(env["DISP0_SPLIT_DRIVER"],"benchmark_suite.py")
            self.assertEqual(env["DISP0_SPLIT_THREADS"],"24")

    def test_explicit_embers_cpu_only_route_and_no_wait_or_gpu_reservation(self):
        argv = recipe.command(Path("/frozen-source"),Path("/fresh"))
        for value in ("--qos=embers","--account=gts-cs207-chemx",
                      "--partition=cpu-small","--cpus-per-task=24","--mem=160G",
                      "--time=02:00:00","--exclusive",
                      "--constraint=core24&gold6226&localNVMe","--no-requeue"):
            self.assertIn(value,argv)
        self.assertFalse(any(value.startswith(("--gres","--gpus","--dependency")) for value in argv))
        self.assertEqual(argv[-1],"/frozen-source/devtools/disp0/phoenix_validation/split_job.sbatch")

    def test_oracle_peer_configuration_changes_threads_only(self):
        gpu = {"schema":1,"device_family":"H200","threads":8,
               "identity_contract":"process-owned-kernel-sealed-snapshots-v1",
               "stage":"/whole-stage","sealed_alias_parent":"/existing-diagnostic-parent"}
        cpu = recipe.cpu_configuration(gpu)
        require_oracle_config(cpu,gpu)
        self.assertEqual(gpu["threads"],8)
        self.assertEqual(cpu["threads"],24)
        for wrong in ({**gpu,"threads":4},{**gpu,"device_family":"RTX3090"},
                      {**gpu,"identity_contract":"fresh-bytes-v1"}):
            with self.assertRaises(RuntimeError):
                recipe.cpu_configuration(wrong)

    def test_finalized_producer_is_frozen_by_small_document_hashes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture(root)
            with patch.object(recipe,"load_point",return_value=({},DIMENSIONS)):
                config,manifest,seals = recipe.producer(root,"pin")
            self.assertEqual(config["threads"],8)
            self.assertEqual(manifest["science_pin"],"pin")
            self.assertEqual(seals["gpu_manifest_sha256"],digest(root/"benchmark/manifest.json"))
            self.assertEqual(seals["gpu_job_root"],str(root/"benchmark/execution"))
            (root/"job-status.env").write_text("exit_code=1\nterm_received=0\n")
            with self.assertRaisesRegex(RuntimeError,"Finalized"):
                recipe.producer(root,"pin")

    def test_changed_payload_hash_and_fabricated_timer_summary_reject(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            benchmark,pilot = fixture(root)
            result = root/"benchmark/execution/benchmark-result.json"
            benchmark["cases"]["peptide"]["timing"]["median_seconds"] = 1.
            save(result,benchmark)
            with patch.object(recipe,"load_point",return_value=({},DIMENSIONS)):
                with self.assertRaisesRegex(RuntimeError,"changed final"):
                    recipe.producer(root,"pin")
                pilot["benchmark_sha256"] = digest(result)
                save(root/"pilot-result.json",pilot)
                with self.assertRaisesRegex(RuntimeError,"actual energy timers"):
                    recipe.producer(root,"pin")

    def test_duplicate_process_even_with_updated_receipts_cannot_be_a_repeat(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            benchmark,pilot = fixture(root)
            row = benchmark["cases"]["peptide"]["samples"][1]
            path = Path(row["path"])/"result.json"
            sample = json.loads(path.read_text());sample["pid"] = 100
            save(path,sample);row["result_sha256"] = digest(path)
            result = root/"benchmark/execution/benchmark-result.json"
            save(result,benchmark);pilot["benchmark_sha256"] = digest(result)
            save(root/"pilot-result.json",pilot)
            with patch.object(recipe,"load_point",return_value=({},DIMENSIONS)):
                with self.assertRaisesRegex(RuntimeError,"provenance/timing"):
                    recipe.producer(root,"pin")


if __name__ == "__main__":
    unittest.main()
