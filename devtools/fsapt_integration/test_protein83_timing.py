import importlib.util
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


load("phoenix_timing")
timing = load("protein83_timing")
submit = load("submit_protein83")


class Protein83Contracts(unittest.TestCase):
    def test_two_samples_never_three_or_dropped(self):
        self.assertEqual(timing.two_sample_summary([10., 20.])["median_seconds"], 15.)
        for values in ([10.], [1., 2., 3.], [0., 2.], [1., float("nan")]):
            with self.assertRaises(ValueError):
                timing.two_sample_summary(values)

    def test_cpu_roles_never_reserve_gpu(self):
        for role in ("cpu0", "cpu1", "compare"):
            argv = submit.command(Path("/source"), Path("/run"), role, ["123"])
            for flag in ("--cpus-per-task=48", "--mem=360G", "--partition=cpu-gnr",
                         "--constraint=graniterapids", "--time=08:00:00", "--qos=embers"):
                self.assertIn(flag, argv)
            self.assertFalse(any(flag.startswith("--gres") for flag in argv))

    def test_gpu_and_artifact_dependencies(self):
        argv = submit.command(Path("/source"), Path("/run"), "gpu", ["123"])
        for flag in ("--cpus-per-task=8", "--mem=160G", "--partition=gpu-h200",
                     "--constraint=H200", "--gres=gpu:h200:1", "--time=04:00:00"):
            self.assertIn(flag, argv)
        compare = submit.command(Path("/source"), Path("/run"), "compare", ["1", "2", "3"])
        self.assertIn("--dependency=afterany:1:2:3", compare)
        self.assertIn("--kill-on-invalid-dep=yes", compare)
        self.assertNotIn("afterok", " ".join(compare))
        with self.assertRaises(ValueError):
            submit.command(Path("/source"), Path("/run"), "gpu", ["bad"])

    def test_speedup_uses_sealed_worker_seconds_and_exact_inputs(self):
        doc = dict(status="EXECUTED_REQUIRES_INDEPENDENT_COMPARISON",
                   source_pin="pin", job_id="123", hostname="node", arm="cpu",
                   runtime_config_sha256="config", input_manifest_sha256="inputs",
                   system="protein83", scaled=True, basis_profile="jun-cc-pvdz",
                   factor_cache_bytes=0, dispersion_block_size=24, orbital_dimensions={"ao":922},
                   benchmark=True, benchmark_whole_energy_seconds=100.,
                   options={"basis":"jun-cc-pvdz","e_convergence":10,"d_convergence":12,
                            "puream":True,"use_cuest":False})
        kwargs = dict(pin="pin", job="123", node="node", arm="cpu", config_sha="config",
                      input_sha="inputs", dimensions={"ao":922})
        self.assertEqual(timing.verified_sample_seconds(doc, {"seconds":100.}, **kwargs), 100.)
        with self.assertRaises(RuntimeError):
            timing.verified_sample_seconds(doc, {"seconds":1.}, **kwargs)
        for key, value in (("input_manifest_sha256","different"), ("job_id","124"),
                           ("runtime_config_sha256","other"), ("basis_profile","def2-svp")):
            with self.assertRaises(RuntimeError):
                timing.verified_sample_seconds({**doc,key:value}, {"seconds":100.}, **kwargs)
        with self.assertRaises(RuntimeError):
            timing.verified_sample_seconds(
                {**doc,"options":{**doc["options"],"d_convergence":10}}, {"seconds":100.}, **kwargs)


if __name__ == "__main__":
    unittest.main()
