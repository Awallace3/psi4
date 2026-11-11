import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


worker = load("timing_worker")
timing = load("phoenix_timing")
submit = load("submit_phoenix_timing")


class TimingContracts(unittest.TestCase):
    def records(self, gpu=False):
        values = {
            "Dimer SCF": 4., "Monomer A SCF": 3., "Monomer B SCF": 2.,
            "Build JK": 1., "elst": 1., "exch": 1., "ind": 5.,
            "delta HF assembly": 2.,
        }
        if gpu:
            values[" GPU F-SAPT"] = 12.
        else:
            values.update({"Localize Orbitals": 1., "Partition": 1.,
                           " F-SAPT Localization (IBO)": 1.,
                           " F-SAPT Electrostatics": 1., " F-SAPT Exchange": 1.,
                           " F-SAPT Induction": 1., " F-SAPT Dispersion": 1.})
        return {"SAPT(DFT) Energy;SAPT(DFT):" + key:
                {"wall_time": value, "n_calls": 1} for key, value in values.items()}

    def test_cpu_disjoint_ledger(self):
        rows = worker.ledger({}, self.records(), 30.)
        self.assertEqual(rows["canonical_ind"], 3.)
        self.assertEqual(rows["delta_hf_assembly"], 2.)
        self.assertEqual(rows["fragment_localization"], 3.)
        self.assertEqual(sum(value for key, value in rows.items() if key != "total"), 30.)
        self.assertNotIn("canonical_disp", rows)

    def test_gpu_overhead_not_lost_or_double_counted(self):
        trace = {"workflow": {"windows": {name: {"wall_seconds": 1.}
                 for name in ("localization", "elst", "exch", "ind", "disp")}}}
        rows = worker.ledger({}, self.records(True), 40., trace)
        self.assertEqual(rows["gpu_fragment_overhead"], 7.)
        self.assertEqual(sum(value for key, value in rows.items() if key != "total"), 40.)

    def test_missing_duplicated_calls_or_negative_residual_rejected(self):
        records = self.records()
        key = next(iter(records))
        for invalid in ({}, {**records, key: {"wall_time": 4., "n_calls": 2}},
                        {**records, "duplicate;" + key: records[key]}):
            with self.assertRaises(ValueError):
                worker.ledger({}, invalid, 40.)
        with self.assertRaises(ValueError):
            worker.ledger({}, records, 1.)

    def test_timer_snapshot_deltas(self):
        after = self.records()
        before = {key: {"wall_time": 10., "n_calls": 2} for key in after}
        accumulated = {key: {"wall_time": value["wall_time"] + 10., "n_calls": 3}
                       for key, value in after.items()}
        self.assertEqual(worker.ledger(before, accumulated, 30.),
                         worker.ledger({}, after, 30.))

    def test_three_samples_no_dropping_or_residual_ratios(self):
        cpu = [{"total": v, "other": 1.} for v in (4., 6., 8.)]
        gpu = [{"total": v, "other": 1., "gpu_fragment_overhead": 2.} for v in (1., 2., 3.)]
        self.assertEqual(timing.ratios(cpu, gpu)["total"]["speedup"], 3.)
        self.assertEqual(set(timing.ratios(cpu, gpu)), {"total"})
        for values in ([1., 2.], [1., 2., float("nan")], [1., 2., 0.]):
            with self.assertRaises(ValueError):
                timing.summary(values)

    def test_embers_bounded_afterany(self):
        argv = submit.command(Path("/source"), Path("/run"), "123")
        for flag in ("--qos=embers", "--account=gts-cs207-chemx",
                     "--dependency=afterany:123", "--kill-on-invalid-dep=yes",
                     "--cpus-per-task=8", "--constraint=H200", "--gres=gpu:h200:1",
                     "--time=04:00:00", "--signal=TERM@120"):
            self.assertIn(flag, argv)
        self.assertNotIn("afterok", " ".join(argv))
        with self.assertRaises(ValueError):
            submit.command(Path("/source"), Path("/run"), "123;456")

    def test_observer_wraps_original_bootstrap_and_charges_outer_audit(self):
        # Host-only tracing mechanics. No fake scientific result is used as GPU evidence.
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            package = root / "psi4"
            driver = package / "driver/driver.py"
            workflow = package / "driver/procrouting/sapt/sapt_fisapt_workflow_cuda.py"
            workflow.parent.mkdir(parents=True)
            workflow.write_text("# fake source for observer mechanics only\n")
            driver.write_text("def energy(name):\n    return (1., object())\n")
            output = root / "output"
            output.mkdir()
            bootstrap = root / "sealed_bootstrap.py"
            bootstrap.write_text(
                "from pathlib import Path\n"
                f"p = Path({str(driver)!r})\n"
                "scope = {'__file__': str(p)}\n"
                "exec(compile(p.read_text(), str(p), 'exec'), scope)\n"
                "scope['energy']('sapt(dft)')\n"
                f"Path({str(output / 'result.json')!r}).write_text("
                + repr(json.dumps({"status": "EXECUTED_REQUIRES_INDEPENDENT_COMPARISON",
                                   "arm": "cpu", "benchmark": True,
                                   "benchmark_whole_energy_seconds": 100.})) + ")\n")
            build = {
                "python_package": str(package),
                "provenance_sha256": {str(bootstrap): timing.digest(bootstrap)},
                "scientific_python_sha256": {
                    str(path.relative_to(package)): timing.digest(path)
                    for path in (driver, workflow)},
            }
            runtime = types.ModuleType("gpu_runtime")
            runtime.read_config = lambda *args: ({}, build)
            runtime.digest = timing.digest
            tracer = types.ModuleType("fragment_term_trace")
            tracer.TermTrace = lambda *args, **kwargs: object()
            core = types.ModuleType("psi4.core")
            snapshots = iter(({}, self.records()))
            core.get_timer_records = lambda flat: next(snapshots)
            receipt = root / "timers.json"
            argv = ["timing_worker.py", "--bootstrap", str(bootstrap),
                    "--receipt", str(receipt), "--config", str(root / "config.json"),
                    "--config-sha256", "fixture", "--arm", "cpu", "--output", str(output)]
            old_path = sys.path[:]
            try:
                with patch.object(sys, "argv", argv), patch.dict(sys.modules, {
                        "gpu_runtime": runtime, "fragment_term_trace": tracer, "psi4.core": core}):
                    worker.main()
            finally:
                sys.path[:] = old_path
            result = json.loads(receipt.read_text())
            self.assertEqual(result["status"], "PASS_OBSERVED_TIMERS")
            self.assertEqual(result["seconds"]["total"], 100.)
            self.assertLess(result["energy"]["wall_seconds"], 100.)
            self.assertIsNone(sys.gettrace())


if __name__ == "__main__":
    unittest.main()
