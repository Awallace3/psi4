import copy
import tempfile
import unittest
import subprocess
from types import SimpleNamespace
from unittest.mock import patch
import json
from pathlib import Path

from grids import export_grid
from tz_control import attempt, matching_audit, timing_gate, validate_source_record


class ControlTests(unittest.TestCase):
    def test_changed_config_cannot_reach_worker_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.json"
            config.write_text(json.dumps({"worker_timeout_s": 1800}))
            args = SimpleNamespace(psi_env=root, gpu_env=root, psi_package=root, output=root,
                _config_pins={config: "hash-from-before-edit"}, _geometry_pins={"peptide": "geometry"})
            with patch("tz_control.observed_worker") as worker, self.assertRaisesRegex(
                    RuntimeError, "Configuration changed"):
                attempt(args, root, config, "psi4", "peptide", "pbe", "timed", False, 0, {"uuid": "GPU"})
            worker.assert_not_called()

    def test_shell_does_not_launch_after_term(self):
        shell = Path(__file__).with_name("tz_job.sh").read_text()
        helper = shell[shell.index("run_child() {"):shell.index("# The compiled source")]
        result = subprocess.run(["bash", "-c", "terminated=1\n" + helper +
                                 '\nrun_child bash -c "echo BUG_STARTED"'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 143)
        self.assertNotIn("BUG_STARTED", result.stdout)

    def test_four_arm_gate_fails_closed(self):
        proof = dict(verified=True, identical_nonzero_quadrature=True)
        proofs = {basis: {engine: dict(proof) for engine in ("psi4", "pyscf")}
                  for basis in ("cc-pvtz", "aug-cc-pvtz")}
        self.assertTrue(timing_gate(proofs))
        for basis in proofs:
            for engine in proofs[basis]:
                changed = copy.deepcopy(proofs)
                changed[basis][engine]["identical_nonzero_quadrature"] = False
                self.assertFalse(timing_gate(changed))
                del changed[basis][engine]
                self.assertFalse(timing_gate(changed))
        self.assertFalse(timing_gate({}))

    def test_native_source_requires_success_and_actual_count(self):
        record = dict(ok=True, process_returncode=0, grid_points=2,
                      grid_exports={"live-plan-before-scf": dict(stored_points=2,
                          nonzero_weight_points=2, identity={"grid_mode": "native-default"})})
        self.assertEqual(validate_source_record(record)["stored_points"], 2)
        for field, value in (("ok", False), ("process_returncode", 1), ("grid_points", 3)):
            broken = dict(record, **{field: value})
            with self.assertRaises(RuntimeError):
                validate_source_record(broken)
        record["grid_exports"]["live-plan-before-scf"]["identity"]["grid_mode"] = "nominal"
        with self.assertRaises(RuntimeError):
            validate_source_record(record)

    def test_equal_counts_different_weights_refused(self):
        import numpy as np
        identity = dict(engine="psi4", system="protein83", functional="pbe",
                        geometry_sha256="geometry", basis="cc-pvtz", grid_mode="native-default",
                        config_sha256="config", build_sha256="binary")
        with tempfile.TemporaryDirectory() as directory:
            source, candidate = (Path(directory) / name for name in ("source.npz", "candidate.npz"))
            coords = np.array([[0., 0., 0.], [1., 2., 3.]])
            export_grid(source, coords, [1., 2.], identity, phase="live-plan-before-scf",
                        settings={"native": True})
            other = dict(identity, basis="aug-cc-pvtz", engine="pyscf", grid_mode="shared-cuest")
            export_grid(candidate, coords, [1., 3.], other, phase="xc-entry", settings={"fixed": True})
            with self.assertRaisesRegex(RuntimeError, "weights differ"):
                matching_audit(source, candidate)


if __name__ == "__main__":
    unittest.main()
