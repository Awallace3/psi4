"""Exercise the canonical controller's actual launch seam, without a GPU."""
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import sapt0_timing as timing
from sealed_h200 import CONTRACT
from test_sapt0_timing import TimingContracts


class CanonicalSealedLaunch(unittest.TestCase):
    def test_whole_gpu_energy_uses_owner_guard_without_fallback(self):
        psi4 = SimpleNamespace(energy=Mock(return_value="cpu"))
        runtime = SimpleNamespace(guarded_call=Mock(return_value="gpu"))
        self.assertEqual(timing.invoke_energy(psi4, "mol", runtime, {"factory": 1}), "gpu")
        runtime.guarded_call.assert_called_once_with(
            psi4.energy, "sapt(dft)", molecule="mol", return_wfn=True, factory=1)
        psi4.energy.assert_not_called()
        runtime.guarded_call.side_effect = RuntimeError("retired")
        with self.assertRaisesRegex(RuntimeError, "retired"):
            timing.invoke_energy(psi4, "mol", runtime, {})
        psi4.energy.assert_not_called()
        self.assertEqual(timing.invoke_energy(psi4, "mol", None, {}), "cpu")

    def test_canonical_run_uses_early_sealed_loader_for_both_arms(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            aliases = root / "aliases"
            aliases.mkdir(mode=0o700)
            config = {"identity_contract": CONTRACT, "sealed_alias_parent": str(aliases),
                      "threads": 8, "stage": "/whole-stage", "dependency_environment": "/env"}
            args = SimpleNamespace(config=root/"config", config_sha256="frozen",
                                   case=root/"peptide.json", output=root/"pair",
                                   repeats=1, timeout=60, diagnostics=False)
            commands = []
            fixtures = TimingContracts().records()

            def execute(command, **kwargs):
                commands.append(command)
                self.assertTrue(any(Path(arg).name == "sealed_bootstrap.py" for arg in command))
                self.assertIn("--sapt0", command)
                self.assertTrue(kwargs["pass_fds"])
                self.assertIn("DISP0_RETIREMENT_FD", kwargs["env"])
                arm = command[command.index("--arm") + 1]
                child = Path(command[command.index("--output") + 1])
                child.mkdir()
                record = next(row for row in fixtures if row["arm"] == arm)
                timing.save(child/"result.json", record)
                return SimpleNamespace(returncode=0)

            with patch("gpu_runtime.read_config", return_value=(config, {})), \
                    patch("run_gate.environment", return_value={}) as environment, \
                    patch("sapt0_timing.subprocess.run", side_effect=execute), \
                    patch.dict(os.environ, TMPDIR=str(root)):
                timing.run(args)
            self.assertEqual(len(commands), 2)
            self.assertEqual(len({call.kwargs["sealed_alias_directory"]
                                  for call in environment.call_args_list}), 2)
            self.assertEqual(json.loads((args.output/"comparison.json").read_text())["status"],
                             "PASS_MATCHED_SAPT0")


if __name__ == "__main__":
    unittest.main()
