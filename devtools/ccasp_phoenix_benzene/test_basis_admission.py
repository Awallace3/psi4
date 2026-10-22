# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""ENGINE-FREE/MOCK-ONLY tests. No Psi4/NumPy import, native basis or benzene run."""
import ast
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from typing import List, Optional
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TOOLS = load("mock_only_basis_tools", "basis_admission.py")
PILOT = load("mock_only_corrected_pilot", "pilot.py")
RAW = {name: TOOLS.raw_gbs_elements((ROOT / "psi4/share/psi4/basis" / name).read_text())
       for name in TOOLS.contract()["basis_files"]}


class FakeMolecule:
    # Fictitious collinear centres, NOT the private benzene geometry.
    def natom(self): return 12
    def x(self, i): return float(i)
    def y(self, i): return 0.
    def z(self, i): return 0.


class FakeShell:
    def __init__(self, raw, pure, shift=0.):
        self.am, self.nprimitive, self.pure = raw["l"], raw["nprimitive"], pure
        self.exponents = [float(r[0].replace("D", "E")) for r in raw["raw_exponent_coefficient_tokens"]]
        # Arbitrary mock values; NOT a GBS-to-native normalization formula.
        self.coefficients = [.125 + shift] * self.nprimitive
    def is_pure(self): return self.pure
    def exp(self, k): return self.exponents[k]
    def coef(self, k): return self.coefficients[k]


class FakeBasis:
    def __init__(self, role, pure, molecule, shift=0.):
        name = "aug-cc-pvtz.gbs" if role == "main" else "aug-cc-pvtz-ri.gbs"
        self.role, self.pure, self.mol = role, pure, molecule
        self.shells = [(i, FakeShell(s, pure, shift))
                       for i in range(12) for s in RAW[name]["C" if i < 6 else "H"]]
    def molecule(self): return self.mol
    def has_puream(self): return self.pure
    def nshell(self): return len(self.shells)
    def max_am(self): return max(s.am for _, s in self.shells)
    def shell_to_center(self, i): return self.shells[i][0]
    def shell(self, i): return self.shells[i][1]
    def nao(self): return sum((s.am + 1) * (s.am + 2) // 2 for _, s in self.shells)
    def nbf(self): return sum(2 * s.am + 1 for _, s in self.shells) if self.pure else self.nao()


class FakeCore:
    def __init__(self, value=True, changed=True):
        self.value, self.changed, self.aux_shift = value, changed, 0.
        self.built = []
        self.BasisSet = SimpleNamespace(build=self.build)
    def get_global_option(self, key): return self.value
    def has_global_option_changed(self, key): return self.changed
    def set_global_option(self, key, value): self.value, self.changed = bool(value), True
    def revoke_global_option_changed(self, key): self.changed = False
    def build(self, molecule, key, name, puream):
        pure = self.value if self.changed else bool(puream)  # documented precedence
        role = "auxiliary" if name.lower().endswith("-ri") else "main"
        basis = FakeBasis(role, pure, molecule, self.aux_shift if role == "auxiliary" else 0.)
        self.built.append(basis)
        return basis


def options_state(core):
    # Execute ONLY the frozen OptionState/OptionsState classes on FakeCore.
    # This tests real restore logic without importing the scientific package.
    tree = ast.parse((ROOT / "psi4/driver/p4util/optproc.py").read_text())
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name in ("OptionState", "OptionsState")]
    scope = dict(core=core, List=List, Optional=Optional, ValidationError=ValueError)
    exec(compile(ast.Module(body=classes, type_ignores=[]), "frozen-options-classes", "exec"), scope)
    return scope["OptionsState"]


class BasisAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.stage = Path(self.temp.name) / "stage"
        for name in TOOLS.contract()["basis_files"]:
            target = self.stage / "share/psi4/basis" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT / "psi4/share/psi4/basis" / name).read_bytes())
        self.core, self.molecule = FakeCore(), FakeMolecule()
        self.options = options_state(self.core)
        self.binding = {"native": {"commit": "MOCK"}, "driver": {"commit": "MOCK"}}

    def manifest(self):
        receipt = {}
        TOOLS.basis_files(ROOT, self.stage, receipt)
        TOOLS.build_bases(self.core, self.options, self.molecule, receipt)
        return dict(schema="ccasp-benzene-native-basis-admission/1",
                    status="basis_metadata_only_complete", no_scf=True,
                    binding=copy.deepcopy(self.binding), admission=copy.deepcopy(receipt))

    def driver_fragment(self, manifest):
        """Execute the ACTUAL pre-SCF driver statements on mocks, including energy seam."""
        tree = ast.parse((HERE / "pilot.py").read_text())
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
        body = next(n for n in main.body if isinstance(n, ast.Try)).body
        start = next(i for i, n in enumerate(body) if isinstance(n, ast.Expr)
                     and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute)
                     and n.value.func.attr == "set_options")
        end = next(i for i, n in enumerate(body) if isinstance(n, ast.Assign)
                   and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute)
                   and n.value.func.attr == "energy")
        events = []
        energy = mock.Mock(side_effect=lambda *a, **k: (events.append("energy") or (0., object())))
        psi4 = SimpleNamespace(
            set_options=lambda cfg: self.core.set_global_option("PUREAM", cfg["puream"]),
            energy=energy)
        scope = dict(psi4=psi4, cfg=json.loads((HERE / "pilot.json").read_text()),
            core=self.core, OptionsState=self.options, molecule=self.molecule,
            args=SimpleNamespace(), cfg_path=HERE / "pilot.json", source=ROOT, stage=self.stage,
            admission={}, manifest=manifest, basis_binding=lambda *_: self.binding,
            tools=TOOLS, recipe=mock.Mock(return_value="MOCK retained recipe"),
            write_json=lambda *a: events.append("basis-receipt"), out=Path(self.temp.name))
        self.energy, self.events, self.scope = energy, events, scope
        exec(compile(ast.Module(body=body[start:end + 1], type_ignores=[]),
                     "actual-pilot-admission-fragment", "exec"), scope)
        return scope

    def test_canonical_static_counts_1146_912_252_and_main_414(self):
        aux = TOOLS.static_counts(RAW["aug-cc-pvtz-ri.gbs"])
        main = TOOLS.static_counts(RAW["aug-cc-pvtz.gbs"])
        self.assertEqual(aux["C"]["shells_by_l"], {"0": 9, "1": 7, "2": 6, "3": 4, "4": 2})
        self.assertEqual(aux["H"]["shells_by_l"], {"0": 5, "1": 4, "2": 3, "3": 2})
        self.assertEqual(aux["6C6H"], dict(nshell=252, cartesian=1146, spherical=912))
        self.assertEqual(main["6C6H"], dict(nshell=138, cartesian=480, spherical=414))

    def test_mock_unscoped_override_and_scoped_cartesian_success(self):
        wrong = self.core.BasisSet.build(self.molecule, "DF_BASIS_SCF", "aug-cc-pvtz-ri", puream=0)
        self.assertEqual((wrong.nbf(), wrong.nao()), (912, 1146))
        audit = {}
        correct = TOOLS.scoped_auxiliary(self.core, self.options, self.molecule, audit)
        self.assertEqual((correct.nbf(), correct.nao(), correct.nshell()), (1146, 1146, 252))
        self.assertFalse(correct.has_puream())
        self.assertEqual(audit["before"], audit["after"])

    def test_success_restores_both_flags_for_all_four_input_states(self):
        for value in (False, True):
            for changed in (False, True):
                with self.subTest(value=value, changed=changed):
                    core = FakeCore(value, changed)
                    audit = {}
                    actual = TOOLS.scoped_auxiliary(core, options_state(core), self.molecule, audit)
                    self.assertEqual(actual.nbf(), 1146)
                    self.assertEqual(TOOLS.puream_state(core), dict(value=value, changed=changed))
                    self.assertEqual(audit["during"], dict(value=False, changed=True))

    def test_error_restores_both_flags_and_same_error_for_all_input_states(self):
        error = RuntimeError("controlled mock builder")
        def fail(): raise error
        for value in (False, True):
            for changed in (False, True):
                with self.subTest(value=value, changed=changed):
                    core, audit = FakeCore(value, changed), {}
                    with self.assertRaises(RuntimeError) as caught:
                        TOOLS.scoped_auxiliary(core, options_state(core), self.molecule, audit, builder=fail)
                    self.assertIs(caught.exception, error)
                    self.assertEqual(TOOLS.puream_state(core), dict(value=value, changed=changed))
                    self.assertTrue(audit["restored_exactly"])

    def test_912_nbf_is_not_admitted_using_1146_nao(self):
        basis = FakeBasis("auxiliary", True, self.molecule)
        with self.assertRaisesRegex(ValueError, r"ACTUAL.*912.*1146.*nshell"):
            TOOLS.check_shape(TOOLS.native_snapshot(basis), TOOLS.contract()["auxiliary"], "AUX")

    def test_actual_driver_fragment_admits_recipe_before_one_energy(self):
        scope = self.driver_fragment(self.manifest())
        self.energy.assert_called_once()
        self.assertEqual(self.events, ["basis-receipt", "energy"])
        scope["recipe"].assert_called_once()
        self.assertEqual(scope["part"], "MOCK retained recipe")
        self.assertFalse(scope["recipe"].call_args.kwargs["auxiliary_native"].has_puream())

    def test_matching_counts_but_changed_shell_coefficients_refuse_before_energy(self):
        manifest = self.manifest()
        self.core.aux_shift = .125
        with self.assertRaisesRegex(ValueError, "exact ordered native shell"):
            self.driver_fragment(manifest)
        self.energy.assert_not_called()
        self.scope["recipe"].assert_not_called()
        self.assertEqual(self.scope["admission"]["auxiliary"]["metadata"],
                         manifest["admission"]["auxiliary"]["metadata"])

    def test_changed_canonical_file_hash_refuses_before_energy(self):
        manifest = self.manifest()
        path = self.stage / "share/psi4/basis/aug-cc-pvtz-ri.gbs"
        path.write_text(path.read_text() + "\n! mock hash corruption, counts unchanged\n")
        with self.assertRaisesRegex(ValueError, "basis file hash mismatch"):
            self.driver_fragment(manifest)
        self.energy.assert_not_called()

    def test_changed_source_binding_refuses_before_energy(self):
        manifest = self.manifest()
        manifest["binding"]["native"]["commit"] = "other"
        with self.assertRaisesRegex(ValueError, "binding mismatch"):
            self.driver_fragment(manifest)
        self.energy.assert_not_called()

    def test_changed_representation_refuses_before_energy(self):
        manifest = self.manifest()
        # Only this mock builder returns an intentionally wrong spherical AUX.
        original = self.core.build
        def wrong(molecule, key, name, puream):
            return FakeBasis("auxiliary", True, molecule) if name.endswith("-ri") else original(
                molecule, key, name, puream)
        self.core.BasisSet.build = wrong
        with self.assertRaisesRegex(ValueError, "ACTUAL.*912"):
            self.driver_fragment(manifest)
        self.energy.assert_not_called()
        self.assertEqual(TOOLS.puream_state(self.core), dict(value=True, changed=True))

    def test_tampered_reference_fingerprint_refuses_before_energy(self):
        manifest = self.manifest()
        manifest["admission"]["auxiliary"]["ordered_shells"][0]["native_normalized_coefficients_hex"][0] = "0x1p+0"
        with self.assertRaisesRegex(ValueError, "corrupt native shell fingerprint"):
            self.driver_fragment(manifest)
        self.energy.assert_not_called()

    def test_post_scf_same_main_passes_and_same_count_changed_shell_fails(self):
        main = FakeBasis("main", True, self.molecule)
        reference = TOOLS.native_snapshot(main)
        TOOLS.compare_scf_main(main, reference)
        receipt = {}
        with self.assertRaisesRegex(ValueError, "SCF wavefunction MAIN"):
            TOOLS.compare_scf_main(FakeBasis("main", True, self.molecule, shift=.125), reference, receipt)
        self.assertEqual(receipt["actual_scf_main"]["metadata"], reference["metadata"])

    def test_native_coefficients_and_raw_gbs_are_separately_labeled(self):
        receipt = self.manifest()["admission"]
        self.assertIn("NOT native normalization", receipt["basis_files"]["aug-cc-pvtz.gbs"]["raw_coefficient_convention"])
        self.assertIn("native-normalized", receipt["auxiliary"]["coefficient_convention"])
        self.assertNotEqual(receipt["auxiliary"]["fingerprint_sha256"],
                            receipt["basis_files"]["aug-cc-pvtz-ri.gbs"]["raw_shell_sha256"])

    def test_recipe_not_reconstructed_after_energy_in_actual_driver(self):
        tree = ast.parse((HERE / "pilot.py").read_text())
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
        calls = [n for n in ast.walk(main) if isinstance(n, ast.Call)]
        energy = [n for n in calls if isinstance(n.func, ast.Attribute) and n.func.attr == "energy"]
        self.assertEqual(len(energy), 1)
        self.assertFalse([n for n in calls if isinstance(n.func, ast.Name) and n.func.id == "recipe"
                          and n.lineno > energy[0].lineno])

    def test_binding_distinguishes_corrected_driver_and_accepted_native_roots(self):
        fixture = Path(self.temp.name) / "mock-fixture.json"
        fixture.write_text("{}")
        args = SimpleNamespace(
            source="/accepted/native", commit="native-commit", native_tree="native-tree",
            stage="/accepted/native/build_ccasp_phoenix_benzene/stage", core_sha256="native-core",
            driver_source=str(ROOT), driver_commit="driver-commit", driver_tree="driver-tree",
            fixture=str(fixture))
        binding = PILOT.basis_binding(args, HERE / "pilot.json")
        self.assertEqual(binding["driver"]["commit"], "driver-commit")
        self.assertEqual(binding["native"]["commit"], "native-commit")
        self.assertNotEqual(binding["driver"]["source"], binding["native"]["source"])
        self.assertEqual(binding["driver"]["files_sha256"]["basis_admission.py"],
                         TOOLS.file_sha(HERE / "basis_admission.py"))

    def test_probe_has_no_scf_or_integral_property_calls(self):
        tree = ast.parse((HERE / "basis_probe.py").read_text())
        forbidden = {"energy", "oeprop", "native_partition", "IsaGrid", "MintsHelper",
                     "compute_energy", "native_constrained_ov"}
        self.assertFalse([n for n in ast.walk(tree) if isinstance(n, ast.Call)
                          and isinstance(n.func, ast.Attribute) and n.func.attr in forbidden])
        text = (HERE / "basis_probe.sbatch").read_text()
        for flag in ("--cpus-per-task=2", "--mem=8G", "--time=00:10:00",
                     "--nodelist=atl1-1-02-007-23-1", "--qos=embers"):
            self.assertIn(flag, text)

    def test_probe_systemexit_zero_is_a_nonzero_failure_receipt(self):
        probe = load("mock_only_basis_probe", "basis_probe.py")
        root = Path(self.temp.name) / "probe"
        (root / "metadata").mkdir(parents=True)
        with mock.patch.object(probe, "RUN", root), mock.patch.object(
                probe, "run", side_effect=SystemExit(0)), mock.patch.object(
                sys, "argv", ["probe", "run", "--commit", "mock", "--tree", "mock"]), \
                contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(probe.main(), 1)
        result = json.loads((root / "metadata/bootstrap-failure.json").read_text())
        self.assertIn("SystemExit", result["traceback"])
        self.assertTrue(result["no_scf"])


if __name__ == "__main__":
    unittest.main()
