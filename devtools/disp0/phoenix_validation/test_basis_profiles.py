import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from fragment_gate import options
from full_inputs import load_point
from prepare_inputs import (BASIS, BASIS_PROFILES, EXPECTED, PROFILE_COUNTS,
                            input_basis_profile, prepare, spherical_counts)


BASIS_DIRECTORY = Path(__file__).resolve().parents[3]/"psi4/share/psi4/basis"


class BasisProfileTests(unittest.TestCase):
    def fixture(self, root):
        # Independent literal compositions, with no Psi4 execution/import.
        fragments = {
            "peptide": ({"C":3,"H":7,"N":1,"O":1}, {"C":3,"H":7,"N":1,"O":1}),
            "nanotube": ({"C":2,"H":4}, {"C":24,"H":12}),
            "protein83": ({"C":12,"H":24,"N":4,"O":4}, {"C":18,"H":20,"O":1}),
        }
        systems = {}
        for name, pair in fragments.items():
            text = []
            for index, composition in enumerate(pair):
                if index:
                    text.append("--")
                text.append("0 1")
                for element, count in composition.items():
                    text.extend(f"{element} {i} {index} 0" for i in range(count))
            systems[name] = "\n".join(text+[
                "units angstrom", "symmetry c1", "no_reorient", "no_com"])+"\n"
        path = root/"systems.py"
        path.write_text("SYSTEMS = "+repr({**systems, "protein157":"DO NOT RUN"})+"\n")
        return path

    def test_profiles_count_real_basis_files_and_roundtrip_all_selected_systems(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            module = self.fixture(root)
            for profile in BASIS_PROFILES:
                output = root/profile
                doc = prepare(module, BASIS_DIRECTORY, output, basis_profile=profile)
                self.assertEqual(input_basis_profile(doc), profile)
                self.assertEqual(set(doc["systems"]), set(EXPECTED))
                path = output/"inputs.json"
                checksum = hashlib.sha256(path.read_bytes()).hexdigest()
                for system, counts in PROFILE_COUNTS[profile].items():
                    point, dimensions = load_point(path, checksum, system)
                    self.assertEqual(tuple(point["spherical_basis_counts"][role]
                                           for role in ("primary","jkfit","rifit")), counts)
                    self.assertEqual(dimensions["ao"], counts[0])
                    self.assertEqual(dimensions["atoms"], EXPECTED[system][0])
                if profile == "aug-cc-pvdz":
                    self.assertEqual(doc["schema"], 1)
                    self.assertEqual(doc["basis_names"], BASIS)
                    self.assertNotIn("basis_profile", doc)
                else:
                    self.assertEqual(doc["schema"], 2)
                with self.assertRaises(ValueError):
                    load_point(path, checksum, "protein157")

    def test_arms_differ_only_by_backend_and_preserve_accuracy_options(self):
        original = options(False, True)
        for profile, names in BASIS_PROFILES.items():
            cpu, gpu = options(False, True, profile), options(True, True, profile)
            self.assertEqual({k:v for k,v in cpu.items() if k!="use_cuest"},
                             {k:v for k,v in gpu.items() if k!="use_cuest"})
            self.assertEqual(cpu["basis"], names["primary"])
            self.assertEqual(cpu["df_basis_scf"], names["jkfit"])
            self.assertEqual(cpu["df_basis_sapt"], names["jkfit"])
            self.assertEqual(cpu["df_basis_mp2"], names["rifit"])
            for key in original.keys()-{"basis","df_basis_scf","df_basis_sapt","df_basis_mp2"}:
                self.assertEqual(cpu[key], original[key])
        with self.assertRaises(ValueError):
            options(True, True, "unknown")

    def test_refrozen_tampered_counts_or_mixed_fitting_sets_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            doc = prepare(self.fixture(root), BASIS_DIRECTORY, root/"svp",
                          basis_profile="def2-svp")
            def check(value, error):
                path = root/"tampered.json"
                path.write_text(json.dumps(value))
                with self.assertRaises(error):
                    load_point(path, hashlib.sha256(path.read_bytes()).hexdigest(), "protein83")
            changed = json.loads(json.dumps(doc))
            changed["systems"]["protein83"]["spherical_basis_counts"]["primary"] += 1
            check(changed, RuntimeError)
            changed = json.loads(json.dumps(doc))
            changed["basis_names"]["jkfit"] = "aug-cc-pvdz-jkfit"
            check(changed, ValueError)
            changed = json.loads(json.dumps(doc))
            changed["basis_files"]["primary"]["filename"] = "def2-tzvp.gbs"
            check(changed, ValueError)
            changed = json.loads(json.dumps(doc))
            changed["schema"] = 1
            check(changed, ValueError)

    def test_primary_spdf_bound_is_not_relaxed_for_def2_auxiliary_g_shells(self):
        with self.assertRaises(ValueError):
            spherical_counts(BASIS_DIRECTORY/"def2-universal-jkfit.gbs")
        counts = spherical_counts(BASIS_DIRECTORY/"def2-universal-jkfit.gbs", maximum_l=4)
        self.assertEqual(counts["maximum_l"]["C"], 4)
        self.assertEqual(counts["spherical_functions"]["C"], 75)


if __name__ == "__main__":
    unittest.main()
