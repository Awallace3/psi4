import importlib.util
import json
import os
from pathlib import Path
import tempfile
import subprocess
import unittest

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("compare", ROOT / "compare.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class CompareTests(unittest.TestCase):
    def setUp(self):
        self.c = json.loads((ROOT / "config.json").read_text())
        self.c["repeats"] = 1

    def record(self, engine, **kwargs):
        r = dict(engine=engine, system="peptide", functional="pbe", profile=False,
                 ok=True, process_returncode=0, scf_wall_s=2 if engine == "psi4" else 1,
                 calculation_wall_s=3 if engine == "psi4" else 2,
                 process_wall_s=4, scf_iterations=15, energy_hartree=-100,
                 nbf=210, naux=600, functional_metadata=dict(
                     omega=0, short_range_exchange=0, long_range_exchange=0,
                     components=self.c["functionals"]["pbe"]["components"]))
        r.update(kwargs)
        return r

    def row(self, records):
        return next(r for r in m.summarize(records, self.c, ["peptide"]) if r["functional"] == "pbe")

    def test_complete_ratio(self):
        row = self.row([self.record(e) for e in m.ENGINES])
        self.assertTrue(row["complete"])
        self.assertTrue(row["comparison_eligible"])
        self.assertEqual(row["psi4_over_pyscf_scf_time"], 2)
        self.assertEqual(row["psi4_over_pyscf_calculation_time"], 1.5)

    def test_teardown_failure_is_not_success(self):
        row = self.row([self.record("psi4"), self.record("pyscf", process_returncode=-11)])
        self.assertFalse(row["complete"])
        self.assertNotIn("psi4_over_pyscf_scf_time", row)

    def test_instrumentation_excluded(self):
        row = self.row([self.record(e) for e in m.ENGINES] + [
            self.record("pyscf", profile=True, scf_wall_s=100)])
        self.assertEqual(row["psi4_over_pyscf_scf_time"], 2)

    def test_aux_mismatch_suppresses_ratio(self):
        row = self.row([self.record("psi4"), self.record("pyscf", naux=601)])
        self.assertFalse(row["comparison_eligible"])
        self.assertNotIn("psi4_over_pyscf_scf_time", row)

    def test_functional_mismatch_suppresses_ratio(self):
        p = self.record("pyscf")
        p["functional_metadata"]["omega"] = .3
        row = self.row([self.record("psi4"), p])
        self.assertFalse(row["comparison_eligible"])

    def test_missing_repeat_remains_incomplete(self):
        self.c["repeats"] = 3
        self.assertFalse(self.row([self.record(e) for e in m.ENGINES])["complete"])

    def test_components_mismatch_suppresses_ratio(self):
        p = self.record("pyscf")
        p["functional_metadata"]["components"] = {"LDA_X": 1}
        row = self.row([self.record("psi4"), p])
        self.assertFalse(row["comparison_eligible"])
        self.assertNotIn("psi4_over_pyscf_calculation_time", row)

    def test_vwn_variants_cannot_pass_component_validation(self):
        with self.assertRaises(AssertionError):
            m.verify_components({"LDA_X": 1, "LDA_C_VWN": 1},
                                self.c["functionals"]["svwn"]["components"])

    def test_malformed_result_is_a_failure(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "result.json"
            p.write_text("{not json")
            r = m.load_result(p, dict(engine="pyscf"))
            self.assertFalse(r["ok"])
            self.assertEqual(r["engine"], "pyscf")

    def test_wrong_worker_identity_is_a_failure(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "result.json"
            p.write_text(json.dumps(dict(engine="psi4", ok=False)))
            self.assertFalse(m.load_result(p, dict(engine="pyscf"))["ok"])

    def test_energy_review_is_separate(self):
        row = self.row([self.record("psi4"), self.record("pyscf", energy_hartree=-99)])
        self.assertTrue(row["complete"])
        self.assertTrue(row["energy_review"])

    def test_no_pruning_and_native_fp64(self):
        opts = m.psi_options(self.c)
        self.assertEqual(opts["dft_pruning_scheme"], "none")
        self.assertFalse(opts["cuest_mixed_precision"])

    def test_vwn_rpa_explicit(self):
        self.assertEqual(self.c["functionals"]["b3lyp"]["pyscf"], "HYB_GGA_XC_B3LYP")
        self.assertEqual(self.c["functionals"]["svwn"]["pyscf"], "LDA_X,LDA_C_VWN_RPA")
        self.assertEqual(set(x["rung"] for x in self.c["functionals"].values()),
                         {"lda", "gga", "mgga", "hybrid", "range-separated"})

    def test_xyz_size_guard(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.xyz"
            p.write_text("1\ncharge=0 mult=1\nH 0 0 0\n")
            with self.assertRaises(AssertionError):
                m.xyz(p, "peptide")

    def test_real_suite_geometry_counts(self):
        suite = Path("/home/awallace43/data/timing_test_suite/geometries")
        if not suite.is_dir():
            self.skipTest("Local timing suite unavailable")
        for system, n in m.ATOMS.items():
            atom = m.xyz(suite / f"{system}.xyz", system)
            atoms = [line.split()[0] for line in atom.splitlines()]
            self.assertEqual(len(atoms), n)
            self.assertEqual(sum(5 if e == "H" else 14 for e in atoms), m.NBF[system])

    def test_nested_timer_counts_are_not_fabricated(self):
        class Obj:
            def f(self):
                return 7
        obj = Obj()
        t = m.Timings(lambda: None)
        t.wrap(obj, "f", "f")
        self.assertEqual(obj.f(), 7)
        self.assertEqual(t.data["f"]["calls"], 1)

    def test_psiapi_setup_accepts_unset_pythonpath_under_nounset(self):
        payload = (ROOT / "payload.sh").read_text()
        path_setup = payload.split("export PYTHONNOUSERSITE=1\n", 1)[1].split(
            "export LD_LIBRARY_PATH=", 1)[0]
        psiapi_setup = payload.split('psi_exe=', 1)[1].split(
            'mkdir -p "$TMPDIR/inputs/geometries"', 1)[0]
        with tempfile.TemporaryDirectory() as d:
            stage = Path(d) / "stage"
            (stage / "bin").mkdir(parents=True)
            exe = stage / "bin/psi4"
            # The actual historical staged binary emits this same expansion.
            exe.write_text("#!/bin/bash\nprintf '%s\\n' 'export PATH=/stage/bin:$PATH PYTHONPATH=/stage/lib:$PYTHONPATH'\n")
            exe.chmod(0o755)
            env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
            env.update(COMPARE_PSI_PACKAGE=str(stage / "lib/psi4"),
                       COMPARE_PSI_ENV=d, COMPARE_PSI_CUDA_LIB=d)
            script = path_setup + "psi_exe=" + psiapi_setup + '\n[[ "$PYTHONPATH" == /stage/lib:* ]]'
            p = subprocess.run(["bash", "-euc", script], env=env,
                               capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stderr)

    def test_statically_linked_cupy_runtime_identity(self):
        class Runtime:
            getDeviceCount = staticmethod(lambda: 1)
            getDevice = staticmethod(lambda: 0)
            getDeviceProperties = staticmethod(lambda _: {
                "uuid": bytes.fromhex("951a11c240ee3fcbf71f49849e744cb1")})
            runtimeGetVersion = staticmethod(lambda: 12080)
        self.assertEqual(m.cuda_identity(Runtime())["uuid"],
                         "GPU-951a11c2-40ee-3fcb-f71f-49849e744cb1")

    def test_multiple_visible_gpus_are_rejected(self):
        class Runtime:
            getDeviceCount = staticmethod(lambda: 2)
            getDevice = staticmethod(lambda: 0)
        with self.assertRaises(AssertionError):
            m.cuda_identity(Runtime())


if __name__ == "__main__":
    unittest.main()
