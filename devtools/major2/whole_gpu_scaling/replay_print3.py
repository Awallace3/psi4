"""Explicit external-fixture parser replay; never full allocation acceptance.

Run with --artifacts /absolute/path/to/failure-14182594/artifacts.
No archive is bundled, inferred, skipped, or modified. No scientific imports.
"""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import re
import unittest

import config  # establishes the same helper import path as the controller
import science as legacy


SUITE = Path("/home/awallace43/data/timing_test_suite/geometries/systems.py")
SUITE_SHA = "72668390d24f134316c8e779931a2f65efcd9289ecea8f1cf20a72733262e3bb"
GEOMETRY_SHA = "14b4041b272a66bea7daf3d010ffd695a6340fd554906b0d7309de4c80b2acab"
HASHES = {
    "profile.txt": "1111c0e9c572c4e15868e792f4cf877b68cb5c627f0e69dba803908d30c5b8d3",
    "psi4.out": "f5a09c362b09a0c0878afc31eb0bf5e807d4fbd1777e41ebdbb99b7df268e6c9",
    "result.json": "674aabc204343050f53e8db14498d1ac784f2868e25c416d38cc89a6e351736e",
    "timer.dat": "1d2364d21ea4c8282fcf46099092e3037b482abafb21c3551a7a738cf3f17313",
}
# Frozen specification, independent of result.json and the runner under test.
OPTIONS = {
    "basis": "6-31+g**", "scf_type": "df", "SCF_SUBTYPE": "AUTO", "reference": "rhf",
    "DF_BASIS_SCF": "def2-universal-jkfit", "DF_BASIS_MP2": "aug-cc-pvdz-ri", "PUREAM": True,
    "D_CONVERGENCE": 8, "E_CONVERGENCE": 9, "MAXITER": 150, "ORBITAL_OPTIMIZER_PACKAGE": "INTERNAL",
    "SAD_PRINT": 1, "SAPT_DFT_FUNCTIONAL": "pbe0", "SAPT_DFT_DO_DHF": True,
    "SAPT_DFT_USE_EINSUMS": True, "SAPT_DFT_GRAC_SHIFT_A": .136, "SAPT_DFT_GRAC_SHIFT_B": .136,
    "SAPT_DFT_INDUCTION_TYPE": "CPKS", "CUEST_MIXED_PRECISION": False,
    "USE_CUEST": True, "CUEST_XC": True, "CUEST_SAD": True,
    "DFT_RADIAL_POINTS": 75, "DFT_SPHERICAL_POINTS": 302,
    "DFT_RADIAL_POINTS_INITIAL": 35, "DFT_SPHERICAL_POINTS_INITIAL": 194,
    "DFT_GRID_SWITCH_FACTOR": 1.0,
}
STAGE = ("/storage/project/r-cs207-0/awallace43/gits/"
         "psi4.saptdft-cuest-major2-jkexception-baseline-20261018/"
         "build_major2_cuda_jkexception/stage/lib/psi4")
EXPECTED = dict(
    source_commit="c56708fc6f6e8fcf2801868f0d3cb49bdb1bb0a2",
    runner_sha256="c002cce1f2bb0bb39a81b1954505e79c7627f81f3cf51fd7128dc84ce306c087",
    suite_systems_sha256=SUITE_SHA, geometry_sha256=GEOMETRY_SHA,
    slurm_job_id="14182594", hostname="atl1-1-03-020-18-0.pace.gatech.edu",
    system="peptide", suite_system="peptide", psi4_module=STAGE+"/__init__.py",
    method="sapt(dft)-d4(i)", basis="6-31+g**",
    trace_print_options={"global": 3, "scf": 3, "sapt": 3},
    science_options_sha256="e8954d0c9a6267a3b1593ffd7bde745b82293c8f68629caa7945535adbb39346",
    grid_policy=dict(initial="35x194", final="75x302", switch_factor=1.0, minimum_final_iterations=2),
    policy_checks=dict(problems=[], switched_scfs=2),
    iterator=dict(path=STAGE+"/driver/procrouting/scf_proc/scf_iterator.py",
                  sha256="0cb954e4f1cbf3b1c4b95c6b9652a51da08d2467df62041c978a6d229b0051d7"),
    natom=24, nbf=250, naux_jkfit=1010, naux_ri=1042, nbf_monomer_a=125, nbf_monomer_b=125,
)


def load_fixture(root):
    """Fail loudly on unavailable, changed, or independently unpinned evidence."""
    assert root.is_absolute() and root.is_dir(), "explicit existing absolute artifacts path required"
    out = root/"raw/sessions/whole-baseline-peptide-r1"
    for name, digest in HASHES.items():
        assert hashlib.sha256((out/name).read_bytes()).hexdigest() == digest, name
    receipt = json.loads((root/"raw/whole-baseline-peptide-r1.COMPLETE.json").read_text())
    child_exit = json.loads((root/"raw/whole-baseline-peptide-r1.exit.json").read_text())
    assert receipt["output_hashes"] == HASHES
    assert receipt["evidence_kind"] == "parent post-exit completed output custody"
    assert receipt["child_exit"] == child_exit
    assert child_exit["exit"] == 0 and child_exit["interrupted"] is False
    assert child_exit["pid"] == receipt["pid"] == 3551173
    assert hashlib.sha256(SUITE.read_bytes()).hexdigest() == SUITE_SHA
    tree = ast.parse(SUITE.read_text())
    systems = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "SYSTEMS" for t in n.targets))
    geom = systems["peptide"]
    assert hashlib.sha256(geom.encode()).hexdigest() == GEOMETRY_SHA
    return [json.loads((out/"result.json").read_text()), (out/"psi4.out").read_text(),
            (out/"timer.dat").read_text(), (out/"profile.txt").read_text(),
            copy.deepcopy(EXPECTED), copy.deepcopy(OPTIONS), geom]


class LegacyReplay(unittest.TestCase):
    def test_unchanged_real_output_fails_first_legacy_token(self):
        with self.assertRaisesRegex(RuntimeError, re.escape("Spherical Harmonics?: true")) as caught:
            legacy.validate_record(*self.values)
        print("LEGACY RED (unchanged bytes):", caught.exception)

    def test_second_legacy_token_is_independently_absent(self):
        # A token-only diagnostic, NOT a modified record or a validator bypass.
        text = self.values[1]
        for token in ("Spherical Harmonics?: true", "Basis Set: 6-31+G**"):
            self.assertNotIn(token, text)
            with self.assertRaisesRegex(RuntimeError, re.escape(token)) as caught:
                legacy.require(token in text, "missing backend/options print: "+token)
            print("LEGACY token-only RED (unchanged bytes; not record acceptance):", caught.exception)


class OwnedReplay(unittest.TestCase):
    def reject_change(self, change, message, trajectory=False):
        from validation import validate_record
        values = copy.deepcopy(self.values)
        change(values)
        self.assertNotEqual(values, self.values, "negative mutation must actually change evidence")
        with self.assertRaisesRegex(RuntimeError, message):
            validate_record(*values, candidate=False)
            if trajectory:
                import repeatability
                repeatability.trajectory(values[1])

    def replace_text(self, old, new, message):
        self.assertIn(old, self.values[1])
        self.reject_change(lambda v: v.__setitem__(1, v[1].replace(old, new, 1)), message)

    def test_unchanged_four_artifacts(self):
        from validation import validate_record
        from trace import parse, timer_regions
        import repeatability
        verdict = validate_record(*self.values, candidate=False)
        trajectory = repeatability.trajectory(self.values[1])
        self.assertEqual([len(v["rows"]) for v in trajectory],
                         [v["iterations"]+1 for v in self.values[0]["proofs"]])
        traced = parse(self.values[1], "baseline")
        self.assertGreater(traced["observed_created"], 0)
        self.assertGreater(timer_regions(self.values[0])["SCF: JK native initialize"]["n_calls"], 0)
        self.assertEqual(len(verdict["scfs"]), 5)
        self.assertEqual(len(verdict["solvers"]), 2)
        print("OWNED GREEN: unchanged four real artifacts + trajectory/trace/regions;"
              " parser replay ONLY, NOT full allocation acceptance.")

    def test_every_frozen_identity_gate(self):
        for key, value in EXPECTED.items():
            with self.subTest(key=key):
                self.reject_change(lambda v: v[0].__setitem__(key, None), "record identity/options: "+key)

    def test_options_and_runtime_policy(self):
        self.reject_change(lambda v: v[0]["options"].__setitem__("E_CONVERGENCE", 8),
                           "scientific options changed")
        for key, value in (("ok", False), ("arm", "cpu"), ("threads", 9), ("memory", "99 GiB")):
            with self.subTest(key=key):
                self.reject_change(lambda v: v[0].__setitem__(key, value), "runtime policy")

    def test_wrong_or_missing_backend_and_cpu_fallback(self):
        for key, value in (("gpu_builder_seen", False), ("sad_gpu_builder_seen", False),
                           ("cpu_builder_seen", True), ("direct_seen", True), ("disk_df_builder_seen", True)):
            with self.subTest(key=key):
                self.reject_change(lambda v: v[0].__setitem__(key, value), "GPU backend record")
        for token in ("cuESTJK: GPU-Accelerated", "NVIDIA H200",
                      "Threads set to 8 by Python driver.", "Memory set to 100.000 GiB by Python driver."):
            with self.subTest(missing=token):
                self.reject_change(lambda v: v.__setitem__(1, v[1].replace(token, "WRONG")),
                                   "missing backend/options print")
        self.replace_text("SAD J/K backend: cuESTJK", "SAD J/K backend: WRONG", "actual SAD backend")
        self.reject_change(lambda v: v.__setitem__(1, re.sub(r"SAD J/K backend: \S+", "", v[1])),
                           "actual SAD backend")
        for token in ("MemDFJK: Density-Fitted", "DiskDFJK", "DirectJK"):
            with self.subTest(fallback=token):
                self.reject_change(lambda v: v.__setitem__(1, v[1]+"\n"+token), "CPU fallback")

    def test_perturbed_and_reordered_printed_geometry(self):
        atoms = list(re.finditer(r"^[ \t]*[A-Z][a-z]?(?:[ \t]+[-0-9.]+){4}[ \t]*$", self.values[1], re.M))
        self.assertGreaterEqual(len(atoms), 24)
        first, second = atoms[:2]
        fields = first[0].split()
        fields[1] = str(float(fields[1])+.001)
        self.replace_text(first[0], "    "+"    ".join(fields), "geometry coordinates")
        def reorder(v):
            t = v[1]
            v[1] = t[:first.start()]+second[0]+t[first.end():second.start()]+first[0]+t[second.end():]
        self.reject_change(reorder, "geometry coordinates")

    def test_primary_wrong_basis_false_and_so_on_real_output(self):
        self.replace_text("Name                   = 6-31+G**",
                          "Name                   = WRONG", "PRINT3: primary basis name")
        self.replace_text("Spherical Harmonics    = TRUE",
                          "Spherical Harmonics    = FALSE", "PRINT3: primary spherical")
        self.replace_text("Number of SO           = 250",
                          "Number of SO           = 260", "PRINT3: primary SO count")

    def test_missing_duplicate_displaced_and_late_real_primary(self):
        text = self.values[1]
        match = re.search(r"^[ \t]*==> Primary Basis <==.*?^[ \t]*-Contraction Scheme:[ \t]*\n",
                          text, re.M | re.S)
        self.assertIsNotNone(match)
        block = match[0]
        self.replace_text(block, "", "PRINT3: exactly five")
        self.replace_text(block, block+block, "PRINT3: exactly five")
        def move(v, destination):
            t = v[1].replace(block, "", 1)
            v[1] = t[:destination(t)]+block+t[destination(t):]
        for destination, gate in (
            (lambda t: 0, "PRINT3: displaced/reordered"),
            (lambda t: t.index("SAPT(DFT): delta HF Monomer A"), "PRINT3: displaced/reordered"),
            (lambda t: t.index("\n", t.index("@DF-RHF Final Energy:"))+1, "PRINT3: displaced/reordered"),
            (lambda t: t.index("@DF-RHF iter SAD:"), "PRINT3: late primary"),
        ):
            with self.subTest(gate=gate, destination=destination):
                # For late-before-first-row, place after the first row instead.
                if gate == "PRINT3: late primary":
                    destination = lambda t: t.index("\n", t.index("@DF-RHF iter SAD:"))+1
                self.reject_change(lambda v: move(v, destination), gate)

    def test_bad_missing_aux_role_identity_on_real_output(self):
        for old, new in (("Role: JKFIT", "Role: WRONG"), ("Role: RIFIT", "Role: WRONG"),
                         ("Keyword: DF_BASIS_MP2", "Keyword: DF_BASIS_SCF"),
                         ("Name: AUG-CC-PVDZ-RI", "Name: WRONG"),
                         ("Keyword: DF_BASIS_SCF", "")):
            with self.subTest(new=new):
                self.replace_text(old, new, "PRINT3:")
        self.reject_change(lambda v: v.__setitem__(1, re.sub(
            r"[ \t]*=> Loading Basis Set <=\n\n[ \t]*Name: AUG-CC-PVDZ-RI\n"
            r"[ \t]*Role: RIFIT\n[ \t]*Keyword: DF_BASIS_MP2\n(?:[ \t]*atoms[^\n]*\n)+",
            "AUG-CC-PVDZ-RI\n", v[1])), "PRINT3: missing orbital/JKFIT/RIFIT")

    def test_real_banner_and_primary_displaced_together(self):
        text = self.values[1]
        banner = re.search(r"^[ \t]*SAPT\(DFT\): delta HF Dimer[ \t]*\n", text, re.M)[0]
        block = re.search(r"^[ \t]*==> Primary Basis <==.*?^[ \t]*-Contraction Scheme:[ \t]*\n",
                          text, re.M | re.S)[0]
        def move(v):
            t = v[1].replace(banner, "", 1).replace(block, "", 1)
            pos = t.index("\n", t.index("@DF-RHF iter SAD:"))+1
            v[1] = t[:pos]+banner+block+t[pos:]
        self.reject_change(move, "PRINT3: late primary block")

    def test_real_blank_separated_loader_conflicts(self):
        row = re.search(r"^[ \t]+atoms [^\n]+\n", self.values[1], re.M)[0]
        for field in ("Role: WRONG", "Name: WRONG", "Keyword: WRONG"):
            with self.subTest(field=field):
                self.replace_text(row, row+"\n    "+field+"\n", "PRINT3:.*loading fields")

    def test_timer_values_calls_and_paths(self):
        root = "SAPT(DFT) Energy"
        for field, value, gate in (("wall_time", 6.124, "printed timer values"),
                                    ("n_calls", 99, "printed timer calls"),
                                    ("timer_path", ["WRONG"], "timer identity/orphan")):
            with self.subTest(field=field):
                # Use a child timer so wall/energy coverage does not mask value comparison.
                key = next(k for k in self.values[0]["timer_records"] if k != root)
                self.reject_change(lambda v: v[0]["timer_records"][key].__setitem__(field, value), gate)
        printed = self.values[2]
        last_root = printed.rindex(root)
        row = re.search(r"^SAPT\(DFT\) Energy.*$", printed[last_root:], re.M)[0]
        calls = re.search(r"\d+ calls", row)[0]
        for replacement, gate in ((row.replace(calls, "99 calls"), "printed timer calls"),
                                   (row.replace(root, "WRONG Energy"), "flat/hierarchical timer roots")):
            with self.subTest(gate=gate):
                self.reject_change(lambda v: v.__setitem__(2, printed[:last_root]+
                                   printed[last_root:].replace(row, replacement, 1)), gate)
        self.reject_change(lambda v: v[0]["timer_records"][root].__setitem__("wall_time", float("nan")),
                           "timer finiteness")

    def test_scf_grid_solver_and_nonfinite(self):
        self.reject_change(lambda v: v[0]["proofs"][0].__setitem__("converged", False), "SCF proof binding")
        self.replace_text("Energy and wave function converged.", "", "SCF/DFT count")
        self.replace_text("XC-FINAL", "XC-INITIAL", "switch chronology")
        self.reject_change(lambda v: v[0]["proofs"][3].__setitem__("final_grid_after", False),
                           "grid proof differs")
        self.reject_change(lambda v: v[0]["cpks_solvers"][0].__setitem__("converged", False),
                           "solver histories disagree")
        solver_start = self.values[1].index("SAPT Coupled Induction Solver")
        residual = re.search(r"^[ \t]+Guess[ \t]+(\S+)", self.values[1][solver_start:], re.M)
        self.assertIsNotNone(residual)
        pos = solver_start+residual.start(1)
        old = residual[1]
        self.reject_change(lambda v: v.__setitem__(1, v[1][:pos]+"nan"+v[1][pos+len(old):]),
                           "nonfinite residual")
        for key in ("wall_s", "ru_maxrss_gib"):
            with self.subTest(key=key):
                self.reject_change(lambda v: v[0].__setitem__(key, float("nan")), "wall/RSS")
        self.reject_change(lambda v: v[0]["cpks_solve_calls_s"].__setitem__(0, float("inf")),
                           "solver timing coverage")
        self.reject_change(lambda v: v[0]["components_hartree"].__setitem__("SAPT ELST ENERGY", float("nan")),
                           "energy components finite")
        row = re.search(r"@DF-RHF iter[ \t]+1:[ \t]+(\S+)", self.values[1])
        self.assertIsNotNone(row)
        self.reject_change(lambda v: v.__setitem__(1, v[1][:row.start(1)]+"nan"+v[1][row.end(1):]),
                           "nonfinite trajectory", trajectory=True)

    def test_fatal_grac_components_and_profile(self):
        self.reject_change(lambda v: v.__setitem__(1, v[1]+"\nSCFConvergenceError"), "fatal output")
        self.replace_text("GRAC shift supplied by the user (not computed).", "", "GRAC")
        self.reject_change(lambda v: v[0].__setitem__("returned_energy_hartree", -1.0), "energy sum")
        # Only the final [mEh] component row is authoritative, not earlier prose.
        row = re.findall(r"^[ \t]+Electrostatics[ \t]+[-0-9.]+[ \t]+\[mEh\].*$",
                         self.values[1], re.M)[-1]
        self.replace_text(row, row.replace("Electrostatics", "CORRUPT"), "printed components")
        self.reject_change(lambda v: v.__setitem__(3, ""), "missing Einsums profile")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--legacy-only", action="store_true")
    args = parser.parse_args()
    values = load_fixture(args.artifacts)
    suite = unittest.TestSuite()
    for cls in ([LegacyReplay] if args.legacy_only else [LegacyReplay, OwnedReplay]):
        cls.values = values
        suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(cls))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
