"""Local fake-file/command tests ONLY. No Psi4 import, GPU call or scheduler command."""
import copy
import ast
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

import campaign
import policy as p
import science as s

HERE = Path(__file__).resolve().parent


def manifest():
    files = {
        "devtools/saptdft_cpks_suite_run.py": dict(mode="100644", sha256=p.RUNNER_SHA),
        "devtools/cpks_gpu_build.sh": dict(mode="100644", sha256=p.BUILD_SHA),
    }
    return dict(schema=1, route=p.ROUTE, gates=p.GATES, run_root=p.RUN, systems=p.SYSTEMS,
                suite=p.H0+"/geometries", suite_sha256=p.SUITE_SHA, sequence=p.sequence(),
                scratch_parent="/storage/scratch1/3/awallace43/cpks-major2",
                arms={arm: dict(source=p.SOURCES[arm], pin=pin, backend="gpu",
                                build=p.SOURCES[arm]+"/build_major2_cuda",
                                files=files, files_digest=p.digest(files))
                      for arm, pin in [("baseline", p.BASE), ("candidate", p.OBSERVED_CANDIDATE)]},
                harness=dict(path=p.HARNESS, pin="a"*40, files=files, files_digest=p.digest(files)),
                dependency_hashes=p.LIBS, common_selectors=p.COMMON, fixed_selector=p.FIXED, fixed_k=p.FIXED_K,
                runtime_policy=dict(threads=8, memory="100 GiB", cache="enabled-shared-job-1GiB",
                                    prewarm=False, print_override=False, affinity_override=False),
                approval=dict(state="APPROVED", review_id="FAKE-UNIT-TEST", candidate_pin=p.OBSERVED_CANDIDATE))


def fixture():
    """Deliberately synthetic chemistry-shaped bytes, never scientific evidence."""
    root = "SAPT(DFT) Energy"
    stages = ["Dimer SCF", "Monomer A SCF", "Monomer B SCF", " Monomer A DFT", " Monomer B DFT"]
    timers = {}

    def add(path, wall):
        values = path.split(";")
        timers[path] = dict(wall_time=wall, user_time=0., system_time=0., n_calls=1,
                            timer_path=values, timer_name=values[-1], level=len(values)-1,
                            parent_id=";".join(values[:-1]) or None)
    add(root, 10.)
    for stage in stages:
        path = root + ";SAPT(DFT):" + stage
        add(path, 1.)
        add(path+";SCF: Native finalize", .1)
        add(path+";SCF: Native finalize;RHF: Lagrangian", .04)
        add(path+";SCF: Native finalize;RHF: Finalize cleanup", .06)
    add(root+";SAPT(DFT):Dimer SCF;SCF: JK native initialize", .1)
    add(root+";SAPT(DFT): Monomer A DFT;cuEST XC: CUDA LibXC Functional", .1)
    printed = "SAPT(DFT) Energy: 0.000u 0.000s 10.000w 1 calls\n"

    def emit(parent=None):
        lines = ""
        for path, v in timers.items():
            if v["parent_id"] == parent:
                lines += "| " * v["level"] + v["timer_name"] + f": 0.000u 0.000s {v['wall_time']:.3f}w 1 calls\n"
                lines += emit(path)
        return lines
    printed += emit()
    geometry = "H 0 0 0\nH 0 0 1\n"
    text = """cuESTJK: GPU-Accelerated
NVIDIA H200
Threads set to 8 by Python driver.
Memory set to 100.000 GiB by Python driver.
Spherical Harmonics?: true
Basis Set: 6-31+G**
Name: DEF2-UNIVERSAL-JKFIT
SAD J/K backend: cuESTJK
Geometry (in Angstrom), charge = 0, multiplicity = 1:
H 0 0 0 1
H 0 0 1 1
Running in synthetic-local-test mode
"""
    solver = """SAPT Coupled Induction Solver
Convergence = 1e-8
Time [s]
Guess .02 .03 0
1 .0001 .0002 1
2 1e-9 2e-9 1
----
"""
    proofs = []
    for i, stage in enumerate(s.STAGES):
        text += "SAPT(DFT): " + stage + "\n"
        typ = "DF-RHF" if i < 3 else "DF-RKS"
        for j in range(3 if i < 3 else 4):
            if i >= 3 and j == 2:
                text += "Near convergence on initial XC grid; continuing on final XC grid.\n"
            text += f"@{typ} iter {'SAD' if j == 0 else j}: 0"
            if i >= 3:
                text += " XC-INITIAL" if j < 2 else " XC-FINAL"
            text += "\n"
        text += f"Energy and wave function converged.\n@{typ} Final Energy: -1\n"
        proofs.append(dict(converged=True, nbf=250, iterations=2 if i < 3 else 3,
                           switched=i >= 3, functional="HF" if i < 3 else "PBE0",
                           initial=None if i < 3 else 2, final=None if i < 3 else 2,
                           final_grid_after=True))
        if i >= 3:
            text += "==> DFT Potential <==\nGRAC shift supplied by the user (not computed).\n"
            text += f"Monomer {'A' if i == 3 else 'B'} GRAC Shift 0.136000\n"
        if i in [2, 4]:
            text += solver
    for name, value in zip(["Electrostatics", "Exchange", "Induction", "Dispersion", "Total SAPT(DFT)"],
                           [-10, 20, -30, -40, -60]):
        text += f"  {name} {value:.9f} [mEh]\n"
    options = dict(SAPT_DFT_INDUCTION_TYPE="CPKS", CUEST_MIXED_PRECISION=False)
    data = dict(source_commit=p.OBSERVED_CANDIDATE, arm="gpu", threads=8, memory="100 GiB",
                system="peptide", options=copy.deepcopy(options), ok=True, gpu_builder_seen=True, sad_gpu_builder_seen=True,
                cpu_builder_seen=False, direct_seen=False, disk_df_builder_seen=False, nbf=250,
                nbf_monomer_a=125, nbf_monomer_b=125, proofs=proofs, cpks_solvers=s.solvers(text),
                grac_shifts_hartree=dict(A=.136, B=.136), wall_s=10.1, ru_maxrss_gib=1.,
                cpks_solve_calls_s=[.1, .2], cpks_solve_s=.3, components_hartree=dict(zip(s.COMPONENTS, [-.01, .02, -.03, -.04, -.06])),
                returned_energy_hartree=-.06, timer_records=timers)
    data.update(geometry_sha256=p.hashlib.sha256(geometry.encode()).hexdigest(), slurm_job_id="12345")
    expected = dict(source_commit=p.OBSERVED_CANDIDATE, arm="gpu", system="peptide",
                    geometry_sha256=data["geometry_sha256"], slurm_job_id="12345")
    return data, text, printed, "Timing information:\n 1ms : 1 calls : 1ms per call : Fake GEMM\n", expected, options, geometry


class HarnessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=os.environ["MAJOR2_TEST_TMP"])
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    @staticmethod
    def cleanup_fake_process(proc):
        if proc.poll() is None:
            children = Path(f"/proc/{proc.pid}/task/{proc.pid}/children")
            for child in children.read_text().split() if children.exists() else []:
                try:
                    os.killpg(int(child), signal.SIGKILL)
                except ProcessLookupError:
                    pass
            proc.kill()
        proc.communicate(timeout=5)

    def test_policy_manifest_and_pending(self):
        p.validate_manifest(manifest())
        m = manifest()
        m["approval"] = dict(state="PENDING", review_id="", candidate_pin=None)
        p.validate_manifest(m, executable=False)
        with self.assertRaisesRegex(RuntimeError, "NOT approved"):
            p.validate_manifest(m)

    def test_manifest_negative_controls(self):
        mutations = [
            lambda x: x["arms"]["candidate"].update(pin="saptdft_cuest_major2_gpu"),
            lambda x: x["arms"]["candidate"].update(pin="6171b59"),
            lambda x: x["arms"]["candidate"].update(backend="cpu"),
            lambda x: x["arms"]["candidate"].update(build=p.SOURCES["baseline"]+"/build_major2_cuda"),
            lambda x: x["approval"].update(candidate_pin=p.BASE),
            lambda x: x["harness"].update(pin="HEAD"),
            lambda x: x.update(run_root=p.H0),
            lambda x: x.update(suite_sha256="0"*64),
            lambda x: x["gates"].update(component_abs_eh=1e-5),
        ]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                m = copy.deepcopy(manifest())
                mutate(m)
                with self.assertRaises(RuntimeError):
                    p.validate_manifest(m)

    def test_24_unique_gpu_identifiers_and_balance(self):
        seq = p.sequence()
        self.assertEqual(len(seq), 24)
        self.assertEqual(len({x["label"] for x in seq}), 24)
        self.assertEqual(sum(x["arm"] == "baseline" for x in seq[::2]), 6)
        for system in p.SYSTEMS:
            for arm in ["baseline", "candidate"]:
                self.assertEqual(sorted(x["repeat"] for x in seq if x["system"] == system and x["arm"] == arm), [1, 2, 3])
        self.assertTrue(all(x["backend"] == "gpu" and not x["smoke"] for x in seq))
        for a, b in zip(seq[::2], seq[1::2]):
            self.assertEqual((a["system"], a["repeat"]), (b["system"], b["repeat"]))
            self.assertNotEqual(a["arm"], b["arm"])

    def test_fake_git_source_pins_cleanliness_and_file_manifest(self):
        subprocess.run(["git", "init", "-q", str(self.root)], check=True)
        f = self.root/"fake.txt"
        f.write_text("source\n")
        subprocess.run(["git", "-C", str(self.root), "add", "fake.txt"], check=True)
        subprocess.run(["git", "-C", str(self.root), "-c", "user.name=Local Test", "-c",
                        "user.email=local@example.invalid", "commit", "-qm", "fake fixture"], check=True)
        pin = p.git(self.root, "rev-parse", "HEAD")
        files = p.file_manifest(self.root)
        p.check_source(self.root, pin, files)
        with self.assertRaises(RuntimeError):
            p.check_source(self.root, "0"*40, files)
        f.write_text("dirty\n")
        with self.assertRaises(RuntimeError):
            p.check_source(self.root, pin, files)
        with self.assertRaises(RuntimeError):
            p.check_source(self.root, "HEAD")

    def test_environment_and_exact_helper_argv(self):
        env = p.runtime_env(dict(CPATH="/cuda12", GCC_ROOT="wrong", PYTHONPATH="/old",
                                 LD_LIBRARY_PATH="/old", OMP_PROC_BIND="inherited", PATH="/usr/bin"), self.root)
        self.assertNotIn("CPATH", env)
        self.assertNotIn("GCC_ROOT", env)
        self.assertNotIn("PYTHONPATH", env)
        self.assertNotIn("/old", env["LD_LIBRARY_PATH"])
        self.assertEqual(env["OMP_PROC_BIND"], "inherited")
        self.assertEqual(env["CPKS_CUDA_ARCHS"], "80;89;90;120")
        self.assertEqual(env["CPKS_BUILD_DIR"], "build_major2_cuda")
        for name in ["TMPDIR", "SCRATCH", "PSI_SCRATCH"]:
            self.assertEqual(env[name], str(self.root))
        argv = p.build_argv("/fake source ; not shell", p.BASE, "/fake output")
        self.assertEqual(argv, ["bash", "/fake source ; not shell/devtools/cpks_gpu_build.sh",
                                "/fake source ; not shell", p.E, p.BASE, "/fake output", "cuda"])
        self.assertEqual(env["OMP_NUM_THREADS"], "8")
        self.assertEqual(env["MKL_NUM_THREADS"], "8")

    def test_fake_cmake_cache_policy_and_cuda12_rejection(self):
        expected = {
            "CMAKE_BUILD_TYPE": "Release", "CMAKE_C_COMPILER": p.P+"/software/toolchains/conda-gcc",
            "CMAKE_CXX_COMPILER": p.P+"/software/toolchains/conda-g++", "CMAKE_CUDA_COMPILER": p.CUDA+"/bin/nvcc",
            "CMAKE_CUDA_ARCHITECTURES": "OFF", "CUDAToolkit_ROOT": p.CUDA, "Python_EXECUTABLE": p.E+"/bin/python",
            "Libxc_DIR": p.XC+"/lib64/cmake/Libxc", "Libint2_DIR": p.E+"/lib/cmake/libint2",
            "Einsums_DIR": p.P+"/gits/psi4.saptdft-cuest-lifecycle-20261003/build_lifecycle/stage/share/cmake/Einsums",
            "ENABLE_Einsums": "ON", "ENABLE_cuEST": "ON", "ENABLE_OPENMP": "ON", "ENABLE_dkh": "OFF",
            "ENABLE_XHOST": "OFF", "psi4_CMAKE_UNITY_BUILD": "ON", "LAPACK_LIBRARIES": p.E+"/lib/libmkl_rt.so",
            "CMAKE_C_FLAGS": "-I"+p.CUDA+"/targets/x86_64-linux/include -DPSI4_CUDA13_HEADER_ENV=1",
            "CMAKE_CXX_FLAGS": "-I"+p.CUDA+"/targets/x86_64-linux/include -DPSI4_CUDA13_HEADER_ENV=1",
            "CMAKE_CUDA_FLAGS": " ".join(f"-gencode=arch=compute_{a},code=sm_{a}" for a in [80, 89, 90, 120]),
        }
        outer = "\n".join(f"{k}:STRING={v}" for k, v in expected.items())
        inner = outer+"\nCMAKE_INSTALL_PREFIX:PATH=/fake-new-stage\nPSI4_CUEST_HAS_NONSYMMETRIC_XC:INTERNAL=1"
        p.cmake_policy(outer, inner, "/fake-new-stage")
        for o, i in [(outer.replace("Release", "Debug"), inner),
                     (outer, inner.replace("CUDA13_HEADER_ENV=1", "CUDA12_HEADER_ENV=1")),
                     (outer, inner.replace("sm_120", "sm_90")), (outer, inner.replace("/fake-new-stage", "/old-stage"))]:
            with self.assertRaises(RuntimeError):
                p.cmake_policy(o, i, "/fake-new-stage")

    def test_allocation_resource_route_fail_closed(self):
        text = ("Account=gts-cs207-chemx QOS=embers Partition=gpu-h200 NumCPUs=8 MinMemoryNode=128G "
                "NumNodes=1 NumTasks=1 TimeLimit=04:00:00 Features=H200 AllocTRES=cpu=8,mem=128G,gres/gpu=1,gres/gpu:h200=1")
        env = dict(SLURM_CPUS_PER_TASK="8", SLURM_JOB_ACCOUNT="gts-cs207-chemx", SLURM_JOB_QOS="embers",
                   SLURM_JOB_NUM_NODES="1", SLURM_JOB_START_TIME="1000", SLURM_JOB_END_TIME="15400")
        campaign.allocation(text, env)
        for old, new in [("QOS=embers", "QOS=inferno"), ("NumCPUs=8", "NumCPUs=12"),
                         ("Features=H200", "Features=H100"), ("128G", "64G"), ("gpu-h200", "gpu-h100")]:
            with self.assertRaises(RuntimeError):
                campaign.allocation(text.replace(old, new), env)
        campaign.gpu_line("NVIDIA H200, GPU-abcdef, 143771, fake-driver")
        for bad in ["NVIDIA H100, GPU-abcdef, 143771, fake-driver", "NVIDIA H200, GPU-abcdef, 143771, fake-driver\n"*2]:
            with self.assertRaises(RuntimeError):
                campaign.gpu_line(bad)

    def test_time_admission(self):
        p.admission(14400, 12600)
        p.admission(1500, 1200)
        for left in [1499, -1, float("nan"), float("inf")]:
            with self.assertRaises(RuntimeError):
                p.admission(left, 1200)

    def test_rhf_tolerance_ast_drift_rejected_before_job(self):
        text = "\n".join(f"np.testing.assert_allclose(a,b,atol={a},rtol={r})"
                         for a, r in [[2e-13, 2e-13], [2e-13, 2e-13], [2e-13, 0], [1e-12, 0], [1e-10, 0]])
        self.assertEqual(len(p.rhf_tolerances(ast.parse(text))), 5)
        with self.assertRaisesRegex(RuntimeError, "tolerance policy changed"):
            p.rhf_tolerances(ast.parse(text.replace("atol=2e-13", "atol=1e-5", 1)))

    def junit(self, content):
        f = self.root/"junit.xml"
        f.write_text("<testsuites><testsuite>" + content + "</testsuite></testsuites>")
        return f

    def test_expected_red_specific_assertion_only(self):
        name = "test_cuest_unchanged_initialize_reuses_plan"
        collected = ["fake.py::"+name]
        file = self.junit(f'<testcase name="{name}"><failure>assert _creations(trace) == 1\nE assert 2 == 1</failure></testcase>')
        p.check_junit(file, collected, 1, expected_red=True)
        for content, rc in [
            (f'<testcase name="{name}"><skipped message="no GPU"/></testcase>', 0),
            (f'<testcase name="{name}"><error>ImportError</error></testcase>', 1),
            (f'<testcase name="{name}"><failure>no release_freed_memory binding</failure></testcase>', 1),
            (f'<testcase name="{name}"><failure>assert 0 == 1</failure></testcase>', 1),
            (f'<testcase name="{name}"/>', 0),
        ]:
            with self.subTest(content=content):
                with self.assertRaises(RuntimeError):
                    p.check_junit(self.junit(content), collected, rc, expected_red=True)

    def test_junit_green_requires_all_executed_cases(self):
        f = self.junit('<testcase name="test_fake[a]"/><testcase name="test_fake[b]"/>')
        p.check_junit(f, ["fake.py::test_fake[a]", "fake.py::test_fake[b]"], 0)
        for collected, rc in [([], 0), (["fake.py::test_fake[a]"], 0),
                              (["fake.py::test_fake[a]", "fake.py::test_fake[b]"], 1)]:
            with self.assertRaises(RuntimeError):
                p.check_junit(f, collected, rc)
        with self.assertRaises(RuntimeError):
            p.check_junit(self.junit('<testcase name="test_fake[a]"><skipped/></testcase>'), ["fake.py::test_fake[a]"], 0)

    def test_synthetic_record_positive(self):
        values = fixture()
        verdict = s.validate_record(*values, candidate=True)
        self.assertIn("UNAVAILABLE", verdict["plan_counts"])

    def test_synthetic_record_negative_data_controls(self):
        changes = [
            lambda d: d.update(source_commit=p.BASE),
            lambda d: d.update(threads=24),
            lambda d: d.update(memory="200 GiB"),
            lambda d: d.update(arm="cpu"),
            lambda d: d.update(geometry_sha256="0"*64),
            lambda d: d.update(slurm_job_id="54321"),
            lambda d: d["options"].update(SAPT_DFT_INDUCTION_TYPE="NONE"),
            lambda d: d.update(cpu_builder_seen=True),
            lambda d: d["proofs"][3].update(final=1),
            lambda d: d["proofs"][4].update(final_grid_after=False),
            lambda d: d["proofs"][2].update(converged=False),
            lambda d: d["cpks_solvers"][0]["residuals"][-1].__setitem__(0, 1e-7),
            lambda d: d["components_hartree"].update({"SAPT TOTAL ENERGY": float("nan")}),
            lambda d: d["components_hartree"].update({"SAPT IND ENERGY": .5}),
            lambda d: d["timer_records"].pop("SAPT(DFT) Energy;SAPT(DFT):Dimer SCF"),
            lambda d: d.update(wall_s=float("nan")),
        ]
        for change in changes:
            values = list(fixture())
            change(values[0])
            with self.subTest(change=change):
                with self.assertRaises(RuntimeError):
                    s.validate_record(*values, candidate=True)

    def test_synthetic_chronology_backend_geometry_solver_negatives(self):
        replacements = [
            ("XC-FINAL", "XC-INITIAL"),
            ("@DF-RKS iter 3:", "@DF-RKS iter 1:"),
            ("NVIDIA H200", "NVIDIA H100"),
            ("H 0 0 1 1", "H 0 0 2 1"),
            ("2 1e-9 2e-9", "3 1e-9 2e-9"),
            ("1e-8", "1e-6"),
            ("Name: DEF2-UNIVERSAL-JKFIT", "Name: OTHER"),
            ("Threads set to 8", "Threads set to 2"),
            ("Monomer A GRAC Shift 0.136000", "Monomer A GRAC Shift 0.100000"),
        ]
        for old, new in replacements:
            values = list(fixture())
            values[1] = values[1].replace(old, new)
            with self.subTest(old=old):
                with self.assertRaises(RuntimeError):
                    s.validate_record(*values, candidate=True)

    def test_candidate_timers_required_without_invented_counts(self):
        values = list(fixture())
        name = next(x for x in values[0]["timer_records"] if x.endswith(";RHF: Lagrangian"))
        del values[0]["timer_records"][name]
        with self.assertRaisesRegex(RuntimeError, "RHF"):
            s.validate_record(*values, candidate=True)
        values = list(fixture())
        values[2] = values[2].replace("0.040w", "0.080w")
        with self.assertRaises(RuntimeError):
            s.validate_record(*values, candidate=True)

    def test_paired_and_repeat_numerical_iteration_gates(self):
        a = fixture()[0]
        s.compare(a, copy.deepcopy(a))
        b = copy.deepcopy(a)
        b["components_hartree"]["SAPT ELST ENERGY"] += 1.01e-8
        with self.assertRaisesRegex(RuntimeError, "energy difference"):
            s.compare(a, b)
        for owner in ["proofs", "cpks_solvers"]:
            b = copy.deepcopy(a)
            b[owner][0]["iterations"] += 1
            with self.assertRaisesRegex(RuntimeError, "STOP_FOR_REVIEW"):
                s.compare(a, b)
        records = []
        for row in p.sequence():
            d = copy.deepcopy(a)
            d["wall_s"] = (10 if row["arm"] == "baseline" else 8) + row["repeat"]-2
            records.append(dict(row, data=d))
        result = s.summary(records)
        self.assertAlmostEqual(result["systems"]["peptide"]["improvement_percent"], 20.)
        self.assertEqual(len(result["systems"]["peptide"]["adjacent_pair_ratios"]), 3)
        self.assertIsNone(result["cpu_gpu_ratio"])
        with self.assertRaises(RuntimeError):
            s.summary(records[:-1])
        records[-1]["data"]["components_hartree"]["SAPT TOTAL ENERGY"] += 2e-8
        with self.assertRaises(RuntimeError):
            s.summary(records)

    def fake_engine(self):
        engine = object.__new__(campaign.Engine)
        engine.root = self.root
        engine.raw = self.root/"raw"
        engine.raw.mkdir()
        engine.scratch = self.root
        engine.env = dict(os.environ)
        engine.end = time.time()+1000
        engine.signalled = False
        engine.active = None
        return engine

    def test_fake_command_argv_exit_and_partial_preservation(self):
        engine = self.fake_engine()
        engine.run_command([sys.executable, "-c", "import sys; print(sys.argv[1])", "a b;no-shell"], "fake")
        self.assertEqual((engine.raw/"fake.log").read_text().strip(), "a b;no-shell")
        with self.assertRaises(RuntimeError):
            engine.run_command([sys.executable, "-c", "print('partial',flush=True); raise SystemExit(7)"], "bad")
        self.assertIn("partial", (engine.raw/"bad.log").read_text())
        self.assertEqual(p.read_json(engine.raw/"bad.exit.json")["exit"], 7)
        with self.assertRaises(FileExistsError):
            engine.run_command([sys.executable, "-c", "pass"], "bad")

    def test_fake_command_timeout_preserves_partial(self):
        engine = self.fake_engine()
        with self.assertRaises(subprocess.TimeoutExpired):
            engine.run_command([sys.executable, "-c", "import signal;print('partial-timeout',flush=True);signal.pause()"],
                               "timeout", timeout=.15)
        self.assertIn("partial-timeout", (engine.raw/"timeout.log").read_text())
        self.assertEqual(p.read_json(engine.raw/"timeout.exit.json")["interrupted"], "TimeoutExpired")
        self.assertIsNone(engine.active)

    def test_fake_pytest_plugin_executes_and_rejects_skip_and_collection_drift(self):
        config = self.root/"fake.ini"
        config.write_text("[pytest]\n")
        test = self.root/"test_fake.py"
        test.write_text("def test_ok():\n assert 1==1\n")
        env = dict(os.environ, PYTHONPATH=str(HERE))
        base = [sys.executable, "-B", "-m", "pytest", "-c", str(config), "--noconftest",
                "-q", "-p", "no:cacheprovider", "-p", "pytest_evidence", str(test)]
        for mode in ["pass", "skip", "drift"]:
            out = self.root/mode
            out.mkdir()
            if mode == "skip":
                test.write_text("import pytest\ndef test_ok():\n pytest.skip('fake absent GPU')\n")
            env.update(MAJOR2_PYTEST_EVIDENCE=str(out/"execution.json"), MAJOR2_EXPECTED_COUNT="2" if mode == "drift" else "1")
            proc = subprocess.run(base+["--junitxml="+str(out/"junit.xml")], cwd=self.root, env=env,
                                  capture_output=True, text=True, timeout=20)
            evidence = p.read_json(out/"execution.json")
            if mode == "pass":
                self.assertEqual(proc.returncode, 0, proc.stderr)
                p.check_junit(out/"junit.xml", evidence["collected"], proc.returncode)
                self.assertEqual(next(iter(evidence["phases"].values()))["call"], "passed")
            elif mode == "skip":
                self.assertEqual(proc.returncode, 0)
                with self.assertRaises(RuntimeError):
                    p.check_junit(out/"junit.xml", evidence["collected"], proc.returncode)
            else:
                self.assertNotEqual(proc.returncode, 0)
                self.assertFalse(evidence["phases"])

    def test_probe_rejects_fake_wrong_arm_before_chemistry(self):
        package = self.root/"psi4"
        package.mkdir()
        (package/"__init__.py").write_text("from . import core\n")
        (package/"core.py").write_text("release_freed_memory = None\n")
        env = dict(os.environ, PYTHONPATH=str(self.root), MAJOR2_EXPECTED_STAGE=str(self.root/"other-stage"))
        proc = subprocess.run([sys.executable, "-B", str(HERE/"probe.py"), str(self.root/"probe.json")],
                              cwd=self.root, env=env, capture_output=True, text=True, timeout=15)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("wrong arm/package import", proc.stderr)
        self.assertFalse((self.root/"probe.json").exists())

    def test_fake_raw_inventory_covers_builds_rechecked(self):
        engine = self.fake_engine()
        (engine.raw/"fake.log").write_text("partial")
        (engine.root/"build-baseline").mkdir()
        (engine.root/"build-baseline/BUILD_OK").write_text("FAKE")
        engine.inventory()
        inventory = p.read_json(self.root/"raw-inventory.json")
        self.assertEqual(set(inventory), {"raw/fake.log", "build-baseline/BUILD_OK"})
        self.assertTrue(p.read_json(self.root/"raw-inventory-recheck.json")["verified"])

    def test_batch_directives_no_submission_no_requeue(self):
        text = campaign.batch_text("a"*64)
        file = self.root/"fake.sbatch"
        file.write_text(text)
        subprocess.run(["bash", "-n", str(file)], check=True)
        for directive in ["--qos=embers", "--account=gts-cs207-chemx", "--gres=gpu:h200:1",
                          "--constraint=H200", "--mem=128G", "--cpus-per-task=8", "--time=04:00:00", "--no-requeue"]:
            self.assertIn("#SBATCH "+directive, text)
        self.assertNotIn("sbatch ", text)
        self.assertIn(p.HARNESS+"/devtools/major2/supervise.sh", text)

    def test_term_forwarding_no_false_completed_preserves_partial(self):
        code = ("import signal,sys,pathlib,time\n"
                "p=pathlib.Path(sys.argv[1]);(p/'partial').write_text('keep')\n"
                "def stop(*a):\n (p/'forwarded').write_text('TERM');raise SystemExit(143)\n"
                "signal.signal(signal.SIGTERM,stop)\nprint('READY',flush=True)\n"
                # Avoid signal.pause's check-to-syscall lost-wakeup window in
                # Python's deferred handler dispatch. This is only a dummy wait.
                "while True: time.sleep(.01)\n")
        proc = subprocess.Popen(["bash", str(HERE/"supervise.sh"), str(self.root),
                                 sys.executable, "-u", "-c", code, str(self.root)],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.addCleanup(self.cleanup_fake_process, proc)
        self.assertEqual(proc.stdout.readline().strip(), "READY")
        proc.send_signal(signal.SIGTERM)
        out, err = proc.communicate(timeout=10)
        self.assertEqual(proc.returncode, 143, (out, err))
        self.assertTrue((self.root/"partial").exists())
        self.assertTrue((self.root/"forwarded").exists())
        self.assertTrue((self.root/"FAILED").exists())
        self.assertFalse((self.root/"COMPLETED").exists())
        self.assertIn("signalled=1", (self.root/"batch-exit.txt").read_text())

    def test_term_before_child_pid_registration_deterministic(self):
        fifo = self.root/"ready.fifo"
        os.mkfifo(fifo)
        bash_env = self.root/"fake-bash-env.sh"
        # DEBUG hook freezes the exact launch-to-$! registration window.
        # FIFO guarantees the payload installed TERM handling and saved partials.
        bash_env.write_text(
            "trap 'if [[ $BASH_COMMAND == \"child=\\$!\" ]]; then "
            "read -r ready < \"$MAJOR2_TEST_FIFO\"; kill -TERM \"$$\"; fi' DEBUG\n")
        code = ("import signal,sys,pathlib,time\np=pathlib.Path(sys.argv[1])\n"
                "def stop(*a):\n (p/'forwarded').write_text('TERM');raise SystemExit(143)\n"
                "signal.signal(signal.SIGTERM,stop)\n(p/'partial').write_text('keep')\n"
                "with (p/'ready.fifo').open('w') as f:f.write('READY\\n')\n"
                "while True: time.sleep(.01)\n")
        env = dict(os.environ, BASH_ENV=str(bash_env), MAJOR2_TEST_FIFO=str(fifo))
        proc = subprocess.Popen(["bash", str(HERE/"supervise.sh"), str(self.root),
                                 sys.executable, "-u", "-c", code, str(self.root)], env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.addCleanup(self.cleanup_fake_process, proc)
        out, err = proc.communicate(timeout=2)
        self.assertEqual(proc.returncode, 143, (out, err))
        self.assertTrue((self.root/"partial").exists())
        self.assertTrue((self.root/"forwarded").exists())
        self.assertFalse((self.root/"COMPLETED").exists())
        self.assertTrue((self.root/"FAILED").exists())

    def test_supervisor_failed_and_success_payloads(self):
        for code in [7, 0]:
            directory = self.root/str(code)
            directory.mkdir()
            proc = subprocess.run(["bash", str(HERE/"supervise.sh"), str(directory),
                                   sys.executable, "-c", f"raise SystemExit({code})"], capture_output=True)
            self.assertEqual(proc.returncode, code)
            self.assertEqual((directory/"COMPLETED").exists(), code == 0)
            self.assertEqual((directory/"FAILED").exists(), code != 0)
        # A second attempt cannot leave a false new success or overwrite old evidence.
        old = (directory/"COMPLETED").read_bytes()
        retry = subprocess.run(["bash", str(HERE/"supervise.sh"), str(directory), "/usr/bin/false"], capture_output=True)
        self.assertNotEqual(retry.returncode, 0)
        self.assertEqual((directory/"COMPLETED").read_bytes(), old)

    def test_no_network_build_filter_inherited_by_fake_child(self):
        code = ("import socket,errno,pathlib,sys\n"
                "pathlib.Path(sys.argv[1]).write_text('local IO allowed')\n"
                "try: socket.socket()\n"
                "except PermissionError as e:\n assert e.errno==errno.EPERM;print('DENIED')\n"
                "else: raise RuntimeError('network not denied')\n")
        result = subprocess.run([sys.executable, "-B", str(HERE/"no_network.py"),
                                 sys.executable, "-c", code, str(self.root/"local")],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "DENIED")
        self.assertTrue((self.root/"local").exists())

    def test_sampler_fake_supported_optional_fields_and_actual_pid(self):
        executable = self.root/"nvidia-smi"
        executable.write_text("#!"+sys.executable+"\n"
            "import os,sys,signal\n"
            "if '--help-query-gpu' in sys.argv: print('\"power.draw\"');raise SystemExit()\n"
            "if any(x.startswith('--query-compute-apps=') for x in sys.argv):\n"
            " print('GPU-fake, 123, fake-chemistry, 20',flush=True);os.kill(os.getppid(),signal.SIGTERM)\n"
            "else:\n"
            " fields=next(x.split('=',1)[1].split(',') for x in sys.argv if x.startswith('--query-gpu='))\n"
            " if os.getenv('FAKE_OPTIONAL_ERROR') and 'power.draw' in fields and len(fields)>1:\n"
            "  print('optional field unsupported',file=sys.stderr);raise SystemExit(1)\n"
            " print(', '.join(os.getenv('FAKE_UUID','GPU-fake') if x=='uuid' else '20' for x in fields))\n")
        executable.chmod(0o700)
        env = dict(os.environ, PATH=str(self.root)+":"+os.environ["PATH"])
        out = self.root/"telemetry.jsonl"
        result = subprocess.run([sys.executable, "-B", str(HERE/"sample.py"), str(out), "GPU-fake"],
                                env=env, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        rows = [json.loads(line) for line in out.read_text().splitlines()]
        self.assertIn("clocks.sm", rows[0]["unsupported_optional"])
        self.assertIn("power.draw", rows[0]["fields"])
        self.assertEqual(rows[1]["compute_apps"][0]["pid"], "123")
        self.assertEqual(rows[1]["values"]["uuid"], "GPU-fake")
        fallback = self.root/"fallback.jsonl"
        result = subprocess.run([sys.executable, "-B", str(HERE/"sample.py"), str(fallback), "GPU-fake"],
                                env=dict(env, FAKE_OPTIONAL_ERROR="1"), capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        last = json.loads(fallback.read_text().splitlines()[-1])
        self.assertIn("unsupported", last["optional_query_error"])
        wrong = subprocess.run([sys.executable, "-B", str(HERE/"sample.py"), str(self.root/"wrong.jsonl"), "GPU-fake"],
                               env=dict(env, FAKE_UUID="GPU-other"), capture_output=True, text=True, timeout=10)
        self.assertNotEqual(wrong.returncode, 0)
        self.assertIn("UUID changed", wrong.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
