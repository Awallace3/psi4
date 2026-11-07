"""Pure stdlib tests: no Psi4 imports, calculations, network, or cluster tools.

Run: python3 -I -B devtools/isapol_native_profile/test_profile.py
Optional readonly real fixtures: PROFILE_FIXTURE_ROOT=/path/to/native-perf-inputs001
"""
import copy
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock
import sys
import subprocess
import atexit


ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("native_profile_under_test", ROOT / "profile.py")
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)
t = profile.telemetry


def cwd_probe(root, mode, failure):
    """Subprocess-only: exercise real run() to import, using fake Psi4 modules."""
    root = Path(root)
    scratch, output = root / "scratch", root / "output"
    fixture, build, qualification, qce = (root / name for name in ("fixture", "build", "qual", "qce"))
    for directory in (fixture, build, qualification, qce):
        directory.mkdir()
    claimed = {"function_kwargs": {"input_data": {"schema_name": "qcschema_input", "schema_version": 1}}}
    (fixture / "claimed-input.json").write_text(json.dumps(claimed))
    (build / "build-identity.json").write_text("{}")
    (qualification / "installed-validation.json").write_text("{}")
    request = {"test": "fake request, NEVER a scientific fixture"}
    data = dict(molecule={}, specification={"keywords": {"function_kwargs": {"native_monomer": request}}})
    metadata = dict(input_sha256="pure-test", monomer_id="monomer-0001", science_memory_gib=20)
    imported = dict(package=str(build / "env/psi4/__init__.py"), core=str(build / "env/psi4/core.so"))
    core = types.SimpleNamespace(__file__=imported["core"], set_num_threads=lambda *a, **k: None,
                                 IOManager=types.SimpleNamespace(shared_object=lambda: types.SimpleNamespace(
                                     set_default_path=lambda *a: None)))
    psi = types.ModuleType("psi4")
    psi.__file__, psi.__path__, psi.core = imported["package"], [], core
    psi.set_memory = lambda *a, **k: None
    schema = types.ModuleType("psi4.schema_wrapper")
    schema._read_output = lambda path: "fake output"
    raw = b"mock transport, no quantum data"
    import base64
    checksum = hashlib.sha256(raw).hexdigest()
    envelope = dict(request=request, artifacts={"tensors.npz": dict(
        data=base64.b64encode(raw).decode(), size=len(raw), sha256=checksum)},
        result=dict(tensors_sha256=checksum, resources={},
                    provenance=dict(response="reference_h2h1", partition={})))
    def fake_schema(*args, **kwargs):
        if failure == "run":
            raise RuntimeError("mock scientific failure")
        Path("psi.fake.clean").write_text("fake runtime-relative write")
        return types.SimpleNamespace(model_dump_json=lambda: json.dumps(dict(
            success=True, extras={"psi4:native_monomer": envelope})))
    schema.run_qcschema = fake_schema
    psi.schema_wrapper = schema
    driver, routing = types.ModuleType("psi4.driver"), types.ModuleType("psi4.driver.procrouting")
    driver.__path__, routing.__path__ = [], []
    isa = types.ModuleType("psi4.driver.procrouting.isapol_schema")
    isa.implementation_identity = lambda: {"sha256": profile.IMPLEMENTATION}
    isa.admit_native_schema = lambda *a, **k: None
    native = types.ModuleType("psi4.driver.procrouting.isapol_native_partition")
    native.DRHO_REFINEMENT_ITERATIONS = 32
    psi.driver, driver.procrouting = driver, routing
    routing.isapol_schema, routing.isapol_native_partition = isa, native
    modules = {"psi4": psi, "psi4.schema_wrapper": schema, "psi4.driver": driver,
               "psi4.driver.procrouting": routing,
               "psi4.driver.procrouting.isapol_schema": isa,
               "psi4.driver.procrouting.isapol_native_partition": native}
    args = types.SimpleNamespace(
        fixtures=str(fixture), build_manifest=str(build / "build-identity.json"),
        qualification=str(qualification / "installed-validation.json"), qcengine_package=str(qce),
        qcengine_harness_sha256="pure-test", output=str(output), scratch=str(scratch),
        sample_seconds=.1, threads=1, run=mode == "run", instrumentation="off")
    # Guard bypasses are ONLY stdlib mocks inside this pure subprocess test.
    environment = dict(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                       SLURM_JOB_ID="fake-test-not-a-real-job", SLURM_CPUS_PER_TASK="1")
    real_import = __import__
    reached = False
    def intercept(name, *a, **k):
        nonlocal reached
        if name == "psi4" and not reached:
            reached = True
            assert Path.cwd() == scratch / "cwd", "actual pre-import path did not isolate CWD"
            assert (Path.cwd() / ".owned-profile-cwd").is_file()
            (root / "import-reached.json").write_text(json.dumps(dict(cwd=os.getcwd(), mode=mode)))
            # Simulate core finalization AFTER run() returns/raises and mock scopes unwind.
            def finalize():
                with Path("timer.dat").open("a") as stream:
                    stream.write("fake core finalizer\n")
                Path("psi.finalizer.clean").write_text("fake finalizer")
            atexit.register(finalize)
            if failure == "import":
                raise RuntimeError("mock partial import failure")
        return real_import(name, *a, **k)
    with mock.patch.dict(sys.modules, modules), mock.patch.dict(os.environ, environment), \
         mock.patch.object(sys, "prefix", str(build / "env")), \
         mock.patch.object(sys, "pycache_prefix", str(scratch / "cache")), \
         mock.patch.object(profile, "read_fixtures", return_value=(data, envelope, metadata)), \
         mock.patch.object(profile, "verify_receipts", return_value=({}, imported)), \
         mock.patch.object(profile, "source_audit", return_value={"recognized_qcengine_wrapper_timer": True}), \
         mock.patch.object(profile.shutil, "which", return_value=str(build / "env/bin/psi4")), \
         mock.patch.object(profile, "thread_snapshot", return_value={"process_threads": 1}), \
         mock.patch("builtins.__import__", side_effect=intercept):
        profile.run(args)
    assert reached
    assert Path.cwd() == scratch / "cwd", "CWD restored before deferred core finalization"


class ProfileTests(unittest.TestCase):
    def test_real_preimport_path_keeps_check_run_and_failure_finalizers_owned(self):
        for mode, failure in (("check", ""), ("run", ""), ("check", "import"),
                              ("run", "import"), ("run", "run")):
            with self.subTest(mode=mode, failure=failure), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                protected = root / "protected"
                protected.mkdir()
                (protected / "sentinel").write_text("unchanged")
                result = subprocess.run(
                    [sys.executable, "-I", "-B", str(Path(__file__).resolve()),
                     "--cwd-probe", str(root), mode, failure],
                    cwd=protected, text=True, capture_output=True, timeout=15)
                self.assertEqual(result.returncode == 0, not bool(failure), result.stderr)
                self.assertEqual(sorted(p.name for p in protected.iterdir()), ["sentinel"])
                self.assertEqual((protected / "sentinel").read_text(), "unchanged")
                reached = json.loads((root / "import-reached.json").read_text())
                self.assertEqual(reached["cwd"], str(root / "scratch/cwd"))
                self.assertTrue((root / "scratch/cwd/timer.dat").exists())
                self.assertTrue((root / "scratch/cwd/psi.finalizer.clean").exists())
                if mode == "check" and not failure:
                    self.assertEqual(json.loads((root / "output/manifest.json").read_text())["status"],
                                     "ADMITTED_NOT_EXECUTED")
                if mode == "run" and failure != "import":
                    status = json.loads((root / "output/summary.json").read_text())["status"]
                    self.assertEqual(status, "FAILED_OR_INTERRUPTED" if failure else "COMPUTED_NOT_ACCEPTED")
    def test_pool_fallback_queries_only_already_loaded_readonly_symbols(self):
        getter = lambda: 4
        class Library:
            def __getattr__(self, name):
                if name == "openblas_get_num_threads":
                    return getter
                raise AttributeError(name)
        original = Path.read_text
        def text(path, *args, **kwargs):
            if str(path) == "/proc/self/maps":
                return "0000-1000 r-xp 0000 00:00 0 /fake/libopenblas.so\n"
            return original(path, *args, **kwargs)
        fake = types.SimpleNamespace(core=types.SimpleNamespace(
            get_num_threads=lambda: 4, _isa_blas_max_threads=lambda: 0))
        with mock.patch.object(Path, "read_text", text), mock.patch("ctypes.CDLL", return_value=Library()) as cdll:
            snapshot = profile.thread_snapshot(fake)
        cdll.assert_called_once_with("/fake/libopenblas.so", mode=os.RTLD_NOLOAD | os.RTLD_LOCAL)
        self.assertEqual(snapshot["readonly_pool_queries"][0]["queries"],
                         {"openblas_get_num_threads": 4})

    def test_probe_layout_matches_frozen_source_without_importing_science(self):
        routing = ROOT.parents[1] / "psi4/driver/procrouting"
        for filename, function, expected in (
            ("isapol_native_partition.py", "native_partition",
             dict(drho=1, density=1, samples=1, controller=2, declared=1,
                  initial=1, trajectory=1, provenance=1)),
            ("isapol_pfit_stream.py", "fitted_point_targets", dict(provider=1, packed=1)),
        ):
            tree = ast.parse((routing / filename).read_text())
            target = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function)
            counts = {}
            for node in ast.walk(target):
                if isinstance(node, ast.Assign):
                    for name in node.targets:
                        if isinstance(name, ast.Name) and name.id in expected:
                            counts[name.id] = counts.get(name.id, 0) + 1
            self.assertEqual(counts, expected)
    def test_line_probes_preserve_results_exceptions_and_trace_restore(self):
        def work(fail=False):
            first = object()
            middle = first
            if fail:
                raise ValueError("scientific exception")
            first = middle
            stop = None
            return first
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            recorder = t.Recorder(path)
            before = sys.gettrace()
            probes = dict(first=("first operation", "second operation"), middle=None, stop=None)
            with t.line_intervals(work, recorder, probes,
                                  {"middle": lambda local: {"first_present": "first" in local}}):
                self.assertIsNotNone(work())
            self.assertIs(sys.gettrace(), before)
            with self.assertRaisesRegex(ValueError, "scientific exception"):
                with t.line_intervals(work, recorder, probes):
                    work(True)
            self.assertIs(sys.gettrace(), before)
            recorder.close()
            events = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertTrue(any(e.get("name") == "second operation" for e in events))
            self.assertTrue(any(e.get("values") == {"first_present": True} for e in events))
            with self.assertRaisesRegex(RuntimeError, "layout differs"):
                with t.line_intervals(work, recorder, {"missing": None}):
                    work()
    def test_input_translation_preserves_science_protocols_and_ownership(self):
        raw = dict(molecule=dict(symbols=["H"], geometry=[0, 0, 0]),
                   specification=dict(driver="properties", model={"method": "pbe0"},
                                      keywords={"x": [1, 2]}, protocols={"stdout": False}))
        before = copy.deepcopy(raw)
        normalized = profile.canonical_input(raw)
        self.assertEqual(raw, before)
        self.assertEqual(normalized["specification"], raw["specification"])
        normalized["specification"]["keywords"]["x"][0] = 9
        self.assertEqual(raw, before)
        flat = dict(schema_name="qcschema_input", schema_version=1, **raw["specification"],
                    molecule=raw["molecule"])
        self.assertEqual(profile.canonical_input(flat)["specification"], raw["specification"])
        with self.assertRaisesRegex(ValueError, "nested input"):
            profile.canonical_input(dict(raw, schema_version=1))
        with self.assertRaisesRegex(ValueError, "campaign receipt"):
            profile.canonical_input(dict(raw, dataset_id=357))

    def test_case_validation_rejects_fdds_identity_and_held_monomer(self):
        request = dict(implementation_sha256=profile.IMPLEMENTATION,
                       resources={"max_bytes": 12 * 1024**3})
        raw = dict(molecule={}, specification=dict(
            driver="properties", model=dict(method="pbe0", basis="aug-cc-pvtz"),
            keywords={"function_kwargs": {"native_monomer": request}}))
        envelope = dict(schema="native-monomer-artifacts/2", request=request,
                        implementation={"sha256": profile.IMPLEMENTATION},
                        result={"provenance": {"response": "reference_h2h1"}})
        meta = dict(schema="native-profile-case/1", monomer_id="monomer-0001", science_memory_gib=20)
        profile.validate_case(raw, envelope, meta)
        for key, value in (("monomer_id", "monomer-0026"), ("science_memory_gib", 60)):
            with self.assertRaises(ValueError):
                profile.validate_case(raw, envelope, dict(meta, **{key: value}))
        bad = copy.deepcopy(envelope)
        bad["result"]["provenance"]["response"] = "native_fdds"
        with self.assertRaisesRegex(ValueError, "H2H1"):
            profile.validate_case(raw, bad, meta)
        bad = copy.deepcopy(raw)
        bad["specification"]["keywords"]["function_kwargs"]["native_monomer"]["implementation_sha256"]["core"] = "0"*64
        with self.assertRaises(ValueError):
            profile.validate_case(bad, envelope, meta)

    def test_spans_nested_failure_returns_and_patch_restoration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            recorder = t.Recorder(path)
            patches = t.Patches()
            sentinel = object()
            module = types.SimpleNamespace(__name__="fake", f=lambda value: value)
            original = module.f
            patches.wrap(module, "f", recorder)
            mapping = {"driver": original}
            patches.mapping(mapping, "driver", module.f, "dispatch")
            try:
                with recorder.span("outer"):
                    self.assertIs(mapping["driver"](sentinel), sentinel)
                    with self.assertRaisesRegex(RuntimeError, "original"):
                        with recorder.span("bad"):
                            raise RuntimeError("original")
            finally:
                patches.restore()
                recorder.close()
            self.assertIs(module.f, original)
            self.assertIs(mapping["driver"], original)
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            outer = next(r for r in rows if r.get("name") == "outer" and r["event"] == "begin")
            inner = next(r for r in rows if r.get("name") == "fake.f" and r["event"] == "begin")
            self.assertEqual(inner["parent"], outer["id"])
            self.assertTrue(any(r.get("status") == "failed" for r in rows))
            self.assertEqual(recorder.stack, [])

    def test_silent_stages_capture_without_changing_original_calls(self):
        class Log:
            def __init__(self):
                self.calls = []
            def stage(self, name, parameters=(), level=1):
                self.stage_end(level=level)
                self.calls.append(("stage", name, parameters, level))
                return "stage-return"
            def stage_end(self, level=1):
                self.calls.append(("end", level))
            def stage_failed(self, error, level=1):
                self.calls.append(("failed", type(error).__name__, level))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            original = Log.stage
            recorder, patches = t.Recorder(path), t.Patches()
            t.install_stages(patches, Log, recorder)
            log = Log()
            self.assertEqual(log.stage("ISA", [("x", 2)], level=0), "stage-return")
            log.stage("response")
            log.stage_failed(ValueError("failure"))
            log.stage("unfinished")
            patches.restore()
            recorder.close()
            self.assertIs(Log.stage, original)
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            ends = [row for row in rows if row["event"] == "stage_end"]
            self.assertEqual([(r["name"], r["status"]) for r in ends],
                             [("ISA", "complete"), ("response", "failed"), ("unfinished", "unclosed")])
            self.assertIn(("stage", "ISA", [("x", 2)], 0), log.calls)

    def test_scratch_never_follows_links_or_deletes_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "scratch"
            root.mkdir()
            (root / "a").write_bytes(b"12345")
            external = Path(directory) / "external"
            external.mkdir()
            (external / "not-counted").write_bytes(b"0"*100)
            (root / "link").symlink_to(external, target_is_directory=True)
            snapshot = t.scratch_snapshot(root)
            self.assertEqual(snapshot["logical_bytes"], 5)
            self.assertEqual(snapshot["files"], 1)
            self.assertEqual(snapshot["symlinks"], 1)
            sampler = t.Sampler(root, Path(directory) / "samples.jsonl", 1)
            sampler.start()
            result = sampler.stop()
            self.assertGreaterEqual(result["samples"], 2)
            self.assertFalse(result["errors"])
            self.assertEqual((external / "not-counted").stat().st_size, 100)
            self.assertTrue((root / "a").exists())

    def test_sampler_open_failure_is_not_silent(self):
        with tempfile.TemporaryDirectory() as directory:
            sampler = t.Sampler(directory, Path(directory) / "missing" / "samples", 1)
            sampler.start()
            result = sampler.stop()
            self.assertTrue(result["errors"])

    def test_source_attribution_records_actual_files_and_requires_harness_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "programs").mkdir()
            (root / "util.py").write_text(
                'def compute_wrapper():\n'
                '    comp_time = time.time()\n'
                '    yield metadata\n'
                '    metadata["wall_time"] = time.time() - comp_time\n'
                'def handle_output_metadata():\n'
                '    provenance_augments["wall_time"] = metadata["wall_time"]\n')
            (root / "compute.py").write_text(
                'def compute():\n'
                '    with compute_wrapper() as metadata:\n'
                '        executor.compute(input_data, config)\n')
            harness = root / "programs/psi4.py"
            harness.write_text("class Harness:\n    def compute(self):\n        pass\n")
            result = profile.source_audit(root, profile.digest(harness))
            self.assertTrue(result["recognized_qcengine_wrapper_timer"])
            self.assertIn("not native-stage", result["attribution"])
            with self.assertRaisesRegex(ValueError, "deployed harness"):
                profile.source_audit(root, "0"*64)
            (root / "util.py").write_text((root / "util.py").read_text().replace("time.time()", "clock()"))
            self.assertFalse(profile.source_audit(root, profile.digest(harness))["recognized_qcengine_wrapper_timer"])

    def test_full_export_sidecars_and_bad_checksum(self):
        import base64
        with tempfile.TemporaryDirectory() as directory:
            raw = b"synthetic transport fixture, not scientific NPZ"
            checksum = hashlib.sha256(raw).hexdigest()
            envelope = dict(artifacts={"tensors.npz": dict(data=base64.b64encode(raw).decode(),
                                                          size=len(raw), sha256=checksum)},
                            result={"tensors_sha256": checksum})
            result = dict(success=True, extras={"psi4:native_monomer": envelope})
            profile.preserve_exports(result, Path(directory))
            self.assertEqual((Path(directory) / "server-tensors.npz").read_bytes(), raw)
            self.assertEqual(json.loads((Path(directory) / "server-native-envelope.json").read_text()), envelope)
            result["extras"]["psi4:native_monomer"]["artifacts"]["tensors.npz"]["size"] += 1
            with self.assertRaisesRegex(ValueError, "checksum"):
                profile.preserve_exports(result, Path(directory))

    @unittest.skipUnless(os.environ.get("PROFILE_FIXTURE_ROOT"), "optional immutable fixture copies not supplied")
    def test_actual_copied_accepted_fixtures_readonly(self):
        root = Path(os.environ["PROFILE_FIXTURE_ROOT"])
        inventory = profile.load(ROOT / "fixtures.json")
        for name, pins in inventory.items():
            with self.subTest(monomer=name):
                data, envelope, metadata = profile.read_fixtures(root / name)
                self.assertEqual(metadata["monomer_id"], name)
                self.assertEqual(envelope["request"]["implementation_sha256"], profile.IMPLEMENTATION)
                self.assertEqual(data["specification"]["protocols"]["error_correction"]["default_policy"], False)
                self.assertEqual({f: profile.digest(root / name / f) for f in pins}, pins)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--cwd-probe":
        cwd_probe(*sys.argv[2:])
    else:
        unittest.main(verbosity=2)
