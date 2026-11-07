"""Stdlib/fake APIs only. Never import Psi4/NumPy or evaluate a native kernel."""
import array
import builtins
import copy
import importlib.util
import json
import os
from pathlib import Path
import signal
import struct
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest import mock
import warnings
import zipfile

HERE = Path(__file__).resolve().parent
s = importlib.util.spec_from_file_location("replay", HERE/"replay.py")
r = importlib.util.module_from_spec(s)
s.loader.exec_module(r)


def fixture():
    meta = dict(schema="molecular-q-capture/1", acceptance=False, implementation=r.ops.OLD_ID,
                build_result_sha256=r.BUILD_SHA, provenance="exact signed sample fixture",
                basis=dict(name="test", origin="effective coefficients", representation="Spherical",
                           shells=[dict(centre=i % 3, l=0, first=i, count=1) for i in range(246)]),
                sites=[dict(label=str(i), rank=4, screen=[] if i == 1 else [2, 0]) for i in range(3)],
                shell_layout=[[i, 1, i % 3, 0] for i in range(246)],
                fixture_sha256={"fixture": "f"*64}, adapter_code_sha256="a"*64,
                original_output={"test": True}, charge_error_observed=0.)
    shapes = r.expected_shapes(meta)
    arrays = {n: (shape, [0.]*__import__("math").prod(shape)) for n, shape in shapes.items()}
    arrays["points"][1][:4] = [-0., 5.e-324, -5.e-324, 1.]
    arrays["exponents"][1][:] = [1.]*246
    arrays["coefficients"][1][:] = [-.7, 5.e-324, -0.]+[1.]*243
    return meta, arrays


class FakeCore:
    IsaGaussianShell = types.SimpleNamespace
    IsaMultipoleSamples = types.SimpleNamespace
    IsaMultipoleSite = types.SimpleNamespace
    IsaBasisRole = types.SimpleNamespace(MolecularAux="MolecularAux")
    IsaBasisRepresentation = types.SimpleNamespace(Spherical="Spherical")

    @staticmethod
    def IsaExplicitBasis(role, rep, centres, shells):
        return types.SimpleNamespace(nfunction=246, shell_layout=lambda: [
            [i, 1, s.centre, s.l] for i, s in enumerate(shells)], shells=shells, centres=centres)


class Sampler:
    def __init__(self, *unused):
        self.errors, self.peaks = [], {}
    def start(self):
        pass
    def stop(self):
        return {"errors": []}


class Tests(unittest.TestCase):
    def test_observer_original_code_only_disable_before_io_and_trace_still_works(self):
        calls, traces = [], []
        def target(x):
            value = x
            return (value,)
        original = target
        def trace(frame, event, arg):
            if frame.f_code is target.__code__:
                traces.append(event)
            return trace
        def capture(local, returned):
            self.assertIsNone(sys.getprofile())
            self.assertIs(sys.gettrace(), trace)
            calls.append((local["value"], returned))
        previous = sys.gettrace()
        try:
            sys.settrace(trace)
            with r.ReturnObserver(target, capture) as obs:
                self.assertEqual(target(-0.), (-0.,))
            self.assertEqual(obs.count, 1)
            self.assertEqual(len(calls), 1)
            self.assertIs(target, original)
            self.assertIn("line", traces)
            self.assertIsNone(sys.getprofile())
        finally:
            sys.settrace(previous)
            sys.setprofile(None)

    def test_exceptional_return_none_preserves_science_exception(self):
        def target():
            raise LookupError("original science error")
        capture = mock.Mock()
        with self.assertRaisesRegex(LookupError, "original science error"):
            with r.ReturnObserver(target, capture):
                target()
        capture.assert_not_called()
        self.assertIsNone(sys.getprofile())

    def test_callback_error_latched_without_retained_frame_even_when_caller_catches(self):
        def target():
            return (1,)
        def fail(*args):
            raise OSError("capture write failed")
        obs = r.ReturnObserver(target, fail)
        with self.assertRaisesRegex(ValueError, "capture write failed"):
            with obs:
                try:
                    target()
                except OSError:
                    pass
        self.assertEqual(obs.error["type"], "OSError")
        self.assertEqual(set(vars(obs)), {"code", "capture", "count", "error"})
        self.assertIsNone(sys.getprofile())

    def test_missing_capture_or_existing_profile_fails_without_replacing_it(self):
        def target():
            return (1,)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            with r.ReturnObserver(target, mock.Mock()):
                pass
        def previous(*args):
            pass
        sys.setprofile(previous)
        try:
            with self.assertRaisesRegex(ValueError, "existing profile"):
                with r.ReturnObserver(target, mock.Mock()):
                    target()
            self.assertIs(sys.getprofile(), previous)
        finally:
            sys.setprofile(None)

    def test_binary_roundtrip_exact_negative_zero_subnormal_and_stream_chunks(self):
        with mock.patch.object(r, "N", 2), tempfile.TemporaryDirectory() as d:
            meta, data = fixture()
            file = Path(d)/"one.npz"
            digest = r.archive_write(file, meta, data)
            got, values = r.archive_read(file, digest)
            self.assertEqual(got["sites"][1]["screen"], [])  # screen ALL, not all centres.
            for name, (_, value) in data.items():
                self.assertEqual(r.bits(values[name]), r.bits(value))
            self.assertNotEqual(r.bits([0.]), r.bits([-0.]))
            self.assertEqual(r.bits(values["points"])[0], 6*8)
            self.assertTrue(all(len(c) <= 65536*8 for c in r.blocks([1.] * 100000)))
            with self.assertRaises(FileExistsError):
                r.archive_write(file, meta, data)
            _, audit = r.archive_read(file, digest, materialize=False)
            self.assertEqual(audit, {})
            class NoncontiguousView:
                @property
                def flat(self):
                    return iter([2., -0., 5.e-324, 1.])
            r.same(NoncontiguousView(), [2., -0., 5.e-324, 1.])
            with self.assertRaisesRegex(ValueError, "flat index 1"):
                r.same(NoncontiguousView(), [2., 0., 5.e-324, 1.])

    def test_capture_reads_actual_native_fields_before_dedup_without_sum(self):
        class Grid:
            shape = (3, 2)
            def __init__(self):
                self.data = [[1., 2.], [3., 4.], [5., 6.]]
            def __len__(self):
                return 3
            def __getitem__(self, n):
                return self.data[n]
            def __iter__(self):
                return iter(self.data)
        # Exercise the actual capture function with fake native getter objects.
        with mock.patch.object(r, "N", 2), tempfile.TemporaryDirectory() as d:
            meta, data = fixture()
            aux = types.SimpleNamespace(centres=[[0., 0., 0.]]*3, name="test", origin="literal",
                representation="Spherical", shells=[types.SimpleNamespace(centre=i%3, l=0,
                    exponents=[1.], coefficients=[-.7]) for i in range(246)])
            points, weights, total, sampled = [[0., 0., 0.], [1., -0., 0.]], [1., -0.], [9., 11.], Grid()
            sites = [types.SimpleNamespace(label=str(i), origin=[0., 0., 0.], rank=4,
                samples=types.SimpleNamespace(points=points, weights=weights, shape=sampled[i],
                                             shape_sum=list(total), auxiliary_sites=[])) for i in range(3)]
            values = types.SimpleNamespace(shape=(75, 246))
            q = types.SimpleNamespace(origins=[[0., 0., 0.]]*3, denominator_cutoff=1.e-36,
                                      values=types.SimpleNamespace(np=values))
            local = dict(auxiliary=aux, qsites=sites, points=points, weights=weights, sampled=sampled,
                         total=total, rank=4, cutoff=1.e-36, provenance="literal",
                         response_aux=types.SimpleNamespace(nfunction=246, shell_layout=lambda: []))
            output = dict(origins_bits=r.small_hex(q.origins), cutoff_bits=r.small_hex([1.e-36]))
            with mock.patch.object(r, "metadata", return_value=output), \
                 mock.patch.object(r, "same", side_effect=lambda a,b: None if a is values else (
                     None if list(r.blocks(a)) == list(r.blocks(b))
                     else (_ for _ in ()).throw(ValueError("bits differ")))), \
                 mock.patch.object(r, "archive_write", return_value="f"*64) as write:
                r.capture(local, (q, values, 0.), Path(d)/"archive", {}, "c"*64)
                self.assertIs(write.call_args.args[2]["shape_sum"][1], total)  # no np.sum/reduction.
                self.assertEqual(write.call_args.args[1]["sites"][0]["screen"], [])
                sites[0].samples.shape_sum = [9., 10.]  # cannot silently dedup unequal native denominator.
                with self.assertRaisesRegex(ValueError, "bits differ"):
                    r.capture(local, (q, values, 0.), Path(d)/"archive2", {}, "c"*64)

    def test_literal_basis_coefficients_screens_and_reconstruction(self):
        with mock.patch.object(r, "N", 2), tempfile.TemporaryDirectory() as d:
            meta, data = fixture()
            file = Path(d)/"a.npz"
            _, values = r.archive_read(file, r.archive_write(file, meta, data))
            aux, sites = r.reconstruct(FakeCore, meta, values)
            self.assertEqual(r.bits(aux.shells[0].coefficients), r.bits([-.7]))
            self.assertEqual(r.bits(aux.shells[1].coefficients), r.bits([5.e-324]))
            self.assertEqual(r.bits(aux.shells[2].coefficients), r.bits([-0.]))
            self.assertEqual(sites[1].samples.auxiliary_sites, [])
            self.assertEqual(sites[0].samples.auxiliary_sites, [2, 0])
            class Bad(FakeCore):
                class IsaGaussianShell:
                    def __setattr__(self, name, value):
                        object.__setattr__(self, name, [1.] if name == "coefficients" else value)
            with self.assertRaisesRegex(ValueError, "bits differ"):
                r.reconstruct(Bad, meta, values)

    def rewrite(self, path, change):
        with zipfile.ZipFile(path) as z:
            members = [(n, z.read(n)) for n in z.namelist()]
        changed = change(members)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as z:
                for name, value in changed:
                    z.writestr(name, value)

    def test_archive_missing_unknown_duplicate_compressed_or_tampered_fails(self):
        with mock.patch.object(r, "N", 2), tempfile.TemporaryDirectory() as d:
            meta, data = fixture()
            changes = [
                lambda m: m[:-1], lambda m: m+[("../unexpected", b"x")],
                lambda m: m+[m[0]], lambda m: [(n, v[:-8]+b"\0"*8 if n == "coefficients.npy" else v) for n,v in m],
                lambda m: [(n, v.replace(b"'<f8'", b"'|O8'") if n == "points.npy" else v) for n,v in m],
            ]
            for i, change in enumerate(changes):
                path = Path(d)/str(i)
                digest = r.archive_write(path, meta, data)
                self.rewrite(path, change)
                with self.assertRaises(ValueError):
                    r.archive_read(path, r.ops.sha(path))
                with self.assertRaises(ValueError):
                    r.archive_read(path, "0"*64)
            path = Path(d)/"compressed"
            r.archive_write(path, meta, data)
            with zipfile.ZipFile(path) as z:
                members = {n:z.read(n) for n in z.namelist()}
            with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
                for n,v in members.items(): z.writestr(n,v)
            with self.assertRaisesRegex(ValueError, "compression"):
                r.archive_read(path, r.ops.sha(path))

    def test_all_headers_and_budgets_validated_before_materialization_and_write_cap(self):
        with mock.patch.object(r, "N", 2), tempfile.TemporaryDirectory() as d:
            meta, data = fixture()
            path = Path(d)/"bad"
            r.archive_write(path, meta, data)
            def modify(members):
                result = []
                for n,v in members:
                    if n == "manifest.json":
                        doc = json.loads(v); doc["arrays"]["Q"]["shape"] = [75, 10**12]
                        v = json.dumps(doc).encode()
                    result.append((n,v))
                return result
            self.rewrite(path, modify)
            with mock.patch.object(r.array, "array", side_effect=AssertionError("allocated before checking ALL shapes")):
                with self.assertRaisesRegex(ValueError, "shape/dtype"):
                    r.archive_read(path, r.ops.sha(path))
            with (Path(d)/"bounded").open("wb") as f, mock.patch.object(r, "LIMIT", 9):
                stream = r.CappedStream(f)
                stream.write(b"12345678")
                with self.assertRaisesRegex(ValueError, "incremental"):
                    stream.write(b"xx")
                self.assertEqual(f.tell(), 8)
            with mock.patch.object(r, "LIMIT", 1024):
                with self.assertRaisesRegex(ValueError, "budget"):
                    r.archive_write(Path(d)/"oversized", meta, data)
            self.assertFalse((Path(d)/"oversized").exists())
            # Reject oversized/unknown central directories before ZipInfo heap growth.
            bad = Path(d)/"central"
            r.archive_write(bad, meta, data)
            payload = bytearray(bad.read_bytes())
            payload[-14:-12] = struct.pack("<H", 65535)  # EOCD total-entry count.
            bad.write_bytes(payload)
            with mock.patch.object(r.zipfile, "ZipFile", side_effect=AssertionError("parsed unbounded directory")):
                with self.assertRaisesRegex(ValueError, "directory inventory"):
                    r.archive_read(bad, r.ops.sha(bad))

    def test_failed_or_partial_full_capture_never_admits_fixture(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            r.ops.write(root/"summary.json", dict(status="FAILED"))
            with self.assertRaisesRegex(ValueError, "entire OLD capture"):
                r.capture_ready(root, "e"*64, {})

    def test_adapter_span_count_missing_exceptional_or_repeated_fails(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/"events"
            begin = dict(event="begin", name="isapol_distribution._stockholder_q", id=1)
            end = dict(event="end", name=begin["name"], id=1, status="returned")
            for rows, good in (([begin,end],True),([],False),([begin],False),
                               ([begin,dict(end,status="failed")],False),
                               ([begin,end,dict(begin,id=2),dict(end,id=2)],False)):
                path.write_text("\n".join(map(json.dumps,rows)))
                if good:
                    self.assertEqual(r.adapter_span(path),1)
                else:
                    with self.assertRaisesRegex(ValueError,"exactly one"):
                        r.adapter_span(path)

    def args(self, root):
        return types.SimpleNamespace(source=root/"source", root=root/"out", scratch_root=root/"scratch",
            profiler_source=root/"profile", fixtures=root/"fixtures", compiled_source=root/"compiled",
            commit="c", tree="t", build_result_sha256=r.BUILD_SHA, launch_token="test", exit_code=0,
            threads=8, operation="capture", arm="old", affinity=json.dumps(list(range(8))))

    def test_real_preimport_path_owns_cwd_and_preserves_parent_on_import_exception(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); args = self.args(root)
            protected = root/"protected"; protected.mkdir()
            (protected/"sentinel").write_bytes(b"protected")
            original = Path.cwd()
            try:
                os.chdir(protected)
                importer = builtins.__import__
                def failed_import(name, *a, **kw):
                    if name == "psi4":
                        self.assertEqual(Path.cwd(), args.scratch_root/"cwd")
                        raise ImportError("fake Psi4 import failure")
                    return importer(name, *a, **kw)
                p = types.SimpleNamespace(read_fixtures=lambda unused: ({},{},{"monomer_id":"monomer-0001"}))
                with mock.patch.object(r, "source_gate"), mock.patch.object(r.b, "build_gate"), \
                     mock.patch.object(r.b, "profile", return_value=p), \
                     mock.patch.object(builtins, "__import__", side_effect=failed_import), \
                     mock.patch.dict(os.environ, SLURM_JOB_ID="fake", SLURM_CPUS_PER_TASK="8",
                         OMP_NUM_THREADS="8", MKL_NUM_THREADS="8", OPENBLAS_NUM_THREADS="1"), \
                     mock.patch.object(sys, "flags", types.SimpleNamespace(isolated=True)), \
                     mock.patch.object(sys, "pycache_prefix", str(args.scratch_root/"cache")):
                    with self.assertRaises(ImportError):
                        r.child(args)  # Actual child->enter_child->pre-import path, not a substitute.
                    self.assertEqual(Path.cwd(), args.scratch_root/"cwd")
                    self.assertEqual(list(protected.iterdir()), [protected/"sentinel"])
                    self.assertEqual((protected/"sentinel").read_bytes(), b"protected")
            finally:
                os.chdir(original)

    def controller(self, failure=False, mismatch=False):
        with tempfile.TemporaryDirectory() as d:
            args = self.args(Path(d))
            calls = []
            qbytes = b"\0"*(75*246*8)
            qsha = r.hashlib.sha256(qbytes).hexdigest()
            captured = dict(arrays={"Q": {"sha256": qsha}}, original_output={"fake": True})
            p = types.SimpleNamespace(telemetry=types.SimpleNamespace(Sampler=Sampler))
            def fake_run(runner, name, command):
                calls.append(name)
                out = args.root/name; out.mkdir()
                if failure:
                    raise ValueError("child process exited1")
                item = dict(status="RETURNED_NOT_ACCEPTED", capture_sha256="e"*64, fixed_bits_equal=True)
                if name != "capture":
                    item.update(Q_bits_sha256=qsha, output_metadata={"fake": True})
                    (out/"Q.npy").write_bytes(r.header((75,246))+(b"x"+qbytes[1:] if mismatch else qbytes))
                r.ops.write(out/"summary.json", item)
            original = Path.cwd()
            try:
                with mock.patch.object(r, "source_gate"), mock.patch.object(r, "seal", return_value={}) as seal, \
                     mock.patch.object(r.b, "profile", return_value=p), mock.patch.object(r.b, "build_gate",
                         return_value={"identity": {"implementation": {"test": "fake"}}}), \
                     mock.patch.object(r, "archive_read", return_value=(captured, {})), \
                     mock.patch.object(r, "saved_capture_Q", return_value=r.header((75,246))+qbytes), \
                     mock.patch.object(r.ops, "allocation", return_value={}), \
                     mock.patch.dict(os.environ, SLURM_JOB_ID="fake", SLURM_CPUS_PER_TASK="8"), \
                     mock.patch.object(os, "sched_getaffinity", return_value=set(range(8))), \
                     mock.patch.object(r.ops.Runner, "run", fake_run):
                    if failure or mismatch:
                        with self.assertRaises(ValueError):
                            r.run(args)
                    else:
                        r.run(args)
                    self.assertEqual(seal.call_count, 2)
                    status = r.finalize(args)
                    self.assertEqual(status, 1 if failure or mismatch else 0)
                    self.assertEqual((args.root/"COMPLETED").exists(), not (failure or mismatch))
                    if not (failure or mismatch):
                        self.assertEqual(calls, ["capture"]+[
                            f"replay-{n}-{arm}" for n in (1,2,4,8) for arm in ("old","candidate")])
                    self.assertFalse(r.ops.read(args.root/"result.json")["acceptance"])
            finally:
                os.chdir(original)

    def test_eight_isolated_replays_no_candidate_scf(self):
        self.controller()

    def test_child_failure_preserves_post_integrity_check_and_no_success(self):
        self.controller(failure=True)

    def test_controller_does_not_trust_success_summary_when_saved_Q_differs(self):
        self.controller(mismatch=True)

    def test_source_and_frozen_science_guards_no_fullgrid_or_denominator_recomputation(self):
        text = (HERE/"replay.py").read_text()
        self.assertIn('target = distribution._stockholder_q', text)
        self.assertIn('with ReturnObserver(target, snapshot)', text)
        self.assertNotIn("np.sum", text)
        self.assertNotIn("final_shape_samples(", text)
        self.assertNotIn(".evaluate(", text)
        self.assertNotIn("settrace(", text)
        self.assertNotIn("setattr(psi4.core", text)
        self.assertNotIn("8558cc", text)  # Actual identity is read, not a replacement constant.
        self.assertEqual(r.BUILD_SHA, "ab8a01fef0bb715bf31e084ae7f9c56251b9cae1471f3d1b4389a795ebd041cf")
        for path, digest in r.ops.FROZEN.items():
            self.assertEqual(r.ops.sha(HERE.parents[1]/path), digest)
        self.assertNotIn("numpy", sys.modules)
        self.assertNotIn("psi4", sys.modules)

    def test_source_allowlist_and_literal_build_gate_are_not_optional(self):
        args = self.args(Path("/fake"))
        args.source = HERE.parents[1]
        changed = "\n".join(sorted(r.ALLOWED))
        def git(path, *argv):
            if path == r.b.PROFILE_SOURCE:
                return {"HEAD": r.b.PROFILE_COMMIT, "HEAD^{tree}": r.b.PROFILE_TREE}.get(argv[-1], "")
            if argv[:1] == ("merge-base",):
                return r.BASE
            if argv[:1] == ("diff",):
                return changed
            if argv == ("rev-parse", "--show-toplevel"):
                return str(args.source)
            return {"HEAD": args.commit, "HEAD^{tree}": args.tree}.get(argv[-1], "")
        with mock.patch.object(r.ops, "git", side_effect=git), mock.patch.object(r.ops, "source_gate") as compiled:
            r.source_gate(args)
            compiled.assert_called_once_with(r.b.COMPILED_SOURCE, r.b.BASE_COMMIT, r.b.COMPILED_TREE)
            args.build_result_sha256 = "1"*64
            with self.assertRaisesRegex(ValueError, "literal"):
                r.source_gate(args)
            args.build_result_sha256 = r.BUILD_SHA
            changed += "\npsi4/driver/schema_wrapper.py"
            with self.assertRaisesRegex(ValueError, "frozen"):
                r.source_gate(args)

    def test_term_int_during_finalization_before_after_marker_never_succeeds(self):
        for phase in ("before", "after"):
            for sig in (signal.SIGTERM, signal.SIGINT):
                with self.subTest(phase=phase, signal=sig), tempfile.TemporaryDirectory() as d:
                    root = Path(d)
                    helper = root/"source/devtools/isapol_q_molecular_replay/replay.py"
                    helper.parent.mkdir(parents=True)
                    helper.write_text("""
import argparse,importlib.util,os,pathlib,signal,sys,time
p=argparse.ArgumentParser();p.add_argument("op")
for n in ("source","commit","tree","root","scratch-root","launch-token","build-result-sha256"):p.add_argument("--"+n)
p.add_argument("--exit-code",type=int);a=p.parse_args();a.root=pathlib.Path(a.root)
s=importlib.util.spec_from_file_location("real",os.environ["REAL_REPLAY"])
m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.source_gate=lambda *a:None
state=pathlib.Path(os.environ["STATE"])
if a.op=="run":
 a.root.mkdir();m.ops.write(a.root/"owner.json",dict(token=a.launch_token,mode="molecular-replay"))
 m.ops.write(a.root/"result.json",dict(status=m.SUCCESS,launch_token=a.launch_token,
 old_and_candidate_unchanged=True,fixed_bits_equal=True,acceptance=False,adoption=False,
 build_result_sha256=m.BUILD_SHA));sys.exit(0)
if a.exit_code==0:
 def stop(*a):
  (state/"TERM").write_text("forwarded");sys.exit(143)
 signal.signal(signal.SIGTERM,stop)
 if os.environ["PHASE"]=="after":m.finalize(a)
 (state/"FINALIZING").write_text("ready")
 while not (state/"RELEASE").exists():time.sleep(.01)
sys.exit(m.finalize(a))
""")
                    fake = root/"srun"
                    fake.write_text("#!/usr/bin/python3\nimport os,sys\na=sys.argv[1:]\n"
                                    "while a[0].startswith('--'):a.pop(0)\nos.execv(a[0],a)\n")
                    fake.chmod(0o700)
                    wrapper = root/"wrapper"
                    wrapper.write_text((HERE/"replay.sbatch").read_text().replace("/opt/slurm/current/bin/srun",str(fake)))
                    process = subprocess.Popen(["bash",str(wrapper),"--source",str(root/"source"),
                        "--commit","c","--tree","t","--root",str(root/"out"),
                        "--scratch-root",str(root/"scratch"),"--build-result-sha256",r.BUILD_SHA],cwd=root,
                        env=dict(os.environ,REAL_REPLAY=str(HERE/"replay.py"),STATE=str(root),PHASE=phase),
                        stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,start_new_session=True)
                    try:
                        deadline = time.monotonic()+8
                        while not (root/"FINALIZING").exists() and time.monotonic()<deadline:
                            if process.poll() is not None:break
                            time.sleep(.01)
                        self.assertTrue((root/"FINALIZING").exists())
                        process.send_signal(sig)
                        stdout,stderr = process.communicate(timeout=8)
                        self.assertEqual(process.returncode,143,stderr)
                        self.assertTrue((root/"TERM").exists())
                        self.assertFalse((root/"out/COMPLETED").exists())
                        self.assertEqual(r.ops.read(root/"out/FAILED")["exit_code"],143)
                    finally:
                        (root/"RELEASE").touch()
                        if process.poll() is None:
                            os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=5)

    def test_finite_wrapper_and_no_new_affinity_knob(self):
        text = (HERE/"replay.sbatch").read_text()
        subprocess.run(["bash","-n",str(HERE/"replay.sbatch")],check=True)
        for s in ("--qos=embers","--mem=32G","--cpus-per-task=8","--time=02:00:00","--no-requeue"):
            self.assertIn("#SBATCH "+s,text)
        self.assertNotIn("OMP_PROC_BIND",text)
        self.assertNotIn("OMP_PLACES",text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
