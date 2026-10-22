# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""Engine-free temporary runtime/basis toys. No scientific runtime is activated."""
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def load(name,file):
    spec = importlib.util.spec_from_file_location(name,HERE/file)
    item = importlib.util.module_from_spec(spec)
    sys.modules[name] = item
    spec.loader.exec_module(item)
    return item


R = load("toy_integrated_runtime","integrated_runtime.py")
J = load("toy_integrated_job","integrated_job.py")
B = load("toy_reused_basis_fakes","test_basis_admission.py")


class IntegratedRuntimeTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.source,self.accepted = self.root/"source",self.root/"accepted-stage"
        for root in (self.source/"psi4/driver",self.accepted/"lib/psi4/driver"):
            root.mkdir(parents=True)
        (self.source/"psi4/driver/__init__.py").write_bytes(b"# NEW toy driver\n")
        (self.source/"psi4/driver/new_hook.py").write_bytes(b"# NEW toy hook\n")
        (self.accepted/"lib/psi4/driver/__init__.py").write_bytes(b"# OLD toy driver\n")
        (self.accepted/"lib/psi4/__init__.py").write_bytes(b"# configured toy, NOT raw CMake\n")
        (self.accepted/"lib/psi4/extras.py").write_bytes(b"# configured toy extras\n")
        (self.accepted/"lib/psi4/core.cpython-313-x86_64-linux-gnu.so").write_bytes(b"TOY NOT ELF")
        (self.accepted/"bin").mkdir()
        (self.accepted/"bin/psi4").write_bytes(b"# TOY configured launcher\n")
        (self.accepted/"share/psi4").mkdir(parents=True)
        self.runtime = self.root/"runtime"
        self.binding = dict(source=str(self.source),commit="1"*40,tree="2"*40,
            native=dict(core_sha256=R.sha(self.accepted/"lib/psi4/core.cpython-313-x86_64-linux-gnu.so")))
        # Independent, frozen toy Git-tree response. It does NOT track later
        # filesystem omissions/additions/mutations or the composed manifest.
        records = []
        for relative in ("psi4/driver/__init__.py","psi4/driver/new_hook.py"):
            data = (self.source/relative).read_bytes()
            oid = hashlib.sha1(b"blob "+str(len(data)).encode()+b"\0"+data).hexdigest()
            records.append(f"100644 blob {oid}\t{relative}".encode()+b"\0")
        self.git_tree = b"".join(records)
        self.inventory_git = mock.Mock(side_effect=self.committed_inventory_reply)
        patch = mock.patch.object(R.subprocess,"check_output",self.inventory_git)
        patch.start(); self.addCleanup(patch.stop)
        self.stack = mock.patch.multiple(R.OLD,STAGE=self.accepted,
            CORE_SHA=self.binding["native"]["core_sha256"])
        self.stack.start()
        self.addCleanup(self.stack.stop)
        patch = mock.patch.dict(R.PLAN,{"runtime":str(self.runtime)})
        patch.start(); self.addCleanup(patch.stop)

    def committed_inventory_reply(self, argv, **kwargs):
        self.assertEqual(argv,["git","-C",str(self.source),"ls-tree","-rz",
                               self.binding["commit"],"--","psi4/driver"])
        self.assertFalse(kwargs)
        return self.git_tree

    def composed(self):
        manifest = R.compose(self.source,self.accepted,self.runtime/"stage",self.binding)
        file = self.runtime/"runtime-manifest.json"
        pin = R.write(file,manifest)
        return file,pin,manifest

    def test_composition_is_exclusive_configured_root_and_complete_new_driver(self):
        path,pin,manifest = self.composed()
        stage = self.runtime/"stage"
        self.assertEqual((stage/"lib/psi4/__init__.py").read_bytes(),b"# configured toy, NOT raw CMake\n")
        self.assertEqual((stage/"lib/psi4/driver/__init__.py").read_bytes(),b"# NEW toy driver\n")
        self.assertTrue((stage/"lib/psi4/driver/new_hook.py").is_file())
        self.assertTrue((stage/"lib/psi4/core.cpython-313-x86_64-linux-gnu.so").is_symlink())
        self.assertEqual((stage/"bin/psi4").read_bytes(),b"# TOY configured launcher\n")
        self.assertEqual(manifest["copied_bytes"],sum(r.get("nbytes",0) for r in manifest["records"].values()))
        self.assertFalse(manifest["native_compiled_from_new_commit"])
        self.assertFalse(manifest["core_hash_verified_during_composition"])
        self.assertEqual(R.verify_runtime(path,pin,self.binding,hash_core=True),manifest)
        with self.assertRaises(FileExistsError):
            R.compose(self.source,self.accepted,stage,self.binding)
        self.assertEqual((self.accepted/"lib/psi4/driver/__init__.py").read_bytes(),b"# OLD toy driver\n")

    def test_unexpected_python_bytecode_and_prefix_files_are_refused(self):
        path,pin,_ = self.composed()
        stage = self.runtime/"stage"
        for relative in ("lib/psi4/unreviewed.py","lib/psi4/unreviewed.pyc","lib/extra.py"):
            with self.subTest(relative=relative):
                file = stage/relative
                file.write_bytes(b"TOY unexpected")
                with self.assertRaisesRegex(ValueError,"unexpected"):
                    R.verify_runtime(path,pin,self.binding)
                file.unlink()  # temporary test fixture only

    def test_wrong_core_link_and_hash_are_refused(self):
        path,pin,_ = self.composed()
        core = self.runtime/"stage/lib/psi4/core.cpython-313-x86_64-linux-gnu.so"
        core.unlink()
        core.symlink_to(self.accepted/"lib/psi4/extras.py")
        with self.assertRaisesRegex(ValueError,"core link"):
            R.verify_runtime(path,pin,self.binding,hash_core=True)
        core.unlink()
        core.symlink_to(self.accepted/"lib/psi4/core.cpython-313-x86_64-linux-gnu.so")
        core.resolve().write_bytes(b"CHANGED TOY CORE NOT ELF")
        with self.assertRaisesRegex(ValueError,"core hash"):
            R.verify_runtime(path,pin,self.binding,hash_core=True)

    def test_wrong_data_link_and_changed_python_bytes_are_refused(self):
        path,pin,_ = self.composed()
        data = self.runtime/"stage/share/psi4"
        data.unlink(); data.symlink_to(self.source,target_is_directory=True)
        with self.assertRaisesRegex(ValueError,"data root"):
            R.verify_runtime(path,pin,self.binding)
        data.unlink(); data.symlink_to(self.accepted/"share/psi4",target_is_directory=True)
        python = self.runtime/"stage/lib/psi4/driver/new_hook.py"
        python.chmod(0o600); python.write_bytes(b"CHANGED TOY")
        with self.assertRaisesRegex(ValueError,"hash differs"):
            R.verify_runtime(path,pin,self.binding)

    def test_runtime_manifest_pin_and_new_binding_mismatch_are_refused(self):
        path,pin,_ = self.composed()
        with self.assertRaisesRegex(ValueError,"hash/type/size"):
            R.verify_runtime(path,"0"*64,self.binding)
        changed = copy.deepcopy(self.binding); changed["commit"] = "3"*40
        with self.assertRaisesRegex(ValueError,"binding differs"):
            R.verify_runtime(path,pin,changed)

    def test_inconsistent_declared_core_hash_is_refused_even_with_recomputed_manifest_pin(self):
        path,pin,manifest = self.composed()
        manifest["records"]["lib/psi4/core.cpython-313-x86_64-linux-gnu.so"]["sha256"] = "0"*64
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError,"core.*declaration"):
            R.verify_runtime(path,R.sha(path),self.binding,hash_core=True)

    def test_nonregular_metadata_is_rejected_before_hashing(self):
        fifo = self.root/"metadata.fifo"
        os.mkfifo(fifo)
        with mock.patch.object(R,"sha",side_effect=AssertionError("must reject FIFO before hashing")):
            with self.assertRaisesRegex(ValueError,"type"):
                R.pinned_json(fifo,"0"*64)

    def test_launcher_fifo_is_rejected_without_blocking_open_or_hash(self):
        launcher = self.accepted/"bin/psi4"
        launcher.unlink()
        os.mkfifo(launcher)
        original_open, original_path_open = os.open, Path.open
        def nonblocking_only(path, flags, *args, **kwargs):
            if Path(path) == launcher:
                self.assertTrue(flags & os.O_NONBLOCK)
                self.assertTrue(flags & os.O_NOFOLLOW)
            return original_open(path,flags,*args,**kwargs)
        def reject_blocking_path_open(path, *args, **kwargs):
            if path == launcher:
                self.fail("launcher copy reached blocking Path.open before type validation")
            return original_path_open(path,*args,**kwargs)
        with mock.patch.object(R.os,"open",side_effect=nonblocking_only), \
             mock.patch.object(Path,"open",reject_blocking_path_open):
            with self.assertRaisesRegex(ValueError,"regular"):
                self.composed()

    def test_launcher_symlink_is_rejected_instead_of_followed(self):
        launcher = self.accepted/"bin/psi4"
        launcher.unlink()
        launcher.symlink_to(self.accepted/"lib/psi4/extras.py")
        with self.assertRaises((ValueError,OSError)):
            self.composed()

    def test_oversized_launcher_refuses_before_destination_creation(self):
        launcher = self.accepted/"bin/psi4"
        with launcher.open("wb") as stream:
            stream.truncate(32*1024**2+1)  # sparse engine-free size fixture
        with self.assertRaisesRegex(ValueError,"byte bound"):
            self.composed()
        self.assertFalse((self.runtime/"stage/bin/psi4").exists())

    def test_launcher_bytes_share_unchanged_32MiB_package_allowance(self):
        launcher = self.accepted/"bin/psi4"
        with launcher.open("wb") as stream:
            stream.truncate(32*1024**2)  # individually fits, package+launcher does not
        with self.assertRaisesRegex(ValueError,"byte bound"):
            self.composed()
        self.assertFalse((self.runtime/"stage/bin/psi4").exists())

    def test_unreadable_traversal_error_is_raised_not_silently_omitted(self):
        error = PermissionError("TOY unreadable driver subtree")
        def unreadable_walk(root, **kwargs):
            if kwargs.get("onerror") is not None:
                kwargs["onerror"](error)
            return iter(())  # os.walk's old silent-default behavior
        with mock.patch.object(R.os,"walk",side_effect=unreadable_walk):
            with self.assertRaises(PermissionError) as caught:
                R.regular_files(self.source/"psi4/driver")
        self.assertIs(caught.exception,error)

    def test_omitted_committed_driver_source_file_is_refused(self):
        (self.source/"psi4/driver/new_hook.py").unlink()
        with self.assertRaisesRegex(ValueError,"committed driver.*inventory"):
            self.composed()

    def test_extra_uncommitted_driver_source_file_is_refused(self):
        (self.source/"psi4/driver/extra.py").write_bytes(b"# uncommitted TOY\n")
        with self.assertRaisesRegex(ValueError,"committed driver.*inventory"):
            self.composed()

    def test_altered_committed_driver_bytes_are_refused_against_git_blob(self):
        (self.source/"psi4/driver/new_hook.py").write_bytes(b"# changed TOY under same name\n")
        with self.assertRaisesRegex(ValueError,"committed driver Python blob identity"):
            self.composed()

    def test_incomplete_stage_and_rewritten_manifest_cannot_hide_committed_file(self):
        path,pin,manifest = self.composed()
        rel = "lib/psi4/driver/new_hook.py"
        (self.runtime/"stage"/rel).unlink()
        del manifest["records"][rel]
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError,"committed driver.*inventory"):
            R.verify_runtime(path,R.sha(path),self.binding)

    def test_extra_driver_in_stage_manifest_and_source_cannot_override_committed_inventory(self):
        path,pin,manifest = self.composed()
        source = self.source/"psi4/driver/extra.py"
        source.write_bytes(b"# uncommitted TOY\n")
        staged = self.runtime/"stage/lib/psi4/driver/extra.py"
        staged.write_bytes(source.read_bytes())
        manifest["records"]["lib/psi4/driver/extra.py"] = dict(role="new committed Python tree",
            source=str(source),sha256=R.sha(source),nbytes=source.stat().st_size)
        manifest["copied_bytes"] += source.stat().st_size
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError,"committed driver.*inventory"):
            R.verify_runtime(path,R.sha(path),self.binding)

    def test_malformed_or_symlink_git_inventory_cannot_be_trusted(self):
        original = self.git_tree
        for data in (original[:-1], original+original, original.replace(b"100644",b"120000",1),
                     original.replace(b"psi4/driver/new_hook.py",b"../untrusted.py",1)):
            with self.subTest(data=data):
                self.git_tree = data
                with self.assertRaises(ValueError):
                    R.committed_driver_inventory(self.source,self.binding["commit"])

    def test_explicit_source_bytecode_exclusions_remain_allowed(self):
        (self.source/"psi4/driver/new_hook.pyc").write_bytes(b"TOY ignored bytecode")
        cache = self.source/"psi4/driver/__pycache__"
        cache.mkdir()
        (cache/"ignored.pyc").write_bytes(b"TOY ignored cache")
        path,pin,manifest = self.composed()
        self.assertFalse(any("__pycache__" in name or name.endswith(".pyc") for name in manifest["records"]))
        R.verify_runtime(path,pin,self.binding)

    def test_successful_entry_does_not_publish_a_failure_receipt(self):
        tree = ast.parse((HERE/"integrated_job.py").read_text())
        entry = tree.body[-1]
        metadata = self.root/"run/metadata"
        metadata.mkdir(parents=True)
        fake = SimpleNamespace(PLAN={"run":str(metadata.parent)},write=mock.Mock())
        env = dict(__name__="__main__",main=lambda:0,sys=sys,R=fake,Path=Path,os=os,traceback=__import__("traceback"))
        with mock.patch.dict(os.environ,dict(SLURM_JOB_ID="TOY",USER="awallace43",HOME="/storage/home/TOY")):
            with self.assertRaises(SystemExit) as exit:
                exec(compile(ast.Module(body=[entry],type_ignores=[]),"actual-entry","exec"),env)
        self.assertEqual(exit.exception.code,0)
        fake.write.assert_not_called()

    def test_compute_guard_refuses_local_before_scientific_import(self):
        before = {name for name in ("psi4","numpy","adcc","einsums") if name in sys.modules}
        with mock.patch.dict(os.environ,{"SLURM_JOB_ID":""}):
            with self.assertRaisesRegex(ValueError,"Phoenix compute"):
                R.compute_guard()
        self.assertEqual(before,{name for name in ("psi4","numpy","adcc","einsums") if name in sys.modules})

    def test_driver_has_one_scf_one_pipeline_and_export_only_after_pipeline(self):
        tree = ast.parse((HERE/"integrated_driver.py").read_text())
        fn = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="capture_science")
        calls = [n for n in ast.walk(fn) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
        self.assertEqual(sum(n.func.attr=="energy" for n in calls),1)
        self.assertEqual(sum(n.func.attr=="oeprop" for n in calls),1)
        self.assertEqual(sum(n.func.attr=="export" for n in calls),1)
        oeprop = next(n for n in calls if n.func.attr=="oeprop")
        export = next(n for n in calls if n.func.attr=="export")
        self.assertLess(oeprop.lineno,export.lineno)
        self.assertIn("diagnostic_sink",{k.arg for k in oeprop.keywords})
        self.assertFalse(any(n.func.attr in ("solve","energy") and n is not next(c for c in calls if c.func.attr=="energy")
                             for n in calls))
        script = (HERE/"integrated_capture.sbatch").read_text()
        for token in ("--qos=embers","--cpus-per-task=8","--mem=48G","--time=06:00:00",
                      "--nodelist=atl1-1-02-007-23-1","child_ok=0","child_ok=1"):
            self.assertIn(token,script)
        self.assertNotIn("sbatch ",script)  # no submission/retry in job

    def test_source_allowlist_and_native_tree_identity_fail_closed(self):
        def git(path,*args):
            if args[:2] == ("diff","--name-only"):
                return "psi4/src/DO_NOT_CHANGE.cc"
            return "unchanged-tree"
        with mock.patch.dict(R.PLAN,{"source":str(R.ROOT)}), \
             mock.patch.object(R.OLD,"identity"), mock.patch.object(R.OLD,"git",side_effect=git):
            with self.assertRaisesRegex(ValueError,"allowlist"):
                R.source_binding("a"*40,"b"*40)
        def git_tree_changed(path,*args):
            if args[:2] == ("diff","--name-only"):
                return R.PLAN["routing"][0]
            return "old-tree" if args[-1].startswith(R.OLD.BUILD_COMMIT) else "changed-tree"
        with mock.patch.dict(R.PLAN,{"source":str(R.ROOT)}), \
             mock.patch.object(R.OLD,"identity"), mock.patch.object(R.OLD,"git",side_effect=git_tree_changed):
            with self.assertRaisesRegex(ValueError,"native/GBS"):
                R.source_binding("a"*40,"b"*40)

    def test_initializer_wiring_is_same_process_and_guarded_before_imports(self):
        tree = ast.parse((HERE/"integrated_runtime.py").read_text())
        fn = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="initialize")
        statements = [ast.unparse(n) for n in fn.body]
        text = "\n".join(statements)
        first_import = next(i for i,t in enumerate(statements) if t.startswith("import numpy"))
        self.assertTrue(statements[0].startswith("compute_guard()"))
        self.assertLess(next(i for i,t in enumerate(statements) if "ctypes.CDLL" in t),first_import)
        self.assertLess(next(i for i,t in enumerate(statements) if "retained_handles.append(handle)" in t),first_import)
        self.assertIn("RTLD_NOW | os.RTLD_GLOBAL",text)
        self.assertIn("libEinsums.so.1.1.3",text)
        self.assertIn("before-capture",text)
        self.assertIn("binding['routing_sha256']",text)
        self.assertNotIn("subprocess.run",text)  # pilot not launched in another interpreter
        # The only subprocess is the staged, import-free --psiapi helper.
        self.assertIn("'--psiapi'",text)

    def test_term_safe_script_exit_and_launch_guards_static_only(self):
        # Source-contract regression only: no actual signal/process lifecycle test.
        text = (HERE/"integrated_capture.sbatch").read_text()
        finalizer = text.split("on_exit() {",1)[1].split("\n}",1)[0]
        self.assertIn("trap - EXIT",finalizer)
        self.assertIn('exit "$rc"',finalizer)
        self.assertLess(finalizer.index("rc=$?"),finalizer.index("trap - EXIT"))
        launch = text.index("/opt/slurm/current/bin/srun")
        self.assertIn("if (( term_received )); then exit 143; fi",text[:launch])
        child_assigned = text.index("child=$!",launch)
        wait = text.index('wait "$child"',child_assigned)
        self.assertIn('if (( term_received )); then kill -TERM "$child" || true; fi',
                      text[child_assigned:wait])
        cleanup = text[text.index("rc=$?",wait):text.index("set -e\n",wait)]
        self.assertIn("trap '' TERM INT HUP",cleanup)
        self.assertLess(cleanup.index("trap '' TERM INT HUP"),cleanup.index('wait "$child"'))

    def test_real_job_attempts_final_closure_and_preserves_primary_exception(self):
        error = RuntimeError("TOY original strict LW exception")
        for closure_fails in (False,True):
            with self.subTest(closure_fails=closure_fails), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root/"metadata").mkdir()
                binding = {"new":"TOY binding"}
                runtime_sha,derivative_sha = "a"*64,"b"*64
                R.write(root/"metadata/preparation.json",dict(binding=binding,
                    runtime_manifest_sha256=runtime_sha,derivative_sha256=derivative_sha,
                    script_sha256=R.sha(HERE/"integrated_capture.sbatch")))
                retained = []
                token = object()
                def initialize(*args):
                    args[-1].append(token)
                    retained.append(args[-1])
                    return object(),token
                def snapshot(*args,**kwargs):
                    self.assertIs(retained[0][0],token)
                    if closure_fails:
                        raise ValueError("TOY final loader refusal")
                fake = SimpleNamespace(PLAN=dict(run=str(root),runtime=str(root/"runtime"),
                    prior_probe_sha256="c"*64),compute_guard=mock.Mock(),
                    OLD=SimpleNamespace(BUILD_COMMIT="native15cc",CORE_SHA="coreb283",
                        snapshot=mock.Mock(side_effect=snapshot),check_driver_result=R.OLD.check_driver_result),
                    source_binding=mock.Mock(return_value=binding),sha=R.sha,write=R.write,
                    pinned_json=mock.Mock(return_value={"same":"derived"}),
                    derived_document=mock.Mock(return_value={"same":"derived"}),
                    initialize=mock.Mock(side_effect=initialize),verify_runtime=mock.Mock(),
                    module=mock.Mock(return_value=SimpleNamespace(capture_science=mock.Mock(side_effect=error))))
                with mock.patch.object(J,"R",fake),mock.patch.dict(os.environ,{"SLURM_JOB_ID":"TOY"}):
                    self.assertEqual(J.run("1"*40,"2"*40,runtime_sha,derivative_sha),1)
                fake.OLD.snapshot.assert_called_once_with("after-capture",receipt_run=root)
                result = json.loads((root/"results/result.json").read_text())
                self.assertEqual(result["error"],"RuntimeError: TOY original strict LW exception")
                self.assertIn("TOY original strict LW exception",result["traceback"])
                self.assertFalse(result["science_pass"])
                self.assertEqual(result["after_runtime_closure"],"failed" if closure_fails else "passed")
                fake.module.return_value.capture_science.assert_called_once()


class CommittedDriverInventoryTests(unittest.TestCase):
    def test_actual_local_git_inventory_reads_objects_without_scientific_import(self):
        inventory = R.committed_driver_inventory(ROOT,"ffc6f86948cd0da6bd876073ab28e856386ac810")
        self.assertEqual(len(inventory),166)
        self.assertIn(Path("driver/procrouting/isapol_bounded.py"),inventory)
        path = ROOT/"psi4/driver/procrouting/isapol_bounded.py"
        self.assertEqual(R.small_regular(path,32*1024**2)["git_blob_sha1"],
                         inventory[Path("driver/procrouting/isapol_bounded.py")])


class DerivedAdmissionTests(unittest.TestCase):
    """Fake normalized shells and fictitious centres; never native measurements."""
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.stage = self.root/"accepted-stage"
        for name in R.BASIS.contract()["basis_files"]:
            f = self.stage/"share/psi4/basis"/name
            f.parent.mkdir(parents=True,exist_ok=True)
            f.write_bytes((ROOT/"psi4/share/psi4/basis"/name).read_bytes())
        patch = mock.patch.multiple(R.OLD,SOURCE=ROOT,STAGE=self.stage)
        patch.start(); self.addCleanup(patch.stop)
        self.core,self.molecule = B.FakeCore(),B.FakeMolecule()
        self.options = B.options_state(self.core)
        receipt = {}
        R.BASIS.basis_files(ROOT,self.stage,receipt)
        R.BASIS.build_bases(self.core,self.options,self.molecule,receipt)
        self.core.built.clear()
        cfg = json.loads((HERE/"pilot.json").read_text())
        binding = dict(native=dict(source=str(ROOT),commit=R.OLD.BUILD_COMMIT,tree=R.OLD.BUILD_TREE,
                            stage=str(self.stage),core_sha256=R.OLD.CORE_SHA),
            driver=dict(source=R.PLAN["prior_driver_source"],commit=R.PLAN["prior_driver_commit"],
                tree=R.PLAN["prior_driver_tree"],files_sha256=R.PLAN["prior_driver_files"]),
            fixture_sha256=cfg["fixture_sha256"],config_sha256=R.sha(HERE/"pilot.json"))
        self.original = dict(schema="ccasp-benzene-native-basis-admission/1",
            status="basis_metadata_only_complete",no_scf=True,binding=binding,admission=receipt)
        self.probe = self.root/"toy-probe.json"
        self.probe_sha = R.write(self.probe,self.original)
        patch = mock.patch.dict(R.PLAN,dict(prior_probe=str(self.probe),prior_probe_sha256=self.probe_sha,
            fingerprints={role:receipt[role]["fingerprint_sha256"] for role in ("main","auxiliary")}))
        patch.start(); self.addCleanup(patch.stop)
        self.new = dict(source="/TOY/new-driver",commit="1"*40,tree="2"*40)
        self.runtime_sha = "3"*64
        self.derivative = self.root/"toy-derived.json"
        self.derivative_sha = R.write(self.derivative,R.derived_document(self.new,self.runtime_sha))

    def test_derivative_preserves_old_binding_and_admits_exact_toy_fingerprints(self):
        builder = mock.Mock(return_value="retained toy recipe")
        audit = {}
        _,part = R.admit_basis(self.core,self.options,self.molecule,audit,self.new,self.runtime_sha,
                               self.derivative,self.derivative_sha,builder)
        self.assertEqual(part,"retained toy recipe")
        self.assertEqual(len(self.core.built),2)
        builder.assert_called_once()
        self.assertEqual(R.sha(self.probe),self.probe_sha)
        self.assertEqual(audit["main"],self.original["admission"]["main"])
        self.assertEqual(audit["auxiliary"],self.original["admission"]["auxiliary"])
        self.assertTrue(audit["derivative_admission"]["no_extra_probe_job"])
        self.assertEqual(audit["auxiliary_puream_scope"]["before"],audit["auxiliary_puream_scope"]["after"])

    def test_wrong_new_driver_or_runtime_binding_refuses_before_basis_build(self):
        for new,runtime_sha in ((dict(self.new,commit="4"*40),self.runtime_sha),(self.new,"5"*64)):
            with self.subTest(new=new):
                with self.assertRaisesRegex(ValueError,"derivative admission"):
                    R.admit_basis(self.core,self.options,self.molecule,{},new,runtime_sha,
                        self.derivative,self.derivative_sha,mock.Mock())
                self.assertFalse(self.core.built)

    def test_corrupt_original_or_derivative_bytes_refuse_before_basis_build(self):
        with self.assertRaisesRegex(ValueError,"hash/type/size"):
            R.admit_basis(self.core,self.options,self.molecule,{},self.new,self.runtime_sha,
                self.derivative,"0"*64,mock.Mock())
        self.probe.write_bytes(self.probe.read_bytes()+b" ")
        with self.assertRaisesRegex(ValueError,"hash/type/size"):
            R.admit_basis(self.core,self.options,self.molecule,{},self.new,self.runtime_sha,
                self.derivative,self.derivative_sha,mock.Mock())
        self.assertFalse(self.core.built)

    def test_old_driver_binding_is_checked_entirely_even_after_repinning_toy(self):
        self.original["binding"]["driver"]["commit"] = "e"*40
        self.probe.write_text(json.dumps(self.original))
        with mock.patch.dict(R.PLAN,{"prior_probe_sha256":R.sha(self.probe)}):
            with self.assertRaisesRegex(ValueError,"entire original"):
                R.derived_document(self.new,self.runtime_sha)
        self.assertFalse(self.core.built)

    def test_same_counts_altered_toy_shells_refuse_before_recipe_or_scf(self):
        self.core.aux_shift = .125
        builder = mock.Mock()
        with self.assertRaisesRegex(ValueError,"exact ordered native shell"):
            R.admit_basis(self.core,self.options,self.molecule,{},self.new,self.runtime_sha,
                self.derivative,self.derivative_sha,builder)
        builder.assert_not_called()


if __name__ == "__main__":
    unittest.main()
