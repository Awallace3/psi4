"""Construction schedule/comparison CPU spies, not GPU performance evidence."""
import copy
import ast
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import test_cuest_plan_owner as fakes
import construction_probe as probe


class Session(fakes.Session):
    def __init__(self, **kwargs):
        super().__init__()
        self.cleanup_complete = False
        self.runtime.cudaMemGetInfo = lambda:(0,1024**3-self.live_bytes,2*1024**3)
    def __enter__(self):
        return self
    def __exit__(self,kind,error,tb):
        self.closed = True
        self.cleanup_complete = not self.native_owners and self.live_bytes == 0
        if not self.cleanup_complete and error is None:
            raise RuntimeError("CPU-spy session has live resources")
        return False


class ConstructionProbe(unittest.TestCase):
    def test_water_fixture_is_the_existing_six_atom_gate_control(self):
        tree = ast.parse((HERE.parent/"phoenix_validation/fragment_gate.py").read_text())
        water = next(ast.literal_eval(node.value) for node in tree.body
                     if isinstance(node,ast.Assign)
                     and any(isinstance(target,ast.Name) and target.id == "WATER"
                             for target in node.targets))
        self.assertEqual((HERE/"construction_water.dat").read_text(),water)

    def fixture(self):
        session = Session()
        native = fakes.Native(session)
        fakes.abi.configure_factory_abi(native)
        maps = {"cuda_mapped_sha256":{session.blas_path:session.blas_sha256}}
        guard = patch.object(fakes.abi,"audit_factory_maps",return_value=maps)
        guard.start()
        self.addCleanup(guard.stop)
        def api(**kwargs):
            self.assertIs(kwargs["session"],session)
            return fakes.abi._FactoryAPI(session,native,{},{"components":[]},maps)
        config = {"cuest_root":"unused","cuest_archive":"unused",
                  "cuda_manifest_sha256":"unused","device_memory_limit_bytes":1024**3,
                  "native_persistent_bytes":4096}
        runtime = types.SimpleNamespace(config=config,plans=fakes.owner_module,
            abi=types.SimpleNamespace(admit_factory_api=api),
            cuda=types.SimpleNamespace(CUDASession=lambda **kwargs:session),
            manifest={},blas={"path":"unused","sha256":"unused"},
            require_device_admission=Mock(),device_identity=lambda session:{"uuid":"CPU-spy"})
        primary = fakes.basis_fakes.Basis()
        auxiliary = fakes.basis_fakes.Basis([fakes.basis_fakes.Shell(0,0),
            fakes.basis_fakes.Shell(0,1),fakes.basis_fakes.Shell(1,0)])
        report = {"status":"PASS_CONSTRUCTION_ONLY","jit_compile_threads":4,
            "source_pin":"frozen","runtime_config_sha256":"frozen","geometry_sha256":"frozen",
            "basis":"primary","jkfit":"jk","rifit":"ri",
            "basis_dimensions":{"primary":primary.nbf(),"JKFIT":auxiliary.nbf(),"RIFIT":auxiliary.nbf()},
            "basis_files":{name:{"path":name,"sha256":"frozen"} for name in ("primary","JKFIT","RIFIT")},
            "cache_policy":"unknown native cache"}
        return runtime,session,native,primary,auxiliary,report

    def run_arm(self,threads):
        runtime,session,native,primary,auxiliary,report = self.fixture()
        report["jit_compile_threads"] = threads
        probe.measure_pair(runtime,primary,auxiliary,auxiliary,threads,report)
        return report,session,native

    def test_same_session_two_owned_contexts_reverse_cleanup_no_compute(self):
        report,session,native = self.run_arm(4)
        samples = report["memory_samples"]
        self.assertEqual([r["label"] for r in samples],[
            "session_before_native","JKFIT_live","JKFIT_RIFIT_coexist",
            "after_RIFIT_cleanup","after_JKFIT_cleanup"])
        self.assertEqual([r["tracked_live_bytes"] for r in samples],[0,96,192,96,0])
        self.assertTrue(all(r["sampled_not_peak"] for r in samples))
        self.assertEqual(native.events.count("create:context"),2)
        self.assertEqual(native.events.count("destroy:context"),2)
        self.assertEqual(native.objects,{})
        self.assertEqual(session.native_owners,[])
        for name in ("cuestDFMOIntegralsCompute","cuestDFMOIntegralsComputeWorkspaceQuery"):
            getattr(native,name).assert_not_called()
        for owner in report["owners"].values():
            self.assertEqual(owner["context_parameter_readback"]["jit_compile_threads"],4)
            self.assertTrue(owner["cleanup_complete"])
            self.assertIn("cleanup_total",owner["host_phase_timings"]["phases"])
        self.assertEqual(report["session_cleanup"]["peak_explicit_bytes"],192)

    def test_paired_comparison_requires_same_inputs_policy_readback_and_cleanup(self):
        one,_,_ = self.run_arm(1)
        four,_,_ = self.run_arm(4)
        self.assertEqual(probe.compare([one,four])["status"],"PASS_PAIRED_CONSTRUCTION_ONLY")
        changes = [
            lambda r:r.update(geometry_sha256="changed"),
            lambda r:r.update(basis_dimensions={"primary":999}),
            lambda r:r["basis_files"]["JKFIT"].update(sha256="changed"),
            lambda r:r["owners"]["RIFIT"].update(auxiliary_fingerprint="changed"),
            lambda r:r["owners"]["JKFIT"]["plan_parameter_readback"].update(fitting_algorithm=0),
            lambda r:r["owners"]["RIFIT"]["context_parameter_readback"].update(jit_compile_threads=1),
            lambda r:r["owners"]["RIFIT"].update(cleanup_complete=False),
            lambda r:r["session_cleanup"].update(live_bytes=1),
            lambda r:r.update(status="FAILED"),
        ]
        for index,change in enumerate(changes):
            broken = copy.deepcopy(four)
            change(broken)
            with self.subTest(index=index),self.assertRaises(RuntimeError):
                probe.compare([one,broken])
        with self.assertRaises(RuntimeError):
            probe.compare([one])

    def test_second_plan_query_failure_unwinds_both_contexts_and_keeps_timings(self):
        runtime,session,native,primary,auxiliary,report = self.fixture()
        original = native.cuestDFIntPlanCreateWorkspaceQuery.side_effect
        count = 0
        def query(*args):
            nonlocal count
            count += 1
            return 23 if count == 2 else original(*args)
        native.cuestDFIntPlanCreateWorkspaceQuery.side_effect = query
        with self.assertRaisesRegex(RuntimeError,"workspace query failed: 23"):
            probe.measure_pair(runtime,primary,auxiliary,auxiliary,4,report)
        self.assertEqual(report["owners"]["RIFIT"]["host_phase_timings"]["phases"]["build_total"]["failures"],1)
        self.assertEqual(native.objects,{})
        self.assertEqual(session.live_bytes,0)
        self.assertTrue(session.cleanup_complete)
        self.assertTrue(all(row["cleanup_complete"] for row in report["owners"].values()))

    def test_basis_builder_explicit_spherical_metadata_only(self):
        psi4 = types.SimpleNamespace(geometry=Mock(),core=types.SimpleNamespace(BasisSet=Mock()))
        bases = probe.build_bases(psi4,"geometry","primary","jk","ri")
        psi4.geometry.assert_called_once_with("geometry")
        psi4.geometry.return_value.update_geometry.assert_called_once_with()
        self.assertEqual(len(bases),3)
        calls = psi4.core.BasisSet.build.call_args_list
        self.assertEqual([c.kwargs for c in calls],[{"puream":1}]*3)
        self.assertEqual(calls[1].args[1:],("DF_BASIS_SCF","jk","JKFIT","primary"))
        self.assertEqual(calls[2].args[1:],("DF_BASIS_MP2","ri","RIFIT","primary"))


if __name__ == "__main__":
    unittest.main()
