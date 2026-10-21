"""CPU ctypes/parameter/map spies, NOT actual SDK or native factory execution."""
import ctypes as ct
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch

import test_cuest250_identity as identity_fixtures

SAPT = Path(__file__).resolve().parents[2]/"psi4/driver/procrouting/sapt"
PACKAGE = "_disp0_factory_abi_fixtures"
package = types.ModuleType(PACKAGE); package.__path__ = [str(SAPT)]
sys.modules[PACKAGE] = package
abi = importlib.import_module(PACKAGE+".sapt_cuest_factory_abi")


class Native:
    def __init__(self):
        base = ("cuestParametersCreate","cuestParametersDestroy",
                "cuestDFMOIntegralsCompute","cuestDFMOIntegralsComputeWorkspaceQuery",
                "cuestGetMajorVersion","cuestGetMinorVersion","cuestGetPatchVersion",
                "cuestGetMathMode")
        for name in (*base,*abi.FACTORY_ABI):
            setattr(self,name,Mock(return_value=0))
        self.values,self.configured = {},[]
        self.query_code,self.configure_code = 0,0
        self.bad_readback = None
        self.math_mode,self.math_code,self.ignore_set = 0,0,False
        self.cuestParametersConfigure.side_effect = self.configure
        self.cuestParametersQuery.side_effect = self.query
        self.cuestSetMathMode.side_effect = self.set_math
        self.cuestGetMathMode.side_effect = self.get_math
    def configure(self,kind,parameters,attribute,value,size):
        raw = ct.string_at(value,size)
        self.configured.append((kind,parameters.value,attribute,size,raw))
        self.values[kind,parameters.value,attribute] = raw
        return self.configure_code
    def query(self,kind,parameters,attribute,value,size):
        if self.query_code: return self.query_code
        raw = self.values[kind,parameters.value,attribute]
        assert len(raw) == size
        ct.memmove(value,b"\0"*size if attribute == self.bad_readback else raw,size)
        return 0
    def set_math(self,handle,mode):
        assert isinstance(handle,ct.c_void_p)
        if not self.ignore_set: self.math_mode = mode
        return self.math_code
    def get_math(self,handle,out):
        out._obj.value = self.math_mode
        return self.math_code


class FactoryABI(unittest.TestCase):
    def fixture(self):
        native = Native()
        abi.configure_factory_abi(native)
        session = types.SimpleNamespace(check=Mock(),blas_path="CPU BLAS",blas_sha256="0"*64)
        return abi._FactoryAPI(session,native,{},{"components":[]},{}),native

    def test_constructor_and_workspace_abis_preserve_pointer_levels_and_uint64_counts(self):
        _,native = self.fixture()
        self.assertEqual(native.cuestCreate.argtypes,[abi.H,abi.PH])
        self.assertEqual(native.cuestAOShellCreate.argtypes,
                         [abi.H,ct.c_int32,ct.c_uint64,ct.c_uint64,abi.PD,abi.PD,abi.H,abi.PH])
        self.assertEqual(native.cuestAOBasisCreate.argtypes,
                         [abi.H,ct.c_uint64,abi.PU64,abi.PH,abi.H,abi.PW,abi.PW,abi.PH])
        self.assertEqual(native.cuestAOPairListCreateWorkspaceQuery.argtypes,
                         [abi.H,abi.H,ct.c_uint64,abi.PD,ct.c_double,abi.H,abi.PQ,abi.PQ,abi.PH])
        self.assertEqual(native.cuestDFIntPlanCreate.argtypes,
                         [abi.H]*5+[abi.PW,abi.PW,abi.PH])
        self.assertEqual((ct.sizeof(abi._Descriptor),ct.sizeof(abi._Workspace)),(16,32))
        for name in abi.FACTORY_ABI:
            self.assertIs(getattr(native,name).restype,ct.c_int)
        del native.cuestDFIntPlanCreate
        with self.assertRaisesRegex(RuntimeError,"factory capability missing"):
            abi.configure_factory_abi(native)

    def test_context_pointer_attributes_are_addresses_of_typed_values(self):
        api,native = self.fixture()
        result = api.configure_context_parameters(123,stream=0xabcdef,jit_compile_threads=1)
        self.assertEqual(result,dict(cuda_stream=0xabcdef,cublas=None,cusolver=None,jit_compile_threads=1))
        self.assertEqual([(k,p,a,s) for k,p,a,s,_ in native.configured],
                         [(0,123,0,8),(0,123,1,8),(0,123,2,8),(0,123,8,4)])
        self.assertEqual(int.from_bytes(native.configured[0][-1],sys.byteorder),0xabcdef)
        native.cuestCreate.assert_not_called()

    def test_df_policy_explicit_matrixpower_full_coulomb_relative_cutoff_readback(self):
        api,native = self.fixture()
        result = api.configure_df_plan_parameters(456)
        self.assertEqual(result,dict(fitting_cutoff=1e-12,relative_conditioning=1,
                                    exchange_fraction=1.,fitting_algorithm=1,
                                    lrc_exchange_fraction=0.,lrc_omega=0.,
                                    three_index_integral_direct=0))
        self.assertEqual([r[3] for r in native.configured],[8,4,8,4,8,8,4])
        self.assertEqual([r[2] for r in native.configured],list(range(7)))
        self.assertTrue(all(r[0] == 5 and r[1] == 456 for r in native.configured))
        self.assertEqual(native.cuestParametersQuery.call_count,7)
        native.cuestDFIntPlanCreate.assert_not_called()

    def test_parameter_errors_or_wrong_readback_permanently_disable_execution(self):
        for failure in ("configure","query","readback","exception"):
            api,native = self.fixture()
            if failure == "configure": native.configure_code = 7
            if failure == "query": native.query_code = 8
            if failure == "readback": native.bad_readback = 3
            if failure == "exception": native.cuestParametersQuery.side_effect = RuntimeError("fixture exception")
            with self.subTest(failure=failure),self.assertRaises(RuntimeError):
                api.configure_df_plan_parameters(456)
            self.assertTrue(api.failed)
            calls = native.cuestParametersConfigure.call_count
            with self.assertRaisesRegex(RuntimeError,"Invalidated"):
                api.configure_df_plan_parameters(456)
            self.assertEqual(native.cuestParametersConfigure.call_count,calls)

    def test_owned_context_precision_requires_same_pid_setter_success_and_readback(self):
        api,native = self.fixture()
        self.assertEqual(api.set_owned_context_fp64(123,handle_pid=os.getpid()),1)
        native.cuestCreate.assert_not_called()
        for getter in ("Major","Minor","Patch"):
            getattr(native,"cuestGet"+getter+"Version").assert_not_called()
        for failure in ("setter","readback"):
            api,native = self.fixture()
            if failure == "setter": native.math_code = 9
            else: native.ignore_set = True
            with self.subTest(failure=failure),self.assertRaises(RuntimeError):
                api.set_owned_context_fp64(123,handle_pid=os.getpid())
            self.assertTrue(api.failed)
        api,native = self.fixture()
        with self.assertRaisesRegex(ValueError,"same-process"):
            api.set_owned_context_fp64(123,handle_pid=os.getpid()+1)
        native.cuestSetMathMode.assert_not_called()

    def test_invalid_python_arguments_or_foreign_process_never_call_native(self):
        api,native = self.fixture()
        for value in (True,0,-1,1.5):
            with self.subTest(value=value),self.assertRaises(ValueError):
                api.configure_df_plan_parameters(value)
        for threads in (True,0,257,1.5):
            with self.assertRaises(ValueError):
                api.configure_context_parameters(123,stream=7,jit_compile_threads=threads)
        native.cuestParametersConfigure.assert_not_called()
        api.pid += 1
        with self.assertRaisesRegex(RuntimeError,"foreign-process"):
            api.configure_df_plan_parameters(123)
        self.assertTrue(api.failed)

    def test_map_or_session_failure_disables_existing_api(self):
        api,native = self.fixture()
        with patch.object(abi,"audit_factory_maps",side_effect=RuntimeError("foreign mapping")):
            with self.assertRaisesRegex(RuntimeError,"foreign mapping"): api.assert_identity()
        with self.assertRaisesRegex(RuntimeError,"Invalidated"):
            api.configure_df_plan_parameters(123)
        native.cuestParametersConfigure.assert_not_called()
        api,_ = self.fixture(); api.session.check.side_effect = RuntimeError("changed device")
        with self.assertRaisesRegex(RuntimeError,"changed device"): api.check()
        self.assertTrue(api.failed)

    def test_admission_never_constructs_hidden_version_context_or_invokes_getters(self):
        native = Native()
        session = types.SimpleNamespace(check=Mock(),blas_path="CPU BLAS",blas_sha256="0"*64)
        manifest = {"scope":"CPU ADMISSION SPY"}
        sha = hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(",",":")).encode()).hexdigest()
        maps = {"cuda_mapped_sha256":{session.blas_path:session.blas_sha256}}
        with patch.object(abi.identity,"verify_files",return_value={"native_path":"CPU NATIVE"}),patch.object(
                abi,"library",return_value=(native,"CPU NATIVE")),patch.object(
                abi,"audit_factory_maps",return_value=maps) as audit:
            api = abi.admit_factory_api(session=session,sdk_root="CPU SDK",archive="CPU ARCHIVE",
                                       cuda_manifest=manifest,cuda_manifest_sha256=sha)
        self.assertFalse(api.failed); audit.assert_called_once()
        for name in (*abi.FACTORY_ABI,"cuestParametersCreate","cuestGetMajorVersion",
                     "cuestGetMinorVersion","cuestGetPatchVersion","cuestGetMathMode"):
            getattr(native,name).assert_not_called()
        manifest["scope"] = "MUTATED CALLER"
        self.assertEqual(api.manifest["scope"],"CPU ADMISSION SPY")
        with self.assertRaisesRegex(ValueError,"frozen receipt"):
            abi.admit_factory_api(session=session,sdk_root="",archive="",
                                  cuda_manifest=manifest,cuda_manifest_sha256=sha)

    def test_all_factory_symbols_require_canonical_mapped_ownership(self):
        _,native,proof,manifest,text = identity_fixtures.Identity.fixture(self)
        stat = Path(proof["native_path"]).stat()
        for name,args in abi.FACTORY_ABI.items():
            function = ct.CFUNCTYPE(ct.c_int,*args)(lambda *args:0)
            setattr(native,name,function)
            address = ct.cast(function,ct.c_void_p).value
            text += (f"\n{address:x}-{address+1:x} r-xp 0 "
                     f"{os.major(stat.st_dev):x}:{os.minor(stat.st_dev):x} "
                     f"{stat.st_ino} {proof['native_path']}")
        with patch.object(abi.identity,"NATIVE_SHA256",abi.identity.file_digest(proof["native_path"])):
            result = abi.audit_factory_maps(native,proof,manifest,text=text)
            self.assertEqual(set(result["factory_symbols"]),set(abi.FACTORY_ABI))
            address = ct.cast(native.cuestCreate,ct.c_void_p).value
            wrong = "\n".join(line.replace(proof["native_path"],"/foreign/constructor.so")
                              if line.startswith(f"{address:x}-") else line
                              for line in text.splitlines())
            with self.assertRaisesRegex(RuntimeError,"factory symbol outside canonical0.2.50: cuestCreate"):
                abi.audit_factory_maps(native,proof,manifest,text=wrong)


if __name__ == "__main__":
    unittest.main()
