"""CPU adapter tests for explicit immutable-worker contract selection."""
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import gpu_runtime


class ContractSelection(unittest.TestCase):
    def fixture(self, directory):
        sdk=directory/"sdk"
        native=sdk/"lib/13/libcuest.so.0.2.50"
        native.parent.mkdir(parents=True)
        native.write_bytes(b"native fixture")
        blas=directory/"cuda/libcublas.so.13"
        blas.parent.mkdir()
        blas.write_bytes(b"BLAS fixture")
        config={"cuest_root":str(sdk)}
        manifest={"components":[{"path":str(blas),"sha256":"b"*64}]}
        owner=Mock()
        module=types.SimpleNamespace(current_owned_runtime=Mock(return_value=None),
            admit_owned_runtime=Mock(return_value=owner))
        library=object()
        cuda=types.SimpleNamespace(library=Mock(return_value=(library,str(native))))
        abi=types.SimpleNamespace(__package__="installed.sapt",
            identity=types.SimpleNamespace(NATIVE_SHA256="a"*64),audit_factory_maps=Mock())
        return config,manifest,module,owner,cuda,abi,native,blas,library

    def test_owned_mode_pins_all_files_and_audits_without_version_probe_context(self):
        with tempfile.TemporaryDirectory() as temporary:
            config,manifest,module,owner,cuda,abi,native,blas,library=self.fixture(Path(temporary))
            config["identity_contract"]="process-owned-kernel-read-leases-v1"
            with patch.object(gpu_runtime.importlib,"import_module",return_value=module) as load:
                result=gpu_runtime.activate_identity_contract(config,manifest,cuda,abi)
            self.assertIs(result,owner)
            load.assert_called_once_with("installed.sapt.sapt_cuest_owned_runtime")
            module.admit_owned_runtime.assert_called_once_with(
                {str(native):"a"*64,str(blas):"b"*64},dedicated_worker=True)
            cuda.library.assert_called_once_with(str(native),"a"*64,owner=owner)
            abi.audit_factory_maps.assert_called_once_with(
                library,{"native_path":str(native)},manifest,owner=owner)

    def test_legacy_default_does_not_acquire_leases_or_load_libraries(self):
        with tempfile.TemporaryDirectory() as temporary:
            config,manifest,module,_,cuda,abi,*_=self.fixture(Path(temporary))
            with patch.object(gpu_runtime.importlib,"import_module",return_value=module):
                self.assertIsNone(gpu_runtime.activate_identity_contract(config,manifest,cuda,abi))
            module.admit_owned_runtime.assert_not_called()
            cuda.library.assert_not_called()
            abi.audit_factory_maps.assert_not_called()

    def test_missing_legacy_module_is_allowed_but_owned_mode_or_missing_dependency_is_not(self):
        with tempfile.TemporaryDirectory() as temporary:
            config,manifest,_,_,cuda,abi,*_=self.fixture(Path(temporary))
            name="installed.sapt.sapt_cuest_owned_runtime"
            with patch.object(gpu_runtime.importlib,"import_module",
                    side_effect=ModuleNotFoundError("old coherent stage",name=name)):
                self.assertIsNone(gpu_runtime.activate_identity_contract(config,manifest,cuda,abi))
                config["identity_contract"]="process-owned-kernel-read-leases-v1"
                with self.assertRaisesRegex(RuntimeError,"coherently installed"):
                    gpu_runtime.activate_identity_contract(config,manifest,cuda,abi)
            config.pop("identity_contract")
            with patch.object(gpu_runtime.importlib,"import_module",
                    side_effect=ModuleNotFoundError("broken dependency",name="dependency")):
                with self.assertRaises(ModuleNotFoundError):
                    gpu_runtime.activate_identity_contract(config,manifest,cuda,abi)
            cuda.library.assert_not_called()

    def test_no_downgrade_unknown_policy_or_duplicate_owned_locator(self):
        with tempfile.TemporaryDirectory() as temporary:
            config,manifest,module,owner,cuda,abi,native,*_=self.fixture(Path(temporary))
            module.current_owned_runtime.return_value=owner
            with patch.object(gpu_runtime.importlib,"import_module",return_value=module):
                with self.assertRaisesRegex(RuntimeError,"downgrade"):
                    gpu_runtime.activate_identity_contract(config,manifest,cuda,abi)
                config["identity_contract"]="unknown"
                with self.assertRaisesRegex(ValueError,"Unknown"):
                    gpu_runtime.activate_identity_contract(config,manifest,cuda,abi)
                config["identity_contract"]="process-owned-kernel-read-leases-v1"
                manifest["components"].append({"path":str(native),"sha256":"a"*64})
                with self.assertRaisesRegex(RuntimeError,"Duplicate"):
                    gpu_runtime.activate_identity_contract(config,manifest,cuda,abi)
            module.admit_owned_runtime.assert_not_called()
            cuda.library.assert_not_called()

    def test_unadapted_native_call_is_gated_only_in_owned_mode(self):
        runtime=gpu_runtime.AuditedRuntime.__new__(gpu_runtime.AuditedRuntime)
        self.assertEqual(runtime.guarded_call(pow,3,2),9)
        owner=types.SimpleNamespace(guarded_call=Mock(side_effect=lambda fn,*a,**kw:fn(*a,**kw)))
        runtime.identity_owner=owner
        self.assertEqual(runtime.guarded_call(pow,3,2),9)
        owner.guarded_call.assert_called_once_with(pow,3,2)


if __name__=="__main__":unittest.main()
