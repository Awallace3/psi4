"""CPU native-resource graph/failure spies. No SDK/CUDA/DFMO execution."""
import ctypes as ct
import gc
import importlib
import math
import os
from pathlib import Path
import sys
import types
import unittest
import weakref
from unittest.mock import Mock, patch

import numpy as np

import test_cuda_backend as cpu
import test_cuest_basis as basis_fakes
import test_cuest_factory_abi as api_fakes

SAPT = Path(__file__).resolve().parents[2]/"psi4/driver/procrouting/sapt"
PACKAGE = "_disp0_plan_owner_fixtures"
package = types.ModuleType(PACKAGE); package.__path__ = [str(SAPT)]
sys.modules[PACKAGE] = package
owner_module = importlib.import_module(PACKAGE+".sapt_cuest_plan_owner")
abi = owner_module.abi
runtime = importlib.import_module(PACKAGE+".sapt_cuda_runtime")


class Session(cpu.CPUSession):
    retain_native_owner = runtime.CUDASession.retain_native_owner
    release_native_owner = runtime.CUDASession.release_native_owner
    rt = staticmethod(runtime.CUDASession.rt)
    def __init__(self):
        super().__init__(None)
        self.native_owners,self.scope_depth = [],0
        self.native_bytes,self.stream,self.device = 4096,7,0
        self.blas_path,self.blas_sha256 = "CPU BLAS","0"*64
        self.runtime = types.SimpleNamespace(cudaGetDevice=lambda:(0,0))
        self.syncs,self.fail_sync = 0,False
        self.device_workspace_limit = 1024
    def empty(self,shape,*,dtype=np.float64,zero=False):
        if self.live_bytes + math.prod(shape)*np.dtype(dtype).itemsize > self.device_workspace_limit:
            raise MemoryError("CPU fixture device workspace bound")
        return super().empty(shape,dtype=dtype,zero=zero)
    def synchronize_device(self):
        self.syncs += 1
        if self.fail_sync: raise RuntimeError("CPU fixture synchronization failed")


class Native(api_fakes.Native):
    def __init__(self,session):
        super().__init__()
        self.session = session
        self.objects,self.events,self.shell_inputs = {},[],[]
        self.destroy_attempts,self.host_workspace_refs = [],{}
        self.next_handle = 1000
        self.fail_at,self.fail_destroy = None,None
        self.ambiguous_failure,self.bad_nbf = False,False
        self.success_null_at,self.huge_device_query = None,None
        self.version = (0,2,50)
        self.cuestParametersCreate.side_effect = lambda kind,out:self.create("parameters"+str(kind),out)
        self.cuestParametersDestroy.side_effect = lambda kind,h:self.destroy("parameters"+str(kind),h)
        self.cuestCreate.side_effect = lambda params,out:self.create("context",out,[params.value])
        self.cuestDestroy.side_effect = lambda h:self.destroy("context",h)
        self.cuestAOShellCreate.side_effect = self.shell
        self.cuestAOShellDestroy.side_effect = lambda h:self.destroy("shell",h)
        self.cuestAOBasisCreateWorkspaceQuery.side_effect = lambda *a:self.workspace_query("basis",a)
        self.cuestAOBasisCreate.side_effect = self.basis
        self.cuestAOBasisDestroy.side_effect = lambda h:self.destroy("basis",h)
        self.cuestAOPairListCreateWorkspaceQuery.side_effect = lambda *a:self.workspace_query("pair",a)
        self.cuestAOPairListCreate.side_effect = self.pair
        self.cuestAOPairListDestroy.side_effect = lambda h:self.destroy("pair",h)
        self.cuestDFIntPlanCreateWorkspaceQuery.side_effect = lambda *a:self.workspace_query("plan",a)
        self.cuestDFIntPlanCreate.side_effect = self.plan
        self.cuestDFIntPlanDestroy.side_effect = lambda h:self.destroy("plan",h)
        self.cuestQuery.side_effect = self.query_basis
        for i,name in enumerate(("Major","Minor","Patch")):
            def getter(h,out,i=i):
                out._obj.value = self.version[i]
                return 0
            getattr(self,"cuestGet"+name+"Version").side_effect = getter
    def create(self,kind,out,deps=(),data=None,workspace=None):
        self.events.append("create:"+kind)
        if self.success_null_at == "create:"+kind: return 0
        failed = self.fail_at == "create:"+kind
        if not failed or self.ambiguous_failure:
            handle = self.next_handle; self.next_handle += 1
            out._obj.value = handle
            current = self.session.native_owners[-1].resources[-1]
            inputs = tuple(weakref.ref(value) for value in current.keepalive
                           if isinstance(value,(np.ndarray,ct.Array)))
            for resource in self.session.native_owners[-1].resources:
                for value in resource.keepalive:
                    if isinstance(value,dict) and isinstance(value.get("host"),np.ndarray):
                        array = value["host"]
                        self.host_workspace_refs[array.ctypes.data] = weakref.ref(array)
            self.objects[handle] = dict(kind=kind,deps=set(deps),data=data,
                                       workspace=workspace,input_refs=inputs)
        return 17 if failed else 0
    def destroy(self,kind,handle):
        self.events.append("destroy:"+kind)
        self.destroy_attempts.append((kind,handle.value))
        if self.fail_destroy == kind: return 19
        value = handle.value
        assert self.objects[value]["kind"] == kind
        assert not any(value in item["deps"] for key,item in self.objects.items() if key != value)
        for ref in self.objects[value]["input_refs"]:
            assert ref() is not None, "native host input released before destruction"
        workspaces = self.objects[value]["workspace"]
        for workspace in workspaces or ():
            if workspace.device_bytes:
                assert any(a.pointer == workspace.device and not a.freed for a in self.session.allocations), \
                    "persistent/creation-temporary device workspace freed before native destruction"
            if workspace.host_bytes:
                assert self.host_workspace_refs[workspace.host]() is not None, \
                    "persistent/creation-temporary host workspace freed before native destruction"
        del self.objects[value]
        return 0
    def shell(self,context,pure,am,nprimitive,exps,coefficients,params,out):
        e = np.ctypeslib.as_array(exps,shape=(nprimitive,)).copy()
        c = np.ctypeslib.as_array(coefficients,shape=(nprimitive,)).copy()
        self.shell_inputs.append((pure,am,e,c))
        return self.create("shell",out,[context.value,params.value],
                           dict(pure=bool(pure),am=am,nprimitive=nprimitive))
    def workspace_query(self,kind,args):
        self.events.append("query:"+kind)
        if self.fail_at == "query:"+kind: return 23
        persistent,temporary,out = args[-3:]
        assert out is None
        persistent._obj.host_bytes,persistent._obj.device_bytes = 8,16
        if self.huge_device_query == kind: persistent._obj.device_bytes = 4096
        temporary._obj.host_bytes,temporary._obj.device_bytes = 4,8
        return 0
    def basis(self,context,ncenter,counts,handles,params,persistent,temporary,out):
        ids = list(handles)
        assert len(ids) == sum(counts)
        shells = [self.objects[h]["data"] for h in ids]
        pure = all(s["pure"] for s in shells)
        cart = all(not s["pure"] for s in shells)
        ncart = sum((s["am"]+1)*(s["am"]+2)//2 for s in shells)
        nbf = sum(2*s["am"]+1 if s["pure"] else (s["am"]+1)*(s["am"]+2)//2 for s in shells)
        attrs = [int(pure),int(cart),int(not pure and not cart),ncenter,len(ids),nbf,ncart,
                 sum(s["nprimitive"] for s in shells),max(s["am"] for s in shells)]
        return self.create("basis",out,[context.value,params.value,*ids],
                           attrs,(persistent._obj,temporary._obj))
    def query_basis(self,context,kind,basis,attribute,out,size):
        assert kind == 2
        assert size == (4 if attribute < 3 else 8)
        value = self.objects[basis.value]["data"][attribute]
        out._obj.value = value + (1 if self.bad_nbf and attribute == 5 else 0)
        return 0
    def pair(self,context,basis,ncenter,xyz,threshold,params,persistent,temporary,out):
        assert threshold == 0.0
        actual = np.ctypeslib.as_array(xyz,shape=(3*ncenter,))
        np.testing.assert_array_equal(actual,[0.,0.,0.,1.5,-.3,.7])
        return self.create("pair",out,[context.value,basis.value,params.value],
                           workspace=(persistent._obj,temporary._obj))
    def plan(self,context,primary,auxiliary,pair,params,persistent,temporary,out):
        return self.create("plan",out,[context.value,primary.value,auxiliary.value,pair.value,params.value],
                           workspace=(persistent._obj,temporary._obj))


class PlanOwner(unittest.TestCase):
    def fixture(self,**kwargs):
        session = Session(); native = Native(session)
        abi.configure_factory_abi(native)
        api = abi._FactoryAPI(session,native,{},{"components":[]},{})
        maps = {"cuda_mapped_sha256":{session.blas_path:session.blas_sha256}}
        patcher = patch.object(abi,"audit_factory_maps",return_value=maps)
        patcher.start(); self.addCleanup(patcher.stop)
        primary = basis_fakes.Basis()
        auxiliary = basis_fakes.Basis([basis_fakes.Shell(0,0),basis_fakes.Shell(0,1),
                                      basis_fakes.Shell(1,0)])
        owner = owner_module.DiagnosticPlanOwner(
            api,primary,auxiliary,coefficient_candidate="UNVALIDATED_PSI4_COEF",**kwargs)
        return owner,session,native,primary,auxiliary

    def test_full_graph_native_policy_dimensions_candidate_coefficients_and_accounting(self):
        owner,session,native,primary,auxiliary = self.fixture()
        with owner as active:
            self.assertIs(active,owner)
            self.assertIs(session.native_owners[0],owner)
            self.assertGreater(owner.handle,0); self.assertGreater(owner.plan,0)
            self.assertEqual(owner.version,(0,2,50))
            report = owner.report()
            self.assertFalse(report["production_provider_admitted"])
            self.assertFalse(report["basis_layout_metric_rank_admitted"])
            self.assertFalse(report["opaque_native_internals_measured"])
            self.assertEqual(report["known_persistent_device_workspace_bytes"],64)
            self.assertEqual(report["known_retained_creation_temporary_device_bytes"],32)
            self.assertEqual(report["known_host_workspace_bytes"],48)
            self.assertEqual(session.live_bytes,96)
            self.assertEqual(report["plan_parameter_readback"]["fitting_algorithm"],1)
            np.testing.assert_array_equal(native.shell_inputs[0][3],[.31,-.8])
            native.cuestDFMOIntegralsCompute.assert_not_called()
            native.cuestDFMOIntegralsComputeWorkspaceQuery.assert_not_called()
        self.assertEqual(native.objects,{})
        self.assertEqual(session.native_owners,[])
        self.assertEqual(session.live_bytes,0)
        self.assertTrue(owner.cleanup_complete)
        with self.assertRaisesRegex(RuntimeError,"Closed/invalidated"): _ = owner.plan
        before = list(native.events); owner.close(); self.assertEqual(native.events,before)

    def test_create_and_workspace_query_failures_clean_only_known_resources(self):
        for phase in ("create:parameters0","create:parameters1","create:parameters2",
                      "create:parameters3","create:parameters5","create:context","create:shell","create:basis",
                      "create:pair","create:plan","query:basis","query:pair","query:plan"):
            owner,session,native,*_ = self.fixture()
            native.fail_at = phase
            with self.subTest(phase=phase),self.assertRaisesRegex(RuntimeError,"failed"):
                with owner: self.fail("failed factory yielded an owner")
            self.assertEqual(native.objects,{})
            self.assertEqual(session.live_bytes,0)
            self.assertEqual(session.native_owners,[])
            self.assertTrue(owner.cleanup_complete)
            self.assertTrue(session.failed)
            with self.assertRaisesRegex(RuntimeError,"Closed/invalidated"): owner.build()

    def test_wrong_native_version_or_basis_dimension_rejects_and_unwinds(self):
        for failure in ("version","dimension"):
            owner,session,native,*_ = self.fixture()
            if failure == "version": native.version = (0,2,2)
            else: native.bad_nbf = True
            with self.subTest(failure=failure),self.assertRaises(RuntimeError):
                with owner: self.fail("unadmitted native state")
            self.assertEqual(native.objects,{})
            self.assertEqual(session.live_bytes,0)
            self.assertTrue(owner.cleanup_complete)

    def test_host_budget_failure_releases_previously_created_native_resources(self):
        owner,session,native,*_ = self.fixture(host_workspace_limit=1)
        with self.assertRaisesRegex(MemoryError,"host workspace"):
            with owner: self.fail("budget failure yielded")
        self.assertEqual(native.objects,{})
        self.assertEqual(session.live_bytes,0)
        self.assertEqual(session.native_owners,[])

    def test_ambiguous_nonnull_failure_output_retains_every_dependency_without_guessing(self):
        owner,session,native,*_ = self.fixture()
        native.fail_at,native.ambiguous_failure = "create:plan",True
        with self.assertRaisesRegex(RuntimeError,"create failed: 17") as error:
            with owner: self.fail("ambiguous creation yielded")
        self.assertIn("speculative cleanup",error.exception.__notes__[0])
        self.assertFalse(owner.cleanup_complete)
        self.assertTrue(owner.cleanup_failed)
        self.assertIn("DF plan",owner.report()["ambiguous_create_outputs"])
        self.assertTrue(native.objects)
        self.assertGreater(session.live_bytes,0)
        self.assertIs(session.native_owners[0],owner)
        self.assertFalse(any(e.startswith("destroy:") for e in native.events))
        with self.assertRaisesRegex(RuntimeError,"explicit retry_cleanup"): owner.close()
        with self.assertRaisesRegex(RuntimeError,"speculative cleanup"): owner.retry_cleanup()
        self.assertFalse(any(e.startswith("destroy:") for e in native.events))

    def test_failed_destroy_retains_graph_and_requires_explicit_cleanup_only_retry(self):
        owner,session,native,*_ = self.fixture()
        owner.build(); native.fail_destroy = "plan"
        with self.assertRaisesRegex(RuntimeError,"DF plan destroy failed"): owner.close()
        self.assertEqual(native.events[-1],"destroy:plan")
        self.assertGreater(session.live_bytes,0)
        before = list(native.events)
        with self.assertRaisesRegex(RuntimeError,"explicit retry_cleanup"): owner.close()
        self.assertEqual(native.events,before)
        native.fail_destroy = None
        owner.retry_cleanup()
        self.assertTrue(owner.cleanup_complete)
        self.assertEqual(native.objects,{})
        self.assertEqual(session.live_bytes,0)
        self.assertEqual(native.events.count("destroy:plan"),2)
        with self.assertRaisesRegex(RuntimeError,"Closed/invalidated"): owner.build()

    def test_failed_synchronization_destroys_nothing_until_explicit_cleanup_retry(self):
        owner,session,native,*_ = self.fixture()
        owner.build(); session.fail_sync = True
        with self.assertRaisesRegex(RuntimeError,"synchronization failed"): owner.close()
        self.assertFalse(any(e.startswith("destroy:") for e in native.events))
        self.assertIs(session.native_owners[0],owner)
        session.fail_sync = False
        owner.retry_cleanup()
        self.assertTrue(owner.cleanup_complete)
        self.assertEqual(native.objects,{})

    def test_outer_primary_error_is_preserved_when_native_destruction_also_fails(self):
        owner,session,native,*_ = self.fixture()
        with self.assertRaisesRegex(ValueError,"original computation") as error:
            with owner:
                native.fail_destroy = "plan"
                raise ValueError("original computation")
        self.assertIn("DF plan destroy failed",error.exception.__notes__[0])
        self.assertFalse(owner.cleanup_complete)
        self.assertGreater(session.live_bytes,0)

    def test_changed_basis_inputs_invalidate_handles_without_computation_retry(self):
        owner,session,native,primary,_ = self.fixture()
        owner.build(); primary.shells[0].coefficients[0] += .1
        with self.assertRaisesRegex(ValueError,"changed"): owner.assert_live()
        self.assertTrue(owner.failed); self.assertTrue(session.failed)
        primary.shells[0].coefficients[0] -= .1
        with self.assertRaisesRegex(RuntimeError,"Closed/invalidated"): _ = owner.handle
        owner.close()
        self.assertTrue(owner.cleanup_complete)

    def test_explicit_candidate_allowance_single_owner_and_temporary_scope_guards(self):
        owner,session,native,primary,auxiliary = self.fixture()
        with self.assertRaisesRegex(RuntimeError,"already has a plan owner"):
            owner_module.DiagnosticPlanOwner(
                owner.api,primary,auxiliary,coefficient_candidate="UNVALIDATED_PSI4_COEF")
        owner.close()
        for allowance,candidate in ((4096,"guessed normalization"),(0,"UNVALIDATED_PSI4_COEF"),
                                    (True,"UNVALIDATED_PSI4_COEF")):
            session.native_bytes = allowance
            api = abi._FactoryAPI(session,native,{},{"components":[]},{})
            with self.subTest(allowance=allowance,candidate=candidate),self.assertRaises(ValueError):
                owner_module.DiagnosticPlanOwner(api,primary,auxiliary,coefficient_candidate=candidate)
            self.assertEqual(session.native_owners,[])
        session.native_bytes = 4096
        api = abi._FactoryAPI(session,native,{},{"components":[]},{})
        with runtime.CUDASession.scope(session):
            with self.assertRaisesRegex(ValueError,"outside temporary"):
                owner_module.DiagnosticPlanOwner(
                    api,primary,auxiliary,coefficient_candidate="UNVALIDATED_PSI4_COEF")
        native.cuestCreate.assert_not_called()
        native.cuestParametersCreate.assert_not_called()

    def test_null_success_output_is_ambiguous_and_never_passed_to_version_getters(self):
        owner,session,native,*_ = self.fixture()
        native.success_null_at = "create:context"
        with self.assertRaisesRegex(RuntimeError,"null object") as error:
            with owner: self.fail("null native handle yielded")
        self.assertIn("speculative cleanup",error.exception.__notes__[0])
        self.assertEqual(owner.report()["ambiguous_create_outputs"],["context"])
        native.cuestGetMajorVersion.assert_not_called()
        self.assertFalse(any(e.startswith("destroy:") for e in native.events))
        self.assertIs(session.native_owners[0],owner)

    def test_cleanup_map_device_and_pid_failures_retain_resources_without_native_destroy(self):
        for failure in ("maps","device","pid"):
            owner,session,native,*_ = self.fixture()
            owner.build()
            if failure == "maps":
                guard = patch.object(abi,"audit_factory_maps",side_effect=RuntimeError("foreign mapping"))
            elif failure == "device":
                guard = patch.object(session.runtime,"cudaGetDevice",return_value=(0,1))
            else:
                guard = patch.object(owner_module.os,"getpid",return_value=owner.pid+1)
            with self.subTest(failure=failure),guard,self.assertRaises(RuntimeError):
                owner.close()
            self.assertFalse(any(e.startswith("destroy:") for e in native.events))
            self.assertGreater(session.live_bytes,0)
            owner.retry_cleanup()
            self.assertEqual(native.objects,{})
            self.assertEqual(session.live_bytes,0)

    def test_device_workspace_budget_failure_releases_partial_host_and_native_owners(self):
        owner,session,native,*_ = self.fixture()
        native.huge_device_query = "plan"
        with self.assertRaisesRegex(MemoryError,"device workspace bound"):
            with owner: self.fail("device budget failure yielded")
        self.assertEqual(native.objects,{})
        self.assertEqual(session.live_bytes,0)
        self.assertTrue(owner.cleanup_complete)

    def test_workspace_free_failure_retains_parents_without_redestroying_finished_child(self):
        owner,session,native,*_ = self.fixture()
        owner.build()
        allocation = session.allocations[-1]
        original = allocation.free
        allocation.free = Mock(side_effect=RuntimeError("workspace free failure"))
        with self.assertRaisesRegex(RuntimeError,"workspace free failure"): owner.close()
        self.assertEqual(native.events.count("destroy:plan"),1)
        self.assertNotIn("destroy:context",native.events)
        self.assertFalse(owner.cleanup_complete)
        self.assertGreater(session.live_bytes,0)
        allocation.free = original
        owner.retry_cleanup()
        self.assertEqual(native.events.count("destroy:plan"),1)
        self.assertEqual(native.objects,{})
        self.assertEqual(session.live_bytes,0)

    def test_every_destructor_stage_can_retain_and_explicitly_resume_parent_cleanup(self):
        for kind in ("pair","basis","shell","context","parameters0","parameters1",
                     "parameters2","parameters3","parameters5"):
            owner,session,native,*_ = self.fixture()
            owner.build(); native.fail_destroy = kind
            with self.subTest(kind=kind),self.assertRaisesRegex(RuntimeError,"destroy failed"):
                owner.close()
            self.assertIs(session.native_owners[0],owner)
            self.assertFalse(owner.cleanup_complete)
            native.fail_destroy = None
            owner.retry_cleanup()
            self.assertEqual(native.objects,{})
            self.assertEqual(session.live_bytes,0)
            self.assertTrue(owner.cleanup_complete)

    def test_df_parameter_configuration_or_readback_failure_never_constructs_plan(self):
        for failure in ("configure","readback"):
            owner,session,native,*_ = self.fixture()
            if failure == "configure":
                original = native.cuestParametersConfigure.side_effect
                def configure(kind,*args):
                    return 27 if kind == 5 else original(kind,*args)
                native.cuestParametersConfigure.side_effect = configure
            else:
                native.bad_readback = 3
            with self.subTest(failure=failure),self.assertRaises(RuntimeError):
                with owner: self.fail("wrong plan parameter policy yielded")
            native.cuestDFIntPlanCreate.assert_not_called()
            self.assertEqual(native.objects,{})
            self.assertEqual(session.live_bytes,0)

    def test_post_destroy_sync_failure_retains_inputs_and_parents_without_redestroy(self):
        owner,session,native,*_ = self.fixture()
        owner.build()
        original = session.synchronize_device
        def synchronize():
            if native.events and native.events[-1] == "destroy:plan":
                raise RuntimeError("post-destroy synchronization failure")
            original()
        session.synchronize_device = synchronize
        with self.assertRaisesRegex(RuntimeError,"post-destroy synchronization"): owner.close()
        self.assertEqual(owner.report()["retired_waiting_for_device_sync"],["DF plan"])
        self.assertEqual(native.events.count("destroy:plan"),1)
        self.assertGreater(session.live_bytes,0)
        session.synchronize_device = original
        owner.retry_cleanup()
        self.assertEqual(native.events.count("destroy:plan"),1)
        self.assertEqual(native.objects,{})
        self.assertEqual(owner.retired_waiting_for_sync,[])
        self.assertEqual(session.live_bytes,0)

    def test_unobserved_create_or_destroy_status_never_triggers_speculative_cleanup(self):
        owner,session,native,*_ = self.fixture()
        original = native.cuestCreate.side_effect
        def interrupted_create(params,out):
            original(params,out)
            raise RuntimeError("unobserved create status")
        native.cuestCreate.side_effect = interrupted_create
        with self.assertRaisesRegex(RuntimeError,"unobserved create status"):
            with owner: self.fail("ambiguous constructor yielded")
        self.assertIn("context",owner.report()["ambiguous_create_outputs"])
        self.assertFalse(any(e.startswith("destroy:") for e in native.events))
        with self.assertRaisesRegex(RuntimeError,"speculative cleanup"): owner.retry_cleanup()
        owner,session,native,*_ = self.fixture()
        owner.build()
        original = native.cuestDFIntPlanDestroy.side_effect
        def interrupted_destroy(handle):
            original(handle)
            raise RuntimeError("unobserved destroy status")
        native.cuestDFIntPlanDestroy.side_effect = interrupted_destroy
        with self.assertRaisesRegex(RuntimeError,"unobserved destroy status"): owner.close()
        self.assertIn("DF plan",owner.report()["ambiguous_destroy_outcomes"])
        with self.assertRaisesRegex(RuntimeError,"speculative cleanup"): owner.retry_cleanup()
        self.assertEqual(native.events.count("destroy:plan"),1)
        self.assertGreater(session.live_bytes,0)

    def test_precreated_owner_cannot_delay_build_or_enter_into_temporary_scope(self):
        for entry in ("build","enter"):
            owner,session,native,*_ = self.fixture()
            native.fail_destroy = "plan"  # No native destructor may be reached.
            with self.subTest(entry=entry),runtime.CUDASession.scope(session):
                with self.assertRaisesRegex(ValueError,"construction.*outside temporary"):
                    if entry == "build":
                        owner.build()
                    else:
                        with owner: self.fail("temporary-scope construction yielded")
                native.cuestCreate.assert_not_called()
                native.cuestParametersCreate.assert_not_called()
                self.assertEqual(session.live_bytes,0)
                self.assertEqual(native.objects,{})
            self.assertEqual(session.scope_depth,0)
            owner.close()

    def test_existing_native_workspaces_survive_cleanup_failure_inside_later_scope(self):
        owner,session,native,*_ = self.fixture()
        owner.build()
        persistent_bytes = session.live_bytes
        with runtime.CUDASession.scope(session):
            scratch = session.empty((7,),dtype=np.uint8)
            native.fail_destroy = "plan"
            with self.assertRaisesRegex(RuntimeError,"destroy failed"): owner.close()
            self.assertEqual(session.live_bytes,persistent_bytes+7)
        self.assertTrue(scratch.allocation.freed)
        self.assertEqual(session.live_bytes,persistent_bytes)
        self.assertIs(session.native_owners[0],owner)
        self.assertTrue(native.objects)
        native.fail_destroy = None
        owner.retry_cleanup()
        self.assertEqual(session.live_bytes,0)

    def test_shell_basis_and_pair_host_inputs_survive_gc_until_retry_synchronization(self):
        for kind,label,input_count in (("shell","auxiliary shell2",2),
                                       ("basis","auxiliary basis",2),
                                       ("pair","primary pair list",1)):
            owner,session,native,*_ = self.fixture()
            owner.build()
            def capture():
                resource = next(r for r in owner.resources if r.label == label)
                return resource.handle.value,[weakref.ref(v) for v in resource.keepalive[:input_count]]
            handle,refs = capture()
            # Mock call history must not accidentally keep the real inputs alive.
            for value in vars(native).values():
                if isinstance(value,Mock): value.reset_mock()
            original = session.synchronize_device
            def synchronize():
                if native.events and native.events[-1] == "destroy:"+kind:
                    raise RuntimeError("post-destroy input synchronization failure")
                original()
            session.synchronize_device = synchronize
            try:
                owner.close()
            except RuntimeError as error:
                self.assertIn("post-destroy input synchronization",str(error))
                error.__traceback__ = None
            else:
                self.fail("expected post-destroy synchronization failure")
            gc.collect()
            with self.subTest(kind=kind):
                self.assertTrue(all(ref() is not None for ref in refs))
                self.assertEqual(owner.report()["retired_waiting_for_device_sync"],[label])
                self.assertEqual(native.destroy_attempts.count((kind,handle)),1)
            session.synchronize_device = original
            owner.retry_cleanup()
            gc.collect()
            self.assertTrue(all(ref() is None for ref in refs))
            self.assertEqual(native.destroy_attempts.count((kind,handle)),1)


if __name__ == "__main__":
    unittest.main()
