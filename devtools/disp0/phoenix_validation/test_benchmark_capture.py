"""Snapshot deferral contracts without importing/executing a native Psi4 core."""
import ast
from contextlib import ExitStack, nullcontext
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import numpy as np

from fragment_gate import FIELDS, validate_factor_cache


def nested_function(name, namespace):
    tree = ast.parse(Path(__file__).with_name("fragment_gate.py").read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    node = next(n for n in ast.walk(main) if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                 "isolated-snapshot-contract", "exec"), namespace)
    return namespace[name]


class SnapshotContracts(unittest.TestCase):
    def test_fragment_wrapper_observes_budget_and_executes_inside_cpu_forbidden_guards(self):
        core_names=("DFHelper","FISAPT","MintsHelper","IBOLocalizer2","ExternalPotential")
        term_names=("localization","flocalization","felst","fexch","find","fdisp0")
        core=SimpleNamespace(**dict.fromkeys(core_names,object()))
        terms=SimpleNamespace(**dict.fromkeys(term_names,object()))
        forbidden=Mock(side_effect=AssertionError("CPU route forbidden"))
        output={"fixture":"output"}
        data={"Disp20,u":-.01,"Exch-Disp20,u":.001,"FISAPT execution":{"fixture":True}}
        def original(cache,*args,**kwargs):
            self.assertEqual(kwargs["factor_cache_bytes"],4*1024**3)
            self.assertTrue(all(getattr(core,k) is forbidden for k in core_names))
            self.assertTrue(all(getattr(terms,k) is forbidden for k in term_names))
            return output,data
        namespace={"core":core,"terms":terms,"forbidden":forbidden,
            "ExitStack":ExitStack,"patch":patch,"original":Mock(side_effect=original),
            "raw_snapshot":Mock(),"capture":Mock(),"phases":[],"result":{},
            "time":SimpleNamespace(perf_counter=Mock(side_effect=(1.,2.))),
            "args":SimpleNamespace(factor_cache_bytes=4*1024**3,optimization_probe=False)}
        fragment=nested_function("fragment",namespace)
        module="psi4.driver.procrouting.sapt.sapt_fisapt_workflow_cuda"
        with patch.dict("sys.modules",{module:SimpleNamespace(HF_FIELDS=())}):
            self.assertEqual(fragment({},factor_cache_bytes=4*1024**3),(output,data))
            self.assertEqual(namespace["phases"][0]["seconds"],1.)
            namespace["capture"].assert_called_once_with(output)
            for kwargs in ({},{"factor_cache_bytes":64*1024**2}):
                namespace["time"].perf_counter=Mock(return_value=3.)
                with self.assertRaisesRegex(RuntimeError,"Public driver did not forward"):
                    fragment({},**kwargs)
            self.assertEqual(namespace["original"].call_count,1)
        self.assertTrue(all(getattr(core,k) is not forbidden for k in core_names))
        self.assertTrue(all(getattr(terms,k) is not forbidden for k in term_names))

    def test_optimization_probe_preserves_actual_and_compares_all_elst_fields(self):
        for mismatch in (None,"qa","qb","matrix","total"):
            with self.subTest(mismatch=mismatch):
                fields={key:np.ones((1,)) for key in ("qa","qb","matrix","total")}
                actual=SimpleNamespace(**fields,work_plan={"point_potential_calls":3})
                reference=SimpleNamespace(**{key:value.copy() for key,value in fields.items()},
                                          work_plan={"point_potential_calls":6})
                if mismatch:
                    getattr(reference,mismatch)[0]+=1e-5
                session=SimpleNamespace(scope=Mock(side_effect=nullcontext),
                                        download=lambda value:value)
                provider=SimpleNamespace(session=session)
                original=Mock(side_effect=(actual,reference))
                report={}
                probe=nested_function("paired_elst",
                    {"original_elst":original,"np":np,"result":report})
                indices=((0,),(1,2))
                if mismatch:
                    with self.assertRaises(AssertionError):
                        probe(provider,active_atoms=indices,jkfit_basis="JKFIT")
                    self.assertEqual(report,{})
                else:
                    self.assertIs(probe(provider,active_atoms=indices,jkfit_basis="JKFIT"),actual)
                    control=report["optimization_controls"]
                    self.assertEqual(set(control["felst_maximum_absolute_errors"]),set(fields))
                    self.assertTrue(all(x==0. for x in control["felst_maximum_absolute_errors"].values()))
                self.assertEqual(original.call_args_list[0].kwargs,
                                 {"active_atoms":indices,"jkfit_basis":"JKFIT"})
                self.assertEqual(original.call_args_list[1].kwargs,
                                 {"active_atoms":None,"jkfit_basis":"JKFIT"})
                session.scope.assert_called_once_with()

    def test_explicit_cache_cap_does_not_change_cpu_reference_or_accept_unbounded_inputs(self):
        for budget in (0,64*1024**2,4*1024**3):
            validate_factor_cache("gpu",budget)
        validate_factor_cache("cpu",0)
        for arm,budget in (("cpu",1),("gpu",-1),("gpu",4*1024**3+1),
                           ("gpu",True),("gpu",1.5),("gpu","4096"),("unknown",0)):
            with self.subTest(arm=arm,budget=budget),self.assertRaises(ValueError):
                validate_factor_cache(arm,budget)

    def test_gate_exercises_public_cache_keyword_instead_of_injecting_it_in_the_wrapper(self):
        tree=ast.parse(Path(__file__).with_name("fragment_gate.py").read_text())
        main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="main")
        fragment=next(n for n in ast.walk(main)
                      if isinstance(n,ast.FunctionDef) and n.name=="fragment")
        assignments=[target for n in ast.walk(fragment) if isinstance(n,ast.Assign)
                     for target in n.targets]
        self.assertFalse(any(isinstance(n,ast.Subscript) and isinstance(n.value,ast.Name)
                             and n.value.id=="kwargs" for n in assignments))
        mapping=next(n for n in ast.walk(main) if isinstance(n,ast.Dict) and any(
            isinstance(k,ast.Constant) and k.value=="cuda_dispersion_factor_cache_bytes"
            for k in n.keys))
        value=next(v for k,v in zip(mapping.keys,mapping.values)
                   if isinstance(k,ast.Constant) and k.value=="cuda_dispersion_factor_cache_bytes")
        self.assertEqual(ast.unparse(value),"args.factor_cache_bytes")

    def test_retains_only_output_owners_no_DFHelper_JK_or_pre_timer_array_access(self):
        class Owner:
            @property
            def np(self):
                raise AssertionError("Snapshot array accessed before timer stop")
        owner = Owner()
        deferred = []
        namespace = {"args": SimpleNamespace(benchmark=True),
                     "deferred_cache": deferred, "FIELDS": FIELDS}
        capture = nested_function("capture", namespace)
        capture({"Elst_AB": owner, "Uocc_A": owner, "wfn_A": owner,
                 "Cvir_A": owner, "dfh": object(), "jk": object(), "large_unrelated": object()})
        self.assertEqual(set(deferred[0]), {"Elst_AB", "Uocc_A", "wfn_A", "Cvir_A"})
        self.assertIs(deferred[0]["Elst_AB"], owner)

    def test_raw_HF_snapshot_retains_exact_requested_owners_without_copy_or_IO(self):
        class Owner:
            @property
            def np(self):
                raise AssertionError("Raw HF copied before timer stop")
        owner = Owner()
        deferred = []
        np = Mock()
        namespace = {"args": SimpleNamespace(benchmark=True),
                     "deferred_raw": deferred, "np": np}
        capture = nested_function("raw_snapshot", namespace)
        capture({"S": owner, "J_A": owner, "unrelated": object()}, ("S", "J_A"))
        self.assertEqual(set(deferred[0]), {"S", "J_A"})
        self.assertIs(deferred[0]["S"], owner)
        np.savez.assert_not_called()
        np.array.assert_not_called()

    def test_CPU_pair_arrays_are_retained_not_copied_inside_benchmark_timer(self):
        disp, exch = object(), object()
        np = Mock()
        namespace = {"args": SimpleNamespace(benchmark=True), "np": np,
                     "store_components": lambda cache, *args, **kwargs: None}
        store = nested_function("components", namespace)
        cache = {}
        store(cache, disp, exch)
        self.assertIs(cache["Disp20 occupied pairs"], disp)
        self.assertIs(cache["Exch-Disp20 occupied pairs"], exch)
        np.array.assert_not_called()


if __name__ == "__main__":
    unittest.main()
