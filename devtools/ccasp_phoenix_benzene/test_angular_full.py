# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""Metadata/stub regressions only. No native import, integrals or molecular grids."""
import ast
import copy
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
ROUTING = HERE.parents[1]/"psi4/driver/procrouting"


def load(name, file):
    spec = importlib.util.spec_from_file_location(name,HERE/file)
    item = importlib.util.module_from_spec(spec)
    sys.modules[name] = item
    spec.loader.exec_module(item)
    return item


J = load("angular_test_job","angular_job.py")
R = J.R
P = load("angular_test_profile","angular_profile.py")
D = load("angular_test_driver","angular_driver.py")
S = load("angular_test_shared","integrated_driver.py")


def definitions(file, names, env):
    nodes = [n for n in ast.parse((ROUTING/file).read_text()).body
             if isinstance(n,(ast.FunctionDef,ast.ClassDef,ast.Assign))
             and (getattr(n,"name",None) in names or
                  isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in names for t in n.targets))]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(ROUTING/file),"exec"),env)
    return env


def resource_definitions():
    return definitions("isapol_bounded_response.py",
        {"RESOURCE_CAPACITIES","BoundedResources","_Ledger"},{"dataclass":dataclass})


def planner_evidence():
    """Execute unmodified function ASTs; only shell metadata and len(grid) exist."""
    part = ModuleType("_angular_metadata.isapol_native_partition")
    part.DRHO_REFINEMENT_ITERATIONS = 10
    env = definitions("isapol_distribution.py",{"_width","isa_resource_plan"},
                      {"__package__":"_angular_metadata"})
    def basis(width):
        return NS(representation="Spherical",shells=[NS(l=0)]*width)
    widths = [109]*6+[49]*6
    recipe = NS(sites=[NS(atomic=basis(w),shape=basis(12 if i<6 else 8))
                       for i,w in enumerate(widths)],
        auxiliary=basis(1146),grid=NS(radial_points=100,spherical_points=200),
        controller=NS(max_iterations=120))
    class Grid:
        def __init__(self, rows): self.rows = rows
        def __len__(self): return self.rows
    main = NS(nbf=lambda:414,nshell=lambda:138)
    wfn = NS(basisset=lambda:main)
    evidence = {}
    with mock.patch.dict(sys.modules,{"_angular_metadata.isapol_native_partition":part}):
        for name,h in (("baseline",1408920),("candidate",2325912)):
            numeric,work = env["isa_resource_plan"](wfn,recipe,basis(1146),Grid(h),4)
            # Caller Q grid exactly once; both response copies, MAIN temporaries,
            # planned native state and diagnostics coexist with standalone ISA.
            shared = 32*h+2*32*273240+32*414*414+106842112+32*1024**2
            evidence[name] = dict(q_points=h,standalone_ISA_bytes=numeric,ISA_work=work,
                                  shared_bytes=shared,total_bytes=numeric+shared)
    return evidence


class ResourceTests(unittest.TestCase):
    def test_actual_planner_exact_bytes_and_real_ledger_24_refusal_25_admission(self):
        evidence = planner_evidence()
        self.assertEqual(evidence["baseline"],dict(q_points=1408920,
            standalone_ISA_bytes=23398743536,ISA_work=43607749049472,
            shared_bytes=208454016,total_bytes=23607197552))
        self.assertEqual(evidence["candidate"],dict(q_points=2325912,
            standalone_ISA_bytes=26333117936,ISA_work=44868796447872,
            shared_bytes=237797760,total_bytes=26570915696))
        env = resource_definitions()
        candidate = evidence["candidate"]
        core = NS(print_out=lambda text:None)
        with mock.patch.dict(sys.modules,{"psi4":NS(core=core)}):
            for gib,admitted in ((24,False),(25,True)):
                budget = env["BoundedResources"](gib*1024**3,64000000000000,68719476736,capacity="large")
                ledger = env["_Ledger"](budget,reserved=candidate["shared_bytes"])
                if admitted:
                    ledger.admit("ISA/Q",candidate["standalone_ISA_bytes"],candidate["ISA_work"])
                    self.assertEqual(ledger.peak,26570915696)
                else:
                    with self.assertRaisesRegex(ValueError,"shared numeric byte resource limit"):
                        ledger.admit("ISA/Q",candidate["standalone_ISA_bytes"],candidate["ISA_work"])
        self.assertEqual(25*1024**3,26843545600)
        self.assertEqual(26843545600-candidate["total_bytes"],272629904)
        # Remaining complete-chain work is conditional source-review evidence,
        # not a grid measurement or a prediction of the strict physical gates.
        conditional_work = candidate["ISA_work"]+17469106735400
        self.assertEqual(conditional_work,62337903183272)
        self.assertLess(conditional_work,64000000000000)
        self.assertLess(6968466000,68719476736)
        print("ACTUAL_PLANNER_METADATA_EVIDENCE "+json.dumps(evidence,sort_keys=True))


class ProfileTests(unittest.TestCase):
    def test_exact_two_changes_and_no_mutable_old_namespace(self):
        cfg,identity = P.configuration(R)
        parent = json.loads((HERE/"pilot.json").read_text())
        restored = copy.deepcopy(cfg)
        restored["isa"]["q_grid"] = parent["isa"]["q_grid"]
        restored["resources"]["psi4_memory_bytes"] = parent["resources"]["psi4_memory_bytes"]
        self.assertEqual(restored,parent)
        self.assertEqual(R.sha(HERE/"pilot.json"),P.PARENT_SHA)
        self.assertEqual(identity["effective_config_sha256"],P.canonical_sha(cfg))
        self.assertEqual(identity["identity_sha256"],P.canonical_sha({k:v for k,v in identity.items()
                                                                   if k!="identity_sha256"}))
        self.assertNotEqual(identity["effective_config_sha256"],P.PARENT_SHA)
        old = J.bootstrap
        old_plan = copy.deepcopy(old.PLAN)
        self.assertNotEqual(R.PLAN["source"],old.PLAN["source"])
        self.assertNotEqual(R.PLAN["runtime"],old.PLAN["runtime"])
        self.assertIsNone(old.effective_profile())
        self.assertEqual(old.PLAN,old_plan)
        cfg["isa"]["q_grid"][1] = 999
        self.assertEqual(P.configuration(R)[0]["isa"]["q_grid"],[200,974])

    def test_override_memory_thread_parent_tampering_refused(self):
        for mutate in (
            lambda p:p["overrides"].update({"isa.q_grid":[300,974]}),
            lambda p:p["allocation"].update(psi4_memory_bytes=24*1024**3),
            lambda p:p["allocation"].update(numerical_threads=8),
            lambda p:p["allocation"].update(qos="inferno"),
            lambda p:p["native"].update(core_sha256="0"*64),
            lambda p:p.update(parent_profile_sha256="0"*64)):
            bad = copy.deepcopy(R.PLAN)
            mutate(bad)
            with mock.patch.object(R,"PLAN",bad),self.assertRaisesRegex(ValueError,"exact explicit"):
                P.configuration(R)

    def test_derivative_binds_effective_identity_without_relabeling_original(self):
        _,identity = P.configuration(R)
        original = {"binding":{"original_profile":P.PARENT_SHA}}
        binding = dict(effective_profile=identity,commit="a"*40)
        with mock.patch.object(R,"prior_probe",return_value=original):
            doc = R.derived_document(binding,"b"*64)
            self.assertEqual(doc["original_probe"]["binding"],original["binding"])
            self.assertEqual(doc["effective_profile"],identity)
            self.assertEqual(doc["runtime_manifest_sha256"],"b"*64)
            with self.assertRaisesRegex(ValueError,"effective profile"):
                R.derived_document(dict(binding,effective_profile={}),"b"*64)
        self.assertEqual(original,{"binding":{"original_profile":P.PARENT_SHA}})

    def test_argv_real_pins_and_guard_before_outputs(self):
        args = J.argv("a"*40,"b"*40,"c"*64,"d"*64)
        self.assertIn("--test-only",args)
        self.assertTrue(args[4].endswith("/angular_full.sbatch"))
        with self.assertRaises(ValueError):
            J.argv("a"*40,"b"*40,None,None)
        with mock.patch.object(R,"compute_guard",side_effect=ValueError("local")), \
             mock.patch.object(R,"source_binding") as binding:
            with self.assertRaisesRegex(ValueError,"local"):
                J.run("a"*40,"b"*40,"c"*64,"d"*64)
            binding.assert_not_called()

    def test_stdlib_only_import(self):
        command = ("import importlib.util,sys; "
            f"s=importlib.util.spec_from_file_location('j',{str(HERE/'angular_job.py')!r}); "
            "m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m);"
            "assert not {'numpy','psi4','adcc','einsums'} & set(sys.modules)")
        subprocess.run([sys.executable,"-I","-S","-B","-c",command],check=True)


class PipelineTests(unittest.TestCase):
    def test_actual_angular_adapter_explicitly_selects_shared_pipeline_and_validator(self):
        cfg,identity = P.configuration(R)
        shared = NS(capture_science=mock.Mock(return_value={"science_pass":False}))
        rt = NS(PLAN=R.PLAN,HERE=HERE,OLD=R.OLD,sha=R.sha,compute_guard=mock.Mock(),write=mock.Mock(),
                module=mock.Mock(side_effect=[P,shared]))
        modules = {"psi4.driver.procrouting":NS(isapol_lw=NS(PRODUCTION_TOLERANCE=1e-6))}
        with mock.patch.dict(sys.modules,modules):
            result = D.capture_science(rt,object(),{"effective_profile":identity},"runtime","derivative")
        kwargs = shared.capture_science.call_args.kwargs
        self.assertEqual(kwargs["configuration"],cfg)
        self.assertEqual(kwargs["profile_identity"],identity)
        self.assertIs(kwargs["validate_result"],D.validate_complete)
        self.assertEqual(result["effective_profile"],identity)
        self.assertFalse(result["science_pass"])

    def test_real_shared_driver_binds_25GiB_Q974_and_unchanged_pipeline(self):
        """Small stand-ins reach actual energy/oeprop/export callsites; never native."""
        import numpy  # load before sys.modules patching; never import a quantum engine
        cfg,identity = P.configuration(R)
        budget = resource_definitions()["BoundedResources"]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/"results").mkdir()
            scratch = root/"scratch"
            scratch.mkdir()
            fixture = root/"private-toy.json"
            fixture.write_text(json.dumps(dict(source_sha256=cfg["source_dataframe_sha256"],
                system_id="24_Benzene-Benzene_pi-pi_1.00",dataframe_index=3806,
                fragment="A",charge=0,multiplicity=1)))
            grid_options = NS()
            core = NS(__file__=str(root/"TOY-NOT-ELF"),IsaGridOptions=lambda:grid_options,
                IsaGrid=lambda *a:NS(x=lambda:[1.],y=lambda:[2.],z=lambda:[3.],w=lambda:[4.]),
                CasimirGrid=lambda *a:"TOY quadrature")
            modules = {
                "psi4":NS(core=core),
                "psi4.driver.p4util":NS(OptionsState=object()),
                "psi4.driver.procrouting.isapol_bounded_response":NS(BoundedResources=budget),
                "psi4.driver.procrouting.isapol_native":NS(Quadrature=NS(from_casimir=lambda q:q)),
                "psi4.driver.procrouting.isapol_native_propagator":NS(KernelSmoothing=lambda *a:a)}
            wfn = NS(functional=lambda:NS(grac_shift=lambda:0.),basisset=lambda:NS(nbf=lambda:414))
            psi4 = NS(set_output_file=mock.Mock(),set_memory=mock.Mock(),set_num_threads=mock.Mock(),
                set_options=mock.Mock(),energy=mock.Mock(return_value=(-1.,wfn)),
                oeprop=mock.Mock(),atomic_property_result=mock.Mock(return_value="TOY public result"))
            prior = NS(make_molecule=mock.Mock(return_value=(NS(clone=lambda:None),[])),
                recipe=mock.Mock(return_value="TOY recipe"),export=mock.Mock())
            sink = NS(written=123)
            sink_module = NS(DiagnosticSink=mock.Mock(return_value=sink))
            def admit(core,options,molecule,audit,*args):
                audit.update(basis_files={},main={"fingerprint_sha256":"main"},
                             auxiliary={"fingerprint_sha256":"aux"})
                return None,NS(auxiliary="TOY admitted AUX")
            rt = NS(HERE=HERE,ROOT=root,PLAN=dict(run=str(root),scratch=str(scratch),
                fixture=str(fixture),runtime=str(root/"runtime"),
                allocation=R.PLAN["allocation"],prior_probe_sha256="probe"),
                OLD=R.OLD,compute_guard=mock.Mock(),module=mock.Mock(side_effect=[prior,sink_module]),
                admit_basis=admit,write=R.write,sha=lambda p:(
                    cfg["fixture_sha256"] if Path(p)==fixture else R.sha(p) if Path(p).exists() else "e"*64),
                BASIS=NS(compare_scf_main=mock.Mock()))
            validator = mock.Mock()
            with mock.patch.dict(sys.modules,modules), \
                 mock.patch.dict(os.environ,SLURM_JOB_ID="TOY"), \
                 mock.patch.object(S.os,"statvfs",return_value=NS(f_bavail=401*1024**3,f_frsize=1)):
                result = S.capture_science(rt,psi4,dict(commit="a"*40,tree="b"*40), "r","d",configuration=cfg,
                    profile_identity=identity,validate_result=validator)
            self.assertTrue(result["science_pass"])  # synthetic call completion only
            psi4.set_memory.assert_called_once_with(26843545600)
            psi4.set_num_threads.assert_called_once_with(1)
            psi4.energy.assert_called_once()
            psi4.oeprop.assert_called_once()
            kwargs = psi4.oeprop.call_args.kwargs
            self.assertEqual(kwargs["resources"].max_bytes,26843545600)
            self.assertEqual(kwargs["resources"].max_work,64000000000000)
            self.assertEqual(kwargs["resources"].max_io_bytes,68719476736)
            self.assertEqual((grid_options.radial_points,grid_options.spherical_points),(200,974))
            self.assertEqual((kwargs["radial_points"],kwargs["spherical_points"]),(100,200))
            self.assertEqual(kwargs["response"],"reference_h2h1")
            self.assertEqual(kwargs["localization_rank_limit"],2)
            self.assertTrue(kwargs["recoupled_dispersion"])
            self.assertIs(kwargs["diagnostic_sink"],sink)
            validator.assert_called_once_with("TOY public result","TOY quadrature",cfg)
            prior.export.assert_called_once()
            self.assertEqual(sink_module.DiagnosticSink.call_args.args[1]["config_sha256"],
                             identity["effective_config_sha256"])

    def complete_result(self):
        refinement = NS(model=NS(static_reference_sha256="static",penalty_selection="static_reference"))
        return NS(frequencies=tuple(range(11)),local_tensors=[0]*11,refinements=[refinement]*11,
            refined_tensors=[0]*11,diagnostics=[{"localization_residual":1e-8} for _ in range(11)],
            dispersion=NS(pairs=[0]*144),recoupled_dispersion=NS(pairs=[0]*144),
            resources=dict(maximum_numeric_plan=26570915696,charged_work=62337903183272,
                           charged_io_bytes=6968466000))

    def test_incomplete_or_bad_physical_receipts_cannot_publish(self):
        cfg,_ = P.configuration(R)
        result = self.complete_result()
        quadrature = NS(frequencies=result.frequencies)
        D.validate_complete(result,quadrature,cfg)
        for mutate in (lambda r:r.local_tensors.pop(),
                       lambda r:r.refinements[0].model.__setattr__("penalty_selection","dynamic"),
                       lambda r:r.diagnostics[0].update(localization_residual=1e-5),
                       lambda r:r.diagnostics[0].update(localization_residual=float("nan")),
                       lambda r:setattr(r,"recoupled_dispersion",None),
                       lambda r:r.resources.update(maximum_numeric_plan=26*1024**3)):
            bad = copy.deepcopy(result)
            mutate(bad)
            with self.assertRaises(ValueError):
                D.validate_complete(bad,quadrature,cfg)

    def test_actual_shell_terminal_receipts_success_failure_TERM(self):
        script = (HERE/"angular_full.sbatch").read_text()
        for value in ("--account=gts-cs207-chemx","--partition=cpu-small","--qos=embers",
                      "--cpus-per-task=8","--mem=48G","--time=06:00:00",
                      "--nodelist=atl1-1-02-007-23-1","--no-requeue","-I -S -B"):
            self.assertIn(value,script)
        self.assertNotIn("--dependency",script)
        self.assertNotIn("sbatch ",script)
        for value in (R.PLAN["source"],R.PLAN["run"],R.PLAN["scratch"]):
            self.assertIn(value,script)
        traps = script.split("term_received=0\n",1)[1].split('test "$#" = 4',1)[0]
        for action,code in (("child_ok=1; exit 0",0),("exit 2",2),
                            ("child_ok=1; kill -TERM $$; exit 0",143)):
            with tempfile.TemporaryDirectory() as tmp:
                (Path(tmp)/"metadata").mkdir()
                text = "set -euo pipefail\nRUN="+tmp+"\nterm_received=0\n"+traps+action
                process = subprocess.run(["bash"],input=text,text=True,capture_output=True)
                self.assertEqual(process.returncode,code,process.stderr)
                terminal = json.loads((Path(tmp)/"metadata/batch-terminal.json").read_text())
                self.assertEqual(terminal["complete"],code==0)
                self.assertEqual(terminal["exit_code"],code)

    def test_job_retains_primary_failure_capture_and_final_closure(self):
        for failure in ("science","loader","closure","none"):
            with self.subTest(failure=failure),tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root/"metadata").mkdir()
                binding = {"effective_profile":{"synthetic":True}}
                R.write(root/"metadata/preparation.json",dict(binding=binding,
                    runtime_manifest_sha256="c"*64,derivative_sha256="d"*64,
                    script_sha256=R.sha(HERE/"angular_full.sbatch")))
                retained = []
                token = object()
                def initialize(*args):
                    args[-1].append(token)
                    retained.append(args[-1])
                    if failure=="loader":
                        raise ValueError("TOY loader refusal")
                    return object(),token
                def science(*args):
                    marker = root/"results/diagnostic/node-01/pre-lw-complete.json"
                    marker.parent.mkdir(parents=True)
                    R.write(marker,{"properties_accepted":False})
                    if failure=="science":
                        raise RuntimeError("TOY original strict LW refusal")
                    return dict(science_pass=True,status="complete")
                def snapshot(*args,**kwargs):
                    self.assertIs(retained[0][0],token)
                    if failure in ("science","closure"):
                        raise ValueError("TOY final closure refusal")
                with mock.patch.dict(R.PLAN,run=str(root)), \
                     mock.patch.dict(os.environ,SLURM_JOB_ID="TOY"), \
                     mock.patch.object(R,"compute_guard"), \
                     mock.patch.object(R,"source_binding",return_value=binding), \
                     mock.patch.object(R,"pinned_json",return_value={}), \
                     mock.patch.object(R,"derived_document",return_value={}), \
                     mock.patch.object(R,"initialize",side_effect=initialize), \
                     mock.patch.object(R,"verify_runtime"), \
                     mock.patch.object(R,"module",return_value=NS(capture_science=science)), \
                     mock.patch.object(R.OLD,"snapshot",side_effect=snapshot) as closure:
                    self.assertEqual(J.run("a"*40,"b"*40,"c"*64,"d"*64),0 if failure=="none" else 1)
                closure.assert_called_once()
                record = json.loads((root/"results/result.json").read_text())
                terminal = json.loads((root/"metadata/payload-terminal.json").read_text())
                self.assertEqual(terminal["complete"],failure=="none")
                if failure=="science":
                    self.assertIn("original strict LW",record["error"])
                    self.assertIn("final closure",record["closure_error"])
                    self.assertEqual(len(record["complete_pre_lw_capture_receipts"]),1)
                    self.assertFalse(record["science_pass"])


if __name__ == "__main__":
    unittest.main()
