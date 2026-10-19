"""Synthetic host checks, not GPU evidence. No scientific stack import/device call."""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import config as s
import controller
import custody
import execute
from provenance import expected_runner, verify_runner
from statistics_report import classify, summary
from trace import parse, timer_regions
import science
from test_harness import fixture


def synthetic_rows():
    rows = []
    for v in s.SEQUENCE:
        data = copy.deepcopy(fixture()[0])
        data["system"] = v["system"]
        data["nbf"] = s.p.SYSTEMS[v["system"]][1]
        data["wall_s"] = (10 if v["arm"] == "baseline" else 8)+v["repeat"]
        rows.append(dict(v,data=data,validation={"timer_cut":{"synthetic":True}},
                         trace={"synthetic":True},pid=100+len(rows)))
    return rows


class WholeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=os.environ["MAJOR2_TEST_TMP"])
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_24_sequence_unique_exact_inherited_order(self):
        self.assertEqual(len(s.SEQUENCE),24)
        self.assertEqual(len({v["label"] for v in s.SEQUENCE}),24)
        self.assertEqual([{**v,"label":v["label"].replace("whole-","major2-")}
                          for v in s.SEQUENCE],s.p.sequence())

    def test_adjacent_pairs_same_system_repeat_opposite_arms(self):
        for i in range(0,24,2):
            a,b = s.SEQUENCE[i:i+2]
            self.assertEqual((a["system"],a["repeat"]),(b["system"],b["repeat"]))
            self.assertEqual({a["arm"],b["arm"]},{"baseline","candidate"})
        self.assertEqual(sum(s.SEQUENCE[i]["arm"] == "baseline" for i in range(0,24,2)),6)

    def test_three_each_build_each_system(self):
        for system in s.p.SYSTEMS:
            for arm in ("baseline","candidate"):
                self.assertEqual(sorted(v["repeat"] for v in s.SEQUENCE
                    if v["system"] == system and v["arm"] == arm),[1,2,3])

    def test_budget_all_caps_and_reserves(self):
        self.assertEqual(sum(s.CAPS[v["system"]] for v in s.SEQUENCE),15480)
        self.assertEqual(s.remaining_budget(0),18420)
        s.admit(21600,0,prolog=True)
        with self.assertRaises(RuntimeError):
            s.admit(14400,0,prolog=True)
        self.assertEqual(s.remaining_budget(24),1500)

    def test_budget_exact_boundary_and_nonfinite_fail(self):
        for i in range(25):
            need = s.remaining_budget(i)
            s.admit(need,i)
            for value in (need-.001,float("nan"),float("inf")):
                with self.assertRaises(RuntimeError):
                    s.admit(value,i)

    def test_runner_exact_minimal_delta(self):
        v = verify_runner(s.ROOT)
        self.assertEqual(v["original_sha256"],s.p.RUNNER_SHA)
        self.assertEqual((s.HERE/"runner.py").read_text(),
                         expected_runner((s.ROOT/"devtools/saptdft_cpks_suite_run.py").read_text()))
        self.assertEqual(v["patch"].count("+    psi4.set_options"),1)

    def test_timing_boundary_no_added_energy_hooks(self):
        text = (s.HERE/"runner.py").read_text()
        exact = ('        start = time.perf_counter()\n'
                 '        energy = psi4.energy("sapt(dft)-d4(i)", molecule=molecule)\n'
                 '        record["wall_s"] = time.perf_counter() - start\n')
        self.assertIn(exact,text)
        self.assertEqual(text.count("psi4.energy("),1)
        for forbidden in ("jk.initialize(", "jk.finalize(", "HF.form_G", "return_wfn=True"):
            self.assertNotIn(forbidden,text)

    def test_native_print_source_paths_and_sad_limit(self):
        source = s.ROOT
        iterator = (source/"psi4/driver/procrouting/scf_proc/scf_iterator.py").read_text()
        self.assertIn("jk.set_print(self.get_print())",iterator)
        self.assertIn('if (options["PRINT"].has_changed()) jk->set_print(options.get_int("PRINT"));',
                      (source/"psi4/src/psi4/libfock/jk.cc").read_text())
        sad = (source/"psi4/src/psi4/libscf_solver/sad.cc").read_text()
        self.assertIn("set_print",sad)
        self.assertIn("internal SAD",parse("", "baseline")["coverage"]["sad"])
        self.assertEqual(s.POLICY["print_options"],{"PRINT":3,"SCF__PRINT":3,"SAPT__PRINT":3})

    def test_original_proof_and_solver_wrappers_verbatim(self):
        old = ast.parse((s.ROOT/"devtools/saptdft_cpks_suite_run.py").read_text())
        new = ast.parse((s.HERE/"runner.py").read_text())
        for name in ("install_iterator_proofs","solver_histories"):
            a = next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name == name)
            b = next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name == name)
            self.assertEqual(ast.dump(a),ast.dump(b))

    def test_actual_completed_trace_line_fixtures(self):
        f = s.load(s.HERE/"trace-fixtures.json")
        for arm,created,reused in (("baseline",6,0),("candidate",1,5)):
            value = parse("\n".join(f[arm+"_lines"]),arm)
            self.assertEqual(value["observed_created"],created)
            self.assertEqual(value["observed_explicit_reused"],reused)
        self.assertIn("NOT whole-energy",f["provenance"])

    def test_trace_resets_not_global_ids(self):
        text = ("SAPT(DFT): delta HF Dimer\n    cuESTJK plan created #1\n"
                "    Device memory, DF int plan: 2 [GiB]\n"
                "SAPT(DFT): DFT Monomer A\n    cuESTJK plan created #1\n"
                "    Device memory, DF int plan: 2 [GiB]\n"
                "    cuESTJK plan reused #1\n")
        v = parse(text,"candidate")
        self.assertEqual(v["observed_created"],2)
        self.assertEqual([x["local_counter"] for x in v["evidence"] if x["kind"] == "created"],[1,1])
        self.assertEqual(v["evidence"][-1]["context"],"DFT Monomer A")

    def test_trace_zero_hits_valid_no_hypothetical_opportunities(self):
        text = "    cuESTJK plan created #1\n    Device memory, DF int plan: 2 [GiB]\n"
        self.assertEqual(parse(text,"candidate")["observed_explicit_reused"],0)

    def test_trace_banner_agreement_fatal(self):
        for text in ("cuESTJK plan created #1\n", "Device memory, DF int plan: 1\n"):
            with self.assertRaises(RuntimeError):
                parse(text,"candidate")

    def test_trace_malformed_and_baseline_candidate_lines_fatal(self):
        for arm,text in (("candidate","cuESTJK plan reused #x"),
                         ("candidate","prefix cuESTJK plan created #1 suffix"),
                         ("baseline","cuESTJK plan reused #1")):
            with self.assertRaises(RuntimeError):
                parse(text,arm)

    def test_raw_lines_numbers_sha_and_passed_object_separate(self):
        line = "    Device memory, DF int plan:     42.946 [GiB]"
        text = "\nSAPT(DFT): delta HF Dimer\n"+line+"\nRe-using passed JK object instead of rebuilding\n"
        v = parse(text,"baseline")
        self.assertEqual(v["evidence"][0]["line_number"],3)
        self.assertEqual(v["evidence"][0]["sha256"],hashlib.sha256(line.encode()).hexdigest())
        self.assertEqual(v["raw_log_sha256"],hashlib.sha256(text.encode()).hexdigest())
        self.assertEqual(v["driver_passed_object_messages"],1)
        self.assertEqual(v["observed_explicit_reused"],0)

    def test_descriptive_statistics_absolute_reduction(self):
        r = summary(synthetic_rows())
        for v in r["systems"].values():
            self.assertEqual(v["baseline"]["samples_s"],[11,12,13])
            self.assertEqual(v["candidate"]["samples_s"],[9,10,11])
            self.assertEqual(v["seconds_saved"],2)
            self.assertAlmostEqual(v["percent_reduction"],100/6)
            self.assertEqual(v["baseline_over_candidate"],1.2)
            self.assertEqual(len(v["all_nine_cross_arm_comparisons"]),9)
            self.assertTrue(all(len(x)==3 for x in v["within_arm_comparisons"].values()))
        self.assertIsNone(r["confidence_intervals"])

    def test_no_deletion_on_finite_component_gate_failure(self):
        rows = synthetic_rows()
        rows[1]["data"]["components_hartree"][science.COMPONENTS[0]] += 1e-7
        result = summary(rows)
        self.assertEqual(len(rows),24)
        self.assertEqual(len(result["systems"]["peptide"]["all_nine_cross_arm_comparisons"]),9)
        self.assertEqual(result["systems"]["peptide"]["qualification"],
                         s.POLICY["failed_gate_qualification"])

    def test_count_changes_reported_all_five_and_two(self):
        a,b = [copy.deepcopy(fixture()[0]) for _ in range(2)]
        b["proofs"][3]["iterations"] += 1
        b["cpks_solvers"][1]["iterations"] += 2
        v = classify(a,b)
        self.assertEqual(v["iteration_gate"],"different")
        self.assertEqual(len(v["b_counts"]["scfs"]),5)
        self.assertEqual(len(v["b_counts"]["solvers"]),2)

    def test_old_compare_and_gates_unchanged(self):
        a,b = [copy.deepcopy(fixture()[0]) for _ in range(2)]
        b["proofs"][0]["iterations"] += 1
        self.assertEqual(s.p.GATES["iteration_changes"],"STOP_FOR_REVIEW")
        self.assertEqual(s.p.GATES["component_abs_eh"],1e-8)
        with self.assertRaises(RuntimeError):
            science.compare(a,b)
        self.assertFalse(classify(a,b)["matched_work_acceptance"])

    def test_production_controller_never_calls_old_cross_record_stopper(self):
        tree = ast.parse((s.HERE/"controller.py").read_text())
        self.assertFalse(any(isinstance(v,ast.Call) and isinstance(v.func,ast.Attribute) and
                             v.func.attr == "compare" for v in ast.walk(tree)))
        method = next(v for cls in tree.body if isinstance(cls,ast.ClassDef)
                      for v in cls.body if isinstance(v,ast.FunctionDef) and v.name == "execute")
        text = ast.unparse(method)
        self.assertLess(text.index("self.records.append(value)"),text.index("summary(self.records)"))

    def test_numeric_scf_trajectory_residual_validation(self):
        import re
        text = re.sub(r"(iter (?:SAD|\d+):) 0",r"\1 -1.0 1e-11 1e-10",fixture()[1])
        parsed = controller.r.trajectory(text)
        self.assertEqual([len(v["rows"]) for v in parsed],[3,3,3,4,4])
        for before,after in (("1e-11 1e-10","nan 1e-10"),("1e-11 1e-10","1e-7 1e-6")):
            with self.assertRaises(RuntimeError):
                controller.r.trajectory(text.replace(before,after))

    def test_max_component_and_total_delta(self):
        a,b = [copy.deepcopy(fixture()[0]) for _ in range(2)]
        b["components_hartree"][science.COMPONENTS[2]] += 2e-7
        b["returned_energy_hartree"] += 1e-7
        v = classify(a,b)
        self.assertEqual(v["max_component"],science.COMPONENTS[2])
        self.assertAlmostEqual(v["max_component_abs_delta_eh"],2e-7)
        self.assertAlmostEqual(v["total_signed_b_minus_a_eh"],1e-7)

    def test_missing_duplicate_reordered_records_fatal(self):
        for rows in (synthetic_rows()[:-1],synthetic_rows()[::-1],
                     synthetic_rows()[:-1]+[synthetic_rows()[0]]):
            with self.assertRaises(RuntimeError):
                summary(rows)

    def test_nan_nonpositive_wall_invalid_and_wrong_nbf_fatal(self):
        for key,value in (("wall_s",float("nan")),("wall_s",0),("ok",False),("nbf",251)):
            rows = synthetic_rows()
            rows[0]["data"][key] = value
            with self.assertRaises(RuntimeError):
                summary(rows)

    def test_nonfinite_component_fatal(self):
        a,b = [copy.deepcopy(fixture()[0]) for _ in range(2)]
        for value in (float("nan"),float("inf")):
            b["components_hartree"][science.COMPONENTS[0]] = value
            with self.assertRaises(RuntimeError):
                classify(a,b)

    def test_nonconvergence_nan_residual_and_nonterminal_grid_fatal(self):
        values = fixture()
        science.validate_record(*values,candidate=True)
        for mutation in ("scf","solver","nan","grid"):
            v = copy.deepcopy(values)
            if mutation == "scf":
                v[0]["proofs"][0]["converged"] = False
            elif mutation == "solver":
                v = list(v)
                v[1] = v[1].replace("1e-9 2e-9","1e-7 2e-7")
            elif mutation == "nan":
                v[0]["components_hartree"][science.COMPONENTS[0]] = float("nan")
            else:
                v = list(v)
                v[1] = v[1].replace("XC-FINAL","XC-INITIAL",1)
            with self.assertRaises(RuntimeError):
                science.validate_record(*v,candidate=True)

    def test_timer_regions_do_not_add_inclusive_subphase_twice(self):
        d = fixture()[0]
        v = timer_regions(d)
        self.assertEqual(v["SCF: Native finalize"]["n_calls"],5)
        self.assertAlmostEqual(v["SCF: Native finalize"]["wall_s"],.5)
        self.assertEqual(len(v["RHF subphases"]),10)

    def test_timer_cut_additivity_and_provenance(self):
        d,_,printed,*_ = fixture()
        cut = science.timers(d,printed,True)
        self.assertAlmostEqual(sum(cut["top_level_cut_s"].values())+
            cut["energy_unattributed_s"]+cut["outside_energy_s"],d["wall_s"])
        rows = synthetic_rows()
        r = summary(rows)["systems"]["peptide"]
        self.assertEqual(r["baseline"]["representative_label"],"whole-baseline-peptide-r2")
        self.assertEqual(r["baseline"]["representative_timer_cut"],{"synthetic":True})

    def test_nested_native_region_selection_rejected(self):
        d = fixture()[0]
        key = next(k for k in d["timer_records"] if k.endswith(";SCF: Native finalize"))
        d["timer_records"][key+";SCF: Native finalize"] = d["timer_records"][key]
        with self.assertRaises(RuntimeError):
            timer_regions(d)

    def test_no_matrix_observer_sampler_or_build_path(self):
        self.assertIs(controller.Engine.arm_command,controller.c.Engine.arm_command)
        text = (s.HERE/"controller.py").read_text()
        for call in ("self.start_sampler(", "self.build(", "self.tests(", "self.adopt("):
            self.assertNotIn(call,text)
        self.assertIn("r.verify_artifact(",text)
        self.assertIs(controller.Engine.identify_gpu,controller.rc.Engine.identify_gpu)
        self.assertIs(controller.Engine.verify_build,controller.c.Engine.verify_build)

    def test_early_env_before_scientific_imports(self):
        t = (s.HERE/"execute.py").read_text()
        self.assertLess(t.index("receipt = early("),t.index("\n    import probe\n"))
        self.assertLess(t.index("probe.main()"),t.index('runpy.run_path('))
        self.assertIn('cuda.synchronize()  # outside unchanged primary',t)

    def test_exclusive_pid_real_uuid_not_global_index(self):
        with patch.object(execute.os,"getpid",return_value=123), patch.object(
                execute.subprocess,"check_output",return_value="GPU-own, 123\n"):
            self.assertEqual(execute.exclusive_pid("GPU-own"),dict(pid=123,uuid="GPU-own"))
        for output in ("GPU-other, 123\n","GPU-own, 456\n","GPU-own, 123\nGPU-own, 456\n",""):
            with patch.object(execute.subprocess,"check_output",return_value=output):
                with self.assertRaises(RuntimeError):
                    execute.exclusive_pid("GPU-own")

    def test_env_locator_126_allowed_numerical_knobs_fatal(self):
        s.n.check_env({"CUDA_ROOT":"/site/cuda12.6","CUDA_HOME":s.n.CUDA})
        for key in ("CUDA_LAUNCH_BLOCKING","CUBLAS_WORKSPACE_CONFIG","OMP_PLACES"):
            with self.assertRaises(RuntimeError):
                s.n.check_env({key:"forbidden"})

    def test_18_root_accounting_exact_no_unsubmitted_confusion(self):
        self.assertEqual(len(custody.FAILED),14)
        self.assertEqual(len(custody.COMPLETED),4)
        self.assertEqual(len(set(custody.FAILED+custody.COMPLETED)),18)
        self.assertIn("14182594",custody.FAILED)
        self.assertNotIn("14182594",custody.COMPLETED)
        self.assertIn("14179309",custody.COMPLETED)
        output = "\n".join("|".join((j+suf,state,code)) for jobs,state,code in (
            (custody.FAILED,"FAILED","1:0"),(custody.COMPLETED,"COMPLETED","0:0"))
            for j in jobs for suf in ("",".batch"))
        with patch.object(s.n,"output",return_value=output):
            self.assertEqual(len(custody.accounting()),36)
        for job in ("14179309","14182594"):
            with patch.object(s.n,"output",return_value=output.replace(job+".batch","missing.batch")):
                with self.assertRaises(RuntimeError):
                    custody.accounting()

    def test_exact_batch_six_hours_and_parent_only_entry(self):
        text = custody.batch_text("a"*64)
        for v in ("--time=06:00:00","--qos=embers","--gres=gpu:h200:1","--cpus-per-task=8",
                  "--mem=128G","--signal=TERM@300","--no-requeue"):
            self.assertIn(v,text)
        self.assertNotIn("sbatch ",text)
        self.assertIn("whole_gpu_scaling/supervise.sh",text)
        subprocess.run(["bash","-n"],input=text,text=True,check=True)

    def test_allocation_six_hours_not_old_four_hours(self):
        text = ("Account=gts-cs207-chemx QOS=embers Partition=gpu-h200,gpu-v100 NumCPUs=8 "
                "MinMemoryNode=128G NumNodes=1 NumTasks=1 TimeLimit=06:00:00 Features=H200 "
                "AllocTRES=cpu=8,gres/gpu:h200=1")
        env = dict(SLURM_CPUS_PER_TASK="8",SLURM_JOB_ACCOUNT="gts-cs207-chemx",
                   SLURM_JOB_QOS="embers",SLURM_JOB_NUM_NODES="1",
                   SLURM_JOB_START_TIME="1",SLURM_JOB_END_TIME="21601")
        controller.allocation(text,env)
        for old,new in (("06:00:00","04:00:00"),("embers","inferno"),("NumCPUs=8","NumCPUs=9"),
                        ("Features=H200","Features=A100")):
            with self.assertRaises(RuntimeError):
                controller.allocation(text.replace(old,new),env)

    def test_one_shot_lock_is_permanent(self):
        lock = self.root/"attempt"
        custody.permanent_lock(lock)
        with self.assertRaises(FileExistsError):
            custody.permanent_lock(lock)
        self.assertTrue(lock.is_dir())

    def test_parent_guard_rejects_before_any_scheduler_action(self):
        with patch.dict(os.environ,{},clear=True), patch.object(custody,"audit") as audit:
            with self.assertRaises(RuntimeError):
                custody.parent_submit("a"*40,"b"*64,"c"*64)
            audit.assert_not_called()

    def test_accepted_id_is_fsynced_before_controller_query(self):
        (self.root/"logs").mkdir()
        mh = "b"*64
        def query(*args):
            self.assertEqual((self.root/"logs/job-id.txt").read_text(),"123456\n")
            self.assertEqual((self.root/"logs/job-id.txt").stat().st_mode & 0o222,0)
            self.assertTrue((self.root/"logs/parent-attempt").is_dir())
            return ("JobId=123456 QOS=embers Account=gts-cs207-chemx TimeLimit=06:00:00 "
                    "Partition=gpu-h200 Command="+str(self.root)+"/campaign.sbatch")
        with patch.object(s,"RUN",str(self.root)), patch.dict(os.environ,MAJOR2_WHOLE_PARENT_REVIEW=mh), \
             patch.object(custody,"audit",return_value={"synthetic":True}), \
             patch.object(custody.subprocess,"run",return_value=SimpleNamespace(
                 returncode=0,stdout="123456\n",stderr="")), patch.object(s.n,"output",side_effect=query):
            custody.parent_submit("a"*40,mh,"c"*64)

    def test_ambiguous_submit_no_second_attempt(self):
        (self.root/"logs").mkdir()
        mh = "b"*64
        with patch.object(s,"RUN",str(self.root)), patch.dict(os.environ,MAJOR2_WHOLE_PARENT_REVIEW=mh), \
             patch.object(custody,"audit",return_value={"synthetic":True}), \
             patch.object(custody.subprocess,"run",side_effect=subprocess.TimeoutExpired("sbatch",35)) as run:
            with self.assertRaises(subprocess.TimeoutExpired):
                custody.parent_submit("a"*40,mh,"c"*64)
            with self.assertRaises(FileExistsError):
                custody.parent_submit("a"*40,mh,"c"*64)
            self.assertEqual(run.call_count,1)

    def test_numeric_id_parser_rejects_noise(self):
        self.assertEqual(custody.accept_id("1234;phoenix-slurm\n"),"1234")
        for value in ("Submitted batch job 1234\n","1234\n1235\n","0\n","1234 \n"):
            with self.assertRaises(RuntimeError):
                custody.accept_id(value)

    def test_durable_exclusive_finite_and_immutable(self):
        file = self.root/"receipt"
        s.durable(file,{"x":1})
        self.assertEqual(file.stat().st_mode & 0o222,0)
        with self.assertRaises(FileExistsError):
            s.durable(file,{"x":2})
        with self.assertRaises(ValueError):
            s.durable(self.root/"nan",{"x":float("nan")})

    def fake_engine(self):
        e = controller.Engine.__new__(controller.Engine)
        e.signalled = False
        e.raw = self.root
        e.scratch = self.root
        e.env = os.environ.copy()
        e.end = __import__("time").time()+4000
        e.active = None
        return e

    def test_child_failure_retains_partial_and_exit_receipt(self):
        e = self.fake_engine()
        code = "from pathlib import Path;Path('partial').write_text('partial');raise SystemExit(7)"
        with self.assertRaises(RuntimeError):
            e.run_command([sys.executable,"-c",code],"synthetic-fail",timeout=5)
        self.assertEqual(s.load(self.root/"synthetic-fail.exit.json")["exit"],7)
        self.assertTrue((self.root/"partial").exists())

    def test_child_timeout_retains_signal_receipt(self):
        e = self.fake_engine()
        with self.assertRaises(subprocess.TimeoutExpired):
            e.run_command([sys.executable,"-c","import time;time.sleep(10)"],"synthetic-timeout",timeout=.05)
        self.assertTrue(s.load(self.root/"synthetic-timeout.exit.json")["interrupted"])

    def test_term_deadline_handlers_fatal(self):
        with self.assertRaises(RuntimeError):
            controller.deadline()
        with self.assertRaises(RuntimeError):
            execute.stop(signal.SIGTERM,None)
        e = self.fake_engine()
        with self.assertRaises(RuntimeError):
            e.stop(signal.SIGTERM,None)
        self.assertTrue(e.signalled)

    def test_supervisor_false_success_without_endpoint_rejected(self):
        result = subprocess.run(["bash",str(s.HERE/"supervise.sh"),str(self.root),
            sys.executable,"-c","pass"],capture_output=True,text=True,timeout=10)
        self.assertNotEqual(result.returncode,0)
        self.assertFalse((self.root/"COMPLETED").exists())
        self.assertEqual(s.load(self.root/"batch-exit.json"),dict(exit=1,signalled=0))

    def test_supervisor_fsync_complete_and_no_overwrite(self):
        s.durable(self.root/"WHOLE-COMPLETE.json",dict(processes=24,energy_calls=24,
            individually_valid_records=24,observational_only=True,no_dropped_samples=True))
        s.durable(self.root/"WHOLE-REPORT.json",{"synthetic":True})
        args = ["bash",str(s.HERE/"supervise.sh"),str(self.root),sys.executable,"-c","pass"]
        subprocess.run(args,check=True,timeout=10)
        self.assertEqual(s.load(self.root/"batch-exit.json"),dict(exit=0,signalled=0))
        self.assertEqual((self.root/"COMPLETED").stat().st_mode & 0o222,0)
        self.assertNotEqual(subprocess.run(args,capture_output=True,timeout=10).returncode,0)

    def test_supervisor_term_partial_no_false_completion(self):
        import time
        code = ("import signal,time,pathlib,sys\n"
                "signal.signal(signal.SIGTERM,lambda *_:sys.exit(143))\n"
                "pathlib.Path('partial').write_text('ready')\n"
                "while True:time.sleep(.01)\n")
        proc = subprocess.Popen(["bash",str(s.HERE/"supervise.sh"),str(self.root),
            sys.executable,"-c",code],cwd=self.root)
        try:
            limit = time.monotonic()+5
            while not (self.root/"partial").exists() and time.monotonic()<limit:
                time.sleep(.01)
            self.assertTrue((self.root/"partial").exists())
            proc.send_signal(signal.SIGTERM)
            self.assertNotEqual(proc.wait(timeout=10),0)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)
        self.assertFalse((self.root/"COMPLETED").exists())
        self.assertTrue((self.root/"partial").exists())
        self.assertEqual(s.load(self.root/"batch-exit.json")["signalled"],1)

    def test_no_actual_stack_import_from_preparation_modules(self):
        self.assertNotIn("psi4",sys.modules)
        self.assertNotIn("cupy",sys.modules)
        self.assertNotIn("torch",sys.modules)


if __name__ == "__main__":
    unittest.main()
