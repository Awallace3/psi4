"""Offline harness controls, not native CPU timing/correctness evidence."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import compare_cpu
import cpu_benchmark
import submit_cpu
import validate_cpu_allocation
import cpu_pilot_proof
from gpu_runtime import digest


class CPUComparison(unittest.TestCase):
    def test_pilot_reuse_fingerprints_exclude_only_post_timing_validation(self):
        text=Path(cpu_benchmark.__file__).read_text()
        original=cpu_pilot_proof.timed_parts(text)
        self.assertEqual(original,cpu_pilot_proof.timed_parts(
            text.replace('err_msg="Canonical component algebra vs interaction energy"',
                         'err_msg="post-timing diagnostic changed"')))
        self.assertNotEqual(original,cpu_pilot_proof.timed_parts(
            text.replace('expected="einsums.fdisp0"','expected="post-timing change"')))
        self.assertNotEqual(original,cpu_pilot_proof.timed_parts(
            text.replace('energy,wfn=psi4.energy("sapt(dft)"','energy,wfn=psi4.energy("other"')))
        self.assertNotEqual(original,cpu_pilot_proof.timed_parts(
            text.replace('fisapt_fsapt_ind_response=False','fisapt_fsapt_ind_response=True')))
        with self.assertRaises(RuntimeError):
            cpu_pilot_proof.timed_parts(text.replace('        expected=','        renamed='))
        with self.assertRaises(RuntimeError):
            cpu_pilot_proof.timed_parts(text.replace("def validate_component_ledger",
                "@change_environment()\ndef validate_component_ledger"))

    def test_pilot_requalification_binds_actual_matrix_sums_and_each_sample_identity(self):
        expected={"source_pin":"science","input_manifest_sha256":"inputs",
            "dependency_proof_sha256":"config","threads":24,"psi4_memory":"40 GiB",
            "stage":"/stage","node":"node","geometry_sha256":"geometry","frozen_occupied":[0,0]}
        with tempfile.TemporaryDirectory() as directory:
            pilot=Path(directory)
            for repeat in range(3):
                for arm in ("cpp","einsums"):
                    root=pilot/f"r{repeat}-{arm}";root.mkdir()
                    matrices={name:np.ones((2,2))*.1 for name in ("Elst","Exch","IndAB","IndBA","Disp")}
                    np.savez(root/"matrices.npz",**matrices)
                    value={**expected,"status":"COMPLETED_CPU_REQUIRES_PAIRED_VALIDATION",
                        "system":"peptide","arm":arm,"repeat":repeat,
                        "options":{**cpu_benchmark.production_options(arm),"fisapt_fsapt_filepath":"none"},
                        "matrices_sha256":digest(root/"matrices.npz"),
                        "signed_matrix_sums":{name:.4 for name in matrices},"canonical":{},"energy_Eh":-.01}
                    (root/"result.json").write_text(json.dumps(value))
            callback=lambda sums,canonical,energy:{"checked_actual_sums":sums}
            proof=cpu_pilot_proof.revalidate_samples(pilot,expected,callback)
            self.assertEqual(len(proof),6)
            self.assertEqual(proof["r0-cpp"]["revalidated_component_ledger"]["checked_actual_sums"]["Elst"],.4)
            path=pilot/"r0-cpp/result.json";original=json.loads(path.read_text())
            for key,value in (("arm","einsums"),("repeat",1),("system","nanotube"),
                              ("input_manifest_sha256","foreign"),("geometry_sha256","foreign")):
                path.write_text(json.dumps({**original,key:value}))
                with self.subTest(key=key),self.assertRaises(RuntimeError):
                    cpu_pilot_proof.revalidate_samples(pilot,expected,callback)
            stale={**original,"signed_matrix_sums":{**original["signed_matrix_sums"],"Elst":.5}}
            path.write_text(json.dumps(stale))
            with self.assertRaises(AssertionError):
                cpu_pilot_proof.revalidate_samples(pilot,expected,callback)

    def test_cancelled_total_does_not_impose_stricter_DF_accuracy_than_components(self):
        # Counterexample reconstructed from nanotube14324682: observed net
        # residual0.64329819 microEh and localized electrostatic residual0.55154.
        canonical={"Elst10,r":-.028776417821518407,"Exch10":.06380549626997682,
            "Ind20,r":-.03837080179904888,"Exch-Ind20,r":.035556549843709,
            "Delta HF Correction":-.003955312576267869,
            "Ind20,r (A<-B)":-.027772966836654943,"Exch-Ind20,r (A<-B)":.02582080324813478,
            "Ind20,r (A->B)":-.010597834962393938,"Exch-Ind20,r (A->B)":.009735746595574222,
            "Disp20,u":-.03984732454923503,"Exch-Disp20,u":.00832731373642325}
        scale=(canonical["Ind20,r"]+canonical["Exch-Ind20,r"]+canonical["Delta HF Correction"])/(
            canonical["Ind20,r"]+canonical["Exch-Ind20,r"])
        sums={"Elst":canonical["Elst10,r"],"Exch":canonical["Exch10"],
              "Disp":canonical["Disp20,u"]+canonical["Exch-Disp20,u"]}
        for side,direction in (("AB","A<-B"),("BA","A->B")):
            sums["Ind"+side]=scale*(canonical[f"Ind20,r ({direction})"]+
                                    canonical[f"Exch-Ind20,r ({direction})"])
        energy=sum(sums.values())
        sums["Elst"]+=5.51541518409e-7
        sums["IndAB"]+=6.4329819e-7-5.51541518409e-7
        actual=cpu_benchmark.validate_component_ledger(sums,canonical,energy)
        self.assertAlmostEqual(actual["raw_fragment_sum_residual_Eh"],6.4329819e-7,places=14)
        # Opposing induction errors must NOT be hidden by a matching total.
        with self.assertRaises(AssertionError):
            cpu_benchmark.validate_component_ledger(
                {**sums,"IndAB":sums["IndAB"]+5e-6,"IndBA":sums["IndBA"]-5e-6},canonical,energy)
        with self.assertRaises(AssertionError):
            cpu_benchmark.validate_component_ledger({**sums,"Elst":sums["Elst"]+2e-6},canonical,energy)
        with self.assertRaises(AssertionError):
            cpu_benchmark.validate_component_ledger(sums,canonical,energy+1e-5)

    def test_effective_allocation_must_match_explicit_paid_shape(self):
        check=validate_cpu_allocation
        manifest={"account":"gts-cs207-chemx","qos":"inferno","memory_GiB":64,"wall_time":"01:00:00"}
        job={"Account":"gts-cs207-chemx","QOS":"inferno","Partition":"cpu-small","NumNodes":"1",
             "NumCPUs":"24","NumTasks":"1","CPUs/Task":"24","OverSubscribe":"NO",
             "TimeLimit":"01:00:00","MinMemoryNode":"64G"}
        check.validate_job(job,manifest)
        for key,value in (("QOS","embers"),("Account","other"),("NumCPUs","48"),
                          ("OverSubscribe","YES"),("TimeLimit","02:00:00"),("MinMemoryNode","160G")):
            with self.subTest(key=key),self.assertRaises(RuntimeError):
                check.validate_job({**job,key:value},manifest)
        self.assertEqual(check.memory_bytes("65536M"),64*1024**3)
        self.assertEqual(check.memory_bytes("65536"),64*1024**3)
        self.assertEqual(check.seconds("1-02:03:04"),93784)
        self.assertEqual(check.fields('SOURCE="/dev/nvme1n1[/scratch/job]" FSTYPE="xfs"')["FSTYPE"],"xfs")

    def test_larger_points_require_successful_exact_pilot_and_all_terminal_steps(self):
        outcome={"status":"PASS_PRODUCTION_CPU_COMPARISON","system":"peptide",
            "accepted_cpu_sample_count":6,"helper_pin":"helper","source_pin":"science",
            "input_manifest_sha256":"inputs","dependency_proof_sha256":"config"}
        kwargs={"helper_pin":"helper","source_pin":"science","inputs_sha256":"inputs","config_sha256":"config"}
        status="exit_code=0\nterm_received=0\n"
        submit_cpu.validate_pilot(outcome,status,**kwargs)
        for key,value in (("status","FAILED"),("accepted_cpu_sample_count",2),
                          ("helper_pin","foreign"),("input_manifest_sha256","changed")):
            with self.subTest(key=key),self.assertRaises(RuntimeError):
                submit_cpu.validate_pilot({**outcome,key:value},status,**kwargs)
        with self.assertRaises(RuntimeError):
            submit_cpu.validate_pilot(outcome,"exit_code=143\nterm_received=1\n",**kwargs)
        accounting="123|COMPLETED|0:0|\n123.batch|COMPLETED|0:0|\n123.0|COMPLETED|0:0|\n"
        submit_cpu.validate_accounting(accounting,"123")
        for bad in (accounting.replace("123.0","missing"),accounting.replace("COMPLETED","RUNNING"),
                    accounting.replace("0:0","1:0")):
            with self.assertRaises(RuntimeError):submit_cpu.validate_accounting(bad,"123")

    def test_production_science_matches_except_fragment_dispatch(self):
        left=cpu_benchmark.production_options("einsums")
        right=cpu_benchmark.production_options("cpp")
        self.assertEqual({**left,"sapt_dft_do_fsapt":"FISAPT"},right)
        self.assertFalse(left["fisapt_fsapt_ind_response"])
        self.assertTrue(left["fisapt_fsapt_ind_scale"])
        self.assertTrue(left["fisapt_fsapt_exch_scale"])
        self.assertTrue(left["sapt_dft_do_dhf"])
        self.assertFalse(left["use_cuest"])
        self.assertEqual(left["minao_basis"],"CC-PVTZ-MINAO")
        self.assertEqual(right["fisapt_fdisp_algorithm"],"BLOCKED")

    def test_exclusive_phase_totals_never_double_count_nested_setup(self):
        clock=iter((0.,1.,3.,5.))
        tape=cpu_benchmark.PhaseTape(lambda:next(clock))
        inner=tape.wrap("localize",lambda:42)
        outer=tape.wrap("setup",lambda:inner())
        self.assertEqual(outer(),42)
        records={v["name"]:v for v in tape.records}
        self.assertEqual(records["setup"]["inclusive_seconds"],5.)
        self.assertEqual(records["setup"]["exclusive_seconds"],3.)
        self.assertEqual(records["localize"]["exclusive_seconds"],2.)
        self.assertEqual(tape.stack,[])

    def test_phase_failure_does_not_keep_argument_or_frame(self):
        tape=cpu_benchmark.PhaseTape()
        def fail(obj):raise ValueError("original")
        with self.assertRaisesRegex(ValueError,"original"):tape.wrap("failed",fail)(object())
        self.assertEqual(tape.stack,[])
        self.assertEqual(set(tape.records[0]),{"name","inclusive_seconds","exclusive_seconds"})

    def test_CPU_allocations_are_fixed_24_threads_no_GPU_and_bounded(self):
        for system,wall,mem in (("peptide","01:00:00",64),("nanotube","02:00:00",64),
                                ("protein83","08:00:00",160)):
            argv=submit_cpu.command(Path("/source"),Path("/run"),system,
                                    qos="inferno",account="gts-cs207-chemx")
            for value in ("--cpus-per-task=24","--exclusive","--qos=inferno",
                          "--time="+wall,"--mem="+str(mem)+"G",
                          "--constraint=core24&gold6226&localNVMe"):
                self.assertIn(value,argv)
            self.assertFalse(any(v.startswith(("--gres","--gpus","--array")) for v in argv))
            psi_mem=int(submit_cpu.SHAPES[system][2].split()[0])
            self.assertLessEqual(psi_mem,.65*mem)

    def test_orbital_signs_and_permutations_are_not_rotated_energy_entries(self):
        co=np.eye(4)[:,:2]
        right=co[:,[1,0]]*np.array([-1.,1.])
        np.testing.assert_array_equal(compare_cpu.orbital_permutation(co,right,np.eye(4)),[1,0])
        theta=.3
        mixed=co@np.array([[np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]])
        with self.assertRaises(AssertionError):
            compare_cpu.orbital_permutation(co,mixed,np.eye(4))
        with self.assertRaisesRegex(RuntimeError,"frozen"):
            compare_cpu.orbital_permutation(co,right,np.eye(4),nfrozen=1)

    def test_pair_comparison_checks_matrices_not_just_totals(self):
        rng=np.random.default_rng(724)
        with tempfile.TemporaryDirectory() as directory:
            left,right=Path(directory)/"cpp",Path(directory)/"einsums"
            left.mkdir();right.mkdir()
            s=np.eye(4);a=s[:,:2];b=s[:,2:]
            matrices={"S":s,"Locc_A":a,"Locc_B":b,
                      "QA":rng.normal(size=(2,2)),"QB":rng.normal(size=(2,2))}
            for name in ("Elst","Exch","IndAB","IndBA","Disp"):
                matrices[name]=rng.normal(size=(5,5))
            mapping=np.array([1,0]);indices=np.array([0,1,3,2,4])
            permuted={**matrices,"Locc_A":a[:,mapping],"Locc_B":-b[:,mapping],
                "QA":matrices["QA"][:,mapping],"QB":matrices["QB"][:,mapping]}
            for name in ("Elst","Exch","IndAB","IndBA","Disp"):
                permuted[name]=matrices[name][np.ix_(indices,indices)]
            def save(path,values,arm):
                np.savez(path/"matrices.npz",**values)
                document={"status":"COMPLETED_CPU_REQUIRES_PAIRED_VALIDATION",
                    "source_pin":"science","geometry_sha256":"geometry",
                    "input_manifest_sha256":"inputs","threads":24,"psi4_memory":"40 GiB",
                    "node":"paired-node","frozen_occupied":[0,0],
                    "canonical":{name:-.01 for name in cpu_benchmark.SCALARS},
                    "stage":"/stage","energy_Eh":-.02,"matrices_sha256":digest(path/"matrices.npz"),
                    "options":{"sapt_dft_do_fsapt":arm,"fisapt_fsapt_filepath":"none",
                               "fisapt_fsapt_ind_response":False}}
                (path/"result.json").write_text(json.dumps(document))
            save(left,matrices,"FISAPT");save(right,permuted,"SAPTDFT")
            result=compare_cpu.compare(left,right)
            self.assertEqual(max(result["maximum_absolute_matrix_errors_Eh"].values()),0.)
            permuted["Disp"]=permuted["Disp"].copy()
            permuted["Disp"][0,0]+=.001;permuted["Disp"][0,1]-=.001
            save(right,permuted,"SAPTDFT")
            with self.assertRaises(AssertionError):compare_cpu.compare(left,right)


if __name__=="__main__":unittest.main()
