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
from gpu_runtime import digest


class CPUComparison(unittest.TestCase):
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
