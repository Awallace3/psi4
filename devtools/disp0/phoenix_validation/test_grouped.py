"""The user's regrouping changes allocation counts, not scientific engines."""
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from grouped_jobs import entries,main
from run_full import save
from split_jobs import validate_allocation
from submit_grouped import job_specs,prepare
from submit_split import command


class GroupedContracts(unittest.TestCase):
    def test_one_GPU_shared_small_CPU_and_separate_protein_CPU_keep_total_caps(self):
        specs=job_specs()
        self.assertEqual(list(specs),["gpu","cpu-small","cpu-protein83"])
        self.assertEqual(specs["gpu"]["systems"],["nanotube","protein83"])
        self.assertEqual(specs["cpu-small"]["systems"],["nanotube"])
        self.assertEqual(specs["cpu-protein83"]["systems"],["protein83"])
        self.assertEqual(sum(v["allocation_wall_seconds"] for v in specs.values() if v["kind"]=="gpu"),2*28800)
        self.assertEqual(sum(v["allocation_wall_seconds"] for v in specs.values() if v["kind"]=="cpu"),6*28800)
        self.assertEqual(entries("gpu",specs["gpu"]["systems"]),[("nanotube",None),("protein83",None)])
        self.assertEqual(entries("cpu",["nanotube"]),[("nanotube",0),("nanotube",1),("nanotube",2)])
        for kind,wall,flag in (("gpu",57600,"--time=16:00:00"),("cpu",86400,"--time=24:00:00")):
            argv=command(Path("/source"),Path("/run"),kind,"gts-cs207-chemx","inferno",
                         None if kind=="gpu" else "123",wall_seconds=wall)
            self.assertIn(flag,argv)
            self.assertEqual(any(a.startswith("--gres") for a in argv),kind=="gpu")

    def test_grouped_allocation_validates_actual_extended_wall_not_per_case_cap(self):
        job={"Account":"gts-cs207-chemx","QOS":"inferno","Partition":"cpu-small",
             "NumNodes":"1","NumCPUs":"24","CPUs/Task":"24","NumTasks":"1",
             "TimeLimit":"1-00:00:00","MinMemoryNode":"160G","AllocTRES":"cpu=24,mem=160G"}
        manifest={"kind":"cpu","account":"gts-cs207-chemx","qos":"inferno","allocation_wall_seconds":86400}
        with patch.dict(os.environ,{"SLURM_JOB_GPUS":""}):
            validate_allocation(job,manifest)
            with self.assertRaises(RuntimeError):validate_allocation({**job,"TimeLimit":"08:00:00"},manifest)

    def test_grouped_preparation_calls_no_scheduler_and_skips_completed_peptide(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);source=base/"source";source.mkdir()
            paths=[]
            for name in ("config","inputs","release"):
                p=base/(name+".json");p.write_text("{}");paths.append(p)
            args=SimpleNamespace(source=source,root=base/"plan",config=paths[0],config_sha256="c",
                inputs=paths[1],inputs_sha256="i",release=paths[2],release_sha256="r")
            with patch("submit_grouped.source_pin",return_value="helper"),patch(
                    "submit_grouped.read_config",return_value=({"source_pin":"science"},{})),patch(
                    "submit_grouped.require_release"),patch("submit_grouped.load_point") as points,patch(
                    "submit_grouped.subprocess.run") as scheduler:
                prepare(args);scheduler.assert_not_called()
                self.assertEqual([c.args[-1] for c in points.call_args_list],["nanotube","protein83"])
            self.assertEqual(len(json.loads((args.root/"plan.json").read_text())["jobs"]),3)

    def exercise(self,kind,fail=None,unsafe=False,terminate=False):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);root=base/"job";root.mkdir()
            scratch=base/"scratch";scratch.mkdir(mode=0o700)
            manifest={"kind":kind,"systems":["nanotube","protein83"] if kind=="gpu" else ["nanotube"],
                "allocation_wall_seconds":57600 if kind=="gpu" else 86400,
                "helper_pin":"helper","science_pin":"science","gpu_group_job_root":"/gpu",
                "inputs":"/inputs","inputs_sha256":"input-sha"}
            (base/"manifest.json").write_text(json.dumps(manifest))
            calls=[]
            def scientific_case(selected,path):
                calls.append((selected["system"],selected["repeat"],selected["kind"]))
                failed=selected["system"]==fail
                if not (failed and unsafe):shutil.rmtree(os.environ["TMPDIR"])
                save(path/"split-result.json",{
                    "status":"FAILED" if failed else "PASS",
                    "scratch_cleanup":{"complete":not (failed and unsafe)}})
                if failed:
                    if terminate:raise InterruptedError("TERM")
                    raise RuntimeError("deliberate point failure")
            with patch.dict(os.environ,{"TMPDIR":str(scratch),"DISP0_SPLIT_SCRATCH_CREATED":str(scratch),
                                        "PSI_SCRATCH":str(scratch),"SCRATCH":str(scratch)}),patch(
                    "grouped_jobs.run",side_effect=scientific_case),patch(
                    "grouped_jobs.load_point",return_value=({},{"dimensions":"fixture"})),patch(
                    "grouped_jobs.compare_repeat",return_value={"PASS":True}),patch.object(
                    sys,"argv",["grouped","--output",str(root)]):
                if fail:
                    with self.assertRaises(InterruptedError if terminate else RuntimeError):main()
                else:main()
            result=json.loads((root/"group-result.json").read_text())
            if unsafe:
                self.assertTrue(scratch.exists())
                self.assertEqual(len(calls),1)
            else:self.assertFalse(scratch.exists())
            if kind=="gpu" and fail and not unsafe and not terminate:
                self.assertTrue((root/"protein83/job-status.env").exists())
                self.assertFalse((root/"nanotube/job-status.env").exists())
            return calls,result

    def test_GPU_group_never_runs_CPU_cases_and_releases_all_owned_scratch(self):
        calls,result=self.exercise("gpu")
        self.assertEqual(calls,[("nanotube",None,"gpu"),("protein83",None,"gpu")])
        self.assertEqual(result["status"],"PASS_GPU_GROUP_AWAITING_CPU")

    def test_small_CPU_group_runs_three_pairs_and_checks_repeats(self):
        calls,result=self.exercise("cpu")
        self.assertEqual(calls,[("nanotube",i,"cpu") for i in range(3)])
        self.assertEqual(len(result["CPU_repeat_checks"]),2)
        self.assertEqual(result["status"],"PASS_CPU_GROUP_REFERENCES_AND_ORACLES")

    def test_failure_can_preserve_another_distinct_GPU_case_but_never_retry(self):
        calls,result=self.exercise("gpu",fail="nanotube")
        self.assertEqual(len(calls),2)
        self.assertEqual(result["status"],"FAILED")
        self.assertTrue(result["scratch_cleanup"]["complete"])

    def test_TERM_or_unproved_case_cleanup_stops_launching_and_never_deletes_live_scratch(self):
        calls,_=self.exercise("gpu",fail="nanotube",terminate=True)
        self.assertEqual(len(calls),1)
        self.exercise("gpu",fail="nanotube",unsafe=True)


if __name__=="__main__":unittest.main()
