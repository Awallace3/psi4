"""Replay the real CPU payload with only OS/job-launch boundaries stubbed."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

HERE=Path(__file__).resolve().parent


class PayloadScratch(unittest.TestCase):
    def test_CPU_uses_scheduler_writable_scratch_not_unowned_scratch_job_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp)
            scheduler=base/"scheduler-writable";scheduler.mkdir(mode=0o700)
            output=base/"outputs";output.mkdir()
            keep=output/"preserve.json";keep.write_text("preserve")
            commands=base/"commands";commands.mkdir()
            environment=base/"environment";(environment/"bin").mkdir(parents=True)
            def executable(path,text):
                path.write_text(text);path.chmod(0o755)
            executable(commands/"git","""#!/bin/sh
case "$*" in
  *"rev-parse HEAD") printf 'fixture-pin\\n';;
  *"status --porcelain") :;;
  *) exit 2;;
esac
""")
            # Reproduce the cluster's permission boundary without attempting
            # to write any real /scratch parent on the local workstation.
            executable(commands/"mkdir","""#!/bin/sh
for target do :; done
case "$target" in
  "$FIXTURE_SCHEDULER_BASE"/*) exec /bin/mkdir "$@";;
  *) printf 'mkdir: Permission denied outside scheduler scratch: %s\\n' "$target" >&2; exit 1;;
esac
""")
            executable(environment/"bin/python","""#!/bin/sh
printf 'SELECTED_TMPDIR=%s\\n' "$TMPDIR"
""")
            env={**os.environ,"PATH":str(commands)+":/usr/bin:/bin",
                 "DISP0_SOURCE":str(base/"source"),"DISP0_PIN":"fixture-pin",
                 "DISP0_JOB_RUN":str(output),"DISP0_GATE_ENV":str(environment),
                 "DISP0_SPLIT_THREADS":"24","DISP0_SPLIT_KIND":"cpu",
                 "SLURM_JOB_ID":"99999999","SLURM_CPUS_PER_TASK":"24","SLURM_JOB_GPUS":"",
                 "TMPDIR":str(scheduler),"SLURM_TMPDIR":str(scheduler),
                 "FIXTURE_SCHEDULER_BASE":str(scheduler)}
            result=subprocess.run(["/bin/bash",str(HERE/"split_payload.sh")],
                env=env,capture_output=True,text=True,timeout=10)
            self.assertEqual(result.returncode,0,result.stderr)
            selected=next(line.split("=",1)[1] for line in result.stdout.splitlines()
                          if line.startswith("SELECTED_TMPDIR="))
            self.assertTrue(Path(selected).is_relative_to(scheduler))
            self.assertEqual(keep.read_text(),"preserve")


if __name__=="__main__":unittest.main()
