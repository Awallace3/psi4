"""Real local process-control tests; no Slurm, network or quantum execution."""
from pathlib import Path
import os
import select
import signal
import subprocess
import sys
import time

import pytest


HELPER = Path(__file__).with_name('term_wait.sh')
CHILD = r"""
from pathlib import Path
import signal,sys,time
marker,mode,delay=sys.argv[1:]
def cleanup(signum, frame):
    time.sleep(float(delay))
    Path(marker).write_text('cleanup completed')
    sys.exit(0)
if mode=='ignore':
    signal.signal(signal.SIGTERM,signal.SIG_IGN)
else:
    signal.signal(signal.SIGTERM,cleanup)
print('READY',flush=True)
if mode.startswith('exit'):
    sys.exit(int(mode[4:]))
while True:
    signal.pause()
"""


def launch(tmp_path, mode, delay='0.25', grace='2'):
    marker = tmp_path/'cleanup'
    script = r'''
set -euo pipefail
source "$1"
setsid "$2" -u -c "$3" "$4" "$5" "$6" &
child=$!
if ccasp_wait_child "$child" "$7"; then rc=0; else rc=$?; fi
exit "$rc"
'''
    proc = subprocess.Popen(
        ['/bin/bash', '-c', script, '_', str(HELPER), sys.executable,
         CHILD, str(marker), mode, delay, grace],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
    ready, _, _ = select.select([proc.stdout], [], [], 4)
    if not ready or proc.stdout.readline().strip() != 'READY':
        proc.kill()
        out, err = proc.communicate(timeout=3)
        pytest.fail('child did not initialize: '+out+err)
    return proc, marker


def test_term_waits_for_delayed_child_cleanup_before_returning_interrupted(tmp_path):
    proc, marker = launch(tmp_path, 'cleanup')
    started = time.monotonic()
    os.kill(proc.pid, signal.SIGTERM)
    out, err = proc.communicate(timeout=6)
    assert proc.returncode == 143, out+err
    assert marker.read_text() == 'cleanup completed'
    assert time.monotonic()-started >= .2


def test_term_cleanup_is_bounded_when_child_ignores_signal(tmp_path):
    proc, marker = launch(tmp_path, 'ignore', grace='1')
    started = time.monotonic()
    os.kill(proc.pid, signal.SIGTERM)
    out, err = proc.communicate(timeout=6)
    assert proc.returncode == 143, out+err
    assert not marker.exists()
    assert .8 <= time.monotonic()-started < 5


@pytest.mark.parametrize('exit_code', [0, 7])
def test_unsignalled_exit_status_is_preserved(tmp_path, exit_code):
    proc, marker = launch(tmp_path, 'exit'+str(exit_code))
    out, err = proc.communicate(timeout=4)
    assert proc.returncode == exit_code, out+err
    assert not marker.exists()


def test_batch_uses_explicit_payload_step_and_phoenix_non_batch_only_warning():
    text = HELPER.with_name('phoenix_pilot.sbatch').read_text()
    assert '#SBATCH --signal=TERM@180' in text
    assert '#SBATCH --signal=B:' not in text
    assert 'setsid /opt/slurm/current/bin/srun ' in text
    assert 'source "$SOURCE/devtools/isapol_s66_campaign/term_wait.sh"' in text
    assert 'ccasp_wait_child "$child" 20' in text
