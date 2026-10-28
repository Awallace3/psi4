"""Real blocking-writer/forced-break contract tests; CPU only, no sysctl edits."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import textwrap
import unittest

from test_cuest_owned_runtime import PREFIX, SAPT


@unittest.skipUnless(sys.platform.startswith("linux"),"Linux kernel lease contract")
class BlockingLeaseBreak(unittest.TestCase):
    def exercise(self, *, suppress_notification, seconds):
        with tempfile.TemporaryDirectory(prefix="disp0-owned-blocking-",
                dir="/dev/shm" if Path("/dev/shm").is_dir() else None) as directory:
            body = f'''
                owner=owned.admit_owned_runtime(pins)
                suppress={suppress_notification!r}
                if suppress:
                    previous=signal.pthread_sigmask(signal.SIG_BLOCK,{{signal.SIGIO}})
                print("READY",flush=True)
                command=sys.stdin.readline()
                if suppress:
                    assert command=="CHECK\\n",command
                    assert signal.sigtimedwait({{signal.SIGIO}},0) is not None
                    signal.pthread_sigmask(signal.SIG_SETMASK,previous)
                owner.check_live()
                print("UNSAFE result published",flush=True)
            '''
            owner = writer = None
            try:
                owner = subprocess.Popen(
                    [sys.executable,"-B","-c",PREFIX+"\n"+textwrap.dedent(body),str(SAPT),directory],
                    stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
                self.assertEqual(owner.stdout.readline(),"READY\n")
                path = Path(directory)/"libfixture.so.1"
                writer = subprocess.Popen([sys.executable,"-B","-c",
                    "import sys\nwith open(sys.argv[1],'wb',buffering=0) as f:f.write(b'forced replacement')\n"
                    "print('WROTE',flush=True)\n",str(path)],
                    stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
                out,err = writer.communicate(timeout=seconds+15)
                self.assertEqual(writer.returncode,0,err)
                self.assertEqual(out,"WROTE\n")
                self.assertEqual(path.read_bytes(),b"forced replacement")
                if suppress_notification:
                    # Owner is still alive and never voluntarily unlocked. The
                    # writer proceeded only after the kernel forced the break.
                    self.assertIsNone(owner.poll())
                    owner.stdin.write("CHECK\n")
                    owner.stdin.flush()
                out,err = owner.communicate(timeout=10)
                self.assertEqual(owner.returncode,86 if suppress_notification else int(signal.SIGIO),
                                 out+"\n"+err)
                self.assertNotIn("UNSAFE",out)
                # No surviving kernel lease after either worker termination.
                descriptor = os.open(path,os.O_WRONLY|os.O_NONBLOCK)
                os.close(descriptor)
            finally:
                for process in (writer,owner):
                    if process is not None and process.poll() is None:
                        process.kill()  # Only our test children, never other PIDs.
                        process.wait(timeout=5)
                    if process is not None:
                        for stream in (process.stdin,process.stdout,process.stderr):
                            if stream is not None:stream.close()

    def test_blocking_writer_is_released_by_fatal_worker_exit(self):
        self.exercise(suppress_notification=False,seconds=5)

    def test_forced_break_cannot_publish_when_notification_was_suppressed(self):
        seconds = int(Path("/proc/sys/fs/lease-break-time").read_text())
        if not 1 <= seconds <= 60:
            self.skipTest("Forced-break integration test needs an existing 1..60-second kernel policy")
        self.exercise(suppress_notification=True,seconds=seconds)


if __name__ == "__main__":
    unittest.main()
