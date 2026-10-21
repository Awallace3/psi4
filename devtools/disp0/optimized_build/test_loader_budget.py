"""CPU-only budget/transport regression; no CUDA library or scheduler needed."""
import importlib.util
import json
import multiprocessing
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent


def driver():
    spec = importlib.util.spec_from_file_location("prepare_inputs", HERE / "prepare_inputs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LoaderBudget(unittest.TestCase):
    def test_large_virtual_mapping_fails_decoder_budget_but_fits_loader_budget(self):
        # Before the fix the closure check inherits the decoder's 1GiB limit.
        # This makes the pre-fix run fail on the actual virtual mapping, not
        # merely on an absent new helper symbol.
        loader_budget = driver().LOADER_AS if (HERE / "prepare_inputs.py").exists() else 1024**3
        code = "import mmap; x=mmap.mmap(-1,1536*1024**2); print('MAPPED_NO_PAGES_TOUCHED'); x.close()"
        results = []
        for budget in (1024**3, loader_budget):
            def limit():
                resource.setrlimit(resource.RLIMIT_AS, (budget, budget))
            results.append(subprocess.run(
                [sys.executable, "-B", "-c", code], preexec_fn=limit,
                capture_output=True, text=True, timeout=5))
        self.assertNotEqual(results[0].returncode, 0)
        self.assertIn("Cannot allocate memory", results[0].stderr)
        self.assertEqual(results[1].returncode, 0, results[1].stderr)
        self.assertIn("MAPPED_NO_PAGES_TOUCHED", results[1].stdout)

    def test_only_canonical_approved_loader_requests_are_accepted(self):
        module = driver()
        with tempfile.TemporaryDirectory() as tmp:
            vendor = Path(tmp)
            lib = vendor / "targets/x86_64-linux/lib"
            lib.mkdir(parents=True)
            (vendor / "lib").mkdir()
            path = lib / "libcusolver.so.12.2.6.9"
            path.write_text("metadata fixture, never loaded")
            env = {"PATH": "/usr/bin:/bin", "LC_ALL": "C",
                   "LD_LIBRARY_PATH": f"{lib}:{vendor}/lib"}
            good = {"argv": ["/usr/bin/ldd", str(path)], "env": env}
            module.validate_request(good, vendor)
            for bad in ({"argv": ["/bin/sh", str(path)], "env": env},
                        {"argv": ["/usr/bin/ldd", str(vendor / "outside.so")], "env": env},
                        {"argv": good["argv"], "env": {**env, "LD_PRELOAD": "/bad.so"}}):
                with self.assertRaises(RuntimeError):
                    module.validate_request(bad, vendor)

    def test_nonzero_loader_reply_preserves_both_streams_and_never_retries(self):
        module = driver()
        response = {"exit": 1, "stdout": "libcublasLt.so.13: failed to map segment",
                    "stderr": "retained stderr", "address_space_bytes": module.LOADER_AS}
        connection = unittest.mock.Mock()
        connection.poll.return_value = True
        connection.recv_bytes.return_value = json.dumps(response).encode()
        trace = module.LoaderTrace(connection, 123)
        with self.assertRaisesRegex(RuntimeError, "failed to map segment") as caught:
            trace(["/usr/bin/ldd", "/fixture"], env={})
        self.assertIn("retained stderr", str(caught.exception))
        self.assertEqual(connection.send_bytes.call_count, 1)
        self.assertEqual(len(trace.evidence["records"]), 1)

    def test_reply_loss_is_fail_closed_without_retry(self):
        module = driver()
        connection = unittest.mock.Mock()
        connection.poll.return_value = False
        trace = module.LoaderTrace(connection, 123)
        with self.assertRaises(TimeoutError):
            trace(["/usr/bin/ldd", "/fixture"], env={})
        self.assertEqual(connection.send_bytes.call_count, 1)

    def test_decoder_caps_are_not_raised_by_new_entrypoint(self):
        module = driver()
        self.assertEqual(module.LOADER_AS, 8 * 1024**3)
        self.assertEqual(module.MAX_AUDIT_BYTES, 1024**2)
        text = (HERE.parent / "provision_sdk_inputs.py").read_text()
        self.assertIn("resource.setrlimit(resource.RLIMIT_AS, (1024**3, 1024**3))", text)
        self.assertIn("signal.alarm(110)", text)
        self.assertIn("MAX_WINDOW_BYTES = 128 * 1024**2", text)

    def test_actual_trace_worker_inherits_wider_budget_before_decoder_is_capped(self):
        # A separate short-lived test process protects the test runner's own
        # hard limit. No ELF is loaded: the worker rejects a non-ldd request.
        code = """
import json,multiprocessing,resource,tempfile
from pathlib import Path
import prepare_inputs as m
with tempfile.TemporaryDirectory() as tmp:
    vendor=Path(tmp)
    (vendor/'targets/x86_64-linux/lib').mkdir(parents=True)
    (vendor/'lib').mkdir()
    ctx=multiprocessing.get_context('fork')
    client,server=ctx.Pipe()
    process=ctx.Process(target=m.loader_worker,args=(server,vendor,vendor/'raw.jsonl',client))
    process.start()
    server.close()
    m.inputs.apply_input_limits()
    client.send_bytes(json.dumps({'argv':['/bin/echo','not executed'],'env':{}}).encode())
    assert client.poll(5)
    response=json.loads(client.recv_bytes(m.MAX_FRAME))
    client.close()
    process.join(2)
    assert process.exitcode == 0
    assert response['exit'] == -1
    assert response['address_space_bytes'] == m.LOADER_AS
    assert resource.getrlimit(resource.RLIMIT_AS)[0] == 1024**3
    print('SEPARATE_BUDGETS_OBSERVED_NO_LIBRARY_LOADED')
"""
        env = dict(__import__("os").environ, PYTHONPATH=str(HERE))
        result = subprocess.run([sys.executable, "-B", "-c", code], env=env,
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("SEPARATE_BUDGETS_OBSERVED_NO_LIBRARY_LOADED", result.stdout)

    def worker_reply(self, effect):
        module = driver()
        with tempfile.TemporaryDirectory() as tmp:
            vendor = Path(tmp)
            lib = vendor / "targets/x86_64-linux/lib"
            lib.mkdir(parents=True)
            (vendor / "lib").mkdir()
            path = lib / "libcusolver.so.12.2.6.9"
            path.write_bytes(b"FAKE_METADATA_ONLY_NOT_LOADED")
            request = {"argv": ["/usr/bin/ldd", str(path)],
                       "env": {"PATH": "/usr/bin:/bin", "LC_ALL": "C",
                               "LD_LIBRARY_PATH": f"{lib}:{vendor}/lib"}}
            ctx = multiprocessing.get_context("fork")
            client, server = ctx.Pipe()
            log = vendor / "raw.jsonl"
            # The patched command is inherited by the fork; the actual worker
            # limit, framing, raw persistence and acknowledgement paths run.
            with patch.object(module.subprocess, "run", side_effect=effect):
                worker = ctx.Process(target=module.loader_worker, args=(server, vendor, log, client))
                worker.start()
            server.close()
            try:
                client.send_bytes(json.dumps(request).encode())
                self.assertTrue(client.poll(5))
                response = json.loads(client.recv_bytes(module.MAX_FRAME))
                worker.join(2)
                self.assertEqual(worker.exitcode, 0)
                self.assertEqual(json.loads(log.read_text()), response)
                return response
            finally:
                client.close()
                if worker.is_alive():
                    worker.terminate()
                worker.join(2)

    def test_worker_persists_timeout_partial_stdout_and_stderr(self):
        def timeout(*args, **kwargs):
            raise subprocess.TimeoutExpired(args[0], 10, output=b"partial mapping stdout",
                                            stderr=b"partial stderr")
        response = self.worker_reply(timeout)
        self.assertEqual(response["status"], "TIMEOUT")
        self.assertEqual(response["exit"], -1)
        self.assertEqual(response["stdout"], "partial mapping stdout")
        self.assertEqual(response["stderr"], "partial stderr")
        self.assertFalse(response["stdout_truncated"])

    def test_worker_preserves_bounded_oversized_output_and_fails_closed(self):
        def oversized(*args, **kwargs):
            return subprocess.CompletedProcess(args[0], 0, "x" * 5000, "retained stderr")
        response = self.worker_reply(oversized)
        self.assertEqual(response["status"], "PASS_OUTPUT_LIMIT")
        self.assertEqual(response["exit"], -1)
        self.assertEqual(response["stdout_bytes"], 5000)
        self.assertEqual(len(response["stdout"]), 4096)
        self.assertTrue(response["stdout_truncated"])
        self.assertEqual(response["stderr"], "retained stderr")


if __name__ == "__main__":
    unittest.main()
