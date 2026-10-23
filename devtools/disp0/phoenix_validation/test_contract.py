"""Offline checks; these are not native build, import or GPU evidence."""
import importlib.util
import hashlib
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location("phoenix_" + name, HERE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Contract(unittest.TestCase):
    def test_header_stream_uses_POSIX_text_order_not_path_component_order(self):
        validate = load("validate")
        with tempfile.TemporaryDirectory() as directory:
            include = Path(directory)
            (include / "cuest").mkdir()
            (include / "cuest.h").write_bytes(b"root header")
            (include / "cuest/context.h").write_bytes(b"context header")
            files = list(include.rglob("*.h"))
            self.assertNotEqual([p.relative_to(include).as_posix() for p in sorted(files)],
                                sorted(p.relative_to(include).as_posix() for p in files))
            stream = b"".join(name.encode() + b"\0" + hashlib.sha256(data).hexdigest().encode() + b"\n"
                              for name, data in (("cuest.h", b"root header"),
                                                 ("cuest/context.h", b"context header")))
            self.assertEqual(validate.header_identity(include),
                             (2, hashlib.sha256(stream).hexdigest()))
            (include / "cuest/context.h").write_bytes(b"changed")
            self.assertNotEqual(validate.header_identity(include)[1],
                                hashlib.sha256(stream).hexdigest())

    def test_job_payload_is_present(self):
        self.assertTrue((HERE / "payload.sh").is_file())
        self.assertIn("phoenix_validation/payload.sh", (HERE / "job.sbatch").read_text())

    def test_explicit_free_cpu_shape_and_absolute_logs(self):
        submit = load("submit")
        argv = submit.command(Path("/source"), Path("/fresh-run"))
        for value in ("--qos=embers", "--account=gts-cs207-chemx", "--mem=48G",
                      "--cpus-per-task=8", "--no-requeue", "--time=03:00:00"):
            self.assertIn(value, argv)
        self.assertFalse(any(x.startswith(("--gres", "--gpus")) for x in argv))
        self.assertIn("--output=/fresh-run/logs/%x-%j.out", argv)

    def test_inherited_scheduler_overrides_are_removed(self):
        self.assertEqual(load("submit").clean_environment(
            {"SBATCH_QOS": "inferno", "SBATCH_GRES": "gpu:1", "PATH": "/bin"}),
            {"PATH": "/bin"})

    def test_inferno_is_an_explicit_same_resource_shape(self):
        submit = load("submit")
        argv = submit.command(Path("/source"), Path("/fresh-run"),
                              qos="inferno", account="gts-cs207")
        self.assertIn("--qos=inferno", argv)
        self.assertNotIn("--qos=embers", argv)
        self.assertIn("--account=gts-cs207", argv)
        self.assertIn("--time=03:00:00", argv)
        self.assertIn("--cpus-per-task=8", argv)
        self.assertFalse(any(x.startswith(("--gres", "--gpus")) for x in argv))

    def test_missing_sdk_headers_reject_before_submission(self):
        submit = load("submit")
        source = HERE.parents[2]
        with tempfile.TemporaryDirectory() as directory:
            sdk = Path(directory)
            with self.assertRaisesRegex(RuntimeError, "cusparse.h"):
                submit.validate_sdk_shape(source, sdk)

    def test_dependency_world_must_match_exact_pin_and_prefix(self):
        validate = load("validate")
        world = {"git_commit": validate.E2_PIN, "version_major": 2,
                 "compiler_major": 14, "library_path": "/e2/lib/libEinsums.so"}
        self.assertEqual(validate.validate_world(world, Path("/e2")),
                         Path("/e2/lib/libEinsums.so"))
        for key, value in (("git_commit", "old"), ("version_major", 1),
                           ("compiler_major", 12), ("library_path", "/foreign/libEinsums.so")):
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                validate.validate_world({**world, key: value}, Path("/e2"))

    def test_proof_files_cannot_overwrite_previous_attempts(self):
        validate = load("validate")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "proof.json"
            validate.save(path, {"status": "PASS"})
            with self.assertRaises(FileExistsError):
                validate.save(path, {"status": "different"})


if __name__ == "__main__":
    unittest.main()
