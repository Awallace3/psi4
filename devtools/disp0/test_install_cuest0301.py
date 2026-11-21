import copy
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    "install_cuest0301", Path(__file__).with_name("install_cuest0301.py")
)
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


class PackageIdentityTests(unittest.TestCase):
    def test_exact_cuda13_packages(self):
        for name, sha in installer.PACKAGES.items():
            record = dict(
                name=name, version=installer.VERSION, build=installer.BUILD,
                subdir="linux-64", sha256=sha,
                url=f"https://conda.anaconda.org/conda-forge/linux-64/{name}-{installer.VERSION}-{installer.BUILD}.conda",
            )
            installer.validate_record(record, name)
            for field in record:
                with self.subTest(package=name, field=field):
                    bad = copy.deepcopy(record)
                    bad[field] = "wrong"
                    with self.assertRaises(RuntimeError):
                        installer.validate_record(bad, name)


if __name__ == "__main__":
    unittest.main()
