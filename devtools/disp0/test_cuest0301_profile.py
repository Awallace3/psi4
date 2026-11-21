"""Host unit tests for authenticated conda-forge cuEST 0.3.0.1 SDK profile."""
import ctypes as ct
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

SAPT = Path(__file__).resolve().parents[2] / "psi4/driver/procrouting/sapt"
PACKAGE = "_disp0_cuest0301_fixtures"
if PACKAGE not in sys.modules:
    pkg = types.ModuleType(PACKAGE)
    pkg.__path__ = [str(SAPT)]
    sys.modules[PACKAGE] = pkg

identity = importlib.import_module(f"{PACKAGE}.sapt_cuest250_identity")
factory_abi = importlib.import_module(f"{PACKAGE}.sapt_cuest_factory_abi")


def _make_conda_0301_fixture(root: Path, *, extra_unrelated_header: bool = True):
    sdk = root / "sdk"
    header_payloads = {
        "include/cuest.h": b"/* cuest 0.3.0.1 root */\n",
        "include/cuest/context/context_api.h": b"/* context */\n",
        "include/cuest/types/cuest_parameter_types.h": b"/* params */\n",
        "include/cuest/types/workspace_api.h": b"/* ws */\n",
        "include/cuest/parameters/parameter_api.h": b"/* param_api */\n",
        "include/cuest/density_fitted_integrals/density_fitted_mo_integrals_compute_api.h": b"/* dfmo */\n",
        "include/cuest/integral_plans/density_fitted_integral_plan_api.h": b"/* dfplan */\n",
    }
    for rel, data in header_payloads.items():
        path = sdk / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    if extra_unrelated_header:
        # Unrelated header in a conda prefix must NOT trigger rejection.
        (sdk / "include/unrelated_pkg.h").write_bytes(b"/* unrelated */\n")
    native = sdk / identity.NATIVE_0301_RELPATH
    native.parent.mkdir(parents=True, exist_ok=True)
    native.write_bytes(b"FAKE_CUEST_0301_ELF_FIXTURE")
    meta_dir = sdk / "conda-meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    for pkg_name, pkg_sha in identity.SDK_0301_PACKAGES.items():
        files = (
            [identity.NATIVE_0301_RELPATH, "lib/libcuest.so.0"]
            if pkg_name == "libcuest"
            else [*sorted(header_payloads), "lib/libcuest.so"]
        )
        record = {
            "name": pkg_name,
            "version": identity.SDK_0301_PACKAGE_VERSION,
            "build": identity.SDK_0301_PACKAGE_BUILD,
            "subdir": "linux-64",
            "sha256": pkg_sha,
            "url": (
                f"https://conda.anaconda.org/conda-forge/linux-64/"
                f"{pkg_name}-{identity.SDK_0301_PACKAGE_VERSION}-{identity.SDK_0301_PACKAGE_BUILD}.conda"
            ),
            "files": files,
        }
        (meta_dir / f"{pkg_name}-{identity.SDK_0301_PACKAGE_VERSION}-{identity.SDK_0301_PACKAGE_BUILD}.json").write_text(
            json.dumps(record)
        )
    records = {
        rel: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        for rel, data in header_payloads.items()
    }
    tree_sha, nul_sha = identity.header_digests_0301(records)
    req = {rel: records[rel]["sha256"] for rel in header_payloads}
    patches = {
        "HEADER_0301_COUNT": len(records),
        "HEADER_0301_JSON_SHA256": tree_sha,
        "HEADER_0301_NUL_SHA256": nul_sha,
        "REQUIRED_HEADERS_0301": req,
        "NATIVE_0301_BYTES": native.stat().st_size,
        "NATIVE_0301_SHA256": hashlib.sha256(native.read_bytes()).hexdigest(),
    }
    return sdk, native, patches


class TestCuEST0301Profile(unittest.TestCase):
    def test_verify_files_0301_admits_sdk_headers_and_ignores_unrelated_prefix_headers(self):
        with tempfile.TemporaryDirectory() as tmp:
            sdk, native, patches = _make_conda_0301_fixture(Path(tmp), extra_unrelated_header=True)
            with patch.multiple(identity, **patches):
                proof = identity.verify_files(sdk, None)
                self.assertEqual(proof["sdk_version"], "0.3.0.1")
                self.assertEqual(proof["sdk_profile"], "conda-forge-0.3.0.1-cuda13")
                self.assertIsNone(proof["archive_sha256"])
                self.assertEqual(proof["native_path"], str(native.resolve()))
                self.assertEqual(proof["native_sha256"], patches["NATIVE_0301_SHA256"])
                self.assertEqual(identity.expected_sdk_version(proof), (0, 3, 0))
                self.assertEqual(identity.expected_native_sha256(proof), patches["NATIVE_0301_SHA256"])

                # Foreign header inside include/cuest/ must fail closed.
                foreign = sdk / "include/cuest/injected.h"
                foreign.write_bytes(b"/* injected */")
                with self.assertRaisesRegex(RuntimeError, "header inventory differs"):
                    identity.verify_files(sdk, None)
                foreign.unlink()

                # Mutating a canonical header must fail closed.
                orig = (sdk / "include/cuest.h").read_bytes()
                (sdk / "include/cuest.h").write_bytes(b"X" * len(orig))
                with self.assertRaisesRegex(RuntimeError, "required header signatures differ"):
                    identity.verify_files(sdk, None)
                (sdk / "include/cuest.h").write_bytes(orig)

                # Wrong package hash in conda-meta must fail closed.
                meta_path = (
                    sdk / "conda-meta"
                    / f"libcuest-{identity.SDK_0301_PACKAGE_VERSION}-{identity.SDK_0301_PACKAGE_BUILD}.json"
                )
                bad_meta = json.loads(meta_path.read_text())
                bad_meta["sha256"] = "0" * 64
                meta_path.write_text(json.dumps(bad_meta))
                with self.assertRaisesRegex(RuntimeError, "Unexpected conda-forge cuEST 0.3.0.1"):
                    identity.verify_files(sdk, None)

    def test_native_version_accepts_030_only_when_expected_and_rejects_mismatch(self):
        vtype = ct.CFUNCTYPE(ct.c_int, ct.c_void_p, ct.POINTER(ct.c_uint32))

        def make_native(ver):
            fns = {}
            for name, val in zip(("Major", "Minor", "Patch"), ver):
                def cb(handle, out, v=val):
                    out[0] = v
                    return 0
                fns[f"cuestGet{name}Version"] = vtype(cb)
            return types.SimpleNamespace(**fns)

        native_030 = make_native((0, 3, 0))
        self.assertEqual(identity.native_version(native_030, 1234, expected=(0, 3, 0)), (0, 3, 0))
        with self.assertRaisesRegex(RuntimeError, "must be0.2.50"):
            identity.native_version(native_030, 1234)
        native_0250 = make_native((0, 2, 50))
        with self.assertRaisesRegex(RuntimeError, "must be 0.3.0"):
            identity.native_version(native_0250, 1234, expected=(0, 3, 0))

    def test_factory_df_plan_configures_0301_weighting_and_disables_emulation(self):
        configured = {}

        def cfg(kind, ptr, attr, buf, size):
            configured[(kind, attr)] = ct.string_at(buf, size)
            return 0

        def qry(kind, ptr, attr, buf, size):
            raw = configured[(kind, attr)]
            ct.memmove(buf, raw, size)
            return 0

        fake_native = types.SimpleNamespace(
            cuestParametersConfigure=cfg,
            cuestParametersQuery=qry,
        )
        fake_session = types.SimpleNamespace(check=lambda: None)
        api_0301 = factory_abi._FactoryAPI(
            fake_session, fake_native, {"sdk_version": "0.3.0.1"}, {}, {}
        )
        res_0301 = api_0301.configure_df_plan_parameters(0x1234)
        self.assertEqual(res_0301["cache_operator_weighting"], 0)
        self.assertEqual(res_0301["three_index_fitting_emulation_mode"], 1)
        self.assertEqual(res_0301["fitting_algorithm"], 1)

        configured.clear()
        api_0250 = factory_abi._FactoryAPI(
            fake_session, fake_native, {"sdk_version": "0.2.50"}, {}, {}
        )
        res_0250 = api_0250.configure_df_plan_parameters(0x1234)
        self.assertNotIn("cache_operator_weighting", res_0250)
        self.assertNotIn("three_index_fitting_emulation_mode", res_0250)

    def test_phoenix_sealed_h200_configuration_supports_0301(self):
        pv = Path(__file__).resolve().parent / "phoenix_validation"
        sys.path.insert(0, str(pv))
        try:
            import sealed_h200
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp).resolve()
                sdk_lib = root / "sdk/lib"
                sdk_lib.mkdir(parents=True)
                (sdk_lib / "libcuest.so.0.3.0").write_bytes(b"fixture")
                rows = []
                for name in sealed_h200.PRELOAD[:-1]:
                    p = root / (name + ".123")
                    p.write_bytes(b"fixture")
                    rows.append({"path": str(p), "sha256": "fixture"})
                base = {
                    "schema": 1, "device_family": "H200", "source_pin": "pin",
                    "cuest_root": str(root / "sdk"), "cuest_archive": None,
                    "cuest_sdk_version": "0.3.0.1", "dependency_environment": "/env",
                }
                build = {"status": "PASS", "pin": "pin"}
                with patch.object(sealed_h200, "require_cuda_manifest", return_value={"components": rows}):
                    cfg = sealed_h200.configuration(base, build, root)
                self.assertEqual(len(cfg["native_preload_paths"]), 9)
                self.assertTrue(cfg["native_preload_paths"][-1].endswith("libcuest.so.0.3.0"))
        finally:
            sys.path.remove(str(pv))


if __name__ == "__main__":
    unittest.main()
