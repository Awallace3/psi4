# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""Synthetic, engine-free tests. No actual molecular/native calculation."""
import importlib.util
from contextlib import contextmanager, ExitStack
import hashlib
import json
from pathlib import Path
import struct
import unittest
import sys
import tempfile
import os
import subprocess
from types import ModuleType, SimpleNamespace
from unittest import mock

SPEC = importlib.util.spec_from_file_location("synthetic_lw_capture", Path(__file__).with_name("lw_capture.py"))
LW = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = LW
SPEC.loader.exec_module(LW)


def f64(shape, values):
    return LW.Float64Array(tuple(shape), b"".join(struct.pack("<d", x) for x in values))


def identities():
    source = dict(source_root="/synthetic/source", source_commit="1" * 40, source_tree="2" * 40)
    return dict(
        native_core=dict(source, path="/synthetic/stage/lib/psi4/core.so",
                         sha256="3" * 64, build_node="synthetic-node"),
        python_boundary=dict(source, path="/synthetic/stage/lib/psi4/driver/procrouting/isapol_lw.py",
                             package_root="/synthetic/stage/lib/psi4", sha256="4" * 64),
        driver=source.copy(), config_sha256="5" * 64, basis_manifest_sha256="6" * 64,
        fixture_sha256="7" * 64)


def inputs():
    values = [0.] * 2500
    values[-1] = -0.
    return dict(tensors=f64((1, 2, 2, 25, 25), values),
        origins=f64((2, 3), [0., 0., 0., 1., 2., 3.]),
        frames=f64((2, 3, 3), [1., 0., 0., 0., 1., 0., 0., 0., 1.] * 2),
        labels=["Left", "Right"], bonds=[(1, 0)], frequency=0.,
        identities=identities(), kind="synthetic")


def asymmetric_inputs():
    kw = inputs()
    data = bytearray(kw["tensors"].data)
    for a, b, t, u, value in ((0, 0, 1, 0, 2.), (0, 1, 1, 0, -5.),
                              (0, 0, 0, 2, -1.), (1, 0, 0, 2, 7.)):
        struct.pack_into("<d", data, (((a * 2 + b) * 25 + t) * 25 + u) * 8, value)
    kw["tensors"] = LW.Float64Array((1, 2, 2, 25, 25), bytes(data))
    return kw


def dense_three_site_inputs():
    """Dense literal-rule toy, not molecular data; all arithmetic is binary-exact."""
    rows = []
    for r in range(75):
        a, t = divmod(r, 25)
        for col in range(75):
            b, u = divmod(col, 25)
            if t == u == 0:
                value = (a+1)*(b+1)/16
            elif u == 0:
                value = (a+1)*(b+1)*t/8
                if (a, t) == (1, 8):
                    value = 6*(b+1)  # potential-axis tie at +36
            elif t == 0:
                value = -(a+1)*(b+1)*u/4
                if (b, u) in ((1, 8), (2, 7)):
                    value = -6*(a+1)  # response-axis ties at -36
            else:
                value = (r*128+col+1)/8 * (-1 if (r+2*col) % 3 == 0 else 1)
            if (r, col) in ((17, 68), (42, 10)):
                value = -0.  # rank4/rank4 and rank4/rank3, respectively
            rows.append(value)
    raw = f64((75,75), rows)
    return dict(raw_site_major=raw, tensors=LW.site_major_to_pairs(raw),
        origins=f64((3,3), [11.,-2.,.5, 12.,3.,-.25, 9.,1.,2.]),
        frames=f64((3,3,3), [0.,-1.,0., 1.,0.,0., 0.,0.,1.,
                              1.,0.,0., 0.,0.,-1., 0.,1.,0.,
                              0.,0.,1., 1.,0.,0., 0.,1.,0.]),
        labels=["S0","S1","S2"], bonds=[(2,0),(1,2)], frequency=.375,
        identities=identities(), kind="synthetic")


def refusal(value="6", off_site="0"):
    return RuntimeError(
        "localize_lw: postcondition exceeds residual tolerance "
        f"(off-site={off_site}, charge-sum={value}, reciprocity=0, molecular-sum=0, "
        f"local-charge={value}, input-sum-rule={value}, charge-sum-transport=0)")


class LosslessCaptureTests(unittest.TestCase):
    def stored(self, tmp, kw=None):
        receipt = LW.snapshot(Path(tmp) / "capture", **(asymmetric_inputs() if kw is None else kw))
        return receipt, LW.load_capture(receipt.path, receipt.manifest_sha256)

    def repin(self, path, manifest):
        encoded = json.dumps(manifest, allow_nan=False).encode()
        (path / "manifest.json").write_bytes(encoded)
        return hashlib.sha256(encoded).hexdigest()

    def test_site_major_requires_transpose_not_reshape_only(self):
        data = bytearray(50 * 50 * 8)
        struct.pack_into("<d", data, (1 * 50 + 27) * 8, 1001.25)  # a0,t1,b1,u2
        raw = LW.Float64Array((50, 50), bytes(data))
        pairs = LW.site_major_to_pairs(raw)
        index = ((0 * 2 + 1) * 25 + 1) * 25 + 2
        self.assertEqual(struct.unpack_from("<d", pairs.data, index * 8)[0], 1001.25)
        self.assertEqual(pairs.shape, (1, 2, 2, 25, 25))
        self.assertEqual(raw.data, bytes(data))

    def test_exact_binary_roundtrip_preserves_signed_zero_and_ordered_bond(self):
        original = inputs()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "capture"
            receipt = LW.snapshot(path, **original)
            capture = LW.load_capture(path, receipt.manifest_sha256)
            for name in ("tensors", "origins", "frames"):
                self.assertEqual(capture.array(name), original[name])
            self.assertEqual(capture.declaration["bonds"], [[1, 0]])
            self.assertEqual(capture.declaration["labels"], ["Left", "Right"])
            self.assertEqual(capture.array("tensors").data[-8:], struct.pack("<d", -0.))

    def test_dense_three_site_mapping_checks_every_full25_element_byte_for_byte(self):
        kw = dense_three_site_inputs()
        raw, pair = kw["raw_site_major"], kw["tensors"]
        before = raw.data
        checked = 0
        # Invert source matrix coordinates, unlike the converter's pair-row copying.
        for row in range(75):
            response_site, response_component = divmod(row, 25)
            for col in range(75):
                potential_site, potential_component = divmod(col, 25)
                source = (row*75+col)*8
                target = (response_site*1875 + potential_site*625
                          + response_component*25 + potential_component)*8
                self.assertEqual(pair.data[target:target+8], raw.data[source:source+8],
                                 (row, col))
                checked += 1
        self.assertEqual(checked, 5625)
        self.assertEqual(raw.data, before)
        self.assertEqual(pair.shape, (1,3,3,25,25))
        for row, col in ((17,68), (42,10)):
            self.assertEqual(raw.data[(row*75+col)*8:(row*75+col+1)*8],
                             struct.pack("<d", -0.))
        self.assertEqual(struct.unpack_from("<d", pair.data,
            (0*1875+2*625+24*25+24)*8)[0], 393.375)  # independent rank4 marker

    def test_dense_signed_axes_discarded_rank_maximum_and_ties(self):
        kw = dense_three_site_inputs()
        arrays, meta = LW.charge_diagnostics(kw["tensors"], kw["labels"])
        # Closed-form sums of weights 1+2+3=6, not the implementation's site loop.
        potential, response = [], []
        for site in range(3):
            for t in range(25):
                potential.append(3*(site+1)/8 if t == 0 else 3*(site+1)*t/4)
                response.append(3*(site+1)/8 if t == 0 else -3*(site+1)*t/2)
        potential[25+8] = 36.
        response[25+8] = response[50+7] = -36.
        for name in ("sum_potential_site", "fsum_potential_site"):
            self.assertEqual(arrays[name], f64((3,25), potential))
        for name in ("sum_response_site", "fsum_response_site"):
            self.assertEqual(arrays[name], f64((3,25), response))
        ordered = meta["native_ordered"]
        for key, axis, signed in (
            ("potential_site_maximum", "sum_potential_site", 36.),
            ("response_site_maximum", "sum_response_site", -36.),
            ("working_maximum", "sum_potential_site", 36.)):
            maximum = ordered[key]
            self.assertEqual((maximum["axis"], maximum["site"], maximum["component"]),
                             (axis, 1, 8))
            self.assertEqual((maximum["label"], maximum["component_name"]), ("S1","22s"))
            self.assertEqual(maximum["signed_value_hex"], signed.hex())
        full = ordered["full_rank4_maximum"]
        self.assertEqual((full["axis"], full["site"], full["component"]),
                         ("sum_response_site", 2, 24))
        self.assertEqual(full["signed_value_hex"], (-108.).hex())
        self.assertEqual(full["component_name"], "44s")
        self.assertGreater(float.fromhex(full["magnitude_hex"]),
                           float.fromhex(ordered["working_maximum"]["magnitude_hex"]))

    def test_dense_roundtrip_retains_nonzero_frequency_and_nonidentity_frames(self):
        kw = dense_three_site_inputs()
        with tempfile.TemporaryDirectory() as tmp:
            receipt = LW.snapshot(Path(tmp)/"dense-toy", **kw)
            capture = LW.load_capture(receipt.path, receipt.manifest_sha256)
            for name in ("raw_site_major", "tensors", "origins", "frames"):
                self.assertEqual(capture.array(name), kw[name])
            self.assertEqual(capture.declaration["kind"], "synthetic")
            self.assertEqual(capture.declaration["frequency_au_hex"], "0x1.8000000000000p-2")
            self.assertEqual(capture.declaration["labels"], ["S0","S1","S2"])
            self.assertEqual(capture.declaration["bonds"], [[2,0],[1,2]])

    def test_finite_binary_tampering_is_rejected_by_checksum(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "capture"
            receipt = LW.snapshot(path, **inputs())
            binary = path / "tensors.f64"
            data = bytearray(binary.read_bytes())
            struct.pack_into("<d", data, 0, 3.25)  # finite, same size, not the captured bytes
            binary.write_bytes(data)
            with self.assertRaisesRegex(ValueError, "checksum"):
                LW.load_capture(path, receipt.manifest_sha256)

    def test_wrong_policy_metadata_is_refused_before_any_binary_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "capture"
            LW.snapshot(path, **inputs())
            (path / "tensors.f64").rename(path / "not-available")
            manifest_path = path / "manifest.json"
            manifest = json.loads(manifest_path.read_bytes())
            manifest["declaration"]["residual_policy"] = "reported_input_sum_rule"
            encoded = json.dumps(manifest).encode()
            manifest_path.write_bytes(encoded)
            with self.assertRaisesRegex(ValueError, "strict declaration"):
                LW.load_capture(path, hashlib.sha256(encoded).hexdigest())

    def test_both_signed_sum_axes_and_site_component_argmax(self):
        tensors = asymmetric_inputs()["tensors"]
        arrays, metadata = LW.charge_diagnostics(tensors, ["Left", "Right"])
        self.assertEqual(struct.unpack_from("<d", arrays["sum_potential_site"].data, 8)[0], -3.)
        self.assertEqual(struct.unpack_from("<d", arrays["sum_response_site"].data, 16)[0], 6.)
        maximum = metadata["native_ordered"]["working_maximum"]
        self.assertEqual((maximum["axis"], maximum["site"], maximum["component"]),
                         ("sum_response_site", 0, 2))
        self.assertEqual((maximum["label"], maximum["component_name"]), ("Left", "11c"))
        self.assertEqual(maximum["signed_value_hex"], (6.).hex())

    def test_strict_refusing_stub_called_once_is_diagnostic_not_properties(self):
        with tempfile.TemporaryDirectory() as tmp:
            receipt = LW.snapshot(Path(tmp) / "capture", **asymmetric_inputs())
            capture = LW.load_capture(receipt.path, receipt.manifest_sha256)
            before = tuple((k, a.data) for k, a in capture.arrays)
            calls = []
            def strict_boundary(**kw):
                calls.append(kw)
                self.assertEqual((kw["input_rank"], kw["localization_rank_limit"]), (4, 2))
                self.assertEqual(kw["residual_policy"], "production")
                self.assertEqual(kw["truncation"], "discard_rank4_rows_and_columns")
                self.assertEqual(kw["tensors"].shape, (1, 2, 2, 25, 25))
                raise refusal()
            outcome = LW.replay(capture, strict_boundary=strict_boundary)
            self.assertEqual(len(calls), 1)
            self.assertEqual(outcome["status"], "EXPECTED_INPUT_SUM_RULE_REFUSAL")
            self.assertTrue(outcome["diagnostic_replay_passed"])
            self.assertFalse(outcome["scientific_properties_accepted"])
            self.assertEqual(before, tuple((k, a.data) for k, a in capture.arrays))

    def test_native_replay_refuses_local_execution_before_scientific_imports(self):
        kw = asymmetric_inputs()
        kw["kind"] = "native_owned"  # guard-only mock declaration, NOT a real capture
        with tempfile.TemporaryDirectory() as tmp:
            receipt = LW.snapshot(Path(tmp) / "capture", **kw)
            capture = LW.load_capture(receipt.path, receipt.manifest_sha256)
            before = {name for name in sys.modules if name in ("numpy", "psi4")}
            with mock.patch.dict(os.environ, {"SLURM_JOB_ID": ""}):
                with self.assertRaisesRegex(ValueError, "Phoenix compute allocation"):
                    LW.native_replay(capture)
            self.assertEqual(before, {name for name in sys.modules if name in ("numpy", "psi4")})

    def test_compensated_reduction_reports_roundoff_without_repairing_input(self):
        data = bytearray(1 * 3 * 3 * 25 * 25 * 8)
        for b, value in enumerate((1e16, 1., -1e16)):
            struct.pack_into("<d", data, ((b * 25 + 1) * 25) * 8, value)
        raw = LW.Float64Array((1,3,3,25,25), bytes(data))
        arrays, meta = LW.charge_diagnostics(raw, ["S0","S1","S2"])
        self.assertEqual(struct.unpack_from("<d", arrays["sum_potential_site"].data, 8)[0], 0.)
        self.assertEqual(struct.unpack_from("<d", arrays["fsum_potential_site"].data, 8)[0], 1.)
        self.assertEqual(raw.data, bytes(data))
        self.assertIn("never substituted", meta["compensated_role"])

    def test_unrelated_exceptions_are_rethrown_unchanged_without_retry(self):
        errors = [RuntimeError("unrelated factor failure"), ValueError(str(refusal())),
                  refusal(off_site="2e-6"), refusal(value="5")]
        with tempfile.TemporaryDirectory() as tmp:
            _, capture = self.stored(tmp)
            for error in errors:
                calls = []
                def boundary(**kw):
                    calls.append(kw)
                    raise error
                with self.subTest(error=str(error)), self.assertRaises(type(error)) as caught:
                    LW.replay(capture, strict_boundary=boundary)
                self.assertIs(caught.exception, error)
                self.assertEqual(len(calls), 1)

    def test_rounded_historical_exception_cannot_stand_in_for_missing_tensor(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, capture = self.stored(tmp, inputs())  # ZERO toy tensor, not P003 data
            error = refusal(value="1.288e-5")
            def boundary(**kw): raise error
            with self.assertRaises(RuntimeError) as caught:
                LW.replay(capture, strict_boundary=boundary)
            self.assertIs(caught.exception, error)

    def test_strict_success_is_unexpected_not_a_property_publication(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, capture = self.stored(tmp)
            calls = []
            def boundary(**kw):
                calls.append(kw)
                return object()
            outcome = LW.replay(capture, strict_boundary=boundary)
            self.assertEqual(len(calls), 1)
            self.assertEqual(outcome["status"], "UNEXPECTED_STRICT_SUCCESS")
            self.assertFalse(outcome["diagnostic_replay_passed"])
            self.assertFalse(outcome["scientific_properties_accepted"])

    def test_input_refusal_does_not_require_every_supplied_residual_to_fail(self):
        kw = inputs()
        data = bytearray(kw["tensors"].data)
        for a in range(2):
            for b in range(2):
                struct.pack_into("<d", data, (a*2+b)*25*25*8,
                                 9.5e-7 if a == b else 9e-8)
        kw["tensors"] = LW.Float64Array((1,2,2,25,25), bytes(data))
        error = RuntimeError(
            "localize_lw: postcondition exceeds residual tolerance "
            "(off-site=9e-8, charge-sum=1.04e-6, reciprocity=0, molecular-sum=0, "
            "local-charge=9.5e-7, input-sum-rule=1.04e-6, charge-sum-transport=0)")
        with tempfile.TemporaryDirectory() as tmp:
            _, capture = self.stored(tmp, kw)
            calls = []
            def boundary(**kw):
                calls.append(kw)
                raise error
            outcome = LW.replay(capture, strict_boundary=boundary)
            self.assertEqual(outcome["status"], "EXPECTED_INPUT_SUM_RULE_REFUSAL")
            self.assertFalse(outcome["scientific_properties_accepted"])
            self.assertEqual(len(calls), 1)

    def test_rounded_owned_residual_at_gate_is_ambiguous_not_input_only(self):
        # A true 1.0000004e-6 violation prints 1e-6 at native six-digit precision.
        with tempfile.TemporaryDirectory() as tmp:
            _, capture = self.stored(tmp)
            for field in ("off-site", "reciprocity", "molecular-sum", "charge-sum-transport"):
                text = str(refusal()).replace(field + "=0", field + "=1e-6")
                error = RuntimeError(text)
                calls = []
                def boundary(**kw):
                    calls.append(kw)
                    raise error
                with self.subTest(field=field):
                    with self.assertRaises(RuntimeError) as caught:
                        LW.replay(capture, strict_boundary=boundary)
                    self.assertIs(caught.exception, error)
                    self.assertEqual(len(calls), 1)

    def test_mutating_boundary_is_detected_and_original_capture_is_immutable(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, capture = self.stored(tmp)
            original = capture.array("tensors").data
            calls = []
            def bad_boundary(**kw):
                calls.append(kw)
                object.__setattr__(kw["tensors"], "data", b"\0" * len(kw["tensors"].data))
                raise refusal()
            with self.assertRaisesRegex(ValueError, "mutated"):
                LW.replay(capture, strict_boundary=bad_boundary)
            self.assertEqual(len(calls), 1)
            self.assertEqual(capture.array("tensors").data, original)

    def test_invalid_shapes_object_arrays_and_nonfinite_data_are_rejected(self):
        for dtype in ("object", "|O", ">f8"):
            with self.subTest(dtype=dtype), self.assertRaises(ValueError):
                LW.Float64Array((1,), struct.pack("<d", 0.), dtype)
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "nonfinite"):
                f64((1,), [value])
        with self.assertRaises(ValueError):
            LW.Float64Array((1,), bytearray(8))
        with self.assertRaises(ValueError):
            LW.Float64Array((True,), bytes(8))
        for width in (9,16):
            kw = inputs()
            kw["tensors"] = f64((1,2,2,width,width), [0.] * (4*width*width))
            with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(ValueError, "shape"):
                LW.snapshot(Path(tmp) / "capture", **kw)

    def test_capture_paths_are_exclusive_and_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "capture"
            path.mkdir()
            sentinel = path / "sentinel"
            sentinel.write_bytes(b"preserve")
            with self.assertRaises(FileExistsError):
                LW.snapshot(path, **inputs())
            self.assertEqual(sentinel.read_bytes(), b"preserve")
            self.assertEqual([p.name for p in path.iterdir()], ["sentinel"])

    def test_metadata_overflow_wrong_dtype_and_pickle_filename_reject_before_binary(self):
        cases = (("shape", [1,2,2,10**30,25]), ("dtype", "object"),
                 ("file", "tensors.pkl"), ("nbytes", LW.MAX_BYTES + 8))
        for field, value in cases:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                receipt, _ = self.stored(tmp)
                manifest = json.loads((receipt.path / "manifest.json").read_bytes())
                manifest["arrays"]["tensors"][field] = value
                (receipt.path / "tensors.f64").rename(receipt.path / "held")
                pin = self.repin(receipt.path, manifest)
                with self.assertRaisesRegex(ValueError, "declaration|shape|size"):
                    LW.load_capture(receipt.path, pin)

    def test_nonfinite_binary_is_rejected_even_with_recomputed_checksums(self):
        with tempfile.TemporaryDirectory() as tmp:
            receipt, _ = self.stored(tmp)
            manifest = json.loads((receipt.path / "manifest.json").read_bytes())
            file = receipt.path / "tensors.f64"
            data = bytearray(file.read_bytes())
            struct.pack_into("<d", data, 0, float("nan"))
            file.write_bytes(data)
            manifest["arrays"]["tensors"]["sha256"] = hashlib.sha256(data).hexdigest()
            pin = self.repin(receipt.path, manifest)
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                LW.load_capture(receipt.path, pin)

    def test_signed_diagnostic_tampering_is_rejected_even_with_recomputed_checksums(self):
        with tempfile.TemporaryDirectory() as tmp:
            receipt, _ = self.stored(tmp)
            manifest = json.loads((receipt.path / "manifest.json").read_bytes())
            file = receipt.path / "sum_response_site.f64"
            data = bytearray(file.read_bytes())
            struct.pack_into("<d", data, 16, 99.)
            file.write_bytes(data)
            manifest["arrays"]["sum_response_site"]["sha256"] = hashlib.sha256(data).hexdigest()
            pin = self.repin(receipt.path, manifest)
            with self.assertRaisesRegex(ValueError, "signed diagnostic"):
                LW.load_capture(receipt.path, pin)

    def test_symlink_array_and_metadata_size_limit_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            receipt, _ = self.stored(tmp)
            file = receipt.path / "origins.f64"
            file.rename(receipt.path / "held")
            file.symlink_to(receipt.path / "held")
            with self.assertRaises(OSError):
                LW.load_capture(receipt.path, receipt.manifest_sha256)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "capture"
            path.mkdir()
            (path / "manifest.json").write_bytes(b" " * (LW.MAX_METADATA_BYTES + 1))
            with self.assertRaisesRegex(ValueError, "size"):
                LW.load_capture(path, "0" * 64)

    def test_fifo_file_is_rejected_without_waiting_for_a_writer(self):
        # The seam verifies nonblocking open flags without a hanging RED test.
        # The real FIFO is opened only once the required flag is present.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            os.mkfifo(path)
            real_open = os.open
            def require_nonblocking(file, flags):
                self.assertTrue(flags & os.O_NONBLOCK, "capture read could block on a FIFO")
                return real_open(file, flags)
            with mock.patch.object(LW.os, "open", side_effect=require_nonblocking):
                with self.assertRaisesRegex(ValueError, "file type"):
                    LW._read(path, LW.MAX_METADATA_BYTES)

    def test_manifest_pin_and_duplicate_json_keys_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            receipt, _ = self.stored(tmp)
            with self.assertRaisesRegex(ValueError, "manifest checksum"):
                LW.load_capture(receipt.path, "0" * 64)
            data = b'{"schema":"x","schema":"y"}'
            (receipt.path / "manifest.json").write_bytes(data)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                LW.load_capture(receipt.path, hashlib.sha256(data).hexdigest())

    def test_optional_raw_site_major_is_preserved_and_cross_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            receipt = LW.synthetic_example(Path(tmp) / "capture")
            capture = LW.load_capture(receipt.path, receipt.manifest_sha256)
            self.assertEqual(LW.site_major_to_pairs(capture.array("raw_site_major")),
                             capture.array("tensors"))
        kw = asymmetric_inputs()
        kw["raw_site_major"] = f64((50,50), [0.] * 2500)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "capture"
            with self.assertRaisesRegex(ValueError, "disagree"):
                LW.snapshot(path, **kw)
            self.assertFalse(path.exists())

    def test_conventions_rank_policy_and_bond_mutations_are_refused(self):
        cases = (("input_rank", 3), ("localization_rank_limit", 3),
                 ("actual_native_input_width", 9), ("dipoles", ["x","y","z"]),
                 ("coordinate_frame", "recentered"), ("bonds", [[0,0]]),
                 ("bonds", [[True,1]]), ("labels", ["Left","Left"]))
        for field, value in cases:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                receipt, _ = self.stored(tmp)
                manifest = json.loads((receipt.path / "manifest.json").read_bytes())
                manifest["declaration"][field] = value
                with self.assertRaises(ValueError):
                    LW.load_capture(receipt.path, self.repin(receipt.path, manifest))

    def test_blank_labels_are_rejected_without_normalizing_or_writing(self):
        kw = inputs()
        kw["labels"] = [" ", "Right"]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "capture"
            with self.assertRaisesRegex(ValueError, "labels"):
                LW.snapshot(path, **kw)
            self.assertFalse(path.exists())

    def test_module_import_and_example_inspection_are_engine_free(self):
        code = """
import importlib.abc, importlib.util, pathlib, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in ('psi4','numpy','adcc','einsums'):
            raise AssertionError('scientific import '+fullname)
sys.meta_path.insert(0,Block())
p=pathlib.Path(sys.argv[1])
s=importlib.util.spec_from_file_location('isolated_capture',p)
m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m)
m.main(['example',sys.argv[2]])
"""
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(__file__).with_name("lw_capture.py")
            r = subprocess.run([sys.executable, "-I", "-S", "-B", "-c", code, str(file),
                                str(Path(tmp) / "example")], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            receipt = json.loads(r.stdout)
            with mock.patch("sys.stdout"):
                self.assertEqual(LW.main(["inspect", receipt["path"], "--manifest-sha256",
                                          receipt["manifest_sha256"]]), 0)


class FakeReadOnlyArray:
    """Test-only NumPy boundary stand-in; NOT native NumPy/ABI evidence."""
    def __init__(self, data, dtype):
        if type(data) is not bytes or dtype != "<f8":
            raise AssertionError("adapter must provide owned binary64 bytes")
        self.buffer = memoryview(data)
        self.dtype = dtype
        self.flags = SimpleNamespace(writeable=False)

    def reshape(self, shape):
        count = 1
        for n in shape:
            count *= n
        if count*8 != len(self.buffer):
            raise AssertionError("adapter reshape changed size")
        self.shape = shape
        return self


class MockedNativeAdapterTests(unittest.TestCase):
    """Fake allocation/modules/Git only. No native admission or scientific import."""
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        package = self.root/"toy-stage/psi4"
        boundary_file = package/"driver/procrouting/isapol_lw.py"
        boundary_file.parent.mkdir(parents=True)
        self.core_file = package/"core.so"
        self.core_file.write_bytes(b"TEST ONLY, NOT ELF")
        boundary_file.write_bytes(b"# TEST ONLY, NOT scientific package source\n")
        self.ids = identities()
        for index, role in enumerate(("native_core", "python_boundary", "driver"), 1):
            self.ids[role].update(source_root=str(self.root/("toy-source-"+role)),
                source_commit=str(index)*40, source_tree=str(index+4)*40)
        self.ids["native_core"].update(path=str(self.core_file),
            sha256=hashlib.sha256(self.core_file.read_bytes()).hexdigest(), build_node="mock-compute")
        self.ids["python_boundary"].update(path=str(boundary_file), package_root=str(package),
            sha256=hashlib.sha256(boundary_file.read_bytes()).hexdigest())
        self.boundary_file = boundary_file
        self.psi4 = ModuleType("psi4")
        self.psi4.__file__ = str(package/"__init__.py")
        self.psi4.core = SimpleNamespace(__file__=str(self.core_file))
        self.numpy = ModuleType("numpy")
        self.numpy.frombuffer = mock.Mock(side_effect=FakeReadOnlyArray)
        self.public = ModuleType("psi4.driver.procrouting.isapol_lw")
        self.public.__file__ = str(boundary_file)
        self.public.PRODUCTION_TOLERANCE = 1e-6
        self.public.TRUNCATE_RANK4 = "discard_rank4_rows_and_columns"
        self.public.Provenance = mock.Mock(side_effect=lambda *args: SimpleNamespace(args=args))
        self.boundary = mock.Mock(side_effect=refusal())
        self.public.supplied_nonlocal_properties = self.boundary
        self.environment = dict(SLURM_JOB_ID="mock-allocation", USER="awallace43",
            HOME="/storage/home/TEST-ONLY", SLURMD_NODENAME="mock-compute")
        self.hostname = "mock-compute.cluster.invalid"
        self.git_overrides = {}
        self.git = mock.Mock(side_effect=self.git_reply)
        self.lookup = mock.Mock(side_effect=self.import_reply)
        self.capture_index = 0

    def git_reply(self, argv, *, text):
        self.assertTrue(text)
        self.assertEqual(argv[:2], ["git", "-C"])
        item = next(i for role, i in self.ids.items()
                    if role in ("native_core","python_boundary","driver") and i["source_root"] == argv[2])
        query = tuple(argv[3:])
        replies = {("rev-parse","HEAD"): item["source_commit"],
                   ("rev-parse","HEAD^{tree}"): item["source_tree"],
                   ("status","--porcelain"): ""}
        return self.git_overrides.get((argv[2], query), replies[query]) + "\n"

    def import_reply(self, name):
        self.assertEqual(name, "psi4.driver.procrouting.isapol_lw")
        return self.public  # never import a real scientific module

    def capture(self, *, kind="native_owned", build_node="mock-compute", dense=False):
        # native_owned here exercises the guard ONLY; all underlying bytes are toy data.
        kw = dense_three_site_inputs() if dense else asymmetric_inputs()
        kw["kind"] = kind
        kw["identities"] = json.loads(json.dumps(self.ids))
        kw["identities"]["native_core"]["build_node"] = build_node
        self.capture_index += 1
        receipt = LW.snapshot(self.root/f"toy-capture-{self.capture_index}", **kw)
        return LW.load_capture(receipt.path, receipt.manifest_sha256), kw

    @contextmanager
    def runtime(self, *, missing=()):
        for collaborator in (self.git, self.lookup, self.boundary, self.numpy.frombuffer):
            collaborator.reset_mock()
        with ExitStack() as stack:
            stack.enter_context(mock.patch.dict(os.environ, self.environment, clear=True))
            stack.enter_context(mock.patch.object(LW.socket, "gethostname", return_value=self.hostname))
            stack.enter_context(mock.patch.dict(sys.modules, {"psi4": self.psi4, "numpy": self.numpy}))
            for name in missing:
                sys.modules.pop(name, None)
            stack.enter_context(mock.patch.object(LW.subprocess, "check_output", self.git))
            stack.enter_context(mock.patch.object(LW.importlib, "import_module", self.lookup))
            yield

    def refused(self, capture, message):
        with self.assertRaisesRegex(ValueError, message):
            LW.native_replay(capture)
        self.boundary.assert_not_called()
        self.numpy.frombuffer.assert_not_called()

    def assert_sources_checked(self):
        calls = [mock.call(["git","-C",self.ids[role]["source_root"],*query], text=True)
                 for role in ("native_core","python_boundary","driver")
                 for query in (("rev-parse","HEAD"),("rev-parse","HEAD^{tree}"),("status","--porcelain"))]
        self.assertEqual(self.git.call_args_list, calls)

    def test_login_host_is_rejected_even_when_scheduler_and_build_host_match(self):
        self.hostname = self.environment["SLURMD_NODENAME"] = "login-toy"
        capture, _ = self.capture(build_node="login-toy")
        with self.runtime():
            self.refused(capture, "Phoenix compute allocation")
            self.git.assert_not_called()
            self.lookup.assert_not_called()

    def test_scheduler_host_mismatch_is_rejected(self):
        capture, _ = self.capture()
        self.environment["SLURMD_NODENAME"] = "different-compute"
        with self.runtime():
            self.refused(capture, "Phoenix compute allocation")
            self.git.assert_not_called()
            self.lookup.assert_not_called()

    def test_captured_build_node_mismatch_is_rejected(self):
        capture, _ = self.capture(build_node="different-build")
        with self.runtime():
            self.refused(capture, "Phoenix compute allocation")
            self.git.assert_not_called()
            self.lookup.assert_not_called()

    def test_wrong_user_or_home_is_rejected_with_other_allocation_checks_valid(self):
        capture, _ = self.capture()
        for key, value in (("USER","different-user"), ("HOME","/tmp/not-phoenix")):
            previous = self.environment[key]
            self.environment[key] = value
            with self.subTest(key=key), self.runtime():
                self.refused(capture, "Phoenix compute allocation")
                self.git.assert_not_called()
                self.lookup.assert_not_called()
            self.environment[key] = previous

    def test_synthetic_kind_rejection_is_reached_after_valid_allocation_guards(self):
        capture, _ = self.capture(kind="synthetic")
        with self.runtime():
            self.refused(capture, "actual native_owned capture")
            self.git.assert_not_called()
            self.lookup.assert_not_called()

    def test_missing_initialized_modules_is_reached_after_native_kind_guard(self):
        capture, _ = self.capture()
        for missing in (("psi4",), ("numpy",), ("psi4","numpy")):
            with self.subTest(missing=missing), self.runtime(missing=missing):
                self.refused(capture, "preinitialized reviewed runtime")
                self.git.assert_not_called()
                self.lookup.assert_not_called()

    def source_guard_cases(self, query, answer):
        capture, _ = self.capture()
        roles = ("native_core","python_boundary","driver")
        query_order = (("rev-parse","HEAD"),("rev-parse","HEAD^{tree}"),("status","--porcelain"))
        for index, role in enumerate(roles):
            root = self.ids[role]["source_root"]
            self.git_overrides = {(root, query): answer}
            with self.subTest(role=role), self.runtime():
                self.refused(capture, "source identity/clean state differs")
                # Exact query prefix proves the intended role/guard was reached.
                expected = [mock.call(["git","-C",self.ids[previous]["source_root"],*q], text=True)
                            for previous in roles[:index] for q in query_order]
                expected.extend(mock.call(["git","-C",root,*q], text=True)
                                for q in query_order[:query_order.index(query)+1])
                self.assertEqual(self.git.call_args_list, expected)
                self.lookup.assert_not_called()

    def test_each_source_commit_mismatch_is_rejected(self):
        self.source_guard_cases(("rev-parse","HEAD"), "f"*40)

    def test_each_source_tree_mismatch_is_rejected(self):
        self.source_guard_cases(("rev-parse","HEAD^{tree}"), "e"*40)

    def test_each_dirty_source_is_rejected(self):
        self.source_guard_cases(("status","--porcelain"), " M TEST-ONLY")

    def test_wrong_core_path_is_rejected_after_all_sources_match(self):
        capture, _ = self.capture()
        self.psi4.core.__file__ = str(self.root/"wrong-core.so")
        with self.runtime():
            self.refused(capture, "native core/Python package identity differs")
            self.assert_sources_checked()
            self.lookup.assert_not_called()

    def test_wrong_core_hash_is_rejected_after_matching_core_path(self):
        capture, _ = self.capture()
        self.core_file.write_bytes(b"CHANGED TEST ONLY, NOT ELF")
        with self.runtime():
            self.refused(capture, "native core/Python package identity differs")
            self.assert_sources_checked()
            self.lookup.assert_not_called()

    def test_wrong_package_path_is_rejected_after_matching_core_hash(self):
        capture, _ = self.capture()
        self.psi4.__file__ = str(self.root/"wrong-package/__init__.py")
        with self.runtime():
            self.refused(capture, "native core/Python package identity differs")
            self.assert_sources_checked()
            self.lookup.assert_not_called()

    def test_boundary_file_hash_mismatch_is_rejected_before_module_lookup(self):
        capture, _ = self.capture()
        self.boundary_file.write_bytes(b"# changed TEST ONLY boundary\n")
        with self.runtime():
            self.refused(capture, "native core/Python package identity differs")
            self.assert_sources_checked()
            self.lookup.assert_not_called()

    def test_wrong_loaded_boundary_path_is_rejected_after_module_lookup(self):
        capture, _ = self.capture()
        self.public.__file__ = str(self.root/"wrong-boundary.py")
        with self.runtime():
            self.refused(capture, "strict public LW seam identity/default differs")
            self.assert_sources_checked()
            self.lookup.assert_called_once_with("psi4.driver.procrouting.isapol_lw")

    def test_changed_boundary_hash_during_lookup_is_rechecked(self):
        capture, _ = self.capture()
        def lookup_then_change(name):
            module = self.import_reply(name)
            self.boundary_file.write_bytes(b"# changed during TEST ONLY lookup\n")
            return module
        self.lookup.side_effect = lookup_then_change
        with self.runtime():
            self.refused(capture, "strict public LW seam identity/default differs")
            self.assert_sources_checked()
            self.lookup.assert_called_once_with("psi4.driver.procrouting.isapol_lw")

    def test_changed_strict_tolerance_is_rejected_after_boundary_hash_checks(self):
        capture, _ = self.capture()
        self.public.PRODUCTION_TOLERANCE = 2e-6
        with self.runtime():
            self.refused(capture, "strict public LW seam identity/default differs")
            self.assert_sources_checked()
            self.lookup.assert_called_once_with("psi4.driver.procrouting.isapol_lw")

    def test_changed_truncation_is_rejected_after_strict_tolerance_check(self):
        capture, _ = self.capture()
        self.public.TRUNCATE_RANK4 = "TEST-ONLY wrong truncation"
        with self.runtime():
            self.refused(capture, "strict public LW seam identity/default differs")
            self.assert_sources_checked()
            self.lookup.assert_called_once_with("psi4.driver.procrouting.isapol_lw")

    def test_mocked_happy_orchestration_calls_public_strict_boundary_once(self):
        capture, original = self.capture(dense=True)
        before = tuple((name, a.data) for name, a in capture.arrays)
        self.boundary.side_effect = refusal(value="36")  # fake public seam, not native LW
        with self.runtime():
            outcome = LW.native_replay(capture)
            self.assert_sources_checked()
            self.lookup.assert_called_once_with("psi4.driver.procrouting.isapol_lw")
            self.boundary.assert_called_once()
            self.assertEqual(self.boundary.call_args.args, ())
            kw = self.boundary.call_args.kwargs
            self.assertEqual(set(kw), {"labels","origins","frames","bonds","frequencies",
                "tensors","input_rank","truncation","residual_policy","localization_rank_limit","provenance"})
            for name in ("tensors","origins","frames"):
                view = kw[name]
                self.assertEqual(view.shape, original[name].shape)
                self.assertEqual(view.buffer.tobytes(), original[name].data)
                self.assertTrue(view.buffer.readonly)
                self.assertFalse(view.flags.writeable)
                with self.assertRaises(TypeError):
                    view.buffer[0] = 0
            self.assertEqual(self.numpy.frombuffer.call_args_list,
                [mock.call(original[name].data, dtype="<f8") for name in ("tensors","origins","frames")])
            self.assertEqual(kw["tensors"].shape, (1,3,3,25,25))
            self.assertEqual(kw["labels"], ("S0","S1","S2"))
            self.assertEqual(kw["bonds"], ((2,0),(1,2)))
            self.assertEqual(kw["frequencies"], (.375,))
            self.assertEqual((kw["input_rank"],kw["localization_rank_limit"]), (4,2))
            self.assertEqual(kw["truncation"], "discard_rank4_rows_and_columns")
            self.assertEqual(kw["residual_policy"], "production")
            self.public.Provenance.assert_called_once()
            self.assertEqual(kw["provenance"].args[1], capture.manifest_sha256)
            self.assertEqual(outcome["status"], "EXPECTED_INPUT_SUM_RULE_REFUSAL")
            self.assertTrue(outcome["diagnostic_replay_passed"])
            self.assertFalse(outcome["scientific_properties_accepted"])
            self.assertFalse(outcome["binary_source_attestation"])
        self.assertEqual(before, tuple((name, a.data) for name, a in capture.arrays))


if __name__ == "__main__":
    unittest.main()
