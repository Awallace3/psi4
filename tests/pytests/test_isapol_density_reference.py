# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""Pure offline tests of the opt-in diagnostic at its actual checkout seam."""
from dataclasses import FrozenInstanceError
from decimal import Decimal, localcontext
from fractions import Fraction
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


SOURCE = Path(__file__).resolve().parents[2] / "psi4/driver/procrouting/isapol_density_reference.py"
spec = importlib.util.spec_from_file_location("_isapol_density_reference_source", SOURCE)
reference = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = reference
spec.loader.exec_module(reference)
assert Path(reference.__file__).resolve() == SOURCE
print(f"TESTED_MODULE={reference.__file__}; PSI4_IMPORTED={'psi4' in sys.modules}", flush=True)


class FiniteDensityReferenceTests(unittest.TestCase):
    def test_finite_target_is_not_zero_target_or_hard_charge(self):
        args = (np.array([[2.]]), np.array([1.]), np.array([1.]))
        finite = reference.finite_density_reference(
            *args, electron_count=10., charge_penalty=1000.)
        zero = reference.finite_density_reference(
            *args, electron_count=0., charge_penalty=1000.)
        self.assertEqual(finite.coefficients[0], float(Fraction(10001, 1002)))
        self.assertEqual(zero.coefficients[0], float(Fraction(1, 1002)))
        self.assertNotEqual(finite.reference_charge_error, Decimal(0))
        with localcontext() as context:
            context.prec = 80
            expected_error = Decimal.from_float(finite.coefficients[0])-Decimal(10)
        self.assertEqual(finite.charge_error, expected_error)
        self.assertEqual(finite.decimal_digits, 80)
        self.assertEqual((finite.factorizations, finite.right_hand_sides), (1, 2))

    def test_zero_penalty_or_zero_charge_does_not_repair_density(self):
        J, b = np.diag([2., 4.]), np.array([1., 3.])
        for q, penalty in [(np.ones(2), 0.), (np.zeros(2), 1000.)]:
            with self.subTest(q=q, penalty=penalty):
                result = reference.finite_density_reference(
                    J, b, q, electron_count=10., charge_penalty=penalty)
                np.testing.assert_array_equal(result.coefficients, [.5, .75])
                self.assertEqual(result.denominator, Decimal(1))
                self.assertEqual(result.reference_scaled_penalized_residual, 0)

    def test_zero_and_weak_coefficients_are_not_cut_off_or_normalized(self):
        # Water has no exact-zero coefficients. This independent diagonal
        # case covers exact zeros and a tiny nonzero without a fitted floor.
        expected = np.array([0., 1.e-100, 0.])
        result = reference.finite_density_reference(
            np.eye(3), expected.copy(), np.array([0., 0., 1.]),
            electron_count=0., charge_penalty=1000.)
        np.testing.assert_array_equal(result.coefficients.view(np.uint64), expected.view(np.uint64))
        self.assertEqual(result.charge_error, 0)

    def test_nonsymmetric_late_pivot_swap_and_two_rhs(self):
        # J*[1,2,3]=[8,4,13]; J*[1,-1,2]=[4,3,0].
        # Pivot1 swaps rows AFTER storing earlier multipliers. qTc0=10, qTu=1.
        J = np.array([[3., 1., 1.], [1., 0., 1.], [2., 4., 1.]])
        result = reference.finite_density_reference(
            J, np.array([8., 4., 13.]), np.array([4., 3., 0.]),
            electron_count=12., charge_penalty=1000.)
        expected = [float(Fraction(v, 1001)) for v in (3001, 2, 7003)]
        np.testing.assert_array_equal(result.coefficients, expected)
        self.assertEqual(result.denominator, 1001)
        self.assertLess(result.reference_scaled_penalized_residual, Decimal("1e-70"))

    def test_input_and_returned_data_are_immutable_and_fingerprinted(self):
        J, b, q = np.array([[2., .5], [0., 3.]]), np.array([1., 2.]), np.array([1., .25])
        saved = [x.copy() for x in (J, b, q)]
        result = reference.finite_density_reference(J, b, q, electron_count=10., charge_penalty=1000.)
        for actual, old in zip((J, b, q), saved):
            np.testing.assert_array_equal(actual, old)
            self.assertTrue(actual.flags.writeable)
        with self.assertRaises(ValueError):
            result.coefficients[0] = 0.
        with self.assertRaises(ValueError):
            result.coefficients.setflags(write=True)
        with self.assertRaises(FrozenInstanceError):
            result.denominator = Decimal(1)
        self.assertRegex(result.input_sha256, r"^[0-9a-f]{64}$")
        same = reference.finite_density_reference(*saved, electron_count=10., charge_penalty=1000.)
        self.assertEqual(result.input_sha256, same.input_sha256)
        changed = reference.finite_density_reference(*saved, electron_count=0., charge_penalty=1000.)
        self.assertNotEqual(result.input_sha256, changed.input_sha256)

    def test_private_precision_ignores_and_preserves_caller_context(self):
        with localcontext() as context:
            context.prec = 2
            result = reference.finite_density_reference(
                np.array([[3.]]), np.array([1.]), np.array([0.]),
                electron_count=0., charge_penalty=0.)
            self.assertEqual(context.prec, 2)
        self.assertEqual(result.coefficients[0], float(Fraction(1, 3)))
        self.assertEqual(result.decimal_digits, 80)

    def test_residual_diagnostics_describe_returned_rounded_coefficients(self):
        J, b, q = np.array([[2., .5], [0., 3.]]), np.array([1., 2.]), np.array([1., .25])
        result = reference.finite_density_reference(J, b, q, electron_count=10., charge_penalty=4.)
        # Independent tiny explicit intended matrix; dyadic operands make these
        # rounded-vector products exact within110 digits. Module forms no A.
        with localcontext() as context:
            context.prec = 110
            matrix = [[Decimal.from_float(float(v)) for v in row] for row in J]
            rhs, charges, c = ([Decimal.from_float(float(v)) for v in row]
                               for row in (b, q, result.coefficients))
            A = [[v+4*charges[i]*charges[j] for j, v in enumerate(row)]
                 for i, row in enumerate(matrix)]
            target = [v+40*qi for v, qi in zip(rhs, charges)]
            original = max(abs(sum(x*y for x, y in zip(row, c))-v)
                           for row, v in zip(matrix, rhs))
            intended = max(abs(sum(x*y for x, y in zip(row, c))-v)
                           for row, v in zip(A, target))
            charge = sum(x*y for x, y in zip(charges, c))-10
        self.assertEqual(result.original_residual_inf, original)
        self.assertEqual(result.penalized_residual_inf, intended)
        self.assertEqual(result.charge_error, charge)

    def test_invalid_shape_dtype_nonfinite_and_scalar_inputs(self):
        valid = dict(metric=np.eye(2), density_rhs=np.ones(2), charges=np.ones(2),
                     electron_count=10., charge_penalty=1000.)
        for name, bad in (
                ("metric", [[1., 0.], [0., 1.]]), ("metric", np.empty((0, 0))),
                ("metric", np.ones((2, 3))), ("metric", np.eye(2, dtype=np.float32)),
                ("metric", np.eye(2)*np.nan), ("density_rhs", np.ones((2, 1))),
                ("density_rhs", np.ones(3)), ("density_rhs", np.array([np.inf, 1.])),
                ("charges", [1., 1.]), ("charges", np.ones(2, dtype=complex)),
                ("charges", np.array([np.nan, 1.])), ("electron_count", True),
                ("electron_count", -1.), ("electron_count", np.nan),
                ("electron_count", 10**400), ("electron_count", np.array([10.])),
                ("charge_penalty", False), ("charge_penalty", -1.),
                ("charge_penalty", np.inf), ("charge_penalty", 1.+1.j)):
            with self.subTest(name=name, bad=bad), self.assertRaises(ValueError):
                reference.finite_density_reference(**(valid | {name: bad}))

    def test_resource_rejection_precedes_finiteness_scan_or_decimal_solve(self):
        # Zero-stride view, no large buffer. If scanned first this is a
        # nonfinite-input ValueError; profile admission must instead reject.
        J = np.broadcast_to(np.array(np.nan), (257, 257))
        with self.assertRaisesRegex(MemoryError, "profile"):
            reference.finite_density_reference(
                J, np.ones(257), np.ones(257), electron_count=10., charge_penalty=1000.)

    def test_maximum_dimension_is_inside_documented_profile(self):
        result = reference.finite_density_reference(
            np.eye(256), np.ones(256), np.zeros(256), electron_count=0., charge_penalty=0.)
        np.testing.assert_array_equal(result.coefficients, np.ones(256))
        self.assertEqual(result.planned_buffer_bytes, 142606336)
        self.assertEqual(result.estimated_scalar_operations, 38748736)
        self.assertLess(result.planned_buffer_bytes, 256*1024**2)
        self.assertLess(result.estimated_scalar_operations, 500000000)

    def test_singular_J_refused_even_if_penalized_system_is_invertible(self):
        with self.assertRaisesRegex(np.linalg.LinAlgError, "nonsingular J"):
            reference.finite_density_reference(
                np.zeros((1, 1)), np.ones(1), np.ones(1),
                electron_count=10., charge_penalty=1.)
        with self.assertRaisesRegex(np.linalg.LinAlgError, "zero Decimal LU pivot"):
            reference.finite_density_reference(
                np.array([[1., 2.], [2., 4.]]), np.ones(2), np.ones(2),
                electron_count=10., charge_penalty=0.)

    def test_singular_and_near_cancelling_denominators_refused(self):
        with self.assertRaisesRegex(np.linalg.LinAlgError, "singular finite-penalty denominator"):
            reference.finite_density_reference(
                np.array([[-1.]]), np.ones(1), np.ones(1),
                electron_count=0., charge_penalty=1.)
        # qTu=-1+(exact-binary1e-30)^2. The denominator is NONZERO but
        # loses about60 decimal digits; fixed conservative guard must refuse.
        with self.assertRaisesRegex(np.linalg.LinAlgError, "near-cancelling"):
            reference.finite_density_reference(
                np.diag([-1., 1.]), np.array([1., 0.]), np.array([1., 1.e-30]),
                electron_count=0., charge_penalty=1.)

    def test_negative_noncancelling_denominator_is_not_silently_repaired(self):
        result = reference.finite_density_reference(
            np.array([[-1.]]), np.array([2.]), np.ones(1),
            electron_count=10., charge_penalty=2.)
        self.assertEqual(result.denominator, -1)
        np.testing.assert_array_equal(result.coefficients, [22.])

    def test_binary64_coefficient_overflow_is_refused_not_clipped(self):
        with self.assertRaisesRegex(np.linalg.LinAlgError, "coefficient overflow"):
            reference.finite_density_reference(
                np.array([[np.nextafter(0., 1.)]]), np.array([np.finfo(float).max]),
                np.zeros(1), electron_count=0., charge_penalty=0.)

    def test_accepted_direct_builder_bytes_and_no_quantum_imports(self):
        import hashlib
        # Published PR27 source, deliberately NOT cleanup's comparison helper.
        # The diagnostic port must leave this production module untouched.
        self.assertEqual(hashlib.sha256(SOURCE.with_name("isapol_native_factors.py").read_bytes()).hexdigest(),
                         "7d0187360751024d6774307d486beec7ab77c5d2c45f7bffa3fcf30b5f3bbc43")
        self.assertFalse(any(name.split(".")[0] in ("psi4", "adcc", "pylibxc") for name in sys.modules))


class NativeWaterReferenceTests(unittest.TestCase):
    def test_all_246_coefficients_match_correctly_rounded_native_reference(self):
        import hashlib
        import json
        fixture = Path(__file__).resolve().parent / "data_isapol/native_density_reference"
        pins = json.loads((fixture / "water.json").read_text())
        path = fixture / "water.npz"
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), pins["fixture_sha256"])
        with np.load(path, allow_pickle=False) as data:
            self.assertEqual(set(data.files), {"J", "b", "q", "reference"})
            arrays = {name: data[name] for name in data.files}
        for name, array in arrays.items():
            self.assertEqual(array.dtype, np.dtype("<f8"))
            self.assertTrue(np.isfinite(array).all())
            self.assertEqual(hashlib.sha256(array.tobytes()).hexdigest(), pins["payload_sha256"][name])
        arrays = {name: array.astype(np.float64, copy=False) for name, array in arrays.items()}
        self.assertEqual(arrays["J"].shape, (246, 246))
        self.assertEqual(len(pins["intended_reference_110_decimal"]), 246)
        # Exact Decimal -> binary64 conversion, not allclose or residual gating.
        rounded = np.array([float(Decimal(v)) for v in pins["intended_reference_110_decimal"]])
        np.testing.assert_array_equal(arrays["reference"].view(np.uint64), rounded.view(np.uint64))
        result = reference.finite_density_reference(
            arrays["J"], arrays["b"], arrays["q"],
            electron_count=pins["electron_count"], charge_penalty=pins["charge_penalty"])
        self.assertEqual(result.coefficients.shape, (246,))
        np.testing.assert_array_equal(result.coefficients.view(np.uint64), rounded.view(np.uint64))
        self.assertEqual(result.input_sha256, pins["diagnostic_input_sha256"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
