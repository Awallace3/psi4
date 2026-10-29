"""Small deterministic CPU contracts. No native Psi4/XC/chemistry proof."""
import importlib
from pathlib import Path
import sys
import types
import unittest
from types import SimpleNamespace

import numpy as np

SAPT = Path(__file__).resolve().parents[2] / "psi4/driver/procrouting/sapt"
PACKAGE = "_cpks_variant_contracts"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(SAPT)]
sys.modules[PACKAGE] = package
cpks = importlib.import_module(PACKAGE + ".sapt_fisapt_cpks")


class Matrix:
    def __init__(self, array):
        self.np = np.array(array, dtype=np.float64, copy=True)

    @staticmethod
    def from_array(array):
        return Matrix(array)


class DenseFixture:
    """Explicit synthetic coupled Hessian, not a simulated XC qualification."""
    mode = cpks.Mode.CPKS
    metadata = {"evidence": "deterministic algebra fixture only"}

    def __init__(self, hessian, shape):
        self.hessian = hessian
        self.gap = np.diag(hessian).reshape(shape).copy()

    def apply(self, trial):
        return (self.hessian @ trial.ravel()).reshape(self.gap.shape)


def fixture(seed=42):
    rng = np.random.default_rng(seed)
    shape = (3, 2)
    a = rng.normal(size=(6, 6))
    hessian = np.diag(np.linspace(2, 4, 6)) + .1 * (a @ a.T)
    sources = rng.normal(size=(5, *shape))
    sources[-1] = 0
    # A 2x2 QR Q is a symmetric reflection and cannot expose U/U.T mistakes.
    rotation = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    assert not np.allclose(rotation, rotation.T)
    target = rotation.T @ sources.sum(axis=0)
    target_x = rng.normal(size=shape)
    exact = -np.linalg.solve(hessian, sources.reshape(5, -1).T).T.reshape(sources.shape)
    local = np.einsum("il,siv->slv", rotation, exact)
    ind = 2 * np.einsum("slv,lv->ls", local, target)
    ex = 2 * np.einsum("slv,lv->ls", local, target_x)
    return DenseFixture(hessian, shape), sources, rotation, target, target_x, exact, ind, ex


def run(operator, sources, rotation, target, target_x, exact, ind, ex, **kwargs):
    options = dict(total_source=sources.sum(axis=0),
                   canonical_induction=float(ind.sum()),
                   canonical_exchange_induction=float(ex.sum()),
                   rtol=1e-12, atol=1e-14)
    options.update(kwargs)
    return cpks.source_resolved(operator, sources, rotation, target, target_x, **options)


class SourceContracts(unittest.TestCase):
    def test_individual_source_pair_cells_sign_gauge_and_true_residual(self):
        for seed in (42, 81):
            data = fixture(seed)
            op, sources, rotation, target, target_x, exact, ind, ex = data
            result = run(*data)
            np.testing.assert_allclose(result.amplitudes, exact, rtol=1e-10, atol=1e-12)
            np.testing.assert_allclose(result.induction, ind, rtol=1e-10, atol=1e-12)
            np.testing.assert_allclose(result.exchange_induction, ex, rtol=1e-10, atol=1e-12)
            wrong_local = np.einsum("li,siv->slv", rotation, exact)
            wrong_cells = 2 * np.einsum("slv,lv->ls", wrong_local, target)
            self.assertFalse(np.allclose(wrong_cells, result.induction))
            for rhs, amplitude, diagnostic in zip(sources, result.amplitudes, result.diagnostics):
                residual = np.linalg.norm(op.apply(amplitude) + rhs)
                self.assertLessEqual(residual, max(1e-14, 1e-12 * np.linalg.norm(rhs)))
                self.assertEqual(residual, diagnostic["absolute_residual"])
            self.assertEqual(result.diagnostics[-1]["iterations"], 0)
            self.assertEqual(result.metadata["response_scale"], 1)
            self.assertFalse(result.metadata["normalized"])
            self.assertLess(result.metadata["source_additivity_max_abs"], 1e-11)

    def test_raw_snapshots_not_mutated_by_export(self):
        result = run(*fixture())
        exported = cpks.export_direction(
            result.induction, result.exchange_induction, mode=result.mode,
            direction="A<-B", delta_hf_scale=1.2)
        np.testing.assert_array_equal(result.induction, exported["raw_induction"])
        self.assertFalse(result.induction.flags.writeable)
        with self.assertRaises(ValueError):
            result.induction[0, 0] = 0
        with self.assertRaises(TypeError):
            result.metadata["normalized"] = True
        self.assertIn("approximation", exported["metadata"]["delta_hf_redistribution"])

    def test_linearity_signed_rescaling_and_source_regrouping(self):
        data = fixture()
        op, sources, rotation, target, target_x, exact, ind, ex = data
        baseline = run(*data)
        # Split a physical source into signed subpieces without changing total RHS.
        split = np.concatenate((sources[:1] * 1.7, sources[:1] * -.7, sources[1:]))
        split_result = cpks.source_resolved(
            op, split, rotation, target, target_x, total_source=sources.sum(axis=0),
            canonical_induction=float(ind.sum()), canonical_exchange_induction=float(ex.sum()))
        for field in ("induction", "exchange_induction"):
            values = getattr(split_result, field)
            np.testing.assert_allclose(values[:, 0] + values[:, 1],
                                       getattr(baseline, field)[:, 0], atol=1e-10)
        np.testing.assert_allclose(split_result.amplitudes[0], 1.7 * exact[0], atol=1e-10)
        np.testing.assert_allclose(split_result.amplitudes[1], -.7 * exact[0], atol=1e-10)

    def test_separate_components_cannot_hide_equal_combined_total(self):
        data = fixture()
        with self.assertRaisesRegex(RuntimeError, "separate canonical"):
            run(*data, canonical_induction=float(data[-2].sum()) + .1,
                canonical_exchange_induction=float(data[-1].sum()) - .1)

    def test_missing_rhs_bad_target_gauge_and_budget_fail_before_solve(self):
        data = fixture()
        for options, error in (({"total_source": data[1].sum(axis=0) + .1}, RuntimeError),
                               ({"max_response_bytes": 1}, MemoryError),
                               ({"check_atol": -1}, ValueError)):
            with self.assertRaises(error):
                run(*data, **options)
        bad = list(data)
        bad[2] = np.eye(data[2].shape[0]) * 2
        with self.assertRaisesRegex(RuntimeError, "gauge"):
            run(*bad)
        bad = list(data)
        bad[3] = bad[3] + .01
        with self.assertRaisesRegex(RuntimeError, "target/gauge"):
            run(*bad)

    def test_nonpositive_curvature_and_nonconvergence(self):
        data = list(fixture())
        data[0] = DenseFixture(-np.eye(6), data[1].shape[1:])
        data[0].gap = np.ones(data[1].shape[1:])
        with self.assertRaisesRegex(RuntimeError, "curvature"):
            run(*data)
        with self.assertRaisesRegex(RuntimeError, "iteration limit"):
            run(*fixture(), max_iterations=1)

    def test_zero_rhs_and_zero_operator_offset(self):
        op = DenseFixture(np.eye(6), (2, 3))
        z = np.zeros((2, 3))
        result = cpks.source_resolved(op, z[None], np.eye(2), z, z, total_source=z,
                                      canonical_induction=0., canonical_exchange_induction=0.)
        self.assertEqual(result.diagnostics[0]["absolute_residual"], 0)
        self.assertEqual(result.aggregate_diagnostic["iterations"], 0)
        op.apply = lambda trial: trial + 1.
        with self.assertRaisesRegex(RuntimeError, "true physical residual"):
            cpks.source_resolved(op, z[None], np.eye(2), z, z, total_source=z,
                                 canonical_induction=0., canonical_exchange_induction=0.)

    def test_cancelling_sources_keep_nonzero_raw_exchange_cells(self):
        op = DenseFixture(np.eye(6), (2, 3))
        source = np.arange(1., 7.).reshape(2, 3)
        sources = np.stack((source, -source))
        z = np.zeros((2, 3))
        result = cpks.source_resolved(
            op, sources, np.eye(2), z, np.ones((2, 3)), total_source=z,
            canonical_induction=0., canonical_exchange_induction=0.)
        self.assertEqual(result.aggregate_diagnostic["iterations"], 0)
        self.assertEqual(float(result.exchange_induction.sum()), 0.)
        self.assertGreater(float(np.max(np.abs(result.exchange_induction))), 1.)
        output = cpks.export_direction(result.induction, result.exchange_induction,
                                      mode=cpks.Mode.CPKS, direction="A<-B")
        np.testing.assert_array_equal(output["combined"], result.exchange_induction)

    def test_true_residual_rejects_changing_operator(self):
        op, *rest = fixture()
        original = op.apply
        def action(x):
            # A malicious operator changes its action on the final physical sign.
            return original(x) if np.vdot(x, rest[0][0]) >= 0 else original(x) + .01
        op.apply = action
        with self.assertRaises(RuntimeError):
            run(op, *rest)


class ScalingContracts(unittest.TestCase):
    def test_equal_normalized_totals_do_not_qualify_partition(self):
        data = fixture()
        result = run(*data)
        op, sources, rotation, target, target_x, *_ = data
        unc = -sources / op.gap
        local = np.einsum("il,siv->slv", rotation, unc)
        ind = 2 * np.einsum("slv,lv->ls", local, target)
        ex = 2 * np.einsum("slv,lv->ls", local, target_x)
        total = float(result.induction.sum() + result.exchange_induction.sum())
        compatibility = cpks.export_direction(ind, ex, mode=cpks.Mode.UNCOUPLED_SCALED,
                                               direction="A<-B", reference="KS", canonical_combined=total)
        genuine = cpks.export_direction(result.induction, result.exchange_induction,
                                       mode=cpks.Mode.CPKS, direction="A<-B")
        difference = compatibility["combined"] - genuine["combined"]
        self.assertAlmostEqual(float(difference.sum()), 0., places=12)
        self.assertGreater(float(np.max(np.abs(difference))), .01)
        self.assertFalse(np.allclose(compatibility["combined"], genuine["combined"]))

    def test_direction_specific_combined_not_component_scaling(self):
        ind = np.array([[-3., -1.]])
        ex = np.array([[1., .5]])
        for direction, total in (("A<-B", -5.), ("B<-A", -1.25)):
            output = cpks.export_direction(ind, ex, mode=cpks.Mode.UNCOUPLED_SCALED,
                                            direction=direction, reference="KS", canonical_combined=total,
                                            delta_hf_scale=1.1)
            self.assertEqual(output["metadata"]["response_scale"], total / -2.5)
            self.assertAlmostEqual(float(output["combined"].sum()), 1.1 * total)
            np.testing.assert_array_equal(output["raw_induction"], ind)

    def test_cphf_to_cpks_total_is_labeled_not_genuine_cpks(self):
        output = cpks.export_direction(np.array([[-1.]]), np.array([[.2]]),
                                       mode=cpks.Mode.CPHF_TO_CPKS_TOTAL,
                                       direction="B<-A", reference="KS", canonical_combined=-1.6)
        self.assertEqual(output["metadata"]["response_scale"], 2)
        self.assertNotEqual(output["metadata"]["mode"], cpks.Mode.CPKS.value)
        self.assertEqual(output["metadata"]["response_kernel"], "HF-like on fixed KS")
        for mode, reference in ((cpks.Mode.CPHF, "KS"), (cpks.Mode.CPKS, "HF"),
                                (cpks.Mode.CPHF_TO_CPKS_TOTAL, None)):
            with self.assertRaises(ValueError):
                cpks.export_direction(np.array([[-1.]]), np.array([[.2]]),
                                      mode=mode, direction="B<-A", reference=reference)

    def test_coupled_modes_reject_even_identity_response_ratio(self):
        for mode in (cpks.Mode.CPHF, cpks.Mode.CPKS):
            with self.assertRaisesRegex(ValueError, "legacy response ratio"):
                cpks.export_direction(np.array([[-1.]]), np.zeros((1, 1)),
                                      mode=mode, direction="A<-B", canonical_combined=-1.)

    def test_cancellation_and_undefined_ratios_fail_closed(self):
        for ind, ex in ((0., 0.), (-1., 1.), (-1., 1. - 1e-13)):
            for desired in (0., -2.):
                with self.assertRaisesRegex(ValueError, "Undefined/cancellation"):
                    cpks.export_direction(np.array([[ind]]), np.array([[ex]]),
                                          mode=cpks.Mode.UNCOUPLED_SCALED,
                                          direction="A<-B", reference="KS", canonical_combined=desired)
        output = cpks.export_direction(np.array([[-1.]]), np.zeros((1, 1)),
                                       mode=cpks.Mode.UNCOUPLED_SCALED,
                                       direction="A<-B", reference="KS", canonical_combined=0.)
        self.assertEqual(output["metadata"]["response_scale"], 0.)

    def test_invalid_scaling_policy(self):
        for options in ({"canonical_combined": None}, {"canonical_combined": float("nan")},
                        {"delta_hf_scale": float("inf")}, {"ratio_atol": -1.},
                        {"direction": "AB"}):
            args = dict(mode=cpks.Mode.UNCOUPLED_SCALED, direction="A<-B",
                        reference="KS", canonical_combined=-2.)
            args.update(options)
            with self.assertRaises(ValueError):
                cpks.export_direction(np.array([[-1.]]), np.zeros((1, 1)), **args)
        with self.assertRaises(FloatingPointError):
            cpks.export_direction(np.full((2, 2), 1e308), np.zeros((2, 2)),
                                  mode=cpks.Mode.CPKS, direction="A<-B")


def native_double(name="PBE", **flags):
    """Only adapter admission/sign contracts, not real wavefunction semantics."""
    functional = SimpleNamespace(
        name=lambda: name, is_x_lrc=lambda: flags.get("lrc", False),
        is_c_lrc=lambda: flags.get("c_lrc", False),
        is_c_hybrid=lambda: flags.get("c_hybrid", False),
        is_c_scs_hybrid=lambda: flags.get("c_scs", False),
        is_meta=lambda: flags.get("meta", False), needs_vv10=lambda: flags.get("vv10", False),
        needs_grac=lambda: flags.get("grac", False), grac_shift=lambda: flags.get("shift", 0.),
        needs_xc=lambda: flags.get("xc", name != "HF"),
        is_x_hybrid=lambda: flags.get("hybrid", name == "HF"),
        x_alpha=lambda: flags.get("alpha", 1. if name == "HF" else 0.),
        x_beta=lambda: flags.get("beta", 0.))
    co, cv = np.eye(3)[:, :1], np.eye(3)[:, 1:]
    assigned = []
    wfn = SimpleNamespace(
        functional=lambda: functional, nirrep=lambda: 1, nalpha=lambda: 1, nbeta=lambda: 1,
        same_a_b_orbs=lambda: True, same_a_b_dens=lambda: True,
        Ca_subset=lambda basis, space: Matrix(co if space == "OCC" else cv),
        epsilon_a_subset=lambda basis, space: Matrix([-.5] if space == "OCC" else [.2, .6]),
        Fa=lambda: Matrix(np.diag([-.5, .2, .6])), set_jk=assigned.append,
        has_external_cpscf_terms=lambda: False,
        cphf_Hx=lambda values: [Matrix(-np.array([[.7, 1.1]]) * values[0].np)])
    core = SimpleNamespace(Matrix=Matrix, get_global_option=lambda key: False,
                           get_global_option_list=lambda: ["BRIANQC_ENABLE"])
    return core, wfn, object(), assigned


class NativeAdapterContracts(unittest.TestCase):
    def make(self, core, wfn, jk, **kwargs):
        args = dict(grac_shift=0., orbital_shift=0., external_response=False)
        args.update(kwargs)
        return cpks.NativeMonomerCPKS(core, wfn, jk, **args)

    def test_sign_and_HF_limit_labels(self):
        for name, mode in (("PBE", cpks.Mode.CPKS), ("SVWN", cpks.Mode.CPKS),
                           ("HF", cpks.Mode.CPHF)):
            core, wfn, jk, assigned = native_double(name)
            op = self.make(core, wfn, jk)
            trial = np.array([[.3, -.9]])
            np.testing.assert_allclose(op.apply(trial), op.gap * trial)
            self.assertEqual(op.mode, mode)
            self.assertEqual(assigned, [jk])
        core, wfn, jk, _ = native_double("HF", beta=1.)
        operator = self.make(core, wfn, jk)
        self.assertEqual(operator.mode, cpks.Mode.CPHF)
        self.assertEqual(operator.metadata["response_exchange_beta_effective"], 0.)

    def test_stored_external_terms_fail_even_with_current_options_disabled(self):
        core, wfn, jk, assigned = native_double()
        wfn.has_external_cpscf_terms = lambda: True
        with self.assertRaisesRegex(NotImplementedError, "Stored external"):
            self.make(core, wfn, jk)
        self.assertEqual(assigned, [])
        del wfn.has_external_cpscf_terms
        with self.assertRaisesRegex(RuntimeError, "matching build"):
            self.make(core, wfn, jk)

    def test_reference_validation_scratch_requires_explicit_admission(self):
        core, wfn, jk, assigned = native_double()
        with self.assertRaisesRegex(MemoryError, "scratch budget"):
            self.make(core, wfn, jk, max_validation_bytes=1)
        self.assertEqual(assigned, [])
        self.assertGreater(cpks._validation_work_bytes(1001, 1, 1000), 64 * 1024**2)

    def test_unsupported_functionals_fail_before_setting_JK(self):
        for name, flags in (("B3LYP", {}), ("PBE", {"alpha": .25}),
                            ("PBE", {"beta": .2}), ("PBE", {"hybrid": True}),
                            ("PBE", {"lrc": True}), ("PBE", {"c_hybrid": True}),
                            ("PBE", {"c_lrc": True}), ("PBE", {"c_scs": True}),
                            ("PBE", {"meta": True}), ("PBE", {"vv10": True}),
                            ("PBE", {"grac": True}), ("PBE", {"shift": .1}),
                            ("HF", {"alpha": .25}), ("HF", {"xc": True}),
                            ("HF", {"beta": float("nan")}), ("HF", {"beta": float("inf")})):
            core, wfn, jk, assigned = native_double(name, **flags)
            with self.assertRaises(NotImplementedError):
                self.make(core, wfn, jk)
            self.assertEqual(assigned, [])

    def test_shifts_external_response_symmetry_open_shell_accelerator_fail(self):
        for kwargs in ({"grac_shift": .1}, {"orbital_shift": .1},
                       {"grac_shift": float("nan")}, {"external_response": True}):
            core, wfn, jk, assigned = native_double()
            with self.assertRaises(NotImplementedError):
                self.make(core, wfn, jk, **kwargs)
            self.assertEqual(assigned, [])
        for method, value in (("nirrep", 2), ("nbeta", 0), ("same_a_b_orbs", False),
                              ("same_a_b_dens", False)):
            core, wfn, jk, assigned = native_double()
            setattr(wfn, method, lambda: value)
            with self.assertRaises(NotImplementedError):
                self.make(core, wfn, jk)
        core, wfn, jk, _ = native_double()
        core.get_global_option = lambda key: True
        with self.assertRaisesRegex(ValueError, "BrianQC"):
            self.make(core, wfn, jk)

    def test_Fock_gap_mismatch_not_normalized_away(self):
        core, wfn, jk, assigned = native_double()
        wfn.Fa = lambda: Matrix(np.diag([-.5, .3, .6]))
        with self.assertRaisesRegex(RuntimeError, "Fock/gap"):
            self.make(core, wfn, jk)
        self.assertEqual(assigned, [])

    def test_bad_action_shape_and_missing_JK(self):
        core, wfn, jk, _ = native_double()
        with self.assertRaises(ValueError):
            self.make(core, wfn, None)
        op = self.make(core, wfn, jk)
        with self.assertRaises(ValueError):
            op.apply(np.ones((1, 2), dtype=np.float32))
        wfn.cphf_Hx = lambda values: []
        with self.assertRaisesRegex(RuntimeError, "vector count"):
            op.apply(np.ones((1, 2)))

    def test_optional_BrianQC_option_absent_and_enabled(self):
        core, wfn, jk, _ = native_double()
        core.get_global_option_list = lambda: []
        def get(name):
            raise AssertionError("Queried an option absent in CPU-only builds")
        core.get_global_option = get
        self.make(core, wfn, jk)
        core.get_global_option_list = lambda: ["BRIANQC_ENABLE"]
        core.get_global_option = lambda key: key == "BRIANQC_ENABLE"
        with self.assertRaisesRegex(ValueError, "BrianQC"):
            self.make(core, wfn, jk)


class SourceBuilderContracts(unittest.TestCase):
    def test_nuclear_sign_electronic_spin_factor_and_fixed_source_gauge(self):
        rng = np.random.default_rng(19)
        b = rng.normal(size=(4, 3, 3))
        b = (b + b.transpose(0, 2, 1)) / 2
        eri = np.einsum("Qmn,Qrs->mnrs", b, b)
        co, cv = np.eye(3)[:, :1], np.eye(3)[:, 1:]
        lo = rng.normal(size=(3, 2))
        left, right = [], []
        nuclear = np.diag([-2., -1., -.3]) + .1
        class Potential:
            def addCharge(self, charge, *xyz):
                self.charge = charge
            def computePotentialMatrix(self, primary):
                return Matrix(self.charge * nuclear)
        jk = SimpleNamespace(
            basisset=lambda: object(), C_clear=lambda: (left.clear(), right.clear()),
            C_left_add=lambda x: left.append(x.np), C_right_add=lambda x: right.append(x.np),
            compute=lambda: None,
            J=lambda: [Matrix(np.einsum("mnrs,rs->mn", eri, left[0] @ right[0].T))])
        core = SimpleNamespace(Matrix=Matrix, ExternalPotential=Potential,
                               get_global_option=lambda key: False,
                               get_global_option_list=lambda: [])
        sources = cpks.build_sources(core, jk, co, cv, lo, [(2., (0., 0., 0.))])
        np.testing.assert_allclose(sources[0], 2 * co.T @ nuclear @ cv)
        for index, orbital in enumerate(lo.T, 1):
            expected = 2 * np.einsum("mnrs,r,s->mn", eri, orbital, orbital)
            np.testing.assert_allclose(sources[index], co.T @ expected @ cv)
        total_j = np.einsum("mnrs,rs->mn", eri, lo @ lo.T)
        np.testing.assert_allclose(sources.sum(axis=0), co.T @ (2*nuclear + 2*total_j) @ cv)
        self.assertEqual(left, [])
        self.assertEqual(right, [])
        core.get_global_option_list = lambda: ["BRIANQC_ENABLE"]
        core.get_global_option = lambda key: key == "BRIANQC_ENABLE"
        with self.assertRaisesRegex(ValueError, "BrianQC"):
            cpks.build_sources(core, jk, co, cv, lo, [])


if __name__ == "__main__":
    unittest.main()
