"""Native-find Python seam and source-wiring contracts; no native qualification."""
import ast
import hashlib
import importlib
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from test_fsapt_cpks_reference import PACKAGE, SAPT, Matrix

find = importlib.import_module(PACKAGE + ".sapt_fisapt_cpks_find")
ROOT = SAPT.parents[3]


def fixture():
    options = {
        ("FISAPT", "FISAPT_FSAPT_IND_XC_RESPONSE"): True,
        ("FISAPT", "FISAPT_FSAPT_IND_RESPONSE"): True,
        ("FISAPT", "FISAPT_LINK_ASSIGNMENT"): "C",
        ("FISAPT", "SSAPT0_SCALE"): False,
        ("FISAPT", "FISAPT_EXCH_PARPERP"): False,
        ("SAPT", "SAPT_DFT_DO_FSAPT"): "FISAPT",
        ("SAPT", "SAPT_DFT_INDUCTION_TYPE"): "CPKS",
        ("SAPT", "SAPT_DFT_FUNCTIONAL"): "PBE",
        ("SAPT", "SAPT_DFT_DO_DDFT"): False,
        ("SAPT", "SAPT_DFT_GRAC_COMPUTE"): "NONE",
        ("SAPT", "SAPT_DFT_GRAC_SHIFT_A"): 0.,
        ("SAPT", "SAPT_DFT_GRAC_SHIFT_B"): 0.,
        ("SAPT", "DO_IND_EXCH_SINF"): False,
        ("SAPT", "CPHF_R_CONVERGENCE"): 1e-10,
        ("SAPT", "MAXITER"): 128, ("SCF", "LEVEL_SHIFT"): 0.,
        ("SCF", "PCM"): False, ("SCF", "PE"): False, ("SCF", "DDX"): False,
    }
    core = SimpleNamespace(
        Matrix=Matrix, get_option=lambda mod, name: options[mod, name],
        get_global_option=lambda name: False, get_global_option_list=lambda: ["BRIANQC_ENABLE"],
        get_memory=lambda: 256*1024**2)
    def get_option(module, name):
        core.current_module = module
        return options[module, name]
    core.get_option = get_option
    core.prepare_options_for_module = lambda module: setattr(core, "current_module", module)
    functional = SimpleNamespace(
        name=lambda: "PBE", is_x_lrc=lambda: False, is_c_lrc=lambda: False,
        is_c_hybrid=lambda: False, is_c_scs_hybrid=lambda: False, is_meta=lambda: False,
        needs_vv10=lambda: False, needs_grac=lambda: False, grac_shift=lambda: 0.,
        is_x_hybrid=lambda: False, needs_xc=lambda: True,
        x_alpha=lambda: 0., x_beta=lambda: 0.)
    cache, monomers = {}, {}
    for side, occupied in (("A", 1), ("B", 2)):
        co, cv = np.eye(3)[:, :occupied], np.eye(3)[:, occupied:]
        energy = np.array([-.8, -.2, .5])
        eo, ev = energy[:occupied], energy[occupied:]
        gap = ev[None, :] - eo[:, None]
        hessian = np.diag(gap.ravel()) + .03*np.ones((gap.size, gap.size))
        jk = object()
        wfn = SimpleNamespace(
            nirrep=lambda: 1, nalpha=lambda n=occupied: n, nbeta=lambda n=occupied: n,
            same_a_b_orbs=lambda: True, same_a_b_dens=lambda: True,
            functional=lambda: functional,
            Ca_subset=lambda basis, space, o=co, v=cv: Matrix(o if space == "OCC" else v),
            epsilon_a_subset=lambda basis, space, o=eo, v=ev: Matrix(o if space == "OCC" else v),
            Fa=lambda e=energy: Matrix(np.diag(e)),
            has_external_cpscf_terms=lambda: False,
            jk=lambda j=jk: j, set_jk=lambda j: None,
            cphf_Hx=lambda xs, h=hessian, shape=gap.shape: [
                Matrix(-(h @ x.np.ravel()).reshape(shape)) for x in xs])
        wfn.hessian = hessian
        monomers[side] = wfn
        for key, value in (("Cocc_", co), ("Cvir_", cv), ("eps_occ_", eo), ("eps_vir_", ev)):
            cache[key + side] = Matrix(value)
        factor = 1 if side == "A" else 2
        cache["V_" + side] = Matrix(factor*np.array([[.1, .02, -.03], [.02, .2, .04], [-.03, .04, .3]]))
        cache["J_" + side] = Matrix(factor*np.array([[.2, -.01, .01], [-.01, .1, -.02], [.01, -.02, .2]]))
    dimer = SimpleNamespace(has_potential_variable=lambda side: False,
                            molecule=lambda: SimpleNamespace(
                                natom=lambda: 2, get_fragments=lambda: [(0, 1), (1, 2)]))
    return core, options, dimer, monomers, cache


def deliver(driver, *, omit=None, offset=0.):
    positive = {}
    for side, state in driver.state.items():
        weights = [.3, -.2, .9] + [0.]*(state["expected_sources"]-3)
        positive[side] = []
        for index, weight in enumerate(weights):
            if (side, index) == omit:
                continue
            rhs = weight*state["total"] + offset
            # Match DFHelper's singleton tensor axis on some incoming matrices.
            matrix = Matrix(rhs[None] if index == 1 else rhs)
            positive[side].append(driver.solve(side, index, matrix).np)
    return positive


class FindCallbacks(unittest.TestCase):
    def test_original_live_references_distinct_JK_sign_zero_and_singleton_source_axis(self):
        core, _, dimer, monomers, cache = fixture()
        driver = find.FindCPKS(core, dimer, monomers, cache)
        positive = deliver(driver)
        for side, state in driver.state.items():
            self.assertIs(state["operator"].wfn, monomers[side])
            self.assertIs(state["operator"].jk, monomers[side].jk())
            expected = np.linalg.solve(monomers[side].hessian, state["total"].ravel()).reshape(state["total"].shape)
            np.testing.assert_allclose(sum(positive[side]), expected, atol=1e-12)
            np.testing.assert_array_equal(positive[side][-1], np.zeros_like(expected))
        metrics = driver.finish()
        self.assertEqual(metrics["CPKS SOURCES (A<-B)"], 5.)
        self.assertEqual(metrics["CPKS SOURCES (B<-A)"], 4.)
        self.assertLess(metrics["CPKS MAX TRUE RESIDUAL (A<-B)"], 1e-12)
        with self.assertRaisesRegex(ValueError, "single-use"):
            driver.finish()
        with self.assertRaises(ValueError):
            driver.solve("A", 0, Matrix(driver.state["A"]["total"]))

    def test_missing_duplicate_wrong_order_and_wrong_total_fail_before_export(self):
        for side in ("A", "B"):
            core, _, dimer, monomers, cache = fixture()
            driver = find.FindCPKS(core, dimer, monomers, cache)
            with self.assertRaisesRegex(ValueError, "slot order"):
                driver.solve(side, 1, Matrix(driver.state[side]["total"]))
        core, _, dimer, monomers, cache = fixture()
        driver = find.FindCPKS(core, dimer, monomers, cache)
        deliver(driver, omit=("B", 3))
        with self.assertRaisesRegex(RuntimeError, "Incomplete"):
            driver.finish()
        driver = find.FindCPKS(core, dimer, monomers, cache)
        deliver(driver, offset=.001)
        with self.assertRaisesRegex(RuntimeError, "completeness"):
            driver.finish()

    def test_same_shape_different_canonical_reference_is_rejected(self):
        core, _, dimer, monomers, cache = fixture()
        cache["Cocc_B"].np *= -1
        with self.assertRaisesRegex(RuntimeError, "canonical reference"):
            find.FindCPKS(core, dimer, monomers, cache)

    def test_zero_rhs_still_gets_an_independent_physical_residual_check(self):
        core, _, dimer, monomers, cache = fixture()
        driver = find.FindCPKS(core, dimer, monomers, cache)
        shape = driver.state["A"]["operator"].gap.shape
        driver.state["A"]["operator"].apply = lambda trial: np.ones(shape)
        with self.assertRaisesRegex(RuntimeError, "physical true residual"):
            driver.solve("A", 0, Matrix(np.zeros(shape)))
        self.assertEqual(driver.state["A"]["sources"], 0)

    def test_memory_shift_external_and_bad_convergence_fail_closed(self):
        core, options, dimer, monomers, cache = fixture()
        core.get_memory = lambda: 1
        with self.assertRaises(MemoryError):
            find.FindCPKS(core, dimer, monomers, cache)
        core.get_memory = lambda: 256*1024**2
        for key, value in ((("SCF", "LEVEL_SHIFT"), .1), (("SAPT", "CPHF_R_CONVERGENCE"), 1e-6)):
            previous = options[key]
            options[key] = value
            with self.assertRaises((ValueError, NotImplementedError)):
                find.FindCPKS(core, dimer, monomers, cache)
            options[key] = previous
        dimer.has_potential_variable = lambda side: side == "A"
        with self.assertRaises(NotImplementedError):
            find.FindCPKS(core, dimer, monomers, cache)

    def test_quadratic_validation_memory_is_admitted_before_operator_construction(self):
        core, _, dimer, monomers, cache = fixture()
        # Views allocate no giant Fock/orbital matrix; old 20*ov accounting would pass.
        cache["Cocc_A"] = SimpleNamespace(np=np.broadcast_to(np.zeros(1), (1001, 1)))
        cache["Cvir_A"] = SimpleNamespace(np=np.broadcast_to(np.zeros(1), (1001, 1000)))
        with self.assertRaisesRegex(MemoryError, "bounded CPU response-array"):
            find.FindCPKS(core, dimer, monomers, cache)

    def test_effective_SCF_embedding_options_override_disabled_globals(self):
        core, options, _, _, _ = fixture()
        core.get_global_option_list = lambda: ["PCM", "PE", "DDX"]
        for name in ("PCM", "PE", "DDX"):
            options["SCF", name] = True
            with self.assertRaisesRegex(NotImplementedError, "External response"):
                find.validate_request(core)
            options["SCF", name] = False

    def test_XC_off_registers_nothing_and_true_flag_requires_matching_binding(self):
        core, options, dimer, monomers, cache = fixture()
        options["FISAPT", "FISAPT_FSAPT_IND_XC_RESPONSE"] = False
        self.assertIsNone(find.install(core, object(), dimer, monomers["A"], monomers["B"], cache))
        options["FISAPT", "FISAPT_FSAPT_IND_XC_RESPONSE"] = True
        with self.assertRaisesRegex(RuntimeError, "matching native build"):
            find.install(core, object(), dimer, monomers["A"], monomers["B"], cache)
        self.assertEqual(core.current_module, "FISAPT")
        recorded = []
        fisapt = SimpleNamespace(set_cpks_response_callbacks=lambda *callbacks: recorded.extend(callbacks))
        driver = find.install(core, fisapt, dimer, monomers["A"], monomers["B"], cache)
        self.assertIs(recorded[0].__self__, driver)
        self.assertIs(recorded[1].__self__, driver)
        self.assertEqual(core.current_module, "FISAPT")

    def test_explicit_backend_and_mode_rejections(self):
        core, options, _, _, _ = fixture()
        for key, value in ((("SAPT", "SAPT_DFT_DO_FSAPT"), "SAPTDFT"),
                           (("SAPT", "SAPT_DFT_INDUCTION_TYPE"), "CPHF"),
                           (("SAPT", "SAPT_DFT_FUNCTIONAL"), "PBE0"),
                           (("SAPT", "SAPT_DFT_DO_DDFT"), True),
                           (("SAPT", "SAPT_DFT_GRAC_SHIFT_A"), .1),
                           (("SAPT", "DO_IND_EXCH_SINF"), True),
                           (("FISAPT", "FISAPT_FSAPT_IND_RESPONSE"), False),
                           (("FISAPT", "SSAPT0_SCALE"), True)):
            previous = options[key]
            options[key] = value
            with self.assertRaises((ValueError, NotImplementedError)):
                find.validate_request(core)
            options[key] = previous
        core.get_global_option = lambda name: name == "BRIANQC_ENABLE"
        with self.assertRaisesRegex(ValueError, "CPU reference"):
            find.validate_request(core)


class NativeWiringSourceContracts(unittest.TestCase):
    def test_actual_setup_hook_passes_original_objects_only_when_opted_in(self):
        core, options, dimer, monomers, cache = fixture()
        tree = ast.parse((SAPT / "saptdft_fisapt.py").read_text())
        function = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef) and node.name == "setup_fisapt_object")
        hook = next(node for node in function.body if isinstance(node, ast.If)
                    and any(isinstance(child, ast.ImportFrom)
                            and any(alias.name == "install" for alias in child.names)
                            for child in ast.walk(node)))
        code = compile(ast.fix_missing_locations(ast.Module(body=[hook], type_ignores=[])), "<setup-hook>", "exec")
        fisapt = object()
        namespace = {"__package__": PACKAGE, "core": core, "fisapt": fisapt, "wfn": dimer,
                     "wfn_A": monomers["A"], "wfn_B": monomers["B"], "cache": cache}
        with patch.object(find, "install") as register:
            options["FISAPT", "FISAPT_FSAPT_IND_XC_RESPONSE"] = False
            exec(code, namespace)
            register.assert_not_called()
            options["FISAPT", "FISAPT_FSAPT_IND_XC_RESPONSE"] = True
            exec(code, namespace)
            for actual, expected in zip(register.call_args.args,
                                        (core, fisapt, dimer, monomers["A"], monomers["B"], cache)):
                self.assertIs(actual, expected)

    def test_native_source_buffers_external_slot_and_zero_rhs_guards(self):
        text = (ROOT / "psi4/src/psi4/fisapt/fisapt.cc").read_text()
        entry = text[text.index("void FISAPT::find()"):text.index("// => Sizing <= //", text.index("void FISAPT::find()"))]
        self.assertLess(entry.index('set_current_module("FISAPT")'), entry.index('get_bool("FISAPT_FSAPT_IND_RESPONSE")'))
        loop = text[text.index("// ==> Master Loop over perturbing atoms"):text.index("// => Induction scaling")]
        self.assertIn('fill_tensor("WBar", wB, {C, C + 1})', loop)
        self.assertIn('fill_tensor("WAbs", wA, {C, C + 1})', loop)
        self.assertNotIn('fill_tensor("WAbs", wB', loop)
        self.assertIn('if (C < nB + nb1 + 1)', loop)
        self.assertIn('if (C < nA + na1 + 1)', loop)
        self.assertLess(loop.index("wB->zero()"), loop.index('fill_tensor("WBar"'))
        self.assertLess(loop.index("wA->zero()"), loop.index('fill_tensor("WAbs"'))
        self.assertIn('if (b2A == 0.0 && b2B == 0.0) return;', text)
        self.assertLess(text.index('cpks_finish_()'), text.index('// => Induction scaling'))
        self.assertIn('matrices_["CPKS_Ind20r_AB_raw"]', text)
        self.assertIn('ind_xc ? Sdelta : sIndHF / IndSAPT0', text)
        self.assertIn('reference_->set_array_variable("FSAPT_" + item.first, item.second)', text)
        self.assertIn('reference_->set_scalar_variable("FSAPT " + item.first, item.second)', text)
        native_find = text[text.index("void FISAPT::find()"):text.index("void FISAPT::fdisp()")]
        self.assertLess(native_find.index("raw_doubles"), native_find.index('add_disk_tensor("WBar"'))

    def test_driver_guard_precedes_reference_work_in_both_entrypoints(self):
        tree = ast.parse((SAPT / "sapt_proc.py").read_text())
        for name in ("_run_sapt_dft", "sapt_dft"):
            function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
            calls = [node for node in ast.walk(function) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
            guard = next(node for node in calls if node.func.id == "validate_request")
            setup = next(node for node in calls if node.func.id in ("run_scf", "sapt_dft_header"))
            self.assertLess(guard.lineno, setup.lineno)

    def test_driver_guard_restores_SAPT_option_context_even_with_XC_off(self):
        core, options, _, _, _ = fixture()
        tree = ast.parse((SAPT / "sapt_proc.py").read_text())
        for enabled in (False, True):
            options["FISAPT", "FISAPT_FSAPT_IND_XC_RESPONSE"] = enabled
            for name in ("_run_sapt_dft", "sapt_dft"):
                function = next(node for node in tree.body
                                if isinstance(node, ast.FunctionDef) and node.name == name)
                index, hook = next((i, node) for i, node in enumerate(function.body)
                                   if isinstance(node, ast.If)
                                   and any(isinstance(child, ast.Constant)
                                           and child.value == "FISAPT_FSAPT_IND_XC_RESPONSE"
                                           for child in ast.walk(node.test)))
                restore = function.body[index+1]
                code = compile(ast.fix_missing_locations(
                    ast.Module(body=[hook, restore], type_ignores=[])), "<driver-option-context>", "exec")
                exec(code, {"__package__": PACKAGE, "core": core, "kwargs": {},
                            "external_potentials": None, "do_delta_dft": False})
                self.assertEqual(core.current_module, "SAPT")

    def test_direct_driver_rejects_deltaDFT_parameter_before_reference_work(self):
        core, _, _, _, _ = fixture()
        tree = ast.parse((SAPT / "sapt_proc.py").read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "sapt_dft")
        hook = next(n for n in function.body if isinstance(n, ast.If)
                    and any(isinstance(c, ast.Constant) and c.value == "FISAPT_FSAPT_IND_XC_RESPONSE"
                            for c in ast.walk(n.test)))
        code = compile(ast.fix_missing_locations(ast.Module(body=[hook], type_ignores=[])), "<guard>", "exec")
        with self.assertRaisesRegex(ValueError, "deltaDFT"):
            exec(code, {"__package__": PACKAGE, "core": core, "do_delta_dft": True,
                        "ValidationError": ValueError, "external_potentials": None})

    def test_default_off_and_legacy_kernel_formula_are_not_reinterpreted(self):
        options = (ROOT / "psi4/src/read_options.cc").read_text()
        self.assertIn('add_bool("FISAPT_FSAPT_IND_XC_RESPONSE", false)', options)
        self.assertIn('add_bool("FISAPT_FSAPT_IND_RESPONSE", false)', options)
        text = (ROOT / "psi4/src/psi4/fisapt/fisapt.cc").read_text()
        product = text[text.index("CPHF_FISAPT::product("):]
        self.assertIn("Jv->scale(4.0)", product)
        self.assertIn("Jv->subtract(Kv)", product)
        self.assertIn("Jv->subtract(Kv->transpose())", product)
        self.assertNotIn("compute_Vx", product)
        # Exact function bytes from the assigned d2a98abe35 base: XC-off
        # must not quietly change response exchange or Hessian algebra.
        body = text[text.index("std::map<std::string, std::shared_ptr<Matrix> > CPHF_FISAPT::product("):
                    text.index("// method for updating/setting matrices_")]
        self.assertEqual(hashlib.sha256(body.encode()).hexdigest(),
                         "4269d4f0c61816b8b534afdce5032045ecb37e324c20a871a6e22105b7af17e8")


if __name__ == "__main__":
    unittest.main()
