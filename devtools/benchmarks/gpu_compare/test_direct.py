import contextlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from compare import Timings, psi_options, pyscf_timing_policy
from direct import audit_direct_contractions, jk_mode, make_rks
from tz_control import strategy_modes


class Array:
    pass


class DirectTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((Path(__file__).parent / "config.json").read_text())
        self.config["jk_mode"] = "conventional-direct"
        self.mf = SimpleNamespace(with_df=None, _eri=None)
        self.cp = SimpleNamespace(ndarray=Array)
        self.mol = SimpleNamespace(omega=0.)
        self.dm = Array()
        def get_j(mol, dm, hermi=1, vhfopt=None):
            return Array()
        def get_k(mol, dm, hermi=1, vhfopt=None, omega=None, lr_factor=None, sr_factor=None):
            return Array()
        self.j = SimpleNamespace(get_j=get_j)
        self.k = SimpleNamespace(get_k=get_k)
        self.forbidden = SimpleNamespace(build=lambda: None)
        self.record = dict(functional_metadata=dict(
            omega=0., short_range_exchange=0., long_range_exchange=0.))

    def policy(self):
        return audit_direct_contractions(self.mf, self.record, self.cp, self.j, self.k,
                                         [(self.forbidden, "build")])

    def test_mode_does_not_relabel_integral_direct_df(self):
        self.assertEqual(jk_mode({}), "density-fitted")
        self.assertEqual(jk_mode(dict(df_mode="native-strategy")), "density-fitted")
        for c in (dict(jk_mode="direct"), dict(self.config, df_mode="stored-df")):
            with self.assertRaises(ValueError):
                jk_mode(c)
        self.assertEqual(strategy_modes(self.config), [None])
        with self.assertRaises(ValueError):
            strategy_modes(self.config, ["stored-df"])

    def test_psi4_direct_options_need_no_auxiliary_basis(self):
        self.config.pop("auxbasis")
        self.config.pop("df_fitting_condition")
        options = psi_options(self.config)
        self.assertEqual(options["scf_type"], "direct")
        self.assertFalse(options["df_scf_guess"])
        self.assertTrue(options["save_jk"])
        self.assertEqual(options["guess"], "core")
        self.assertEqual(options["cuest_direct_jit"], "off")
        self.assertNotIn("df_basis_scf", options)
        self.assertNotIn("df_fitting_condition", options)

    def test_pyscf_direct_construction_is_undecorated(self):
        class RKS:
            def __init__(self, mol, xc):
                self.xc = xc
            def density_fit(self, **kwargs):
                raise AssertionError("Direct worker attempted density fitting")
        self.assertEqual(make_rks(SimpleNamespace(RKS=RKS), self.mol, "PBE", self.config).xc, "PBE")

    def test_pure_j_is_verified_and_restored(self):
        original = self.j.get_j
        forbidden = self.forbidden.build
        with self.policy():
            self.j.get_j(self.mol, self.dm)
        self.assertTrue(self.record["direct_verified"])
        self.assertEqual(self.record["direct_route_calls"]["j_completed"], 1)
        self.assertIs(self.j.get_j, original)
        self.assertIs(self.forbidden.build, forbidden)

    def test_hybrid_requires_consumed_k(self):
        self.record["functional_metadata"]["short_range_exchange"] = .2
        with self.assertRaisesRegex(RuntimeError, "not verified"):
            with self.policy():
                self.j.get_j(self.mol, self.dm)
        with self.policy():
            self.j.get_j(self.mol, self.dm)
            self.k.get_k(self.mol, self.dm)
        self.assertTrue(self.record["direct_verified"])

    def test_rsh_requires_consumed_nonzero_omega(self):
        self.record["functional_metadata"].update(omega=.3, long_range_exchange=1.)
        with self.assertRaisesRegex(RuntimeError, "not verified"):
            with self.policy():
                self.j.get_j(self.mol, self.dm)
                self.k.get_k(self.mol, self.dm, omega=0.)
        with self.policy():
            self.j.get_j(self.mol, self.dm)
            self.k.get_k(self.mol, self.dm, 1, None, .3, 1., .2)
        self.assertEqual(self.record["direct_route_calls"]["rsh_k_completed"], 1)
        self.assertEqual(self.record["direct_requests"][-1]["omega"], .3)

    def test_fallback_is_fatal_even_when_worker_catches_it(self):
        with self.assertRaisesRegex(RuntimeError, "not verified"):
            with self.policy():
                try:
                    self.forbidden.build()
                except RuntimeError:
                    pass
                self.j.get_j(self.mol, self.dm)
        self.assertEqual(self.record["direct_route_calls"]["forbidden_attempts"], 1)

    def test_failed_entry_is_not_completed_and_hooks_restore(self):
        def broken(mol, dm):
            raise ValueError("kernel failed")
        self.j.get_j = broken
        with self.assertRaises(ValueError):
            with self.policy():
                self.j.get_j(self.mol, self.dm)
        self.assertEqual(self.record["direct_route_calls"]["j_attempted"], 1)
        self.assertEqual(self.record["direct_route_calls"]["j_completed"], 0)
        self.assertFalse(self.record["direct_verified"])
        self.assertIs(self.j.get_j, broken)

    def test_missing_calls_cpu_density_and_existing_df_rejected(self):
        with self.assertRaises(RuntimeError):
            with self.policy():
                pass
        with self.assertRaisesRegex(RuntimeError, "GPU density"):
            with self.policy():
                self.j.get_j(self.mol, object())
        self.mf.with_df = object()
        with self.assertRaisesRegex(RuntimeError, "forbids DF"):
            with self.policy():
                pass

    def test_no_stored_eri_switch_allowed(self):
        with self.assertRaisesRegex(RuntimeError, "stored ERIs"):
            with self.policy():
                self.j.get_j(self.mol, self.dm)
                self.mf._eri = object()

    def test_kernel_calls_are_required_when_enabled(self):
        library = SimpleNamespace(kernel=lambda: 0)
        original = library.kernel
        kernels = [(library, "kernel", "j")]
        with self.assertRaisesRegex(RuntimeError, "not verified"):
            with audit_direct_contractions(self.mf, self.record, self.cp,
                                           self.j, self.k, [], kernels):
                self.j.get_j(self.mol, self.dm)
        with audit_direct_contractions(self.mf, self.record, self.cp,
                                       self.j, self.k, [], kernels):
            self.j.get_j(self.mol, self.dm)
            library.kernel()
        self.assertEqual(self.record["direct_route_calls"]["gpu_j_kernel_calls"], 1)
        self.assertIs(library.kernel, original)

    def test_profile_validates_bound_dispatch_and_restores_after_failure(self):
        outer = self
        class RKS:
            def get_j(self, mol, dm):
                return outer.j.get_j(mol, dm)
            get_k = get_j
            get_jk = get_j
            eig = get_j
        mf = RKS()
        mf.with_df = mf._eri = None
        mf._numint = SimpleNamespace(nr_rks=lambda: None)
        mf.grids = SimpleNamespace(build=lambda: None)
        original_nr_rks = mf._numint.nr_rks
        @contextlib.contextmanager
        def policy():
            self.assertIs(mf.get_j.__func__, RKS.get_j)
            with audit_direct_contractions(mf, self.record, self.cp,
                                           self.j, self.k, []):
                yield
        for fail in (False, True):
            timers = Timings(lambda: None)
            with self.subTest(fail=fail):
                try:
                    with pyscf_timing_policy(policy(), mf, timers, True, True):
                        mf.get_j(self.mol, self.dm)
                        if fail:
                            raise ValueError("SCF failed")
                except ValueError:
                    self.assertTrue(fail)
                self.assertIs(mf.get_j.__func__, RKS.get_j)
                self.assertNotIn("get_j", mf.__dict__)
                self.assertIs(mf._numint.nr_rks, original_nr_rks)
                self.assertEqual(timers.data["j_inclusive"]["calls"], 1)
                self.assertEqual(self.record["direct_verified"], not fail)


if __name__ == "__main__":
    unittest.main()
