"""CPU public-API fakes: structural snapshots, NOT native basis conversion."""
from dataclasses import FrozenInstanceError
import importlib
from pathlib import Path
import sys
import types
import unittest

SAPT = Path(__file__).resolve().parents[2]/"psi4/driver/procrouting/sapt"
PACKAGE = "_disp0_factory_basis_fixtures"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(SAPT)]
sys.modules[PACKAGE] = package
basis_api = importlib.import_module(PACKAGE+".sapt_cuest_basis")


class Shell:
    def __init__(self, center, am, pure=True):
        self.ncenter, self.am, self.pure = center, am, pure
        self.xyz = (1.5*center, -0.3*center, 0.7*center)
        self.exponents = [0.17, 2.3]
        self.coefficients = [0.31, -0.8]
        self.original = [0.41, -0.7]
        self.erd = [0.51, -0.6]
        self.function_index = 0
    @property
    def nprimitive(self): return len(self.exponents)
    @property
    def ncartesian(self): return (self.am+1)*(self.am+2)//2
    @property
    def nfunction(self): return 2*self.am+1 if self.pure else self.ncartesian
    def coord(self, k): return self.xyz[k]
    def exp(self, p): return self.exponents[p]
    def coef(self, p): return self.coefficients[p]
    def original_coef(self, p): return self.original[p]
    def erd_coef(self, p): return self.erd[p]
    def is_pure(self): return self.pure
    def is_cartesian(self): return not self.pure


class Basis:
    def __init__(self, shells=None, ncenter=2):
        self.shells = shells if shells is not None else [
            Shell(0,0), Shell(0,1), Shell(0,2), Shell(1,0), Shell(1,1)]
        self.ncenter, self.ecp = ncenter, False
        self.molecule_coords = [(1.5*c,-0.3*c,0.7*c) for c in range(ncenter)]
        offset = cart = 0
        self.cart_offsets = []
        for shell in self.shells:
            shell.function_index = offset
            self.cart_offsets.append(cart)
            offset += shell.nfunction
            cart += shell.ncartesian
    def has_ECP(self): return self.ecp
    def n_ecp_core(self): return 0
    def molecule(self):
        return types.SimpleNamespace(natom=lambda:self.ncenter,
                                     x=lambda c:self.molecule_coords[c][0],
                                     y=lambda c:self.molecule_coords[c][1],
                                     z=lambda c:self.molecule_coords[c][2])
    def nshell(self): return len(self.shells)
    def nbf(self): return sum(s.nfunction for s in self.shells)
    def nao(self): return sum(s.ncartesian for s in self.shells)
    def nprimitive(self): return sum(s.nprimitive for s in self.shells)
    def max_am(self): return max(s.am for s in self.shells)
    def shell(self, i): return self.shells[i]
    def shell_to_center(self, i): return self.shells[i].ncenter
    def shell_to_basis_function(self, i): return self.shells[i].function_index
    def shell_to_ao_function(self, i): return self.cart_offsets[i]
    def __getattr__(self, name):
        raise AssertionError("Unexpected factory access (no DFHelper/integrals): "+name)


class BasisPreflight(unittest.TestCase):
    def test_snapshot_uses_actual_nbf_and_retains_all_uninterpreted_coefficients(self):
        basis = Basis()
        result = basis_api.snapshot_basis(basis, role="primary")
        self.assertEqual((result.nbf,result.ncartesian),(13,14))
        self.assertIs(result.owner,basis)
        self.assertEqual(result.shells_per_center,(3,2))
        self.assertEqual(result.shells[2].function_offset,4)
        self.assertEqual(result.shells[3].cartesian_offset,10)
        self.assertEqual(result.shells[0].psi4_coefficients,(0.31,-0.8))
        self.assertEqual(result.shells[0].original_coefficients,(0.41,-0.7))
        self.assertEqual(result.shells[0].erd_coefficients,(0.51,-0.6))
        self.assertEqual(result.fingerprint,basis_api.snapshot_basis(basis,role="primary").fingerprint)
        with self.assertRaises(FrozenInstanceError): result.nbf = 0

    def test_cartesian_primary_rejects_even_with_spherical_auxiliary(self):
        primary = Basis([Shell(0,0,False),Shell(0,1,False),Shell(0,2,False),Shell(1,0,False)])
        auxiliary = Basis([Shell(0,0),Shell(0,1),Shell(1,0)])
        with self.assertRaisesRegex(NotImplementedError,"spherical primary"):
            basis_api.snapshot_pair(primary,auxiliary)

    def test_spherical_guards_check_flags_even_when_sp_dimensions_are_equal(self):
        for role in ("primary","auxiliary"):
            for am in (0,1):
                basis = Basis([Shell(0,am,False)],1)
                self.assertEqual(basis.nbf(),basis.nao())
                with self.assertRaisesRegex(NotImplementedError,"spherical "+role):
                    basis_api.require_spherical_basis(basis,role=role)
        basis_api.require_spherical_basis(Basis(),role="primary")

    def test_ecp_cartesian_auxiliary_mixed_missing_center_holes_and_order_reject(self):
        ecp = Basis(); ecp.ecp = True
        examples = [
            (ecp,"primary","ECP"),
            (Basis([Shell(0,0,False)],1),"auxiliary","spherical auxiliary"),
            (Basis([Shell(0,0),Shell(0,1,False)],1),"primary","spherical primary"),
            (Basis([Shell(0,0)],2),"primary","Every native"),
            (Basis([Shell(0,0),Shell(0,2)],1),"primary","holes"),
            (Basis([Shell(0,0),Shell(1,0),Shell(0,1)]),"primary","non-center"),
        ]
        for basis,role,message in examples:
            with self.subTest(message=message),self.assertRaisesRegex(NotImplementedError,message):
                basis_api.snapshot_basis(basis,role=role)

    def test_bad_offsets_and_same_center_coordinate_disagreement_reject(self):
        basis = Basis(); basis.cart_offsets[2] += 1
        with self.assertRaisesRegex(ValueError,"offsets"):
            basis_api.snapshot_basis(basis,role="primary")
        basis = Basis(); basis.shells[1].xyz = (7.,8.,9.)
        with self.assertRaisesRegex(ValueError,"different coordinates"):
            basis_api.snapshot_basis(basis,role="primary")

    def test_primary_auxiliary_geometry_and_center_count_must_match(self):
        for auxiliary in (Basis([Shell(0,0)],1), Basis()):
            if auxiliary.ncenter == 2:
                for shell in auxiliary.shells:
                    if shell.ncenter == 1: shell.xyz = (8.,9.,10.)
            with self.assertRaisesRegex(ValueError,"geometry|centers"):
                basis_api.snapshot_pair(Basis(),auxiliary)

    def test_nonfinite_zero_or_negative_primitive_data_reject(self):
        for field,value in (("exponents",0.),("exponents",-1.),("exponents",float("inf")),
                            ("coefficients",float("nan")),("original",float("inf")),
                            ("erd",float("nan"))):
            basis = Basis(); getattr(basis.shells[0],field)[0] = value
            with self.subTest(field=field,value=value),self.assertRaises(ValueError):
                basis_api.snapshot_basis(basis,role="primary")
        basis = Basis(); basis.shells[0].coefficients = [0.,0.]
        with self.assertRaisesRegex(ValueError,"Zero-contraction"):
            basis_api.snapshot_basis(basis,role="primary")

    def test_fingerprint_detects_changed_inputs_without_aliasing_old_snapshot(self):
        for variant in ("coefficients","original","erd"):
            with self.subTest(variant=variant):
                basis = Basis(); snapshot = basis_api.snapshot_basis(basis,role="primary")
                snapshot.assert_unchanged()
                getattr(basis.shells[0],variant)[0] += .1
                self.assertEqual(snapshot.shells[0].psi4_coefficients,(0.31,-0.8))
                self.assertEqual(snapshot.shells[0].original_coefficients,(0.41,-0.7))
                self.assertEqual(snapshot.shells[0].erd_coefficients,(0.51,-0.6))
                with self.assertRaisesRegex(ValueError,"changed"):
                    snapshot.assert_unchanged()
                self.assertNotEqual(snapshot.fingerprint,
                                    basis_api.snapshot_basis(basis,role="primary").fingerprint)

    def test_molecule_only_or_shell_only_geometry_changes_reject(self):
        for change in ("molecule","shell"):
            basis = Basis(); snapshot = basis_api.snapshot_basis(basis,role="primary")
            if change == "molecule":
                basis.molecule_coords[1] = (8.,9.,10.)
            else:
                for shell in basis.shells:
                    if shell.ncenter == 1: shell.xyz = (8.,9.,10.)
            with self.subTest(change=change),self.assertRaisesRegex(ValueError,"Shell/molecule geometry"):
                basis_api.snapshot_basis(basis,role="primary")
            with self.assertRaisesRegex(ValueError,"changed"):
                snapshot.assert_unchanged()
        primary,auxiliary = Basis(),Basis()
        auxiliary.molecule_coords[1] = (8.,9.,10.)
        with self.assertRaisesRegex(ValueError,"Shell/molecule geometry"):
            basis_api.snapshot_pair(primary,auxiliary)

    def test_consistent_shell_and_molecule_move_changes_fingerprint(self):
        basis = Basis(); snapshot = basis_api.snapshot_basis(basis,role="primary")
        basis.molecule_coords[1] = (8.,9.,10.)
        for shell in basis.shells:
            if shell.ncenter == 1: shell.xyz = (8.,9.,10.)
        fresh = basis_api.snapshot_basis(basis,role="primary")
        self.assertEqual(fresh.coordinates,fresh.molecule_coordinates)
        self.assertNotEqual(snapshot.fingerprint,fresh.fingerprint)
        with self.assertRaisesRegex(ValueError,"changed"):
            snapshot.assert_unchanged()

    def test_invalid_counts_and_roles_reject_before_native_resources_exist(self):
        with self.assertRaisesRegex(ValueError,"role"):
            basis_api.snapshot_basis(Basis(),role="RIFIT")
        for count in (True,0,-1,1.5):
            basis = Basis(); basis.ncenter = count
            with self.subTest(count=count),self.assertRaises(ValueError):
                basis_api.snapshot_basis(basis,role="primary")
        basis = Basis(); basis.nbf = lambda:999
        with self.assertRaisesRegex(ValueError,"aggregate"):
            basis_api.snapshot_basis(basis,role="primary")


if __name__ == "__main__":
    unittest.main()
