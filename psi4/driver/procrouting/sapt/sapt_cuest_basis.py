"""Pure structural preflight for a future explicit cuEST0.2.50 factory.

No Psi4/CUDA imports, native calls, integrals, transforms or factor generation.
These snapshots deliberately DO NOT select a normalization/component convention
or admit a production provider. All three Psi4 coefficient variants are retained
for the independent numerical conversion gate.
"""
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import operator


def _count(value, label, *, minimum=0, maximum=2**63-1):
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer, not bool")
    try:
        value = operator.index(value)
    except TypeError as error:
        raise ValueError(f"{label} must be an integer") from error
    if not minimum <= value <= maximum:
        raise ValueError(f"{label} outside supported bounds")
    return value


def _real(value, label, *, positive=False):
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite real data")
    value = float(value)
    if not math.isfinite(value) or (positive and value <= 0):
        raise ValueError(f"{label} must be finite" + (" and positive" if positive else ""))
    return value


@dataclass(frozen=True)
class ShellSnapshot:
    index: int
    center: int
    am: int
    pure: bool
    nfunction: int
    ncartesian: int
    function_offset: int
    cartesian_offset: int
    xyz: tuple
    exponents: tuple
    psi4_coefficients: tuple
    original_coefficients: tuple
    erd_coefficients: tuple


@dataclass(frozen=True)
class BasisSnapshot:
    role: str
    ncenter: int
    nbf: int
    ncartesian: int
    shells: tuple
    coordinates: tuple
    molecule_coordinates: tuple
    fingerprint: str
    # Retention/identity only, not part of the mathematical fingerprint.
    owner: object = field(compare=False, repr=False)

    @property
    def shells_per_center(self):
        return tuple(sum(shell.center == center for shell in self.shells)
                     for center in range(self.ncenter))

    def assert_unchanged(self):
        try:
            fresh = snapshot_basis(self.owner, role=self.role)
        except (ValueError, NotImplementedError) as error:
            raise ValueError("Psi4 basis changed since native-factory preflight") from error
        if fresh.fingerprint != self.fingerprint:
            raise ValueError("Psi4 basis changed since native-factory preflight")


def snapshot_basis(basis, *, role):
    """Read exported Psi4 properties/methods without guessing native conventions.

    Initial structural scope rejects ECP, non-center-contiguous shell order,
    missing centers, angular-momentum holes, and mixed shell representations.
    Auxiliary shells must already be pure; no implicit Cartesian conversion.
    Shell and owning-molecule coordinates must agree exactly; independently
    moved/stale bases are unsupported. This is NOT a cuEST coordinate-unit proof.
    """
    if role not in ("primary", "auxiliary"):
        raise ValueError("Basis role must be primary or auxiliary")
    if basis.has_ECP() or basis.n_ecp_core():
        raise NotImplementedError("cuEST factory ECP conversion is unsupported")
    molecule = basis.molecule()
    ncenter = _count(molecule.natom(), "center count", minimum=1)
    molecule_coordinates = tuple(
        tuple(_real(getattr(molecule, axis)(center), "molecule coordinate") for axis in ("x","y","z"))
        for center in range(ncenter))
    nshell = _count(basis.nshell(), "shell count", minimum=1)
    nbf = _count(basis.nbf(), "basis function count", minimum=1)
    ncartesian = _count(basis.nao(), "Cartesian function count", minimum=1)
    shells, centers, angular = [], {}, {center: set() for center in range(ncenter)}
    function_offset = cartesian_offset = primitives = 0
    previous_center = -1
    for index in range(nshell):
        shell = basis.shell(index)
        center = _count(basis.shell_to_center(index), "shell center", maximum=ncenter-1)
        if center != shell.ncenter:
            raise ValueError("Shell center disagrees with basis mapping")
        if center < previous_center:
            raise NotImplementedError("Native AO permutation for non-center-contiguous shells is unadmitted")
        previous_center = center
        am = _count(shell.am, "angular momentum", maximum=10)
        pure, cartesian = shell.is_pure(), shell.is_cartesian()
        if not isinstance(pure, bool) or not isinstance(cartesian, bool) or pure == cartesian:
            raise ValueError("Shell representation flags are inconsistent")
        if role == "auxiliary" and not pure:
            raise NotImplementedError("cuEST DF auxiliary basis must already be pure")
        nprimitive = _count(shell.nprimitive, "primitive count", minimum=1)
        nf = _count(shell.nfunction, "shell function count", minimum=1)
        nc = _count(shell.ncartesian, "shell Cartesian count", minimum=1)
        if nc != (am+1)*(am+2)//2 or nf != (2*am+1 if pure else nc):
            raise ValueError("Shell angular momentum/function dimensions disagree")
        offsets = (_count(basis.shell_to_basis_function(index), "basis shell offset"),
                   _count(shell.function_index, "shell function offset"),
                   _count(basis.shell_to_ao_function(index), "Cartesian shell offset"))
        if offsets != (function_offset, function_offset, cartesian_offset):
            raise ValueError("Shell offsets require an unadmitted AO permutation")
        xyz = tuple(_real(shell.coord(k), "shell coordinate") for k in range(3))
        if center in centers and centers[center] != xyz:
            raise ValueError("Shells on one center have different coordinates")
        centers[center] = xyz
        exponents = tuple(_real(shell.exp(p), "primitive exponent", positive=True)
                          for p in range(nprimitive))
        variants = tuple(tuple(_real(getter(p), "contraction coefficient")
                               for p in range(nprimitive))
                         for getter in (shell.coef, shell.original_coef, shell.erd_coef))
        if not any(variants[0]):
            raise ValueError("Zero-contraction shell is unsupported")
        shells.append(ShellSnapshot(index, center, am, pure, nf, nc,
                                    function_offset, cartesian_offset, xyz,
                                    exponents, *variants))
        angular[center].add(am)
        function_offset += nf
        cartesian_offset += nc
        primitives += nprimitive
    if set(centers) != set(range(ncenter)):
        raise NotImplementedError("Every native basis center must have shells")
    for values in angular.values():
        if values != set(range(max(values)+1)):
            raise NotImplementedError("cuEST basis angular-momentum holes are unsupported")
    if len({shell.pure for shell in shells}) != 1:
        raise NotImplementedError("cuEST basis cannot mix pure and Cartesian shells")
    if (function_offset != nbf or cartesian_offset != ncartesian
            or primitives != _count(basis.nprimitive(), "basis primitive count", minimum=1)
            or max(shell.am for shell in shells) != basis.max_am()):
        raise ValueError("Basis and shell aggregate dimensions disagree")
    coordinates = tuple(centers[c] for c in range(ncenter))
    if coordinates != molecule_coordinates:
        raise ValueError("Shell/molecule geometry differ; independently moved or stale bases are unsupported")
    payload = {"schema": "psi4-shell-structure-v1", "role": role,
               "ncenter": ncenter, "nbf": nbf, "ncartesian": ncartesian,
               "shells": [asdict(shell) for shell in shells],
               "coordinates": coordinates,
               "molecule_coordinates": molecule_coordinates,
               "native_normalization_layout_units_admitted": False}
    fingerprint = hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return BasisSnapshot(role, ncenter, nbf, ncartesian, tuple(shells),
                         coordinates, molecule_coordinates, fingerprint, basis)


def snapshot_pair(primary, auxiliary):
    """Require one common center/geometry ordering without producing any factors."""
    primary_snapshot = snapshot_basis(primary, role="primary")
    auxiliary_snapshot = snapshot_basis(auxiliary, role="auxiliary")
    if (primary_snapshot.ncenter != auxiliary_snapshot.ncenter
            or primary_snapshot.coordinates != auxiliary_snapshot.coordinates):
        raise ValueError("Primary/RI centers or shell geometry differ")
    return primary_snapshot, auxiliary_snapshot
