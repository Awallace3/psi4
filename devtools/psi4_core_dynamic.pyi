# Typed declarations for APIs attached to psi4.core by the Python driver.

class BasisSet:
    def __getattr__(self, name: str) -> typing.Any: ...
    @typing.overload
    @staticmethod
    def build(mol: Molecule, key: str | None = None, target: str | collections.abc.Callable | None = None, fitrole: str = "ORBITAL", other: str | collections.abc.Callable | None = None, puream: int = -1, return_atomlist: typing.Literal[False] = False, *, quiet: bool = False) -> BasisSet: ...
    @typing.overload
    @staticmethod
    def build(mol: Molecule, key: str | None = None, target: str | collections.abc.Callable | None = None, fitrole: str = "ORBITAL", other: str | collections.abc.Callable | None = None, puream: int = -1, return_atomlist: typing.Literal[True] = True, *, quiet: bool = False) -> list[BasisSet]: ...

class CIVector:
    @property
    def np(self) -> numpy.ndarray:
        """Zero-copy NumPy view of the CI vector buffer."""

class CUHF(HF):
    def compute_orbital_gradient(self, save_fock: bool, max_diis_vectors: int) -> float: ...
    def diis(self, Dnorm: float) -> typing.Any: ...

class CubeProperties:
    def compute_properties(self) -> None:
        """Compute requested cube properties and manage output files."""

class Dimension:
    @classmethod
    def from_list(cls, dims: tuple[int, ...] | list[int] | numpy.ndarray | Dimension, name: str = "New Dimension") -> typing.Self:
        """Construct a Dimension from integer sizes."""
    def to_tuple(self) -> tuple[int, ...]:
        """Return dimension sizes as a tuple."""
    def __iter__(self) -> collections.abc.Iterator[int]: ...

class FISAPT:
    def compute_energy(self, jk_obj: JK, *, external_potentials: typing.Any = None) -> typing.Any: ...
    def fdrop(self, external_potentials: typing.Any = None) -> typing.Any: ...
    def plot(self) -> typing.Any: ...
    def save_variables_to_wfn(self, ref_wfn: Wavefunction, external_potentials: typing.Any = None) -> typing.Any: ...

class HF(Wavefunction):
    iteration_energies: list[float]
    def initialize(self) -> typing.Any: ...
    def compute_energy(self) -> float: ...
    def initialize_jk(self, memory: int, jk: JK | None = None) -> typing.Any: ...
    def iterations(self, e_conv: float | None = None, d_conv: float | None = None) -> typing.Any: ...
    def finalize_energy(self) -> float: ...
    def print_energies(self) -> None: ...
    def print_preiterations(self, small: bool = False) -> None: ...
    def validate_diis(self) -> None: ...

class JK:
    @staticmethod
    def build(orbital_basis: BasisSet, aux: BasisSet | None = None, jk_type: str | None = None, do_wK: bool | None = None, memory: int | None = None) -> JK:
        """Construct a JK object from an orbital basis."""

class Matrix:
    @staticmethod
    def doublet(A: Matrix, B: Matrix, transA: bool, transB: bool) -> Matrix:
        """Multiply two matrices; deprecated in favor of core.doublet."""
    @staticmethod
    def triplet(A: Matrix, B: Matrix, C: Matrix, transA: bool, transB: bool, transC: bool) -> Matrix:
        """Multiply three matrices; deprecated in favor of core.triplet."""
    def __iter__(self) -> typing.Never: ...
    def __getitem__(self, key: typing.Any) -> typing.Never: ...
    def __getattr__(self, name: str) -> typing.Any: ...
    @classmethod
    def from_array(cls, arr: numpy.ndarray | list[numpy.ndarray], name: str = "New Matrix", dim1: list | tuple | Dimension | None = None, dim2: Dimension | None = None) -> typing.Self:
        """Construct from a NumPy array or a list of irrep arrays."""
    @classmethod
    def from_list(cls, values: list) -> typing.Self:
        """Construct from values convertible to a NumPy array."""
    def to_array(self, copy: bool = True, dense: bool = False) -> numpy.ndarray | list[numpy.ndarray]:
        """Return a copy or view as NumPy arrays."""
    @property
    def np(self) -> numpy.ndarray:
        """Zero-copy NumPy view for an object with one irrep."""
    @property
    def nph(self) -> list[numpy.ndarray]:
        """Zero-copy NumPy views, one for each irrep."""
    @property
    def shape(self) -> tuple[int, ...] | tuple[tuple[int, ...], ...]:
        """Shape of the NumPy view or views."""
    @property
    def __array_interface__(self) -> dict[str, typing.Any]:
        """NumPy array interface for an object with one irrep."""
    def np_write(self, filename: str | None = None, prefix: str = "") -> dict[str, typing.Any] | None:
        """Write to a NumPy archive or return packed data."""
    @classmethod
    def np_read(cls, filename: str, prefix: str = "") -> typing.Self:
        """Read from a NumPy compressed or uncompressed file."""
    def to_serial(self) -> dict[str, typing.Any]:
        """Serialize this object to a dictionary."""
    @classmethod
    def from_serial(cls, json_data: dict[str, typing.Any]) -> typing.Self:
        """Construct from serialized data."""
    def chain_dot(self, *args: Matrix, trans: collections.abc.Iterable[bool] | None = None) -> Matrix:
        """Chain matrix products, optionally transposing individual arguments."""

class Molecule:
    def __setattr__(self, name: str, value: typing.Any) -> None: ...
    def __getattr__(self, name: str) -> typing.Any: ...
    def to_arrays(self, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        """Return array representations of this molecule."""
    def to_dict(self, force_c1: bool = False, force_units: bool = False, np_out: bool = True, quiet: bool = False) -> dict[str, typing.Any]:
        """Serialize this molecule to a dictionary."""
    def BFS(self, *args: typing.Any, **kwargs: typing.Any) -> typing.Any: ...
    def B787(self, *args: typing.Any, **kwargs: typing.Any) -> typing.Any: ...
    def scramble(self, *args: typing.Any, **kwargs: typing.Any) -> typing.Any: ...
    @classmethod
    def from_arrays(cls, *args: typing.Any, **kwargs: typing.Any) -> typing.Self: ...
    @classmethod
    def from_string(cls, molstr: str, *args: typing.Any, **kwargs: typing.Any) -> typing.Self: ...
    def to_string(self, *args: typing.Any, **kwargs: typing.Any) -> str: ...
    @typing.overload
    @classmethod
    def from_schema(cls, molschema: dict[str, typing.Any], return_dict: typing.Literal[False] = False, nonphysical: bool = False, verbose: int = 1) -> typing.Self: ...
    @typing.overload
    @classmethod
    def from_schema(cls, molschema: dict[str, typing.Any], return_dict: typing.Literal[True] = True, nonphysical: bool = False, verbose: int = 1) -> tuple[typing.Self, dict[str, typing.Any]]: ...
    def to_schema(self, *args: typing.Any, **kwargs: typing.Any) -> typing.Any: ...
    def run_dftd3(self, *args: typing.Any, **kwargs: typing.Any) -> typing.Any: ...
    def run_sdftd3(self, *args: typing.Any, **kwargs: typing.Any) -> typing.Any: ...
    def run_dftd4(self, *args: typing.Any, **kwargs: typing.Any) -> typing.Any: ...
    def run_gcp(self, *args: typing.Any, **kwargs: typing.Any) -> typing.Any: ...
    @staticmethod
    def format_molecule_for_mol(*args: typing.Any, **kwargs: typing.Any) -> str: ...

class OEProp:
    valid_methods: typing.ClassVar[list[str]]

class RHF(HF):
    def compute_orbital_gradient(self, save_fock: bool, max_diis_vectors: int) -> float: ...
    def diis(self, Dnorm: float) -> typing.Any: ...

class ROHF(HF):
    def compute_orbital_gradient(self, save_fock: bool, max_diis_vectors: int) -> float: ...
    def diis(self, Dnorm: float) -> typing.Any: ...

class UHF(HF):
    def stability_analysis(self) -> typing.Any: ...
    def compute_orbital_gradient(self, save_fock: bool, max_diis_vectors: int) -> float: ...
    def diis(self, Dnorm: float) -> typing.Any: ...

class VBase:
    def get_np_xyzw(self) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]:
        """Return grid x, y, z, and weight arrays."""

class Vector(ProtoVector):
    def __iter__(self) -> typing.Never: ...
    def __getitem__(self, key: typing.Any) -> typing.Never: ...
    def __getattr__(self, name: str) -> typing.Any: ...
    @classmethod
    def from_array(cls, arr: numpy.ndarray | list[numpy.ndarray], name: str = "New Matrix", dim1: list | tuple | Dimension | None = None, dim2: Dimension | None = None) -> typing.Self:
        """Construct from a NumPy array or a list of irrep arrays."""
    @classmethod
    def from_list(cls, values: list) -> typing.Self:
        """Construct from values convertible to a NumPy array."""
    def to_array(self, copy: bool = True, dense: bool = False) -> numpy.ndarray | list[numpy.ndarray]:
        """Return a copy or view as NumPy arrays."""
    @property
    def np(self) -> numpy.ndarray:
        """Zero-copy NumPy view for an object with one irrep."""
    @property
    def nph(self) -> list[numpy.ndarray]:
        """Zero-copy NumPy views, one for each irrep."""
    @property
    def shape(self) -> tuple[int, ...] | tuple[tuple[int, ...], ...]:
        """Shape of the NumPy view or views."""
    @property
    def __array_interface__(self) -> dict[str, typing.Any]:
        """NumPy array interface for an object with one irrep."""
    def np_write(self, filename: str | None = None, prefix: str = "") -> dict[str, typing.Any] | None:
        """Write to a NumPy archive or return packed data."""
    @classmethod
    def np_read(cls, filename: str, prefix: str = "") -> typing.Self:
        """Read from a NumPy compressed or uncompressed file."""
    def to_serial(self) -> dict[str, typing.Any]:
        """Serialize this object to a dictionary."""
    @classmethod
    def from_serial(cls, json_data: dict[str, typing.Any]) -> typing.Self:
        """Construct from serialized data."""

class Wavefunction:
    def __getattr__(self, name: str) -> typing.Any: ...
    @staticmethod
    def build(mol: Molecule, basis: str | BasisSet | None = None, *, quiet: bool = False) -> Wavefunction:
        """Build a wavefunction from a molecule and optional basis."""
    def get_scratch_filename(self, filenumber: int) -> str:
        """Return the canonical scratch path for a file number."""
    @staticmethod
    def from_file(wfn_data: str | dict[str, typing.Any]) -> Wavefunction:
        """Build a wavefunction from serialized data or a filename."""
    def to_file(self, filename: str | None = None) -> dict[str, dict[str, typing.Any]]:
        """Serialize this wavefunction, optionally writing it to a file."""
    def frequencies(self) -> dict[str, numpy.ndarray] | None:
        """Return vibrational frequency-analysis results when available."""
    def write_nbo(self, name: str) -> None:
        """Write wavefunction information in NBO format."""
    def write_molden(self, filename: str | None = None, do_virtual: bool | None = None, use_natural: bool = False) -> None:
        """Write wavefunction information in Molden format."""
    def has_variable(self, key: str) -> bool:
        """Whether a scalar or array QCVariable has been set on this wavefunction."""
    def variable(self, key: str) -> float | Matrix | numpy.ndarray:
        """Return a scalar or array QCVariable from this wavefunction."""
    def set_variable(self, key: str, val: float | Matrix | numpy.ndarray) -> None:
        """Set a scalar or array QCVariable on this wavefunction."""
    def del_variable(self, key: str) -> None:
        """Remove a scalar or array QCVariable from this wavefunction."""
    def variables(self, include_deprecated_keys: bool = False) -> dict[str, float | Matrix | numpy.ndarray]:
        """Return all QCVariables set on this wavefunction."""

EXTERN: typing.Any | None

def set_global_option_python(key: str, value: typing.Any) -> None:
    """Set a global option whose value may be a Python object."""

def has_variable(key: str) -> bool:
    """Whether a scalar or array QCVariable has been set in global memory."""

def variable(key: str) -> float | Matrix | numpy.ndarray:
    """Return a scalar or array QCVariable from global memory."""

def set_variable(key: str, val: float | Matrix | numpy.ndarray) -> None:
    """Set a scalar or array QCVariable in global memory."""

def del_variable(key: str) -> None:
    """Remove a scalar or array QCVariable from global memory."""

def variables(include_deprecated_keys: bool = False) -> dict[str, float | Matrix | numpy.ndarray]:
    """Return all QCVariables set in global memory."""
