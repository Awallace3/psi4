"""Tests for the persistent cache of MBIS free-atom reference volumes.

The property under test is not "the cache is fast" but "the cache is honest": a hit has to be
indistinguishable from the computation it replaced, and anything that could make it distinguishable
-- a different basis, a different grid, a damaged file -- has to produce a miss instead.  Several
of the tests below therefore assert that something does *not* hit.
"""

import json
import multiprocessing
import os

import pytest

import psi4
from psi4.driver.p4util import free_atom_cache as fac
from psi4.driver.p4util import prop_util

pytestmark = [pytest.mark.psi, pytest.mark.api]

# A cheap but complete MBIS setup: the free-atom machinery does not care how hard the molecule was.
BASE_OPTIONS = {
    "basis": "cc-pvdz",
    "scf_type": "df",
    "e_convergence": 10,
    "d_convergence": 8,
    "mbis_radial_points": 75,
    "mbis_spherical_points": 302,
    "scf_properties": ["MBIS_CHARGES", "MBIS_VOLUME_RATIOS"],
}


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    """Point the cache at a scratch directory, and leave the user's own cache alone."""
    monkeypatch.setenv("PSI4_FREE_ATOM_CACHE_PATH", str(tmp_path / "free_atom_volumes"))
    monkeypatch.setenv("PSI4_FREE_ATOM_CACHE", "READWRITE")
    fac.clear()
    yield fac.path()
    fac.clear()


@pytest.fixture(autouse=True)
def fresh_options():
    psi4.core.clean_options()
    psi4.core.clean()
    psi4.core.clean_variables()
    yield
    psi4.core.clean_options()


def _water():
    return psi4.geometry("""
0 1
O  0.000  0.000  0.000
H  0.000  0.000  0.960
H  0.906  0.000 -0.317
units angstrom
""")


def _dummy_key(**overrides):
    fields = dict(symbol="O", Z=8, multiplicity=3, reference="UHF", theory="PBE",
                  basis_hash="0" * 64)
    fields.update(overrides)
    return fac.build_key(**fields)


# ==> 1. A hit is indistinguishable from a miss <==


def test_hit_equals_miss(cache_dir):
    psi4.set_options(BASE_OPTIONS)

    _water()
    _, wfn = psi4.energy("pbe", return_wfn=True)
    first = wfn.array_variable("MBIS VOLUME RATIOS").np.ravel().copy()
    assert len(fac.list_entries()) == 2, "one entry each for O and H"

    # Clearing only the in-memory half forces the second run through the files.
    fac.clear(memory_only=True)
    psi4.core.clean()
    psi4.core.clean_variables()

    atomic_runs = []
    wrapped = prop_util._run_free_atom

    def counting(*args, **kwargs):
        atomic_runs.append(args[0])
        return wrapped(*args, **kwargs)

    prop_util._run_free_atom = counting
    try:
        _water()
        _, wfn = psi4.energy("pbe", return_wfn=True)
    finally:
        prop_util._run_free_atom = wrapped
    second = wfn.array_variable("MBIS VOLUME RATIOS").np.ravel().copy()

    assert atomic_runs == [], f"cache hit still ran atomic SCFs for {atomic_runs}"
    # Bit-for-bit, not merely close: the cached value is the same double that was computed.
    assert (first == second).all(), f"{first} != {second}"


def test_prewarm_is_reused_by_a_molecule(cache_dir):
    """The entries prewarm() writes have to be the ones a real job looks for.

    They very nearly were not: the changed-option sweep noticed that psi4's own SCF bookkeeping had
    touched INTS_TOLERANCE, so an entry written before any SCF had run in the process could never be
    found by one written after.  Prewarming is the cache's whole reason to exist on a cluster, so
    this asserts the round trip and not just that prewarm() returns a number.
    """
    psi4.set_options(BASE_OPTIONS)
    warmed = fac.prewarm(["O", "H"], "pbe", basis="cc-pvdz")
    assert sorted(warmed) == ["H", "O"]
    assert len(fac.list_entries()) == 2

    fac.clear(memory_only=True)
    psi4.core.clean()
    psi4.core.clean_variables()

    atomic_runs = []
    wrapped = prop_util._run_free_atom

    def counting(*args, **kwargs):
        atomic_runs.append(args[0])
        return wrapped(*args, **kwargs)

    prop_util._run_free_atom = counting
    try:
        _water()
        _, wfn = psi4.energy("pbe", return_wfn=True)
    finally:
        prop_util._run_free_atom = wrapped

    assert atomic_runs == [], f"prewarmed entries missed; recomputed {atomic_runs}"
    assert wfn.scalar_variable("MBIS FREE ATOM O VOLUME") == warmed["O"]
    assert wfn.scalar_variable("MBIS FREE ATOM H VOLUME") == warmed["H"]


def test_prewarm_is_reused_with_scf_type_left_at_its_default(cache_dir):
    """The same round trip with SCF_TYPE unset, which is how anyone would actually run it.

    Every other test here sets SCF_TYPE explicitly, and that hid a real failure: a job promotes an
    unset SCF_TYPE to DF before asking for free-atom volumes, so its key records DF, while
    prewarm() read the raw default PK and wrote entries under it -- entries that no job could find,
    and that were labelled with an algorithm they had not been computed with.
    """
    psi4.set_options({name: value for name, value in BASE_OPTIONS.items() if name != "scf_type"})
    warmed = fac.prewarm(["O", "H"], "pbe", basis="cc-pvdz")

    entries = fac.list_entries()
    assert len(entries) == 2
    assert {entry["key"]["options"]["SCF_TYPE"] for entry in entries} == {"DF"}, \
        "an entry has to be keyed on the algorithm that produced it"
    # Resolving it must not leave it resolved behind prewarm's back, or the next job in the process
    # would silently be a DF job.
    assert not psi4.core.has_global_option_changed("SCF_TYPE")

    fac.clear(memory_only=True)
    psi4.core.clean()
    psi4.core.clean_variables()

    atomic_runs = []
    wrapped = prop_util._run_free_atom

    def counting(*args, **kwargs):
        atomic_runs.append(args[0])
        return wrapped(*args, **kwargs)

    prop_util._run_free_atom = counting
    try:
        _water()
        _, wfn = psi4.energy("pbe", return_wfn=True)
    finally:
        prop_util._run_free_atom = wrapped

    assert atomic_runs == [], f"prewarmed entries missed; recomputed {atomic_runs}"
    assert wfn.scalar_variable("MBIS FREE ATOM O VOLUME") == warmed["O"]


def test_off_mode_does_not_write(cache_dir, monkeypatch):
    monkeypatch.setenv("PSI4_FREE_ATOM_CACHE", "OFF")
    psi4.set_options(BASE_OPTIONS)
    _water()
    psi4.energy("pbe")
    assert fac.list_entries() == []


def test_read_mode_does_not_write(cache_dir, monkeypatch):
    monkeypatch.setenv("PSI4_FREE_ATOM_CACHE", "READ")
    psi4.set_options(BASE_OPTIONS)
    _water()
    psi4.energy("pbe")
    assert fac.list_entries() == [], "READ must consult the cache without adding to it"


# ==> 2. Everything that changes the number changes the key <==


@pytest.mark.parametrize("options", [
    pytest.param({"puream": False}, id="pueram"),
    pytest.param({"df_basis_scf": "cc-pvtz-jkfit"}, id="df_basis_scf"),
    pytest.param({"scf_type": "pk"}, id="scf_type"),
    pytest.param({"mbis_radial_points": 99}, id="mbis_radial_points"),
    pytest.param({"mbis_spherical_points": 590}, id="mbis_spherical_points"),
    pytest.param({"mbis_d_convergence": 6}, id="mbis_d_convergence"),
    pytest.param({"mbis_screening_threshold": 1.0e-10}, id="mbis_screening_threshold"),
    pytest.param({"mbis_anderson": False}, id="mbis_anderson"),
    pytest.param({"d_convergence": 6}, id="d_convergence"),
    pytest.param({"e_convergence": 8}, id="e_convergence"),
    pytest.param({"dft_spherical_points": 590}, id="dft_spherical_points"),
    pytest.param({"dft_radial_points": 99}, id="dft_radial_points"),
    pytest.param({"dft_pruning_scheme": "robust"}, id="dft_pruning_scheme"),
    pytest.param({"dft_nuclear_scheme": "becke"}, id="dft_nuclear_scheme"),
    pytest.param({"freeze_core": True}, id="freeze_core"),
    pytest.param({"relativistic": "x2c"}, id="relativistic"),
    pytest.param({"stability_analysis": "check"}, id="stability_analysis"),
    pytest.param({"guess": "core"}, id="guess"),
    pytest.param({"maxiter": 200}, id="maxiter"),
    # Not on the hand-maintained whitelist; caught only by the sweep over changed options, which is
    # the backstop that keeps a forgotten option from silently producing a wrong hit.
    pytest.param({"incfock": True}, id="incfock (sweep)"),
    pytest.param({"df_scf_guess": False}, id="df_scf_guess (sweep)"),
])
def test_key_discriminates(cache_dir, options):
    psi4.set_options(BASE_OPTIONS)
    baseline = fac.key_hash(_dummy_key())
    psi4.set_options(options)
    assert fac.key_hash(_dummy_key()) != baseline


@pytest.mark.parametrize("field", [
    pytest.param({"symbol": "N"}, id="element"),
    pytest.param({"Z": 7}, id="Z"),
    pytest.param({"multiplicity": 1}, id="multiplicity"),
    pytest.param({"reference": "RHF"}, id="reference"),
    pytest.param({"theory": "B3LYP"}, id="theory"),
    pytest.param({"basis_hash": "1" * 64}, id="basis"),
    pytest.param({"salt": "mine"}, id="salt"),
])
def test_key_discriminates_on_identity(cache_dir, field):
    psi4.set_options(BASE_OPTIONS)
    assert fac.key_hash(_dummy_key(**field)) != fac.key_hash(_dummy_key())


@pytest.mark.parametrize("options", [
    pytest.param({"print": 3}, id="print"),
    pytest.param({"writer_file_label": "scratch"}, id="writer_file_label"),
    pytest.param({"debug": 1}, id="debug"),
    pytest.param({"cubeprop_tasks": ["density"]}, id="cubeprop_tasks"),
    pytest.param({"molden_with_virtual": False}, id="molden_with_virtual"),
    pytest.param({"geom_maxiter": 77}, id="geom_maxiter"),
])
def test_key_ignores_the_irrelevant(cache_dir, options):
    """Options on the ignore list must not fragment the cache.

    A spurious miss is only ever a wasted SCF, so this list is short and each member is on it
    because it provably cannot reach a converged atomic density.
    """
    psi4.set_options(BASE_OPTIONS)
    baseline = fac.key_hash(_dummy_key())
    psi4.set_options(options)
    assert fac.key_hash(_dummy_key()) == baseline


def test_key_is_indifferent_to_the_basis_name(cache_dir):
    """Renaming a basis must not change the key; the key is built from the functions, not the name."""
    psi4.set_options(BASE_OPTIONS)
    baseline = fac.key_hash(_dummy_key())
    psi4.set_options({"basis": "aug-cc-pvtz"})
    assert fac.key_hash(_dummy_key()) == baseline


# ==> 3. Two bases of the same name are not the same basis <==


def _content_hash_of(body, name):
    mol = psi4.geometry("0 1\nHe 0.0 0.0 0.0\n")
    psi4.basis_helper(body, name=name)
    basisset = psi4.core.BasisSet.build(mol, "ORBITAL", psi4.core.get_global_option("BASIS"), quiet=True)
    return fac.basis_content_hash(basisset, 0)


def test_same_basis_name_different_contents(cache_dir):
    """The whole reason the key hashes shells rather than `mol.basis_on_atom()`.

    That method reports the *block* name, so these two inputs are indistinguishable by it, and a
    name-keyed cache would hand the second run the first run's numbers.
    """
    first = _content_hash_of("assign cc-pvdz", "userdef")
    psi4.core.clean_options()
    second = _content_hash_of("assign cc-pvtz", "userdef")
    assert first is not None and second is not None
    assert first != second


def test_different_basis_name_same_contents(cache_dir):
    """...and the converse, which is what makes anonymous blocks cacheable at all.

    psi4 names an unnamed ``basis {}`` block with a fresh random suffix on every run, so under a
    name-keyed cache such an input could never hit.
    """
    first = _content_hash_of("assign cc-pvdz", "alpha")
    psi4.core.clean_options()
    second = _content_hash_of("assign cc-pvdz", "beta")
    assert first is not None
    assert first == second


# ==> 4. Labelled atoms <==


def test_labelled_atoms_get_their_own_volumes(cache_dir):
    """Two atoms of one element with different bases need different free-atom references.

    `assign H1 cc-pvtz` attaches a basis to a *label*, so identity here has to be the label, and
    grouping has to be by the functions each label ends up with.
    """
    psi4.set_options(BASE_OPTIONS)
    psi4.basis_helper("assign cc-pvdz\nassign H1 cc-pvtz", name="mixed")
    psi4.geometry("0 1\nH1 0.0 0.0 0.0\nH2 0.0 0.0 0.74\nunits angstrom\n")

    _, wfn = psi4.energy("pbe", return_wfn=True)

    one = wfn.scalar_variable("MBIS FREE ATOM H1 VOLUME")
    two = wfn.scalar_variable("MBIS FREE ATOM H2 VOLUME")
    assert one != two, "H1 (cc-pVTZ) and H2 (cc-pVDZ) must not share a free-atom volume"
    assert len(fac.list_entries()) == 2

    ratios = wfn.array_variable("MBIS VOLUME RATIOS").np.ravel()
    assert ratios[0] != ratios[1]


def test_single_atom_ion_gets_a_ratio(cache_dir):
    """An ion is not its own free-atom reference, so its ratio is meaningful and must be produced."""
    psi4.set_options(BASE_OPTIONS)
    psi4.geometry("-1 1\nCl 0.0 0.0 0.0\n")

    _, wfn = psi4.energy("pbe", return_wfn=True)

    ratio = wfn.array_variable("MBIS VOLUME RATIOS").np.ravel()[0]
    assert ratio > 1.2, f"an anion is more diffuse than the neutral atom, got {ratio}"


# ==> 5. Damage produces a miss, never an exception <==


@pytest.mark.parametrize("damage", [
    pytest.param(lambda text: text[:len(text) // 2], id="truncated"),
    pytest.param(lambda text: "this is not json", id="not json"),
    pytest.param(lambda text: json.dumps({**json.loads(text), "schema": 99}), id="future schema"),
    pytest.param(lambda text: json.dumps({**json.loads(text), "value": "not a number"}), id="bad value"),
    pytest.param(lambda text: json.dumps({k: v for k, v in json.loads(text).items() if k != "key"}), id="no key"),
    pytest.param(lambda text: json.dumps({**json.loads(text), "key": {**json.loads(text)["key"], "Z": 99}}),
                 id="key disagrees"),
    pytest.param(lambda text: "", id="empty"),
])
def test_damaged_entry_is_a_miss(cache_dir, damage):
    psi4.set_options(BASE_OPTIONS)
    key = _dummy_key()
    fac.store(key, 12.5)
    fac.clear(memory_only=True)

    entry = cache_dir / f"{fac.key_hash(key)}.json"
    entry.write_text(damage(entry.read_text()))

    assert fac.lookup(key) is None


def test_missing_directory_is_a_miss(cache_dir):
    psi4.set_options(BASE_OPTIONS)
    assert fac.lookup(_dummy_key()) is None
    assert not cache_dir.exists(), "a lookup must not create the cache directory"


def test_uncacheable_basis_yields_no_key(cache_dir):
    """A basis whose contents cannot be hashed must refuse to produce a key at all."""
    assert fac.build_key("O", 8, 3, "UHF", "PBE", None) is None
    assert fac.lookup(None) is None
    fac.store(None, 1.0)  # must not raise, must not write
    assert fac.list_entries() == []


# ==> 6. Concurrent writers <==


def _write_one(args):
    directory, payload = args
    os.environ["PSI4_FREE_ATOM_CACHE_PATH"] = directory
    os.environ["PSI4_FREE_ATOM_CACHE"] = "READWRITE"
    fac.store(payload, 19.651190345897877)


def test_concurrent_writers_leave_one_file(cache_dir):
    """The realistic workload: a few hundred cluster workers racing on the same few elements.

    They write byte-identical content, so the race is benign and no lock is taken; what has to
    hold is that the directory afterwards contains one valid entry and no half-written debris.
    """
    psi4.set_options(BASE_OPTIONS)
    key = _dummy_key()
    cache_dir.mkdir(parents=True, exist_ok=True)

    context = multiprocessing.get_context("fork")
    with context.Pool(8) as pool:
        pool.map(_write_one, [(str(cache_dir), key)] * 8)

    assert sorted(p.name for p in cache_dir.glob("*.json")) == [f"{fac.key_hash(key)}.json"]
    assert list(cache_dir.glob("*.tmp")) == [], "a crashed or lost race must not leave debris"

    fac.clear(memory_only=True)
    assert fac.lookup(key)["value"] == 19.651190345897877


# ==> Inspection API <==


def test_list_and_clear(cache_dir):
    psi4.set_options(BASE_OPTIONS)
    fac.store(_dummy_key(), 1.0)
    fac.store(_dummy_key(symbol="N", Z=7), 2.0)

    entries = fac.list_entries()
    assert sorted(e["key"]["element"] for e in entries) == ["N", "O"]
    assert all("file" in e for e in entries)

    assert fac.clear() == 2
    assert fac.list_entries() == []
