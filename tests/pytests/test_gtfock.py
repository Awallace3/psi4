"""End-to-end tests for the optional GTFock MPI J/K engine.

The GTFock path is opt-in: it exists only when Psi4 was configured with
``-DENABLE_GTFock=ON``. ``test_gtfock_is_optional`` runs everywhere and asserts
the default build reports no GTFock and needs no MPI; everything else skips
through the standard ``uusing("gtfock")`` add-on marker.

Every test here drives the linked library rather than a header or a stub: they
all go through ``psi4.core``/``JK`` into ``libgtfock.so``. The tests that need a
GTFock engine of their own do so in a subprocess, because a process gets exactly
one engine; the multi-rank test launches ``mpirun`` on the installed Psi4 and
reads back what each rank's own GTFock engine reported.
"""

import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

import psi4
from addons import uusing

pytestmark = [pytest.mark.psi, pytest.mark.api]

# GTFock's ERIs come from Simint through a different code path than Psi4's
# Libint2, so the two agree to integral-accumulation roundoff rather than
# bitwise. Psi4's own JK-vs-JK comparisons use 1e-9 on matrix elements; the
# prototype measured max|dJ| ~ 4e-15 and max|dK| ~ 3e-15 on this case, so 1e-9
# leaves six orders of headroom while still catching any real disagreement.
JK_TOL = 1.0e-9
# Energies accumulate over nbf^2 matrix elements and SCF iterations, so allow a
# little more than the per-element tolerance. 1e-9 Eh is far below chemical
# accuracy and matches the tolerance the GTFock milestone-1 SCF regression uses.
ENERGY_TOL = 1.0e-9

GEOMETRY = """
0 1
O   0.000000000000   0.000000000000  -0.068516219320
H   0.000000000000  -0.790689573744   0.543701060715
H   0.000000000000   0.790689573744   0.543701060715
units angstrom
symmetry c1
no_reorient
no_com
"""


def _water():
    return psi4.geometry(GEOMETRY)


@pytest.fixture
def gtfock_mpi():
    """Bring MPI up through Psi4's own mpi4py entry point, or skip."""
    from psi4.driver import gtfock

    try:
        gtfock.initialize()
    except gtfock.GTFockNotAvailable as exc:
        pytest.skip(str(exc))
    return gtfock


def test_gtfock_is_optional():
    """A build without GTFock must say so and must not need MPI to say it.

    This is the optionality guard: it runs in the default suite, where
    ``psi4.core`` is not linked against GTFock, Simint, or MPI at all.
    """
    enabled = psi4.core.gtfock_enabled()
    assert enabled == psi4.addons("gtfock"), "psi4.addons and core disagree about GTFock"

    from psi4.driver import gtfock

    assert gtfock.available() == enabled
    if not enabled:
        # Reporting must work without MPI ever having been initialized.
        assert gtfock.fock_builds() == 0
        assert gtfock.mpi_info() == {"rank": -1, "size": -1, "initialized": False}
        with pytest.raises(gtfock.GTFockNotAvailable):
            gtfock.initialize()


# Run in a subprocess with mpi4py made unimportable, to show the GTFock module
# degrades to a clear error instead of being a hard dependency of `import psi4`.
_NO_MPI4PY_PROBE = """
import sys

class _Block:
    def find_spec(self, name, path=None, target=None):
        if name == "mpi4py" or name.startswith("mpi4py."):
            raise ImportError("mpi4py blocked for this test")
        return None

sys.meta_path.insert(0, _Block())
for mod in [m for m in sys.modules if m == "mpi4py" or m.startswith("mpi4py.")]:
    del sys.modules[mod]

import psi4
from psi4.driver import gtfock

try:
    gtfock.initialize()
except gtfock.GTFockNotAvailable as exc:
    print("RAISED", exc)
else:
    print("NO-RAISE")
"""


def test_gtfock_module_works_without_mpi4py():
    """Importing Psi4 and its GTFock module must not require mpi4py.

    mpi4py is only needed to *run* the GTFock path. With it unimportable,
    ``import psi4`` and ``from psi4.driver import gtfock`` must still succeed and
    ``initialize()`` must fail with an actionable message rather than an
    ImportError traceback.
    """
    probe = subprocess.run([sys.executable, "-c", _NO_MPI4PY_PROBE],
                           capture_output=True, text=True)
    assert probe.returncode == 0, f"{probe.stdout}\n{probe.stderr}"
    assert probe.stdout.startswith("RAISED"), f"{probe.stdout}\n{probe.stderr}"
    if psi4.core.gtfock_enabled():
        assert "mpi4py" in probe.stdout


# Compare GTFock's J/K against DirectJK's for one basis, in a process of its
# own. A process gets exactly one GTFock engine (see `Prototype scope` in
# doc/sphinxman/source/gtfock.rst), so two bases cannot be compared from the
# same interpreter; each parametrization forks a fresh Psi4 instead. This still
# drives the shipped, linked libgtfock.so through psi4.core/JK -- it is the same
# code path the in-process tests use, just isolated.
_JK_COMPARE_PROBE = r'''
import json
import sys

import numpy as np

import psi4
from psi4.driver import gtfock

basis, scratch = sys.argv[1], sys.argv[2]
psi4.core.IOManager.shared_object().set_default_path(scratch)
psi4.core.set_output_file(scratch + "/jk_probe.out", False)
psi4.set_num_threads(1)

gtfock.initialize()

psi4.geometry({geometry!r})
psi4.set_options({{"basis": basis, "puream": False, "df_scf_guess": False,
                  "guess": "core", "e_convergence": 1e-10, "d_convergence": 1e-9}})

# A converged density is a more demanding input than a random matrix: it has
# the sparsity pattern GTFock's screening actually acts on.
psi4.set_options({{"scf_type": "pk"}})
_, wfn = psi4.energy("scf", return_wfn=True)
primary = wfn.basisset()
Cocc = wfn.Ca_subset("AO", "OCC")


def jk_of(scf_type):
    psi4.set_options({{"scf_type": scf_type}})
    jk = psi4.core.JK.build_JK(primary, None)
    jk.set_do_K(True)
    jk.initialize()
    jk.C_clear()
    jk.C_left_add(Cocc)
    jk.compute()
    return jk, np.array(jk.J()[0]), np.array(jk.K()[0])


ref_jk, J_ref, K_ref = jk_of("direct")
builds_before = gtfock.fock_builds()
gt_jk, J_gt, K_gt = jk_of("gtfock")

print("PSI4-GTFOCK-JSON " + json.dumps({{
    "ref_name": ref_jk.name(),
    "gt_name": gt_jk.name(),
    "fock_builds": gtfock.fock_builds() - builds_before,
    "max_dJ": float(np.max(np.abs(J_gt - J_ref))),
    "max_dK": float(np.max(np.abs(K_gt - K_ref))),
    "max_am": max(primary.shell(s).am for s in range(primary.nshell())),
    "nbf": primary.nbf(),
}}), flush=True)
'''.format(geometry=GEOMETRY)


@uusing("gtfock")
@pytest.mark.parametrize("basis,expected_max_am", [("sto-3g", 1), ("6-31G*", 2)])
def test_gtfock_jk_matches_directjk(gtfock_mpi, tmp_path, basis, expected_max_am):
    """J and K from GTFock must match Psi4's own DirectJK for the same density.

    ``6-31G*`` carries a Cartesian ``d`` shell, which is where GTFock's and
    Psi4's conventions could next diverge: the ordering inside a six-function
    ``d`` block, and the normalization Simint applies to the raw contraction
    coefficients the shim hands it. A permuted or mis-scaled J/K would show up
    here rather than silently in a user's energy.
    """
    probe = subprocess.run(
        [sys.executable, "-c", _JK_COMPARE_PROBE, basis, str(tmp_path)],
        capture_output=True, text=True, timeout=900)
    assert probe.returncode == 0, f"{probe.stdout}\n{probe.stderr}"

    lines = [line for line in probe.stdout.splitlines() if line.startswith("PSI4-GTFOCK-JSON ")]
    assert len(lines) == 1, f"{probe.stdout}\n{probe.stderr}"
    report = json.loads(lines[0].split(" ", 1)[1])

    # The basis really did reach the angular momentum this case is here to cover.
    assert report["max_am"] == expected_max_am
    # Guard against a silent fallback to Psi4's own integrals.
    assert report["ref_name"] == "DirectJK"
    assert report["gt_name"] == "GTFockJK"
    assert report["fock_builds"] == 1

    assert report["max_dJ"] < JK_TOL
    assert report["max_dK"] < JK_TOL


@uusing("gtfock")
def test_gtfock_rhf_energy_matches_reference(gtfock_mpi):
    """A full RHF driven by GTFock must land on Psi4's own RHF energy."""
    _water()
    # save_jk keeps the JK object alive past HF::finalize() so its class can be
    # inspected; otherwise wfn.jk() is None and the fallback check is vacuous.
    common = {"basis": "sto-3g", "puream": False, "df_scf_guess": False,
              "guess": "core", "e_convergence": 1e-10, "d_convergence": 1e-9,
              "save_jk": True}

    psi4.set_options({**common, "scf_type": "pk"})
    e_ref = psi4.energy("scf")

    builds_before = gtfock_mpi.fock_builds()
    psi4.set_options({**common, "scf_type": "gtfock"})
    e_gtfock, wfn = psi4.energy("scf", return_wfn=True)

    assert wfn.jk().name() == "GTFockJK"
    assert gtfock_mpi.fock_builds() > builds_before, "GTFock never ran"
    assert abs(e_gtfock - e_ref) < ENERGY_TOL


@uusing("gtfock")
@pytest.mark.parametrize("basis", ["cc-pVDZ", "sto-3g"])
def test_gtfock_refuses_spherical_basis(gtfock_mpi, basis):
    """GTFock's Simint path is Cartesian, so every spherical basis must be refused.

    ``sto-3g`` is the s/p-only case: its shell counts do match GTFock's 2l+1
    sizing, but Simint fills a p shell as px, py, pz while Psi4 orders pure
    shells by m, so it would come back permuted rather than merely mis-sized.
    It has to raise, not return a wrong energy.
    """
    mol = _water()
    primary = psi4.core.BasisSet.build(mol, "ORBITAL", basis, puream=True)
    assert primary.has_puream()
    psi4.set_options({"scf_type": "gtfock"})
    jk = psi4.core.JK.build_JK(primary, None)
    jk.set_do_K(True)
    jk.initialize()
    jk.C_clear()
    jk.C_left_add(psi4.core.Matrix.from_array(np.zeros((primary.nbf(), 1))))

    builds_before = gtfock_mpi.fock_builds()
    with pytest.raises(RuntimeError, match="spherical"):
        jk.compute()
    assert gtfock_mpi.fock_builds() == builds_before


@uusing("gtfock")
def test_gtfock_refuses_high_angular_momentum(gtfock_mpi):
    """A shell above GTFock's angular-momentum ceiling must raise, not corrupt memory.

    libcint indexes GTFock's per-thread shell-pair work lists as
    ``l_P * (l_max + 1) + l_Q`` into a table sized for ``l_max``, with no bound
    check, so an ``h`` shell (``l = 5``) indexes past the end of that array.
    Cartesian ``cc-pV5Z`` puts ``h`` functions on oxygen, and ``puream false`` is
    exactly what the GTFock docs tell users to set, so this is reachable from a
    documented configuration. It has to raise before GTFock is ever entered.
    """
    mol = _water()
    primary = psi4.core.BasisSet.build(mol, "ORBITAL", "cc-pV5Z", puream=False)
    assert not primary.has_puream()
    assert max(primary.shell(s).am for s in range(primary.nshell())) >= 5
    psi4.set_options({"scf_type": "gtfock"})
    jk = psi4.core.JK.build_JK(primary, None)
    jk.set_do_K(True)
    jk.initialize()
    jk.C_clear()
    jk.C_left_add(psi4.core.Matrix.from_array(np.zeros((primary.nbf(), 1))))

    builds_before = gtfock_mpi.fock_builds()
    with pytest.raises(RuntimeError, match="angular momentum"):
        jk.compute()
    assert gtfock_mpi.fock_builds() == builds_before


@uusing("gtfock")
def test_gtfock_refuses_nonsymmetric_reuse(gtfock_mpi):
    """A reused engine that assumes density symmetry must refuse C_left != C_right.

    GTFock is created with its symmetry flag fixed, so once an engine has been
    built for symmetric densities a later SOSCF/response-style build with
    ``C_left != C_right`` would exploit a symmetry the density no longer has.
    That has to raise, not return a plausible-looking wrong J/K.
    """
    mol = _water()
    primary = psi4.core.BasisSet.build(mol, "ORBITAL", "sto-3g", puream=False)
    psi4.set_options({"scf_type": "gtfock"})
    jk = psi4.core.JK.build_JK(primary, None)
    jk.set_do_K(True)
    jk.initialize()

    rng = np.random.RandomState(0)
    Cl = psi4.core.Matrix.from_array(rng.rand(primary.nbf(), 1))
    Cr = psi4.core.Matrix.from_array(rng.rand(primary.nbf(), 1))

    # First build is symmetric, which is what pins the engine's symmetry flag.
    jk.C_clear()
    jk.C_left_add(Cl)
    jk.compute()

    jk.C_clear()
    jk.C_left_add(Cl)
    jk.C_right_add(Cr)
    builds_before = gtfock_mpi.fock_builds()
    with pytest.raises(RuntimeError, match="C_left != C_right"):
        jk.compute()
    # The refusal must come before GTFock is handed the asymmetric density.
    assert gtfock_mpi.fock_builds() == builds_before


@uusing("gtfock")
def test_gtfock_refuses_nonsymmetric_engine(gtfock_mpi):
    """A first-ever GTFock build with C_left != C_right must refuse too.

    The sibling of ``test_gtfock_refuses_nonsymmetric_reuse``: here nothing has
    pinned the symmetry flag, so libfock would otherwise create the engine in
    GTFock's nosymm mode, whose post-build symmetrization is missing upstream.
    """
    mol = _water()
    primary = psi4.core.BasisSet.build(mol, "ORBITAL", "sto-3g", puream=False)
    psi4.set_options({"scf_type": "gtfock"})
    jk = psi4.core.JK.build_JK(primary, None)
    jk.set_do_K(True)
    jk.initialize()

    rng = np.random.RandomState(1)
    jk.C_clear()
    jk.C_left_add(psi4.core.Matrix.from_array(rng.rand(primary.nbf(), 1)))
    jk.C_right_add(psi4.core.Matrix.from_array(rng.rand(primary.nbf(), 1)))

    builds_before = gtfock_mpi.fock_builds()
    with pytest.raises(RuntimeError, match="nosymm"):
        jk.compute()
    assert gtfock_mpi.fock_builds() == builds_before


@uusing("gtfock")
def test_gtfock_refuses_wrongly_shaped_matrix(gtfock_mpi):
    """Every Psi4 <-> GTMatrix transfer must refuse a non-``nbf x nbf`` C1 block.

    ``SetP``, ``GetJ`` and ``GetK`` all move ``nbf*nbf`` contiguous doubles
    through ``Matrix::pointer(0)``, so a matrix of any other shape would be read
    or written past its end rather than fail. They share one guard; this drives
    it through the linked library on the side libfock can actually reach, by
    handing a GTFock JK orbitals whose AO row count does not match the basis.

    ``GetJ``/``GetK`` call the same guard as a backstop for direct C++ users of
    ``gtfock_interface.h``. libfock itself cannot trip that half: ``JK`` sizes
    ``J_ao_``/``K_ao_`` from the very ``D_ao_`` that ``SetP`` validates, so the
    density gate below always fires first.
    """
    mol = _water()
    primary = psi4.core.BasisSet.build(mol, "ORBITAL", "sto-3g", puream=False)
    psi4.set_options({"scf_type": "gtfock"})
    jk = psi4.core.JK.build_JK(primary, None)
    jk.set_do_K(True)
    jk.initialize()

    assert primary.nbf() > 2
    rng = np.random.RandomState(3)
    jk.C_clear()
    jk.C_left_add(psi4.core.Matrix.from_array(rng.rand(primary.nbf() - 2, 1)))

    builds_before = gtfock_mpi.fock_builds()
    with pytest.raises(RuntimeError, match="density matrix"):
        jk.compute()
    # The refusal has to come before GTFock is handed the mis-sized block.
    assert gtfock_mpi.fock_builds() == builds_before


def _oversubscribe_flag(mpirun):
    """``--oversubscribe`` is an Open MPI spelling; MPICH's mpiexec rejects it.

    Both implementations install a program called ``mpirun``, so ask the
    launcher on PATH whether it accepts the flag rather than guessing from its
    name. A test box usually has fewer cores than the ranks below ask for, and
    Open MPI refuses to place them without it.
    """
    try:
        probe = subprocess.run([mpirun, "--oversubscribe", "-n", "1", sys.executable, "-c", ""],
                               capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return []
    return ["--oversubscribe"] if probe.returncode == 0 else []


@uusing("gtfock")
@pytest.mark.parametrize("nranks", [2])
def test_gtfock_multirank_mpirun(tmp_path, nranks):
    """Run the whole Python -> mpi4py -> Psi4 -> GTFock path under mpirun.

    Asserts, per rank: mpi4py and Psi4's linked MPI agree on rank/size; GTFock
    handed this rank a distinct slice of the AO matrix; libfock really built a
    GTFockJK; GTFock ran at least one Fock build; and the energy matches Psi4's
    own single-process RHF.
    """
    mpirun = shutil.which("mpirun") or shutil.which("mpiexec")
    if mpirun is None:
        pytest.skip("no mpirun/mpiexec on PATH")
    pytest.importorskip("mpi4py")

    launch = [mpirun] + _oversubscribe_flag(mpirun)

    driver = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gtfock_mpi_driver.py")

    # Reference from a plain single-process Psi4, computed with Psi4's own PK
    # integrals, so the comparison is against Psi4 rather than against GTFock.
    _water()
    psi4.set_options({"basis": "sto-3g", "puream": False, "scf_type": "pk",
                      "df_scf_guess": False, "guess": "core",
                      "e_convergence": 1e-10, "d_convergence": 1e-9})
    e_ref = psi4.energy("scf")

    env = dict(os.environ)
    # One OpenMP thread per rank: the ranks are oversubscribed on a test box and
    # each must reach GTFock's collective calls in the same order.
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"

    proc = subprocess.run(
        launch + ["-n", str(nranks), sys.executable, driver, str(tmp_path)],
        capture_output=True, text=True, env=env, timeout=900)
    assert proc.returncode == 0, f"mpirun failed:\n{proc.stdout}\n{proc.stderr}"

    reports = [json.loads(line.split(" ", 1)[1])
               for line in proc.stdout.splitlines() if line.startswith("PSI4-GTFOCK-JSON ")]
    assert len(reports) == nranks, f"expected {nranks} rank reports, got {len(reports)}:\n{proc.stdout}"

    seen_ranks = set()
    seen_blocks = set()
    for report in reports:
        rank = report["mpi4py_rank"]
        seen_ranks.add(rank)
        assert report["mpi4py_size"] == nranks
        # Python's MPI and the MPI that GTFock is linked against are the same one.
        assert report["core_mpi"] == {"rank": rank, "size": nranks, "initialized": True}
        # libfock built the GTFock engine, not a fallback.
        assert report["jk_name"] == "GTFockJK"
        assert report["fock_builds"] > 0
        assert report["process_grid"] != [-1, -1]
        seen_blocks.add(tuple(report["local_block"]))
        assert abs(report["scf_energy"] - e_ref) < ENERGY_TOL

    assert seen_ranks == set(range(nranks))
    # GTFock partitioned the AO matrix instead of replicating it on every rank.
    assert len(seen_blocks) == nranks, f"ranks share an AO block: {seen_blocks}"
