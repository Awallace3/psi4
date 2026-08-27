"""End-to-end tests for the optional GTFock MPI J/K engine.

The GTFock path is opt-in: it exists only when Psi4 was configured with
``-DENABLE_GTFock=ON``. ``test_gtfock_is_optional`` runs everywhere and asserts
the default build reports no GTFock and needs no MPI; everything else skips
through the standard ``uusing("gtfock")`` add-on marker.

Every test here drives the linked library rather than a header or a stub: they
all go through ``psi4.core``/``JK`` into ``libgtfock.so``. The tests that need a
GTFock engine of their own do so in a subprocess, because a process gets exactly
one engine; the multi-rank tests launch ``mpirun`` on the installed Psi4 and read
back what each rank's own GTFock engine reported.

``test_gtfock_rank_count_invariance`` is the rank-count evidence: it sweeps one,
two and four ranks on a water hexamer and compares every rank's energy against a
single-process Psi4 PK reference. ``tests/pytests/gtfock_benchmark.py`` runs the
same sweep with timings; it is a script rather than a test because it takes
minutes.
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
# This is the cross-engine tolerance for a single compact molecule only; see
# CLUSTER_ENGINE_TOL for why an extended system needs a looser one.
ENERGY_TOL = 1.0e-9
# Agreement between GTFock and Psi4's own integrals is *not* uniform: it is
# roundoff-level for a single compact molecule in any of sto-3g, 6-31G or 6-31G*
# (measured ~1e-14 Eh), and degrades to a few times 1e-6 Eh for a six-water
# cluster in 6-31G/6-31G*. GTFock's integral layer prints its own screening
# settings at engine creation ("Screen method: 2 / Screen tol: 1.0e-14") and
# prunes Simint primitive pairs at that fixed tolerance, which Psi4's
# INTS_TOLERANCE does not reach: sweeping INTS_TOLERANCE from 1e-8 to 1e-16 moves
# the cluster energy by under 1e-7 Eh. The truncation is therefore per primitive
# pair and accumulates with the number of well-separated centres, which is why it
# is invisible on one water and visible on six. That is an accuracy property of
# the GTFock stack, present at one rank, and nothing to do with distribution --
# so the cross-engine comparison on the cluster gets its own stated tolerance,
# while rank-count invariance keeps a tight one.
CLUSTER_ENGINE_TOL = 1.0e-5
# GTFock against GTFock at a different rank count. The gather-and-broadcast SCF
# is replicated, so this should be reordered-summation noise and nothing more;
# measured spread is ~1e-12 Eh at 390 basis functions. This is the tolerance that
# carries the distributed-correctness claim, so it stays tight.
RANK_INVARIANCE_TOL = 1.0e-9

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


# The rank-count sweep needs a system GTFock can actually decompose: at four
# ranks it wants a 2x2 process grid whose per-rank AO panel is still large enough
# for GTFock to split into several task blocks. Cartesian 6-31G* on the water
# hexamer gives 60 shells and 114 basis functions, so each of the four ranks owns
# a strict sub-block of that matrix, itself split into more than one task block in
# each dimension. Water/STO-3G, the smoke case above, has 5 shells and would hand
# each rank a single degenerate block.
HEXAMER_BASIS = "6-31G*"


def _hexamer():
    """The MPI driver's own water hexamer.

    Taken from the driver rather than copied, so the single-process reference and
    the GTFock ranks cannot drift onto different geometries.
    """
    from gtfock_mpi_driver import _MOLECULES

    return psi4.geometry(_MOLECULES["water6"])


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
        assert gtfock.decomposition() == {"grid": [-1, -1], "block": [-1, -1, -1, -1],
                                          "task_shape": [-1, -1, -1]}
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
def test_gtfock_hybrid_dft_energy_matches_reference(gtfock_mpi):
    """A hybrid-DFT SCF driven by GTFock must land on Psi4's own energy.

    GTFock returns plain J and K; the hybrid's exchange fraction is applied
    afterwards by ``RHF::form_G`` (``G = J - alpha*K + V_xc``), which is where
    every J/K engine gets scaled. So the interesting question is not whether the
    scaling code runs but whether the DFT entry point in ``scf_iterator.py``
    hands GTFock a task it supports: ``set_do_K(is_x_hybrid())`` is true and
    ``set_do_wK(is_x_lrc())`` is false for B3LYP, which is exactly the one shape
    GTFock can answer. This pins that, and pins that the exchange really was
    scaled -- an ``x_alpha`` of zero would make the test pass vacuously.
    """
    _water()
    common = {"basis": "sto-3g", "puream": False, "df_scf_guess": False,
              "guess": "core", "e_convergence": 1e-10, "d_convergence": 1e-9,
              "save_jk": True}

    psi4.set_options({**common, "scf_type": "pk"})
    e_ref, ref_wfn = psi4.energy("b3lyp", return_wfn=True)
    # B3LYP is a hybrid and is not range-separated: K yes, wK no.
    assert ref_wfn.functional().is_x_hybrid()
    assert not ref_wfn.functional().is_x_lrc()
    assert ref_wfn.functional().x_alpha() > 0.0

    builds_before = gtfock_mpi.fock_builds()
    psi4.set_options({**common, "scf_type": "gtfock"})
    e_gtfock, wfn = psi4.energy("b3lyp", return_wfn=True)

    assert wfn.jk().name() == "GTFockJK"
    # K yes, wK no: JK only allocates the matrices it was asked for, so these
    # two show which half of the exchange the DFT entry point requested.
    assert len(wfn.jk().K()) == 1, "the hybrid did not ask GTFock for exchange"
    assert len(wfn.jk().wK()) == 0, "a wK matrix appeared for a global hybrid"
    assert gtfock_mpi.fock_builds() > builds_before, "GTFock never ran"
    assert abs(e_gtfock - e_ref) < ENERGY_TOL


@uusing("gtfock")
@pytest.mark.parametrize("functional", ["wb97x", "cam-b3lyp"])
def test_gtfock_refuses_range_separated_functionals(gtfock_mpi, functional):
    """Range-separated functionals need wK, which GTFock cannot produce.

    Psi4's superfunctional builder refuses any ``SCF_TYPE`` outside its
    wK-capable list, so the refusal lands before a JK is even constructed and
    names both the functional class and the offending ``SCF_TYPE``. Pinning it
    here is what keeps a future GTFock build from quietly returning a
    hybrid-shaped energy for a functional whose long-range exchange was silently
    dropped -- which is the failure mode that matters, since the number would
    look plausible.
    """
    _water()
    psi4.set_options({"basis": "sto-3g", "puream": False, "scf_type": "gtfock",
                      "df_scf_guess": False, "guess": "core"})

    builds_before = gtfock_mpi.fock_builds()
    with pytest.raises(psi4.ValidationError, match="range-separated"):
        psi4.energy(functional)
    assert gtfock_mpi.fock_builds() == builds_before, "GTFock ran before the refusal"


@uusing("gtfock")
def test_gtfock_refuses_wk_directly(gtfock_mpi):
    """The JK layer itself must refuse wK, for callers that skip the DFT driver.

    ``test_gtfock_refuses_range_separated_functionals`` covers the path Psi4's
    own DFT code takes. This covers the one a direct ``JK`` user takes, where no
    superfunctional exists to be validated: ``GTFockJK::compute_JK`` refuses
    before touching GTFock, so the refusal cannot be reached by way of a wrong
    number.
    """
    mol = _water()
    primary = psi4.core.BasisSet.build(mol, "ORBITAL", "sto-3g", puream=False)
    psi4.set_options({"scf_type": "gtfock"})
    jk = psi4.core.JK.build_JK(primary, None)
    jk.set_do_K(True)
    jk.set_do_wK(True)
    jk.set_omega(0.3)
    jk.initialize()

    rng = np.random.RandomState(4)
    jk.C_clear()
    jk.C_left_add(psi4.core.Matrix.from_array(rng.rand(primary.nbf(), 1)))

    builds_before = gtfock_mpi.fock_builds()
    with pytest.raises(RuntimeError, match="range-separated"):
        jk.compute()
    assert gtfock_mpi.fock_builds() == builds_before


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


def _mpi_launch():
    """The ``mpirun`` invocation prefix, or skip if this box has no MPI."""
    mpirun = shutil.which("mpirun") or shutil.which("mpiexec")
    if mpirun is None:
        pytest.skip("no mpirun/mpiexec on PATH")
    pytest.importorskip("mpi4py")
    return [mpirun] + _oversubscribe_flag(mpirun)


def _run_mpi_driver(launch, nranks, scratch, args=()):
    """Run ``gtfock_mpi_driver.py`` under mpirun and return one report per rank."""
    driver = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gtfock_mpi_driver.py")

    env = dict(os.environ)
    # One OpenMP thread per rank: the ranks may be oversubscribed on a test box
    # and each must reach GTFock's collective calls in the same order.
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"

    os.makedirs(scratch, exist_ok=True)
    proc = subprocess.run(
        launch + ["-n", str(nranks), sys.executable, driver, str(scratch), *args],
        capture_output=True, text=True, env=env, timeout=3600)
    assert proc.returncode == 0, f"mpirun -n {nranks} failed:\n{proc.stdout}\n{proc.stderr}"

    reports = [json.loads(line.split(" ", 1)[1])
               for line in proc.stdout.splitlines() if line.startswith("PSI4-GTFOCK-JSON ")]
    assert len(reports) == nranks, f"expected {nranks} rank reports, got {len(reports)}:\n{proc.stdout}"
    return sorted(reports, key=lambda report: report["mpi4py_rank"])


@uusing("gtfock")
@pytest.mark.parametrize("nranks", [2])
def test_gtfock_multirank_mpirun(tmp_path, nranks):
    """Run the whole Python -> mpi4py -> Psi4 -> GTFock path under mpirun.

    Asserts, per rank: mpi4py and Psi4's linked MPI agree on rank/size; GTFock
    handed this rank a distinct slice of the AO matrix; libfock really built a
    GTFockJK; GTFock ran at least one Fock build; and the energy matches Psi4's
    own single-process RHF.
    """
    launch = _mpi_launch()

    # Reference from a plain single-process Psi4, computed with Psi4's own PK
    # integrals, so the comparison is against Psi4 rather than against GTFock.
    _water()
    psi4.set_options({"basis": "sto-3g", "puream": False, "scf_type": "pk",
                      "df_scf_guess": False, "guess": "core",
                      "e_convergence": 1e-10, "d_convergence": 1e-9})
    e_ref = psi4.energy("scf")

    reports = _run_mpi_driver(launch, nranks, tmp_path)

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


@uusing("gtfock")
@pytest.mark.parametrize("method", ["scf", "b3lyp"])
def test_gtfock_rank_count_invariance(tmp_path, method):
    """The same SCF energy at one, two and four ranks, and Psi4's own answer.

    This is the distributed-correctness evidence. J and K are still gathered on
    rank 0 and broadcast, so in principle every rank runs an identical replicated
    SCF and the energy cannot depend on the rank count -- but the *integrals*
    behind those matrices are computed on a decomposition that does change with
    rank count, and GTFock screens shell quartets against a density-weighted
    bound per task block. So the answer is invariant only if the decomposition is
    correct; a mis-assigned block or a dropped task would show up here as an
    energy that moves with the rank count. That is the claim this test carries,
    and it is held to ``RANK_INVARIANCE_TOL``.

    Every rank is separately checked against a single-process Psi4 PK reference,
    so agreement between rank counts cannot be agreement on a shared error. That
    check is held to the looser ``CLUSTER_ENGINE_TOL``, because GTFock's integral
    layer and Psi4's do not agree to roundoff on a cluster this size for reasons
    that have nothing to do with rank count -- see the comment on that constant.
    The two tolerances are the point: the distributed claim is tight, the
    cross-engine claim is loose and says so.

    The four-rank case additionally has to show a 2x2 process grid, four distinct
    AO panels, and a per-rank panel that GTFock split into more than one block in
    each direction -- otherwise the sweep would be measuring a system too small
    to decompose. ``b3lyp`` covers the hybrid-DFT path through the same J/K.
    """
    launch = _mpi_launch()

    # Reference from a plain single-process Psi4 with its own PK integrals, so
    # the comparison is against Psi4 rather than against GTFock.
    _hexamer()
    psi4.set_options({"basis": HEXAMER_BASIS, "puream": False, "scf_type": "pk",
                      "df_scf_guess": False, "guess": "core",
                      "e_convergence": 1e-10, "d_convergence": 1e-9})
    e_ref = psi4.energy(method)

    driver_args = ["--molecule", "water6", "--basis", HEXAMER_BASIS, "--method", method]
    results = {nranks: _run_mpi_driver(launch, nranks, os.path.join(str(tmp_path), f"n{nranks}"),
                                      driver_args)
               for nranks in (1, 2, 4)}

    energies = []
    for nranks, reports in results.items():
        panels = set()
        for report in reports:
            rank = report["mpi4py_rank"]
            assert report["mpi4py_size"] == nranks
            # Python's MPI and the MPI GTFock is linked against are the same one.
            assert report["core_mpi"] == {"rank": rank, "size": nranks, "initialized": True}
            # libfock built the GTFock engine, not a fallback.
            assert report["jk_name"] == "GTFockJK", f"n={nranks} rank {rank} used {report['jk_name']}"
            assert report["fock_builds"] > 0, f"n={nranks} rank {rank} ran no GTFock build"
            assert report["method"] == method
            panels.add(tuple(report["local_block"]))
            energies.append(report["scf_energy"])
            assert abs(report["scf_energy"] - e_ref) < CLUSTER_ENGINE_TOL, (
                f"n={nranks} rank {rank}: {report['scf_energy']!r} vs PK {e_ref!r}")

        # GTFock partitioned the AO matrix instead of replicating it.
        assert len(panels) == nranks, f"n={nranks}: ranks share an AO panel: {panels}"
        # ... and each rank's own panel was itself blocked, so the run exercises
        # GTFock's task decomposition rather than one block per rank.
        nblks_row, nblks_col = reports[0]["local_task_shape"][:2]
        assert nblks_row > 1 and nblks_col > 1, (
            f"n={nranks}: one AO block per rank ({nblks_row}x{nblks_col}); "
            "this system is too small to prove anything about the decomposition")

    assert results[4][0]["process_grid"] == [2, 2]
    assert results[2][0]["process_grid"] == [1, 2]
    # The distributed claim: every rank of every rank count agrees with every
    # other to roundoff, so nothing about the decomposition leaked into the
    # energy. This is the tight tolerance.
    assert max(energies) - min(energies) < RANK_INVARIANCE_TOL, (
        f"{method} energy depends on the rank count: spread "
        f"{max(energies) - min(energies):.3e} Eh over {energies}")


def _hpc_record(rank, **overrides):
    """One ``gtfock_hpc_benchmark.py`` per-rank record, with the fields the reducer reads.

    The serialized record is that script's output contract: it is what lands in
    ``<json-out>.rank<N>.json`` and what the collector consumes.
    """
    record = {
        "system": "peptide", "arm": "gtfock", "basis": "6-31+G**", "method": "scf",
        "nbf": 260, "nshell": 128, "puream": False, "jk_name": "GTFockJK",
        "ranks": 2, "rank": rank, "threads_per_rank": 12, "total_cores": 24,
        "iterations": 11, "jk_calls": 11,
        "scf_energy": -757.5, "scf_wall_seconds": 100.0 + rank,
        "jk_wall_seconds": 50.0 + rank, "peak_rss_mb": 900.0 + 200.0 * rank,
        "host": "atl1-1-01-002-8-0", "slurm_job_id": "12400108",
        "slurm_nodelist": "atl1-1-01-002-8-0",
    }
    record.update(overrides)
    return record


def _write_hpc_records(directory, records):
    directory.mkdir(parents=True, exist_ok=True)
    for record in records:
        path = directory / f"peptide_gtfock_n{record['ranks']}.rank{record['rank']}.json"
        path.write_text(json.dumps(record))
    return str(directory)


def test_hpc_collector_reduces_one_run(tmp_path):
    """One run reduces to one row: wall clock is the slowest rank, memory the node.

    The two memory columns are the two the documentation publishes as
    ``RSS/rank`` and ``RSS node``, so both the worst rank and the sum over ranks
    are pinned here rather than only one of them.
    """
    import gtfock_hpc_collect

    run = _write_hpc_records(tmp_path / "peptide_12400108",
                             [_hpc_record(0), _hpc_record(1)])

    points = gtfock_hpc_collect.load_points([run])

    assert len(points) == 1
    assert points[0]["scf_wall_s"] == 101.0
    assert points[0]["jk_wall_s"] == 51.0
    assert points[0]["peak_rss_max_mb"] == 1100.0
    assert points[0]["peak_rss_sum_mb"] == 2000.0
    assert points[0]["slurm_job_id"] == "12400108"


def test_hpc_collector_refuses_records_from_two_jobs(tmp_path):
    """A repeated sweep must abort, not silently reduce two jobs into one row.

    The maxima and the summed memory would otherwise span unrelated hardware,
    and the documentation tables are generated from exactly these rows.
    """
    import gtfock_hpc_collect

    first = _write_hpc_records(tmp_path / "peptide_12395891",
                               [_hpc_record(0, slurm_job_id="12395891"),
                                _hpc_record(1, slurm_job_id="12395891")])
    second = _write_hpc_records(tmp_path / "peptide_12400108",
                                [_hpc_record(0), _hpc_record(1)])

    with pytest.raises(SystemExit, match="more than one run"):
        gtfock_hpc_collect.load_points([first, second])


def test_hpc_collector_refuses_the_same_directory_twice(tmp_path):
    """Passing one directory twice would double the node-memory column."""
    import gtfock_hpc_collect

    run = _write_hpc_records(tmp_path / "peptide_12400108",
                             [_hpc_record(0), _hpc_record(1)])

    with pytest.raises(SystemExit, match="more than one record for rank"):
        gtfock_hpc_collect.load_points([run, run])


def test_hpc_collector_reduces_one_multinode_run(tmp_path):
    """One job whose ranks sit on different nodes is one point, not two runs.

    Run identity is the job id and its nodelist; the per-rank ``host`` differs as
    soon as a launch spans more than one node, and that is a distributed point
    rather than a repeated sweep.
    """
    import gtfock_hpc_collect

    run = _write_hpc_records(
        tmp_path / "peptide_12400108",
        [_hpc_record(0, host="atl1-1-01-002-8-0",
                     slurm_nodelist="atl1-1-01-002-[8-9]-0"),
         _hpc_record(1, host="atl1-1-01-002-9-0",
                     slurm_nodelist="atl1-1-01-002-[8-9]-0")])

    points = gtfock_hpc_collect.load_points([run])

    assert len(points) == 1
    assert points[0]["scf_wall_s"] == 101.0
    assert points[0]["jk_wall_s"] == 51.0
    assert points[0]["peak_rss_max_mb"] == 1100.0
    assert points[0]["peak_rss_sum_mb"] == 2000.0
    assert points[0]["slurm_nodelist"] == "atl1-1-01-002-[8-9]-0"


def test_hpc_collector_refuses_two_runs_without_a_job_id(tmp_path):
    """Off SLURM there is no job id, so the result directory identifies the run.

    Two interactive attempts that each died after writing a different rank's file
    would otherwise union to a complete-looking {0, 1} under one identity, and
    the row's maximum wall clock and summed memory would span both attempts.
    """
    import gtfock_hpc_collect

    first = _write_hpc_records(
        tmp_path / "attempt_one",
        [_hpc_record(1, slurm_job_id=None, slurm_nodelist=None)])
    second = _write_hpc_records(
        tmp_path / "attempt_two",
        [_hpc_record(0, slurm_job_id=None, slurm_nodelist=None)])

    with pytest.raises(SystemExit, match="more than one run"):
        gtfock_hpc_collect.load_points([first, second])
