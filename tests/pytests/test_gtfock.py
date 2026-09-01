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

The ``test_gtfock_df_*`` tests cover the second, separate engine: ``SCF_TYPE
GTFOCK_DF``, which distributes a fitted three-index tensor by auxiliary function
instead of distributing four-centre integrals. It has its own add-on marker,
``uusing("gtfock_df")``, because a GTFock install can predate ``libgtfockdf`` and
lack it entirely. Unlike the exact path it needs no subprocess isolation: a
``PDF_t`` keeps no file-scope state, so a process may hold several engines at
once and may destroy them, which ``test_gtfock_df_engine_is_reentrant`` pins
down.
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
# GTFock's density fitting against Psi4's own MemDFJK, given the *same* fitting
# basis. This is a much tighter comparison than the exact-path one above: both
# sides expand in the same auxiliary space, and J and K are invariant to the
# rotation within it that distinguishes GTFock's eigendecomposition of the
# Coulomb metric from DFHelper's, so only the three-centre integrals and the
# contraction order differ. Measured on water/cc-pVDZ with cc-pVDZ-JKFIT:
# max|dJ| ~ 1e-12, max|dK| ~ 2e-13, dE ~ 5e-12 Eh. 1e-9 keeps three orders of
# headroom over the largest of those while still catching a real disagreement.
DF_JK_TOL = 1.0e-9
DF_ENERGY_TOL = 1.0e-9
# The DF orbital/fitting pair the DF tests use. Both must be Cartesian, so the
# tests build them with puream=False; cc-pVDZ-JKFIT tops out at f (l = 3), well
# inside GTFDF_maxSupportedAM().
DF_BASIS = "cc-pVDZ"
DF_FITTING_BASIS = "cc-pVDZ-JKFIT"

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


# --------------------------------------------------------------------------
# SCF_TYPE GTFOCK_DF: the distributed density-fitted engine.
#
# A separate engine from the exact path above, and separately optional: a
# GTFock install can predate libgtfockdf, so these carry their own add-on
# marker. They also need no subprocess isolation -- see the module docstring.
# --------------------------------------------------------------------------


def _df_bases(orbital=DF_BASIS, fitting=DF_FITTING_BASIS, mol=None):
    """A Cartesian orbital/fitting pair on water, which is what the DF engine takes."""
    mol = _water() if mol is None else mol
    primary = psi4.core.BasisSet.build(mol, "ORBITAL", orbital, puream=False)
    auxiliary = psi4.core.BasisSet.build(mol, "DF_BASIS_SCF", fitting, puream=False)
    assert not primary.has_puream() and not auxiliary.has_puream()
    return primary, auxiliary


def _df_jk(primary, auxiliary, scf_type):
    """A built, initialized JK of the requested type over this basis pair."""
    psi4.set_options({"scf_type": scf_type})
    jk = psi4.core.JK.build_JK(primary, auxiliary)
    jk.set_do_K(True)
    jk.initialize()
    return jk


def test_gtfock_df_is_optional():
    """A build without the DF engine must say so, and must not need MPI to say it.

    The sibling of ``test_gtfock_is_optional``, and separate from it on purpose:
    ``libgtfockdf`` is optional *within* ``ENABLE_GTFock``, so the two flags can
    legitimately disagree and each needs its own guard.
    """
    enabled = psi4.core.gtfock_df_enabled()
    assert enabled == psi4.addons("gtfock_df"), "psi4.addons and core disagree about GTFock DF"

    from psi4.driver import gtfock

    assert gtfock.df_available() == enabled
    # The DF engine cannot exist without the GTFock it is built on top of.
    assert not enabled or psi4.core.gtfock_enabled()
    if not enabled:
        # Reporting must work without MPI ever having been initialized.
        assert gtfock.df_jk_builds() == 0
        assert gtfock.df_partition() == {"nbf": -1, "naux": -1, "nlocal_aux": -1,
                                         "nmetric_null": -1, "nlocal_pairs": -1,
                                         "local_tensor_doubles": 0}


@uusing("gtfock_df")
def test_gtfock_df_jk_matches_memdfjk(gtfock_mpi):
    """J and K from GTFock's DF engine must match Psi4's MemDFJK on the same fitting basis.

    Given one auxiliary basis the two sides are expanding in the same space, so
    this is a far tighter comparison than the exact path's Simint-vs-Libint2 one:
    the rotation within the auxiliary space that separates GTFock's
    eigendecomposition of ``(P|Q)`` from DFHelper's cancels out of both J and K.
    What is left is three-centre integral accuracy and contraction order, and
    that is what ``DF_JK_TOL`` is sized for.
    """
    primary, auxiliary = _df_bases()

    # A converged density, so the comparison runs on the matrix an SCF actually
    # feeds the engine rather than on a random one.
    psi4.set_options({"basis": DF_BASIS, "puream": False, "df_basis_scf": DF_FITTING_BASIS,
                      "scf_type": "mem_df", "df_scf_guess": False, "guess": "core",
                      "e_convergence": 1e-10, "d_convergence": 1e-9})
    _, wfn = psi4.energy("scf", return_wfn=True)
    Cocc = wfn.Ca_subset("AO", "OCC")

    ref = _df_jk(primary, auxiliary, "mem_df")
    ref.C_clear()
    ref.C_left_add(Cocc)
    ref.compute()

    builds_before = gtfock_mpi.df_jk_builds()
    gt = _df_jk(primary, auxiliary, "gtfock_df")
    # Building the fitted tensor is setup, not a J/K build.
    assert gtfock_mpi.df_jk_builds() == builds_before
    gt.C_clear()
    gt.C_left_add(Cocc)
    gt.compute()

    # Guard against a silent fallback to Psi4's own DF.
    assert ref.name() == "MemDFJK"
    assert gt.name() == "GTFockDFJK"
    assert gtfock_mpi.df_jk_builds() == builds_before + 1

    partition = gtfock_mpi.df_partition()
    assert partition["nbf"] == primary.nbf()
    assert partition["naux"] == auxiliary.nbf()

    assert np.max(np.abs(np.array(gt.J()[0]) - np.array(ref.J()[0]))) < DF_JK_TOL
    assert np.max(np.abs(np.array(gt.K()[0]) - np.array(ref.K()[0]))) < DF_JK_TOL


@uusing("gtfock_df")
def test_gtfock_df_rhf_energy_matches_mem_df(gtfock_mpi):
    """A full RHF driven by the DF engine must land on Psi4's own DF energy.

    ``mem_df`` rather than ``pk`` is the right reference: both sides are making
    the same fitting approximation in the same auxiliary basis, so any
    disagreement here is an implementation difference and not the DF error.
    """
    _water()
    common = {"basis": DF_BASIS, "puream": False, "df_basis_scf": DF_FITTING_BASIS,
              "df_scf_guess": False, "guess": "core", "e_convergence": 1e-10,
              "d_convergence": 1e-9, "save_jk": True}

    psi4.set_options({**common, "scf_type": "mem_df"})
    e_ref = psi4.energy("scf")

    builds_before = gtfock_mpi.df_jk_builds()
    psi4.set_options({**common, "scf_type": "gtfock_df"})
    e_gtfock, wfn = psi4.energy("scf", return_wfn=True)

    assert wfn.jk().name() == "GTFockDFJK"
    assert gtfock_mpi.df_jk_builds() > builds_before, "the GTFock DF engine never ran"
    # proc.py has to have put a real fitting basis on the wavefunction; the
    # zero basis it hands the exact path would leave the engine with naux == 0.
    assert wfn.get_basisset("DF_BASIS_SCF").nbf() > 0
    assert abs(e_gtfock - e_ref) < DF_ENERGY_TOL


@uusing("gtfock_df")
def test_gtfock_df_uhf_energy_matches_mem_df(gtfock_mpi):
    """An open-shell UHF must match too, which is the multiple-density path.

    ``PDF_computeJK`` takes one density at a time, so ``GTFockDFJK::compute_JK``
    loops over ``D_ao_``. RHF has one entry and would not notice the loop being
    wrong; UHF has two, and a loop that reused the alpha density or overwrote
    the alpha J would show up here.

    The doublet also drives the ``nocc == 0`` guard on nothing, so
    ``test_gtfock_df_handles_an_empty_occupied_block`` covers that separately.
    """
    psi4.geometry("""
0 2
O   0.000000000000   0.000000000000  -0.068516219320
H   0.000000000000   0.790689573744   0.543701060715
units angstrom
symmetry c1
no_reorient
no_com
""")
    common = {"basis": DF_BASIS, "puream": False, "df_basis_scf": DF_FITTING_BASIS,
              "reference": "uhf", "df_scf_guess": False, "guess": "core",
              "e_convergence": 1e-10, "d_convergence": 1e-9, "save_jk": True}

    psi4.set_options({**common, "scf_type": "mem_df"})
    e_ref = psi4.energy("scf")

    builds_before = gtfock_mpi.df_jk_builds()
    psi4.set_options({**common, "scf_type": "gtfock_df"})
    e_gtfock, wfn = psi4.energy("scf", return_wfn=True)

    assert wfn.jk().name() == "GTFockDFJK"
    # Two densities per iteration, so at least two builds per SCF cycle.
    assert gtfock_mpi.df_jk_builds() >= builds_before + 2
    assert abs(e_gtfock - e_ref) < DF_ENERGY_TOL


@uusing("gtfock_df")
def test_gtfock_df_handles_an_empty_occupied_block(gtfock_mpi):
    """A spin with no electrons must give K = 0, not a degenerate DGEMM.

    A hydrogen atom UHF has ``nbeta == 0``, so the beta build reaches
    ``PDF_computeJK`` with an ``nocc == 0`` ``Cocc``. ``GTFockDFJK`` drops K for
    that build and leaves the zeroed matrix in place rather than handing GTFock a
    zero-column orbital block. The energy is the check that it dropped only K:
    the beta *Coulomb* term is still needed and is not zero.
    """
    psi4.geometry("""
0 2
H   0.0   0.0   0.0
units angstrom
symmetry c1
no_reorient
no_com
""")
    common = {"basis": DF_BASIS, "puream": False, "df_basis_scf": DF_FITTING_BASIS,
              "reference": "uhf", "df_scf_guess": False, "guess": "core",
              "e_convergence": 1e-10, "d_convergence": 1e-9, "save_jk": True}

    psi4.set_options({**common, "scf_type": "mem_df"})
    e_ref = psi4.energy("scf")

    psi4.set_options({**common, "scf_type": "gtfock_df"})
    e_gtfock, wfn = psi4.energy("scf", return_wfn=True)

    assert wfn.jk().name() == "GTFockDFJK"
    assert wfn.nbetapi().sum() == 0, "this case is only meaningful with an empty beta block"
    assert abs(e_gtfock - e_ref) < DF_ENERGY_TOL


@uusing("gtfock_df")
def test_gtfock_df_builds_its_engine_in_initialize(gtfock_mpi):
    """The fitted tensor must be built by ``initialize()``, not lazily on first ``compute()``.

    This is the whole reason ``GTFockDFJK`` differs from ``GTFockJK`` in where it
    creates its engine, and it is a timing claim as much as a correctness one.
    Psi4 runs ``preiterations()`` from ``JK::initialize()``, *outside* the
    ``"JK: JK"`` timer that wraps ``compute()``. Building the tensor there is
    what puts GTFock's DF setup cost in the same place ``MemDFJK`` puts its own,
    so the two engines' ``JK: JK`` numbers mean the same thing and can be
    compared. An engine built lazily inside the first ``compute()`` would land
    that one-off cost inside the timer and make the first SCF iteration look
    enormous and every later one artificially cheap.

    ``df_partition()`` reports the last engine any rank created, so the
    pre-``initialize()`` assertion is that ``build_JK`` changed nothing -- an
    earlier test in the same process may already have left a partition behind.
    """
    primary, auxiliary = _df_bases()
    before = gtfock_mpi.df_partition()
    builds_before = gtfock_mpi.df_jk_builds()

    psi4.set_options({"scf_type": "gtfock_df"})
    jk = psi4.core.JK.build_JK(primary, auxiliary)
    jk.set_do_K(True)
    # Constructing the JK is not what creates the engine.
    assert gtfock_mpi.df_partition() == before

    jk.initialize()
    after = gtfock_mpi.df_partition()
    assert after["nbf"] == primary.nbf()
    assert after["naux"] == auxiliary.nbf()
    assert after["local_tensor_doubles"] > 0, "initialize() did not build a fitted tensor"
    # ... and it did so without computing any J or K.
    assert gtfock_mpi.df_jk_builds() == builds_before


@uusing("gtfock_df")
def test_gtfock_df_engine_is_reentrant(gtfock_mpi):
    """Two DF engines may be alive at once, and a released one may be rebuilt.

    The exact path cannot do either: ``fock_task.c`` keeps its state in
    file-scope globals, so a process gets one ``PFock_t`` for its lifetime and
    the shim's singleton is deliberately never destroyed. ``PDF_t`` has no such
    globals, which is what lets ``GTFockDFJK`` own its engine outright, build it
    in ``preiterations()`` and drop it in ``postiterations()``. If that ever
    stopped being true this test would deadlock or corrupt the second engine
    rather than fail quietly, so it is worth pinning.
    """
    primary, auxiliary = _df_bases()
    small, small_aux = _df_bases(orbital="6-31G")

    first = _df_jk(primary, auxiliary, "gtfock_df")
    second = _df_jk(small, small_aux, "gtfock_df")
    # Two live engines over different basis pairs.
    assert gtfock_mpi.df_partition()["nbf"] == small.nbf()
    assert small.nbf() != primary.nbf()

    rng = np.random.RandomState(11)
    for jk, basis in ((first, primary), (second, small)):
        jk.C_clear()
        jk.C_left_add(psi4.core.Matrix.from_array(rng.rand(basis.nbf(), 2)))
        jk.compute()
        assert np.max(np.abs(np.array(jk.J()[0]))) > 0.0

    # postiterations() releases the engine, and initialize() may build another.
    first.finalize()
    with pytest.raises(RuntimeError, match="before initialize"):
        first.compute()
    # That refusal also has to leave the object reusable, which is a claim about
    # JK::compute() rather than about this engine: it aliases C_right_ to C_left_
    # and opens the "JK: JK" timer around compute_JK, and until both were made to
    # unwind, a refused build turned the next one into a bogus "non-symmetric
    # densities" error and the timer left at exit deadlocked the interpreter.
    first.initialize()
    first.compute()
    assert gtfock_mpi.df_partition()["nbf"] == primary.nbf()


@uusing("gtfock_df")
@pytest.mark.parametrize("spherical", ["orbital", "fitting"])
def test_gtfock_df_refuses_spherical_basis(gtfock_mpi, spherical):
    """Both bases must be Cartesian, and each must be screened on its own.

    The DF engine drives Simint over a *pair* of basis sets, so there are two
    places a spherical basis can enter. Psi4's ``PUREAM`` normally applies to
    both at once, which would hide a check that only looked at one of them;
    these two cases build the pair by hand so exactly one side is spherical.
    """
    mol = _water()
    puream = {"orbital": spherical == "orbital", "fitting": spherical == "fitting"}
    primary = psi4.core.BasisSet.build(mol, "ORBITAL", DF_BASIS, puream=puream["orbital"])
    auxiliary = psi4.core.BasisSet.build(mol, "DF_BASIS_SCF", DF_FITTING_BASIS,
                                         puream=puream["fitting"])
    assert primary.has_puream() != auxiliary.has_puream()

    psi4.set_options({"scf_type": "gtfock_df"})
    jk = psi4.core.JK.build_JK(primary, auxiliary)
    jk.set_do_K(True)

    # The refusal is at engine creation, so it comes out of initialize().
    with pytest.raises(RuntimeError, match="spherical"):
        jk.initialize()


@uusing("gtfock_df")
def test_gtfock_df_refuses_high_angular_momentum(gtfock_mpi):
    """A shell above Simint's generated ceiling must raise, not read past its tables.

    ``GTFDF_maxSupportedAM()`` reports ``SIMINT_OSTEI_MAXAM`` from the linked
    Simint. Simint dispatches through generated tables indexed by angular
    momentum with no bound check, so a shell above it reads past them rather
    than failing. Cartesian ``cc-pV6Z`` puts ``i`` functions (``l = 6``) on
    oxygen, one above the ceiling this build was generated for.

    Note this ceiling is *not* the exact path's: libcint hardcodes its own,
    lower ``_SIMINT_OSTEI_MAXAM``, which is why ``cc-pV5Z`` is enough to trip
    ``test_gtfock_refuses_high_angular_momentum`` and is not enough here.
    """
    mol = _water()
    primary = psi4.core.BasisSet.build(mol, "ORBITAL", "cc-pV6Z", puream=False)
    auxiliary = psi4.core.BasisSet.build(mol, "DF_BASIS_SCF", DF_FITTING_BASIS, puream=False)
    assert max(primary.shell(sh).am for sh in range(primary.nshell())) >= 6

    psi4.set_options({"scf_type": "gtfock_df"})
    jk = psi4.core.JK.build_JK(primary, auxiliary)
    jk.set_do_K(True)

    with pytest.raises(RuntimeError, match="angular momentum"):
        jk.initialize()


@uusing("gtfock_df")
def test_gtfock_df_refuses_wk(gtfock_mpi):
    """GTFock fits the plain Coulomb operator, so range separation must be refused."""
    primary, auxiliary = _df_bases()
    psi4.set_options({"scf_type": "gtfock_df"})
    jk = psi4.core.JK.build_JK(primary, auxiliary)
    jk.set_do_K(True)
    jk.set_do_wK(True)
    jk.set_omega(0.3)
    jk.initialize()

    rng = np.random.RandomState(5)
    jk.C_clear()
    jk.C_left_add(psi4.core.Matrix.from_array(rng.rand(primary.nbf(), 1)))

    builds_before = gtfock_mpi.df_jk_builds()
    with pytest.raises(RuntimeError, match="range-separated"):
        jk.compute()
    assert gtfock_mpi.df_jk_builds() == builds_before


@uusing("gtfock_df")
def test_gtfock_df_refuses_nonsymmetric_density(gtfock_mpi):
    """K needs ``D = Cocc Cocc^T``, so ``C_left != C_right`` must raise.

    ``PDF_computeJK`` contracts ``B`` against a single occupied block for K,
    which is only the exchange matrix when the density factorizes that way. A
    response-style build with two different coefficient blocks does not, and
    would come back a plausible-looking wrong K rather than an error.

    J is unaffected -- it needs only the density -- but the guard fires on the
    whole build, because a caller asking for both would silently get one right
    matrix and one wrong one.
    """
    primary, auxiliary = _df_bases()
    jk = _df_jk(primary, auxiliary, "gtfock_df")

    rng = np.random.RandomState(6)
    jk.C_clear()
    jk.C_left_add(psi4.core.Matrix.from_array(rng.rand(primary.nbf(), 2)))
    jk.C_right_add(psi4.core.Matrix.from_array(rng.rand(primary.nbf(), 2)))

    builds_before = gtfock_mpi.df_jk_builds()
    with pytest.raises(RuntimeError, match="non-symmetric"):
        jk.compute()
    assert gtfock_mpi.df_jk_builds() == builds_before


@uusing("gtfock_df")
def test_gtfock_df_computes_j_without_k(gtfock_mpi):
    """A J-only build must skip K entirely and still match MemDFJK.

    ``PDF_computeJK`` takes NULL for either output, and a J-only caller never
    supplies an occupied block -- so this is the path where asking GTFock for
    ``Cocc`` anyway would dereference nothing. It is also the path a
    Coulomb-only method such as a pure functional's ``form_G`` takes.
    """
    primary, auxiliary = _df_bases()
    psi4.set_options({"basis": DF_BASIS, "puream": False, "df_basis_scf": DF_FITTING_BASIS,
                      "scf_type": "mem_df", "df_scf_guess": False, "guess": "core",
                      "e_convergence": 1e-10, "d_convergence": 1e-9})
    _, wfn = psi4.energy("scf", return_wfn=True)
    Cocc = wfn.Ca_subset("AO", "OCC")

    matrices = {}
    for scf_type in ("mem_df", "gtfock_df"):
        psi4.set_options({"scf_type": scf_type})
        jk = psi4.core.JK.build_JK(primary, auxiliary)
        jk.set_do_J(True)
        jk.set_do_K(False)
        jk.initialize()
        jk.C_clear()
        jk.C_left_add(Cocc)
        jk.compute()
        matrices[scf_type] = np.array(jk.J()[0])

    assert np.max(np.abs(matrices["gtfock_df"] - matrices["mem_df"])) < DF_JK_TOL


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

    Three ranks are in the sweep because a prime rank count is the one case
    ``split_procs()`` cannot factor squarely: it starts from
    ``floor(sqrt(nprocs))`` and decrements until the row count divides, which for
    a prime means falling all the way to a 1xN grid. That is a different branch
    from the 2x2 case and a plausible place to refuse a rank count outright, so
    the grid assertion below pins it to 1x3 rather than leaving it to be
    discovered by a user with three nodes.
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
               for nranks in (1, 2, 3, 4)}

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
    assert results[3][0]["process_grid"] == [1, 3]
    assert results[2][0]["process_grid"] == [1, 2]
    # The distributed claim: every rank of every rank count agrees with every
    # other to roundoff, so nothing about the decomposition leaked into the
    # energy. This is the tight tolerance.
    assert max(energies) - min(energies) < RANK_INVARIANCE_TOL, (
        f"{method} energy depends on the rank count: spread "
        f"{max(energies) - min(energies):.3e} Eh over {energies}")


@uusing("gtfock_df")
def test_gtfock_df_rank_count_invariance(tmp_path):
    """The DF engine's energy and its partition must both add up at every rank count.

    The distributed-correctness evidence for ``SCF_TYPE GTFOCK_DF``, and it
    carries two claims the exact path's sweep cannot.

    The first is the energy, held to ``RANK_INVARIANCE_TOL``. The fitted tensor
    is cut by auxiliary function and each rank contracts only its own slice, so
    J and K are completed by an ``MPI_Allreduce``; the rank count changes which
    partial sums land on which rank and in what order, and nothing else. An
    energy that moved with the rank count would mean a slice was dropped or
    double-counted.

    The second is the partition itself, and it is exact rather than approximate:
    the auxiliary functions are dealt out in one contiguous block per rank, so
    ``nlocal_aux`` must sum to ``naux`` with nothing left over and nothing
    counted twice. The same holds for ``nlocal_pairs`` over the AO-element
    partition used before the redistribution, and for the tensor slices, which
    must sum to the whole ``naux x npair`` tensor. A rank-count-dependent
    off-by-one in either partition is invisible in the energy -- the missing
    auxiliary function's contribution is small -- but shows up here immediately.

    ``nmetric_null`` is checked to be identical on every rank because the metric
    is inverted redundantly: every rank eigendecomposes the same replicated
    ``(P|Q)`` so that they agree bit for bit and need no further communication
    to stay in step. A rank that dropped a different number of vectors than its
    peers would be contracting against a different fitting space.
    """
    launch = _mpi_launch()

    driver_args = ["--molecule", "water6", "--scf-type", "gtfock_df",
                   "--basis", DF_BASIS, "--df-basis", DF_FITTING_BASIS]
    results = {nranks: _run_mpi_driver(launch, nranks,
                                       os.path.join(str(tmp_path), f"n{nranks}"), driver_args)
               for nranks in (1, 2, 3, 4)}

    # The one-rank run is the undistributed baseline every partition sum is
    # compared against, so read the totals off it rather than hardcoding them.
    whole = results[1][0]["df_partition"]
    assert whole["nlocal_aux"] == whole["naux"]

    energies = []
    for nranks, reports in results.items():
        for report in reports:
            rank = report["mpi4py_rank"]
            assert report["mpi4py_size"] == nranks
            assert report["core_mpi"] == {"rank": rank, "size": nranks, "initialized": True}
            # libfock built the DF engine, not a fallback and not the exact path.
            assert report["jk_name"] == "GTFockDFJK", f"n={nranks} rank {rank} used {report['jk_name']}"
            assert report["fock_builds"] > 0, f"n={nranks} rank {rank} ran no DF build"
            assert report["df_partition"]["naux"] == whole["naux"]
            assert report["df_partition"]["nbf"] == whole["nbf"]
            # Every rank inverted the same metric and kept the same fitting space.
            assert report["df_partition"]["nmetric_null"] == whole["nmetric_null"], (
                f"n={nranks} rank {rank} dropped a different number of metric vectors")
            energies.append(report["scf_energy"])

        partitions = [report["df_partition"] for report in reports]
        # The auxiliary partition covers naux exactly once.
        assert sum(part["nlocal_aux"] for part in partitions) == whole["naux"], (
            f"n={nranks}: auxiliary functions do not partition naux")
        # ... and so does the AO-element partition the three-centre integrals use.
        assert sum(part["nlocal_pairs"] for part in partitions) == whole["nlocal_pairs"], (
            f"n={nranks}: AO pair elements do not partition the whole set")
        # ... and the slices add up to the whole fitted tensor.
        assert (sum(part["local_tensor_doubles"] for part in partitions)
                == whole["local_tensor_doubles"]), f"n={nranks}: tensor slices do not tile"
        if nranks > 1:
            # The tensor really was cut up rather than replicated on every rank.
            assert max(part["local_tensor_doubles"] for part in partitions) \
                < whole["local_tensor_doubles"], f"n={nranks}: a rank holds the whole tensor"

    assert max(energies) - min(energies) < RANK_INVARIANCE_TOL, (
        f"the DF energy depends on the rank count: spread "
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


def test_hpc_collector_refuses_a_truncated_point(tmp_path):
    """A job killed at its wall limit leaves fewer rank files than it declares.

    The survivors carry one job id and one nodelist and no rank repeats, so the
    run-identity and duplicate guards both pass; only the completeness check
    stops the row, whose node memory would otherwise be summed over half the
    ranks the ``ranks`` column claims.
    """
    import gtfock_hpc_collect

    run = _write_hpc_records(tmp_path / "peptide_12400108",
                             [_hpc_record(0, ranks=4), _hpc_record(1, ranks=4)])

    with pytest.raises(SystemExit, match="is incomplete"):
        gtfock_hpc_collect.load_points([run])
