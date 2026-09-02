"""Run one point of the GTFock HPC rank-scaling benchmark and report per rank.

One invocation is one SCF: one system, one basis, one J/K algorithm, one rank
count. The orchestration -- which points to run and in which order -- lives in
the SLURM script (:source:`tests/pytests/gtfock_hpc_phoenix.slurm`), so this
script stays a single measurement that reports and does not judge::

    # GTFock, four ranks, six OpenMP threads each
    mpirun -n 4 python gtfock_hpc_benchmark.py --system peptide --arm gtfock \
        --threads 6 --memory "40 GB" --scratch runs/pep_gtfock_n4

    # the reference arm, one process with all the cores
    python gtfock_hpc_benchmark.py --system peptide --arm direct \
        --threads 24 --memory "160 GB" --scratch runs/pep_direct

Every process prints one ``PSI4-GTFOCK-JSON <json>`` line and, with
``--json-out``, writes the same record to a per-rank file for
:source:`tests/pytests/gtfock_hpc_collect.py` to gather. The record carries the
whole SCF wall clock, the J/K-build wall clock on its own, the iteration count,
the final energy and this process's peak RSS, so a differing iteration count or a
memory blow-up cannot hide inside a wall-clock ratio.

The two systems are dimers from the SAPT(DFT) + DFT-D4 timing set: a
twelve-atom-per-monomer peptide backbone pair and an ethene molecule on a carbon
nanotube fragment. They are run as single closed-shell molecules, not as SAPT --
this measures a Fock build, not an interaction energy.

The basis is 6-31+G**, which |PSIfour| ships as a Cartesian basis
(``psi4/share/psi4/basis/6-31pgss.gbs`` declares ``cartesian``), so it needs no
``puream`` override to run through GTFock's Cartesian-only Simint path. The
driver asserts that, since a silently spherical basis would be refused by the
engine at best and be a different computation at worst.
"""

import argparse
import json
import os
import platform
import socket
import sys
import time

import psi4

# Geometries transcribed from the SAPT(DFT) timing set (ids 144 and 154), in
# angstrom, C1 with no reorientation and no centre-of-mass shift: GTFock's calls
# are collective, so a geometry that differed between ranks would deadlock rather
# than disagree.
_SYSTEMS = {
    # Two 12-atom peptide-backbone fragments (6 C, 2 N, 2 O, 14 H). 260 basis
    # functions in 6-31+G**.
    "peptide": """
0 1
    C      3.281507600000     1.445273000000     1.034710730000
    C      1.604507600000     0.361273004000    -2.219289270000
    C      2.275507600000     0.774273004000     0.103710727000
    O      1.326507600000     0.135273004000     0.560710727000
    N      2.500507600000     0.907273004000    -1.203289270000
    H      3.279507600000     0.902273004000     1.980710730000
    H      3.307507600000     1.437273000000    -1.515289270000
    H      0.579507596000     0.630273004000    -1.981289270000
    H      4.297647600000     1.415103000000     0.614526727000
    H      2.980587600000     2.483703000000     1.237488730000
    H      1.855712800000     0.766963004000    -3.210406270000
    H      1.647067600000    -0.734436996000    -2.306570270000
    C     -1.669492400000    -2.208727000000    -0.834289273000
    C     -1.959492400000     0.854273004000     1.385710730000
    C     -2.391492400000    -0.402726996000     0.645710727000
    O     -3.564492400000    -0.771726996000     0.627710727000
    N     -1.417492400000    -1.039727000000    -0.000289273110
    H     -0.470492404000    -0.681726996000     0.081710726900
    H     -0.735492404000    -2.756727000000    -0.957289273000
    H     -2.384492400000    -2.865727000000    -0.336289273000
    H     -0.871492404000     0.898273004000     1.327710730000
    H     -2.060812400000    -1.976247000000    -1.835698270000
    H     -2.342852400000     1.773423000000     0.918589727000
    H     -2.197492400000     0.842013004000     2.459585730000
units angstrom
symmetry c1
no_reorient
no_com
""",
    # Ethene on a carbon-nanotube fragment (26 C, 16 H). 574 basis functions in
    # 6-31+G**, which is ~140 per rank at four ranks -- comfortably more than one
    # AO block each, which is what makes the decomposition worth measuring.
    "nanotube": """
0 1
    H     -0.923387110000     2.620325580000     1.228608870000
    H      0.923387110000     2.620325580000     1.228608870000
    H      0.923387110000     2.620325580000    -1.228608870000
    H     -0.923387110000     2.620325580000    -1.228608870000
    C      0.000000000000     2.620325580000    -0.666616630000
    C      0.000000000000     2.620325580000     0.666616630000
    C      3.330973000000    -1.379737000000    -0.710500000000
    C      3.330973000000    -1.379737000000     0.710500000000
    C     -3.330973000000    -1.379737000000    -0.710500000000
    C     -3.330973000000    -1.379737000000     0.710500000000
    C      2.355354000000    -0.631118000000    -2.843523000000
    C      2.355354000000    -0.631118000000    -1.422523000000
    C      2.355354000000    -0.631118000000     1.422523000000
    C      2.355354000000    -0.631118000000     2.843523000000
    C     -2.355354000000    -0.631118000000    -2.843523000000
    C     -2.355354000000    -0.631118000000    -1.422523000000
    C     -2.355354000000    -0.631118000000     1.422523000000
    C     -2.355354000000    -0.631118000000     2.843523000000
    C      1.219221000000    -0.160516000000    -3.555546500000
    C      1.219221000000    -0.160516000000    -0.710500000000
    C      1.219221000000    -0.160516000000     0.710500000000
    C      1.219221000000    -0.160516000000     3.555546500000
    C     -1.219221000000    -0.160516000000    -3.555546500000
    C     -1.219221000000    -0.160516000000    -0.710500000000
    C     -1.219221000000    -0.160516000000     0.710500000000
    C     -1.219221000000    -0.160516000000     3.555546500000
    C      0.000000000000    -0.000002999906    -2.843523000000
    C      0.000000000000    -0.000002999906    -1.422523000000
    C      0.000000000000    -0.000002999906     1.422523000000
    C      0.000000000000    -0.000002999906     2.843523000000
    H      3.902336500000    -2.124352200000    -1.253932500000
    H      3.902336500000    -2.124352200000     1.253932500000
    H     -3.902336500000    -2.124352200000    -1.253932500000
    H     -3.902336500000    -2.124352200000     1.253932500000
    H      3.099969000000    -1.202481400000    -3.386955800000
    H      3.099969000000    -1.202481400000     3.386955800000
    H     -3.099969000000    -1.202481400000    -3.386955800000
    H     -3.099969000000    -1.202481400000     3.386955800000
    H      1.219221000000    -0.160516000000    -4.640086500000
    H      1.219221000000    -0.160516000000     4.640086500000
    H     -1.219221000000    -0.160516000000    -4.640086500000
    H     -1.219221000000    -0.160516000000     4.640086500000
units angstrom
symmetry c1
no_reorient
no_com
""",
    # A 157-atom fragment pair carved out of PDB 3acx (52 C, 13 O, 12 N, 80 H),
    # id 157 in the same timing set: a 118-atom peptide and a 39-atom ligand
    # fragment, run as one closed-shell molecule like the other two. 1863 basis
    # functions in 6-31+G**, 3.2x the nanotube, which is the point -- on the 2x2
    # process grid four ranks factor into, each rank owns roughly a 932x932
    # block of the AO matrix, so the partitioning has real work to hide the
    # gather-and-broadcast behind. The source coordinates carry three decimals;
    # they are copied verbatim rather than padded, since the trailing digits
    # would be invented.
    "protein157": """
0 1
    C        51.125       17.185       51.848
    H        50.199       17.199       52.426
    H        51.687       16.291       52.121
    H        50.872       17.141       50.790
    C        51.929       18.454       52.210
    O        51.729       19.001       53.293
    N        52.756       18.933       51.263
    H        52.854       18.417       50.399
    C        53.495       20.187       51.364
    H        54.206       20.237       50.541
    H        54.064       20.200       52.294
    C        52.573       21.415       51.296
    O        52.829       22.386       52.007
    N        51.497       21.374       50.486
    H        51.335       20.559       49.911
    C        50.492       22.441       50.399
    H        51.026       23.376       50.214
    C        49.547       22.207       49.224
    H        50.099       22.117       48.288
    H        48.955       21.301       49.351
    H        48.850       23.039       49.111
    C        49.682       22.621       51.694
    O        49.402       23.760       52.060
    N        49.351       21.516       52.385
    H        49.598       20.603       52.026
    C        48.671       21.557       53.684
    H        47.823       22.241       53.598
    C        48.120       20.192       54.090
    H        47.602       20.245       55.048
    H        47.407       19.823       53.353
    H        48.916       19.452       54.183
    C        49.596       22.100       54.803
    O        49.116       22.807       55.689
    N        50.903       21.802       54.711
    H        51.219       21.214       53.952
    C        51.944       22.268       55.631
    H        51.555       22.215       56.650
    C        53.139       21.324       55.530
    H        53.585       21.344       54.534
    H        53.918       21.594       56.243
    H        52.849       20.294       55.743
    C        52.409       23.719       55.391
    O        53.117       24.241       56.251
    N        52.051       24.344       54.253
    H        51.460       23.865       53.588
    C        52.575       25.654       53.839
    H        53.662       25.547       53.818
    H        52.303       26.431       54.554
    H        52.215       25.922       52.846
    C        58.425       25.417       46.737
    H        59.127       25.063       45.980
    H        58.912       26.204       47.314
    H        57.552       25.830       46.232
    C        58.065       24.222       47.647
    O        58.867       23.297       47.759
    N        56.856       24.232       48.235
    H        56.243       25.025       48.106
    C        56.344       23.138       49.057
    H        57.158       22.808       49.705
    C        55.221       23.662       49.948
    H        54.336       23.939       49.374
    H        54.931       22.897       50.664
    H        55.538       24.531       50.524
    C        55.891       21.911       48.246
    O        56.118       20.792       48.709
    N        55.316       22.126       47.044
    H        55.125       23.074       46.745
    C        55.057       21.069       46.059
    H        54.467       20.298       46.551
    C        54.264       21.584       44.861
    H        54.065       20.784       44.146
    H        53.299       21.982       45.178
    H        54.789       22.379       44.331
    C        56.365       20.443       45.555
    O        56.469       19.220       45.548
    N        57.333       21.299       45.178
    H        57.142       22.292       45.210
    C        58.662       20.931       44.687
    H        58.517       20.318       43.795
    C        59.400       22.202       44.274
    H        58.848       22.753       43.512
    H        59.560       22.866       45.122
    H        60.379       21.963       43.857
    C        59.464       20.091       45.697
    O        60.132       19.145       45.280
    N        59.349       20.421       46.997
    H        58.802       21.229       47.264
    C        59.959       19.656       48.082
    H        60.986       19.430       47.787
    C        60.040       20.485       49.360
    H        60.626       21.391       49.206
    H        59.049       20.783       49.707
    H        60.513       19.921       50.165
    C        59.253       18.315       48.320
    O        59.947       17.325       48.518
    N        57.912       18.275       48.259
    H        57.391       19.124       48.082
    C        57.147       17.037       48.424
    H        56.086       17.272       48.470
    H        57.425       16.563       49.365
    C        57.408       16.047       47.280
    O        57.513       14.847       47.524
    N        57.578       16.560       46.053
    H        57.455       17.556       45.915
    C        57.949       15.802       44.862
    H        57.313       14.916       44.835
    C        57.617       16.643       43.632
    H        57.849       16.099       42.716
    H        56.557       16.897       43.601
    H        58.187       17.573       43.618
    C        59.409       15.309       44.886
    O        59.662       14.214       44.386
    N        60.319       16.070       45.526
    H        60.039       16.972       45.887
    C        61.694       15.655       45.827
    H        62.151       15.323       44.893
    H        61.699       14.833       46.543
    H        62.266       16.488       46.234
    C        56.282       16.378       57.621
    C        56.834       15.837       56.524
    C        58.279       22.417       53.598
    C        57.370       22.160       52.579
    C        58.747       21.380       54.394
    C        56.925       20.858       52.361
    C        58.292       20.088       54.185
    C        56.429       17.916       51.785
    C        55.922       16.622       51.676
    C        56.729       17.578       54.142
    C        56.063       15.397       55.288
    C        55.187       12.287       51.991
    C        54.688       11.792       53.357
    C        56.040       13.558       52.114
    O        55.220       14.572       52.679
    C        57.361       19.807       53.181
    C        56.838       18.417       53.025
    C        56.221       16.278       54.049
    C        55.805       15.806       52.799
    H        56.889       16.671       58.467
    H        55.215       16.540       57.696
    H        57.907       15.694       56.488
    H        58.628       23.427       53.768
    H        57.013       22.975       51.969
    H        59.462       21.581       55.180
    H        56.226       20.680       51.560
    H        58.681       19.305       54.820
    H        56.499       18.526       50.895
    H        55.597       16.245       50.723
    H        57.004       17.959       55.114
    H        54.998       15.320       55.515
    H        56.396       14.385       55.052
    H        54.336       12.479       51.335
    H        55.773       11.505       51.508
    H        54.050       12.532       53.844
    H        55.518       11.575       54.031
    H        56.413       13.855       51.133
    H        56.912       13.374       52.744
    H        54.103       10.878       53.251
units angstrom
symmetry c1
no_reorient
no_com
""",
}

# Expected basis-function counts, asserted so a geometry or basis mix-up shows up
# in the first second of a queued job rather than in the results table. Keyed on
# the (system, basis) pair, lower-cased, because --basis is a knob: a system has
# one size per basis, not one size.
_EXPECTED_NBF = {
    ("peptide", "6-31+g**"): 260,
    ("nanotube", "6-31+g**"): 574,
    ("protein157", "6-31+g**"): 1863,
    ("protein157", "6-31g**"): 1555,
}

# J/K algorithms this script can put under the same SCF. "gtfock" is the exact
# four-center engine; "gtfock_df" is the distributed density-fitting engine that
# grew out of it, and is the arm that makes the DF comparison a like-for-like
# one; "direct" is Psi4's own exact-ERI builder and is the algorithmic
# apples-to-apples reference for "gtfock"; "df" is Psi4's own density fitting
# and is the reference for "gtfock_df".
_ARMS = {"gtfock": "gtfock", "gtfock_df": "gtfock_df", "direct": "direct",
         "df": "df", "pk": "pk"}

# The arms that bring up MPI and drive a GTFock engine. Everything else is one
# process with every core.
_MPI_ARMS = ("gtfock", "gtfock_df")


def assert_basis_as_sized(system, basis_name, basis) -> None:
    """Refuse the point unless the basis is the Cartesian one it was sized for.

    A spherical basis would be refused by GTFock's Simint path at best and make
    the two arms different computations at worst, and a basis-function count that
    disagrees means the geometry or the basis is not what was measured. Both are
    fatal: a point that is not the point asked for is worse than a missing one.
    """
    if basis.has_puream():
        raise RuntimeError(
            f"{basis_name} came up spherical: GTFock's Simint path is Cartesian "
            "only, and a spherical basis would make the two arms different "
            "computations. Check the basis definition rather than forcing "
            "puream.")
    try:
        expected = _EXPECTED_NBF[(system, basis_name.lower())]
    except KeyError:
        known = ", ".join(f"{s}/{b}" for s, b in sorted(_EXPECTED_NBF))
        raise RuntimeError(
            f"{system}/{basis_name} has no recorded size, so the count below "
            f"cannot be checked against anything. Sized pairs: {known}. Add the "
            f"pair to _EXPECTED_NBF once you have confirmed its basis-function "
            f"count ({basis.nbf()} here) is the intended one.") from None
    if basis.nbf() != expected:
        raise RuntimeError(
            f"{system}/{basis_name} gave {basis.nbf()} basis functions, "
            f"expected {expected}: the geometry or the basis is not what this "
            "benchmark was sized for.")


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--system", choices=sorted(_SYSTEMS), required=True)
    parser.add_argument("--arm", choices=sorted(_ARMS), required=True,
                        help="which J/K algorithm to put under the SCF")
    parser.add_argument("--basis", default="6-31+G**")
    parser.add_argument("--df-basis", default=None,
                        help="fitting basis for the df and gtfock_df arms; the "
                             "default is whatever Psi4 pairs with --basis, built "
                             "Cartesian because the orbital basis is")
    parser.add_argument("--method", default="scf",
                        help="anything psi4.energy accepts that needs only J and K")
    parser.add_argument("--threads", type=int, default=1,
                        help="OpenMP threads for this process; the fixed-core sweep "
                             "holds ranks x threads constant, the multi-node sweep "
                             "holds threads constant and lets the ranks add nodes")
    parser.add_argument("--memory", default="4 GB")
    parser.add_argument("--scratch", default=".",
                        help="directory for per-rank scratch and Psi4 output files")
    parser.add_argument("--json-out", default=None,
                        help="write this rank's record to <json-out>.rank<N>.json")
    return parser.parse_args(argv)


def peak_rss_mb() -> float:
    """This process's high-water resident set size, in MiB.

    VmHWM is the kernel's own peak, so it survives the allocator having already
    given the memory back by the time the SCF returns.
    """
    try:
        with open("/proc/self/status") as handle:
            for line in handle:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return float("nan")


# The timer libfock wraps around ``compute_JK()`` for every JK subclass, and the
# parent it sits under in the production SCF. ``get_timer_records`` is keyed by
# full path and its ``timer_name`` is the leaf only, so the leaf name alone names
# one record per distinct parent rather than one record.
_JK_TIMER = "JK: JK"
_JK_TIMER_PARENT = "HF: Form G"


def jk_build_seconds(arm):
    """The production SCF's J/K timer: its full path, wall clock, and call count.

    ``JK: JK`` means the same thing in every arm -- integrals plus whatever
    communication that arm needs, and nothing of the diagonalization, DIIS or
    guess around it -- which is what makes the arms comparable.

    Which record it is matters as much as the number. SAD builds its own JK for
    the atomic subproblems rather than routing through SCF_TYPE, so a guess-side
    ``JK: JK`` can exist at another path; the SCF's is the one under
    ``HF: Form G``. Selecting by leaf name alone would be selecting by traversal
    order. An ambiguous or absent timer is fatal here, at the point of
    measurement, rather than a plausible-looking wrong number in the table or a
    ``null`` that fails in the reducer hours later.
    """
    matches = {key: record
               for key, record in psi4.core.get_timer_records(False).items()
               if key == _JK_TIMER or key.endswith(f";{_JK_TIMER}")}
    under_scf = {key: record for key, record in matches.items()
                 if key.startswith(f"{_JK_TIMER_PARENT};")}
    chosen = under_scf or matches
    if not chosen:
        raise RuntimeError(
            f"the {arm} arm finished with no '{_JK_TIMER}' timer record, so "
            "there is no J/K wall clock to report for this point. The timer "
            "libfock wraps around compute_JK() is either unavailable in this "
            "build or no longer called that.")
    if len(chosen) > 1:
        raise RuntimeError(
            f"the {arm} arm has more than one '{_JK_TIMER}' timer record "
            f"({', '.join(sorted(chosen))}), so which one is the production "
            f"SCF's J/K build is ambiguous. Expected exactly one under "
            f"'{_JK_TIMER_PARENT}'.")
    key, record = next(iter(chosen.items()))
    return key, record["wall_time"], record["n_calls"]


# Where density fitting does the work the ``JK: JK`` timer does not see.
# ``JK::initialize()`` (libfock/jk.cc:593) calls the subclass ``preiterations()``
# outside ``timer_on("JK: JK")`` (jk.cc:650), and for ``MemDFJK`` that is
# ``dfh_->initialize()`` with method STORE (libfock/MemDFJK.cc:71) -- the whole
# three-index tensor, built and kept. ``DirectJK::preiterations()`` does nothing
# comparable, so ``JK: JK`` covers all of an exact-ERI arm's integral work and
# only the GEMM half of density fitting's. Reporting the two arms' ``jk_wall``
# side by side without this number overstates the DF advantage several-fold.
#
# Two builders answer to ``scf_type df`` and they instrument themselves
# differently, so both vocabularies have to be listened for. ``MemDFJK`` goes
# through ``DFHelper`` and brackets ``DFH: sparsity prep`` and
# ``DFH: initialize()``; ``DiskDFJK`` (lib3index is not involved) brackets
# ``JK: (A|mn)``, ``JK: (A|Q)^-1/2`` and ``JK: (Q|mn)`` in its own
# ``preiterations()`` (libfock/DiskDFJK.cc:430). Watching only the DFHelper
# names reports a setup cost of zero for the disk algorithm, which reads as a
# measured absence and is the one wrong answer this instrumentation can give.
# Every name here is a top-level timer, sibling to ``HF: Form G`` rather than
# nested inside another entry on this list, so summing them double-counts
# nothing. ``DFH: AO Construction`` and ``DFH: AO-Met. Contraction`` are
# deliberately absent: they are children of ``DFH: initialize()``.
# ``GTFockDFJK`` is a third vocabulary, and the one this comparison exists for.
# It brackets its whole engine build -- three-center integrals, the metric
# inverse square root and the redistribution -- under a single top-level timer,
# whose name is read off the module rather than copied, so a rename in the C++
# cannot silently turn this arm's setup cost into a measured zero. The name is
# only available in a build that has the engine, so it is appended when it is.
_DF_SETUP_TIMERS = (
    "DFH: sparsity prep",
    "DFH: initialize()",
    "JK: (A|mn)",
    "JK: (A|Q)^-1/2",
    "JK: (Q|mn)",
)

try:
    from psi4.driver import gtfock as _gtfock_module

    if _gtfock_module.df_available():
        _DF_SETUP_TIMERS = _DF_SETUP_TIMERS + (_gtfock_module.df_setup_timer(),)
except (ImportError, AttributeError):
    # A Psi4 without the GTFock module, or one predating the exported name. The
    # other arms are unaffected; the gtfock_df arm cannot run in such a build
    # anyway, so there is nothing whose setup could go unmeasured.
    pass


def df_setup_records():
    """Every density-fitting setup timer record, by full path.

    More than one record can share a timer name: ``sad_scf_type df`` builds its
    own DFHelper for the atomic subproblems, so a guess-side record can sit at a
    different path than the production JK's. Rather than pick between them by a
    rule this script cannot verify from inside one run, every record is reported
    with its full path, longest first, and the reduction decides in the open. An
    arm with no density fitting anywhere returns an empty list, which is a
    measurement and not a failure.
    """
    def is_setup(key):
        return any(key == name or key.endswith(f";{name}")
                   for name in _DF_SETUP_TIMERS)

    return sorted(
        ({"key": key,
          "wall_seconds": record["wall_time"],
          "n_calls": record["n_calls"]}
         for key, record in psi4.core.get_timer_records(False).items()
         if is_setup(key)),
        key=lambda entry: -entry["wall_seconds"])


def main(argv=None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    if args.arm in _MPI_ARMS:
        # mpi4py's import is what calls MPI_Init_thread; go through Psi4's own
        # entry point so the benchmark exercises the shipped module.
        from psi4.driver import gtfock
        if args.arm == "gtfock_df" and not gtfock.df_available():
            raise RuntimeError(
                "this Psi4 links GTFock but not its distributed density-fitting "
                "engine, so the gtfock_df arm cannot run. Rebuild against a "
                "GTFock that ships libgtfockdf and gtfock_pdf.h.")
        info = gtfock.initialize()
        rank, size = info["rank"], info["size"]
        processor = info["processor"]
    else:
        # The reference arms are one process with every core, so there is no MPI
        # to bring up and nothing to initialize.
        gtfock = None
        rank, size, processor = 0, 1, platform.processor()

    rank_scratch = os.path.join(args.scratch, f"rank{rank}")
    os.makedirs(rank_scratch, exist_ok=True)
    psi4.core.IOManager.shared_object().set_default_path(rank_scratch)
    psi4.core.set_output_file(
        os.path.join(args.scratch, f"psi4_rank{rank}.out"), False)

    psi4.set_num_threads(args.threads)
    psi4.set_memory(args.memory)

    molecule = psi4.geometry(_SYSTEMS[args.system])
    # Every arm gets the same convergence, the same guess and the same integral
    # threshold, so the only thing that differs between two points is the J/K
    # algorithm and the rank count. df_scf_guess is off everywhere: it would
    # otherwise put a DF-SCF in front of the algorithm being measured.
    psi4.set_options({
        "basis": args.basis,
        "scf_type": _ARMS[args.arm],
        "df_scf_guess": False,
        # SAD, and SAD's own atomic J/K pinned to DF, on every point. SAD builds
        # its own JK for the atomic subproblems (libscf_solver/sad.cc) rather
        # than routing through SCF_TYPE, so pinning SAD_SCF_TYPE makes the guess
        # bit-identical work in all three arms -- and replicated on every rank,
        # which is part of what the total-vs-J/K wall clock split below exposes.
        "guess": "sad",
        "sad_scf_type": "df",
        "e_convergence": 1e-8,
        "d_convergence": 1e-7,
        "ints_tolerance": 1e-12,
        "maxiter": 100,
        # Keeps the JK object alive past HF::finalize() so its class can be
        # reported; without it psi4 releases the object and wfn.jk() is None.
        "save_jk": True,
    })
    if args.df_basis:
        # Both fitted arms get the same fitting basis when one is named, so that
        # the only difference between them stays the engine. Left unset, Psi4
        # pairs its own with --basis and builds it Cartesian because the orbital
        # basis is, which is what GTFock's Simint path requires of both.
        psi4.set_options({"df_basis_scf": args.df_basis})

    # Built and checked here, before any integral work, so a mistyped --basis or
    # an edited geometry costs a second of a queued allocation rather than a
    # whole SCF -- and, under the SLURM script's `set -eo pipefail`, rather than
    # the rest of the sweep behind it.
    assert_basis_as_sized(
        args.system, args.basis,
        psi4.core.BasisSet.build(molecule, "ORBITAL", args.basis))

    # Each engine has its own tally; asking the exact one about a DF run would
    # report a confident zero.
    if gtfock is None:
        jk_builds_before = 0
    elif args.arm == "gtfock_df":
        jk_builds_before = gtfock.df_jk_builds()
    else:
        jk_builds_before = gtfock.fock_builds()
    start = time.perf_counter()
    energy, wfn = psi4.energy(args.method, return_wfn=True)
    elapsed = time.perf_counter() - start
    jk_timer_key, jk_wall, jk_calls = jk_build_seconds(args.arm)
    df_setup = df_setup_records()

    # The report describes the basis the SCF actually used, so it is read back
    # off the wavefunction and held to the same guards -- the pre-flight check
    # cannot see a basis the driver swapped underneath it.
    basis = wfn.basisset()
    assert_basis_as_sized(args.system, args.basis, basis)

    report = {
        "system": args.system,
        "arm": args.arm,
        "basis": args.basis,
        "method": args.method,
        "nbf": basis.nbf(),
        "nshell": basis.nshell(),
        "puream": basis.has_puream(),
        "ranks": size,
        "rank": rank,
        "threads_per_rank": args.threads,
        "total_cores": size * args.threads,
        "scf_energy": energy,
        "iterations": wfn.iteration_,
        "scf_wall_seconds": elapsed,
        "jk_wall_seconds": jk_wall,
        "jk_calls": jk_calls,
        "jk_timer_key": jk_timer_key,
        # The DF setup cost, and the part of the SCF no J/K timer covers. The
        # remainder is diagonalization, DIIS and the density build in every arm,
        # plus -- in the df arm only -- the three-index tensor construction,
        # which is why the two are reported together rather than one alone.
        "df_setup_records": df_setup,
        "df_setup_total_seconds": sum(r["wall_seconds"] for r in df_setup),
        "scf_remainder_seconds": elapsed - jk_wall,
        "peak_rss_mb": peak_rss_mb(),
        "jk_name": wfn.jk().name(),
        "df_basis_scf": psi4.core.get_global_option("DF_BASIS_SCF") or "(auto)",
        "host": socket.gethostname(),
        "processor": processor,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_nodelist": os.environ.get("SLURM_JOB_NODELIST"),
    }
    if gtfock is not None:
        report["core_mpi"] = gtfock.mpi_info()
    if args.arm == "gtfock":
        report.update({
            "fock_builds": gtfock.fock_builds() - jk_builds_before,
            "process_grid": psi4.core.gtfock_process_grid(),
            "local_block": psi4.core.gtfock_local_block(),
            "local_task_shape": psi4.core.gtfock_local_task_shape(),
        })
    elif args.arm == "gtfock_df":
        # The DF engine partitions the auxiliary index, not the AO matrix, so it
        # has no process grid and no local AO block to report. The fields are
        # left out rather than filled with a placeholder the reducer would have
        # to tell apart from a measurement. What replaces them is the partition
        # itself: nlocal_aux summing to naux over the ranks is what says the
        # fitted tensor was distributed rather than replicated, and it is the
        # one claim this arm makes that a wall clock cannot check.
        report.update({
            "fock_builds": gtfock.df_jk_builds() - jk_builds_before,
            "df_partition": gtfock.df_partition(),
            # The setup timer above is the total; this is where it went. Only
            # some of these phases divide over ranks, so the breakdown is what
            # says whether a setup that stopped improving has stopped because
            # of the replicated metric factorization or something else.
            "df_setup_phases": gtfock.df_setup_phases(),
            # And the same question asked of the iterations rather than the
            # build: how much of J/K was arithmetic (jk_local), how much was
            # waiting for the slowest rank (jk_skew), and how much was the
            # reduction (jk_comm). Summed over calls; fock_builds above divides
            # them into a per-build cost.
            "df_jk_phases": gtfock.df_jk_phases(),
        })

    print("PSI4-GTFOCK-JSON " + json.dumps(report), flush=True)
    if args.json_out:
        with open(f"{args.json_out}.rank{rank}.json", "w") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)

    # Releasing the GTFock engine is collective, so drop the wavefunction here,
    # at the same point on every rank, rather than at interpreter teardown.
    del wfn
    return 0


if __name__ == "__main__":
    sys.exit(main())
