#!/usr/bin/env python3
"""Run the native Psi4 atomic-polarizability route on water, by hand.

Everything this script computes is written into the Psi4 output file, not just to the
terminal, so the SCF triple, the pipeline's own stage banners, and the parity scoreboard
all land in one readable log next to each other.

    python run_example_water.py                       # shipped defaults
    python run_example_water.py faithful-cloud        # EQUAL_VOLUME + VOLUME vdW 2->4
    python run_example_water.py basis-space           # ISA Algorithm A, shipped cloud
    python run_example_water.py faithful-basis-space   # both at once
    python run_example_water.py --list                # show the arms and exit
    python run_example_water.py <arm> --dry-run       # resolve options, run nothing

``--dry-run`` answers "can this psi4 build run this arm?" without paying for the run.
Keywords are global options, so a build that predates one rejects it even when the value
asked for is that keyword's own default; the script drops those and reports which arms
the build genuinely cannot express.

Environment
-----------
``OMP_NUM_THREADS=1`` is required for bit-reproducibility: threaded BLAS reorders the
reductions inside the response solve and every downstream artifact drifts at the 1e-8
level while still looking self-consistent. The script sets Psi4's own thread count to 1
and warns if the environment variable disagrees.

Runtime is roughly 6-7 minutes per arm on one core: three aug-cc-pVTZ SCFs on a 590/99
DFT grid, an ISA fixed point on a 100/24/32 atom grid, a dense FDDS response solve at
eleven frequencies, and a constrained WSM refinement at each of them.

Nothing here reads CamCASP, ORIENT, PFIT or CASIMIR. The reference numbers are the
literals already checked into ``tests/pytests/test_atomic_polarizabilities.py``, quoted
below so this file runs standalone from any directory.
"""

import os
import sys
import time

import numpy as np
import psi4
from psi4.driver.procrouting import atomic_polarizability as native_driver

# --------------------------------------------------------------------------------------
# Geometry: the reviewed parity geometry, in bohr, with symmetry and orientation frozen.
# `no_com`/`no_reorient`/`symmetry c1` matter -- the anisotropic blocks are published in
# the *molecular* frame, so letting Psi4 reorient would silently rotate the output.
# --------------------------------------------------------------------------------------
GEOMETRY = """
0 1
O  0.00000000  0.0  0.00000000
H -1.45365196  0.0 -1.12168732
H  1.45365196  0.0 -1.12168732
symmetry c1
no_com
no_reorient
units bohr
"""

# --------------------------------------------------------------------------------------
# The reviewed protocol. Grid quality is never inherited silently: the DFT grid is
# binding once diffuse functions are present (aug-cc-pVDZ sticks at an LW charge-sum
# residual of 1.2e-5 on 302/50 no matter how dense the ISA grid gets), so both grids are
# pinned here rather than left at their defaults.
# --------------------------------------------------------------------------------------
PARITY_PROTOCOL = {
    "basis": "aug-cc-pvtz",
    "scf_type": "pk",
    "e_convergence": 1.0e-10,
    "d_convergence": 1.0e-9,
    "dft_spherical_points": 590,
    "dft_radial_points": 99,
    "dft_density_tolerance": 1.0e-12,
    "atomic_polarizability_partition": "ISA",
    "atomic_polarizability_isa_radial_points": 100,
    "atomic_polarizability_isa_angular_polar_points": 24,
    "atomic_polarizability_isa_angular_azimuthal_points": 32,
    "atomic_polarizability_localization_tolerance": 1.0e-6,
}

# --------------------------------------------------------------------------------------
# Arms. Each one names *every* keyword it varies, because Psi4 options are global and
# sticky: a script that sets only the keyword it is studying will silently inherit the
# previous arm's setting and produce a scoreboard that attributes the change to the wrong
# lever. One arm per process is the only safe protocol.
# --------------------------------------------------------------------------------------
ARMS = {
    "default": {
        "doc": "shipped defaults: BOHR 4.5-11.5, LINEAR shells, UNIFORM nodes, real-space ISA",
        "options": {
            "atomic_polarizability_fit_radial_units": "BOHR",
            "atomic_polarizability_fit_inner_limit": 4.5,
            "atomic_polarizability_fit_outer_limit": 11.5,
            "atomic_polarizability_fit_radial_spacing": "LINEAR",
            "atomic_polarizability_fit_angular_weighting": "UNIFORM",
            "atomic_polarizability_isa_method": "REAL_SPACE",
        },
    },
    "faithful-cloud": {
        "doc": "volume-faithful cloud inside the reference window: vdW 2->4, EQUAL_VOLUME, VOLUME",
        "options": {
            "atomic_polarizability_fit_radial_units": "VDW",
            "atomic_polarizability_fit_inner_limit": 2.0,
            "atomic_polarizability_fit_outer_limit": 4.0,
            "atomic_polarizability_fit_radial_spacing": "EQUAL_VOLUME",
            "atomic_polarizability_fit_angular_weighting": "VOLUME",
            "atomic_polarizability_isa_method": "REAL_SPACE",
        },
    },
    "basis-space": {
        "doc": "ISA Algorithm A on the shipped cloud: one keyword, no cloud change",
        "options": {
            "atomic_polarizability_fit_radial_units": "BOHR",
            "atomic_polarizability_fit_inner_limit": 4.5,
            "atomic_polarizability_fit_outer_limit": 11.5,
            "atomic_polarizability_fit_radial_spacing": "LINEAR",
            "atomic_polarizability_fit_angular_weighting": "UNIFORM",
            "atomic_polarizability_isa_method": "BASIS_SPACE_A",
        },
    },
    "faithful-basis-space": {
        "doc": "both levers: best measured static and C6 anywhere, but C8 leaves its band",
        "options": {
            "atomic_polarizability_fit_radial_units": "VDW",
            "atomic_polarizability_fit_inner_limit": 2.0,
            "atomic_polarizability_fit_outer_limit": 4.0,
            "atomic_polarizability_fit_radial_spacing": "EQUAL_VOLUME",
            "atomic_polarizability_fit_angular_weighting": "VOLUME",
            "atomic_polarizability_isa_method": "BASIS_SPACE_A",
        },
    },
}

# --------------------------------------------------------------------------------------
# Not every psi4 build has every keyword. The arms above name each keyword they vary,
# including the ones they leave at the shipped default, because options are global and
# sticky. That is right for reproducibility and wrong for portability: on a build that
# predates a keyword, asking for its own default is still an error. So record what the
# default is, and which branch introduced the keyword, and let the arm degrade.
#
# A keyword missing from the build is only safe to drop when the arm wanted the default:
# the build then behaves as though it had been set, because the behaviour the keyword
# selects at its default value is exactly the behaviour that was hardwired before it
# existed. Wanting anything else needs a build that can express it.
# --------------------------------------------------------------------------------------
KEYWORD_DEFAULTS = {
    "atomic_polarizability_fit_radial_units": "BOHR",
    "atomic_polarizability_fit_inner_limit": 4.5,
    "atomic_polarizability_fit_outer_limit": 11.5,
    "atomic_polarizability_fit_radial_spacing": "LINEAR",
    "atomic_polarizability_fit_angular_weighting": "UNIFORM",
    "atomic_polarizability_isa_method": "REAL_SPACE",
}
KEYWORD_ORIGIN = {
    "atomic_polarizability_fit_radial_spacing":
        "11cfeed59 'feat: add equal-volume fit-point radial spacing' "
        "(branch worktree-fit-radial-spacing)",
    "atomic_polarizability_fit_angular_weighting":
        "d6100119b 'feat: add volume-centroid shells and volume angular weighting' "
        "(branch worktree-fit-radial-spacing)",
    "atomic_polarizability_isa_method":
        "90bd2a175 'feat: add basis-space ISA Algorithm A' (branch camcasp)",
}

# --------------------------------------------------------------------------------------
# Reference values, copied from tests/pytests/test_atomic_polarizabilities.py. These are
# an *external molecular reference*, not a snapshot of our own output: agreement is a
# statement about the finite-rank model, not about transcription. The bands below are the
# reviewed acceptance widths, which is why they are wide -- the higher-rank coefficients
# carry a documented model deficit, not a bug.
# --------------------------------------------------------------------------------------
LABELS = ("xx", "xy", "xz", "yy", "yz", "zz")
SITES = ("O", "H1", "H2")

ORACLE_STATIC = np.asarray([
    [7.041967041199, 0.0, 0.0, 7.473775078471, 0.0, 7.128954933164],  # O
    [1.587044944101, 0.0, 0.645265189422, 0.760937870807, 0.0, 1.240793691747],  # H1
    [1.587044944101, 0.0, -0.645265189422, 0.760937870807, 0.0, 1.240793691747],  # H2
])
STATIC_BAND = 0.16

ORACLE_CN = {
    "ATOMIC C6": np.asarray([[26.48176709, 4.142316899, 4.142316899],
                             [4.142316899, 0.6514696683, 0.6514696683],
                             [4.142316899, 0.6514696683, 0.6514696683]]),
    "ATOMIC C8": np.asarray([[490.4584355, 65.08315227, 65.08315227],
                             [65.08315227, 8.463255173, 8.463255173],
                             [65.08315227, 8.463255173, 8.463255173]]),
    "ATOMIC C10": np.asarray([[9673.248403, 1262.304843, 1262.304843],
                              [1262.304843, 168.1889023, 168.1889023],
                              [1262.304843, 168.1889023, 168.1889023]]),
    "ATOMIC C12": np.asarray([[150417.3729, 18759.27627, 18759.27627],
                              [18759.27627, 2278.795679, 2278.795679],
                              [18759.27627, 2278.795679, 2278.795679]]),
}
CN_BANDS = {"ATOMIC C6": 0.11, "ATOMIC C8": 0.27, "ATOMIC C10": 0.37, "ATOMIC C12": 0.47}


def log(line=""):
    """Write one line to the Psi4 output file *and* to the terminal."""
    psi4.core.print_out(line + "\n")
    print(line, flush=True)


def build_banner():
    """Identify the psi4 that actually got imported, since PYTHONPATH decides that.

    Worth printing: several builds of this fork coexist on a typical dev box and they do
    not all carry the same keywords, so a scoreboard is only interpretable next to the
    commit that produced it. The plain __version__ is the same string on every branch,
    which is why this reaches for the long form that carries the abbreviated hash.
    """
    try:
        from psi4.metadata import __version_long as version
    except Exception:  # noqa: BLE001
        version = psi4.__version__
    return [f"psi4 {version}",
            f"     from {os.path.dirname(psi4.__file__)}"]


def keyword_exists(key):
    """True when the loaded build declares this option at all."""
    try:
        psi4.core.get_global_option(key.upper())
    except Exception:  # noqa: BLE001 -- psi4 raises a bare RuntimeError here
        return False
    return True


def resolve_options(options):
    """Split the requested options into (usable, dropped-as-default, unavailable)."""
    usable, dropped, unavailable = {}, [], []
    for key, value in options.items():
        if keyword_exists(key):
            usable[key] = value
        elif key in KEYWORD_DEFAULTS and KEYWORD_DEFAULTS[key] == value:
            dropped.append(key)
        else:
            unavailable.append(key)
    return usable, dropped, unavailable


def worst_relative(actual, oracle):
    """Largest relative deviation over the nonzero reference entries, and where it sits."""
    actual = np.asarray(actual)
    mask = oracle != 0.0
    relative = np.zeros_like(oracle, dtype=float)
    relative[mask] = np.abs(actual[mask] - oracle[mask]) / np.abs(oracle[mask])
    flat = int(np.argmax(relative))
    return float(relative.max()), np.unravel_index(flat, relative.shape)


def main():
    argv = [a for a in sys.argv[1:] if a]
    dry_run = "--dry-run" in argv
    argv = [a for a in argv if a != "--dry-run"]
    if argv and argv[0] in ("--list", "-l", "--help", "-h"):
        print("arms:")
        for name, arm in ARMS.items():
            print(f"  {name:22s} {arm['doc']}")
        return 0
    arm_name = argv[0] if argv else "default"
    if arm_name not in ARMS:
        print(f"unknown arm {arm_name!r}; one of {', '.join(ARMS)}", file=sys.stderr)
        return 2
    arm = ARMS[arm_name]

    output = os.path.abspath(f"water_atomic_polarizability_{arm_name}.out")
    psi4.core.set_output_file(output, False)
    psi4.set_num_threads(1)
    # Memory is deliberately left at Psi4's default here. The driver raises it to
    # PIPELINE_MEMORY_BYTES (4 GB) around the fit stage on its own and restores it
    # afterwards, because the WSM resource gate refuses to run when the estimated SVD
    # peak exceeds half the configured memory -- about 0.45 GB on the default cloud.
    # Setting it globally instead would change what the SCFs see and stop these numbers
    # being comparable with the measured scoreboard.

    threads = os.environ.get("OMP_NUM_THREADS")
    log("=" * 88)
    log(f"Native atomic-polarizability route on H2O -- arm {arm_name!r}")
    log(f"  {arm['doc']}")
    log("=" * 88)
    if threads != "1":
        log(f"WARNING: OMP_NUM_THREADS={threads!r}, not '1'. Threaded BLAS reorders the")
        log("         response reductions and every artifact below drifts at ~1e-8 while")
        log("         still looking self-consistent. Re-run with OMP_NUM_THREADS=1 before")
        log("         comparing anything at parity precision.")
        log("")

    for line in build_banner():
        log(line)
    log("")

    requested = {**PARITY_PROTOCOL, **arm["options"]}
    options, dropped, unavailable = resolve_options(requested)
    if unavailable:
        log(f"ERROR: arm {arm_name!r} needs keywords this psi4 build does not have:")
        for key in unavailable:
            origin = KEYWORD_ORIGIN.get(key, "a newer branch")
            log(f"    {key.upper():55s} wanted {requested[key]!r}")
            log(f"    {'':55s} introduced by {origin}")
        log("")
        log("Rebuild psi4 from a branch that carries them -- worktree-fit-radial-spacing")
        log("carries all of them -- or run an arm that does not need them. Try")
        log("`python run_example_water.py <arm> --dry-run` to test an arm against a build.")
        return 3

    psi4.core.clean_variables()
    try:
        psi4.set_options(options)
    except Exception as exc:  # noqa: BLE001 -- psi4 wraps these in ValidationError
        log(f"ERROR: this psi4 build rejected arm {arm_name!r}'s options.")
        log("A declared keyword can still refuse a value: a build may know")
        log("ATOMIC_POLARIZABILITY_FIT_RADIAL_UNITS and not list VDW among its choices.")
        log("")
        log(str(exc))
        return 4
    molecule = psi4.geometry(GEOMETRY)

    log("Options in force (everything else is at its Psi4 default):")
    for key in sorted(options):
        log(f"    {key.upper():55s} {options[key]}")
    if dropped:
        log("")
        log("Not set -- absent from this build, and this arm wanted the default anyway,")
        log("so the build's hardwired behaviour is already what was asked for:")
        for key in sorted(dropped):
            log(f"    {key.upper():55s} would have been {requested[key]!r}")
    log("")
    if dry_run:
        log(f"--dry-run: this build can run arm {arm_name!r}. Stopping before the SCFs.")
        return 0

    log("Pipeline: three PBE0 SCFs (neutral precursor -> cation -> GRAC-corrected)")
    log("  -> FDDS response kernel at 11 frequencies (25% CHF + 75% ALDA)")
    log("  -> ISA partition -> LW localization -> WSM refinement -> Casimir-Polder Cn")
    log("")

    start = time.time()
    wfn = native_driver.atomic_polarizabilities(molecule=molecule)
    elapsed = time.time() - start
    log(f"pipeline finished in {elapsed:.1f} s")
    log("")

    # ---- static dipole polarizabilities ------------------------------------------------
    static = np.asarray(wfn.array_variable("ATOMIC POLARIZABILITIES"))
    log("ATOMIC POLARIZABILITIES  (natom, 6) in a.u., packed xx xy xz yy yz zz")
    log("-" * 88)
    log(f"{'site':>6} {'comp':>5} {'reference':>16} {'ours':>16} {'relative':>12}")
    for i, site in enumerate(SITES):
        for j, label in enumerate(LABELS):
            if ORACLE_STATIC[i, j] == 0.0:
                continue
            reference = ORACLE_STATIC[i, j]
            ours = static[i, j]
            log(f"{site:>6} {label:>5} {reference:16.9f} {ours:16.9f} "
                f"{(ours - reference) / reference:+12.5f}")
    worst_static, cell = worst_relative(static, ORACLE_STATIC)
    verdict = "PASS" if worst_static <= STATIC_BAND else "FAIL"
    log("-" * 88)
    log(f"worst static component {worst_static:.6f} at {SITES[cell[0]]}.{LABELS[cell[1]]}"
        f"   band {STATIC_BAND:.2f}  {verdict}")
    log("")

    # ---- isotropic dispersion coefficients ---------------------------------------------
    log("Isotropic dispersion coefficients, site-pair matrices in a.u.")
    log("-" * 88)
    log(f"{'coefficient':>12} {'worst rel':>11} {'band':>7} {'verdict':>8} {'worst cell':>12}")
    scoreboard = []
    for name, reference in ORACLE_CN.items():
        ours = np.asarray(wfn.array_variable(name))
        worst, pair = worst_relative(ours, reference)
        band = CN_BANDS[name]
        passes = worst <= band
        scoreboard.append((name, worst, band, passes))
        log(f"{name:>12} {worst:11.4f} {band:7.2f} {'PASS' if passes else 'FAIL':>8} "
            f"{SITES[pair[0]] + '-' + SITES[pair[1]]:>12}")
    log("-" * 88)
    for name, reference in ORACLE_CN.items():
        ours = np.asarray(wfn.array_variable(name))
        log(f"{name} (ours):")
        for i, site in enumerate(SITES):
            log("    " + f"{site:>3} " + " ".join(f"{v:16.7f}" for v in ours[i]))
    log("")

    # ---- what else was published -------------------------------------------------------
    frequencies = np.asarray(wfn.array_variable("ATOMIC POLARIZABILITY FREQUENCIES")).ravel()
    dynamic = np.asarray(wfn.array_variable("ATOMIC DYNAMIC POLARIZABILITIES"))
    anisotropic = np.asarray(wfn.array_variable("ATOMIC ANISOTROPIC POLARIZABILITIES"))
    labels = wfn.array_variable("ATOMIC DISPERSION LABELS")
    log("Also published on this wavefunction and in the global QCVariable store:")
    log(f"    ATOMIC POLARIZABILITY FREQUENCIES         {frequencies.shape}  "
        f"{np.array2string(frequencies, precision=6, max_line_width=200)}")
    log(f"    ATOMIC DYNAMIC POLARIZABILITIES           {dynamic.shape}")
    log(f"    ATOMIC ANISOTROPIC POLARIZABILITIES       {anisotropic.shape}"
        "   (rank 1-3 real-spherical, molecular frame)")
    log(f"    ATOMIC DISPERSION COEFFICIENTS / LABELS   "
        f"{np.asarray(wfn.array_variable('ATOMIC DISPERSION COEFFICIENTS')).shape} / "
        f"{np.asarray(labels).shape}")
    log("    Real-spherical component order: 10 11c 11s 20 21c 21s 22c 22s 30 31c 31s"
        " 32c 32s 33c 33s")
    log("")

    # ---- summary line ------------------------------------------------------------------
    log("=" * 88)
    cells = [f"static {worst_static:.4f}"] + [f"{n.split()[-1]} {w:.4f}"
                                              for n, w, _, _ in scoreboard]
    bands_passed = int(worst_static <= STATIC_BAND) + sum(p for _, _, _, p in scoreboard)
    log(f"arm {arm_name!r}: " + "  ".join(cells))
    log(f"{bands_passed} of 5 bands pass.  Full log: {output}")
    log("=" * 88)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
