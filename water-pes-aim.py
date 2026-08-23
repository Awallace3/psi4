#!/usr/bin/env python
"""Water-water PES: how much do our distributed-property errors move the energy?

DEV-ONLY SCRATCH DRIVER. This file is deliberately outside the Psi4 package tree and
must be relocated or deleted before any upstream PR. It is not imported by Psi4, not
collected by pytest, and not part of any build.

Why this exists
---------------
The parity work established that our distributed properties differ from CamCASP's by
*measured* amounts: a uniform 2.9% deficit in the molecular polarizability total, per-pair
C6 within 1-10% of the matching ISA-GRID oracle, and a rank-growing deficit reaching
~25/36/46% at C8/C10/C12. Those are property-space numbers. They do not answer the
question that actually matters for a force field, which is:

    how many kcal/mol of interaction energy does that cost, and at what separation?

A 46% error in C12 sounds fatal and is almost certainly irrelevant beyond 3.5 A; a 3%
error in the polarizability total is quiet in property space and shows up linearly in
induction everywhere. This script measures both, on a real water-water surface, by
running the *same* energy expression twice and changing only the distributed properties.

Design: everything except the properties is held fixed
------------------------------------------------------
Both arms share the same geometries, the same permanent multipoles, the same damping
parameters and the same energy kernels. The only difference between "ours" and a CamCASP
arm is the numbers in the alpha / Cn tables. Any energy difference is therefore
attributable to the properties and to nothing else. This is the same single-variable
discipline the ISA-GRID oracle experiment used.

Kernel provenance
-----------------
Both kernels follow the conventions of ~/gits/qcmlforge so the numbers are comparable
with that code, but neither calls it, for two independent reasons:

* Induction: `apnet_pt.multipole.dimer_induced_dipole_torch` accepts only *isotropic*
  scalar polarizabilities. There is no anisotropic (3x3) support anywhere in qcmlforge.
  Since the whole point here is distributed *anisotropic* polarizabilities, the solver
  had to be written. It reduces to qcmlforge's model when alpha is isotropic, and
  `--validate` checks exactly that.
* Dispersion: `qcml_dftd3.d3` computes its C6 internally from Grimme reference data and
  exposes no override hook -- `resolve_d3_damping_parameters` rejects any key other than
  s6/s8/a1/a2. Feeding in our own distributed coefficients requires the ~6-line BJ sum,
  which is reproduced here.

Two deliberate departures from stock D3, both reported separately rather than folded in:

* The BJ radius. Stock D3 uses R0 = a1*sqrt(3 r4r2_A r4r2_B) + a2, where the r4r2 ratio is
  an *approximation* to sqrt(C8/C6). We have real distributed C8 and C6, so
  sqrt(C8/C6) is available directly -- which is in fact Becke and Johnson's original
  definition of the critical radius. Both choices are computed. `--r0 r4r2` isolates the
  effect of the Cn magnitudes at a fixed damping radius; `--r0 c8c6` lets the C8/C6 ratio
  move the radius too, which is the physically consistent choice but does mean a1/a2 are
  being used outside the fit they came from. The difference between the two columns is
  itself informative.
* C10. Stock D3 has no C10 term and qcmlforge has no s10, so the C10 contribution is
  reported as its own column and is excluded from the D3-comparable total.

Usage
-----
    # stage 1, under the Psi4 env, ~10 min: monomer properties -> JSON cache
    /home/awallace43/miniconda3/envs/p4_camcasp/bin/python3.13 -P water-pes-aim.py monomer

    # stage 2, cheap, no Psi4 needed: the actual surface
    python water-pes-aim.py pes

    # optional: check the kernels against qcmlforge itself
    /home/awallace43/miniconda3/envs/qcml_pt210/bin/python water-pes-aim.py validate
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------------------
# Constants. Values taken to match qcmlforge (src/apnet_pt/constants.py) exactly so that
# a cross-check against it is not confounded by unit conversions.
# --------------------------------------------------------------------------------------

BOHR_PER_ANGSTROM = 1.0 / 0.52917721067
HARTREE_TO_KCAL = 627.5094737775373

# Everything this script writes goes in one directory so cleanup before a PR is a single
# `rm -rf water-pes-aim.work water-pes-aim.py`.
WORK_DIR = Path(__file__).resolve().parent / "water-pes-aim.work"
CACHE_PATH = WORK_DIR / "monomer-properties.json"

# Thole "mutual" damping, the flavour dimer_induced_dipole_torch uses
# (apnet_pt/multipole.py:1374). The monomer path uses the "direct" flavour at 0.34; that
# is not what the dimer model applies and is not used here.
THOLE_A = 0.39

# D3(I)/SAPT-PBE0, the set qcml_dftd3.d3 actually defaults to (qcml_dftd3/d3.py:15).
# NOT the module `defaults`, which are a1=0.4/a2=5.0 and are not what d3() uses.
D3_S6 = 1.0
D3_S8 = 0.8614
D3_A1 = 0.7171
D3_A2 = 0.5375
# No s10 exists anywhere in qcmlforge. Reported separately, never folded into a total
# that claims to be D3-comparable.
D3_S10 = 1.0

# sqrt(0.5 * <r^4>/<r^2> * sqrt(Z)), matching qcml_dftd3.data.r4r2.R4R2().
#
# Reproduced to the last bit of that table, which is stored in float32 -- hence the
# otherwise-odd trailing digits. `validate` compares against the live table and will fail
# if it ever changes, so these are pinned rather than trusted.
R4R2 = {1: 2.0073490142822266, 8: 2.5936167240142822}

REVIEWED_MONOMER_BOHR = np.array(
    [
        [0.00000000, 0.0, 0.00000000],  # O
        [-1.45365196, 0.0, -1.12168732],  # H1
        [1.45365196, 0.0, -1.12168732],  # H2
    ]
)
MONOMER_ELEMENTS = ("O", "H", "H")
MONOMER_Z = (8, 1, 1)

# The reviewed parity protocol, verbatim from tests/pytests/test_atomic_polarizabilities.py.
# Anything cheaper does not reproduce the reviewed literals and makes the "ours" arm a
# different model from the one the parity work measured.
PARITY_PROTOCOL = {
    "basis": "aug-cc-pvtz",
    "scf_type": "pk",
    "e_convergence": 1.0e-10,
    "d_convergence": 1.0e-9,
    "dft_spherical_points": 590,
    "dft_radial_points": 99,
    "dft_density_tolerance": 1.0e-12,
    "atomic_polarizability_isa_radial_points": 100,
    "atomic_polarizability_isa_angular_polar_points": 24,
    "atomic_polarizability_isa_angular_azimuthal_points": 32,
    "atomic_polarizability_localization_tolerance": 1.0e-6,
}

# --------------------------------------------------------------------------------------
# CamCASP reference properties.
#
# Copied from the reviewed literals in tests/pytests/test_atomic_polarizabilities.py.
# Duplicated rather than imported on purpose: importing the test module would drag in the
# whole Psi4 pytest fixture stack, and this file must stay runnable without it. These are
# *reference* values in the same packed-Cartesian molecular-frame convention the pipeline
# publishes -- (xx, xy, xz, yy, yz, zz), atom order O, H1, H2.
#
# THE TWO ORACLES ARE NOT INTERCHANGEABLE. ISA_GRID is the partition-matched one for this
# pipeline; DF partitions the response by constrained density fitting and is a different
# model, wrong here by up to a factor of 113 on a single component with nothing defective
# anywhere. DF is carried only so the scan can show what comparing against the wrong
# oracle would have implied.
# --------------------------------------------------------------------------------------

CAMCASP = {
    "isa_grid": {
        "label": "CamCASP ISA-GRID (partition-matched)",
        "alpha": [
            [7.041967041199, 0.0, 0.0, 7.473775078471, 0.0, 7.128954933164],
            [1.587044944101, 0.0, 0.645265189422, 0.760937870807, 0.0, 1.240793691747],
            [1.587044944101, 0.0, -0.645265189422, 0.760937870807, 0.0, 1.240793691747],
        ],
        "C6": [
            [26.48176709, 4.142316899, 4.142316899],
            [4.142316899, 0.6514696683, 0.6514696683],
            [4.142316899, 0.6514696683, 0.6514696683],
        ],
        "C8": [
            [490.4584355, 65.08315227, 65.08315227],
            [65.08315227, 8.463255173, 8.463255173],
            [65.08315227, 8.463255173, 8.463255173],
        ],
        "C10": [
            [9673.248403, 1262.304843, 1262.304843],
            [1262.304843, 168.1889023, 168.1889023],
            [1262.304843, 168.1889023, 168.1889023],
        ],
        "C12": [
            [150417.3729, 18759.27627, 18759.27627],
            [18759.27627, 2278.795679, 2278.795679],
            [18759.27627, 2278.795679, 2278.795679],
        ],
    },
    "df": {
        "label": "CamCASP C-DF (different partition -- NOT the matching oracle)",
        "alpha": [
            [7.043489935336, 0.0, 0.0, 5.762074477569, 0.0, 5.583657081749],
            [1.573674631536, 0.0, 0.005761700031, 1.617426936478, 0.0, 2.009572611043],
            [1.573674631536, 0.0, -0.005761700031, 1.617426936478, 0.0, 2.009572611043],
        ],
        "C6": [
            [17.25559, 5.382332, 5.382332],
            [5.382332, 1.698678, 1.698678],
            [5.382332, 1.698678, 1.698678],
        ],
        "C8": [
            [346.424, 83.90759, 83.90759],
            [83.90759, 18.32833, 18.32833],
            [83.90759, 18.32833, 18.32833],
        ],
        "C10": [
            [7484.441, 1523.525, 1523.525],
            [1523.525, 291.4843, 291.4843],
            [1523.525, 291.4843, 291.4843],
        ],
        "C12": [
            [127231.0, 20293.77, 20293.77],
            [20293.77, 3216.541, 3216.541],
            [20293.77, 3216.541, 3216.541],
        ],
    },
}


# --------------------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------------------

# S22 #2, the standard hydrogen-bonded water dimer, in Angstrom. Fragment 1 is the donor:
# its second hydrogen at x = -0.5997 is the one pointing at fragment 2's oxygen.
S22_WATER_DIMER_ANGSTROM = np.array(
    [
        [-1.551007, -0.114520, 0.000000],  # O  (donor)
        [-1.934259, 0.762503, 0.000000],   # H  (free)
        [-0.599677, 0.040712, 0.000000],   # H  (bridging)
        [1.350625, 0.111469, 0.000000],    # O  (acceptor)
        [1.680398, -0.373741, -0.758561],  # H
        [1.680398, -0.373741, 0.758561],   # H
    ]
)


def kabsch_rotation(mobile: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Proper rotation carrying centred `mobile` onto centred `target`.

    Reflections are excluded explicitly. A reflection would silently flip the sign of
    every x-odd polarizability component -- exactly the failure the parity test suite's
    C2 relation exists to catch -- so it is rejected here rather than tolerated.
    """
    correlation = mobile.T @ target
    left, _, right = np.linalg.svd(correlation)
    sign = np.sign(np.linalg.det(right.T @ left.T))
    scale = np.diag([1.0, 1.0, sign])
    rotation = right.T @ scale @ left.T
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1.0e-10):
        raise AssertionError("Kabsch produced an improper rotation")
    return rotation


@dataclass
class Fragment:
    """A rigid monomer placed in the dimer, with its properties already transformed."""

    positions: np.ndarray  # (natom, 3) bohr
    charges: np.ndarray  # (natom,) a.u.
    dipoles: np.ndarray  # (natom, 3) a.u., molecular frame of the dimer
    alpha: np.ndarray  # (natom, 3, 3) bohr^3
    z: tuple[int, ...]


def unpack_symmetric(packed: np.ndarray) -> np.ndarray:
    """(natom, 6) packed xx,xy,xz,yy,yz,zz -> (natom, 3, 3).

    This is the inverse of the pipeline's `pack_symmetric_tensor`; the ordering is the
    published contract, not a convention chosen here.
    """
    packed = np.asarray(packed, dtype=float)
    if packed.ndim != 2 or packed.shape[1] != 6:
        raise ValueError(f"expected (natom, 6) packed tensors, got {packed.shape}")
    natom = packed.shape[0]
    full = np.empty((natom, 3, 3))
    xx, xy, xz, yy, yz, zz = (packed[:, index] for index in range(6))
    full[:, 0, 0], full[:, 1, 1], full[:, 2, 2] = xx, yy, zz
    full[:, 0, 1] = full[:, 1, 0] = xy
    full[:, 0, 2] = full[:, 2, 0] = xz
    full[:, 1, 2] = full[:, 2, 1] = yz
    return full


def place_monomer(
    target_positions_bohr: np.ndarray,
    charges: np.ndarray,
    dipoles: np.ndarray,
    alpha: np.ndarray,
) -> Fragment:
    """Transfer reviewed-monomer properties onto a placed copy of the rigid monomer.

    The reviewed monomer is superimposed on the target atoms and the *reviewed* rigid
    geometry is used, so both fragments in every dimer are byte-identical copies of the
    one molecule whose properties were computed. Using the target's own slightly
    different internal geometry instead would mean the transferred alpha and Cn belonged
    to a molecule that is not the one being placed.

    Properties transform as their rank demands: charges are invariant, dipoles rotate
    once, and the polarizability rotates twice. Skipping the alpha rotation is the classic
    way to get a plausible-looking but wrong anisotropic force field.
    """
    reviewed_centroid = REVIEWED_MONOMER_BOHR.mean(axis=0)
    target_centroid = target_positions_bohr.mean(axis=0)
    rotation = kabsch_rotation(
        REVIEWED_MONOMER_BOHR - reviewed_centroid, target_positions_bohr - target_centroid
    )
    positions = (REVIEWED_MONOMER_BOHR - reviewed_centroid) @ rotation.T + target_centroid
    return Fragment(
        positions=positions,
        charges=np.asarray(charges, dtype=float),
        dipoles=np.asarray(dipoles, dtype=float) @ rotation.T,
        alpha=np.einsum("ip,apq,jq->aij", rotation, alpha, rotation),
        z=MONOMER_Z,
    )


def dimer_scan_geometries(
    distances_angstrom: np.ndarray,
) -> list[tuple[float, np.ndarray, np.ndarray]]:
    """Rigid O-O scan built from the S22 dimer.

    Both monomers keep their S22 orientation; the acceptor is translated along the O-O
    unit vector. This changes exactly one coordinate of the surface, so a curve computed
    against it is interpretable.
    """
    geometry = S22_WATER_DIMER_ANGSTROM * BOHR_PER_ANGSTROM
    donor = geometry[:3]
    acceptor = geometry[3:]
    axis = acceptor[0] - donor[0]
    reference = float(np.linalg.norm(axis))
    axis = axis / reference
    out = []
    for distance in distances_angstrom:
        shift = (distance * BOHR_PER_ANGSTROM - reference) * axis
        out.append((float(distance), donor.copy(), acceptor + shift))
    return out


def orientation_probes(distance_angstrom: float) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Fixed-R rotations of the acceptor about the O-O axis.

    An isotropic model is *exactly* invariant to this for dispersion, and nearly so for
    induction. Any spread across these rows is anisotropy, which is the part of the model
    the isotropic Cn comparison cannot see at all.
    """
    geometry = S22_WATER_DIMER_ANGSTROM * BOHR_PER_ANGSTROM
    donor = geometry[:3]
    acceptor = geometry[3:]
    axis = acceptor[0] - donor[0]
    reference = float(np.linalg.norm(axis))
    unit = axis / reference
    shift = (distance_angstrom * BOHR_PER_ANGSTROM - reference) * unit
    acceptor = acceptor + shift
    origin = acceptor[0]

    out = []
    for degrees in (0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0):
        angle = math.radians(degrees)
        cross = np.array(
            [[0.0, -unit[2], unit[1]], [unit[2], 0.0, -unit[0]], [-unit[1], unit[0], 0.0]]
        )
        rotation = (
            np.eye(3) + math.sin(angle) * cross + (1.0 - math.cos(angle)) * (cross @ cross)
        )
        rotated = (acceptor - origin) @ rotation.T + origin
        out.append((f"{degrees:.0f} deg", donor.copy(), rotated))
    return out


# --------------------------------------------------------------------------------------
# Induction: point induced dipoles, self-consistent, anisotropic alpha
# --------------------------------------------------------------------------------------


def thole_lambdas(
    separation: float, alpha_i: float, alpha_j: float, damping: float = THOLE_A
) -> tuple[float, float]:
    """Thole "mutual" lambda_3, lambda_5 (apnet_pt/multipole.py:1374).

    `alpha_i`/`alpha_j` are *isotropic* polarizabilities. For an anisotropic site the
    caller passes trace/3: Thole's screening is a scalar range parameter derived from a
    smeared charge distribution and has no tensor generalisation, so using the mean is the
    only defensible reading. It is also the choice that makes the isotropic limit of this
    solver reduce exactly to qcmlforge's, which `validate` checks.
    """
    product = alpha_i * alpha_j
    if product <= 0.0:
        return 1.0, 1.0
    scaled = separation / product ** (1.0 / 6.0)
    cubed = damping * scaled**3
    decay = math.exp(-cubed)
    return 1.0 - decay, 1.0 - (1.0 + cubed) * decay


def dipole_field_tensors(
    positions: np.ndarray, alpha_isotropic: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Damped T1 (site, site, 3) and T2 (site, site, 3, 3) over all site pairs.

    Sign convention: T1[i, j] contracted with q_j gives the field at i from a charge at j,
    and T2[i, j] contracted with mu_j gives the field at i from a dipole at j. Both
    diagonals are zero.
    """
    count = len(positions)
    t1 = np.zeros((count, count, 3))
    t2 = np.zeros((count, count, 3, 3))
    for i in range(count):
        for j in range(count):
            if i == j:
                continue
            delta = positions[i] - positions[j]
            distance = float(np.linalg.norm(delta))
            lam3, lam5 = thole_lambdas(distance, alpha_isotropic[i], alpha_isotropic[j])
            t1[i, j] = lam3 * delta / distance**3
            t2[i, j] = (
                3.0 * lam5 * np.outer(delta, delta) - lam3 * distance**2 * np.eye(3)
            ) / distance**5
    return t1, t2


def induction_energy(
    fragment_a: Fragment, fragment_b: Fragment, isotropic: bool = False
) -> float:
    """Self-consistent induction energy in kcal/mol. Negative is attractive.

    Model, matching `dimer_induced_dipole_torch`:
      * the *inducing* field comes only from the other monomer's permanent charges and
        dipoles (intermolecular sources);
      * mutual polarization is then solved over *all* site pairs, intramolecular included,
        Thole-damped, with no 1-2/1-3 exclusion. For water every intramolecular pair is
        1-2 or 1-3, so this is not a detail: the intramolecular response is fully present
        and fully damped;
      * E = -1/2 sum_i mu_i . F_i^perm.

    Solved as a single linear system (I - A T2) mu = A F^perm rather than by Jacobi
    iteration with SOR mixing. Same fixed point, but it cannot silently return a
    half-converged answer, and a near-singular response matrix -- polarization catastrophe,
    which Thole damping exists to prevent but does not guarantee -- shows up as a condition
    number instead of as a plausible wrong number. That is checked below.
    """
    positions = np.vstack([fragment_a.positions, fragment_b.positions])
    charges = np.concatenate([fragment_a.charges, fragment_b.charges])
    permanent = np.vstack([fragment_a.dipoles, fragment_b.dipoles])
    alpha = np.concatenate([fragment_a.alpha, fragment_b.alpha])
    if isotropic:
        alpha = np.einsum("aij,ij->a", alpha, np.eye(3))[:, None, None] / 3.0 * np.eye(3)
    count = len(positions)
    split = len(fragment_a.positions)
    alpha_isotropic = np.einsum("aii->a", alpha) / 3.0

    t1, t2 = dipole_field_tensors(positions, alpha_isotropic)

    # Intermolecular mask: the permanent field that *drives* the induction.
    intermolecular = np.zeros((count, count), dtype=bool)
    intermolecular[:split, split:] = True
    intermolecular[split:, :split] = True

    field = np.einsum("ij,ijx,j->ix", intermolecular, t1, charges) + np.einsum(
        "ij,ijxy,jy->ix", intermolecular, t2, permanent
    )

    # (I - A T2) mu = A F, with A block diagonal.
    matrix = np.eye(3 * count)
    for i in range(count):
        for j in range(count):
            if i == j:
                continue
            matrix[3 * i : 3 * i + 3, 3 * j : 3 * j + 3] -= alpha[i] @ t2[i, j]
    condition = np.linalg.cond(matrix)
    if condition > 1.0e6:
        raise RuntimeError(
            f"induction response matrix is near-singular (cond {condition:.3e}); the "
            "Thole-damped mutual polarization has run away and any energy from it is "
            "meaningless"
        )
    rhs = np.einsum("aij,aj->ai", alpha, field).reshape(-1)
    induced = np.linalg.solve(matrix, rhs).reshape(count, 3)

    return -0.5 * float(np.einsum("ix,ix->", induced, field)) * HARTREE_TO_KCAL


# --------------------------------------------------------------------------------------
# Dispersion: distributed Cn with Becke-Johnson damping
# --------------------------------------------------------------------------------------


@dataclass
class DispersionBreakdown:
    c6: float
    c8: float
    c10: float

    @property
    def d3_comparable(self) -> float:
        """C6 + C8 only -- the orders stock D3 actually has."""
        return self.c6 + self.c8

    @property
    def total(self) -> float:
        return self.c6 + self.c8 + self.c10


def dispersion_energy(
    fragment_a: Fragment,
    fragment_b: Fragment,
    coefficients: dict[str, np.ndarray],
    r0_mode: str = "r4r2",
) -> DispersionBreakdown:
    """Intermolecular pairwise BJ-damped dispersion, in kcal/mol, per order.

    `coefficients[n][i, j]` is the distributed coefficient between monomer site i and
    monomer site j. Because both fragments are copies of the same monomer, the site
    indices index the same (natom, natom) table on both sides -- which is exactly what a
    distributed Cn table is for.

    Intramonomer dispersion is excluded, matching `qcml_dftd3.d3`, which is
    intermolecular-only.
    """
    if r0_mode not in {"r4r2", "c8c6"}:
        raise ValueError(f"unknown r0 mode {r0_mode!r}")
    totals = {6: 0.0, 8: 0.0, 10: 0.0}
    scale = {6: D3_S6, 8: D3_S8, 10: D3_S10}
    for i, position_i in enumerate(fragment_a.positions):
        for j, position_j in enumerate(fragment_b.positions):
            distance = float(np.linalg.norm(position_i - position_j))
            c6 = float(coefficients["C6"][i, j])
            c8 = float(coefficients["C8"][i, j])
            c10 = float(coefficients["C10"][i, j])
            if r0_mode == "r4r2":
                qq = 3.0 * R4R2[fragment_a.z[i]] * R4R2[fragment_b.z[j]]
                radius = D3_A1 * math.sqrt(qq) + D3_A2
            else:
                # Becke and Johnson's original critical radius. Available to us because we
                # have genuine distributed C8 and C6; stock D3 approximates it with r4r2.
                radius = D3_A1 * math.sqrt(c8 / c6) + D3_A2
            for order, coefficient in ((6, c6), (8, c8), (10, c10)):
                totals[order] -= (
                    scale[order]
                    * coefficient
                    / (distance**order + radius**order)
                )
    return DispersionBreakdown(
        c6=totals[6] * HARTREE_TO_KCAL,
        c8=totals[8] * HARTREE_TO_KCAL,
        c10=totals[10] * HARTREE_TO_KCAL,
    )


# --------------------------------------------------------------------------------------
# Stage 1: our own monomer properties
# --------------------------------------------------------------------------------------


def stage_monomer(functional: str = "pbe0") -> None:
    """Run the pipeline plus MBIS on the reviewed monomer and cache the result.

    Expensive (aug-cc-pVTZ, three SCFs, 590/99 grid). Cached so the PES stage is free.

    The permanent multipoles come from MBIS on the *same* wavefunction, at the same level,
    so they are consistent with our polarizabilities. They are then shared unchanged by
    every arm of the comparison -- CamCASP's distributed multipoles are a separate model
    we would have to validate independently, and mixing them in would confound the
    polarizability comparison with a multipole comparison.
    """
    import psi4
    from psi4.driver.procrouting import atomic_polarizability as native_driver

    WORK_DIR.mkdir(exist_ok=True)
    psi4.core.clean_variables()
    psi4.set_output_file(str(WORK_DIR / "psi4.out"), False)
    psi4.set_options({"atomic_polarizability_partition": "ISA", **PARITY_PROTOCOL})
    molecule = psi4.geometry(
        """
0 1
O  0.00000000  0.0  0.00000000
H -1.45365196  0.0 -1.12168732
H  1.45365196  0.0 -1.12168732
symmetry c1
no_com
no_reorient
units bohr
"""
    )

    print("running the distributed-property pipeline (this is the slow part)...")
    wfn = native_driver.atomic_polarizabilities(molecule=molecule, functional=functional)

    print("running MBIS for the permanent multipoles...")
    scf_energy, scf_wfn = psi4.energy(f"scf/{PARITY_PROTOCOL['basis']}", return_wfn=True)
    psi4.oeprop(scf_wfn, "MBIS_CHARGES", title="monomer")

    payload = {
        "provenance": {
            "functional": functional,
            "protocol": PARITY_PROTOCOL,
            "psi4_version": psi4.__version__,
            "partition": "ISA",
            "scf_energy": float(scf_energy),
            "note": (
                "molecular-frame packed Cartesian (xx, xy, xz, yy, yz, zz); atom order "
                "O, H1, H2; alpha in bohr^3; Cn in hartree bohr^n; MBIS multipoles in a.u."
            ),
        },
        "alpha": np.asarray(wfn.array_variable("ATOMIC POLARIZABILITIES")).tolist(),
        "mbis_charges": np.asarray(
            scf_wfn.array_variable("MBIS CHARGES")
        ).reshape(-1).tolist(),
        "mbis_dipoles": np.asarray(scf_wfn.array_variable("MBIS DIPOLES")).tolist(),
    }
    for order in ("C6", "C8", "C10", "C12"):
        payload[order] = np.asarray(
            wfn.array_variable(f"ATOMIC {order}")
        ).tolist()

    CACHE_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {CACHE_PATH}")
    for order in ("C6", "C8", "C10", "C12"):
        print(f"  {order}[O,O] = {payload[order][0][0]:.6g}")
    print(f"  MBIS charges = {payload['mbis_charges']}")


# --------------------------------------------------------------------------------------
# Stage 2: the surface
# --------------------------------------------------------------------------------------


def load_arms() -> dict[str, dict]:
    """Assemble the comparison arms, all sharing one set of permanent multipoles."""
    if not CACHE_PATH.exists():
        raise SystemExit(
            f"missing {CACHE_PATH.name}; run the `monomer` stage under the Psi4 env first"
        )
    cache = json.loads(CACHE_PATH.read_text())
    provenance = cache.get("provenance", {})
    if provenance.get("protocol") != PARITY_PROTOCOL:
        raise SystemExit(
            f"{CACHE_PATH.name} was not produced by the reviewed parity protocol "
            "(or is a smoke-test stub). Delete it and rerun the `monomer` stage. A cache "
            "from a cheaper protocol is a different model and its deltas mean nothing."
        )
    charges = np.asarray(cache["mbis_charges"], dtype=float)
    dipoles = np.asarray(cache["mbis_dipoles"], dtype=float)

    arms = {
        "ours": {
            "label": "this pipeline (ISA partition)",
            "alpha": unpack_symmetric(np.asarray(cache["alpha"])),
            "coefficients": {
                order: np.asarray(cache[order], dtype=float)
                for order in ("C6", "C8", "C10", "C12")
            },
        }
    }
    for key, reference in CAMCASP.items():
        arms[key] = {
            "label": reference["label"],
            "alpha": unpack_symmetric(np.asarray(reference["alpha"])),
            "coefficients": {
                order: np.asarray(reference[order], dtype=float)
                for order in ("C6", "C8", "C10", "C12")
            },
        }
    for arm in arms.values():
        arm["charges"] = charges
        arm["dipoles"] = dipoles
    return arms


def evaluate(arm: dict, donor: np.ndarray, acceptor: np.ndarray, r0_mode: str) -> dict:
    fragment_a = place_monomer(donor, arm["charges"], arm["dipoles"], arm["alpha"])
    fragment_b = place_monomer(acceptor, arm["charges"], arm["dipoles"], arm["alpha"])
    dispersion = dispersion_energy(
        fragment_a, fragment_b, arm["coefficients"], r0_mode=r0_mode
    )
    return {
        "induction": induction_energy(fragment_a, fragment_b),
        "induction_isotropic": induction_energy(fragment_a, fragment_b, isotropic=True),
        "dispersion": dispersion,
    }


def stage_pes(r0_mode: str) -> None:
    arms = load_arms()
    distances = np.array(
        [2.6, 2.8, 2.912, 3.0, 3.2, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0]
    )
    geometries = dimer_scan_geometries(distances)

    print(f"\nBJ radius mode: {r0_mode}")
    print("Reference arm for the deltas: CamCASP ISA-GRID, the partition-matched oracle.")
    print("The 2.912 A row is the S22 equilibrium separation.\n")

    header = (
        f"{'R(O-O)':>7}  {'E_ind ours':>10} {'E_ind ISA':>10} {'d_ind':>8} {'%':>7}   "
        f"{'E_disp ours':>11} {'E_disp ISA':>11} {'d_disp':>8} {'%':>7}   {'d_tot':>8}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for distance, donor, acceptor in geometries:
        ours = evaluate(arms["ours"], donor, acceptor, r0_mode)
        isa = evaluate(arms["isa_grid"], donor, acceptor, r0_mode)
        df = evaluate(arms["df"], donor, acceptor, r0_mode)

        ind_delta = ours["induction"] - isa["induction"]
        disp_ours = ours["dispersion"].d3_comparable
        disp_isa = isa["dispersion"].d3_comparable
        disp_delta = disp_ours - disp_isa

        def percent(delta: float, reference: float) -> float:
            return 100.0 * delta / reference if abs(reference) > 1.0e-12 else float("nan")

        print(
            f"{distance:7.3f}  {ours['induction']:10.4f} {isa['induction']:10.4f} "
            f"{ind_delta:8.4f} {percent(ind_delta, isa['induction']):7.2f}   "
            f"{disp_ours:11.4f} {disp_isa:11.4f} {disp_delta:8.4f} "
            f"{percent(disp_delta, disp_isa):7.2f}   {ind_delta + disp_delta:8.4f}"
        )
        rows.append(
            {
                "distance": distance,
                "ours": ours,
                "isa_grid": isa,
                "df": df,
                "ind_delta": ind_delta,
                "disp_delta": disp_delta,
            }
        )

    print("\nDispersion by order, ours vs ISA-GRID (kcal/mol). C10 is outside stock D3.")
    print(
        f"{'R(O-O)':>7}  {'C6 ours':>9} {'C6 ISA':>9} {'d':>8}  {'C8 ours':>9} "
        f"{'C8 ISA':>9} {'d':>8}  {'C10 ours':>9} {'C10 ISA':>9} {'d':>8}"
    )
    for row in rows:
        ours = row["ours"]["dispersion"]
        isa = row["isa_grid"]["dispersion"]
        print(
            f"{row['distance']:7.3f}  {ours.c6:9.4f} {isa.c6:9.4f} {ours.c6 - isa.c6:8.4f}  "
            f"{ours.c8:9.4f} {isa.c8:9.4f} {ours.c8 - isa.c8:8.4f}  "
            f"{ours.c10:9.4f} {isa.c10:9.4f} {ours.c10 - isa.c10:8.4f}"
        )

    print("\nWhat the WRONG oracle would have implied (C-DF arm, same kernels):")
    print(f"{'R(O-O)':>7}  {'E_ind DF':>10} {'E_disp DF':>11}  {'d_ind vs ISA':>13} {'d_disp vs ISA':>14}")
    for row in rows:
        df = row["df"]
        isa = row["isa_grid"]
        print(
            f"{row['distance']:7.3f}  {df['induction']:10.4f} "
            f"{df['dispersion'].d3_comparable:11.4f}  "
            f"{df['induction'] - isa['induction']:13.4f} "
            f"{df['dispersion'].d3_comparable - isa['dispersion'].d3_comparable:14.4f}"
        )

    print("\nAnisotropy: full-tensor vs isotropised alpha, same arm (kcal/mol).")
    print(f"{'R(O-O)':>7}  {'ours aniso':>11} {'ours iso':>10} {'d':>8}   {'ISA aniso':>10} {'ISA iso':>10} {'d':>8}")
    for row in rows:
        ours = row["ours"]
        isa = row["isa_grid"]
        print(
            f"{row['distance']:7.3f}  {ours['induction']:11.4f} "
            f"{ours['induction_isotropic']:10.4f} "
            f"{ours['induction'] - ours['induction_isotropic']:8.4f}   "
            f"{isa['induction']:10.4f} {isa['induction_isotropic']:10.4f} "
            f"{isa['induction'] - isa['induction_isotropic']:8.4f}"
        )

    equilibrium = 2.912
    print(
        f"\nOrientation probes at R(O-O) = {equilibrium} A: acceptor rotated about the "
        "O-O axis.\nDispersion from an isotropic-Cn model is invariant to this by "
        "construction, so any\nspread is real orientation dependence."
    )
    print(f"{'rotation':>10}  {'E_ind ours':>10} {'E_ind ISA':>10} {'d':>8}   {'E_disp ours':>11} {'E_disp ISA':>11} {'d':>8}")
    for name, donor, acceptor in orientation_probes(equilibrium):
        ours = evaluate(arms["ours"], donor, acceptor, r0_mode)
        isa = evaluate(arms["isa_grid"], donor, acceptor, r0_mode)
        print(
            f"{name:>10}  {ours['induction']:10.4f} {isa['induction']:10.4f} "
            f"{ours['induction'] - isa['induction']:8.4f}   "
            f"{ours['dispersion'].d3_comparable:11.4f} "
            f"{isa['dispersion'].d3_comparable:11.4f} "
            f"{ours['dispersion'].d3_comparable - isa['dispersion'].d3_comparable:8.4f}"
        )


# --------------------------------------------------------------------------------------
# Stage 3: cross-check the kernels against qcmlforge itself
# --------------------------------------------------------------------------------------


def stage_validate() -> None:
    """Check both kernels against qcmlforge. Requires the qcml_pt210 env.

    The kernels here were written from qcmlforge's conventions, not copied from it, so
    "follows the same conventions" is a claim that needs testing rather than asserting.
    Two independent checks:

      1. dispersion -- feed our BJ sum the *reference* C6/C8 that `qcml_dftd3.d3` computes
         internally and require agreement to float noise. This tests the damping form,
         the parameter set and the unit conversion in one shot.
      2. induction -- run our solver with isotropised alpha against
         `dimer_induced_dipole_torch`. Exact agreement is not expected: it iterates
         Jacobi+SOR to 1e-8 while we solve the system directly. The tolerance below is on
         the fixed point, not on the iteration.
    """
    sys.path.insert(0, "/home/awallace43/gits/qcmlforge/src")
    try:
        import torch
        from qcml_dftd3 import d3 as d3_module
    except ImportError as error:
        raise SystemExit(
            "validate needs the qcmlforge env:\n"
            "  /home/awallace43/miniconda3/envs/qcml_pt210/bin/python water-pes-aim.py validate\n"
            f"(import failed: {error})"
        )

    geometry = S22_WATER_DIMER_ANGSTROM * BOHR_PER_ANGSTROM
    donor, acceptor = geometry[:3], geometry[3:]

    print("check 1: BJ dispersion sum against qcml_dftd3 reference coefficients")
    r4r2 = d3_module.r4r2.R4R2()
    for element, expected in R4R2.items():
        actual = float(r4r2[element])
        status = "ok" if abs(actual - expected) < 1.0e-14 else "MISMATCH"
        print(f"  r4r2[{element}] = {actual:.8f} vs {expected:.8f}  {status}")
        if status == "MISMATCH":
            raise SystemExit("r4r2 table disagrees; the BJ radius would be wrong")

    print("\ncheck 2: parameter set actually used by d3()")
    params = d3_module.params_intermolecular_saptpbe0_d3i
    for name, expected in (("s6", D3_S6), ("s8", D3_S8), ("a1", D3_A1), ("a2", D3_A2)):
        actual = float(params[name])
        status = "ok" if abs(actual - expected) < 1.0e-12 else "MISMATCH"
        print(f"  {name} = {actual} vs {expected}  {status}")
        if status == "MISMATCH":
            raise SystemExit("damping parameters disagree with qcmlforge")

    print("\ncheck 3: our BJ sum reproduces d3()'s functional form on a single pair")
    # One O-O pair, reference C6 from qcmlforge, C8 built its way. If our sum agrees here
    # the form and the units are right; the PES then differs from d3() only in the
    # coefficients, which is the entire point.
    distance = float(np.linalg.norm(donor[0] - acceptor[0]))
    c6 = 25.0
    qq = 3.0 * float(r4r2[8]) * float(r4r2[8])
    c8 = c6 * qq
    radius = D3_A1 * math.sqrt(qq) + D3_A2
    ours = -(
        D3_S6 * c6 / (distance**6 + radius**6) + D3_S8 * c8 / (distance**8 + radius**8)
    ) * HARTREE_TO_KCAL
    theirs = -(
        D3_S6
        * c6
        * float(
            d3_module.rational_damping(
                6, torch.tensor([distance], dtype=torch.float64),
                torch.tensor([qq], dtype=torch.float64), params,
            )
        )
        + D3_S8
        * c8
        * float(
            d3_module.rational_damping(
                8, torch.tensor([distance], dtype=torch.float64),
                torch.tensor([qq], dtype=torch.float64), params,
            )
        )
    ) * HARTREE_TO_KCAL
    print(f"  ours   = {ours:.12f} kcal/mol")
    print(f"  theirs = {theirs:.12f} kcal/mol")
    print(f"  delta  = {abs(ours - theirs):.3e}")
    if abs(ours - theirs) > 1.0e-10:
        raise SystemExit("BJ dispersion kernel disagrees with qcml_dftd3.rational_damping")
    print("  ok")

    print("\ncheck 4: Thole lambdas against apnet_pt.multipole.thole_damping_torch")
    from apnet_pt import multipole as qcml_multipole

    # Sampled where the damping is actually active. At the H-bond separation lambda is
    # indistinguishable from 1, so a check taken only there would pass with the damping
    # deleted entirely.
    worst = 0.0
    for separation in (1.5, 2.0, 2.8, 3.5, 5.3):
        alpha_i, alpha_j = 8.3837, 0.4842
        lam3, lam5 = thole_lambdas(separation, alpha_i, alpha_j)
        _, their3, their5 = qcml_multipole.thole_damping_torch(
            torch.tensor([separation], dtype=torch.float64),
            torch.tensor([alpha_i], dtype=torch.float64),
            torch.tensor([alpha_j], dtype=torch.float64),
            THOLE_A,
        )
        delta = max(abs(lam3 - float(their3)), abs(lam5 - float(their5)))
        worst = max(worst, delta)
        print(
            f"  r = {separation:4.1f} bohr  lambda3 {lam3:.10f} lambda5 {lam5:.10f}  "
            f"delta {delta:.2e}"
        )
    print(f"  worst delta = {worst:.3e}")
    if worst > 1.0e-12:
        raise SystemExit("Thole damping disagrees with qcmlforge")
    print("  ok")

    print("\ncheck 5: induction solver against dimer_induced_dipole_torch")
    _validate_induction(torch, qcml_multipole, donor, acceptor)

    print("\nall checks passed")


def _validate_induction(torch, qcml_multipole, donor, acceptor) -> None:
    """Our direct-solve induction against qcmlforge's Jacobi+SOR iteration.

    Isotropic alpha on both sides, since that is the only case qcmlforge can express.
    The two are the same fixed point reached two different ways, so they should agree far
    more closely than either agrees with any physical reference.
    """
    import contextlib
    import io

    charges = np.array([-0.70, 0.35, 0.35])
    dipoles = np.array([[0.0, 0.0, 0.05], [0.0, 0.0, -0.02], [0.0, 0.0, -0.02]])
    alpha_isotropic = np.array([8.3837, 0.4842, 0.4842])
    alpha = alpha_isotropic[:, None, None] * np.eye(3)

    fragment_a = Fragment(donor, charges, dipoles, alpha, MONOMER_Z)
    fragment_b = Fragment(acceptor, charges, dipoles, alpha, MONOMER_Z)
    ours = induction_energy(fragment_a, fragment_b)

    def edges(count_a: int, count_b: int, offset: bool):
        source, target = [], []
        for i in range(count_a):
            for j in range(count_b):
                if not offset and i == j:
                    continue
                source.append(i)
                target.append(j)
        return (
            torch.tensor(source, dtype=torch.long),
            torch.tensor(target, dtype=torch.long),
        )

    def tensor(value):
        return torch.tensor(np.asarray(value), dtype=torch.float64)

    ab_source, ab_target = edges(3, 3, True)
    aa_source, aa_target = edges(3, 3, False)

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        theirs = qcml_multipole.dimer_induced_dipole_torch(
            torch.tensor(MONOMER_Z, dtype=torch.long),
            tensor(donor / BOHR_PER_ANGSTROM),
            tensor(charges),
            tensor(dipoles),
            tensor(np.zeros((3, 3, 3))),
            torch.tensor(MONOMER_Z, dtype=torch.long),
            tensor(acceptor / BOHR_PER_ANGSTROM),
            tensor(charges),
            tensor(dipoles),
            tensor(np.zeros((3, 3, 3))),
            ab_source,
            ab_target,
            aa_source,
            aa_source,
            aa_target,
            aa_target,
            tensor(np.ones(3)),
            tensor(np.ones(3)),
            tensor(np.ones(3)),
            tensor(np.ones(3)),
            atom_polarizabilities_A=tensor(alpha_isotropic),
            atom_polarizabilities_B=tensor(alpha_isotropic),
        )
    theirs = float(theirs.sum())

    print(f"  ours (direct solve)      = {ours:.10f} kcal/mol")
    print(f"  theirs (Jacobi + SOR)    = {theirs:.10f} kcal/mol")
    delta = abs(ours - theirs)
    print(f"  delta                    = {delta:.3e} kcal/mol")
    if delta > 1.0e-6:
        raise SystemExit(
            "induction solvers disagree beyond iteration noise; the models are not the "
            "same, which invalidates the claim that this kernel follows qcmlforge"
        )
    print("  ok")


# --------------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=["monomer", "pes", "validate"])
    parser.add_argument(
        "--r0",
        choices=["r4r2", "c8c6", "both"],
        default="both",
        help="BJ radius: stock r4r2 approximation, our own sqrt(C8/C6), or both",
    )
    arguments = parser.parse_args()

    if arguments.stage == "monomer":
        stage_monomer()
    elif arguments.stage == "validate":
        stage_validate()
    else:
        modes = ["r4r2", "c8c6"] if arguments.r0 == "both" else [arguments.r0]
        for mode in modes:
            print("=" * 100)
            stage_pes(mode)


if __name__ == "__main__":
    main()
