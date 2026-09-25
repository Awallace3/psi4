# Copyright (c) 2007-2026 The Psi4 Developers.
# SPDX-License-Identifier: LGPL-3.0-only
"""Optional parity of the atom-defined frames against external Tinker and OpenMM.

Skipped by default. The Tinker check needs ``PSI4_TINKER_SOURCE`` pointing at a
built Tinker ``source/`` directory (``libtinker.a`` plus its ``.mod`` files),
gfortran and FFTW (``PSI4_TINKER_FFTW_LIBDIR`` if FFTW is not on the default
path). It links a small black-box driver that calls Tinker's own
``chkpole``/``rotpole``/``rotmat``; nothing of Tinker is copied here. The
OpenMM check needs ``openmm`` importable. Validated against Tinker 25.5 and
OpenMM 8.2.
"""
import os
import shutil
import subprocess

import numpy as np
import pytest

_DRIVER = """\
      program dumpframes
      use atoms
      use mpole
      implicit none
      integer i,j,k,ixyz
      real*8 a(3,3)
      logical planar
      call initial
      call getcart (ixyz)
      call mechanic
      call chkpole
      call rotpole ('MPOLE')
      do i = 1, n
         if (pollist(i) .ne. 0) then
            call rotmat (i,a,planar)
            write (*,'(a,i6,1x,a8)') 'SITE',i,polaxe(i)
            write (*,'(a,9es25.16)') 'FRAME',((a(j,k),k=1,3),j=1,3)
            write (*,'(a,13es25.16)') 'LOCAL',(pole(j,i),j=1,13)
            write (*,'(a,13es25.16)') 'GLOBAL',(rpole(j,i),j=1,13)
         end if
      end do
      call final
      end
"""

# (symbols, Angstrom geometry, bonds, one (kind, z, x, y) per site).
_BASE = {
    "water": (["O", "H", "H"], [[0., 0., .117], [0., .757, -.467], [0., -.757, -.467]],
              [(0, 1), (0, 2)],
              [("bisector", 1, 2, None), ("z_then_x", 0, 2, None), ("z_then_x", 0, 1, None)]),
    "ammonia": (["N", "H", "H", "H"],
                [[0., 0., .116], [0., .939, -.271], [.813, -.470, -.271], [-.813, -.470, -.271]],
                [(0, 1), (0, 2), (0, 3)],
                [("three_fold", 1, 2, 3), ("z_then_x", 0, 2, None), ("z_only", 0, None, None),
                 ("z_bisect", 0, 1, 2)]),
    "methylamine": (["C", "N", "H", "H", "H", "H", "H"],
                    [[-.702, .018, 0.], [.757, -.066, 0.], [-1.09, 1.036, 0.],
                     [-1.101, -.482, .889], [-1.101, -.482, -.889], [1.130, .421, .810],
                     [1.130, .421, -.810]],
                    [(0, 1), (0, 2), (0, 3), (0, 4), (1, 5), (1, 6)],
                    [("z_then_x", 1, 2, 3), ("z_bisect", 0, 5, 6), ("z_then_x", 0, 1, None),
                     ("bisector", 0, 1, None), ("z_only", 0, None, None),
                     ("z_then_x", 1, 0, None), ("three_fold", 1, 0, 5)]),
    "chfclbr": (["C", "H", "F", "Cl", "Br"],
                [[0., 0., 0.], [0., 0., 1.09], [1.30, 0., -.46], [-.83, 1.44, -.59],
                 [-.97, -1.68, -.64]],
                [(0, 1), (0, 2), (0, 3), (0, 4)],
                [("z_then_x", 2, 3, 4), ("z_then_x", 0, 2, None), ("z_only", 0, None, None),
                 ("bisector", 0, 2, None), ("global", None, None, None)]),
}
_TINKER_KIND = {"z_then_x": "Z-then-X", "bisector": "Bisector", "z_bisect": "Z-Bisect",
                "three_fold": "3-Fold", "z_only": "Z-Only", "global": "None"}


def _systems():
    rng = np.random.default_rng(101)
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    rotation = q*np.sign(np.diag(r))
    rotation *= np.linalg.det(rotation)
    out = {}
    for name, (symbols, xyz, bonds, frames) in _BASE.items():
        for tag, moved in (("lab", np.asarray(xyz)),
                           ("rotated", np.asarray(xyz) @ rotation.T + [1.3, -.4, 2.2])):
            out[f"{name}_{tag}"] = (symbols, moved, bonds, frames)
            if name == "chfclbr":
                out[f"{name}_mirror_{tag}"] = (symbols, moved*[1., -1., 1.], bonds, frames)
    for tag, c in (("below", .700), ("above", .714)):  # either side of Z_ONLY_SWITCH
        z = np.array([c, .6*np.sqrt(1-c*c), .8*np.sqrt(1-c*c)])
        out[f"hcl_{tag}"] = (["Cl", "H"], np.array([[0., 0., 0.], 1.27*z]), [(0, 1)],
                             [("z_only", 1, None, None), ("z_only", 0, None, None)])
    return out


def _multipoles(n):
    rng = np.random.default_rng(17)
    out = []
    for _ in range(n):
        b = rng.normal(size=(3, 3))
        out.append((rng.normal(size=3), b + b.T - np.trace(b)*2/3*np.eye(3)))
    return out


def _resolve(xyz, frames):
    from psi4.driver.procrouting.isapol_geometry import LocalFrame, resolve_local_frames
    return resolve_local_frames(xyz, [LocalFrame(i, k, z=z, x=x, y=y)
                                      for i, (k, z, x, y) in enumerate(frames)])


def _racah(dipole, quad):
    t = quad
    return np.concatenate(([dipole[2], dipole[0], dipole[1]],
                           [t[2, 2], 2*t[0, 2]/np.sqrt(3), 2*t[1, 2]/np.sqrt(3),
                            (t[0, 0]-t[1, 1])/np.sqrt(3), 2*t[0, 1]/np.sqrt(3)]))


def _tinker_refs(kind, z, x, y):
    # Tinker's multipole-parameter axis-type sign convention; atom type = index + 1.
    z, x, y = (None if v is None else v+1 for v in (z, x, y))
    if kind == "global":
        return [0, 0]
    if kind == "z_only":
        return [z, 0]
    if kind == "z_then_x":
        return [z, x] + ([] if y is None else [y])
    if kind == "bisector":
        return [z, -x]
    return [z, -x, -y] if kind == "z_bisect" else [-z, -x, -y]


@pytest.fixture(scope="module")
def dumpframes(tmp_path_factory):
    source = os.environ.get("PSI4_TINKER_SOURCE")
    if not source or not os.path.isfile(os.path.join(source, "libtinker.a")):
        pytest.skip("PSI4_TINKER_SOURCE does not name a built Tinker source directory")
    if shutil.which("gfortran") is None:
        pytest.skip("gfortran unavailable")
    work = tmp_path_factory.mktemp("tinker")
    (work/"dumpframes.f").write_text(_DRIVER)
    fftw = os.environ.get("PSI4_TINKER_FFTW_LIBDIR")
    subprocess.run(["gfortran", "-O2", "-fopenmp", f"-I{source}", "dumpframes.f", "-o",
                    "dumpframes", os.path.join(source, "libtinker.a")] +
                   ([f"-L{fftw}"] if fftw else []) + ["-lfftw3_threads", "-lfftw3"],
                   cwd=work, check=True, capture_output=True)
    return work/"dumpframes"


@pytest.mark.parametrize("name", sorted(_systems()))
def test_frames_and_rotated_moments_match_tinker(dumpframes, tmp_path, name):
    from psi4 import core
    from psi4.driver.procrouting.isapol_geometry import local_y_reflection
    symbols, xyz, bonds, frames = _systems()[name]
    poles = _multipoles(len(symbols))
    prm = ["forcefield TEST"] + [f'atom {i+1} {i+1} {s} "{s}{i+1}" 1 1.0 1'
                                 for i, s in enumerate(symbols)]
    for i, (frame, (d, t)) in enumerate(zip(frames, poles)):
        prm += [f"multipole {i+1} " + " ".join(map(str, _tinker_refs(*frame))) + " 0.0",
                " ".join(f"{v:.15f}" for v in d), f"{t[0,0]:.15f}",
                f"{t[1,0]:.15f} {t[1,1]:.15f}", f"{t[2,0]:.15f} {t[2,1]:.15f} {t[2,2]:.15f}"]
    (tmp_path/"t.prm").write_text("\n".join(prm)+"\n")
    (tmp_path/"t.key").write_text("parameters t.prm\nmultipoleterm only\n")
    neighbors = [[b+1 for a, b in bonds if a == i] + [a+1 for a, b in bonds if b == i]
                 for i in range(len(symbols))]
    (tmp_path/"t.xyz").write_text(f"{len(symbols)} {name}\n" + "".join(
        f"{i+1} {s} {r[0]:.15f} {r[1]:.15f} {r[2]:.15f} {i+1} {' '.join(map(str, nb))}\n"
        for i, (s, r, nb) in enumerate(zip(symbols, xyz, neighbors))))
    out = subprocess.run([str(dumpframes), "t.xyz", "-k", "t.key"], cwd=tmp_path, check=True,
                         capture_output=True, text=True).stdout
    sites = []
    for line in out.splitlines():
        fields = line.split()
        if fields and fields[0] == "SITE":
            sites.append({"kind": fields[2] if len(fields) > 2 else "None"})
        elif fields and fields[0] in ("FRAME", "LOCAL", "GLOBAL"):
            sites[-1][fields[0]] = np.array(fields[1:], dtype=float)
    assert len(sites) == len(symbols)

    resolved = _resolve(xyz, frames)
    signs = local_y_reflection(2)[1:]
    for i, (site, (d, t)) in enumerate(zip(sites, poles)):
        assert site["kind"] == _TINKER_KIND[frames[i][0]]
        np.testing.assert_allclose(resolved.frames[i], site["FRAME"].reshape(3, 3),
                                   atol=1e-12, rtol=0)
        # Tinker stores the dipole in e*Ang and the quadrupole in e*Ang^2/3.
        bohr = np.linalg.norm(site["LOCAL"][1:4])/np.linalg.norm(d)
        scale = np.concatenate(([bohr]*3, [bohr**2/3]*5))
        # A positive y-axis type is inverted exactly where the handedness is +1.
        local = _racah(d, t)*(signs if resolved.handedness[i] == 1 else 1.)
        np.testing.assert_allclose(
            _racah(site["LOCAL"][1:4], site["LOCAL"][4:].reshape(3, 3))/scale, local,
            atol=1e-11, rtol=0)
        rotation = np.asarray(core.isa_multipole_rotation(2, resolved.frames[i].tolist()))
        np.testing.assert_allclose(
            rotation[1:9, 1:9] @ local,
            _racah(site["GLOBAL"][1:4], site["GLOBAL"][4:].reshape(3, 3))/scale,
            atol=1e-11, rtol=0)


@pytest.mark.parametrize("name", sorted(_systems()))
def test_frames_match_openmm_up_to_documented_conventions(name):
    openmm = pytest.importorskip("openmm")
    from psi4.driver.procrouting.isapol_geometry import Z_ONLY_SWITCH
    amoeba = openmm.AmoebaMultipoleForce
    axis = {"z_then_x": amoeba.ZThenX, "bisector": amoeba.Bisector,
            "z_bisect": amoeba.ZBisect, "three_fold": amoeba.ThreeFold,
            "z_only": amoeba.ZOnly, "global": amoeba.NoAxisType}
    symbols, xyz, bonds, frames = _systems()[name]
    resolved = _resolve(xyz, frames)

    def lab(dipoles):
        system, force = openmm.System(), amoeba()
        force.setPolarizationType(amoeba.Direct)
        for (kind, z, x, y), d in zip(frames, dipoles):
            if kind == "three_fold":  # OpenMM seeds x from its x reference
                z, x = x, z
            system.addParticle(1.)
            force.addMultipole(0., list(d), [0.]*9, axis[kind], *(-1 if v is None else v
                                                                  for v in (z, x, y)),
                               .39, 0., 0.)
        system.addForce(force)
        context = openmm.Context(system, openmm.VerletIntegrator(1e-3),
                                 openmm.Platform.getPlatformByName("Reference"))
        context.setPositions(np.asarray(xyz)*.1)
        return np.array([[float(v) for v in mu]
                         for mu in force.getLabFramePermanentDipoles(context)])

    columns = [lab(np.tile(e, (len(symbols), 1))) for e in np.eye(3)]
    for i, (kind, z, *_) in enumerate(frames):
        if kind == "z_only":
            u = xyz[z] - xyz[i]
            if Z_ONLY_SWITCH < abs(u[0])/np.linalg.norm(u) <= .866:
                continue  # OpenMM switches lab axes at 0.866, Tinker at 0.707
        probed = np.column_stack([c[i] for c in columns])
        mirror = np.diag([1., -1., 1.]) if resolved.handedness[i] == 1 else np.eye(3)
        np.testing.assert_allclose(resolved.frames[i] @ mirror, probed, atol=1e-12, rtol=0)
