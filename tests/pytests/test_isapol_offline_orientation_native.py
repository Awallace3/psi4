"""Analytic controls through the adapter's actual core call seam."""
import itertools
import math
import unittest
import numpy as np
from psi4 import core
from psi4.driver.procrouting import isapol_offline_orientation as adapter


def rotation(axis, angle):
    u = np.array(axis, dtype=float)
    u /= np.linalg.norm(u)
    x, y, z = u
    k = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    return np.eye(3) + math.sin(angle)*k + (1-math.cos(angle))*(k@k)


F = rotation([1, 2, -1], .73)
G = rotation([-2, 1, 3], -.39)
Q = rotation([2, -1, 1], .52)


def site(a, label="S", origin=(0., 0., 0.), frame=F, ranks=(1, 2)):
    return dict(label=label, ranks=list(ranks), origin=list(origin), frame=frame,
                responses=[np.zeros_like(a), a])


def model(sites, q=np.eye(3), t=(0., 0., 0.)):
    return adapter.placed_model(core, sites, [0., 1.], q, t, "synthetic invariant")


def calc(a, b, weight=.37):
    return adapter.contract(core, a, b, [0., weight], 10)


def coeff(output):
    return np.array([[c["value"] for c in p["coefficients"]] for p in output["pairs"]])


def energy(output):
    return np.array([[c["energy"] for c in p["coefficients"]] for p in output["pairs"]])


def tensors():
    rng = np.random.default_rng(907)
    a, b = rng.normal(size=(8, 8)), rng.normal(size=(8, 8))
    return a+a.T, b+b.T


class InvariantTests(unittest.TestCase):
    def test_joint_noncommuting_rotation_translation_and_whole_model_exchange(self):
        aa, bb = tensors()
        a = [site(aa, "A0", [.1, .2, -.3]), site(.3*aa, "A1", [-.4, .7, -.1], G)]
        b = [site(bb, "B0", [3.2, -.7, 2.9], G), site(.7*bb, "B1", [2.6, -.4, 3.8])]
        self.assertFalse(np.allclose(F@G, G@F))
        ab = calc(model(a), model(b))
        rotated = calc(model(a, Q), model(b, Q))
        translated = calc(model(a, t=[4., -3., 2.]), model(b, t=[4., -3., 2.]))
        ba = calc(model(b), model(a))
        reverse = {(p["site_b"], p["site_a"]): p for p in ba["pairs"]}
        np.testing.assert_allclose(coeff(ab), coeff(rotated), rtol=4e-11, atol=4e-10)
        np.testing.assert_allclose(coeff(ab), coeff(translated), rtol=4e-11, atol=4e-10)
        for p in ab["pairs"]:
            r = reverse[p["site_a"], p["site_b"]]
            np.testing.assert_allclose([x["value"] for x in p["coefficients"]],
                                       [x["value"] for x in r["coefficients"]],
                                       rtol=3e-12, atol=3e-11)
        self.assertAlmostEqual(ab["total_energy"], rotated["total_energy"], places=11)
        self.assertAlmostEqual(ab["total_energy"], ba["total_energy"], places=11)

    def test_single_pair_distance_scaling_each_order_including_odd(self):
        aa, bb = tensors()
        a = model([site(aa)])
        r = np.array([2., -3., 5.])
        first = calc(a, model([site(bb, origin=r, frame=G)]))
        second = calc(a, model([site(bb, origin=2*r, frame=G)]))
        np.testing.assert_allclose(coeff(first), coeff(second), rtol=2e-13, atol=2e-12)
        np.testing.assert_allclose(energy(first), energy(second)*2.**np.arange(6, 11),
                                   rtol=2e-13, atol=2e-14)
        self.assertGreater(np.max(np.abs(coeff(first)[0, 1::2])), 1e-3)
        self.assertEqual([len(c["missing_rank_quadruples"])
                          for c in first["pairs"][0]["coefficients"]], [0, 0, 4, 16, 34])

    def test_independent_cartesian_dipole_control(self):
        # Matrices below are xyz Cartesian. Adapter input is z,x,y.
        aa = np.array([[2., .3, -.7], [.3, 1., .8], [-.7, .8, -1.]])
        bb = np.array([[1., -.6, .2], [-.6, 4., -.3], [.2, -.3, 2.]])
        r = np.array([.3, -.6, 1.2])
        u = r/np.linalg.norm(r)
        tau = np.eye(3)-3*np.outer(u, u)
        expected = .37*np.trace((F@aa@F.T)@tau@(G@bb@G.T)@tau.T)
        perm = [2, 0, 1]
        a = model([site(aa[np.ix_(perm, perm)], ranks=(1,))])
        b = model([site(bb[np.ix_(perm, perm)], origin=r, frame=G, ranks=(1,))])
        actual = calc(a, b)
        np.testing.assert_allclose(coeff(actual)[0, 0], expected, rtol=3e-13, atol=3e-13)
        np.testing.assert_allclose(actual["total_energy"],
                                   -expected/np.linalg.norm(r)**6, rtol=3e-13)
        np.testing.assert_allclose(coeff(actual)[0, 1:], 0., atol=0.)
        control = np.asarray(core.isa_anisotropic_interaction(1, 1, r).array_interface()[0])
        np.testing.assert_allclose(control, tau[np.ix_(perm, perm)]/np.linalg.norm(r)**3,
                                   rtol=2e-14, atol=2e-15)

    def test_axial_c7_literal_multiplicity_and_physical_reversal(self):
        a, b = np.zeros((8, 8)), np.zeros((8, 8))
        a[0, 0], b[0, 0] = 2., 3.
        a[0, 3] = a[3, 0] = .4
        b[0, 3] = b[3, 0] = -.7
        first = calc(model([site(a, frame=np.eye(3))]),
                     model([site(b, origin=[0., 0., 2.], frame=np.eye(3))]), weight=1.)
        reverse = calc(model([site(a, frame=np.eye(3))]),
                       model([site(b, origin=[0., 0., -2.], frame=np.eye(3))]), weight=1.)
        # tau11=-2,tau12=3,tau21=-3; both reciprocal mixed blocks occur.
        self.assertAlmostEqual(coeff(first)[0, 1], 31.2, places=12)
        np.testing.assert_allclose(coeff(reverse), coeff(first)*[1, -1, 1, -1, 1],
                                   rtol=3e-13, atol=3e-13)

    def test_static_exclusion_and_cp_normalization_analytic_lorentz(self):
        nodes, w = np.polynomial.legendre.leggauss(40)
        omega = (1+nodes)/(1-nodes)
        cp = w/(np.pi*(1-nodes)**2)  # Jacobian and 1/(2pi) exactly once.
        def lorentz(scale, pole, label, origin):
            return dict(label=label, ranks=[1], origin=origin, frame=np.eye(3),
                        responses=[np.eye(3)*1e100] +
                        [np.eye(3)*scale*pole*pole/(pole*pole+o*o) for o in omega])
        a = adapter.placed_model(core, [lorentz(2., .8, "A", [0., 0., 0.])],
                                 [0., *omega], np.eye(3), [0., 0., 0.], "Lorentz A")
        b = adapter.placed_model(core, [lorentz(3., 1.7, "B", [0., 0., 2.])],
                                 [0., *omega], np.eye(3), [0., 0., 0.], "Lorentz B")
        actual = adapter.contract(core, a, b, [0., *cp], 6)
        expected = 6*2*3*.8*1.7/(4*(.8+1.7))
        np.testing.assert_allclose(coeff(actual)[0, 0], expected, rtol=3e-13)

    def test_independent_orientation_average_trace_control(self):
        a = np.array([[2., .3, -.4], [.3, 5., .8], [-.4, .8, 11.]])
        b = np.diag([3., 7., 13.])
        frames = []
        for perm in itertools.permutations(range(3)):
            for signs in itertools.product([-1, 1], repeat=3):
                f = np.eye(3)[:, perm] @ np.diag(signs)
                if np.linalg.det(f) > 0:
                    frames.append(f)
        cycle = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
        aa = model([site(a, f"A{i}", frame=f, ranks=(1,)) for i, f in enumerate(frames)])
        bb = model([site(b, f"B{i}", origin=[.3, -.7, 1.1],
                         frame=np.linalg.matrix_power(cycle, i), ranks=(1,)) for i in range(3)])
        values = coeff(calc(aa, bb))[:, 0]
        self.assertGreater(np.ptp(values), 1.)
        np.testing.assert_allclose(values.mean(), .37*6*np.trace(a)/3*np.trace(b)/3,
                                   rtol=3e-13, atol=3e-12)
