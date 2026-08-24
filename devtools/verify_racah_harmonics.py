import numpy as np
from scipy.special import sph_harm_y as SH   # SH(l, m, theta, phi)


def racah_real(l, x, y, z):
    """Real Racah regular solid harmonics built from C_lm = sqrt(4pi/(2l+1)) Y_lm r^l,
    inverted with the transform the code states at atomic_polarizability.cc:5738-5741:
        C_{l,+m} = (-1)^m (R_lmc + i R_lms)/sqrt2
        C_{l,-m} =        (R_lmc - i R_lms)/sqrt2
    """
    r = np.sqrt(x * x + y * y + z * z)
    th = np.arccos(z / r)
    ph = np.arctan2(y, x)
    N = np.sqrt(4 * np.pi / (2 * l + 1)) * r**l
    C = {m: N * SH(l, m, th, ph) for m in range(-l, l + 1)}
    out = [C[0].real]
    for m in range(1, l + 1):
        Rc = ((-1) ** m * C[m] + C[-m]) / np.sqrt(2)
        Rs = ((-1) ** m * C[m] - C[-m]) / (1j * np.sqrt(2))
        out += [Rc.real, Rs.real]
    return out


def hardcoded(x, y, z):
    """Verbatim transcription of regular_harmonics(), atomic_polarizability.cc:213-229."""
    rho2 = x * x + y * y + z * z
    return [
        1.0, z, x, y,
        (3.0 * z * z - rho2) / 2.0, np.sqrt(3.0) * x * z, np.sqrt(3.0) * y * z,
        np.sqrt(3.0) * (x * x - y * y) / 2.0, np.sqrt(3.0) * x * y,
        (5.0 * z**3 - 3.0 * z * rho2) / 2.0,
        np.sqrt(3.0 / 8.0) * x * (5.0 * z * z - rho2),
        np.sqrt(3.0 / 8.0) * y * (5.0 * z * z - rho2),
        np.sqrt(15.0) * z * (x * x - y * y) / 2.0, np.sqrt(15.0) * x * y * z,
        np.sqrt(10.0) * x * (x * x - 3.0 * y * y) / 4.0,
        np.sqrt(10.0) * y * (3.0 * x * x - y * y) / 4.0,
    ]


labels = ["00", "10", "11c", "11s", "20", "21c", "21s", "22c", "22s",
          "30", "31c", "31s", "32c", "32s", "33c", "33s"]
rng = np.random.default_rng(7)
worst = np.zeros(16)
for _ in range(400):
    x, y, z = rng.normal(size=3) * 1.7
    ref = (racah_real(0, x, y, z) + racah_real(1, x, y, z)
           + racah_real(2, x, y, z) + racah_real(3, x, y, z))
    worst = np.maximum(worst, np.abs(np.array(hardcoded(x, y, z)) - np.array(ref)))

print(f"{'cmp':>5} {'max |code - Racah ref|':>24}")
for lab, w in zip(labels, worst):
    print(f"{lab:>5} {w:24.3e}   {'OK' if w < 1e-12 else '<<< MISMATCH'}")
print(f"\nsqrt(10)/4 = {np.sqrt(10)/4:.15f}   sqrt(5/8) = {np.sqrt(5/8):.15f}")
