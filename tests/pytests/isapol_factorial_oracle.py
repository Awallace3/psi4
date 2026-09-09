"""Table-free angular algebra shared by independent ISA tests.

Moved unchanged from test_isapol_recoupled_dispersion.py. No Psi4 import,
reference files, production tables, or runtime dependencies beyond NumPy.
Psi4 additions: Copyright (c) 2026 The Psi4 Developers. LGPL-3.0-only.
"""
import math

import numpy as np


def cg(j1,m1,j2,m2,J,M):
    """Independent factorial Wigner 3j formula (Condon--Shortley complex basis)."""
    if m1+m2 != M or not abs(j1-j2) <= J <= j1+j2 or abs(M)>J:
        return 0.
    f = math.factorial
    pref = math.sqrt((2*J+1)*f(j1+j2-J)*f(j1-j2+J)*f(-j1+j2+J)/f(j1+j2+J+1))
    pref *= math.sqrt(f(j1+m1)*f(j1-m1)*f(j2+m2)*f(j2-m2)*f(J+M)*f(J-M))
    total = 0.
    for z in range(j1+j2+J+1):
        den = (z,j1+j2-J-z,j1-m1-z,j2+m2-z,J-j2+m1+z,J-j1-m2+z)
        if min(den) >= 0:
            total += (-1)**z/math.prod(f(k) for k in den)
    return pref*total


def real_transform(l):
    """Real Racah no-CS functions in CS |m>, m=-l..l. Rank1 is z,x,y."""
    U = np.zeros((2*l+1,2*l+1),complex)
    U[0,l] = 1.
    for m in range(1,l+1):
        U[2*m-1,l+m], U[2*m-1,l-m] = (-1)**m/math.sqrt(2), 1/math.sqrt(2)
        U[2*m,l+m], U[2*m,l-m] = (-1)**m/(1j*math.sqrt(2)), -1/(1j*math.sqrt(2))
    return U


def independent_g(l,p):
    result = np.zeros(((l+p+1)**2,2*l+1,2*p+1),complex)
    for L in range(abs(l-p),l+p+1):
        spherical = np.zeros((2*L+1,2*l+1,2*p+1))
        for m in range(-l,l+1):
            for n in range(-p,p+1):
                if abs(m+n)<=L:
                    spherical[m+n+L,m+l,n+p] = cg(l,m,p,n,L,m+n)
        result[L*L:(L+1)**2] = np.einsum('vM,Mab,ka,qb->vkq',
            real_transform(L), spherical, real_transform(l).conj(), real_transform(p).conj())
    return result
