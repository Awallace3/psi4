import pytest
import psi4
import numpy as np


def test_asapt():
    mol = psi4.geometry(
        """ 
    0 1
     C     0.0000000    0.0000000    3.0826195
     H     0.5868776    0.8381742    3.4463772
     H    -1.0193189    0.0891638    3.4463772
     H     0.0000000    0.0000000    1.9966697
     H     0.4324413   -0.9273380    3.4463772
     --
     0 1
     C     1.3932178    0.0362913   -0.6332803
     C     0.7280364   -1.1884015   -0.6333017
     C    -0.6651797   -1.2247077   -0.6332803
     C    -1.3932041   -0.0362972   -0.6333017
     C    -0.7280381    1.1884163   -0.6332803
     C     0.6651677    1.2246987   -0.6333017
     H     2.4742737    0.0644484   -0.6317240
     H     1.2929588   -2.1105409   -0.6317401
     H    -1.1813229   -2.1750081   -0.6317240
     H    -2.4742614   -0.0644647   -0.6317401
     H    -1.2929508    2.1105596   -0.6317240
     H     1.1813026    2.1750056   -0.6317401
    units angstrom
"""
    )
    psi4.set_options(
        {
            "basis": "jun-cc-pvdz",
            "df_basis_scf": "jun-cc-pvdz-jkfit",
            "df_basis_sapt": "jun-cc-pvdz-ri",
            "df_basis_elst": "jun-cc-pvdz-jkfit",
            "local_convergence": 10,
        }
    )
    e, wfn = psi4.energy("asapt", return_wfn=True)
    Elst_AB = wfn.array_variable("Elst_AB").np
    Exch_AB = wfn.array_variable("Exch_AB").np
    IndAB_AB = wfn.array_variable("IndAB_AB").np
    IndBA_AB = wfn.array_variable("IndBA_AB").np
    Disp_AB = wfn.array_variable("Disp_AB").np

    # detach atomic populations from the wavefunction
    # datatype: numpy arrays of size [NA x 1] and [NB x 1] respectively.
    Pop_A = wfn.array_variable("Pop_A").np
    Pop_B = wfn.array_variable("Pop_B").np

    # save numpy arrays to file
    np.save('Elst_AB.npy', Elst_AB)
    np.save('Exch_AB.npy', Exch_AB)
    np.save('IndAB_AB.npy', IndAB_AB)
    np.save('IndBA_AB.npy', IndBA_AB)
    np.save('Disp_AB.npy', Disp_AB)
    np.save('Pop_A.npy', Pop_A)
    np.save('Pop_B.npy', Pop_B)
    assert False


