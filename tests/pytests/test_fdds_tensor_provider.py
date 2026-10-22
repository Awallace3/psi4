"""Independent physical oracles for the FDDS integral-supplier seam."""
import numpy as np
import pytest
import psi4

from test_cuest_fdds import fdds_factory

pytestmark = [pytest.mark.psi, pytest.mark.api, pytest.mark.quick]


@pytest.mark.parametrize("hybrid", [False, True])
@pytest.mark.parametrize("occupied", [(3, 3), (2, 3)], ids=["equal", "unequal"])
def test_fdds_native_integral_supplier(fdds_factory, hybrid, occupied):
    obj = fdds_factory(hybrid, occupied=occupied)
    primary, auxiliary, cache = fdds_factory.inputs
    n, q = primary.nbf(), auxiliary.nbf()
    mints = psi4.core.MintsHelper(primary)
    zero = psi4.core.BasisSet.zero_ao_basis_set()
    # This oracle does not read an integral tensor back from the supplier.
    ao = np.asarray(mints.ao_eri(auxiliary, zero, primary, primary)).reshape(q, n, n)
    eigenvalues, eigenvectors = np.linalg.eigh(np.asarray(obj.metric()))
    keep = (eigenvalues > 0) & (np.abs(eigenvalues) >= 1.e-12 * np.max(np.abs(eigenvalues)))
    half = (eigenvectors[:, keep] / np.sqrt(eigenvalues[keep])) @ eigenvectors[:, keep].T
    errors = []
    for monomer, raw_name, fit_names in (
            ("A", "arQ", ("aaR", "arR", "rrR")),
            ("B", "bsQ", ("bbR", "bsR", "ssR"))):
        co, cv = (np.asarray(cache[key + monomer]) for key in ("Cocc_", "Cvir_"))
        o, v = co.shape[1], cv.shape[1]
        raw = np.einsum("qmn,mi,nj->ijq", ao, co, cv)

        def check(name, expected):
            actual = obj.get_tensor_pqQ(name, expected.shape).np
            assert actual.shape == expected.shape  # Public NumPy layout contract.
            errors.append(float(np.max(np.abs(actual - expected))))
            np.testing.assert_allclose(actual, expected, rtol=0, atol=2.e-11)

        check(raw_name, raw)
        # A first-axis prefix exercises a real provider slice, not only full reads.
        check(raw_name, raw[:1])
        if hybrid:
            check("raQ" if monomer == "A" else "sbQ", raw.transpose(1, 0, 2))
            check("Qar" if monomer == "A" else "Qbs", raw.transpose(2, 0, 1))
            for name, left, right in zip(fit_names, (co, co, cv), (co, cv, cv)):
                fitted = np.einsum("qmn,mi,nj,qp->ijp", ao, left, right, half)
                check(name, fitted)
        # Repeated reads after the hybrid fitting round must retain raw bytes.
        check(raw_name, raw)
        assert (o, v) == fdds_factory.dimensions[monomer]
    print("FDDS native supplier maximum absolute tensor error / a.u.:", max(errors))
