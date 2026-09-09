"""Small AO-adaptation BLAS scope; no process-wide thread mutation."""
from concurrent.futures import ThreadPoolExecutor
from threading import Event
import numpy as np
import pytest
import psi4
from psi4.driver import p4util
from psi4.driver.procrouting.isapol_native_partition import adapt_main
from test_isapol_native_performance import threads


def test_thread_local_blas_scope_restores_success_exception_and_nested_calls():
    with threads(4):
        before = psi4.core._isa_blas_max_threads()
        expected = 1 if before else 0  # non-MKL builds deliberately do nothing
        assert psi4.core._isa_serial_blas_call(psi4.core._isa_blas_max_threads) == expected
        assert psi4.core._isa_blas_max_threads() == before
        def nested():
            assert psi4.core._isa_blas_max_threads() == expected
            assert psi4.core._isa_serial_blas_call(psi4.core._isa_blas_max_threads) == expected
            assert psi4.core._isa_blas_max_threads() == expected
            raise ValueError('intentional callback failure')
        with pytest.raises(ValueError, match='intentional'):
            psi4.core._isa_serial_blas_call(nested)
        assert psi4.core._isa_blas_max_threads() == before
        assert psi4.core.get_num_threads() == 4


def test_blas_override_does_not_affect_another_python_thread():
    with threads(4):
        before = psi4.core._isa_blas_max_threads()
        entered, release = Event(), Event()
        def callback():
            entered.set()
            assert release.wait(10)
            return psi4.core._isa_blas_max_threads()
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(psi4.core._isa_serial_blas_call, callback)
            try:
                assert entered.wait(10)
                assert psi4.core._isa_blas_max_threads() == before
            finally:
                release.set()
            assert future.result() == (1 if before else 0)
        assert psi4.core._isa_blas_max_threads() == before


def test_actual_water_ao_transform_is_mkl_thread_invariant():
    if psi4.core._isa_blas_max_threads() == 0:
        pytest.skip('Thread-local adaptation policy is currently MKL-specific')
    settings = {'basis':'cc-pvdz','reference':'rhf','scf_type':'pk',
                'e_convergence':1e-10,'d_convergence':1e-10}
    saved = p4util.OptionsState(*[[key.upper()] for key in settings])
    try:
        psi4.set_options(settings)
        mol = psi4.geometry('''0 1
O 0 0 0
H -1.45365196 0 -1.12168732
H 1.45365196 0 -1.12168732
units bohr
symmetry c1
no_com
no_reorient
''')
        with threads(1):
            _, wfn = psi4.energy('hf', molecule=mol, return_wfn=True)
            reference = adapt_main(wfn, caller_converged=True)
        for count in (2,4,8):
            with threads(count):
                before = psi4.core._isa_blas_max_threads()
                actual = adapt_main(wfn, caller_converged=True)
                np.testing.assert_array_equal(actual.transform, reference.transform)
                np.testing.assert_array_equal(actual.occupied, reference.occupied)
                assert psi4.core._isa_blas_max_threads() == before
                assert psi4.core.get_num_threads() == count
    finally:
        saved.restore()
