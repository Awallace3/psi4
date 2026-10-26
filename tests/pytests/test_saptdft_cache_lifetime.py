"""HF cache must not overlap the DFT stage unless FISAPT needs that cache."""
import weakref

import pytest
import psi4
from psi4.driver.procrouting.sapt import sapt_proc

pytestmark = [pytest.mark.psi, pytest.mark.api, pytest.mark.quick, pytest.mark.saptdft]


@pytest.mark.parametrize("fsapt_type", ["NONE", "FISAPT"])
def test_hf_cache_lifetime_at_dft_boundary(monkeypatch, fsapt_type):
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.set_num_threads(2)
    mol = psi4.geometry("""
        Ne
        --
        Ne 1 4.5
        units bohr
    """)
    psi4.set_options({
        "basis": "sto-3g", "scf_type": "df", "use_cuest": False,
        "sapt_dft_grac_shift_a": 0.203293, "sapt_dft_grac_shift_b": 0.203293,
        "sapt_dft_do_dhf": True, "sapt_dft_do_hybrid": False,
        # Stop before DFT; do not invoke the unrelated FISAPT/FDDS guard.
        "sapt_dft_do_disp": False,
        "sapt_dft_use_einsums": False, "sapt_dft_induction_type": "CPHF",
        "sapt_dft_do_fsapt": fsapt_type, "orbital_optimizer_package": "internal",
    })

    class TrackedCache(dict):
        pass

    class ReachedDFTBoundary(Exception):
        pass

    references = []
    build_cache = sapt_proc.sapt_jk_terms.build_sapt_jk_cache
    scf_helper = sapt_proc.scf_helper

    def tracked_cache(*args, **kwargs):
        cache = TrackedCache(build_cache(*args, **kwargs))
        references.append(weakref.ref(cache))
        return cache

    def check_boundary(*args, **kwargs):
        if kwargs.get("banner") == "SAPT(DFT): DFT Monomer A":
            assert len(references) == 1
            # Assert before the driver frame unwinds; otherwise destruction at
            # function exit would conceal a cache retained across both stages.
            assert (references[0]() is not None) == (fsapt_type == "FISAPT")
            raise ReachedDFTBoundary
        return scf_helper(*args, **kwargs)

    monkeypatch.setattr(sapt_proc.sapt_jk_terms, "build_sapt_jk_cache", tracked_cache)
    monkeypatch.setattr(sapt_proc, "scf_helper", check_boundary)
    try:
        with pytest.raises(ReachedDFTBoundary):
            psi4.energy("sapt(dft)", molecule=mol)
    finally:
        psi4.core.clean_options()
        psi4.core.clean()
