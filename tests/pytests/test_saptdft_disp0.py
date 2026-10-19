"""Signed SAPT0 dispersion tests; CPU graph tests are NOT GPU evidence."""

import importlib
import json
import gc

import numpy as np
import pytest
import psi4

ein = pytest.importorskip("einsums")
if not hasattr(ein, "RuntimeTensorD"):
    pytest.skip("These tests require the actual Einsums2 runtime APIs", allow_module_level=True)

terms = importlib.import_module("psi4.driver.procrouting.sapt.sapt_jk_terms_ein")
mp2 = importlib.import_module("psi4.driver.procrouting.sapt.sapt_mp2_terms_ein")
gpu_module = importlib.import_module("psi4.driver.procrouting.sapt.sapt_dispersion_gpu")
sapt_proc = importlib.import_module("psi4.driver.procrouting.sapt.sapt_proc")

pytestmark = [pytest.mark.psi, pytest.mark.api, pytest.mark.saptdft]
KERNEL_RTOL, KERNEL_ATOL = 1e-11, 1e-12
PHYSICAL_RTOL, PHYSICAL_ATOL = 1e-8, 1e-8


@pytest.mark.parametrize("extents", [
    (93, 93, 47, 47, 24, 24),  # review reproducer: 9 shapes before alignment
    (101, 83, 37, 29, 5, 7),
    (13, 11, 3, 4, 24, 24),  # DF memory limit smaller than compute edge
])
def test_disp0_df_boundaries_bound_cached_shapes(extents):
    nr, ns, *_ = extents
    mr, ms, br, bs = terms._dispersion_block_extents(*extents)
    rshapes = {min(br, min(mr, nr - r) - r0)
               for r in range(0, nr, mr) for r0 in range(0, min(mr, nr - r), br)}
    sshapes = {min(bs, min(ms, ns - s) - s0)
               for s in range(0, ns, ms) for s0 in range(0, min(ms, ns - s), bs)}
    assert len(rshapes) <= 2 and len(sshapes) <= 2
    assert sum(r * s for r in rshapes for s in sshapes) <= 4 * br * bs


@pytest.mark.parametrize("nrb,nsb", [(1, 1), (3, 4), (2, 1)])
def test_disp0_default_replay_packing_and_freshness(nrb, nsb):
    """Use real default CPU graph replay, strided packs and pool-owned outputs."""
    rng = np.random.default_rng(9136)
    na, nb, nq = 3, 2, 11
    nk = 2 * nq + 2
    shapes = [(nk, nrb * na), (nk, nsb * nb),
              (nk, nsb * na), (nk, nrb * nb)]
    parents = [ein.asarray(rng.normal(size=(rows + 4, cols + 3))) for rows, cols in shapes]
    factors = [p[2:2 + rows, 1:1 + cols] for p, (rows, cols) in zip(parents, shapes)]
    pool = ein.MemoryPool(8192, "disp0-test")
    v = pool.empty([nrb * na, nsb * nb], name="V")
    x = pool.empty([nrb * na, nsb * nb], name="X")
    w = pool.empty([nsb * na, nrb * nb], name="W")
    contractions = gpu_module.BlockContractions(device="cpu")
    for iteration in range(3):
        if iteration:
            for i, parent in enumerate(parents):
                np.asarray(parent)[...] *= (-0.3 if i % 2 else 1.2)
        # Beta=0 must overwrite stale/poisoned outputs on every replay.
        for output in (v, x, w):
            np.asarray(output)[...] = 893.0
        contractions.products(*factors, nq, v, x, w)
        af, fa, bc, cb = [np.asarray(a) for a in factors]
        for actual, reference in ((v, af[:nq].T @ fa[nq:2 * nq]),
                                  (x, af.T @ fa), (w, bc.T @ cb)):
            np.testing.assert_allclose(np.asarray(actual), reference,
                                       rtol=KERNEL_RTOL, atol=KERNEL_ATOL)
    report = contractions.report(1.0)
    assert report["block_replays"] == 3
    assert report["planned_gpu_contractions"] == 0
    assert report["planned_cpu_contractions"] == 9
    assert report["actual_gpu_execution_verified"] is False
    # The cached graphs/outputs are live borrows. Reset must fail instead of
    # destroying their storage; after dropping the replay owner it must work.
    with pytest.raises(RuntimeError):
        pool.reset()
    del contractions, v, x, w, output, actual
    gc.collect()
    assert pool.bytes_used == 0
    pool.reset()


def test_disp0_one_cache_main_edge_main_replay():
    rng = np.random.default_rng(3819)
    replay = gpu_module.BlockContractions(device="cpu")
    buffers = {}
    for nr, ns in [(3, 4), (1, 2), (3, 4), (2, 1), (3, 4)]:
        na, nb, nq = 3, 2, 11
        nk = 2 * nq + 2
        key = nr, ns
        if key not in buffers:
            factors = [ein.create_zero_tensor("factor", [nk, width])
                       for width in (nr * na, ns * nb, ns * na, nr * nb)]
            outputs = [ein.create_zero_tensor("out", list(shape)) for shape in
                       [(nr * na, ns * nb), (nr * na, ns * nb), (ns * na, nr * nb)]]
            buffers[key] = factors, outputs
        factors, outputs = buffers[key]
        for factor in factors:
            np.asarray(factor)[...] = rng.normal(size=factor.shape)
        for output in outputs:
            np.asarray(output)[...] = 1293
        replay.products(*factors, nq, *outputs)
        af, fa, bc, cb = [np.asarray(f) for f in factors]
        for actual, expected in zip(outputs, [af[:nq].T @ fa[nq:2 * nq], af.T @ fa, bc.T @ cb]):
            np.testing.assert_allclose(np.asarray(actual), expected, rtol=KERNEL_RTOL, atol=KERNEL_ATOL)
    report = replay.report(1.0)
    assert report["cached_graph_shapes"] == 3
    assert report["block_replays"] == 5


WATER_NE = """
0 1
O 0.0 0.0 0.0
H 0.0 0.0 0.96
H 0.91 0.0 -0.24
--
0 1
Ne 2.8 1.1 1.5
symmetry c1
no_reorient
no_com
"""

ETHANE_WATER = """
0 1
C 0.000 0.000 0.000
H 1.090 0.000 0.000
C -0.770 1.330 0.000
H -1.860 1.330 0.000
H -0.390 1.870 0.890
H -0.390 1.870 -0.890
H -0.390 -0.540 0.890
H -0.390 -0.540 -0.890
--
0 1
O 0.200 -0.300 3.400
H 0.900 -0.100 4.000
H -0.500 -0.700 3.950
symmetry c1
no_reorient
no_com
"""

NE_WATER = """
0 1
Ne 2.8 1.1 1.5
--
0 1
O 0.0 0.0 0.0
H 0.0 0.0 0.96
H 0.91 0.0 -0.24
symmetry c1
no_reorient
no_com
"""


@pytest.mark.parametrize("geometry,basis,frozen,puream", [
    (WATER_NE, "cc-pvdz", False, True),
    (WATER_NE.replace("2.8 1.1 1.5", "3.1 0.9 1.7"), "cc-pvdz", False, True),
    (NE_WATER, "cc-pvdz", False, True),
    (ETHANE_WATER, "6-31g*", True, False),
], ids=["water-ne", "water-ne-shifted", "ne-water-reversed", "ethane-water-cart-fc"])
def test_disp0_hf_canonical_signed_native_oracle(geometry, basis, frozen, puream,
                                                monkeypatch, record_property):
    """One SCF/cache per geometry; compare both signed components, not totals."""
    captured, comparisons = {}, []
    build = terms.build_sapt_jk_cache
    entry = mp2.df_mp2_einsums_dispersion

    def capture_cache(dimer, wfn_a, wfn_b, *args, **kwargs):
        captured["wfn_a"] = wfn_a
        return build(dimer, wfn_a, wfn_b, *args, **kwargs)

    def compare(dimer, cache, nfrozen_a=0, nfrozen_b=0, do_print=True):
        aux = psi4.core.BasisSet.build(
            dimer.molecule(), "DF_BASIS_MP2", psi4.core.get_option("DFMP2", "DF_BASIS_MP2"),
            "RIFIT", psi4.core.get_global_option("BASIS"))
        wfn_a = captured["wfn_a"]
        reference = mp2.df_mp2_fisapt_dispersion(
            wfn_a, wfn_a.basisset(), aux, cache, nfrozen_a, nfrozen_b, False)
        assert reference["Disp20,u"] < 0
        for edge in (1, 5, 500):
            for replay in (None, "cpu"):
                actual = terms.dispersion0(
                    cache, dimer, nfrozen_a, nfrozen_b, block_size=edge,
                    _graph_device=replay, do_print=False)
                for label in ("Disp20,u", "Exch-Disp20,u"):
                    delta = float(actual[label] - reference[label])
                    comparisons.append({"edge": edge, "replay": replay, "label": label,
                                        "reference": reference[label], "actual": actual[label], "delta": delta})
                    np.testing.assert_allclose(actual[label], reference[label],
                                               rtol=PHYSICAL_RTOL, atol=PHYSICAL_ATOL,
                                               err_msg=f"{label}, edge={edge}, replay={replay}")
        result = entry(dimer, cache, nfrozen_a, nfrozen_b, do_print)
        for label in ("Disp20,u", "Exch-Disp20,u"):
            np.testing.assert_allclose(result[label], reference[label],
                                       rtol=PHYSICAL_RTOL, atol=PHYSICAL_ATOL)
        # Force several non-aligned DF boundaries independently of the compute
        # cap. The implementation must align them, retain <=4 work/cache shapes
        # and still reproduce both signed native components.
        bounded = terms.dispersion0(
            cache, dimer, nfrozen_a, nfrozen_b, block_size=5, _df_block_size=7,
            _graph_device="cpu", do_print=False)
        report = bounded["Disp0 execution"]
        assert report["cached_graph_shapes"] <= 4 and report["cached_work_shapes"] <= 4
        assert report["df_block_extents"] == [5, 5]
        for label in ("Disp20,u", "Exch-Disp20,u"):
            comparisons.append({"edge": 5, "df_cap": 7, "replay": "cpu", "label": label,
                                "reference": reference[label], "actual": bounded[label],
                                "delta": float(bounded[label] - reference[label])})
            np.testing.assert_allclose(bounded[label], reference[label],
                                       rtol=PHYSICAL_RTOL, atol=PHYSICAL_ATOL)
        # Independent asymmetric freezing (one A occupied, zero B occupied).
        asymmetric_ref = mp2.df_mp2_fisapt_dispersion(
            wfn_a, wfn_a.basisset(), aux, cache, 1, 0, False)
        asymmetric = terms.dispersion0(
            cache, dimer, 1, 0, block_size=5, _graph_device="cpu", do_print=False)
        for label in ("Disp20,u", "Exch-Disp20,u"):
            comparisons.append({"edge": 5, "frozen": [1, 0], "replay": "cpu", "label": label,
                                "reference": asymmetric_ref[label], "actual": asymmetric[label],
                                "delta": float(asymmetric[label] - asymmetric_ref[label])})
            np.testing.assert_allclose(asymmetric[label], asymmetric_ref[label],
                                       rtol=PHYSICAL_RTOL, atol=PHYSICAL_ATOL)
        zero = terms.dispersion0(cache, dimer, cache["Cocc_A"].shape[1], nfrozen_b, do_print=False)
        assert zero["Disp20,u"] == 0.0 and zero["Exch-Disp20,u"] == 0.0
        no_virtual = dict(cache)
        no_virtual["Cvir_A"] = psi4.core.Matrix("empty virtual", cache["Cvir_A"].shape[0], 0)
        zero = terms.dispersion0(no_virtual, dimer, nfrozen_a, nfrozen_b, do_print=False)
        assert zero["Disp20,u"] == 0.0 and zero["Exch-Disp20,u"] == 0.0
        return result

    monkeypatch.setattr(terms, "build_sapt_jk_cache", capture_cache)
    monkeypatch.setattr(mp2, "df_mp2_einsums_dispersion", compare)
    psi4.set_options({
        "basis": basis, "puream": puream, "scf_type": "df", "freeze_core": frozen,
        "SAPT_DFT_FUNCTIONAL": "HF", "SAPT_DFT_DO_DHF": False,
        "SAPT_DFT_INDUCTION_TYPE": "NONE", "SAPT_DFT_MP2_DISP_ALG": "EINSUMS",
        "SAPT_DFT_USE_EINSUMS": True, "ORBITAL_OPTIMIZER_PACKAGE": "INTERNAL",
    })
    try:
        psi4.energy("sapt(dft)", molecule=psi4.geometry(geometry))
        assert len(comparisons) == 16
    finally:
        # Keep every discrepancy, including successful comparisons, in XML.
        record_property("signed_comparisons", json.dumps(comparisons))


@pytest.mark.parametrize("options,message", [
    ({"SAPT_DFT_FUNCTIONAL": "PBE0"}, "requires SAPT_DFT_FUNCTIONAL=HF"),
    ({"SAPT_DFT_USE_EINSUMS": False}, "requires SAPT_DFT_MP2_DISP_ALG=EINSUMS"),
    ({"SAPT_DFT_EINSUMS_DISP_BLOCK": 0}, "must be positive"),
    ({"DO_DISP_EXCH_SINF": True}, "does not support DO_DISP_EXCH_SINF"),
    ({"SAPT_DFT_DO_FSAPT": "SAPTDFT"}, "does not yet support F-SAPT"),
])
def test_disp0_unsupported_dispatch_fails_explicitly(options, message):
    psi4.set_options({"basis": "sto-3g", "SAPT_DFT_FUNCTIONAL": "HF",
                      "SAPT_DFT_MP2_DISP_ALG": "EINSUMS", **options})
    with pytest.raises(psi4.ValidationError, match=message):
        psi4.energy("sapt(dft)", molecule=psi4.geometry(WATER_NE))


@pytest.mark.parametrize("options,do_dft,do_disp,message", [
    ({}, True, True, "requires SAPT_DFT_FUNCTIONAL=HF"),
    ({}, False, False, "ordinary SAPT_DFT_DO_DISP=true"),
    ({"SAPT_DFT_USE_EINSUMS": False}, False, True, "requires SAPT_DFT_MP2_DISP_ALG=EINSUMS"),
    ({"DO_DISP_EXCH_SINF": True}, False, True, "does not support DO_DISP_EXCH_SINF"),
    ({"SAPT_DFT_DO_FSAPT": "FISAPT"}, False, True, "does not yet support F-SAPT"),
    ({"SAPT_DFT_MP2_DISP_ALG": "SAPT", "SAPT_DFT_EINSUMS_GPU": True},
     False, True, "requires SAPT_DFT_MP2_DISP_ALG=EINSUMS"),
])
def test_disp0_direct_entrypoint_rejects_before_wavefunction_use(options, do_dft, do_disp, message):
    psi4.set_options({"SAPT_DFT_MP2_DISP_ALG": "EINSUMS", **options})
    with pytest.raises(psi4.ValidationError, match=message):
        sapt_proc.sapt_dft(None, None, None, do_dft=do_dft, do_disp=do_disp)


def test_disp0_cuda_request_rejects_mock():
    if "mock" not in ein.gpu.device_name().lower():
        pytest.skip("Rejection test requires the CPU/mock Einsums runtime")
    with pytest.raises(RuntimeError, match="coherent CUDA-enabled Einsums2"):
        gpu_module.BlockContractions()
