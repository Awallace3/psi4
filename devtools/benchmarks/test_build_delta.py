import json

import pytest

import build_delta as mod


def stats(median, spread=0.0):
    return {"median": median, "min": median - spread, "max": median + spread}


def row(system="benzene", basis="aug-cc-pvdz", nbf=384, cpu=120.0, gpu=20.0,
        cpu_spread=0.5, gpu_spread=0.1, cpu_host=14900.0, gpu_host=1279.0,
        device=2878.0, delta=2.077e-06, delta_repeats=None):
    built = {
        "system": system, "basis": basis, "nbf": nbf,
        "wall_s": {"cpu": stats(cpu, cpu_spread), "gpu": stats(gpu, gpu_spread)},
        "speedup": cpu / gpu,
        "max_abs_delta_hartree": delta,
        "memory": {
            "host_peak_rss_mib": {"cpu": {"peak_rss_mib": stats(cpu_host, 8.0)},
                                  "gpu": {"peak_rss_mib": stats(gpu_host, 2.0)}},
            "device_peak_mib": stats(device, 400.0),
        },
    }
    if delta_repeats is not None:
        built["components"] = {
            "SAPT EXCH ENERGY": {"max_abs_delta_hartree": 0.0,
                                 "paired_deltas_hartree": [0.0, 0.0, 0.0]},
            "SAPT DISP ENERGY": {"max_abs_delta_hartree": delta,
                                 "paired_deltas_hartree": list(delta_repeats)},
        }
    return built


def summary(*rows):
    return {"rows": list(rows)}


def test_a_change_larger_than_the_scatter_is_reported_as_a_ratio():
    payload = mod.compare(summary(row(cpu_host=14900.0)),
                          summary(row(cpu_host=11000.0)))
    peak = payload["rows"][0]["cpu_host_peak_mib"]
    assert peak["resolved"]
    assert peak["delta"] == pytest.approx(-3900.0)
    assert peak["ratio"] == pytest.approx(11000.0 / 14900.0)


def test_a_change_inside_the_scatter_is_not_called_a_change():
    # Three repeats on a shared cluster do not resolve a sub-percent move, and
    # a report that prints one as a ratio invites a conclusion from noise.
    payload = mod.compare(summary(row(cpu=120.0, cpu_spread=1.0)),
                          summary(row(cpu=120.9, cpu_spread=1.0)))
    walls = payload["rows"][0]["cpu_wall_s"]
    assert not walls["resolved"]
    assert "(~)" in mod.cell(walls)


def test_the_noise_band_widens_with_the_measured_scatter():
    quiet = mod.compare(summary(row(cpu=120.0, cpu_spread=0.1)),
                        summary(row(cpu=121.0, cpu_spread=0.1)))
    noisy = mod.compare(summary(row(cpu=120.0, cpu_spread=5.0)),
                        summary(row(cpu=121.0, cpu_spread=5.0)))
    assert quiet["rows"][0]["cpu_wall_s"]["resolved"]
    assert not noisy["rows"][0]["cpu_wall_s"]["resolved"]


def test_a_metric_with_no_repeat_spread_is_flagged_as_unscattered():
    # A single-repeat tree has min == max, so the band is zero and every
    # difference "clears" it. The flag says the band was never measured.
    payload = mod.compare(summary(row(cpu_spread=0.0)), summary(row(cpu=121.0, cpu_spread=0.0)))
    assert payload["rows"][0]["cpu_wall_s"]["scatter_measured"] is True
    assert mod.change({"median": 1.0}, {"median": 2.0})["scatter_measured"] is False


def test_identical_numerics_is_asserted_not_assumed():
    same = mod.compare(summary(row(delta=2.077e-06)), summary(row(delta=2.077e-06)))
    assert same["identical_numerics"]
    moved = mod.compare(summary(row(delta=2.077e-06)), summary(row(delta=3.0e-06)))
    assert not moved["identical_numerics"]
    assert moved["numeric_drift"][0]["treatment"] == pytest.approx(3.0e-06)
    assert "do not agree numerically" in mod.markdown(moved, "c", "t")


def test_a_numeric_move_inside_the_repeats_own_scatter_is_not_a_disagreement():
    # The CPU-vs-GPU delta is a difference of two threaded reductions, so its
    # last digits move between repeats of a single build. Demanding exact
    # equality across builds would report that wobble as a correctness change.
    control = row(delta=5.0e-08, delta_repeats=[4.0e-08, 5.0e-08, 6.0e-08])
    treatment = row(delta=5.04e-08, delta_repeats=[4.1e-08, 5.04e-08, 6.1e-08])
    payload = mod.compare(summary(control), summary(treatment))
    assert payload["identical_numerics"]
    assert payload["rows"][0]["max_abs_delta_hartree"]["scatter_measured"] is True
    assert "within" in mod.markdown(payload, "c", "t")


def test_a_numeric_move_beyond_the_scatter_is_still_caught():
    control = row(delta=5.0e-08, delta_repeats=[4.99e-08, 5.0e-08, 5.01e-08])
    treatment = row(delta=9.0e-08, delta_repeats=[8.99e-08, 9.0e-08, 9.01e-08])
    payload = mod.compare(summary(control), summary(treatment))
    assert not payload["identical_numerics"]
    assert payload["numeric_drift"][0]["noise_band"] == pytest.approx(2.0e-10)
    assert "scatter" in mod.markdown(payload, "c", "t")


def test_without_per_repeat_deltas_the_comparison_stays_exact():
    # No `components` block means the scatter was never measured, and the
    # honest fallback is the strict test rather than an invented tolerance.
    payload = mod.compare(summary(row(delta=5.0e-08)), summary(row(delta=5.000001e-08)))
    assert not payload["identical_numerics"]
    assert payload["rows"][0]["max_abs_delta_hartree"]["scatter_measured"] is False


def test_cases_present_in_only_one_job_are_named_rather_than_dropped():
    payload = mod.compare(summary(row(), row(system="water", basis="cc-pvdz", nbf=48)),
                          summary(row()))
    assert payload["control_only"] == ["water:cc-pvdz"]
    assert payload["treatment_only"] == []
    assert "water:cc-pvdz" in mod.markdown(payload, "c", "t")


def test_a_differing_basis_dimension_means_it_is_not_the_same_calculation():
    payload = mod.compare(summary(row(nbf=384)), summary(row(nbf=386)))
    assert not payload["rows"][0]["nbf_agrees"]
    assert "Basis dimension differs" in mod.markdown(payload, "c", "t")


def test_rows_are_ordered_by_problem_size():
    payload = mod.compare(
        summary(row(), row(system="water", basis="cc-pvdz", nbf=48)),
        summary(row(), row(system="water", basis="cc-pvdz", nbf=48)))
    assert [r["nbf"] for r in payload["rows"]] == [48, 384]


def test_a_missing_memory_block_renders_rather_than_raises():
    bare = row()
    bare.pop("memory")
    payload = mod.compare(summary(bare), summary(bare))
    assert payload["rows"][0]["gpu_device_peak_mib"] is None
    assert "—" in mod.markdown(payload, "c", "t")


def test_both_labels_appear_so_the_direction_is_never_ambiguous():
    payload = mod.compare(summary(row()), summary(row()))
    text = mod.markdown(payload, "premerge d91b5f8e81", "merge ee6161a3b6")
    assert "premerge d91b5f8e81" in text and "merge ee6161a3b6" in text
    assert text.index("premerge d91b5f8e81") < text.index("merge ee6161a3b6")


def test_the_cli_writes_json_and_can_fail_on_numeric_drift(tmp_path, capsys):
    control = tmp_path / "control.json"
    treatment = tmp_path / "treatment.json"
    control.write_text(json.dumps(summary(row(delta=2.0e-06))))
    treatment.write_text(json.dumps(summary(row(delta=9.0e-06))))
    out = tmp_path / "delta.json"
    import sys
    argv = sys.argv
    sys.argv = ["build_delta.py", str(control), str(treatment), "--output", str(out),
                "--require-identical-numerics"]
    try:
        with pytest.raises(SystemExit):
            mod.main()
    finally:
        sys.argv = argv
    assert json.loads(out.read_text())["identical_numerics"] is False
