# Temporary numerical acceptance policy

User authorization: allow **1e-3 tolerance for struggling stages for now** to
unblock end-to-end implementation, recording eventual tightening TODOs.
Subsequent user authorization permits **Drho-C-only 1e-2**, via the explicit
`provisional-drho-1e-2` profile. It changes only the existing Drho-C forward-error
allowlist; metric/RHS/electron/residual and structural checks remain strict.
Other stages remain at1e-3, and the original `provisional-1e-3` profile retains its
meaning even for Drho-C. The new profile is rejected for other stages.

This supersedes earlier blanket instructions not to relax numerical gates, but
does not replace the original strict targets or retrospectively change evidence.

## Application

- Opt in explicitly for struggling numerical comparisons only. Keep the existing
  error metric/normalization and units; 1e-3 is not blanket absolute error or an
  implicit change to relative-error denominators. Existing passing tests retain
  their original tolerances.
- Report original strict tolerance/result, provisional tolerance/result, measured
  errors, stage/input/source provenance, and a tightening TODO. Write fresh reports;
  do not overwrite historical failures or label provisional passes strict parity.
- Use a separately named provisional profile, not looser solver stopping criteria,
  altered ranks, regularization, density rescaling or silent symmetrization.
- Preserve dimension, finiteness, reciprocity, PSD policy, rank/conditioning,
  ownership, source/identity, cache-generation, schema and safety checks. Missing
  algorithms, native producers or published localization definitions cannot pass
  by numerical tolerance alone.
- Legacy callable trajectory comparisons retain their custom `tolerance` argument
  for compatibility. Nondefault tolerances produce `legacy-custom-tolerance`
  reports with `profile_certified=false`, not named strict/provisional certificate
  flags. Custom tolerances cannot be combined with the provisional profile, and
  the certification CLI offers only the fixed profiles.
- Values above the explicitly selected limit (1e-3, or Drho-C-only1e-2) remain
  failures. Do not round, enlarge tolerance, switch metrics,
  exclude failing coefficient checks or substitute a downstream observable merely
  to claim acceptance. Downstream end-to-end errors must be measured separately;
  per-stage1e-3 does not guarantee end-to-end1e-3.
- Provisional integration is allowed where the algorithm exists and the explicit
  provisional numerical gate actually passes. Public driver registration still
  requires a functional pipeline and truthful experimental/acceptance labeling.

## Initial backlog and historical indicators

These are prior measurements, not fresh provisional acceptance:

- TODO9: raw-tail parameters: retain strict scaled1e-9 target; recorded deviations
  around1e-8 may qualify provisionally after complete comparator checks.
- TODO10: native Drho-C coefficients/density: historical coefficient scaled error
  **0.0012802188514176112 exceeds1e-3**, despite smaller represented-density errors.
  This does not become a provisional pass under the authorized threshold.
- TODO11: native OV coefficients/density: recorded coefficient scaled error
  2.3019798321950356e-5 may qualify provisionally; native API/producer completeness
  remains a separate requirement.
- TODO12: tighten every newly provisional downstream property and end-to-end gate
  to its original strict target, with stage-specific error budgets and reruns.

TODO12's "stage-specific error budgets" half is now **measured**, though not yet
met. `psi4.driver.procrouting.isapol_budget` derives, per intermediate and per
property group, the precision that intermediate must be known to for a stated
property tolerance, by perturbing it and rebuilding the whole downstream through
the shipped objects under the unrelaxed production LW policy. Evidence:
`.pi/audit/property-anchored-budget.json`; specification: SPEC.md §8. Against a
1e-6 property tolerance the historical indicators stand as:

- TODO10 (Drho-C, 0.0012802188514176112, absolute metric) needs **2.78e-9**
  (A=3.60e2, binding on alpha_iso_rank3, water direct-OV): **misses by ~6 orders**.
- TODO11 (fitted OV, 2.3019798321950356e-5, absolute metric) needs **6.21e-10**
  (A=1.61e3, binding on C10, He fitted chain): **misses by ~4.5 orders**.
- TODO9 (raw tails, 2.3588804665973028e-8) is **left uncompared**. The
  max-scaled metric it was recorded in is not a well-posed question for the
  shape samples: their elements span many decades and the max-scaled
  amplification diverges as the probe shrinks (5.6e4 -> 2.4e5 -> 6.3e5 over
  eps 1e-6 -> 1e-10) instead of converging. In the elementwise-relative metric,
  which does converge, the requirement is 1.03e-4 -- but the recorded error is
  not in that metric and the budget refuses to compare across metrics. TODO9
  additionally associates a joint-tail *parameter* error with a *sample array*;
  closing it needs the shape samples re-measured against the same-input
  reference elementwise-relatively.

Neither comparison is a new provisional pass, and none of these amplifications
is a gate: each is a first-order directional lower bound, so meeting the derived
precision is necessary and not sufficient. What they do establish is that the
per-stage provisional profiles above must not be read as implying any 1e-6
end-to-end property bound. TODO12 remains open on the tightening half.

## Measured results before the Drho-C-only exception

Task8's explicit profile is implemented. The combined ISA/FDDS suite passes
1061 tests, including legacy custom-tolerance compatibility regressions. Strict
numerical oracle tests and existing supplied-stage tolerances remain unchanged.

Current reports establish:
- Retained fixed-density trajectory recomparison: strict FAIL, provisional PASS;
  maximum joint-tail scaled error2.3588804665973028e-8. Not a new controller run.
- Fresh native C++ Drho-C measurement: strict FAIL, provisional FAIL;
  coefficient scaled error0.0012802188514176112 under the old1e-3 profile.
  A subsequent fresh measurement under `provisional-drho-1e-2` PASSED with
  the same coefficient error; strict parity remains FAIL. Evidence:
  `.pi/audit/provisional-parent-drho-1e-2-v1.json`. The exception's combined
  suite passed1064 tests in6.39s. TODO10 remains for strict parity.
- Fresh native-integral/supplied-C NumPy OV diagnostic: strict FAIL, provisional
  PASS; coefficient scaled error2.3019798321950356e-5. This is not a production
  C++ OV API or native response acceptance.

Bounded evidence: `tests/pytests/data_isapol/psi4_provisional_acceptance_evidence.json`.
The profile/measurement implementation was independently reviewed; its single
custom-tolerance compatibility finding was then fixed and regression-tested.
TODO9–12 retain strictness restoration work. Full end-to-end acceptance remains
false; missing native producers and published localization are not waived.
