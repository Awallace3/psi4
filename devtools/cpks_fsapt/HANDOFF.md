# Compact native CPKS F-SAPT — draft PR #38

https://github.com/Awallace3/psi4/pull/38 targets `saptdft_ein_fi_option_d4`
(`d7b5a01d4c`). Candidate branch `fsapt-cpks-native-cpu`; no merge authorized.
The earlier Python-heavy candidate is retained at `26f042d4b2`.

## Native seam

`set_cpks_monomers(A, B)` borrows the original canonical monomers and live JKs.
Existing `CPHF_FISAPT` PCG now selects `-monomer.cphf_Hx()` for XC on; otherwise
the old gap+4J−K−Kᵀ product remains unchanged. Native `find()` retains source
construction, the shared localized gauge and bare-SAPT readout. No additional
gap/J/K or hybrid fraction is applied to the original monomer action.

Python is only admission/wiring: one 43-line helper, versus three modules /
572 lines before. The duplicate Python PCG and unused source/export framework
are removed. Source buffers, final slots and zero-RHS fixes remain.
Both flags default off; default uncoupled/scaled behavior is unchanged.

Full raw directional induction/exchange-induction, independent physical true
residuals, source completeness and amplitude additivity precede scaling.
The response factor is one; uniform deltaHF remains a separately labeled
approximation. Raw matrices and scaling/error metadata survive on the returned
wavefunction. Extra PCG, Fock-validation and snapshot arrays are admitted
before tensor construction against min(64 MiB, native FISAPT budget / 8).

Admission remains restricted C1, unembedded two-fragment HF/SVWN/PBE references.
Effective SCF options and stored external-potential/CPSCF state are checked.
Hybrids/PBE0, range separation, meta-GGA, GRAC/level shifts, external response,
sSAPT/Sinf readout and deltaDFT redistribution are rejected.

## Verification scope

```
python -m unittest discover -s tests/pytests -p 'test_fsapt_cpks*.py' -v
```

The algebra tests compile the actual native PCG/action methods against a tiny
Matrix fixture, checking dense solutions, source readouts/additivity, zero
directions, curvature, gaps and true residuals. This is a small standalone
test compilation, NOT a Psi4 build or native chemistry qualification.
Runs **bfe7043c8**, final committed-candidate **b9a0dce3f**:
**9 pass, 3 matching-build tests skipped**. The legacy
J/K product tail is byte-identical to target `d7b5a01d4c` (SHA256
`b313ba78bdeaf2149f4804e3e00000709c33f1eea33adcedd693a9a9daa2aee6`).

Actual-kernel agreement **b218c2c40** uses the pinned donor's original monomer
`cphf_Hx`, native IBO gauge and bare-SAPT readout on H2–water/STO-3G.
Compare historical Python PCG (`26f042d4b2`) with extracted compiled native PCG,
before all scaling, grid 75/302, configured CC-PVTZ-MINAO, two CPU threads /
256 MiB. Source/pair matrices and separate canonical totals agree:

| Functional | Max old/new raw pair difference (Eh) | Max actual true residual |
|---|---:|---:|
| HF | 1.085e-19 | 1.670e-13 |
| SVWN | 1.085e-19 | 1.225e-13 |
| PBE | 1.085e-19 | 1.461e-13 |
| PBE0 | 4.066e-20 | 1.011e-12 |

Maximum amplitude difference is 3.470e-17; separate canonical-component
mismatch ≤6.560e-16 Eh. All true residuals pass unchanged
`max(1e-12, 1e-10*||RHS||)`. This qualifies solver agreement on actual donor
kernels, NOT this whole native integration or production PBE0 admission.
Receipt: `/tmp/compact-cpks-agreement-vhy_a9ar/comparison.json`,
SHA256 `f303ce6200c5a0316d996fde07c7bbb67262d8831eb262468d698941199755da`.
Checker: `/tmp/compare_compact_cpks.py`; no installed source was overlaid.
The receipt's four candidate/fixture source hashes match the checked code.
Setup-only failed checker attempts used a literal `minao` basis name and a
nonexistent Python `Matrix.ncol()`; corrected to configured MINAO_BASIS and
`.np.shape`, without changing response policy. Static review
**dacd9b50cb65770e661e6f029d51f309a** found no actionable native-code defects.

Frozen-source phenol receipt rechecks **b8f445005**: **5 pass**, retaining both
jun/aDZ tables and the explicitly failed HF all-component scope. This is
saved-data arithmetic, not a new candidate phenol run.

Matching-build HF/SVWN/PBE tests check independent physical source cells,
production dispatch and deltaHF on/off, but remain explicitly gated:

```
PSI4_RUN_FSAPT_CPKS_NATIVE=1 python -m unittest discover \
  -s tests/pytests -p 'test_fsapt_cpks_native.py' -v
```

**Merge blocker:** no matching built native integration exists locally.
Do not substitute the donor core or infer qualification from algebra tests.
No heavy native build/install, GPU or cluster execution is authorized.
Commits retain `[skip ci]` to avoid requesting remote build resources.

Earlier phenol jun/aDZ evidence remains on `saptdft-cpks-fsapt-breakdown`
(`98a40d0f6b`, numerical receipts `dbf8ddb6d1`); it is donor-runtime evidence,
not execution of this compact native FISAPT integration. The separate HF
electrostatic export mismatch remains flagged, not normalized away.
Report: https://cos-4a16505.tail7d58ac.ts.net/docs/saptdft/cpks/cpks-fsapt.html#phenol-aug
