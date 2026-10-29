# CPU/native CPKS F-SAPT draft split

- Draft PR: https://github.com/Awallace3/psi4/pull/38
  (base/head and draft status verified; no CI build checks requested).
- Base: `saptdft_ein_fi_option_d4`, `d7b5a01d4c9dc7b17832f91d0c32f0bf78deecfd`.
- Candidate: `fsapt-cpks-native-cpu`, prepared in the existing isolated CPKS
  worktree. No other checkout, environment, target-branch commit or running
  calculation was changed.
- Source investigation retained on `saptdft-cpks-fsapt-breakdown`; numerical
  augmented-basis results committed there in `dbf8ddb6d1`.
- Selective port of native integration `cb4edba686`, original-monomer adapter
  and focused contracts. A dedicated CPU-only solver replaces the source
  branch's shared response-helper dependency. No accelerator/backend code,
  build changes, dispersion kernels, manuscript or scientific-run harness.

## Scientific mode / compatibility

New `FISAPT_FSAPT_IND_XC_RESPONSE` is **false** by default. Existing default
uncoupled/scaled weights are unchanged. With coupled response but XC off,
the old full-exchange Hessian algebra is unchanged, even on KS inputs.
Correct source buffers, complete final slots and zero-RHS handling apply to
both coupled paths; erroneous older cells are not preservation targets.

XC on borrows the original whole-monomer `cphf_Hx()` and live canonical JK,
solves each RHS in the canonical gauge, then uses the native common localized
readout. One operator per monomer; no isolated-density XC. Raw directional
induction and exchange-induction, source completeness/additivity and true
residuals are checked before scaling. Full response factor is one; uniform
deltaHF remains a separately labeled approximation. Raw arrays/metadata survive
on the returned wavefunction.

Production admission remains unshifted HF/SVWN/PBE, restricted closed-shell C1,
two unembedded fragments, native C assignment and S2 readout. Hybrids/PBE0,
range separation, meta-GGA, GRAC/level shifts, external response, sSAPT and
deltaDFT redistribution are rejected.
Admission checks module-effective SCF embedding options and read-only stored
monomer external-potential/PCM/CPSCF state, not merely current global flags.
The response preflight includes quadratic Fock/diagonal validation scratch
before constructing those arrays, including highly asymmetric ov spaces.

## Evidence and merge blockers

Candidate CPU contract run **b0cbe969e**: **39 passed, 3 matching-build native
tests skipped** (42 total, unittest discovery command below). Syntax,
whitespace and CPU-only dependency scans pass. The read-only review's
stored-embedding and quadratic-validation-memory findings were fixed and
covered by focused contracts; a follow-up inspection was requested.
Follow-up **d334d46e81584ae49df88679482478b67** found no further actionable
defects in those fixes. It was inspection-only, not build/native qualification;
the draft and matching-build merge blockers remain unchanged.

Completed **exploratory donor-runtime** fixed-PBE0 phenol studies support the
scientific distinction, not this candidate's compiled-native qualification:
XC-only omission retaining 25% exchange gives summed-pair L1 errors
**1.39937% / 1.46189%** at jun-cc-pVDZ / aug-cc-pVDZ, versus scaled uncoupled
**15.65518% / 14.41336%**. Legacy full exchange on KS fails curvature both ways.
Full aug-cc-pVDZ raw component error is ≤3.71e-14 Eh, true residual ≤2.22e-12.
The separate HF all-component electrostatic discrepancy remains explicitly
failed; induction-only checks do not relabel it as a full success.

Report:
https://cos-4a16505.tail7d58ac.ts.net/docs/saptdft/cpks/cpks-fsapt.html#phenol-aug

**Not ready for merge:** native C++/callback ABI has not been compiled or
numerically qualified on a matching runtime. No matching build exists locally;
no native build/install/GPU/SSH/cluster work is authorized by this PR split.
Run the opt-in native HF/SVWN/PBE tests after a matching build is provided:

```
PSI4_RUN_FSAPT_CPKS_NATIVE=1 python -m unittest discover \
  -s tests/pytests -p 'test_fsapt_cpks_native.py' -v
```

Pure CPU contracts can run without importing native Psi4:

```
python -m unittest discover -s tests/pytests -p 'test_fsapt_cpks*.py' -v
```

Those synthetic/API/source contracts must not be presented as native chemical
qualification. Draft PR creation is authorized; merging is not.
Commit messages retain `[skip ci]` to avoid automatically requesting remote
build resources under the current no-build/no-paid-job limits.
