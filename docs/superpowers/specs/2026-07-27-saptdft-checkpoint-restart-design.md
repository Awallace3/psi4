# SAPT(DFT) Checkpoint and Restart Design

**Status:** Proposed
**Date:** 2026-07-27
**Scope:** `psi4/driver/procrouting/sapt/sapt_proc.py` and focused checkpoint tests

## 1. Purpose

SAPT(DFT) checkpointing shall let an interrupted calculation resume without repeating completed SCF calculations. Restarted calculations shall produce the same SAPT and F-SAPT results as uninterrupted calculations.

Converged SCF state and selected downstream results may be stored. JK objects shall never be stored because they can be prohibitively large. JK objects and JK-derived caches shall instead be reconstructed on demand from restored wavefunctions.

The checkpoint implementation shall also prevent accidental reuse across different calculations. A checkpoint from a different molecule, method, basis, option set, software version, or supported add-on version shall fail explicitly rather than be ignored or reused.

## 2. Goals

1. Avoid all repeated SCF iterations for SCF stages recorded as complete.
2. Reconstruct usable RHF/RKS wavefunctions from stored converged state.
3. Rebuild JK objects and JK-derived caches only when unfinished work requires them.
4. Resume scalar SAPT, delta-correction, dispersion, and F-SAPT stages at their actual completion boundaries.
5. Identify equivalent jobs through deterministic QCSchema input serialization.
6. Fail explicitly on incompatible, incomplete, corrupt, or concurrently modified checkpoints.
7. Preserve existing behavior when checkpointing is disabled.
8. Support both native QCSchema execution and ordinary `psi4.energy()` execution.

## 3. Non-goals

1. Serializing or restoring JK objects.
2. Avoiding reconstruction of every transient intermediate. JK objects and SAPT JK caches are deliberately recomputed.
3. Restarting in the middle of an SCF iteration or in the middle of an individual SAPT component routine.
4. Compatibility across different Psi4 versions, checkpoint schema versions, or result-affecting add-on versions.
5. Supporting checkpoint restart with external potentials in the initial implementation. Enabling checkpointing with external potentials shall continue to raise an explicit validation error.
6. Sharing one checkpoint directory among concurrent jobs.

## 4. Terminology

- **Canonical input:** A normalized QCSchema `AtomicInput` containing the complete computational identity of the job.
- **Job identity:** A SHA-256 digest of the deterministic serialization of the canonical input plus the execution compatibility fingerprint.
- **SCF snapshot:** Persisted converged wavefunction state sufficient to reconstruct an SCF subclass without running SCF iterations.
- **Persistent artifact:** A checksummed file referenced by the checkpoint manifest.
- **Transient prerequisite:** Data deliberately recreated after restart, including JK objects and JK-derived caches.
- **Completed stage:** An atomic unit of work whose required artifacts and values were durably written before the manifest recorded completion.

## 5. User-facing behavior

Checkpointing remains opt-in through the configured checkpoint directory. When disabled, SAPT(DFT) follows the existing execution path and shall not create checkpoint artifacts.

When enabled:

1. A new compatible checkpoint directory is initialized before the first checkpointable stage.
2. An existing compatible checkpoint is loaded and validated before any expensive work begins.
3. Completed SCF stages are restored without calling SCF iteration or energy-computation entry points.
4. Completed downstream stages are skipped when their saved outputs satisfy the dependencies of the next unfinished stage.
5. Transient JK state is rebuilt only if unfinished work requires it.
6. A fully completed checkpoint returns the restored result without running SCF or constructing JK.

A job identity mismatch, unsupported version, malformed manifest, missing artifact, checksum mismatch, invalid SCF snapshot, or active writer lock shall raise `ValidationError`. The implementation shall not silently discard the checkpoint and start a fresh calculation.

## 6. Canonical QCSchema job identity

### 6.1 Native QCSchema calls

For a native QCSchema call, the supplied `AtomicInput` is normalized before hashing. Normalization shall:

1. Serialize the molecule through its QCSchema representation, including fragments, charge, multiplicity, geometry, masses, atomic identities, and orientation-related fields.
2. Normalize method and basis names consistently with Psi4's method dispatch.
3. Expand all result-affecting defaults into the keyword map so omitted defaults and explicitly supplied default values identify the same computation.
4. Normalize keyword names and enumerated string values to a stable case.
5. Preserve numerical and array values without lossy string formatting.
6. Include driver, model, protocols, and user-supplied extras, except for checkpoint runtime controls explicitly listed below.
7. Sort object keys during deterministic JSON serialization.

The following runtime controls are excluded from job identity because they do not define the computed result:

- checkpoint directory;
- test-only stop-after stage;
- output filename and print destination;
- memory limit;
- thread count;
- timer and verbosity settings.

All result-affecting SAPT, SCF, DF, FISAPT, dispersion, frozen-core, convergence, and backend-selection options shall be represented in the normalized keyword map.

### 6.2 Ordinary Python API calls

For ordinary `psi4.energy()` calls, the driver shall synthesize a QCSchema `AtomicInput` after method parsing, molecule preparation, and relevant option resolution. The synthesized input shall pass through the same normalization and hashing code as a native QCSchema input.

Equivalent QCSchema and ordinary Python API jobs shall produce the same canonical serialization and job identity. Identity construction shall therefore live in one shared helper rather than in separate API-specific implementations.

### 6.3 Execution compatibility fingerprint

The job digest shall also cover a compatibility fingerprint containing:

- checkpoint schema version;
- exact Psi4 version and development revision when available;
- QCEngine/QCElemental schema model version used for normalization;
- selected einsums versus NumPy implementation;
- versions of result-affecting D3/D4 or other selected add-ons.

The manifest shall retain both the digest and the canonical input/fingerprint used to create it. On mismatch, the error should report the first useful differing fields rather than only the two hashes.

## 7. Manifest and artifact format

The manifest shall be a versioned JSON document with this logical structure:

```json
{
  "schema_version": 1,
  "job_identity": {
    "sha256": "...",
    "canonical_input": {},
    "execution_fingerprint": {}
  },
  "completed_stages": {
    "hf_dimer_scf": {
      "artifacts": ["hf_dimer_wfn"],
      "scalars": ["HF DIMER"]
    }
  },
  "scalars": {},
  "artifacts": {
    "hf_dimer_wfn": {
      "path": "hf_dimer.npy",
      "sha256": "...",
      "kind": "scf_snapshot",
      "size": 0
    }
  }
}
```

A stage is complete only if:

1. every required artifact exists;
2. each artifact checksum and size match the manifest;
3. every required scalar or metadata field exists; and
4. every declared dependency is complete.

Unknown stage names, unknown artifact kinds, and unsupported manifest versions are errors.

### 7.1 Atomic writes

For every checkpoint update:

1. Write each new artifact to a uniquely named temporary file in the checkpoint directory.
2. Flush and close it.
3. Compute and record its size and SHA-256 checksum.
4. Atomically rename it to its final artifact path.
5. Write the new manifest to a temporary file.
6. Flush, close, and atomically replace the previous manifest.

The manifest is the commit record. Unreferenced temporary or final artifacts left by an interruption may be removed on the next validated open, but an artifact referenced by the current manifest shall never be removed before a replacement manifest is committed.

### 7.2 Writer exclusion

The manager shall acquire an exclusive checkpoint-directory lock before mutation. If another live writer owns the lock, checkpoint initialization fails. Stale-lock recovery must be explicit and conservative; it shall not permit two processes to write concurrently.

## 8. Stage model

Completion shall be represented as independent named stages with explicit dependencies, not as one ordinal stage. Conditional stages are present only when requested by the canonical input.

The supported stage set includes:

### Setup and SCF stages

- `grac_monomer_a`
- `grac_monomer_b`
- `hf_dimer_scf`
- `hf_monomer_a_scf`
- `hf_monomer_b_scf`
- `dimer_localization_scf`
- `monomer_a_dft_scf`
- `monomer_b_dft_scf`
- `delta_dft_dimer_scf`
- `delta_dft_monomer_a_scf`
- `delta_dft_monomer_b_scf`
- `delta_dft`

### SAPT(HF) stages

- `hf_sapt_elst`
- `hf_sapt_exch`
- `hf_sapt_ind`

The HF SAPT JK construction is transient and is not a completed persistent stage.

### SAPT(DFT) stages

- `elst`
- `exch`
- `ind`
- `disp`
- `d3`
- `d4`

The SAPT JK object and SAPT JK cache construction are transient and are not persistent completed stages.

### F-SAPT stages

- `fsapt_setup`
- `fsapt_elst`
- `fsapt_exch`
- `fsapt_ind`
- `fsapt_disp`
- `fsapt_final`

### Completion

- `final`

Dependencies shall describe actual data requirements. For example, `exch` depends on restored monomer wavefunctions and `elst`, while `fsapt_exch` depends on `fsapt_setup`, `fsapt_elst`, and its required restored matrices. Optional execution paths shall not imply completion of stages that were not requested.

## 9. SCF snapshot persistence and restoration

### 9.1 Persisted state

Each completed SCF stage shall store enough converged state to reconstruct the correct RHF or RKS subclass without iteration. At minimum, the snapshot contract shall cover:

- molecule and primary basis identity;
- required auxiliary basis assignments or enough canonical input data to rebuild them;
- reference and functional identity;
- alpha and beta MO coefficients;
- alpha and beta orbital energies;
- occupations and electron counts;
- dimensions and symmetry metadata;
- converged SCF energy;
- required density or orbital-space information used downstream;
- wavefunction variables consumed by SAPT, F-SAPT, or reporting;
- a snapshot-format version.

The snapshot shall not contain a JK object or a serialized representation of JK internal state.

### 9.2 Rehydration

Restoration shall:

1. Deserialize the stored base wavefunction data.
2. Reconstruct the functional from the canonical method/options.
3. Create the appropriate SCF subclass through `scf_wavefunction_factory` or a focused shared rehydration helper built on the same factory logic.
4. Restore converged matrices, vectors, occupations, energies, basis assignments, and variables.
5. Validate dimensions, reference type, functional identity, molecule identity, and all required fields.
6. Return an object supporting downstream SCF subclass methods, including `set_jk()` and the appropriate `cphf_Hx()` implementation.

Rehydration shall not call `compute_energy()`, `scf_iterator()`, `scf_helper()`, `run_scf()`, or any equivalent SCF convergence path.

If a required subclass cannot be reconstructed, restart fails. It shall not clear the checkpoint and rerun the SCF.

### 9.3 SCF consumers

The restored wavefunctions shall be usable by:

- SAPT JK cache construction;
- electrostatics and exchange;
- CPHF/CPKS induction after a rebuilt JK is attached;
- MP2 or FDDS dispersion;
- localization and F-SAPT setup;
- delta corrections and final reporting.

## 10. JK and transient-cache reconstruction

JK handling is governed by these invariants:

1. No checkpoint artifact or manifest entry may contain a JK object.
2. No checkpoint write may traverse or materialize JK internals.
3. A restored wavefunction initially has no attached JK.
4. JK is constructed lazily only when the next unfinished operation requires it.
5. The implementation rebuilds the correct J/K/wK configuration, omega value, basis, and backend from canonical input and restored wavefunctions.
6. The rebuilt JK is attached to the restored monomer SCF subclasses before induction invokes `cphf_Hx()`.
7. Distinct monomer JK objects are rebuilt when differing range-separated settings require them.
8. JK objects are finalized through the existing cleanup behavior.

SAPT JK caches are likewise transient. They may be regenerated even when earlier scalar SAPT stages are complete, but regeneration must not trigger SCF iterations or recomputation of a completed component. The cache builder should construct only the prerequisite data needed by the next unfinished stage when practical; this optimization is recommended but not required for initial correctness.

A `final` checkpoint shall restore and return final variables without constructing JK or rebuilding a JK cache.

## 11. Scalar and matrix restoration

Scalar results shall be stored in a lossless JSON-compatible representation. Every stage shall declare the scalar keys it produces and requires.

Large matrices needed to skip completed F-SAPT stages shall be stored as dedicated checksummed array artifacts rather than hidden inside wavefunction variables. On restart, the manager restores those matrices into the cache under their original keys before executing the next unfinished stage.

Expected F-SAPT artifacts include, as applicable:

- localized orbital and partition metadata required by later stages;
- `Elst_AB`;
- `Exch_AB`;
- `IndAB_AB`;
- `IndBA_AB`;
- `Disp_AB`;
- fragment charges and dimensions required for final variable publication.

Every F-SAPT stage shall guard execution with its completion record. Persisting a stage name without restoring and reusing its outputs is not valid checkpoint support.

## 12. Restart data flow

1. Parse the requested SAPT(DFT) job and resolve relevant options.
2. Build or obtain the canonical QCSchema input.
3. Compute the job identity and execution fingerprint.
4. Open and lock the checkpoint directory.
5. If a manifest exists, validate identity, schema, stage graph, artifacts, and checksums.
6. Restore all completed SCF snapshots required by unfinished work.
7. Restore completed scalar and matrix results.
8. Determine the next unfinished stage from explicit dependencies and requested options.
9. If the checkpoint is `final`, publish restored variables and return immediately.
10. Otherwise, lazily build JK and transient caches when the next unfinished stage requires them.
11. Execute only unfinished persistent stages.
12. Commit each completed stage atomically.
13. Commit `final`, publish variables, release resources, and return.

## 13. Error handling

The following conditions shall raise a descriptive `ValidationError`:

- canonical input or execution fingerprint mismatch;
- malformed manifest JSON;
- unsupported checkpoint schema;
- unknown completed stage;
- incomplete dependency graph;
- missing artifact;
- artifact size or checksum mismatch;
- unsupported snapshot format;
- missing required SCF state;
- SCF subclass reconstruction failure;
- concurrent writer;
- unsupported external potentials.

Errors should identify the checkpoint directory, affected stage or artifact, and corrective action. Identity errors should report useful differing QCSchema fields. The implementation shall never treat one of these errors as a cache miss and silently recompute completed SCFs.

## 14. Testing requirements

### 14.1 Numerical equivalence

For representative HF, DFT, delta-HF, delta-DFT, dispersion, D3/D4, and F-SAPT configurations:

- run an uninterrupted reference calculation;
- stop after each supported persistent stage;
- restart in a fresh Python process;
- compare all published scalar variables and F-SAPT matrix outputs to the uninterrupted reference at existing project tolerances.

### 14.2 Proof that SCF is not repeated

Restart tests shall patch or instrument all relevant SCF entry points so they raise if invoked for a completed SCF stage. This includes `scf_helper`, `run_scf`, `compute_energy`, and the SCF iterator path as appropriate.

Each SCF checkpoint case passes only if restart reaches the correct final result while those guards remain active.

### 14.3 Proof of JK policy

Tests shall verify that:

- no manifest entry or artifact represents JK state;
- no JK-sized object is serialized through wavefunction checkpointing;
- JK construction occurs after restart when unfinished induction or another dependent stage requires it;
- restored SCF subclasses accept the rebuilt JK and complete CPHF/CPKS;
- restart from `final` constructs no JK.

### 14.4 Stage behavior

Tests shall assert the manifest's completed-stage set after every forced stop. D3 and D4 must be included. F-SAPT tests shall instrument completed F-SAPT routines to fail if rerun and shall verify restoration of required matrix artifacts.

### 14.5 Identity and corruption

Tests shall cover:

- equivalent native QCSchema and Python API calls producing the same identity;
- omitted defaults versus explicitly supplied defaults;
- changed geometry, fragments, charge, multiplicity, method, basis, relevant option, backend, or add-on version causing explicit mismatch failure;
- malformed manifest;
- missing, truncated, or checksum-invalid artifact;
- unsupported schema or snapshot version;
- interruption before artifact rename;
- interruption after artifact rename but before manifest replacement;
- lock contention and stale-lock handling.

### 14.6 Regression behavior

Existing SAPT(DFT) tests shall continue to pass with checkpointing disabled. Checkpoint tests shall use small systems for routine CI but must include at least one integration test that restarts in a separate process to rule out accidental reuse of in-memory Psi4 state.

## 15. Acceptance criteria

The implementation is accepted when all of the following are demonstrated:

1. Every completed SCF stage resumes without any SCF iteration or energy recomputation.
2. Restored RHF/RKS subclasses support downstream `set_jk()` and `cphf_Hx()` behavior.
3. JK objects are absent from all checkpoint files and are rebuilt only when required.
4. A completed final checkpoint returns without SCF or JK construction.
5. All supported restart stages produce results equivalent to uninterrupted execution.
6. F-SAPT stages restore and reuse their required matrices rather than rerunning completed routines.
7. D3 and D4 completion is represented and honored correctly.
8. Native QCSchema and equivalent ordinary Python API jobs share one canonical identity implementation.
9. Any incompatible or corrupt checkpoint fails explicitly.
10. Existing non-checkpointed SAPT(DFT) behavior remains unchanged.

## 16. Implementation boundaries

The checkpoint manager should isolate four responsibilities behind focused interfaces:

1. **Identity builder:** produces canonical QCSchema input, execution fingerprint, and digest.
2. **Manifest/artifact store:** validates, locks, reads, and atomically commits checkpoint state.
3. **SCF snapshot adapter:** serializes converged SCF state and reconstructs usable SCF subclasses without iteration.
4. **Stage coordinator:** declares dependencies, restores outputs, and decides which persistent stage executes next.

The SAPT computational routines should not manipulate JSON, locks, checksums, or filenames directly. They should query stage completion, request typed restored outputs, and commit typed stage outputs through the coordinator.

Unrelated SAPT refactoring is outside scope.
