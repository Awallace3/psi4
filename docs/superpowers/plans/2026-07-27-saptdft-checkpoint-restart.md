# SAPT(DFT) Checkpoint Restart Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement durable SAPT(DFT) restart that restores completed SCF calculations, never serializes JK, rebuilds JK lazily, validates equivalent jobs through canonical QCSchema serialization, and resumes scalar and F-SAPT stages without rerunning completed work.

**Architecture:** Add a focused `saptdft_checkpoint.py` module that owns canonical identity, the atomic manifest/artifact store, SCF snapshots, and explicit stage completion. Keep computation in `sapt_proc.py`, replacing its ordinal inline manager with typed stage queries and commits. Rehydrate RHF/RKS objects from a fresh base wavefunction plus copied converged state; directly passing a deserialized base wavefunction to `scf_wavefunction_factory()` is prohibited because it has been reproduced to segfault.

**Tech Stack:** Python 3.14, Psi4 core Python bindings, QCEngine/QCElemental QCSchema v2, NumPy artifacts, pytest, SHA-256 JSON manifests.

## Global Constraints

- Preserve the approved design at `docs/superpowers/specs/2026-07-27-saptdft-checkpoint-restart-design.md`.
- Preserve and build on the existing uncommitted checkpoint changes in `psi4/driver/procrouting/sapt/sapt_proc.py` and `tests/pytests/test_fsaptdft.py`; never reset, stash, or discard them.
- Completed SCF stages must never call `scf_helper`, `run_scf`, `compute_energy`, or an SCF iterator during restart.
- JK objects and JK internals must never be serialized; JK and SAPT JK caches are transient and rebuilt only for unfinished work.
- Any job-identity mismatch, corrupt artifact, unsupported version, invalid dependency, or concurrent writer must raise `ValidationError`, never silently restart from scratch.
- Native QCSchema and ordinary `psi4.energy()` calls must use the same canonical identity builder.
- External-potential checkpointing remains explicitly unsupported.
- Use atomic artifact-first, manifest-last commits with SHA-256 and size validation.
- Run source-backed tests through the existing build at `build_saptdft_ein_fi_option_d4_cp`; rebuild/install Python files before focused tests.

---

### Task 1: Canonical identity and atomic checkpoint store

**Files:**
- Create: `psi4/driver/procrouting/sapt/saptdft_checkpoint.py`
- Modify: `psi4/driver/procrouting/sapt/CMakeLists.txt`
- Modify: `psi4/driver/p4util/procutil.py`
- Test: `tests/pytests/test_fsaptdft.py`

**Interfaces:**
- Produces: `build_saptdft_job_identity(*, name, molecule, function_kwargs=None, atomic_input=None) -> dict` with `canonical_input`, `execution_fingerprint`, and `sha256`.
- Produces: `SAPTDFTCheckpoint(path: Path, identity: dict)` with `open()`, `is_complete(stage)`, `restore_scalars(keys)`, `restore_array(name)`, `commit_stage(stage, *, scalars=None, arrays=None, wavefunctions=None)`, and `close()`.
- Produces: `SAPTDFT_STAGE_DEFINITIONS: dict[str, StageDefinition]` and validation of explicit dependencies.

- [ ] **Step 1: Add failing identity/store tests**

Add tests that instantiate the module without running SAPT and assert deterministic identity, runtime-control exclusion, changed-geometry mismatch, manifest schema, checksum failure, unknown stage failure, artifact-first interruption behavior, and lock contention. Use a tiny Ne dimer and `p4util.state_to_atomicinput(dtype=2, driver="energy", method="sapt(dft)", ...)` as the ordinary-API schema source.

- [ ] **Step 2: Run the new tests and verify failure**

Run:

```bash
cmake --build build_saptdft_ein_fi_option_d4_cp -j2
cd /tmp && PYTHONPATH=$OLDPWD/build_saptdft_ein_fi_option_d4_cp/stage/lib \
  python -m pytest -q $OLDPWD/tests/pytests/test_fsaptdft.py -k 'checkpoint_identity or checkpoint_store or checkpoint_lock'
```

Expected: failure because `saptdft_checkpoint.py` and its interfaces do not exist.

- [ ] **Step 3: Implement canonicalization and store**

Use QCSchema v2 model serialization (`model_dump(mode="json")` where available, with the repository's v1/v2 compatibility conventions) and deterministic `json.dumps(..., sort_keys=True, separators=(",", ":"), allow_nan=False)`. Strip only checkpoint directory, stop-after, output, memory, threads, timer, and verbosity controls. Include exact Psi4/checkpoint schema versions and selected backend/add-on versions in the execution fingerprint.

Implement artifact writes as unique temp file → flush/fsync → SHA-256/size → `os.replace()` final artifact → temporary manifest → flush/fsync → `os.replace()` manifest. Acquire an exclusive lock with PID metadata before mutation; a live lock raises `ValidationError`.

- [ ] **Step 4: Run focused tests and verify pass**

Run the command from Step 2. Expected: all selected tests pass.

- [ ] **Step 5: Commit**

```bash
git add psi4/driver/procrouting/sapt/saptdft_checkpoint.py \
        psi4/driver/procrouting/sapt/CMakeLists.txt \
        psi4/driver/p4util/procutil.py tests/pytests/test_fsaptdft.py
git commit -m "feat: add SAPTDFT checkpoint store"
```

### Task 2: Safe RHF/RKS snapshot rehydration without SCF

**Files:**
- Modify: `psi4/driver/procrouting/sapt/saptdft_checkpoint.py`
- Modify: `psi4/driver/procrouting/proc.py`
- Test: `tests/pytests/test_fsaptdft.py`

**Interfaces:**
- Produces: `capture_scf_snapshot(wfn, *, reference: str, method: str) -> dict[str, object]`.
- Produces: `rehydrate_scf_wavefunction(snapshot, *, method: str, reference: str) -> core.HF`.
- Consumes: checkpoint wavefunction artifacts and canonical identity from Task 1.

- [ ] **Step 1: Add failing RHF and RKS round-trip tests**

For small converged RHF and RKS wavefunctions, serialize and reload the base wavefunction, invoke `rehydrate_scf_wavefunction`, and assert the result is the correct SCF subclass, supports `set_jk` and `cphf_Hx`, and exactly restores energy, `Ca/Cb`, `Da/Db`, `Fa/Fb`, orbital energies, dimensions, and required variables. Monkeypatch SCF convergence entry points to raise during rehydration.

Add a regression test in a subprocess asserting that this unsafe operation is never used:

```python
loaded = core.Wavefunction.from_file(path)
# prohibited: scf_wavefunction_factory(method, loaded, reference)
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
cmake --build build_saptdft_ein_fi_option_d4_cp -j2
cd /tmp && PYTHONPATH=$OLDPWD/build_saptdft_ein_fi_option_d4_cp/stage/lib \
  python -m pytest -q $OLDPWD/tests/pytests/test_fsaptdft.py -k 'checkpoint_rehydrate'
```

Expected: failure because safe rehydration is not implemented.

- [ ] **Step 3: Implement the safe reconstruction path**

The implementation must follow this proven-safe shape:

```python
loaded = core.Wavefunction.from_file(path)
fresh_base = core.Wavefunction.build(loaded.molecule(), loaded.basisset())
rehydrated = scf_wavefunction_factory(method, fresh_base, reference)
# Copy converged matrices/vectors/dimensions/energy/QCVariables from loaded.
```

Copy into the fresh subclass's allocated matrices and vectors; reconstruct functional and auxiliary basis state from canonical options. Never pass `loaded` directly into `scf_wavefunction_factory()`. Validate molecule, basis, reference, dimensions, snapshot version, and required fields before return. A failure raises `ValidationError`; it never requests SCF recomputation.

- [ ] **Step 4: Run focused tests and verify pass**

Run the Step 2 command. Expected: RHF and RKS round trips pass with SCF guards active.

- [ ] **Step 5: Commit**

```bash
git add psi4/driver/procrouting/sapt/saptdft_checkpoint.py \
        psi4/driver/procrouting/proc.py tests/pytests/test_fsaptdft.py
git commit -m "feat: rehydrate checkpointed SCF wavefunctions"
```

### Task 3: Integrate explicit stages and SCF-skipping restart into SAPT(DFT)

**Files:**
- Modify: `psi4/driver/procrouting/sapt/sapt_proc.py`
- Modify: `psi4/driver/procrouting/sapt/saptdft_checkpoint.py`
- Create: `tests/pytests/fsaptdft_checkpoint_worker.py`
- Test: `tests/pytests/test_fsaptdft.py`

**Interfaces:**
- Consumes: identity/store and SCF snapshot APIs from Tasks 1–2.
- Produces: persistent setup, SCF, HF-SAPT, SAPT(DFT), delta-DFT, D3, D4, dispersion, and `final` stage records.
- Produces: subprocess worker modes `reference`, `stop`, `restart`, and `restart_with_guards`, returning a JSON summary.

- [ ] **Step 1: Add fresh-process failing restart tests**

Replace numerical-only restart assertions with subprocess tests. After stopping at each SCF stage, restart with `scf_helper`, `run_scf`, and SCF iteration/energy entry points guarded to raise. Assert final energies match an uninterrupted reference, manifest `completed_stages` is correct, and `d3`/`d4` are represented when selected.

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
cmake --build build_saptdft_ein_fi_option_d4_cp -j2
cd /tmp && PYTHONPATH=$OLDPWD/build_saptdft_ein_fi_option_d4_cp/stage/lib \
  python -m pytest -q $OLDPWD/tests/pytests/test_fsaptdft.py -k 'checkpoint_restart_skips_scf or checkpoint_stage_dependencies or checkpoint_d3_d4'
```

Expected: current ordinal manager clears restored state or reruns SCF, so guards fail.

- [ ] **Step 3: Replace the ordinal manager and integrate restored SCFs**

Remove `_SAPTDFT_CHECKPOINT_STAGES`, the inline `_SAPTDFTCheckpoint`, ordinal `reached()`, and persistent `build_jk`/`hf_sapt_jk` stages. Initialize the new manager with canonical identity before expensive work. For every completed SCF stage, rehydrate its RHF/RKS object and restore scalar energy instead of invoking SCF. Commit each new SCF snapshot only after convergence completes.

Guard every scalar stage independently through explicit dependencies. Add `d3` and `d4` definitions. A complete `final` stage restores the dimer result variables and returns before any SCF or JK setup.

- [ ] **Step 4: Run focused restart tests and verify pass**

Run the Step 2 command. Expected: all selected restart/stage tests pass in fresh processes.

- [ ] **Step 5: Commit**

```bash
git add psi4/driver/procrouting/sapt/sapt_proc.py \
        psi4/driver/procrouting/sapt/saptdft_checkpoint.py \
        tests/pytests/fsaptdft_checkpoint_worker.py tests/pytests/test_fsaptdft.py
git commit -m "feat: resume SAPTDFT from explicit stages"
```

### Task 4: Lazy transient JK reconstruction and F-SAPT artifact reuse

**Files:**
- Modify: `psi4/driver/procrouting/sapt/sapt_proc.py`
- Modify: `psi4/driver/procrouting/sapt/saptdft_checkpoint.py`
- Modify: `tests/pytests/fsaptdft_checkpoint_worker.py`
- Test: `tests/pytests/test_fsaptdft.py`

**Interfaces:**
- Consumes: restored SCF subclasses and explicit stage decisions.
- Produces: lazy `build_saptdft_jk(...) -> core.JK` behavior with no persistent JK representation.
- Produces: dedicated array artifacts for `Elst_AB`, `Exch_AB`, `IndAB_AB`, `IndBA_AB`, and `Disp_AB` plus required localization/fragment metadata.

- [ ] **Step 1: Add failing JK and F-SAPT reuse tests**

Instrument `core.JK.build` and completed F-SAPT routines. Assert restart before unfinished induction rebuilds and attaches JK exactly when needed; restart from `final` builds no JK; no artifact or serialized wavefunction contains JK state; and completed F-SAPT electrostatics/exchange/induction/dispersion routines are not called after restart.

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
cmake --build build_saptdft_ein_fi_option_d4_cp -j2
cd /tmp && PYTHONPATH=$OLDPWD/build_saptdft_ein_fi_option_d4_cp/stage/lib \
  python -m pytest -q $OLDPWD/tests/pytests/test_fsaptdft.py -k 'checkpoint_jk or checkpoint_fsapt or checkpoint_final'
```

Expected: current code eagerly rebuilds or reruns F-SAPT stages and lacks restorable array artifacts.

- [ ] **Step 3: Implement lazy JK and typed F-SAPT restoration**

Construct J/K/wK/omega/backend settings from canonical options only after the next unfinished stage is known. Attach the rebuilt JK to restored monomer SCF subclasses before induction calls `cphf_Hx`; build a separate monomer-B JK when range-separated settings differ. Keep SAPT JK caches transient.

Persist F-SAPT matrices as checksummed array artifacts and restore them under their original cache keys before the next unfinished F-SAPT stage. Add `is_complete()` guards around setup, electrostatics, exchange, induction, dispersion, and final publication.

- [ ] **Step 4: Run focused tests and verify pass**

Run the Step 2 command. Expected: selected JK/F-SAPT/final tests pass.

- [ ] **Step 5: Commit**

```bash
git add psi4/driver/procrouting/sapt/sapt_proc.py \
        psi4/driver/procrouting/sapt/saptdft_checkpoint.py \
        tests/pytests/fsaptdft_checkpoint_worker.py tests/pytests/test_fsaptdft.py
git commit -m "feat: rebuild JK and reuse FSAPT checkpoints"
```

### Task 5: QCSchema handoff, mismatch coverage, and full regression validation

**Files:**
- Modify: `psi4/driver/schema_wrapper.py`
- Modify: `psi4/driver/p4util/procutil.py`
- Modify: `psi4/driver/procrouting/sapt/sapt_proc.py`
- Modify: `tests/pytests/fsaptdft_checkpoint_worker.py`
- Test: `tests/pytests/test_fsaptdft.py`
- Test: `tests/pytests/test_saptdft.py`

**Interfaces:**
- Consumes: `build_saptdft_job_identity` from Task 1.
- Produces: private validated `AtomicInput` handoff from QCSchema execution to SAPT(DFT), while direct Python execution synthesizes the equivalent v2 input through `state_to_atomicinput`.

- [ ] **Step 1: Add failing API-equivalence and error tests**

Cover equivalent QCSchema/direct identities, omitted versus explicit defaults, mismatch diagnostics for geometry/fragments/charge/multiplicity/method/basis/options/backend/version, malformed/truncated/checksum-invalid artifacts, unsupported manifest/snapshot versions, interrupted writes, lock contention, and external-potential rejection.

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
cmake --build build_saptdft_ein_fi_option_d4_cp -j2
cd /tmp && PYTHONPATH=$OLDPWD/build_saptdft_ein_fi_option_d4_cp/stage/lib \
  python -m pytest -q $OLDPWD/tests/pytests/test_fsaptdft.py -k 'checkpoint_identity or checkpoint_mismatch or checkpoint_corruption'
```

Expected: native QCSchema context is not yet handed to the shared identity builder or one of the explicit errors is missing.

- [ ] **Step 3: Complete QCSchema integration and diagnostics**

Pass the original validated v2 `AtomicInput` through a private driver kwarg that cannot affect computation and is stripped from identity runtime controls. For direct calls, synthesize the equivalent v2 input after method and relevant option resolution. Keep canonicalization centralized. Report useful differing canonical fields with every identity mismatch.

- [ ] **Step 4: Run focused and regression suites**

Run:

```bash
cmake --build build_saptdft_ein_fi_option_d4_cp -j2
cd /tmp && PYTHONPATH=$OLDPWD/build_saptdft_ein_fi_option_d4_cp/stage/lib \
  python -m pytest -q $OLDPWD/tests/pytests/test_fsaptdft.py -k checkpoint
cd /tmp && PYTHONPATH=$OLDPWD/build_saptdft_ein_fi_option_d4_cp/stage/lib \
  python -m pytest -q $OLDPWD/tests/pytests/test_saptdft.py -k qcschema
```

Then run at least one existing non-checkpoint HF/F-SAPT test selected from the collected test names in `test_fsaptdft.py`. Expected: all commands pass and checkpoint-disabled results remain unchanged.

- [ ] **Step 5: Verify the approved spec line-by-line**

Confirm every acceptance criterion in Section 15 of the approved design has implementation and test evidence. Search for placeholders and confirm there is no fallback that clears incompatible state and silently reruns SCF.

- [ ] **Step 6: Commit**

```bash
git add psi4/driver/schema_wrapper.py psi4/driver/p4util/procutil.py \
        psi4/driver/procrouting/sapt/sapt_proc.py \
        psi4/driver/procrouting/sapt/saptdft_checkpoint.py \
        tests/pytests/fsaptdft_checkpoint_worker.py \
        tests/pytests/test_fsaptdft.py tests/pytests/test_saptdft.py
git commit -m "test: validate SAPTDFT checkpoint restart"
```
