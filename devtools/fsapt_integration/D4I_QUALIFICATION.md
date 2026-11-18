# F-SAPT(HF/PBE0)-D4(I)/cc-pVDZ on cuEST

## Methods and evidence

Both methods use the production scalar induction definition, CPHF/CPKS with delta-HF. F-SAPT uses the default rescaled uncoupled fragment partition. The four signed scalar components are ELST, EXCH, IND and intermolecular empirical DISP. Fragment induction closure uses the sum of both directions. The HF monomer wavefunctions already supply the SAPT(HF) monomers. PBE0 also requires iterative GRAC, two Kohn–Sham monomer SCFs, and an HF interaction calculation for delta-HF.

The timing ledger gives inclusive wall seconds for a fresh energy call per row, hartrees for signed energies, and fragment sum minus scalar for closure. Blank entries denote inactive terms or unavailable instrumentation. The atom-pair extraction timer sits **inside** D4. The same pair matrix supplies the scalar dispersion and fragment dispersion. Wall totals also include driver overhead outside these timers. Each row is a single observation. [Timing ledger](D4I_TERM_TIMES.csv) · [Worker](d4i_timing.py) · [Timer definitions](d4i_summary.py)

| Operation | Executed implementation for the GPU arm |
|---|---|
| GRAC A/B, PBE0 monomers A/B | Iterative SCFs with cuEST DF-J/K. XC uses the cuEST grid. XC functional evaluation depends on the mapped LibXC build. |
| HF dimer, HF monomers A/B | cuEST DF-J/K for all three. Orbital updates include host work. |
| Scalar elst / exch | Electrostatics contracts cached densities and potentials on the host. Exchange combines cuEST JK intermediates and host matrix work. |
| Scalar ind | Host-driven response iteration with cuEST JK for the Hessian action and XC response for PBE0. The HF response used for PBE0 delta-HF has separate timers. |
| Scalar disp / fdisp | CPU D4 atom-pair evaluation and intermolecular pair extraction. Both methods skip the quantum dispersion solver. |
| felst | GPU one-electron nuclear potentials and localized-orbital Coulomb builds on the cuEST fragment path. Host projections and matrix assembly remain. |
| fexch | The tiled DFMO path transforms pairs of localized occupied and projected orbitals and reduces their auxiliary-index products on device. Other exchange contributions include host linear algebra. |
| find | Nuclear potentials and orbital J sources use cuEST. Projection, disk-backed perturbations, uncoupled response and rescaling still contribute. |

## Observed fragment costs

The RTX 3090 measurements are debugging observations with two host threads, CUDA 13.5 and host LibXC. Absolute Phoenix throughput requires a matched allocation and CUDA LibXC. The CPU reference in job 15799432 uses eight Phoenix CPU cores. The two water-grouped samples are CPU smoke tests on cachy. Its cuEST-enabled binary ran the host DF arm with USE_CUEST disabled.

| System and functional | Fragment integral path | Total s | FElst s | FExch s | FInd s | Max closure, hartree |
|---|---|---:|---:|---:|---:|---:|
| protein83 HF, 766 AOs | Orbital J/K | 192.00 | 4.83 | 87.63 | 8.90 | 6.53e-12 |
| protein83 HF, 766 AOs | Tiled DFMO | 108.47 | 4.51 | 4.58 | 8.15 | 2.30e-12 |
| protein83 PBE0, 766 AOs | Orbital J/K | 488.26 | 5.72 | 89.32 | 8.22 | 8.19e-12 |
| CDK2 2 Å pocket HF, 995 AOs | Tiled DFMO | 229.11 | 10.91 | 11.81 | 19.39 | 5.38e-9 |

The CDK2 pilot uses ligand A, 56 QM protein atoms on B, and 8,977 MM charges assigned to B. On HF the largest closure residual is electrostatic. On protein83 HF with tiled DFMO, the scalar induction timer is 38.99 s, larger than all three fragment timers combined. On protein83 PBE0 with orbital J/K, GRAC takes 83.36 and 42.28 s, scalar induction takes 99.69 s and the extra HF induction takes 40.70 s.

At fixed SCF orbitals, water HF/PBE0, a peptide with uneven DFMO tiles, and a reversed-occupancy dimer all agree with host transformed-integral fragment matrices to within 9e-13 hartree per entry. This comparison includes exchange before scalar rescaling. Separate energy calls also exercise the public driver, SCF reproducibility, and the external MM fragment. Passing scalar closure alone would not establish correct individual exchange entries.

## Full pocket memory

The 16 prepared CDK2 complexes contain 555–566 QM atoms, 5,187–5,300 spherical orbital basis functions and 25,361–25,938 auxiliary functions. The following sizes count named double-precision tensors. Peak RAM, scratch and device memory include additional allocations.

| Tensor or stage | Full CDK2 space per ligand | Tiled fragment path |
|---|---:|---|
| Induction occupied–virtual DF factors, Aar and Abs | 831–900 GiB | Replaced by localized Coulomb builds |
| Exchange projected factors, Bab and Bba | 30.2–41.0 GiB | Fit and reduce in GPU tiles |
| Induction potential slabs, WAbs and WBar | 23.8–27.2 GiB | Still disk-backed |

A cuEST 0.3.0.1 workspace query for the full CDK2 ligand 0000 geometry (5,209 AOs, 25,509 auxiliary functions) returned 394,265,141,248 persistent and 343,383,764,992 temporary device bytes for the cached DF integral plan. These overlap during creation. The integral-direct plan returned 5,453,289,984 and 20,824,834,304 bytes respectively. This query used the molecular coordinates, shell lists and screening from Psi4's JK builder and stopped before allocation. It measures plan creation only. For ligand 0000 each occupied column contributes 0.990 GiB of fitted factors per side of the exchange contraction. The tiled path recomputes these in blocks, so a single-column probe cannot establish whole-SCF throughput.

## Reproduction

The production Phoenix source is `c504418811bdcfb24b9bc82d8273f0f6c00e6112`. Its existing installations use cuEST 0.2.50. Modern builds and results are isolated under `/storage/project/r-cs207-0/awallace43/runs/fsapt-d4i-0301-20261117`. The runtime check requires the mapped cuEST library SHA-256 `cb810a15dff2d0c9c0c83cb56336bf80d77a74ace53f6e4d30921726679fc221` and CUDA runtime `13050`. Phoenix GPU recipes also check the mapped CUDA build of LibXC 7.1.2. Raw per-run results include the geometry digest, options, component energies, native timer paths and fragment matrices.

The timing-suite geometries come from `~/data/timing_test_suite/geometries/systems.py`, SHA-256 `72668390d24f134316c8e779931a2f65efcd9289ecea8f1cf20a72733262e3bb`. The CDK2 pilot uses the prepared `cutoff_scan/c2/cdk2_0000` input with coordinates for Psi4 external potentials in bohr. The two charges in the water test form a separate embedded correctness control.

## Device-memory qualification

`CUEST_DF_INTEGRAL_DIRECT = true` selects the low-memory cuEST plan explicitly. The SDK's symmetric exchange workspace query rejected a direct plan with `CUEST_STATUS_INVALID_PLAN`. Psi4 therefore obtains fitted factors with cuEST DFMO and contracts them in occupied tiles for symmetric and nonsymmetric K. The variable exchange workspace on the cached path also uses the available device memory. The default remains the cached plan. The fragment option `FISAPT_FSAPT_INTEGRAL_ALGORITHM = CUEST_DFMO` can be selected independently.

Four AO-matrix regressions compared J and K with CPU DF for symmetric and nonsymmetric densities, at exchange fractions 1 and 0.25. All four passed with a maximum permitted absolute error of 1e-8 hartree. A diagnostic build with two occupied columns per tile also passed all four tests using five columns, exercising two full tiles and a final one-column tile. Direct-plan HF and PBE0 embedded-water energy calls agreed with independent cached-plan calculations within 2e-9 hartree in total energy and 1.5e-8 hartree per fragment entry. The six embedded fragment regression cases covering CPU DF and GPU orbital J/K and tiled DFMO also passed.

The first reduced CDK2 PBE0 attempt completed the HF and DFT reference work and reached coupled induction, where the cached implementation requested 7.478 GiB with 7.081 GiB free on the shared RTX 3090. A retry with adaptive exchange scratch was terminated during SCF and produced no energy. Neither attempt supplies a PBE0 CDK2 timing. The full-pocket single-occupied J/K probe, full SCF, and ligand-ranking throughput require separate measurements.

The full-pocket J/K probe selects `CUEST_DF_INTEGRAL_DIRECT=True`. A full fragment energy would additionally use `FISAPT_FSAPT_INTEGRAL_ALGORITHM="CUEST_DFMO"`. The reduced runs in the timing ledger use cached J/K unless `integral_direct` is true. The fragment `AUTO` policy selects the tiled route when the estimated occupied–virtual factors exceed the FISAPT memory budget. It can select conventional DF on a small system with a large host memory budget.

The scalar candidate controls use the non-Einsums driver. The matched production control requests the Einsums backend used by the production source. Both use a fresh process, cc-pVDZ/cc-pVDZ-JKFIT, SAD, the same embedding, and iterative GRAC for PBE0. The candidate and production each have their own measured plain arm. The Phoenix eight-core CPU plain protein83 calculation took 263.90 s for HF and 1,069.31 s for PBE0. Comparing the independent protein83 RTX results to those CPU controls, the largest scalar differences were 1.94e-8 hartree for HF induction and 4.66e-7 hartree for PBE0 exchange.

The source containing both fragment DFMO and integral-direct J/K is `f0d4f7c06506150e595b22db4c0c60bf8485de33`. The cachy debug build overlays the changed native objects and Python driver onto an existing cuEST 0.3.0.1 Psi4 build. The clean Phoenix Release build targets x86-64-v3 and the CUDA architecture set supported by its toolkit. [Candidate source](https://github.com/Awallace3/psi4/tree/f0d4f7c06506150e595b22db4c0c60bf8485de33)

A portable Git bundle for the exact production reference is stored at `production-reference-c504418811.bundle` in the Phoenix run directory. Its prerequisite is the shared ancestor `ca21ef8645a73263281480454b730085506adca5` and its SHA-256 is `85ec4dc8012672900061c564b6f6726d1bb6eda00ac7404ec45a53812f282413`. The local copy passed `git bundle verify`.

## Phoenix job provenance

| Job | Inputs and purpose |
|---|---|
| 15799432 | Eight-core CPU controls on embedded water, peptide and protein83. The one-hour allocation timed out in the last protein83 PBE0 fragment case. The other eleven fresh energy calls produced result files. |
| 15801202 | Dedicated latest-candidate protein83 PBE0 CPU fragment rerun with sixteen threads and node-local scratch. The earlier process was observed in Lustre I/O wait. |
| 15800725 | Completed both isolated Release builds in 26 min 22 s. Verified the candidate and reference imports with their staged QCEElemental 0.50.4 and QCEngine 0.50.0. The reference also imports its staged Einsums backend. |
| 15800764 | Earlier free GPU comparison. Its candidate environment imports successfully. Its production phase places the older environment QCEElemental first and would fail before CDK2. |
| 15800861 | CPU controls for both methods on the real reduced CDK2 pocket. Its optional production phase uses the earlier import order. |
| 15801531 | Reduced CDK2 HF with the corrected residue grouping, orbital population export and node-local scratch. |
| 15799433 | Free GPU control for the earlier localized-orbital J/K implementation, including peptide and protein157. |

GPU jobs write one-second utilization and memory samples, per-case Psi4 logs, result JSON, fragment matrices and scalar/fragment checks under `job-<id>` in the run directory. The submitted GPU comparison checks scalar agreement and fragment closure with a 1e-6 hartree absolute limit. CUDA LibXC and the cuEST JK banner are checked by the worker. The per-term timers measure blocks containing both host and device work.


## Residue grouping and the MM column

The public group analysis now treats MM sites as zero-dispersion sites and preserves their electrostatic and induction contributions. The B-side external weight vector follows the column count of the energy matrix. The same input group dictionary can be reused across ligands. A regression with rectangular component matrices verified the public wavefunction result and the flattened, lowercase QCSchema result, including the grouped total and the MM contribution. The timing worker accepts `--groups` and saves the table in kcal/mol, times group analysis separately and records the two orbital population matrices for later regrouping. [Grouping implementation](https://github.com/Awallace3/psi4/commit/c077b9130959406f67a0ee3796b3543cb544c446)

For the real reduced HF pocket the stored component matrices have shape (205, 207). Summing the MM column over ligand A gives −0.0115980910 hartree electrostatic and −0.0138678371 hartree induction. Exchange and empirical dispersion against the MM sites are zero. These values come directly from the fragment matrices. The per-residue input for job 15801531 was matched element-by-element and coordinate-by-coordinate to the 101 atoms in the capped QM PDB. It assigns one ligand group, four protein residue groups and two explicit cap groups. The file is `groups/cdk2_0000_c2-groups.json` in the Phoenix run directory.

For embedded water the public grouping call took 0.00087 s for HF and 0.00105 s for PBE0 in separate CPU smoke tests. The MM group received exactly zero empirical dispersion in both. The fresh-energy wall timer excludes this optional post-processing.

At the 2026-11-17 checkpoint the GPU jobs were pending. The corrected grouped GPU recipe is `gpu-grouped.sbatch`. It puts each build's staged Python packages first, measures the embedded CDK2 pilot before protein83, captures the residue table and continues to independent cases when a case fails. The companion five-minute CPU recipe `production-smoke.sbatch` checks the production worker and timer export. Both request the free embers QoS. Admission retries record the real allocation IDs in `gpu-grouped.jobid` and `production-smoke.jobid` once the scheduler accepts them. The local campaign collector writes completed measurements to `.pi/d4i-nov17/phoenix-live/term-times.csv`. The full CDK2 ligand ranking and the matched Phoenix GPU throughput remain unqualified.
