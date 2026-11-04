# Recovered original H200/CPU driver timers

Phoenix SSH access returned on2026-11-03. All six original per-worker
`timer.dat` files exist and were retrieved read-only; their downloaded hashes
match the remote originals. **No benchmark or oracle was rerun.**

Scientific pin: `927283f8ed0a94d42179c35dd2a1a5bce674347e`.
Method: scaled HF-reference F-SAPT/aug-cc-pVDZ, spherical, exact per-source
coupled response. Peptide:24atoms,356primary basis functions.
CPU:24Gold6226cores; accelerator:oneH200plus8host cores.

The parser in `devtools/disp0/cachy/profile_sapt_steps.py` accepts one nonempty
flat epoch and reads the wall column, not accumulated user CPU seconds.
The duplicate nested tree is not added. Each listed stage has one call.
Core timers have0.001s precision.

| Driver window | CPU r0/r1/r2 (s) | H200 r0/r1/r2 (s) | CPU median (s) | H200 median (s) | Ratio of medians |
|---|---|---|---:|---:|---:|
| Dimer SCF | 8.188 / 5.859 / 5.839 | 5.799 / 5.281 / 5.479 | 5.859 | 5.479 | 1.069× |
| Monomer A SCF | 2.728 / 2.719 / 2.736 | 1.113 / 1.225 / 1.122 | 2.728 | 1.122 | 2.431× |
| Monomer B SCF | 2.715 / 2.700 / 2.731 | 1.078 / 1.105 / 1.160 | 2.715 | 1.105 | 2.457× |
| SAPT JK/cache | 0.304 / 0.302 / 0.300 | 0.108 / 0.108 / 0.109 | 0.302 | 0.108 | 2.796× |
| Canonical electrostatics | 0.003 / 0.003 / 0.003 | 0.002 / 0.002 / 0.002 | 0.003 | 0.002 | 1.500× |
| Canonical exchange | 0.251 / 0.247 / 0.244 | 0.028 / 0.033 / 0.034 | 0.247 | 0.033 | 7.485× |
| Canonical induction | 3.691 / 3.670 / 3.661 | 0.396 / 0.392 / 0.399 | 3.670 | 0.396 | 9.268× |
| Dispersion dispatch/metadata | 0.109 / 0.104 / 0.105 | 0.060 / 0.059 / 0.059 | 0.105 | 0.059 | 1.780× |

The electrostatics ratio has limited meaning at2–3ms timer resolution.
Dispatch/metadata is **not fragment dispersion**. These ratios describe
driver windows with different CPU resources, not isolated kernel speedups.

The core GPU fragment timer is28.526/28.473/28.567s. It encloses the full
continuation and is slightly wider than the previously retained JSON
fragment interval28.488587549/28.442927432/28.534608316s. Do not silently
substitute one boundary for the other. Original whole-call7.004× and
JSON-fragment8.555× aggregate ratios remain unchanged.

**The H200 elst/exch/ind/disp fragment subterms remain unresolved** in these
core timer files. CPU fragment terms are present and agree with the existing
higher-resolution JSON intervals to timer rounding. Cachy's separate traced
fragment decomposition is not substituted into H200 data.

## Artifact locations and hashes

Base:
`/storage/project/r-cs207-0/awallace43/runs/saptdft-cuest-disp0`.

- CPU root:
  `phoenix-sealed-cpu-qualification-927283f8ed/job-14991749/`.
- GPU root:
  `phoenix-sealed-h200-927283f8ed/job-14946636/benchmark/execution/`.
- Each file is `<root>/peptide-r{0,1,2}/timer.dat`.
- Local downloaded copies:
  `/tmp/fsapt-h200-driver-timers-20261103/{cpu,gpu}-r{0,1,2}-timer.dat`.
  These are analysis copies, not runtime dependencies.

| Artifact | Bytes | SHA256 |
|---|---:|---|
| CPU r0 | 23713 | `3adef7bf26e28de7106b874fa0a802c5d022094772b11b64ef279713f0cbc3b3` |
| CPU r1 | 23713 | `71c0c597f153753f19f37d9d2c1fce0b3b61cb5b964963e8fe626ff38d604e0a` |
| CPU r2 | 23713 | `528096c1ba26955f0b6e3d1fd10df30840fd0423bd2c9782d72af9281a97b7d3` |
| H200 r0 | 7790 | `486123a626ee381a06de1a806f3a6dca4379d4060b4775cf32db3d2836339c92` |
| H200 r1 | 7790 | `75bbfb5df98ccba01c151f2b6106074a131911427bd18fc56424c7b3181aa7d3` |
| H200 r2 | 7790 | `794b85e91abfb7731ac859c02ca624d498588646d3a36f0c5dc7635da589490f` |

Every embedded core wall value and all six hashes in
`~/docs/saptdft/cuest/fsapt.html` were checked against these raw artifacts.

## Backend audit: cuEST is active, cold totals conceal steady JK work

The user challenged the near-1× dimer ratio. Read-only retrieval of all six
original `psi4.out` files confirms:

- Every GPU output prints `cuESTJK: GPU-Accelerated Density-Fitted J/K Matrices`
  before the first dimer iteration; it prints88cuESTJK calls in total.
- Both monomers print `Re-using passed JK object instead of rebuilding`.
- CPU outputs instead identify in-core `MemDFJK`; no `cuESTJK` calls occur.
- At the executed927283f8ed pin, `fragment_gate.options()` sets
  `USE_CUEST=True` for the GPU arm, false for CPU, with mixed precision disabled.
  `SCF_TYPE=DF` describes the algorithm, **not evidence of a CPU fallback**.
- `sapt_proc.py:603,613,633` shares the dimer's saved JK with both monomers.
  `sapt_proc.py:1023–1029` passes `wfn_B.jk()` into `sapt_dft`.
  In this executed route `SAPT(DFT):Build JK` therefore encloses SAPT cache
  construction using that object, not initial JK-object setup. The generic
  function *can* initialize an object when none is supplied (`1511–1520`).

The original labeling “JK object / cache” was misleading. It is now labeled
**SAPT JK cache (not initial object setup)**, and the dimer timer is labeled
**cold-inclusive**. No raw timing is changed or startup cost removed.

### Nested paths, not additional flat contributions

| Nested timer path suffix | Calls/run | CPU r0/r1/r2 (s) | H200 r0/r1/r2 (s) | Ratio of medians |
|---|---:|---|---|---:|
| Dimer → HF: Form core H | 1 | .064 / .059 / .067 | 2.156 / 1.802 / 1.847 | 0.035× |
| Dimer → HF: Form G → JK: JK | 18 | 1.461 / 1.453 / 1.462 | 1.458 / 1.451 / 1.452 | 1.006× |
| Dimer → HF: DIIS | 17 | 3.286 / 1.145 / 1.107 | .918 / .843 / .992 | 1.247× |
| Monomer A → HF: Form G → JK: JK | 17 | .931 / .918 / .926 | .088 / .088 / .089 | 10.523× |
| Monomer B → HF: Form G → JK: JK | 17 | .931 / .922 / .927 | .086 / .086 / .087 | 10.779× |
| Build JK → JK: JK | 1 | .217 / .214 / .213 | .010 / .010 / .010 | 21.400× |

These paths are selected from the nested tree, not the global flat `JK: JK`
aggregate. They are explanatory child windows; never add them to their
already-inclusive parent rows or mix them into the full-call stack.

### First-use dimer call from the original native log

| GPU repeat | First JK total (ms) | First J (ms) | First K (ms) | Following17 JK calls range (ms) | Following17 median (ms) |
|---|---:|---:|---:|---|---:|
| r0 | 1370.46 | 1366.50 | 2.06 | 5.05–5.52 | 5.10 |
| r1 | 1362.32 | 1358.30 | 2.07 | 5.08–5.40 | 5.14 |
| r2 | 1362.82 | 1358.78 | 2.07 | 5.10–5.37 | 5.19 |

Thus the cold dimer JK interval really includes a first-call1.36s stall;
subsequent calls are about5ms. Core-H construction also has a first-use cost,
and DIIS/other host-side SCF work remains in the enclosing dimer timer.
The exact opaque-library split between initialization and compilation is
**not** measured. No artificial warmed dimer-SCF ratio is reconstructed by
subtracting these costs from an independently timed CPU total.

All36displayed nested wall/call-count pairs, all native first/later-call
statistics, cuEST backend markers and both reuse markers were checked against
the original local artifacts. Six output SHA256 values were independently
matched to the remote originals; no science code or experiment was changed.

| Original `psi4.out` | SHA256 |
|---|---|
| CPU r0 | `0df34325348c7bcb322cd3ea9f2f31e2ed6c639234a72d35ce42863fc5c3a021` |
| CPU r1 | `68607c01f0659b3fbde990d29f4eab5b6508602e435cd7a0e1cbd040eb526b02` |
| CPU r2 | `7829395a5ff84362b3cb7441c03ae70f0423513cd9d9ea1c0cfdf003e5c54f56` |
| H200 r0 | `94af32a63a49051f61c8d5be3387a1365afefb2a7070a84598db66a1d0a8bc56` |
| H200 r1 | `1a8db2ac640d979069f706ab49fb0fb5f868c795f956e9945e7a527c765ebb9e` |
| H200 r2 | `7a9de2f14a6c7176d072bf2130475b030928d4a4e827315d986cbbf32fdddac8` |
