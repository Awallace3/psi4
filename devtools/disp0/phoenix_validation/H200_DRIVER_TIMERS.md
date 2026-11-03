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
