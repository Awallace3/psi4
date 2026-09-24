| Tree | Node | Threads | DGEMM GF/s per core | Triad GB/s | Scalar Miter/s | Live MHz |
|---|---|---:|---:|---:|---:|---:|
| M1-core6-h200-job13358747 | atl1-1-02-014-9-0.pace.gatech.edu | 8 | 84.3 | 14.4 | 47.4 | 2800.0 |
| M2-cpu24-core6-job13358750 | atl1-1-02-008-2-2.pace.gatech.edu | 8 | 76.5 | 9.3 | 24.7 | 1200.0 |
| M3-premerge-core6-h200-job13367763 | atl1-1-02-012-9-0.pace.gatech.edu | 8 | 84.2 | 14.6 | 47.3 | 2800.0 |
| M4-core6-h200-cuest022-job13376151 | atl1-1-02-012-23-0.pace.gatech.edu | 8 | 24.5 | 4.2 | 13.4 | 1800.0 |
| M4-core6-h200-cuest022-job13395711 | atl1-1-02-012-2-0.pace.gatech.edu | 8 | 84.9 | 14.5 | 47.4 | 2800.0 |
| M5-premerge-core6-h200-cuest022-job13376152 | atl1-1-02-012-23-0.pace.gatech.edu | 8 | 24.5 | 4.9 | 10.0 | 1800.0 |
| M5-premerge-core6-h200-cuest022-job13395712 | atl1-1-03-019-2-0.pace.gatech.edu | 8 | 84.2 | 12.1 | 47.0 | 2800.0 |
| P1-protein157-h200-job13376153 | atl1-1-02-012-23-0.pace.gatech.edu | 8 | 24.4 | 5.1 | 13.4 | 800.0 |
| P1-protein157-h200-job13395713 | atl1-1-02-014-9-0.pace.gatech.edu | 8 | 24.6 | 5.2 | 13.5 | 2800.0 |
| P1-protein157-h200-job13429862 | atl1-1-03-020-18-0.pace.gatech.edu | 8 | 84.1 | 12.1 | 35.1 | 2800.0 |
| P1-protein157-h200-job13480138 | atl1-1-03-020-11-0.pace.gatech.edu | 8 | 84.2 | 12.0 | 47.3 | 2800.0 |
| P2-protein157-h200-job13395714 | atl1-1-02-012-2-0.pace.gatech.edu | 8 | 83.3 | 14.6 | 47.3 | 2800.0 |
| P2-protein157-h200-job13429863 | atl1-1-03-018-14-0.pace.gatech.edu | 8 | 84.2 | 12.0 | 35.1 | 2797.2 |
| P2-protein157-h200-job13480139 | atl1-1-03-020-11-0.pace.gatech.edu | 8 | 84.3 | 11.8 | 47.5 | 2800.0 |
| P3-protein157-h200-job13395715 | atl1-1-03-019-2-0.pace.gatech.edu | 8 | 84.2 | 12.0 | 47.5 | 2800.0 |

Verdict: **mismatched** (worst pairwise ratio 4.76×, tolerance 1.25×).

Host speed changed between the start and end of the run, so this tree's own cases are not mutually comparable:

- `M4-core6-h200-cuest022-job13376151`: dgemm_gflops_per_core 1.00×, live_mhz 2.25×, scalar_miter_s 1.00×, stream_gb_s 1.21×
- `M4-core6-h200-cuest022-job13395711`: dgemm_gflops_per_core 1.00×, live_mhz 1.00×, scalar_miter_s 1.33×, stream_gb_s 1.00×
- `M5-premerge-core6-h200-cuest022-job13376152`: dgemm_gflops_per_core 1.00×, live_mhz 2.25×, scalar_miter_s 1.00×, stream_gb_s 1.00×
